use ark_std::UniformRand;
use ark_std::{end_timer, start_timer};
use ff::Field;
use ff::PrimeField;
use group::Curve;
use rand_core::OsRng;
use rand_core::RngCore;
use rayon::iter::IndexedParallelIterator;
use rayon::iter::IntoParallelRefMutIterator;
use rayon::iter::ParallelIterator;
use rayon::prelude::IntoParallelIterator;
use rayon::prelude::IntoParallelRefIterator;
use rayon::slice::ParallelSlice;
use std::env::var;
use std::ops::RangeTo;
use std::sync::atomic::AtomicUsize;
use std::sync::Mutex;
use std::time::Instant;
use std::{iter, sync::atomic::Ordering};

use super::{
    circuit::{
        Advice, Any, Assignment, Circuit, Column, ConstraintSystem, Fixed, FloorPlanner, Instance,
        Selector,
    },
    lookup, permutation, vanishing, ChallengeBeta, ChallengeGamma, ChallengeTheta, ChallengeX,
    ChallengeY, Error, ProvingKey,
};
use crate::arithmetic::eval_polynomial_st;
use crate::plonk::lookup::prover::Permuted;
use crate::{
    arithmetic::{eval_polynomial, BaseExt, CurveAffine, FieldExt},
    plonk::Assigned,
};
use crate::{
    plonk::Expression,
    poly::{
        self,
        commitment::{Blind, Params},
        multiopen::{self, ProverQuery},
        Coeff, ExtendedLagrangeCoeff, LagrangeCoeff, Polynomial, Rotation,
    },
};
use crate::{
    poly::batch_invert_assigned,
    transcript::{EncodedChallenge, TranscriptWrite},
};

thread_local! {
    pub static GPU_GROUP_ID: std::cell::Cell<usize>  =  std::cell::Cell::new(0);
}

lazy_static! {
    pub static ref N_GPU: usize = usize::from_str_radix(
        &std::env::var("HALO2_PROOFS_N_GPU").unwrap_or({
            #[cfg(feature = "cuda")]
            {
                ec_gpu_gen::rust_gpu_tools::Device::all().len().to_string()
            }
            #[cfg(not(feature = "cuda"))]
            {
                "1".to_owned()
            }
        }),
        10
    )
    .unwrap();
    pub static ref MSM_LOCK: Vec<Mutex<()>> =
        (0..*N_GPU).into_iter().map(|_| Mutex::new(())).collect();
}

/// This creates a proof for the provided `circuit` when given the public
/// parameters `params` and the proving key [`ProvingKey`] that was
/// generated previously for the same circuit. The provided `instances`
/// are zero-padded internally.
pub fn create_proof<
    C: CurveAffine,
    E: EncodedChallenge<C>,
    R: RngCore,
    T: TranscriptWrite<C, E>,
    ConcreteCircuit: Circuit<C::Scalar>,
>(
    params: &Params<C>,
    pk: &ProvingKey<C>,
    circuits: &[ConcreteCircuit],
    instances: &[&[&[C::Scalar]]],
    mut rng: R,
    transcript: &mut T,
) -> Result<(), Error> {
    for instance in instances.iter() {
        if instance.len() != pk.vk.cs.num_instance_columns {
            return Err(Error::InvalidInstances);
        }
    }

    println!("k is {}", pk.vk.domain.k());

    // Hash verification key into transcript
    pk.vk.hash_into(transcript)?;

    let domain = &pk.vk.domain;
    let mut meta = ConstraintSystem::default();
    let config = ConcreteCircuit::configure(&mut meta);

    // Selector optimizations cannot be applied here; use the ConstraintSystem
    // from the verification key.
    let meta = &pk.vk.cs;

    struct InstanceSingle<C: CurveAffine> {
        pub instance_values: Vec<Polynomial<C::Scalar, LagrangeCoeff>>,
        pub instance_polys: Vec<Polynomial<C::Scalar, Coeff>>,

        #[cfg(not(feature = "cuda"))]
        pub instance_cosets: Vec<Polynomial<C::Scalar, ExtendedLagrangeCoeff>>,
    }

    let timer = start_timer!(|| "instance");
    let instance: Vec<InstanceSingle<C>> = instances
        .iter()
        .map(|instance| -> Result<InstanceSingle<C>, Error> {
            let instance_values = instance
                .iter()
                .map(|values| {
                    let mut poly = domain.empty_lagrange();
                    assert_eq!(poly.len(), params.n as usize);
                    if values.len() > (poly.len() - (meta.blinding_factors() + 1)) {
                        return Err(Error::InstanceTooLarge);
                    }
                    for (poly, value) in poly.iter_mut().zip(values.iter()) {
                        *poly = *value;
                    }
                    Ok(poly)
                })
                .collect::<Result<Vec<_>, _>>()?;
            let instance_commitments_projective: Vec<_> = instance_values
                .iter()
                .map(|poly| params.commit_lagrange(poly))
                .collect();
            let mut instance_commitments =
                vec![C::identity(); instance_commitments_projective.len()];
            C::Curve::batch_normalize(&instance_commitments_projective, &mut instance_commitments);
            let instance_commitments = instance_commitments;
            drop(instance_commitments_projective);

            for commitment in &instance_commitments {
                transcript.common_point(*commitment)?;
            }

            let instance_polys: Vec<_> = instance_values
                .iter()
                .map(|poly| {
                    let lagrange_vec = domain.lagrange_from_vec(poly.to_vec());
                    domain.lagrange_to_coeff(lagrange_vec)
                })
                .collect();

            #[cfg(not(feature = "cuda"))]
            let instance_cosets: Vec<_> = instance_polys
                .iter()
                .map(|poly| domain.coeff_to_extended(poly.clone()))
                .collect();

            Ok(InstanceSingle {
                instance_values,
                instance_polys,

                #[cfg(not(feature = "cuda"))]
                instance_cosets,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    end_timer!(timer);
    let timer = start_timer!(|| "advice");
    struct AdviceSingle<C: CurveAffine> {
        pub advice_polys: Vec<Polynomial<C::Scalar, Coeff>>,

        #[cfg(not(feature = "cuda"))]
        pub advice_cosets: Vec<Polynomial<C::Scalar, ExtendedLagrangeCoeff>>,
    }

    let get_scalar_bits = |x: C::Scalar| {
        let repr = x.to_repr();
        let max_scalar_repr_ref: &[u8] = repr.as_ref();
        max_scalar_repr_ref
            .iter()
            .enumerate()
            .fold(0, |acc, (idx, v)| {
                if *v == 0 {
                    acc
                } else {
                    idx * 8 + 8 - v.leading_zeros() as usize
                }
            })
    };

    let find_max_scalar_bits = |x: &Vec<C::Scalar>| {
        get_scalar_bits(x.iter().fold(C::Scalar::zero(), |acc, x| acc.max(*x)))
    };

    let advice: Vec<Vec<Polynomial<C::Scalar, LagrangeCoeff>>> = circuits
        .iter()
        .zip(instances.iter())
        .map(|(circuit, instances)| {
            struct WitnessCollection<'a, F: Field> {
                k: u32,
                pub advice: Vec<Polynomial<F, LagrangeCoeff>>,
                instances: &'a [&'a [F]],
                usable_rows: RangeTo<usize>,
                _marker: std::marker::PhantomData<F>,
            }

            impl<'a, F: Field> Assignment<F> for WitnessCollection<'a, F> {
                fn enter_region<NR, N>(&mut self, _: N)
                where
                    NR: Into<String>,
                    N: FnOnce() -> NR,
                {
                    // Do nothing; we don't care about regions in this context.
                }

                fn exit_region(&mut self) {
                    // Do nothing; we don't care about regions in this context.
                }

                fn enable_selector<A, AR>(
                    &mut self,
                    _: A,
                    _: &Selector,
                    _: usize,
                ) -> Result<(), Error>
                where
                    A: FnOnce() -> AR,
                    AR: Into<String>,
                {
                    // We only care about advice columns here

                    Ok(())
                }

                fn query_instance(
                    &self,
                    column: Column<Instance>,
                    row: usize,
                ) -> Result<Option<F>, Error> {
                    if !self.usable_rows.contains(&row) {
                        return Err(Error::not_enough_rows_available(self.k));
                    }

                    self.instances
                        .get(column.index())
                        .and_then(|column| column.get(row))
                        .map(|v| Some(*v))
                        .ok_or(Error::BoundsFailure)
                }

                fn assign_advice<V, VR, A, AR>(
                    &mut self,
                    _: A,
                    column: Column<Advice>,
                    row: usize,
                    to: V,
                ) -> Result<(), Error>
                where
                    V: FnOnce() -> Result<VR, Error>,
                    VR: Into<Assigned<F>>,
                    A: FnOnce() -> AR,
                    AR: Into<String>,
                {
                    if !self.usable_rows.contains(&row) {
                        return Err(Error::not_enough_rows_available(self.k));
                    }

                    let assigned: Assigned<F> = to()?.into();
                    let v = if let Some(inv) = assigned.denominator() {
                        assigned.numerator() * inv.invert().unwrap()
                    } else {
                        assigned.numerator()
                    };

                    *self
                        .advice
                        .get_mut(column.index())
                        .and_then(|v| v.get_mut(row))
                        .ok_or(Error::BoundsFailure)? = v;

                    Ok(())
                }

                fn assign_fixed<V, VR, A, AR>(
                    &mut self,
                    _: A,
                    _: Column<Fixed>,
                    _: usize,
                    _: V,
                ) -> Result<(), Error>
                where
                    V: FnOnce() -> Result<VR, Error>,
                    VR: Into<Assigned<F>>,
                    A: FnOnce() -> AR,
                    AR: Into<String>,
                {
                    // We only care about advice columns here

                    Ok(())
                }

                fn copy(
                    &mut self,
                    _: Column<Any>,
                    _: usize,
                    _: Column<Any>,
                    _: usize,
                ) -> Result<(), Error> {
                    // We only care about advice columns here

                    Ok(())
                }

                fn fill_from_row(
                    &mut self,
                    _: Column<Fixed>,
                    _: usize,
                    _: Option<Assigned<F>>,
                ) -> Result<(), Error> {
                    Ok(())
                }

                fn push_namespace<NR, N>(&mut self, _: N)
                where
                    NR: Into<String>,
                    N: FnOnce() -> NR,
                {
                    // Do nothing; we don't care about namespaces in this context.
                }

                fn pop_namespace(&mut self, _: Option<String>) {
                    // Do nothing; we don't care about namespaces in this context.
                }
            }

            let unusable_rows_start = params.n as usize - (meta.blinding_factors() + 1);

            let mut witness = WitnessCollection {
                k: params.k,
                advice: vec![domain.empty_lagrange(); meta.num_advice_columns],
                instances,
                // The prover will not be allowed to assign values to advice
                // cells that exist within inactive rows, which include some
                // number of blinding factors and an extra row for use in the
                // permutation argument.
                usable_rows: ..unusable_rows_start,
                _marker: std::marker::PhantomData,
            };

            // Synthesize the circuit to obtain the witness and other information.
            ConcreteCircuit::FloorPlanner::synthesize(
                &mut witness,
                circuit,
                config.clone(),
                meta.constants.clone(),
            )
            .unwrap();

            let mut advice = witness.advice;

            let timer = start_timer!(|| "rng");
            advice.par_iter_mut().for_each(|advice| {
                for cell in &mut advice[unusable_rows_start..] {
                    *cell = C::Scalar::from(u16::rand(&mut OsRng) as u64);
                }
            });
            end_timer!(timer);

            let timer = start_timer!(|| "find column max bits");
            let max_bits = advice
                .par_iter()
                .map(|x| find_max_scalar_bits(&x.values))
                .collect::<Vec<_>>();
            end_timer!(timer);

            let timer = start_timer!(|| "commit_lagrange");
            let advice_commitments_projective: Vec<_> = advice
                .iter()
                .zip(max_bits.into_iter())
                .enumerate()
                .map(|(idx, (poly, max_bits))| {
                    GPU_GROUP_ID.set(idx);
                    params.commit_lagrange_with_bound(poly, max_bits)
                })
                .collect();
            end_timer!(timer);

            let timer = start_timer!(|| "advice_commitments_projective");
            let mut advice_commitments = vec![C::identity(); advice_commitments_projective.len()];
            C::Curve::batch_normalize(&advice_commitments_projective, &mut advice_commitments);
            let advice_commitments = advice_commitments;
            drop(advice_commitments_projective);
            end_timer!(timer);

            for commitment in &advice_commitments {
                transcript.write_point(*commitment).unwrap();
            }

            advice
        })
        .collect::<Vec<_>>();

    // Sample theta challenge for keeping lookup columns linearly independent
    let theta: ChallengeTheta<_> = transcript.squeeze_challenge_scalar();

    end_timer!(timer);
    let timer = start_timer!(|| format!("lookups {}", pk.vk.cs.lookups.len()));
    let (lookups, lookups_commitments): (
        Vec<Vec<Vec<lookup::prover::Permuted<C>>>>,
        Vec<Vec<Vec<[C; 2]>>>,
    ) = instance
        .iter()
        .zip(advice.iter())
        .map(|(instance, advice)| -> (Vec<_>, Vec<_>) {
            // Construct and commit to permuted values for each lookup
            let groups = *N_GPU * 2;
            let chunk_size = (pk.vk.cs.lookups.len() + groups - 1) / groups;
            pk.vk
                .cs
                .lookups
                .par_chunks(chunk_size)
                .map(|lookups| {
                    lookups
                        .into_par_iter()
                        .map(|lookup| {
                            lookup
                                .commit_permuted(
                                    pk,
                                    params,
                                    domain,
                                    theta,
                                    &advice,
                                    &pk.fixed_values,
                                    &instance.instance_values,
                                    &mut OsRng,
                                )
                                .unwrap()
                        })
                        .unzip()
                })
                .unzip()
        })
        .unzip();

    lookups_commitments.into_iter().for_each(|x| {
        x.iter().for_each(|x| {
            x.iter().for_each(|x| {
                transcript.write_point(x[0]).unwrap();
                transcript.write_point(x[1]).unwrap();
            })
        })
    });
    end_timer!(timer);

    // Sample beta challenge
    let beta: ChallengeBeta<_> = transcript.squeeze_challenge_scalar();
    // Sample gamma challenge
    let gamma: ChallengeGamma<_> = transcript.squeeze_challenge_scalar();

    let timer = start_timer!(|| "lookups commit product");
    let lookups: Vec<Vec<_>> = lookups
        .into_iter()
        .map(|lookups| {
            lookups
                .into_par_iter()
                .enumerate()
                .map(|(idx, lookups)| {
                    lookups
                        .into_par_iter()
                        .map(|lookup| {
                            lookup
                                .commit_product(pk, params, beta, gamma, idx % *N_GPU)
                                .unwrap()
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<Vec<_>>>()
        })
        .collect::<Vec<_>>();
    end_timer!(timer);

    let timer = start_timer!(|| "lookups add blinding value");
    let lookups: Vec<Vec<_>> = lookups
        .into_iter()
        .map(|lookups| {
            lookups
                .into_iter()
                .map(|lookups| {
                    lookups
                        .into_iter()
                        .map(|(l0, l1, mut z)| {
                            for _ in 0..pk.vk.cs.blinding_factors() {
                                z.push(C::Scalar::random(&mut rng))
                            }
                            (l0, l1, pk.vk.domain.lagrange_from_vec(z))
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<Vec<_>>>();
    end_timer!(timer);

    let (lookups, permutations) = std::thread::scope(|s| {
        let permutations = s.spawn(|| {
            // prepare permutation value.
            instance
                .iter()
                .zip(advice.iter())
                .map(|(instance, advice)| {
                    pk.vk.cs.permutation.commit(
                        params,
                        pk,
                        &pk.permutation,
                        &advice,
                        &pk.fixed_values,
                        &instance.instance_values,
                        beta,
                        gamma.clone(),
                        &mut OsRng,
                    )
                })
                .collect::<Result<Vec<_>, _>>()
                .unwrap()
        });

        let timer = start_timer!(|| "lookups msm");
        let lookups_z_commitments = lookups
            .iter()
            .flat_map(|lookups| {
                lookups
                    .into_iter()
                    .flat_map(|lookups| {
                        lookups
                            .into_iter()
                            .enumerate()
                            .map(|(idx, l)| {
                                GPU_GROUP_ID.set(idx);
                                params.commit_lagrange(&l.2).to_affine()
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        end_timer!(timer);

        let timer = start_timer!(|| "lookups fft");
        let lookups = lookups
            .into_iter()
            .map(|lookups| {
                lookups
                    .into_par_iter()
                    .enumerate()
                    .flat_map(|(idx, lookups)| {
                        GPU_GROUP_ID.set(idx);
                        lookups
                            .into_iter()
                            .map(
                                |(
                                    lookup_permuted_input,
                                    lookup_permuted_table,
                                    lookups_product,
                                )| {
                                    lookup::prover::Committed {
                                        permuted_input_poly: pk
                                            .vk
                                            .domain
                                            .lagrange_to_coeff_st(lookup_permuted_input),
                                        permuted_table_poly: pk
                                            .vk
                                            .domain
                                            .lagrange_to_coeff_st(lookup_permuted_table),
                                        product_poly: pk
                                            .vk
                                            .domain
                                            .lagrange_to_coeff_st(lookups_product),
                                    }
                                },
                            )
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        end_timer!(timer);

        let timer = start_timer!(|| "permutation commit");
        let permutations = permutations
            .join()
            .expect("permutations thread failed unexpectedly");

        let permutations = permutations
            .into_iter()
            .map(|permutations| {
                permutation::prover::Committed {
                    sets: permutations
                        .into_iter()
                        .map(|z| {
                            let permutation_product_commitment_projective =
                                params.commit_lagrange(&z);
                            let z = domain.lagrange_to_coeff(z);
                            let permutation_product_poly = z.clone();

                            let permutation_product_coset = domain.coeff_to_extended(z.clone());

                            let permutation_product_commitment =
                                permutation_product_commitment_projective.to_affine();

                            // Hash the permutation product commitment
                            transcript
                                .write_point(permutation_product_commitment)
                                .unwrap();

                            permutation::prover::CommittedSet {
                                permutation_product_poly,
                                permutation_product_coset,
                            }
                        })
                        .collect(),
                }
            })
            .collect::<Vec<_>>();
        end_timer!(timer);

        lookups_z_commitments
            .into_iter()
            .for_each(|lookups_z_commitment| transcript.write_point(lookups_z_commitment).unwrap());

        (lookups, permutations)
    });

    let timer = start_timer!(|| "vanishing commit");
    // Commit to the vanishing argument's random polynomial for blinding h(x_3)
    let vanishing = vanishing::Argument::commit(params, domain, rng, transcript)?;

    // Obtain challenge for keeping all separate gates linearly independent
    let y: ChallengeY<_> = transcript.squeeze_challenge_scalar();

    end_timer!(timer);
    let timer = start_timer!(|| "h_poly");
    // Evaluate the h(X) polynomial

    let advice = advice
        .into_iter()
        .map(|advice| {
            let timer = start_timer!(|| "lagrange_to_coeff_st");
            let advice_polys: Vec<_> = advice
                .into_par_iter()
                .enumerate()
                .map(|(idx, poly)| {
                    GPU_GROUP_ID.set(idx);
                    domain.lagrange_to_coeff_st(poly)
                })
                .collect();
            end_timer!(timer);

            #[cfg(not(feature = "cuda"))]
            let advice_cosets: Vec<_> = advice_polys
                .iter()
                .map(|poly| domain.coeff_to_extended(poly.clone()))
                .collect();

            AdviceSingle::<C> {
                advice_polys,
                #[cfg(not(feature = "cuda"))]
                advice_cosets,
            }
        })
        .collect::<Vec<_>>();

    #[cfg(feature = "cuda")]
    let h_poly = pk.ev.evaluate_h(
        pk,
        advice.iter().map(|a| &a.advice_polys).collect(),
        instance.iter().map(|i| &i.instance_polys).collect(),
        *y,
        *beta,
        *gamma,
        *theta,
        &lookups,
        &permutations,
    );

    #[cfg(not(feature = "cuda"))]
    let h_poly = pk.ev.evaluate_h(
        pk,
        advice.iter().map(|a| &a.advice_cosets).collect(),
        instance.iter().map(|i| &i.instance_cosets).collect(),
        *y,
        *beta,
        *gamma,
        *theta,
        &lookups,
        &permutations,
    );

    end_timer!(timer);
    let timer = start_timer!(|| "vanishing construct");
    // Construct the vanishing argument's h(X) commitments
    let vanishing = vanishing.construct(params, domain, h_poly, transcript)?;

    let x: ChallengeX<_> = transcript.squeeze_challenge_scalar();
    let xn = x.pow(&[params.n as u64, 0, 0, 0]);
    end_timer!(timer);

    let timer = start_timer!(|| "eval poly");

    let mut inputs = vec![];

    // Compute and hash instance evals for each circuit instance
    for instance in instance.iter() {
        // Evaluate polynomials at omega^i x
        meta.instance_queries.iter().for_each(|&(column, at)| {
            inputs.push((
                &instance.instance_polys[column.index()],
                domain.rotate_omega(*x, at),
            ))
        })
    }

    // Compute and hash advice evals for each circuit instance
    for advice in advice.iter() {
        // Evaluate polynomials at omega^i x
        meta.advice_queries.iter().for_each(|&(column, at)| {
            inputs.push((
                &advice.advice_polys[column.index()],
                domain.rotate_omega(*x, at),
            ))
        })
    }

    // Compute and hash fixed evals (shared across all circuit instances)
    meta.fixed_queries.iter().for_each(|&(column, at)| {
        inputs.push((&pk.fixed_polys[column.index()], domain.rotate_omega(*x, at)))
    });

    for eval in inputs
        .into_par_iter()
        .map(|(a, b)| eval_polynomial_st(a, b))
        .collect::<Vec<_>>()
    {
        transcript.write_scalar(eval)?;
    }

    end_timer!(timer);
    let timer = start_timer!(|| "eval poly vanishing");
    let vanishing = vanishing.evaluate(x, xn, domain, transcript)?;

    end_timer!(timer);
    let timer = start_timer!(|| "eval poly permutation");
    // Evaluate common permutation data
    pk.permutation.evaluate(x, transcript)?;

    // Evaluate the permutations, if any, at omega^i x.
    let permutations: Vec<permutation::prover::Evaluated<C>> = permutations
        .into_iter()
        .map(|permutation| -> Result<_, _> { permutation.construct().evaluate(pk, x, transcript) })
        .collect::<Result<Vec<_>, _>>()?;

    end_timer!(timer);

    let timer = start_timer!(|| "eval poly lookups");
    // Evaluate the lookups, if any, at omega^i x.
    let (lookups, evals): (
        Vec<Vec<lookup::prover::Evaluated<C>>>,
        Vec<Vec<Vec<C::ScalarExt>>>,
    ) = lookups
        .into_iter()
        .map(|lookups| lookups.into_par_iter().map(|p| p.evaluate(pk, x)).unzip())
        .unzip();
    evals.into_iter().for_each(|evals| {
        evals.into_iter().for_each(|evals| {
            evals
                .into_iter()
                .for_each(|eval| transcript.write_scalar(eval).unwrap())
        })
    });
    end_timer!(timer);

    let timer = start_timer!(|| "multi open");
    let instances = instance
        .iter()
        .zip(advice.iter())
        .zip(permutations.iter())
        .zip(lookups.iter())
        .flat_map(|(((instance, advice), permutation), lookups)| {
            iter::empty()
                .chain(
                    pk.vk
                        .cs
                        .instance_queries
                        .iter()
                        .map(move |&(column, at)| ProverQuery {
                            point: domain.rotate_omega(*x, at),
                            rotation: at,
                            poly: &instance.instance_polys[column.index()],
                        }),
                )
                .chain(
                    pk.vk
                        .cs
                        .advice_queries
                        .iter()
                        .map(move |&(column, at)| ProverQuery {
                            point: domain.rotate_omega(*x, at),
                            rotation: at,
                            poly: &advice.advice_polys[column.index()],
                        }),
                )
                .chain(permutation.open(pk, x))
                .chain(lookups.iter().flat_map(move |p| p.open(pk, x)).into_iter())
        })
        .chain(
            pk.vk
                .cs
                .fixed_queries
                .iter()
                .map(|&(column, at)| ProverQuery {
                    point: domain.rotate_omega(*x, at),
                    rotation: at,
                    poly: &pk.fixed_polys[column.index()],
                }),
        )
        .chain(pk.permutation.open(x))
        // We query the h(X) polynomial at x
        .chain(vanishing.open(x));

    let res = multiopen::create_proof(params, transcript, instances).map_err(|_| Error::Opening);
    end_timer!(timer);

    res
}
