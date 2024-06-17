use crate::helpers::AssignWitnessCollection;
use crate::plonk::range_check::RangeCheckRelAssigner;
use ark_std::iterable::Iterable;
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
use rayon::prelude::ParallelSliceMut;
use rayon::slice::ParallelSlice;
use std::collections::{BTreeMap, HashMap};
use std::env::var;
use std::fs::File;
use std::iter::FromIterator;
use std::ops::RangeTo;
use std::sync::atomic::AtomicUsize;
use std::sync::{Arc, Condvar, Mutex};
use std::time::Instant;
use std::{iter, sync::atomic::Ordering};

use super::range_check::{Argument, RangeCheckRel};
use super::{
    circuit::{
        Advice, Any, Assignment, Circuit, Column, ConstraintSystem, Fixed, FloorPlanner, Instance,
        Selector,
    },
    lookup, permutation, shuffle, vanishing, ChallengeBeta, ChallengeGamma, ChallengeTheta,
    ChallengeX, ChallengeY, Error, ProvingKey,
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
    pub static ref GPU_LOCK: Mutex<Vec<usize>> =
        Mutex::new(Vec::from_iter((0..*N_GPU).into_iter()));
    pub static ref GPU_COND_VAR: Condvar = Condvar::new();
}

#[derive(Debug)]
pub struct InstanceSingle<C: CurveAffine> {
    pub instance_values: Vec<Polynomial<C::Scalar, LagrangeCoeff>>,
    pub instance_polys: Vec<Polynomial<C::Scalar, Coeff>>,

    #[cfg(not(feature = "cuda"))]
    pub instance_cosets: Vec<Polynomial<C::Scalar, ExtendedLagrangeCoeff>>,
}

pub fn create_single_instances<C: CurveAffine, E: EncodedChallenge<C>, T: TranscriptWrite<C, E>>(
    params: &Params<C>,
    pk: &ProvingKey<C>,
    instances: &[&[&[C::Scalar]]],
    transcript: &mut T,
) -> Result<Vec<InstanceSingle<C>>, Error> {
    for instance in instances.iter() {
        if instance.len() != pk.vk.cs.num_instance_columns {
            return Err(Error::InvalidInstances);
        }
    }

    // Hash verification key into transcript
    pk.vk.hash_into(transcript)?;

    let domain = &pk.vk.domain;

    // Selector optimizations cannot be applied here; use the ConstraintSystem
    // from the verification key.
    let meta = &pk.vk.cs;

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
    Ok(instance)
}

fn sort<Scalar: FieldExt>(
    origin_advice: &[Scalar],
    sort_advice: &mut [Scalar],
    argument: &RangeCheckRel<Scalar>,
) {
    let mut field_to_u32_map = HashMap::<[u64; 4], u32>::with_capacity(
        argument.max.0 as usize - argument.min.0 as usize + 1,
    );
    let mut count_map = vec![0; argument.max.0 as usize - argument.min.0 as usize + 1];

    let mut start = argument.min.1;
    for i in argument.min.0..=argument.max.0 {
        let raw = unsafe { std::mem::transmute::<_, &[u64; 4]>(&start) };

        field_to_u32_map.insert(*raw, i);
        start += Scalar::one();
    }

    for value in &*origin_advice {
        let v = unsafe { std::mem::transmute::<_, &[u64; 4]>(value) };
        let v = field_to_u32_map.get(v).unwrap();

        count_map[(*v - argument.min.0 as u32) as usize] += 1;
    }

    let mut start = argument.min.1;

    let mut offset = 0;
    count_map.into_iter().for_each(|cnt| {
        for _ in 0..cnt {
            sort_advice[offset] = start;
            offset += 1;
        }

        start += Scalar::one();
    });
}

/// This creates a proof for the provided `circuit` when given the public
/// parameters `params` and the proving key [`ProvingKey`] that was
/// generated previously for the same circuit. The provided `instances`
/// are zero-padded internally.
pub fn create_proof_ext<
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
    use_gwc: bool,
) -> Result<(), Error> {
    let domain = &pk.vk.domain;

    let timer = start_timer!(|| "instance");
    let instance = create_single_instances(params, pk, instances, transcript)?;
    end_timer!(timer);

    let meta = &pk.vk.cs;

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
            let unusable_rows_start = params.n as usize - (meta.blinding_factors() + 1);

            let timer = start_timer!(|| "prepare collection");
            let mut advice: Vec<_> = (0..meta.num_advice_columns)
                .into_par_iter()
                .map(|_| domain.empty_lagrange())
                .collect();

            generate_advice_from_synthesize(
                params,
                pk,
                circuit,
                instances,
                &advice
                    .iter_mut()
                    .map(|x| (&mut x.values[..]) as *mut [_])
                    .collect::<Vec<_>>()[..],
            );
            end_timer!(timer);

            let named = &pk.vk.cs.named_advices;

            let timer = start_timer!(|| "rng");
            advice.par_iter_mut().enumerate().for_each(|(i, advice)| {
                if named.iter().find(|n| n.1 as usize == i).is_none() {
                    for cell in &mut advice[unusable_rows_start..] {
                        *cell = C::Scalar::from(u16::rand(&mut OsRng) as u64);
                    }
                }
            });
            end_timer!(timer);

            let timer = start_timer!(|| "commit_lagrange");
            let advice_commitments_projective: Vec<_> = advice
                .par_iter()
                .map(|advice| {
                    let max_bits = find_max_scalar_bits(&advice.values);
                    params.commit_lagrange_with_bound(advice, max_bits)
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
    let (lookups, lookups_commitments): (Vec<Vec<lookup::prover::Permuted<C>>>, Vec<Vec<[C; 2]>>) =
        instance
            .iter()
            .zip(advice.iter())
            .map(|(instance, advice)| -> (Vec<_>, Vec<_>) {
                pk.vk
                    .cs
                    .lookups
                    .par_iter()
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
            .unzip();

    lookups_commitments.into_iter().for_each(|x| {
        x.iter().for_each(|x| {
            transcript.write_point(x[0]).unwrap();
            transcript.write_point(x[1]).unwrap();
        })
    });
    end_timer!(timer);

    let shuffle_groups = pk.vk.cs.shuffles.group(pk.vk.cs.degree());
    let timer = start_timer!(|| format!(
        "total shuffles {},groups:{}",
        pk.vk.cs.shuffles.0.len(),
        shuffle_groups.len()
    ));
    let shuffles: Vec<Vec<shuffle::prover::Compressed<C>>> = instance
        .iter()
        .zip(advice.iter())
        .map(|(instance, advice)| -> Vec<_> {
            shuffle_groups
                .par_iter()
                .map(|shuffle| {
                    shuffle
                        .compress(
                            pk,
                            params,
                            theta,
                            &advice,
                            &pk.fixed_values,
                            &instance.instance_values,
                        )
                        .unwrap()
                })
                .collect()
        })
        .collect();

    end_timer!(timer);

    // Sample beta challenge
    let beta: ChallengeBeta<_> = transcript.squeeze_challenge_scalar();
    // Sample gamma challenge
    let gamma: ChallengeGamma<_> = transcript.squeeze_challenge_scalar();

    let (lookups, shuffles, permutations) = std::thread::scope(|s| {
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

        let timer = start_timer!(|| "lookups commit product");
        let lookups: Vec<Vec<_>> = lookups
            .into_iter()
            .map(|lookups| {
                lookups
                    .into_par_iter()
                    .map(|lookup| lookup.commit_product(pk, params, beta, gamma).unwrap())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        end_timer!(timer);

        let timer = start_timer!(|| "lookups add blinding value");
        let lookups: Vec<Vec<_>> = lookups
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
            .collect::<Vec<Vec<_>>>();
        end_timer!(timer);

        let timer = start_timer!(|| "lookups msm and fft");
        let (lookups_z_commitments, lookups): (Vec<Vec<_>>, Vec<Vec<_>>) = lookups
            .into_iter()
            .map(|lookups| {
                lookups
                    .into_par_iter()
                    .map(|l| {
                        let (product_poly, c) = params.commit_lagrange_and_ifft(
                            l.2,
                            &pk.vk.domain.get_omega_inv(),
                            &pk.vk.domain.ifft_divisor,
                        );
                        let c = c.to_affine();
                        (
                            c,
                            lookup::prover::Committed {
                                permuted_input_poly: pk.vk.domain.lagrange_to_coeff_st(l.0),
                                permuted_table_poly: pk.vk.domain.lagrange_to_coeff_st(l.1),
                                product_poly,
                            },
                        )
                    })
                    .collect::<Vec<_>>()
                    .into_iter()
                    .unzip()
            })
            .unzip();
        end_timer!(timer);

        let timer = start_timer!(|| "shuffles commit product");
        let shuffles: Vec<Vec<_>> = shuffles
            .into_iter()
            .map(|shuffles| {
                shuffles
                    .into_par_iter()
                    .map(|shuffle| shuffle.commit_product(pk, params, beta).unwrap())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        end_timer!(timer);

        let timer = start_timer!(|| "shuffles add blinding value");
        let shuffles: Vec<Vec<_>> = shuffles
            .into_iter()
            .map(|shuffles| {
                shuffles
                    .into_iter()
                    .map(|mut z| {
                        for _ in 0..pk.vk.cs.blinding_factors() {
                            z.push(C::Scalar::random(&mut rng))
                        }
                        assert_eq!(z.len(), params.n as usize);
                        pk.vk.domain.lagrange_from_vec(z)
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<Vec<_>>>();
        end_timer!(timer);

        let timer = start_timer!(|| "shuffles msm and fft");
        let (shuffles_z_commitments, shuffles): (Vec<Vec<_>>, Vec<Vec<_>>) = shuffles
            .into_iter()
            .map(|shuffles| {
                shuffles
                    .into_par_iter()
                    .map(|l| {
                        let (product_poly, c) = params.commit_lagrange_and_ifft(
                            l,
                            &pk.vk.domain.get_omega_inv(),
                            &pk.vk.domain.ifft_divisor,
                        );
                        let c = c.to_affine();
                        (c, shuffle::prover::Committed { product_poly })
                    })
                    .collect::<Vec<_>>()
                    .into_iter()
                    .unzip()
            })
            .unzip();
        end_timer!(timer);

        let timer = start_timer!(|| "permutation commit");
        let permutations = permutations
            .join()
            .expect("permutations thread failed unexpectedly");

        let permutations: Vec<_> = permutations
            .into_iter()
            .map(|permutations| {
                let (c, sets): (Vec<_>, _) = permutations
                    .into_par_iter()
                    .map(|z| {
                        let (permutation_product_poly, permutation_product_commitment_projective) =
                            params.commit_lagrange_and_ifft(
                                z,
                                &pk.vk.domain.get_omega_inv(),
                                &pk.vk.domain.ifft_divisor,
                            );

                        #[cfg(not(feature = "cuda"))]
                        let permutation_product_coset =
                            domain.coeff_to_extended(permutation_product_poly.clone());

                        let permutation_product_commitment =
                            permutation_product_commitment_projective.to_affine();

                        (
                            permutation_product_commitment,
                            permutation::prover::CommittedSet {
                                permutation_product_poly,
                                #[cfg(not(feature = "cuda"))]
                                permutation_product_coset,
                            },
                        )
                    })
                    .unzip();
                (c, permutation::prover::Committed { sets })
            })
            .collect();

        for (cl, _) in permutations.iter() {
            for c in cl {
                transcript.write_point(*c).unwrap();
            }
        }

        let permutations: Vec<_> = permutations.into_iter().map(|x| x.1).collect();
        end_timer!(timer);

        lookups_z_commitments
            .into_iter()
            .for_each(|lookups_z_commitments| {
                lookups_z_commitments
                    .into_iter()
                    .for_each(|lookups_z_commitment| {
                        transcript.write_point(lookups_z_commitment).unwrap()
                    })
            });
        shuffles_z_commitments
            .into_iter()
            .for_each(|shuffles_z_commitments| {
                shuffles_z_commitments
                    .into_iter()
                    .for_each(|shuffles_z_commitment| {
                        transcript.write_point(shuffles_z_commitment).unwrap()
                    })
            });

        (lookups, shuffles, permutations)
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
                .map(|poly| domain.lagrange_to_coeff_st(poly))
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
        &shuffles,
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
        &shuffles,
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

    let timer = start_timer!(|| "eval poly shuffles");
    // Evaluate the shuffles, if any, at omega^i x.
    let (shuffles, evals): (
        Vec<Vec<shuffle::prover::Evaluated<C>>>,
        Vec<Vec<Vec<C::ScalarExt>>>,
    ) = shuffles
        .into_iter()
        .map(|shuffles| shuffles.into_par_iter().map(|s| s.evaluate(pk, x)).unzip())
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
        .zip(shuffles.iter())
        .flat_map(|((((instance, advice), permutation), lookups), shuffles)| {
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
                .chain(shuffles.iter().flat_map(move |p| p.open(pk, x)).into_iter())
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

    let res = if use_gwc {
        multiopen::gwc::create_proof(params, transcript, instances).map_err(|_| Error::Opening)
    } else {
        multiopen::shplonk::create_proof(params, transcript, instances).map_err(|_| Error::Opening)
    };
    end_timer!(timer);

    res
}

/// This creates a proof for the provided `circuit` when given the public
/// parameters `params` and the proving key [`ProvingKey`] that was
/// generated previously for the same circuit. The provided `instances`
/// are zero-padded internally.
pub fn create_proof_with_shplonk<
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
    rng: R,
    transcript: &mut T,
) -> Result<(), Error> {
    create_proof_ext(params, pk, circuits, instances, rng, transcript, false)
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
    rng: R,
    transcript: &mut T,
) -> Result<(), Error> {
    create_proof_ext(params, pk, circuits, instances, rng, transcript, true)
}

/// generate and write witness to files
pub fn create_witness<C: CurveAffine, ConcreteCircuit: Circuit<C::Scalar>>(
    params: &Params<C>,
    pk: &ProvingKey<C>,
    circuit: &ConcreteCircuit,
    instances: &[&[C::Scalar]],
    fd: &mut File,
) -> Result<(), Error> {
    let meta = &pk.vk.cs;
    let unusable_rows_start = params.n as usize - (meta.blinding_factors() + 1);
    AssignWitnessCollection::store_witness(
        params,
        pk,
        instances,
        unusable_rows_start,
        circuit,
        fd,
    )?;
    Ok(())
}

/// create_proof based on vkey and witness
pub fn create_proof_from_witness<
    C: CurveAffine,
    E: EncodedChallenge<C>,
    R: RngCore,
    T: TranscriptWrite<C, E>,
>(
    params: &Params<C>,
    pk: &ProvingKey<C>,
    instances: &[&[&[C::Scalar]]],
    mut rng: R,
    transcript: &mut T,
    fd: &mut File,
    use_gwc: bool,
) -> Result<(), Error> {
    let meta = &pk.vk.cs;
    let domain = &pk.vk.domain;

    let timer = start_timer!(|| "create single instances");
    let instance = create_single_instances(params, pk, instances, transcript)?;

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

    let advice: Vec<Vec<Polynomial<C::Scalar, LagrangeCoeff>>> = instances
        .iter()
        .map(|_| -> Vec<Polynomial<C::Scalar, LagrangeCoeff>> {
            let unusable_rows_start = params.n as usize - (meta.blinding_factors() + 1);

            let mut advice = AssignWitnessCollection::fetch_witness(params, fd)
                .expect("fetch witness should not fail");

            let timer = start_timer!(|| "rng");
            advice.par_iter_mut().for_each(|advice| {
                for cell in &mut advice[unusable_rows_start..] {
                    *cell = C::Scalar::from(u16::rand(&mut OsRng) as u64);
                }
            });
            end_timer!(timer);

            let timer = start_timer!(|| "commit_lagrange");
            let advice_commitments_projective: Vec<_> = advice
                .par_iter()
                .map(|advice| {
                    let max_bits = find_max_scalar_bits(&advice.values);
                    params.commit_lagrange_with_bound(advice, max_bits)
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
    let (lookups, lookups_commitments): (Vec<Vec<lookup::prover::Permuted<C>>>, Vec<Vec<[C; 2]>>) =
        instance
            .iter()
            .zip(advice.iter())
            .map(|(instance, advice)| -> (Vec<_>, Vec<_>) {
                pk.vk
                    .cs
                    .lookups
                    .par_iter()
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
            .unzip();

    lookups_commitments.into_iter().for_each(|x| {
        x.iter().for_each(|x| {
            transcript.write_point(x[0]).unwrap();
            transcript.write_point(x[1]).unwrap();
        })
    });
    end_timer!(timer);

    let shuffle_groups = pk.vk.cs.shuffles.group(pk.vk.cs.degree());
    let timer = start_timer!(|| format!(
        "total shuffles {}, groups {}",
        pk.vk.cs.shuffles.0.len(),
        shuffle_groups.len()
    ));
    let shuffles: Vec<Vec<shuffle::prover::Compressed<C>>> = instance
        .iter()
        .zip(advice.iter())
        .map(|(instance, advice)| -> Vec<_> {
            shuffle_groups
                .par_iter()
                .map(|shuffle| {
                    shuffle
                        .compress(
                            pk,
                            params,
                            theta,
                            &advice,
                            &pk.fixed_values,
                            &instance.instance_values,
                        )
                        .unwrap()
                })
                .collect()
        })
        .collect();

    end_timer!(timer);

    // Sample beta challenge
    let beta: ChallengeBeta<_> = transcript.squeeze_challenge_scalar();
    // Sample gamma challenge
    let gamma: ChallengeGamma<_> = transcript.squeeze_challenge_scalar();

    let (lookups, shuffles, permutations) = std::thread::scope(|s| {
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

        let timer = start_timer!(|| "lookups commit product");
        let lookups: Vec<Vec<_>> = lookups
            .into_iter()
            .map(|lookups| {
                lookups
                    .into_par_iter()
                    .map(|lookup| lookup.commit_product(pk, params, beta, gamma).unwrap())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        end_timer!(timer);

        let timer = start_timer!(|| "lookups add blinding value");
        let lookups: Vec<Vec<_>> = lookups
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
            .collect::<Vec<Vec<_>>>();
        end_timer!(timer);

        let timer = start_timer!(|| "lookups msm and fft");
        let (lookups_z_commitments, lookups): (Vec<Vec<_>>, Vec<Vec<_>>) = lookups
            .into_iter()
            .map(|lookups| {
                lookups
                    .into_par_iter()
                    .map(|l| {
                        let (product_poly, c) = params.commit_lagrange_and_ifft(
                            l.2,
                            &pk.vk.domain.get_omega_inv(),
                            &pk.vk.domain.ifft_divisor,
                        );
                        let c = c.to_affine();
                        (
                            c,
                            lookup::prover::Committed {
                                permuted_input_poly: pk.vk.domain.lagrange_to_coeff_st(l.0),
                                permuted_table_poly: pk.vk.domain.lagrange_to_coeff_st(l.1),
                                product_poly,
                            },
                        )
                    })
                    .collect::<Vec<_>>()
                    .into_iter()
                    .unzip()
            })
            .unzip();
        end_timer!(timer);

        let timer = start_timer!(|| "shuffles commit product");
        let shuffles: Vec<Vec<_>> = shuffles
            .into_iter()
            .map(|shuffles| {
                shuffles
                    .into_par_iter()
                    .map(|shuffle| shuffle.commit_product(pk, params, beta).unwrap())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        end_timer!(timer);

        let timer = start_timer!(|| "shuffles add blinding value");
        let shuffles: Vec<Vec<_>> = shuffles
            .into_iter()
            .map(|shuffles| {
                shuffles
                    .into_iter()
                    .map(|mut z| {
                        for _ in 0..pk.vk.cs.blinding_factors() {
                            z.push(C::Scalar::random(&mut rng))
                        }
                        assert_eq!(z.len(), params.n as usize);
                        pk.vk.domain.lagrange_from_vec(z)
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<Vec<_>>>();
        end_timer!(timer);

        let timer = start_timer!(|| "shuffles msm and fft");
        let (shuffles_z_commitments, shuffles): (Vec<Vec<_>>, Vec<Vec<_>>) = shuffles
            .into_iter()
            .map(|shuffles| {
                shuffles
                    .into_par_iter()
                    .map(|l| {
                        let (product_poly, c) = params.commit_lagrange_and_ifft(
                            l,
                            &pk.vk.domain.get_omega_inv(),
                            &pk.vk.domain.ifft_divisor,
                        );
                        let c = c.to_affine();
                        (c, shuffle::prover::Committed { product_poly })
                    })
                    .collect::<Vec<_>>()
                    .into_iter()
                    .unzip()
            })
            .unzip();
        end_timer!(timer);

        let timer = start_timer!(|| "permutation commit");
        let permutations = permutations
            .join()
            .expect("permutations thread failed unexpectedly");

        let permutations: Vec<_> = permutations
            .into_iter()
            .map(|permutations| {
                let (c, sets): (Vec<_>, _) = permutations
                    .into_par_iter()
                    .map(|z| {
                        let (permutation_product_poly, permutation_product_commitment_projective) =
                            params.commit_lagrange_and_ifft(
                                z,
                                &pk.vk.domain.get_omega_inv(),
                                &pk.vk.domain.ifft_divisor,
                            );

                        #[cfg(not(feature = "cuda"))]
                        let permutation_product_coset =
                            domain.coeff_to_extended(permutation_product_poly.clone());

                        let permutation_product_commitment =
                            permutation_product_commitment_projective.to_affine();

                        (
                            permutation_product_commitment,
                            permutation::prover::CommittedSet {
                                permutation_product_poly,
                                #[cfg(not(feature = "cuda"))]
                                permutation_product_coset,
                            },
                        )
                    })
                    .unzip();
                (c, permutation::prover::Committed { sets })
            })
            .collect();

        for (cl, _) in permutations.iter() {
            for c in cl {
                transcript.write_point(*c).unwrap();
            }
        }

        let permutations: Vec<_> = permutations.into_iter().map(|x| x.1).collect();
        end_timer!(timer);

        lookups_z_commitments
            .into_iter()
            .for_each(|lookups_z_commitments| {
                lookups_z_commitments
                    .into_iter()
                    .for_each(|lookups_z_commitment| {
                        transcript.write_point(lookups_z_commitment).unwrap()
                    })
            });
        shuffles_z_commitments
            .into_iter()
            .for_each(|shuffles_z_commitments| {
                shuffles_z_commitments
                    .into_iter()
                    .for_each(|shuffles_z_commitment| {
                        transcript.write_point(shuffles_z_commitment).unwrap()
                    })
            });

        (lookups, shuffles, permutations)
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
                .map(|poly| domain.lagrange_to_coeff_st(poly))
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
        &shuffles,
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
        &shuffles,
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

    let timer = start_timer!(|| "eval poly shuffles");
    // Evaluate the shuffles, if any, at omega^i x.
    let (shuffles, evals): (
        Vec<Vec<shuffle::prover::Evaluated<C>>>,
        Vec<Vec<Vec<C::ScalarExt>>>,
    ) = shuffles
        .into_iter()
        .map(|shuffles| shuffles.into_par_iter().map(|s| s.evaluate(pk, x)).unzip())
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
        .zip(shuffles.iter())
        .flat_map(|((((instance, advice), permutation), lookups), shuffles)| {
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
                .chain(shuffles.iter().flat_map(move |p| p.open(pk, x)).into_iter())
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

    let res = if use_gwc {
        multiopen::gwc::create_proof(params, transcript, instances).map_err(|_| Error::Opening)
    } else {
        multiopen::shplonk::create_proof(params, transcript, instances).map_err(|_| Error::Opening)
    };
    end_timer!(timer);

    res
}

pub fn generate_advice_from_synthesize<'a, C: CurveAffine, ConcreteCircuit: Circuit<C::Scalar>>(
    params: &'a Params<C>,
    pk: &'a ProvingKey<C>,
    circuit: &'a ConcreteCircuit,
    instances: &'a [&'a [C::Scalar]],
    advices: &'a [*mut [C::Scalar]],
) {
    use Assigned;
    use Column;
    use Error;
    use FloorPlanner;

    let mut meta = ConstraintSystem::default();
    let config = ConcreteCircuit::configure(&mut meta);

    let meta = &pk.vk.cs;

    let first_unassigned_offset = (0..meta.num_advice_columns)
        .into_iter()
        .map(|_| Arc::new(AtomicUsize::new(0)))
        .collect::<Vec<_>>();

    #[derive(Clone)]
    struct WitnessCollection<'a, F: Field> {
        pub advice: &'a [*mut [F]],
        first_unassigned_offset: &'a [Arc<AtomicUsize>],
        instances: &'a [&'a [F]],
        usable_rows: core::ops::RangeTo<usize>,
        _marker: std::marker::PhantomData<F>,
    }

    impl<'a, F: Field> Assignment<F> for WitnessCollection<'a, F> {
        fn is_in_prove_mode(&self) -> bool {
            true
        }

        fn enter_region<NR, N>(&self, _: N)
        where
            NR: Into<String>,
            N: FnOnce() -> NR,
        {
        }
        fn exit_region(&self) {}

        fn enable_selector<A, AR>(&self, _: A, _: &Selector, _: usize) -> Result<(), Error>
        where
            A: FnOnce() -> AR,
            AR: Into<String>,
        {
            Ok(())
        }

        fn query_instance(&self, column: Column<Instance>, row: usize) -> Result<Option<F>, Error> {
            if !self.usable_rows.contains(&row) {
                assert!(false)
            }

            Ok(self
                .instances
                .get(column.index())
                .and_then(|column| column.get(row))
                .map(|v| Some(*v))
                .unwrap())
        }

        fn assign_advice<V, VR, A, AR>(
            &self,
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
                assert!(false)
            }

            let assigned: Assigned<F> = to()?.into();
            let v = if let Some(inv) = assigned.denominator() {
                assigned.numerator() * inv.invert().unwrap()
            } else {
                assigned.numerator()
            };

            *self
                .advice
                .get(column.index())
                .and_then(|v| unsafe { (*v).as_mut().unwrap() }.get_mut(row))
                .ok_or(Error::BoundsFailure)? = v;

            self.first_unassigned_offset
                .get(column.index())
                .and_then(|offset| Some(offset.fetch_max(row + 1, Ordering::Relaxed)))
                .unwrap();

            Ok(())
        }

        fn assign_fixed<V, VR, A, AR>(
            &self,
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

        fn copy(&self, _: Column<Any>, _: usize, _: Column<Any>, _: usize) -> Result<(), Error> {
            // We only care about advice columns here

            Ok(())
        }

        fn fill_from_row(
            &self,
            _: Column<Fixed>,
            _: usize,
            _: Option<Assigned<F>>,
        ) -> Result<(), Error> {
            Ok(())
        }

        fn push_namespace<NR, N>(&self, _: N)
        where
            NR: Into<String>,
            N: FnOnce() -> NR,
        {
            // Do nothing; we don't care about namespaces in this context.
        }

        fn pop_namespace(&self, _: Option<String>) {
            // Do nothing; we don't care about namespaces in this context.
        }
    }

    let unusable_rows_start = params.n as usize - (meta.blinding_factors() + 1);

    let mut witness = WitnessCollection {
        advice: advices,
        first_unassigned_offset: &first_unassigned_offset[..],
        instances,
        // The prover will not be allowed to assign values to advice
        // cells that exist within inactive rows, which include some
        // number of blinding factors and an extra row for use in the
        // permutation argument.
        usable_rows: ..unusable_rows_start,
        _marker: std::marker::PhantomData,
    };

    let timer = start_timer!(|| "synthesize");
    // Synthesize the circuit to obtain the witness and other information.
    ConcreteCircuit::FloorPlanner::synthesize(
        &mut witness,
        circuit,
        config.clone(),
        meta.constants.clone(),
    )
    .unwrap();
    end_timer!(timer);

    let timer = start_timer!(|| "synthesize sort range check col");
    {
        let last_active_offset =
            pk.get_vk().domain.n - (pk.get_vk().cs.blinding_factors() as u64 + 1) - 1;

        // Assign all values within range to unused cells.
        pk.vk.cs.range_check.0.iter().for_each(|argument| {
            let origin_column_index = argument.origin.index;

            let first_unassigned_offset = first_unassigned_offset
                .get(origin_column_index)
                .unwrap()
                .load(Ordering::Relaxed);

            let advice = unsafe {
                advices
                    .get(argument.origin.index)
                    .unwrap()
                    .as_mut()
                    .unwrap()
            };

            let mut offset = last_active_offset as usize;

            let assigner: RangeCheckRelAssigner = argument.into();
            let mut iter = assigner.into_iter();

            while let Some(value) = iter.next() {
                *advice.get_mut(offset).unwrap() = C::ScalarExt::from(value as u64);
                offset -= 1;
            }

            assert!(first_unassigned_offset <= offset);
        });

        let timer = start_timer!(|| "sort range check");
        {
            let mut advices = advices
                .iter()
                .map(|advice| unsafe { advice.as_mut() }.unwrap())
                .collect::<Vec<_>>();

            let advices = advices
                .iter_mut()
                .enumerate()
                .filter_map(|(column_index, advice)| {
                    pk.vk
                        .cs
                        .range_check
                        .0
                        .iter()
                        .find(|argument| {
                            argument.origin.index == column_index
                                || argument.sort.index == column_index
                        })
                        .map(|argument| (argument, column_index, advice))
                })
                .collect::<Vec<_>>();

            advices.into_par_iter().chunks(2).for_each(|mut chunks| {
                let (range_check_rel, column_index_a, advice_a) = chunks.pop().unwrap();
                let (_, _, advice_b) = chunks.pop().unwrap();

                let (origin_advice, sort_advice) = if column_index_a == range_check_rel.origin.index
                {
                    (
                        &(advice_a)[0..unusable_rows_start],
                        &mut (advice_b)[0..unusable_rows_start],
                    )
                } else {
                    (
                        &(advice_b)[0..unusable_rows_start],
                        &mut (advice_a)[0..unusable_rows_start],
                    )
                };

                sort::<C::ScalarExt>(origin_advice, sort_advice, range_check_rel);
            });
            end_timer!(timer);
        };
    }
    end_timer!(timer);
}
