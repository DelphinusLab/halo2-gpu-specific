use super::super::{
    circuit::Expression, ChallengeBeta, ChallengeGamma, ChallengeTheta, ChallengeX, Error,
    ProvingKey,
};
use super::Argument;
use crate::arithmetic::{batch_invert, eval_polynomial_st};
use crate::plonk::evaluation::{evaluate, evaluate_with_theta};
use crate::poly::Basis;
use crate::{
    arithmetic::{eval_polynomial, parallelize, BaseExt, CurveAffine, FieldExt},
    poly::{
        commitment::Params, multiopen::ProverQuery, Coeff, EvaluationDomain, ExtendedLagrangeCoeff,
        LagrangeCoeff, Polynomial, Rotation,
    },
    transcript::{EncodedChallenge, TranscriptWrite},
};
use ark_std::iterable::Iterable;
use ark_std::UniformRand;
use ark_std::{end_timer, start_timer};
use ff::PrimeField;
use group::{
    ff::{BatchInvert, Field},
    Curve,
};
use rand_core::RngCore;
use rayon::prelude::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator,
    IntoParallelRefMutIterator, ParallelIterator, ParallelSlice, ParallelSliceMut,
};
use std::any::TypeId;
use std::convert::TryInto;
use std::num::ParseIntError;
use std::ops::{Deref, Index};
use std::{
    collections::BTreeMap,
    iter,
    ops::{Mul, MulAssign},
};

#[derive(Debug)]
pub(in crate::plonk) struct CompressedInputExpressionSet<C: CurveAffine>(
    pub Vec<Polynomial<C::Scalar, LagrangeCoeff>>,
);

#[derive(Debug)]
pub(in crate::plonk) struct Compressed<C: CurveAffine> {
    compressed_table_expression: Polynomial<C::Scalar, LagrangeCoeff>,
    compressed_input_expression_sets: Vec<CompressedInputExpressionSet<C>>,
    pub(in crate::plonk) multiplicity_expression: Polynomial<C::Scalar, LagrangeCoeff>,
}

#[derive(Debug)]
pub(in crate::plonk) struct Committed<C: CurveAffine> {
    pub(in crate::plonk) multiplicity_poly: Polynomial<C::Scalar, Coeff>,
    // grand sum poly
    pub(in crate::plonk) z_poly_set: Vec<Polynomial<C::Scalar, Coeff>>,
}

pub(in crate::plonk) struct Evaluated<C: CurveAffine> {
    constructed: Committed<C>,
}

impl<F: FieldExt> Argument<F> {
    /// Given a Lookup with input expressions set [[A_0, A_1, ..., A_{m-1}]..]  and table expressions
    /// [S_0, S_1, ..., S_{m-1}], this method
    /// - constructs A_compressed_1 = \theta^{m-1} A_0 + theta^{m-2} A_1 + ... + \theta A_{m-2} + A_{m-1}
    ///              A_compressed_2 = ...
    /// - constructs S_compressed = \theta^{m-1} S_0 + theta^{m-2} S_1 + ... + \theta S_{m-2} + S_{m-1}
    /// - constructs Multiplicity expression to count input expressions set's multiplicity in table expression
    pub(in crate::plonk) fn compress<'a, C, R: RngCore>(
        &self,
        pk: &ProvingKey<C>,
        params: &Params<C>,
        theta: ChallengeTheta<C>,
        advice_values: &'a [Polynomial<C::Scalar, LagrangeCoeff>],
        fixed_values: &'a [Polynomial<C::Scalar, LagrangeCoeff>],
        instance_values: &'a [Polynomial<C::Scalar, LagrangeCoeff>],
        mut rng: R,
    ) -> Result<(Compressed<C>, C), Error>
    where
        C: CurveAffine<ScalarExt = F>,
        C::Curve: Mul<F, Output = C::Curve> + MulAssign<F>,
    {
        let blinding_factors = pk.vk.cs.blinding_factors();
        let usable_row = params.n as usize - blinding_factors - 1;
        let compress_expressions = |expressions: &[Expression<C::Scalar>]| {
            pk.vk.domain.lagrange_from_vec(evaluate_with_theta(
                expressions,
                params.n as usize,
                1,
                fixed_values,
                advice_values,
                instance_values,
                *theta,
            ))
        };

        // Get values of input expressions involved in the lookup and compress them
        let compressed_input_expression_sets = self
            .input_expressions_sets
            .iter()
            .map(|set| {
                set.0
                    .iter()
                    .map(|compress| compress_expressions(compress))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        // Get values of table expressions involved in the lookup and compress them
        let compressed_table_expression = compress_expressions(&self.table_expressions);

        // compute m(X)
        // for table has repeated elements case, mi will locate the only binary searched one.
        // e.g. table=[0,1,2,3,4,4,5], by binary search algorithm, it will locate the index=5, only count m[5]+=1,m[4]=0
        // refer to https://github.com/scroll-tech/halo2/halo2_proofs/src/plonk/mv_lookup
        let timer = start_timer!(|| "lookup table sort");
        let mut sorted_table_with_indices = compressed_table_expression
            .par_iter()
            .take(usable_row)
            .enumerate()
            .map(|(i, t)| (t, i))
            .collect::<Vec<_>>();
        sorted_table_with_indices.par_sort_by_key(|(&t, _)| t);
        end_timer!(timer);

        let timer = start_timer!(|| "lookup construct m(X) values");
        let mut multiplicity_values: Vec<F> =
            (0..params.n).into_par_iter().map(|_| F::zero()).collect();
        let num_threads = rayon::current_num_threads();
        let chunk_size = (usable_row + num_threads - 1) / num_threads;
        let res = compressed_input_expression_sets
            .iter()
            .flatten()
            .map(|input_expression| {
                input_expression.as_ref()[0..usable_row]
                    .par_chunks(chunk_size)
                    .map(|values| {
                        let mut map_count: BTreeMap<usize, usize> = BTreeMap::new();
                        let mut map_cache: BTreeMap<_, usize> = BTreeMap::new();
                        let mut hit_count = 0;
                        for fi in values {
                            let index = if let Some(idx) = map_cache.get(fi) {
                                hit_count += 1;
                                *idx
                            } else {
                                let index = sorted_table_with_indices
                                    .binary_search_by_key(&fi, |&(t, _)| t)
                                    .expect("logup binary_search_by_key should hit");
                                let index = sorted_table_with_indices[index].1;
                                map_cache.insert(fi, index);
                                index
                            };
                            map_count
                                .entry(index)
                                .and_modify(|count| *count += 1)
                                .or_insert(1);
                        }
                        (map_count, hit_count, values.len())
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let mut total_count = 0;
        let mut total_hit_count = 0;
        res.iter().for_each(|trees| {
            trees.iter().for_each(|(tree, hit_count, total)| {
                tree.iter().for_each(|(index, count)| {
                    multiplicity_values[*index] += F::from(*count as u64);
                    total_hit_count += *hit_count;
                    total_count += *total;
                })
            })
        });
        end_timer!(timer, || format!(
            "cache ratio:{}%",
            total_hit_count * 100 / total_count
        ));

        #[cfg(feature = "sanity-checks")]
        {
            let random = F::random(&mut rng);
            let res = (0..usable_row)
                .into_par_iter()
                .map(|r| {
                    let inputs =
                        compressed_input_expression_sets
                            .iter()
                            .fold(F::zero(), |acc, set| {
                                // ∑ 1/(f_i(X)+beta)
                                let sum = set.iter().fold(F::zero(), |acc, input| {
                                    acc + (input[r] + random).invert().unwrap()
                                });
                                acc + sum
                            });
                    // ∑ 1/(φ_i(X)) - m(X) / τ(X)))
                    inputs
                        - ((compressed_table_expression[r] + random).invert().unwrap()
                            * multiplicity_values[r])
                })
                .collect::<Vec<_>>();
            let last_z = res.iter().fold(F::zero(), |acc, v| acc + v);
            assert_eq!(last_z, F::zero());
        }

        // Closure to construct commitment to vector of values
        let commit_values = |values: &Polynomial<C::Scalar, LagrangeCoeff>, max_bits: usize| {
            params
                .commit_lagrange_with_bound(values, max_bits)
                .to_affine()
        };

        multiplicity_values.truncate(usable_row);

        let m_max_bits = expression_max_bits::<C>(&multiplicity_values[..]);
        multiplicity_values.extend(
            (0..(blinding_factors + 1)).map(|_| C::Scalar::from(u16::rand(&mut rng) as u64)),
        );
        assert_eq!(multiplicity_values.len(), params.n as usize);
        let multiplicity_expression = pk.vk.domain.lagrange_from_vec(multiplicity_values);

        // Commit to m expression
        let multiplicity_commitment = commit_values(&multiplicity_expression, m_max_bits);

        Ok((
            Compressed {
                compressed_table_expression,
                compressed_input_expression_sets: compressed_input_expression_sets
                    .into_iter()
                    .map(CompressedInputExpressionSet)
                    .collect::<Vec<_>>(),
                multiplicity_expression,
            },
            multiplicity_commitment,
        ))
    }
}

impl<C: CurveAffine> Compressed<C> {
    /// Given a Lookup with input expressions, table expressions, and the multiplicity
    /// expression, this method constructs the grand sum polynomial over the lookup.
    pub(in crate::plonk) fn commit_z(
        self,
        pk: &ProvingKey<C>,
        params: &Params<C>,
        beta: ChallengeBeta<C>,
    ) -> Result<(Polynomial<C::Scalar, LagrangeCoeff>, Vec<Vec<C::Scalar>>), Error> {
        let blinding_factors = pk.vk.cs.blinding_factors();
        /*
            φ_i(X) = f_i(X) + beta
            τ(X) = t(X) + beta
            LHS = τ(X) * Π(φ_i(X)) * (ϕ(gX) - ϕ(X))
            RHS = τ(X) * Π(φ_i(X)) * (∑ 1/(φ_i(X)) - m(X) / τ(X))))
        */

        let mut grand_sum_base = vec![C::Scalar::zero(); params.n as usize];

        for input in self.compressed_input_expression_sets[0].0.iter() {
            let mut fi_sum = vec![C::Scalar::zero(); params.n as usize];
            parallelize(&mut fi_sum, |sum_values, start| {
                for (sum, input_value) in sum_values.iter_mut().zip(input[start..].iter()) {
                    *sum += &(*beta + input_value);
                }
            });
            batch_invert(&mut fi_sum);
            parallelize(&mut grand_sum_base, |sum_values, start| {
                for (sum, fi) in sum_values.iter_mut().zip(fi_sum[start..].iter()) {
                    *sum += fi;
                }
            });
        }

        let mut table_sum = vec![C::Scalar::zero(); params.n as usize];
        parallelize(&mut table_sum, |sum_values, start| {
            for (sum, table) in sum_values
                .iter_mut()
                .zip(self.compressed_table_expression[start..].iter())
            {
                *sum += &(*beta + table);
            }
        });
        batch_invert(&mut table_sum);
        parallelize(&mut grand_sum_base, |sum_values, start| {
            for ((sum, table), m) in sum_values
                .iter_mut()
                .zip(table_sum[start..].iter())
                .zip(self.multiplicity_expression[start..].iter())
            {
                // (Σ 1/(φ_i(X)) - m(X) / τ(X))
                *sum = *sum - &(*table * m);
            }
        });
        /*
             omit table in extra input set
             φ_i(X) = f_i(X) + beta
             LHS = Π(φ_i(X)) * (ϕ(gX) - ϕ(X))
             RHS = Π(φ_i(X)) * (∑ 1/(φ_i(X)))
        */
        let mut grand_sum_extra_set = vec![
            vec![C::Scalar::zero(); params.n as usize];
            self.compressed_input_expression_sets.len() - 1
        ];
        for (i, set) in self
            .compressed_input_expression_sets
            .iter()
            .skip(1)
            .enumerate()
        {
            for input in set.0.iter() {
                let mut fi_sum = vec![C::Scalar::zero(); params.n as usize];
                parallelize(&mut fi_sum, |sum_values, start| {
                    for (sum, input) in sum_values.iter_mut().zip(input[start..].iter()) {
                        *sum += &(*beta + input);
                    }
                });
                batch_invert(&mut fi_sum);
                parallelize(&mut grand_sum_extra_set[i], |sum_values, start| {
                    for (sum, expression_value) in sum_values.iter_mut().zip(fi_sum[start..].iter())
                    {
                        *sum += expression_value;
                    }
                });
            }
        }

        // usable rows = n - blinding_rows -1
        // z[0]=zero, the last row for z is the usable row + 1
        // { |--- usable rows --|z[last]|-- blinding rows --| }
        let u = (params.n as usize) - (blinding_factors + 1);
        let mut last_z = C::Scalar::zero();
        let raw_zs = iter::once(&grand_sum_base)
            .chain(grand_sum_extra_set.iter())
            .map(|grand_sum| {
                let z = iter::once(&last_z)
                    .chain(grand_sum)
                    .scan(C::Scalar::zero(), |state, v| {
                        *state = *state + v;
                        Some(*state)
                    })
                    .take(params.n as usize - blinding_factors)
                    .collect::<Vec<_>>();
                last_z = z[u];
                z
            })
            .collect::<Vec<_>>();

        #[cfg(feature = "sanity-checks")]
        {
            // While in Lagrange basis, check that grand sum is correctly constructed
            /*
                 φ_i(X) = f_i(X) + α
                 τ(X) = t(X) + α
                 LHS = τ(X) * Π(φ_i(X)) * (ϕ(gX) - ϕ(X))
                 RHS = τ(X) * Π(φ_i(X)) * (∑ 1/(φ_i(X)) - m(X) / τ(X))))

                 extend inputs:
                 φ_i(X) = f_i(X) + α
                 LHS = Π(φ_i(X)) * (ϕ(gX) - ϕ(X))
                 RHS = Π(φ_i(X)) * (∑ 1/(φ_i(X)))

            */
            let z_first = raw_zs.first().unwrap();
            let z_last = raw_zs.last().unwrap();
            // l_0(X) * (z_first(X)) = 0
            assert_eq!(z_first[0], C::Scalar::zero());
            // l_last(X) * (z_last(X)) = 0
            assert_eq!(z_last[u], C::Scalar::zero());

            let mut inputs_set_sums = self
                .compressed_input_expression_sets
                .iter()
                .map(|input_expression_set| {
                    (0..u)
                        .into_par_iter()
                        .map(|i| {
                            input_expression_set
                                .0
                                .iter()
                                .map(|input| (input[i] + *beta).invert().unwrap())
                                .fold(C::Scalar::zero(), |acc, e| acc + e)
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();

            inputs_set_sums[0]
                .par_iter_mut()
                .zip(self.compressed_table_expression.par_iter())
                .zip(self.multiplicity_expression.par_iter())
                .for_each(|((phi, table), m)| {
                    *phi = *phi - *m * &(*beta + table).invert().unwrap()
                });

            inputs_set_sums
                .par_iter()
                .enumerate()
                .for_each(|(i, inputs_set_sum)| {
                    inputs_set_sum
                        .par_iter()
                        .enumerate()
                        .for_each(|(j, phi_sum)| {
                            assert_eq!(raw_zs[i][j + 1], *phi_sum + raw_zs[i][j]);
                        })
                });

            raw_zs
                .iter()
                .skip(1)
                .zip(raw_zs.iter())
                .for_each(|(z, z_pre)| assert_eq!(z[0], z_pre[u]))
        }

        Ok((self.multiplicity_expression, raw_zs))
    }
}

impl<C: CurveAffine> Committed<C> {
    pub(in crate::plonk) fn evaluate(
        self,
        pk: &ProvingKey<C>,
        x: ChallengeX<C>,
    ) -> (Evaluated<C>, Vec<C::ScalarExt>) {
        let domain = &pk.vk.domain;
        let blinding_factors = pk.vk.cs.blinding_factors();
        let x_next = domain.rotate_omega(*x, Rotation::next());
        let x_last = domain.rotate_omega(*x, Rotation(-((blinding_factors + 1) as i32)));

        let mut eval_set = vec![(&self.multiplicity_poly, *x)];

        let mut iter = self.z_poly_set.iter();
        while let Some(poly) = iter.next() {
            eval_set.push((poly, *x));
            eval_set.push((poly, x_next));
            if iter.len() > 0 {
                eval_set.push((poly, x_last));
            }
        }

        let evals = eval_set
            .into_par_iter()
            .map(|(a, b)| eval_polynomial_st(a, b))
            .collect();

        (Evaluated { constructed: self }, evals)
    }
}

impl<C: CurveAffine> Evaluated<C> {
    pub(in crate::plonk) fn open<'a>(
        &'a self,
        pk: &'a ProvingKey<C>,
        x: ChallengeX<C>,
    ) -> impl Iterator<Item = ProverQuery<'a, C>> + Clone {
        let blinding_factors = pk.vk.cs.blinding_factors();
        let x_next = pk.vk.domain.rotate_omega(*x, Rotation::next());
        let x_last = pk
            .vk
            .domain
            .rotate_omega(*x, Rotation(-((blinding_factors + 1) as i32)));

        iter::empty()
            // Open lookup m poly commitments at x
            .chain(Some(ProverQuery {
                point: *x,
                rotation: Rotation::cur(),
                poly: &self.constructed.multiplicity_poly,
            }))
            .chain(self.constructed.z_poly_set.iter().flat_map(move |z_poly| {
                iter::empty() // Open lookup grand sum commitments at x
                    .chain(Some(ProverQuery {
                        point: *x,
                        rotation: Rotation::cur(),
                        poly: z_poly,
                    }))
                    // Open lookup grand sum commitments at x_next
                    .chain(Some(ProverQuery {
                        point: x_next,
                        rotation: Rotation::next(),
                        poly: z_poly,
                    }))
            }))
            .chain(
                self.constructed
                    .z_poly_set
                    .iter()
                    .rev()
                    .skip(1)
                    .map(move |z_poly| ProverQuery {
                        point: x_last,
                        rotation: Rotation(-((blinding_factors + 1) as i32)),
                        poly: z_poly,
                    }),
            )
    }
}

fn expression_max_bits<C: CurveAffine>(expression: &[C::Scalar]) -> usize {
    let max_val = *expression.iter().reduce(|a, b| a.max(b)).unwrap();

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

    16.max(get_scalar_bits(max_val))
}
