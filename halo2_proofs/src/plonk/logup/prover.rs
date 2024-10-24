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
    IntoParallelRefMutIterator, ParallelIterator, ParallelSliceMut,
};
use std::any::TypeId;
use std::convert::TryInto;
use std::num::ParseIntError;
use std::ops::Index;
use std::{
    collections::BTreeMap,
    iter,
    ops::{Mul, MulAssign},
};

#[derive(Debug)]
pub(in crate::plonk) struct Compressed<C: CurveAffine> {
    compressed_table_expression: Polynomial<C::Scalar, LagrangeCoeff>,
    compressed_input_expression_set: Vec<Polynomial<C::Scalar, LagrangeCoeff>>,
    pub(in crate::plonk) multiplicity_expression: Polynomial<C::Scalar, LagrangeCoeff>,
}

#[derive(Debug)]
pub(in crate::plonk) struct Committed<C: CurveAffine> {
    pub(in crate::plonk) grand_sum_poly: Polynomial<C::Scalar, Coeff>,
    pub(in crate::plonk) multiplicity_poly: Polynomial<C::Scalar, Coeff>,
}

pub(in crate::plonk) struct Evaluated<C: CurveAffine> {
    constructed: Committed<C>,
}

impl<F: FieldExt> Argument<F> {
    /// Given a Lookup with input expressions set [[A_0, A_1, ..., A_{m-1}]..]  and table expressions
    /// [S_0, S_1, ..., S_{m-1}], this method
    /// - constructs A_compressed_1 = \theta^{m-1} A_0 + theta^{m-2} A_1 + ... + \theta A_{m-2} + A_{m-1}
    ///   A_compressed_2...
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
        // Closure to get values of expressions and compress them
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
        let compressed_input_expression_set = self
            .input_expressions_set
            .0
            .iter()
            .map(|compress| compress_expressions(compress))
            .collect::<Vec<_>>();

        // Get values of table expressions involved in the lookup and compress them
        let compressed_table_expression = compress_expressions(&self.table_expressions);

        // compute m(X)
        // for table has repeated elements case, mi will locate the only binary searched one.
        // e.g. table=[0,1,2,3,4,4,5], by binary search algorithm, it will locate the index=5, only count m[5]+=1,m[4]=0
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
        let mut multiplicity_values: Vec<F> = {
            use std::sync::atomic::{AtomicU64, Ordering};
            let m_values: Vec<AtomicU64> = (0..params.n).map(|_| AtomicU64::new(0)).collect();
            //todo check ConstraintSystemFailure report to outside by try_for_each?
            for compressed_input_expression in compressed_input_expression_set.iter() {
                compressed_input_expression
                    .par_iter()
                    .take(usable_row)
                    .try_for_each(|fi| -> Result<(), Error> {
                        let index = sorted_table_with_indices
                            .binary_search_by_key(&fi, |&(t, _)| t)
                            .map_err(|_| Error::ConstraintSystemFailure)?;
                        let index = sorted_table_with_indices[index].1;
                        m_values[index].fetch_add(1, Ordering::Relaxed);
                        Ok(())
                    })?;
            }

            m_values
                .par_iter()
                .map(|mi| F::from(mi.load(Ordering::Relaxed)))
                .collect()
        };
        end_timer!(timer);
        // Closure to construct commitment to vector of values
        let commit_values = |values: &Polynomial<C::Scalar, LagrangeCoeff>, max_bits: usize| {
            params
                .commit_lagrange_with_bound(values, max_bits)
                .to_affine()
        };
        multiplicity_values.truncate(usable_row);

        let m_max_bits = expression_max_bits::<C>(&multiplicity_values);
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
                compressed_input_expression_set,
                multiplicity_expression,
            },
            multiplicity_commitment,
        ))
    }
}

impl<C: CurveAffine> Compressed<C> {
    /// Given a Lookup with input expressions, table expressions, and the multiplicity
    /// expression, this method constructs the grand sum polynomial over the lookup.
    pub(in crate::plonk) fn commit_grand_sum(
        self,
        pk: &ProvingKey<C>,
        params: &Params<C>,
        beta: ChallengeBeta<C>,
    ) -> Result<(Polynomial<C::Scalar, LagrangeCoeff>, Vec<C::Scalar>), Error> {
        let blinding_factors = pk.vk.cs.blinding_factors();
        /*
            φ_i(X) = f_i(X) + beta
            τ(X) = t(X) + beta
            LHS = τ(X) * Π(φ_i(X)) * (ϕ(gX) - ϕ(X))
            RHS = τ(X) * Π(φ_i(X)) * (∑ 1/(φ_i(X)) - m(X) / τ(X))))
        */

        let mut grand_sum = vec![C::Scalar::zero(); params.n as usize];

        for input in self.compressed_input_expression_set.iter() {
            let mut fi_sum = vec![C::Scalar::zero(); params.n as usize];
            parallelize(&mut fi_sum, |sum, start| {
                for (sum_value, expression_value) in sum.iter_mut().zip(input[start..].iter()) {
                    *sum_value += &(*beta + expression_value);
                }
            });
            batch_invert(&mut fi_sum);
            parallelize(&mut grand_sum, |sum, start| {
                for (sum_value, expression_value) in sum.iter_mut().zip(fi_sum[start..].iter()) {
                    *sum_value += expression_value;
                }
            });
        }

        let mut table_sum = vec![C::Scalar::zero(); params.n as usize];
        parallelize(&mut table_sum, |sum, start| {
            for (sum_value, expression_value) in sum
                .iter_mut()
                .zip(self.compressed_table_expression[start..].iter())
            {
                *sum_value += &(*beta + expression_value);
            }
        });
        batch_invert(&mut table_sum);
        parallelize(&mut grand_sum, |sum, start| {
            for ((sum_value, table_value), m_value) in sum
                .iter_mut()
                .zip(table_sum[start..].iter())
                .zip(self.multiplicity_expression[start..].iter())
            {
                // (Σ 1/(φ_i(X)) - m(X) / τ(X))
                *sum_value = *sum_value - &(*table_value * m_value);
            }
        });

        let z = iter::once(C::Scalar::zero())
            .chain(grand_sum)
            .scan(C::Scalar::zero(), |state, v| {
                *state = *state + v;
                Some(*state)
            })
            .take(params.n as usize - blinding_factors)
            .collect::<Vec<_>>();

        #[cfg(feature = "sanity-checks")]
        {
            // While in Lagrange basis, check that grand sum is correctly constructed
            let u = (params.n as usize) - (blinding_factors + 1);
            /*
                 φ_i(X) = f_i(X) + α
                 τ(X) = t(X) + α
                 LHS = τ(X) * Π(φ_i(X)) * (ϕ(gX) - ϕ(X))
                 RHS = τ(X) * Π(φ_i(X)) * (∑ 1/(φ_i(X)) - m(X) / τ(X))))
            */
            // l_0(X) * (1 - z(X)) = 0
            assert_eq!(z[0], C::Scalar::zero());

            let mut phi_chunk_sums = (0..u)
                .into_iter()
                .map(|i| {
                    self.compressed_input_expression_set
                        .iter()
                        .map(|input| (input[i] + *beta).invert().unwrap())
                        .fold(C::Scalar::zero(), |acc, e| acc + e)
                })
                .collect::<Vec<_>>();

            for ((phi, table), m) in phi_chunk_sums
                .iter_mut()
                .zip(self.compressed_table_expression.iter())
                .zip(self.m_expression.iter())
            {
                *phi = *phi - *m * &(*table + *beta).invert().unwrap();
            }

            for (i, phi_sum) in phi_chunk_sums.iter().enumerate() {
                assert_eq!(z[i + 1], *phi_sum + z[i]);
            }
            assert_eq!(z[u], C::Scalar::zero())
        }

        Ok((self.multiplicity_expression, z))
    }
}

impl<C: CurveAffine> Committed<C> {
    pub(in crate::plonk) fn evaluate(
        self,
        pk: &ProvingKey<C>,
        x: ChallengeX<C>,
    ) -> (Evaluated<C>, Vec<C::ScalarExt>) {
        let domain = &pk.vk.domain;
        let x_next = domain.rotate_omega(*x, Rotation::next());

        let evals = vec![
            (&self.grand_sum_poly, *x),
            (&self.grand_sum_poly, x_next),
            (&self.multiplicity_poly, *x),
        ]
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
        let x_next = pk.vk.domain.rotate_omega(*x, Rotation::next());

        iter::empty()
            // Open lookup grand sum commitments at x
            .chain(Some(ProverQuery {
                point: *x,
                rotation: Rotation::cur(),
                poly: &self.constructed.grand_sum_poly,
            }))
            // Open lookup grand sum commitments at x_next
            .chain(Some(ProverQuery {
                point: x_next,
                rotation: Rotation::next(),
                poly: &self.constructed.grand_sum_poly,
            }))
            // Open lookup m poly commitments at x
            .chain(Some(ProverQuery {
                point: *x,
                rotation: Rotation::cur(),
                poly: &self.constructed.multiplicity_poly,
            }))
    }
}

fn expression_max_bits<C: CurveAffine>(expression: &Vec<C::Scalar>) -> usize {
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
