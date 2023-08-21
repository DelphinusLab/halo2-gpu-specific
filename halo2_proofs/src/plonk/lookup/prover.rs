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
pub(in crate::plonk) struct Permuted<C: CurveAffine> {
    compressed_input_expression: Polynomial<C::Scalar, LagrangeCoeff>,
    pub(in crate::plonk) permuted_input_expression: Polynomial<C::Scalar, LagrangeCoeff>,
    compressed_table_expression: Polynomial<C::Scalar, LagrangeCoeff>,
    pub(in crate::plonk) permuted_table_expression: Polynomial<C::Scalar, LagrangeCoeff>,
}

#[derive(Debug)]
pub(in crate::plonk) struct Committed<C: CurveAffine> {
    pub(in crate::plonk) permuted_input_poly: Polynomial<C::Scalar, Coeff>,
    pub(in crate::plonk) permuted_table_poly: Polynomial<C::Scalar, Coeff>,
    pub(in crate::plonk) product_poly: Polynomial<C::Scalar, Coeff>,
}

pub(in crate::plonk) struct Evaluated<C: CurveAffine> {
    constructed: Committed<C>,
}

impl<F: FieldExt> Argument<F> {
    /// Given a Lookup with input expressions [A_0, A_1, ..., A_{m-1}] and table expressions
    /// [S_0, S_1, ..., S_{m-1}], this method
    /// - constructs A_compressed = \theta^{m-1} A_0 + theta^{m-2} A_1 + ... + \theta A_{m-2} + A_{m-1}
    ///   and S_compressed = \theta^{m-1} S_0 + theta^{m-2} S_1 + ... + \theta S_{m-2} + S_{m-1},
    /// - permutes A_compressed and S_compressed using permute_expression_pair() helper,
    ///   obtaining A' and S', and
    /// - constructs Permuted<C> struct using permuted_input_value = A', and
    ///   permuted_table_expression = S'.
    /// The Permuted<C> struct is used to update the Lookup, and is then returned.
    pub(in crate::plonk) fn commit_permuted<'a, C, R: RngCore>(
        &self,
        pk: &ProvingKey<C>,
        params: &Params<C>,
        domain: &EvaluationDomain<C::Scalar>,
        theta: ChallengeTheta<C>,
        advice_values: &'a [Polynomial<C::Scalar, LagrangeCoeff>],
        fixed_values: &'a [Polynomial<C::Scalar, LagrangeCoeff>],
        instance_values: &'a [Polynomial<C::Scalar, LagrangeCoeff>],
        mut rng: R,
    ) -> Result<(Permuted<C>, [C; 2]), Error>
    where
        C: CurveAffine<ScalarExt = F>,
        C::Curve: Mul<F, Output = C::Curve> + MulAssign<F>,
    {
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

        // Closure to construct commitment to vector of values
        let commit_values = |values: &Polynomial<C::Scalar, LagrangeCoeff>, max_bits: usize| {
            params
                .commit_lagrange_with_bound(values, max_bits)
                .to_affine()
        };

        // Get values of input expressions involved in the lookup and compress them
        let compressed_input_expression = compress_expressions(&self.input_expressions);

        // Get values of table expressions involved in the lookup and compress them
        let compressed_table_expression = compress_expressions(&self.table_expressions);

        // Permute compressed (InputExpression, TableExpression) pair
        let (
            permuted_input_expression,
            permuted_table_expression,
            permuted_input_expression_max_bits,
            permuted_table_expression_max_bits,
        ) = permute_expression_pair::<C, _>(
            pk,
            params,
            domain,
            &mut rng,
            &compressed_input_expression,
            &compressed_table_expression,
        )?;

        // Commit to permuted input expression
        let permuted_input_commitment = commit_values(
            &permuted_input_expression,
            permuted_input_expression_max_bits,
        );

        // Commit to permuted table expression
        let permuted_table_commitment = commit_values(
            &permuted_table_expression,
            permuted_table_expression_max_bits,
        );

        Ok((
            Permuted {
                compressed_input_expression,
                permuted_input_expression,
                compressed_table_expression,
                permuted_table_expression,
            },
            [permuted_input_commitment, permuted_table_commitment],
        ))
    }
}

impl<C: CurveAffine> Permuted<C> {
    /// Given a Lookup with input expressions, table expressions, and the permuted
    /// input expression and permuted table expression, this method constructs the
    /// grand product polynomial over the lookup. The grand product polynomial
    /// is used to populate the Product<C> struct. The Product<C> struct is
    /// added to the Lookup and finally returned by the method.
    pub(in crate::plonk) fn commit_product(
        self,
        pk: &ProvingKey<C>,
        params: &Params<C>,
        beta: ChallengeBeta<C>,
        gamma: ChallengeGamma<C>,
    ) -> Result<
        (
            Polynomial<C::Scalar, LagrangeCoeff>,
            Polynomial<C::Scalar, LagrangeCoeff>,
            Vec<C::Scalar>,
        ),
        Error,
    > {
        let blinding_factors = pk.vk.cs.blinding_factors();

        // Goal is to compute the products of fractions
        //
        // Numerator: (\theta^{m-1} a_0(\omega^i) + \theta^{m-2} a_1(\omega^i) + ... + \theta a_{m-2}(\omega^i) + a_{m-1}(\omega^i) + \beta)
        //            * (\theta^{m-1} s_0(\omega^i) + \theta^{m-2} s_1(\omega^i) + ... + \theta s_{m-2}(\omega^i) + s_{m-1}(\omega^i) + \gamma)
        // Denominator: (a'(\omega^i) + \beta) (s'(\omega^i) + \gamma)
        //
        // where a_j(X) is the jth input expression in this lookup,
        // where a'(X) is the compression of the permuted input expressions,
        // s_j(X) is the jth table expression in this lookup,
        // s'(X) is the compression of the permuted table expressions,
        // and i is the ith row of the expression.
        let mut lookup_product = vec![C::Scalar::zero(); params.n as usize];

        #[cfg(not(feature = "cuda"))]
        {
            // Denominator uses the permuted input expression and permuted table expression
            parallelize(&mut lookup_product, |lookup_product, start| {
                for ((lookup_product, permuted_input_value), permuted_table_value) in lookup_product
                    .iter_mut()
                    .zip(self.permuted_input_expression[start..].iter())
                    .zip(self.permuted_table_expression[start..].iter())
                {
                    *lookup_product =
                        (*beta + permuted_input_value) * &(*gamma + permuted_table_value);
                }
            });

            // Batch invert to obtain the denominators for the lookup product
            // polynomials
            batch_invert(&mut lookup_product);

            // Finish the computation of the entire fraction by computing the numerators
            // (\theta^{m-1} a_0(\omega^i) + \theta^{m-2} a_1(\omega^i) + ... + \theta a_{m-2}(\omega^i) + a_{m-1}(\omega^i) + \beta)
            // * (\theta^{m-1} s_0(\omega^i) + \theta^{m-2} s_1(\omega^i) + ... + \theta s_{m-2}(\omega^i) + s_{m-1}(\omega^i) + \gamma)
            parallelize(&mut lookup_product, |product, start| {
                for (i, product) in product.iter_mut().enumerate() {
                    let i = i + start;

                    *product *= &(self.compressed_input_expression[i] + &*beta);
                    *product *= &(self.compressed_table_expression[i] + &*gamma);
                }
            });
        }

        // Denominator uses the permuted input expression and permuted table expression
        for ((lookup_product, permuted_input_value), permuted_table_value) in lookup_product
            .iter_mut()
            .zip(self.permuted_input_expression.iter())
            .zip(self.permuted_table_expression.iter())
        {
            *lookup_product = (*beta + permuted_input_value) * &(*gamma + permuted_table_value);
        }

        // Batch invert to obtain the denominators for the lookup product
        // polynomials
        lookup_product.batch_invert();

        // Finish the computation of the entire fraction by computing the numerators
        // (\theta^{m-1} a_0(\omega^i) + \theta^{m-2} a_1(\omega^i) + ... + \theta a_{m-2}(\omega^i) + a_{m-1}(\omega^i) + \beta)
        // * (\theta^{m-1} s_0(\omega^i) + \theta^{m-2} s_1(\omega^i) + ... + \theta s_{m-2}(\omega^i) + s_{m-1}(\omega^i) + \gamma)
        for ((lookup_product, compressed_input_value), compressed_table_value) in lookup_product
            .iter_mut()
            .zip(self.compressed_input_expression.iter())
            .zip(self.compressed_table_expression.iter())
        {
            *lookup_product *=
                (*beta + compressed_input_value) * &(*gamma + compressed_table_value);
        }

        /*
        #[cfg(feature = "cuda")]
        {
            use ec_gpu_gen::fft::FftKernel;
            use ec_gpu_gen::rust_gpu_tools::program_closures;
            use ec_gpu_gen::rust_gpu_tools::Device;
            use ec_gpu_gen::threadpool::Worker;
            use ec_gpu_gen::EcResult;
            use group::Curve;
            use pairing::bn256::Fr;

            let device = Device::all()[_gpu_idx];
            let program = ec_gpu_gen::program!(device).expect("Cannot create programs!");
            let kern = FftKernel::<Fr>::create(vec![program]).expect("Cannot initialize kernel!");

            let compute_units = device.compute_units() as usize;
            let local_work_size = 128usize;
            let work_units =
                (compute_units * local_work_size * 2) / local_work_size * local_work_size as usize;
            let len = self.permuted_input_expression.len();
            let slot_len = ((len + work_units - 1) / work_units) as usize;

            let closures = program_closures!(|program,
                                              input: (&[Fr], &[Fr], &[Fr], &[Fr], &mut [Fr])|
             -> EcResult<()> {
                let permuted_input = program.create_buffer_from_slice(input.0)?;
                let permuted_table = program.create_buffer_from_slice(input.1)?;
                let compressed_input = program.create_buffer_from_slice(input.2)?;
                let compressed_table = program.create_buffer_from_slice(input.3)?;
                let beta_gamma = program.create_buffer_from_slice(&vec![*beta, *gamma])?;

                let global_work_size = work_units / local_work_size;

                let kernel_name = format!("{}_calc_lookup_z", "Bn256_Fr");
                let kernel = program.create_kernel(
                    &kernel_name,
                    global_work_size as usize,
                    local_work_size as usize,
                )?;
                kernel
                    .arg(&permuted_input)
                    .arg(&permuted_table)
                    .arg(&compressed_input)
                    .arg(&compressed_table)
                    .arg(&beta_gamma)
                    .arg(&(len as u32))
                    .arg(&(slot_len as u32))
                    .run()?;

                let mut lookup_product_to_inv = vec![Fr::one(); params.n as usize];
                let mut lookup_product_to_inv_packed = vec![Fr::zero(); work_units];
                program.read_into_buffer(&permuted_input, input.4)?;
                program.read_into_buffer(&permuted_table, &mut lookup_product_to_inv)?;

                for i in 0..work_units.min((len + slot_len - 1) / slot_len) {
                    lookup_product_to_inv_packed[i] = lookup_product_to_inv[i * slot_len];
                }

                lookup_product_to_inv_packed.iter_mut().batch_invert();

                for (i, lookup_product_to_inv) in
                    lookup_product_to_inv_packed.into_iter().enumerate()
                {
                    for j in i * slot_len..((i + 1) * slot_len).min(len) {
                        input.4[j] *= lookup_product_to_inv;
                    }
                }

                Ok(())
            });

            kern.kernels[0]
                .program
                .run(closures, unsafe {
                    (
                        std::mem::transmute::<_, &[Fr]>(&self.permuted_input_expression[..]),
                        std::mem::transmute::<_, &[Fr]>(&self.permuted_table_expression[..]),
                        std::mem::transmute::<_, &[Fr]>(&self.compressed_input_expression[..]),
                        std::mem::transmute::<_, &[Fr]>(&self.compressed_table_expression[..]),
                        std::mem::transmute::<_, &mut [Fr]>(&mut lookup_product[..]),
                    )
                })
                .unwrap();
        }
        */

        // The product vector is a vector of products of fractions of the form
        //
        // Numerator: (\theta^{m-1} a_0(\omega^i) + \theta^{m-2} a_1(\omega^i) + ... + \theta a_{m-2}(\omega^i) + a_{m-1}(\omega^i) + \beta)
        //            * (\theta^{m-1} s_0(\omega^i) + \theta^{m-2} s_1(\omega^i) + ... + \theta s_{m-2}(\omega^i) + s_{m-1}(\omega^i) + \gamma)
        // Denominator: (a'(\omega^i) + \beta) (s'(\omega^i) + \gamma)
        //
        // where there are m input expressions and m table expressions,
        // a_j(\omega^i) is the jth input expression in this lookup,
        // a'j(\omega^i) is the permuted input expression,
        // s_j(\omega^i) is the jth table expression in this lookup,
        // s'(\omega^i) is the permuted table expression,
        // and i is the ith row of the expression.
        // Compute the evaluations of the lookup product polynomial
        // over our domain, starting with z[0] = 1
        let z = iter::once(C::Scalar::one())
            .chain(lookup_product)
            .scan(C::Scalar::one(), |state, cur| {
                *state *= &cur;
                Some(*state)
            })
            // Take all rows including the "last" row which should
            // be a boolean (and ideally 1, else soundness is broken)
            .take(params.n as usize - blinding_factors)
            .collect::<Vec<_>>();

        #[cfg(feature = "sanity-checks")]
        // This test works only with intermediate representations in this method.
        // It can be used for debugging purposes.
        {
            // While in Lagrange basis, check that product is correctly constructed
            let u = (params.n as usize) - (blinding_factors + 1);

            // l_0(X) * (1 - z(X)) = 0
            assert_eq!(z[0], C::Scalar::one());

            // z(\omega X) (a'(X) + \beta) (s'(X) + \gamma)
            // - z(X) (\theta^{m-1} a_0(X) + ... + a_{m-1}(X) + \beta) (\theta^{m-1} s_0(X) + ... + s_{m-1}(X) + \gamma)
            for i in 0..u {
                let mut left = z[i + 1];
                let permuted_input_value = &self.permuted_input_expression[i];

                let permuted_table_value = &self.permuted_table_expression[i];

                left *= &(*beta + permuted_input_value);
                left *= &(*gamma + permuted_table_value);

                let mut right = z[i];
                let mut input_term = self.compressed_input_expression[i];
                let mut table_term = self.compressed_table_expression[i];

                input_term += &(*beta);
                table_term += &(*gamma);
                right *= &(input_term * &table_term);

                assert_eq!(left, right);
            }

            // l_last(X) * (z(X)^2 - z(X)) = 0
            // Assertion will fail only when soundness is broken, in which
            // case this z[u] value will be zero. (bad!)
            assert_eq!(z[u], C::Scalar::one());
        }

        Ok((
            self.permuted_input_expression,
            self.permuted_table_expression,
            z,
        ))
    }
}

impl<C: CurveAffine> Committed<C> {
    pub(in crate::plonk) fn evaluate(
        self,
        pk: &ProvingKey<C>,
        x: ChallengeX<C>,
    ) -> (Evaluated<C>, Vec<C::ScalarExt>) {
        let domain = &pk.vk.domain;
        let x_inv = domain.rotate_omega(*x, Rotation::prev());
        let x_next = domain.rotate_omega(*x, Rotation::next());

        let evals = vec![
            (&self.product_poly, *x),
            (&self.product_poly, x_next),
            (&self.permuted_input_poly, *x),
            (&self.permuted_input_poly, x_inv),
            (&self.permuted_table_poly, *x),
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
        let x_inv = pk.vk.domain.rotate_omega(*x, Rotation::prev());
        let x_next = pk.vk.domain.rotate_omega(*x, Rotation::next());

        iter::empty()
            // Open lookup product commitments at x
            .chain(Some(ProverQuery {
                point: *x,
                rotation: Rotation::cur(),
                poly: &self.constructed.product_poly,
            }))
            // Open lookup input commitments at x
            .chain(Some(ProverQuery {
                point: *x,
                rotation: Rotation::cur(),
                poly: &self.constructed.permuted_input_poly,
            }))
            // Open lookup table commitments at x
            .chain(Some(ProverQuery {
                point: *x,
                rotation: Rotation::cur(),
                poly: &self.constructed.permuted_table_poly,
            }))
            // Open lookup input commitments at x_inv
            .chain(Some(ProverQuery {
                point: x_inv,
                rotation: Rotation::prev(),
                poly: &self.constructed.permuted_input_poly,
            }))
            // Open lookup product commitments at x_next
            .chain(Some(ProverQuery {
                point: x_next,
                rotation: Rotation::next(),
                poly: &self.constructed.product_poly,
            }))
    }
}

type ExpressionPair<F> = (
    Polynomial<F, LagrangeCoeff>,
    Polynomial<F, LagrangeCoeff>,
    usize,
    usize,
);

fn sort_get_max<F: FieldExt>(value: &mut Vec<F>) -> F {
    let max = *value.iter().reduce(|a, b| a.max(b)).unwrap();

    value.sort_unstable_by(|a, b| unsafe {
        let a: &[u64; 4] = std::mem::transmute(a);
        let b: &[u64; 4] = std::mem::transmute(b);
        a.cmp(b)
    });

    max
}

/// Given a vector of input values A and a vector of table values S,
/// this method permutes A and S to produce A' and S', such that:
/// - like values in A' are vertically adjacent to each other; and
/// - the first row in a sequence of like values in A' is the row
///   that has the corresponding value in S'.
/// This method returns (A', S') if no errors are encountered.
fn permute_expression_pair<C: CurveAffine, R: RngCore>(
    pk: &ProvingKey<C>,
    params: &Params<C>,
    domain: &EvaluationDomain<C::Scalar>,
    mut rng: R,
    input_expression: &Polynomial<C::Scalar, LagrangeCoeff>,
    table_expression: &Polynomial<C::Scalar, LagrangeCoeff>,
) -> Result<ExpressionPair<C::Scalar>, Error> {
    let blinding_factors = pk.vk.cs.blinding_factors();
    let usable_rows = params.n as usize - (blinding_factors + 1);

    let mut permuted_input_expression: Vec<C::Scalar> = input_expression.to_vec();
    permuted_input_expression.truncate(usable_rows);

    let mut sorted_table_coeffs = table_expression.to_vec();
    sorted_table_coeffs.truncate(usable_rows);

    let max_input = sort_get_max(&mut permuted_input_expression);
    let max_table = sort_get_max(&mut sorted_table_coeffs);

    let mut permuted_table_coeffs = vec![None; usable_rows];

    let unique_input_values = permuted_input_expression
        .iter()
        .zip(permuted_table_coeffs.iter_mut())
        .enumerate()
        .filter_map(|(row, (input_value, table_value))| {
            // If this is the first occurrence of `input_value` in the input expression
            if row == 0 || *input_value != permuted_input_expression[row - 1] {
                *table_value = Some(*input_value);
                Some(*input_value)
            } else {
                None
            }
        })
        .collect::<Vec<_>>();

    let mut i_unique_input_value = 0;
    let mut i_sorted_table_coeffs = 0;
    for v in permuted_table_coeffs.iter_mut() {
        while i_unique_input_value < unique_input_values.len()
            && unique_input_values[i_unique_input_value]
                == sorted_table_coeffs[i_sorted_table_coeffs]
        {
            i_unique_input_value += 1;
            i_sorted_table_coeffs += 1;
        }
        if v.is_none() {
            *v = Some(sorted_table_coeffs[i_sorted_table_coeffs]);
            i_sorted_table_coeffs += 1;
        }
    }

    while i_unique_input_value < unique_input_values.len()
        && unique_input_values[i_unique_input_value] == sorted_table_coeffs[i_sorted_table_coeffs]
    {
        i_unique_input_value += 1;
        i_sorted_table_coeffs += 1;
    }

    //assert!(i_unique_input_value == unique_input_values.len());

    let mut permuted_table_coeffs = permuted_table_coeffs
        .iter()
        .filter_map(|x| *x)
        .collect::<Vec<_>>();

    permuted_input_expression
        .extend((0..(blinding_factors + 1)).map(|_| C::Scalar::from(u16::rand(&mut rng) as u64)));
    permuted_table_coeffs
        .extend((0..(blinding_factors + 1)).map(|_| C::Scalar::from(u16::rand(&mut rng) as u64)));
    assert_eq!(permuted_input_expression.len(), params.n as usize);
    assert_eq!(permuted_table_coeffs.len(), params.n as usize);

    #[cfg(feature = "sanity-checks")]
    {
        let mut last = None;
        for (a, b) in permuted_input_expression
            .iter()
            .zip(permuted_table_coeffs.iter())
            .take(usable_rows)
        {
            if *a != *b {
                assert_eq!(*a, last.unwrap());
            }
            last = Some(*a);
        }
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

    Ok((
        domain.lagrange_from_vec(permuted_input_expression),
        domain.lagrange_from_vec(permuted_table_coeffs),
        16.max(get_scalar_bits(max_input)),
        16.max(get_scalar_bits(max_table)),
    ))
}
