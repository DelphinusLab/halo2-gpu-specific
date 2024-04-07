use std::iter;

use super::super::{
    circuit::Expression, ChallengeBeta, ChallengeGamma, ChallengeTheta, ChallengeX,
};
use super::ArgumentGroup;
use crate::{
    arithmetic::{BaseExt, CurveAffine, FieldExt},
    plonk::{Error, VerifyingKey},
    poly::{multiopen::VerifierQuery, Rotation},
    transcript::{EncodedChallenge, TranscriptRead},
};
use ff::Field;

#[derive(Debug)]
pub struct Committed<C: CurveAffine> {
    pub product_commitment: C,
}

#[derive(Debug)]
pub struct Evaluated<C: CurveAffine> {
    pub committed: Committed<C>,
    pub product_eval: C::Scalar,
    pub product_next_eval: C::Scalar,
}

impl<F: FieldExt> ArgumentGroup<F> {
    pub fn read_product_commitment<
        C: CurveAffine,
        E: EncodedChallenge<C>,
        T: TranscriptRead<C, E>,
    >(
        &self,
        transcript: &mut T,
    ) -> Result<Committed<C>, Error> {
        let product_commitment = transcript.read_point()?;

        Ok(Committed { product_commitment })
    }
}

impl<C: CurveAffine> Committed<C> {
    pub fn evaluate<E: EncodedChallenge<C>, T: TranscriptRead<C, E>>(
        self,
        transcript: &mut T,
    ) -> Result<Evaluated<C>, Error> {
        let product_eval = transcript.read_scalar()?;
        let product_next_eval = transcript.read_scalar()?;

        Ok(Evaluated {
            committed: self,
            product_eval,
            product_next_eval,
        })
    }
}

impl<C: CurveAffine> Evaluated<C> {
    pub(in crate::plonk) fn expressions<'a>(
        &'a self,
        l_0: C::Scalar,
        l_last: C::Scalar,
        l_blind: C::Scalar,
        argument: &'a ArgumentGroup<C::Scalar>,
        theta: ChallengeTheta<C>,
        beta: ChallengeBeta<C>,
        advice_evals: &[C::Scalar],
        fixed_evals: &[C::Scalar],
        instance_evals: &[C::Scalar],
    ) -> impl Iterator<Item = C::Scalar> + 'a {
        let active_rows = C::Scalar::one() - (l_last + l_blind);
        let product_expression = || {
            // (\theta^{m-1} s_0(X) + ... + s_{m-1}(X))
            let compress_expressions = |expressions: &[Expression<C::Scalar>]| {
                expressions
                    .iter()
                    .map(|expression| {
                        expression.evaluate(
                            &|scalar| scalar,
                            &|_| panic!("virtual selectors are removed during optimization"),
                            &|index, _, _| fixed_evals[index],
                            &|index, _, _| advice_evals[index],
                            &|index, _, _| instance_evals[index],
                            &|a| -a,
                            &|a, b| a + &b,
                            &|a, b| a() * &b(),
                            &|a, scalar| a * &scalar,
                        )
                    })
                    .fold(C::Scalar::zero(), |acc, eval| acc * &*theta + &eval)
            };
            let challenges: Vec<C::Scalar> = (0..argument.0.len())
                .map(|i| beta.pow_vartime([1 + i as u64, 0, 0, 0]))
                .collect();
            let (product_shuffle, product_input) = argument
                .0
                .iter()
                .zip(challenges.iter())
                .map(|(argument, lcx)| {
                    (
                        compress_expressions(&argument.shuffle_expressions) + lcx,
                        compress_expressions(&argument.input_expressions) + lcx,
                    )
                })
                .fold((C::Scalar::one(), C::Scalar::one()), |acc, v| {
                    (acc.0 * v.0, acc.1 * v.1)
                });
            let left = self.product_next_eval * &product_shuffle;
            let right = self.product_eval * &product_input;
            (left - &right) * &active_rows
        };

        std::iter::empty()
            .chain(
                // l_0(X) * (1 - z'(X)) = 0
                Some(l_0 * &(C::Scalar::one() - &self.product_eval)),
            )
            .chain(
                // l_last(X) * (z(X)^2 - z(X)) = 0
                Some(l_last * &(self.product_eval.square() - &self.product_eval)),
            )
            .chain(
                // (1 - (l_last(X) + l_blind(X))) *
                //( z(\omega X) (s1(X)+\beta)(s2(X)+\beta^2) - z(X) (a1(X)+\beta)(a2(X)+\beta^2))
                Some(product_expression()),
            )
    }

    pub(in crate::plonk) fn queries<'r>(
        &'r self,
        vk: &'r VerifyingKey<C>,
        x: ChallengeX<C>,
    ) -> impl Iterator<Item = VerifierQuery<'r, C>> + Clone {
        let x_next = vk.domain.rotate_omega(*x, Rotation::next());

        iter::empty()
            // Open lookup product commitment at x
            .chain(Some(VerifierQuery::new_commitment(
                &self.committed.product_commitment,
                *x,
                Rotation::cur(),
                self.product_eval,
            )))
            // Open lookup product commitment at \omega x
            .chain(Some(VerifierQuery::new_commitment(
                &self.committed.product_commitment,
                x_next,
                Rotation::next(),
                self.product_next_eval,
            )))
    }
}
