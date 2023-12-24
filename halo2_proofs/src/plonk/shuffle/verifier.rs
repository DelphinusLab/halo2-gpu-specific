use std::iter;

use super::super::{
    circuit::Expression, ChallengeBeta, ChallengeGamma, ChallengeTheta, ChallengeX,
};
use super::Argument;
use crate::{
    arithmetic::{CurveAffine, FieldExt},
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

impl<F: FieldExt> Argument<F> {
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
        argument: &'a Argument<C::Scalar>,
        theta: ChallengeTheta<C>,
        gamma: ChallengeGamma<C>,
        advice_evals: &[C::Scalar],
        fixed_evals: &[C::Scalar],
        instance_evals: &[C::Scalar],
    ) -> impl Iterator<Item = C::Scalar> + 'a {
        let active_rows = C::Scalar::one() - (l_last + l_blind);
        let product_expression = || {
            // z(\omega X) (\theta^{m-1} s_0(X) + ... + s_{m-1}(X) + \gamma)
            // - z(X) (\theta^{m-1} a_0(X) + ... + a_{m-1}(X) + \gamma)
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
            let left = self.product_next_eval
                * &(compress_expressions(&argument.shuffle_expressions) + &*gamma);
            let right =
                self.product_eval * &(compress_expressions(&argument.input_expressions) + &*gamma);
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
                // (1 - (l_last(X) + l_blind(X))) * ( z(\omega X) (s(X) + \gamma) - z(X) (a(X) + \gamma))
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
