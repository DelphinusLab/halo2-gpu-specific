use std::iter;

use super::super::{
    circuit::Expression, ChallengeBeta, ChallengeGamma, ChallengeTheta, ChallengeX,
};
use super::Argument;
use crate::{
    arithmetic::{BaseExt, CurveAffine, FieldExt},
    plonk::{Error, VerifyingKey},
    poly::{multiopen::VerifierQuery, Rotation},
    transcript::{EncodedChallenge, TranscriptRead},
};

use group::{
    ff::{BatchInvert, Field},
    Curve,
};
#[derive(Debug)]
pub struct MultiplicityCommitment<C: CurveAffine>(pub C);

#[derive(Debug)]
pub struct Committed<C: CurveAffine> {
    pub grand_sum_commitment: C,
    pub multiplicity_commitment: MultiplicityCommitment<C>,
}

#[derive(Debug)]
pub struct Evaluated<C: CurveAffine> {
    pub committed: Committed<C>,
    pub grand_sum_eval: C::Scalar,
    pub grand_sum_next_eval: C::Scalar,
    pub multiplicity_eval: C::Scalar,
}

impl<F: FieldExt> Argument<F> {
    pub fn read_m_commitments<C: CurveAffine, E: EncodedChallenge<C>, T: TranscriptRead<C, E>>(
        &self,
        transcript: &mut T,
    ) -> Result<MultiplicityCommitment<C>, Error> {
        Ok(MultiplicityCommitment(transcript.read_point()?))
    }
}

impl<C: CurveAffine> MultiplicityCommitment<C> {
    pub fn read_grand_sum_commitment<E: EncodedChallenge<C>, T: TranscriptRead<C, E>>(
        self,
        transcript: &mut T,
    ) -> Result<Committed<C>, Error> {
        let grand_sum_commitment = transcript.read_point()?;

        Ok(Committed {
            multiplicity_commitment: self,
            grand_sum_commitment,
        })
    }
}

impl<C: CurveAffine> Committed<C> {
    pub fn evaluate<E: EncodedChallenge<C>, T: TranscriptRead<C, E>>(
        self,
        transcript: &mut T,
    ) -> Result<Evaluated<C>, Error> {
        let grand_sum_eval = transcript.read_scalar()?;
        let grand_sum_next_eval = transcript.read_scalar()?;
        let multiplicity_eval = transcript.read_scalar()?;

        Ok(Evaluated {
            committed: self,
            grand_sum_eval,
            grand_sum_next_eval,
            multiplicity_eval,
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
        beta: ChallengeBeta<C>,
        advice_evals: &[C::Scalar],
        fixed_evals: &[C::Scalar],
        instance_evals: &[C::Scalar],
    ) -> impl Iterator<Item = C::Scalar> + 'a {
        let active_rows = C::Scalar::one() - (l_last + l_blind);

        let product_expression = || {
            let z_gx_minus_zx = self.grand_sum_next_eval - self.grand_sum_eval;
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
            /*
                 φ_i(X) = f_i(X) + α
                 τ(X) = t(X) + α
                 LHS = τ(X) * Π(φ_i(X)) * (ϕ(gX) - ϕ(X))
                 RHS = τ(X) * Π(φ_i(X)) * (∑ 1/(φ_i(X)) - m(X) / τ(X))))
                 <=>
                 LHS = (τ(X) * (ϕ(gX) - ϕ(X)) + m(x)) *Π(φ_i(X))
                 RHS = τ(X) * Π(φ_i(X)) * (∑ 1/(φ_i(X)))
            */
            let mut phi = argument
                .input_expressions_set
                .0
                .iter()
                .map(|express| compress_expressions(express) + &*beta)
                .collect::<Vec<_>>();
            let tau = compress_expressions(&argument.table_expressions) + &*beta;
            let product_fi = phi.iter().fold(C::Scalar::one(), |acc, e| acc * e);
            phi.batch_invert();
            let sum_invert_fi = phi.iter().fold(C::Scalar::zero(), |acc, e| acc + e);

            let left = (tau * z_gx_minus_zx + &self.multiplicity_eval) * product_fi;
            let right = tau * product_fi * sum_invert_fi;
            (left - &right) * &active_rows
        };

        std::iter::empty()
            .chain(
                // l_0(X) * z[0](X) = 0
                Some(l_0 * &self.grand_sum_eval),
            )
            .chain(
                // l_last(X) * z[last](X) = 0
                Some(l_last * &self.grand_sum_eval),
            )
            .chain(
                /*
                     φ_i(X) = f_i(X) + α
                     τ(X) = t(X) + α
                     LHS = τ(X) * Π(φ_i(X)) * (ϕ(gX) - ϕ(X))
                     RHS = τ(X) * Π(φ_i(X)) * (∑ 1/(φ_i(X)) - m(X) / τ(X))))
                */
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
            .chain(Some(VerifierQuery::new_commitment(
                &self.committed.grand_sum_commitment,
                *x,
                Rotation::cur(),
                self.grand_sum_eval,
            )))
            .chain(Some(VerifierQuery::new_commitment(
                &self.committed.grand_sum_commitment,
                x_next,
                Rotation::next(),
                self.grand_sum_next_eval,
            )))
            .chain(Some(VerifierQuery::new_commitment(
                &self.committed.multiplicity_commitment.0,
                *x,
                Rotation::cur(),
                self.multiplicity_eval,
            )))
    }
}
