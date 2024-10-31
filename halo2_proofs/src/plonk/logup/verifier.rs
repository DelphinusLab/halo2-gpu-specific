use std::iter;

use super::super::{
    circuit::Expression, ChallengeBeta, ChallengeGamma, ChallengeTheta, ChallengeX,
};
use super::{Argument, InputExpressionSet};
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
    pub grand_sum_commitment_set: Vec<C>,
    pub multiplicity_commitment: MultiplicityCommitment<C>,
}

#[derive(Debug)]
pub struct GrandSumEvalSet<C: CurveAffine> {
    pub commitment: C,
    pub eval: C::Scalar,
    pub next_eval: C::Scalar,
    pub last_eval: Option<C::Scalar>,
}

#[derive(Debug)]
pub struct Evaluated<C: CurveAffine> {
    pub grand_sum_sets: Vec<GrandSumEvalSet<C>>,
    pub multiplicity_commitment: C,
    pub multiplicity_eval: C::Scalar,
}

impl<F: FieldExt> Argument<F> {
    pub fn read_m_commitments<C: CurveAffine, E: EncodedChallenge<C>, T: TranscriptRead<C, E>>(
        &self,
        transcript: &mut T,
    ) -> Result<MultiplicityCommitment<C>, Error> {
        Ok(MultiplicityCommitment(transcript.read_point()?))
    }

    pub fn read_grand_sum_commitment<
        C: CurveAffine,
        E: EncodedChallenge<C>,
        T: TranscriptRead<C, E>,
    >(
        &self,
        transcript: &mut T,
        multiplicity_commitment: MultiplicityCommitment<C>,
    ) -> Result<Committed<C>, Error> {
        let mut grand_sum_commitment_set = vec![];
        self.input_expressions_sets
            .iter()
            .try_for_each(|_| -> Result<_, Error> {
                grand_sum_commitment_set.push(transcript.read_point()?);
                Ok(())
            })?;

        Ok(Committed {
            multiplicity_commitment,
            grand_sum_commitment_set,
        })
    }
}

impl<C: CurveAffine> Committed<C> {
    pub fn evaluate<E: EncodedChallenge<C>, T: TranscriptRead<C, E>>(
        self,
        transcript: &mut T,
    ) -> Result<Evaluated<C>, Error> {
        let multiplicity_eval = transcript.read_scalar()?;

        let mut grand_sum_sets = vec![];
        let mut iter = self.grand_sum_commitment_set.into_iter();
        while let Some(commitment) = iter.next() {
            let eval = transcript.read_scalar()?;
            let next_eval = transcript.read_scalar()?;
            let last_eval = if iter.len() > 0 {
                Some(transcript.read_scalar()?)
            } else {
                None
            };
            grand_sum_sets.push(GrandSumEvalSet {
                commitment,
                eval,
                next_eval,
                last_eval,
            })
        }

        Ok(Evaluated {
            grand_sum_sets,
            multiplicity_commitment: self.multiplicity_commitment.0,
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
        advice_evals: &'a [C::Scalar],
        fixed_evals: &'a [C::Scalar],
        instance_evals: &'a [C::Scalar],
    ) -> impl Iterator<Item = C::Scalar> + 'a {
        let active_rows = C::Scalar::one() - (l_last + l_blind);

        let compress_expressions = move |expressions: &[Expression<C::Scalar>]| {
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

        let product_expression = || {
            let first_set = self.grand_sum_sets.first().unwrap();
            let z_gx_minus_zx = first_set.next_eval - first_set.eval;
            /*
                 φ_i(X) = f_i(X) + α
                 τ(X) = t(X) + α
                 LHS = τ(X) * Π(φ_i(X)) * (ϕ(gX) - ϕ(X))
                 RHS = τ(X) * Π(φ_i(X)) * (∑ 1/(φ_i(X)) - m(X) / τ(X))))
                 <=>
                 LHS = (τ(X) * (ϕ(gX) - ϕ(X)) + m(x)) *Π(φ_i(X))
                 RHS = τ(X) * Π(φ_i(X)) * (∑ 1/(φ_i(X)))
            */
            let mut phi = argument.input_expressions_sets[0]
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

        let extend_product_expression = move |(grand_sum_set, input_express_set): (
            &GrandSumEvalSet<C>,
            &InputExpressionSet<C::Scalar>,
        )| {
            let z_gx_minus_zx = grand_sum_set.next_eval - grand_sum_set.eval;
            /*
                 φ_i(X) = f_i(X) + α
                 LHS = Π(φ_i(X)) * (ϕ(gX) - ϕ(X))
                 RHS = Π(φ_i(X)) * (∑ 1/(φ_i(X)))
            */
            let mut phi = input_express_set
                .0
                .iter()
                .map(|express| compress_expressions(express) + &*beta)
                .collect::<Vec<_>>();
            let product_fi = phi.iter().fold(C::Scalar::one(), |acc, e| acc * e);
            phi.batch_invert();
            let sum_invert_fi = phi.iter().fold(C::Scalar::zero(), |acc, e| acc + e);

            let left = z_gx_minus_zx;
            let right = sum_invert_fi;
            (left - &right) * product_fi * &active_rows
        };

        std::iter::empty()
            .chain(
                // l_0(X) * z[0](X) = 0
                Some(l_0 * &self.grand_sum_sets.first().unwrap().eval),
            )
            .chain(
                // l_last(X) * z[last](X) = 0
                Some(l_last * &self.grand_sum_sets.last().unwrap().eval),
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
            .chain(
                self.grand_sum_sets
                    .iter()
                    .skip(1)
                    .zip(self.grand_sum_sets.iter())
                    .map(move |(set, last_set)| l_0 * (set.eval - &last_set.last_eval.unwrap())),
            )
            .chain(
                self.grand_sum_sets
                    .iter()
                    .zip(argument.input_expressions_sets.iter())
                    .skip(1)
                    .map(extend_product_expression),
            )
    }

    pub(in crate::plonk) fn queries<'r>(
        &'r self,
        vk: &'r VerifyingKey<C>,
        x: ChallengeX<C>,
    ) -> impl Iterator<Item = VerifierQuery<'r, C>> + Clone {
        let blinding_factors = vk.cs.blinding_factors();
        let x_next = vk.domain.rotate_omega(*x, Rotation::next());
        let x_last = vk
            .domain
            .rotate_omega(*x, Rotation(-((blinding_factors + 1) as i32)));

        iter::empty()
            .chain(Some(VerifierQuery::new_commitment(
                &self.multiplicity_commitment,
                *x,
                Rotation::cur(),
                self.multiplicity_eval,
            )))
            .chain(self.grand_sum_sets.iter().flat_map(move |set| {
                iter::empty()
                    .chain(Some(VerifierQuery::new_commitment(
                        &set.commitment,
                        *x,
                        Rotation::cur(),
                        set.eval,
                    )))
                    .chain(Some(VerifierQuery::new_commitment(
                        &set.commitment,
                        x_next,
                        Rotation::next(),
                        set.next_eval,
                    )))
            }))
            .chain(self.grand_sum_sets.iter().rev().skip(1).map(move |set| {
                VerifierQuery::new_commitment(
                    &set.commitment,
                    x_last,
                    Rotation(-((blinding_factors + 1) as i32)),
                    set.last_eval.unwrap(),
                )
            }))
    }
}
