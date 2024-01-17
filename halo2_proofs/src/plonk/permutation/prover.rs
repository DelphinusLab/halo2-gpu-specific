use ark_std::{end_timer, start_timer};
use group::{
    ff::{BatchInvert, Field},
    Curve,
};
use rand_core::RngCore;
use rayon::prelude::ParallelSlice;
use rayon::prelude::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};
use std::iter::{self, ExactSizeIterator};

use super::super::{circuit::Any, ChallengeBeta, ChallengeGamma, ChallengeX};
use super::{Argument, ProvingKey};
use crate::{
    arithmetic::{
        batch_invert, eval_polynomial, mul_acc, parallelize, BaseExt, CurveAffine, FieldExt,
    },
    plonk::{self, Error},
    poly::{
        commitment::Params, multiopen::ProverQuery, Coeff, ExtendedLagrangeCoeff, LagrangeCoeff,
        Polynomial, Rotation,
    },
    transcript::{EncodedChallenge, TranscriptWrite},
};

pub(crate) struct CommittedSet<C: CurveAffine> {
    pub(crate) permutation_product_poly: Polynomial<C::Scalar, Coeff>,
    #[cfg(not(feature = "cuda"))]
    pub(crate) permutation_product_coset: Polynomial<C::Scalar, ExtendedLagrangeCoeff>,
}

pub(crate) struct Committed<C: CurveAffine> {
    pub(crate) sets: Vec<CommittedSet<C>>,
}

pub struct ConstructedSet<C: CurveAffine> {
    permutation_product_poly: Polynomial<C::Scalar, Coeff>,
}

pub(crate) struct Constructed<C: CurveAffine> {
    sets: Vec<ConstructedSet<C>>,
}

pub(crate) struct Evaluated<C: CurveAffine> {
    constructed: Constructed<C>,
}

impl Argument {
    pub(in crate::plonk) fn commit<C: CurveAffine, R: RngCore>(
        &self,
        params: &Params<C>,
        pk: &plonk::ProvingKey<C>,
        pkey: &ProvingKey<C>,
        advice: &[Polynomial<C::Scalar, LagrangeCoeff>],
        fixed: &[Polynomial<C::Scalar, LagrangeCoeff>],
        instance: &[Polynomial<C::Scalar, LagrangeCoeff>],
        beta: ChallengeBeta<C>,
        gamma: ChallengeGamma<C>,
        mut rng: R,
    ) -> Result<Vec<Polynomial<C::ScalarExt, LagrangeCoeff>>, Error> {
        let domain = &pk.vk.domain;

        // How many columns can be included in a single permutation polynomial?
        // We need to multiply by z(X) and (1 - (l_last(X) + l_blind(X))). This
        // will never underflow because of the requirement of at least a degree
        // 3 circuit for the permutation argument.
        assert!(pk.vk.cs.degree() >= 3);
        let chunk_len = pk.vk.cs.degree() - 2;
        let blinding_factors = pk.vk.cs.blinding_factors();

        let mut sets = vec![];

        let raw_zs = self
            .columns
            .par_chunks(chunk_len)
            .zip(pkey.permutations.par_chunks(chunk_len))
            .enumerate()
            .map(|(i, (columns, permutations))| {
                // Each column gets its own delta power.
                let mut delta_omega = C::Scalar::DELTA.pow(&[i as u64 * chunk_len as u64, 0, 0, 0]);

                // Goal is to compute the products of fractions
                //
                // (p_j(\omega^i) + \delta^j \omega^i \beta + \gamma) /
                // (p_j(\omega^i) + \beta s_j(\omega^i) + \gamma)
                //
                // where p_j(X) is the jth column in this permutation,
                // and i is the ith row of the column.

                let mut modified_values = vec![C::Scalar::one(); params.n as usize];

                // Iterate over each column of the permutation
                for (&column, permuted_column_values) in columns.iter().zip(permutations.iter()) {
                    let values = match column.column_type() {
                        Any::Advice => advice,
                        Any::Fixed => fixed,
                        Any::Instance => instance,
                    };
                    for i in 0..params.n as usize {
                        modified_values[i] *= &(*beta * permuted_column_values[i]
                            + &*gamma
                            + values[column.index()][i]);
                    }
                }

                // Invert to obtain the denominator for the permutation product polynomial
                modified_values.iter_mut().batch_invert();

                // Iterate over each column again, this time finishing the computation
                // of the entire fraction by computing the numerators
                for &column in columns.iter() {
                    let omega = domain.get_omega();
                    let values = match column.column_type() {
                        Any::Advice => advice,
                        Any::Fixed => fixed,
                        Any::Instance => instance,
                    };
                    for i in 0..params.n as usize {
                        modified_values[i] *=
                            &(delta_omega * &*beta + &*gamma + values[column.index()][i]);
                        delta_omega *= &omega;
                    }
                    delta_omega *= &C::Scalar::DELTA;
                }

                // The modified_values vector is a vector of products of fractions
                // of the form
                //
                // (p_j(\omega^i) + \delta^j \omega^i \beta + \gamma) /
                // (p_j(\omega^i) + \beta s_j(\omega^i) + \gamma)
                //
                // where i is the index into modified_values, for the jth column in
                // the permutation

                // Compute the evaluations of the permutation product polynomial
                // over our domain, starting with z[0] = 1

                let mut z = vec![C::Scalar::zero(); params.n as usize];
                z.iter_mut().enumerate().for_each(|(i, z)| {
                    if i > 0 {
                        *z = modified_values[i - 1];
                    }
                });
                z
            })
            .collect::<Vec<_>>();

        // Track the "last" value from the previous column set
        let mut last_z = C::Scalar::one();

        for mut z in raw_zs.into_iter() {
            z[0] = last_z;
            for i in 0..z.len() - 1 {
                z[i + 1] = z[i] * z[i + 1];
            }

            let mut z = domain.lagrange_from_vec(z);
            // Set blinding factors
            for z in &mut z[params.n as usize - blinding_factors..] {
                *z = C::Scalar::random(&mut rng);
            }
            // Set new last_z
            last_z = z[params.n as usize - (blinding_factors + 1)];

            sets.push(z);
        }

        Ok(sets)
    }
}

impl<C: CurveAffine> Committed<C> {
    pub(in crate::plonk) fn construct(self) -> Constructed<C> {
        Constructed {
            sets: self
                .sets
                .into_iter()
                .map(|set| ConstructedSet {
                    permutation_product_poly: set.permutation_product_poly,
                })
                .collect(),
        }
    }
}

impl<C: CurveAffine> super::ProvingKey<C> {
    pub(in crate::plonk) fn open(
        &self,
        x: ChallengeX<C>,
    ) -> impl Iterator<Item = ProverQuery<'_, C>> + Clone {
        self.polys.iter().map(move |poly| ProverQuery {
            point: *x,
            rotation: Rotation::cur(),
            poly,
        })
    }

    pub(in crate::plonk) fn evaluate<E: EncodedChallenge<C>, T: TranscriptWrite<C, E>>(
        &self,
        x: ChallengeX<C>,
        transcript: &mut T,
    ) -> Result<(), Error> {
        // Hash permutation evals
        for eval in self.polys.iter().map(|poly| eval_polynomial(poly, *x)) {
            transcript.write_scalar(eval)?;
        }

        Ok(())
    }
}

impl<C: CurveAffine> Constructed<C> {
    pub(in crate::plonk) fn evaluate<E: EncodedChallenge<C>, T: TranscriptWrite<C, E>>(
        self,
        pk: &plonk::ProvingKey<C>,
        x: ChallengeX<C>,
        transcript: &mut T,
    ) -> Result<Evaluated<C>, Error> {
        let domain = &pk.vk.domain;
        let blinding_factors = pk.vk.cs.blinding_factors();

        {
            let mut sets = self.sets.iter();

            while let Some(set) = sets.next() {
                let permutation_product_eval = eval_polynomial(&set.permutation_product_poly, *x);

                let permutation_product_next_eval = eval_polynomial(
                    &set.permutation_product_poly,
                    domain.rotate_omega(*x, Rotation::next()),
                );

                // Hash permutation product evals
                for eval in iter::empty()
                    .chain(Some(&permutation_product_eval))
                    .chain(Some(&permutation_product_next_eval))
                {
                    transcript.write_scalar(*eval)?;
                }

                // If we have any remaining sets to process, evaluate this set at omega^u
                // so we can constrain the last value of its running product to equal the
                // first value of the next set's running product, chaining them together.
                if sets.len() > 0 {
                    let permutation_product_last_eval = eval_polynomial(
                        &set.permutation_product_poly,
                        domain.rotate_omega(*x, Rotation(-((blinding_factors + 1) as i32))),
                    );

                    transcript.write_scalar(permutation_product_last_eval)?;
                }
            }
        }

        Ok(Evaluated { constructed: self })
    }
}

impl<C: CurveAffine> Evaluated<C> {
    pub(in crate::plonk) fn open<'a>(
        &'a self,
        pk: &'a plonk::ProvingKey<C>,
        x: ChallengeX<C>,
    ) -> impl Iterator<Item = ProverQuery<'a, C>> + Clone {
        let blinding_factors = pk.vk.cs.blinding_factors();
        let x_next = pk.vk.domain.rotate_omega(*x, Rotation::next());
        let x_last = pk
            .vk
            .domain
            .rotate_omega(*x, Rotation(-((blinding_factors + 1) as i32)));

        iter::empty()
            .chain(self.constructed.sets.iter().flat_map(move |set| {
                iter::empty()
                    // Open permutation product commitments at x and \omega x
                    .chain(Some(ProverQuery {
                        point: *x,
                        rotation: Rotation::cur(),
                        poly: &set.permutation_product_poly,
                    }))
                    .chain(Some(ProverQuery {
                        point: x_next,
                        rotation: Rotation::next(),
                        poly: &set.permutation_product_poly,
                    }))
            }))
            // Open it at \omega^{last} x for all but the last set. This rotation is only
            // sensical for the first row, but we only use this rotation in a constraint
            // that is gated on l_0.
            .chain(
                self.constructed
                    .sets
                    .iter()
                    .rev()
                    .skip(1)
                    .flat_map(move |set| {
                        Some(ProverQuery {
                            point: x_last,
                            rotation: Rotation(-((blinding_factors + 1) as i32)),
                            poly: &set.permutation_product_poly,
                        })
                    }),
            )
    }
}
