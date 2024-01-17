use ark_std::{end_timer, start_timer};
use ff::Field;
use group::Curve;
use rayon::prelude::*;

use super::{Argument, ProvingKey, VerifyingKey};
use crate::{
    arithmetic::{mul_acc, parallelize, CurveAffine, FieldExt},
    plonk::{Any, Column, Error},
    poly::{
        commitment::{Blind, Params},
        EvaluationDomain,
    },
};

#[derive(Debug, Clone)]
pub struct Assembly {
    pub(crate) columns: Vec<Column<Any>>,
    pub(crate) mapping: Vec<Vec<(u32, u32)>>,
    pub(crate) aux: Vec<Vec<(u32, u32)>>,
    pub(crate) sizes: Vec<Vec<usize>>,
}

impl Assembly {
    pub(crate) fn new(n: usize, p: &Argument) -> Self {
        // Initialize the copy vector to keep track of copy constraints in all
        // the permutation arguments.
        let mut columns = vec![];
        for i in 0..p.columns.len() {
            // Computes [(i, 0), (i, 1), ..., (i, n - 1)]
            columns.push((0..n).map(|j| (i as u32, j as u32)).collect());
        }

        // Before any equality constraints are applied, every cell in the permutation is
        // in a 1-cycle; therefore mapping and aux are identical, because every cell is
        // its own distinguished element.
        Assembly {
            columns: p.columns.clone(),
            mapping: columns.clone(),
            aux: columns,
            sizes: vec![vec![1usize; n]; p.columns.len()],
        }
    }

    pub(crate) fn copy(
        &mut self,
        left_column: Column<Any>,
        left_row: usize,
        right_column: Column<Any>,
        right_row: usize,
    ) -> Result<(), Error> {
        let left_column = self
            .columns
            .iter()
            .position(|c| c == &left_column)
            .ok_or(Error::ColumnNotInPermutation(left_column))?;
        let right_column = self
            .columns
            .iter()
            .position(|c| c == &right_column)
            .ok_or(Error::ColumnNotInPermutation(right_column))?;

        // Check bounds
        if left_row >= self.mapping[left_column].len()
            || right_row >= self.mapping[right_column].len()
        {
            return Err(Error::BoundsFailure);
        }

        // See book/src/design/permutation.md for a description of this algorithm.

        let mut left_cycle = self.aux[left_column][left_row];
        let mut right_cycle = self.aux[right_column][right_row];

        // If left and right are in the same cycle, do nothing.
        if left_cycle == right_cycle {
            return Ok(());
        }

        if self.sizes[left_cycle.0 as usize][left_cycle.1 as usize]
            < self.sizes[right_cycle.0 as usize][right_cycle.1 as usize]
        {
            std::mem::swap(&mut left_cycle, &mut right_cycle);
        }

        // Merge the right cycle into the left one.
        self.sizes[left_cycle.0 as usize][left_cycle.1 as usize] +=
            self.sizes[right_cycle.0 as usize][right_cycle.1 as usize];
        let mut i = right_cycle;
        loop {
            self.aux[i.0 as usize][i.1 as usize] = left_cycle;
            i = self.mapping[i.0 as usize][i.1 as usize];
            if i == right_cycle {
                break;
            }
        }

        let tmp = self.mapping[left_column][left_row];
        self.mapping[left_column][left_row] = self.mapping[right_column][right_row];
        self.mapping[right_column][right_row] = tmp;

        Ok(())
    }

    pub(crate) fn build_vk<C: CurveAffine>(
        self,
        params: &Params<C>,
        domain: &EvaluationDomain<C::Scalar>,
        p: &Argument,
    ) -> VerifyingKey<C> {
        // Compute [omega^0, omega^1, ..., omega^{params.n - 1}]
        let mut omega_powers = Vec::with_capacity(params.n as usize);
        {
            let mut cur = C::Scalar::one();
            for _ in 0..params.n {
                omega_powers.push(cur);
                cur *= &domain.get_omega();
            }
        }

        // Compute [omega_powers * \delta^0, omega_powers * \delta^1, ..., omega_powers * \delta^m]
        let mut delta_omegas = Vec::with_capacity(p.columns.len());
        {
            let mut cur = C::Scalar::one();
            for _ in 0..p.columns.len() {
                let mut omega_powers = omega_powers.clone();
                for o in &mut omega_powers {
                    *o *= &cur;
                }

                delta_omegas.push(omega_powers);

                cur *= &C::Scalar::DELTA;
            }
        }

        // Pre-compute commitments for the URS.
        let mut commitments = vec![];
        for i in 0..p.columns.len() {
            // Computes the permutation polynomial based on the permutation
            // description in the assembly.
            let mut permutation_poly = domain.empty_lagrange();
            for (j, p) in permutation_poly.iter_mut().enumerate() {
                let (permuted_i, permuted_j) = self.mapping[i][j];
                *p = delta_omegas[permuted_i as usize][permuted_j as usize];
            }

            // Compute commitment to permutation polynomial
            commitments.push(params.commit_lagrange(&permutation_poly).to_affine());
        }
        VerifyingKey { commitments }
    }

    pub(crate) fn build_pk<C: CurveAffine>(
        self,
        params: &Params<C>,
        domain: &EvaluationDomain<C::Scalar>,
        p: &Argument,
    ) -> ProvingKey<C> {
        // Compute [omega^0, omega^1, ..., omega^{params.n - 1}]
        let timer = start_timer!(|| "prepare delta_omegas");
        let mut deltas = vec![C::Scalar::one()];
        for _ in 1..p.columns.len() {
            deltas.push(C::Scalar::DELTA * deltas.last().unwrap());
        }

        let mut delta_omegas = vec![vec![]; p.columns.len()];
        let omega = domain.get_omega();
        delta_omegas.par_iter_mut().enumerate().for_each(|(i, x)| {
            x.push(deltas[i]);
            for _ in 1..params.n {
                x.push(omega * x.last().unwrap())
            }
        });
        end_timer!(timer);

        let timer = start_timer!(|| "prepare permutations");
        // Compute permutation polynomials, convert to coset form.
        let mut permutations = vec![];
        for i in 0..p.columns.len() {
            // Computes the permutation polynomial based on the permutation
            // description in the assembly.
            let mut permutation_poly = domain.empty_lagrange();

            parallelize(&mut permutation_poly, |permutation_poly, start| {
                permutation_poly.iter_mut().enumerate().for_each(|(j, p)| {
                    let j = start + j;
                    let (permuted_i, permuted_j) = self.mapping[i][j];
                    *p = delta_omegas[permuted_i as usize][permuted_j as usize];
                });
            });

            // Store permutation polynomial and precompute its coset evaluation
            permutations.push(permutation_poly.clone());
        }
        end_timer!(timer);

        let timer = start_timer!(|| "prepare poly");
        let polys: Vec<_> = permutations
            .par_iter()
            .map(|permutation_poly| domain.lagrange_to_coeff_st(permutation_poly.clone()))
            .collect();
        end_timer!(timer);

        #[cfg(not(feature = "cuda"))]
        let cosets = polys
            .par_iter()
            .map(|poly| domain.coeff_to_extended(poly.clone()))
            .collect();

        ProvingKey {
            permutations,
            polys,

            #[cfg(not(feature = "cuda"))]
            cosets,
        }
    }
}
