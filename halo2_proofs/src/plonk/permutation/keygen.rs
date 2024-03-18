use std::{
    cell::RefCell,
    collections::{BTreeSet, HashMap, HashSet},
    rc::Rc,
};

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

#[derive(Debug)]
pub(crate) struct ParallelAssembly {
    n: usize,
    columns: Vec<Column<Any>>,
    aux: Vec<Vec<Option<Rc<RefCell<Vec<(u32, u32)>>>>>>,
    cycles: BTreeSet<Rc<RefCell<Vec<(u32, u32)>>>>,
}

impl ParallelAssembly {
    pub(crate) fn new(n: usize, p: &Argument) -> Self {
        Self {
            n,
            columns: p.columns.clone(),
            aux: vec![vec![None; n]; p.columns.len()],
            cycles: BTreeSet::default(),
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
        if left_row >= self.n || right_row >= self.n {
            return Err(Error::BoundsFailure);
        }

        let left_cycle = self.aux[left_column][left_row].clone();
        let right_cycle = self.aux[right_column][right_row].clone();

        // If left and right are in the same cycle, do nothing.
        if left_cycle.is_some() && right_cycle.is_some() {
            if Rc::ptr_eq(left_cycle.as_ref().unwrap(), right_cycle.as_ref().unwrap()) {
                return Ok(());
            }
        }

        let left_cycle = left_cycle.unwrap_or_else(|| {
            let cycle = Rc::new(RefCell::new(vec![(left_column as u32, left_row as u32)]));

            self.aux[left_column][left_row] = Some(cycle.clone());
            cycle
        });
        let right_cycle = right_cycle.unwrap_or_else(|| {
            let cycle = Rc::new(RefCell::new(vec![(right_column as u32, right_row as u32)]));

            self.aux[right_column][right_row] = Some(cycle.clone());
            cycle
        });

        let (small_cycle, big_cycle) =
            if left_cycle.borrow().len() <= right_cycle.borrow_mut().len() {
                (left_cycle, right_cycle)
            } else {
                (right_cycle, left_cycle)
            };

        // merge small cycle into big cycle
        self.cycles.remove(&small_cycle);
        self.cycles.insert(big_cycle.clone());

        for (col, row) in small_cycle.borrow().iter() {
            self.aux[*col as usize][*row as usize] = Some(big_cycle.clone());
        }

        let mut small_cycle = Rc::try_unwrap(small_cycle).unwrap().into_inner();
        big_cycle.borrow_mut().append(&mut small_cycle);

        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct Assembly {
    pub(crate) mapping: Vec<Vec<(u32, u32)>>,
}

impl From<ParallelAssembly> for Assembly {
    fn from(assembly: ParallelAssembly) -> Self {
        let mut mapping: Vec<Vec<(u32, u32)>> = vec![];

        for i in 0..assembly.columns.len() {
            // Computes [(i, 0), (i, 1), ..., (i, n - 1)]
            mapping.push((0..assembly.n).map(|j| (i as u32, j as u32)).collect());
        }

        for cycle in assembly.cycles {
            let mut first = None;

            let mut cycle = cycle.borrow_mut();
            cycle.sort();

            let mut iter = cycle.iter().peekable();
            while let Some(origin) = iter.next() {
                if first.is_none() {
                    first = Some(*origin);
                }

                if let Some(permuted) = iter.peek() {
                    mapping[origin.0 as usize][origin.1 as usize] = **permuted;
                } else {
                    // It's last element
                    mapping[origin.0 as usize][origin.1 as usize] = first.unwrap();
                }
            }
        }

        Self { mapping }
    }
}

impl Assembly {
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

#[test]
fn test_parallel_assembly() {
    let col0 = Column::new(0, Any::Advice);
    let col1 = Column::new(1, Any::Advice);
    let col2 = Column::new(2, Any::Advice);

    let argument = Argument {
        columns: vec![col0, col1, col2],
    };

    let mut assembly = ParallelAssembly::new(2, &argument);

    // A <-> B
    assembly.copy(col0, 0, col1, 0).unwrap();

    // C <-> D
    assembly.copy(col1, 1, col2, 1).unwrap();

    // B <-> C
    assembly.copy(col1, 0, col1, 1).unwrap();

    let assembly: Assembly = assembly.into();

    assert_eq!(assembly.mapping[0][0], (1, 0));
    assert_eq!(assembly.mapping[0][1], (0, 1));
    assert_eq!(assembly.mapping[1][0], (1, 1));
    assert_eq!(assembly.mapping[1][1], (2, 1));
    assert_eq!(assembly.mapping[2][0], (2, 0));
    assert_eq!(assembly.mapping[2][1], (0, 0));
}
