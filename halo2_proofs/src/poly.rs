//! Contains utilities for performing arithmetic over univariate polynomials in
//! various forms, including computing commitments to them and provably opening
//! the committed polynomials at arbitrary points.

use crate::arithmetic::parallelize;
use crate::plonk::Assigned;

use ark_std::{end_timer, start_timer};
use group::ff::{BatchInvert, Field};
use pairing::arithmetic::FieldExt;
use rayon::iter::*;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::ops::{Add, Deref, DerefMut, Index, IndexMut, Mul, RangeFrom, RangeFull, Sub};

pub mod commitment;
mod domain;
mod msm;
pub mod multiopen;

pub use domain::*;
pub use msm::{PairMSM, MSM};

/// This is an error that could occur during proving or circuit synthesis.
// TODO: these errors need to be cleaned up
#[derive(Debug)]
pub enum Error {
    /// OpeningProof is not well-formed
    OpeningError,
    /// Caller needs to re-sample a point
    SamplingError,
}

/// The basis over which a polynomial is described.
pub trait Basis: Copy + Debug + Send + Sync {}

/// The polynomial is defined as coefficients
#[derive(Clone, Copy, Debug)]
pub struct Coeff;
impl Basis for Coeff {}

/// The polynomial is defined as coefficients of Lagrange basis polynomials
#[derive(Clone, Copy, Debug)]
pub struct LagrangeCoeff;
impl Basis for LagrangeCoeff {}

/// The polynomial is defined as coefficients of Lagrange basis polynomials in
/// an extended size domain which supports multiplication
#[derive(Clone, Copy, Debug)]
pub struct ExtendedLagrangeCoeff;
impl Basis for ExtendedLagrangeCoeff {}

/// The polynomial is defined as coefficients of Lagrange basis polynomials in
/// an extended size domain which supports multiplication
#[derive(Clone, Copy, Debug)]
pub struct PreparedExtendedLagrangeCoeff;
impl Basis for PreparedExtendedLagrangeCoeff {}

/// Represents a univariate polynomial defined over a field and a particular
/// basis.
#[derive(Clone, Debug)]
pub struct Polynomial<F, B> {
    pub values: Vec<F>,
    pub _marker: PhantomData<B>,
}

impl<F, B> Index<usize> for Polynomial<F, B> {
    type Output = F;

    fn index(&self, index: usize) -> &F {
        self.values.index(index)
    }
}

impl<F, B> IndexMut<usize> for Polynomial<F, B> {
    fn index_mut(&mut self, index: usize) -> &mut F {
        self.values.index_mut(index)
    }
}

impl<F, B> Index<RangeFrom<usize>> for Polynomial<F, B> {
    type Output = [F];

    fn index(&self, index: RangeFrom<usize>) -> &[F] {
        self.values.index(index)
    }
}

impl<F, B> IndexMut<RangeFrom<usize>> for Polynomial<F, B> {
    fn index_mut(&mut self, index: RangeFrom<usize>) -> &mut [F] {
        self.values.index_mut(index)
    }
}

impl<F, B> Index<RangeFull> for Polynomial<F, B> {
    type Output = [F];

    fn index(&self, index: RangeFull) -> &[F] {
        self.values.index(index)
    }
}

impl<F, B> IndexMut<RangeFull> for Polynomial<F, B> {
    fn index_mut(&mut self, index: RangeFull) -> &mut [F] {
        self.values.index_mut(index)
    }
}

impl<F, B> Deref for Polynomial<F, B> {
    type Target = [F];

    fn deref(&self) -> &[F] {
        &self.values[..]
    }
}

impl<F, B> DerefMut for Polynomial<F, B> {
    fn deref_mut(&mut self) -> &mut [F] {
        &mut self.values[..]
    }
}

impl<F, B> Polynomial<F, B> {
    pub fn new(values: Vec<F>) -> Self {
        Polynomial {
            values,
            _marker: PhantomData,
        }
    }
    /// Iterate over the values, which are either in coefficient or evaluation
    /// form depending on the basis `B`.
    pub fn iter(&self) -> impl Iterator<Item = &F> {
        self.values.iter()
    }

    /// Iterate over the values mutably, which are either in coefficient or
    /// evaluation form depending on the basis `B`.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut F> {
        self.values.iter_mut()
    }

    /// Gets the size of this polynomial in terms of the number of
    /// coefficients used to describe it.
    pub fn num_coeffs(&self) -> usize {
        self.values.len()
    }
}

pub(crate) fn batch_invert_assigned<F: FieldExt>(
    assigned: Vec<Polynomial<Assigned<F>, LagrangeCoeff>>,
) -> Vec<Polynomial<F, LagrangeCoeff>> {
    let mut assigned_denominators: Vec<_> = assigned
        .par_iter()
        .map(|f| {
            f.iter()
                .map(|value| value.denominator())
                .collect::<Vec<_>>()
        })
        .collect();

    assigned_denominators.par_iter_mut().for_each(|f| {
        f.iter_mut()
            // If the denominator is trivial, we can skip it, reducing the
            // size of the batch inversion.
            .filter_map(|d| d.as_mut())
            .batch_invert();
    });

    assigned
        .iter()
        .zip(assigned_denominators.into_iter())
        .map(|(poly, inv_denoms)| poly.invert(inv_denoms))
        .collect()
}

impl<F: Field> Polynomial<Assigned<F>, LagrangeCoeff> {
    pub(crate) fn invert(&self, inv_denoms: Vec<Option<F>>) -> Polynomial<F, LagrangeCoeff> {
        Polynomial {
            values: self
                .values
                .par_iter()
                .zip(inv_denoms.into_par_iter())
                .map(|(a, inv_den)| {
                    inv_den.map_or_else(|| a.numerator(), |inv_den| a.numerator() * inv_den)
                })
                .collect(),
            _marker: self._marker,
        }
    }
}

impl<'a, F: Field, B: Basis> Add<&'a Polynomial<F, B>> for Polynomial<F, B> {
    type Output = Polynomial<F, B>;

    fn add(mut self, rhs: &'a Polynomial<F, B>) -> Polynomial<F, B> {
        parallelize(&mut self.values, |lhs, start| {
            for (lhs, rhs) in lhs.iter_mut().zip(rhs.values[start..].iter()) {
                *lhs += *rhs;
            }
        });

        self
    }
}

impl<'a, F: Field, B: Basis> Sub<&'a Polynomial<F, B>> for Polynomial<F, B> {
    type Output = Polynomial<F, B>;

    fn sub(mut self, rhs: &'a Polynomial<F, B>) -> Polynomial<F, B> {
        parallelize(&mut self.values, |lhs, start| {
            for (lhs, rhs) in lhs.iter_mut().zip(rhs.values[start..].iter()) {
                *lhs -= *rhs;
            }
        });

        self
    }
}

impl<'a, F: Field> Polynomial<F, LagrangeCoeff> {
    /// Rotates the values in a Lagrange basis polynomial by `Rotation`
    pub fn rotate(&self, rotation: Rotation) -> Polynomial<F, LagrangeCoeff> {
        let mut values = self.values.clone();
        if rotation.0 < 0 {
            values.rotate_right((-rotation.0) as usize);
        } else {
            values.rotate_left(rotation.0 as usize);
        }
        Polynomial {
            values,
            _marker: PhantomData,
        }
    }
}

impl<'a, F: Field, B: Basis> Sub<F> for &'a Polynomial<F, B> {
    type Output = Polynomial<F, B>;

    fn sub(self, rhs: F) -> Polynomial<F, B> {
        let mut res = self.clone();
        res.values[0] -= rhs;
        res
    }
}

impl<'a, F: Field, B: Basis> Mul<F> for Polynomial<F, B> {
    type Output = Polynomial<F, B>;

    fn mul(mut self, rhs: F) -> Polynomial<F, B> {
        parallelize(&mut self.values, |lhs, _| {
            for lhs in lhs.iter_mut() {
                *lhs *= rhs;
            }
        });

        self
    }
}

/// Describes the relative rotation of a vector. Negative numbers represent
/// reverse (leftmost) rotations and positive numbers represent forward (rightmost)
/// rotations. Zero represents no rotation.
#[derive(Copy, Clone, Debug, PartialEq, PartialOrd, Eq, Ord)]
pub struct Rotation(pub i32);

impl Rotation {
    /// The current location in the evaluation domain
    pub fn cur() -> Rotation {
        Rotation(0)
    }

    /// The previous location in the evaluation domain
    pub fn prev() -> Rotation {
        Rotation(-1)
    }

    /// The next location in the evaluation domain
    pub fn next() -> Rotation {
        Rotation(1)
    }
}
