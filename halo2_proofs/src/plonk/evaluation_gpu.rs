use super::Expression;
use crate::multicore;
use crate::plonk::lookup::prover::Committed;
use crate::plonk::permutation::Argument;
use crate::plonk::{lookup, permutation, Any, ProvingKey};
use crate::poly::Basis;
use crate::{
    arithmetic::{eval_polynomial, parallelize, BaseExt, CurveAffine, FieldExt},
    poly::{
        commitment::Params, multiopen::ProverQuery, Coeff, EvaluationDomain, ExtendedLagrangeCoeff,
        LagrangeCoeff, Polynomial, Rotation,
    },
    transcript::{EncodedChallenge, TranscriptWrite},
};
use ark_std::{end_timer, start_timer};
use ec_gpu_gen::fft::FftKernel;
use ec_gpu_gen::rust_gpu_tools::cuda::Buffer;
use ec_gpu_gen::rust_gpu_tools::Device;
use group::prime::PrimeCurve;
use group::{
    ff::{BatchInvert, Field},
    Curve,
};
use std::any::TypeId;
use std::collections::BTreeSet;
use std::convert::TryInto;
use std::iter::FromIterator;
use std::num::ParseIntError;
use std::{cmp, slice};
use std::{
    collections::BTreeMap,
    iter,
    ops::{Index, Mul, MulAssign},
};

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum ProveExpressionUnit {
    /// This is a fixed column queried at a certain relative location
    Fixed {
        /// Column index
        column_index: usize,
        /// Rotation of this query
        rotation: Rotation,
    },
    /// This is an advice (witness) column queried at a certain relative location
    Advice {
        /// Column index
        column_index: usize,
        /// Rotation of this query
        rotation: Rotation,
    },
    /// This is an instance (external) column queried at a certain relative location
    Instance {
        /// Column index
        column_index: usize,
        /// Rotation of this query
        rotation: Rotation,
    },
}

#[derive(Clone, Debug)]
pub enum ProveExpression<F> {
    /// This is a constant polynomial
    Constant(F),
    Unit(ProveExpressionUnit),
    /// This is the sum of two polynomials
    Sum(Box<ProveExpression<F>>, Box<ProveExpression<F>>),
    /// This is the product of two polynomials
    Product(Box<ProveExpression<F>>, Box<ProveExpression<F>>),
    /// This is a scaled polynomial
    Scaled(Box<ProveExpression<F>>, F),
    Y(F, u32),
}

impl<F: FieldExt> ProveExpression<F> {
    pub(crate) fn _eval_gpu<C: CurveAffine<ScalarExt = F>>(
        &self,
        pk: &ProvingKey<C>,
        advice: Vec<&Vec<Polynomial<F, LagrangeCoeff>>>,
        instance: Vec<&Vec<Polynomial<F, LagrangeCoeff>>>,
        y: F,
    ) -> Buffer<F> {
        unimplemented!()
    }

    pub(crate) fn eval_gpu<C: CurveAffine<ScalarExt = F>>(
        &self,
        pk: &ProvingKey<C>,
        advice: Vec<&Vec<Polynomial<F, LagrangeCoeff>>>,
        instance: Vec<&Vec<Polynomial<F, LagrangeCoeff>>>,
        y: F,
    ) -> Polynomial<F, ExtendedLagrangeCoeff> {
        use pairing::bn256::Fr;

        let mut values = pk.vk.domain.empty_extended();

        let closures =
            ec_gpu_gen::rust_gpu_tools::program_closures!(
                |program, input: &mut [F]| -> ec_gpu_gen::EcResult<()> {
                    let values_buf = self._eval_gpu(pk, advice, instance, y);
                    program.read_into_buffer(&values_buf, input)?;
                    Ok(())
                }
            );

        let devices = Device::all();
        let programs = devices
            .iter()
            .map(|device| ec_gpu_gen::program!(device))
            .collect::<Result<_, _>>()
            .expect("Cannot create programs!");
        let kern = FftKernel::<Fr>::create(programs).expect("Cannot initialize kernel!");

        kern.kernels[0]
            .program
            .run(closures, &mut values.values[..])
            .unwrap();
        values
    }

    pub(crate) fn new() -> Self {
        Self::Constant(F::zero())
    }

    pub(crate) fn from_expr(e: &Expression<F>) -> Self {
        match e {
            Expression::Constant(x) => ProveExpression::Constant(*x),
            Expression::Selector(_) => unreachable!(),
            Expression::Fixed {
                column_index,
                rotation,
                ..
            } => Self::Unit(ProveExpressionUnit::Fixed {
                column_index: *column_index,
                rotation: *rotation,
            }),
            Expression::Advice {
                column_index,
                rotation,
                ..
            } => Self::Unit(ProveExpressionUnit::Advice {
                column_index: *column_index,
                rotation: *rotation,
            }),
            Expression::Instance {
                column_index,
                rotation,
                ..
            } => Self::Unit(ProveExpressionUnit::Instance {
                column_index: *column_index,
                rotation: *rotation,
            }),
            Expression::Negated(e) => {
                ProveExpression::Scaled(Box::new(Self::from_expr(e)), -F::one())
            }
            Expression::Sum(l, r) => {
                ProveExpression::Sum(Box::new(Self::from_expr(l)), Box::new(Self::from_expr(r)))
            }
            Expression::Product(l, r) => {
                ProveExpression::Product(Box::new(Self::from_expr(l)), Box::new(Self::from_expr(r)))
            }
            Expression::Scaled(l, r) => ProveExpression::Scaled(Box::new(Self::from_expr(l)), *r),
        }
    }

    pub(crate) fn add_gate(self, e: &Expression<F>) -> Self {
        Self::Sum(
            Box::new(Self::Product(
                Box::new(self),
                Box::new(Self::Y(F::zero(), 1)),
            )),
            Box::new(Self::from_expr(e)),
        )
    }

    fn ___reconstruct(coeff: (F, u32)) -> Self {
        Self::Y(coeff.0, coeff.1)
    }

    fn ____reconstruct(u: ProveExpressionUnit, c: u32) -> Self {
        if c == 1 {
            Self::Unit(u)
        } else {
            Self::Product(
                Box::new(Self::Unit(u.clone())),
                Box::new(Self::____reconstruct(u, c - 1)),
            )
        }
    }

    fn __reconstruct(mut us: BTreeMap<ProveExpressionUnit, u32>, coeff: (F, u32)) -> Self {
        if us.len() == 0 {
            Self::___reconstruct(coeff)
        } else {
            let p = us.pop_first().unwrap();
            Self::Product(
                Box::new(Self::____reconstruct(p.0, p.1)),
                Box::new(Self::__reconstruct(us, coeff)),
            )
        }
    }

    fn _reconstruct(mut tree: Vec<(BTreeMap<ProveExpressionUnit, u32>, (F, u32))>) -> Self {
        if tree.len() == 1 {
            let u = tree.pop().unwrap();
            return Self::__reconstruct(u.0, u.1);
        }

        // find max
        let mut map = BTreeMap::new();

        for (us, _) in tree.iter() {
            for (u, _) in us {
                if let Some(c) = map.get_mut(u) {
                    *c = *c + 1;
                } else {
                    map.insert(u, 1);
                }
            }
        }

        let mut max_u = (*map.first_entry().unwrap().key()).clone();
        let mut max_c = 0;

        for (u, c) in map {
            if c > max_c {
                max_c = c;
                max_u = u.clone();
            }
        }

        let mut l = vec![];
        let mut r = vec![];

        for (mut k, v) in tree {
            let c = k.remove(&max_u);
            match c {
                Some(1) => {
                    l.push((k, v));
                }
                Some(c) => {
                    k.insert(max_u.clone(), c - 1);
                    l.push((k, v));
                }
                None => {
                    r.push((k, v));
                }
            }
        }

        let l = Self::_reconstruct(l);
        let l = Self::Product(Box::new(Self::Unit(max_u)), Box::new(l));
        if r.len() == 0 {
            l
        } else {
            let r = Self::_reconstruct(r);
            Self::Sum(Box::new(l), Box::new(r))
        }
    }

    pub(crate) fn reconstruct(tree: BTreeMap<Vec<ProveExpressionUnit>, (F, u32)>) -> Self {
        let tree = tree
            .into_iter()
            .map(|(us, v)| {
                let mut map = BTreeMap::new();
                for u in us {
                    if let Some(c) = map.get_mut(&u) {
                        *c = *c + 1;
                    } else {
                        map.insert(u, 1);
                    }
                }
                (map, v)
            })
            .collect();

        Self::_reconstruct(tree)
    }

    pub(crate) fn get_degree(&self) -> (u32, u32) {
        match self {
            ProveExpression::Constant(_) => unreachable!(),
            ProveExpression::Unit(_) => (1, 0),
            ProveExpression::Sum(l, r) => {
                let l = l.get_degree();
                let r = r.get_degree();
                (l.0 + r.0 + 1, l.0 + r.0)
            }
            ProveExpression::Product(l, r) => {
                let l = l.get_degree();
                let r = r.get_degree();
                (l.0 + r.0 + 1, l.0 + r.0)
            }
            ProveExpression::Scaled(_, _) => unreachable!(),
            ProveExpression::Y(_, _) => (0, 0),
        }
    }

    // u32 is order of y
    pub(crate) fn flatten(self) -> BTreeMap<Vec<ProveExpressionUnit>, (F, u32)> {
        match self {
            ProveExpression::Constant(c) => {
                if c == F::zero() {
                    BTreeMap::from_iter(vec![(vec![], (c, 0))].into_iter())
                } else {
                    BTreeMap::new()
                }
            }
            ProveExpression::Unit(u) => {
                BTreeMap::from_iter(vec![(vec![u], (F::one(), 0))].into_iter())
            }
            ProveExpression::Sum(l, r) => {
                let mut l = l.flatten();
                let r = r.flatten();

                for (rk, (f, y_order)) in r.into_iter() {
                    if let Some(coeff) = l.get_mut(&rk) {
                        coeff.0 *= f;
                        coeff.1 += y_order;
                    } else {
                        l.insert(rk, (f, y_order));
                    }
                }
                l
            }
            ProveExpression::Product(l, r) => {
                let l = l.flatten();
                let r = r.flatten();

                let mut res = BTreeMap::<Vec<_>, (_, _)>::new();

                for (lk, (lf, ly_order)) in l.into_iter() {
                    for (rk, (rf, ry_order)) in r.clone().into_iter() {
                        let mut k = vec![lk.clone(), rk.clone()].concat();
                        k.sort();
                        let f = rf * lf;
                        let y_order = ry_order + ly_order;
                        if let Some(coeff) = res.get_mut(&k) {
                            coeff.0 *= f;
                            coeff.1 += y_order;
                        } else {
                            res.insert(k, (f, y_order));
                        }
                    }
                }
                res
            }
            ProveExpression::Scaled(l, r) => {
                let mut l = l.flatten();

                for l in l.iter_mut() {
                    l.1 .0 = l.1 .0 * r;
                }
                l
            }
            ProveExpression::Y(f, order) => {
                BTreeMap::from_iter(vec![(vec![], (f, order))].into_iter())
            }
        }
    }
}
