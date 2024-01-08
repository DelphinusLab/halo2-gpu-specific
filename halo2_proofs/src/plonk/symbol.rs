use super::Expression;
use crate::multicore;
use crate::plonk::{lookup, permutation, Any, ProvingKey};
use crate::poly::Basis;
use crate::arithmetic::FieldExt;
use crate::poly::Rotation;

/*
use crate::{
    arithmetic::{eval_polynomial, parallelize, BaseExt, CurveAffine, FieldExt},
    poly::{
        commitment::Params, multiopen::ProverQuery, Coeff, EvaluationDomain, ExtendedLagrangeCoeff,
        LagrangeCoeff, Polynomial, Rotation,
    },
    transcript::{EncodedChallenge, TranscriptWrite},
};
*/
use ark_std::{end_timer, start_timer};
use group::prime::PrimeCurve;
use group::{
    ff::{BatchInvert, Field},
    Curve,
};
use std::collections::{BTreeSet, HashMap, LinkedList, HashSet};
use std::convert::TryInto;
use std::iter::FromIterator;
use std::num::ParseIntError;
use std::{cmp, slice};
use std::{
    collections::BTreeMap,
    iter,
    ops::{Index, Mul, MulAssign},
};

#[derive(Clone, Debug)]
pub enum Bop {
    Sum,
    Product,
}

#[derive(Debug)]
pub(crate) struct ComplexityProfiler {
    mul: usize,
    sum: usize,
    scale: usize,
    unit: usize,
    y: usize,
    pub(crate) ref_cnt: HashMap<usize, u32>,
}

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

impl ProveExpressionUnit {
    pub fn get_group(&self) -> usize {
        match self {
            ProveExpressionUnit::Fixed { column_index, .. } => column_index << 2,
            ProveExpressionUnit::Advice { column_index, .. } => (column_index << 2) + 1,
            ProveExpressionUnit::Instance { column_index, .. } => (column_index << 2) + 2,
        }
    }

    pub fn get_rotation(&self) -> i32 {
        match self {
            ProveExpressionUnit::Fixed { rotation , .. } => rotation.0,
            ProveExpressionUnit::Advice { rotation, .. } => rotation.0,
            ProveExpressionUnit::Instance { rotation, .. } => rotation.0,
        }
    }

    pub fn is_fix(&self) -> bool {
        match self {
            ProveExpressionUnit::Fixed { .. } => true,
            _ => false
        }

    }

    pub fn to_string(&self) -> String {
        match self {
            ProveExpressionUnit::Fixed { column_index, rotation } => format!("f{}-{}", column_index, rotation.0),
            ProveExpressionUnit::Advice { column_index, rotation } => format!("a{}-{}", column_index, rotation.0),
            ProveExpressionUnit::Instance { column_index, rotation } => format!("i{}-{}", column_index, rotation.0),
        }
    }

    pub fn key_to_string(key: usize) -> String {
        let typ = key & 0x3;
        let column_index = key >> 2;
        match typ {
            0 => format!("f{}", column_index),
            1 => format!("a{}", column_index),
            2 => format!("i{}", column_index),
            _ => unreachable!("invalid typ")
        }
    }
}

#[derive(Clone, Debug)]
pub enum ProveExpression<F> {
    Unit(ProveExpressionUnit),
    Op(Box<ProveExpression<F>>, Box<ProveExpression<F>>, Bop),
    Y(BTreeMap<u32, F>), // gate:a=0, gate:2a=0 --> a + y*2a + y^2*3a = a*(f(y))
    Scale(Box<ProveExpression<F>>, BTreeMap<u32, F>),
}

impl<F:FieldExt> ProveExpression<F> {
    pub fn to_string(&self) -> String {
        let c = if let Some ((uid, exprs)) = self.flat_unique_unit_scale() {
            if exprs.len() > 1 {
                Some(format!("Smix({})", uid.to_string()))
            } else {
                None
            }
        } else {
            None
        };
        c.map_or(
            match &self {
                ProveExpression::Unit(a) => {
                    format!("u({})", a.to_string())
                },
                ProveExpression::Op(a, b, bop) => {
                    match bop {
                        Bop::Sum => format!("({} + {})", a.to_string(), b.to_string()),
                        Bop::Product => format!("({} * {})", a.to_string(), b.to_string()),
                    }
                },
                ProveExpression::Y(_) => format!("Y"),
                ProveExpression::Scale(a, _b) => format!("S({})", a.to_string()),
            },
            |x| x
        )
    }
    pub fn string_of_bundle(v: &Vec<ProveExpressionUnit>) -> String {
        let mut full = "".to_string();
        for e in v.iter() {
            full = format!("{}{}", e.to_string(), full);
        }
        format!("<{}>", full)
    }
    pub fn prefetch_units(&self, cache: &mut BTreeMap<usize, ProveExpression<F>>) {
        match &self {
            ProveExpression::Unit(a) => {
                if let None = cache.get(&a.get_group()) {
                    cache.insert(a.get_group(), ProveExpression::Unit(a.clone()));
                }
            },
            ProveExpression::Op(a, b, _) => {
                a.prefetch_units(cache);
                b.prefetch_units(cache);
            },
            ProveExpression::Y(_) => (),
            ProveExpression::Scale(a, _b) => a.prefetch_units(cache)
        }
    }
    pub fn calculate_units(&self, cache: &mut BTreeMap<usize, usize>) {
        match &self {
            ProveExpression::Unit(a) => {
                match cache.get(&a.get_group()) {
                    None => cache.insert(a.get_group(), 1),
                    Some(k) => cache.insert(a.get_group(), k+1),
                };
            },
            ProveExpression::Op(a, b, _) => {
                a.calculate_units(cache);
                b.calculate_units(cache);
            },
            ProveExpression::Y(_) => (),
            ProveExpression::Scale(a, _b) => a.calculate_units(cache)
        }
    }

    pub fn calculate_balances(&self) -> HashSet<usize> {
        match &self {
            ProveExpression::Unit(a) => {
                let mut s = HashSet::new();
                s.insert(a.get_group());
                s
            },
            ProveExpression::Op(a, b, _) => {
                let sa = a.calculate_balances();
                let sb = b.calculate_balances();
                let si = sa.intersection(&sb).count();
                sa.union(&sb).collect::<HashSet<_>>()
                    .into_iter()
                    .map(|x| *x)
                    .collect()
            },
            ProveExpression::Y(_) => HashSet::new(),
            ProveExpression::Scale(a, _b) => a.calculate_balances()
        }
    }

    pub fn flat_unique_unit_scale(&self) -> Option<(Self, Vec<Self>)> {
        match &self {
            ProveExpression::Op(a, b, Bop::Sum) => {
                if let Some ((ua, mut va)) = a.flat_unique_unit_scale() {
                    if let Some ((ub, mut vb)) = b.flat_unique_unit_scale() {
                        if ua.get_unit().unwrap().get_group() == ub.get_unit().unwrap().get_group() {
                            va.append(&mut vb);
                            Some ((ua, va))
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                } else {
                    None
                }
            },
            ProveExpression::Scale(a, _) => match a.as_ref().clone() {
                ProveExpression::Unit(_) => Some ((a.as_ref().clone(), vec![self.clone()])),
                _ => None
            },
            _ => None
        }
    }

    pub fn get_unit(&self) -> Option<ProveExpressionUnit> {
        match &self {
            ProveExpression::Unit(a) => {
                Some(a.clone())
            },
            _ => None
        }
    }

    pub fn get_unit_rotation(&self) -> Option<i32> {
        match &self {
            ProveExpression::Unit(a) => {
                Some(a.get_rotation())
            },
            ProveExpression::Scale(a, ys) => a.get_unit_rotation(),
            _ => None
        }
    }

    // ys = 0, y, y^2, ... y^k
    // y is the scale coeff cache
    // scale = y[key] * v where k, v in ys
    pub fn get_scale_coeff(&self, y: &mut Vec<F>) -> Option<F> {
        match &self {
            ProveExpression::Scale(a, ys) => {
                    let max_y_order = ys.keys().max().unwrap();
                    for _ in (y.len() as u32)..max_y_order + 1 {
                        y.push(y[1] * y.last().unwrap());
                    }

                    let c = ys.iter().fold(F::zero(), |acc, (y_order, f)| {
                        acc + y[*y_order as usize] * f
                    });
                    Some (c)
            },
            _ => None,
        }
    }


    pub fn depth(&self) -> usize {
        match &self {
            ProveExpression::Unit(a) => {
                1
            },
            ProveExpression::Op(a, b, _) => {
                let sa = a.depth();
                let sb = b.depth();
                if sa >= sb {
                    //println!("left is heavier {} {}", sa, sb);
                    sa + 1
                }
                else {
                    //println!("right is heavier {} {}", sa, sb);
                    sb + 1
                }
            },
            ProveExpression::Y(_) => 0,
            ProveExpression::Scale(a, _b) => a.depth()
        }
    }

    fn get_atomic_unit(&self) -> Option<Self> {
        match &self {
            ProveExpression::Unit(a) => Some(self.clone()),
            ProveExpression::Op(_, _, _) => None,
            ProveExpression::Y(_) => None,
            ProveExpression::Scale(a, _) => a.get_atomic_unit()
        }
    }

    fn get_unit_group(&self) -> Option<usize>{
        self.get_atomic_unit().map(|x| x.get_unit().unwrap().get_group())
    }
}


impl<F: FieldExt> ProveExpression<F> {
    pub(crate) fn new() -> Self {
        ProveExpression::Y(BTreeMap::from_iter(vec![(0, F::zero())].into_iter()))
    }

    pub(crate) fn from_expr(e: &Expression<F>) -> Self {
        match e {
            Expression::Constant(x) => {
                ProveExpression::Y(BTreeMap::from_iter(vec![(0, *x)].into_iter()))
            }
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
            Expression::Negated(e) => ProveExpression::Op(
                Box::new(Self::from_expr(e)),
                Box::new(ProveExpression::Y(BTreeMap::from_iter(
                    vec![(0, -F::one())].into_iter(),
                ))),
                Bop::Product,
            ),
            Expression::Sum(l, r) => ProveExpression::Op(
                Box::new(Self::from_expr(l)),
                Box::new(Self::from_expr(r)),
                Bop::Sum,
            ),
            Expression::Product(l, r) => ProveExpression::Op(
                Box::new(Self::from_expr(l)),
                Box::new(Self::from_expr(r)),
                Bop::Product,
            ),
            Expression::Scaled(l, r) => ProveExpression::Op(
                Box::new(Self::from_expr(l)),
                Box::new(ProveExpression::Y(BTreeMap::from_iter(
                    vec![(0, *r)].into_iter(),
                ))),
                Bop::Product,
            ),
        }
    }

    pub(crate) fn add_gate(self, e: &Expression<F>) -> Self {
        Self::Op(
            Box::new(Self::Op(
                Box::new(self),
                Box::new(ProveExpression::Y(BTreeMap::from_iter(
                    vec![(1, F::one())].into_iter(),
                ))),
                Bop::Product,
            )),
            Box::new(Self::from_expr(e)),
            Bop::Sum,
        )
    }

    // max r deep: 0
    fn reconstruct_coeff(coeff: BTreeMap<u32, F>) -> Self {
        ProveExpression::Y(coeff)
    }

    // max r deep: 1
    fn reconstruct_unit(u: ProveExpressionUnit, c: u32) -> Self {
        if c >= 3 {
            println!("find large c {}", c);
        }

        if c == 1 {
            Self::Unit(u)
        } else {
            Self::Op(
                Box::new(Self::reconstruct_unit(u.clone(), c - 1)),
                Box::new(Self::Unit(u)),
                Bop::Product,
            )
        }
    }

    // max r deep: 1
    fn reconstruct_units(mut us: BTreeMap<ProveExpressionUnit, u32>) -> Self {
        let u = us.pop_first().unwrap();

        let mut l = Self::reconstruct_unit(u.0, u.1);

        for (u, c) in us {
            for _ in 0..c {
                l = Self::Op(Box::new(l), Box::new(Self::Unit(u.clone())), Bop::Product);
            }
        }

        l
    }

    // max r deep: 1
    fn reconstruct_units_coeff(
        us: BTreeMap<ProveExpressionUnit, u32>,
        coeff: BTreeMap<u32, F>,
    ) -> Self {
        let res = if us.len() == 0 {
            Self::reconstruct_coeff(coeff)
        } else {
            Self::Scale(Box::new(Self::reconstruct_units(us)), coeff)
        };

        assert!(res.get_r_deep() <= 1);
        res
    }

    fn is_scale_with_same_unit(
        acc: &ProveExpression<F>,
        x: &ProveExpression<F>
    ) -> bool {
        match acc {
            Self::Scale(a, _) => {
                match (a.as_ref().clone(), x.clone()) {
                    (Self::Unit(a), Self::Scale(ub, _)) => {
                        match ub.as_ref().clone() {
                            Self::Unit(b) => if a.get_group() == b.get_group() {
                                println!("find merge {} {}", a.to_string(), b.to_string());
                                true
                            } else {
                                false
                            },
                            _ => false
                        }
                    },
                    _ => false
                }
            },
            _ => false
        }
    }


    fn reconstruct_tree(
        mut tree: Vec<(BTreeMap<ProveExpressionUnit, u32>, BTreeMap<u32, F>)>,
        r_deep_limit: u32,
    ) -> Self {
        if tree.len() == 1 {
            let u = tree.pop().unwrap();
            return Self::reconstruct_units_coeff(u.0, u.1);
        }

        if r_deep_limit > 2 {
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

            if max_c > 1 {
                let mut picked = vec![];
                let mut other = vec![];

                for (mut k, v) in tree {
                    let c = k.remove(&max_u);
                    match c {
                        Some(1) => {
                            picked.push((k, v));
                        }
                        Some(c) => {
                            k.insert(max_u.clone(), c - 1);
                            picked.push((k, v));
                        }
                        None => {
                            other.push((k, v));
                        }
                    }
                }

                let picked = Self::reconstruct_tree(picked, r_deep_limit - 1);
                let mut r = Self::Op(Box::new(picked), Box::new(Self::Unit(max_u)), Bop::Product);

                if other.len() > 0 {
                    r = Self::Op(
                        Box::new(Self::reconstruct_tree(other, r_deep_limit)),
                        Box::new(r),
                        Bop::Sum,
                    );
                }

                return r;
            }
        }

        let c = tree.clone()
            .into_iter()
            .map(|(k, ys)| Self::reconstruct_units_coeff(k, ys))
            .collect::<Vec<_>>();

        let mut atomic_group:HashMap<usize, Self> = HashMap::new();
	    let mut non_atomic = None;

        for e in c.into_iter() {
            if let Some(g) = e.get_unit_group() { 
                if let Some(es) = atomic_group.get(&g) {
                    atomic_group.insert(g, Self::Op(Box::new(es.clone()), Box::new(e), Bop::Sum));
                } else {
                    atomic_group.insert(g, e);
                }
            } else {
                non_atomic = non_atomic.map_or(
                    Some(e.clone()), |x| {
                        Some(Self::Op(Box::new(x), Box::new(e), Bop::Sum))
                    }
                );
            }
        }

        let mut acc = None;
        for v in atomic_group.values() {
            acc = acc.map_or(
                Some(v.clone()),
                |x| {
                    Some(Self::Op(Box::new(x), Box::new(v.clone()), Bop::Sum))
                }
            )
        }
        acc = acc.map_or(
            non_atomic.clone(),
            |x| {
                non_atomic.map_or(
                    Some(x.clone()),
                    |y| Some(Self::Op(Box::new(x), Box::new(y), Bop::Sum))
                )
            }
        );

        acc.unwrap()
    }

    pub(crate) fn reconstruct(tree: &[(Vec<ProveExpressionUnit>, BTreeMap<u32, F>)]) -> Self {
        let tree = tree
            .into_iter()
            .map(|(us, v)| {
                let mut map = BTreeMap::new();
                for u in us {
                    if let Some(c) = map.get_mut(u) {
                        *c = *c + 1;
                    } else {
                        map.insert(u.clone(), 1);
                    }
                }
                (map, v.clone())
            })
            .collect();

        let r_deep = std::env::var("HALO2_PROOF_GPU_EVAL_R_DEEP").unwrap_or("6".to_owned());
        let r_deep = u32::from_str_radix(&r_deep, 10).expect("Invalid HALO2_PROOF_GPU_EVAL_R_DEEP");
        Self::reconstruct_tree(tree, r_deep)
    }

    pub(crate) fn get_complexity(&self) -> ComplexityProfiler {
        match self {
            ProveExpression::Unit(u) => ComplexityProfiler {
                mul: 0,
                sum: 0,
                scale: 0,
                unit: 1,
                y: 0,
                ref_cnt: HashMap::from_iter(vec![(u.get_group(), 1)]),
            },
            ProveExpression::Op(l, r, op) => {
                let mut l = l.get_complexity();
                let r = r.get_complexity();
                for (k, v) in r.ref_cnt {
                    if let Some(lv) = l.ref_cnt.get_mut(&k) {
                        *lv += v;
                    } else {
                        l.ref_cnt.insert(k, v);
                    }
                }
                l.scale += r.scale;
                l.mul += r.mul;
                l.sum += r.sum;
                l.y += r.y;
                l.unit += r.unit;
                match op {
                    Bop::Sum => l.sum += 1,
                    Bop::Product => l.mul += 1,
                };
                l
            }
            ProveExpression::Y(_) => ComplexityProfiler {
                mul: 0,
                sum: 0,
                scale: 0,
                unit: 0,
                y: 1,
                ref_cnt: HashMap::from_iter(vec![]),
            },
            ProveExpression::Scale(l, _) => {
                let mut l = l.get_complexity();
                l.scale += 1;
                l
            }
        }
    }

    pub(crate) fn get_r_deep(&self) -> u32 {
        match self {
            ProveExpression::Unit(_) => 0,
            ProveExpression::Op(l, r, _) => {
                let l = l.get_r_deep();
                let r = r.get_r_deep();
                u32::max(l, r + 1)
            }
            ProveExpression::Y(_) => 0,
            ProveExpression::Scale(l, _) => {
                let l = l.get_r_deep();
                u32::max(l, 1)
            }
        }
    }

    fn ys_add_assign(l: &mut BTreeMap<u32, F>, r: BTreeMap<u32, F>) {
        for r in r {
            if let Some(f) = l.get_mut(&r.0) {
                *f = r.1 + &*f;
            } else {
                l.insert(r.0, r.1);
            }
        }
    }

    fn ys_mul(l: &BTreeMap<u32, F>, r: &BTreeMap<u32, F>) -> BTreeMap<u32, F> {
        let mut res = BTreeMap::new();

        for l in l {
            for r in r {
                let order = l.0 + r.0;
                let f = *l.1 * r.1;
                if let Some(origin_f) = res.get_mut(&order) {
                    *origin_f = f + &*origin_f;
                } else {
                    res.insert(order, f);
                }
            }
        }

        res
    }

    // u32 is order of y
    pub(crate) fn flatten(self) -> BTreeMap<Vec<ProveExpressionUnit>, BTreeMap<u32, F>> {
        match self {
            ProveExpression::Unit(u) => BTreeMap::from_iter(
                vec![(
                    vec![u],
                    BTreeMap::from_iter(vec![(0, F::one())].into_iter()),
                )]
                .into_iter(),
            ),
            ProveExpression::Op(l, r, Bop::Sum) => {
                let mut l = l.flatten();
                let r = r.flatten();

                for (rk, rys) in r.into_iter() {
                    if let Some(lys) = l.get_mut(&rk) {
                        Self::ys_add_assign(lys, rys);
                    } else {
                        l.insert(rk, rys);
                    }
                }
                l
            }
            ProveExpression::Op(l, r, Bop::Product) => {
                let l = l.flatten();
                let r = r.flatten();

                let mut res = BTreeMap::new();

                for (lk, lys) in l.into_iter() {
                    for (rk, rys) in r.clone().into_iter() {
                        let mut k = vec![lk.clone(), rk.clone()].concat();
                        k.sort();
                        let ys = Self::ys_mul(&lys, &rys);
                        if let Some(origin_ys) = res.get_mut(&k) {
                            Self::ys_add_assign(origin_ys, ys);
                        } else {
                            res.insert(k, ys);
                        }
                    }
                }
                res
            }
            ProveExpression::Y(ys) => BTreeMap::from_iter(vec![(vec![], ys)].into_iter()),
            ProveExpression::Scale(_, _) => unreachable!(),
        }
    }
}
impl<F:FieldExt> ProveExpression<F> {
    pub(crate) fn disjoint(es: &Vec<ProveExpressionUnit>, src: &Vec<ProveExpressionUnit>) -> bool {
       let mut disjoint = true;
       for e in src {
           if es.iter().find(|x| x.get_group() == e.get_group() && !e.is_fix()).is_some() {
               disjoint = false;
               break
           }
       }
       disjoint
    }

    pub(crate) fn disjoint_group(gs: &Vec<(Vec<ProveExpressionUnit>, BTreeMap<u32, F>)>, src: &Vec<ProveExpressionUnit>) -> bool {
        let mut disjoint = true;
        for g in gs {
            if Self::disjoint(&g.0, src) {
                disjoint = false;
                break;
            }
        }
        disjoint
    }

    pub(crate) fn mk_group(tree: &Vec<(Vec<ProveExpressionUnit>, BTreeMap<u32, F>)>) -> Vec<Vec<(Vec<ProveExpressionUnit>, BTreeMap<u32, F>)>> {
        let mut vs: Vec<Vec<(Vec<ProveExpressionUnit>, BTreeMap<u32, F>)>> = vec![];
        for (es, m) in tree {
            let mut ns = vec![];
            let mut c = vec![(es.clone(), m.clone())];
            for s in vs.clone().into_iter() {
                if Self::disjoint_group(&s, es) {
                    c.append(&mut s.clone());
                } else {
                    ns.push(s.clone());
                }
            }
            ns.push(c);
            vs = ns
        }
        println!("total vs group is {}", vs.len());
        vs
    }

}
