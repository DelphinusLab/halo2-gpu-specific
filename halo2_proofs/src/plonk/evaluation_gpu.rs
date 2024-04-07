use super::Expression;
use crate::multicore;
use crate::plonk::evaluation::LcChallenge;
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
use group::prime::PrimeCurve;
use group::{
    ff::{BatchInvert, Field},
    Curve,
};
use std::any::TypeId;
use std::collections::{BTreeSet, HashMap, LinkedList};
use std::convert::TryInto;
use std::iter::FromIterator;
use std::num::ParseIntError;
use std::rc::Rc;
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

impl ProveExpressionUnit {
    pub fn get_group(&self) -> usize {
        match self {
            ProveExpressionUnit::Fixed { column_index, .. } => column_index << 2,
            ProveExpressionUnit::Advice { column_index, .. } => (column_index << 2) + 1,
            ProveExpressionUnit::Instance { column_index, .. } => (column_index << 2) + 2,
        }
    }
}

#[derive(Clone, Debug)]
pub enum Bop {
    Sum,
    Product,
}

#[derive(Clone, Debug)]
pub enum ProveExpression<F> {
    Unit(ProveExpressionUnit),
    Op(Box<ProveExpression<F>>, Box<ProveExpression<F>>, Bop),
    Y(BTreeMap<u32, F>),
    Scale(Box<ProveExpression<F>>, BTreeMap<u32, F>),
}

#[derive(Clone, Debug)]
pub enum LookupProveExpression<F> {
    Expression(ProveExpression<F>),
    LcTheta(Box<LookupProveExpression<F>>, Box<LookupProveExpression<F>>),
    //(a+challenge^p)*b
    LcChallenge(
        Box<LookupProveExpression<F>>,
        Box<LookupProveExpression<F>>,
        LcChallenge,
        usize,
    ),
    //a+challenge
    AddChallenge(Box<LookupProveExpression<F>>, LcChallenge),
}

#[cfg(feature = "cuda")]
impl<F: FieldExt> LookupProveExpression<F> {
    pub(crate) fn _eval_gpu<C: CurveAffine<ScalarExt = F>>(
        &self,
        pk: &ProvingKey<C>,
        program: &Program,
        advice: &Vec<Polynomial<F, Coeff>>,
        instance: &Vec<Polynomial<F, Coeff>>,
        y: &mut Vec<F>,
        beta: F,
        theta: F,
        gamma: F,
        unit_cache: &mut Cache<Buffer<F>>,
        allocator: &mut LinkedList<Buffer<F>>,
        helper: &mut ExtendedFFTHelper<F>,
    ) -> EcResult<(Rc<Buffer<F>>, i32)> {
        let size = 1u32 << pk.vk.domain.extended_k();
        let local_work_size = 128;
        let global_work_size = size / local_work_size;

        match self {
            LookupProveExpression::Expression(e) => e._eval_gpu_buffer(
                pk, program, advice, instance, y, unit_cache, allocator, helper,
            ),
            LookupProveExpression::LcTheta(l, r) => {
                let l = l._eval_gpu(
                    pk, program, advice, instance, y, beta, theta, gamma, unit_cache, allocator,
                    helper,
                )?;
                let r = r._eval_gpu(
                    pk, program, advice, instance, y, beta, theta, gamma, unit_cache, allocator,
                    helper,
                )?;
                let res = if r.1 == 0 && Rc::strong_count(&r.0) == 1 {
                    r.0.clone()
                } else if l.1 == 0 && Rc::strong_count(&l.0) == 1 {
                    l.0.clone()
                } else {
                    Rc::new(allocator.pop_front().unwrap_or_else(|| unsafe {
                        program.create_buffer::<F>(size as usize).unwrap()
                    }))
                };
                let theta = program.create_buffer_from_slice(&vec![theta])?;
                let kernel_name = format!("{}_eval_lctheta", "Bn256_Fr");
                //let timer = start_timer!(|| kernel_name.clone());
                let kernel = program.create_kernel(
                    &kernel_name,
                    global_work_size as usize,
                    local_work_size as usize,
                )?;
                kernel
                    .arg(res.as_ref())
                    .arg(l.0.as_ref())
                    .arg(r.0.as_ref())
                    .arg(&l.1)
                    .arg(&r.1)
                    .arg(&size)
                    .arg(&theta)
                    .run()?;

                if Rc::strong_count(&l.0) == 1 {
                    allocator.push_back(Rc::try_unwrap(l.0).unwrap())
                }

                if Rc::strong_count(&r.0) == 1 {
                    allocator.push_back(Rc::try_unwrap(r.0).unwrap())
                }

                //end_timer!(timer);
                Ok((res, 0))
            }
            LookupProveExpression::LcChallenge(l, r, lcx, p) => {
                let l = l._eval_gpu(
                    pk, program, advice, instance, y, beta, theta, gamma, unit_cache, allocator,
                    helper,
                )?;
                let r = r._eval_gpu(
                    pk, program, advice, instance, y, beta, theta, gamma, unit_cache, allocator,
                    helper,
                )?;
                let res = if r.1 == 0 && Rc::strong_count(&r.0) == 1 {
                    r.0.clone()
                } else if l.1 == 0 && Rc::strong_count(&l.0) == 1 {
                    l.0.clone()
                } else {
                    Rc::new(allocator.pop_front().unwrap_or_else(|| unsafe {
                        program.create_buffer::<F>(size as usize).unwrap()
                    }))
                };
                let mut x = match lcx {
                    LcChallenge::Beta => beta,
                    LcChallenge::Gamma => gamma,
                };
                if *p > 1 {
                    x = x.pow(&[*p as u64, 0, 0, 0]);
                }
                let beta = program.create_buffer_from_slice(&vec![x])?;
                let kernel_name = format!("{}_eval_lcbeta", "Bn256_Fr");
                //let timer = start_timer!(|| kernel_name.clone());
                let kernel = program.create_kernel(
                    &kernel_name,
                    global_work_size as usize,
                    local_work_size as usize,
                )?;
                kernel
                    .arg(res.as_ref())
                    .arg(l.0.as_ref())
                    .arg(r.0.as_ref())
                    .arg(&l.1)
                    .arg(&r.1)
                    .arg(&size)
                    .arg(&beta)
                    .run()?;

                if Rc::strong_count(&l.0) == 1 {
                    allocator.push_back(Rc::try_unwrap(l.0).unwrap())
                }

                if Rc::strong_count(&r.0) == 1 {
                    allocator.push_back(Rc::try_unwrap(r.0).unwrap())
                }
                //end_timer!(timer);
                Ok((res, 0))
            }
            LookupProveExpression::AddChallenge(l, lcx) => {
                let l = l._eval_gpu(
                    pk, program, advice, instance, y, beta, theta, gamma, unit_cache, allocator,
                    helper,
                )?;
                let res = if l.1 == 0 && Rc::strong_count(&l.0) == 1 {
                    l.0.clone()
                } else {
                    Rc::new(allocator.pop_front().unwrap_or_else(|| unsafe {
                        program.create_buffer::<F>(size as usize).unwrap()
                    }))
                };
                let x = match lcx {
                    LcChallenge::Beta => beta,
                    LcChallenge::Gamma => gamma,
                };
                let gamma = program.create_buffer_from_slice(&vec![x])?;
                let kernel_name = format!("{}_eval_addgamma", "Bn256_Fr");
                //let timer = start_timer!(|| kernel_name.clone());
                let kernel = program.create_kernel(
                    &kernel_name,
                    global_work_size as usize,
                    local_work_size as usize,
                )?;
                kernel
                    .arg(res.as_ref())
                    .arg(l.0.as_ref())
                    .arg(&l.1)
                    .arg(&size)
                    .arg(&gamma)
                    .run()?;

                if Rc::strong_count(&l.0) == 1 {
                    allocator.push_back(Rc::try_unwrap(l.0).unwrap())
                }

                //end_timer!(timer);
                Ok((res, 0))
            }
        }
    }
}

#[cfg(feature = "cuda")]
use ec_gpu_gen::{
    fft::FftKernel, rust_gpu_tools::cuda::Buffer, rust_gpu_tools::cuda::Program,
    rust_gpu_tools::Device, rust_gpu_tools::LocalBuffer, EcResult,
};

#[cfg(feature = "cuda")]
#[derive(Debug, PartialEq, Clone)]
pub(crate) enum CacheAction {
    Cache,
    Drop,
}

#[cfg(feature = "cuda")]
pub(crate) struct Cache<T> {
    data: BTreeMap<usize, (Rc<T>, usize)>,
    ts: usize,
    bound: usize,
    hit: usize,
    miss: usize,
    access: Vec<(usize, CacheAction)>,
}

#[cfg(feature = "cuda")]
impl<T> Cache<T> {
    pub fn access(&mut self, k: usize) {
        self.access.push((k, CacheAction::Cache));
    }

    pub fn analyze(&mut self) {
        let mut to_update = true;
        let mut try_count = 100000;
        let timer = start_timer!(|| "cache policy analysis");
        while try_count > 0 && to_update {
            try_count -= 1;
            to_update = false;
            let mut _hit = 0;
            let mut _miss = 0;
            let mut sim: BTreeMap<usize, (usize, usize)> = BTreeMap::new();
            let mut new_access = self.access.clone();
            for (ts, (k, action)) in self.access.iter().enumerate() {
                if let Some(x) = sim.get_mut(&k) {
                    _hit += 1;
                    x.0 = ts;
                    if *action == CacheAction::Drop {
                        sim.remove(&k);
                    }
                } else {
                    _miss += 1;
                    if *action == CacheAction::Cache {
                        if sim.len() == self.bound {
                            for (ts, (k, _)) in self.access.iter().enumerate().skip(ts) {
                                if let Some(e) = sim.get_mut(&k) {
                                    if e.1 > ts {
                                        e.1 = ts;
                                    }
                                }
                            }

                            let (_, last_ts) =
                                sim.iter().fold((0, 0), |(max_latest_access, ts), e| {
                                    if e.1 .1 > max_latest_access {
                                        (e.1 .1, e.1 .0)
                                    } else {
                                        (max_latest_access, ts)
                                    }
                                });

                            if self.access[last_ts].1 != CacheAction::Drop {
                                new_access[last_ts].1 = CacheAction::Drop;
                                to_update = true;
                                break;
                            }
                        }

                        sim.insert(*k, (ts, self.access.len()));
                    }
                }
            }

            if !to_update {
                for (_, (ts, _)) in sim.iter() {
                    new_access[*ts].1 = CacheAction::Drop;
                }
            }

            self.access = new_access;
        }
        end_timer!(timer);
    }
}

#[cfg(feature = "cuda")]
impl<T: std::fmt::Debug> Cache<T> {
    pub fn new(bound: usize) -> Cache<T> {
        Self {
            data: BTreeMap::new(),
            ts: 0,
            bound,
            hit: 0,
            miss: 0,
            access: vec![],
        }
    }

    pub fn get(&mut self, key: usize) -> (Option<Rc<T>>, CacheAction) {
        let action = self
            .access
            .get(self.ts)
            .map(|x| {
                assert!(key == self.access[self.ts].0);
                x.1.clone()
            })
            .unwrap_or(CacheAction::Cache);
        self.ts += 1;
        let res = if let Some(x) = self.data.get_mut(&key) {
            self.hit += 1;
            x.1 = self.ts;
            Some(x.0.clone())
        } else {
            self.miss += 1;
            None
        };

        (res, action)
    }

    pub fn update(&mut self, key: usize, value: T, on_drop: impl FnOnce(T) -> ()) -> Rc<T> {
        let value = Rc::new(value);
        if self.data.len() < self.bound {
            self.data.insert(key, (value.clone(), self.ts));
        } else {
            let min_ts = self
                .data
                .iter()
                .reduce(|a, b| if a.1 .1 > b.1 .1 { b } else { a })
                .unwrap()
                .0
                .clone();
            let drop_value = self.data.remove(&min_ts).unwrap().0;
            if Rc::strong_count(&drop_value) == 1 {
                on_drop(Rc::try_unwrap(drop_value).unwrap());
            }
            self.data.insert(key, (value.clone(), self.ts));
        }
        value
    }
}

#[cfg(feature = "cuda")]
impl<F: FieldExt> ProveExpression<F> {
    pub(crate) fn gen_cache_policy(&self, unit_cache: &mut Cache<Buffer<F>>) {
        match self {
            ProveExpression::Unit(u) => unit_cache.access(u.get_group()),
            ProveExpression::Op(l, r, _) => {
                l.gen_cache_policy(unit_cache);
                r.gen_cache_policy(unit_cache);
            }
            ProveExpression::Y(_) => {}
            ProveExpression::Scale(l, _) => {
                l.gen_cache_policy(unit_cache);
            }
        }
    }

    pub(crate) fn eval_gpu<C: CurveAffine<ScalarExt = F>>(
        &self,
        gpu_idx: usize,
        pk: &ProvingKey<C>,
        advice: &Vec<Polynomial<F, Coeff>>,
        instance: &Vec<Polynomial<F, Coeff>>,
        y: F,
    ) -> Polynomial<F, ExtendedLagrangeCoeff> {
        let closures = ec_gpu_gen::rust_gpu_tools::program_closures!(|program,
                                                                      input: &mut [F]|
         -> ec_gpu_gen::EcResult<
            (),
        > {
            let mut ys = vec![F::one(), y];

            let cache_size = std::env::var("HALO2_PROOF_GPU_EVAL_CACHE").unwrap_or("5".to_owned());
            let cache_size =
                usize::from_str_radix(&cache_size, 10).expect("Invalid HALO2_PROOF_GPU_EVAL_CACHE");
            let mut unit_cache = Cache::new(cache_size);
            self.gen_cache_policy(&mut unit_cache);
            unit_cache.analyze();

            let mut helper = gen_do_extended_fft(pk, program)?;
            let values_buf = self._eval_gpu(
                pk,
                program,
                advice,
                instance,
                &mut ys,
                &mut unit_cache,
                &mut LinkedList::new(),
                &mut helper,
            )?;
            program.read_into_buffer(&values_buf.0.unwrap().0, input)?;

            Ok(())
        });

        let mut values = pk.vk.domain.empty_extended();
        let devices = Device::all();

        let device = devices[gpu_idx % devices.len()];
        let programs = vec![ec_gpu_gen::program!(device).unwrap()];
        let kern =
            FftKernel::<pairing::bn256::Fr>::create(programs).expect("Cannot initialize kernel!");
        kern.kernels[0]
            .program
            .run(closures, &mut values.values[..])
            .unwrap();
        values
    }

    pub(crate) fn _eval_gpu_buffer<C: CurveAffine<ScalarExt = F>>(
        &self,
        pk: &ProvingKey<C>,
        program: &Program,
        advice: &Vec<Polynomial<F, Coeff>>,
        instance: &Vec<Polynomial<F, Coeff>>,
        y: &mut Vec<F>,
        unit_cache: &mut Cache<Buffer<F>>,
        allocator: &mut LinkedList<Buffer<F>>,
        helper: &mut ExtendedFFTHelper<F>,
    ) -> EcResult<(Rc<Buffer<F>>, i32)> {
        let size = 1u32 << pk.vk.domain.extended_k();
        let local_work_size = 128;
        let global_work_size = size / local_work_size;

        let v = self._eval_gpu(
            pk, program, advice, instance, y, unit_cache, allocator, helper,
        )?;
        match v {
            (Some((l, rot_l)), Some(r)) => {
                let res = unsafe { program.create_buffer::<F>(size as usize)? };
                let c = program.create_buffer_from_slice(&vec![r])?;
                let kernel_name = format!("{}_eval_sum_c", "Bn256_Fr");
                let kernel = program.create_kernel(
                    &kernel_name,
                    global_work_size as usize,
                    local_work_size as usize,
                )?;
                kernel
                    .arg(&res)
                    .arg(l.as_ref())
                    .arg(&rot_l)
                    .arg(&c)
                    .arg(&size)
                    .run()?;
                Ok((Rc::new(res), 0))
            }
            (Some((l, rot_l)), None) => Ok((l, rot_l)),
            (None, Some(r)) => {
                let res = unsafe { program.create_buffer::<F>(size as usize)? };
                let c = program.create_buffer_from_slice(&vec![r])?;
                let kernel_name = format!("{}_eval_constant", "Bn256_Fr");
                let kernel = program.create_kernel(
                    &kernel_name,
                    global_work_size as usize,
                    local_work_size as usize,
                )?;
                kernel.arg(&res).arg(&c).run()?;
                Ok((Rc::new(res), 0))
            }
            _ => {
                unreachable!()
            }
        }
    }

    pub(crate) fn _eval_gpu<C: CurveAffine<ScalarExt = F>>(
        &self,
        pk: &ProvingKey<C>,
        program: &Program,
        advice: &Vec<Polynomial<F, Coeff>>,
        instance: &Vec<Polynomial<F, Coeff>>,
        y: &mut Vec<F>,
        unit_cache: &mut Cache<Buffer<F>>,
        allocator: &mut LinkedList<Buffer<F>>,
        helper: &mut ExtendedFFTHelper<F>,
    ) -> EcResult<(Option<(Rc<Buffer<F>>, i32)>, Option<F>)> {
        let size = 1u32 << pk.vk.domain.extended_k();
        let local_work_size = 128;
        let global_work_size = size / local_work_size;
        let rot_scale = 1 << (pk.vk.domain.extended_k() - pk.vk.domain.k());

        match self {
            ProveExpression::Op(l, r, op) => {
                let l = l._eval_gpu(
                    pk, program, advice, instance, y, unit_cache, allocator, helper,
                )?;
                let r = r._eval_gpu(
                    pk, program, advice, instance, y, unit_cache, allocator, helper,
                )?;
                //let timer = start_timer!(|| format!("gpu eval sum {} {:?} {:?}", size, l.0, r.0));
                let res = match (l.0, r.0) {
                    (Some(l), Some(r)) => {
                        let kernel_name = match op {
                            Bop::Sum => format!("{}_eval_sum", "Bn256_Fr"),
                            Bop::Product => format!("{}_eval_mul", "Bn256_Fr"),
                        };
                        let kernel = program.create_kernel(
                            &kernel_name,
                            global_work_size as usize,
                            local_work_size as usize,
                        )?;

                        let res = if r.1 == 0 && Rc::strong_count(&r.0) == 1 {
                            r.0.clone()
                        } else if l.1 == 0 && Rc::strong_count(&l.0) == 1 {
                            l.0.clone()
                        } else {
                            Rc::new(allocator.pop_front().unwrap_or_else(|| unsafe {
                                program.create_buffer::<F>(size as usize).unwrap()
                            }))
                        };

                        kernel
                            .arg(res.as_ref())
                            .arg(l.0.as_ref())
                            .arg(r.0.as_ref())
                            .arg(&l.1)
                            .arg(&r.1)
                            .arg(&size)
                            .run()?;

                        if Rc::strong_count(&l.0) == 1 {
                            allocator.push_back(Rc::try_unwrap(l.0).unwrap())
                        }

                        if Rc::strong_count(&r.0) == 1 {
                            allocator.push_back(Rc::try_unwrap(r.0).unwrap())
                        }

                        Ok((Some((res, 0)), None))
                    }
                    (None, None) => match op {
                        Bop::Sum => Ok((None, Some(l.1.unwrap() + r.1.unwrap()))),
                        Bop::Product => Ok((None, Some(l.1.unwrap() * r.1.unwrap()))),
                    },
                    (None, Some(b)) | (Some(b), None) => {
                        let c = l.1.or(r.1).unwrap();
                        let c = program.create_buffer_from_slice(&vec![c])?;
                        let kernel_name = match op {
                            Bop::Sum => format!("{}_eval_sum_c", "Bn256_Fr"),
                            Bop::Product => format!("{}_eval_mul_c", "Bn256_Fr"),
                        };
                        let kernel = program.create_kernel(
                            &kernel_name,
                            global_work_size as usize,
                            local_work_size as usize,
                        )?;

                        let res = if b.1 == 0 && Rc::strong_count(&b.0) == 1 {
                            b.0.clone()
                        } else {
                            Rc::new(allocator.pop_front().unwrap_or_else(|| unsafe {
                                program.create_buffer::<F>(size as usize).unwrap()
                            }))
                        };

                        kernel
                            .arg(res.as_ref())
                            .arg(b.0.as_ref())
                            .arg(&b.1)
                            .arg(&c)
                            .arg(&size)
                            .run()?;

                        if Rc::strong_count(&b.0) == 1 {
                            allocator.push_back(Rc::try_unwrap(b.0).unwrap())
                        }

                        Ok((Some((res, 0)), None))
                    }
                };
                //end_timer!(timer);

                res
            }
            ProveExpression::Y(ys) => {
                let max_y_order = ys.keys().max().unwrap();
                for _ in (y.len() as u32)..max_y_order + 1 {
                    y.push(y[1] * y.last().unwrap());
                }

                //let timer = start_timer!(|| format!("gpu eval c {}", size));
                let c = ys.iter().fold(F::zero(), |acc, (y_order, f)| {
                    acc + y[*y_order as usize] * f
                });
                //end_timer!(timer);
                Ok((None, Some(c)))
            }
            ProveExpression::Unit(u) => {
                let group = u.get_group();
                let (cache, cache_action) = unit_cache.get(group);
                let (values, rotation) = if let Some(cached_values) = cache {
                    match u {
                        ProveExpressionUnit::Fixed { rotation, .. }
                        | ProveExpressionUnit::Advice { rotation, .. }
                        | ProveExpressionUnit::Instance { rotation, .. } => {
                            (cached_values, *rotation)
                        }
                    }
                } else {
                    let (origin_values, rotation) = match u {
                        ProveExpressionUnit::Fixed {
                            column_index,
                            rotation,
                        } => (&pk.fixed_polys[*column_index], rotation),
                        ProveExpressionUnit::Advice {
                            column_index,
                            rotation,
                        } => (&advice[*column_index], rotation),
                        ProveExpressionUnit::Instance {
                            column_index,
                            rotation,
                        } => (&instance[*column_index], rotation),
                    };

                    let buffer = do_extended_fft(pk, program, origin_values, allocator, helper)?;

                    let value = if cache_action == CacheAction::Cache {
                        unit_cache.update(group, buffer, |buffer| allocator.push_back(buffer))
                    } else {
                        Rc::new(buffer)
                    };

                    let res = (value, *rotation);
                    //end_timer!(timer);
                    res
                };
                Ok((Some((values, rotation.0 * rot_scale)), None))
            }
            ProveExpression::Scale(l, ys) => {
                let l = l._eval_gpu(
                    pk, program, advice, instance, y, unit_cache, allocator, helper,
                )?;
                let l = l.0.unwrap();
                let max_y_order = ys.keys().max().unwrap();
                for _ in (y.len() as u32)..max_y_order + 1 {
                    y.push(y[1] * y.last().unwrap());
                }

                //let timer = start_timer!(|| "gpu eval c");
                let c = ys.iter().fold(F::zero(), |acc, (y_order, f)| {
                    acc + y[*y_order as usize] * f
                });
                let c = program.create_buffer_from_slice(&vec![c])?;

                let kernel_name = format!("{}_eval_scale", "Bn256_Fr");
                let kernel = program.create_kernel(
                    &kernel_name,
                    global_work_size as usize,
                    local_work_size as usize,
                )?;

                let res = if l.1 == 0 && Rc::strong_count(&l.0) == 1 {
                    l.0.clone()
                } else {
                    Rc::new(allocator.pop_front().unwrap_or_else(|| unsafe {
                        program.create_buffer::<F>(size as usize).unwrap()
                    }))
                };
                kernel
                    .arg(res.as_ref())
                    .arg(l.0.as_ref())
                    .arg(&l.1)
                    .arg(&size)
                    .arg(&c)
                    .run()?;

                if Rc::strong_count(&l.0) == 1 {
                    allocator.push_back(Rc::try_unwrap(l.0).unwrap())
                }

                Ok((Some((res, 0)), None))
            }
        }
    }
}

#[cfg(feature = "cuda")]
pub(crate) struct ExtendedFFTHelper<F: FieldExt> {
    pub(crate) origin_value_buffer: Rc<Buffer<F>>,
    pub(crate) coset_powers_buffer: Rc<Buffer<F>>,
    pub(crate) pq_buffer: Rc<Buffer<F>>,
    pub(crate) omegas_buffer: Rc<Buffer<F>>,
}

#[cfg(feature = "cuda")]
pub(crate) fn gen_do_extended_fft<F: FieldExt, C: CurveAffine<ScalarExt = F>>(
    pk: &ProvingKey<C>,
    program: &Program,
) -> EcResult<ExtendedFFTHelper<F>> {
    const MAX_LOG2_RADIX: u32 = 8;
    const LOG2_MAX_ELEMENTS: usize = 32;

    let domain = &pk.vk.domain;
    let coset_powers = [domain.g_coset, domain.g_coset_inv];
    let coset_powers_buffer = Rc::new(program.create_buffer_from_slice(&coset_powers[..])?);

    let log_n = pk.vk.domain.extended_k();
    let n = 1 << log_n;
    let omega = pk.vk.domain.get_extended_omega();
    let max_deg = cmp::min(MAX_LOG2_RADIX, log_n);
    // Precalculate:
    // [omega^(0/(2^(deg-1))), omega^(1/(2^(deg-1))), ..., omega^((2^(deg-1)-1)/(2^(deg-1)))]
    let mut pq = vec![F::zero(); 1 << max_deg >> 1];
    let twiddle = omega.pow_vartime([(n >> max_deg) as u64]);
    pq[0] = F::one();
    if max_deg > 1 {
        pq[1] = twiddle;
        for i in 2..(1 << max_deg >> 1) {
            pq[i] = pq[i - 1];
            pq[i].mul_assign(&twiddle);
        }
    }
    let pq_buffer = Rc::new(program.create_buffer_from_slice(&pq)?);

    // Precalculate [omega, omega^2, omega^4, omega^8, ..., omega^(2^31)]
    let mut omegas = vec![F::zero(); 32];
    omegas[0] = omega;
    for i in 1..LOG2_MAX_ELEMENTS {
        omegas[i] = omegas[i - 1].pow_vartime([2u64]);
    }
    let omegas_buffer = Rc::new(program.create_buffer_from_slice(&omegas)?);

    let origin_value_buffer = Rc::new(unsafe { program.create_buffer::<F>(1 << domain.k())? });

    Ok(ExtendedFFTHelper {
        origin_value_buffer,
        omegas_buffer,
        coset_powers_buffer,
        pq_buffer,
    })
}

#[cfg(feature = "cuda")]
pub(crate) fn do_extended_fft<F: FieldExt, C: CurveAffine<ScalarExt = F>>(
    pk: &ProvingKey<C>,
    program: &Program,
    origin_values: &Polynomial<F, Coeff>,
    allocator: &mut LinkedList<Buffer<F>>,
    helper: &mut ExtendedFFTHelper<F>,
) -> EcResult<Buffer<F>> {
    let origin_size = 1u32 << pk.vk.domain.k();
    let extended_size = 1u32 << pk.vk.domain.extended_k();
    let local_work_size = 128;
    let global_work_size = extended_size / local_work_size;

    //let timer = start_timer!(|| "gpu eval unit");
    let values = allocator
        .pop_front()
        .unwrap_or_else(|| unsafe { program.create_buffer::<F>(extended_size as usize).unwrap() });

    let origin_values_buffer = Rc::get_mut(&mut helper.origin_value_buffer).unwrap();
    program.write_from_buffer(origin_values_buffer, &origin_values.values)?;

    do_distribute_powers_zeta(
        pk,
        program,
        origin_values_buffer,
        &helper.coset_powers_buffer,
    )?;

    let kernel_name = format!("{}_eval_fft_prepare", "Bn256_Fr");
    let kernel = program.create_kernel(
        &kernel_name,
        global_work_size as usize,
        local_work_size as usize,
    )?;
    kernel
        .arg(origin_values_buffer)
        .arg(&values)
        .arg(&origin_size)
        .run()?;

    let domain = &pk.vk.domain;
    do_fft_pure(
        program,
        values,
        domain.extended_k(),
        allocator,
        &helper.pq_buffer,
        &helper.omegas_buffer,
    )
}

#[cfg(feature = "cuda")]
pub(crate) fn do_fft_pure<F: FieldExt>(
    program: &Program,
    values: Buffer<F>,
    log_n: u32,
    allocator: &mut LinkedList<Buffer<F>>,
    pq_buffer: &Buffer<F>,
    omegas_buffer: &Buffer<F>,
) -> EcResult<Buffer<F>> {
    do_fft_core(
        program,
        values,
        log_n,
        None,
        allocator,
        pq_buffer,
        omegas_buffer,
    )
}

#[cfg(feature = "cuda")]
pub(crate) fn do_distribute_powers_zeta<F: FieldExt, C: CurveAffine<ScalarExt = F>>(
    pk: &ProvingKey<C>,
    program: &Program,
    values: &mut Buffer<F>,
    coset_powers_buffer: &Buffer<F>,
) -> EcResult<()> {
    let size = 1u32 << pk.vk.domain.k();
    let local_work_size = 128;
    let global_work_size = size / local_work_size;

    let kernel_name = format!("{}_distribute_powers_zeta", "Bn256_Fr");
    let kernel = program.create_kernel(
        &kernel_name,
        global_work_size as usize,
        local_work_size as usize,
    )?;
    kernel.arg(values).arg(coset_powers_buffer).arg(&3).run()?;
    Ok(())
}

#[cfg(feature = "cuda")]
pub(crate) fn _do_ifft<F: FieldExt>(
    program: &Program,
    values: Buffer<F>,
    log_n: u32,
    omega_inv: F,
    allocator: &mut LinkedList<Buffer<F>>,
    pq_buffer: &Buffer<F>,
    omegas_buffer: &Buffer<F>,
) -> EcResult<Buffer<F>> {
    do_fft_core(
        program,
        values,
        log_n,
        Some(omega_inv),
        allocator,
        pq_buffer,
        omegas_buffer,
    )
}

#[cfg(feature = "cuda")]
pub(crate) fn do_fft_core<F: FieldExt>(
    program: &Program,
    values: Buffer<F>,
    log_n: u32,
    divisor: Option<F>,
    allocator: &mut LinkedList<Buffer<F>>,
    pq_buffer: &Buffer<F>,
    omegas_buffer: &Buffer<F>,
) -> EcResult<Buffer<F>> {
    let n = 1 << log_n;
    const MAX_LOG2_RADIX: u32 = 8;
    const MAX_LOG2_LOCAL_WORK_SIZE: u32 = 7;

    let mut src_buffer = values;
    let mut dst_buffer = allocator
        .pop_front()
        .unwrap_or_else(|| unsafe { program.create_buffer::<F>(n).unwrap() });
    // The precalculated values pq` and `omegas` are valid for radix degrees up to `max_deg`
    let max_deg = cmp::min(MAX_LOG2_RADIX, log_n);

    //let timer = start_timer!(|| format!("fft main {}", log_n));
    // Specifies log2 of `p`, (http://www.bealto.com/gpu-fft_group-1.html)
    let mut log_p = 0u32;
    // Each iteration performs a FFT round
    while log_p < log_n {
        // 1=>radix2, 2=>radix4, 3=>radix8, ...
        let deg = cmp::min(max_deg, log_n - log_p);

        let n = 1u32 << log_n;
        let local_work_size = 1 << cmp::min(deg - 1, MAX_LOG2_LOCAL_WORK_SIZE);
        let global_work_size = n >> deg;
        let kernel_name = format!("{}_radix_fft", "Bn256_Fr");
        let kernel = program.create_kernel(
            &kernel_name,
            global_work_size as usize,
            local_work_size as usize,
        )?;
        kernel
            .arg(&src_buffer)
            .arg(&dst_buffer)
            .arg(pq_buffer)
            .arg(omegas_buffer)
            .arg(&LocalBuffer::<F>::new(1 << deg))
            .arg(&n)
            .arg(&log_p)
            .arg(&deg)
            .arg(&max_deg)
            .run()?;

        log_p += deg;
        std::mem::swap(&mut src_buffer, &mut dst_buffer);
    }

    if let Some(divisor) = divisor {
        let divisor = vec![divisor];
        let divisor_buffer = program.create_buffer_from_slice(&divisor[..])?;
        let local_work_size = 128;
        let global_work_size = n / local_work_size;
        let kernel_name = format!("{}_eval_mul_c", "Bn256_Fr");
        let kernel = program.create_kernel(
            &kernel_name,
            global_work_size as usize,
            local_work_size as usize,
        )?;
        kernel
            .arg(&src_buffer)
            .arg(&src_buffer)
            .arg(&0u32)
            .arg(&divisor_buffer)
            .arg(&(n as u32))
            .run()?;
    }

    allocator.push_back(dst_buffer);

    Ok(src_buffer)
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

impl<F: FieldExt> ProveExpression<F> {
    pub(crate) fn new() -> Self {
        ProveExpression::Y(BTreeMap::from_iter(vec![(0, F::zero())].into_iter()))
    }

    pub fn from_expr(e: &Expression<F>) -> Self {
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

        return tree
            .into_iter()
            .map(|(k, ys)| Self::reconstruct_units_coeff(k, ys))
            .reduce(|acc, x| Self::Op(Box::new(acc), Box::new(x), Bop::Sum))
            .unwrap();
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
    pub fn flatten(self) -> BTreeMap<Vec<ProveExpressionUnit>, BTreeMap<u32, F>> {
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
            ProveExpression::Scale(x, ys) => {
                // as Product
                ProveExpression::Op(x, Box::new(ProveExpression::Y(ys)), Bop::Product).flatten()
            }
        }
    }
}
