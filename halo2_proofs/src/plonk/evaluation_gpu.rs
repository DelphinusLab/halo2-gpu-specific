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
use group::prime::PrimeCurve;
use group::{
    ff::{BatchInvert, Field},
    Curve,
};
use std::any::TypeId;
use std::collections::{BTreeSet, HashMap, LinkedList, HashSet};
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

use super::symbol::Bop;
use super::symbol::ProveExpression;
use super::symbol::ProveExpressionUnit;

/*
pub fn est_cache<F>(e: &ProveExpression<F>) -> usize where F: FieldExt {
    let cache_size = std::env::var("HALO2_PROOF_GPU_EVAL_CACHE").unwrap_or("5".to_owned());
    let cache_size = usize::from_str_radix(&cache_size, 10).expect("Invalid HALO2_PROOF_GPU_EVAL_CACHE");
    let mut unit_cache = Cache::new(cache_size);
    e.gen_cache_policy(&mut unit_cache);
    unit_cache.analyze()
}
*/

#[derive(Clone, Debug)]
pub enum LookupProveExpression<F> {
    Expression(ProveExpression<F>),
    LcTheta(Box<LookupProveExpression<F>>, Box<LookupProveExpression<F>>),
    LcBeta(Box<LookupProveExpression<F>>, Box<LookupProveExpression<F>>),
    AddGamma(Box<LookupProveExpression<F>>),
}

impl<F:FieldExt> ToString for LookupProveExpression<F> {
    fn to_string(&self) -> String {
        match &self {
            LookupProveExpression::Expression(a) => {
                format!("(lookupexpr: {})", a.to_string())
            },
            LookupProveExpression::LcTheta(a, b) => {
                format!("(lookuptheta: {}; {})", a.to_string(), b.to_string())
            },
            LookupProveExpression::LcBeta(a, b) => {
                format!("(lookupbeta: {}; {})", a.to_string(), b.to_string())
            },
            LookupProveExpression::AddGamma(g) => {
                format!("(lookupgamma: {})", g.to_string())
            },
        }
    }
}

#[cfg(feature = "cuda")]
impl<F: FieldExt> LookupProveExpression<F> {
    pub(crate) fn _eval_gpu<C: CurveAffine<ScalarExt = F>>(
        &self,
        pk: &ProvingKey<C>,
        program: &Program,
        memory_cache: &BTreeMap<usize, usize>,
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
        println!("FIND_ME {}", stringify!(_eval_gpu));
        println!("{}", self.to_string());

        let size = 1u32 << pk.vk.domain.extended_k();
        let local_work_size = 128;
        let global_work_size = size / local_work_size;


        //println!("_eval gpu lookup_expr: {}", self.to_string());
        //let timer = start_timer!(|| "eval lookup expr");

        let c = match self {
            LookupProveExpression::Expression(e) => e._eval_gpu_buffer(
                pk, program, memory_cache,
                advice, instance, y, unit_cache, allocator, helper,
            ),
            LookupProveExpression::LcTheta(l, r) => {
                let l = l._eval_gpu(
                    pk, program, memory_cache,
                    advice, instance, y, beta, theta, gamma, unit_cache, allocator,
                    helper,
                )?;
                let r = r._eval_gpu(
                    pk, program, memory_cache,
                    advice, instance, y, beta, theta, gamma, unit_cache, allocator,
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
            LookupProveExpression::LcBeta(l, r) => {
                let l = l._eval_gpu(
                    pk, program, memory_cache,
                    advice, instance, y, beta, theta, gamma, unit_cache, allocator,
                    helper,
                )?;
                let r = r._eval_gpu(
                    pk, program, memory_cache,
                    advice, instance, y, beta, theta, gamma, unit_cache, allocator,
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
                let beta = program.create_buffer_from_slice(&vec![beta])?;
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
            LookupProveExpression::AddGamma(l) => {
                let l = l._eval_gpu(
                    pk, program, memory_cache, advice, instance, y,
                    beta, theta, gamma, unit_cache, allocator,
                    helper,
                )?;
                let res = if l.1 == 0 && Rc::strong_count(&l.0) == 1 {
                    l.0.clone()
                } else {
                    Rc::new(allocator.pop_front().unwrap_or_else(|| unsafe {
                        program.create_buffer::<F>(size as usize).unwrap()
                    }))
                };
                let gamma = program.create_buffer_from_slice(&vec![gamma])?;
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
        };
        //end_timer!(timer);
        c
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
    pub data: BTreeMap<usize, (Rc<T>, usize)>,
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

    pub fn analyze(&mut self) -> usize {
        let mut to_update = true;
        let mut try_count = 100000;
        let timer = start_timer!(|| "cache policy analysis");
        let mut miss = 0;
        while try_count > 0 && to_update {
            try_count -= 1;
            to_update = false;
            let mut _hit = 0;
            miss = 0;
            // e -> (latest_time_stamp, order_in_queue)
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
                    miss += 1;
                    if *action == CacheAction::Cache {
                        if sim.len() == self.bound {
                            for (ts, (k, _)) in self.access.iter().enumerate().skip(ts) {
                                if let Some(e) = sim.get_mut(&k) {
                                    if e.1 > ts {
                                        e.1 = ts;
                                    }
                                }
                            }

                            // select a better non visited element in the future
                            let (_, last_ts) =
                                sim.iter().fold((0, 0), |(max_latest_access, ts), e| {
                                    // order_inqueue
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
        println!("simulated miss is {}", miss);
        end_timer!(timer);
        miss
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
               if key != self.access[self.ts].0 {
                   println!("access is {:?}", self.access);
               }
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
        let handle_flat = if let Some ((uid, exprs)) = self.flat_unique_unit_scale() {
            if exprs.len() > 1 {
                uid.gen_cache_policy(unit_cache);
                Some (())
            } else {
                None
            }
        } else {
            None
        };

        if handle_flat.is_none() {
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
    }

    pub(crate) fn eval_gpu<C: CurveAffine<ScalarExt = F>>(
        &self,
        group_idx: usize,
        pk: &ProvingKey<C>,
        memory_cache: &BTreeMap<usize, usize>,
        advice: &Vec<Polynomial<F, Coeff>>,
        instance: &Vec<Polynomial<F, Coeff>>,
        y: F,
    ) -> (Polynomial<F, ExtendedLagrangeCoeff>, BTreeMap<usize, (Rc<Buffer<F>>, usize)>) {
        //let timer = start_timer!(|| format!("evalu_gpu {}", self.to_string()));
        let closures = ec_gpu_gen::rust_gpu_tools::program_closures!(|program,
                                                                      input: &mut [F]|
         -> ec_gpu_gen::EcResult<BTreeMap<usize, (Rc<Buffer<F>>, usize)>> {
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
                memory_cache,
                advice,
                instance,
                &mut ys,
                &mut unit_cache,
                &mut LinkedList::new(),
                &mut helper,
            )?;
            program.read_into_buffer(&values_buf.0.unwrap().0, input)?;


            {
                use std::fs::File;
                use std::io::Write;


                let mut f = match File::create("input.txt") {
                    Ok(file) => file,
                    Err(e) => {
                        panic!("Failed to create file: {}", e);
                    }
                };
                let dd = format!("{:?}", input);
                match f.write_all(dd.as_bytes()) {
                    Ok(_) => println!("Data has been written to the file."),
                    Err(e) => println!("Error occurred while writing to the file: {}", e),
                }
            }

            println!("cache: {:?}", unit_cache.data.keys().collect::<Vec<_>>().into_iter().map(|x| ProveExpressionUnit::key_to_string(*x)).collect::<Vec<_>>());

            Ok(unit_cache.data)
        });

        let mut values = pk.vk.domain.empty_extended();
        let devices = Device::all();
        let programs = devices
            .iter()
            .map(|device| ec_gpu_gen::program!(device))
            .collect::<Result<_, _>>()
            .expect("Cannot create programs!");
        let kern =
            FftKernel::<pairing::bn256::Fr>::create(programs).expect("Cannot initialize kernel!");

        let gpu_idx = group_idx % kern.kernels.len();
        println!("GPU IDX = {:?}", gpu_idx);
        let data = kern.kernels[gpu_idx]
            .program
            .run(closures, &mut values.values[..])
            .unwrap();
        //end_timer!(timer);
        (values, data)
    }

    pub(crate) fn _eval_gpu_buffer<C: CurveAffine<ScalarExt = F>>(
        &self,
        pk: &ProvingKey<C>,
        program: &Program,
        memory_cache: &BTreeMap<usize, usize>,
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
            pk, program, memory_cache,
            advice, instance, y, unit_cache, allocator, helper,
        )?;
        match v {
            (Some((l, rot_l)), Some(r)) => {
                let res = unsafe { program.create_buffer::<F>(size as usize)? };
                let c = program.create_buffer_from_slice(&vec![r])?;
                //let timer = start_timer!(|| "sum eval");
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
                //end_timer!(timer);
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

    fn eval_flat_scale<C: CurveAffine<ScalarExt = F>>(
        &self,
        pk: &ProvingKey<C>,
        program: &Program,
        memory_cache: &BTreeMap<usize, usize>,
        advice: &Vec<Polynomial<F, Coeff>>,
        instance: &Vec<Polynomial<F, Coeff>>,
        y: &mut Vec<F>,
        unit_cache: &mut Cache<Buffer<F>>,
        allocator: &mut LinkedList<Buffer<F>>,
        helper: &mut ExtendedFFTHelper<F>,
    ) -> EcResult<Option<Rc<Buffer<F>>>> {
        let size = 1u32 << pk.vk.domain.extended_k();
        let local_work_size = 128;
        let global_work_size = size / local_work_size;
        let rot_scale = 1 << (pk.vk.domain.extended_k() - pk.vk.domain.k());

        if let Some ((uid, exprs)) = self.flat_unique_unit_scale() {
            if exprs.len() > 1 {
                let cs = exprs.iter().map(|x| x.get_scale_coeff(y).unwrap()).collect::<Vec<F>>();
                let rot = exprs.iter().map(|x| x.get_unit_rotation().unwrap() * rot_scale).collect::<Vec<i32>>();
                let c = program.create_buffer_from_slice(&cs)?;
                let r = program.create_buffer_from_slice(&rot)?;
                let clen = cs.len() as u32;

                let l = uid._eval_gpu(
                    pk, program, memory_cache,
                    advice, instance, y, unit_cache, allocator, helper,
                )?.0.unwrap();

                let kernel_name = format!("{}_eval_batch_scale", "Bn256_Fr");
                let kernel = program.create_kernel(
                    &kernel_name,
                    global_work_size as usize,
                    local_work_size as usize,
                )?;

                let res = Rc::new(allocator.pop_front().unwrap_or_else(|| unsafe {
                        program.create_buffer::<F>(size as usize).unwrap()
                    }));

                //let timer = start_timer!(|| "batch eval scale");
                kernel
                    .arg(res.as_ref())
                    .arg(l.0.as_ref())
                    .arg(&r)
                    .arg(&clen)
                    .arg(&size)
                    .arg(&c)
                    .run()?;

                if Rc::strong_count(&l.0) == 1 {
                    allocator.push_back(Rc::try_unwrap(l.0).unwrap())
                }

                //end_timer!(timer);

                Ok(Some(res))
            } else {
                Ok(None)
            }
        } else {
            Ok(None)
        }
    }


    pub(crate) fn _eval_gpu<C: CurveAffine<ScalarExt = F>>(
        &self,
        pk: &ProvingKey<C>,
        program: &Program,
        memory_cache: &BTreeMap<usize, usize>,
        advice: &Vec<Polynomial<F, Coeff>>,
        instance: &Vec<Polynomial<F, Coeff>>,
        y: &mut Vec<F>,
        unit_cache: &mut Cache<Buffer<F>>,
        allocator: &mut LinkedList<Buffer<F>>,
        helper: &mut ExtendedFFTHelper<F>,
    ) -> EcResult<(Option<(Rc<Buffer<F>>, i32)>, Option<F>)> {
        if self.to_string() == "(Smix(u(a24-3)) * u(f16-0))".to_string() {
            println!("FIND_ME ProveExpression");
            println!("{}", self.to_string());
        }
        let size = 1u32 << pk.vk.domain.extended_k();
        let local_work_size = 128;
        let global_work_size = size / local_work_size;
        let rot_scale = 1 << (pk.vk.domain.extended_k() - pk.vk.domain.k());
        if let Some(v) = self.eval_flat_scale (
                pk,
                program,
                memory_cache,
                advice,
                instance,
                y,
                unit_cache,
                allocator,
                helper
            )? {
            Ok((Some((v, 0)), None))
        } else {
            match self {
                ProveExpression::Op(l, r, op) => {
                    let l = l._eval_gpu(
                        pk, program, memory_cache,
                        advice, instance, y, unit_cache, allocator, helper,
                    )?;
                    let r = r._eval_gpu(
                        pk, program, memory_cache,
                        advice, instance, y, unit_cache, allocator, helper,
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
                    //
                    if self.to_string() == "(Smix(u(a24-3)) * u(f16-0))".to_string() {
                        println!("FIND_ME ProveExpression::Op");
                        println!("{:?}", res);

                        let bb = res.as_ref().unwrap().0.as_ref().unwrap().0.as_ref();
                        let d = 1048576;

                        let mut xx = vec![F::zero(); d];
                        let mut yy = xx.as_mut_slice();

                        program.read_into_buffer(&bb, &mut yy)?;

                        {
                            use std::fs::File;
                            use std::io::Write;

                            let mut f = match File::create("a24_mul_f16.txt") {
                                Ok(file) => file,
                                Err(e) => {
                                    panic!("Failed to create file: {}", e);
                                }
                            };
                            let dd = format!("{:?}", yy);
                            match f.write_all(dd.as_bytes()) {
                                Ok(_) => println!("Data has been written to the file."),
                                Err(e) => println!("Error occurred while writing to the file: {}", e),
                            }
                        }

                    }

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
                        //let timer = start_timer!(|| format!("processing unit {} hit", u.to_string()));
                        let v = match u {
                            ProveExpressionUnit::Fixed { rotation, .. }
                            | ProveExpressionUnit::Advice { rotation, .. }
                            | ProveExpressionUnit::Instance { rotation, .. } => {
                                (cached_values, *rotation)
                            }
                        };
                        //end_timer!(timer);
                        v
                    } else {
                        //let timer = start_timer!(|| format!("processing unit {} miss", u.to_string()));
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

                        let buffer =  do_extended_fft(pk, program, origin_values, allocator, helper)?;
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
                        pk, program, memory_cache,
                        advice, instance, y, unit_cache, allocator, helper,
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
    const LOG2_MAX_ELEMENTS: usize = 32;

    let domain = &pk.vk.domain;
    let coset_powers = [domain.g_coset, domain.g_coset_inv];
    let coset_powers_buffer = Rc::new(program.create_buffer_from_slice(&coset_powers[..])?);

    let log_n = pk.vk.domain.extended_k();
    let n = 1 << log_n;
    let omega = pk.vk.domain.get_extended_omega();

    let max_log2_radix: u32 = if log_n % 8 == 1 {
        7
    } else {
        8
    };

    let max_deg = cmp::min(max_log2_radix, log_n);
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

/*
 * 1. Loading from memory is slow.
 * 2. Leave the code block for potential use case
#[cfg(feature = "cuda")]
pub(crate) fn load_unit_from_mem_cache<F: FieldExt, C: CurveAffine<ScalarExt = F>>(
    pk: &ProvingKey<C>,
    program: &Program,
    allocator: &mut LinkedList<Buffer<F>>,
    poly: &Polynomial<F, ExtendedLagrangeCoeff>
) -> EcResult<Buffer<F>> {
    let domain = &pk.vk.domain;
    let extended_size = 1u32 << pk.vk.domain.extended_k();

    let mut values = allocator
        .pop_front()
        .unwrap_or_else(|| unsafe { program.create_buffer::<F>(extended_size as usize).unwrap() });

    program.write_from_buffer(&mut values, &poly.values)?;
    Ok(values)
}
*/


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

    //let timerall = start_timer!(|| "gpu eval unit");
    let values = allocator
        .pop_front()
        .unwrap_or_else(|| unsafe { program.create_buffer::<F>(extended_size as usize).unwrap() });

    let origin_values_buffer = Rc::get_mut(&mut helper.origin_value_buffer).unwrap();
    //let timer = start_timer!(|| "write from buffer");
    program.write_from_buffer(origin_values_buffer, &origin_values.values)?;
    //end_timer!(timer);

    //let timer = start_timer!(|| "distribute powers zeta");
    do_distribute_powers_zeta(
        pk,
        program,
        origin_values_buffer,
        &helper.coset_powers_buffer,
    )?;
    //end_timer!(timer);

    //let timer = start_timer!(|| "eval fft prepare");
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
    //end_timer!(timer);

    //let timer = start_timer!(|| "do fft pure");
    let domain = &pk.vk.domain;
    let a = do_fft_pure(
        program,
        values,
        domain.extended_k(),
        allocator,
        &helper.pq_buffer,
        &helper.omegas_buffer,
    );
    //end_timer!(timer);
    //end_timer!(timerall);
    a
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
    let max_log2_radix = if log_n % 8 == 1 {
        7 // scale the local worker size of the last round
    } else {
        8
    };
    let max_log2_local_work_size: u32 = 6;

    let mut src_buffer = values;
    let mut dst_buffer = allocator
        .pop_front()
        .unwrap_or_else(|| unsafe { program.create_buffer::<F>(n).unwrap() });
    // The precalculated values pq` and `omegas` are valid for radix degrees up to `max_deg`
    let max_deg = cmp::min(max_log2_radix, log_n);

    // Specifies log2 of `p`, (http://www.bealto.com/gpu-fft_group-1.html)
    let mut log_p = 0u32;
    // Each iteration performs a FFT round
    while log_p < log_n {
        // 1=>radix2, 2=>radix4, 3=>radix8, ...
        let deg = cmp::min(max_deg, log_n - log_p);

        let n = 1u32 << log_n;
        let local_work_size = 1 << cmp::min(deg - 1, max_log2_local_work_size);
        let global_work_size = n >> deg;
        let kernel_name = format!("{}_radix_fft", "Bn256_Fr");
        let kernel = program.create_kernel(
            &kernel_name,
            global_work_size as usize,
            local_work_size as usize,
        )?;

        //let timer = start_timer!(|| format!("fft round {} of {}, deg is {}", log_p, log_n, deg));
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
        //end_timer!(timer); 
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
