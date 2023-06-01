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
use std::collections::{BTreeSet, HashMap};
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
    fn get_group(&self) -> usize {
        match self {
            ProveExpressionUnit::Fixed { column_index, .. } => column_index << 2,
            ProveExpressionUnit::Advice { column_index, .. } => (column_index << 2) + 1,
            ProveExpressionUnit::Instance { column_index, .. } => (column_index << 2) + 2,
        }
    }
}

#[derive(Clone, Debug)]
pub enum ProveExpression<F> {
    Unit(ProveExpressionUnit),
    /// This is the sum of two polynomials
    Sum(Box<ProveExpression<F>>, Box<ProveExpression<F>>),
    /// This is the product of two polynomials
    Product(Box<ProveExpression<F>>, Box<ProveExpression<F>>),
    Y(BTreeMap<u32, F>),
    Scale(Box<ProveExpression<F>>, BTreeMap<u32, F>),
}

#[derive(Clone, Debug)]
pub enum LookupProveExpression<F> {
    Expression(ProveExpression<F>),
    LcTheta(Box<LookupProveExpression<F>>, Box<LookupProveExpression<F>>),
    LcBeta(Box<LookupProveExpression<F>>, Box<LookupProveExpression<F>>),
    AddGamma(Box<LookupProveExpression<F>>),
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
    ) -> EcResult<(Buffer<F>, i32)> {
        let size = 1u32 << pk.vk.domain.extended_k();
        let local_work_size = 128;
        let global_work_size = size / local_work_size;

        match self {
            LookupProveExpression::Expression(e) => {
                e._eval_gpu_buffer(pk, program, advice, instance, y, unit_cache)
            }
            LookupProveExpression::LcTheta(l, r) => {
                let l = l._eval_gpu(
                    pk, program, advice, instance, y, beta, theta, gamma, unit_cache,
                )?;
                let r = r._eval_gpu(
                    pk, program, advice, instance, y, beta, theta, gamma, unit_cache,
                )?;
                let res = unsafe { program.create_buffer::<F>(size as usize)? };
                let theta = program.create_buffer_from_slice(&vec![theta])?;
                let kernel_name = format!("{}_eval_lctheta", "Bn256_Fr");
                //let timer = start_timer!(|| kernel_name.clone());
                let kernel = program.create_kernel(
                    &kernel_name,
                    global_work_size as usize,
                    local_work_size as usize,
                )?;
                kernel
                    .arg(&res)
                    .arg(&l.0)
                    .arg(&r.0)
                    .arg(&l.1)
                    .arg(&r.1)
                    .arg(&size)
                    .arg(&theta)
                    .run()?;
                //end_timer!(timer);
                Ok((res, 0))
            }
            LookupProveExpression::LcBeta(l, r) => {
                let l = l._eval_gpu(
                    pk, program, advice, instance, y, beta, theta, gamma, unit_cache,
                )?;
                let r = r._eval_gpu(
                    pk, program, advice, instance, y, beta, theta, gamma, unit_cache,
                )?;
                let res = unsafe { program.create_buffer::<F>(size as usize)? };
                let beta = program.create_buffer_from_slice(&vec![beta])?;
                let kernel_name = format!("{}_eval_lcbeta", "Bn256_Fr");
                //let timer = start_timer!(|| kernel_name.clone());
                let kernel = program.create_kernel(
                    &kernel_name,
                    global_work_size as usize,
                    local_work_size as usize,
                )?;
                kernel
                    .arg(&res)
                    .arg(&l.0)
                    .arg(&r.0)
                    .arg(&l.1)
                    .arg(&r.1)
                    .arg(&size)
                    .arg(&beta)
                    .run()?;
                //end_timer!(timer);
                Ok((res, 0))
            }
            LookupProveExpression::AddGamma(l) => {
                let res = unsafe { program.create_buffer::<F>(size as usize)? };
                let l = l._eval_gpu(
                    pk, program, advice, instance, y, beta, theta, gamma, unit_cache,
                )?;
                let gamma = program.create_buffer_from_slice(&vec![gamma])?;
                let kernel_name = format!("{}_eval_addgamma", "Bn256_Fr");
                //let timer = start_timer!(|| kernel_name.clone());
                let kernel = program.create_kernel(
                    &kernel_name,
                    global_work_size as usize,
                    local_work_size as usize,
                )?;
                kernel
                    .arg(&res)
                    .arg(&l.0)
                    .arg(&l.1)
                    .arg(&size)
                    .arg(&gamma)
                    .run()?;
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

pub(crate) struct Cache<T> {
    data: BTreeMap<usize, (Rc<T>, usize)>,
    ts: usize,
    bound: usize,
}

impl<T> Cache<T> {
    pub fn new(bound: usize) -> Cache<T> {
        Self {
            data: BTreeMap::new(),
            ts: 0,
            bound,
        }
    }

    pub fn get(&mut self, key: usize) -> Option<Rc<T>> {
        self.ts += 1;
        if let Some(x) = self.data.get_mut(&key) {
            x.1 = self.ts;
            Some(x.0.clone())
        } else {
            None
        }
    }

    pub fn update(&mut self, key: usize, value: T) {
        self.ts += 1;
        if self.data.len() < self.bound {
            self.data.insert(key, (Rc::new(value), self.ts));
        } else {
            let min_ts = self
                .data
                .iter()
                .reduce(|a, b| if a.1 .1 > b.1 .1 { b } else { a })
                .unwrap()
                .0
                .clone();
            self.data.remove(&min_ts);
            self.data.insert(key, (Rc::new(value), self.ts));
        }
    }
}

#[cfg(feature = "cuda")]
impl<F: FieldExt> ProveExpression<F> {
    pub(crate) fn eval_gpu<C: CurveAffine<ScalarExt = F>>(
        &self,
        group_idx: usize,
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

            //let mut unit_cache = BTreeMap::<usize, Buffer<F>>::new();
            let cache_size = std::env::var("HALO2_PROOF_GPU_EVAL_CACHE").unwrap_or("5".to_owned());
            let cache_size =
                usize::from_str_radix(&cache_size, 10).expect("Invalid HALO2_PROOF_GPU_EVAL_CACHE");
            let mut unit_cache = Cache::new(cache_size);
            /*
                for i in 0..usize::min(cache_size as usize, pk.ev.unit_ref_count.len()) {
                    let group = pk.ev.unit_ref_count[i].0;
                    let t = group & 0x3;
                    let column_index = group >> 2;
                    let origin_values = if t == 0 {
                        pk.fixed_polys[column_index].clone()
                    } else if t == 1 {
                        advice[column_index].clone()
                    } else if t == 2 {
                        instance[column_index].clone()
                    } else {
                        unreachable!()
                    };

                    //let timer = start_timer!(|| "gpu eval unit");
                    let values = unsafe { program.create_buffer::<F>(size as usize)? };

                    //let timer = start_timer!(|| "coeff_to_extended_without_fft");
                    let origin_values = pk.vk.domain.coeff_to_extended_without_fft(origin_values);
                    //end_timer!(timer);

                    let origin_values = program.create_buffer_from_slice(&origin_values.values)?;

                    let kernel_name = format!("{}_eval_fft_prepare", "Bn256_Fr");
                    let kernel = program.create_kernel(
                        &kernel_name,
                        global_work_size as usize,
                        local_work_size as usize,
                    )?;
                    kernel
                        .arg(&origin_values)
                        .arg(&values)
                        .arg(&origin_size)
                        .run()?;
                    let values = Self::do_fft(pk, program, values)?;
                    unit_cache.insert(group, values);
                }
            */
            let values_buf =
                self._eval_gpu(pk, program, advice, instance, &mut ys, &mut unit_cache)?;
            program.read_into_buffer(&values_buf.0.unwrap().0, input)?;

            Ok(())
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
        kern.kernels[gpu_idx]
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
    ) -> EcResult<(Buffer<F>, i32)> {
        let size = 1u32 << pk.vk.domain.extended_k();
        let local_work_size = 128;
        let global_work_size = size / local_work_size;
        let v = self._eval_gpu(pk, program, advice, instance, y, unit_cache)?;
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
                    .arg(&l)
                    .arg(&rot_l)
                    .arg(&c)
                    .arg(&size)
                    .run()?;
                Ok((res, 0))
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
                Ok((res, 0))
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
    ) -> EcResult<(Option<(Buffer<F>, i32)>, Option<F>)> {
        let origin_size = 1u32 << pk.vk.domain.k();
        let size = 1u32 << pk.vk.domain.extended_k();
        let local_work_size = 128;
        let global_work_size = size / local_work_size;
        let rot_scale = 1 << (pk.vk.domain.extended_k() - pk.vk.domain.k());

        match self {
            ProveExpression::Sum(l, r) => {
                let l = l._eval_gpu(pk, program, advice, instance, y, unit_cache)?;
                let r = r._eval_gpu(pk, program, advice, instance, y, unit_cache)?;
                //let timer = start_timer!(|| format!("gpu eval sum {} {:?} {:?}", size, l.0, r.0));
                let res = match (l.0, r.0) {
                    (Some(l), Some(r)) => {
                        let kernel_name = format!("{}_eval_sum", "Bn256_Fr");
                        let kernel = program.create_kernel(
                            &kernel_name,
                            global_work_size as usize,
                            local_work_size as usize,
                        )?;

                        let res = if r.1 == 0 {
                            kernel
                                .arg(&r.0)
                                .arg(&l.0)
                                .arg(&r.0)
                                .arg(&l.1)
                                .arg(&r.1)
                                .arg(&size)
                                .run()?;
                            r.0
                        } else {
                            if l.1 == 0 {
                                kernel
                                    .arg(&l.0)
                                    .arg(&l.0)
                                    .arg(&r.0)
                                    .arg(&l.1)
                                    .arg(&r.1)
                                    .arg(&size)
                                    .run()?;
                                l.0
                            } else {
                                let res = unsafe { program.create_buffer::<F>(size as usize)? };
                                kernel
                                    .arg(&res)
                                    .arg(&l.0)
                                    .arg(&r.0)
                                    .arg(&l.1)
                                    .arg(&r.1)
                                    .arg(&size)
                                    .run()?;
                                res
                            }
                        };
                        Ok((Some((res, 0)), None))
                    }
                    (None, None) => Ok((None, Some(l.1.unwrap() + r.1.unwrap()))),
                    (None, Some(b)) | (Some(b), None) => {
                        let c = l.1.or(r.1).unwrap();
                        let c = program.create_buffer_from_slice(&vec![c])?;
                        let kernel_name = format!("{}_eval_sum_c", "Bn256_Fr");
                        let kernel = program.create_kernel(
                            &kernel_name,
                            global_work_size as usize,
                            local_work_size as usize,
                        )?;
                        let res = if b.1 == 0 {
                            kernel
                                .arg(&b.0)
                                .arg(&b.0)
                                .arg(&b.1)
                                .arg(&c)
                                .arg(&size)
                                .run()?;
                            b.0
                        } else {
                            let res = unsafe { program.create_buffer::<F>(size as usize)? };
                            kernel
                                .arg(&res)
                                .arg(&b.0)
                                .arg(&b.1)
                                .arg(&c)
                                .arg(&size)
                                .run()?;
                            res
                        };

                        Ok((Some((res, 0)), None))
                    }
                };
                //end_timer!(timer);

                res
            }
            ProveExpression::Product(l, r) => {
                let l = l._eval_gpu(pk, program, advice, instance, y, unit_cache)?;
                let r = r._eval_gpu(pk, program, advice, instance, y, unit_cache)?;

                //let timer = start_timer!(|| format!("gpu eval mul {}", size));
                let res = match (l.0, r.0) {
                    (Some(l), Some(r)) => {
                        let kernel_name = format!("{}_eval_mul", "Bn256_Fr");
                        let kernel = program.create_kernel(
                            &kernel_name,
                            global_work_size as usize,
                            local_work_size as usize,
                        )?;

                        let res = if r.1 == 0 {
                            kernel
                                .arg(&r.0)
                                .arg(&l.0)
                                .arg(&r.0)
                                .arg(&l.1)
                                .arg(&r.1)
                                .arg(&size)
                                .run()?;
                            r.0
                        } else {
                            if l.1 == 0 {
                                kernel
                                    .arg(&l.0)
                                    .arg(&l.0)
                                    .arg(&r.0)
                                    .arg(&l.1)
                                    .arg(&r.1)
                                    .arg(&size)
                                    .run()?;
                                l.0
                            } else {
                                let res = unsafe { program.create_buffer::<F>(size as usize)? };
                                kernel
                                    .arg(&res)
                                    .arg(&l.0)
                                    .arg(&r.0)
                                    .arg(&l.1)
                                    .arg(&r.1)
                                    .arg(&size)
                                    .run()?;
                                res
                            }
                        };
                        Ok((Some((res, 0)), None))
                    }
                    (None, None) => Ok((None, Some(l.1.unwrap() * r.1.unwrap()))),
                    (None, Some(b)) | (Some(b), None) => {
                        let c = l.1.or(r.1).unwrap();
                        let res = unsafe { program.create_buffer::<F>(size as usize)? };
                        let c = program.create_buffer_from_slice(&vec![c])?;
                        let kernel_name = format!("{}_eval_mul_c", "Bn256_Fr");
                        let kernel = program.create_kernel(
                            &kernel_name,
                            global_work_size as usize,
                            local_work_size as usize,
                        )?;
                        kernel
                            .arg(&res)
                            .arg(&b.0)
                            .arg(&b.1)
                            .arg(&c)
                            .arg(&size)
                            .run()?;
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
                let (values, rotation) = if let Some(cached_values) = unit_cache.get(group) {
                    let values = unsafe { program.create_buffer::<F>(size as usize)? };
                    let kernel_name = format!("{}_eval_fft_prepare", "Bn256_Fr");
                    let kernel = program.create_kernel(
                        &kernel_name,
                        global_work_size as usize,
                        local_work_size as usize,
                    )?;
                    kernel
                        .arg(cached_values.as_ref())
                        .arg(&values)
                        .arg(&size)
                        .run()?;
                    match u {
                        ProveExpressionUnit::Fixed { rotation, .. }
                        | ProveExpressionUnit::Advice { rotation, .. }
                        | ProveExpressionUnit::Instance { rotation, .. } => (values, *rotation),
                    }
                } else {
                    let (origin_values, rotation) = match u {
                        ProveExpressionUnit::Fixed {
                            column_index,
                            rotation,
                        } => (pk.fixed_polys[*column_index].clone(), rotation),
                        ProveExpressionUnit::Advice {
                            column_index,
                            rotation,
                        } => (advice[*column_index].clone(), rotation),
                        ProveExpressionUnit::Instance {
                            column_index,
                            rotation,
                        } => (instance[*column_index].clone(), rotation),
                    };

                    //let timer = start_timer!(|| "gpu eval unit");
                    let values = unsafe { program.create_buffer::<F>(size as usize)? };

                    //let timer = start_timer!(|| "coeff_to_extended_without_fft");
                    let origin_values = pk.vk.domain.coeff_to_extended_without_fft(origin_values);
                    //end_timer!(timer);

                    let origin_values = program.create_buffer_from_slice(&origin_values.values)?;

                    let kernel_name = format!("{}_eval_fft_prepare", "Bn256_Fr");
                    let kernel = program.create_kernel(
                        &kernel_name,
                        global_work_size as usize,
                        local_work_size as usize,
                    )?;
                    kernel
                        .arg(&origin_values)
                        .arg(&values)
                        .arg(&origin_size)
                        .run()?;
                    let buffer = Self::do_fft(pk, program, values)?;
                    let buffer_cached = unsafe { program.create_buffer::<F>(size as usize)? };

                    let kernel_name = format!("{}_eval_fft_prepare", "Bn256_Fr");
                    let kernel = program.create_kernel(
                        &kernel_name,
                        global_work_size as usize,
                        local_work_size as usize,
                    )?;
                    kernel.arg(&buffer).arg(&buffer_cached).arg(&size).run()?;

                    unit_cache.update(group, buffer_cached);
                    let res = (buffer, *rotation);
                    //end_timer!(timer);
                    res
                };
                Ok((Some((values, rotation.0 * rot_scale)), None))
            }
            ProveExpression::Scale(l, ys) => {
                let l = l._eval_gpu(pk, program, advice, instance, y, unit_cache)?;
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
                let values = if l.1 == 0 {
                    kernel
                        .arg(&l.0)
                        .arg(&l.0)
                        .arg(&l.1)
                        .arg(&size)
                        .arg(&c)
                        .run()?;
                    l.0
                } else {
                    let values = unsafe { program.create_buffer::<F>(size as usize)? };
                    kernel
                        .arg(&values)
                        .arg(&l.0)
                        .arg(&l.1)
                        .arg(&size)
                        .arg(&c)
                        .run()?;
                    values
                };
                Ok((Some((values, 0)), None))
            }
        }
    }

    pub(crate) fn do_fft<C: CurveAffine<ScalarExt = F>>(
        pk: &ProvingKey<C>,
        program: &Program,
        values: Buffer<F>,
    ) -> EcResult<Buffer<F>> {
        let log_n = pk.vk.domain.extended_k();
        let n = 1 << log_n;
        let omega = pk.vk.domain.get_extended_omega();
        const MAX_LOG2_RADIX: u32 = 8;
        const LOG2_MAX_ELEMENTS: usize = 32;
        const MAX_LOG2_LOCAL_WORK_SIZE: u32 = 7;

        let mut src_buffer = values;
        let mut dst_buffer = unsafe { program.create_buffer::<F>(n)? };
        // The precalculated values pq` and `omegas` are valid for radix degrees up to `max_deg`
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
        let pq_buffer = program.create_buffer_from_slice(&pq)?;

        // Precalculate [omega, omega^2, omega^4, omega^8, ..., omega^(2^31)]
        let mut omegas = vec![F::zero(); 32];
        omegas[0] = omega;
        for i in 1..LOG2_MAX_ELEMENTS {
            omegas[i] = omegas[i - 1].pow_vartime([2u64]);
        }
        let omegas_buffer = program.create_buffer_from_slice(&omegas)?;

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
                .arg(&pq_buffer)
                .arg(&omegas_buffer)
                .arg(&LocalBuffer::<F>::new(1 << deg))
                .arg(&n)
                .arg(&log_p)
                .arg(&deg)
                .arg(&max_deg)
                .run()?;

            log_p += deg;
            std::mem::swap(&mut src_buffer, &mut dst_buffer);
        }

        Ok(src_buffer)
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
            Expression::Negated(e) => ProveExpression::Product(
                Box::new(Self::from_expr(e)),
                Box::new(ProveExpression::Y(BTreeMap::from_iter(
                    vec![(0, -F::one())].into_iter(),
                ))),
            ),
            Expression::Sum(l, r) => {
                ProveExpression::Sum(Box::new(Self::from_expr(l)), Box::new(Self::from_expr(r)))
            }
            Expression::Product(l, r) => {
                ProveExpression::Product(Box::new(Self::from_expr(l)), Box::new(Self::from_expr(r)))
            }
            Expression::Scaled(l, r) => ProveExpression::Product(
                Box::new(Self::from_expr(l)),
                Box::new(ProveExpression::Y(BTreeMap::from_iter(
                    vec![(0, *r)].into_iter(),
                ))),
            ),
        }
    }

    pub(crate) fn add_gate(self, e: &Expression<F>) -> Self {
        Self::Sum(
            Box::new(Self::Product(
                Box::new(self),
                Box::new(ProveExpression::Y(BTreeMap::from_iter(
                    vec![(1, F::one())].into_iter(),
                ))),
            )),
            Box::new(Self::from_expr(e)),
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
            Self::Product(
                Box::new(Self::reconstruct_unit(u.clone(), c - 1)),
                Box::new(Self::Unit(u)),
            )
        }
    }

    // max r deep: 1
    fn reconstruct_units(mut us: BTreeMap<ProveExpressionUnit, u32>) -> Self {
        let u = us.pop_first().unwrap();

        let mut l = Self::reconstruct_unit(u.0, u.1);

        for (u, c) in us {
            for _ in 0..c {
                l = Self::Product(Box::new(l), Box::new(Self::Unit(u.clone())));
            }
        }

        l
    }

    // max r deep: 1
    fn reconstruct_units_coeff(
        us: BTreeMap<ProveExpressionUnit, u32>,
        coeff: BTreeMap<u32, F>,
    ) -> Self {
        if us.len() == 0 {
            Self::reconstruct_coeff(coeff)
        } else {
            Self::Scale(Box::new(Self::reconstruct_units(us)), coeff)
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

        let mut l = Self::reconstruct_tree(l, r_deep_limit);
        l = Self::Product(Box::new(l), Box::new(Self::Unit(max_u)));

        if r_deep_limit <= 3 {
            for (k, ys) in r {
                l = Self::Sum(Box::new(l), Box::new(Self::reconstruct_units_coeff(k, ys)));
            }
        } else {
            if r.len() > 0 {
                l = Self::Sum(
                    Box::new(l),
                    Box::new(Self::reconstruct_tree(r, r_deep_limit - 1)),
                );
            }
        }

        l
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

        let r_deep = std::env::var("HALO2_PROOF_GPU_EVAL_R_DEEP").unwrap_or("5".to_owned());
        let r_deep = u32::from_str_radix(&r_deep, 10).expect("Invalid HALO2_PROOF_GPU_EVAL_R_DEEP");
        Self::reconstruct_tree(tree, r_deep)
    }

    pub(crate) fn get_complexity(&self) -> (u32, u32, u32, u32, HashMap<usize, u32>) {
        match self {
            ProveExpression::Unit(u) => (0, 0, 1, 0, HashMap::from_iter(vec![(u.get_group(), 1)])),
            ProveExpression::Product(l, r) => {
                let mut l = l.get_complexity();
                let r = r.get_complexity();
                for (k, v) in r.4 {
                    if let Some(lv) = l.4.get_mut(&k) {
                        *lv += v;
                    } else {
                        l.4.insert(k, v);
                    }
                }
                (l.0 + r.0 + 1, l.1 + r.1, l.2 + r.2, l.3 + r.3, l.4)
            }
            ProveExpression::Sum(l, r) => {
                let mut l = l.get_complexity();
                let r = r.get_complexity();
                for (k, v) in r.4 {
                    if let Some(lv) = l.4.get_mut(&k) {
                        *lv += v;
                    } else {
                        l.4.insert(k, v);
                    }
                }
                (l.0 + r.0, l.1 + r.1 + 1, l.2 + r.2, l.3 + r.3, l.4)
            }
            ProveExpression::Y(_) => (0, 0, 0, 1, HashMap::new()),
            ProveExpression::Scale(l, _) => {
                let l = l.get_complexity();
                (l.0 + 1, l.1, l.2, l.3, l.4)
            }
        }
    }

    pub(crate) fn get_r_deep(&self) -> u32 {
        match self {
            ProveExpression::Unit(_) => 0,
            ProveExpression::Sum(l, r) => {
                let l = l.get_r_deep();
                let r = r.get_r_deep();
                u32::max(l, r + 1)
            }
            ProveExpression::Product(l, r) => {
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
            ProveExpression::Sum(l, r) => {
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
            ProveExpression::Product(l, r) => {
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
