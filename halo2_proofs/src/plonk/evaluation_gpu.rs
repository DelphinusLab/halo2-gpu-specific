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
use ec_gpu_gen::rust_gpu_tools::cuda::Program;
use ec_gpu_gen::rust_gpu_tools::Device;
use ec_gpu_gen::rust_gpu_tools::LocalBuffer;
use ec_gpu_gen::EcResult;
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
        program: &Program,
        advice: &Vec<Polynomial<F, LagrangeCoeff>>,
        instance: &Vec<Polynomial<F, LagrangeCoeff>>,
        y: &mut Vec<F>,
    ) -> EcResult<(Buffer<F>, i32)> {
        let origin_size = 1u32 << pk.vk.domain.k();
        let size = 1u32 << pk.vk.domain.extended_k();
        let local_work_size = 32;
        let global_work_size = size / local_work_size;

        match self {
            ProveExpression::Sum(l, r) => {
                let l = l._eval_gpu(pk, program, advice, instance, y)?;
                let r = r._eval_gpu(pk, program, advice, instance, y)?;
                let kernel_name = format!("{}_eval_sum", "Bn256_Fr");
                let kernel = program.create_kernel(
                    &kernel_name,
                    global_work_size as usize,
                    local_work_size as usize,
                )?;
                kernel
                    .arg(&l.0)
                    .arg(&r.0)
                    .arg(&l.1)
                    .arg(&r.1)
                    .arg(&size)
                    .run()?;
                Ok((l.0, 0))
            }
            ProveExpression::Product(l, r) => {
                let l = l._eval_gpu(pk, program, advice, instance, y)?;
                let r = r._eval_gpu(pk, program, advice, instance, y)?;
                let kernel_name = format!("{}_eval_mul", "Bn256_Fr");
                let kernel = program.create_kernel(
                    &kernel_name,
                    global_work_size as usize,
                    local_work_size as usize,
                )?;
                kernel
                    .arg(&l.0)
                    .arg(&r.0)
                    .arg(&l.1)
                    .arg(&r.1)
                    .arg(&size)
                    .run()?;
                Ok((l.0, 0))
            }
            ProveExpression::Y(f, order) => {
                for _ in (y.len() as u32)..order + 1 {
                    y.push(y[1] * y.last().unwrap());
                }
                let values = unsafe { program.create_buffer::<F>(size as usize)? };
                let c = program.create_buffer_from_slice(&vec![y[*order as usize] * f])?;
                let kernel_name = format!("{}_eval_constant", "Bn256_Fr");
                let kernel = program.create_kernel(
                    &kernel_name,
                    global_work_size as usize,
                    local_work_size as usize,
                )?;
                kernel.arg(&values).arg(&c).run()?;
                Ok((values, 0))
            }
            ProveExpression::Unit(u) => match u {
                ProveExpressionUnit::Fixed {
                    column_index,
                    rotation,
                } => {
                    let origin_values =
                        program.create_buffer_from_slice(&pk.fixed_polys[*column_index].values)?;
                    let values = unsafe { program.create_buffer::<F>(size as usize)? };
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
                    Ok((Self::do_fft(pk, program, values)?, rotation.0))
                }
                ProveExpressionUnit::Advice {
                    column_index,
                    rotation,
                } => {
                    let origin_values =
                        program.create_buffer_from_slice(&advice[*column_index].values)?;
                    let values = unsafe { program.create_buffer::<F>(size as usize)? };
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
                    Ok((Self::do_fft(pk, program, values)?, rotation.0))
                }
                ProveExpressionUnit::Instance {
                    column_index,
                    rotation,
                } => {
                    let origin_values =
                        program.create_buffer_from_slice(&instance[*column_index].values)?;
                    let values = unsafe { program.create_buffer::<F>(size as usize)? };
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
                    Ok((Self::do_fft(pk, program, values)?, rotation.0))
                }
            },
            ProveExpression::Scaled(_, _) => unreachable!(),
            ProveExpression::Constant(_) => unreachable!(),
        }
    }

    pub(crate) fn do_fft<C: CurveAffine<ScalarExt = F>>(
        pk: &ProvingKey<C>,
        program: &Program,
        values: Buffer<F>,
    ) -> EcResult<Buffer<F>> {
        let log_n = pk.vk.domain.extended_k();
        let n = 1 << log_n;
        let omega = pk.vk.domain.get_omega();
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

    pub(crate) fn eval_gpu<C: CurveAffine<ScalarExt = F>>(
        &self,
        pk: &ProvingKey<C>,
        advice: Vec<&Vec<Polynomial<F, LagrangeCoeff>>>,
        instance: Vec<&Vec<Polynomial<F, LagrangeCoeff>>>,
        y: F,
    ) -> Polynomial<F, ExtendedLagrangeCoeff> {
        use pairing::bn256::Fr;

        let mut values = pk.vk.domain.empty_extended();

        let closures = ec_gpu_gen::rust_gpu_tools::program_closures!(|program,
                                                                      input: &mut [F]|
         -> ec_gpu_gen::EcResult<
            (),
        > {
            let mut ys = vec![F::one(), y];
            let values_buf = self._eval_gpu(pk, program, advice[0], instance[0], &mut ys)?;
            program.read_into_buffer(&values_buf.0, input)?;
            Ok(())
        });

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
