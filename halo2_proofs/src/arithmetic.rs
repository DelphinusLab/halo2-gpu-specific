//! This module provides common utilities, traits and structures for group,
//! field and polynomial arithmetic.

use std::{ops::Mul, sync::Arc};

use super::multicore;
use ark_std::{end_timer, start_timer};
pub use ff::Field;
use group::{
    cofactor::CofactorCurveAffine,
    ff::{BatchInvert, PrimeField},
    Group as _,
};
pub use pairing::arithmetic::*;
use pairing::bn256::G1Affine;
use rayon::prelude::*;

fn multiexp_serial<C: CurveAffine>(coeffs: &[C::Scalar], bases: &[C], acc: &mut C::Curve) {
    let coeffs: Vec<_> = coeffs.iter().map(|a| a.to_repr()).collect();

    let c = if bases.len() < 4 {
        1
    } else if bases.len() < 32 {
        3
    } else {
        (f64::from(bases.len() as u32)).ln().ceil() as usize
    };

    fn get_at<F: PrimeField>(segment: usize, c: usize, bytes: &F::Repr) -> usize {
        let skip_bits = segment * c;
        let skip_bytes = skip_bits / 8;

        if skip_bytes >= 32 {
            return 0;
        }

        let mut v = [0; 8];
        for (v, o) in v.iter_mut().zip(bytes.as_ref()[skip_bytes..].iter()) {
            *v = *o;
        }

        let mut tmp = u64::from_le_bytes(v);
        tmp >>= skip_bits - (skip_bytes * 8);
        tmp = tmp % (1 << c);

        tmp as usize
    }

    let segments = (256 / c) + 1;

    for current_segment in (0..segments).rev() {
        for _ in 0..c {
            *acc = acc.double();
        }

        #[derive(Clone, Copy)]
        enum Bucket<C: CurveAffine> {
            None,
            Affine(C),
            Projective(C::Curve),
        }

        impl<C: CurveAffine> Bucket<C> {
            fn add_assign(&mut self, other: &C) {
                *self = match *self {
                    Bucket::None => Bucket::Affine(*other),
                    Bucket::Affine(a) => Bucket::Projective(a + *other),
                    Bucket::Projective(mut a) => {
                        a += *other;
                        Bucket::Projective(a)
                    }
                }
            }

            fn add(self, mut other: C::Curve) -> C::Curve {
                match self {
                    Bucket::None => other,
                    Bucket::Affine(a) => {
                        other += a;
                        other
                    }
                    Bucket::Projective(a) => other + &a,
                }
            }
        }

        let mut buckets: Vec<Bucket<C>> = vec![Bucket::None; (1 << c) - 1];

        for (coeff, base) in coeffs.iter().zip(bases.iter()) {
            let coeff = get_at::<C::Scalar>(current_segment, c, coeff);
            if coeff != 0 {
                buckets[coeff - 1].add_assign(base);
            }
        }

        // Summation by parts
        // e.g. 3a + 2b + 1c = a +
        //                    (a) + b +
        //                    ((a) + b) + c
        let mut running_sum = C::Curve::identity();
        for exp in buckets.into_iter().rev() {
            running_sum = exp.add(running_sum);
            *acc = *acc + &running_sum;
        }
    }
}

/// Performs a small multi-exponentiation operation.
/// Uses the double-and-add algorithm with doublings shared across points.
pub fn small_multiexp<C: CurveAffine>(coeffs: &[C::Scalar], bases: &[C]) -> C::Curve {
    let coeffs: Vec<_> = coeffs.iter().map(|a| a.to_repr()).collect();
    let mut acc = C::Curve::identity();

    // for byte idx
    for byte_idx in (0..32).rev() {
        // for bit idx
        for bit_idx in (0..8).rev() {
            acc = acc.double();
            // for each coeff
            for coeff_idx in 0..coeffs.len() {
                let byte = coeffs[coeff_idx].as_ref()[byte_idx];
                if ((byte >> bit_idx) & 1) != 0 {
                    acc += bases[coeff_idx];
                }
            }
        }
    }

    acc
}

pub fn gpu_multiexp2<C: CurveAffine>(coeffs: &[C::Scalar], bases: &[C]) -> C::Curve {
    use ec_gpu_gen::{
        fft::FftKernel, multiexp::MultiexpKernel, rust_gpu_tools::Device, threadpool::Worker,
    };
    use group::Curve;
    use pairing::bn256::Fr;

    let timer = start_timer!(|| "msm");

    let devices = Device::all();
    let programs = devices
        .iter()
        .map(|device| ec_gpu_gen::program!(device))
        .collect::<Result<_, _>>()
        .expect("Cannot create programs!");
    let mut kern =
        MultiexpKernel::<G1Affine>::create(programs, &devices).expect("Cannot initialize kernel!");
    let pool = Worker::new();

    let _coeffs = [Arc::new(
        coeffs.iter().map(|x| x.to_repr()).collect::<Vec<_>>(),
    )];
    let _coeffs: &Arc<Vec<[u8; 32]>> = unsafe { std::mem::transmute(&_coeffs) };
    let bases: &[G1Affine] = unsafe { std::mem::transmute(bases) };
    let bases = Arc::new(Vec::from(bases));

    let a = [kern.multiexp(&pool, bases, _coeffs.clone(), 0).unwrap()];
    let res: &[C::Curve] = unsafe { std::mem::transmute(&a[..]) };
    end_timer!(timer);
    res[0]
}

pub fn gpu_multiexp<C: CurveAffine>(coeffs: &[C::Scalar], bases: &[C]) -> C::Curve {
    use ec_gpu_gen::{
        fft::FftKernel, multiexp::SingleMultiexpKernel, rust_gpu_tools::Device, threadpool::Worker,
    };
    use group::Curve;
    use pairing::bn256::Fr;

    let device = Device::all()[0];
    let programs = ec_gpu_gen::program!(device).unwrap();
    let kern = SingleMultiexpKernel::<G1Affine>::create(programs, device, None)
        .expect("Cannot initialize kernel!");

    let _coeffs = coeffs.iter().map(|x| x.to_repr()).collect::<Vec<_>>();
    let _coeffs: &[[u8; 32]] = unsafe { std::mem::transmute(&_coeffs[..]) };
    let bases: &[G1Affine] = unsafe { std::mem::transmute(bases) };

    let a = [kern.multiexp(bases, _coeffs).unwrap()];
    let res: &[C::Curve] = unsafe { std::mem::transmute(&a[..]) };
    res[0]
}

pub fn best_multiexp_gpu_cond<C: CurveAffine>(coeffs: &[C::Scalar], bases: &[C]) -> C::Curve {
    if false && coeffs.len() > 1 << 14 {
        gpu_multiexp(coeffs, bases)
    } else {
        best_multiexp(coeffs, bases)
    }
}

/// Performs a multi-exponentiation operation.
///
/// This function will panic if coeffs and bases have a different length.
///
/// This will use multithreading if beneficial.
pub fn best_multiexp<C: CurveAffine>(coeffs: &[C::Scalar], bases: &[C]) -> C::Curve {
    assert_eq!(coeffs.len(), bases.len());

    let num_threads = multicore::current_num_threads();
    if coeffs.len() > num_threads {
        let chunk = coeffs.len() / num_threads;
        let num_chunks = coeffs.chunks(chunk).len();
        let mut results = vec![C::Curve::identity(); num_chunks];
        multicore::scope(|scope| {
            let chunk = coeffs.len() / num_threads;

            for ((coeffs, bases), acc) in coeffs
                .chunks(chunk)
                .zip(bases.chunks(chunk))
                .zip(results.iter_mut())
            {
                scope.spawn(move |_| {
                    multiexp_serial(coeffs, bases, acc);
                });
            }
        });
        results.iter().fold(C::Curve::identity(), |a, b| a + b)
    } else {
        let mut acc = C::Curve::identity();
        multiexp_serial(coeffs, bases, &mut acc);
        acc
    }
}

pub fn gpu_fft<G: Group>(a: &mut [G], omega: G::Scalar, log_n: u32) {
    use ec_gpu_gen::{fft::FftKernel, rust_gpu_tools::Device};
    use pairing::bn256::Fr;
    let devices = Device::all();
    let programs = devices
        .iter()
        .map(|device| ec_gpu_gen::program!(device))
        .collect::<Result<_, _>>()
        .expect("Cannot create programs!");
    let mut kern = FftKernel::<Fr>::create(programs).expect("Cannot initialize kernel!");
    let a: &mut [Fr] = unsafe { std::mem::transmute(a) };
    let omega: &Fr = unsafe { std::mem::transmute(&omega) };
    kern.radix_fft_many(&mut [a], &[*omega], &[log_n])
        .expect("GPU FFT failed!");
}

#[cfg(test)]
fn omega<F: PrimeField>(num_coeffs: usize) -> F {
    // Compute omega, the 2^exp primitive root of unity
    let exp = (num_coeffs as f32).log2().floor() as u32;
    let mut omega = F::root_of_unity();
    for _ in exp..F::S {
        omega = omega.square();
    }
    omega
}

#[test]
fn test_fft() {
    use ec_gpu_gen::{fft::FftKernel, rust_gpu_tools::Device};
    use ff::PrimeField;
    use group::ff::Field;
    use pairing::bn256::Fr;
    let mut rng = rand::thread_rng();
    let devices = Device::all();

    let log_d = 24;
    let d = 1 << log_d;
    let mut coeffs = (0..d).map(|_| Fr::random(&mut rng)).collect::<Vec<_>>();
    let omega = omega::<Fr>(coeffs.len());

    let programs = devices
        .iter()
        .map(|device| ec_gpu_gen::program!(device))
        .collect::<Result<_, _>>()
        .expect("Cannot create programs!");
    let mut kern = FftKernel::<Fr>::create(programs).expect("Cannot initialize kernel!");

    for _ in 0..10 {
        if false {
            best_fft(&mut coeffs, omega, log_d);
        } else {
            kern.radix_fft_many(&mut [&mut coeffs], &[omega], &[log_d])
                .expect("GPU FFT failed!");
        }
    }
}

#[test]
fn test_mul_batch() {
    use ec_gpu_gen::{fft::FftKernel, rust_gpu_tools::Device};
    use ff::PrimeField;
    use group::ff::Field;
    use pairing::bn256::Fr;
    let devices = Device::all();

    let log_d = 23;
    let d = 1 << log_d;
    let mut coeffs = (0..d).map(|_| Fr::one()).collect::<Vec<_>>();
    let rhs = Fr::from(2);

    let programs = devices
        .iter()
        .map(|device| ec_gpu_gen::program!(device))
        .collect::<Result<_, _>>()
        .expect("Cannot create programs!");
    let mut kern = FftKernel::<Fr>::create(programs).expect("Cannot initialize kernel!");

    if false {
        for _ in 0..20 {
            coeffs.par_iter_mut().for_each(|x| *x = x.double());
        }
    } else {
        kern.mul_by_field(&mut coeffs, &rhs, log_d).unwrap();
    }

    assert_eq!(coeffs[1], Fr::from(1u64 << 20));
}

/// Performs a radix-$2$ Fast-Fourier Transformation (FFT) on a vector of size
/// $n = 2^k$, when provided `log_n` = $k$ and an element of multiplicative
/// order $n$ called `omega` ($\omega$). The result is that the vector `a`, when
/// interpreted as the coefficients of a polynomial of degree $n - 1$, is
/// transformed into the evaluations of this polynomial at each of the $n$
/// distinct powers of $\omega$. This transformation is invertible by providing
/// $\omega^{-1}$ in place of $\omega$ and dividing each resulting field element
/// by $n$.
///
/// This will use multithreading if beneficial.

pub fn best_fft<G: Group>(a: &mut [G], omega: G::Scalar, log_n: u32) {
    if true {
        gpu_fft(a, omega, log_n);
        return;
    }

    fn bitreverse(mut n: usize, l: usize) -> usize {
        let mut r = 0;
        for _ in 0..l {
            r = (r << 1) | (n & 1);
            n >>= 1;
        }
        r
    }

    let threads = multicore::current_num_threads();
    let log_threads = log2_floor(threads);
    let n = a.len() as usize;
    assert_eq!(n, 1 << log_n);

    for k in 0..n {
        let rk = bitreverse(k, log_n as usize);
        if k < rk {
            a.swap(rk, k);
        }
    }

    //let timer1 = start_timer!(|| format!("prepare do fft {}", log_n));
    // precompute twiddle factors
    let mut twiddles: Vec<_> = (0..(n / 2) as usize)
        .into_iter()
        .map(|_| G::Scalar::one())
        .collect();

    let chunck_size = 1 << 14;
    let block_size = 1 << 10;

    if n / 2 < chunck_size {
        for i in 1..n / 2 {
            twiddles[i] = twiddles[i - 1] * omega;
        }
    } else {
        for i in 1..chunck_size {
            twiddles[i] = twiddles[i - 1] * omega;
        }

        let base = twiddles[chunck_size - 1] * omega;
        let mut chunks = twiddles.chunks_mut(chunck_size);
        let mut prev = chunks.next().unwrap();

        chunks.for_each(|curr| {
            curr.par_chunks_mut(block_size)
                .enumerate()
                .for_each(|(i, v)| {
                    v.iter_mut().enumerate().for_each(|(j, v)| {
                        *v = base * prev[i * block_size + j];
                    });
                });
            prev = curr;
        });
    }

    if log_n <= log_threads {
        let mut chunk = 2_usize;
        let mut twiddle_chunk = (n / 2) as usize;
        for _ in 0..log_n {
            a.chunks_mut(chunk).for_each(|coeffs| {
                let (left, right) = coeffs.split_at_mut(chunk / 2);

                // case when twiddle factor is one
                let (a, left) = left.split_at_mut(1);
                let (b, right) = right.split_at_mut(1);
                let t = b[0];
                b[0] = a[0];
                a[0].group_add(&t);
                b[0].group_sub(&t);

                left.iter_mut()
                    .zip(right.iter_mut())
                    .enumerate()
                    .for_each(|(i, (a, b))| {
                        let mut t = *b;
                        t.group_scale(&twiddles[(i + 1) * twiddle_chunk]);
                        *b = *a;
                        a.group_add(&t);
                        b.group_sub(&t);
                    });
            });
            chunk *= 2;
            twiddle_chunk /= 2;
        }
    } else {
        recursive_butterfly_arithmetic(a, n, 1, &twiddles, 0)
    }
}

pub fn recursive_butterfly_arithmetic<G: Group>(
    a: &mut [G],
    n: usize,
    twiddle_chunk: usize,
    twiddles: &[G::Scalar],
    level: u32,
) {
    if n == 2 {
        let t = a[1];
        a[1] = a[0];
        a[0].group_add(&t);
        a[1].group_sub(&t);
    } else {
        let (left, right) = a.split_at_mut(n / 2);

        rayon::join(
            || recursive_butterfly_arithmetic(left, n / 2, twiddle_chunk * 2, twiddles, level + 1),
            || recursive_butterfly_arithmetic(right, n / 2, twiddle_chunk * 2, twiddles, level + 1),
        );

        // case when twiddle factor is one
        let (a, left) = left.split_at_mut(1);
        let (b, right) = right.split_at_mut(1);
        let t = b[0];
        b[0] = a[0];
        a[0].group_add(&t);
        b[0].group_sub(&t);

        let chunk_size = 512;
        if n > chunk_size << 2 && level < 4 {
            left.par_chunks_mut(chunk_size)
                .zip(right.par_chunks_mut(chunk_size))
                .enumerate()
                .for_each(|(i, (left, right))| {
                    left.iter_mut()
                        .zip(right.iter_mut())
                        .enumerate()
                        .for_each(|(j, (a, b))| {
                            let mut t = *b;
                            t.group_scale(&twiddles[(i * chunk_size + j + 1) * twiddle_chunk]);
                            *b = *a;
                            a.group_add(&t);
                            b.group_sub(&t);
                        });
                });
        } else {
            left.iter_mut()
                .zip(right.iter_mut())
                .enumerate()
                .for_each(|(i, (a, b))| {
                    let mut t = *b;
                    t.group_scale(&twiddles[(i + 1) * twiddle_chunk]);
                    *b = *a;
                    a.group_add(&t);
                    b.group_sub(&t);
                });
        }
    }
}

/// This evaluates a provided polynomial (in coefficient form) at `point`.
pub fn eval_polynomial<F: Field>(poly: &[F], point: F) -> F {
    fn evaluate<F: Field>(poly: &[F], point: F) -> F {
        poly.iter()
            .rev()
            .fold(F::zero(), |acc, coeff| acc * point + coeff)
    }
    let n = poly.len();
    let num_threads = multicore::current_num_threads();
    if n * 2 < num_threads {
        evaluate(poly, point)
    } else {
        let chunk_size = (n + num_threads - 1) / num_threads;
        let mut parts = vec![F::zero(); num_threads];
        multicore::scope(|scope| {
            for (chunk_idx, (out, poly)) in
                parts.chunks_mut(1).zip(poly.chunks(chunk_size)).enumerate()
            {
                scope.spawn(move |_| {
                    let start = chunk_idx * chunk_size;
                    out[0] = evaluate(poly, point) * point.pow_vartime(&[start as u64, 0, 0, 0]);
                });
            }
        });
        parts.iter().fold(F::zero(), |acc, coeff| acc + coeff)
    }
}

/// This computes the inner product of two vectors `a` and `b`.
///
/// This function will panic if the two vectors are not the same size.
pub fn compute_inner_product<F: Field>(a: &[F], b: &[F]) -> F {
    // TODO: parallelize?
    assert_eq!(a.len(), b.len());

    let mut acc = F::zero();
    for (a, b) in a.iter().zip(b.iter()) {
        acc += (*a) * (*b);
    }

    acc
}

/// Divides polynomial `a` in `X` by `X - b` with
/// no remainder.
pub fn kate_division<'a, F: Field, I: IntoIterator<Item = &'a F>>(a: I, mut b: F) -> Vec<F>
where
    I::IntoIter: DoubleEndedIterator + ExactSizeIterator,
{
    b = -b;
    let a = a.into_iter();

    let mut q = vec![F::zero(); a.len() - 1];

    let mut tmp = F::zero();
    for (q, r) in q.iter_mut().rev().zip(a.rev()) {
        let mut lead_coeff = *r;
        lead_coeff.sub_assign(&tmp);
        *q = lead_coeff;
        tmp = lead_coeff;
        tmp.mul_assign(&b);
    }

    q
}

/// This simple utility function will parallelize an operation that is to be
/// performed over a mutable slice.
pub fn parallelize<T: Send, F: Fn(&mut [T], usize) + Send + Sync + Clone>(v: &mut [T], f: F) {
    let n = v.len();
    let num_threads = multicore::current_num_threads();
    let mut chunk = (n as usize) / num_threads;
    if chunk < num_threads {
        chunk = n as usize;
    }

    multicore::scope(|scope| {
        for (chunk_num, v) in v.chunks_mut(chunk).enumerate() {
            let f = f.clone();
            scope.spawn(move |_| {
                let start = chunk_num * chunk;
                f(v, start);
            });
        }
    });
}

fn log2_floor(num: usize) -> u32 {
    assert!(num > 0);

    let mut pow = 0;

    while (1 << (pow + 1)) <= num {
        pow += 1;
    }

    pow
}

/// Returns coefficients of an n - 1 degree polynomial given a set of n points
/// and their evaluations. This function will panic if two values in `points`
/// are the same.
pub fn lagrange_interpolate<F: FieldExt>(points: &[F], evals: &[F]) -> Vec<F> {
    assert_eq!(points.len(), evals.len());
    if points.len() == 1 {
        // Constant polynomial
        return vec![evals[0]];
    } else {
        let mut denoms = Vec::with_capacity(points.len());
        for (j, x_j) in points.iter().enumerate() {
            let mut denom = Vec::with_capacity(points.len() - 1);
            for x_k in points
                .iter()
                .enumerate()
                .filter(|&(k, _)| k != j)
                .map(|a| a.1)
            {
                denom.push(*x_j - x_k);
            }
            denoms.push(denom);
        }
        // Compute (x_j - x_k)^(-1) for each j != i
        denoms.iter_mut().flat_map(|v| v.iter_mut()).batch_invert();

        let mut final_poly = vec![F::zero(); points.len()];
        for (j, (denoms, eval)) in denoms.into_iter().zip(evals.iter()).enumerate() {
            let mut tmp: Vec<F> = Vec::with_capacity(points.len());
            let mut product = Vec::with_capacity(points.len() - 1);
            tmp.push(F::one());
            for (x_k, denom) in points
                .iter()
                .enumerate()
                .filter(|&(k, _)| k != j)
                .map(|a| a.1)
                .zip(denoms.into_iter())
            {
                product.resize(tmp.len() + 1, F::zero());
                for ((a, b), product) in tmp
                    .iter()
                    .chain(std::iter::once(&F::zero()))
                    .zip(std::iter::once(&F::zero()).chain(tmp.iter()))
                    .zip(product.iter_mut())
                {
                    *product = *a * (-denom * x_k) + *b * denom;
                }
                std::mem::swap(&mut tmp, &mut product);
            }
            assert_eq!(tmp.len(), points.len());
            assert_eq!(product.len(), points.len() - 1);
            for (final_coeff, interpolation_coeff) in final_poly.iter_mut().zip(tmp.into_iter()) {
                *final_coeff += interpolation_coeff * eval;
            }
        }
        final_poly
    }
}

pub(crate) fn evaluate_vanishing_polynomial<F: FieldExt>(roots: &[F], z: F) -> F {
    fn evaluate<F: FieldExt>(roots: &[F], z: F) -> F {
        roots.iter().fold(F::one(), |acc, point| (z - point) * acc)
    }
    let n = roots.len();
    let num_threads = multicore::current_num_threads();
    if n * 2 < num_threads {
        evaluate(roots, z)
    } else {
        let chunk_size = (n + num_threads - 1) / num_threads;
        let mut parts = vec![F::one(); num_threads];
        multicore::scope(|scope| {
            for (out, roots) in parts.chunks_mut(1).zip(roots.chunks(chunk_size)) {
                scope.spawn(move |_| out[0] = evaluate(roots, z));
            }
        });
        parts.iter().fold(F::one(), |acc, part| acc * part)
    }
}

#[cfg(test)]
use rand_core::OsRng;

#[cfg(test)]
use pairing::bn256::Fr as Fp;
use rayon::prelude::IntoParallelRefIterator;

#[test]
fn test_lagrange_interpolate() {
    let rng = OsRng;

    let points = (0..5).map(|_| Fp::random(rng)).collect::<Vec<_>>();
    let evals = (0..5).map(|_| Fp::random(rng)).collect::<Vec<_>>();

    for coeffs in 0..5 {
        let points = &points[0..coeffs];
        let evals = &evals[0..coeffs];

        let poly = lagrange_interpolate(points, evals);
        assert_eq!(poly.len(), points.len());

        for (point, eval) in points.iter().zip(evals) {
            assert_eq!(eval_polynomial(&poly, *point), *eval);
        }
    }
}
