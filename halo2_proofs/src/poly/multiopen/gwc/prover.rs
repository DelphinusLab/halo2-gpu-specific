use super::{construct_intermediate_sets, ChallengeV, Query};
use crate::arithmetic::{
    eval_polynomial, eval_polynomial_st, kate_division, CurveAffine, FieldExt,
};
use crate::poly::multiopen::ProverQuery;
use crate::poly::Rotation;
use crate::poly::{commitment::Params, Coeff, Polynomial};
use crate::transcript::{EncodedChallenge, TranscriptWrite};

use ark_std::{end_timer, start_timer};
use ff::Field;
use group::Curve;
use rayon::iter::*;
use std::io;
use std::marker::PhantomData;
use std::sync::Mutex;

/// Create a multi-opening proof
pub fn create_proof<'a, I, C: CurveAffine, E: EncodedChallenge<C>, T: TranscriptWrite<C, E>>(
    params: &Params<C>,
    transcript: &mut T,
    queries: I,
) -> io::Result<()>
where
    I: IntoIterator<Item = ProverQuery<'a, C>>,
{
    let v: ChallengeV<_> = transcript.squeeze_challenge_scalar();
    let commitment_data = construct_intermediate_sets(queries);

    let zero = || Polynomial::<C::Scalar, Coeff> {
        values: vec![C::Scalar::zero(); params.n as usize],
        _marker: PhantomData,
    };

    let mut ws = vec![C::identity(); commitment_data.len()];

    commitment_data
        .par_iter()
        .zip(ws.par_iter_mut())
        .for_each(|(commitment_at_a_point, w)| {
            let z = commitment_at_a_point.point;

            #[cfg(not(feature = "cuda"))]
            let (poly_batch, eval_batch) = {
                let mut poly_batch = zero();
                let mut eval_batch = C::Scalar::zero();
                for (query, eval) in commitment_at_a_point.queries.iter().zip(evals.iter()) {
                    assert_eq!(query.get_point(), z);

                    let poly = query.get_commitment().poly;
                    poly_batch = poly_batch * *v + poly;
                    eval_batch = eval_batch * *v + eval;
                }
                (poly_batch, eval_batch)
            };

            #[cfg(feature = "cuda")]
            let poly_batch = {
                if commitment_at_a_point.queries.len() < 10 {
                    let mut poly_batch = zero();
                    for query in commitment_at_a_point.queries.iter() {
                        assert_eq!(query.get_point(), z);

                        let poly = query.get_commitment().poly;
                        poly_batch = poly_batch * *v + poly;
                    }
                    poly_batch
                } else {
                    use crate::arithmetic::acquire_gpu;
                    use crate::arithmetic::release_gpu;
                    use crate::plonk::{GPU_COND_VAR, GPU_LOCK};
                    use ec_gpu_gen::rust_gpu_tools::program_closures;
                    use ec_gpu_gen::{
                        fft::FftKernel, multiexp::SingleMultiexpKernel, rust_gpu_tools::Device,
                        threadpool::Worker,
                    };
                    use group::Curve;
                    use pairing::bn256::Fr;

                    let mut poly_batch = zero();

                    let gpu_idx = acquire_gpu();
                    let closures =
                        program_closures!(|program,
                                           input: &mut [C::ScalarExt]|
                         -> ec_gpu_gen::EcResult<()> {
                            let size = params.n as usize;
                            let local_work_size = 128;
                            let global_work_size = size / local_work_size;
                            let vl = vec![*v];
                            let v_buffer = program.create_buffer_from_slice(&vl[..])?;
                            let mut it = commitment_at_a_point.queries.iter();
                            let mut tmp_buffer = unsafe { program.create_buffer(size)? };
                            let query = it.next().unwrap();
                            let res_buffer = program.create_buffer_from_slice(
                                &query.get_commitment().poly.values[..],
                            )?;
                            for query in it {
                                let kernel_name = format!("{}_eval_mul_c", "Bn256_Fr");
                                let kernel = program.create_kernel(
                                    &kernel_name,
                                    global_work_size as usize,
                                    local_work_size as usize,
                                )?;
                                kernel
                                    .arg(&res_buffer)
                                    .arg(&res_buffer)
                                    .arg(&0)
                                    .arg(&v_buffer)
                                    .arg(&(size as u32))
                                    .run()?;

                                program.write_from_buffer(
                                    &mut tmp_buffer,
                                    &query.get_commitment().poly.values[..],
                                )?;

                                let kernel_name = format!("{}_eval_sum", "Bn256_Fr");
                                let kernel = program.create_kernel(
                                    &kernel_name,
                                    global_work_size as usize,
                                    local_work_size as usize,
                                )?;
                                kernel
                                    .arg(&res_buffer)
                                    .arg(&res_buffer)
                                    .arg(&tmp_buffer)
                                    .arg(&0)
                                    .arg(&0)
                                    .arg(&(size as u32))
                                    .run()?;
                            }
                            program.read_into_buffer(&res_buffer, input)?;
                            Ok(())
                        });

                    let devices = Device::all();
                    let device = devices[gpu_idx % devices.len()];
                    let program = ec_gpu_gen::program!(device).unwrap();
                    program
                        .run(closures, unsafe {
                            std::mem::transmute::<_, &mut [C::ScalarExt]>(
                                &mut poly_batch.values[..],
                            )
                        })
                        .unwrap();

                    release_gpu(gpu_idx);
                    poly_batch
                }
            };

            let eval_batch =
                eval_polynomial_st(&poly_batch, commitment_at_a_point.queries[0].get_point());

            let poly_batch = &poly_batch - eval_batch;
            let witness_poly = Polynomial {
                values: kate_division(&poly_batch.values, z),
                _marker: PhantomData,
            };

            *w = params.commit(&witness_poly).to_affine();
        });

    for w in ws {
        transcript.write_point(w)?;
    }

    Ok(())
}

#[doc(hidden)]
#[derive(Copy, Clone, Debug)]
pub struct PolynomialPointer<'a, C: CurveAffine> {
    poly: &'a Polynomial<C::Scalar, Coeff>,
}

impl<'a, C: CurveAffine> PartialEq for PolynomialPointer<'a, C> {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self.poly, other.poly)
    }
}

impl<'a, C: CurveAffine> Query<C::Scalar> for ProverQuery<'a, C> {
    type Commitment = PolynomialPointer<'a, C>;

    fn get_point(&self) -> C::Scalar {
        self.point
    }
    fn get_rotation(&self) -> Rotation {
        self.rotation
    }
    fn get_eval(&self) -> C::Scalar {
        eval_polynomial_st(self.poly, self.get_point())
    }
    fn get_commitment(&self) -> Self::Commitment {
        PolynomialPointer { poly: self.poly }
    }
}
