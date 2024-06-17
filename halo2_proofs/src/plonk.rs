//! This module provides an implementation of a variant of (Turbo)[PLONK][plonk]
//! that is designed specifically for the polynomial commitment scheme described
//! in the [Halo][halo] paper.
//!
//! [halo]: https://eprint.iacr.org/2019/1021
//! [plonk]: https://eprint.iacr.org/2019/953

use blake2b_simd::Params as Blake2bParams;

use crate::arithmetic::{BaseExt, CurveAffine, FieldExt};
use crate::helpers::{read_cs, read_u32, write_cs, CurveRead, ParaSerializable, Serializable};
use crate::poly::{
    commitment::Params, Coeff, EvaluationDomain, ExtendedLagrangeCoeff, LagrangeCoeff,
    PinnedEvaluationDomain, Polynomial,
};
use crate::transcript::{ChallengeScalar, EncodedChallenge, Transcript};

pub mod circuit;
pub mod evaluation_gpu;

mod assigned;
mod error;
mod evaluation;
mod keygen;
pub(crate) mod lookup;
pub(crate) mod permutation;
pub mod range_check;
pub(crate) mod shuffle;
mod vanishing;

mod prover;
mod verifier;

pub use assigned::*;
pub use circuit::*;
pub use error::*;
pub use keygen::*;
pub use prover::*;
pub use verifier::*;

use std::fs::File;
use std::io;

use self::evaluation::Evaluator;
use self::permutation::keygen::Assembly;

/// This is a verifying key which allows for the verification of proofs for a
/// particular circuit.
#[derive(Debug, Clone)]
pub struct VerifyingKey<C: CurveAffine> {
    pub domain: EvaluationDomain<C::Scalar>,
    pub fixed_commitments: Vec<C>,
    pub permutation: permutation::VerifyingKey<C>,
    pub cs: ConstraintSystem<C::Scalar>,
}

impl<C: CurveAffine> VerifyingKey<C> {
    /// Writes a verifying key to a buffer.
    pub fn write<W: io::Write>(&self, writer: &mut W) -> io::Result<()> {
        for commitment in &self.fixed_commitments {
            writer.write_all(commitment.to_bytes().as_ref())?;
        }
        self.permutation.write(writer)?;

        Ok(())
    }

    /// Reads a verification key from a buffer.
    pub fn read<R: io::Read, ConcreteCircuit: Circuit<C::Scalar>>(
        reader: &mut R,
        params: &Params<C>,
    ) -> io::Result<Self> {
        let (domain, cs, _) = keygen::create_domain::<C, ConcreteCircuit>(params);

        let fixed_commitments: Vec<_> = (0..cs.num_fixed_columns)
            .map(|_| C::read(reader))
            .collect::<Result<_, _>>()?;

        let permutation = permutation::VerifyingKey::read(reader, &cs.permutation)?;

        Ok(VerifyingKey {
            domain,
            fixed_commitments,
            permutation,
            cs,
        })
    }

    /// Hashes a verification key into a transcript.
    pub fn hash_into<E: EncodedChallenge<C>, T: Transcript<C, E>>(
        &self,
        transcript: &mut T,
    ) -> io::Result<()> {
        let mut hasher = Blake2bParams::new()
            .hash_length(64)
            .personal(b"Halo2-Verify-Key")
            .to_state();

        let s = format!("{:?}", self.pinned());

        hasher.update(&(s.len() as u64).to_le_bytes());
        hasher.update(s.as_bytes());

        // Hash in final Blake2bState
        transcript.common_scalar(C::Scalar::from_bytes_wide(hasher.finalize().as_array()))?;

        Ok(())
    }

    /// Obtains a pinned representation of this verification key that contains
    /// the minimal information necessary to reconstruct the verification key.
    pub fn pinned(&self) -> PinnedVerificationKey<'_, C> {
        PinnedVerificationKey {
            base_modulus: C::Base::MODULUS,
            scalar_modulus: C::Scalar::MODULUS,
            domain: self.domain.pinned(),
            fixed_commitments: &self.fixed_commitments,
            permutation: &self.permutation,
            cs: self.cs.pinned(),
        }
    }
}

#[derive(Debug)]
pub struct CircuitData<C: CurveAffine> {
    vkey: VerifyingKey<C>,
    fixed: Vec<Polynomial<C::Scalar, LagrangeCoeff>>,
    permutation: permutation::keygen::Assembly,
}

impl<C: CurveAffine> CircuitData<C> {
    pub fn new<ConcreteCircuit: Circuit<C::Scalar>>(
        params: &Params<C>,
        vkey: VerifyingKey<C>,
        circuit: &ConcreteCircuit,
    ) -> io::Result<Self> {
        let (fixed, permutation) = generate_pk_info(params, &vkey, circuit).unwrap();

        Ok(CircuitData {
            vkey,
            fixed,
            permutation,
        })
    }

    pub fn read_vkey(reader: &mut File) -> io::Result<VerifyingKey<C>> {
        let j = read_u32(reader)?;
        let k = read_u32(reader)?;
        let domain: EvaluationDomain<C::Scalar> = EvaluationDomain::new(j, k);
        let cs = read_cs::<C, _>(reader)?;

        let fixed_commitments: Vec<_> = (0..cs.num_fixed_columns)
            .map(|_| C::read(reader))
            .collect::<Result<_, _>>()?;

        let permutation = permutation::VerifyingKey::read(reader, &cs.permutation)?;

        Ok(VerifyingKey {
            domain,
            cs,
            fixed_commitments,
            permutation,
        })
    }

    pub fn read(reader: &mut File) -> io::Result<Self> {
        let vkey = Self::read_vkey(reader)?;

        let fixed = Vec::fetch(reader)?;
        let permutation = Assembly::vec_fetch(reader)?;

        Ok(CircuitData {
            vkey,
            fixed,
            permutation,
        })
    }

    pub fn write(&self, fd: &mut File) -> io::Result<()> {
        use std::io::Write;

        let j = (self.vkey.domain.get_quotient_poly_degree() + 1) as u32; // quotient_poly_degree is j-1
        let k = self.vkey.domain.k() as u32;
        fd.write(&mut j.to_le_bytes())?;
        fd.write(&mut k.to_le_bytes())?;
        write_cs::<C, _>(&self.vkey.cs, fd)?;

        self.vkey.write(fd)?;

        self.fixed.store(fd)?;
        self.permutation.vec_store(fd)?;

        Ok(())
    }

    pub fn into_proving_key(self, params: &Params<C>) -> ProvingKey<C> {
        keygen_pk_from_info(params, &self.vkey, self.fixed, self.permutation).unwrap()
    }

    pub fn get_vkey(&self) -> &VerifyingKey<C> {
        &self.vkey
    }
}

/// Minimal representation of a verification key that can be used to identify
/// its active contents.
#[allow(dead_code)]
#[derive(Debug)]
pub struct PinnedVerificationKey<'a, C: CurveAffine> {
    base_modulus: &'static str,
    scalar_modulus: &'static str,
    domain: PinnedEvaluationDomain<'a, C::Scalar>,
    cs: PinnedConstraintSystem<'a, C::Scalar>,
    fixed_commitments: &'a Vec<C>,
    permutation: &'a permutation::VerifyingKey<C>,
}
/// This is a proving key which allows for the creation of proofs for a
/// particular circuit.
#[derive(Debug)]
pub struct ProvingKey<C: CurveAffine> {
    pub vk: VerifyingKey<C>,

    pub l_active_row: Polynomial<C::Scalar, ExtendedLagrangeCoeff>,

    #[cfg(not(feature = "cuda"))]
    l0: Polynomial<C::Scalar, ExtendedLagrangeCoeff>,
    #[cfg(not(feature = "cuda"))]
    l_last: Polynomial<C::Scalar, ExtendedLagrangeCoeff>,

    #[cfg(feature = "cuda")]
    pub l0: Polynomial<C::Scalar, Coeff>,
    #[cfg(feature = "cuda")]
    pub l_last: Polynomial<C::Scalar, Coeff>,

    pub fixed_values: Vec<Polynomial<C::Scalar, LagrangeCoeff>>,
    pub fixed_polys: Vec<Polynomial<C::Scalar, Coeff>>,

    #[cfg(not(feature = "cuda"))]
    fixed_cosets: Vec<Polynomial<C::Scalar, ExtendedLagrangeCoeff>>,
    pub permutation: permutation::ProvingKey<C>,
    pub ev: Evaluator<C>,
}

impl<C: CurveAffine> ProvingKey<C> {
    /// Get the underlying [`VerifyingKey`].
    pub fn get_vk(&self) -> &VerifyingKey<C> {
        &self.vk
    }
}

impl<C: CurveAffine> VerifyingKey<C> {
    /// Get the underlying [`EvaluationDomain`].
    pub fn get_domain(&self) -> &EvaluationDomain<C::Scalar> {
        &self.domain
    }
}

#[derive(Clone, Copy, Debug)]
struct Theta;
type ChallengeTheta<F> = ChallengeScalar<F, Theta>;

#[derive(Clone, Copy, Debug)]
struct Beta;
type ChallengeBeta<F> = ChallengeScalar<F, Beta>;

#[derive(Clone, Copy, Debug)]
struct Gamma;
type ChallengeGamma<F> = ChallengeScalar<F, Gamma>;

#[derive(Clone, Copy, Debug)]
struct Y;
type ChallengeY<F> = ChallengeScalar<F, Y>;

#[derive(Clone, Copy, Debug)]
struct X;
type ChallengeX<F> = ChallengeScalar<F, X>;
