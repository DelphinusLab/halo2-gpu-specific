use halo2_proofs::{
    arithmetic::FieldExt,
    circuit::{floor_planner::V1, Layouter},
    dev::MockProver,
    plonk::*,
    poly::commitment::{Params, ParamsVerifier},
    transcript::{Blake2bRead, Blake2bWrite, Challenge255},
};
use pairing::bn256::{Bn256, Fr as Fp, G1Affine};
use rand::Rng;
use rand_core::OsRng;

use std::marker::PhantomData;

const K: usize = 18;

#[derive(Clone)]
struct RangeCheckConfig {
    l_0: Column<Fixed>,
    l_active: Column<Fixed>,
    l_last_active: Column<Fixed>,
    adv: Column<Advice>,
    l_last_offset: usize,
}

struct TestCircuit<F: FieldExt> {
    _mark: PhantomData<F>,
}

impl<F: FieldExt> Circuit<F> for TestCircuit<F> {
    type Config = RangeCheckConfig;
    type FloorPlanner = V1;

    fn without_witnesses(&self) -> Self {
        Self { _mark: PhantomData }
    }

    fn configure(meta: &mut ConstraintSystem<F>) -> RangeCheckConfig {
        let l_0 = meta.fixed_column();
        let l_active = meta.fixed_column();
        let l_last_active = meta.fixed_column();

        let adv = meta.advice_column_range(
            l_0,
            l_active,
            l_last_active,
            (0, F::from(0)),
            (u16::MAX as u32, F::from(u16::MAX as u64)),
            (2, F::from(2)),
        );

        let l_last_offset = (1 << K) - (meta.blinding_factors() + 1);

        RangeCheckConfig {
            l_0,
            l_active,
            l_last_active,
            l_last_offset,
            adv,
        }
    }

    fn synthesize(
        &self,
        config: RangeCheckConfig,
        layouter: impl Layouter<F>,
    ) -> Result<(), Error> {
        layouter.assign_region(
            || "region",
            |region| {
                region.assign_fixed(|| "l_0", config.l_0, 0, || Ok(F::one()))?;
                region.assign_fixed(
                    || "l_last_active",
                    config.l_last_active,
                    config.l_last_offset - 1,
                    || Ok(F::one()),
                )?;
                for offset in 0..config.l_last_offset {
                    region.assign_fixed(|| "l_active", config.l_active, offset, || Ok(F::one()))?;
                }

                let mut rng = OsRng;

                for offset in 0..u64::from(u16::MAX) {
                    let value = rng.gen_range(0..=u16::MAX);
                    region.assign_advice(
                        || "advice",
                        config.adv,
                        offset as usize,
                        || Ok(F::from(value as u64)),
                    )?;
                }

                Ok(())
            },
        )?;

        Ok(())
    }
}

fn main() {
    let k = 18;
    let public_inputs_size = 0;

    let circuit: TestCircuit<Fp> = TestCircuit { _mark: PhantomData };

    let prover = MockProver::run(k, &circuit, vec![]).unwrap();
    assert!(prover.verify().is_ok());

    // Initialize the polynomial commitment parameters
    let params: Params<G1Affine> = Params::<G1Affine>::unsafe_setup::<Bn256>(k);
    let params_verifier: ParamsVerifier<Bn256> = params.verifier(public_inputs_size).unwrap();

    // Initialize the proving key
    let vk = keygen_vk(&params, &circuit).expect("keygen_vk should not fail");
    let pk = keygen_pk(&params, vk, &circuit).expect("keygen_pk should not fail");

    // Create a proof
    let mut transcript = Blake2bWrite::<_, _, Challenge255<_>>::init(vec![]);

    create_proof(&params, &pk, &[circuit], &[&[]], OsRng, &mut transcript)
        .expect("proof generation should not fail");

    let proof = transcript.finalize();

    let strategy = SingleVerifier::new(&params_verifier);
    let mut transcript = Blake2bRead::<_, _, Challenge255<_>>::init(&proof[..]);

    verify_proof(
        &params_verifier,
        pk.get_vk(),
        strategy,
        &[&[]],
        &mut transcript,
    )
    .unwrap();
}
