use halo2_proofs::arithmetic::FieldExt;
use halo2_proofs::circuit::{Chip, Layouter, Region, SimpleFloorPlanner};
use halo2_proofs::dev::MockProver;
use halo2_proofs::plonk::*;
use pairing::bn256::Fr as Fp;
use pairing::bn256::{Bn256, G1Affine};

use std::marker::PhantomData;

use halo2_proofs::poly::{
    commitment::{Params, ParamsVerifier},
    Rotation,
};
use halo2_proofs::transcript::{Blake2bRead, Blake2bWrite, Challenge255};
use rand_core::OsRng;

#[derive(Clone, Debug)]
struct ShuffleChip<F: FieldExt> {
    config: ShuffleConfig,
    _marker: PhantomData<F>,
}

#[derive(Clone, Debug)]
struct ShuffleConfig {
    input_0: Column<Advice>,
    input_1: Column<Advice>,
    shuffle_0: Column<Advice>,
    shuffle_1: Column<Advice>,
    s_input: Column<Fixed>,
    s_shuffle: Column<Fixed>,
}

impl<F: FieldExt> Chip<F> for ShuffleChip<F> {
    type Config = ShuffleConfig;
    type Loaded = ();

    fn config(&self) -> &Self::Config {
        &self.config
    }
    fn loaded(&self) -> &Self::Loaded {
        &()
    }
}

impl<F: FieldExt> ShuffleChip<F> {
    fn construct(config: ShuffleConfig) -> Self {
        Self {
            config,
            _marker: PhantomData,
        }
    }

    fn configure(
        meta: &mut ConstraintSystem<F>,
        input_0: Column<Advice>,
        input_1: Column<Advice>,
        shuffle_0: Column<Advice>,
        shuffle_1: Column<Advice>,
        s_input: Column<Fixed>,
        s_shuffle: Column<Fixed>,
    ) -> ShuffleConfig {
        //need at least one gate or GPU will panic
        meta.create_gate("", |meta| {
            let input_0 = meta.query_advice(input_0, Rotation::cur());
            let input_1 = meta.query_advice(input_1, Rotation::cur());
            let s_input = meta.query_fixed(s_input, Rotation::cur());
            vec![s_input * (input_0 * F::from(10) - input_1)]
        });

        meta.shuffle("shuffle", |meta| {
            let input_0 = meta.query_advice(input_0, Rotation::cur());
            let shuffle_0 = meta.query_advice(shuffle_0, Rotation::cur());
            let input_1 = meta.query_advice(input_1, Rotation::cur());
            let shuffle_1 = meta.query_advice(shuffle_1, Rotation::cur());
            let s_input = meta.query_fixed(s_input, Rotation::cur());
            let s_shuffle = meta.query_fixed(s_shuffle, Rotation::cur());

            [
                (s_input.clone() * input_0, s_shuffle.clone() * shuffle_0),
                (s_input * input_1, s_shuffle * shuffle_1),
            ]
            .to_vec()
        });

        ShuffleConfig {
            input_0,
            input_1,
            shuffle_0,
            shuffle_1,
            s_input,
            s_shuffle,
        }
    }
}

#[derive(Default, Clone, Debug)]
struct MyCircuit<F: FieldExt> {
    input0: Vec<F>,
    input1: Vec<F>,
    shuffle0: Vec<F>,
    shuffle1: Vec<F>,
}

impl<F: FieldExt> MyCircuit<F> {
    fn construct() -> Self {
        Self {
            input0: [1, 2, 4, 1].map(|x| F::from(x as u64)).to_vec(),
            shuffle0: [4, 1, 1, 2].map(|x| F::from(x as u64)).to_vec(),
            input1: [10, 20, 40, 10].map(|x| F::from(x as u64)).to_vec(),
            shuffle1: [40, 10, 10, 20].map(|x| F::from(x as u64)).to_vec(),
        }
    }
}

impl<F: FieldExt> Circuit<F> for MyCircuit<F> {
    type Config = ShuffleConfig;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self::default()
    }

    fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
        let [input_0, input_1, shuffle_0, shuffle_1] = [(); 4].map(|_| meta.advice_column());
        let [s_input, s_shuffle] = [(); 2].map(|_| meta.fixed_column());

        ShuffleChip::configure(
            meta, input_0, input_1, shuffle_0, shuffle_1, s_input, s_shuffle,
        )
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<F>,
    ) -> Result<(), Error> {
        let ch = ShuffleChip::<F>::construct(config.clone());

        layouter.assign_region(
            || "inputs",
            |mut region: Region<'_, F>| {
                for (i, (input0, input1)) in self.input0.iter().zip(self.input1.iter()).enumerate()
                {
                    region.assign_advice(|| "", ch.config.input_0, i, || Ok(*input0))?;
                    region.assign_advice(|| "", ch.config.input_1, i, || Ok(*input1))?;

                    region.assign_fixed(|| "", ch.config.s_input, i, || Ok(F::from(1)))?;
                }
                Ok(())
            },
        )?;
        layouter.assign_region(
            || "shuffles",
            |mut region: Region<'_, F>| {
                for (i, (shuffle0, shuffle1)) in
                    self.shuffle0.iter().zip(self.shuffle1.iter()).enumerate()
                {
                    region.assign_advice(|| "", ch.config.shuffle_0, i, || Ok(*shuffle0))?;
                    region.assign_advice(|| "", ch.config.shuffle_1, i, || Ok(*shuffle1))?;

                    region.assign_fixed(|| "", ch.config.s_shuffle, i, || Ok(F::from(1)))?;
                }
                Ok(())
            },
        )
    }
}

fn test_prover(k: u32, circuit: MyCircuit<Fp>) {
    let public_inputs_size = 0;
    // Initialize the polynomial commitment parameters
    let params: Params<G1Affine> = Params::<G1Affine>::unsafe_setup::<Bn256>(k);
    let params_verifier: ParamsVerifier<Bn256> = params.verifier(public_inputs_size).unwrap();

    let vk = keygen_vk(&params, &circuit).unwrap();
    let pk = keygen_pk(&params, vk, &circuit).unwrap();

    let mut transcript = Blake2bWrite::<_, _, Challenge255<_>>::init(vec![]);

    create_proof(&params, &pk, &[circuit], &[&[]], OsRng, &mut transcript)
        .expect("proof generation should not fail");

    let proof = transcript.finalize();

    let strategy = SingleVerifier::new(&params_verifier);
    let mut transcript = Blake2bRead::<_, _, Challenge255<_>>::init(&proof[..]);

    assert!(verify_proof(
        &params_verifier,
        pk.get_vk(),
        strategy,
        &[&[]],
        &mut transcript,
    )
    .is_ok());
}

fn main() {
    // The number of rows in our circuit cannot exceed 2^k
    let k = 10;

    let circuit = MyCircuit::<Fp>::construct();

    let prover = MockProver::run(k, &circuit, vec![]).unwrap();
    assert_eq!(prover.verify(), Ok(()));

    test_prover(k, circuit);
}
