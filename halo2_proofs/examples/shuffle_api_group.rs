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
    inputs: Vec<Column<Advice>>,
    shuffles: Vec<Column<Advice>>,
    s_inputs: Vec<Column<Fixed>>,
    s_shuffles: Vec<Column<Fixed>>,
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
        inputs: &[Column<Advice>],
        shuffles: &[Column<Advice>],
        s_inputs: &[Column<Fixed>],
        s_shuffles: &[Column<Fixed>],
    ) -> ShuffleConfig {
        //need at least one gate or GPU will panic
        meta.create_gate("", |meta| {
            let input_0 = meta.query_advice(inputs[0], Rotation::cur());
            let input_1 = meta.query_advice(inputs[1], Rotation::cur());
            let s_input = meta.query_fixed(s_inputs[0], Rotation::cur());
            vec![s_input * (input_0 - input_1)]
        });

        meta.shuffle("shuffle1", |meta| {
            let input_0 = meta.query_advice(inputs[0], Rotation::cur());
            let shuffle_0 = meta.query_advice(shuffles[0], Rotation::cur());
            let input_1 = meta.query_advice(inputs[1], Rotation::cur());
            let shuffle_1 = meta.query_advice(shuffles[1], Rotation::cur());

            [(input_0, shuffle_0), (input_1, shuffle_1)].to_vec()
        });

        meta.shuffle("shuffle2", |meta| {
            let input = meta.query_advice(inputs[2], Rotation::cur());
            let shuffle = meta.query_advice(shuffles[2], Rotation::cur());
            [(input, shuffle)].to_vec()
        });

        meta.shuffle("shuffle3", |meta| {
            let input = meta.query_advice(inputs[3], Rotation::cur());
            let shuffle = meta.query_advice(shuffles[3], Rotation::cur());
            let s_input = meta.query_fixed(s_inputs[0], Rotation::cur());
            let s_shuffle = meta.query_fixed(s_shuffles[0], Rotation::cur());
            [(input * s_input, shuffle * s_shuffle)].to_vec()
        });

        meta.shuffle("shuffle4", |meta| {
            let input = meta.query_advice(inputs[4], Rotation::cur());
            let shuffle = meta.query_advice(shuffles[4], Rotation::cur());
            let s_input0 = meta.query_fixed(s_inputs[0], Rotation::cur());
            let s_shuffle0 = meta.query_fixed(s_shuffles[0], Rotation::cur());
            let s_input1 = meta.query_fixed(s_inputs[1], Rotation::cur());
            let s_shuffle1 = meta.query_fixed(s_shuffles[1], Rotation::cur());
            [(
                input * s_input0 * s_input1,
                shuffle * s_shuffle0 * s_shuffle1,
            )]
            .to_vec()
        });

        ShuffleConfig {
            inputs: inputs.to_vec(),
            shuffles: shuffles.to_vec(),
            s_inputs: s_inputs.to_vec(),
            s_shuffles: s_shuffles.to_vec(),
        }
    }
}

#[derive(Default, Clone, Debug)]
struct MyCircuit<F: FieldExt> {
    input0: Vec<F>,
    input1: Vec<F>,
}

impl<F: FieldExt> MyCircuit<F> {
    fn construct() -> Self {
        Self {
            input0: [1, 2, 4, 1].map(|x| F::from(x as u64)).to_vec(),
            input1: [4, 1, 1, 2].map(|x| F::from(x as u64)).to_vec(),
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
        let inputs: Vec<_> = (0..5).into_iter().map(|_| meta.advice_column()).collect();
        let shuffles: Vec<_> = (0..5).into_iter().map(|_| meta.advice_column()).collect();
        let s_inputs: Vec<_> = (0..2).into_iter().map(|_| meta.fixed_column()).collect();
        let s_shuffles: Vec<_> = (0..2).into_iter().map(|_| meta.fixed_column()).collect();
        ShuffleChip::configure(meta, &inputs, &shuffles, &s_inputs, &s_shuffles)
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
                    region.assign_advice(|| "", ch.config.inputs[0], i, || Ok(*input0))?;
                    region.assign_advice(|| "", ch.config.inputs[1], i, || Ok(*input0))?;
                    region.assign_advice(|| "", ch.config.inputs[2], i, || Ok(*input0))?;
                    region.assign_advice(|| "", ch.config.inputs[3], i, || Ok(*input0))?;
                    region.assign_advice(|| "", ch.config.inputs[4], i, || Ok(*input0))?;

                    region.assign_advice(|| "", ch.config.shuffles[0], i, || Ok(*input1))?;
                    region.assign_advice(|| "", ch.config.shuffles[1], i, || Ok(*input1))?;
                    region.assign_advice(|| "", ch.config.shuffles[2], i, || Ok(*input1))?;
                    region.assign_advice(|| "", ch.config.shuffles[3], i, || Ok(*input1))?;
                    region.assign_advice(|| "", ch.config.shuffles[4], i, || Ok(*input1))?;

                    region.assign_fixed(|| "", ch.config.s_inputs[0], i, || Ok(F::from(1)))?;
                    region.assign_fixed(|| "", ch.config.s_shuffles[0], i, || Ok(F::from(1)))?;
                    region.assign_fixed(|| "", ch.config.s_inputs[1], i, || Ok(F::from(1)))?;
                    region.assign_fixed(|| "", ch.config.s_shuffles[1], i, || Ok(F::from(1)))?;
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
