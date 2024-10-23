use halo2_proofs::arithmetic::FieldExt;
use halo2_proofs::circuit::{floor_planner::V1, Chip, Layouter, Region};
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
struct SimpleChip<F: FieldExt> {
    config: SimpleConfig,
    _marker: PhantomData<F>,
}

#[derive(Clone, Debug)]
struct SimpleConfig {
    input_0: Column<Advice>,
    input_1: Column<Advice>,
    lookup_0: Column<Advice>,
    s_0: Column<Fixed>,
    s_1: Column<Fixed>,
    s_table0: TableColumn,
}

impl<F: FieldExt> Chip<F> for SimpleChip<F> {
    type Config = SimpleConfig;
    type Loaded = ();

    fn config(&self) -> &Self::Config {
        &self.config
    }
    fn loaded(&self) -> &Self::Loaded {
        &()
    }
}

impl<F: FieldExt> SimpleChip<F> {
    fn construct(config: SimpleConfig) -> Self {
        Self {
            config,
            _marker: PhantomData,
        }
    }

    fn configure(
        meta: &mut ConstraintSystem<F>,
        input_0: Column<Advice>,
        input_1: Column<Advice>,
        lookup_0: Column<Advice>,
        s_0: Column<Fixed>,
        s_1: Column<Fixed>,
        s_table0: TableColumn,
    ) -> SimpleConfig {
        meta.create_gate("", |meta| {
            let input_0 = meta.query_advice(input_0, Rotation::cur());
            let input_1 = meta.query_advice(input_1, Rotation::cur());
            let s0 = meta.query_fixed(s_0, Rotation::cur());
            vec![s0 * (input_0 * F::from(1) - input_1)]
        });

        meta.lookup("table1", |meta| {
            let input_0 = meta.query_advice(input_0, Rotation::cur());
            [(input_0, s_table0)].to_vec()
        });
        meta.lookup("table2", |meta| {
            let input_1 = meta.query_advice(input_1, Rotation::cur());
            [(input_1 * F::from(2), s_table0)].to_vec()
        });
        meta.lookup("table3", |meta| {
            let lookup_0 = meta.query_advice(lookup_0, Rotation::cur());
            [(lookup_0, s_table0)].to_vec()
        });

        meta.lookup_any("any", |meta| {
            let input_0 = meta.query_advice(input_0, Rotation::cur());
            let input_1 = meta.query_advice(input_1, Rotation::cur());
            let lookup_0 = meta.query_advice(lookup_0, Rotation::cur());
            let s0 = meta.query_fixed(s_0, Rotation::cur());
            let s1 = meta.query_fixed(s_1, Rotation::cur());

            [
                (s0.clone() * input_0.clone(), s0 * input_1),
                (s1.clone() * input_0, s1 * lookup_0),
            ]
            .to_vec()
        });

        SimpleConfig {
            input_0,
            input_1,
            lookup_0,
            s_0,
            s_1,
            s_table0,
        }
    }
}

#[derive(Default, Clone, Debug)]
struct MyCircuit<F: FieldExt> {
    _marker: PhantomData<F>,
}

impl<F: FieldExt> Circuit<F> for MyCircuit<F> {
    type Config = SimpleConfig;
    type FloorPlanner = V1;

    fn without_witnesses(&self) -> Self {
        Self::default()
    }

    fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
        let [input_0, input_1, lookup_0, lookup_1] = [(); 4].map(|_| meta.advice_column());
        let [s_0, s_1] = [(); 2].map(|_| meta.fixed_column());
        let [s_table0, s_table1] = [(); 2].map(|_| meta.lookup_table_column());
        SimpleChip::configure(meta, input_0, input_1, lookup_0, s_0, s_1, s_table0)
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<F>,
    ) -> Result<(), Error> {
        let ch = SimpleChip::<F>::construct(config.clone());

        layouter.assign_region(
            || "inputs",
            |mut region: &Region<'_, F>| {
                region.assign_advice(|| "", ch.config.input_0, 0, || Ok(F::from(1 as u64)))?;
                region.assign_advice(|| "", ch.config.input_1, 0, || Ok(F::from(1 as u64)))?;
                region.assign_fixed(|| "", ch.config.s_0, 0, || Ok(F::from(1)))?;

                region.assign_advice(|| "", ch.config.input_0, 1, || Ok(F::from(3 as u64)))?;
                region.assign_advice(|| "", ch.config.lookup_0, 1, || Ok(F::from(3 as u64)))?;
                region.assign_fixed(|| "", ch.config.s_1, 1, || Ok(F::from(1)))?;

                Ok(())
            },
        )?;
        layouter.assign_table(
            || "common range table",
            |table| {
                for i in 0..9 {
                    table.assign_cell(
                        || "range tag",
                        ch.config.s_table0,
                        i,
                        || Ok(F::from(i as u64)),
                    )?;
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
    let k = 4;

    // let circuit = MyCircuit::<Fp>::construct();
    let circuit = MyCircuit::<Fp> {
        _marker: PhantomData,
    };

    let prover = MockProver::run(k, &circuit, vec![]).unwrap();
    assert_eq!(prover.verify(), Ok(()));

    test_prover(k, circuit);
}
