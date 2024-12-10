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
    inputs: [Column<Advice>; 6],
    s_0: Column<Fixed>,
    s_1: Column<Fixed>,
    table: TableColumn,
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
        inputs: [Column<Advice>; 6],
        s_0: Column<Fixed>,
        s_1: Column<Fixed>,
        table: TableColumn,
    ) -> SimpleConfig {
        meta.create_gate("", |meta| {
            let input_0 = meta.query_advice(inputs[0], Rotation::cur());
            let input_1 = meta.query_advice(inputs[1], Rotation::cur());
            let s0 = meta.query_fixed(s_0, Rotation::cur());
            vec![s0 * (input_0 * F::from(1) - input_1)]
        });

        //set 0
        meta.lookup("table0", |meta| {
            let input_0 = meta.query_advice(inputs[0], Rotation::cur());
            [(input_0, table)].to_vec()
        });

        //set 1
        meta.lookup("table1", |meta| {
            let input_1 = meta.query_advice(inputs[1], Rotation::cur());
            [(input_1 * F::from(2), table)].to_vec()
        });
        meta.lookup("table2", |meta| {
            let input_2 = meta.query_advice(inputs[2], Rotation::cur());
            [(input_2, table)].to_vec()
        });

        //set 2
        meta.lookup("table3", |meta| {
            let input_3 = meta.query_advice(inputs[3], Rotation::cur());
            [(input_3 * F::from(10), table)].to_vec()
        });
        meta.lookup("table4", |meta| {
            let input_4 = meta.query_advice(inputs[4], Rotation::cur());
            [(input_4, table)].to_vec()
        });

        //set 3
        meta.lookup("table5", |meta| {
            let input_5 = meta.query_advice(inputs[5], Rotation::cur());
            [(input_5, table)].to_vec()
        });

        SimpleConfig {
            inputs,
            s_0,
            s_1,
            table,
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
        let inputs = [(); 6].map(|_| meta.advice_column());
        let [s_0, s_1] = [(); 2].map(|_| meta.fixed_column());
        let [table] = [(); 1].map(|_| meta.lookup_table_column());
        SimpleChip::configure(meta, inputs, s_0, s_1, table)
    }

    fn synthesize(&self, config: Self::Config, layouter: impl Layouter<F>) -> Result<(), Error> {
        let ch = SimpleChip::<F>::construct(config.clone());

        layouter.assign_region(
            || "inputs",
            |region: &Region<'_, F>| {
                for i in 0..6 {
                    region.assign_advice(
                        || "",
                        ch.config.inputs[i],
                        0,
                        || Ok(F::from(1 as u64)),
                    )?;
                }
                region.assign_fixed(|| "", ch.config.s_0, 0, || Ok(F::from(1)))?;

                for i in 0..6 {
                    region.assign_advice(
                        || "",
                        ch.config.inputs[i],
                        1,
                        || Ok(F::from(3 as u64)),
                    )?;
                }
                region.assign_fixed(|| "", ch.config.s_1, 1, || Ok(F::from(1)))?;

                Ok(())
            },
        )?;
        layouter.assign_table(
            || "common range table",
            |table| {
                for i in 0..100 {
                    table.assign_cell(
                        || "range tag",
                        ch.config.table,
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
    let k = 10;

    let circuit = MyCircuit::<Fp> {
        _marker: PhantomData,
    };

    let prover = MockProver::run(k, &circuit, vec![]).unwrap();
    assert_eq!(prover.verify(), Ok(()));

    test_prover(k, circuit);
}
