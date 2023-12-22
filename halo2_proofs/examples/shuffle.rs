use halo2_proofs::arithmetic::FieldExt;
use halo2_proofs::circuit::{Cell, Chip, Layouter, Region, SimpleFloorPlanner};
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

#[derive(Clone)]
struct MyConfig {
    input: Column<Advice>,
    shuffle: Column<Advice>,
    input_q: Column<Advice>,
    s: Column<Fixed>,
    s2: Column<Fixed>,
}

/// The full circuit implementation.
///
/// In this struct we store the private input variables. We use `Option<F>` because
/// they won't have any value during key generation. During proving, if any of these
/// were `None` we would get an error.
#[derive(Default, Clone, Debug)]
struct MyCircuit<F: FieldExt> {
    _a: PhantomData<F>,
}

impl<F: FieldExt> MyCircuit<F> {
    fn construct() -> Self {
        return Self { _a: PhantomData };
    }
}

impl<F: FieldExt> Circuit<F> for MyCircuit<F> {
    // Since we are using a single chip for everything, we can just reuse its config.
    type Config = MyConfig;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self::default()
    }

    fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
        // We create the two advice columns that FieldChip uses for I/O.
        let advice = [
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
        ];
        let select = meta.fixed_column();
        let select0 = meta.fixed_column();
        let select2 = meta.fixed_column();

        meta.create_gate("sum equals to instance", |meta| {
            let sel = meta.query_fixed(select0, Rotation::cur());
            let a = meta.query_advice(advice[0], Rotation(0));
            let b = meta.query_advice(advice[1], Rotation(0));
            vec![sel * (a - b)]
        });

        // meta.lookup_any("a to b lookup", |meta| {
        //     let a = meta.query_advice(advice[0], Rotation(0));
        //     let b = meta.query_advice(advice[1], Rotation(0));
        //     vec![(a, b)]
        // });
        //
        //
        // meta.enable_equality(advice[0]);
        // meta.enable_equality(advice[1]);

        meta.shuffle("shuffle1", |cell| {
            let exp_a = cell.query_advice(advice[0], Rotation::cur());
            let exp_b = cell.query_advice(advice[1], Rotation::cur());
            let q = cell.query_fixed(select, Rotation::cur());
            let exp_c = cell.query_advice(advice[2], Rotation::cur());
            // vec![(exp_a, exp_b),(exp_c,q)]
            vec![(q* exp_a, exp_b)]
            // vec![(exp_a, exp_b)]
            // vec![(exp_a, exp_b)]
        });

        meta.shuffle("shuffle2", |cell| {
            let exp_a = cell.query_advice(advice[0], Rotation::cur());
            // let exp_b = cell.query_advice(advice[1], Rotation::cur());
            let q = cell.query_fixed(select2, Rotation::cur());
            let exp_c = cell.query_advice(advice[2], Rotation::cur());
            // vec![(exp_a, exp_b),(exp_c,q)]
            vec![(q* exp_a, exp_c)]
            // vec![(exp_a, exp_b)]
            // vec![(exp_a, exp_b)]
        });

        Self::Config {
            input: advice[0],
            shuffle: advice[1],
            input_q: advice[2],
            s: select,
            s2:select2,
        }
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<F>,
    ) -> Result<(), Error> {
        // layouter.enabled()
        layouter.assign_region(
            || "sys",
            |mut region: Region<'_, F>| {
                let end = 10;
                for i in 0..end {
                    // region.enable_selector(||"sel",&config.s,i);
                    // config.s.enable(&mut region, i);
                    // region.assign_fixed(||"",config.s,i,||Ok(F::from(1).into()))?;
                    // region.assign_advice(||"",config.input_q,i,||Ok(F::from((i as u64+1) *10).into()))?;
                    region.assign_advice(
                        || "input",
                        config.input,
                        i,
                        || Ok(F::from((i as u64 + 1)).into()),
                    )?;
                    // region.assign_advice(
                    //     || "shuffle",
                    //     config.shuffle,
                    //     i,
                    //     || Ok(F::from(i as u64+1).into()),
                    // )?;
                    region.assign_advice(
                        || "shuffle",
                        config.shuffle,
                        i,
                        || Ok(F::from((end - i) as u64).into()),
                    )?;
                    region.assign_fixed(|| "", config.s, i, || Ok(F::from(1).into()))?;
                }
                // region.assign_fixed(|| "", config.s, 0, || Ok(F::from(1).into()))?;
                // region.assign_fixed(|| "", config.s, 1, || Ok(F::from(1).into()))?;
                for i in end..end+10{
                    region.assign_advice(
                        || "input",
                        config.input,
                        i,
                        || Ok(F::from((i as u64 + 1)).into()),
                    )?;
                    region.assign_advice(
                        || "shuffle",
                        config.input_q,
                        i,
                        || Ok(F::from( i as u64 +1).into()),
                    )?;
                    region.assign_fixed(|| "", config.s2, i, || Ok(F::from(1).into()))?;
                }
                Ok(())
            },
        )
    }
}

fn test_prover(K: u32, circuit: MyCircuit<Fp>) {
    let public_inputs_size = 0;
    // Initialize the polynomial commitment parameters
    let params: Params<G1Affine> = Params::<G1Affine>::unsafe_setup::<Bn256>(K);
    let params_verifier: ParamsVerifier<Bn256> = params.verifier(public_inputs_size).unwrap();

    let vk = keygen_vk(&params, &circuit).unwrap();
    let pk = keygen_pk(&params, vk, &circuit).unwrap();

    let mut transcript = Blake2bWrite::<_, _, Challenge255<_>>::init(vec![]);

    create_proof(&params, &pk, &[circuit], &[&[]], OsRng, &mut transcript)
        .expect("proof generation should not fail");

    let proof = transcript.finalize();

    let strategy = SingleVerifier::new(&params_verifier);
    let mut transcript = Blake2bRead::<_, _, Challenge255<_>>::init(&proof[..]);
    let a = verify_proof(
        &params_verifier,
        pk.get_vk(),
        strategy,
        &[&[]],
        &mut transcript,
    );
    println!("a={:?}", a);
    // assert!(verify_proof(
    //     &params_verifier,
    //     pk.get_vk(),
    //     strategy,
    //     &[&[]],
    //     &mut transcript,
    // )
    // .is_ok());
}

fn main() {
    // The number of rows in our circuit cannot exceed 2^k. Since our example
    // circuit is very small, we can pick a very small value here.
    let k = 10;

    // Instantiate the circuit with the private inputs.

    let circuit = MyCircuit::<Fp>::construct();

    let prover = MockProver::run(k, &circuit, vec![]).unwrap();
    assert_eq!(prover.verify(), Ok(()));
    // Arrange the public input. We expose the multiplication result in row 0
    // of the instance column, so we position it there in our public inputs.
    // let mut public_inputs = vec![c];

    // Given the correct public input, our circuit will verify.
    // let prover = MockProver::run(k, &circuit, vec![public_inputs.clone()]).unwrap();
    // assert_eq!(prover.verify(), Ok(()));
    test_prover(k, circuit);
}
