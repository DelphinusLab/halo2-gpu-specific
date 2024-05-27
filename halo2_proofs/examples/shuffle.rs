use group::ff::BatchInvert;
use halo2_proofs::arithmetic::FieldExt;
use halo2_proofs::circuit::{Layouter, SimpleFloorPlanner};
use halo2_proofs::dev::MockProver;
use halo2_proofs::plonk::*;
use pairing::bn256::Fr as Fp;
use pairing::bn256::{Bn256, G1Affine};

use halo2_proofs::poly::{
    commitment::{Params, ParamsVerifier},
    Rotation,
};
use halo2_proofs::transcript::{Blake2bRead, Blake2bWrite, Challenge255};
use rand_core::{OsRng, RngCore};

fn rand_2d_array<F: FieldExt, R: RngCore, const W: usize, const H: usize>(
    rng: &mut R,
) -> [[F; H]; W] {
    [(); W].map(|_| [(); H].map(|_| F::random(&mut *rng)))
}

fn shuffled<F: FieldExt, R: RngCore, const W: usize, const H: usize>(
    original: [[F; H]; W],
    rng: &mut R,
) -> [[F; H]; W] {
    let mut shuffled = original;

    for row in (1..H).rev() {
        let rand_row = (rng.next_u32() as usize) % row;
        for column in shuffled.iter_mut() {
            column.swap(row, rand_row);
        }
    }

    shuffled
}

#[derive(Clone)]
struct MyConfig<const W: usize, const T: usize, const B: usize> {
    q_shuffle: Column<Fixed>,
    q_first: Column<Fixed>,
    q_last: Column<Fixed>,
    original: [Column<Advice>; W],
    shuffled: [Column<Advice>; W],
    z: Column<Advice>,
}

impl<const W: usize, const T: usize, const B: usize> MyConfig<W, T, B> {
    fn configure<F: FieldExt>(meta: &mut ConstraintSystem<F>) -> Self {
        let [q_shuffle, q_first, q_last] = [(); 3].map(|_| meta.fixed_column());
        let original = [(); W].map(|_| meta.advice_column());
        let shuffled = [(); W].map(|_| meta.advice_column());
        let theta = Expression::Constant(F::from(T as u64));
        let beta = Expression::Constant(F::from(B as u64));
        let z = meta.advice_column();

        meta.create_gate("z should start with 1", |meta| {
            let q_first = meta.query_fixed(q_first, Rotation::cur());
            let z = meta.query_advice(z, Rotation::cur());
            let one = Expression::Constant(F::one());

            vec![q_first * (one - z)]
        });

        meta.create_gate("z should end with 1", |meta| {
            let one = Expression::Constant(F::one());
            let q_last = meta.query_fixed(q_last, Rotation::cur());
            let z = meta.query_advice(z, Rotation::cur());
            vec![q_last * (one - z)]
        });

        meta.create_gate("z should have valid transition", |meta| {
            let q_shuffle = meta.query_fixed(q_shuffle, Rotation::cur());
            let original = original.map(|advice| meta.query_advice(advice, Rotation::cur()));
            let shuffled = shuffled.map(|advice| meta.query_advice(advice, Rotation::cur()));
            let z_cur = meta.query_advice(z, Rotation::cur());
            let z_next = meta.query_advice(z, Rotation::next());

            // Compress
            let original = original
                .iter()
                .cloned()
                .reduce(|acc, a| acc * theta.clone() + a)
                .unwrap();
            let shuffled = shuffled
                .iter()
                .cloned()
                .reduce(|acc, a| acc * theta.clone() + a)
                .unwrap();

            vec![q_shuffle * (z_cur * (original + beta.clone()) - z_next * (shuffled + beta))]
        });

        Self {
            q_shuffle,
            q_first,
            q_last,
            original,
            shuffled,
            z,
        }
    }
}

#[derive(Clone)]
struct MyCircuit<F: FieldExt, const W: usize, const H: usize, const T: usize, const B: usize> {
    original: [[F; H]; W],
    shuffled: [[F; H]; W],
}

impl<F: FieldExt, const W: usize, const H: usize, const T: usize, const B: usize> Default
    for MyCircuit<F, W, H, T, B>
{
    fn default() -> Self {
        Self {
            original: [[F::zero(); H]; W],
            shuffled: [[F::zero(); H]; W],
        }
    }
}

impl<F: FieldExt, const W: usize, const H: usize, const T: usize, const B: usize>
    MyCircuit<F, W, H, T, B>
{
    fn rand<R: RngCore>(rng: &mut R) -> Self {
        let original = rand_2d_array::<F, _, W, H>(rng);
        let shuffled = shuffled(original, rng);

        Self {
            original: original,
            shuffled: shuffled,
        }
    }
}

impl<F: FieldExt, const W: usize, const H: usize, const T: usize, const B: usize> Circuit<F>
    for MyCircuit<F, W, H, T, B>
{
    type Config = MyConfig<W, T, B>;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self::default()
    }

    fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
        MyConfig::configure(meta)
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<F>,
    ) -> Result<(), Error> {
        let theta = F::from(T as u64);
        let beta = F::from(B as u64);

        layouter.assign_region(
            || "Shuffle original into shuffled",
            |mut region| {
                region.assign_fixed(|| "", config.q_first, 0, || Ok(F::one()))?;
                region.assign_fixed(|| "", config.q_last, H, || Ok(F::one()))?;
                for offset in 0..H {
                    region.assign_fixed(|| "", config.q_shuffle, offset, || Ok(F::one()))?;
                }

                for (idx, (&column, values)) in
                    config.original.iter().zip(self.original.iter()).enumerate()
                {
                    for (offset, &value) in values.iter().enumerate() {
                        region.assign_advice(
                            || format!("original[{}][{}]", idx, offset),
                            column,
                            offset,
                            || Ok(value),
                        )?;
                    }
                }
                for (idx, (&column, values)) in
                    config.shuffled.iter().zip(self.shuffled.iter()).enumerate()
                {
                    for (offset, &value) in values.iter().enumerate() {
                        region.assign_advice(
                            || format!("shuffled[{}][{}]", idx, offset),
                            column,
                            offset,
                            || Ok(value),
                        )?;
                    }
                }

                let z = Some(self.original)
                    .zip(Some(self.shuffled))
                    .map(|(original, shuffled)| {
                        let mut product = vec![F::zero(); H];
                        for (idx, product) in product.iter_mut().enumerate() {
                            let mut compressed = F::zero();
                            for value in shuffled.iter() {
                                compressed *= theta;
                                compressed += value[idx];
                            }

                            *product = compressed + beta
                        }

                        product.iter_mut().batch_invert();

                        for (idx, product) in product.iter_mut().enumerate() {
                            let mut compressed = F::zero();
                            for value in original.iter() {
                                compressed *= theta;
                                compressed += value[idx];
                            }

                            *product *= compressed + beta
                        }

                        #[allow(clippy::let_and_return)]
                        let z = std::iter::once(F::one())
                            .chain(product)
                            .scan(F::one(), |state, cur| {
                                *state *= &cur;
                                Some(*state)
                            })
                            .collect::<Vec<_>>();

                        #[cfg(feature = "sanity-checks")]
                        assert_eq!(F::one(), *z.last().unwrap());

                        z
                    })
                    .unwrap();

                for (offset, value) in z.iter().enumerate() {
                    region.assign_advice(
                        || format!("z[{}]", offset),
                        config.z,
                        offset,
                        || Ok(*value),
                    )?;
                }

                Ok(())
            },
        )
    }
}

fn test_prover<const W: usize, const H: usize, const T: usize, const B: usize>(
    k: u32,
    circuit: MyCircuit<Fp, W, H, T, B>,
) {
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
    const W: usize = 4;
    const H: usize = 32;

    const THETA: usize = 111;
    const BETA: usize = 222;

    // The number of rows in our circuit cannot exceed 2^k
    let k = 10;

    let circuit = MyCircuit::<Fp, W, H, THETA, BETA>::rand(&mut OsRng);

    let prover = MockProver::run(k, &circuit, vec![]).unwrap();
    assert_eq!(prover.verify(), Ok(()));

    test_prover::<W, H, THETA, BETA>(k, circuit);
}
