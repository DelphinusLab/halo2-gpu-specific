use ec_gpu_gen::SourceBuilder;
use pairing::bn256::{Fr, Fq, G1Affine};

fn main() {
    let source_builder = SourceBuilder::new()
        .add_fft::<Fr>()
        .add_multiexp::<G1Affine, Fq>();
        ec_gpu_gen::generate(&source_builder);
}
