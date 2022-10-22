use ec_gpu_gen::SourceBuilder;
use pairing::bn256::Fr;

fn main() {
    let source_builder = SourceBuilder::new().add_fft::<Fr>();
    ec_gpu_gen::generate(&source_builder);
}
