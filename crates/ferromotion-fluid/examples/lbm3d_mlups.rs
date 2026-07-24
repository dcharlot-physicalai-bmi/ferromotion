//! D3Q19 GPU throughput: MLUPs (million lattice-site updates/s).
//! Run: `cargo run -p ferromotion-fluid --features gpu --release --example lbm3d_mlups`
use ferromotion_fluid::lbm3d_gpu::LbmGpu3;
use std::time::Instant;

fn main() {
    for n in [32usize, 48, 64, 96] {
        let Some(g) = LbmGpu3::new(n, n, n, 0.8) else {
            println!("no GPU available");
            return;
        };
        g.run(32);
        let _ = g.velocities(); // warmup + sync
        let steps = if n <= 48 { 400 } else { 150 };
        let t = Instant::now();
        g.run(steps);
        let _ = g.velocities(); // include full completion
        let mlups = (n * n * n * steps) as f64 / t.elapsed().as_secs_f64() / 1e6;
        println!("GPU D3Q19 {n:>3}^3  {mlups:>10.1} MLUPs");
    }
}
