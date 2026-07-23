//! LBM throughput: MLUPs (million lattice-site updates per second), GPU vs the CPU reference.
//! Run: `cargo run -p ferromotion-fluid --features gpu --release --example lbm_mlups`
//!
//! The number reported is honest wall-clock over full submitted work including queue sync —
//! not kernel-time-only.

use ferromotion_fluid::lbm::{LbmBc, LbmD2Q9};
use ferromotion_fluid::lbm_gpu::LbmGpu;
use std::time::Instant;

fn main() {
    // CPU reference baseline (single thread), modest grid.
    let n = 256;
    let mut c = LbmD2Q9::new(n, n, 0.8, LbmBc::Cavity { lid_u: 0.1 });
    let steps = 200;
    let t = Instant::now();
    for _ in 0..steps {
        c.step();
    }
    let cpu_mlups = (n * n * steps) as f64 / t.elapsed().as_secs_f64() / 1e6;
    println!("CPU  {n:>5}x{n:<5} {cpu_mlups:>10.1} MLUPs (single thread)");

    for n in [256usize, 512, 1024, 2048] {
        let Some(g) = LbmGpu::new(n, n, 0.8, LbmBc::Cavity { lid_u: 0.1 }) else {
            println!("no GPU available");
            return;
        };
        // Warmup (pipeline compile, first submits), then timed.
        g.run(64);
        let _ = g.velocities(); // sync
        let steps = if n <= 512 { 2000 } else { 500 };
        let t = Instant::now();
        g.run(steps);
        let _ = g.velocities(); // sync — include full completion in the clock
        let mlups = (n * n * steps) as f64 / t.elapsed().as_secs_f64() / 1e6;
        println!("GPU  {n:>5}x{n:<5} {mlups:>10.1} MLUPs ({:.0}x CPU)", mlups / cpu_mlups);
    }
}
