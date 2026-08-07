//! **The planar/spatial ratio against direction count** — because the lesson quotes it at three sampling densities.
//!
//! The lesson text carries "0.92x at 64 directions against 0.58x at 20000" and a narration line saying the ratio reads
//! nine-tenths at three contacts and eleven-tenths at four. Those are 64-direction values. This reproduces the lab's
//! exact contact set and prints both metrics and their ratio at each density, so the copy can quote measured numbers.
//!
//!   cargo run -p ferromotion-core --example grasp_ratio_convergence --release

use ferromotion_core::{force_closure_q1_planar_subspace, force_closure_q1_spatial, GraspContact3};
use nalgebra::Vector3;

const FACETS: usize = 8; // matches grasp_reality_lab::FACETS
const RADIUS: f64 = 0.05; // matches grasp_reality_lab::RADIUS

/// The lab's `contact_set()` at `spread = 0` (coplanar), transcribed so this measures the same geometry.
fn contact_set(n: usize, mu: f64, spread: f64) -> Vec<GraspContact3> {
    (0..n)
        .map(|i| {
            let az = 2.0 * std::f64::consts::PI * (i as f64) / (n as f64);
            let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
            let el = sign * spread * 0.9;
            let (ce, se) = (el.cos(), el.sin());
            let pos = Vector3::new(RADIUS * ce * az.cos(), RADIUS * ce * az.sin(), RADIUS * se);
            GraspContact3::hard(pos, -pos.normalize(), mu)
        })
        .collect()
}

fn main() {
    let mu = 0.5; // the lab's default
    println!("coplanar (spread 0), mu = {mu}, 8 facets, radius {RADIUS} m");
    println!("  dirs    contacts   Q1 spatial   Q1 planar    planar/spatial");
    for dirs in [64usize, 1024, 4096, 20_000] {
        for n in [3usize, 4, 5] {
            let cs = contact_set(n, mu, 0.0);
            let s = force_closure_q1_spatial(&cs, FACETS, dirs);
            let p = force_closure_q1_planar_subspace(&cs, FACETS, dirs);
            let r = if s > 0.0 { p / s } else { f64::NAN };
            println!("  {dirs:>5}      {n}        {s:.6}     {p:.6}     {r:.4}x");
        }
    }

    println!();
    println!("Does the ratio cross one at any of these densities?");
    for dirs in [64usize, 1024, 4096, 20_000] {
        let mut lo = f64::INFINITY;
        let mut hi: f64 = 0.0;
        for n in 3..=7 {
            let cs = contact_set(n, mu, 0.0);
            let s = force_closure_q1_spatial(&cs, FACETS, dirs);
            let p = force_closure_q1_planar_subspace(&cs, FACETS, dirs);
            if s > 0.0 {
                let r = p / s;
                lo = lo.min(r);
                hi = hi.max(r);
            }
        }
        println!("  {dirs:>5} dirs, contacts 3..7: ratio in [{lo:.4}, {hi:.4}]  crosses 1.0: {}", lo < 1.0 && hi > 1.0);
    }
}
