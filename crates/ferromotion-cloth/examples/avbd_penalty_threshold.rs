//! **How large does the AVBD penalty have to be, relative to the stiffness it stands in for?**
//!
//! `AugmentedVbd::step` builds each vertex block with `penalty` in place of the spring's true stiffness, and carries the
//! difference in a multiplier clamped to `+/- stiffness * rest`. So the penalty is not a free knob: it is the stiffness
//! the Newton step actually sees. Too small and the block is far too soft for the constraint it is meant to enforce.
//!
//! This was found by writing a test that only asserted the result was FINITE. It was: `23192.7883 m` of finite, on a
//! chain whose rest length is `1.0 m`. Finite is not the same as physical, which is the same lesson the module's own
//! "stable is not accurate" note makes about plain VBD.
//!
//!   cargo run -p ferromotion-cloth --example avbd_penalty_threshold --release

use ferromotion_cloth::vbd::{AugmentedVbd, Spring, VbdSolver, Vertex};
use nalgebra::Vector3;

/// A chain pinned at the origin, laid out along +x, all springs at the same stiffness.
fn chain(n: usize, link: f64, mass: f64, stiffness: f64) -> (Vec<Vertex>, Vec<Spring>) {
    let mut verts = Vec::with_capacity(n + 1);
    verts.push(Vertex::pinned_at(Vector3::zeros()));
    for i in 1..=n {
        verts.push(Vertex::new(Vector3::new(i as f64 * link, 0.0, 0.0), mass));
    }
    let springs =
        (0..n).map(|i| Spring { i, j: i + 1, rest: link, stiffness }).collect();
    (verts, springs)
}

fn main() {
    let (n, link, mass) = (10usize, 0.1f64, 0.05f64);
    let rest_total = n as f64 * link;
    let solver = VbdSolver::new(1.0 / 60.0, 4).unwrap();

    println!("10-link chain, link {link} m (rest total {rest_total} m), mass {mass} kg, 4 sweeps, 200 steps");
    println!("A 'usable' result is a max extent within 3x the rest total; anything beyond that has diverged.");
    println!();
    println!("  stiffness   penalty   penalty/stiffness   max extent (m)   at clamp   verdict");

    for stiffness in [1e2f64, 1e4, 1e7] {
        for penalty in [1e0f64, 1e2, 1e4, 1e6, 1e8, 1e10] {
            let (verts0, springs) = chain(n, link, mass, stiffness);
            let mut avbd = AugmentedVbd::new(solver, springs.len(), penalty).unwrap();
            let mut verts = verts0.clone();
            for _ in 0..200 {
                avbd.step(&mut verts, &springs);
            }
            let ext = verts
                .iter()
                .map(|v| v.position.norm())
                .fold(0.0f64, |a, b| if a.is_nan() || b.is_nan() { f64::NAN } else { a.max(b) });
            let at_clamp = springs
                .iter()
                .enumerate()
                .filter(|(si, s)| avbd.multipliers[*si].abs() >= s.stiffness * s.rest * (1.0 - 1e-9))
                .count();
            let verdict = if !ext.is_finite() {
                "NaN"
            } else if ext > 3.0 * rest_total {
                "DIVERGED"
            } else {
                "usable"
            };
            println!(
                "  {stiffness:>9.0e}   {penalty:>7.0e}   {:>17.0e}   {ext:>14.4}   {at_clamp:>3} of {}   {verdict}",
                penalty / stiffness,
                springs.len()
            );
        }
        println!();
    }

    println!("Bisect the threshold at stiffness 1e7, so the rule can be stated as a ratio rather than a number:");
    for stiffness in [1e2f64, 1e4, 1e7] {
        let (mut lo, mut hi) = (1e0f64, 1e12f64); // lo diverges, hi is usable
        for _ in 0..48 {
            let mid = (lo.ln() + hi.ln()).exp().sqrt(); // geometric midpoint
            let (verts0, springs) = chain(n, link, mass, stiffness);
            let mut avbd = AugmentedVbd::new(solver, springs.len(), mid).unwrap();
            let mut verts = verts0.clone();
            for _ in 0..200 {
                avbd.step(&mut verts, &springs);
            }
            let ext = verts
                .iter()
                .map(|v| v.position.norm())
                .fold(0.0f64, |a, b| if a.is_nan() || b.is_nan() { f64::NAN } else { a.max(b) });
            if ext.is_finite() && ext <= 3.0 * rest_total {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        println!("  stiffness {stiffness:>8.0e}: usable from penalty ~{hi:.3e}  (ratio penalty/stiffness ~{:.2})", hi / stiffness);
    }
}
