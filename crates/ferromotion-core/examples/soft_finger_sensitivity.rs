//! **Does Q1 respond to mu_torsion at all, and over what range?**
//!
//! The companion example measured the mixed-convention overstatement (a factor `sqrt(1+mu^2)`, 1.80x at mu = 1.5) and
//! then reported that correcting it changes Q1 by nothing at all, to six decimals, in twelve cases. A null result that
//! clean is usually a dead probe, not a fact, so this measures the sensitivity directly before the null is believed:
//! sweep `mu_torsion` finely and print the response. If Q1 is flat in `mu_torsion` over the corrected interval, the
//! null is real and the mixed convention is latent. If it is not flat, the first probe was not varying what it thought.
//!
//!   cargo run -p ferromotion-core --example soft_finger_sensitivity --release

use ferromotion_core::{force_closure_q1_spatial, GraspContact3};
use nalgebra::Vector3;

fn tetra_grasp(mu: f64, mu_torsion: f64) -> Vec<GraspContact3> {
    let dirs = [
        Vector3::new(1.0, 0.0, -0.35),
        Vector3::new(-0.5, 0.866, -0.35),
        Vector3::new(-0.5, -0.866, -0.35),
        Vector3::new(0.0, 0.0, 1.0),
    ];
    dirs
        .iter()
        .map(|d| {
            let p = d.normalize();
            GraspContact3 { pos: p, normal: -p, mu, mu_torsion }
        })
        .collect()
}

fn main() {
    let dirs = 20_000;

    println!("== A. Q1 against mu_torsion, finely, at mu = 0.3 (where the coarse sweep DID move) ==");
    println!("   The correction multiplies mu_torsion by 1/sqrt(1+mu^2) = {:.6}", 1.0 / 1.09f64.sqrt());
    println!("   mu_torsion   Q1              d(Q1) from previous");
    let mut prev = f64::NAN;
    for mt in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.2873, 0.29, 0.30, 0.35, 0.40, 0.60, 1.00] {
        let q = force_closure_q1_spatial(&tetra_grasp(0.3, mt), 24, dirs);
        let d = if prev.is_nan() { 0.0 } else { q - prev };
        println!("   {mt:8.4}     {q:.9}     {d:+.3e}");
        prev = q;
    }

    println!();
    println!("== B. The two values the correction compares, printed to full precision ==");
    for mu in [0.3f64, 0.5, 1.0, 1.5] {
        let scale = 1.0 / (1.0 + mu * mu).sqrt();
        for mt in [0.1f64, 0.3, 1.0] {
            let a = force_closure_q1_spatial(&tetra_grasp(mu, mt), 24, dirs);
            let b = force_closure_q1_spatial(&tetra_grasp(mu, mt * scale), 24, dirs);
            println!(
                "   mu={mu:3.1} mt={mt:4.2} -> {:.4}   Q1(as shipped)={a:.12}  Q1(corrected)={b:.12}  diff={:+.3e}",
                mt * scale,
                a - b
            );
        }
    }

    println!();
    println!("== C. Where does mu_torsion START to bind? Sweep far past any plausible value ==");
    println!("   mu     mu_torsion at which Q1 first exceeds its mu_torsion=0 value");
    for mu in [0.3f64, 0.5, 1.0, 1.5] {
        let base = force_closure_q1_spatial(&tetra_grasp(mu, 0.0), 24, dirs);
        let mut first = None;
        let mut mt = 0.0f64;
        while mt <= 4.0 {
            let q = force_closure_q1_spatial(&tetra_grasp(mu, mt), 24, dirs);
            if q > base * (1.0 + 1e-9) {
                first = Some((mt, q));
                break;
            }
            mt += 0.02;
        }
        match first {
            Some((m, q)) => println!("   {mu:3.1}    binds from mu_torsion ~ {m:.2}  (Q1 {base:.6} -> {q:.6})"),
            None => println!("   {mu:3.1}    never binds up to mu_torsion = 4.0 (Q1 stays {base:.6})"),
        }
    }

    println!();
    println!("== D. Sanity: the probe IS varying the contacts it builds ==");
    for mt in [0.0f64, 0.3] {
        let cs = tetra_grasp(0.3, mt);
        let sum: f64 = cs.iter().map(|c| c.mu_torsion).sum();
        println!("   requested mu_torsion {mt:4.2} -> sum over {} contacts = {sum:.4}", cs.len());
    }
}
