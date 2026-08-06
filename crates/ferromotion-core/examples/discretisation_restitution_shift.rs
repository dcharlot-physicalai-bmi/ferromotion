//! **The measurement that decides whether a smoothing-gap certificate can close.**
//!
//! A provable bound on the penalty-vs-rigid gap is achievable, but the tractable routes all bound the *continuous*
//! penalty ODE — which is piecewise affine, so its flow is closed-form. The tube's fixture, and every simulator a
//! practitioner actually runs, uses a *fixed-step* rollout instead.
//!
//! If those two realise the same restitution, the distinction is bookkeeping. If they do not, a certificate built on
//! the continuous bound does not cover the fixed-step trajectory, and the extra term has to be bounded too. This
//! program measures the difference, and compares it against the constraint margin the certificate has to fit inside.
//!
//! Run with `cargo run --release -p ferromotion-core --example discretisation_restitution_shift`.

use ferromotion_core::{AdaptiveOptions, AdaptivePenalty, PenaltyMass, GRAVITY};

/// The tube fixture's damping ratio, so the realised restitution holds still as stiffness sweeps.
const ZETA: f64 = 0.1606;
/// The constraint margin the smoothing-tube certificate has to fit inside, from
/// `ferromotion-control/examples/smoothing_tube_certificate.rs`.
const MARGIN: f64 = 2.6e-2;

fn main() {
    println!("=== Do the fixed-step and continuous penalty models realise the same restitution? ===\n");
    println!("A provable gap bound is available for the CONTINUOUS ODE (piecewise affine, closed-form flow).");
    println!("The fixture samples the FIXED-STEP map. This is the difference between the two objects.\n");
    println!("constraint margin the certificate must fit inside: {MARGIN:.1e}\n");
    println!(
        "{:>10}  {:>8}  {:>11}  {:>11}  {:>11}  {:>12}  {:>10}",
        "stiffness", "w*dt", "e (fixed)", "e (cont.)", "de", "de*v [m/s]", "vs margin"
    );

    let v = (2.0 * GRAVITY * 1.0f64).sqrt();
    for &k in &[1e4f64, 1e5, 1e6, 1e7] {
        let d = 2.0 * ZETA * k.sqrt();
        // The fixture's own timestep rule.
        let dt: f64 = (1e-3f64).min(0.2 / k.sqrt());
        let omega_dt = k.sqrt() * dt;

        let fixed = PenaltyMass::new(GRAVITY, k, d, dt).expect("fixed");
        let cont = AdaptivePenalty::new(GRAVITY, k, d).expect("continuous");
        let e_fix = fixed.effective_restitution(v);
        let e_con = cont.effective_restitution(v, AdaptiveOptions::with_tolerance(1e-11));

        match (e_fix, e_con) {
            (Some(a), Some(b)) => {
                let de = (a - b).abs();
                // In velocity units: the rebound speed differs by de * impact speed.
                let dv = de * v;
                println!(
                    "{k:>10.0e}  {omega_dt:>8.3}  {a:>11.6}  {b:>11.6}  {de:>11.3e}  {dv:>12.3e}  {:>9.2}x",
                    dv / MARGIN
                );
            }
            _ => println!("{k:>10.0e}  {omega_dt:>8.3}  {:>11}  {:>11}  {:>11}  {:>12}  {:>10}", "-", "-", "-", "-", "FAILED"),
        }
    }

    println!("\nRead the last column. Any value above 1.0 means the discretisation shift ALONE exceeds the margin the");
    println!("certificate has to fit inside, so a bound proved for the continuous ODE cannot cover a fixed-step");
    println!("rollout however tight the bound itself is. The route out is not a better bound: it is to run the design");
    println!("rollout under a tolerance, which is the same conclusion the gradient work reached from the other side.");
}
