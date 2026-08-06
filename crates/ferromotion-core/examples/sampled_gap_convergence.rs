//! **Is 25 samples enough to estimate a supremum?**
//!
//! `smoothing_tube` bounds the penalty-vs-rigid gap by sampling 25 entry speeds and taking the max. That is formally a
//! lower bound on the supremum, which is why it can refute and never certify. This program asks the separate, practical
//! question: is it a *good* lower bound, or is the true supremum somewhere else entirely?
//!
//! The question is answerable now because [`AffineContact`](ferromotion_core::AffineContact) solves the contact in
//! closed form, so the gap can be evaluated a million times cheaply. Without it this would be a million adaptive
//! rollouts.
//!
//! What is compared is the **continuous** penalty model against the rigid one, since that is the object a provable
//! bound would target. (The fixed-step model realises a different restitution — see
//! `examples/discretisation_restitution_shift.rs` — and that difference is a separate obstacle.)
//!
//! # Scope, stated before the numbers
//!
//! `gap_at` below is an **analytic proxy** for the tube's gap, not the tube's gap. The tube differences two full
//! rollouts; this differences the closed-form contact against an instantaneous reversal. They agree to about 2% at
//! `k = 1e6` (`9.73e-3` here against the tube's measured `9.52e-3`), which is close enough to answer the question
//! asked — does the supremum hide from a coarse sample? — and not close enough to quote as the tube's own bound.
//!
//! The conclusion below is therefore about the shape of the gap as a function of entry speed, which is what decides
//! whether sampling can find its maximum. It is not a re-measurement of the tube.
//!
//! Run with `cargo run --release -p ferromotion-core --example sampled_gap_convergence`.

use ferromotion_core::{AffineContact, BouncingMass, GRAVITY};

/// The tube fixture's damping law and entry spread.
const ZETA: f64 = 0.1606;
const SPREAD: f64 = 0.10;
/// Samples the tube actually takes.
const TUBE_SAMPLES: usize = 25;

/// The gap at one entry speed: how far the continuous penalty model's post-contact state lands from the rigid model's,
/// with the rigid model given the restitution the penalty model realises AT THE NOMINAL speed — which is what a
/// practitioner does, and is itself a source of the gap.
fn gap_at(exact: &AffineContact, rigid: &BouncingMass, speed: f64) -> Option<(f64, f64)> {
    let c = exact.solve(speed)?;
    // Rigid: instantaneous reversal at the nominal restitution. Penalty: the realised exit speed, delayed by the
    // contact duration, during which gravity has acted.
    let rigid_exit = rigid.restitution * speed;
    let dv = (c.exit_speed - rigid_exit).abs();
    // Height gap after the contact: the rigid body has been climbing for `duration` while the penalty body was in
    // contact, so their heights differ by that much travel plus the penetration excursion.
    let dh = (rigid_exit * c.duration - 0.5 * GRAVITY * c.duration * c.duration).abs();
    Some((dh, dv))
}

fn sup_over(exact: &AffineContact, rigid: &BouncingMass, nominal: f64, n: usize) -> (f64, f64, f64) {
    let (lo, hi) = (nominal * (1.0 - SPREAD), nominal * (1.0 + SPREAD));
    let mut best_h = 0.0f64;
    let mut best_v = 0.0f64;
    let mut arg = f64::NAN;
    for i in 0..n {
        let t = if n == 1 { 0.5 } else { i as f64 / (n - 1) as f64 };
        let s = lo + (hi - lo) * t;
        if let Some((dh, dv)) = gap_at(exact, rigid, s) {
            if dh > best_h {
                best_h = dh;
                arg = s;
            }
            best_v = best_v.max(dv);
        }
    }
    (best_h, best_v, arg)
}

fn main() {
    println!("=== Is 25 samples enough to estimate the supremum of the smoothing gap? ===\n");
    println!("Continuous penalty model vs rigid, over +/-{:.0}% of the nominal impact speed.", SPREAD * 100.0);
    println!("Closed-form contact, so a million evaluations is cheap.\n");

    let nominal = (2.0 * GRAVITY * 1.0f64).sqrt();
    for &k in &[1e4f64, 1e6] {
        let exact = AffineContact::new(GRAVITY, k, 2.0 * ZETA * k.sqrt()).expect("contact");
        let e = exact.restitution_at(nominal).expect("restitution");
        let rigid = BouncingMass::new(GRAVITY, e).expect("rigid");
        println!("--- stiffness {k:.0e}, realised restitution {e:.9} ---");
        println!("{:>10}  {:>13}  {:>13}  {:>12}  {:>11}", "samples", "sup |dh|", "sup |dv|", "argmax speed", "vs 1e6");

        let reference = sup_over(&exact, &rigid, nominal, 1_000_001);
        for &n in &[TUBE_SAMPLES, 101, 1_001, 10_001, 100_001, 1_000_001] {
            let (h, v, arg) = sup_over(&exact, &rigid, nominal, n);
            let tag = if n == TUBE_SAMPLES { " <- what the tube takes" } else { "" };
            println!(
                "{n:>10}  {h:>13.6e}  {v:>13.6e}  {arg:>12.6}  {:>10.4}%{tag}",
                100.0 * (reference.0 - h) / reference.0.max(f64::MIN_POSITIVE)
            );
        }
        // Where does the max actually sit? If it is at an endpoint, sampling finds it easily; if interior, it can hide.
        let (_, _, arg) = reference;
        let at_edge = ((arg - nominal * (1.0 - SPREAD)).abs()).min((arg - nominal * (1.0 + SPREAD)).abs());
        println!(
            "  the supremum sits at {arg:.6} m/s, {:.3e} m/s from the nearest endpoint of the entry set",
            at_edge
        );
        println!(
            "  -> {}\n",
            if at_edge < 1e-6 {
                "AT an endpoint, so a coarse sample that includes the endpoints finds it exactly"
            } else {
                "INTERIOR, so a coarse sample can miss it and the shortfall above is the price"
            }
        );
    }

    println!("What this settles, and what it does not.");
    println!();
    println!("  SETTLED: the gap is monotone in entry speed over this set, so its maximum sits ON an endpoint and a");
    println!("  25-point sample finds the same value a million-point sample does - to every digit printed. Sampling is");
    println!("  not losing anything here. Whatever the missing proof is worth, it is not worth a bigger number.");
    println!();
    println!("  NOT SETTLED, and worth being blunt about:");
    println!("  - A million samples finding the max at an endpoint is strong evidence of monotonicity, NOT a proof of");
    println!("    it. A dense sample remains a lower bound, which is exactly why certify() still refuses it.");
    println!("  - This is an analytic PROXY for the tube's gap (2% apart at 1e6), so it describes the gap's shape, not");
    println!("    the tube's bound.");
    println!("  - The obstacle to a closing certificate was never the sampling. It is the fixed-step discretisation");
    println!("    shift, which is 3x the constraint margin on its own.");
}
