//! **A gradient that is exactly zero, and what it costs to invent one.**
//!
//! `adaptive_gradient_decomposition` dealt with a contact gradient that is *wrong*. This one deals with a contact
//! gradient that is *absent*. A finger not touching an object exerts no force on it, the object does not move, and the
//! derivative of anything about the object with respect to anything about the finger is identically zero. Gradient
//! descent from that configuration cannot take a first step, and no integrator, tolerance or timestep changes that.
//!
//! The published remedy is to apply an artificial contact force at positive signed distance, in the **backward pass
//! only**, so the trajectory stays exactly physical while the gradient can feel the object from a distance. This
//! program measures three things about that:
//!
//! 1. that the true gradient really is zero, confirmed by an independent oracle,
//! 2. that descent goes from impossible to solved,
//! 3. what the pre-contact gradient costs in fidelity, since it is a search heuristic and not a derivative of anything.
//!
//! Run with `cargo run --release -p ferromotion-core --example contacts_from_distance`.

use ferromotion_core::{descend, CfdProfile, PushTask};

fn main() {
    let task = PushTask::out_of_reach();
    let threshold = task.contact_threshold();
    println!("=== A gradient that is exactly zero ===\n");
    println!("finger at {:.3} m, object at {:.3} m, target {:.3} m", task.finger_start, task.object_start, task.target);
    println!("contact needs u >= {threshold:.5} m/s within the horizon\n");

    part_one_the_gradient_is_zero(&task, threshold);
    part_two_the_oracle_cannot_be_trusted_small(&task, threshold);
    part_three_descent(&task, threshold);
    part_four_what_the_heuristic_costs(&task, threshold);
}

/// The failure, established with two independent methods so it is not an artefact of either.
fn part_one_the_gradient_is_zero(task: &PushTask, threshold: f64) {
    println!("--- 1. Out of reach, the derivative is zero and not merely small -----------------");
    println!("{:>10}  {:>9}  {:>14}  {:>14}  {:>9}", "u", "touches", "analytic", "fd (h=1e-3)", "objective");
    for frac in [0.25, 0.5, 0.9, 1.1, 2.0] {
        let u = frac * threshold;
        let (_, touched) = task.rollout(u);
        println!(
            "{u:>10.5}  {:>9}  {:>14.6e}  {:>14.6e}  {:>9.3e}",
            touched,
            task.gradient(u, None),
            task.gradient_finite_difference(u, 1e-3),
            task.objective(u)
        );
    }
    println!("\n  Zero is not a small number here. There is no direction to descend in.\n");
}

/// A finite difference is the obvious oracle and on this objective it is only usable on a plateau. Worth its own
/// section because the natural instinct is to shrink the probe, and shrinking it makes the answer worse.
fn part_two_the_oracle_cannot_be_trusted_small(task: &PushTask, threshold: f64) {
    println!("--- 2. The oracle is only usable on a plateau -----------------------------------");
    println!("At u = 2x the threshold the objective depends on u partly through the INTEGER number of");
    println!("timesteps spent in contact. A small probe divides a tiny jump by a tiny h and reports 1/h.\n");
    let u = 2.0 * threshold;
    println!("{:>8}  {:>14}  {:>12}", "h", "fd", "vs previous");
    let mut prev = f64::NAN;
    for e in 1..9 {
        let h = 10f64.powi(-e);
        let fd = task.gradient_finite_difference(u, h);
        let change = if prev.is_finite() { format!("{:.3}", (fd - prev).abs() / fd.abs().max(1e-30)) } else { "-".into() };
        println!("{h:>8.0e}  {fd:>14.6e}  {change:>12}");
        prev = fd;
    }
    println!("\n  Usable plateau: h in [1e-2, 1e-4]. Below 1e-5 the number is the probe, not the derivative.");
    println!("  The analytic gradient instead converges under TIMESTEP refinement, which is its own evidence:");
    println!("  {:>10}  {:>8}  {:>14}", "dt", "steps", "analytic");
    for (dt, steps) in [(1e-4, 4_000usize), (2e-5, 20_000), (5e-6, 80_000)] {
        let t = PushTask { dt, steps, ..*task };
        println!("  {dt:>10.0e}  {steps:>8}  {:>14.6e}", t.gradient(2.0 * t.contact_threshold(), None));
    }
    println!();
}

/// The result that decides whether the method is worth having.
fn part_three_descent(task: &PushTask, threshold: f64) {
    println!("--- 3. Descent, from a configuration where the gradient is dead -----------------");
    println!("Backtracking line search that only accepts steps reducing the TRUE objective, so an");
    println!("artificial force may propose a direction and may not grade the outcome.\n");
    let u0 = 0.5 * threshold;
    println!("start: u = {u0:.5}, objective {:.6e}, touching {}\n", task.objective(u0), task.rollout(u0).1);
    println!("{:>22}  {:>10}  {:>13}  {:>8}  {:>7}", "gradient", "final u", "objective", "touched", "moved");

    let runs = [
        ("true (no profile)", None),
        ("cfd reach 0.02 m", Some(CfdProfile::reaching(0.02))),
        ("cfd reach 0.06 m", Some(CfdProfile::reaching(0.06))),
        ("cfd reach 0.12 m", Some(CfdProfile::reaching(0.12))),
    ];
    for (name, profile) in runs {
        let r = descend(task, u0, profile, 60, 1e3);
        println!(
            "{name:>22}  {:>10.5}  {:>13.6e}  {:>8}  {:>7}",
            r.u, r.objective, r.touched, r.moved
        );
    }
    // The 0.02 m profile does not move either, and that is a real constraint rather than a tuning detail: at this
    // starting parameter the finger's closest approach still leaves a 0.04 m gap, so a 0.02 m reach never sees the
    // object. A pre-contact gradient is not free - you have to know roughly how far away the thing is.
    let closest = task.object_start - task.contact_distance - (task.finger_start + u0 * task.dt * task.steps as f64);
    println!("\n  The true gradient cannot move. Neither can a reach shorter than the closest approach, which here");
    println!("  is {closest:.4} m: the 0.02 m profile is blind and behaves exactly like no profile. Reach has to");
    println!("  exceed the gap you are trying to close, so this buys a first step in exchange for a prior.\n");
}

/// The honest accounting: a pre-contact gradient is not the derivative of anything the robot experiences, so the
/// question is how far it points wrong once contact is real.
fn part_four_what_the_heuristic_costs(task: &PushTask, threshold: f64) {
    println!("--- 4. What the heuristic costs once contact is real ----------------------------");
    println!("In contact the artificial force is behind the finger, not between it and the object, so the two");
    println!("gradients should agree closely. The gap is the price of leaving the profile switched on.\n");
    println!("{:>10}  {:>14}  {:>14}  {:>9}  note", "u", "true", "cfd (0.06 m)", "ratio");
    let profile = CfdProfile::reaching(0.06);
    for frac in [1.1, 1.5, 2.0, 3.0] {
        let u = frac * threshold;
        let (t, c) = (task.gradient(u, None), task.gradient(u, Some(profile)));
        // A ratio against a near-zero denominator is not a fidelity statement, so it is labelled rather than quoted.
        let note = if t.abs() < 0.1 * 8.1e-2 { "ratio meaningless: true gradient near zero" } else { "" };
        println!("{u:>10.5}  {t:>14.6e}  {c:>14.6e}  {:>8.4}x  {note}", c / t);
    }
    println!("\n  Close enough to descend on, and not identical: the profile is a search aid that should be");
    println!("  switched off once contact is established, which is a decision the caller has to make.");
}
