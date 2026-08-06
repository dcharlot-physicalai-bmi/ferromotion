//! **Can a policy built on a smoothed contact be certified against the nonsmooth dynamics?**
//!
//! Today's decomposition established that a fixed-step penalty rollout reports a contact Jacobian wrong by more than
//! the magnitude of the Jacobian itself, and often of the wrong sign, while the same contact model under an error
//! tolerance reports the exact one. That is a statement about a number. This program asks what it costs a certificate.
//!
//! One vertical mass, one impact, a ceiling it must not exceed on the way back up. The smoothing gap between the
//! penalty model and the rigid one is measured, propagated as a reachable tube, and checked against the ceiling. Two
//! things vary: the contact stiffness, and **which impact Jacobian the tube is propagated through**. A certificate is
//! computed from a linearisation, so a wrong linearisation does not merely produce a worse design - it produces a
//! wrong certificate.
//!
//! An earlier version of this program also designed a feedback gain by pole placement on each candidate Jacobian. That
//! experiment was mis-specified: it applied a gain tuned on the impact map to the free-flight steps where it does not
//! belong, and produced tube widths of `1e21` from a closed loop whose spectral radius was `0.5`. It was removed
//! rather than tuned into looking sensible.
//!
//! Run with `cargo run --release -p ferromotion-control --example smoothing_tube_certificate`.

use ferromotion_control::{
    certify, nominal_activity, propagate_tube, GapBound, GapEvidence, HalfSpace, TubeStep, TubeVerdict, Zonotope,
};
use ferromotion_core::{AdaptiveOptions, AdaptivePenalty, BouncingMass, PenaltyMass, GRAVITY};
use nalgebra::{DMatrix, DVector};

/// Drop height, so the impact speed is closed form.
const H0: f64 = 1.0;
/// Control step. The impact is resolved inside one control step, which is what a real controller sees.
const DT: f64 = 1e-3;
/// Damping ratio held fixed as stiffness sweeps, so the realised restitution stays put.
const ZETA: f64 = 0.1606;
/// Ceiling the mass must not exceed after rebounding. Chosen just above the nominal apex so the certificate is a
/// real question rather than a formality.
const CEILING: f64 = 0.46;
/// Control steps after the impact over which the constraint is checked. Must be long enough for the mass to reach
/// its apex: at the realised restitution the rise takes about 285 ms, so a 60-step (60 ms) horizon made the ceiling
/// constraint unreachable and every certificate over it vacuously true.
const HORIZON: usize = 340;

fn impact_speed() -> f64 {
    (2.0 * GRAVITY * H0).sqrt()
}

fn damping_for(k: f64) -> f64 {
    2.0 * ZETA * k.sqrt()
}

fn main() {
    println!("=== Does the gradient you designed on decide whether the certificate closes? ===\n");
    println!("one vertical mass, dropped {H0} m, impacting at {:.4} m/s", impact_speed());
    println!("control step {DT} s, ceiling {CEILING} m, horizon {HORIZON} steps after impact\n");

    for &k in &[1e4, 1e6] {
        run_stiffness(k);
    }

    println!("--- What the evidence type refuses to do ---------------------------------------");
    soundness_demonstration();
}

fn run_stiffness(k: f64) {
    let d = damping_for(k);
    let v = impact_speed();
    let penalty = PenaltyMass::new(GRAVITY, k, d, DT.min(0.2 / k.sqrt())).expect("penalty");
    let adaptive = AdaptivePenalty::new(GRAVITY, k, d).expect("adaptive");
    let e = penalty.effective_restitution(v).expect("restitution");
    let rigid = BouncingMass::new(GRAVITY, e.clamp(0.0, 1.0)).expect("rigid");

    // Start strictly in free flight just above the plane, not exactly on it. The one-sided damper makes the contact
    // force jump discontinuously at h = 0, so a state sitting exactly on the kink drives the adaptive controller
    // below its minimum step and it correctly refuses. Entering from above is also what a real trajectory does.
    let h_start = 1e-4;
    let v_start = -(v * v - 2.0 * GRAVITY * h_start).sqrt();
    let x_pre = [h_start, v_start];
    // Free flight down to the plane, plus enough time to complete the contact and leave it.
    let window = (v - v_start.abs()) / GRAVITY + 8.0 * std::f64::consts::PI / k.sqrt();

    // All three routes map the SAME start state over the SAME window, so the only difference is how the derivative
    // of that map is obtained. Using the bare saltation matrix for the exact route would not be comparable: it has
    // dh/dv = 0 identically, which is a property of the impact instant, not of this map.
    let exact = to_matrix(rigid.jacobian_saltation(x_pre, window).expect("saltation composition"));
    let fixed = to_matrix(penalty.jacobian_autodiff(x_pre, window));
    let tol = to_matrix(
        adaptive
            .jacobian_richardson(x_pre, window, 1e-7, AdaptiveOptions::with_tolerance(1e-11))
            .expect("adaptive jacobian"),
    );

    println!("--- stiffness {k:.0e}, realised restitution {e:.4} ------------------------------");
    println!("the impact map, three ways (rows are [dh/dh-, dh/dv-; dv/dh-, dv/dv-]):");
    for (name, m) in [("exact saltation", &exact), ("fixed-step autodiff", &fixed), ("tolerance-driven", &tol)] {
        println!("  {name:>20}  [{:>10.4} {:>10.4} ; {:>10.4} {:>10.4}]", m[(0, 0)], m[(0, 1)], m[(1, 0)], m[(1, 1)]);
    }

    // The gap the tube has to absorb: how far the smoothed one-step map lands from the rigid one, sampled around the
    // nominal impact state. Sampled, so this can refute but never certify - which is the point of the last section.
    let gap = sampled_gap(&penalty, &rigid, x_pre, window);
    println!(
        "\n  sampled smoothing gap (max half-width {:.3e} over {} states)",
        gap.magnitude(),
        match gap.evidence {
            GapEvidence::Sampled { samples } => samples,
            _ => 0,
        }
    );

    // Free flight over one control step is exact for both models, so those steps carry no gap.
    let flight = DMatrix::from_row_slice(2, 2, &[1.0, DT, 0.0, 1.0]);
    let zero_gap = GapBound::assume_bound(&DVector::zeros(2), 0.0, 0.0).expect("zero gap");
    // The impact map's linearisation residual. Zero is an ASSERTION here, and an unproved one: nothing in this
    // workspace yet bounds how far the true hybrid transition departs from its saltation-composed linearisation.
    let asserted_residual = GapBound::assume_bound(&DVector::zeros(2), 0.0, 0.0).expect("residual");
    // The flight steps' zero gap is justified by "both models integrate the same quadratic" — which holds only above
    // the plane. Measured on this very fixture, the first three flight steps have reachable sets containing h < 0, so
    // the justification fails there and the zero is false. Attaching the precondition makes certify() say so.
    let above_plane = HalfSpace::new(DVector::from_vec(vec![-1.0, 0.0]), 0.0);

    // A small spread of entry states, so the tube starts with a real width and its growth is meaningful.
    let x0 = Zonotope::from_interval(
        &DVector::from_vec(vec![-1e-4, -1e-3]),
        &DVector::from_vec(vec![1e-4, 1e-3]),
    );
    let ceiling = vec![HalfSpace::new(DVector::from_vec(vec![1.0, 0.0]), CEILING)];

    // The experiment. The gap is identical in all three rows; the ONLY thing that changes is which impact Jacobian
    // the tube is propagated through. A certificate is computed from a linearisation, so a wrong linearisation does
    // not merely produce a worse design - it produces a wrong certificate.
    println!("\n  {:>20}  {:>11}  {:>11}  {:>9}  verdict", "tube propagated via", "final width", "vs truth", "growth");
    let mut truth_width = f64::NAN;
    for (name, m) in [("exact saltation", &exact), ("fixed-step autodiff", &fixed), ("tolerance-driven", &tol)] {
        // The impact map is a LINEARISATION of a hybrid transition, so its residual is asserted, not structural.
        let mut steps = vec![TubeStep::new(m.clone(), gap.clone(), asserted_residual.clone())];
        for _ in 0..HORIZON {
            steps.push(
                TubeStep::linear(flight.clone(), zero_gap.clone())
                    .expect("flight step")
                    .requiring(above_plane.clone()),
            );
        }
        let Some(tube) = propagate_tube(&x0, &steps) else {
            println!("  {name:>20}  propagation refused");
            continue;
        };
        let nominal = nominal_after_impact(&rigid, v, tube.sets.len());
        let verdict = certify(&nominal, &tube, &ceiling);
        if truth_width.is_nan() {
            truth_width = tube.final_width();
        }
        let label = match &verdict {
            TubeVerdict::Certified { margin } => format!("CERTIFIED (margin {margin:.3e})"),
            TubeVerdict::Refuted { step, violation, .. } => format!("REFUTED at step {step} by {violation:.3e}"),
            TubeVerdict::Undecided { reason } => format!("undecided: {reason:?}"),
        };
        println!(
            "  {name:>20}  {:>11.3e}  {:>10.2}x  {:>9.3}  {label}",
            tube.final_width(),
            tube.final_width() / truth_width,
            tube.growth()
        );
    }
    // Every row above is Undecided, because the gap was sampled. That is the module working, not failing: a tube
    // built on a lower bound is not a certificate. The useful follow-up question is whether PROVING the gap would be
    // worth the effort, and that is answerable without proving it - assume a sound bound of the measured magnitude and
    // see what verdict it would buy. A conditional result, labelled as one.
    let conditional = GapBound::assume_bound(&gap.half_width, 0.0, 0.0).expect("conditional gap");
    let mut steps = vec![TubeStep::new(exact.clone(), conditional, asserted_residual.clone())];
    for _ in 0..HORIZON {
        steps.push(
            TubeStep::linear(flight.clone(), zero_gap.clone())
                .expect("flight step")
                .requiring(above_plane.clone()),
        );
    }
    if let Some(tube) = propagate_tube(&x0, &steps) {
        let nominal = nominal_after_impact(&rigid, v, tube.sets.len());
        let slack = nominal_activity(&nominal, &ceiling);
        println!(
            "  IF a sound bound of magnitude {:.3e} were proved: {:?}",
            gap.magnitude(),
            certify(&nominal, &tube, &ceiling)
        );
        println!(
            "  nominal slack {slack:.3e} against tube width {:.3e} (ratio {:.2}) - the constraint is {}",
            tube.final_width(),
            slack / tube.final_width(),
            if slack < 10.0 * tube.final_width() { "genuinely active" } else { "UNREACHABLE: the certificate would be vacuous" }
        );
        println!("  (conditional on a Lipschitz argument this module does not supply)");
    }

    println!();
}

/// The nominal trajectory the constraint is checked against: rebound at the realised restitution, then free flight.
fn nominal_after_impact(rigid: &BouncingMass, v: f64, steps: usize) -> Vec<DVector<f64>> {
    let mut out = Vec::with_capacity(steps);
    let v_plus = rigid.restitution * v;
    for i in 0..steps {
        let t = i as f64 * DT;
        let h = v_plus * t - 0.5 * GRAVITY * t * t;
        let vel = v_plus - GRAVITY * t;
        out.push(DVector::from_vec(vec![h.max(0.0), vel]));
    }
    out
}

/// Sample the deviation of the smoothed impact map's *output* from the rigid one, over a spread of entry states. The
/// result is a lower bound on the true supremum, and is labelled as such by [`GapBound::from_samples`].
fn sampled_gap(penalty: &PenaltyMass, rigid: &BouncingMass, x_pre: [f64; 2], window: f64) -> GapBound {
    let mut residuals = Vec::new();
    for i in 0..25 {
        // A spread of entry speeds around the nominal, which is what a tube around a trajectory has to cover.
        let scale = 0.9 + 0.2 * (i as f64) / 24.0;
        let entry = [x_pre[0], x_pre[1] * scale];
        let smooth = penalty.rollout(entry, window);
        let (flown, _) = rigid.flow(entry, window);
        let r = DVector::from_vec(vec![smooth[0] - flown[0], smooth[1] - flown[1]]);
        if r.iter().all(|x| x.is_finite()) {
            residuals.push(r);
        }
    }
    GapBound::from_samples(&residuals).expect("at least one finite residual")
}



fn to_matrix(j: [[f64; 2]; 2]) -> DMatrix<f64> {
    DMatrix::from_row_slice(2, 2, &[j[0][0], j[0][1], j[1][0], j[1][1]])
}

/// The point of the evidence type, shown rather than described: the same tube, the same constraint, the same
/// enormous margin, and two different verdicts depending only on how the gap was established.
fn soundness_demonstration() {
    let tiny = DVector::from_vec(vec![1e-9, 1e-9]);
    let sampled = GapBound::from_samples(std::slice::from_ref(&tiny)).expect("sampled");
    let proved = GapBound::assume_bound(&tiny, 0.0, 0.0).expect("proved");
    let contract = DMatrix::from_row_slice(2, 2, &[0.5, 0.0, 0.0, 0.5]);
    let nominal: Vec<DVector<f64>> = (0..11).map(|_| DVector::zeros(2)).collect();
    let wide = vec![HalfSpace::new(DVector::from_vec(vec![1.0, 0.0]), 1e6)];

    for (name, gap) in [("sampled at 1 state", sampled), ("proved by Lipschitz", proved)] {
        let steps: Vec<TubeStep> =
            (0..10).map(|_| TubeStep::linear(contract.clone(), gap.clone()).unwrap()).collect();
        let tube = propagate_tube(&Zonotope::point(DVector::zeros(2)), &steps).expect("tube");
        println!(
            "  gap {name:>22}: tube width {:.2e}, constraint slack ~1e6  ->  {:?}",
            tube.final_width(),
            certify(&nominal, &tube, &wide)
        );
    }
    println!("\n  Identical geometry. The verdict tracks the evidence, not the margin.");
}
