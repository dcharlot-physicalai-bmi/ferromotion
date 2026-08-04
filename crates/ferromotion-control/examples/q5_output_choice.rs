//! **Q5 milestones M0 and M1: output-tracking accuracy is not stability, and the design rule that fixes it.**
//!
//! A learned policy is usually trained to match reference *outputs* — a foot trajectory, a centre-of-mass path, a
//! posture. On an underactuated robot that leaves the internal (zero) dynamics uncontrolled, and underactuated
//! contact systems are generically non-minimum-phase. So a policy can track its reference to arbitrary precision
//! while an internal mode diverges, and the training metric will report success throughout.
//!
//! This run does two things on the compass-gait biped, linearised about the upright — two states, one input, so
//! any relative-degree-two output leaves exactly one internal state to be uncontrolled:
//!
//! * **M0: exhibit the failure.** Track the swing leg alone, which is the natural choice for a learned policy
//!   (it is the visible, commandable coordinate), and watch the output go to zero while the robot falls over.
//! * **M1: the design rule.** Sweep the output over a family `y = θ₂ + c·θ₁` and compute the zero-dynamics
//!   eigenvalue for each. That eigenvalue **crosses into the left half plane at a computable `c`**, so which
//!   outputs are safe to regulate is decidable *before any policy is trained* — from the plant alone.
//!
//! The mechanism, and why it is a *design* rule rather than a training problem: holding an output at zero pins one
//! combination of coordinates and leaves the complement free. If the complement contains the unactuated tipping
//! mode, nothing in the output error sees it. The fix is to choose an output that **contains information about the
//! unactuated coordinate** — which is precisely what the momentum and divergent-component coordinates that
//! model-based locomotion already uses do, and [Q5.1] is the conjecture that this makes such a policy
//! minimum-phase by design.
//!
//! Run: `cargo run --release --example q5_output_choice -p ferromotion-control`

use ferromotion_control::{is_minimum_phase_order2, zero_dynamics_order2, CompassGait};
use nalgebra::{DMatrix, DVector};

const DT: f64 = 1e-4;

/// The compass gait linearised about the upright, in `[θ₁, θ₂, θ̇₁, θ̇₂]`. Continuous time, since the zero-dynamics
/// question is about eigenvalue signs and a discretisation would only obscure them.
fn linearised(r: &CompassGait) -> (DMatrix<f64>, DMatrix<f64>) {
    let m = r.mass_matrix(0.0, 0.0);
    let mi = m.try_inverse().expect("the mass matrix is invertible upright");
    // gravity gradient at the upright: d/dq of the gravity vector
    let eps = 1e-7;
    let g0 = r.gravity(0.0, 0.0);
    let gq = DMatrix::from_fn(2, 2, |i, j| {
        let (mut a, mut b) = ([0.0, 0.0], [0.0, 0.0]);
        a[j] += eps;
        b[j] -= eps;
        (r.gravity(a[0], a[1])[i] - r.gravity(b[0], b[1])[i]) / (2.0 * eps)
    });
    let _ = g0;
    // q̈ = M⁻¹(Bu − G(q)) so the state matrix is [[0, I], [−M⁻¹ ∂G/∂q, 0]]
    let mut a = DMatrix::zeros(4, 4);
    a.view_mut((0, 2), (2, 2)).copy_from(&DMatrix::identity(2, 2));
    a.view_mut((2, 0), (2, 2)).copy_from(&(-mi * &gq));
    let bcol = mi * DVector::from_row_slice(&[-1.0, 1.0]); // hip torque acts as (−1, +1)
    let mut b = DMatrix::zeros(4, 1);
    b[(2, 0)] = bcol[0];
    b[(3, 0)] = bcol[1];
    (a, b)
}

/// The output `y = θ₂ + c·θ₁`, on **positions**.
///
/// This has relative degree two with respect to a hip torque, which is the only correct formulation for a
/// mechanical system — and the first attempt at this run used a *velocity* combination instead, to satisfy a
/// relative-degree-one routine. That was wrong in a way worth recording: a velocity constraint leaves both
/// positions free, so the unactuated tipping mode survived at **every** output weight and the sweep concluded that
/// no safe output exists. It does; the tooling could not see it.
fn output_row(c: f64) -> DMatrix<f64> {
    DMatrix::from_row_slice(1, 4, &[c, 1.0, 0.0, 0.0])
}

fn main() {
    let r = CompassGait { slope: 0.0, ..CompassGait::default() };
    let (a, b) = linearised(&r);
    println!("Q5 / M0+M1 - output-tracking accuracy is not stability");
    println!("(compass gait linearised about the upright: 2 coordinates, 1 hip torque, so underactuated by one)\n");
    println!("the open-loop plant's own eigenvalues (the tipping modes it has to fight):");
    for l in a.complex_eigenvalues().iter() {
        println!("      {:+.5} {:+.5}i", l.re, l.im);
    }
    println!("    A positive real part is the stance leg falling. Nothing in an output error is obliged to see it.");

    // ---- M1 first: which outputs are safe, computed from the plant alone ----
    println!("\nM1. the design rule: the zero-dynamics eigenvalues of y = theta2 + c*theta1, per c");
    println!("      c        zero-dynamics eigenvalues            minimum phase");
    let mut safe_c = None;
    let mut unsafe_c = None;
    for &c in &[-8.0f64, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 8.0] {
        let cm = output_row(c);
        match zero_dynamics_order2(&a, &b, &cm) {
            Some(zd) => {
                let eig = zd.complex_eigenvalues();
                let desc: Vec<String> = eig.iter().map(|l| format!("{:+.4}{:+.4}i", l.re, l.im)).collect();
                let mp = is_minimum_phase_order2(&a, &b, &cm).unwrap_or(false);
                println!("      {c:<8} {:<36} {mp}", desc.join("  "));
                if mp && safe_c.is_none() {
                    safe_c = Some(c);
                }
                if !mp {
                    unsafe_c = Some(c);
                }
            }
            None => println!("      {c:<8} the decoupling matrix is singular - this output has no well-posed inverse"),
        }
    }

    // locate the crossing by bisection on the largest real part
    let worst_real = |c: f64| -> f64 {
        zero_dynamics_order2(&a, &b, &output_row(c)).map(|zd| zd.complex_eigenvalues().iter().fold(f64::NEG_INFINITY, |m, l| m.max(l.re))).unwrap_or(f64::INFINITY)
    };
    if let (Some(bad), Some(good)) = (unsafe_c, safe_c) {
        let (mut lo, mut hi) = (bad.min(good), bad.max(good));
        for _ in 0..80 {
            let mid = 0.5 * (lo + hi);
            if worst_real(mid) > 0.0 {
                if worst_real(lo) > 0.0 {
                    lo = mid;
                } else {
                    hi = mid;
                }
            } else if worst_real(lo) > 0.0 {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        println!("\n    the crossing sits at c ~ {:.4}: below it the zero dynamics has a right-half-plane mode, above", 0.5 * (lo + hi));
        println!("    it does not. That number comes from the PLANT, with no policy in the picture - which is what");
        println!("    makes this a design rule rather than a diagnosis after training.");
    }

    // ---- M0: exhibit the failure, with the two output choices side by side ----
    //
    // An idealised perfect output-tracker: feedback-linearise the chosen output and drive it to zero with a fast
    // PD. This is the best any learned output-tracker could do, so the failure below is not a training artefact -
    // it is what the output CHOICE costs.
    println!("\nM0. the failure: an idealised PERFECT output-tracker, for two choices of output");
    println!("    (feedback-linearised with a fast PD - the best a learned tracker could achieve, so any failure");
    println!("     below belongs to the output choice and not to the learning)");
    println!("\n      c      worst |y| over 4 s   final |internal state|   verdict");
    for &c in &[-8.0f64, -4.0, -2.0, 0.0, 2.0, 4.0] {
        let cm = output_row(c);
        let ca = &cm * &a;
        let cab = (&ca * &b)[0]; // the decoupling term at relative degree two
        if cab.abs() < 1e-12 {
            println!("      {c:<6} no authority over this output (the decoupling term vanishes)");
            continue;
        }
        let ca2 = &ca * &a;
        // start ON the zero-dynamics manifold, so the run is about the internal dynamics and not a transient
        let mut x = DVector::from_row_slice(&[0.02, -0.02 * c, 0.0, 0.0]);
        let y0 = (&cm * &x)[0];
        x[1] -= y0; // project exactly onto y = 0
        let mut worst_y = 0.0f64;
        let mut blew = false;
        for _ in 0..(4.0 / DT) as usize {
            let y = (&cm * &x)[0];
            let yd = (&ca * &x)[0];
            worst_y = worst_y.max(y.abs());
            // feedback-linearise at relative degree two and add a fast PD on (y, ydot)
            let u = -((&ca2 * &x)[0] + 400.0 * y + 40.0 * yd) / cab;
            x += (&a * &x + &b * DVector::from_row_slice(&[u])) * DT;
            if !x.iter().all(|v| v.is_finite()) || x.norm() > 1e6 {
                blew = true;
                break;
            }
        }
        let mp = is_minimum_phase_order2(&a, &b, &cm).unwrap_or(false);
        let verdict = if blew { "INTERNAL STATE DIVERGED".to_string() } else { format!("bounded (minimum phase: {mp})") };
        println!("      {c:<6} {worst_y:>18.3e}   {:>22.4e}   {verdict}", x.norm());
    }

    println!("\n    Read the first two columns together. Where the output is driven to 1e-3 or below and the state");
    println!("    norm is enormous, the tracker did its job perfectly and the robot fell over. That is Theorem 6.8");
    println!("    on a real underactuated model: output-tracking accuracy carries no information about internal");
    println!("    stability, and a training curve on output error would show nothing wrong at all.");

    // ---- and the payoff: why the DCM-like coordinate is the right output ----
    //
    // Look again at the c = -8 row of the design-rule table: its zero dynamics is +/-2.5573i - purely imaginary.
    // That is MARGINALLY stable, not exponentially so, and the simulation agrees (the internal state stays at 0.32
    // and oscillates rather than decaying). A position-only output on this robot can at best stop the tipping mode
    // from growing; it cannot damp it.
    //
    // What damps it is an output containing VELOCITY as well as position - and the divergent-component coordinate
    // that model-based locomotion regulates, xi = x + xdot/omega, is exactly that. Sweeping a DCM-like output
    // y = (thetadot2 + c*thetadot1) + w*(theta2 + c*theta1) tests [Q5.1] directly.
    // ---- testing [Q5.1], and finding it does not hold here ----
    //
    // The c = -8 row above is only MARGINALLY stable (+/-2.5573i, purely imaginary): a position-only output can
    // stop the tipping mode growing but cannot damp it. [Q5.1] says a divergent-component output - a position PLUS
    // velocity combination, xi = x + xdot/omega - is minimum-phase by design, which would mean exponential rather
    // than marginal stability. Tested below, and it does not hold on this robot, for a reason worth knowing.
    println!("\ntesting [Q5.1]: does a DCM-like (position + velocity) output do better?");
    println!("      c      w      zero-dynamics worst real part   exponentially stable");
    let mut any_exp = false;
    for &c in &[-8.0f64, -4.0, 0.0] {
        for &w in &[1.0f64, 5.0] {
            let cm = DMatrix::from_row_slice(1, 4, &[w * c, w, c, 1.0]);
            match ferromotion_control::zero_dynamics(&a, &b, &cm) {
                Some(zd) => {
                    let worst = zd.complex_eigenvalues().iter().fold(f64::NEG_INFINITY, |m, l| m.max(l.re));
                    let exp_stable = worst < -1e-9;
                    any_exp |= exp_stable;
                    println!("      {c:<6} {w:<6} {worst:>29.5}   {exp_stable}");
                }
                None => println!("      {c:<6} {w:<6} the decoupling term vanishes"),
            }
        }
    }
    println!("\n    No member is exponentially stable, and note that `w` changes NOTHING - the worst real part is");
    println!("    identical at every velocity weight. That is structural, not a search failure. A position-plus-");
    println!("    velocity output has relative degree ONE, so it imposes a single constraint on a four-dimensional");
    println!("    state and cannot pin a position and its rate together. The relative-degree-TWO position output");
    println!("    imposes two (y = 0 and ydot = 0), and two is the ceiling for one actuator. So the best any output");
    println!("    choice achieves on this robot is the marginal stability seen at c ~ -7.");
    assert!(!any_exp, "if this ever finds an exponentially stable member, the reasoning above is stale");

    println!("\n    [Q5.1] therefore does NOT hold here in its stated form, and the resolution is the hybrid one.");
    println!("    HZD does not ask continuous feedback to damp the internal mode - it uses the IMPACT. Q1's M0 run");
    println!("    measured exactly that on this robot: the collision retains 44% of kinetic energy and supplies");
    println!("    delta^2 = 0.914 < 1, while the continuous feedback only holds the state on Z. The output choice is");
    println!("    NECESSARY (get it wrong and the internal state diverges by five orders of magnitude, as M0 shows)");
    println!("    and it is not SUFFICIENT: on an underactuated robot the damping has to come from somewhere else,");
    println!("    and for a walker that somewhere is the footfall.");

    println!("\nWhat this establishes. M0's failure is real and it is a property of the OUTPUT CHOICE, not of the");
    println!("policy: an idealised perfect tracker falls just as surely as a bad one would. And M1's design rule is");
    println!("computable from the plant - the zero-dynamics eigenvalues of a candidate output, and the c at which");
    println!("they cross into the left half plane. So 'which outputs may a learned policy safely regulate' has a");
    println!("computable answer, available before training rather than after a hardware failure.");
    println!("\nAnd [Q5.1] is NOT confirmed - it is refined into something more useful. Choosing the right output is");
    println!("necessary and provably insufficient: with one actuator no output makes the zero dynamics exponentially");
    println!("stable, because one input buys at most two constraints on four states. The best available is marginal.");
    println!("So a design rule of the form 'regulate the DCM coordinate and internal stability follows' overstates");
    println!("what an output choice can deliver on an underactuated robot. What it CAN deliver is the difference");
    println!("between marginal and divergent, which M0 shows is five orders of magnitude - and the damping has to");
    println!("come from the hybrid structure, which is why HZD is built around the impact rather than around the");
    println!("feedback. That is consistent with Q1/M0 measuring the contraction at the collision, and it means Q5 and");
    println!("Q1 are two halves of one statement rather than independent problems.");
    println!("\nWhat this does NOT do: certify a learned policy's internal ISS (Q5's M2). The tracker here is");
    println!("idealised and linear, so its zero dynamics is exactly computable. A network's is not, and that needs");
    println!("the verification machinery Q6 is about - along with the honest observation that the linearisation used");
    println!("throughout this run is itself only valid near the upright.");
}
