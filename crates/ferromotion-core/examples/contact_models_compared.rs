//! **Penalty contact vs hard interior-point contact, on the same walking robot.**
//!
//! The quadruped's scripted crawl walks forward under the compliant penalty stepper
//! ([`tree_floating_contact_step`]) and barely moves under the hard interior-point stepper
//! ([`whole_body_contact_step`]). Same body, same gait, same torques. This runs both side by side and
//! instruments what actually differs: how deep the feet sink, how much a stance foot slips, and how
//! much forward momentum survives each step.
//!
//! Locomotion is friction doing work against the ground, so a stance foot that slips cannot propel.
//! The question this answers is whether the hard solver is failing to grip (a defect) or whether the
//! compliant model is propelling for a reason the hard one legitimately does not reproduce.
//!
//! Run: `cargo run --release --example contact_models_compared -p ferromotion-core`

use ferromotion_core::{whole_body_contact_step_pgs, quadruped, quadruped_trot_tau, tree_floating_contact_step, whole_body_contact_step, whole_body_contact_step_checked, whole_body_forward_kinematics, LinkInertia, WholeBodyContactPoint};
use nalgebra::{Isometry3, Matrix3, Point3, Vector3, Vector6};

const DT: f64 = 2e-4;
const STEPS: usize = 10000; // 2 s
const FLOOR: f64 = 0.0;
const STAND_Z: f64 = 0.60;

fn base_inertia() -> LinkInertia {
    LinkInertia { mass: 8.0, com: Vector3::zeros(), inertia: Matrix3::from_diagonal(&Vector3::new(0.08, 0.08, 0.12)) }
}

/// World positions of the four feet.
fn feet_world(joints: &[ferromotion_core::Joint], parent: &[isize], base: Isometry3<f64>, q: &[f64]) -> Vec<Vector3<f64>> {
    let w = whole_body_forward_kinematics(joints, parent, base, q);
    (0..4).map(|leg| (w[leg * 2 + 1] * Point3::new(0.0, 0.0, -0.3)).coords).collect()
}

struct Report {
    travel: f64,
    /// total backward slip of feet while they are loaded (planted): the anti-propulsion measure
    stance_slip: f64,
    /// worst penetration below the floor (negative = sunk in)
    worst_pen: f64,
    /// mean height of the lowest foot, a proxy for whether the gait is actually bearing weight
    mean_low_foot: f64,
}

fn run(hard: bool, mu: f64) -> Report {
    let (joints, inertia, parent, feet) = quadruped();
    let n = joints.len();
    let bi = base_inertia();
    let g = Vector3::new(0.0, 0.0, -9.81);
    let ipm: Vec<WholeBodyContactPoint> = feet.iter().map(|&(b, o, _)| WholeBodyContactPoint::on(b, o, mu)).collect();
    let pen: Vec<(usize, Vector3<f64>, f64)> = feet.iter().map(|&(b, o, _)| (b, o, mu)).collect();

    let mut base = Isometry3::translation(0.0, 0.0, STAND_Z);
    let (mut v0, mut q, mut qd) = (Vector6::zeros(), vec![0.0; n], vec![0.0; n]);
    let mut prev = feet_world(&joints, &parent, base, &q);
    let (mut slip, mut worst_pen, mut low_sum) = (0.0f64, 0.0f64, 0.0f64);

    for k in 0..STEPS {
        let t = k as f64 * DT;
        let tau = quadruped_trot_tau(&q, &qd, std::f64::consts::TAU * t);
        let (b1, v1, q1, qd1) = if hard {
            whole_body_contact_step(&joints, &inertia, &parent, &bi, base, v0, &q, &qd, &tau, &ipm, FLOOR, DT, 1e-6, g)
        } else {
            tree_floating_contact_step(&joints, &inertia, &parent, &bi, base, v0, &q, &qd, &tau, &pen, FLOOR, 1.5e4, 120.0, DT, g)
        };
        base = b1;
        v0 = v1;
        q = q1;
        qd = qd1;
        if !base.translation.vector.iter().all(|x| x.is_finite()) {
            return Report { travel: f64::NAN, stance_slip: f64::NAN, worst_pen: f64::NAN, mean_low_foot: f64::NAN };
        }
        let now = feet_world(&joints, &parent, base, &q);
        let mut lowest = f64::INFINITY;
        for i in 0..4 {
            worst_pen = worst_pen.min(now[i].z - FLOOR);
            lowest = lowest.min(now[i].z);
            // a foot within a millimetre of the ground is bearing load; any backward travel it makes
            // there is slip rather than propulsion
            if now[i].z < FLOOR + 1e-3 {
                let dx = now[i].x - prev[i].x;
                if dx < 0.0 {
                    slip += -dx;
                }
            }
        }
        low_sum += lowest;
        prev = now;
    }
    Report { travel: base.translation.x, stance_slip: slip, worst_pen, mean_low_foot: low_sum / STEPS as f64 }
}

/// Trace the hard solver at a given friction coefficient, reporting the first step where the body
/// gains energy it was never given: the signature of a contact solve that has stopped converging.
fn trace_blowup(mu: f64) {
    let (joints, inertia, parent, feet) = quadruped();
    let n = joints.len();
    let bi = base_inertia();
    let g = Vector3::new(0.0, 0.0, -9.81);
    let ipm: Vec<WholeBodyContactPoint> = feet.iter().map(|&(b, o, _)| WholeBodyContactPoint::on(b, o, mu)).collect();
    let mut base = Isometry3::translation(0.0, 0.0, STAND_Z);
    let (mut v0, mut q, mut qd) = (Vector6::zeros(), vec![0.0; n], vec![0.0; n]);
    let mut worst_speed = 0.0f64;
    for k in 0..STEPS {
        let t = k as f64 * DT;
        let tau = quadruped_trot_tau(&q, &qd, std::f64::consts::TAU * t);
        let (b1, v1, q1, qd1) = whole_body_contact_step(&joints, &inertia, &parent, &bi, base, v0, &q, &qd, &tau, &ipm, FLOOR, DT, 1e-6, g);
        let sp = v1.norm();
        // a contact can never add speed to a body that is only being held up
        if sp > 6.0 && sp > worst_speed * 2.0 && worst_speed > 0.0 {
            println!("  mu={mu}: speed jumped {:.2} -> {:.2} m/s at step {k} (t={:.3}s), base z {:.3} -> {:.3}",
                worst_speed, sp, t, base.translation.z, b1.translation.z);
            return;
        }
        worst_speed = worst_speed.max(sp);
        base = b1; v0 = v1; q = q1; qd = qd1;
        if !base.translation.vector.iter().all(|x| x.is_finite()) { println!("  mu={mu}: NaN at step {k}"); return; }
    }
    println!("  mu={mu}: no sudden jump; peak speed {:.2} m/s, final base z {:.3}", worst_speed, base.translation.z);
}

/// The invariant a contact cannot violate: with no actuation, total mechanical energy may fall
/// (friction dissipates) but must never rise. Drop the quadruped with zero torque and watch the
/// energy ledger across a sweep of friction coefficients.
fn energy_ledger(mu: f64) -> (f64, f64) {
    let (joints, inertia, parent, feet) = quadruped();
    let n = joints.len();
    let bi = base_inertia();
    let g = Vector3::new(0.0, 0.0, -9.81);
    let ipm: Vec<WholeBodyContactPoint> = feet.iter().map(|&(b, o, _)| WholeBodyContactPoint::on(b, o, mu)).collect();
    let mut base = Isometry3::translation(0.0, 0.0, STAND_Z);
    let (mut v0, mut q, mut qd) = (Vector6::zeros(), vec![0.0; n], vec![0.0; n]);
    let tau = vec![0.0; n];
    // total mass for the gravitational term (base + links)
    let mtot: f64 = bi.mass + inertia.iter().map(|l| l.mass).sum::<f64>();
    let energy = |base: &Isometry3<f64>, v0: &Vector6<f64>, qd: &[f64]| {
        let kin = 0.5 * bi.mass * v0.fixed_rows::<3>(3).norm_squared()
            + 0.5 * 0.1 * qd.iter().map(|v| v * v).sum::<f64>();
        kin + mtot * 9.81 * base.translation.z
    };
    let e0 = energy(&base, &v0, &qd);
    let mut worst_gain = 0.0f64;
    for _ in 0..5000 {
        let (b1, v1, q1, qd1) = whole_body_contact_step(&joints, &inertia, &parent, &bi, base, v0, &q, &qd, &tau, &ipm, FLOOR, DT, 1e-6, g);
        base = b1; v0 = v1; q = q1; qd = qd1;
        if !base.translation.vector.iter().all(|x| x.is_finite()) { return (f64::NAN, f64::NAN); }
        worst_gain = worst_gain.max(energy(&base, &v0, &qd) - e0);
    }
    (e0, worst_gain)
}

/// Test the prediction: if the high-friction blow-up is the actuated gait fighting a rigid constraint
/// under an explicit integrator, then shrinking the timestep should cure it, while the contact model
/// stays untouched.
fn dt_sensitivity(mu: f64, dt: f64) -> (f64, f64) {
    let (joints, inertia, parent, feet) = quadruped();
    let n = joints.len();
    let bi = base_inertia();
    let g = Vector3::new(0.0, 0.0, -9.81);
    let ipm: Vec<WholeBodyContactPoint> = feet.iter().map(|&(b, o, _)| WholeBodyContactPoint::on(b, o, mu)).collect();
    let mut base = Isometry3::translation(0.0, 0.0, STAND_Z);
    let (mut v0, mut q, mut qd) = (Vector6::zeros(), vec![0.0; n], vec![0.0; n]);
    let steps = (1.0 / dt) as usize; // a fixed 1 s of simulated time
    let mut peak = 0.0f64;
    for k in 0..steps {
        let t = k as f64 * dt;
        let tau = quadruped_trot_tau(&q, &qd, std::f64::consts::TAU * t);
        let (b1, v1, q1, qd1) = whole_body_contact_step(&joints, &inertia, &parent, &bi, base, v0, &q, &qd, &tau, &ipm, FLOOR, dt, 1e-6, g);
        base = b1; v0 = v1; q = q1; qd = qd1;
        if !base.translation.vector.iter().all(|x| x.is_finite()) { return (f64::NAN, f64::NAN); }
        peak = peak.max(v1.norm());
    }
    (peak, base.translation.z)
}

/// The decisive question: when the driven body runs away at high friction, is the contact solve still
/// returning a valid complementarity solution (so the MODEL permits it) or has it quietly stopped
/// converging (so it is a solver problem)? Report the worst solver health at the moment speed peaks.
fn health_at_blowup(mu: f64) {
    let (joints, inertia, parent, feet) = quadruped();
    let n = joints.len();
    let bi = base_inertia();
    let g = Vector3::new(0.0, 0.0, -9.81);
    let ipm: Vec<WholeBodyContactPoint> = feet.iter().map(|&(b, o, _)| WholeBodyContactPoint::on(b, o, mu)).collect();
    let mut base = Isometry3::translation(0.0, 0.0, STAND_Z);
    let (mut v0, mut q, mut qd) = (Vector6::zeros(), vec![0.0; n], vec![0.0; n]);
    let (mut peak, mut res_at_peak, mut feas_at_peak, mut worst_res, mut worst_feas) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, f64::INFINITY);
    for k in 0..STEPS {
        let t = k as f64 * DT;
        let tau = quadruped_trot_tau(&q, &qd, std::f64::consts::TAU * t);
        let ((b1, v1, q1, qd1), (res, feas)) = whole_body_contact_step_checked(&joints, &inertia, &parent, &bi, base, v0, &q, &qd, &tau, &ipm, FLOOR, DT, 1e-6, g);
        base = b1; v0 = v1; q = q1; qd = qd1;
        if !base.translation.vector.iter().all(|x| x.is_finite()) { break; }
        worst_res = worst_res.max(res);
        worst_feas = worst_feas.min(feas);
        if v1.norm() > peak { peak = v1.norm(); res_at_peak = res; feas_at_peak = feas; }
    }
    let verdict = if worst_res > 1e-6 { "SOLVER not converging" } else { "solver converged: the MODEL allows this" };
    println!("  mu={mu:<4} peak {peak:8.2} m/s | at peak: complementarity {res_at_peak:.2e}, feasibility {feas_at_peak:+.2e} | worst over run: res {worst_res:.2e}, feas {worst_feas:+.2e}  -> {verdict}");
}

/// The same driven gait, solved by the Gauss-Seidel sweep instead of the interior-point core.
fn pgs_run(mu: f64) {
    let (joints, inertia, parent, feet) = quadruped();
    let n = joints.len();
    let bi = base_inertia();
    let g = Vector3::new(0.0, 0.0, -9.81);
    let pts: Vec<WholeBodyContactPoint> = feet.iter().map(|&(b, o, _)| WholeBodyContactPoint::on(b, o, mu)).collect();
    let mut base = Isometry3::translation(0.0, 0.0, STAND_Z);
    let (mut v0, mut q, mut qd) = (Vector6::zeros(), vec![0.0; n], vec![0.0; n]);
    let mut warm: Option<Vec<nalgebra::Vector3<f64>>> = None;
    let (mut peak, mut worst_viol) = (0.0f64, 0.0f64);
    for k in 0..STEPS {
        let t = k as f64 * DT;
        let tau = quadruped_trot_tau(&q, &qd, std::f64::consts::TAU * t);
        let r = whole_body_contact_step_pgs(&joints, &inertia, &parent, &bi, base, v0, &q, &qd, &tau, &pts, FLOOR, DT, 60, g, warm.as_deref());
        base = r.base; v0 = r.v0; q = r.q.clone(); qd = r.qd.clone();
        warm = Some(r.impulses);
        if !base.translation.vector.iter().all(|x| x.is_finite()) { println!("  mu={mu}: DIVERGED at step {k}"); return; }
        peak = peak.max(v0.norm());
        worst_viol = worst_viol.max(r.violation);
    }
    println!("  mu={mu:<4} peak {peak:7.2} m/s | travel {:6.3} m | worst violation {worst_viol:.2e} | final base z {:.3}", base.translation.x, base.translation.z);
}

fn main() {
    println!("The same quadruped and the same crawl gait, stepped by two contact models");
    println!("(2 s, dt = {DT} s; penalty = spring-damper, hard = interior-point non-penetration + cone)\n");
    println!("{:<28} {:>10} {:>13} {:>14} {:>15}", "model", "travel m", "stance slip m", "worst pen mm", "mean low foot m");
    for &mu in &[0.6, 0.9, 1.5] {
        for &hard in &[false, true] {
            let r = run(hard, mu);
            let name = format!("{} mu={mu}", if hard { "hard interior-point" } else { "penalty spring-damper" });
            println!("{:<28} {:>10.3} {:>13.3} {:>14.2} {:>15.3}", name, r.travel, r.stance_slip, r.worst_pen * 1000.0, r.mean_low_foot);
        }
    }
    println!("\nwhere the hard solver loses it:");
    for &mu in &[0.9, 1.2, 1.5, 2.0] { trace_blowup(mu); }
    println!("\nenergy ledger, dropped with ZERO torque (a contact may dissipate, never generate):");
    for &mu in &[0.3, 0.6, 0.9, 1.0, 1.2, 1.5] {
        let (e0, gain) = energy_ledger(mu);
        let verdict = if !gain.is_finite() { "DIVERGED" } else if gain > 0.05 * e0.abs() { "ENERGY CREATED" } else { "ok" };
        println!("  mu={mu:<4} start {e0:8.2} J, worst gain {gain:+10.3} J   {verdict}");
    }
    println!("\ntimestep sensitivity at high friction (1 s of the same gait, hard contact):");
    for &mu in &[1.2, 1.5] {
        for &dt in &[2e-4, 1e-4, 5e-5, 2e-5] {
            let (peak, z) = dt_sensitivity(mu, dt);
            let verdict = if !peak.is_finite() { "diverged" } else if peak > 5.0 { "unstable" } else { "stable" };
            println!("  mu={mu}  dt={dt:<8} peak speed {peak:9.2} m/s, final base z {z:8.3}   {verdict}");
        }
    }
    println!("\nis the runaway a solver failure or a model limit? (kappa = 1e-6, so a converged solve has residual ~1e-6)");
    for &mu in &[0.9, 1.2, 1.5, 2.0] { health_at_blowup(mu); }
    println!("\nthe SAME driven gait through the Gauss-Seidel sweep (the interior-point core ran away here):");
    for &mu in &[0.9, 1.2, 1.5, 2.0] { pgs_run(mu); }
    println!("\nA foot that slips backward while loaded is cancelling the stride. Compare the slip columns:");
    println!("if the hard model slips far more at the same friction coefficient, it is not gripping, and");
    println!("the gait has nothing to push against.");
}
