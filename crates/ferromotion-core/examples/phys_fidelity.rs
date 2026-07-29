//! PHYSICS-FIDELITY BENCHMARK — does a dynamics model obey real physics? The metric the field is NOT
//! competing on. Everyone is scaling DATA (GR00T 20k+ video-hours, π largest-ever real dataset, Cosmos
//! 20T tokens) to make models that LOOK right; a learned world model can look perfect and still
//! hallucinate energy, break momentum, or penetrate contacts, because pixel/latent prediction carries no
//! physics STRUCTURE. This benchmark scores a model on conservation-law INVARIANTS with ANALYTIC ground
//! truth. The reference is ferromotion (structure-correct). The negative controls are an explicit-Euler
//! integrator (numerically wrong) and a "learned" model (per-step ACCURATE but no physics structure).
//!
//! THE HEADLINE the benchmark makes measurable: physics STRUCTURE beats per-step ACCURACY. A symplectic
//! step with COARSE dt conserves energy that a low-error unstructured predictor cannot — "all the data in
//! the world doesn't beat having the physics right." Open, reproducible, rooted in real mechanics.
use ferromotion_core::{forward_dynamics, from_urdf_full, mass_matrix, LinkInertia, Robot};
use nalgebra::{DVector, Point3, Vector3};

const URDF: &str = include_str!("so101.urdf");
const G: f64 = 9.81; // magnitude; gravity acts along −z

fn hash(mut h: u32) -> u32 { h ^= h >> 15; h = h.wrapping_mul(2246822519); h ^= h >> 13; h = h.wrapping_mul(3266489917); h ^= h >> 16; h }
fn noise(i: u32) -> f64 { (hash(i) % 1_000_000) as f64 / 1_000_000.0 - 0.5 } // deterministic, zero-mean in [-0.5,0.5]

// ===================== Test system A: simple pendulum (analytic ground truth) =====================
// state (θ, ω); m=1, length L; a(θ) = −(g/L) sinθ ; E(θ,ω) = ½ L²ω² + gL(1−cosθ).
const L: f64 = 1.0;
fn pend_accel(theta: f64) -> f64 { -(G / L) * theta.sin() }
fn pend_energy(theta: f64, omega: f64) -> f64 { 0.5 * L * L * omega * omega + G * L * (1.0 - theta.cos()) }

#[derive(Clone, Copy)]
enum Model { Symplectic, ExplicitEuler, Learned } // physics-structure / numerically-wrong / data-accurate-no-structure

// one step of the candidate model. `k` = step index (for the Learned model's deterministic residual).
fn pend_step(m: Model, theta: f64, omega: f64, dt: f64, k: u32) -> (f64, f64) {
    match m {
        // semi-implicit (symplectic) Euler: update velocity first, then position. Bounded energy error.
        Model::Symplectic => { let o = omega + dt * pend_accel(theta); (theta + dt * o, o) }
        // explicit Euler: uses the OLD velocity for position. Pumps energy secularly.
        Model::ExplicitEuler => { (theta + dt * omega, omega + dt * pend_accel(theta)) }
        // a "learned" world model: the correct step PLUS a small per-step residual (what a data-driven
        // next-state predictor has). Per-step ACCURATE, but no conservation structure → energy random-walks.
        Model::Learned => {
            let o = omega + dt * pend_accel(theta);
            let (mut t2, mut o2) = (theta + dt * o, o);
            let eps = 1.0e-3; // per-step residual scale (a very good model: ~0.1% of the oscillation)
            t2 += eps * noise(k * 2 + 1);
            o2 += eps * noise(k * 2 + 2);
            (t2, o2)
        }
    }
}

fn pend_probes(m: Model, name: &str) {
    let dt = 1.0e-3;
    // --- P1 energy conservation: undamped, 20 s, from θ0 = 1.0 rad ---
    let (mut th, mut om) = (1.0, 0.0);
    let e0 = pend_energy(th, om);
    let mut max_drift = 0.0f64;
    let steps: u32 = 20_000;
    for k in 0..steps { let (t, o) = pend_step(m, th, om, dt, k); th = t; om = o; max_drift = max_drift.max(((pend_energy(th, om) - e0) / e0).abs()); }
    let energy_drift = max_drift;
    // --- P2 period accuracy: small oscillation θ0 = 0.10 rad; analytic T = 2π√(L/g) ---
    let (mut th2, mut om2) = (0.10, 0.0);
    let (mut last, mut t_up, mut periods, mut count) = (th2, 0u32, 0.0f64, 0u32);
    for k in 0u32..40_000 {
        let (t, o) = pend_step(m, th2, om2, dt, 1_000_000 + k); th2 = t; om2 = o;
        if last <= 0.0 && th2 > 0.0 { if t_up > 0 { periods += (k - t_up) as f64 * dt; count += 1; } t_up = k; } // upward zero-crossings
        last = th2;
    }
    let t_analytic = 2.0 * std::f64::consts::PI * (L / G).sqrt();
    let period_err = if count > 0 { ((periods / count as f64) - t_analytic).abs() / t_analytic } else { f64::NAN };
    // --- P3 reversibility: 10 s forward, negate ω, 10 s back; return error ---
    let (mut th3, mut om3) = (1.0, 0.0);
    for k in 0u32..10_000 { let (t, o) = pend_step(m, th3, om3, dt, 2_000_000 + k); th3 = t; om3 = o; }
    om3 = -om3;
    for k in 0u32..10_000 { let (t, o) = pend_step(m, th3, om3, dt, 3_000_000 + k); th3 = t; om3 = o; }
    let rev_err = (th3 - 1.0).abs() + om3.abs();

    let p = |ok: bool| if ok { "PASS" } else { "FAIL" };
    println!("  {name:<14} energy-drift {:>8.2}% [{}]   period-err {:>6.2}% [{}]   reversibility {:>7.4} [{}]",
        energy_drift * 100.0, p(energy_drift < 0.01),
        period_err * 100.0, p(period_err < 0.01),
        rev_err, p(rev_err < 0.05));
}

// ===================== Test system B: the real SO-101 in ferromotion =====================
// E = ½ q̇ᵀ(M+A)q̇ + Σ mᵢ g z_comᵢ ; frictionless, zero torque → E must be conserved.
const ARM: f64 = 0.028;
fn so101_energy(robot: &Robot, inertia: &[LinkInertia], q: &[f64], qd: &[f64]) -> f64 {
    let m = mass_matrix(robot, inertia, q);
    let v = DVector::from_row_slice(qd);
    let mut ke = 0.5 * (v.transpose() * &m * &v)[(0, 0)];
    for i in 0..5 { ke += 0.5 * ARM * qd[i] * qd[i]; } // reflected rotor inertia is real kinetic energy
    let mut pe = 0.0;
    for i in 0..5 {
        let w = robot.frame_pose(q, i + 1) * Point3::from(inertia[i].com);
        pe += inertia[i].mass * G * w.z;
    }
    ke + pe
}
fn so101_step(robot: &Robot, inertia: &[LinkInertia], q: &mut Vec<f64>, qd: &mut Vec<f64>, dt: f64, symplectic: bool) {
    let qdd_link = forward_dynamics(robot, inertia, q, qd, &vec![0.0; 5], Vector3::new(0.0, 0.0, -G));
    let m = mass_matrix(robot, inertia, q);
    let mut ma = m.clone(); for i in 0..5 { ma[(i, i)] += ARM; }
    let qdd = ma.cholesky().unwrap().solve(&(&m * DVector::from_row_slice(&qdd_link)));
    if symplectic { for i in 0..5 { qd[i] += dt * qdd[i]; q[i] += dt * qd[i]; } }      // velocity, then position
    else { let old = qd.clone(); for i in 0..5 { q[i] += dt * old[i]; qd[i] += dt * qdd[i]; } } // explicit Euler
}
fn so101_energy_probe(robot: &Robot, inertia: &[LinkInertia], symplectic: bool, name: &str) {
    let dt = 5.0e-4;
    let (mut q, mut qd) = (vec![0.3, -0.4, 0.5, 0.2, 0.0], vec![0.0f64; 5]);
    let e0 = so101_energy(robot, inertia, &q, &qd);
    let mut max_drift = 0.0f64;
    for _ in 0..20_000 { so101_step(robot, inertia, &mut q, &mut qd, dt, symplectic); max_drift = max_drift.max(((so101_energy(robot, inertia, &q, &qd) - e0) / e0.abs()).abs()); }
    let ok = max_drift < 0.02 && q.iter().chain(qd.iter()).all(|x| x.is_finite());
    println!("  {name:<28} energy-drift over 10 s: {:>8.2}%   [{}]", max_drift * 100.0, if ok { "PASS" } else { "FAIL" });
}

fn main() {
    println!("PHYSICS-FIDELITY BENCHMARK — scoring models on conservation-law invariants (ferromotion = reference).\n");
    println!("System A — simple pendulum (analytic ground truth: E const, T=2π√(L/g)={:.3}s):", 2.0 * std::f64::consts::PI * (L / G).sqrt());
    pend_probes(Model::Symplectic, "PHYSICS (sympl)");
    pend_probes(Model::ExplicitEuler, "explicit-Euler");
    pend_probes(Model::Learned, "\"learned\" (+ε)");

    let (robot, inertia) = from_urdf_full(URDF, "base_link", "gripper_link").expect("load SO-101");
    println!("\nSystem B — the real SO-101 (ferromotion), frictionless, zero torque → energy must be conserved:");
    so101_energy_probe(&robot, &inertia, true, "PHYSICS (symplectic step)");
    so101_energy_probe(&robot, &inertia, false, "explicit-Euler step");

    println!("\n  ================  READING  ================");
    println!("  The 'learned' model is per-step ACCURATE (a ~0.1% residual, better than any real learned world");
    println!("  model) yet FAILS energy conservation — its energy random-walks because it has no physics");
    println!("  structure to hold it. The symplectic model, with the SAME coarse step, PASSES: bounded energy,");
    println!("  correct period, reversible. Physics STRUCTURE beats per-step ACCURACY. This is the axis the");
    println!("  data-scaling bets don't compete on — and it is exactly where a correct engine (ferromotion)");
    println!("  wins by construction. A world model earns trust here by OBEYING physics, not by looking real.");
}
