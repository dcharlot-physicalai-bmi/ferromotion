//! The scoring core, shared by every submission. The benchmark system here is the frictionless pendulum
//! (θ̈ = −(g/L)sinθ, L=1): total energy is exactly constant and the flow is exactly time-reversible, so the
//! invariants have a closed-form ground truth. A submission is any type implementing `Model` (advance the
//! state one step). The harness — never the submission — computes the score, so standings cannot be gamed.
use serde::Serialize;

pub const G: f64 = 9.81;
pub fn energy(th: f64, w: f64) -> f64 { 0.5 * w * w + G * (1.0 - th.cos()) }
pub fn accel(th: f64) -> f64 { -G * th.sin() }

/// High-accuracy reference step (RK4 with tiny substeps) for the one-step accuracy score.
pub fn truth_step(th: f64, w: f64, dt: f64) -> (f64, f64) {
    let (mut t, mut v) = (th, w); let h = dt / 50.0;
    for _ in 0..50 {
        let (k1t, k1v) = (v, accel(t));
        let (k2t, k2v) = (v + 0.5 * h * k1v, accel(t + 0.5 * h * k1t));
        let (k3t, k3v) = (v + 0.5 * h * k2v, accel(t + 0.5 * h * k2t));
        let (k4t, k4v) = (v + h * k3v, accel(t + h * k3t));
        t += h / 6.0 * (k1t + 2.0 * k2t + 2.0 * k3t + k4t);
        v += h / 6.0 * (k1v + 2.0 * k2v + 2.0 * k3v + k4v);
    }
    (t, v)
}

/// Metadata a submission declares.
pub struct Meta { pub name: &'static str, pub author: &'static str, pub kind: &'static str }

/// A submission: advance the pendulum state (θ, ω) forward by `dt`.
pub trait Model { fn step(&self, th: f64, w: f64, dt: f64) -> (f64, f64); }

#[derive(Serialize, Clone)]
pub struct Report {
    pub rank: usize,
    pub name: String,
    pub author: String,
    pub kind: String,
    pub energy_drift: f64,     // max relative energy drift over the rollout (0 = perfect)
    pub reversibility: f64,    // forward-then-reverse return error (0 = perfect)
    pub one_step_rmse: f64,    // per-step accuracy vs the RK4 reference
    pub pass: bool,            // energy_drift < 5% AND reversibility < 5%
}

/// Score a model on the conservation invariants. Deterministic.
pub fn score(meta: &Meta, m: &dyn Model) -> Report {
    let dt = 0.02;
    // energy conservation over an 8 s rollout from a large-amplitude release
    let (th0, w0) = (2.0, 0.0); let e0 = energy(th0, w0);
    let (mut th, mut w) = (th0, w0); let mut edrift = 0.0f64;
    for _ in 0..400 {
        let (a, b) = m.step(th, w, dt); th = a; w = b;
        if !th.is_finite() || !w.is_finite() { edrift = f64::INFINITY; break; }
        edrift = edrift.max(((energy(th, w) - e0) / e0).abs());
    }
    // time-reversibility: forward 200, then 200 with reversed time — should return to start
    let (mut th, mut w) = (th0, w0);
    for _ in 0..200 { let (a, b) = m.step(th, w, dt); th = a; w = b; }
    for _ in 0..200 { let (a, b) = m.step(th, w, -dt); th = a; w = b; }
    let rev = if th.is_finite() && w.is_finite() { ((th - th0).powi(2) + (w - w0).powi(2)).sqrt() } else { f64::INFINITY };
    // one-step accuracy vs the RK4 reference over sampled states
    let mut acc = 0.0f64; let n = 400u32;
    for k in 0..n {
        let th = -3.0 + 6.0 * (k as f64 / n as f64);
        let w = -4.0 + 8.0 * (((k * 7) % n) as f64 / n as f64);
        let (pt, pw) = m.step(th, w, dt); let (tt, tw) = truth_step(th, w, dt);
        acc += (pt - tt).powi(2) + (pw - tw).powi(2);
    }
    acc = (acc / n as f64).sqrt();
    let pass = edrift < 0.05 && rev < 0.05 && edrift.is_finite();
    Report { rank: 0, name: meta.name.into(), author: meta.author.into(), kind: meta.kind.into(),
        energy_drift: edrift, reversibility: rev, one_step_rmse: acc, pass }
}
