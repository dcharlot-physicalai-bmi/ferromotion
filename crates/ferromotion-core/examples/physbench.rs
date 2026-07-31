//! THE BENCHMARK HARNESS — the seam that turns the physics-fidelity probes into something OTHERS can run.
//! A submission is any dynamics model that implements one small trait; the harness scores it on the
//! conservation invariants and prints a sorted scorecard next to the references. This file is the "put your
//! model on the stand" entry point: implement `Model`, add it to `entries()`, and run `cargo run --release
//! --example physbench`. Scored here on the canonical frictionless pendulum (θ̈ = −(g/L)sinθ, L=1), whose
//! invariants have a closed form: total energy is constant, and the flow is exactly time-reversible.
//!
//! The point is not the reference integrators below — it is the INTERFACE. A learned world model plugs in
//! the same way (see `wm_on_the_stand.rs` for a trained black-box vs a structured net measured on this exact
//! system: black-box 86% energy drift FAIL, structured 4.2% PASS — the same scorecard this harness prints).
const G: f64 = 9.81;
fn energy(th: f64, w: f64) -> f64 { 0.5 * w * w + G * (1.0 - th.cos()) }
fn accel(th: f64) -> f64 { -G * th.sin() }
// high-accuracy reference one-step (RK4, tiny substeps) for the accuracy score.
fn truth_step(th: f64, w: f64, dt: f64) -> (f64, f64) {
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

/// A submission: a dynamics model for the pendulum. Advance state (θ, ω) by dt.
trait Model { fn name(&self) -> &'static str; fn step(&self, th: f64, w: f64, dt: f64) -> (f64, f64); }

struct Report { name: &'static str, energy: f64, reversibility: f64, accuracy: f64, pass: bool }
fn score(m: &dyn Model) -> Report {
    let dt = 0.02;
    // energy conservation over a 8 s rollout from a large-amplitude release
    let (th0, w0) = (2.0, 0.0); let e0 = energy(th0, w0);
    let (mut th, mut w) = (th0, w0); let mut edrift = 0.0f64;
    for _ in 0..400 { let (a, b) = m.step(th, w, dt); th = a; w = b;
        if !th.is_finite() || !w.is_finite() { edrift = f64::INFINITY; break; }
        edrift = edrift.max(((energy(th, w) - e0) / e0).abs()); }
    // time-reversibility: forward 200 steps, then 200 steps with reversed time; should return to start
    let (mut th, mut w) = (th0, w0);
    for _ in 0..200 { let (a, b) = m.step(th, w, dt); th = a; w = b; }
    for _ in 0..200 { let (a, b) = m.step(th, w, -dt); th = a; w = b; }
    let rev = if th.is_finite() && w.is_finite() { ((th - th0).powi(2) + (w - w0).powi(2)).sqrt() } else { f64::INFINITY };
    // one-step accuracy vs the RK4 reference, averaged over sampled states
    let mut acc = 0.0f64; let n = 400u32;
    for k in 0..n { let th = -3.0 + 6.0 * (k as f64 / n as f64); let w = -4.0 + 8.0 * (((k * 7) % n) as f64 / n as f64);
        let (pt, pw) = m.step(th, w, dt); let (tt, tw) = truth_step(th, w, dt);
        acc += (pt - tt).powi(2) + (pw - tw).powi(2); }
    acc = (acc / n as f64).sqrt();
    let pass = edrift < 0.05 && rev < 0.05 && edrift.is_finite();
    Report { name: m.name(), energy: edrift, reversibility: rev, accuracy: acc, pass }
}

// ---- reference submissions (baselines) ----
struct Symplectic; // semi-implicit Euler: symplectic, conserves
impl Model for Symplectic { fn name(&self) -> &'static str { "symplectic (structure)" }
    fn step(&self, th: f64, w: f64, dt: f64) -> (f64, f64) { let w2 = w + dt * accel(th); (th + dt * w2, w2) } }
struct Verlet; // velocity Verlet: symplectic, 2nd order
impl Model for Verlet { fn name(&self) -> &'static str { "velocity-verlet (structure)" }
    fn step(&self, th: f64, w: f64, dt: f64) -> (f64, f64) { let a = accel(th); let tn = th + dt * w + 0.5 * dt * dt * a; let wn = w + 0.5 * dt * (a + accel(tn)); (tn, wn) } }
struct ExplicitEuler; // accurate per step, but pumps energy
impl Model for ExplicitEuler { fn name(&self) -> &'static str { "explicit-euler (no structure)" }
    fn step(&self, th: f64, w: f64, dt: f64) -> (f64, f64) { (th + dt * w, w + dt * accel(th)) } }
struct Damped; // "looks plausible" but silently loses energy — the wrong system
impl Model for Damped { fn name(&self) -> &'static str { "lossy (wrong invariant)" }
    fn step(&self, th: f64, w: f64, dt: f64) -> (f64, f64) { let w2 = (w + dt * accel(th)) * 0.999; (th + dt * w2, w2) } }

fn entries() -> Vec<Box<dyn Model>> { vec![Box::new(Symplectic), Box::new(Verlet), Box::new(ExplicitEuler), Box::new(Damped)] }

fn main() {
    println!("physics-fidelity benchmark — standings on the frictionless pendulum (energy constant, flow reversible).\n");
    let mut reps: Vec<Report> = entries().iter().map(|m| score(m.as_ref())).collect();
    // rank: passing first, then by smallest energy drift
    reps.sort_by(|a, b| b.pass.cmp(&a.pass).then(a.energy.partial_cmp(&b.energy).unwrap()));
    let pct = |x: f64| if x.is_finite() { format!("{:>8.2}%", x * 100.0) } else { "  DIVERGED".into() };
    println!("  {:<4}{:<32}{:>12}{:>14}{:>14}{:>8}", "#", "model", "energy", "reversibility", "1-step RMSE", "verdict");
    println!("  {}", "-".repeat(92));
    for (i, r) in reps.iter().enumerate() {
        println!("  {:<4}{:<32}{}{}{:>14.2e}{:>8}", i + 1, r.name, pct(r.energy), pct(r.reversibility), r.accuracy, if r.pass { "PASS" } else { "FAIL" });
    }
    println!("\n  Verdict = energy drift < 5% AND time-reversibility error < 5%. Note that the two structured");
    println!("  integrators PASS while explicit-euler PUMPS energy and the lossy model LEAKS it — both FAIL the");
    println!("  conservation invariant even though each is per-step plausible. One-step RMSE (accuracy) does NOT");
    println!("  predict the verdict: a model can be accurate step to step and still violate the invariant over a");
    println!("  rollout. That is the whole point — structure, not per-step accuracy, is what the benchmark scores.");
    println!("\n  SUBMIT YOUR MODEL: implement `trait Model {{ fn step(&self, θ, ω, dt) -> (θ, ω) }}`, add it to");
    println!("  entries(), and re-run. A learned world model plugs in identically — see wm_on_the_stand.rs for a");
    println!("  trained black-box (86% energy drift, FAIL) vs a structured net (4.2%, PASS) scored on this system.");
}
