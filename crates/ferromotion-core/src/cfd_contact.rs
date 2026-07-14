//! **Contacts-from-distance (CFD)** — informative pre-contact gradients, a clean-room take on the
//! DiffMJX idea (*Differentiable Simulation of Hard Contacts with Soft Gradients*, arXiv 2506.14186).
//!
//! Hard contact makes the impulse `λ = max(0, v − φ/dt)`: zero, with zero gradient, whenever the
//! bodies are separated. A controller that hasn't *reached* contact yet therefore gets no gradient
//! signal telling it to close the gap — optimization stalls. CFD keeps the **forward** pass exactly
//! hard (physically correct — no phantom forces at a distance), but in the **backward** pass uses a
//! penalty that activates from a reach distance `d`, via a straight-through estimator: the impulse's
//! surrogate `softplus_ε(v·dt − φ + d)/dt` has a sigmoid derivative that is non-zero while the gap is
//! within `d` of contact, and → the true hard gradient once deep in contact. This is what lets a
//! gradient escape the "not touching yet" plateau. Pure Rust → WASM-clean.

/// Parameters of the CFD normal-contact model.
#[derive(Clone, Copy, Debug)]
pub struct CfdContact {
    pub dt: f64,
    /// Reach distance: the backward pass produces gradient while the gap is within this of contact.
    pub reach: f64,
    /// Backward-pass smoothing (softplus temperature).
    pub eps: f64,
}

/// One step of a 1-DoF mass falling toward a floor under gravity `g`, resolving an inelastic normal
/// contact. Carries forward-mode sensitivities `dv/dθ, dφ/dθ` for both the **CFD** backward pass and
/// the **naive** (true hard) one, so their difference is observable. Returns the contact impulse and
/// its two derivative contributions.
fn contact_step(c: &CfdContact, g: f64, v: &mut f64, phi: &mut f64, dv_c: &mut f64, dphi_c: &mut f64, dv_n: &mut f64, dphi_n: &mut f64) -> (f64, f64, f64) {
    // Gravity (constant → no θ-sensitivity).
    *v += g * c.dt;

    // Forward: exact hard impulse (zero when separated).
    let s = *v - *phi / c.dt;
    let lam = s.max(0.0);

    // CFD backward: sigmoid of the reach-shifted penetration → gradient reaches across the gap.
    let arg = (*v * c.dt - *phi + c.reach) / c.eps;
    let sig = 1.0 / (1.0 + (-arg).exp());
    let dlam_c = sig * *dv_c - (sig / c.dt) * *dphi_c;
    // Naive backward: the true hard indicator (0 when separated → vanishing).
    let ind = if s > 0.0 { 1.0 } else { 0.0 };
    let dlam_n = ind * *dv_n - (ind / c.dt) * *dphi_n;

    // Post-contact velocity v⁺ = v − λ, then advance the gap (φ⁺ = φ − v⁺·dt ≥ 0 by construction).
    let vplus = *v - lam;
    let dvplus_c = *dv_c - dlam_c;
    let dvplus_n = *dv_n - dlam_n;
    *phi -= vplus * c.dt;
    *dphi_c -= dvplus_c * c.dt;
    *dphi_n -= dvplus_n * c.dt;
    *v = vplus;
    *dv_c = dvplus_c;
    *dv_n = dvplus_n;

    (lam, dlam_c, dlam_n)
}

/// Roll out `n` steps from initial downward velocity `theta` and gap `y0`, accumulating the total
/// contact impulse `J` and its derivative `dJ/dθ` under both the **CFD** and **naive** backward
/// passes. `J` itself is always the exact hard-contact value.
pub fn rollout_impulse(c: &CfdContact, g: f64, theta: f64, y0: f64, n: usize) -> (f64, f64, f64) {
    let (mut v, mut phi) = (theta, y0);
    let (mut dv_c, mut dphi_c, mut dv_n, mut dphi_n) = (1.0, 0.0, 1.0, 0.0);
    let (mut j, mut dj_c, mut dj_n) = (0.0, 0.0, 0.0);
    for _ in 0..n {
        let (lam, dl_c, dl_n) = contact_step(c, g, &mut v, &mut phi, &mut dv_c, &mut dphi_c, &mut dv_n, &mut dphi_n);
        j += lam;
        dj_c += dl_c;
        dj_n += dl_n;
    }
    (j, dj_c, dj_n)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model() -> (CfdContact, f64, f64, usize) {
        // Floor 1 m below; over 40 steps of free fall the ball descends ≈0.78 m → it does NOT reach
        // the floor from rest: separated the whole rollout (the stall regime).
        (CfdContact { dt: 0.01, reach: 0.6, eps: 0.05 }, 9.81, 1.0, 40)
    }

    #[test]
    fn forward_pass_is_exact_hard_contact() {
        let (c, g, y0, n) = model();
        // From rest it never reaches the floor → zero impulse, exactly (no phantom force at distance).
        let (j0, ..) = rollout_impulse(&c, g, 0.0, y0, n);
        assert_eq!(j0, 0.0, "phantom contact force at a distance: J = {j0}");
        // With a hard downward launch it does reach the floor → positive impulse.
        let (j1, ..) = rollout_impulse(&c, g, 4.0, y0, n);
        assert!(j1 > 0.0, "should make contact when launched hard: J = {j1}");
    }

    #[test]
    fn naive_gradient_vanishes_but_cfd_reaches() {
        let (c, g, y0, n) = model();
        let (j, dj_cfd, dj_naive) = rollout_impulse(&c, g, 0.0, y0, n);
        assert_eq!(j, 0.0);
        // Naive gradient is exactly zero (never in contact) — the stall.
        assert!(dj_naive.abs() < 1e-12, "naive gradient should vanish: {dj_naive}");
        // CFD gradient is positive and substantial — it "sees" the approaching floor.
        assert!(dj_cfd > 1e-2, "CFD gradient should reach across the gap: {dj_cfd}");
    }

    #[test]
    fn cfd_gradient_escapes_the_stall() {
        let (c, g, y0, n) = model();
        let target = 3.0;
        // Naive gradient ascent from rest stays stuck (gradient is zero).
        let mut theta_n = 0.0;
        for _ in 0..50 {
            let (_, _, dj_n) = rollout_impulse(&c, g, theta_n, y0, n);
            theta_n += 0.5 * dj_n;
        }
        let (j_naive, ..) = rollout_impulse(&c, g, theta_n, y0, n);
        assert!(j_naive < 1e-9, "naive optimizer should stay stalled: J = {j_naive}");

        // CFD gradient ascent climbs out of the plateau and reaches the target impulse.
        let mut theta_c = 0.0;
        for _ in 0..200 {
            let (j, dj_c, _) = rollout_impulse(&c, g, theta_c, y0, n);
            if j >= target {
                break;
            }
            theta_c += 0.2 * dj_c;
        }
        let (j_cfd, ..) = rollout_impulse(&c, g, theta_c, y0, n);
        assert!(j_cfd >= target, "CFD optimizer should reach the target: J = {j_cfd} (θ={theta_c:.3})");
    }

    #[test]
    fn cfd_gradient_has_the_correct_sign_and_converges_to_truth_in_contact() {
        let (c, g, y0, n) = model();
        // Deep-contact regime: launched hard, in contact for most of the rollout.
        let theta = 6.0;
        let (j, dj_cfd, dj_naive) = rollout_impulse(&c, g, theta, y0, n);
        // A finite-difference check that dJ/dθ > 0 (more launch speed ⇒ more accumulated impulse).
        let eps = 1e-5;
        let (jp, ..) = rollout_impulse(&c, g, theta + eps, y0, n);
        let fd = (jp - j) / eps;
        assert!(fd > 0.0 && dj_cfd > 0.0, "sign wrong: fd={fd}, cfd={dj_cfd}");
        // Deep in contact the sigmoid saturates → CFD and naive gradients agree.
        assert!((dj_cfd - dj_naive).abs() / dj_naive.abs().max(1e-9) < 0.05, "CFD ≠ hard gradient in deep contact: cfd={dj_cfd}, naive={dj_naive}");
    }
}
