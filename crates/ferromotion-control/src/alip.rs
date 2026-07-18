//! **ALIP + H-LIP** — angular-momentum templates for dynamic walking. The **Angular-momentum Linear
//! Inverted Pendulum** (Gong & Grizzle, 2020) rewrites the LIP in the state `(x, L)` — CoM position
//! relative to the stance foot, and **angular momentum about the contact point** — instead of `(x, ẋ)`.
//! The win is practical: `L` about the contact does *not* jump at foot impact and is far less noisy than
//! CoM velocity on a robot with heavy legs, so one-step-ahead foot-placement prediction is much more
//! accurate. The continuous dynamics are linear,
//!
//! ```text
//!   d/dt [x; L] = [[0, 1/(mH)], [mgH·? ...]] …  →  ẋ = L/(mH),  L̇ = mg·x
//! ```
//!
//! and the **step-to-step (S2S)** map (integrate over a step, then place the next foot) is an LTI system.
//! **H-LIP** (Xiong & Ames, T-RO 2022) closes the loop on that S2S map with a **deadbeat** foot-placement
//! gain that drives the walker onto its periodic orbit in a finite number of steps — the modern recipe for
//! underactuated / point-foot walking. Verified against the closed-form pendulum divergence rate `ω=√(g/H)`
//! and the deadbeat spectral radius (0). Pure `nalgebra` → WASM-clean.

use nalgebra::{Matrix2, Vector2};

/// An ALIP walker: point mass `m` at constant CoM height `H`, gravity `g`. State is `[x, L]` with `x` the
/// CoM position relative to the stance foot and `L` the angular momentum about the contact point.
#[derive(Clone, Copy, Debug)]
pub struct Alip {
    pub mass: f64,
    pub height: f64,
    pub g: f64,
    /// Step duration.
    pub t_step: f64,
}

impl Alip {
    /// The natural frequency `ω = √(g/H)` (the LIP eigen-rate).
    pub fn omega(&self) -> f64 {
        (self.g / self.height).sqrt()
    }

    /// The **within-step** state-transition matrix `M = exp(A·T)` (closed form via `cosh/sinh`), for
    /// `A = [[0, 1/(mH)], [mg, 0]]` — the LIP flow in `(x, L)` coordinates.
    pub fn s2s_matrix(&self) -> Matrix2<f64> {
        let (m, h) = (self.mass, self.height);
        let (a, b) = (1.0 / (m * h), m * self.g); // ẋ = a·L, L̇ = b·x
        let w = (a * b).sqrt(); // = ω
        let t = self.t_step;
        let (ch, sh) = (( w * t).cosh(), (w * t).sinh());
        Matrix2::new(ch, a / w * sh, b / w * sh, ch)
    }

    /// One step of the S2S dynamics: flow over a step, then place the next foot a distance `u` ahead of the
    /// CoM (so `x⁺ = x(T) − u`, while `L` is continuous across impact). Returns the post-impact `[x, L]`.
    pub fn step(&self, sigma: &Vector2<f64>, u: f64) -> Vector2<f64> {
        let end = self.s2s_matrix() * sigma;
        Vector2::new(end.x - u, end.y)
    }

    /// The **H-LIP deadbeat** foot-placement gain `K` (`u = K·σ`) that places both closed-loop S2S poles at
    /// the origin — the walker reaches its orbit in ≤2 steps. Derived by direct pole placement on
    /// `M − [1;0]·K`.
    pub fn deadbeat_gain(&self) -> Vector2<f64> {
        let m = self.s2s_matrix();
        // char. poly of M − [1;0]K = z² − (M11−K1+M22)z + [(M11−K1)M22 − (M12−K2)M21]; set both to 0.
        let k1 = m[(0, 0)] + m[(1, 1)]; // trace ⇒ M11−K1 = −M22
        let k2 = m[(0, 1)] + m[(1, 1)] * m[(1, 1)] / m[(1, 0)];
        Vector2::new(k1, k2)
    }

    /// The closed-loop S2S matrix under a foot-placement gain `K` (`u = K·σ`): `M − [1;0]·Kᵀ`.
    pub fn closed_loop(&self, k: &Vector2<f64>) -> Matrix2<f64> {
        let mut m = self.s2s_matrix();
        m[(0, 0)] -= k.x;
        m[(0, 1)] -= k.y;
        m
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn walker() -> Alip {
        Alip { mass: 30.0, height: 0.8, g: 9.81, t_step: 0.3 }
    }

    #[test]
    fn the_uncontrolled_pendulum_diverges_at_the_natural_rate() {
        // With no stepping (u=0, fixed foot), the LIP falls: the unstable eigenvalue of the S2S map is
        // e^{ωT}. Verify the larger eigenvalue matches the closed-form pendulum rate.
        let a = walker();
        let m = a.s2s_matrix();
        let eig = m.eigenvalues().unwrap();
        let lambda_max = eig.iter().cloned().fold(0.0_f64, f64::max);
        let expect = (a.omega() * a.t_step).exp();
        assert!((lambda_max - expect).abs() < 1e-9, "divergence rate {lambda_max} vs e^{{ωT}} = {expect}");
        // symplectic-ish: eigenvalues are reciprocal (e^{ωT}, e^{−ωT}) ⇒ det = 1
        assert!((m.determinant() - 1.0).abs() < 1e-9, "LIP flow should be area-preserving: det {}", m.determinant());
    }

    #[test]
    fn the_deadbeat_gain_makes_the_s2s_dynamics_nilpotent() {
        // THE INVARIANT. Under the H-LIP deadbeat gain both closed-loop S2S poles are at 0. For a 2×2 that
        // is exactly nilpotency: by Cayley–Hamilton (trace = det = 0) the closed-loop matrix squares to 0.
        let a = walker();
        let k = a.deadbeat_gain();
        let cl = a.closed_loop(&k);
        assert!(cl.trace().abs() < 1e-9, "closed-loop trace (Σ poles) should be 0: {}", cl.trace());
        assert!(cl.determinant().abs() < 1e-9, "closed-loop det (∏ poles) should be 0: {}", cl.determinant());
        assert!((cl * cl).norm() < 1e-9, "deadbeat ⇒ closed-loop matrix is nilpotent (M² = 0)");
    }

    #[test]
    fn deadbeat_stepping_recovers_balance_in_two_steps() {
        // THE HEADLINE. From a leaning initial state, deadbeat foot placement drives the walker onto its
        // orbit (here: rest, σ=0) in at most 2 steps (the state dimension).
        let a = walker();
        let k = a.deadbeat_gain();
        let mut sigma = Vector2::new(0.08, 4.0); // leaning forward, with forward angular momentum
        for _ in 0..2 {
            let u = k.dot(&sigma);
            sigma = a.step(&sigma, u);
        }
        assert!(sigma.norm() < 1e-6, "deadbeat should reach the orbit in 2 steps: {sigma:?}");
    }

    #[test]
    fn angular_momentum_is_continuous_across_the_step() {
        // The ALIP selling point: L about the contact does not jump at foot placement (only x does).
        let a = walker();
        let sigma = Vector2::new(0.05, 3.0);
        let end = a.s2s_matrix() * sigma;
        let post = a.step(&sigma, 0.12);
        assert!((post.y - end.y).abs() < 1e-12, "L jumped at impact: {} vs {}", post.y, end.y);
        assert!((post.x - (end.x - 0.12)).abs() < 1e-12, "x should shift by the footstep");
    }
}
