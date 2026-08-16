//! **Geometric phase of a rolling sphere** — reorienting a fingertip by driving it in a closed loop.
//!
//! Murray, Li & Sastry (1994), *A Mathematical Introduction to Robotic Manipulation*, §8.4.3 and Theorem 8.6
//! (Gauss-Bonnet). This is the payoff of the rolling-contact kinematics in [`crate::rolling_contact`]: roll a
//! spherical finger around a **closed** path on its own surface, return the contact to exactly where it
//! started on both bodies, and the *contact angle* `ψ` has nonetheless changed. Net reorientation from a
//! net-zero motion, which is what makes dexterous in-hand repositioning possible at all.
//!
//! For a sphere of radius `ρ` on a plane, MLS gives `M_o = I`, `T_o = 0`, and
//! `M_f = diag(ρ, ρ·cos u_f)`, `T_f = [0, −(1/ρ)·tan u_f]`. Substituting into `ψ̇ = T_f M_f α̇_f` collapses to
//!
//! ```text
//! ψ̇ = −sin(u_f) · v̇_f
//! ```
//!
//! **which contains no `ρ`** — the phase is independent of the sphere's radius, so a result proved for the
//! unit sphere holds for every sphere. `M_f = diag(ρ, ρ cos u_f)` identifies the chart as *latitude*: `u_f`
//! measured from the equator, area element `cos u du dv`.
//!
//! # Gauss-Bonnet, and the branch it is stated up to
//!
//! MLS Theorem 8.6: for a closed path enclosing a cap `Ω`, `Δψ = −Area(Ω)`, the area measured on the unit
//! sphere. Taken literally that is off by `2π` for the latitude family, and the reason is worth being precise
//! about rather than smoothing over.
//!
//! Integrating exactly around the latitude circle `u_f = u₀`, `v_f: 0 → 2π`, gives `Δψ = −2π·sin u₀`. The cap
//! from `u₀` to the pole has area `A = 2π(1 − sin u₀)`, so **`Δψ = A − 2π`**. The two caps sum to `4π`, so
//! `−Area(far cap) = A − 4π ≡ A (mod 2π)`. Both readings therefore agree **modulo 2π**, and which one is
//! literal depends on which cap "enclosed" designates and on the traversal direction. Since `ψ` is an angle,
//! congruence mod `2π` is the only statement that can be meaningful — but a caller comparing a raw `Δψ`
//! against a raw area will see a `2π` discrepancy and think something is broken, so [`geometric_phase`]
//! returns the *unwrapped* integral and [`cap_area_from_latitude`] gives the area, with the relation tested
//! explicitly.

use std::f64::consts::PI;

/// `ψ̇ = −sin(u_f)·v̇_f` — MLS eq. (8.36), for a sphere rolling on a plane. No `ρ` appears.
pub fn psi_dot(u_f: f64, v_f_dot: f64) -> f64 {
    -u_f.sin() * v_f_dot
}

/// The contact-angle change accumulated by rolling a spherical finger along a path on its own surface.
///
/// `path` is sampled `(u_f, v_f)` in the latitude chart. Integrated with the trapezoid rule on
/// `ψ̇ dt = −sin(u) dv`, which is exact for the piecewise-linear interpolant of `sin u` between samples and so
/// converges as the sampling refines.
///
/// The result is **unwrapped**: it accumulates past `±2π` rather than folding, because the winding is the
/// physically meaningful part — a path traversed twice reorients twice as far.
pub fn geometric_phase(path: &[(f64, f64)]) -> f64 {
    path.windows(2)
        .map(|w| {
            let ((u0, v0), (u1, v1)) = (w[0], w[1]);
            // ∫ −sin u dv over the segment, trapezoid in sin u
            -0.5 * (u0.sin() + u1.sin()) * (v1 - v0)
        })
        .sum()
}

/// Area on the **unit** sphere of the cap from latitude `u₀` up to the pole: `2π(1 − sin u₀)`.
///
/// The latitude chart's area element is `cos u du dv`, so this is `∫₀^{2π} ∫_{u₀}^{π/2} cos u du dv`.
pub fn cap_area_from_latitude(u0: f64) -> f64 {
    2.0 * PI * (1.0 - u0.sin())
}

/// Lift a path on the finger to the full contact trajectory `η = (α_f, α_o, ψ)` — MLS eq. (8.35).
///
/// For a sphere on a plane the rolling constraint determines `α_o` and `ψ` uniquely from `α_f`, which is the
/// "well defined lifting map" MLS describes. Returns the `(α_o, ψ)` history alongside the given `α_f`.
///
/// `α̇_o = M_o⁻¹ R_ψ M_f α̇_f` with `M_o = I`, so the plane-side contact traces the finger path scaled by the
/// finger metric and turned by the contact angle — the reason a closed loop on the sphere generally leaves an
/// **open** loop on the plane, and vice versa.
pub fn lift_path(path: &[(f64, f64)], rho: f64, psi0: f64) -> Vec<((f64, f64), f64)> {
    let mut out = Vec::with_capacity(path.len());
    let (mut ox, mut oy, mut psi) = (0.0f64, 0.0f64, psi0);
    out.push(((ox, oy), psi));
    for w in path.windows(2) {
        let ((u0, v0), (u1, v1)) = (w[0], w[1]);
        let (du, dv) = (u1 - u0, v1 - v0);
        let um = 0.5 * (u0 + u1);
        // M_f α̇_f for the latitude chart, at the segment midpoint
        let (mu, mv) = (rho * du, rho * um.cos() * dv);
        // R_ψ = [[cos ψ, −sin ψ], [−sin ψ, −cos ψ]] — a reflection, per rolling_contact::r_psi
        let (c, s) = (psi.cos(), psi.sin());
        ox += c * mu - s * mv;
        oy += -s * mu - c * mv;
        psi += -um.sin() * dv;
        out.push(((ox, oy), psi));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sample a latitude circle at `u0`, `v: 0 → 2π·turns`.
    fn latitude_circle(u0: f64, n: usize, turns: f64) -> Vec<(f64, f64)> {
        (0..=n).map(|i| (u0, 2.0 * PI * turns * i as f64 / n as f64)).collect()
    }

    #[test]
    fn a_closed_loop_reorients_the_contact_by_the_enclosed_area_modulo_two_pi() {
        // THE Gauss-Bonnet check, stated in the form that is actually true: Δψ = A − 2π for the latitude
        // family, i.e. Δψ ≡ A (mod 2π). Taking MLS's "Δψ = −Area" literally is off by 2π here, because the
        // two caps sum to 4π so −Area(far) ≡ +Area(near); which cap "enclosed" means fixes the branch.
        for u0 in [0.2, 0.6, 1.0, 1.3] {
            let dpsi = geometric_phase(&latitude_circle(u0, 20_000, 1.0));
            let exact = -2.0 * PI * u0.sin(); // closed form of the line integral
            assert!((dpsi - exact).abs() < 1e-6, "u0={u0}: integrated {dpsi}, exact {exact}");

            let area = cap_area_from_latitude(u0);
            assert!(
                (dpsi - (area - 2.0 * PI)).abs() < 1e-6,
                "u0={u0}: Δψ={dpsi} should equal A − 2π = {}",
                area - 2.0 * PI
            );
            // and the congruence, which is the only frame-free statement for an angle
            let wrap = |x: f64| x.rem_euclid(2.0 * PI);
            assert!((wrap(dpsi) - wrap(area)).abs() < 1e-6, "Δψ ≡ A (mod 2π) must hold");
        }
    }

    #[test]
    fn the_phase_is_independent_of_the_sphere_radius() {
        // MLS notes eq. (8.36) contains no ρ, so a result for the unit sphere holds for any sphere. That is a
        // structural claim about the equation, so test it on the lifted trajectory too, where ρ DOES appear.
        let path = latitude_circle(0.7, 4000, 1.0);
        let base = geometric_phase(&path);
        for rho in [0.005, 0.05, 1.0, 20.0] {
            let lifted = lift_path(&path, rho, 0.0);
            let dpsi = lifted.last().unwrap().1 - lifted[0].1;
            assert!((dpsi - base).abs() < 1e-9, "rho={rho}: phase moved to {dpsi} from {base}");
        }
        // The plane-side path, by contrast, scales linearly with rho — so the invariance above is not
        // vacuous, it is specific to psi.
        let span = |rho: f64| {
            let l = lift_path(&path, rho, 0.0);
            l.iter().map(|((x, y), _)| (x * x + y * y).sqrt()).fold(0.0f64, f64::max)
        };
        let s1 = span(1.0);
        assert!((span(3.0) - 3.0 * s1).abs() < 1e-9 * s1.max(1.0), "the plane-side path must scale with rho");
    }

    #[test]
    fn winding_accumulates_rather_than_folding() {
        // Traversing the same loop twice reorients twice as far. Folding into (−π, π] would destroy that, and
        // the winding is the physically useful part: it is how a finger reorients past one turn.
        let u0 = 0.5;
        let one = geometric_phase(&latitude_circle(u0, 20_000, 1.0));
        let two = geometric_phase(&latitude_circle(u0, 40_000, 2.0));
        assert!((two - 2.0 * one).abs() < 1e-6, "two turns gave {two}, expected 2 × {one}");
        // and reversing the traversal reverses the phase
        let mut rev = latitude_circle(u0, 20_000, 1.0);
        rev.reverse();
        assert!((geometric_phase(&rev) + one).abs() < 1e-6, "reversal must negate the phase");
    }

    #[test]
    fn a_path_enclosing_no_area_produces_no_phase() {
        // Out and back along the same arc encloses nothing, so there is no net reorientation — the degenerate
        // case that separates "geometric phase" from "just integrating something".
        let out: Vec<(f64, f64)> = (0..=500).map(|i| (0.3 + 0.4 * i as f64 / 500.0, 0.8)).collect();
        let mut there_and_back = out.clone();
        let mut back = out;
        back.reverse();
        there_and_back.extend(back.into_iter().skip(1));
        assert!(
            geometric_phase(&there_and_back).abs() < 1e-12,
            "a retraced path encloses no area: {}",
            geometric_phase(&there_and_back)
        );
        // A pure meridian move (dv = 0 throughout) also produces none, since ψ̇ ∝ v̇.
        let meridian: Vec<(f64, f64)> = (0..=500).map(|i| (0.1 + i as f64 / 500.0, 1.1)).collect();
        assert!(geometric_phase(&meridian).abs() < 1e-12, "psi depends only on dv");
        // Degenerate inputs are zero rather than a panic.
        assert_eq!(geometric_phase(&[]), 0.0);
        assert_eq!(geometric_phase(&[(0.3, 0.4)]), 0.0);
    }

    #[test]
    fn rolling_at_the_equator_gives_no_phase_and_at_the_pole_the_most() {
        // sin(u) weights the integrand, so a loop at the equator (u = 0) accumulates nothing however far it
        // travels, while a tight loop near the pole approaches a full −2π. That is the physical content:
        // reorientation comes from latitude, not from distance rolled.
        assert!(geometric_phase(&latitude_circle(0.0, 8000, 1.0)).abs() < 1e-9, "equator gives no phase");
        let near_pole = geometric_phase(&latitude_circle(PI / 2.0 - 1e-4, 8000, 1.0));
        assert!(
            (near_pole + 2.0 * PI).abs() < 1e-3,
            "a loop at the pole should approach −2π, got {near_pole}"
        );
        // monotone in latitude between those extremes
        let mut prev = 0.0;
        for k in 1..=8 {
            let p = geometric_phase(&latitude_circle(k as f64 * PI / 18.0, 4000, 1.0));
            assert!(p < prev, "phase magnitude should grow with latitude: {p} after {prev}");
            prev = p;
        }
    }
}
