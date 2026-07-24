//! **Compressible Euler — the hyperbolic regime** (Honest Fluids — stage 7's other half). The MAC,
//! lattice, particle, and spectral solvers are all incompressible or weakly so; this is the
//! conservation-law family where information travels along characteristics and *shocks* form. A 1-D
//! finite-volume solver with the HLL approximate Riemann flux, verified against the gold-standard
//! compressible CFD benchmark — **Sod's shock tube**, whose exact Riemann solution is known in
//! closed form.
//!
//! State is the conserved vector `U = [ρ, ρu, E]`; the flux is `F = [ρu, ρu²+p, u(E+p)]` with the
//! ideal-gas closure `p = (γ−1)(E − ½ρu²)`. Flux-form ⇒ mass/momentum/energy are conserved to
//! machine precision while no wave has reached a boundary.

/// A 1-D compressible Euler field of `n` cells on `[0, L]`, ratio of specific heats `gamma`.
pub struct Euler1d {
    pub n: usize,
    pub h: f64,
    pub gamma: f64,
    pub rho: Vec<f64>,
    pub mom: Vec<f64>,
    pub e: Vec<f64>,
}

impl Euler1d {
    /// Sod's shock tube: a diaphragm at `x = 0.5` separating `(ρ,u,p) = (1,0,1)` on the left from
    /// `(0.125,0,0.1)` on the right, γ = 1.4.
    pub fn sod(n: usize) -> Self {
        let gamma = 1.4;
        let (mut rho, mut mom, mut e) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        for i in 0..n {
            let x = (i as f64 + 0.5) / n as f64;
            let (r, p) = if x < 0.5 { (1.0, 1.0) } else { (0.125, 0.1) };
            rho[i] = r;
            mom[i] = 0.0;
            e[i] = p / (gamma - 1.0); // u = 0 ⇒ E = p/(γ−1)
        }
        Euler1d { n, h: 1.0 / n as f64, gamma, rho, mom, e }
    }

    fn primitive(&self, i: usize) -> (f64, f64, f64) {
        let r = self.rho[i];
        let u = self.mom[i] / r;
        let p = (self.gamma - 1.0) * (self.e[i] - 0.5 * r * u * u);
        (r, u, p)
    }

    fn flux(r: f64, u: f64, p: f64, e: f64) -> [f64; 3] {
        [r * u, r * u * u + p, u * (e + p)]
    }

    /// HLL flux between cells `l` and `r`.
    fn hll(&self, l: usize, r: usize) -> [f64; 3] {
        let g = self.gamma;
        let (rl, ul, pl) = self.primitive(l);
        let (rr, ur, pr) = self.primitive(r);
        let (al, ar) = ((g * pl / rl).sqrt(), (g * pr / rr).sqrt());
        let sl = (ul - al).min(ur - ar);
        let sr = (ul + al).max(ur + ar);
        let fl = Self::flux(rl, ul, pl, self.e[l]);
        let fr = Self::flux(rr, ur, pr, self.e[r]);
        let ul_v = [self.rho[l], self.mom[l], self.e[l]];
        let ur_v = [self.rho[r], self.mom[r], self.e[r]];
        let mut f = [0.0; 3];
        for k in 0..3 {
            f[k] = if sl >= 0.0 {
                fl[k]
            } else if sr <= 0.0 {
                fr[k]
            } else {
                (sr * fl[k] - sl * fr[k] + sl * sr * (ur_v[k] - ul_v[k])) / (sr - sl)
            };
        }
        f
    }

    /// Maximum wave speed `max(|u| + a)` (for the CFL timestep).
    pub fn max_speed(&self) -> f64 {
        let mut s = 0.0f64;
        for i in 0..self.n {
            let (r, u, p) = self.primitive(i);
            s = s.max(u.abs() + (self.gamma * p / r).sqrt());
        }
        s
    }

    /// One explicit finite-volume step (transmissive boundaries).
    #[allow(clippy::needless_range_loop)] // face index addresses cell pairs (f-1, f)
    pub fn step(&mut self, dt: f64) {
        let n = self.n;
        // face fluxes F[0..=n]; transmissive ⇒ ghost = edge cell.
        let mut faces = vec![[0.0; 3]; n + 1];
        for f in 1..n {
            faces[f] = self.hll(f - 1, f);
        }
        faces[0] = self.hll(0, 0);
        faces[n] = self.hll(n - 1, n - 1);
        let c = dt / self.h;
        for i in 0..n {
            self.rho[i] -= c * (faces[i + 1][0] - faces[i][0]);
            self.mom[i] -= c * (faces[i + 1][1] - faces[i][1]);
            self.e[i] -= c * (faces[i + 1][2] - faces[i][2]);
        }
    }

    /// Density / velocity / pressure at cell `i` (for readout and verification).
    pub fn state(&self, i: usize) -> (f64, f64, f64) {
        self.primitive(i)
    }

    /// Totals (conserved while no wave reaches a boundary).
    pub fn totals(&self) -> [f64; 3] {
        [self.rho.iter().sum::<f64>() * self.h, self.mom.iter().sum::<f64>() * self.h, self.e.iter().sum::<f64>() * self.h]
    }
}

#[cfg(test)]
mod verification {
    use super::*;

    /// Run Sod to `t` and return the field.
    fn run_sod(n: usize, t: f64) -> Euler1d {
        let mut f = Euler1d::sod(n);
        let mut time = 0.0;
        while time < t {
            let dt = (0.4 * f.h / f.max_speed()).min(t - time);
            f.step(dt);
            time += dt;
        }
        f
    }

    /// Average primitive state over a spatial window `[x0, x1]`.
    fn window_avg(f: &Euler1d, x0: f64, x1: f64) -> (f64, f64, f64) {
        let (mut r, mut u, mut p, mut c) = (0.0, 0.0, 0.0, 0.0);
        for i in 0..f.n {
            let x = (i as f64 + 0.5) / f.n as f64;
            if x >= x0 && x <= x1 {
                let (ri, ui, pi) = f.state(i);
                r += ri;
                u += ui;
                p += pi;
                c += 1.0;
            }
        }
        (r / c, u / c, p / c)
    }

    /// Sod's shock tube against the exact Riemann solution (Toro, textbook constants at t = 0.2):
    /// the star region has p* ≈ 0.30313, u* ≈ 0.92745, post-shock density ρ*_R ≈ 0.26557, and the
    /// shock sits at x ≈ 0.850. p* and u* are continuous across the contact, so HLL nails them even
    /// though it smears the contact itself.
    #[test]
    fn sod_shock_tube_matches_exact_riemann() {
        let f = run_sod(600, 0.2);

        // Star region between the contact (x≈0.685) and the shock (x≈0.850): p*, u*, ρ*_R.
        let (rho_star, u_star, p_star) = window_avg(&f, 0.70, 0.82);
        eprintln!("Sod star region: rho {rho_star:.4} (exact 0.2657)  u {u_star:.4} (0.9274)  p {p_star:.4} (0.3031)");
        assert!((p_star - 0.30313).abs() / 0.30313 < 0.03, "star pressure off: {p_star}");
        assert!((u_star - 0.92745).abs() / 0.92745 < 0.03, "star velocity off: {u_star}");
        assert!((rho_star - 0.26557).abs() / 0.26557 < 0.05, "post-shock density off: {rho_star}");

        // Undisturbed far states (waves haven't arrived): left (1,0,1), right (0.125,0,0.1).
        let (rl, _, pl) = window_avg(&f, 0.02, 0.12);
        let (rr, _, pr) = window_avg(&f, 0.92, 0.98);
        assert!((rl - 1.0).abs() < 0.01 && (pl - 1.0).abs() < 0.01, "left state disturbed");
        assert!((rr - 0.125).abs() < 0.01 && (pr - 0.1).abs() < 0.01, "right state disturbed");

        // Shock captured: a sharp density drop crossing x ≈ 0.85.
        let (rho_ahead, _, _) = window_avg(&f, 0.87, 0.90);
        assert!(rho_star > 2.0 * rho_ahead, "shock not captured (no density jump)");
    }

    /// Flux-form conservation, read honestly. While the waves stay interior the boundary states are
    /// undisturbed: `u = 0` there, so the mass flux `ρu` and the energy flux `u(E+p)` both vanish —
    /// **mass and energy are conserved to machine precision**. The momentum flux is `ρu²+p = p`,
    /// which does NOT vanish: the boundary pressures push, so total momentum changes by exactly the
    /// pressure impulse `(p_L − p_R)·t`. The solver reproduces both facts.
    #[test]
    fn conserves_mass_and_energy_momentum_tracks_boundary_impulse() {
        let mut f = Euler1d::sod(400);
        let t0 = f.totals();
        let t_end = 0.2;
        let mut time = 0.0;
        while time < t_end {
            let dt = (0.4 * f.h / f.max_speed()).min(t_end - time);
            f.step(dt);
            time += dt;
        }
        let t1 = f.totals();
        let mass_drift = (t1[0] - t0[0]).abs();
        let energy_drift = (t1[2] - t0[2]).abs();
        let mom_change = t1[1] - t0[1];
        let impulse = (1.0 - 0.1) * t_end; // (p_L − p_R)·t
        eprintln!("Sod: mass drift {mass_drift:.2e}, energy drift {energy_drift:.2e}, Δmomentum {mom_change:.4} vs impulse {impulse:.4}");
        assert!(mass_drift < 1e-10, "mass not conserved: {mass_drift}");
        assert!(energy_drift < 1e-10, "energy not conserved: {energy_drift}");
        assert!((mom_change - impulse).abs() < 1e-3, "momentum change ≠ boundary pressure impulse");
    }
}
