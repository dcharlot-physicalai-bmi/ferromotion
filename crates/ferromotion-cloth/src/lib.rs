//! ferromotion-cloth — a **differentiable FEM thin-shell cloth** solver (StVK membrane + bending),
//! the third material domain alongside the Aquarium fluid and MPM crates (in the spirit of
//! Diffclothai / DaXBench).
//!
//! The cloth is a triangle mesh. Each triangle is a **constant-strain membrane element** with a
//! St. Venant–Kirchhoff material: from the deformation gradient `F = Ds·Dm⁻¹` (3×2) come the Green
//! strain `E = ½(FᵀF − I)`, the 2nd-Piola–Kirchhoff stress `S = 2μE + λ tr(E) I`, the 1st-PK stress
//! `P = F·S`, and the exact nodal forces `H = −A₀·P·Dm⁻ᵀ`. Out-of-plane stiffness comes from bending
//! springs across quad diagonals. Everything is a smooth function of the vertex positions, so the
//! solver is differentiable — and because `S` is **linear in the Lamé parameters**, an outcome's
//! gradient w.r.t. the material stiffness `μ` is exact and available in closed form (verified against
//! finite differences to machine precision). Pure `nalgebra` → WASM-clean.

use nalgebra::{Matrix2, Vector3};

/// A triangle-mesh cloth with a StVK membrane and bending springs.
#[derive(Clone)]
pub struct ClothSim {
    pub x: Vec<Vector3<f64>>,
    pub v: Vec<Vector3<f64>>,
    pub pinned: Vec<bool>,
    tris: Vec<[usize; 3]>,
    dm_inv: Vec<Matrix2<f64>>,
    area: Vec<f64>,
    /// Bending springs `(i, j, rest length)`.
    bends: Vec<(usize, usize, f64)>,
    pub mass: f64,
    pub mu: f64,
    pub lambda: f64,
    pub k_bend: f64,
    pub damping: f64,
    pub dt: f64,
    pub gravity: Vector3<f64>,
}

impl ClothSim {
    /// A flat rectangular cloth of `nx × ny` vertices with spacing `h`, in the x–y plane.
    pub fn grid(nx: usize, ny: usize, h: f64, mass: f64, mu: f64, lambda: f64, k_bend: f64, dt: f64) -> Self {
        let idx = |i: usize, j: usize| j * nx + i;
        let mut x = Vec::new();
        for j in 0..ny {
            for i in 0..nx {
                x.push(Vector3::new(i as f64 * h, j as f64 * h, 0.0));
            }
        }
        let n = x.len();
        let mut tris = Vec::new();
        for j in 0..ny - 1 {
            for i in 0..nx - 1 {
                tris.push([idx(i, j), idx(i + 1, j), idx(i, j + 1)]);
                tris.push([idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)]);
            }
        }
        // Rest membrane matrices (material coords = the flat rest positions' x,y).
        let (mut dm_inv, mut area) = (Vec::new(), Vec::new());
        for t in &tris {
            let (a, b, c) = (x[t[0]], x[t[1]], x[t[2]]);
            let dm = Matrix2::new(b.x - a.x, c.x - a.x, b.y - a.y, c.y - a.y);
            area.push(0.5 * dm.determinant().abs());
            dm_inv.push(dm.try_inverse().expect("degenerate rest triangle"));
        }
        // Bending springs across quad diagonals (vertices two apart).
        let mut bends = Vec::new();
        for j in 0..ny {
            for i in 0..nx {
                if i + 2 < nx {
                    bends.push((idx(i, j), idx(i + 2, j), 2.0 * h));
                }
                if j + 2 < ny {
                    bends.push((idx(i, j), idx(i, j + 2), 2.0 * h));
                }
            }
        }
        Self { x, v: vec![Vector3::zeros(); n], pinned: vec![false; n], tris, dm_inv, area, bends, mass, mu, lambda, k_bend, damping: 0.0, dt, gravity: Vector3::new(0.0, 0.0, -9.81) }
    }

    /// Elastic (membrane + bending) potential energy at positions `x`.
    pub fn elastic_energy(&self, x: &[Vector3<f64>]) -> f64 {
        let mut w = 0.0;
        for (ti, t) in self.tris.iter().enumerate() {
            let e = self.green_strain(x, ti, t);
            let tr = e[(0, 0)] + e[(1, 1)];
            let fro = e.iter().map(|v| v * v).sum::<f64>();
            w += self.area[ti] * (self.mu * fro + 0.5 * self.lambda * tr * tr);
        }
        for &(i, j, l0) in &self.bends {
            let d = (x[i] - x[j]).norm() - l0;
            w += 0.5 * self.k_bend * d * d;
        }
        w
    }

    fn green_strain(&self, x: &[Vector3<f64>], ti: usize, t: &[usize; 3]) -> Matrix2<f64> {
        let (a, b, c) = (x[t[0]], x[t[1]], x[t[2]]);
        let ds = nalgebra::Matrix3x2::from_columns(&[b - a, c - a]);
        let f = ds * self.dm_inv[ti]; // 3×2 deformation gradient
        0.5 * (f.transpose() * f - Matrix2::identity())
    }

    /// Elastic forces (membrane + bending) at positions `x`. If `d_dmu`, also returns `∂force/∂μ`.
    fn elastic_forces(&self, x: &[Vector3<f64>], d_dmu: bool) -> (Vec<Vector3<f64>>, Vec<Vector3<f64>>) {
        let n = x.len();
        let (mut f, mut df) = (vec![Vector3::zeros(); n], vec![Vector3::zeros(); n]);
        for (ti, t) in self.tris.iter().enumerate() {
            let (a, b, c) = (x[t[0]], x[t[1]], x[t[2]]);
            let ds = nalgebra::Matrix3x2::from_columns(&[b - a, c - a]);
            let fdef = ds * self.dm_inv[ti];
            let e = 0.5 * (fdef.transpose() * fdef - Matrix2::identity());
            let tr = e[(0, 0)] + e[(1, 1)];
            let s = 2.0 * self.mu * e + self.lambda * tr * Matrix2::identity(); // 2nd PK
            let p = fdef * s; // 1st PK (3×2)
            let h = -self.area[ti] * p * self.dm_inv[ti].transpose(); // 3×2 → forces on b, c
            let (f1, f2) = (h.column(0).into_owned(), h.column(1).into_owned());
            f[t[1]] += f1;
            f[t[2]] += f2;
            f[t[0]] -= f1 + f2;
            if d_dmu {
                // S is linear in μ: ∂S/∂μ = 2E ⇒ ∂P/∂μ = F·2E ⇒ ∂force/∂μ from ∂H/∂μ.
                let dp = fdef * (2.0 * e);
                let dh = -self.area[ti] * dp * self.dm_inv[ti].transpose();
                let (d1, d2) = (dh.column(0).into_owned(), dh.column(1).into_owned());
                df[t[1]] += d1;
                df[t[2]] += d2;
                df[t[0]] -= d1 + d2;
            }
        }
        for &(i, j, l0) in &self.bends {
            let d = x[i] - x[j];
            let len = d.norm();
            if len > 1e-12 {
                let fb = -self.k_bend * (len - l0) * (d / len);
                f[i] += fb;
                f[j] -= fb;
            }
        }
        (f, df)
    }

    /// Total force per vertex (elastic + gravity + damping).
    fn total_force(&self, x: &[Vector3<f64>], v: &[Vector3<f64>]) -> Vec<Vector3<f64>> {
        let (mut f, _) = self.elastic_forces(x, false);
        for i in 0..x.len() {
            f[i] += self.mass * self.gravity - self.damping * v[i];
        }
        f
    }

    /// Advance one timestep (semi-implicit Euler); pinned vertices are held fixed.
    pub fn step(&mut self) {
        let f = self.total_force(&self.x, &self.v);
        for i in 0..self.x.len() {
            if self.pinned[i] {
                self.v[i] = Vector3::zeros();
                continue;
            }
            self.v[i] += self.dt * f[i] / self.mass;
            self.x[i] += self.dt * self.v[i];
        }
    }

    pub fn kinetic_energy(&self) -> f64 {
        0.5 * self.mass * self.v.iter().map(|v| v.norm_squared()).sum::<f64>()
    }

    /// Total kinetic energy after one step, and its exact `∂/∂μ` (the elastic stress is linear in μ).
    pub fn ke_and_dke_dmu(&self) -> (f64, f64) {
        let (fe, dfe) = self.elastic_forces(&self.x, true);
        let (mut ke, mut dke) = (0.0, 0.0);
        for i in 0..self.x.len() {
            if self.pinned[i] {
                continue;
            }
            let f = fe[i] + self.mass * self.gravity - self.damping * self.v[i];
            let vn = self.v[i] + self.dt * f / self.mass;
            let dvn = self.dt * dfe[i] / self.mass; // gravity/damping are μ-independent
            ke += 0.5 * self.mass * vn.norm_squared();
            dke += self.mass * vn.dot(&dvn);
        }
        (ke, dke)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_cloth() -> ClothSim {
        ClothSim::grid(4, 4, 0.1, 0.02, 500.0, 300.0, 20.0, 1e-3)
    }

    #[test]
    fn membrane_force_is_the_exact_energy_gradient() {
        // Perturb the cloth out of plane, then check force == −∂(elastic energy)/∂x by finite diff.
        let mut c = small_cloth();
        for (k, xi) in c.x.iter_mut().enumerate() {
            xi.z += 0.01 * ((k * 7 % 5) as f64 - 2.0);
            xi.x += 0.005 * ((k * 3 % 4) as f64 - 1.5);
        }
        let (f, _) = c.elastic_forces(&c.x, false);
        let eps = 1e-6;
        let mut max_err = 0.0f64;
        for i in 0..c.x.len() {
            for d in 0..3 {
                let mut xp = c.x.clone();
                let mut xm = c.x.clone();
                xp[i][d] += eps;
                xm[i][d] -= eps;
                let fd = -(c.elastic_energy(&xp) - c.elastic_energy(&xm)) / (2.0 * eps);
                max_err = max_err.max((f[i][d] - fd).abs());
            }
        }
        assert!(max_err < 1e-4, "force ≠ −∇energy: max err {max_err:.2e}");
    }

    #[test]
    fn pinned_cloth_drapes_to_equilibrium() {
        // Pin the two top corners; the cloth sags under gravity and (with damping) settles.
        let mut c = small_cloth();
        c.damping = 0.05;
        let (nx, ny) = (4, 4);
        c.pinned[(ny - 1) * nx] = true; // top-left
        c.pinned[(ny - 1) * nx + nx - 1] = true; // top-right
        let z_top = c.x[(ny - 1) * nx].z;
        let mut prev_energy = f64::INFINITY;
        for step in 0..4000 {
            c.step();
            if step % 500 == 499 {
                let e = c.elastic_energy(&c.x) - c.mass * c.gravity.z * c.x.iter().map(|p| p.z).sum::<f64>();
                assert!(e <= prev_energy + 1e-6, "total energy increased");
                prev_energy = e;
            }
        }
        // Free bottom-middle vertex hangs well below the pinned corners; motion has settled.
        assert!(c.x[0].z < z_top - 0.02, "cloth did not sag: z = {}", c.x[0].z);
        assert!(c.kinetic_energy() < 1e-4, "cloth did not settle: KE = {}", c.kinetic_energy());
    }

    #[test]
    fn material_gradient_matches_finite_difference() {
        // Analytic ∂KE/∂μ of one step vs central FD — the differentiable-cloth check.
        let mut c = small_cloth();
        for (k, xi) in c.x.iter_mut().enumerate() {
            xi.z += 0.02 * ((k * 5 % 7) as f64 - 3.0);
        }
        for (k, vi) in c.v.iter_mut().enumerate() {
            *vi = Vector3::new(0.1 * ((k % 3) as f64 - 1.0), 0.05, -0.1);
        }
        let (_, dke) = c.ke_and_dke_dmu();
        let eps = 1e-2;
        let ke_at = |mu: f64| {
            let mut s = c.clone();
            s.mu = mu;
            s.ke_and_dke_dmu().0
        };
        let fd = (ke_at(c.mu + eps) - ke_at(c.mu - eps)) / (2.0 * eps);
        let rel = (dke - fd).abs() / fd.abs().max(1e-12);
        eprintln!("cloth ∂KE/∂μ: analytic={dke:.6e}, fd={fd:.6e}, rel_err={rel:.2e}");
        assert!(dke.abs() > 1e-9, "gradient trivially zero");
        assert!(rel < 1e-6, "material gradient wrong: analytic {dke} vs fd {fd}");
    }
}
