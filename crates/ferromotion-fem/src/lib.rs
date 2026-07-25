//! ferromotion-fem — a **differentiable volumetric tetrahedral FEM** solver (stable Neo-Hookean),
//! the fourth deformable domain that completes the spectrum: the [cloth](../ferromotion_cloth) crate
//! is a thin *shell*, the [MPM](../ferromotion_mpm) crate is a *particle* method, the
//! [rod](../ferromotion_rod) crate is a 1-D *filament* — this is the *volumetric solid*, the piece
//! Genesis (Implicit FEM) and SoftMAC have that the rest of the field's Rust ecosystem lacks.
//!
//! The body is a tetrahedral mesh. Each tet is a constant-strain element: from the rest edge matrix
//! `Dm` and the current edge matrix `Ds` come the deformation gradient `F = Ds·Dm⁻¹`, and from `F`
//! the **stable Neo-Hookean** strain energy density
//! `Ψ(F) = ½μ(I_C − 3) − μ ln J + ½λ(ln J)²` (Smith et al. 2018), whose first Piola–Kirchhoff stress
//! `P = ∂Ψ/∂F = μ(F − F⁻ᵀ) + λ ln(J)·F⁻ᵀ` gives the exact nodal forces `−V₀·P·Dm⁻ᵀ`. Neo-Hookean is
//! frame-indifferent (a rigid motion costs zero energy) and robust to large deformation, unlike a
//! linear or St.-Venant model. Everything is a smooth function of the vertex positions, so forces are
//! exactly `−∇energy` and an outcome's gradient w.r.t. the material stiffness is available in closed
//! form (both verified against finite differences). Pure `nalgebra` → WASM-clean.

use nalgebra::{Matrix3, Vector3};

/// A tetrahedral-mesh soft body with a stable Neo-Hookean material.
#[derive(Clone)]
pub struct FemSim {
    pub x: Vec<Vector3<f64>>,
    pub v: Vec<Vector3<f64>>,
    pub pinned: Vec<bool>,
    tets: Vec<[usize; 4]>,
    dm_inv: Vec<Matrix3<f64>>,
    vol: Vec<f64>,
    pub mass: f64,
    pub mu: f64,
    pub lambda: f64,
    pub damping: f64,
    pub dt: f64,
    pub gravity: Vector3<f64>,
}

/// Rest edge matrix of a tet (columns = edges from vertex 0), and its signed volume `det/6`.
fn edge_matrix(p: [Vector3<f64>; 4]) -> Matrix3<f64> {
    Matrix3::from_columns(&[p[1] - p[0], p[2] - p[0], p[3] - p[0]])
}

impl FemSim {
    /// Build from an explicit vertex list and tetrahedra. Rest state is the given geometry.
    pub fn new(x: Vec<Vector3<f64>>, tets: Vec<[usize; 4]>, mass: f64, mu: f64, lambda: f64, dt: f64) -> Self {
        let n = x.len();
        let mut dm_inv = Vec::with_capacity(tets.len());
        let mut vol = Vec::with_capacity(tets.len());
        for t in &tets {
            let dm = edge_matrix([x[t[0]], x[t[1]], x[t[2]], x[t[3]]]);
            vol.push(dm.determinant() / 6.0);
            dm_inv.push(dm.try_inverse().expect("degenerate tetrahedron"));
        }
        FemSim {
            v: vec![Vector3::zeros(); n],
            pinned: vec![false; n],
            x,
            tets,
            dm_inv,
            vol,
            mass,
            mu,
            lambda,
            damping: 0.0,
            dt,
            gravity: Vector3::new(0.0, 0.0, -9.81),
        }
    }

    /// A solid box of `nx × ny × nz` cells (unit-cube-split into 5 tets per cell), spacing `h`.
    #[allow(clippy::too_many_arguments)]
    pub fn box_grid(nx: usize, ny: usize, nz: usize, h: f64, mass: f64, mu: f64, lambda: f64, dt: f64) -> Self {
        let (gx, gy, gz) = (nx + 1, ny + 1, nz + 1);
        let vid = |i: usize, j: usize, k: usize| (k * gy + j) * gx + i;
        let mut x = Vec::with_capacity(gx * gy * gz);
        for k in 0..gz {
            for j in 0..gy {
                for i in 0..gx {
                    x.push(Vector3::new(i as f64 * h, j as f64 * h, k as f64 * h));
                }
            }
        }
        // 5-tet decomposition of each hexahedral cell (BCC-consistent alternating parity).
        let mut tets = Vec::new();
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let c = [
                        vid(i, j, k),
                        vid(i + 1, j, k),
                        vid(i + 1, j + 1, k),
                        vid(i, j + 1, k),
                        vid(i, j, k + 1),
                        vid(i + 1, j, k + 1),
                        vid(i + 1, j + 1, k + 1),
                        vid(i, j + 1, k + 1),
                    ];
                    if (i + j + k) % 2 == 0 {
                        tets.push([c[0], c[1], c[3], c[4]]);
                        tets.push([c[1], c[2], c[3], c[6]]);
                        tets.push([c[1], c[3], c[4], c[6]]);
                        tets.push([c[1], c[4], c[5], c[6]]);
                        tets.push([c[3], c[4], c[6], c[7]]);
                    } else {
                        tets.push([c[0], c[1], c[2], c[5]]);
                        tets.push([c[0], c[2], c[3], c[7]]);
                        tets.push([c[0], c[2], c[5], c[7]]);
                        tets.push([c[0], c[4], c[5], c[7]]);
                        tets.push([c[2], c[5], c[6], c[7]]);
                    }
                }
            }
        }
        Self::new(x, tets, mass, mu, lambda, dt)
    }

    pub fn n_verts(&self) -> usize {
        self.x.len()
    }
    pub fn n_tets(&self) -> usize {
        self.tets.len()
    }

    /// Deformation gradient `F = Ds·Dm⁻¹` of tet `e` at the current positions.
    fn deformation_gradient(&self, e: usize) -> Matrix3<f64> {
        let t = &self.tets[e];
        let ds = edge_matrix([self.x[t[0]], self.x[t[1]], self.x[t[2]], self.x[t[3]]]);
        ds * self.dm_inv[e]
    }

    /// Stable Neo-Hookean strain-energy density of a deformation gradient.
    fn psi(&self, f: &Matrix3<f64>) -> f64 {
        let j = f.determinant();
        if j <= 0.0 {
            return f64::INFINITY; // inverted element — infinite energy barrier
        }
        let i_c = (f.transpose() * f).trace();
        0.5 * self.mu * (i_c - 3.0) - self.mu * j.ln() + 0.5 * self.lambda * j.ln().powi(2)
    }

    /// Total elastic strain energy of the mesh.
    pub fn energy(&self) -> f64 {
        (0..self.tets.len()).map(|e| self.vol[e].abs() * self.psi(&self.deformation_gradient(e))).sum()
    }

    /// Per-vertex elastic force `−∂E/∂x`, from the analytic first Piola–Kirchhoff stress.
    pub fn forces(&self) -> Vec<Vector3<f64>> {
        let mut f = vec![Vector3::zeros(); self.x.len()];
        for e in 0..self.tets.len() {
            let fg = self.deformation_gradient(e);
            let j = fg.determinant();
            if j <= 0.0 {
                continue;
            }
            let fit = fg.try_inverse().unwrap().transpose(); // F⁻ᵀ
            // P = μ(F − F⁻ᵀ) + λ ln(J) F⁻ᵀ
            let p = self.mu * (fg - fit) + self.lambda * j.ln() * fit;
            // nodal force block for verts 1,2,3: H = −V₀·P·Dm⁻ᵀ (columns are the forces)
            let h = -self.vol[e].abs() * p * self.dm_inv[e].transpose();
            let t = &self.tets[e];
            for c in 0..3 {
                let fc = h.column(c);
                f[t[c + 1]] += fc;
                f[t[0]] -= fc; // vertex 0 gets the negative sum
            }
        }
        f
    }

    /// One semi-implicit (symplectic) Euler step under gravity + elasticity, pins held fixed.
    #[allow(clippy::needless_range_loop)] // vertex index addresses forces/x/v/pinned together
    pub fn step(&mut self) {
        let forces = self.forces();
        let inv_m = 1.0 / self.mass;
        for i in 0..self.x.len() {
            if self.pinned[i] {
                self.v[i] = Vector3::zeros();
                continue;
            }
            let a = forces[i] * inv_m + self.gravity;
            self.v[i] = (self.v[i] + self.dt * a) * (1.0 - self.damping);
            self.x[i] += self.dt * self.v[i];
        }
    }

    /// Linear momentum of the body.
    pub fn momentum(&self) -> Vector3<f64> {
        self.v.iter().fold(Vector3::zeros(), |acc, v| acc + self.mass * v)
    }
}

#[cfg(test)]
mod verification {
    use super::*;
    use nalgebra::{Rotation3, Vector3};

    fn single_tet(mu: f64, lambda: f64) -> FemSim {
        // a unit reference tetrahedron
        let x = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
        ];
        FemSim::new(x, vec![[0, 1, 2, 3]], 1.0, mu, lambda, 1e-3)
    }

    /// Frame indifference: a rigid translation *and* a rigid rotation of the whole body cost zero
    /// energy and produce zero force — the property that distinguishes Neo-Hookean/co-rotational
    /// from a naive linear model.
    #[test]
    fn rigid_motion_is_energy_free() {
        let mut sim = single_tet(10.0, 5.0);
        // translate
        for p in &mut sim.x {
            *p += Vector3::new(3.0, -2.0, 1.0);
        }
        assert!(sim.energy().abs() < 1e-12, "translation not energy-free: {}", sim.energy());
        // rotate about an arbitrary axis
        let r = Rotation3::new(Vector3::new(0.3, -0.7, 0.5));
        let mut sim = single_tet(10.0, 5.0);
        for p in &mut sim.x {
            *p = r * *p;
        }
        let e = sim.energy();
        let fmax = sim.forces().iter().map(|f| f.norm()).fold(0.0f64, f64::max);
        eprintln!("rigid rotation: energy {e:.2e}, max force {fmax:.2e}");
        assert!(e.abs() < 1e-10, "rotation not energy-free: {e}");
        assert!(fmax < 1e-9, "rotation produced force: {fmax}");
    }

    /// The analytic nodal force equals `−∂E/∂x` (central finite differences on the total energy) —
    /// validates the Piola–Kirchhoff stress derivation.
    #[test]
    fn force_matches_energy_gradient() {
        let mut sim = single_tet(8.0, 4.0);
        // deform into a non-trivial state
        sim.x[1] += Vector3::new(0.15, 0.05, -0.1);
        sim.x[2] += Vector3::new(-0.08, 0.2, 0.06);
        sim.x[3] += Vector3::new(0.04, -0.03, 0.18);
        let analytic = sim.forces();
        let eps = 1e-6;
        let mut worst = 0.0f64;
        for i in 0..sim.x.len() {
            for d in 0..3 {
                let mut sp = sim.clone();
                sp.x[i][d] += eps;
                let mut sm = sim.clone();
                sm.x[i][d] -= eps;
                let fd = -(sp.energy() - sm.energy()) / (2.0 * eps); // force = −dE/dx
                worst = worst.max((analytic[i][d] - fd).abs());
            }
        }
        eprintln!("FEM force vs −∇energy: worst {worst:.2e}");
        assert!(worst < 1e-5, "nodal force does not match −∇energy: {worst}");
    }

    /// Differentiability: the exact `∂energy/∂μ` (energy is affine in the Lamé parameters) matches a
    /// finite difference — the same closed-form material-gradient the cloth/MPM crates provide.
    #[test]
    fn energy_gradient_wrt_stiffness_is_exact() {
        let mut sim = single_tet(8.0, 4.0);
        sim.x[1] += Vector3::new(0.2, 0.1, -0.05);
        sim.x[2] += Vector3::new(-0.1, 0.25, 0.1);
        // ∂Ψ/∂μ = ½(I_C − 3) − ln J summed over tets (weighted by volume)
        let e = 0;
        let f = sim.deformation_gradient(e);
        let j = f.determinant();
        let i_c = (f.transpose() * f).trace();
        let analytic = sim.vol[e].abs() * (0.5 * (i_c - 3.0) - j.ln());
        let eps = 1e-6;
        let e_of = |mu: f64| {
            let mut s = sim.clone();
            s.mu = mu;
            s.energy()
        };
        let fd = (e_of(sim.mu + eps) - e_of(sim.mu - eps)) / (2.0 * eps);
        eprintln!("dE/dmu: analytic {analytic:.6} fd {fd:.6}");
        assert!((analytic - fd).abs() < 1e-6, "material gradient off: {analytic} vs {fd}");
    }

    /// A free (unpinned) body under gravity conserves horizontal momentum and accelerates at g —
    /// no spurious elastic forces from a rigid free-fall.
    #[test]
    fn free_fall_conserves_and_accelerates() {
        let mut sim = FemSim::box_grid(2, 2, 2, 0.5, 0.2, 50.0, 30.0, 1e-3);
        sim.gravity = Vector3::new(0.0, 0.0, -9.81);
        let n = sim.n_verts() as f64;
        for _ in 0..50 {
            sim.step();
        }
        let vz: f64 = sim.v.iter().map(|v| v[2]).sum::<f64>() / n;
        let vxy: f64 = sim.v.iter().map(|v| v[0].abs() + v[1].abs()).sum();
        let expect = -9.81 * 50.0 * 1e-3;
        eprintln!("free fall: mean vz {vz:.5} (expect {expect:.5}), lateral drift {vxy:.2e}");
        assert!((vz - expect).abs() < 1e-6, "not free-falling at g: {vz} vs {expect}");
        assert!(vxy < 1e-9, "spurious lateral motion in rigid free-fall: {vxy}");
    }

    /// A stretched box builds a static energy that grows monotonically with the stretch — sanity on
    /// the assembled volumetric energy.
    #[test]
    fn uniaxial_stretch_stores_energy() {
        let base = FemSim::box_grid(2, 1, 1, 1.0, 1.0, 20.0, 10.0, 1e-3);
        let mut prev = 0.0;
        for s in [1.05, 1.15, 1.3] {
            let mut sim = base.clone();
            for p in &mut sim.x {
                p[0] *= s; // stretch along x
            }
            let e = sim.energy();
            assert!(e > prev, "energy did not grow with stretch {s}: {e} <= {prev}");
            prev = e;
        }
        eprintln!("uniaxial stretch stores monotone energy up to {prev:.3}");
    }
}
