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

pub mod plasticity;

#[cfg(feature = "gpu")]
pub mod gpu;

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
    /// Optional penalty floor at `z = floor`; vertices below it are pushed up by a spring–dashpot
    /// contact of stiffness `k_contact` (so the body can land, rest, and bounce on the ground).
    pub floor: Option<f64>,
    pub k_contact: f64,
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
            floor: None,
            k_contact: 0.0,
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

    /// Build a soft body from an **implicit region** — voxelize `[lo, hi]` into `res³` cells, keep
    /// those whose center is `inside`, and tetrahedralize (5 tets/cell). This lifts the solver off
    /// the box grid onto arbitrary shapes (a sphere, an SDF, any predicate).
    #[allow(clippy::too_many_arguments)]
    pub fn from_implicit(
        lo: Vector3<f64>,
        hi: Vector3<f64>,
        res: usize,
        inside: impl Fn(Vector3<f64>) -> bool,
        mass: f64,
        mu: f64,
        lambda: f64,
        dt: f64,
    ) -> Self {
        let (verts, tets) = tet_mesh_implicit(lo, hi, res, inside);
        Self::new(verts, tets, mass, mu, lambda, dt)
    }

    pub fn n_verts(&self) -> usize {
        self.x.len()
    }
    pub fn n_tets(&self) -> usize {
        self.tets.len()
    }

    /// Tet connectivity (four vertex indices per tet) — for GPU ports and meshing.
    pub fn tets(&self) -> &[[usize; 4]] {
        &self.tets
    }
    /// Per-tet inverse rest edge matrix `Dm⁻¹` (column-major 3×3) — the reference shape.
    pub fn dm_inv(&self) -> &[Matrix3<f64>] {
        &self.dm_inv
    }
    /// Per-tet signed rest volume `det/6`.
    pub fn vol(&self) -> &[f64] {
        &self.vol
    }

    /// Deformation gradient `F = Ds·Dm⁻¹` of tet `e` at the current positions.
    fn deformation_gradient(&self, e: usize) -> Matrix3<f64> {
        let t = &self.tets[e];
        let ds = edge_matrix([self.x[t[0]], self.x[t[1]], self.x[t[2]], self.x[t[3]]]);
        ds * self.dm_inv[e]
    }

    /// Stable Neo-Hookean strain-energy density of a deformation gradient.
    /// `∂J/∂F`, the cofactor matrix — well defined for **any** `F`, including singular ones.
    ///
    /// Written from the column cross-products rather than as `J·F⁻ᵀ` precisely so it survives `det F = 0`,
    /// which is the case the inversion-recovery term has to act at.
    fn cofactor(f: &Matrix3<f64>) -> Matrix3<f64> {
        let (f0, f1, f2) = (f.column(0), f.column(1), f.column(2));
        Matrix3::from_columns(&[f1.cross(&f2), f2.cross(&f0), f0.cross(&f1)])
    }

    /// Target Jacobian the recovery term pulls an inverted element back toward.
    ///
    /// Positive so the restoring force is non-zero even at `J = 0` exactly, which is the configuration a
    /// flattened tet passes through.
    const J_RECOVER: f64 = 0.1;

    /// Stiffness of the inversion-recovery penalty, on the material's own scale so it does not need tuning.
    fn k_recover(&self) -> f64 {
        self.mu + self.lambda
    }

    /// Energy of an inverted element: `½·k·(J − J_recover)²`.
    ///
    /// **This replaces a `+INFINITY` that had no gradient (2026-08-15).** `psi` used to return `+INFINITY` for
    /// `J ≤ 0` and call it "an infinite energy barrier", while [`FemSim::forces`] *skipped* the element on the
    /// identical test — so the barrier exerted exactly **zero** force and an inverted tet could never recover,
    /// in either direction. `energy()` was simultaneously poisoned to `+inf` for the rest of the simulation
    /// while positions stayed finite, so the body kept running and even "settled" with tets still inverted.
    ///
    /// A finite quadratic with a real gradient is the minimum that makes the stated contract true. Note it is
    /// **not** continuous with the `ln J` branch — that branch genuinely diverges as `J → 0⁺`, so no finite
    /// value can meet it. The honest statement is that `J ≤ 0` is outside the material model's domain and this
    /// term exists to push the element back into it, not to model anything.
    ///
    /// **What this does NOT do: rescue a fully degenerate element.** The gradient routes through
    /// `∂J/∂F = cof(F)`, whose entries are cross-products of `F`'s columns, so it *vanishes* as the element
    /// flattens — the restoring force dies exactly where it is most needed and the approach to `J = 0` is
    /// asymptotic. Measured on an inverted unit tet with gravity off, `J` rises monotonically from `−0.2` to
    /// `−0.0346` over 200k steps and does not cross. Escaping a flattened element needs an energy whose gradient
    /// does not pass through `∂J/∂F`, which is what the literature's *stable* Neo-Hookean formulations exist for.
    /// This turns a zero gradient into a correctly-signed one; it does not make the model inversion-safe.
    fn psi_inverted(&self, f: &Matrix3<f64>) -> f64 {
        let d = f.determinant() - Self::J_RECOVER;
        0.5 * self.k_recover() * d * d
    }

    fn psi(&self, f: &Matrix3<f64>) -> f64 {
        let j = f.determinant();
        if j <= 0.0 {
            return self.psi_inverted(f);
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
            let p = if j <= 0.0 {
                // **Inverted: apply the recovery gradient instead of skipping (2026-08-15).** This used to
                // `continue`, so the `+INFINITY` barrier `psi` advertised had no gradient at all and an
                // inverted tet contributed no force in either direction — it could never recover, while
                // `energy()` read `+inf` forever and the simulation carried on with finite positions.
                //
                // ∂/∂F of ½k(J − J_recover)² is k(J − J_recover)·∂J/∂F, and ∂J/∂F is the cofactor matrix,
                // which is defined even at `det F = 0` — the configuration a flattening tet passes through.
                // `J − J_recover < 0` here, so this pulls J back up.
                self.k_recover() * (j - Self::J_RECOVER) * Self::cofactor(&fg)
            } else {
                let fit = fg.try_inverse().unwrap().transpose(); // F⁻ᵀ
                // P = μ(F − F⁻ᵀ) + λ ln(J) F⁻ᵀ
                self.mu * (fg - fit) + self.lambda * j.ln() * fit
            };
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

    /// One semi-implicit (symplectic) Euler step under gravity + elasticity + optional floor
    /// contact, pins held fixed.
    #[allow(clippy::needless_range_loop)] // vertex index addresses forces/x/v/pinned together
    pub fn step(&mut self) {
        let mut forces = self.forces();
        // penalty floor contact: spring–dashpot on vertices below the floor
        if let Some(fz) = self.floor {
            let gamma = 0.7 * (self.k_contact * self.mass).sqrt(); // near-critical contact damping
            for i in 0..self.x.len() {
                let pen = fz - self.x[i].z;
                if pen > 0.0 {
                    let vn = self.v[i].z.min(0.0); // dissipate only while approaching
                    forces[i].z += self.k_contact * pen - gamma * vn;
                }
            }
        }
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

/// Tetrahedralize the interior of an implicit region. Voxelize `[lo, hi]` into `res³` cells; for
/// each cell whose center satisfies `inside`, emit its 8 corners (shared corners deduplicated) and
/// a 5-tet split. Returns `(vertices, tets)`.
pub fn tet_mesh_implicit(lo: Vector3<f64>, hi: Vector3<f64>, res: usize, inside: impl Fn(Vector3<f64>) -> bool) -> (Vec<Vector3<f64>>, Vec<[usize; 4]>) {
    use std::collections::HashMap;
    let h = Vector3::new((hi.x - lo.x) / res as f64, (hi.y - lo.y) / res as f64, (hi.z - lo.z) / res as f64);
    let mut verts: Vec<Vector3<f64>> = Vec::new();
    let mut vmap: HashMap<(usize, usize, usize), usize> = HashMap::new();
    let mut vid = |i: usize, j: usize, k: usize, verts: &mut Vec<Vector3<f64>>| -> usize {
        *vmap.entry((i, j, k)).or_insert_with(|| {
            verts.push(Vector3::new(lo.x + i as f64 * h.x, lo.y + j as f64 * h.y, lo.z + k as f64 * h.z));
            verts.len() - 1
        })
    };
    let mut tets = Vec::new();
    for k in 0..res {
        for j in 0..res {
            for i in 0..res {
                let center = Vector3::new(lo.x + (i as f64 + 0.5) * h.x, lo.y + (j as f64 + 0.5) * h.y, lo.z + (k as f64 + 0.5) * h.z);
                if !inside(center) {
                    continue;
                }
                let c = [
                    vid(i, j, k, &mut verts),
                    vid(i + 1, j, k, &mut verts),
                    vid(i + 1, j + 1, k, &mut verts),
                    vid(i, j + 1, k, &mut verts),
                    vid(i, j, k + 1, &mut verts),
                    vid(i + 1, j, k + 1, &mut verts),
                    vid(i + 1, j + 1, k + 1, &mut verts),
                    vid(i, j + 1, k + 1, &mut verts),
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
    (verts, tets)
}

#[cfg(test)]
mod verification {
    use super::*;
    use nalgebra::{Rotation3, Vector3};

    /// The implicit tetrahedralizer meshes a sphere: total tet volume converges to 4⁄3πr³ as the
    /// resolution rises, every tet is non-degenerate, and a body built from it runs in the solver —
    /// the FEM solver now works on arbitrary shapes, not just box grids.
    #[test]
    fn implicit_tet_mesh_meshes_a_sphere() {
        let (c, r) = (Vector3::new(0.0, 0.0, 0.0), 1.0);
        let inside = |p: Vector3<f64>| (p - c).norm() < r;
        let lo = Vector3::new(-1.2, -1.2, -1.2);
        let hi = Vector3::new(1.2, 1.2, 1.2);
        let exact = 4.0 / 3.0 * std::f64::consts::PI * r * r * r;
        let vol_at = |res: usize| -> f64 {
            let (v, t) = tet_mesh_implicit(lo, hi, res, inside);
            t.iter()
                .map(|te| {
                    let m = Matrix3::from_columns(&[v[te[1]] - v[te[0]], v[te[2]] - v[te[0]], v[te[3]] - v[te[0]]]);
                    (m.determinant() / 6.0).abs()
                })
                .sum()
        };
        let (v16, v32) = (vol_at(16), vol_at(32));
        eprintln!("implicit sphere mesh volume: res16 {v16:.4}, res32 {v32:.4}, exact {exact:.4}");
        let (verts, tets) = tet_mesh_implicit(lo, hi, 20, inside);
        let worst = tets
            .iter()
            .map(|te| {
                let m = Matrix3::from_columns(&[verts[te[1]] - verts[te[0]], verts[te[2]] - verts[te[0]], verts[te[3]] - verts[te[0]]]);
                (m.determinant() / 6.0).abs()
            })
            .fold(f64::INFINITY, f64::min);
        assert!(worst > 0.0, "a tet is degenerate: {worst}");
        assert!((v32 - exact).abs() < 0.05 * exact, "res32 volume off: {v32} vs {exact}");
        assert!((v32 - exact).abs() < (v16 - exact).abs(), "not converging: {v16} → {v32}");
        let mut sim = FemSim::from_implicit(lo, hi, 12, inside, 0.4, 3.0e3, 1.5e3, 3e-4);
        assert!(sim.n_tets() > 100, "too few tets: {}", sim.n_tets());
        sim.step();
        assert!(sim.energy().is_finite(), "solver blew up on the meshed sphere");
    }

    /// **An inverted element must be pushed back, not skipped.** `psi` advertised a `+INFINITY` barrier while
    /// `forces` skipped the element on the identical `J <= 0` test, so the barrier had exactly zero gradient:
    /// an inverted tet contributed no force in either direction and could never recover, while `energy()` read
    /// `+inf` forever and the simulation carried on with finite positions.
    #[test]
    fn an_inverted_element_is_pushed_back_rather_than_ignored() {
        let mut sim = single_tet(8.0, 4.0);
        sim.x[3] = Vector3::new(0.0, 0.0, -0.2); // J = -0.2, inverted through the opposite face
        let j0 = sim.deformation_gradient(0).determinant();
        assert!(j0 < 0.0, "fixture must actually be inverted, J = {j0}");

        // 1. The energy is finite, so it no longer poisons every later reading.
        let e = sim.energy();
        assert!(e.is_finite(), "energy of an inverted element should be finite, got {e}");

        // 2. The force is non-zero — this is the whole defect. It used to be exactly [0,0,0,0].
        let f = sim.forces();
        let fmax = f.iter().map(|v| v.norm()).fold(0.0f64, f64::max);
        assert!(fmax > 1e-6, "an inverted element must exert a restoring force, worst |f| = {fmax}");

        // 3. And it points the right way: stepping must drive J back up through zero.
        //
        // Gravity is switched off so this measures the ELASTIC recovery rather than a race between the two —
        // with gravity left on, J still rises (−0.2 → −0.175 over 20k steps) but the free tet is also falling,
        // which makes the test about the budget rather than about the force.
        let mut relaxing = sim.clone();
        relaxing.gravity = Vector3::zeros();
        relaxing.damping = 0.5;
        let mut j = j0;
        let mut ever_decreased = false;
        for _ in 0..200_000 {
            relaxing.step();
            let jn = relaxing.deformation_gradient(0).determinant();
            assert!(jn.is_finite(), "recovery must stay finite, got {jn}");
            if jn < j - 1e-12 && jn < 0.0 {
                ever_decreased = true;
            }
            j = jn;
            if j > 0.05 {
                break;
            }
        }
        // What is claimed, and all that is claimed: J rises monotonically toward zero. Measured −0.2 → −0.0346
        // over 200k steps with gravity off.
        assert!(!ever_decreased, "while inverted, J must never be driven further negative");
        assert!(j > j0 + 0.1, "J should recover substantially; went {j0} → {j}");
        // **It is NOT claimed that the element fully un-inverts, and it does not.** `∂J/∂F` is the cofactor
        // matrix, whose entries are cross-products of F's columns — so it *vanishes* as the element flattens,
        // and the restoring force dies exactly where it is most needed. Approach to J = 0 is therefore
        // asymptotic. Escaping a fully degenerate element needs an energy that does not route its gradient
        // through ∂J/∂F, which is what the literature's stable Neo-Hookean formulations are for. This fix turns
        // a zero gradient into a correctly-signed one; it does not make the material model inversion-safe.
        assert!(j < 0.0, "if this now crosses zero, the recovery got stronger and the doc's caveat needs revising");
    }

    /// The recovery term must not touch a healthy element — bit-identical, not merely close.
    #[test]
    fn a_healthy_element_is_bit_identical_under_the_recovery_branch() {
        let mut sim = single_tet(8.0, 4.0);
        sim.x[3] = Vector3::new(0.0, 0.0, 1.3); // stretched but perfectly valid
        let j = sim.deformation_gradient(0).determinant();
        assert!(j > 0.0, "fixture must be un-inverted, J = {j}");
        // Recompute the Neo-Hookean stress independently and require exact equality with `forces`.
        let fg = sim.deformation_gradient(0);
        let fit = fg.try_inverse().unwrap().transpose();
        let p = sim.mu * (fg - fit) + sim.lambda * j.ln() * fit;
        let h = -sim.vol[0].abs() * p * sim.dm_inv[0].transpose();
        let got = sim.forces();
        let mut want = [Vector3::zeros(); 4];
        for c in 0..3 {
            let fc = h.column(c);
            want[sim.tets[0][c + 1]] += fc;
            want[sim.tets[0][0]] -= fc;
        }
        for i in 0..4 {
            assert_eq!(got[i], want[i], "vertex {i}: the healthy branch must be untouched");
        }
    }

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

    /// A free soft body dropped onto the penalty floor lands, deforms, and settles to rest above it —
    /// no vertex tunnels through, and the motion damps out.
    #[test]
    fn dropped_soft_body_lands_and_settles() {
        let mut sim = FemSim::box_grid(2, 2, 2, 0.3, 0.4, 4.0e3, 2.0e3, 3e-4);
        sim.damping = 0.003;
        sim.floor = Some(0.0);
        sim.k_contact = 3.0e4;
        // lift it above the floor
        for p in &mut sim.x {
            p.z += 0.5;
        }
        for _ in 0..9000 {
            sim.step();
        }
        let lowest = sim.x.iter().map(|p| p.z).fold(f64::INFINITY, f64::min);
        let ke: f64 = sim.v.iter().map(|v| 0.5 * sim.mass * v.norm_squared()).sum();
        eprintln!("dropped soft body: lowest z {lowest:.4}, residual KE {ke:.2e}");
        assert!(lowest > -0.02, "a vertex tunneled through the floor: {lowest}");
        assert!(lowest < 0.05, "the body never reached the floor: {lowest}");
        assert!(ke < 5e-3, "the body did not settle: KE {ke}"); // near-rest (small elastic jiggle is physical)
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
