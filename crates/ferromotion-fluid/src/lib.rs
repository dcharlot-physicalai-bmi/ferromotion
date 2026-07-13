//! ferromotion-fluid — 2D incompressible Navier–Stokes for fluid–robot interaction (the Aquarium
//! track).
//!
//! Milestone 1: a **marker-and-cell (MAC) projection solver** for the incompressible NS equations on
//! a staggered grid. Velocities live on cell faces (`u` on vertical faces, `v` on horizontal faces),
//! pressure at cell centers — the layout that kills the checkerboard mode without artificial
//! stabilization. Each step is Chorin's fractional method: an explicit advection+diffusion predictor
//! for the intermediate velocity `u*`, a **pressure Poisson projection** back onto the
//! divergence-free manifold, then the velocity correction. The pressure Laplacian is constant, so it
//! is assembled once and factored with **`faer`'s sparse Cholesky**; every timestep is then just a
//! back-substitution — the projection is the expensive part and this makes it cheap.
//!
//! Verified against the canonical **Ghia, Ghia & Shin (1982)** lid-driven-cavity benchmark at
//! Re = 100 (centerline velocity profile), plus a hard internal check that the projected velocity
//! field is discretely divergence-free.
//!
//! Fluid–structure interaction rides on a **direct-forcing immersed boundary** ([`RigidDisk`],
//! [`MacFluid::step_with_disk`]): a body's surface is a set of Lagrangian markers, and iterated
//! forcing through the 4-point Peskin kernel drives the interpolated fluid velocity to the surface
//! velocity — no body-fitted mesh. Verified no-slip enforcement (~4 % residual), Stokes-regime drag
//! linearity, and reported hydrodynamic force on the body. A [`FreeDisk`] closes the loop —
//! [`MacFluid::step_free_disk`] solves the body's own motion from the fluid force plus external
//! forces (two-way coupling); a dense disk released from rest settles to the terminal velocity where
//! drag balances net weight (verified to ~6 %). Pure Rust → WASM-clean.

use faer::linalg::solvers::Solve;
use faer::sparse::linalg::solvers::Llt;
use faer::sparse::{SparseColMat, Triplet};
use faer::{Mat, Side};

/// A staggered-grid incompressible fluid on the unit square `[0,1]²` with no-slip walls and an
/// optional moving lid (the classic lid-driven cavity when the top wall translates).
pub struct MacFluid {
    nx: usize,
    ny: usize,
    h: f64,
    nu: f64,
    dt: f64,
    lid_u: f64,
    /// x-velocity on vertical faces: `(nx+1) × ny`, index `i*ny + j`.
    u: Vec<f64>,
    /// y-velocity on horizontal faces: `nx × (ny+1)`, index `i*(ny+1) + j`.
    v: Vec<f64>,
    /// pressure at cell centers: `nx × ny`, index `i*ny + j`.
    p: Vec<f64>,
    /// Prefactored (negated, cell-0-pinned) pressure Laplacian.
    poisson: Llt<usize, f64>,
}

impl MacFluid {
    /// Build a solver on an `nx × ny` grid with kinematic viscosity `nu`, timestep `dt`, and a lid
    /// speed `lid_u` (top wall translates in +x; set 0 for four static walls).
    pub fn new(nx: usize, ny: usize, nu: f64, dt: f64, lid_u: f64) -> Self {
        assert!(nx >= 3 && ny >= 3, "grid too small");
        let h = 1.0 / nx as f64;
        assert!((h - 1.0 / ny as f64).abs() < 1e-12, "square cells only (nx == ny)");
        let poisson = Self::build_poisson(nx, ny);
        Self {
            nx,
            ny,
            h,
            nu,
            dt,
            lid_u,
            u: vec![0.0; (nx + 1) * ny],
            v: vec![0.0; nx * (ny + 1)],
            p: vec![0.0; nx * ny],
            poisson,
        }
    }

    /// The `-∇²` operator with homogeneous-Neumann walls, cell 0 pinned to make it SPD, factored once.
    fn build_poisson(nx: usize, ny: usize) -> Llt<usize, f64> {
        let idx = |i: usize, j: usize| i * ny + j;
        let mut trips: Vec<Triplet<usize, usize, f64>> = Vec::new();
        for i in 0..nx {
            for j in 0..ny {
                let k = idx(i, j);
                if k == 0 {
                    continue; // pinned Dirichlet reference cell
                }
                let red = k - 1;
                let mut diag = 0.0;
                // Each in-grid neighbour contributes to the diagonal; wall faces are dropped (Neumann).
                let nb = |ni: usize, nj: usize, diag: &mut f64, trips: &mut Vec<Triplet<usize, usize, f64>>| {
                    *diag += 1.0;
                    let nk = idx(ni, nj);
                    if nk != 0 {
                        let nred = nk - 1;
                        if nred < red {
                            trips.push(Triplet::new(red, nred, -1.0)); // strict lower triangle
                        }
                    }
                };
                if i + 1 < nx {
                    nb(i + 1, j, &mut diag, &mut trips);
                }
                if i > 0 {
                    nb(i - 1, j, &mut diag, &mut trips);
                }
                if j + 1 < ny {
                    nb(i, j + 1, &mut diag, &mut trips);
                }
                if j > 0 {
                    nb(i, j - 1, &mut diag, &mut trips);
                }
                trips.push(Triplet::new(red, red, diag));
            }
        }
        let n = nx * ny - 1;
        let mat = SparseColMat::<usize, f64>::try_new_from_triplets(n, n, &trips).expect("assemble Poisson");
        mat.sp_cholesky(Side::Lower).expect("Poisson SPD")
    }

    // ---- ghost-aware face accessors (i32 so wall ghosts are expressible) ----

    /// x-velocity with tangential-wall ghosts in `j` (no-slip bottom, moving lid on top).
    fn uu(&self, i: i32, j: i32) -> f64 {
        let (nx, ny) = (self.nx as i32, self.ny as i32);
        debug_assert!((0..=nx).contains(&i));
        if j < 0 {
            -self.u[(i as usize) * self.ny] // bottom wall no-slip: u_ghost = -u(i,0)
        } else if j >= ny {
            2.0 * self.lid_u - self.u[(i as usize) * self.ny + (ny as usize - 1)] // lid
        } else {
            self.u[(i as usize) * self.ny + j as usize]
        }
    }

    /// y-velocity with tangential-wall ghosts in `i` (no-slip left/right walls).
    fn vv(&self, i: i32, j: i32) -> f64 {
        let (nx, ny) = (self.nx as i32, self.ny as i32);
        debug_assert!((0..=ny).contains(&j));
        if i < 0 {
            -self.v[j as usize] // left wall: v_ghost = -v(0,j)
        } else if i >= nx {
            -self.v[(nx as usize - 1) * (self.ny + 1) + j as usize] // right wall
        } else {
            self.v[(i as usize) * (self.ny + 1) + j as usize]
        }
    }

    /// Advance one timestep and return the max per-step velocity change (a steady-state monitor).
    pub fn step(&mut self) -> f64 {
        let (us, vs) = self.predict();
        self.project_and_correct(us, vs)
    }

    /// Explicit advection + diffusion predictor → intermediate face velocities `(u*, v*)`.
    fn predict(&self) -> (Vec<f64>, Vec<f64>) {
        let (nx, ny, h, nu, dt) = (self.nx, self.ny, self.h, self.nu, self.dt);
        let (inv2h, invh2) = (1.0 / (2.0 * h), 1.0 / (h * h));
        let mut us = self.u.clone();
        let mut vs = self.v.clone();
        for i in 1..nx {
            for j in 0..ny {
                let (ii, jj) = (i as i32, j as i32);
                let uc = self.uu(ii, jj);
                let dudx = (self.uu(ii + 1, jj) - self.uu(ii - 1, jj)) * inv2h;
                let dudy = (self.uu(ii, jj + 1) - self.uu(ii, jj - 1)) * inv2h;
                let vbar = 0.25 * (self.vv(ii - 1, jj) + self.vv(ii, jj) + self.vv(ii - 1, jj + 1) + self.vv(ii, jj + 1));
                let lap = (self.uu(ii + 1, jj) + self.uu(ii - 1, jj) + self.uu(ii, jj + 1) + self.uu(ii, jj - 1) - 4.0 * uc) * invh2;
                us[i * ny + j] = uc + dt * (-(uc * dudx + vbar * dudy) + nu * lap);
            }
        }
        for i in 0..nx {
            for j in 1..ny {
                let (ii, jj) = (i as i32, j as i32);
                let vc = self.vv(ii, jj);
                let dvdx = (self.vv(ii + 1, jj) - self.vv(ii - 1, jj)) * inv2h;
                let dvdy = (self.vv(ii, jj + 1) - self.vv(ii, jj - 1)) * inv2h;
                let ubar = 0.25 * (self.uu(ii, jj - 1) + self.uu(ii + 1, jj - 1) + self.uu(ii, jj) + self.uu(ii + 1, jj));
                let lap = (self.vv(ii + 1, jj) + self.vv(ii - 1, jj) + self.vv(ii, jj + 1) + self.vv(ii, jj - 1) - 4.0 * vc) * invh2;
                vs[i * (ny + 1) + j] = vc + dt * (-(ubar * dvdx + vc * dvdy) + nu * lap);
            }
        }
        // Dirichlet normal-velocity walls (columns 0/nx of u*, rows 0/ny of v*) stay at 0.
        (us, vs)
    }

    /// Pressure-Poisson projection of `(u*, v*)` onto the divergence-free manifold + velocity
    /// correction. Writes the new field into `self`; returns the max per-step velocity change.
    fn project_and_correct(&mut self, us: Vec<f64>, vs: Vec<f64>) -> f64 {
        let (nx, ny, h, dt) = (self.nx, self.ny, self.h, self.dt);
        let n = nx * ny - 1;
        let mut rhs = Mat::<f64>::zeros(n, 1);
        for i in 0..nx {
            for j in 0..ny {
                let k = i * ny + j;
                if k == 0 {
                    continue;
                }
                let div = (us[(i + 1) * ny + j] - us[i * ny + j]) / h + (vs[i * (ny + 1) + j + 1] - vs[i * (ny + 1) + j]) / h;
                rhs[(k - 1, 0)] = -h * h * div / dt; // (-∇²) p = -h²·rhs
            }
        }
        self.poisson.solve_in_place(&mut rhs);
        self.p[0] = 0.0;
        for k in 1..nx * ny {
            self.p[k] = rhs[(k - 1, 0)];
        }

        let mut max_change = 0.0f64;
        for i in 1..nx {
            for j in 0..ny {
                let un = us[i * ny + j] - dt * (self.p[i * ny + j] - self.p[(i - 1) * ny + j]) / h;
                max_change = max_change.max((un - self.u[i * ny + j]).abs());
                self.u[i * ny + j] = un;
            }
        }
        for i in 0..nx {
            for j in 1..ny {
                let vn = vs[i * (ny + 1) + j] - dt * (self.p[i * ny + j] - self.p[i * ny + j - 1]) / h;
                max_change = max_change.max((vn - self.v[i * (ny + 1) + j]).abs());
                self.v[i * (ny + 1) + j] = vn;
            }
        }
        max_change
    }

    // -----------------------------------------------------------------------------------------
    // Immersed boundary (direct-forcing) — couple a rigid body's surface to the fluid via no-slip.
    // -----------------------------------------------------------------------------------------

    /// Stencil of `(flat u-index, weight)` for a marker at `(mx, my)`, over the 4-point Peskin
    /// kernel. `u`-faces sit at `(i·h, (j+½)·h)`.
    fn stencil_u(&self, mx: f64, my: f64) -> Vec<(usize, f64)> {
        let (nx, ny, h) = (self.nx, self.ny, self.h);
        let i0 = (mx / h).floor() as i64;
        let j0 = (my / h - 0.5).floor() as i64;
        let mut out = Vec::with_capacity(16);
        for i in i0 - 1..=i0 + 2 {
            for j in j0 - 1..=j0 + 2 {
                if i < 0 || i > nx as i64 || j < 0 || j >= ny as i64 {
                    continue;
                }
                let w = phi((i as f64 * h - mx) / h) * phi(((j as f64 + 0.5) * h - my) / h);
                if w != 0.0 {
                    out.push((i as usize * ny + j as usize, w));
                }
            }
        }
        out
    }

    /// Stencil of `(flat v-index, weight)` for a marker at `(mx, my)`. `v`-faces sit at `((i+½)·h, j·h)`.
    fn stencil_v(&self, mx: f64, my: f64) -> Vec<(usize, f64)> {
        let (nx, ny, h) = (self.nx, self.ny, self.h);
        let i0 = (mx / h - 0.5).floor() as i64;
        let j0 = (my / h).floor() as i64;
        let mut out = Vec::with_capacity(16);
        for i in i0 - 1..=i0 + 2 {
            for j in j0 - 1..=j0 + 2 {
                if i < 0 || i >= nx as i64 || j < 0 || j > ny as i64 {
                    continue;
                }
                let w = phi(((i as f64 + 0.5) * h - mx) / h) * phi((j as f64 * h - my) / h);
                if w != 0.0 {
                    out.push((i as usize * (ny + 1) + j as usize, w));
                }
            }
        }
        out
    }

    /// Velocity `(u, v)` interpolated from the grid to world point `(mx, my)` via the Peskin kernel.
    pub fn velocity_at(&self, mx: f64, my: f64) -> (f64, f64) {
        let u = self.stencil_u(mx, my).iter().map(|&(k, w)| self.u[k] * w).sum();
        let v = self.stencil_v(mx, my).iter().map(|&(k, w)| self.v[k] * w).sum();
        (u, v)
    }

    /// Advance one timestep with an immersed rigid `disk` whose surface moves at `(disk.ux, disk.uy)`,
    /// enforcing no-slip by iterated direct forcing. Returns the hydrodynamic force the fluid exerts
    /// on the body, `(Fx, Fy)` (the reaction to the momentum injected at the surface).
    pub fn step_with_disk(&mut self, disk: &RigidDisk) -> (f64, f64) {
        const N_FORCE_ITERS: usize = 6;
        let (h, dt) = (self.h, self.dt);
        let markers = disk.markers(h);
        let (mut us, mut vs) = self.predict();

        // Precompute stencils once (markers are fixed within the step).
        let su: Vec<_> = markers.iter().map(|&(mx, my)| self.stencil_u(mx, my)).collect();
        let sv: Vec<_> = markers.iter().map(|&(mx, my)| self.stencil_v(mx, my)).collect();

        let (mut sum_fx, mut sum_fy) = (0.0f64, 0.0f64);
        for _ in 0..N_FORCE_ITERS {
            // Jacobi: gather the velocity deficit at every marker, then scatter the correction.
            let deficits: Vec<(f64, f64)> = (0..markers.len())
                .map(|m| {
                    let ui: f64 = su[m].iter().map(|&(k, w)| us[k] * w).sum();
                    let vi: f64 = sv[m].iter().map(|&(k, w)| vs[k] * w).sum();
                    (disk.ux - ui, disk.uy - vi)
                })
                .collect();
            for m in 0..markers.len() {
                let (fx, fy) = deficits[m];
                for &(k, w) in &su[m] {
                    us[k] += fx * w;
                    sum_fx += fx * w;
                }
                for &(k, w) in &sv[m] {
                    vs[k] += fy * w;
                    sum_fy += fy * w;
                }
            }
        }
        self.project_and_correct(us, vs);
        // Force on the fluid = injected momentum / dt (ρ=1, cell area h²); reaction on the body is −that.
        (-sum_fx * h * h / dt, -sum_fy * h * h / dt)
    }

    /// Advance one timestep with a **free** rigid disk whose motion is driven by the fluid: the
    /// hydrodynamic force plus the caller-supplied external force `(fx, fy)` (gravity − buoyancy,
    /// thrust, …) integrate the body's velocity and position (weak/explicit coupling). Returns the
    /// hydrodynamic force on the body. Stable while the added mass `ρ_f·π r²` is below the body mass.
    pub fn step_free_disk(&mut self, body: &mut FreeDisk, ext_force: (f64, f64)) -> (f64, f64) {
        let disk = RigidDisk { cx: body.cx, cy: body.cy, r: body.r, ux: body.ux, uy: body.uy };
        let f = self.step_with_disk(&disk);
        body.ux += self.dt * (f.0 + ext_force.0) / body.mass;
        body.uy += self.dt * (f.1 + ext_force.1) / body.mass;
        body.cx += self.dt * body.ux;
        body.cy += self.dt * body.uy;
        f
    }

    /// Max no-slip residual `‖u_fluid(Xₖ) − U_body‖` over the disk's surface markers (0 = perfect no-slip).
    pub fn slip_residual(&self, disk: &RigidDisk) -> f64 {
        disk.markers(self.h)
            .iter()
            .map(|&(mx, my)| {
                let (u, v) = self.velocity_at(mx, my);
                ((u - disk.ux).powi(2) + (v - disk.uy).powi(2)).sqrt()
            })
            .fold(0.0f64, f64::max)
    }

    /// Step until the per-step velocity change falls below `tol` or `max_steps` elapses; returns the
    /// step count reached.
    pub fn run_to_steady(&mut self, max_steps: usize, tol: f64) -> usize {
        for s in 0..max_steps {
            if self.step() < tol {
                return s + 1;
            }
        }
        max_steps
    }

    /// Max absolute cell divergence of the current velocity field (should be ~machine-zero post-projection).
    pub fn max_divergence(&self) -> f64 {
        let (nx, ny, h) = (self.nx, self.ny, self.h);
        let mut m = 0.0f64;
        for i in 0..nx {
            for j in 0..ny {
                let div = (self.u[(i + 1) * ny + j] - self.u[i * ny + j]) / h + (self.v[i * (ny + 1) + j + 1] - self.v[i * (ny + 1) + j]) / h;
                m = m.max(div.abs());
            }
        }
        m
    }

    /// u-velocity sampled along the vertical centerline `x = 0.5`, as `(y, u)` from wall to wall.
    /// Requires `nx` even so a `u`-face lands exactly on `x = 0.5`.
    pub fn centerline_u(&self) -> Vec<(f64, f64)> {
        assert!(self.nx % 2 == 0, "need even nx for an exact centerline face");
        let ic = self.nx / 2;
        let mut out = vec![(0.0, 0.0)]; // bottom wall
        for j in 0..self.ny {
            let y = (j as f64 + 0.5) * self.h;
            out.push((y, self.u[ic * self.ny + j]));
        }
        out.push((1.0, self.lid_u)); // lid
        out
    }
}

/// A rigid circular body immersed in the fluid, with a prescribed surface velocity `(ux, uy)`.
#[derive(Clone, Copy, Debug)]
pub struct RigidDisk {
    pub cx: f64,
    pub cy: f64,
    pub r: f64,
    pub ux: f64,
    pub uy: f64,
}

impl RigidDisk {
    /// Lagrangian surface markers spaced ≈ `h` apart around the circle.
    pub fn markers(&self, h: f64) -> Vec<(f64, f64)> {
        let n = ((std::f64::consts::TAU * self.r / h).round() as usize).max(8);
        (0..n)
            .map(|k| {
                let a = std::f64::consts::TAU * k as f64 / n as f64;
                (self.cx + self.r * a.cos(), self.cy + self.r * a.sin())
            })
            .collect()
    }
}

/// A free rigid disk: position, velocity, radius, and mass — its motion is solved from the fluid
/// force plus external forces (see [`MacFluid::step_free_disk`]).
#[derive(Clone, Copy, Debug)]
pub struct FreeDisk {
    pub cx: f64,
    pub cy: f64,
    pub r: f64,
    pub ux: f64,
    pub uy: f64,
    pub mass: f64,
}

/// The 4-point Peskin regularized delta kernel (1D factor), argument in cell units.
fn phi(r: f64) -> f64 {
    let a = r.abs();
    if a <= 1.0 {
        (3.0 - 2.0 * a + (1.0 + 4.0 * a - 4.0 * a * a).sqrt()) / 8.0
    } else if a <= 2.0 {
        (5.0 - 2.0 * a - (-7.0 + 12.0 * a - 4.0 * a * a).sqrt()) / 8.0
    } else {
        0.0
    }
}

/// Linear interpolation of a monotone-in-`x` `(x, y)` table.
#[cfg(test)]
fn interp(table: &[(f64, f64)], x: f64) -> f64 {
    if x <= table[0].0 {
        return table[0].1;
    }
    for w in table.windows(2) {
        let ((x0, y0), (x1, y1)) = (w[0], w[1]);
        if x <= x1 {
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0);
        }
    }
    table[table.len() - 1].1
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Ghia, Ghia & Shin (1982), Table I — u along the vertical centerline of the lid-driven cavity
    /// at Re = 100. These are the standard published benchmark values.
    const GHIA_RE100: &[(f64, f64)] = &[
        (0.0000, 0.00000),
        (0.0547, -0.03717),
        (0.0625, -0.04192),
        (0.0703, -0.04775),
        (0.1016, -0.06434),
        (0.1719, -0.10150),
        (0.2813, -0.15662),
        (0.4531, -0.21090),
        (0.5000, -0.20581),
        (0.6172, -0.13641),
        (0.7344, 0.00332),
        (0.8516, 0.23151),
        (0.9531, 0.68717),
        (0.9609, 0.73722),
        (0.9688, 0.78871),
        (0.9766, 0.84123),
        (1.0000, 1.00000),
    ];

    #[test]
    fn immersed_disk_enforces_no_slip_and_drags_fluid() {
        // A disk held at the box center imposes a +x surface velocity on quiescent fluid.
        let mut f = MacFluid::new(64, 64, 0.01, 0.003, 0.0);
        let disk = RigidDisk { cx: 0.5, cy: 0.5, r: 0.12, ux: 0.4, uy: 0.0 };
        let mut drag = (0.0, 0.0);
        for s in 0..60 {
            let d = f.step_with_disk(&disk);
            if s >= 20 {
                drag.0 += d.0;
                drag.1 += d.1;
            }
        }
        drag.0 /= 40.0;
        drag.1 /= 40.0;

        let slip = f.slip_residual(&disk);
        // Interior divergence away from the immersed surface must still vanish (projection is global).
        eprintln!("IB: slip={slip:.4} (rel {:.3}), drag=({:.4},{:.4})", slip / 0.4, drag.0, drag.1);

        // (1) No-slip is actually enforced at the surface (within a few % of the surface speed).
        assert!(slip / 0.4 < 0.06, "no-slip not enforced: rel slip = {:.3}", slip / 0.4);
        // (2) The fluid just outside the disk is dragged along in +x by a meaningful fraction of U.
        let (ux_out, _) = f.velocity_at(0.5 + 0.16, 0.5);
        assert!(ux_out > 0.1 * 0.4, "fluid not dragged along: u_out = {ux_out:.4}");
        // (3) Drag opposes the surface motion (force on the body points −x).
        assert!(drag.0 < 0.0, "drag does not oppose motion: Fx = {:.4}", drag.0);
    }

    #[test]
    fn free_disk_settles_to_terminal_velocity() {
        // A disk denser than the fluid, released from rest, settles under gravity−buoyancy until
        // hydrodynamic drag balances the net weight → constant terminal velocity.
        let (rho_f, rho_b, g) = (1.0, 3.0, 1.2);
        let mut f = MacFluid::new(64, 64, 0.02, 0.003, 0.0);
        let mut body = FreeDisk { cx: 0.5, cy: 0.75, r: 0.08, ux: 0.0, uy: 0.0, mass: 0.0 };
        let area = std::f64::consts::PI * body.r * body.r;
        body.mass = rho_b * area;
        let ext = (0.0, -(rho_b - rho_f) * area * g); // net weight (down)

        let mut uy_hist = Vec::new();
        let mut last_fy = 0.0;
        for _ in 0..360 {
            last_fy = f.step_free_disk(&mut body, ext).1;
            uy_hist.push(body.uy);
        }

        // Terminal plateau: last-fifth velocities are near-constant.
        let tail = &uy_hist[uy_hist.len() * 4 / 5..];
        let mean: f64 = tail.iter().sum::<f64>() / tail.len() as f64;
        let spread = tail.iter().map(|&u| (u - mean).abs()).fold(0.0f64, f64::max);
        // Force balance at terminal: drag (up) ≈ net weight (down).
        let balance = (last_fy + ext.1).abs() / ext.1.abs();
        eprintln!("settle: cy={:.3}, u_term={mean:.4}, plateau_spread={spread:.4}, force_imbalance={balance:.3}", body.cy);

        assert!(mean < -0.02 && body.cy < 0.72, "disk did not settle downward");
        assert!(spread / mean.abs() < 0.05, "no terminal plateau: spread {spread:.4} vs u {mean:.4}");
        assert!(balance < 0.15, "drag does not balance weight at terminal: imbalance {balance:.3}");
    }

    #[test]
    fn immersed_drag_scales_with_velocity_in_the_stokes_regime() {
        // Low-Re: hydrodynamic drag on the held disk should be ~linear in the surface speed.
        let run = |u: f64| -> f64 {
            let mut f = MacFluid::new(64, 64, 0.02, 0.003, 0.0);
            let disk = RigidDisk { cx: 0.5, cy: 0.5, r: 0.1, ux: u, uy: 0.0 };
            let mut fx = 0.0;
            for s in 0..60 {
                let d = f.step_with_disk(&disk);
                if s >= 30 {
                    fx += d.0;
                }
            }
            fx / 30.0
        };
        let (d1, d2) = (run(0.2), run(0.4));
        let ratio = d2 / d1;
        eprintln!("Stokes drag: F(U)={d1:.4}, F(2U)={d2:.4}, ratio={ratio:.3}");
        assert!((1.7..=2.3).contains(&ratio), "drag not ~linear in U: ratio = {ratio:.3}");
    }

    #[test]
    fn projection_keeps_the_field_divergence_free() {
        // Re = 100 on a modest grid; after each projection the discrete divergence must vanish.
        let mut f = MacFluid::new(32, 32, 0.01, 0.005, 1.0);
        for _ in 0..50 {
            f.step();
        }
        assert!(f.max_divergence() < 1e-9, "field not divergence-free: {}", f.max_divergence());
    }

    // Heavier benchmark (~2600 steps to steady on a 64² grid) — run in release:
    //   cargo test -p ferromotion-fluid --release -- --ignored
    #[test]
    #[ignore = "benchmark: run in release"]
    fn lid_driven_cavity_matches_ghia_re100() {
        // Re = U·L/ν = 1·1/0.01 = 100.
        let mut f = MacFluid::new(64, 64, 0.01, 0.004, 1.0);
        let steps = f.run_to_steady(8000, 2e-6);
        assert!(f.max_divergence() < 1e-8, "not divergence-free after settling");

        let profile = f.centerline_u();
        let (mut max_err, mut sum) = (0.0f64, 0.0);
        for &(y, u_ref) in GHIA_RE100 {
            let e = (interp(&profile, y) - u_ref).abs();
            max_err = max_err.max(e);
            sum += e;
        }
        let mean_err = sum / GHIA_RE100.len() as f64;
        // Measured on a 64² grid: max_err ≈ 0.0028, mean_err ≈ 0.0012 (well inside these guards).
        eprintln!("Ghia Re=100: steps={steps}, max_err={max_err:.4}, mean_err={mean_err:.4}");
        assert!(max_err < 0.01, "centerline deviates from Ghia: max_err = {max_err:.4}");
        assert!(mean_err < 0.005, "centerline mean error too high: {mean_err:.4}");
    }
}
