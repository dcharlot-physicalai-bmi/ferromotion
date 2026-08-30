//! **An elastic surface, so the gel can move where nothing is touching it.**
//!
//! The height field in [`crate`] is geometric: the surface conforms to the indenter and is exactly zero
//! outside its footprint. A real gel does not behave that way, and the crate documentation measures the
//! difference against a 3D elastic solve. The two parts it cannot represent are a decaying tail outside
//! the contact, and an upward **bulge** in a ring around it, worth 6.56% of the press depth at ν = 0.49,
//! which a softplus can never produce because it is non-negative everywhere.
//!
//! This module closes that. The gel is treated as a linear elastic half-layer characterised by one
//! [`Influence`] function, and the contact is *solved* rather than assumed: the pressure comes out
//! non-negative, the contact patch is found rather than prescribed, and the surface is free to rise
//! outside it.
//!
//! # Why a convolution is allowed here
//!
//! Superposition is exact for a linear elastic solid and fails under large deformation, and a GelSight
//! press is around 15% of the gel thickness, which is not obviously small. So it was measured against
//! [`ferromotion-fem`](../ferromotion_fem) rather than assumed:
//!
//! - **Superposition.** The response to two point loads, predicted by shifting and adding the response
//!   to one, matched a direct solve of both to **0.38% rms** of the peak.
//! - **Linearity to a real press.** Sweeping a distributed disc load from a negligible indentation up
//!   to **15.8% of the gel thickness**, the departure from the linear prediction stayed within **1.1%**,
//!   and was non-monotone rather than growing, so it is at the level of the settling tolerance.
//!
//! # How accurate it actually is
//!
//! End to end, against a direct 3D solve of the loads this module computes, on a gel of thickness 0.5:
//!
//! | press depth | % of thickness | rms error vs the 3D solve |
//! |---|---|---|
//! | 0.02 | 4% | **0.70%** |
//! | 0.04 | 8% | 1.04% |
//! | 0.08 | 16% | 1.79% |
//! | 0.12 | 24% | 2.53% |
//!
//! So roughly 1-2% through the range a real sensor works in, degrading smoothly past it. The error
//! tracking press depth is expected rather than a defect: it is the linear response being asked to
//! describe a deformation that is no longer small, and it is why the table stops where the measured
//! linear range does. Nothing here detects that you have gone past it, so if a press is much deeper
//! than a fifth of the gel thickness, treat the surface as indicative.
//!
//! # The influence function is an input
//!
//! [`Influence`] is data, not theory. This crate does not derive it, because the useful closed forms do
//! not describe the object: Boussinesq and Hertz assume an infinite half-space, while a GelSight gel is
//! a thin layer bonded to a rigid backing, and that difference is exactly what produces the bulge. So
//! the kernel is supplied by the caller, measured on the geometry actually being modelled. The
//! `elastic_gel_matches_a_direct_elastic_solve` test shows the measurement end to end with
//! `ferromotion-fem`: load one surface node, relax, read the surface.
//!
//! Because the near field of a surface influence function is singular in the continuum, the discrete
//! kernel depends on the cell it is averaged over. [`Influence`] therefore carries the cell size it was
//! measured at, and [`GelSim::elastic_contact`] refuses a kernel that does not match the gel's own grid
//! rather than silently rescaling one.

use crate::{GelSim, Indenter};

/// A radially symmetric surface influence function for the gel.
///
/// `g[k]` is the **downward** surface deflection at radius `k·cell` produced by a unit downward force
/// spread over one grid cell at the origin. Units are metres per newton.
///
/// `g[0]` is the self-influence of a loaded cell and must be positive: pressing down at a point moves
/// that point down. Later entries may be negative, and for a bonded near-incompressible layer they
/// will be, which is the whole reason this module exists.
#[derive(Clone, Debug)]
pub struct Influence {
    g: Vec<f64>,
    cell: f64,
}

impl Influence {
    /// Build from samples taken at `r = k·cell`.
    ///
    /// Returns `None` unless `cell > 0`, there are at least two samples, and `g[0] > 0`. That last
    /// check is worth having: a sign error in the measurement, which is easy when one convention calls
    /// downward positive and the other calls it negative, otherwise produces a solve that pushes the
    /// indenter away and converges to a contact patch of nothing.
    pub fn from_samples(g: Vec<f64>, cell: f64) -> Option<Influence> {
        (cell > 0.0 && g.len() >= 2 && g[0] > 0.0 && g.iter().all(|v| v.is_finite())).then_some(Influence { g, cell })
    }

    /// The cell size the kernel was measured at.
    pub fn cell(&self) -> f64 {
        self.cell
    }

    /// Reach of the kernel in metres; beyond it the response is taken as zero.
    pub fn reach(&self) -> f64 {
        (self.g.len() - 1) as f64 * self.cell
    }

    /// Deflection per unit cell load at radius `r`, linearly interpolated, zero past [`Self::reach`].
    pub fn at(&self, r: f64) -> f64 {
        let t = r / self.cell;
        let k = t.floor();
        if k < 0.0 {
            return self.g[0];
        }
        let k = k as usize;
        if k + 1 >= self.g.len() {
            return 0.0;
        }
        let f = t - k as f64;
        self.g[k] * (1.0 - f) + self.g[k + 1] * f
    }
}

/// The solved elastic contact: a surface, the pressure that produced it, and what was in contact.
#[derive(Clone, Debug)]
pub struct ElasticContact {
    /// Downward surface displacement on the gel grid, row-major, same layout as
    /// [`GelSim::deformation`]. **May be negative**, where the surface has risen.
    pub h: Vec<f64>,
    /// Contact force per node in newtons, non-negative and zero outside the contact patch.
    pub load: Vec<f64>,
    /// Exact `∂h/∂depth` at the converged contact set, for gradient-based inference.
    pub dh_ddepth: Vec<f64>,
    /// Total normal force carried by the contact.
    pub total_force: f64,
    /// How many nodes ended up in contact.
    pub contact_nodes: usize,
    /// Whether the complementarity iteration met `tol`.
    pub converged: bool,
}

impl GelSim {
    /// Solve the elastic normal contact of a spherical indenter against the gel.
    ///
    /// The unknown is the nodal contact load. Where a node carries load its surface must lie exactly on
    /// the indenter; where it carries none the surface must stay at or below it. That complementarity is
    /// solved by projected Gauss-Seidel, which is why the pressure comes out non-negative and the
    /// contact patch is an output rather than an assumption.
    ///
    /// Only nodes the undeformed indenter already penetrates can carry load. That is a valid restriction
    /// rather than a shortcut: pressing a convex indenter into a surface that sinks under it gives a
    /// contact patch *smaller* than the geometric intersection, so the geometric set is a superset of
    /// the true one. It is also what keeps this fast, since the solve is over the patch while the
    /// surface is evaluated over the whole grid.
    ///
    /// `tol` is **relative** to the largest nodal load, since that scale follows the press depth and
    /// the kernel's units and is not something a caller could bound absolutely. `1e-12` is a good value.
    ///
    /// Returns `None` if the kernel was measured at a different cell size than this gel's grid.
    pub fn elastic_contact(&self, ind: &Indenter, inf: &Influence, iters: usize, tol: f64) -> Option<ElasticContact> {
        let n = self.n;
        let cell = self.cell();
        if (inf.cell() - cell).abs() > 1e-9 * cell.max(1.0) {
            return None; // a kernel for another grid is not this gel's kernel
        }

        // candidate patch: where the undeformed sphere already overlaps the flat surface
        let mut patch: Vec<usize> = Vec::new();
        let mut gap: Vec<f64> = Vec::new(); // target penetration at each candidate
        for iy in 0..n {
            for ix in 0..n {
                let (x, y) = (self.coord(ix), self.coord(iy));
                let rho2 = (x - ind.cx).powi(2) + (y - ind.cy).powi(2);
                if rho2 < ind.radius * ind.radius {
                    let d = (ind.radius * ind.radius - rho2).sqrt() - ind.radius + ind.depth;
                    if d > 0.0 {
                        patch.push(iy * n + ix);
                        gap.push(d);
                    }
                }
            }
        }
        let m = patch.len();
        let mut load = vec![0.0; n * n];
        let mut dh = vec![0.0; n * n];
        if m == 0 {
            return Some(ElasticContact { h: vec![0.0; n * n], load, dh_ddepth: dh, total_force: 0.0, contact_nodes: 0, converged: true });
        }

        // dense influence over the patch: small, since the patch is small
        let xy = |i: usize| (self.coord(i % n), self.coord(i / n));
        let mut k = vec![0.0; m * m];
        for a in 0..m {
            let (xa, ya) = xy(patch[a]);
            for b in 0..m {
                let (xb, yb) = xy(patch[b]);
                k[a * m + b] = inf.at(((xa - xb).powi(2) + (ya - yb).powi(2)).sqrt());
            }
        }

        // Projected Gauss-Seidel on the complementarity problem.
        //
        // `tol` is RELATIVE to the largest nodal load, deliberately. The load scale is roughly
        // `depth / g(0)`, so it moves with both the press and the kernel's units and is not something a
        // caller can pick an absolute bound for: at a depth of 0.12 with a kernel of order 1e-5 the
        // loads are ~1e4, where an absolute 1e-15 sits below f64 resolution and can never be met.
        let mut p = vec![0.0; m];
        let mut converged = false;
        for _ in 0..iters.max(1) {
            let mut worst = 0.0f64;
            let mut scale = 0.0f64;
            for a in 0..m {
                let mut acc = 0.0;
                for b in 0..m {
                    if b != a {
                        acc += k[a * m + b] * p[b];
                    }
                }
                let want = ((gap[a] - acc) / k[a * m + a]).max(0.0);
                worst = worst.max((want - p[a]).abs());
                scale = scale.max(want.abs());
                p[a] = want;
            }
            if worst <= tol * scale.max(f64::MIN_POSITIVE) {
                converged = true;
                break;
            }
        }

        // exact d/d(depth) at the converged active set: G_AA x = 1, then dh = G x
        let active: Vec<usize> = (0..m).filter(|&a| p[a] > 0.0).collect();
        let mut dp = vec![0.0; m];
        for _ in 0..iters.max(1) {
            let mut worst = 0.0f64;
            let mut scale = 0.0f64;
            for &a in &active {
                let mut acc = 0.0;
                for &b in &active {
                    if b != a {
                        acc += k[a * m + b] * dp[b];
                    }
                }
                let want = (1.0 - acc) / k[a * m + a];
                worst = worst.max((want - dp[a]).abs());
                scale = scale.max(want.abs());
                dp[a] = want;
            }
            if worst <= tol * scale.max(f64::MIN_POSITIVE) {
                break;
            }
        }

        // evaluate the surface everywhere from the patch loads
        let mut h = vec![0.0; n * n];
        for iy in 0..n {
            for ix in 0..n {
                let (x, y) = (self.coord(ix), self.coord(iy));
                let (mut acc, mut dacc) = (0.0, 0.0);
                for a in 0..m {
                    let (xa, ya) = xy(patch[a]);
                    let w = inf.at(((x - xa).powi(2) + (y - ya).powi(2)).sqrt());
                    acc += w * p[a];
                    dacc += w * dp[a];
                }
                h[iy * n + ix] = acc;
                dh[iy * n + ix] = dacc;
            }
        }
        for a in 0..m {
            load[patch[a]] = p[a];
        }
        Some(ElasticContact {
            h,
            load,
            dh_ddepth: dh,
            total_force: p.iter().sum(),
            contact_nodes: active.len(),
            converged,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The gel these fixtures use. `cell` is 0.1, matching [`MEASURED`].
    fn gel() -> GelSim {
        GelSim { n: 21, extent: 1.0, beta: 0.02 }
    }

    /// A **measured** influence function, not an invented one.
    ///
    /// Produced by `ferromotion-fem`: a 20x20x5 slab (cell 0.1, thickness 0.5, ν = 0.45, μ = 1e5)
    /// bonded to a rigid backing, one unit downward load on the centre surface node, relaxed to rest,
    /// the surface read and averaged over the four axes. `elastic_gel_matches_a_direct_elastic_solve`
    /// re-measures it live and checks this fixture still matches.
    ///
    /// Note entries 5 through 8 are **negative**: that is the bulge ring, and it is why the geometric
    /// height field in [`crate`] cannot represent a real gel. The last two entries are lifted by the
    /// free edge of a finite 2x2 domain rather than by gel physics.
    ///
    /// An earlier version of these tests used a hand-drawn "decaying with a ring" profile instead. It
    /// produced an influence matrix with 47 of 133 eigenvalues non-positive, so it was not a compliance
    /// at all and the solve could not converge. A plausible shape is not a valid kernel.
    const MEASURED: [f64; 11] = [
        2.436512439e-5,
        6.823104524e-6,
        1.845690022e-6,
        7.773714098e-7,
        1.248365075e-7,
        -7.950505129e-9,
        -8.276880335e-8,
        -5.413622031e-8,
        -2.464986094e-8,
        2.394233059e-8,
        7.294871313e-8,
    ];

    fn measured() -> Influence {
        Influence::from_samples(MEASURED.to_vec(), 0.1).expect("measured kernel is valid")
    }

    #[test]
    fn a_kernel_with_the_wrong_sign_is_refused() {
        // g[0] <= 0 means pressing down moved the loaded point up: a measurement sign error, and it
        // would otherwise converge to a contact patch of nothing rather than fail.
        assert!(Influence::from_samples(vec![-1.0, 0.5], 0.1).is_none());
        assert!(Influence::from_samples(vec![0.0, 0.5], 0.1).is_none());
        assert!(Influence::from_samples(vec![1.0], 0.1).is_none(), "one sample cannot be interpolated");
        assert!(Influence::from_samples(vec![1.0, 0.5], 0.0).is_none());
        assert!(Influence::from_samples(vec![1.0, f64::NAN], 0.1).is_none());
        assert!(Influence::from_samples(vec![1.0, 0.5], 0.1).is_some());
    }

    #[test]
    fn a_kernel_for_another_grid_is_refused() {
        let g = gel();
        let wrong = Influence::from_samples(MEASURED.to_vec(), g.cell() * 2.0).expect("valid samples, wrong grid");
        assert!(g.elastic_contact(&Indenter { cx: 0.0, cy: 0.0, radius: 0.5, depth: 0.12 }, &wrong, 200, 1e-12).is_none());
    }

    /// The whole point: an elastic surface may rise where the geometric one cannot.
    #[test]
    fn the_elastic_surface_can_bulge_upward() {
        let g = gel();
        let inf = measured();
        let ind = Indenter { cx: 0.0, cy: 0.0, radius: 0.5, depth: 0.12 };
        let c = g.elastic_contact(&ind, &inf, 4000, 1e-12).expect("kernel matches the grid");
        assert!(c.converged, "contact solve did not converge");
        let lowest = c.h.iter().cloned().fold(f64::INFINITY, f64::min);
        assert!(lowest < 0.0, "an elastic surface must be able to rise outside the contact, min h = {lowest:e}");
        // and the geometric model, on the identical press, cannot
        let (hg, _) = g.deformation(&ind);
        assert!(hg.iter().cloned().fold(f64::INFINITY, f64::min) >= 0.0, "the geometric field is non-negative by construction");
    }

    /// Complementarity: load is non-negative, lives only on the patch, and where it acts the surface
    /// sits exactly on the indenter.
    #[test]
    fn the_contact_solve_satisfies_complementarity() {
        let g = gel();
        let inf = measured();
        let ind = Indenter { cx: 0.03, cy: -0.05, radius: 0.5, depth: 0.12 };
        let c = g.elastic_contact(&ind, &inf, 6000, 1e-12).expect("kernel matches the grid");
        assert!(c.converged);
        assert!(c.contact_nodes > 8, "expected a real patch, got {}", c.contact_nodes);
        assert!(c.load.iter().all(|&p| p >= 0.0), "contact load must be non-negative");
        assert!(c.total_force > 0.0);

        let n = g.n;
        let mut worst_touch = 0.0f64;
        let mut worst_overlap = 0.0f64;
        for iy in 0..n {
            for ix in 0..n {
                let k = iy * n + ix;
                let (x, y) = (g.coord(ix), g.coord(iy));
                let rho2 = (x - ind.cx).powi(2) + (y - ind.cy).powi(2);
                let target = if rho2 < ind.radius * ind.radius {
                    (ind.radius * ind.radius - rho2).sqrt() - ind.radius + ind.depth
                } else {
                    f64::NEG_INFINITY
                };
                if c.load[k] > 0.0 {
                    // loaded => the surface lies ON the indenter
                    worst_touch = worst_touch.max((c.h[k] - target).abs());
                } else if target > f64::NEG_INFINITY {
                    // Unloaded => the surface must lie at or BELOW the indenter, which in this sign
                    // convention is h >= target: `h` is downward displacement, so a larger `h` is a
                    // surface further from the sphere, not closer to it.
                    worst_overlap = worst_overlap.max(target - c.h[k]);
                }
            }
        }
        assert!(worst_touch < 1e-9, "loaded nodes must sit on the indenter, worst {worst_touch:e}");
        assert!(worst_overlap < 1e-9, "unloaded nodes must stay at or below the indenter (h >= target), worst {worst_overlap:e}");
    }

    /// The reported `∂h/∂depth` is the real derivative, which is what keeps the crate differentiable.
    #[test]
    fn the_depth_gradient_matches_finite_differences() {
        let g = gel();
        let inf = measured();
        let base = Indenter { cx: 0.0, cy: 0.0, radius: 0.5, depth: 0.12 };
        let c = g.elastic_contact(&base, &inf, 6000, 1e-12).expect("kernel matches the grid");
        let eps = 1e-7;
        let mut up = base;
        up.depth += eps;
        let mut dn = base;
        dn.depth -= eps;
        let cu = g.elastic_contact(&up, &inf, 6000, 1e-12).unwrap();
        let cd = g.elastic_contact(&dn, &inf, 6000, 1e-12).unwrap();
        let scale = c.dh_ddepth.iter().cloned().fold(0.0f64, |a, b| a.max(b.abs()));
        assert!(scale > 0.1, "gradient should be O(1), got {scale}");
        let mut worst = 0.0f64;
        for k in 0..c.h.len() {
            let fd = (cu.h[k] - cd.h[k]) / (2.0 * eps);
            worst = worst.max((c.dh_ddepth[k] - fd).abs());
        }
        assert!(worst < 1e-3 * scale, "dh/ddepth vs finite difference: worst {worst:e} on a scale of {scale:e}");
    }

    /// End-to-end against a real 3D elastic solve, with `ferromotion-fem` as a dev-dependency.
    ///
    /// Three things at once, because they only mean something together:
    ///
    /// 1. **The fixture is honest.** [`MEASURED`] is re-measured live and must still match.
    /// 2. **The kernel is a compliance.** Its influence matrix over the contact patch must be positive
    ///    definite. A hand-drawn kernel failed this and no behavioural test noticed.
    /// 3. **The convolution is right.** Take the loads this module solves for, apply those same loads to
    ///    the FEM slab, relax, and the two surfaces must agree. That is what says the contact solve and
    ///    the surface evaluation describe the elastic body they claim to.
    ///
    /// Ignored by default: it relaxes two 3D solves to rest and takes minutes in the debug profile the
    /// main suite uses. It runs in the release `--ignored` lane.
    #[test]
    #[ignore = "relaxes two 3D FEM solves to rest; run in the release --ignored lane"]
    fn elastic_gel_matches_a_direct_elastic_solve() {
        use ferromotion_fem::FemSim;
        use nalgebra::Vector3;

        const NXY: usize = 20;
        const NZ: usize = 5;
        const EXTENT: f64 = 1.0;
        let h = 2.0 * EXTENT / NXY as f64;
        let thickness = NZ as f64 * h;
        let (nu, mu) = (0.45f64, 1.0e5f64);
        let idx = |ix: usize, iy: usize, iz: usize| (iz * (NXY + 1) + iy) * (NXY + 1) + ix;

        // the slab, matching the gel's grid so the kernel transfers without rescaling
        let slab = || {
            let lambda = 2.0 * mu * nu / (1.0 - 2.0 * nu);
            let nv = (NXY + 1) * (NXY + 1) * (NZ + 1);
            let mut s = FemSim::box_grid(NXY, NXY, NZ, h, 0.2 / nv as f64, mu, lambda, 1e-9);
            s.dt = 0.5 * s.stable_timestep();
            s.gravity = Vector3::zeros();
            s.floor = None;
            s.damping_rate = 400.0;
            for p in s.x.iter_mut() {
                p.x -= EXTENT;
                p.y -= EXTENT;
                p.z -= thickness;
            }
            for i in 0..s.x.len() {
                if s.x[i].z <= -thickness + 1e-9 {
                    s.pinned[i] = true; // bonded backing
                }
            }
            s
        };
        let relax = |s: &mut FemSim| {
            let mut last = f64::NAN;
            for _ in 0..4000 {
                for _ in 0..200 {
                    s.step();
                }
                assert!(s.x.iter().all(|p| p.z.is_finite()), "the slab diverged");
                let e = s.energy();
                if (e - last).abs() < 1e-13 * e.abs().max(1e-14) {
                    return;
                }
                last = e;
            }
        };

        // 1. re-measure the influence function
        let mid = NXY / 2;
        let unit = 1.0e-2f64;
        let mut s = slab();
        s.external = vec![Vector3::zeros(); s.x.len()];
        s.external[idx(mid, mid, NZ)].z = -unit;
        relax(&mut s);
        let live: Vec<f64> = (0..=mid)
            .map(|k| {
                let acc: f64 = [(mid + k, mid), (mid - k, mid), (mid, mid + k), (mid, mid - k)]
                    .iter()
                    .map(|&(ix, iy)| -s.x[idx(ix, iy, NZ)].z)
                    .sum();
                acc / 4.0 / unit
            })
            .collect();
        let peak = MEASURED[0];
        let drift = live.iter().zip(MEASURED.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max);
        assert!(
            drift < 1e-3 * peak,
            "the MEASURED fixture no longer matches a live measurement: worst {drift:e} on a peak of {peak:e}"
        );
        // the negative ring is the whole point, so assert it is really there
        assert!(live[6] < 0.0, "the bulge ring should be negative, got g[6] = {:e}", live[6]);

        // 2. the kernel must be a compliance over the patch it will be used on
        let g = gel();
        let inf = Influence::from_samples(live.clone(), h).expect("live kernel is valid");
        // depth 0.08 is 16% of the 0.5 thickness: a real GelSight press, and the top of the range
        // where linearity was measured to hold. Accuracy against depth is tabulated in the module docs.
        let ind = Indenter { cx: 0.0, cy: 0.0, radius: 0.5, depth: 0.08 };
        let c = g.elastic_contact(&ind, &inf, 20_000, 1e-12).expect("kernel matches the grid");
        assert!(c.converged, "the contact solve did not converge, which a non-positive-definite kernel causes");
        assert!(c.contact_nodes > 10, "expected a real patch, got {}", c.contact_nodes);

        // 3. apply this module's own solved loads to the slab and compare surfaces
        let mut t = slab();
        t.external = vec![Vector3::zeros(); t.x.len()];
        for iy in 0..g.n {
            for ix in 0..g.n {
                let load = c.load[iy * g.n + ix];
                if load != 0.0 {
                    t.external[idx(ix, iy, NZ)].z = -load;
                }
            }
        }
        relax(&mut t);
        let mut sq = 0.0f64;
        let mut cnt = 0.0f64;
        let mut hpeak = 0.0f64;
        for iy in 0..g.n {
            for ix in 0..g.n {
                let mine = c.h[iy * g.n + ix];
                let theirs = -t.x[idx(ix, iy, NZ)].z;
                sq += (mine - theirs).powi(2);
                cnt += 1.0;
                hpeak = hpeak.max(theirs.abs());
            }
        }
        let rms = (sq / cnt).sqrt();
        eprintln!("elastic gel vs direct FEM: rms {rms:.4e} on a peak of {hpeak:.4e} ({:.3}%)", 100.0 * rms / hpeak);
        // measured 1.788% at this depth; the bound is that plus headroom, not a guess
        assert!(
            rms < 0.025 * hpeak,
            "the convolution must reproduce the direct solve of its own loads: rms {rms:e} vs peak {hpeak:e} ({:.3}%)",
            100.0 * rms / hpeak
        );
    }

}
