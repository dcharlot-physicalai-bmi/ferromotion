//! **Bundle adjustment** — the joint nonlinear refinement of camera poses *and* 3-D structure that closes
//! every structure-from-motion / visual-SLAM pipeline. After the two-view bootstrap ([`crate::essential`] /
//! [`crate::homography`]), pose-by-resection ([`crate::pnp`]), and triangulation give an initial
//! reconstruction, bundle adjustment minimizes the **total reprojection error** `Σ ‖π(Rᵢ Xⱼ + tᵢ) − xᵢⱼ‖²`
//! over *all* camera poses and *all* points at once — the maximum-likelihood reconstruction under Gaussian
//! pixel noise. It is the single most important optimization in geometric vision.
//!
//! This is a calibrated (normalized-coordinate) Levenberg–Marquardt bundle adjuster on the `SE(3)×ℝ³`
//! manifold: each camera perturbs by a left twist (analytic `[I | −[P_c]×]` Jacobian, shared with
//! [`crate::pnp`]), each point by a Euclidean step (`∂P_c/∂X = R`). Gauge freedom is removed by holding a
//! set of reference cameras fixed (the two bootstrap views fix the world frame *and* the scale). Verified:
//! from a perturbed initialization it recovers the exact poses and structure on noise-free data (the
//! oracle), drives the reprojection error to zero, and lowers it well below the initialization under pixel
//! noise. Pure `nalgebra` → WASM-clean.

use nalgebra::{DMatrix, DVector, Matrix2x3, Matrix3, Matrix3x6, Vector2, Vector3};

fn skew(v: &Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(0.0, -v.z, v.y, v.z, 0.0, -v.x, -v.y, v.x, 0.0)
}
fn so3_exp(phi: &Vector3<f64>) -> Matrix3<f64> {
    let theta = phi.norm();
    if theta < 1e-12 {
        return Matrix3::identity() + skew(phi);
    }
    let k = phi / theta;
    let kx = skew(&k);
    Matrix3::identity() + theta.sin() * kx + (1.0 - theta.cos()) * kx * kx
}

/// A camera pose `X_cam = R·X_world + t`.
#[derive(Clone, Copy, Debug)]
pub struct Camera {
    pub r: Matrix3<f64>,
    pub t: Vector3<f64>,
}

/// One observation: `point` seen in `cam` at normalized image coordinate `px`.
#[derive(Clone, Copy, Debug)]
pub struct Observation {
    pub cam: usize,
    pub point: usize,
    pub px: Vector2<f64>,
}

/// A bundle-adjustment problem: camera poses, 3-D points, observations, and the fixed (gauge) cameras.
#[derive(Clone, Debug)]
pub struct BundleAdjustment {
    pub cameras: Vec<Camera>,
    pub points: Vec<Vector3<f64>>,
    pub observations: Vec<Observation>,
    /// Cameras held constant to fix the gauge (frame + scale) — typically the two bootstrap views.
    pub fixed: Vec<bool>,
}

impl BundleAdjustment {
    /// Root-mean-square reprojection error over all observations.
    pub fn reprojection_rms(&self) -> f64 {
        let mut acc = 0.0;
        for o in &self.observations {
            let pc = self.cameras[o.cam].r * self.points[o.point] + self.cameras[o.cam].t;
            acc += (Vector2::new(pc.x / pc.z, pc.y / pc.z) - o.px).norm_squared();
        }
        (acc / self.observations.len() as f64).sqrt()
    }

    /// Refine all free cameras and all points by Levenberg–Marquardt. Returns the final reprojection RMS.
    pub fn optimize(&mut self, iters: usize) -> f64 {
        // parameter layout: free cameras (6 each) then points (3 each)
        let cam_slot: Vec<Option<usize>> = {
            let mut s = Vec::with_capacity(self.cameras.len());
            let mut next = 0;
            for i in 0..self.cameras.len() {
                if self.fixed.get(i).copied().unwrap_or(false) {
                    s.push(None);
                } else {
                    s.push(Some(next));
                    next += 6;
                }
            }
            s
        };
        let num_free = cam_slot.iter().flatten().count();
        let pt_base = 6 * num_free; // points start after all free-camera parameters
        let p = pt_base + 3 * self.points.len();

        let mut lambda = 1e-3;
        let mut cost = self.sse();

        for _ in 0..iters {
            let mut h = DMatrix::<f64>::zeros(p, p);
            let mut g = DVector::<f64>::zeros(p);
            for o in &self.observations {
                let cam = self.cameras[o.cam];
                let x = self.points[o.point];
                let pc = cam.r * x + cam.t;
                let z = pc.z;
                if z.abs() < 1e-9 {
                    continue;
                }
                let res = Vector2::new(pc.x / z, pc.y / z) - o.px;
                let dpi = Matrix2x3::new(1.0 / z, 0.0, -pc.x / (z * z), 0.0, 1.0 / z, -pc.y / (z * z));
                // point block (always free): 3×3 Hessian, 3-gradient
                let jx = dpi * cam.r; // 2×3
                let px = pt_base + 3 * o.point;
                let hxx = jx.transpose() * jx; // 3×3
                let gx = jx.transpose() * res; // 3
                for a in 0..3 {
                    g[px + a] += gx[a];
                    for b in 0..3 {
                        h[(px + a, px + b)] += hxx[(a, b)];
                    }
                }
                // camera block (if free)
                if let Some(cslot) = cam_slot[o.cam] {
                    let mut dpc = Matrix3x6::zeros();
                    dpc.fixed_view_mut::<3, 3>(0, 0).copy_from(&Matrix3::identity());
                    dpc.fixed_view_mut::<3, 3>(0, 3).copy_from(&(-skew(&pc)));
                    let jc = dpi * dpc; // 2×6
                    let hcc = jc.transpose() * jc; // 6×6
                    let gc = jc.transpose() * res; // 6
                    for a in 0..6 {
                        g[cslot + a] += gc[a];
                        for b in 0..6 {
                            h[(cslot + a, cslot + b)] += hcc[(a, b)];
                        }
                    }
                    // cross term camera↔point
                    let cross = jc.transpose() * jx; // 6×3
                    for a in 0..6 {
                        for b in 0..3 {
                            h[(cslot + a, px + b)] += cross[(a, b)];
                            h[(px + b, cslot + a)] += cross[(a, b)];
                        }
                    }
                }
            }
            // Levenberg–Marquardt damped solve, with backtracking on λ
            loop {
                let mut hl = h.clone();
                for i in 0..p {
                    hl[(i, i)] += lambda * h[(i, i)].max(1e-9);
                }
                let Some(delta) = hl.lu().solve(&(-&g)) else {
                    lambda *= 10.0;
                    if lambda > 1e12 {
                        return self.reprojection_rms();
                    }
                    continue;
                };
                // trial update on a copy
                let (trial_cams, trial_pts) = self.apply_delta(&cam_slot, &delta);
                let new_cost = self.sse_of(&trial_cams, &trial_pts);
                if new_cost < cost {
                    self.cameras = trial_cams;
                    self.points = trial_pts;
                    cost = new_cost;
                    lambda = (lambda * 0.5).max(1e-12);
                    break;
                } else {
                    lambda *= 4.0;
                    if lambda > 1e12 {
                        return self.reprojection_rms();
                    }
                }
            }
        }
        self.reprojection_rms()
    }

    fn apply_delta(&self, cam_slot: &[Option<usize>], delta: &DVector<f64>) -> (Vec<Camera>, Vec<Vector3<f64>>) {
        let mut cams = self.cameras.clone();
        for (i, slot) in cam_slot.iter().enumerate() {
            if let Some(s) = slot {
                let rho = Vector3::new(delta[*s], delta[*s + 1], delta[*s + 2]);
                let phi = Vector3::new(delta[*s + 3], delta[*s + 4], delta[*s + 5]);
                let dr = so3_exp(&phi);
                cams[i].r = dr * cams[i].r;
                cams[i].t = dr * cams[i].t + rho;
            }
        }
        let pt_base = 6 * cam_slot.iter().flatten().count();
        let mut pts = self.points.clone();
        for (j, pj) in pts.iter_mut().enumerate() {
            let b = pt_base + 3 * j;
            *pj += Vector3::new(delta[b], delta[b + 1], delta[b + 2]);
        }
        (cams, pts)
    }

    fn sse(&self) -> f64 {
        self.sse_of(&self.cameras, &self.points)
    }
    fn sse_of(&self, cams: &[Camera], pts: &[Vector3<f64>]) -> f64 {
        let mut acc = 0.0;
        for o in &self.observations {
            let pc = cams[o.cam].r * pts[o.point] + cams[o.cam].t;
            if pc.z.abs() < 1e-9 {
                return f64::INFINITY;
            }
            acc += (Vector2::new(pc.x / pc.z, pc.y / pc.z) - o.px).norm_squared();
        }
        acc
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synth() -> (Vec<Camera>, Vec<Vector3<f64>>, Vec<Observation>) {
        // four cameras along a slight arc, all looking at a cloud ~5 m ahead
        let mut cams = Vec::new();
        for i in 0..4 {
            let a = i as f64 * 0.08;
            let r = so3_exp(&Vector3::new(0.02 * i as f64, a, -0.01 * i as f64));
            let t = Vector3::new(0.4 * i as f64, 0.05 * i as f64, 0.0);
            cams.push(Camera { r, t });
        }
        let pts: Vec<Vector3<f64>> = (0..12)
            .map(|k| {
                let f = k as f64;
                Vector3::new((f * 0.7).sin() * 1.5, (f * 1.3).cos() * 1.2, 5.0 + (f * 0.5).sin())
            })
            .collect();
        let mut obs = Vec::new();
        for (ci, c) in cams.iter().enumerate() {
            for (pj, x) in pts.iter().enumerate() {
                let pc = c.r * x + c.t;
                obs.push(Observation { cam: ci, point: pj, px: Vector2::new(pc.x / pc.z, pc.y / pc.z) });
            }
        }
        (cams, pts, obs)
    }

    fn perturb_cam(c: &Camera, k: f64) -> Camera {
        Camera { r: so3_exp(&Vector3::new(0.05 * k, -0.04 * k, 0.03 * k)) * c.r, t: c.t + Vector3::new(0.1 * k, -0.08 * k, 0.12 * k) }
    }

    #[test]
    fn it_recovers_exact_structure_and_motion_from_clean_data() {
        // THE ORACLE. From a perturbed initialization, BA on noise-free observations recovers the true
        // cameras (2 & 3) and all points to solver precision. Cameras 0 & 1 are fixed to anchor the gauge.
        let (truth_cams, truth_pts, obs) = synth();
        let mut cams = truth_cams.clone();
        cams[2] = perturb_cam(&truth_cams[2], 1.0);
        cams[3] = perturb_cam(&truth_cams[3], -1.2);
        let pts: Vec<Vector3<f64>> = truth_pts.iter().map(|x| x + Vector3::new(0.15, -0.1, 0.2)).collect();
        let mut ba = BundleAdjustment { cameras: cams, points: pts, observations: obs, fixed: vec![true, true, false, false] };
        let rms = ba.optimize(60);
        assert!(rms < 1e-8, "reprojection error should vanish: {rms}");
        for (i, tc) in truth_cams.iter().enumerate().skip(2) {
            assert!((ba.cameras[i].r - tc.r).norm() < 1e-5 && (ba.cameras[i].t - tc.t).norm() < 1e-5, "camera {i} not recovered");
        }
        for (j, x) in truth_pts.iter().enumerate() {
            assert!((ba.points[j] - x).norm() < 1e-5, "point {j} not recovered");
        }
    }

    #[test]
    fn it_drives_the_reprojection_error_down_under_noise() {
        // THE APPLICATION. With deterministic pixel noise, BA lowers the reprojection error far below the
        // (already reasonable) initialization.
        let (truth_cams, truth_pts, mut obs) = synth();
        let mut seed = 5u64;
        let mut noise = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed >> 40) as f64 / (1u64 << 24) as f64 - 0.5) * 0.004
        };
        for o in &mut obs {
            o.px += Vector2::new(noise(), noise());
        }
        let mut cams = truth_cams.clone();
        cams[2] = perturb_cam(&truth_cams[2], 0.5);
        cams[3] = perturb_cam(&truth_cams[3], -0.4);
        let pts: Vec<Vector3<f64>> = truth_pts.iter().map(|x| x + Vector3::new(0.08, -0.05, 0.1)).collect();
        let mut ba = BundleAdjustment { cameras: cams, points: pts, observations: obs, fixed: vec![true, true, false, false] };
        let before = ba.reprojection_rms();
        let after = ba.optimize(60);
        assert!(after < 0.2 * before, "BA should slash reprojection error: {before} → {after}");
    }
}
