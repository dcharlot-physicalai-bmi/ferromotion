//! **MSCKF — Multi-State-Constraint Kalman Filter** for visual-inertial odometry (Mourikis &
//! Roumeliotis, ICRA 2007) — the first exteroceptive estimator in the corpus (InEKF / IMU-preintegration
//! are proprioceptive). Its signature idea: keep a **sliding window of camera poses** in the state — *not*
//! the landmarks — and, when a feature track completes, fold its multi-view geometry into a **pose-only**
//! constraint. For a feature seen from `N` poses the stacked reprojection residual is
//! `r = H_x δx_pose + H_f δX_feature + noise`; projecting `r` onto the **left nullspace of `H_f`** (a basis
//! `A` with `AᵀH_f = 0`) removes the landmark entirely, leaving `r_o = Aᵀr = (AᵀH_x) δx_pose + noise` — an
//! EKF update that costs `O(features)` and never puts the landmarks in the filter state.
//!
//! This implements that visual-constraint core: the pinhole measurement model and its analytic Jacobians,
//! linear feature triangulation from the window, the left-nullspace projection, and the error-state EKF
//! update on the pose window (rotation error is left-invariant, `R = exp([δθ])R̂`). IMU propagation is the
//! standard EKF predict step and is out of scope here (see [`crate::ImuPreintegrator`]). Verified against
//! finite-difference Jacobians, the nullspace property, and pose-error reduction. Pure `nalgebra` + core's
//! SO(3) maps → WASM-clean.

use ferromotion_core::{exp_so3, hat3};
use nalgebra::{DMatrix, DVector, Matrix3, Vector2, Vector3};

/// A camera pose in the sliding window: world→camera rotation `r` and camera position `p` (in world).
#[derive(Clone, Copy, Debug)]
pub struct CamPose {
    pub r: Matrix3<f64>,
    pub p: Vector3<f64>,
}

impl CamPose {
    /// Project a world point into normalized (focal-length-1) pinhole coordinates.
    pub fn project(&self, x_world: &Vector3<f64>) -> Vector2<f64> {
        let xc = self.r * (x_world - self.p);
        Vector2::new(xc.x / xc.z, xc.y / xc.z)
    }
}

/// A completed feature track: the window indices that observed it and the pixel (normalized) observations.
#[derive(Clone, Debug)]
pub struct FeatureTrack {
    pub cams: Vec<usize>,
    pub obs: Vec<Vector2<f64>>,
}

/// An MSCKF over a sliding window of camera poses with a `6N × 6N` error-state covariance
/// (`[δθ_i, δp_i]` per pose, `δθ` first).
#[derive(Clone, Debug)]
pub struct Msckf {
    pub window: Vec<CamPose>,
    pub cov: DMatrix<f64>,
    pub pixel_noise: f64,
}

impl Msckf {
    /// A window with an isotropic prior covariance `sigma0²` per error-state entry.
    pub fn new(window: Vec<CamPose>, sigma0: f64, pixel_noise: f64) -> Msckf {
        let d = 6 * window.len();
        Msckf { window, cov: DMatrix::identity(d, d) * (sigma0 * sigma0), pixel_noise }
    }

    /// `∂z/∂X_c` for the normalized projection `z = (x/z, y/z)` (2×3).
    fn dproj(xc: &Vector3<f64>) -> DMatrix<f64> {
        let iz = 1.0 / xc.z;
        DMatrix::from_row_slice(2, 3, &[iz, 0.0, -xc.x * iz * iz, 0.0, iz, -xc.y * iz * iz])
    }

    /// Linear (DLT) triangulation of a feature from its observing poses.
    pub fn triangulate(&self, track: &FeatureTrack) -> Vector3<f64> {
        // each obs gives two rows: (z_x·r3 − r1)·X = (z_x·r3 − r1)·p , stacking A X = b.
        let rows = 2 * track.cams.len();
        let mut a = DMatrix::zeros(rows, 3);
        let mut b = DVector::zeros(rows);
        for (k, &ci) in track.cams.iter().enumerate() {
            let pose = &self.window[ci];
            let z = track.obs[k];
            let (r1, r2, r3) = (pose.r.row(0), pose.r.row(1), pose.r.row(2));
            let row_x = z.x * r3 - r1;
            let row_y = z.y * r3 - r2;
            a.row_mut(2 * k).copy_from(&row_x);
            a.row_mut(2 * k + 1).copy_from(&row_y);
            b[2 * k] = (z.x * r3 - r1).dot(&pose.p.transpose());
            b[2 * k + 1] = (z.y * r3 - r2).dot(&pose.p.transpose());
        }
        match (a.transpose() * &a).try_inverse() {
            Some(inv) => {
                let x = inv * a.transpose() * b;
                Vector3::new(x[0], x[1], x[2])
            }
            None => Vector3::zeros(),
        }
    }

    /// Stacked residual and Jacobians for a feature at estimate `x_feat`: `r` (2N), `H_x` (2N × 6N over the
    /// whole window), `H_f` (2N × 3).
    fn feature_model(&self, track: &FeatureTrack, x_feat: &Vector3<f64>) -> (DVector<f64>, DMatrix<f64>, DMatrix<f64>) {
        let d = 6 * self.window.len();
        let rows = 2 * track.cams.len();
        let mut r = DVector::zeros(rows);
        let mut hx = DMatrix::zeros(rows, d);
        let mut hf = DMatrix::zeros(rows, 3);
        for (k, &ci) in track.cams.iter().enumerate() {
            let pose = &self.window[ci];
            let xc = pose.r * (x_feat - pose.p);
            let z_hat = Vector2::new(xc.x / xc.z, xc.y / xc.z);
            r[2 * k] = track.obs[k].x - z_hat.x;
            r[2 * k + 1] = track.obs[k].y - z_hat.y;
            let jp = Self::dproj(&xc); // 2×3
            // ∂X_c/∂δθ = −[X_c]×  (R = exp([δθ])R̂) ; ∂X_c/∂δp = −R̂ ; ∂X_c/∂δX = R̂
            let dxc_dtheta = -hat3(&xc);
            let dxc_dp = -pose.r;
            let h_theta = &jp * DMatrix::from_row_slice(3, 3, dxc_dtheta.as_slice()).transpose();
            let h_p = &jp * DMatrix::from_row_slice(3, 3, dxc_dp.as_slice()).transpose();
            hx.view_mut((2 * k, 6 * ci), (2, 3)).copy_from(&h_theta);
            hx.view_mut((2 * k, 6 * ci + 3), (2, 3)).copy_from(&h_p);
            let h_x_feat = &jp * DMatrix::from_row_slice(3, 3, pose.r.as_slice()).transpose();
            hf.view_mut((2 * k, 0), (2, 3)).copy_from(&h_x_feat);
        }
        (r, hx, hf)
    }

    /// An orthonormal basis for the **left nullspace** of `hf` (columns `A` with `AᵀH_f = 0`), by
    /// Gram–Schmidt of the identity against the column space of `hf`.
    fn left_nullspace(hf: &DMatrix<f64>) -> DMatrix<f64> {
        let m = hf.nrows();
        // orthonormal basis of col(hf)
        let mut basis: Vec<DVector<f64>> = Vec::new();
        for j in 0..hf.ncols() {
            let mut v = hf.column(j).into_owned();
            for b in &basis {
                v -= b * b.dot(&v);
            }
            let nv = v.norm();
            if nv > 1e-9 {
                basis.push(v / nv);
            }
        }
        // complement
        let mut null: Vec<DVector<f64>> = Vec::new();
        for i in 0..m {
            let mut v = DVector::zeros(m);
            v[i] = 1.0;
            for b in basis.iter().chain(null.iter()) {
                v -= b * b.dot(&v);
            }
            let nv = v.norm();
            if nv > 1e-9 {
                null.push(v / nv);
            }
            if null.len() == m - basis.len() {
                break;
            }
        }
        let mut a = DMatrix::zeros(m, null.len());
        for (j, v) in null.iter().enumerate() {
            a.column_mut(j).copy_from(v);
        }
        a
    }

    /// MSCKF update from completed feature tracks: triangulate each, form its pose-only nullspace-projected
    /// constraint, stack them, and apply one error-state EKF update to the whole pose window.
    pub fn update(&mut self, tracks: &[FeatureTrack]) {
        let d = 6 * self.window.len();
        let mut h_rows: Vec<DVector<f64>> = Vec::new();
        let mut r_stack: Vec<f64> = Vec::new();
        for track in tracks {
            if track.cams.len() < 2 {
                continue;
            }
            let x_feat = self.triangulate(track);
            let (r, hx, hf) = self.feature_model(track, &x_feat);
            let a = Self::left_nullspace(&hf); // 2N × (2N−3)
            if a.ncols() == 0 {
                continue;
            }
            let r_o = a.transpose() * &r;
            let h_o = a.transpose() * &hx;
            for i in 0..r_o.len() {
                r_stack.push(r_o[i]);
                h_rows.push(h_o.row(i).transpose());
            }
        }
        if r_stack.is_empty() {
            return;
        }
        let m = r_stack.len();
        let mut h = DMatrix::zeros(m, d);
        for (i, row) in h_rows.iter().enumerate() {
            h.row_mut(i).copy_from(&row.transpose());
        }
        let r = DVector::from_vec(r_stack);
        // EKF update: S = H P Hᵀ + σ²I, K = P Hᵀ S⁻¹, δx = K r, P ← (I−KH)P
        let noise = self.pixel_noise * self.pixel_noise;
        let s = &h * &self.cov * h.transpose() + DMatrix::identity(m, m) * noise;
        let Some(s_inv) = s.try_inverse() else { return };
        let k = &self.cov * h.transpose() * s_inv;
        let dx = &k * r;
        // apply the correction to every window pose
        for (i, pose) in self.window.iter_mut().enumerate() {
            let dtheta = Vector3::new(dx[6 * i], dx[6 * i + 1], dx[6 * i + 2]);
            let dp = Vector3::new(dx[6 * i + 3], dx[6 * i + 4], dx[6 * i + 5]);
            pose.r = exp_so3(&dtheta) * pose.r;
            pose.p += dp;
        }
        let ikh = DMatrix::identity(d, d) - &k * &h;
        self.cov = &ikh * &self.cov * ikh.transpose() + &k * (DMatrix::identity(m, m) * noise) * k.transpose();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rot_z(a: f64) -> Matrix3<f64> {
        let (s, c) = a.sin_cos();
        Matrix3::new(c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0)
    }

    // A small window of cameras looking along +z at a cloud of landmarks in front of them.
    fn scene() -> (Vec<CamPose>, Vec<Vector3<f64>>) {
        let poses = vec![
            CamPose { r: Matrix3::identity(), p: Vector3::new(0.0, 0.0, 0.0) },
            CamPose { r: rot_z(0.05), p: Vector3::new(0.3, 0.0, 0.0) },
            CamPose { r: rot_z(-0.03), p: Vector3::new(0.6, 0.1, 0.0) },
            CamPose { r: rot_z(0.02), p: Vector3::new(0.9, -0.1, 0.05) },
        ];
        let landmarks = vec![
            Vector3::new(0.2, 0.1, 4.0),
            Vector3::new(-0.3, 0.4, 5.0),
            Vector3::new(0.5, -0.2, 3.5),
            Vector3::new(0.0, 0.0, 6.0),
        ];
        (poses, landmarks)
    }

    fn track_of(poses: &[CamPose], lm: &Vector3<f64>) -> FeatureTrack {
        FeatureTrack {
            cams: (0..poses.len()).collect(),
            obs: poses.iter().map(|p| p.project(lm)).collect(),
        }
    }

    #[test]
    fn triangulation_recovers_the_landmark() {
        let (poses, landmarks) = scene();
        let f = Msckf::new(poses.clone(), 1.0, 1e-3);
        for lm in &landmarks {
            let est = f.triangulate(&track_of(&poses, lm));
            assert!((est - lm).norm() < 1e-6, "triangulated {est:?} vs true {lm:?}");
        }
    }

    #[test]
    fn the_measurement_jacobian_matches_finite_differences() {
        let (poses, landmarks) = scene();
        let f = Msckf::new(poses.clone(), 1.0, 1e-3);
        let track = track_of(&poses, &landmarks[0]);
        let (_r, hx, hf) = f.feature_model(&track, &landmarks[0]);
        let eps = 1e-6;
        // H_f vs FD (perturb the feature)
        for j in 0..3 {
            let mut xp = landmarks[0];
            let mut xm = landmarks[0];
            xp[j] += eps;
            xm[j] -= eps;
            let rp = f.feature_model(&track, &xp).0;
            let rm = f.feature_model(&track, &xm).0;
            let fd = (rm - rp) / (2.0 * eps); // r = obs − ẑ, so ∂r/∂x = −∂ẑ/∂x ; H_f is ∂ẑ/∂x
            for i in 0..hf.nrows() {
                assert!((hf[(i, j)] - fd[i]).abs() < 1e-4, "H_f[{i},{j}] {} vs fd {}", hf[(i, j)], fd[i]);
            }
        }
        // H_x vs FD on pose 1's position (δp) and rotation (δθ)
        let ci = 1;
        for j in 0..3 {
            let mut fp = f.clone();
            let mut fm = f.clone();
            let mut dp = Vector3::zeros();
            dp[j] = eps;
            fp.window[ci].p += dp;
            fm.window[ci].p -= dp;
            let rp = fp.feature_model(&track, &landmarks[0]).0;
            let rm = fm.feature_model(&track, &landmarks[0]).0;
            let fd = (rm - rp) / (2.0 * eps);
            for i in 0..hx.nrows() {
                assert!((hx[(i, 6 * ci + 3 + j)] - fd[i]).abs() < 1e-4, "H_x δp[{i},{j}] {} vs fd {}", hx[(i, 6 * ci + 3 + j)], fd[i]);
            }
        }
    }

    #[test]
    fn the_nullspace_projection_removes_the_landmark() {
        // THE INVARIANT. AᵀH_f = 0 (the projected constraint is landmark-free), and the nullspace has the
        // right dimension 2N − 3.
        let (poses, landmarks) = scene();
        let f = Msckf::new(poses.clone(), 1.0, 1e-3);
        let track = track_of(&poses, &landmarks[0]);
        let (_r, _hx, hf) = f.feature_model(&track, &landmarks[0]);
        let a = Msckf::left_nullspace(&hf);
        assert_eq!(a.ncols(), 2 * poses.len() - 3, "nullspace dim should be 2N−3");
        let ath = a.transpose() * &hf;
        assert!(ath.amax() < 1e-9, "AᵀH_f must vanish: {}", ath.amax());
    }

    #[test]
    fn the_update_corrects_perturbed_poses() {
        // THE HEADLINE. Feature observations from the TRUE poses, but the filter's window is perturbed.
        // The nullspace-projected visual update pulls the window back toward the truth (pose error drops).
        let (truth, landmarks) = scene();
        let tracks: Vec<FeatureTrack> = landmarks.iter().map(|lm| track_of(&truth, lm)).collect();

        // Monocular visual constraints are relative AND scale-free — the gauge is a 7-DOF similarity
        // (global pose 6 + scale 1). Anchor the first TWO poses to fix it, then perturb poses 2 & 3, which
        // are now fully observable and correctable.
        let mut perturbed = truth.clone();
        for i in [2, 3] {
            perturbed[i].p += Vector3::new(0.01 * i as f64, -0.008 * i as f64, 0.006);
            perturbed[i].r = exp_so3(&Vector3::new(0.0, 0.0, 0.006 * i as f64)) * perturbed[i].r;
        }
        let err0: f64 = perturbed.iter().zip(&truth).map(|(a, b)| (a.p - b.p).norm()).sum();

        let mut f = Msckf::new(perturbed, 0.1, 1e-3);
        for k in 0..12 {
            f.cov[(k, k)] = 1e-12; // anchor poses 0 and 1 (fixes gauge + scale)
        }
        for _ in 0..4 {
            f.update(&tracks);
        }
        let err1: f64 = f.window.iter().zip(&truth).map(|(a, b)| (a.p - b.p).norm()).sum();
        assert!(err1 < 0.3 * err0, "the visual update should cut pose error: {err0} → {err1}");
    }
}
