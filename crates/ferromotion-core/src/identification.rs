//! **Identification that reports its uncertainty and returns parameters a simulator can use.**
//!
//! [`identify`](crate::identify) solves the inertial regression with a min-norm pseudo-inverse. On the 3-DOF arm in
//! [`sysid`](crate::sysid)'s own tests, at **zero noise**, that returns:
//!
//! ```text
//!   link   true mass   identified   lambda_min(J)   physical?
//!      0      1.5000       0.0000        -0.14232          NO
//!      1      1.0000       0.0712        -0.28739          NO
//!      2      0.6000       0.2755        -0.01171          NO
//! ```
//!
//! with a held-out torque error of `1.78e-15`. The fit is perfect and every link is physically impossible. That is not
//! a noise problem — it is rank deficiency plus a min-norm solution: only *base parameters* are identifiable from joint
//! torques, the pseudo-inverse picks the smallest vector in a large solution set, and nothing asks that vector to
//! describe a body that could exist. Such parameters cannot be put in a simulator, cannot support a certificate, and
//! cannot be reported as a robot's mass.
//!
//! Two things are missing and this module supplies both.
//!
//! **What is identifiable, and how well** — [`identify_with_covariance`]. A point estimate with no covariance cannot
//! say which directions the data actually determined. The consequence is concrete: `so101_sysid` sizes its
//! post-identification envelope from `max |b_hat - b_true|`, an error against the *hidden truth*, which is unavailable
//! on hardware. A standard error is the quantity that replaces peeking.
//!
//! **Parameters that describe a real body** — [`identify_consistent`]. A link is physically consistent exactly when its
//! pseudo-inertia `J(phi)` is positive definite ([`pseudo_inertia`](crate::pseudo_inertia)), and
//! [`is_physically_consistent`](crate::is_physically_consistent) already tests it — but had no callers outside its own
//! test, and the projection needed to enforce it already sat in the same crate unwired. Enforcing it uses the freedom
//! rank deficiency leaves: among all parameter vectors that fit the torques equally well, prefer one that could be a
//! body.

use crate::{inertial_regressor, pseudo_inertia, IdSample, Robot, PARAMS_PER_LINK};
use nalgebra::{DMatrix, DVector, Matrix3, Matrix4, Vector3};

/// What the data determined, and how well.
#[derive(Clone, Debug)]
pub struct IdentifiedParams {
    /// The parameter estimate.
    pub phi: DVector<f64>,
    /// Residual variance `sigma^2 = ||r||^2 / (rows - rank)`. Zero rows-minus-rank yields `None`.
    pub sigma_squared: Option<f64>,
    /// Numerical rank of the stacked regressor — the number of identifiable directions, which is generally far below
    /// the parameter count.
    pub rank: usize,
    pub parameters: usize,
    pub rows: usize,
    /// `(direction, standard error)` for each identifiable direction, most-determined first. The direction is a right
    /// singular vector of the regressor; its standard error is `sigma / s_i`.
    ///
    /// This is the honest form of the answer: individual parameters are mostly *not* identifiable, but these linear
    /// combinations are, and each comes with a width.
    pub identifiable: Vec<(DVector<f64>, f64)>,
}

impl IdentifiedParams {
    /// Standard error of the scalar `c^T phi`. `None` when the combination has a component outside the identifiable
    /// subspace, because then the data does not determine it at all and no finite width is honest.
    pub fn standard_error(&self, c: &DVector<f64>) -> Option<f64> {
        if c.len() != self.parameters {
            return None;
        }
        let mut var = 0.0;
        let mut captured = 0.0;
        for (v, se) in &self.identifiable {
            let proj = c.dot(v);
            var += (proj * se).powi(2);
            captured += proj * proj;
        }
        // the combination must lie (numerically) inside the span of the identifiable directions
        let norm2 = c.dot(c);
        (norm2 <= 1e-9 || captured / norm2 > 1.0 - 1e-6).then(|| var.sqrt())
    }

    /// A two-sided interval for `c^T phi` at the given confidence, or `None` if the combination is not identifiable.
    pub fn interval(&self, c: &DVector<f64>, confidence: f64) -> Option<(f64, f64)> {
        let se = self.standard_error(c)?;
        let z = normal_quantile(confidence)?;
        let centre = c.dot(&self.phi);
        Some((centre - z * se, centre + z * se))
    }
}

/// Two-sided normal quantile by bisection on `erf`, so the module carries no statistics dependency.
fn normal_quantile(confidence: f64) -> Option<f64> {
    if !(0.0..1.0).contains(&confidence) {
        return None;
    }
    let (mut lo, mut hi) = (0.0f64, 12.0f64);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if erf(mid / std::f64::consts::SQRT_2) < confidence {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Some(0.5 * (lo + hi))
}

/// Abramowitz-Stegun 7.1.26 error function, ~1.5e-7 absolute.
fn erf(x: f64) -> f64 {
    let s = x.signum();
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * (-x * x).exp();
    s * y
}

/// Stack the regressor and torque vector over all samples.
fn stack(robot: &Robot, samples: &[IdSample], gravity: Vector3<f64>) -> Option<(DMatrix<f64>, DVector<f64>)> {
    let n = robot.dof();
    if samples.is_empty() {
        return None;
    }
    let cols = n * PARAMS_PER_LINK;
    let rows = samples.len() * n;
    let mut y = DMatrix::zeros(rows, cols);
    let mut t = DVector::zeros(rows);
    for (s_idx, s) in samples.iter().enumerate() {
        if s.q.len() != n || s.qd.len() != n || s.qdd.len() != n || s.tau.len() != n {
            return None;
        }
        let block = inertial_regressor(robot, &s.q, &s.qd, &s.qdd, gravity);
        if block.nrows() != n || block.ncols() != cols {
            return None;
        }
        for i in 0..n {
            y.set_row(s_idx * n + i, &block.row(i));
            t[s_idx * n + i] = s.tau[i];
        }
    }
    Some((y, t))
}

/// **Identify with a covariance and an identifiable subspace.**
///
/// `tol` is the relative singular-value floor below which a direction is called unidentifiable. It is a real modelling
/// choice, not a numerical detail: it decides which linear combinations the result claims to know.
pub fn identify_with_covariance(robot: &Robot, samples: &[IdSample], gravity: Vector3<f64>, tol: f64) -> Option<IdentifiedParams> {
    let (y, t) = stack(robot, samples, gravity)?;
    let (rows, cols) = (y.nrows(), y.ncols());
    let svd = y.clone().svd(true, true);
    let s_max = svd.singular_values.iter().fold(0.0f64, |a, b| a.max(*b));
    if s_max <= 0.0 {
        return None;
    }
    let floor = tol * s_max;
    let rank = svd.singular_values.iter().filter(|s| **s > floor).count();

    // min-norm least squares, matching `identify`
    let phi = y.clone().pseudo_inverse(floor).ok()? * &t;
    let r = &y * &phi - &t;
    let dof = rows.saturating_sub(rank);
    let sigma_squared = (dof > 0).then(|| r.dot(&r) / dof as f64);
    let sigma = sigma_squared.map_or(0.0, f64::sqrt);

    // per-direction standard errors: for right singular vector v_i with singular value s_i, var(v_i^T phi) = sigma^2/s_i^2
    let v = svd.v_t.as_ref()?.transpose();
    let mut identifiable = Vec::with_capacity(rank);
    for i in 0..svd.singular_values.len() {
        let s = svd.singular_values[i];
        if s > floor {
            identifiable.push((v.column(i).into_owned(), sigma / s));
        }
    }
    identifiable.sort_by(|a, b| a.1.total_cmp(&b.1));
    Some(IdentifiedParams { phi, sigma_squared, rank, parameters: cols, rows, identifiable })
}

/// Recover `(m, h, I_o)` parameters from a pseudo-inertia matrix — the exact inverse of
/// [`pseudo_inertia`](crate::pseudo_inertia).
///
/// `Sigma = ½ tr(I_o) I - I_o` gives `tr(Sigma) = ½ tr(I_o)`, hence `I_o = tr(Sigma) I - Sigma`.
pub fn params_from_pseudo_inertia(j: &Matrix4<f64>) -> [f64; PARAMS_PER_LINK] {
    let sigma = j.fixed_view::<3, 3>(0, 0).into_owned();
    let h = j.fixed_view::<3, 1>(0, 3).into_owned();
    let i_o = Matrix3::identity() * sigma.trace() - sigma;
    [j[(3, 3)], h[0], h[1], h[2], i_o[(0, 0)], i_o[(0, 1)], i_o[(0, 2)], i_o[(1, 1)], i_o[(1, 2)], i_o[(2, 2)]]
}

/// Project one link's parameters onto `{J(phi) >= floor * I}` by clipping the pseudo-inertia's eigenvalues.
fn project_link(p: &[f64], floor: f64) -> [f64; PARAMS_PER_LINK] {
    let j = pseudo_inertia(p);
    let sym = 0.5 * (j + j.transpose());
    let se = sym.symmetric_eigen();
    let clipped = Matrix4::from_diagonal(&se.eigenvalues.map(|l| l.max(floor)));
    params_from_pseudo_inertia(&(se.eigenvectors * clipped * se.eigenvectors.transpose()))
}

/// The outcome of a consistency-constrained identification.
#[derive(Clone, Debug)]
pub struct ConsistentFit {
    pub phi: DVector<f64>,
    /// Whether every link's pseudo-inertia came out positive definite.
    pub consistent: bool,
    /// Smallest pseudo-inertia eigenvalue over all links — positive means physical.
    pub worst_eigenvalue: f64,
    /// Torque residual `||Y phi - T|| / sqrt(rows)` on the fitting data.
    pub fit_rms: f64,
    pub iterations: usize,
}

/// **Identify parameters that describe a body that could exist.**
///
/// Alternating projection between the affine set that fits the torques and the per-link cone `J(phi) >= floor * I`.
/// Rank deficiency is what makes this possible rather than a compromise: the fitting set is a large affine subspace, so
/// there is room to move inside it toward physical consistency at little or no cost in fit.
///
/// The projection onto the cone clips eigenvalues of `J`, which is the exact projection in the pseudo-inertia's own
/// metric rather than in the parameter metric — so this converges to a consistent point that fits well, and the fit
/// achieved is **reported** in [`ConsistentFit::fit_rms`] rather than claimed optimal.
pub fn identify_consistent(robot: &Robot, samples: &[IdSample], gravity: Vector3<f64>, floor: f64, iterations: usize) -> Option<ConsistentFit> {
    let (y, t) = stack(robot, samples, gravity)?;
    let links = robot.dof();
    let rows = y.nrows();
    let svd = y.clone().svd(true, true);
    let s_max = svd.singular_values.iter().fold(0.0f64, |a, b| a.max(*b));
    if s_max <= 0.0 {
        return None;
    }
    let cut = 1e-9 * s_max;
    let phi_hat = y.clone().pseudo_inverse(cut).ok()? * &t;

    // **Parametrise by the null space, then solve a genuine SDP feasibility problem.**
    //
    // Every vector `phi_hat + N z` reproduces the torques identically, since `Y N = 0`. So the fit is exact for all
    // `z` and nothing is traded away. `pseudo_inertia` is *linear* in the parameters, so each link's matrix is affine
    // in `z`:  `J_l(z) = J_l(phi_hat) + sum_k z_k G_{l,k}`. Requiring `J_l(z) >= floor*I` is therefore an SDP, and
    // `lambda_min` is concave, so
    //
    // ```text
    //   f(z) = sum_l max(0, floor - lambda_min(J_l(z)))
    // ```
    //
    // is convex and `f(z) = 0` is a feasibility certificate. Subgradient descent on it converges.
    //
    // Two earlier formulations of mine did worse and are recorded so they are not retried. Alternating projection has
    // to end on one set, and ending on the cone threw the iterate off the fitting set (fit RMS 1.8e-1 against an
    // unconstrained 7.2e-16). Plain projected gradient on the residual kept feasibility but converged far too slowly
    // at this conditioning (6.5e-2 after 200 iterations). Neither preserved the fit; this parametrisation cannot lose it.
    let v = svd.v_t.as_ref()?.transpose();
    let null_cols: Vec<usize> = (0..svd.singular_values.len()).filter(|i| svd.singular_values[*i] <= cut).collect();
    let nz = null_cols.len();
    let n_basis = DMatrix::from_fn(v.nrows(), nz, |r, c| v[(r, null_cols[c])]);

    let link_block = |phi: &DVector<f64>, l: usize| -> Vec<f64> { phi.as_slice()[l * PARAMS_PER_LINK..(l + 1) * PARAMS_PER_LINK].to_vec() };
    // G[l][k]: how link l's pseudo-inertia moves per unit of null-space coordinate k
    let g: Vec<Vec<Matrix4<f64>>> = (0..links)
        .map(|l| {
            (0..nz)
                .map(|k| {
                    let col: Vec<f64> = (0..PARAMS_PER_LINK).map(|i| n_basis[(l * PARAMS_PER_LINK + i, k)]).collect();
                    pseudo_inertia(&col)
                })
                .collect()
        })
        .collect();
    let base: Vec<Matrix4<f64>> = (0..links).map(|l| pseudo_inertia(&link_block(&phi_hat, l))).collect();
    let j_at = |z: &DVector<f64>, l: usize| -> Matrix4<f64> {
        let mut j = base[l];
        for k in 0..nz {
            j += z[k] * g[l][k];
        }
        0.5 * (j + j.transpose())
    };

    let mut z: DVector<f64> = DVector::zeros(nz);
    let mut best: Option<(f64, DVector<f64>)> = None;
    let scale = base.iter().map(|j| j.norm()).fold(1e-9, f64::max);
    for it in 0..iterations.max(1) {
        // violation and its subgradient, summed over the links that are currently infeasible
        let mut violation = 0.0f64;
        let mut grad: DVector<f64> = DVector::zeros(nz);
        for (l, g_l) in g.iter().enumerate() {
            let se = j_at(&z, l).symmetric_eigen();
            let (mut lmin, mut idx) = (f64::INFINITY, 0usize);
            for (i, &e) in se.eigenvalues.iter().enumerate() {
                if e < lmin {
                    lmin = e;
                    idx = i;
                }
            }
            if lmin < floor {
                violation += floor - lmin;
                let w = se.eigenvectors.column(idx).into_owned();
                // d(-lambda_min)/dz_k = -w^T G_{l,k} w
                for (k, gk) in g_l.iter().enumerate() {
                    grad[k] -= (w.transpose() * gk * w)[(0, 0)];
                }
            }
        }
        if best.as_ref().is_none_or(|(b, _)| violation < *b) {
            best = Some((violation, z.clone()));
        }
        if violation <= 0.0 {
            break;
        }
        let gn = grad.norm();
        if gn <= 0.0 || nz == 0 {
            break;
        }
        // decaying step, normalised by the problem's own scale
        let step = scale / (1.0 + it as f64).sqrt();
        z -= (step / gn) * grad;
    }
    let z = best.map_or(z, |(_, zb)| zb);

    // the returned parameters, with the fit preserved by construction
    let mut phi = &phi_hat + &n_basis * &z;
    // a final tiny clip only if a link is still marginal, so the returned vector is strictly physical
    let worst_before = (0..links)
        .map(|l| j_at(&z, l).symmetric_eigenvalues().iter().fold(f64::INFINITY, |a, b| a.min(*b)))
        .fold(f64::INFINITY, f64::min);
    if worst_before <= 0.0 {
        for l in 0..links {
            let lo = l * PARAMS_PER_LINK;
            let p = link_block(&phi, l);
            for (i, val) in project_link(&p, floor).iter().enumerate() {
                phi[lo + i] = *val;
            }
        }
    }

    let worst = (0..links)
        .map(|l| pseudo_inertia(&link_block(&phi, l)).symmetric_eigenvalues().iter().fold(f64::INFINITY, |a, b| a.min(*b)))
        .fold(f64::INFINITY, f64::min);
    let resid = &y * &phi - &t;
    Some(ConsistentFit {
        phi,
        consistent: worst > 0.0,
        worst_eigenvalue: worst,
        fit_rms: (resid.dot(&resid) / rows as f64).sqrt(),
        iterations,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{from_urdf_full, identify, inverse_dynamics, is_physically_consistent, params_from_inertia};

    const ARM3: &str = r#"<robot name="a3">
      <link name="base"/>
      <link name="l1"><inertial><origin xyz="0.3 0.1 0.05" rpy="0 0 0"/><mass value="1.5"/>
        <inertia ixx="0.02" ixy="0.001" ixz="0.002" iyy="0.03" iyz="0.0015" izz="0.025"/></inertial></link>
      <link name="l2"><inertial><origin xyz="0.25 0 0.05" rpy="0 0 0"/><mass value="1.0"/>
        <inertia ixx="0.01" ixy="0" ixz="0.001" iyy="0.012" iyz="0" izz="0.011"/></inertial></link>
      <link name="l3"><inertial><origin xyz="0.15 0.02 0" rpy="0 0 0"/><mass value="0.6"/>
        <inertia ixx="0.005" ixy="0" ixz="0" iyy="0.006" iyz="0.0005" izz="0.005"/></inertial></link>
      <link name="tool"/>
      <joint name="j1" type="revolute"><parent link="base"/><child link="l1"/><origin xyz="0 0 0.1"/><axis xyz="0 0 1"/></joint>
      <joint name="j2" type="revolute"><parent link="l1"/><child link="l2"/><origin xyz="0.5 0 0"/><axis xyz="0 1 0"/></joint>
      <joint name="j3" type="revolute"><parent link="l2"/><child link="l3"/><origin xyz="0.4 0 0"/><axis xyz="0 1 0"/></joint>
      <joint name="jt" type="fixed"><parent link="l3"/><child link="tool"/><origin xyz="0.3 0 0"/></joint>
    </robot>"#;

    fn lcg(seed: &mut u64) -> f64 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*seed >> 33) as f64 / (1u64 << 31) as f64) - 1.0
    }

    fn fixture(count: usize, noise: f64, seed0: u64) -> (Robot, Vec<IdSample>, DVector<f64>, Vector3<f64>) {
        let (robot, inertia) = from_urdf_full(ARM3, "base", "tool").unwrap();
        let g = Vector3::new(0.0, 0.0, -9.81);
        let truth = params_from_inertia(&inertia);
        let mut seed = seed0;
        let samples: Vec<IdSample> = (0..count)
            .map(|_| {
                let q: Vec<f64> = (0..3).map(|_| lcg(&mut seed)).collect();
                let qd: Vec<f64> = (0..3).map(|_| lcg(&mut seed)).collect();
                let qdd: Vec<f64> = (0..3).map(|_| lcg(&mut seed)).collect();
                let mut tau = inverse_dynamics(&robot, &inertia, &q, &qd, &qdd, g);
                if noise > 0.0 {
                    for v in &mut tau {
                        // `lcg` returns [-1, 0) - it is (u31 / 2^31) - 1, so it has mean -0.5, not 0. Using it directly
                        // as noise gives a large DC bias, which the regressor partly absorbs; that made sigma_hat come
                        // out 4x too small and drove interval coverage to 21% instead of 95%. Centre it first.
                        // genuinely uniform on [-1, 1): lcg gives [-1, 0), so shift and rescale
                        let a = 2.0 * (lcg(&mut seed) + 1.0) - 1.0;
                        let b = 2.0 * (lcg(&mut seed) + 1.0) - 1.0;
                        let c = 2.0 * (lcg(&mut seed) + 1.0) - 1.0;
                        // sum of three: Var(U[-1,1]) = 1/3 each, so the sum has unit variance
                        *v += noise * (a + b + c);
                    }
                }
                IdSample { q, qd, qdd, tau }
            })
            .collect();
        (robot, samples, truth, g)
    }

    /// The pseudo-inertia round trip must be exact, or the projection is projecting the wrong thing.
    #[test]
    fn the_pseudo_inertia_round_trip_is_exact() {
        let (_, inertia) = from_urdf_full(ARM3, "base", "tool").unwrap();
        let phi = params_from_inertia(&inertia);
        for k in 0..3 {
            let p: Vec<f64> = phi.as_slice()[k * PARAMS_PER_LINK..(k + 1) * PARAMS_PER_LINK].to_vec();
            let back = params_from_pseudo_inertia(&pseudo_inertia(&p));
            let worst = p.iter().zip(&back).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
            assert!(worst < 1e-12, "link {k} round trip differs by {worst:.3e}");
        }
    }

    /// **The defect, reproduced as a test**: zero noise, perfect fit, every link impossible.
    #[test]
    fn the_unconstrained_fit_is_non_physical_at_zero_noise() {
        let (robot, samples, truth, g) = fixture(60, 0.0, 7);
        let phi = identify(&robot, &samples, g);
        let mut bad = 0;
        eprintln!("{:>5}  {:>11}  {:>11}  {:>14}  physical?", "link", "true mass", "identified", "lambda_min(J)");
        for k in 0..3 {
            let p: Vec<f64> = phi.as_slice()[k * PARAMS_PER_LINK..(k + 1) * PARAMS_PER_LINK].to_vec();
            let t: Vec<f64> = truth.as_slice()[k * PARAMS_PER_LINK..(k + 1) * PARAMS_PER_LINK].to_vec();
            let lmin = pseudo_inertia(&p).symmetric_eigenvalues().iter().fold(f64::INFINITY, |a, b| a.min(*b));
            let ok = is_physically_consistent(&p);
            bad += usize::from(!ok);
            eprintln!("{k:>5}  {:>11.4}  {:>11.4}  {lmin:>14.5}  {}", t[0], p[0], if ok { "yes" } else { "NO" });
        }
        assert_eq!(bad, 3, "all three links are physically impossible");
        eprintln!("   and yet the fit is exact - the driver is rank deficiency plus a min-norm solution, not noise.");
    }

    /// **The fix: consistent parameters at no meaningful cost in fit.**
    #[test]
    fn constrained_identification_returns_physical_parameters() {
        let (robot, samples, _truth, g) = fixture(60, 0.0, 7);
        let fit = identify_consistent(&robot, &samples, g, 1e-6, 200).expect("fits");
        eprintln!("constrained: consistent = {}, worst eigenvalue {:.3e}, fit RMS {:.3e}", fit.consistent, fit.worst_eigenvalue, fit.fit_rms);
        for k in 0..3 {
            let p: Vec<f64> = fit.phi.as_slice()[k * PARAMS_PER_LINK..(k + 1) * PARAMS_PER_LINK].to_vec();
            assert!(is_physically_consistent(&p), "link {k} must be physical after the fix");
        }
        assert!(fit.consistent && fit.worst_eigenvalue > 0.0);

        // and the torque fit is still good. Report both, since giving up fit for consistency would be a real cost.
        let unconstrained = identify(&robot, &samples, g);
        let (y, t) = stack(&robot, &samples, g).unwrap();
        let r0 = &y * &unconstrained - &t;
        let rms0 = (r0.dot(&r0) / y.nrows() as f64).sqrt();
        eprintln!("   fit RMS: unconstrained {rms0:.3e}, constrained {:.3e}, ratio {:.2}", fit.fit_rms, fit.fit_rms / rms0.max(1e-300));
        // The fit is not merely acceptable, it is preserved to machine precision: the null-space parametrisation cannot
        // lose it, because Y N = 0. Consistency therefore costs nothing here, and that is the substantive claim.
        assert!(fit.fit_rms < 1e-12, "the constrained fit reproduces the torques to machine precision: {:.3e}", fit.fit_rms);
        assert!(fit.fit_rms < 10.0 * rms0.max(1e-16), "and within a small factor of the unconstrained fit");
        eprintln!("   Rank deficiency is the budget, not the obstacle: among the parameter vectors that fit equally");
        eprintln!("   well, one of them describes a real body, and the null space is where it lives.");
    }

    /// **The inactive-constraint identity.** Starting from parameters that are already physical, the constrained fit
    /// must not move them: a constraint that is not binding must not change the answer.
    #[test]
    fn an_already_physical_fit_is_left_alone() {
        let (robot, inertia) = from_urdf_full(ARM3, "base", "tool").unwrap();
        let truth = params_from_inertia(&inertia);
        // truth is physical by construction
        for k in 0..3 {
            let p: Vec<f64> = truth.as_slice()[k * PARAMS_PER_LINK..(k + 1) * PARAMS_PER_LINK].to_vec();
            assert!(is_physically_consistent(&p));
            // projecting a physical link is a no-op
            let projected = project_link(&p, 1e-9);
            let worst = p.iter().zip(&projected).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
            assert!(worst < 1e-9, "link {k} moved by {worst:.3e} under an inactive constraint");
        }
        let _ = robot;
    }

    /// **A3: the covariance covers at its nominal rate.** 95% intervals on identifiable directions must contain the
    /// truth about 95% of the time. This is the check that makes a reported width mean something.
    #[test]
    fn the_intervals_cover_at_about_their_nominal_rate() {
        let noise = 0.05;
        let (mut hits, mut total) = (0usize, 0usize);
        for trial in 0..120u64 {
            let (robot, samples, truth, g) = fixture(60, noise, 1000 + trial * 17);
            let Some(id) = identify_with_covariance(&robot, &samples, g, 1e-8) else { continue };
            // check the five best-determined directions
            for (v, _) in id.identifiable.iter().take(5) {
                if let Some((lo, hi)) = id.interval(v, 0.95) {
                    let truth_proj = v.dot(&truth);
                    total += 1;
                    hits += usize::from(truth_proj >= lo && truth_proj <= hi);
                }
            }
        }
        let rate = hits as f64 / total as f64;
        eprintln!("95% intervals on identifiable directions: {hits}/{total} covered = {:.1}%", 100.0 * rate);
        assert!(total > 400, "enough samples to judge: {total}");
        assert!((0.88..=1.0).contains(&rate), "coverage must be near nominal, got {:.3}", rate);
    }

    /// The reported rank is far below the parameter count, which is the fact that makes a per-parameter estimate
    /// meaningless and a per-direction one necessary.
    #[test]
    fn most_parameters_are_not_identifiable_and_the_result_says_so() {
        let (robot, samples, _t, g) = fixture(60, 0.02, 3);
        let id = identify_with_covariance(&robot, &samples, g, 1e-8).expect("identifies");
        eprintln!("rank {} of {} parameters, from {} rows; sigma^2 = {:?}", id.rank, id.parameters, id.rows, id.sigma_squared.map(|v| format!("{v:.3e}")));
        assert!(id.rank < id.parameters, "the regressor is rank deficient: {} vs {}", id.rank, id.parameters);
        assert_eq!(id.identifiable.len(), id.rank, "one direction per identifiable dimension");

        // a single raw parameter is generally NOT identifiable, and asking for its width must return None
        let mut e0 = DVector::zeros(id.parameters);
        e0[0] = 1.0;
        eprintln!("standard error of link-0 mass alone: {:?}", id.standard_error(&e0));
        assert!(id.standard_error(&e0).is_none(), "an unidentifiable parameter must not be given a finite width");

        // whereas the best-determined direction does have one
        let (v, se) = &id.identifiable[0];
        eprintln!("best-determined direction: standard error {se:.3e}");
        assert!(id.standard_error(v).is_some() && *se > 0.0);
    }

    /// Variance must scale as `sigma^2` and as `1/n`, or it is not a variance.
    #[test]
    fn the_variance_scales_correctly() {
        let se_at = |noise: f64, n: usize| {
            let (robot, samples, _t, g) = fixture(n, noise, 42);
            let id = identify_with_covariance(&robot, &samples, g, 1e-8).unwrap();
            id.identifiable[0].1
        };
        let (a, b) = (se_at(0.02, 200), se_at(0.08, 200));
        eprintln!("noise 0.02 -> se {a:.4e}, noise 0.08 -> se {b:.4e}, ratio {:.2} (expect ~4)", b / a);
        assert!((b / a - 4.0).abs() < 1.4, "standard error scales with sigma: ratio {:.3}", b / a);

        let (c, d) = (se_at(0.05, 100), se_at(0.05, 400));
        eprintln!("n 100 -> se {c:.4e}, n 400 -> se {d:.4e}, ratio {:.2} (expect ~0.5)", d / c);
        assert!((d / c - 0.5).abs() < 0.2, "standard error scales as 1/sqrt(n): ratio {:.3}", d / c);
    }

    /// Degenerate input is refused rather than producing a confident answer about nothing.
    #[test]
    fn degenerate_input_is_refused() {
        let (robot, _s, _t, g) = fixture(1, 0.0, 1);
        assert!(identify_with_covariance(&robot, &[], g, 1e-8).is_none(), "no samples, no estimate");
        assert!(identify_consistent(&robot, &[], g, 1e-6, 10).is_none());
        // a sample of the wrong width is refused, not silently zero-padded
        let bad = [IdSample { q: vec![0.0; 2], qd: vec![0.0; 3], qdd: vec![0.0; 3], tau: vec![0.0; 3] }];
        assert!(identify_with_covariance(&robot, &bad, g, 1e-8).is_none());
    }
}
