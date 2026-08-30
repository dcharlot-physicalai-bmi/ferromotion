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
//! [`ActuatorFit::stderr`] does the same for the *actuator* terms, which had only a point estimate, a
//! trajectory conditioning number and an RMS residual — none of which say how far the plant might be
//! from the fit. Its coverage is verified rather than asserted, and so is the regime where it stops
//! being a valid interval: differentiating an encoder to get rates puts the noise in the regressor
//! instead of the response, and there the width is overconfident by one to two orders of magnitude
//! while the ratio survives as an alarm. Both measurements are on the type.
//!
//! **Parameters that describe a real body** — [`identify_consistent`]. A link is physically consistent exactly when its
//! pseudo-inertia `J(phi)` is positive definite ([`pseudo_inertia`](crate::pseudo_inertia)), and
//! [`is_physically_consistent`](crate::is_physically_consistent) already tests it — but had no callers outside its own
//! test, and the projection needed to enforce it already sat in the same crate unwired. Enforcing it uses the freedom
//! rank deficiency leaves: among all parameter vectors that fit the torques equally well, prefer one that could be a
//! body.

use crate::{inertial_regressor, pseudo_inertia, IdSample, LinkInertia, Robot, PARAMS_PER_LINK};
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

/// **Solve one joint's normal system and report whether the data could support it.**
///
/// Shared by [`identify_actuator`] and [`identify_actuator_with_gain`] so the two cannot drift apart. That is
/// not hypothetical caution: this crate shipped a version where `gendyn` and `dynamics` implemented the same
/// recursion and silently disagreed, because the second copy was written later and one term was missed.
///
/// `ata`/`atb` are the accumulated normal equations, `colnorm` each column's `Σx²`. Returns the solution,
/// a scale-free conditioning number (smallest eigenvalue after unit-normalising every column, so it reports
/// linear dependence rather than the units of `q̈` against `q̇`), and `None` for the solution when the system is
/// too singular to answer — a readable-looking number would be worse.
fn solve_normal(k: usize, ata: &[f64], atb: &[f64], colnorm: &[f64]) -> (Option<DVector<f64>>, f64) {
    let d: Vec<f64> = (0..k).map(|i| colnorm[i].sqrt()).collect();
    // A column with zero norm carries NO information about its parameter, so the system is unidentifiable and
    // must say so. An earlier version substituted a unit basis vector for such a column, which made a DEAD
    // column contribute an eigenvalue of exactly 1 and inflated the conditioning number — the one thing this
    // function exists to prevent. Bail before the eigendecomposition instead.
    if d.iter().any(|x| !x.is_finite() || *x <= 0.0) {
        return (None, 0.0);
    }
    let scaled = DMatrix::from_fn(k, k, |a, b| ata[a * k + b] / (d[a] * d[b]));
    // FIX: a non-finite entry anywhere makes the eigenvalues meaningless, and `f64::min` SKIPS NaN, so an
    // all-NaN spectrum would fold back to the +INFINITY seed and read as perfectly identifiable.
    if scaled.iter().any(|x| !x.is_finite()) {
        return (None, 0.0);
    }
    let conditioning = scaled
        .symmetric_eigenvalues()
        .iter()
        .fold(f64::INFINITY, |x: f64, &y| x.min(y.abs()));
    let conditioning = if conditioning.is_finite() { conditioning } else { 0.0 };
    let m = DMatrix::from_row_slice(k, k, ata);
    let rhs = DVector::from_row_slice(atb);
    (m.lu().solve(&rhs).filter(|_| conditioning > 1e-12), conditioning)
}

/// A motion **planned but not yet run** — the input to [`confounding`].
///
/// No measurement: only what a trajectory generator already knows. That is the point, because the whole value
/// of screening an excitation is doing it before the arm moves.
#[derive(Clone, Debug)]
pub struct PlannedMotion {
    pub q: Vec<f64>,
    pub qd: Vec<f64>,
    pub qdd: Vec<f64>,
}

/// **Which parameters a planned excitation cannot tell apart**, per joint.
#[derive(Clone, Copy, Debug)]
pub struct Confounding {
    pub joint: usize,
    /// Same scale as [`ActuatorFit::conditioning`]: the smallest eigenvalue after unit-normalising each column.
    pub conditioning: f64,
    /// The direction in parameter space the data cannot resolve, unit length, ordered
    /// `(k_t, armature, damping, friction)`. This is the field a scalar cannot replace: a low `conditioning`
    /// says *something* is confounded, and only this says **what**.
    pub direction: [f64; 4],
}

/// Names of the four parameters, in the order [`Confounding::direction`] uses.
pub const ACTUATOR_PARAMETERS: [&str; 4] = ["k_t", "armature", "damping", "friction"];

impl Confounding {
    /// The two parameters carrying most of the unresolvable direction, largest first. This is the answer to
    /// "what is wrong with my trajectory", which is what a caller actually needs.
    pub fn worst_pair(&self) -> (&'static str, &'static str) {
        let mut idx: Vec<usize> = (0..4).collect();
        idx.sort_by(|&a, &b| {
            self.direction[b].abs().partial_cmp(&self.direction[a].abs()).unwrap_or(std::cmp::Ordering::Equal)
        });
        (ACTUATOR_PARAMETERS[idx[0]], ACTUATOR_PARAMETERS[idx[1]])
    }
}

/// **Screen an excitation before running it: which actuator parameters can this motion actually determine?**
///
/// [`ActuatorFit::conditioning`] tells you a fit was degenerate *after* you have collected the data, and it is a
/// scalar, so it cannot say which parameters were confounded. That gap is not academic — it produced a wrong
/// conclusion in this very crate. A test built a constant-velocity motion, saw `conditioning` collapse, and
/// concluded the torque constant was unidentifiable. It was not: the unresolvable direction was
/// `[0.000, 0.000, +0.707, −0.707]`, zero weight on `k_t`, and the real confounding was **damping against
/// friction** — something the three-term fit has too. Reading the direction rather than the scalar would have
/// said so immediately.
///
/// This needs no measurement. The four regression columns are `[I, −q̈, −q̇, −tanh(q̇/ε)]`, and for a *planned*
/// motion all four are computable: the first from the model's own torque divided by `torque_constant`, the rest
/// from the trajectory. So an excitation can be rejected at the desk instead of on the bench.
///
/// **It depends on your prior estimates.** The predicted current comes from the robot's currently-stated
/// actuator terms, so a screening run against wildly wrong priors is approximate. It is reliable about
/// *structural* degeneracy — a column that is identically zero, or two columns that are proportional for
/// geometric reasons — which is the failure mode worth screening for.
///
/// The canonical structural case: a joint whose axis is **parallel to gravity** has `τ_rigid = M·q̈` with `M`
/// constant, so the current column is proportional to `q̈` and `k_t` is inseparable from the armature at any
/// excitation. This reports that as a `(k_t, armature)` pair, which is actionable in a way `conditioning ≈ 0`
/// is not.
pub fn confounding(
    robot: &Robot,
    inertia: &[LinkInertia],
    plan: &[PlannedMotion],
    gravity: Vector3<f64>,
    torque_constant: f64,
) -> Vec<Confounding> {
    let n = robot.dof();
    let mut rigid = robot.clone();
    for j in rigid.joints.iter_mut() {
        *j = j.clone().with_armature(-1.0).with_damping(-1.0).with_friction(-1.0);
    }
    let eps = crate::COULOMB_SMOOTHING;
    const K: usize = 4;
    let mut ata = vec![[0.0f64; K * K]; n];
    let mut colnorm = vec![[0.0f64; K]; n];

    for m in plan {
        if m.q.len() != n || m.qd.len() != n || m.qdd.len() != n {
            continue;
        }
        // What the arm would have to draw, from the model as it currently stands.
        let full = crate::inverse_dynamics(robot, inertia, &m.q, &m.qd, &m.qdd, gravity);
        for i in 0..n {
            let current = if torque_constant.abs() > 0.0 { full[i] / torque_constant } else { 0.0 };
            let c = [current, -m.qdd[i], -m.qd[i], -(m.qd[i] / eps).tanh()];
            for a in 0..K {
                for b in 0..K {
                    ata[i][a * K + b] += c[a] * c[b];
                }
                colnorm[i][a] += c[a] * c[a];
            }
        }
    }
    let _ = &rigid; // the baseline is not needed for identifiability, only the column geometry is

    (0..n)
        .map(|i| {
            let d: Vec<f64> = (0..K).map(|k| colnorm[i][k].sqrt()).collect();
            // A dead column is its own answer: that parameter has no leverage, so the unresolvable direction
            // IS that axis. Reporting a unit vector on it is more useful than refusing.
            if let Some(dead) = (0..K).find(|&k| !d[k].is_finite() || d[k] <= 0.0) {
                let mut dir = [0.0; K];
                dir[dead] = 1.0;
                return Confounding { joint: i, conditioning: 0.0, direction: dir };
            }
            let scaled = DMatrix::from_fn(K, K, |a, b| ata[i][a * K + b] / (d[a] * d[b]));
            if scaled.iter().any(|x| !x.is_finite()) {
                return Confounding { joint: i, conditioning: 0.0, direction: [0.0; K] };
            }
            let se = scaled.symmetric_eigen();
            let mut best = 0usize;
            for k in 1..K {
                if se.eigenvalues[k].abs() < se.eigenvalues[best].abs() {
                    best = k;
                }
            }
            let v = se.eigenvectors.column(best);
            // Sign is arbitrary for an eigenvector; fix it so the largest component is positive and two runs
            // of the same input print the same thing.
            let mut lead = 0usize;
            for k in 1..K {
                if v[k].abs() > v[lead].abs() {
                    lead = k;
                }
            }
            let sign = if v[lead] < 0.0 { -1.0 } else { 1.0 };
            Confounding {
                joint: i,
                conditioning: se.eigenvalues[best].abs(),
                direction: [sign * v[0], sign * v[1], sign * v[2], sign * v[3]],
            }
        })
        .collect()
}

/// A motion sample where the **current** was recorded and the torque was not.
///
/// The distinction is the whole point of [`identify_actuator_with_gain`]. A servo without a torque sensor
/// reports current; converting it to torque needs `k_t`, and inheriting `k_t` from a catalogue bounds every
/// fitted parameter by its own error — measured on the SO-101 wrist, a 10% wrong `k_t` gives a **10.0%** wrong
/// damping (exactly one for one, because at that joint nearly all the torque is actuator rather than
/// rigid-body) and a 7.7% wrong armature. The proportion is not universal — it depends on how much of the
/// joint's torque is actuator versus rigid-body — but nothing averages it out.
#[derive(Clone, Debug)]
pub struct CurrentSample {
    pub q: Vec<f64>,
    pub qd: Vec<f64>,
    pub qdd: Vec<f64>,
    /// Measured motor current per joint (A), referred to the joint output. **Not torque.**
    pub current: Vec<f64>,
}

/// What identifying a joint's actuator terms *and* its torque constant produced.
#[derive(Clone, Copy, Debug)]
pub struct ActuatorGainFit {
    pub joint: usize,
    /// Fitted torque constant `k_t` (N·m/A), referred to the joint output. The parameter that was previously
    /// inherited and is now measured.
    pub torque_constant: f64,
    pub armature: f64,
    pub damping: f64,
    pub friction: f64,
    /// Scale-free identifiability of the **four**-column system. Adding the current column needs it to be
    /// linearly independent of `q̈`, `q̇` and `tanh(q̇/ε)`, which a gravity-loaded or friction-dominated
    /// trajectory can easily violate: at a joint where the current is spent almost entirely on one of those
    /// terms, the two columns are proportional and `k_t` trades off against it.
    pub conditioning: f64,
    pub residual: f64,
    /// All four non-negative, within the same float-zero allowance [`ActuatorFit::physical`] documents.
    pub physical: bool,
}

/// **Identify the actuator terms and the torque constant together, from current rather than torque.**
///
/// This dissolves the limitation [`identify_actuator`] carries. That function needs measured torque; a servo
/// without a torque sensor reports current, and the `k_t` used to convert bounds every fitted parameter by its
/// own error — 1:1 for a joint whose torque is almost all actuator, less for one dominated by its rigid-body
/// terms, and never averaged away. Making `k_t` a *fitted* parameter removes the inherited constant entirely.
///
/// It remains an exact linear fit. Writing the equation of motion with the conversion left in:
///
/// ```text
///   τ_rigid(q, q̇, q̈) = k_t·I − J_a·q̈ − b·q̇ − f·tanh(q̇/ε)
/// ```
///
/// the model-computable rigid-body torque is the target and the four unknowns multiply four columns
/// `[I, −q̈, −q̇, −tanh(q̇/ε)]`. Four columns, still linear, and the joints still decouple.
///
/// **What it costs, and the condition is not the obvious one.** The current column must be linearly independent
/// of the other three — equivalently, the target `τ_rigid` must lie **outside** `span{q̈, q̇, tanh(q̇/ε)}`. What
/// supplies that independence is the **gravity term**, because `G(q)` is a function of position and none of the
/// three columns is. So a gravity-loaded joint is the *best* case for `k_t`, not the worst.
///
/// The case that actually fails is a joint **decoupled from gravity** — an axis parallel to gravity, such as a
/// base yaw or a wrist roll. There `τ_rigid = M·q̈` with `M` constant, so the target is exactly proportional to
/// the `q̈` column and `k_t` is inseparable from the armature no matter how rich the excitation. Measured on a
/// 1-DOF vertical-axis arm under the same two-frequency excitation the oracle test uses: all four parameters
/// come back `NaN`, while [`identify_actuator`] on the identical motion recovers its three exactly.
///
/// Conversely, slowing a gravity-loaded trajectory 10,000x still recovers `k_t` to 1.9e-13 — it is the
/// *damping* that degrades there. So "excite harder" is the wrong instinct: check `conditioning`, and for a
/// gravity-parallel joint measure `k_t` some other way (a locked-rotor and back-EMF pair) and use
/// [`identify_actuator`] instead.
///
/// **What it does not fix.** `q̈` still comes from differentiating position twice, so the guidance in
/// [`identify_actuator`] about reconstructing rates with [`crate::SavGol`] rather than finite differences
/// applies unchanged.
///
/// Same contract as [`identify_actuator`] on two points worth repeating rather than cross-referencing. The
/// rigid-body torque is computed with the robot's **own armature, damping and friction all cleared**, so a
/// model already carrying estimates gets a fresh fit and not a fit of the leftover. And a sample whose slices
/// do not all have length `robot.dof()` is **dropped**, not reshaped — check `residual` and `conditioning`
/// rather than assuming every sample was used.
pub fn identify_actuator_with_gain(
    robot: &Robot,
    inertia: &[LinkInertia],
    samples: &[CurrentSample],
    gravity: Vector3<f64>,
) -> Vec<ActuatorGainFit> {
    let n = robot.dof();
    // Same clearing as `identify_actuator`, and for the same reason: ALL fitted terms must be absent from the
    // baseline or the fit sees a leftover instead of a signal.
    let mut rigid = robot.clone();
    for j in rigid.joints.iter_mut() {
        *j = j.clone().with_armature(-1.0).with_damping(-1.0).with_friction(-1.0);
    }
    let eps = crate::COULOMB_SMOOTHING;

    const K: usize = 4;
    let mut ata = vec![[0.0f64; K * K]; n];
    let mut atb = vec![[0.0f64; K]; n];
    let mut colnorm = vec![[0.0f64; K]; n];
    let mut rows: Vec<Vec<[f64; K + 1]>> = Vec::with_capacity(samples.len());

    for s in samples {
        if s.q.len() != n || s.qd.len() != n || s.qdd.len() != n || s.current.len() != n {
            continue;
        }
        let tr = crate::inverse_dynamics(&rigid, inertia, &s.q, &s.qd, &s.qdd, gravity);
        let mut row = Vec::with_capacity(n);
        for i in 0..n {
            let c = [s.current[i], -s.qdd[i], -s.qd[i], -(s.qd[i] / eps).tanh()];
            let y = tr[i];
            for a in 0..K {
                for b in 0..K {
                    ata[i][a * K + b] += c[a] * c[b];
                }
                atb[i][a] += c[a] * y;
                colnorm[i][a] += c[a] * c[a];
            }
            row.push([c[0], c[1], c[2], c[3], y]);
        }
        rows.push(row);
    }

    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let (sol, conditioning) = solve_normal(K, &ata[i], &atb[i], &colnorm[i]);
        let (kt, a, b, f) = match &sol {
            Some(v) => (v[0], v[1], v[2], v[3]),
            None => (f64::NAN, f64::NAN, f64::NAN, f64::NAN),
        };
        let mut resid_sq = 0.0;
        if kt.is_finite() {
            for row in &rows {
                let c = &row[i];
                let e = c[4] - (kt * c[0] + a * c[1] + b * c[2] + f * c[3]);
                resid_sq += e * e;
            }
        }
        let count = rows.len().max(1);
        out.push(ActuatorGainFit {
            joint: i,
            torque_constant: kt,
            armature: a,
            damping: b,
            friction: f,
            conditioning,
            residual: if kt.is_finite() { (resid_sq / count as f64).sqrt() } else { f64::NAN },
            physical: {
                let floor = -1e-9 * kt.abs().max(a.abs()).max(b.abs()).max(f.abs());
                kt >= floor && a >= floor && b >= floor && f >= floor
            },
        });
    }
    out
}

/// What identifying one joint's actuator terms produced, and whether the data could support it.
#[derive(Clone, Copy, Debug)]
pub struct ActuatorFit {
    pub joint: usize,
    /// Fitted reflected rotor inertia `J_a` (kg·m²) — see [`crate::Joint::armature`].
    pub armature: f64,
    /// Fitted viscous damping `b` (N·m·s/rad) — see [`crate::Joint::damping`].
    pub damping: f64,
    /// Fitted Coulomb friction magnitude `f` (N·m) — see [`crate::Joint::friction`]. Fitted against the same
    /// smoothed basis the dynamics apply, `tanh(q̇/ε)`, so the number means what the simulator will use.
    pub friction: f64,
    /// **Whether the excitation can separate the three terms at all** — and *only* that. The smallest
    /// eigenvalue of the 3×3 normal matrix after each column is normalized to unit length, so it is scale-free
    /// rather than a report on the units of `q̈` against `q̇`. It goes to zero when the columns become linearly
    /// dependent, which is not a contrived case: any purely exponential motion has `q̈ = k·q̇` exactly. When
    /// this is small the fitted values trade off against each other and none is meaningful alone.
    ///
    /// **A good value here does not mean the fit is good.** This is a property of the trajectory, not of the
    /// data quality, and the two come apart badly in practice. Measured on the SO-101 (`so101_reach_rl
    /// --identify`): reconstructing `q̈` by central differences of a 12-bit encoder puts the armature 88–217%
    /// out and often negative, while `conditioning` sits at `1.000` for every one of those rows. Check
    /// [`ActuatorFit::residual`] against your torque scale as well — on that same comparison it moves from
    /// `1e-15` to `1e-1`, which is the signal that something is wrong.
    pub conditioning: f64,
    /// Root-mean-square torque residual after the fit (N·m). Large means the actuator is doing something
    /// neither term describes — friction, backlash, a saturating drive.
    pub residual: f64,
    /// **How wrong each fitted value might be**: one standard error for `(armature, damping, friction)`,
    /// in each parameter's own units.
    ///
    /// `sqrt(diag(σ²(AᵀA)⁻¹))` with `σ² = SSR/(rows − 3)`, the textbook least-squares result, from the
    /// same normal system the fit itself uses.
    ///
    /// **It is a valid interval only under the model least squares assumes**, which is noise in the
    /// measured torque and not in `q̈` or `q̇`. Under that assumption it is correct: measured coverage of
    /// a nominal 95% interval is 94.0–97.3% over 300 trials, and halving or quadrupling the variance
    /// moves it to 67.7% or 100.0%, so the scale is pinned.
    ///
    /// **Reconstructing rates by differentiating an encoder breaks that assumption**, and the interval
    /// then understates the error badly. Noise lands in the regressor rather than the response, which
    /// biases the estimate, and a standard error cannot see a bias. Measured on the SO-101
    /// (`so101_reach_rl --identify`), truth minus estimate in units of this standard error:
    ///
    /// | rate reconstruction | armature error | error / stderr |
    /// |---|---|---|
    /// | 12-bit, central difference | 88–217% | **100–289** |
    /// | 16-bit, central difference | 2.9–48% | 9–30 |
    /// | 12-bit, Savitzky-Golay (50 ms) | 1.5–2.7% | 6–41 |
    ///
    /// So the width is overconfident by one to two orders of magnitude there. What survives is the
    /// **ratio as an alarm**: a value many standard errors from what the model predicts says the
    /// residual is not independent torque noise, which is the signature of regressor noise or a term
    /// the model does not have. Read it as "something is wrong", not as a width.
    ///
    /// That also bounds its use for domain randomisation. Sampling over `estimate ± k·stderr`
    /// randomises over what the data supports **when the torque measurement dominates the error**. With
    /// differentiated rates it would randomise over an interval far too narrow, and around a biased
    /// centre, which is worse than an honestly wide guess.
    ///
    /// [`ActuatorFit::conditioning`] answers a different question again, and only one: whether the
    /// trajectory can separate the three terms at all. It sits at `1.000` through every row above.
    ///
    /// `None` when the fit is unidentifiable, when `AᵀA` is singular, or when there are 3 or fewer
    /// samples, since `σ²` then has no degrees of freedom to be estimated from.
    pub stderr: Option<[f64; 3]>,
    /// Whether all three fitted values are non-negative. A negative rotor inertia, damping or friction is
    /// unphysical; it is **reported rather than clamped**, because a clamped value looks like a measurement.
    ///
    /// A term is allowed to be negative by up to `1e-9` of the largest fitted magnitude, because a least
    /// squares solution for a truly-absent term lands on floating-point noise rather than exactly zero — a
    /// noise-free fit of a frictionless joint returned `-1.4e-15` against terms of order `1e-2`. That allowance
    /// is a **float-zero allowance and not a physical tolerance**: it is thirteen orders below any real value,
    /// so it cannot launder a genuinely negative fit into a physical one.
    pub physical: bool,
}

/// **Identify a joint's reflected rotor inertia and viscous damping from motion data.**
///
/// The term a URDF cannot state is also the term hardest to look up: `J_rotor` for a given servo is rarely
/// published, and a gear ratio squared multiplies whatever error the estimate carries. This measures it instead.
///
/// It is an exact linear fit, not a search, because RNEA is **linear in all three terms**:
///
/// ```text
///   τᵢ = τ_rigid,ᵢ(q, q̇, q̈) + J_aᵢ·q̈ᵢ + bᵢ·q̇ᵢ + fᵢ·tanh(q̇ᵢ/ε)
/// ```
///
/// The Coulomb column is nonlinear in `q̇` and still **linear in the parameter**, which is all a regression
/// needs — `tanh(q̇/ε)` is a fixed function of a measured quantity. So subtracting the rigid-body torque leaves
/// a three-column regression per joint, and the joints **decouple**: each one is its own independent
/// 3-parameter problem. That is the same structure that makes inertial identification a linear regression
/// ([`crate::identify`]), one level further out in the drivetrain.
///
/// Two things this reports rather than hides:
///
/// - **Identifiability.** `q̈` and `q̇` proportional across the samples makes the normal matrix singular and the
///   split between inertia and damping arbitrary. `conditioning` is that number; a small value means the terms
///   are not separable from this data no matter how small the residual is. The converse does **not** hold — see
///   [`ActuatorFit::conditioning`], which reads `1.000` on data that gets the armature 200% wrong.
///
///   The Coulomb column costs conditioning, and it is worth knowing by how much. `tanh(q̇/ε)` with `ε` three
///   orders below the joint rates is very nearly `sign(q̇)`, and for a sinusoidal `q̇` a square wave correlates
///   with it at about 0.9. On the same two-frequency excitation that gave `conditioning ≈ 1.0` for the
///   two-term fit, the three-term fit reports **0.191** — still an exact recovery on noise-free data, but a
///   third of the margin against noise. Friction wants richer excitation than damping does.
///
/// A practical note from measuring this on a real arm's numbers, because it decides whether the method is
/// usable at all: `q̈` is not measured on hardware, it is differentiated twice from position, and quantisation
/// arrives scaled by `1/dt²`. Damping survives that (it multiplies `q̇`, one differentiation) and armature does
/// not. Differentiating with [`crate::SavGol`] instead of a finite difference took the armature error on a
/// 12-bit encoder from 217% to 2.7%; a plain central difference is not good enough at any sample count.
/// - **Physicality.** Negative values come back as-is with `physical: false`, because a clamped zero is
///   indistinguishable from a measured zero.
///
/// The rigid-body torque is computed with the robot's **own armature, damping and friction all cleared**, so
/// passing a model that already carries estimates returns a fresh fit rather than a fit of the leftover. That trap is quiet
/// otherwise: the numbers would look plausible and mean something else.
pub fn identify_actuator(
    robot: &Robot,
    inertia: &[LinkInertia],
    samples: &[IdSample],
    gravity: Vector3<f64>,
) -> Vec<ActuatorFit> {
    let n = robot.dof();
    // Strip any existing estimates so the residual is the FULL actuator contribution, not what is left after
    // whatever the model already claimed.
    let mut rigid = robot.clone();
    for j in rigid.joints.iter_mut() {
        // ALL THREE. Missing one here is silent: the baseline keeps that term, subtracting it removes the
        // signal, and the fit confidently returns zero for it. Caught by the oracle test when friction was
        // added and came back as -1.4e-15 against a truth of 0.08.
        *j = j.clone().with_armature(-1.0).with_damping(-1.0).with_friction(-1.0);
    }

    // Per joint: a 3x3 normal system in (J_a, b, f), accumulated in one pass. Three columns and still a
    // LINEAR fit, because `tanh(q̇/ε)` is a fixed function of the measured rate — the parameter multiplies it.
    let eps = crate::COULOMB_SMOOTHING;
    let mut ata = vec![[0.0f64; 9]; n];
    let mut atb = vec![[0.0f64; 3]; n];
    let mut colnorm = vec![[0.0f64; 3]; n]; // per-column Σx², for a scale-free conditioning number
    let mut resid_sq = vec![0.0f64; n];
    let mut count = 0usize;

    let mut rows: Vec<Vec<[f64; 4]>> = Vec::with_capacity(samples.len()); // per joint: [c0, c1, c2, residual]
    for s in samples {
        if s.q.len() != n || s.qd.len() != n || s.qdd.len() != n || s.tau.len() != n {
            continue; // a malformed sample is dropped rather than silently reshaping the problem
        }
        let tr = crate::inverse_dynamics(&rigid, inertia, &s.q, &s.qd, &s.qdd, gravity);
        let mut row = Vec::with_capacity(n);
        for i in 0..n {
            let c = [s.qdd[i], s.qd[i], (s.qd[i] / eps).tanh()];
            let y = s.tau[i] - tr[i];
            for a in 0..3 {
                for b in 0..3 {
                    ata[i][a * 3 + b] += c[a] * c[b];
                }
                atb[i][a] += c[a] * y;
                colnorm[i][a] += c[a] * c[a];
            }
            row.push([c[0], c[1], c[2], y]);
        }
        rows.push(row);
        count += 1;
    }

    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        // The same solver `identify_actuator_with_gain` uses, so a fix to one reaches both.
        let (sol, conditioning) = solve_normal(3, &ata[i], &atb[i], &colnorm[i]);
        let (armature, damping, friction) = match &sol {
            Some(v) => (v[0], v[1], v[2]),
            // Unidentifiable: say so rather than return three readable-looking numbers.
            None => (f64::NAN, f64::NAN, f64::NAN),
        };

        if armature.is_finite() {
            for row in &rows {
                let c = &row[i];
                let e = c[3] - armature * c[0] - damping * c[1] - friction * c[2];
                resid_sq[i] += e * e;
            }
        }
        // Standard errors from the same normal system: Cov = σ²(AᵀA)⁻¹, σ² = SSR/(rows − 3). The
        // divisor is rows − 3 rather than rows, because three parameters were fitted from this data and
        // the residual is correspondingly optimistic; `residual` above reports plain RMS and is a
        // different quantity.
        let stderr = if armature.is_finite() && count > 3 {
            DMatrix::from_row_slice(3, 3, &ata[i]).try_inverse().and_then(|inv| {
                let s2 = resid_sq[i] / (count - 3) as f64;
                let e = [
                    (s2 * inv[(0, 0)]).sqrt(),
                    (s2 * inv[(1, 1)]).sqrt(),
                    (s2 * inv[(2, 2)]).sqrt(),
                ];
                e.iter().all(|v| v.is_finite()).then_some(e)
            })
        } else {
            None
        };
        out.push(ActuatorFit {
            joint: i,
            armature,
            damping,
            friction,
            conditioning,
            residual: if count > 0 && armature.is_finite() {
                (resid_sq[i] / count as f64).sqrt()
            } else {
                f64::NAN
            },
            stderr,
            physical: {
                let floor = -1e-9 * armature.abs().max(damping.abs()).max(friction.abs());
                armature >= floor && damping >= floor && friction >= floor
            },
        });
    }
    out
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

    /// A two-link arm for the actuator-identification tests.
    fn geared_arm(armature: f64, damping: f64) -> (Robot, Vec<LinkInertia>) {
        const URDF: &str = r#"<robot name="two">
          <link name="base"/>
          <link name="l1"><inertial><origin xyz="0.3 0 0"/><mass value="1.5"/>
            <inertia ixx="0.02" ixy="0" ixz="0" iyy="0.02" iyz="0" izz="0.02"/></inertial></link>
          <link name="l2"><inertial><origin xyz="0.2 0 0"/><mass value="0.8"/>
            <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial></link>
          <link name="tool"/>
          <joint name="j1" type="revolute"><parent link="base"/><child link="l1"/><origin xyz="0 0 0"/>
            <axis xyz="0 1 0"/><limit lower="-3" upper="3" effort="8" velocity="4"/></joint>
          <joint name="j2" type="revolute"><parent link="l1"/><child link="l2"/><origin xyz="0.6 0 0"/>
            <axis xyz="0 1 0"/><limit lower="-3" upper="3" effort="5" velocity="4"/></joint>
          <joint name="jt" type="fixed"><parent link="l2"/><child link="tool"/><origin xyz="0.4 0 0"/></joint>
        </robot>"#;
        let (mut robot, inertia) = crate::from_urdf_full(URDF, "base", "tool").unwrap();
        for (i, j) in robot.joints.iter_mut().enumerate() {
            *j = j
                .clone()
                .with_armature(armature + 0.003 * i as f64)
                .with_damping(damping - 0.1 * i as f64)
                .with_friction(0.08 + 0.02 * i as f64);
        }
        (robot, inertia)
    }

    /// **The oracle: generate torques from known actuator terms, then recover them.**
    ///
    /// This is the same shape of test the inertial identification uses — synthesise from ground truth so the
    /// answer is known, rather than checking that a fit merely converged. The excitation is deliberately
    /// two-frequency, because a single sinusoid cannot separate inertia from damping (see the next test).
    #[test]
    fn identify_actuator_recovers_known_terms() {
        let (robot, inertia) = geared_arm(0.0119, 0.64);
        let g = Vector3::new(0.0, 0.0, -9.81);
        let mut samples = Vec::new();
        for k in 0..400 {
            let t = k as f64 * 0.002;
            // Two frequencies per joint, so q̈ and q̇ are not proportional.
            let q = vec![0.4 * (3.0 * t).sin(), -0.3 * (5.0 * t).cos()];
            let qd = vec![1.2 * (3.0 * t).cos() + 0.5 * (11.0 * t).cos(), 1.5 * (5.0 * t).sin()];
            let qdd = vec![-3.6 * (3.0 * t).sin() - 5.5 * (11.0 * t).sin(), 7.5 * (5.0 * t).cos()];
            // Ground truth INCLUDES the actuator terms, because `robot` states them.
            let tau = crate::inverse_dynamics(&robot, &inertia, &q, &qd, &qdd, g);
            samples.push(IdSample { q, qd, qdd, tau });
        }

        let fits = identify_actuator(&robot, &inertia, &samples, g);
        assert_eq!(fits.len(), 2);
        for (i, f) in fits.iter().enumerate() {
            let (true_a, true_b) = (0.0119 + 0.003 * i as f64, 0.64 - 0.1 * i as f64);
            let true_f = 0.08 + 0.02 * i as f64;
            assert!(f.physical, "joint {i} fit must be physical, got {f:?}");
            assert!(f.conditioning > 1e-3, "joint {i} excitation should be identifiable, got {}", f.conditioning);
            assert!((f.armature - true_a).abs() < 1e-9, "joint {i} armature: {} vs {true_a}", f.armature);
            assert!((f.damping - true_b).abs() < 1e-9, "joint {i} damping: {} vs {true_b}", f.damping);
            assert!((f.friction - true_f).abs() < 1e-9, "joint {i} friction: {} vs {true_f}", f.friction);
            assert!(f.residual < 1e-9, "joint {i} residual {} should be ~0 on noise-free data", f.residual);
        }
    }

    /// **The reported standard error must have the coverage it claims.**
    ///
    /// A number that is merely *returned* is worthless as an uncertainty. The test is frequentist
    /// coverage: with Gaussian torque noise, repeat the identification many times on independent noise
    /// draws and count how often the truth lands inside `estimate ± 1.96·stderr`. If the formula is
    /// right that happens 95% of the time.
    ///
    /// **What it does and does not catch**, mutation-checked rather than assumed. Scaling the variance
    /// by 1/4 drops coverage to 67.7% and by 4 raises it to 100.0%, so a factor error in either
    /// direction fails, including the dangerous one where the reported uncertainty is too confident.
    /// Replacing the `rows − 3` divisor with `rows` does **not** fail, and cannot: at 400 samples that is
    /// a 0.4% change in `σ`, far inside the binomial scatter. The degrees-of-freedom correction earns
    /// its place when samples are few, which is not the regime this fixture is in.
    ///
    /// Noise is added to the torque only, which is the model least squares assumes: error in the
    /// response, not in the regressors. Real encoder differentiation puts noise in `q̈` as well, and
    /// that case is exactly where [`ActuatorFit::conditioning`] reads 1.000 while the armature is
    /// wildly out. This test is about the estimator being correct on its own assumptions.
    #[test]
    fn the_standard_error_covers_the_truth_at_the_rate_it_claims() {
        let (robot, inertia) = geared_arm(0.0119, 0.64);
        let g = Vector3::new(0.0, 0.0, -9.81);
        let sigma = 0.02; // N·m of torque noise

        // clean regressors and clean torques once; only the noise is redrawn per trial
        let mut base = Vec::new();
        for k in 0..400 {
            let t = k as f64 * 0.002;
            let q = vec![0.4 * (3.0 * t).sin(), -0.3 * (5.0 * t).cos()];
            let qd = vec![1.2 * (3.0 * t).cos() + 0.5 * (11.0 * t).cos(), 1.5 * (5.0 * t).sin()];
            let qdd = vec![-3.6 * (3.0 * t).sin() - 5.5 * (11.0 * t).sin(), 7.5 * (5.0 * t).cos()];
            let tau = crate::inverse_dynamics(&robot, &inertia, &q, &qd, &qdd, g);
            base.push(IdSample { q, qd, qdd, tau });
        }

        let mut seed = 0x51ED_5EEDu64;
        let gauss = |seed: &mut u64| {
            // Box-Muller from the module's own uniform generator
            let u1 = ((lcg(seed) + 1.0) / 2.0).max(1e-12);
            let u2 = (lcg(seed) + 1.0) / 2.0;
            (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
        };

        let trials = 300;
        let mut inside = [[0usize; 3]; 2];
        let mut got_stderr = [[0.0f64; 3]; 2];
        for _ in 0..trials {
            let noisy: Vec<IdSample> = base
                .iter()
                .map(|s| IdSample {
                    q: s.q.clone(),
                    qd: s.qd.clone(),
                    qdd: s.qdd.clone(),
                    tau: s.tau.iter().map(|t| t + sigma * gauss(&mut seed)).collect(),
                })
                .collect();
            let fits = identify_actuator(&robot, &inertia, &noisy, g);
            for (i, f) in fits.iter().enumerate() {
                let se = f.stderr.expect("an identifiable fit with 400 samples must report a standard error");
                let truth = [0.0119 + 0.003 * i as f64, 0.64 - 0.1 * i as f64, 0.08 + 0.02 * i as f64];
                let est = [f.armature, f.damping, f.friction];
                for k in 0..3 {
                    got_stderr[i][k] += se[k] / trials as f64;
                    if (est[k] - truth[k]).abs() <= 1.96 * se[k] {
                        inside[i][k] += 1;
                    }
                }
            }
        }

        let names = ["armature", "damping", "friction"];
        for i in 0..2 {
            for k in 0..3 {
                let cov = inside[i][k] as f64 / trials as f64;
                eprintln!("joint {i} {}: coverage {:.1}%, mean stderr {:.3e}", names[k], 100.0 * cov, got_stderr[i][k]);
                // 95% nominal; the binomial standard deviation at n = 300 is 1.3%, so this is ~4 sigma
                assert!(
                    (0.90..=0.99).contains(&cov),
                    "joint {i} {} coverage {:.1}% is not the 95% claimed",
                    names[k],
                    100.0 * cov
                );
            }
        }

        // and it must SCALE: doubling the noise doubles the standard error
        let scaled: Vec<IdSample> = base
            .iter()
            .map(|s| IdSample {
                q: s.q.clone(),
                qd: s.qd.clone(),
                qdd: s.qdd.clone(),
                tau: s.tau.iter().map(|t| t + 2.0 * sigma * gauss(&mut seed)).collect(),
            })
            .collect();
        let big = identify_actuator(&robot, &inertia, &scaled, g);
        let se2 = big[0].stderr.expect("identifiable");
        let ratio = se2[1] / got_stderr[0][1];
        assert!((ratio - 2.0).abs() < 0.4, "doubling the torque noise should double the standard error, got {ratio:.3}x");
    }

    /// **Exponential motion cannot separate the terms, and the fit must say so.**
    ///
    /// For `q̇ = e^{kt}` the acceleration is exactly `k·q̇`, so the inertia and damping columns are parallel and
    /// a whole family of `(J_a, b)` fits equally well. A solver that returns a confident number here is worse
    /// than one that refuses: the value would look like a measurement.
    #[test]
    fn proportional_excitation_is_reported_as_unidentifiable() {
        let (robot, inertia) = geared_arm(0.0119, 0.64);
        let g = Vector3::zeros(); // no gravity, to keep the degeneracy clean
        let k = 3.0;
        let mut samples = Vec::new();
        for n in 0..200 {
            let t = n as f64 * 0.002;
            let v = (k * t).exp();
            let qd = vec![v, 0.5 * v];
            let qdd = vec![k * v, 0.5 * k * v]; // exactly proportional to qd
            let q = vec![v / k, 0.5 * v / k];
            let tau = crate::inverse_dynamics(&robot, &inertia, &q, &qd, &qdd, g);
            samples.push(IdSample { q, qd, qdd, tau });
        }
        let fits = identify_actuator(&robot, &inertia, &samples, g);
        for f in &fits {
            assert!(
                f.conditioning < 1e-6,
                "joint {} should report unidentifiable, got conditioning {}",
                f.joint,
                f.conditioning
            );
        }
        // The control: the SAME arm with two-frequency excitation IS identifiable, so the assertion above is
        // about the data and not about the function always returning zero.
        let mut good = Vec::new();
        for n in 0..200 {
            let t = n as f64 * 0.002;
            let q = vec![0.4 * (3.0 * t).sin(), -0.3 * (5.0 * t).cos()];
            let qd = vec![1.2 * (3.0 * t).cos() + 0.5 * (11.0 * t).cos(), 1.5 * (5.0 * t).sin()];
            let qdd = vec![-3.6 * (3.0 * t).sin() - 5.5 * (11.0 * t).sin(), 7.5 * (5.0 * t).cos()];
            let tau = crate::inverse_dynamics(&robot, &inertia, &q, &qd, &qdd, g);
            good.push(IdSample { q, qd, qdd, tau });
        }
        assert!(identify_actuator(&robot, &inertia, &good, g).iter().all(|f| f.conditioning > 1e-3));
    }

    /// **A model that already carries estimates must get a FRESH fit, not a fit of the leftover.** This is the
    /// quiet trap: the numbers would look plausible and mean something completely different.
    #[test]
    fn an_existing_estimate_does_not_bias_the_fit() {
        let (truth, inertia) = geared_arm(0.0119, 0.64);
        let g = Vector3::new(0.0, 0.0, -9.81);
        let mut samples = Vec::new();
        for n in 0..300 {
            let t = n as f64 * 0.002;
            let q = vec![0.4 * (3.0 * t).sin(), -0.3 * (5.0 * t).cos()];
            let qd = vec![1.2 * (3.0 * t).cos() + 0.5 * (11.0 * t).cos(), 1.5 * (5.0 * t).sin()];
            let qdd = vec![-3.6 * (3.0 * t).sin() - 5.5 * (11.0 * t).sin(), 7.5 * (5.0 * t).cos()];
            let tau = crate::inverse_dynamics(&truth, &inertia, &q, &qd, &qdd, g);
            samples.push(IdSample { q, qd, qdd, tau });
        }
        // Fit against a model claiming WILDLY wrong terms, and against one claiming none. Same answer.
        let (wrong, _) = geared_arm(0.5, 9.0);
        let (mut none, _) = geared_arm(0.0119, 0.64);
        for j in none.joints.iter_mut() {
            *j = j.clone().with_armature(-1.0).with_damping(-1.0);
        }
        let a = identify_actuator(&wrong, &inertia, &samples, g);
        let b = identify_actuator(&none, &inertia, &samples, g);
        for i in 0..2 {
            assert!((a[i].armature - b[i].armature).abs() < 1e-12, "joint {i} armature must not depend on the prior");
            assert!((a[i].damping - b[i].damping).abs() < 1e-12, "joint {i} damping must not depend on the prior");
            assert!((a[i].armature - (0.0119 + 0.003 * i as f64)).abs() < 1e-9);
        }
    }

    /// **A term the three-column basis genuinely cannot express must land in the residual.**
    ///
    /// This test used to inject Coulomb friction, which was the right check when the model had only inertia and
    /// damping — and became vacuous the moment friction became a fitted term, because the fit then recovers it
    /// and the residual drops to `1.5e-15`. So the injected term has to be one the basis cannot represent:
    /// quadratic drag `c·q̇·|q̇|`, which is neither constant in `q̇` nor linear in it.
    ///
    /// The point is not the drag. It is that a residual near zero means "these three terms explain the data",
    /// and the only way to keep that claim honest is to check the residual rises when something else is present.
    #[test]
    fn a_term_outside_the_basis_shows_up_in_the_residual() {
        let (robot, inertia) = geared_arm(0.0119, 0.64);
        let g = Vector3::new(0.0, 0.0, -9.81);
        let drag = 0.05;
        let mut samples = Vec::new();
        for n in 0..300 {
            let t = n as f64 * 0.002;
            let q = vec![0.4 * (3.0 * t).sin(), -0.3 * (5.0 * t).cos()];
            let qd = vec![1.2 * (3.0 * t).cos() + 0.5 * (11.0 * t).cos(), 1.5 * (5.0 * t).sin()];
            let qdd = vec![-3.6 * (3.0 * t).sin() - 5.5 * (11.0 * t).sin(), 7.5 * (5.0 * t).cos()];
            let mut tau = crate::inverse_dynamics(&robot, &inertia, &q, &qd, &qdd, g);
            for i in 0..2 {
                tau[i] += drag * qd[i] * qd[i].abs(); // quadratic: outside span{q̈, q̇, tanh(q̇/ε)}
            }
            samples.push(IdSample { q, qd, qdd, tau });
        }
        let fits = identify_actuator(&robot, &inertia, &samples, g);
        for f in &fits {
            assert!(f.residual > 1e-3, "joint {} should show the unmodelled term, residual {}", f.joint, f.residual);
        }

        // The control: without the drag, the SAME excitation leaves a residual at floating-point zero. Without
        // this, the assertion above could pass on a residual that is always large.
        let mut clean = Vec::new();
        for n in 0..300 {
            let t = n as f64 * 0.002;
            let q = vec![0.4 * (3.0 * t).sin(), -0.3 * (5.0 * t).cos()];
            let qd = vec![1.2 * (3.0 * t).cos() + 0.5 * (11.0 * t).cos(), 1.5 * (5.0 * t).sin()];
            let qdd = vec![-3.6 * (3.0 * t).sin() - 5.5 * (11.0 * t).sin(), 7.5 * (5.0 * t).cos()];
            let tau = crate::inverse_dynamics(&robot, &inertia, &q, &qd, &qdd, g);
            clean.push(IdSample { q, qd, qdd, tau });
        }
        for f in identify_actuator(&robot, &inertia, &clean, g) {
            assert!(f.residual < 1e-9, "joint {} residual {} should be ~0 with nothing extra", f.joint, f.residual);
        }
    }

    /// **The oracle for the four-parameter fit: recover `k_t` alongside the three actuator terms, from current.**
    ///
    /// This is the answer to a limitation the three-term fit carries. Measured on the SO-101, inheriting `k_t`
    /// from a catalogue put the fitted damping wrong by exactly the `k_t` error — 10% in, 10% out, no
    /// averaging-down. Fitting `k_t` removes the inherited constant from the chain entirely.
    #[test]
    fn identify_actuator_with_gain_recovers_the_torque_constant_too() {
        let (robot, inertia) = geared_arm(0.0119, 0.64);
        let g = Vector3::new(0.0, 0.0, -9.81);
        let true_kt = 1.574; // the STS3215 figure, treated here as ground truth to be recovered
        let mut samples = Vec::new();
        for k in 0..500 {
            let t = k as f64 * 0.002;
            let q = vec![0.4 * (3.0 * t).sin(), -0.3 * (5.0 * t).cos()];
            let qd = vec![1.2 * (3.0 * t).cos() + 0.5 * (11.0 * t).cos(), 1.5 * (5.0 * t).sin()];
            let qdd = vec![-3.6 * (3.0 * t).sin() - 5.5 * (11.0 * t).sin(), 7.5 * (5.0 * t).cos()];
            // The arm needs this torque; a real current sensor would read it divided by k_t.
            let tau = crate::inverse_dynamics(&robot, &inertia, &q, &qd, &qdd, g);
            let current: Vec<f64> = tau.iter().map(|x| x / true_kt).collect();
            samples.push(CurrentSample { q, qd, qdd, current });
        }

        let fits = identify_actuator_with_gain(&robot, &inertia, &samples, g);
        assert_eq!(fits.len(), 2);
        for (i, f) in fits.iter().enumerate() {
            let (ta, tb, tf) = (0.0119 + 0.003 * i as f64, 0.64 - 0.1 * i as f64, 0.08 + 0.02 * i as f64);
            assert!(f.physical, "joint {i} must be physical: {f:?}");
            assert!(f.conditioning > 1e-6, "joint {i} four-column conditioning {}", f.conditioning);
            assert!((f.torque_constant - true_kt).abs() < 1e-7, "joint {i} k_t: {} vs {true_kt}", f.torque_constant);
            assert!((f.armature - ta).abs() < 1e-8, "joint {i} armature: {} vs {ta}", f.armature);
            assert!((f.damping - tb).abs() < 1e-8, "joint {i} damping: {} vs {tb}", f.damping);
            assert!((f.friction - tf).abs() < 1e-8, "joint {i} friction: {} vs {tf}", f.friction);
            assert!(f.residual < 1e-8, "joint {i} residual {}", f.residual);
        }
    }

    /// **A wrong `k_t` no longer poisons the other three, which is the whole point.**
    ///
    /// The three-term fit takes torque, so a `k_t` error scales its input and lands 1:1 in the parameters. The
    /// four-term fit takes current, so `k_t` is fitted and there is no inherited constant to be wrong. This
    /// runs both against the same motion and asserts the difference.
    #[test]
    fn fitting_the_gain_removes_the_torque_scale_bias() {
        let (robot, inertia) = geared_arm(0.0119, 0.64);
        let g = Vector3::new(0.0, 0.0, -9.81);
        let true_kt = 1.574;
        let assumed_kt = true_kt * 1.10; // a catalogue figure 10% off

        let mut cur = Vec::new();
        let mut tau_samples = Vec::new();
        for k in 0..500 {
            let t = k as f64 * 0.002;
            let q = vec![0.4 * (3.0 * t).sin(), -0.3 * (5.0 * t).cos()];
            let qd = vec![1.2 * (3.0 * t).cos() + 0.5 * (11.0 * t).cos(), 1.5 * (5.0 * t).sin()];
            let qdd = vec![-3.6 * (3.0 * t).sin() - 5.5 * (11.0 * t).sin(), 7.5 * (5.0 * t).cos()];
            let tau = crate::inverse_dynamics(&robot, &inertia, &q, &qd, &qdd, g);
            let current: Vec<f64> = tau.iter().map(|x| x / true_kt).collect();
            // What a user of the THREE-term fit would supply: current times the wrong constant.
            let tau_wrong: Vec<f64> = current.iter().map(|a| a * assumed_kt).collect();
            cur.push(CurrentSample { q: q.clone(), qd: qd.clone(), qdd: qdd.clone(), current });
            tau_samples.push(IdSample { q, qd, qdd, tau: tau_wrong });
        }

        let biased = identify_actuator(&robot, &inertia, &tau_samples, g);
        let unbiased = identify_actuator_with_gain(&robot, &inertia, &cur, g);
        for i in 0..2 {
            let truth = 0.64 - 0.1 * i as f64;
            let biased_err = (biased[i].damping - truth).abs() / truth;
            let unbiased_err = (unbiased[i].damping - truth).abs() / truth;
            // The biased fit should be off by roughly the k_t error; the unbiased one should be exact.
            assert!(biased_err > 0.02, "joint {i}: the torque-input fit should show the bias, got {biased_err}");
            assert!(unbiased_err < 1e-8, "joint {i}: the current-input fit should be exact, got {unbiased_err}");
            assert!((unbiased[i].torque_constant - true_kt).abs() < 1e-7, "joint {i} k_t recovered");
        }
    }

    /// **A joint decoupled from gravity cannot have its gain identified, and the fit must say so.**
    ///
    /// This test replaces one that could not fail for the reason it claimed. That version used a
    /// constant-velocity trajectory, and an adversarial review found the assertion was satisfied by a
    /// damping-versus-friction collinearity the *three*-term fit already has: on the same motion
    /// [`identify_actuator`] reports `1.1e-16`, and substituting a perfectly independent random current column
    /// still passed. It gave zero coverage of the one condition the four-column API adds.
    ///
    /// The real condition is that `τ_rigid` lie outside `span{q̈, q̇, tanh(q̇/ε)}`, and the **gravity term** is
    /// what supplies that. So the failure case is an axis **parallel to gravity** — a base yaw or wrist roll —
    /// where `τ_rigid = M·q̈` with `M` constant and the current column is exactly proportional to `q̈`.
    #[test]
    fn a_gravity_decoupled_joint_cannot_separate_the_gain() {
        // One revolute joint about z, with gravity also along z: no gravity torque about the axis, ever.
        const VERTICAL: &str = r#"<robot name="yaw">
          <link name="base"/>
          <link name="l1"><inertial><origin xyz="0.3 0 0"/><mass value="1.5"/>
            <inertia ixx="0.02" ixy="0" ixz="0" iyy="0.02" iyz="0" izz="0.02"/></inertial></link>
          <link name="tool"/>
          <joint name="j1" type="revolute"><parent link="base"/><child link="l1"/><origin xyz="0 0 0"/>
            <axis xyz="0 0 1"/><limit lower="-3" upper="3" effort="8" velocity="4"/></joint>
          <joint name="jt" type="fixed"><parent link="l1"/><child link="tool"/><origin xyz="0.4 0 0"/></joint>
        </robot>"#;
        let (mut robot, inertia) = crate::from_urdf_full(VERTICAL, "base", "tool").unwrap();
        for j in robot.joints.iter_mut() {
            *j = j.clone().with_armature(0.0119).with_damping(0.64).with_friction(0.08);
        }
        let g = Vector3::new(0.0, 0.0, -9.81); // parallel to the joint axis
        let true_kt = 1.574;

        // The SAME rich two-frequency excitation the oracle uses, so richness cannot be the explanation.
        let mut cur = Vec::new();
        let mut tau_samples = Vec::new();
        for k in 0..500 {
            let t = k as f64 * 0.002;
            let q = vec![0.4 * (3.0 * t).sin()];
            let qd = vec![1.2 * (3.0 * t).cos() + 0.5 * (11.0 * t).cos()];
            let qdd = vec![-3.6 * (3.0 * t).sin() - 5.5 * (11.0 * t).sin()];
            let tau = crate::inverse_dynamics(&robot, &inertia, &q, &qd, &qdd, g);
            let current: Vec<f64> = tau.iter().map(|x| x / true_kt).collect();
            cur.push(CurrentSample { q: q.clone(), qd: qd.clone(), qdd: qdd.clone(), current });
            tau_samples.push(IdSample { q, qd, qdd, tau });
        }

        let gain_fit = identify_actuator_with_gain(&robot, &inertia, &cur, g);
        assert!(
            gain_fit[0].conditioning < 1e-9,
            "a gravity-decoupled joint must report the gain as unidentifiable, got {}",
            gain_fit[0].conditioning
        );
        assert!(gain_fit[0].torque_constant.is_nan(), "and must refuse rather than return a number");

        // THE CONTROL, and the reason this test is not the vacuous one it replaced: on the very same motion the
        // THREE-term fit is healthy and exact. So the degeneracy is specific to the gain column.
        let three = identify_actuator(&robot, &inertia, &tau_samples, g);
        assert!(three[0].conditioning > 1e-2, "the 3-term fit should be fine here, got {}", three[0].conditioning);
        assert!((three[0].armature - 0.0119).abs() < 1e-9, "armature: {}", three[0].armature);
        assert!((three[0].damping - 0.64).abs() < 1e-9, "damping: {}", three[0].damping);
        assert!((three[0].friction - 0.08).abs() < 1e-9, "friction: {}", three[0].friction);
    }

    /// **Data that determines nothing must not report perfect conditioning.**
    ///
    /// An adversarial review found the worst-possible inputs reading as the best-possible: static poses
    /// (`q̇ = q̈ = 0`) returned `conditioning: 1.000000` with all four parameters `NaN` — a *better* number than
    /// the well-excited oracle earns — because a zero-norm column was being substituted with a unit basis
    /// vector. A stuck-at-zero current sensor reported `9.9e-5`, above the `> 1e-6` bar these tests use to mean
    /// identifiable. Both now report zero.
    #[test]
    fn degenerate_input_reports_zero_conditioning_not_one() {
        let (robot, inertia) = geared_arm(0.0119, 0.64);
        let g = Vector3::new(0.0, 0.0, -9.81);

        let statics: Vec<CurrentSample> = (0..50)
            .map(|k| CurrentSample {
                q: vec![0.01 * k as f64, -0.01 * k as f64],
                qd: vec![0.0, 0.0],
                qdd: vec![0.0, 0.0],
                current: vec![0.3, 0.2],
            })
            .collect();
        for f in identify_actuator_with_gain(&robot, &inertia, &statics, g) {
            assert_eq!(f.conditioning, 0.0, "static poses determine nothing: {f:?}");
            assert!(f.torque_constant.is_nan());
        }

        // A current sensor stuck at zero: k_t has no leverage at all.
        let stuck: Vec<CurrentSample> = (0..200)
            .map(|k| {
                let t = k as f64 * 0.002;
                CurrentSample {
                    q: vec![0.4 * (3.0 * t).sin(), -0.3 * (5.0 * t).cos()],
                    qd: vec![1.2 * (3.0 * t).cos(), 1.5 * (5.0 * t).sin()],
                    qdd: vec![-3.6 * (3.0 * t).sin(), 7.5 * (5.0 * t).cos()],
                    current: vec![0.0, 0.0],
                }
            })
            .collect();
        for f in identify_actuator_with_gain(&robot, &inertia, &stuck, g) {
            assert_eq!(f.conditioning, 0.0, "a dead current column determines nothing: {f:?}");
        }

        // An empty slice, and a non-finite input: neither may produce a finite-looking conditioning.
        for f in identify_actuator_with_gain(&robot, &inertia, &[], g) {
            assert_eq!(f.conditioning, 0.0, "no samples determine nothing");
        }
        // A non-finite input on ONE joint. The first version of this asserted every joint came back zero, which
        // was wrong: poisoning joint 0's current leaves joint 1 with a perfectly ordinary column, and it
        // reported 0.0149. The library was right and the test was wrong. What is worth asserting is the
        // containment — the joints decouple, so poison in one must not reach the other.
        let mut poisoned = stuck.clone();
        poisoned[0].current = vec![f64::INFINITY, 1.0];
        let fits = identify_actuator_with_gain(&robot, &inertia, &poisoned, g);
        assert_eq!(fits[0].conditioning, 0.0, "the poisoned joint must refuse: {:?}", fits[0]);
        assert!(fits[0].torque_constant.is_nan(), "and must not return a number");
        for f in &fits {
            // The real invariant: never an infinite conditioning, on any joint, whatever the input. On a scale
            // where larger means more identifiable, `inf` would clear every threshold a caller writes.
            assert!(f.conditioning.is_finite(), "conditioning must never be infinite: {f:?}");
        }
    }

    /// **The diagnostic must reproduce, automatically, the two confoundings a reviewer derived by hand.**
    ///
    /// Both came out of an adversarial review of this module, and both were invisible to `conditioning` because
    /// it is a scalar. Case one: a constant-velocity motion, where I asserted the *gain* was unidentifiable and
    /// was wrong — the real confounding is damping against friction, which the three-term fit has too. Case
    /// two: a gravity-parallel axis, where the gain genuinely is inseparable, from the armature specifically.
    ///
    /// If this test ever stops distinguishing those two, the diagnostic has lost the only property that makes
    /// it worth more than the scalar it supplements.
    #[test]
    fn confounding_names_the_pair_a_scalar_cannot() {
        let g = Vector3::new(0.0, 0.0, -9.81);
        let kt = 1.574;

        // CASE 1 — constant velocity. q̈ is identically zero and q̇ is constant, so −q̇ and −tanh(q̇/ε) are
        // parallel. The armature column is dead. The reviewer's hand computation: [0, 0, +0.707, −0.707].
        let (arm, inertia) = geared_arm(0.0119, 0.64);
        let flat: Vec<PlannedMotion> = (0..200)
            .map(|k| PlannedMotion {
                q: vec![k as f64 * 0.002, k as f64 * 0.0014],
                qd: vec![1.0, 0.7],
                qdd: vec![0.0, 0.0],
            })
            .collect();
        let c = confounding(&arm, &inertia, &flat, g, kt);
        // The armature column is identically zero, so THAT is the unresolvable axis and the report says so
        // directly rather than through a near-zero eigenvalue.
        assert_eq!(c[0].conditioning, 0.0, "a dead column is not a small eigenvalue, it is no leverage at all");
        assert!(
            c[0].direction[1].abs() > 0.99,
            "the dead direction must be the armature axis, got {:?}",
            c[0].direction
        );
        assert!(
            c[0].direction[0].abs() < 1e-9,
            "and it must carry NO weight on k_t — the conclusion I drew from the scalar was that k_t was the \
             problem, and it was not: {:?}",
            c[0].direction
        );

        // CASE 2 — an axis parallel to gravity. τ_rigid = M·q̈ with M constant, so the current column is
        // proportional to the q̈ column and k_t is inseparable from the ARMATURE specifically.
        const VERTICAL: &str = r#"<robot name="yaw">
          <link name="base"/>
          <link name="l1"><inertial><origin xyz="0.3 0 0"/><mass value="1.5"/>
            <inertia ixx="0.02" ixy="0" ixz="0" iyy="0.02" iyz="0" izz="0.02"/></inertial></link>
          <link name="tool"/>
          <joint name="j1" type="revolute"><parent link="base"/><child link="l1"/><origin xyz="0 0 0"/>
            <axis xyz="0 0 1"/><limit lower="-3" upper="3" effort="8" velocity="4"/></joint>
          <joint name="jt" type="fixed"><parent link="l1"/><child link="tool"/><origin xyz="0.4 0 0"/></joint>
        </robot>"#;
        let (mut yaw, yaw_inertia) = crate::from_urdf_full(VERTICAL, "base", "tool").unwrap();
        for j in yaw.joints.iter_mut() {
            *j = j.clone().with_armature(0.0119).with_damping(0.64).with_friction(0.08);
        }
        let rich: Vec<PlannedMotion> = (0..500)
            .map(|k| {
                let t = k as f64 * 0.002;
                PlannedMotion {
                    q: vec![0.4 * (3.0 * t).sin()],
                    qd: vec![1.2 * (3.0 * t).cos() + 0.5 * (11.0 * t).cos()],
                    qdd: vec![-3.6 * (3.0 * t).sin() - 5.5 * (11.0 * t).sin()],
                }
            })
            .collect();
        let v = confounding(&yaw, &yaw_inertia, &rich, g, kt);
        assert!(v[0].conditioning < 1e-6, "a gravity-parallel axis is degenerate, got {}", v[0].conditioning);
        let pair = v[0].worst_pair();
        assert!(
            (pair == ("k_t", "armature")) || (pair == ("armature", "k_t")),
            "the confounded pair must be k_t against the armature, got {pair:?} from {:?}",
            v[0].direction
        );

        // THE CONTROL that makes the two cases distinguishable rather than both just "degenerate": the SAME
        // rich excitation on a gravity-LOADED arm is well conditioned, because G(q) supplies the independence.
        let loaded: Vec<PlannedMotion> = (0..500)
            .map(|k| {
                let t = k as f64 * 0.002;
                PlannedMotion {
                    q: vec![0.4 * (3.0 * t).sin(), -0.3 * (5.0 * t).cos()],
                    qd: vec![1.2 * (3.0 * t).cos() + 0.5 * (11.0 * t).cos(), 1.5 * (5.0 * t).sin()],
                    qdd: vec![-3.6 * (3.0 * t).sin() - 5.5 * (11.0 * t).sin(), 7.5 * (5.0 * t).cos()],
                }
            })
            .collect();
        for r in confounding(&arm, &inertia, &loaded, g, kt) {
            assert!(
                r.conditioning > 1e-3,
                "joint {} on a gravity-loaded arm should screen as identifiable, got {}",
                r.joint,
                r.conditioning
            );
        }
    }

    /// The screening must agree with what the fit actually does, or it is worse than nothing — a caller would
    /// approve a trajectory the fit then refuses.
    #[test]
    fn the_screening_agrees_with_the_fit_it_screens_for() {
        let g = Vector3::new(0.0, 0.0, -9.81);
        let kt = 1.574;
        let (arm, inertia) = geared_arm(0.0119, 0.64);
        let rich: Vec<(PlannedMotion, CurrentSample)> = (0..400)
            .map(|k| {
                let t = k as f64 * 0.002;
                let q = vec![0.4 * (3.0 * t).sin(), -0.3 * (5.0 * t).cos()];
                let qd = vec![1.2 * (3.0 * t).cos() + 0.5 * (11.0 * t).cos(), 1.5 * (5.0 * t).sin()];
                let qdd = vec![-3.6 * (3.0 * t).sin() - 5.5 * (11.0 * t).sin(), 7.5 * (5.0 * t).cos()];
                let tau = crate::inverse_dynamics(&arm, &inertia, &q, &qd, &qdd, g);
                let current: Vec<f64> = tau.iter().map(|x| x / kt).collect();
                (
                    PlannedMotion { q: q.clone(), qd: qd.clone(), qdd: qdd.clone() },
                    CurrentSample { q, qd, qdd, current },
                )
            })
            .collect();
        let plan: Vec<PlannedMotion> = rich.iter().map(|(p, _)| p.clone()).collect();
        let samples: Vec<CurrentSample> = rich.iter().map(|(_, c)| c.clone()).collect();

        let screened = confounding(&arm, &inertia, &plan, g, kt);
        let fitted = identify_actuator_with_gain(&arm, &inertia, &samples, g);
        for i in 0..2 {
            // The screening predicts the current from the model; the fit sees it measured. On noise-free data
            // built from that same model the two must land on the same conditioning.
            let rel = (screened[i].conditioning - fitted[i].conditioning).abs()
                / fitted[i].conditioning.max(1e-30);
            assert!(
                rel < 1e-6,
                "joint {i}: screening {} vs fit {} — a screen that disagrees with its own fit is worse than none",
                screened[i].conditioning,
                fitted[i].conditioning
            );
        }
    }
}
