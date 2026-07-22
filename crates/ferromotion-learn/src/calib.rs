//! **Gradient-based real-to-sim calibration** — fit a robot's inertial parameters and joint
//! friction to recorded trajectories, by exact gradients through the dynamics.
//!
//! This is the single most validated use of differentiable dynamics in 2026 practice: given
//! samples `(q, q̇, q̈, τ)` from hardware (or a higher-fidelity simulator), find the model that
//! predicts the measured torques — the inverse-dynamics regression formulation of system
//! identification, extended beyond the classical linear-regressor form ([`ferromotion_core`'s
//! `sysid`]) to jointly fit **nonlinear friction** and arbitrary parameter subsets.
//!
//! Machinery: [`ferromotion_core::gendyn`]'s generic-scalar RNEA instantiated at [`Dual`] — each
//! fitted parameter is seeded through the *model* (chain rule through the log-reparameterization
//! included), so every gradient is exact to machine precision. Positivity (masses, inertia
//! diagonals, friction coefficients) holds **by construction**: those parameters are optimized in
//! log space. Coulomb friction is smoothed (`tanh(q̇/w)`) so the model stays differentiable at rest.
//!
//! Honesty about identifiability: for a fixed-base chain only the *base parameters* are
//! recoverable — the fit recovers the dynamics **predictively**, not a unique φ (same caveat as
//! classical linear ID). The report therefore carries a per-parameter Gauss–Newton sensitivity:
//! parameters with near-zero sensitivity are unidentifiable *from this data* and their fitted
//! values should not be read as physical truth. Pure Rust, wasm-clean.

use crate::dual::Dual;
use ferromotion_core::gendyn::{GenModel, Real};
use ferromotion_core::{LinkInertia, Robot};

/// Smoothed joint friction: `τ_f = coulomb·tanh(q̇/W) + viscous·q̇`.
#[derive(Clone, Copy, Debug)]
pub struct JointFriction {
    pub coulomb: f64,
    pub viscous: f64,
}

/// Smoothing width of the Coulomb `tanh` (rad/s) — small enough to look like `sign` at working
/// speeds, large enough to keep gradients finite through zero crossings.
pub const COULOMB_SMOOTH: f64 = 0.05;

/// One recorded sample: joint state, acceleration, and measured torques.
#[derive(Clone, Debug)]
pub struct CalibSample {
    pub q: Vec<f64>,
    pub qd: Vec<f64>,
    pub qdd: Vec<f64>,
    pub tau: Vec<f64>,
}

/// Which parameter groups to fit (everything else stays at its initial value).
#[derive(Clone, Copy, Debug)]
pub struct CalibSpec {
    pub fit_mass: bool,
    pub fit_com: bool,
    pub fit_inertia_diag: bool,
    pub fit_friction: bool,
    pub iters: usize,
    pub lr: f64,
}

impl Default for CalibSpec {
    fn default() -> Self {
        Self { fit_mass: true, fit_com: true, fit_inertia_diag: true, fit_friction: true, iters: 300, lr: 0.03 }
    }
}

/// The calibration result: fitted model, fit quality, and the identifiability signal.
pub struct CalibReport {
    pub inertia: Vec<LinkInertia>,
    pub friction: Vec<JointFriction>,
    /// RMS torque error of the INITIAL model over the samples.
    pub rms_before: f64,
    /// RMS torque error of the fitted model.
    pub rms_after: f64,
    /// Per fitted parameter `(name, Gauss–Newton sensitivity)` — `sqrt(mean (∂τ̂/∂θ)²)`. Near-zero
    /// sensitivity means the data does not constrain that parameter (unidentifiable here).
    pub sensitivity: Vec<(String, f64)>,
}

// One fitted parameter's location in the model.
#[derive(Clone, Copy)]
enum Param {
    LogMass(usize),
    Com(usize, usize),
    LogInertiaDiag(usize, usize),
    LogCoulomb(usize),
    LogViscous(usize),
}

fn param_name(p: Param) -> String {
    match p {
        Param::LogMass(l) => format!("link{l}.mass"),
        Param::Com(l, k) => format!("link{l}.com[{k}]"),
        Param::LogInertiaDiag(l, k) => format!("link{l}.I[{k}{k}]"),
        Param::LogCoulomb(j) => format!("joint{j}.coulomb"),
        Param::LogViscous(j) => format!("joint{j}.viscous"),
    }
}

/// Predicted torques of the generic model + friction at one sample.
fn tau_pred<T: Real>(model: &GenModel<T>, friction: &[(T, T)], q: &[T], qd: &[T], qdd: &[T]) -> Vec<T> {
    let w_inv = T::from_f64(1.0 / COULOMB_SMOOTH);
    let mut tau = model.rnea(q, qd, qdd);
    for (j, t) in tau.iter_mut().enumerate() {
        let (c, v) = friction[j];
        *t = *t + c * (qd[j] * w_inv).tanh() + v * qd[j];
    }
    tau
}

// Decode θ into a model over T, seeding the dual derivative on parameter `seed` with the chain
// rule of the log-reparameterization (d exp(θ)/dθ = exp(θ)).
#[allow(clippy::too_many_arguments)]
fn decode<T: Real>(
    robot: &Robot,
    base_inertia: &[LinkInertia],
    base_friction: &[JointFriction],
    params: &[Param],
    theta: &[f64],
    gravity: [f64; 3],
    seed: Option<usize>,
    mk: impl Fn(f64, f64) -> T, // (value, dvalue/dθ_seed or 0) -> T
) -> (GenModel<T>, Vec<(T, T)>) {
    let mut inertia = base_inertia.to_vec();
    let mut friction = base_friction.to_vec();
    // apply θ values (f64 side)
    for (i, p) in params.iter().enumerate() {
        match *p {
            Param::LogMass(l) => inertia[l].mass = theta[i].exp(),
            Param::Com(l, k) => inertia[l].com[k] = theta[i],
            Param::LogInertiaDiag(l, k) => inertia[l].inertia[(k, k)] = theta[i].exp(),
            Param::LogCoulomb(j) => friction[j].coulomb = theta[i].exp(),
            Param::LogViscous(j) => friction[j].viscous = theta[i].exp(),
        }
    }
    let mut model = GenModel::<T>::from_robot(robot, &inertia, gravity);
    let mut fr: Vec<(T, T)> = friction.iter().map(|f| (mk(f.coulomb, 0.0), mk(f.viscous, 0.0))).collect();
    // seed the derivative
    if let Some(s) = seed {
        match params[s] {
            Param::LogMass(l) => model.links[l].mass = mk(inertia[l].mass, inertia[l].mass),
            Param::Com(l, k) => model.links[l].com.0[k] = mk(inertia[l].com[k], 1.0),
            Param::LogInertiaDiag(l, k) => {
                model.links[l].inertia.0[k][k] = mk(inertia[l].inertia[(k, k)], inertia[l].inertia[(k, k)])
            }
            Param::LogCoulomb(j) => fr[j].0 = mk(friction[j].coulomb, friction[j].coulomb),
            Param::LogViscous(j) => fr[j].1 = mk(friction[j].viscous, friction[j].viscous),
        }
    }
    (model, fr)
}

/// Fit the selected parameter groups to the samples by Adam over exact dual-number gradients.
pub fn calibrate(
    robot: &Robot,
    init_inertia: &[LinkInertia],
    init_friction: &[JointFriction],
    samples: &[CalibSample],
    gravity: [f64; 3],
    spec: CalibSpec,
) -> CalibReport {
    let n = robot.dof();
    // build the fitted-parameter list + initial θ
    let mut params = Vec::new();
    let mut theta = Vec::new();
    for l in 0..n {
        if spec.fit_mass {
            params.push(Param::LogMass(l));
            theta.push(init_inertia[l].mass.max(1e-6).ln());
        }
        if spec.fit_com {
            for k in 0..3 {
                params.push(Param::Com(l, k));
                theta.push(init_inertia[l].com[k]);
            }
        }
        if spec.fit_inertia_diag {
            for k in 0..3 {
                params.push(Param::LogInertiaDiag(l, k));
                theta.push(init_inertia[l].inertia[(k, k)].max(1e-9).ln());
            }
        }
    }
    for j in 0..n {
        if spec.fit_friction {
            params.push(Param::LogCoulomb(j));
            theta.push(init_friction[j].coulomb.max(1e-6).ln());
            params.push(Param::LogViscous(j));
            theta.push(init_friction[j].viscous.max(1e-6).ln());
        }
    }
    let np = params.len();

    let rms_of = |theta: &[f64]| -> f64 {
        let (model, fr) = decode::<f64>(robot, init_inertia, init_friction, &params, theta, gravity, None, |v, _| v);
        let mut se = 0.0;
        let mut cnt = 0.0;
        for s in samples {
            let pred = tau_pred(&model, &fr, &s.q, &s.qd, &s.qdd);
            for j in 0..n {
                se += (pred[j] - s.tau[j]).powi(2);
                cnt += 1.0;
            }
        }
        (se / cnt).sqrt()
    };
    let rms_before = rms_of(&theta);

    // Adam over exact forward-mode gradients: one dual pass per fitted parameter per iteration.
    let (mut m1, mut m2) = (vec![0.0; np], vec![0.0; np]);
    let (b1, b2, eps) = (0.9, 0.999, 1e-8);
    let mut sens = vec![0.0; np];
    for it in 0..spec.iters {
        let mut grad = vec![0.0; np];
        if it + 1 == spec.iters {
            sens.iter_mut().for_each(|s| *s = 0.0);
        }
        for (p, g_out) in grad.iter_mut().enumerate() {
            let (model, fr) = decode::<Dual>(robot, init_inertia, init_friction, &params, &theta, gravity, Some(p), |v, d| Dual { re: v, eps: d });
            let mut g = 0.0;
            let mut cnt = 0.0;
            for s in samples {
                let qc: Vec<Dual> = s.q.iter().map(|&x| Dual::constant(x)).collect();
                let qdc: Vec<Dual> = s.qd.iter().map(|&x| Dual::constant(x)).collect();
                let qddc: Vec<Dual> = s.qdd.iter().map(|&x| Dual::constant(x)).collect();
                let pred = tau_pred(&model, &fr, &qc, &qdc, &qddc);
                for j in 0..n {
                    g += (pred[j].re - s.tau[j]) * pred[j].eps;
                    if it + 1 == spec.iters {
                        sens[p] += pred[j].eps * pred[j].eps;
                    }
                    cnt += 1.0;
                }
            }
            *g_out = g / cnt;
            if it + 1 == spec.iters {
                sens[p] = (sens[p] / cnt).sqrt();
            }
        }
        let t = (it + 1) as f64;
        for p in 0..np {
            m1[p] = b1 * m1[p] + (1.0 - b1) * grad[p];
            m2[p] = b2 * m2[p] + (1.0 - b2) * grad[p] * grad[p];
            let mh = m1[p] / (1.0 - b1.powf(t));
            let vh = m2[p] / (1.0 - b2.powf(t));
            theta[p] -= spec.lr * mh / (vh.sqrt() + eps);
        }
    }
    let rms_after = rms_of(&theta);

    // decode the final model back to plain structures
    let (model, fr) = decode::<f64>(robot, init_inertia, init_friction, &params, &theta, gravity, None, |v, _| v);
    let inertia: Vec<LinkInertia> = model
        .links
        .iter()
        .map(|l| LinkInertia {
            mass: l.mass,
            com: nalgebra::Vector3::new(l.com.0[0], l.com.0[1], l.com.0[2]),
            inertia: nalgebra::Matrix3::from_fn(|r, c| l.inertia.0[r][c]),
        })
        .collect();
    let friction: Vec<JointFriction> = fr.iter().map(|&(c, v)| JointFriction { coulomb: c, viscous: v }).collect();
    let sensitivity = params.iter().zip(&sens).map(|(p, &s)| (param_name(*p), s)).collect();
    CalibReport { inertia, friction, rms_before, rms_after, sensitivity }
}

/// Deterministic excitation trajectories + torques from a ground-truth model — the synthetic data
/// generator for tests and labs (sums of incommensurate sinusoids per joint; torques from the
/// reference dynamics plus smoothed friction, i.e. exactly the model class being fitted).
pub fn excite(
    robot: &Robot,
    inertia: &[LinkInertia],
    friction: &[JointFriction],
    gravity: [f64; 3],
    n_samples: usize,
    phase_seed: f64,
) -> Vec<CalibSample> {
    let n = robot.dof();
    let g = nalgebra::Vector3::new(gravity[0], gravity[1], gravity[2]);
    (0..n_samples)
        .map(|s| {
            let t = s as f64 * 0.03;
            let mut q = vec![0.0; n];
            let mut qd = vec![0.0; n];
            let mut qdd = vec![0.0; n];
            for j in 0..n {
                let (a1, w1) = (0.6 + 0.1 * j as f64, 1.3 + 0.7 * j as f64);
                let (a2, w2) = (0.25, 2.9 + 0.4 * j as f64);
                let ph = phase_seed + j as f64 * 1.7;
                q[j] = a1 * (w1 * t + ph).sin() + a2 * (w2 * t).cos();
                qd[j] = a1 * w1 * (w1 * t + ph).cos() - a2 * w2 * (w2 * t).sin();
                qdd[j] = -a1 * w1 * w1 * (w1 * t + ph).sin() - a2 * w2 * w2 * (w2 * t).cos();
            }
            let mut tau = ferromotion_core::inverse_dynamics(robot, inertia, &q, &qd, &qdd, g);
            for j in 0..n {
                tau[j] += friction[j].coulomb * (qd[j] / COULOMB_SMOOTH).tanh() + friction[j].viscous * qd[j];
            }
            CalibSample { q, qd, qdd, tau }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferromotion_core::{Iso, Joint};
    use nalgebra::{Matrix3, Translation3, UnitQuaternion, Vector3};

    fn test_robot() -> (Robot, Vec<LinkInertia>, Vec<JointFriction>) {
        let mk = |xyz: [f64; 3], rpy: [f64; 3]| {
            Iso::from_parts(
                Translation3::new(xyz[0], xyz[1], xyz[2]),
                UnitQuaternion::from_euler_angles(rpy[0], rpy[1], rpy[2]),
            )
        };
        let joints = vec![
            Joint::revolute(mk([0.0, 0.0, 0.3], [0.0, 0.0, 0.4]), Vector3::z()),
            Joint::revolute(mk([0.1, 0.0, 0.2], [0.3, 0.0, 0.0]), Vector3::y()),
            Joint::revolute(mk([0.2, 0.0, 0.1], [0.0, 0.0, -0.3]), Vector3::y()),
        ];
        let inertia: Vec<LinkInertia> = (0..3)
            .map(|i| {
                let f = i as f64;
                LinkInertia {
                    mass: 2.0 + 0.5 * f,
                    com: Vector3::new(0.03 * f, -0.02, 0.08 + 0.02 * f),
                    inertia: Matrix3::new(
                        0.03 + 0.01 * f, 0.0, 0.0,
                        0.0, 0.04 + 0.005 * f, 0.0,
                        0.0, 0.0, 0.035,
                    ),
                }
            })
            .collect();
        let friction = vec![
            JointFriction { coulomb: 0.4, viscous: 0.15 },
            JointFriction { coulomb: 0.25, viscous: 0.3 },
            JointFriction { coulomb: 0.5, viscous: 0.1 },
        ];
        (Robot { joints, ee_offset: Iso::identity() }, inertia, friction)
    }

    /// Fit the identifiable subset (masses + friction, everything else at truth) from a badly
    /// perturbed start: parameters must RECOVER near-exactly, and the torque RMS must collapse.
    #[test]
    fn masses_and_friction_recover_from_perturbed_start() {
        let (robot, true_inertia, true_friction) = test_robot();
        let g = [0.0, 0.0, -9.81];
        let samples = excite(&robot, &true_inertia, &true_friction, g, 240, 0.0);

        let mut init_inertia = true_inertia.clone();
        for (i, li) in init_inertia.iter_mut().enumerate() {
            li.mass *= 1.0 + 0.3 * if i % 2 == 0 { 1.0 } else { -1.0 };
        }
        let init_friction: Vec<JointFriction> =
            true_friction.iter().map(|f| JointFriction { coulomb: f.coulomb * 1.6, viscous: f.viscous * 0.5 }).collect();

        let spec = CalibSpec { fit_mass: true, fit_com: false, fit_inertia_diag: false, fit_friction: true, iters: 800, lr: 0.05 };
        let rep = calibrate(&robot, &init_inertia, &init_friction, &samples, g, spec);
        assert!(rep.rms_after < rep.rms_before / 100.0, "rms must collapse: {} → {}", rep.rms_before, rep.rms_after);
        for (l, (got, want)) in rep.inertia.iter().zip(&true_inertia).enumerate() {
            let rel = (got.mass - want.mass).abs() / want.mass;
            assert!(rel < 0.02, "link {l} mass: {} vs {} (rel {rel})", got.mass, want.mass);
        }
        for (j, (got, want)) in rep.friction.iter().zip(&true_friction).enumerate() {
            assert!((got.coulomb - want.coulomb).abs() / want.coulomb < 0.05, "joint {j} coulomb");
            assert!((got.viscous - want.viscous).abs() / want.viscous < 0.05, "joint {j} viscous");
        }
        eprintln!("identifiable-subset fit: rms {:.4} → {:.6}", rep.rms_before, rep.rms_after);
    }

    /// Full fit (mass + COM + inertia diagonal + friction) from a perturbed start: the fitted model
    /// must PREDICT — held-out excitation torques near-perfectly — even where individual parameters
    /// are only identifiable in combination (the honest sysid criterion).
    #[test]
    fn full_fit_predicts_held_out_trajectories() {
        let (robot, true_inertia, true_friction) = test_robot();
        let g = [0.0, 0.0, -9.81];
        let train = excite(&robot, &true_inertia, &true_friction, g, 150, 0.0);
        let heldout = excite(&robot, &true_inertia, &true_friction, g, 80, 2.1);

        let mut init_inertia = true_inertia.clone();
        for (i, li) in init_inertia.iter_mut().enumerate() {
            let s = if i % 2 == 0 { 1.0 } else { -1.0 };
            li.mass *= 1.0 + 0.25 * s;
            li.com *= 1.0 - 0.2 * s;
            li.inertia *= 1.0 + 0.3 * s;
        }
        let init_friction: Vec<JointFriction> =
            true_friction.iter().map(|f| JointFriction { coulomb: f.coulomb * 0.6, viscous: f.viscous * 1.5 }).collect();

        let spec = CalibSpec { iters: 500, lr: 0.04, ..Default::default() };
        let rep = calibrate(&robot, &init_inertia, &init_friction, &train, g, spec);

        // held-out predictive check with the fitted model
        let rms_heldout = {
            let mut se = 0.0;
            let mut cnt = 0.0;
            let model = GenModel::<f64>::from_robot(&robot, &rep.inertia, g);
            let fr: Vec<(f64, f64)> = rep.friction.iter().map(|f| (f.coulomb, f.viscous)).collect();
            for s in &heldout {
                let pred = tau_pred(&model, &fr, &s.q, &s.qd, &s.qdd);
                for j in 0..robot.dof() {
                    se += (pred[j] - s.tau[j]).powi(2);
                    cnt += 1.0;
                }
            }
            (se / cnt).sqrt()
        };
        assert!(rep.rms_after < rep.rms_before / 50.0, "train rms: {} → {}", rep.rms_before, rep.rms_after);
        assert!(rms_heldout < rep.rms_before / 50.0, "held-out rms {rms_heldout} vs initial {}", rep.rms_before);
        // the identifiability signal exists and is finite for every fitted parameter
        assert_eq!(rep.sensitivity.len(), 3 * 7 + 3 * 2);
        assert!(rep.sensitivity.iter().all(|(_, s)| s.is_finite()));
        eprintln!("full fit: train rms {:.4} → {:.6}; held-out {:.6}", rep.rms_before, rep.rms_after, rms_heldout);
    }
}
