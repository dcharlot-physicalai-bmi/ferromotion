//! ferromotion-control — control schemes for physical AI, in Rust.
//!
//! The first batch of the control corpus. The model-based controllers build directly on
//! `ferromotion-core`'s dynamics (mass matrix, RNEA bias, Jacobians), so they compose with the rest of
//! the toolkit and stay WASM-clean. See `CONTROL.md` for the full roadmap of methods being ported.

use nalgebra::{DVector, Vector3};
use ferromotion_core::{gravity_vector, inverse_dynamics, mass_matrix, LinkInertia, Robot};

mod pf;
mod mhe;
pub use pf::ParticleFilter;
pub use mhe::{batch_estimate, LinearModel};
mod batch;
pub use batch::{batch_rollout, Cartpole};
#[cfg(feature = "gpu")]
pub mod gpu;
mod alip;
mod abfilter;
mod actuator;
mod backlash;
mod battery;
mod fatigue;
mod mckibben;
mod piezo;
mod sma;
mod rotordynamics;
mod foc;
mod friction;
mod motor_thermal;
mod admittance;
mod algames;
mod assignment;
mod reluqp;
mod trajectory_bundles;
mod boxddp;
mod c3;
mod ccm;
mod cbf;
mod cem;
mod cdpr;
mod centroidal_mpc;
mod covariance_steering;
mod coverage;
mod complementary_filter;
mod cpg;
mod ctr;
mod cw;
mod dcm;
mod dial_mpc;
mod diff_drive;
mod diff_mpc;
mod diff_qp;
mod dircon;
mod dmp;
mod dpilqr;
mod es_mpc;
mod fddp;
mod estimation;
mod frenet;
mod geometric_fabric;
mod geometric_se3;
mod hinf;
mod icem;
mod ilqr;
mod imu_preint;
mod inekf;
mod lqr;
mod marine;
mod msckf;
mod momentum_observer;
mod miqp_footstep;
mod madgwick;
mod mpc;
mod mppi;
mod muscle;
mod orca;
mod osc;
mod path_tracking;
mod placo;
mod proxddp;
mod qp_certificate;
mod qp;
mod quadrotor;
mod rts;
mod sliding_mode;
mod scvx;
mod slip;
mod srbd_mpc;
mod swarm;
mod tinympc;
mod tube_mpc;
mod topp;
mod visual_servo;
mod sensorimotor;
mod wahba;
mod wbc;
mod zmp;
mod smoothing_tube;
mod zonotope;
pub use alip::Alip;
pub use abfilter::{AlphaBeta, AlphaBetaGamma};
pub use actuator::{DcMotor, Delay, SeaJoint};
pub use battery::Battery;
pub use fatigue::{
    damage, damage_from_history, equivalent_amplitude, rainflow, repetitions_to_failure, reversals,
    total_variation, Cycle, MeanCorrection, SnCurve,
};
pub use mckibben::{Mckibben, MAGIC_ANGLE};
pub use piezo::Piezo;
pub use sma::{Direction, Sma, SmaState};
pub use rotordynamics::{gyroscopic_moment, Rotor, WhirlResponse};
pub use foc::{
    clarke, electrical_angle, inverse_clarke, inverse_park, modulation_index, park, spwm_duties,
    spwm_voltage_limit, svpwm_duties, svpwm_voltage_limit, PiCurrent, Pmsm,
};
pub use backlash::{Backlash, Contact};
pub use friction::{LuGre, Stribeck};
pub use motor_thermal::{MotorThermal, ALPHA_COPPER};
pub use admittance::{Admittance, HybridForcePosition};
pub use algames::{AlGames, AlGamesResult, Player};
pub use assignment::{hungarian, Assignment};
pub use reluqp::ReluQp;
pub use trajectory_bundles::TrajectoryBundle;
pub use boxddp::{BoxDdpProblem, BoxDdpReport};
pub use c3::{Lcs, C3};
pub use ctr::{PusherSlider, SmoothedContact};
pub use cbf::{CbfConstraint, CbfFilter, FilterMode, FilterOutcome};
pub use ccm::Ccm;
pub use cdpr::{Cdpr, TensionResult};
pub use cem::{Cem, CemStep};
pub use centroidal_mpc::CentroidalMpc;
pub use complementary_filter::ComplementaryFilter;
pub use covariance_steering::{CovarianceSteering, SteeringPolicy};
pub use coverage::LloydCoverage;
pub use cpg::{CpgNetwork, HopfOscillator};
pub use dcm::{dcm, dcm_control, lipm_omega, plan_dcm, DcmPlan, DcmStep};
pub use cw::Cw;
pub use dial_mpc::DialMpc;
pub use diff_drive::{polar_control, DiffDrive, PolarGains, Unicycle};
pub use diff_mpc::DiffMpc;
pub use diff_qp::{diff_eq_qp, solve_eq_qp, QpGrads, QpSolution};
pub use dircon::{CollocationResult, DirectCollocation};
pub use dmp::Dmp;
pub use dpilqr::{DpilqrResult, PiAgent, PotentialGame};
pub use es_mpc::{log_so3, EsAttitude};
pub use fddp::{FddpProblem, FddpReport};
pub use estimation::{numerical_jacobian, Ekf, KalmanFilter, Ukf};
pub use frenet::{FrenetPath, FrenetPlanner, Quartic, Quintic};
pub use geometric_fabric::GeometricFabric;
pub use geometric_se3::{GeometricSe3, QuadFullState, Reference as Se3Reference};
pub use hinf::Hinf;
pub use icem::{colored_noise, Icem};
pub use ilqr::{solve_ilqr, IlqrProblem, IlqrResult};
pub use imu_preint::{exp_so3, right_jacobian, ImuPreintegrator};
pub use inekf::{riekf_a_matrix, standard_ekf_f, InEkf, Matrix9, Se23, Vector9};
pub use lqr::{dlqr, Lqr};
pub use marine::{los_heading, MarineCraft};
pub use momentum_observer::MomentumObserver;
pub use msckf::{CamPose, FeatureTrack, Msckf};
pub use miqp_footstep::{ConvexRegion, FootstepPlan, FootstepPlanner};
pub use mpc::LinearMpc;
pub use madgwick::Madgwick;
pub use mppi::Mppi;
pub use muscle::HillMuscle;
pub use orca::{orca_line, orca_velocity, Agent as OrcaAgent, Line as OrcaLine};
pub use osc::OperationalSpace;
pub use path_tracking::{pure_pursuit, stanley, Bicycle, CarState, Path as TrackPath};
pub use placo::{PlacoResult, PlacoSolver, PlacoTask};
pub use proxddp::{ConstrainedDdpResult, ConstrainedLqr};
pub use qp_certificate::{certify_qp_loop, common_lyapunov, worst_case_growth, ControlQp, QpCertificate, Verdict};
pub use quadrotor::{flat_to_state, min_snap, FlatState, MinSnap, QuadState};
pub use sliding_mode::{sat, SlidingMode};
pub use slip::{Phase, Slip, SlipState};
pub use scvx::{ScvxOpts, ScvxProblem, ScvxReport};
pub use srbd_mpc::SrbdMpc;
pub use swarm::{consensus_step, formation_step, Graph};
pub use tinympc::TinyMpc;
pub use topp::{topp, ToppPath, ToppResult};
pub use tube_mpc::TubeMpc;
pub use visual_servo::{ibvs_twist, interaction_matrix, Camera};
pub use sensorimotor::{perceive, servo_twist, EyeInHand, FreeEye, Perception};
pub use wahba::{davenport_q_method, triad};
pub use wbc::{CartesianTask, WholeBody};
pub use zmp::{capture_point, CartState, PreviewState, ZmpPreview};
pub use smoothing_tube::{
    certify, escaping_sample, nominal_activity, propagate_tube, reaches_goal, GapBound, GapEvidence, HalfSpace, TubeReport, TubeStep,
    TubeVerdict, UndecidedReason,
};
pub use zonotope::{reach_linear, Zonotope};

/// Classic PID over an n-dimensional error signal (anti-windup-free; the workhorse baseline).
#[derive(Clone, Debug)]
pub struct Pid {
    pub kp: f64,
    pub ki: f64,
    pub kd: f64,
    integral: Vec<f64>,
    prev_err: Option<Vec<f64>>,
}

impl Pid {
    pub fn new(kp: f64, ki: f64, kd: f64) -> Self {
        Self { kp, ki, kd, integral: Vec::new(), prev_err: None }
    }

    /// One control step from the current error; returns the command. `dt` in seconds.
    pub fn step(&mut self, dt: f64, err: &[f64]) -> Vec<f64> {
        if self.integral.len() != err.len() {
            self.integral = vec![0.0; err.len()];
        }
        let mut out = vec![0.0; err.len()];
        for i in 0..err.len() {
            self.integral[i] += err[i] * dt;
            let deriv = match &self.prev_err {
                Some(p) if dt > 0.0 => (err[i] - p[i]) / dt,
                _ => 0.0,
            };
            out[i] = self.kp * err[i] + self.ki * self.integral[i] + self.kd * deriv;
        }
        self.prev_err = Some(err.to_vec());
        out
    }

    pub fn reset(&mut self) {
        self.integral.clear();
        self.prev_err = None;
    }
}

/// Computed-torque (inverse-dynamics) control: feedback-linearizes the arm so the closed loop is
/// `ë + Kd·ė + Kp·e = 0`. `τ = M(q)·(q̈_des + Kp·e + Kd·ė) + C(q,q̇)q̇ + G(q)`.
#[derive(Clone, Debug)]
pub struct ComputedTorque {
    pub kp: f64,
    pub kd: f64,
}

impl ComputedTorque {
    /// **Zero-order-hold damping ratio `kd·T` for control period `T`. Keep it below 2.**
    ///
    /// Same criterion as [`CartesianImpedance::damping_zoh_ratio`], but this controller cancels `M`, so
    /// the closed loop is `ë + kd·ė + kp·e = 0` per joint and the damping RATE the tick holds is `kd`
    /// itself — no mass matrix, no posture, no smallest eigenvalue. That is why this needs neither the
    /// robot nor its inertias.
    ///
    /// **Cancelling `M` is what buys the tick budget.** Measured on the two-link fixture from a 1e-6 rad
    /// nudge at rest: this controller holds its posture at a 40 ms tick (`kd·T` = 1.6) and diverges to
    /// non-finite at 60 ms (2.4), where the impedance controller on the same arm already limit-cycles at
    /// 10 ms. An eightfold coarser control rate, because feedback linearisation takes the small
    /// eigenvalue of the mass matrix out of the damping rate. `None` for a non-finite or non-positive
    /// `dt`.
    pub fn damping_zoh_ratio(&self, dt: f64) -> Option<f64> {
        (dt.is_finite() && dt > 0.0 && (self.kd * dt).is_finite()).then_some(self.kd * dt)
    }

    /// **The largest control period at which this controller can hold a posture**, by the same
    /// condition [`crate::Admittance::stability_limit`] uses: `dt²·ω² + 2·dt·γ ≤ 4`.
    ///
    /// Because this controller cancels `M`, the closed loop is `ë + kd·ė + kp·e = 0` per joint and the
    /// rates ARE the gains: `ω² = kp`, `γ = kd`. No posture, no mass matrix, no robot. Measured on the
    /// two-link fixture at `kp = 400, kd = 40` this returns **41.42 ms**, and the plant holds at 40 ms
    /// and diverges to non-finite at 60 ms. The impedance controller on the same arm is limited to
    /// 8.05 ms, so cancelling `M` buys a fivefold coarser control rate.
    /// CROSSED BY: computed_torque_holds_its_posture_at_a_far_coarser_tick_than_impedance
    pub fn max_stable_dt(&self) -> Option<f64> {
        if !self.kp.is_finite() || !self.kd.is_finite() || self.kp < 0.0 || self.kd < 0.0 {
            return None;
        }
        if self.kp <= 0.0 {
            return Some(if self.kd > 0.0 { 2.0 / self.kd } else { f64::INFINITY });
        }
        Some((-self.kd + (self.kd * self.kd + 4.0 * self.kp).sqrt()) / self.kp)
    }

    pub fn new(kp: f64, kd: f64) -> Self {
        Self { kp, kd }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn torque(
        &self,
        robot: &Robot,
        inertia: &[LinkInertia],
        q: &[f64],
        qd: &[f64],
        q_des: &[f64],
        qd_des: &[f64],
        qdd_des: &[f64],
        gravity: nalgebra::Vector3<f64>,
    ) -> Vec<f64> {
        let n = robot.dof();
        // Desired acceleration from PD on the tracking error.
        let a: Vec<f64> =
            (0..n).map(|i| qdd_des[i] + self.kp * (q_des[i] - q[i]) + self.kd * (qd_des[i] - qd[i])).collect();
        // τ = M·a + bias, bias = C·q̇ + G = RNEA(q, q̇, 0, g).
        let m = mass_matrix(robot, inertia, q);
        let bias = inverse_dynamics(robot, inertia, q, qd, &vec![0.0; n], gravity);
        let ma = &m * DVector::from_row_slice(&a);
        (0..n).map(|i| ma[i] + bias[i]).collect()
    }
}

/// Cartesian impedance control: the tool behaves like a spring-damper toward a target position,
/// with gravity compensation and joint-space damping for null-space stability.
/// `τ = Jₚᵀ(Kp(x_des − x) − Kd·ẋ) + G(q) − Dⱼ·q̇`.
#[derive(Clone, Debug)]
pub struct CartesianImpedance {
    pub kp: f64,
    pub kd: f64,
    pub joint_damping: f64,
}

impl CartesianImpedance {
    pub fn new(kp: f64, kd: f64, joint_damping: f64) -> Self {
        Self { kp, kd, joint_damping }
    }

    pub fn torque(
        &self,
        robot: &Robot,
        inertia: &[LinkInertia],
        q: &[f64],
        qd: &[f64],
        x_des: nalgebra::Vector3<f64>,
        gravity: nalgebra::Vector3<f64>,
    ) -> Vec<f64> {
        let n = robot.dof();
        let tip = robot.fk(q).translation.vector;
        let jp = robot.point_jacobian(q, n, &tip); // 3×n
        let xdot = &jp * DVector::from_row_slice(qd); // 3
        let fv = self.kp * (x_des - tip) - self.kd * Vector3::new(xdot[0], xdot[1], xdot[2]);
        let tau_task = jp.transpose() * DVector::from_row_slice(&[fv.x, fv.y, fv.z]); // n
        let g = gravity_vector(robot, inertia, q, gravity);
        (0..n).map(|i| tau_task[i] + g[i] - self.joint_damping * qd[i]).collect()
    }

    /// **Zero-order-hold damping ratio `λ_max(M⁻¹D)·T` at posture `q` for control period `T`. Keep it
    /// below 2, or the controller will not hold a posture it is already in.**
    ///
    /// A sampled controller holds its torque constant across a tick, and the damping it applies is
    /// therefore stale by up to `T`. When the fastest damping time constant is short compared with the
    /// tick, the closed loop stops being dissipative and a fixed point that is stationary on paper
    /// becomes unstable: a nudge grows into a sustained oscillation and stays there forever. Measured
    /// on the two-link fixture in this module's tests, from a 1e-6 rad nudge at rest:
    ///
    /// | `T` | ratio | outcome |
    /// |---|---|---|
    /// | 5 ms | 1.22 | stable, decays to 2e-10 |
    /// | 10 ms | 2.43 | sustained 0.497 rad limit cycle at 35 rad/s |
    /// | 20 ms | 4.87 | 1.0e4 rad/s |
    /// | 50 ms | 12.17 | non-finite |
    ///
    /// **`D` is not `joint_damping`.** The task-space damper acts through the Jacobian too, so the
    /// effective joint-space damping is `D = Jₚᵀ(Kd·I)Jₚ + Dⱼ·I`. On that fixture the parameter alone
    /// gives 100 s⁻¹ while the effective value is 243 s⁻¹, a factor of 2.4 — reading the parameter
    /// instead of the eigenvalue would put the apparent boundary near 1 and make this criterion look
    /// wrong. `None` if `q` is the wrong length, any input is non-finite, or `M` is not invertible.
    pub fn damping_zoh_ratio(&self, robot: &Robot, inertia: &[LinkInertia], q: &[f64], dt: f64) -> Option<f64> {
        Some(self.rates(robot, inertia, q)?.1 * dt).filter(|r| r.is_finite() && dt.is_finite() && dt > 0.0)
    }

    /// **The largest control period at which this controller can hold posture `q`.** Compare it against
    /// your control period, as with [`crate::Admittance::stability_limit`] and the `max_stable_dt` this
    /// crate's motor, friction and rotordynamics models already expose.
    ///
    /// Same condition those use — `dt²·ω² + 2·dt·γ ≤ 4` for a sampled spring-damper — with the
    /// multi-joint rates `ω² = λ_max(M⁻¹K)` and `γ = λ_max(M⁻¹D)`, where `K = Jₚᵀ(Kp·I)Jₚ` and
    /// `D = Jₚᵀ(Kd·I)Jₚ + Dⱼ·I`. Both act through the Jacobian, so neither is the bare gain: on the
    /// two-link fixture `γ` is 243 s⁻¹ where `joint_damping` alone would suggest 100 s⁻¹.
    ///
    /// Measured on that fixture from a 1e-6 rad nudge at rest, this returns **8.05 ms**, and the plant
    /// holds at 5 ms while settling into a 0.497 rad limit cycle at 35 rad/s at 10 ms. Past the limit
    /// it does not error or clamp — it simply stops holding still, and a position-only convergence
    /// check cannot tell that from success. `None` if `q` is the wrong length, any input is
    /// non-finite, or `M` is not invertible.
    /// CROSSED BY: the_arm_holds_its_own_posture_and_a_coarse_tick_proves_the_probe_can_fail
    pub fn max_stable_dt(&self, robot: &Robot, inertia: &[LinkInertia], q: &[f64]) -> Option<f64> {
        let (w2, gamma) = self.rates(robot, inertia, q)?;
        if w2 <= 0.0 {
            return Some(if gamma > 0.0 { 2.0 / gamma } else { f64::INFINITY });
        }
        Some((-gamma + (gamma * gamma + 4.0 * w2).sqrt()) / w2)
    }

    /// `(ω², γ)`: the stiffness and damping rates this posture presents to a sampled controller.
    fn rates(&self, robot: &Robot, inertia: &[LinkInertia], q: &[f64]) -> Option<(f64, f64)> {
        if q.len() != robot.dof() || !q.iter().all(|v| v.is_finite()) {
            return None;
        }
        let n = robot.dof();
        let tip = robot.fk(q).translation.vector;
        let jp = robot.point_jacobian(q, n, &tip);
        let k_eff = jp.transpose() * (self.kp * &jp);
        let d_eff = jp.transpose() * (self.kd * &jp) + nalgebra::DMatrix::identity(n, n) * self.joint_damping;
        let minv = mass_matrix(robot, inertia, q).try_inverse()?;
        let spectral = |m: nalgebra::DMatrix<f64>| m.complex_eigenvalues().iter().map(|z| z.re.abs()).fold(0.0f64, f64::max);
        let (w2, gamma) = (spectral(&minv * &k_eff), spectral(&minv * &d_eff));
        (w2.is_finite() && gamma.is_finite()).then_some((w2, gamma))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Vector3;
    use ferromotion_core::{forward_dynamics, from_urdf_full};

    const ARM2: &str = r#"<robot name="a2">
      <link name="base"/>
      <link name="l1"><inertial><origin xyz="0.3 0 0" rpy="0 0 0"/><mass value="1.5"/>
        <inertia ixx="0.02" ixy="0" ixz="0" iyy="0.02" iyz="0" izz="0.02"/></inertial></link>
      <link name="l2"><inertial><origin xyz="0.25 0 0" rpy="0 0 0"/><mass value="1.0"/>
        <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial></link>
      <link name="tool"/>
      <joint name="j1" type="revolute"><parent link="base"/><child link="l1"/><origin xyz="0 0 0" rpy="0 0 0"/>
        <axis xyz="0 0 1"/><limit lower="-3.14" upper="3.14" effort="10" velocity="3"/></joint>
      <joint name="j2" type="revolute"><parent link="l1"/><child link="l2"/><origin xyz="0.6 0 0" rpy="0 0 0"/>
        <axis xyz="0 0 1"/><limit lower="-3.14" upper="3.14" effort="10" velocity="3"/></joint>
      <joint name="jt" type="fixed"><parent link="l2"/><child link="tool"/><origin xyz="0.5 0 0" rpy="0 0 0"/></joint>
    </robot>"#;

    #[test]
    fn pid_regulates_a_double_integrator() {
        // Unit mass, force control: ẍ = F. PID should drive x → setpoint.
        let mut pid = Pid::new(12.0, 0.0, 7.0);
        let set = 1.0;
        let (mut x, mut v, dt) = (0.0, 0.0, 1e-3);
        for _ in 0..8000 {
            let f = pid.step(dt, &[set - x])[0];
            v += f * dt;
            x += v * dt;
        }
        assert!((x - set).abs() < 1e-2, "x = {x}");
    }

    #[test]
    fn computed_torque_regulates_to_a_setpoint() {
        let (robot, inertia) = from_urdf_full(ARM2, "base", "tool").unwrap();
        let g = Vector3::new(0.0, 0.0, -9.81);
        let ctrl = ComputedTorque::new(100.0, 20.0);
        let q_des = [0.6, -0.8];
        let (mut q, mut qd, dt) = (vec![0.0, 0.0], vec![0.0, 0.0], 1e-3);
        for _ in 0..4000 {
            let tau = ctrl.torque(&robot, &inertia, &q, &qd, &q_des, &[0.0, 0.0], &[0.0, 0.0], g);
            let qdd = forward_dynamics(&robot, &inertia, &q, &qd, &tau, g);
            for i in 0..2 {
                qd[i] += qdd[i] * dt;
                q[i] += qd[i] * dt;
            }
        }
        let err = ((q[0] - q_des[0]).powi(2) + (q[1] - q_des[1]).powi(2)).sqrt();
        assert!(err < 1e-3, "joint error {err}, q = {q:?}");
    }

    #[test]
    fn cartesian_impedance_pulls_tool_to_target() {
        let (robot, inertia) = from_urdf_full(ARM2, "base", "tool").unwrap();
        let g = Vector3::new(0.0, 0.0, -9.81);
        let ctrl = CartesianImpedance::new(300.0, 40.0, 2.0);
        // A reachable in-plane target (arm lies in the z=0 plane; reach ≈ 1.1 m).
        let x_des = Vector3::new(0.7, 0.5, 0.0);
        let (mut q, mut qd, dt) = (vec![0.3, -0.4], vec![0.0, 0.0], 1e-3);
        for _ in 0..6000 {
            let tau = ctrl.torque(&robot, &inertia, &q, &qd, x_des, g);
            let qdd = forward_dynamics(&robot, &inertia, &q, &qd, &tau, g);
            for i in 0..2 {
                qd[i] += qdd[i] * dt;
                q[i] += qd[i] * dt;
            }
        }
        let tip = robot.fk(&q).translation.vector;
        assert!((tip - x_des).norm() < 5e-3, "tool at {tip:?}, target {x_des:?}");
        // Position alone cannot distinguish settled from cycling, so require it to be AT REST too.
        assert!(qd.iter().all(|v| v.abs() < 1e-3), "reached the target but is still moving: qd = {qd:?}");
        let ratio = ctrl.damping_zoh_ratio(&robot, &inertia, &q, dt).expect("ratio");
        assert!(ratio < 2.0, "this test's own tick must satisfy the ZOH criterion, got {ratio:.3}");
    }

    /// **The plant must hold the posture it is already in, from rest — in VELOCITY, not just position.**
    ///
    /// A standing requirement: a steady-state number measured on a body that is actually limit-cycling
    /// is that oscillation's amplitude, and a position-only convergence check cannot tell the two
    /// apart. This test carries its own POSITIVE CONTROL: the same plant at a coarser control tick
    /// must fail, or the probe proves nothing.
    ///
    /// Holding from the EXACT equilibrium is not the test. There the gravity term cancels bit-exactly,
    /// nothing ever moves, and every tick from 1 ms to 50 ms "passes" — measured. The property that
    /// matters is whether the fixed point is STABLE, so the posture is nudged by 1e-6 rad.
    #[test]
    fn the_arm_holds_its_own_posture_and_a_coarse_tick_proves_the_probe_can_fail() {
        let (robot, inertia) = from_urdf_full(ARM2, "base", "tool").unwrap();
        // In-plane gravity, so holding actually requires torque; along the joint axes it is trivial.
        let g = Vector3::new(0.0, -9.81, 0.0);
        let ctrl = CartesianImpedance::new(300.0, 40.0, 2.0);
        let q0 = [0.3, -0.4];
        let x_des = robot.fk(&q0).translation.vector;

        // hold from a 1e-6 rad nudge at rest, physics fixed at h, control held across `t_ctrl`
        let hold = |t_ctrl: f64| -> (f64, f64, bool) {
            let h = 2e-4;
            let (mut q, mut qd) = (vec![q0[0] + 1e-6, q0[1]], vec![0.0, 0.0]);
            let per = (t_ctrl / h).round().max(1.0) as usize;
            let steps = (3.0 / h) as usize;
            let (mut max_dq, mut max_qd, mut blew_up) = (0.0f64, 0.0f64, false);
            let mut tau = vec![0.0; 2];
            for k in 0..steps {
                if k % per == 0 {
                    tau = ctrl.torque(&robot, &inertia, &q, &qd, x_des, g);
                }
                let qdd = forward_dynamics(&robot, &inertia, &q, &qd, &tau, g);
                for i in 0..2 {
                    qd[i] += qdd[i] * h;
                    q[i] += qd[i] * h;
                }
                if !q.iter().chain(qd.iter()).all(|v| v.is_finite()) {
                    blew_up = true;
                    break; // `f64::max` DISCARDS a NaN, so a diverged run would otherwise report 0.000
                }
                if k > steps / 2 {
                    max_dq = max_dq.max((q[0] - q0[0]).abs().max((q[1] - q0[1]).abs()));
                    max_qd = max_qd.max(qd[0].abs().max(qd[1].abs()));
                }
            }
            (max_dq, max_qd, blew_up)
        };

        let fine = 5e-3;
        let coarse = 1e-2;
        let r_fine = ctrl.damping_zoh_ratio(&robot, &inertia, &q0, fine).expect("ratio");
        let r_coarse = ctrl.damping_zoh_ratio(&robot, &inertia, &q0, coarse).expect("ratio");
        assert!(r_fine < 2.0, "5 ms must satisfy the criterion, got {r_fine:.3}");
        assert!(r_coarse > 2.0, "10 ms must violate it, got {r_coarse:.3}");

        let (dq, qd_max, blew) = hold(fine);
        eprintln!("  5 ms  ratio {r_fine:.3}: max|q-q0| {dq:.3e}, max|qd| {qd_max:.3e}");
        assert!(!blew && dq < 1e-6 && qd_max < 1e-6, "a certified tick must HOLD in velocity too: dq {dq:.3e}, qd {qd_max:.3e}");

        let (dq_c, qd_c, blew_c) = hold(coarse);
        eprintln!("  10 ms ratio {r_coarse:.3}: max|q-q0| {dq_c:.3e}, max|qd| {qd_c:.3e}, non-finite {blew_c}");
        assert!(blew_c || qd_c > 1.0, "POSITIVE CONTROL: the coarse tick must visibly fail, else this probe certifies nothing (got qd {qd_c:.3e})");

        // the parameter alone is not the criterion
        let naive = ctrl.joint_damping * coarse / 0.01991;
        assert!(r_coarse > 2.0 * naive, "the effective damping must exceed the parameter's estimate: {r_coarse:.3} vs {naive:.3}");
    }


    /// **The corollary of the standing rule: a fix in one plant is a hypothesis about the others.**
    ///
    /// Same certification as the impedance controller's, on the same arm, with its own positive
    /// control. This controller cancels `M`, so the criterion is `kd·T` with no inertia in it, and the
    /// measured boundary brackets 2 exactly as the other one does — two structurally different
    /// controllers, one criterion.
    #[test]
    fn computed_torque_holds_its_posture_at_a_far_coarser_tick_than_impedance() {
        let (robot, inertia) = from_urdf_full(ARM2, "base", "tool").unwrap();
        let g = Vector3::new(0.0, -9.81, 0.0);
        let ctrl = ComputedTorque::new(400.0, 40.0);
        let q0 = [0.3, -0.4];

        let hold = |t_ctrl: f64| -> (f64, bool) {
            let h = 2e-4;
            let (mut q, mut qd) = (vec![q0[0] + 1e-6, q0[1]], vec![0.0, 0.0]);
            let per = (t_ctrl / h).round().max(1.0) as usize;
            let steps = (3.0 / h) as usize;
            let (mut mqd, mut blew) = (0.0f64, false);
            let mut tau = vec![0.0; 2];
            for k in 0..steps {
                if k % per == 0 {
                    tau = ctrl.torque(&robot, &inertia, &q, &qd, &q0, &[0.0, 0.0], &[0.0, 0.0], g);
                }
                let qdd = forward_dynamics(&robot, &inertia, &q, &qd, &tau, g);
                for i in 0..2 {
                    qd[i] += qdd[i] * h;
                    q[i] += qd[i] * h;
                }
                if !q.iter().chain(qd.iter()).all(|v| v.is_finite()) {
                    blew = true;
                    break; // `f64::max` discards a NaN, so a diverged run would report 0.000 otherwise
                }
                if k > steps / 2 {
                    mqd = mqd.max(qd[0].abs().max(qd[1].abs()));
                }
            }
            (mqd, blew)
        };

        let (fine, coarse) = (4e-2, 6e-2);
        let r_fine = ctrl.damping_zoh_ratio(fine).expect("ratio");
        let r_coarse = ctrl.damping_zoh_ratio(coarse).expect("ratio");
        assert!(r_fine < 2.0 && r_coarse > 2.0, "40 ms must satisfy and 60 ms must violate: {r_fine:.2}, {r_coarse:.2}");

        let (qd_fine, blew_fine) = hold(fine);
        eprintln!("  40 ms  kd*T {r_fine:.2}: max|qd| {qd_fine:.3e}");
        assert!(!blew_fine && qd_fine < 1e-6, "a certified tick must hold in velocity: {qd_fine:.3e}");

        let (qd_coarse, blew_coarse) = hold(coarse);
        eprintln!("  60 ms  kd*T {r_coarse:.2}: max|qd| {qd_coarse:.3e}, non-finite {blew_coarse}");
        assert!(blew_coarse || qd_coarse > 1.0, "POSITIVE CONTROL: the coarse tick must visibly fail (got {qd_coarse:.3e})");

        // the engineering change, stated as a number: cancelling M buys an 8x coarser tick on this arm
        let imp = CartesianImpedance::new(300.0, 40.0, 2.0);
        let imp_limit = imp.damping_zoh_ratio(&robot, &inertia, &q0, fine).expect("ratio");
        assert!(imp_limit > 2.0, "the impedance controller cannot hold this arm at 40 ms: ratio {imp_limit:.2}");
        eprintln!("  same arm, same 40 ms tick, impedance control: ratio {imp_limit:.2} — cancelling M is what buys the budget");
    }

    /// `ComputedTorque::damping_zoh_ratio` refuses a dt it cannot answer for.
    #[test]
    fn computed_torque_ratio_refuses_unusable_dt() {
        let c = ComputedTorque::new(400.0, 40.0);
        assert_eq!(c.damping_zoh_ratio(1e-3), Some(0.04));
        assert!(c.damping_zoh_ratio(0.0).is_none() && c.damping_zoh_ratio(-1e-3).is_none() && c.damping_zoh_ratio(f64::NAN).is_none());
    }

    /// The house convention (`max_stable_dt`, as in `Admittance::stability_limit` and the motor,
    /// friction and rotordynamics models) must BRACKET the measured boundary for both controllers.
    #[test]
    fn max_stable_dt_brackets_the_measured_boundary_for_both_controllers() {
        let (robot, inertia) = from_urdf_full(ARM2, "base", "tool").unwrap();
        let q0 = [0.3, -0.4];
        let imp = CartesianImpedance::new(300.0, 40.0, 2.0);
        let ct = ComputedTorque::new(400.0, 40.0);
        let imp_lim = imp.max_stable_dt(&robot, &inertia, &q0).expect("limit");
        let ct_lim = ct.max_stable_dt().expect("limit");
        eprintln!("  impedance max_stable_dt = {:.2} ms (holds at 5 ms, limit-cycles at 10 ms)", imp_lim * 1e3);
        eprintln!("  computed torque         = {:.2} ms (holds at 40 ms, diverges at 60 ms)", ct_lim * 1e3);
        assert!((5e-3..10e-3).contains(&imp_lim), "impedance limit must fall between the measured stable and unstable ticks: {imp_lim:.4}");
        assert!((40e-3..60e-3).contains(&ct_lim), "computed-torque limit must too: {ct_lim:.4}");
        assert!(ct_lim > 4.0 * imp_lim, "cancelling M must buy a materially coarser tick: {:.1}x", ct_lim / imp_lim);
        assert!(imp.max_stable_dt(&robot, &inertia, &[f64::NAN, 0.0]).is_none(), "refuses non-finite q");
        assert!(ComputedTorque::new(f64::NAN, 40.0).max_stable_dt().is_none(), "refuses non-finite gains");
    }
    /// `damping_zoh_ratio` refuses inputs it cannot answer for.
    #[test]
    fn damping_zoh_ratio_refuses_unusable_input() {
        let (robot, inertia) = from_urdf_full(ARM2, "base", "tool").unwrap();
        let ctrl = CartesianImpedance::new(300.0, 40.0, 2.0);
        assert!(ctrl.damping_zoh_ratio(&robot, &inertia, &[0.3, -0.4], 1e-3).is_some(), "control");
        assert!(ctrl.damping_zoh_ratio(&robot, &inertia, &[0.3], 1e-3).is_none(), "wrong q length");
        assert!(ctrl.damping_zoh_ratio(&robot, &inertia, &[f64::NAN, 0.0], 1e-3).is_none(), "non-finite q");
        assert!(ctrl.damping_zoh_ratio(&robot, &inertia, &[0.3, -0.4], 0.0).is_none(), "non-positive dt");
        assert!(ctrl.damping_zoh_ratio(&robot, &inertia, &[0.3, -0.4], f64::NAN).is_none(), "non-finite dt");
    }
}

mod hj;
mod hqp;
mod capturability;
mod compass_gait;
mod decomposition;
mod funnel_algebra;
mod hybrid_terminal;
mod hierarchy;
mod embodied;
mod latency;
mod metrology;
mod reliability;
mod regret;
mod lq_regret;
mod envelope;
mod learned_vc;
mod hzd;
mod task_suite;
mod terminal;
mod iss;
mod resclf;
mod koopman;
mod rmpflow;
pub use hj::{solve_brt, HjGrid};
pub use hqp::solve_hqp;
pub use capturability::CaptureParams;
pub use decomposition::{no_recovery_success, Granularity, TaskDecomposition};
pub use funnel_algebra::{compose_chain, required_inflow, sampled_set_escape_rate, samples_for_handoff, set_aware_reliability, ChainCertificate, Incompatibility, Region, SkillFunnel};
pub use hybrid_terminal::{kinetic_metric_residual, lqr_terminal_cost, plastic_impact_terminal, reset_expansion, solve_hybrid_terminal, HybridTerminal};
pub use hierarchy::{hierarchy_cost, margin_of, optimal_slow_period, pareto_frontier, simulate_dual_rate, HierarchyCost, Interface};
pub use embodied::{crossover_seconds, EmbodiedActuator, OwnedCapability};
pub use latency::{delay_margin_samples, delay_margin_seconds, delayed_rho, event_triggered_floor, optimal_compute, uniform_delay_margin, ComputeChoice, sensing_power_floor, SensingBudget};
pub use metrology::{dominant_uncertainty, z_for, task_interval, trials_to_certify_detection, wilson_interval, Interval, MeasuredSkill};
pub use reliability::{compose, detection_equivalent_skill, required_skill, uniform_chain, with_common_cause, ChainOutcome, Skill};
pub use lq_regret::{lqr_gain, place_two, LqLoop};
pub use regret::{log_log_slope, measure_regret, ActionError, ExpertPolicy, Plant, RegretMeasurement, StepCost, Xorshift};
pub use envelope::{best_eta, eiss_envelope, max_disturbance_in_funnel, optimal_split, grid_max_bound, max_tolerable_discrepancy, Envelope, GridBound};
pub use learned_vc::{invariance_defect, train_network, NeuralConstraint, score as score_constraint, train as train_constraint, worst_clearance, GaitGoal, GaitScore, LearnedConstraint};
pub use compass_gait::{CompassGait, GaitState, OutputDynamics, RestrictedCoeffs, RestrictedMap, SwingConstraint, VirtualConstraint};
pub use task_suite::{family_corrected_confidence, locomotion_suite, run_suite, seeds_to_rank, seeds_to_rank_across_suite, Deadbeat, DetunedDeadbeat, Episode, StepPolicy, SuiteScore, Task};
pub use terminal::{check_terminal_ingredients, contact_gap_witness, probe_directions, Dynamics, InputSet, Policy, StageCost, StateFn, TerminalCheck};
pub use resclf::ResClf;
pub use iss::{chunked_imitation_multiplier, disturbance_tube, eiss_step, eiss_ultimate_bound, stochastic_contraction_bound, stochastic_steady_state};
pub use hzd::{hybrid_invariance_residual, hzd_reduction, is_minimum_phase, is_minimum_phase_order2, zero_dynamics, zero_dynamics_order2, HzdReduction, ZeroDynamicsReturnMap};
pub use koopman::{edmd, edmdc, Koopman};
pub use rmpflow::RmpArm;
pub use rts::{RtsSmoother, SmoothResult};
