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
    certify, nominal_activity, propagate_tube, reaches_goal, GapBound, GapEvidence, HalfSpace, TubeReport, TubeStep,
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
pub use latency::{delay_margin_samples, delay_margin_seconds, delayed_rho, event_triggered_floor, optimal_compute, uniform_delay_margin, ComputeChoice};
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
