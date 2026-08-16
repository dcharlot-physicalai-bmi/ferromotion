//! ferromotion-core — robot kinematic optimization in Rust.
//!
//! A revolute/prismatic kinematic chain, forward kinematics on SE(3), an analytic geometric
//! Jacobian, a composable [`Cost`] trait, and a Levenberg–Marquardt solver. Load real robots
//! from URDF (parsed from memory, so it works in the browser too). Pure `nalgebra` +
//! `urdf-rs` — compiles clean to `wasm32-unknown-unknown`.
//!
//! This mirrors PyRoki's spine: a `Robot`, a set of composable costs (pose, joint-limit,
//! posture, …), and a nonlinear-least-squares solve. IK is `solve` over a `PoseCost`.

use nalgebra::{DMatrix, DVector, Isometry3, Translation3, Unit, UnitQuaternion, Vector3, Vector6};

mod aba;
mod adaptive_contact;
mod affine_contact;
mod contacts_from_distance;
mod tree_dynamics;
mod crba;
mod apriltag;
mod bvh;
mod bit_star;
mod bspline;
mod bundle;
mod camera;
mod causal;
mod catmull_rom;
mod cfd_contact;
mod chomp;
mod closed_loop;
mod collision;
mod constraint;
mod constraints;
mod contact_gradient;
mod contact;
mod contact_ipm;
mod contact_pgs;
mod cosserat;
mod costs;
mod dex_retarget;
mod esdf;
mod epa;
mod essential;
mod fft;
mod fitting;
mod foci;
mod gjk;
mod diffik;
mod dcol;
mod despike;
mod dubins;
mod dynamics_workspace;
mod dyn_derivatives;
mod dynamics;
pub mod gendyn;
mod geometry2d;
mod gpmp2;
mod hand_eye;
mod homography;
mod hydroelastic;
mod hybrid_gradient;
mod identification;
mod kinematic_tree;
mod hand_object;
mod hybrid;
mod hybrid_astar;
mod iir;
mod infothermo;
mod grasp_spatial;
mod grasp;
mod great_circle;
mod grid_astar;
mod gcs;
mod geodetic;
mod gnc;
mod iris;
mod icp;
mod ipc;
mod isam;
mod ipm;
mod kiss_icp;
mod kmeans;
mod kdl;
mod lgvi;
mod leg_smoother;
mod manipulability;
mod marginalize;
mod mestimator;
mod mesh3;
mod modal;
mod occupancy;
mod ode;
mod orb;
mod chained_form;
mod grasp_bounds;
mod fourier_steering;
mod geometric_phase;
mod hall_basis;
mod paden_kahan;
mod reciprocal;
mod rolling_contact;
mod pink;
mod planar_contact;
mod pca;
mod pnp;
mod pomdp;
mod polyline;
mod polyroots;
mod pose_graph;
mod quadrature;
mod quat_mean;
mod radar_velocity;
mod randomization;
mod ransac;
mod reeds_shepp;
mod rotation_avg;
mod running_stats;
mod translation_avg;
mod trilateration;
mod savgol;
mod retarget;
mod rigidbody;
mod robot_contact;
mod floating_contact;
mod whole_body_contact;
mod robot_plan;
#[cfg(feature = "gpu")]
pub mod gpu;
mod robust;
mod prm;
mod rrt;
mod cspace_sdf;
mod screw;
mod sdformat;
mod sdf;
mod sensor_render;
mod sdp;
mod sgm;
mod lmi;
mod spatial;
mod sparse;
mod spline_se3;
mod stl;
mod sysid;
mod tdoa;
mod teaser;
mod tensegrity;
mod traj;
mod sampler_bounds;
mod transport;
pub mod transport_geometry;
mod mjcf;
mod usda;
mod urdf;
mod xcorr;
mod xpbd;
pub use aba::{floating_base_forward_dynamics, floating_base_forward_dynamics_ext, forward_dynamics_aba};
pub use tree_dynamics::{tree_floating_forward_dynamics, tree_floating_mass_matrix};
pub use crba::crba;
pub use apriltag::{decode_payload, tag_pose};
pub use cfd_contact::{rollout_impulse, CfdContact};
pub use bit_star::BitStar;
pub use bvh::{Aabb, Bvh};
pub use mesh3::{convex_hull_3d, TriMesh3};
pub use bspline::BSpline;
pub use bundle::{BundleAdjustment, Camera, Observation};
pub use camera::{calibrate, PinholeCamera};
pub use catmull_rom::CatmullRom;
pub use chomp::{Chomp, ChompResult};
pub use closed_loop::{Pin, PlanarLoop};
pub use collision::{CapsuleCollisionCost, PlaneCollisionCost, SphereCollisionCost};
pub use cosserat::CosseratRod;
pub use foci::{collision_cost, collision_grad_p, collision_kernel, overlap_integral, plan as foci_plan, FociPlan, Gaussian3, RobotSplat};
pub use esdf::Esdf;
pub use fft::{fft, fft_real, ifft};
pub use xcorr::{convolve, cross_correlation, time_delay};
pub use fitting::{fit_circle, fit_sphere};
pub use epa::{epa, sphere_penetration, Penetration};
pub use essential::{decompose_essential, eight_point, recover_pose};
pub use gjk::{ccd_toi, gjk, intersects, Ball, ConvexPoints, Cuboid, GjkResult, Support, Translate};
pub use contact_gradient::{jacobian_error, jacobian_relative_error, BouncingMass, PenaltyMass, GRAVITY};
pub use affine_contact::{AffineContact, ExactContact, SpringExit};
pub use contacts_from_distance::{descend, CfdProfile, DescentResult, PushTask};
pub use adaptive_contact::{converged_reference, decompose_gradient_error, AdaptiveError, AdaptiveOptions, AdaptivePenalty, AdaptiveStats, ErrorDecomposition};
pub use contact_ipm::{solve_frictional_ipm, FrictionalStep, StFrictionContact};
pub use sampler_bounds::{consistency_error, flow_matching_w2, minimax_rate, score_to_tv, SamplerError};
pub use transport_geometry::{cramer_distance, distributional_bellman, distributional_bellman_contraction, free_energy, gromov_cost, gromov_wasserstein, gromov_wasserstein_from, jko_step, kantorovich_dual, maximal_wasserstein_1, schrodinger_bridge, total_variation, GromovPlan};
pub use contact_pgs::{solve_contacts_pgs, solve_contacts_pgs_with, PgsContact, PgsResult, PgsStabilization};
pub use contact::{
    solve_contacts, solve_contacts_diff, solve_contacts_friction, Contact, ContactSolve,
    ContactSolveDiff, FrictionContact,
};
pub use diffik::{solve_diffik, DiffIkOptions, DiffIkResult, FrameTaskDef};
pub use dcol::{proximity, proximity_grad_spheres, Primitive};
pub use dynamics::{forward_dynamics, gravity_vector, inverse_dynamics, mass_matrix, LinkInertia};
pub use grasp_spatial::{force_closure_q1_planar_subspace, force_closure_q1_spatial, force_closure_soft_spatial, grasp_matrix, grasp_split, net_wrench, primitive_wrenches_spatial, wrench_rank, GraspContact3, GraspSplit};
pub use grasp::{force_closure_q1, force_closure_soft, primitive_wrenches, GraspContact};
pub use great_circle::{cross_track_distance, destination, haversine_distance, initial_bearing, EARTH_RADIUS};
pub use grid_astar::{astar_grid, octile, path_length as grid_path_length};
pub use hand_eye::hand_eye_calibration;
pub use geometry2d::{convex_hull, min_enclosing_circle, point_in_polygon, polygon_area, polygon_centroid, signed_area, Circle};
pub use gpmp2::Gpmp2;
pub use homography::{apply_homography, homography_dlt, transfer_error};
pub use hydroelastic::{
    equal_spheres_contact, linear_pressure, sphere_plane_contact, sphere_plane_force_closed_form, HydroContact,
};
pub use iir::Biquad;
pub use gcs::{Gcs, GcsPath, HPolytope};
pub use geodetic::{ecef_to_enu, ecef_to_lla, enu_to_ecef, lla_to_ecef, lla_to_enu};
pub use gnc::{gnc_solve, GncResult};
pub use marginalize::{add_factor, GaussianInfo};
pub use mestimator::{barron, RobustKernel};
pub use despike::{hampel_filter, median_filter};
pub use dubins::{dubins_shortest, DubinsPath, Pose as DubinsPose, Seg as DubinsSeg};
pub use dynamics_workspace::{forward_dynamics_in, inverse_dynamics_in, mass_matrix_in, DynamicsWorkspace};
pub use dyn_derivatives::{forward_dynamics_derivatives, id_derivatives};
pub use iris::{ConvexRegion as IrisRegion, Iris};
pub use icp::{covariance_from_normal, umeyama, Icp, IcpResult};
pub use ipc::{barrier, barrier_grad, barrier_hess, IpcFloor};
pub use isam::IncrementalLeastSquares;
pub use ipm::{solve_lcp, solve_lcp_diff, solve_lcp_smoothed};
pub use kdl::resolved_rate;
pub use kiss_icp::KissIcp;
pub use kmeans::{kmeans, KMeans};
pub use occupancy::OccupancyGrid;
pub use ode::{dopri5_step, integrate, OdeSolution};
pub use orb::{
    brief_descriptor, brief_pattern, detect_and_describe, fast_corners, match_descriptors, nms, orient,
    Descriptor, GrayImage, Keypoint,
};
pub use pca::{obb_from_points, pca, Obb, Pca};
pub use pnp::{pnp, pnp_dlt, pnp_gn, reprojection_error};
pub use polyline::{polyline_length, rdp_simplify, resample_uniform};
pub use polyroots::{real_roots, roots};
pub use pose_graph::{Pose2, PoseGraph2D};
pub use quadrature::{gauss_legendre, integrate as gauss_integrate};
pub use quat_mean::average_quaternions;
pub use hand_object::{hand_object_step, HandObjectStep, SphereObject};
pub use kinematic_tree::{tree_from_urdf, tree_ik, KinematicTree, TipTarget, TreeIkResult};
pub use identification::{identify_consistent, identify_with_covariance, params_from_pseudo_inertia, ConsistentFit, IdentifiedParams};
pub use hybrid_gradient::{hybrid_jacobian, HybridGradientOptions, probe_stable_jacobian, split_residual, HybridGradientError, HybridLinearisation, HybridSystem};
pub use hybrid::{compose_monodromy, find_limit_cycle, flow_jacobian, hybrid_certificate, impact_expansion, return_map_jacobian, plastic_impact, plastic_impact_jacobian, poincare_stability, saltation_matrix, transverse_basis, transverse_metric, transverse_restriction, HybridCertificate, HybridEvent};
pub use hybrid_astar::{hybrid_astar, HybridConfig};
pub use reeds_shepp::{path_length as rs_path_length, reeds_shepp, RsSegment};
pub use rotation_avg::{rotation_averaging, spanning_tree_init, RotEdge};
pub use running_stats::RunningStats;
pub use translation_avg::{translation_averaging, TransEdge};
pub use trilateration::trilaterate;
pub use radar_velocity::{ego_velocity_ls, ego_velocity_ransac};
pub use randomization::{AdrSchedule, Lcg, ParameterDistribution};
pub use ransac::{ransac, RansacResult};
pub use savgol::SavGol;
pub use lgvi::LgviBody;
pub use manipulability::{condition_number, force_ellipsoid_axes, isotropy, manipulability_gradient, singular_values, yoshikawa};
pub use modal::{modal_analysis, ModalModel};
pub use mjcf::{from_mjcf_constrained, from_mjcf_full, from_mjcf_str};
pub use usda::{axis_token, parse_usda, robot_from_usda, usda_from_robot, ParseError, Prim, UsdaStage, Value};
pub use urdf::from_urdf_full;
pub use robust::solve_ik_robust;
pub use prm::{PrmStar, Roadmap};
pub use rrt::{RrtResult, RrtStar};
pub use robot_plan::{arm_clearance, arm_spheres, plan_arm_reach, ReachPlanOptions};
pub use constraints::{solve_al, AlOptions, AlResult, PlaneConstraint};
pub use constraint::{constrained_step, constrained_step_with, ConstraintSet, Delassus, Group, Law, LinkSphere, Solver, StepResult};
pub use costs::{Cost, JointLimitCost, PointCost, PoseCost, PostureCost, VectorCost};
pub use rigidbody::RigidBody;
pub use retarget::{FrameTask, Retargeter, VectorRetargeter, VectorTask};
pub use dex_retarget::{
    DexPilotRetargeter, PositionCorr, PositionRetargeter, VectorCorr,
    VectorRetargeter as DexVectorRetargeter,
};
pub use planar_contact::PlanarBody;
pub use robot_contact::RobotContactSim;
pub use floating_contact::{floating_contact_step, quadruped, quadruped_trot_tau, tree_floating_contact_step, FootContact};
pub use whole_body_contact::{whole_body_contact_jacobian, whole_body_contact_step, whole_body_contact_step_checked, whole_body_contact_step_pgs, whole_body_forward_kinematics, WholeBodyContactPoint, WholeBodyStep};
pub use pink::{
    solve_pink, FramePoseTask, PinkOptions, PinkResult, PinkSolver, PinkTask, PostureTask, TaskStack,
};
pub use leg_smoother::{LegSmoother, PriorPose2, RelPose2};
pub use cspace_sdf::{CspaceField, PlanarArm};
pub use chained_form::ChainedForm;
pub use grasp::is_force_closure;
pub use fourier_steering::{alpha_displacement, first_harmonic, integrate_period};
pub use geometric_phase::{cap_area_from_latitude, geometric_phase, lift_path, psi_dot};
pub use hall_basis::{admissible, hall_basis, witt_dimension, LieProduct};
pub use rolling_contact::{contact_derivative, r_psi, ContactRates, GeometricParams};
pub use grasp_bounds::{caratheodory_lower_bound, frictionless_contact_bracket, is_rank_deficient, min_contacts, steinitz_upper_bound, ContactModel};
pub use reciprocal::{are_reciprocal, power, reciprocal_basis, reciprocal_product, reciprocal_product_geometric, swap_wrench_halves, Screw};
pub use paden_kahan::{rotate_about_axis, subproblem1, subproblem2, subproblem3};
pub use screw::{ad, adjoint, exp_se3, exp_so3, hat3, log_se3, log_so3 as screw_log_so3, poe_fk, pose, revolute_axis, rot_of, sclerp, trans_of, vee3};
pub use spline_se3::SplineSE3;
pub use sdf::{op_intersect, op_smooth_union, op_subtract, op_union, Sdf, SdfScene};
pub use sdformat::from_sdf;
pub use sensor_render::{raymarch, DepthCamera, DepthImage, Lidar, LidarScan, RayHit};
pub use sdp::{project_psd, solve_sdp, SdpProblem, SdpSolution};
pub use sgm::{census5x5, cost_volume, disparity_map, StereoParams};
pub use sparse::{solve_factor_graph, SparseFactor, SparseResult};
pub use causal::Scm;
pub use infothermo::{bits_affordable, bode_sensitivity_integral, entropy_production, landauer_energy, max_extractable_work, BOLTZMANN};
pub use pomdp::{best_sensing_action, expected_information_gain, Belief};
pub use stl::Stl;
pub use transport::{sinkhorn, squared_cost, w1_empirical_1d, w2_gaussian, SinkhornPlan};
pub use lmi::{is_hurwitz, is_schur, lyapunov, lyapunov_discrete, solve_lyapunov, solve_lyapunov_discrete};
pub use spatial::{KdTree, VoxelHash};
pub use tensegrity::{Member, Tensegrity};
pub use sysid::{identify, inertial_regressor, is_physically_consistent, params_from_inertia, pseudo_inertia, IdSample, PARAMS_PER_LINK};
pub use tdoa::tdoa_localize;
pub use teaser::{register, Registration};
pub use traj::{TrajectoryProblem, TrajectoryResult};
pub use urdf::from_urdf_str;
pub use xpbd::{DistanceConstraint, Particle as XpbdParticle, XpbdSolver};

/// SE(3) rigid transform.
pub type Iso = Isometry3<f64>;

/// A single-DoF joint.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum JointKind {
    Revolute,
    Prismatic,
}

/// One actuated joint: a fixed `origin` from the parent frame, then motion about/along `axis`.
/// (URDF `fixed` joints are folded into neighbouring origins at load time.)
#[derive(Clone, Debug)]
pub struct Joint {
    /// Fixed transform from the previous joint's frame to this joint's frame.
    pub origin: Iso,
    /// Joint axis, expressed in this joint's frame.
    pub axis: Unit<Vector3<f64>>,
    pub kind: JointKind,
    /// Optional (lower, upper) position limits.
    pub limits: Option<(f64, f64)>,
}

impl Joint {
    pub fn revolute(origin: Iso, axis: Vector3<f64>) -> Self {
        Self { origin, axis: Unit::new_normalize(axis), kind: JointKind::Revolute, limits: None }
    }

    pub fn prismatic(origin: Iso, axis: Vector3<f64>) -> Self {
        Self { origin, axis: Unit::new_normalize(axis), kind: JointKind::Prismatic, limits: None }
    }

    pub fn with_limits(mut self, lower: f64, upper: f64) -> Self {
        self.limits = Some((lower, upper));
        self
    }

    fn motion(&self, q: f64) -> Iso {
        match self.kind {
            JointKind::Revolute => {
                Iso::from_parts(Translation3::identity(), UnitQuaternion::from_axis_angle(&self.axis, q))
            }
            JointKind::Prismatic => {
                Iso::from_parts(Translation3::from(self.axis.into_inner() * q), UnitQuaternion::identity())
            }
        }
    }

    fn transform(&self, q: f64) -> Iso {
        self.origin * self.motion(q)
    }
}

/// A serial kinematic chain terminating in a tool frame (`ee_offset`).
#[derive(Clone, Debug)]
pub struct Robot {
    pub joints: Vec<Joint>,
    pub ee_offset: Iso,
}

impl Robot {
    pub fn dof(&self) -> usize {
        self.joints.len()
    }

    /// End-effector pose for configuration `q`.
    pub fn fk(&self, q: &[f64]) -> Iso {
        let mut t = Iso::identity();
        for (j, &qi) in self.joints.iter().zip(q) {
            t *= j.transform(qi);
        }
        t * self.ee_offset
    }

    /// 6×N world-frame geometric Jacobian: dq → [linear; angular] end-effector velocity.
    pub fn jacobian(&self, q: &[f64]) -> DMatrix<f64> {
        let n = self.dof();
        let mut jac = DMatrix::zeros(6, n);
        let p_ee = self.fk(q).translation.vector;
        let mut t = Iso::identity();
        for (i, (j, &qi)) in self.joints.iter().zip(q).enumerate() {
            let pre = t * j.origin; // this joint's frame, before applying qi
            let z = pre.rotation * j.axis.into_inner(); // joint axis in world
            let p = pre.translation.vector; // joint origin in world
            match j.kind {
                JointKind::Revolute => {
                    let lin = z.cross(&(p_ee - p));
                    jac.fixed_view_mut::<3, 1>(0, i).copy_from(&lin);
                    jac.fixed_view_mut::<3, 1>(3, i).copy_from(&z);
                }
                JointKind::Prismatic => {
                    jac.fixed_view_mut::<3, 1>(0, i).copy_from(&z);
                }
            }
            t = pre * j.motion(qi);
        }
        jac
    }

    /// World pose of the frame after the first `upto` joints (`0..=dof`); `upto = dof` is the
    /// chain end (before the tool offset). Lets costs target any point along the body.
    pub fn frame_pose(&self, q: &[f64], upto: usize) -> Iso {
        let mut t = Iso::identity();
        for (j, &qi) in self.joints.iter().zip(q).take(upto) {
            t *= j.transform(qi);
        }
        t
    }

    /// 3×N position Jacobian of a world point rigidly attached at frame `upto`. Only joints
    /// before `upto` move it; the rest are zero columns.
    pub fn point_jacobian(&self, q: &[f64], upto: usize, world_point: &Vector3<f64>) -> DMatrix<f64> {
        let n = self.dof();
        let mut jac = DMatrix::zeros(3, n);
        let mut t = Iso::identity();
        for (i, (j, &qi)) in self.joints.iter().zip(q).enumerate() {
            if i >= upto {
                break;
            }
            let pre = t * j.origin;
            let z = pre.rotation * j.axis.into_inner();
            let p = pre.translation.vector;
            match j.kind {
                JointKind::Revolute => {
                    let col = z.cross(&(world_point - p));
                    jac.fixed_view_mut::<3, 1>(0, i).copy_from(&col);
                }
                JointKind::Prismatic => {
                    jac.fixed_view_mut::<3, 1>(0, i).copy_from(&z);
                }
            }
            t = pre * j.motion(qi);
        }
        jac
    }
}

/// Twist error `[Δposition; log(R_ee · R_target⁻¹)]` in the world frame.
pub(crate) fn pose_error(current: &Iso, target: &Iso) -> Vector6<f64> {
    let dp = current.translation.vector - target.translation.vector;
    let dr = (current.rotation * target.rotation.inverse()).scaled_axis();
    Vector6::new(dp.x, dp.y, dp.z, dr.x, dr.y, dr.z)
}

// ---------------------------------------------------------------------------
// Generic nonlinear-least-squares solver over composable costs.
// ---------------------------------------------------------------------------

/// Options for the [`solve`] Levenberg–Marquardt loop.
#[derive(Clone, Copy, Debug)]
pub struct SolveOptions {
    pub max_iters: usize,
    pub tol: f64,
    pub lambda0: f64,
}

impl Default for SolveOptions {
    fn default() -> Self {
        Self { max_iters: 200, tol: 1e-10, lambda0: 1e-2 }
    }
}

/// Outcome of a [`solve`].
#[derive(Clone, Debug)]
pub struct SolveResult {
    pub q: Vec<f64>,
    /// Final stacked-residual norm.
    pub error: f64,
    pub iters: usize,
    pub converged: bool,
}

fn stacked_dim(robot: &Robot, costs: &[Box<dyn Cost>]) -> usize {
    costs.iter().map(|c| c.dim(robot)).sum()
}

fn stacked_residual(robot: &Robot, costs: &[Box<dyn Cost>], q: &[f64]) -> DVector<f64> {
    let mut r = DVector::zeros(stacked_dim(robot, costs));
    let mut off = 0;
    for c in costs {
        let d = c.dim(robot);
        r.rows_mut(off, d).copy_from(&c.residual(robot, q));
        off += d;
    }
    r
}

fn stacked_jacobian(robot: &Robot, costs: &[Box<dyn Cost>], q: &[f64]) -> DMatrix<f64> {
    let n = robot.dof();
    let mut j = DMatrix::zeros(stacked_dim(robot, costs), n);
    let mut off = 0;
    for c in costs {
        let d = c.dim(robot);
        j.view_mut((off, 0), (d, n)).copy_from(&c.jacobian(robot, q));
        off += d;
    }
    j
}

/// Levenberg–Marquardt minimization of the stacked cost residuals, from seed `q0`.
pub fn solve(robot: &Robot, costs: &[Box<dyn Cost>], q0: &[f64], opts: &SolveOptions) -> SolveResult {
    let n = robot.dof();
    let mut q = DVector::from_row_slice(q0);
    let mut lambda = opts.lambda0;

    let mut r = stacked_residual(robot, costs, q.as_slice());
    let mut cost = r.norm_squared();
    let mut iters = 0;

    'outer: for it in 0..opts.max_iters {
        iters = it + 1;
        if r.norm() < opts.tol {
            break;
        }
        let j = stacked_jacobian(robot, costs, q.as_slice());
        let jt = j.transpose();
        let jtj = &jt * &j;
        let g = &jt * &r;

        loop {
            let mut a = jtj.clone();
            for d in 0..n {
                a[(d, d)] += lambda;
            }
            let dq = match a.clone().cholesky() {
                Some(ch) => ch.solve(&g),
                None => a.lu().solve(&g).unwrap_or_else(|| DVector::zeros(n)),
            };
            let q_new = &q - &dq;
            let r_new = stacked_residual(robot, costs, q_new.as_slice());
            let cost_new = r_new.norm_squared();
            if cost_new < cost {
                q = q_new;
                r = r_new;
                cost = cost_new;
                lambda = (lambda * 0.5).max(1e-12);
                break;
            }
            lambda *= 3.0;
            if lambda > 1e12 {
                break 'outer; // stalled — no step improves the cost
            }
        }
    }

    let err = r.norm();
    SolveResult { q: q.as_slice().to_vec(), error: err, iters, converged: err < 1e-4 }
}

// ---------------------------------------------------------------------------
// Convenience IK API (pose-only), preserved for the WASM bindings.
// ---------------------------------------------------------------------------

/// Options for [`solve_ik`].
#[derive(Clone, Copy, Debug)]
pub struct IkOptions {
    pub max_iters: usize,
    pub tol: f64,
    pub lambda0: f64,
    pub pos_weight: f64,
    pub rot_weight: f64,
}

impl Default for IkOptions {
    fn default() -> Self {
        Self { max_iters: 200, tol: 1e-10, lambda0: 1e-2, pos_weight: 1.0, rot_weight: 1.0 }
    }
}

/// Outcome of an IK solve.
pub type IkResult = SolveResult;

/// Levenberg–Marquardt inverse kinematics: drive the end-effector to `target` from seed `q0`.
pub fn solve_ik(robot: &Robot, target: &Iso, q0: &[f64], opts: &IkOptions) -> IkResult {
    let costs: Vec<Box<dyn Cost>> =
        vec![Box::new(PoseCost::new(*target, opts.pos_weight, opts.rot_weight))];
    let so = SolveOptions { max_iters: opts.max_iters, tol: opts.tol, lambda0: opts.lambda0 };
    solve(robot, &costs, q0, &so)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn planar_3r() -> Robot {
        let z = Vector3::z();
        let link = |l: f64| Iso::from_parts(Translation3::new(l, 0.0, 0.0), UnitQuaternion::identity());
        Robot {
            joints: vec![
                Joint::revolute(Iso::identity(), z),
                Joint::revolute(link(1.0), z),
                Joint::revolute(link(1.0), z),
            ],
            ee_offset: link(1.0),
        }
    }

    #[test]
    fn fk_at_zero_is_fully_extended() {
        let r = planar_3r();
        let p = r.fk(&[0.0, 0.0, 0.0]).translation.vector;
        assert!((p - Vector3::new(3.0, 0.0, 0.0)).norm() < 1e-9, "got {p:?}");
    }

    #[test]
    fn jacobian_matches_finite_difference() {
        let r = planar_3r();
        let q = [0.2, -0.4, 0.7];
        let analytic = r.jacobian(&q);
        let eps = 1e-6;
        for i in 0..3 {
            let mut qp = q;
            qp[i] += eps;
            let dp = (r.fk(&qp).translation.vector - r.fk(&q).translation.vector) / eps;
            for row in 0..3 {
                assert!((analytic[(row, i)] - dp[row]).abs() < 1e-4, "col {i} row {row}");
            }
        }
    }

    #[test]
    fn ik_reaches_a_reachable_pose() {
        let r = planar_3r();
        let q_true = [0.3f64, -0.5, 0.4];
        let target = r.fk(&q_true);
        let res = solve_ik(&r, &target, &[0.0, 0.0, 0.0], &IkOptions::default());
        assert!(res.converged, "did not converge: err={} iters={}", res.error, res.iters);
        let e = pose_error(&r.fk(&res.q), &target);
        assert!(e.norm() < 1e-4, "residual pose error {}", e.norm());
    }
}
