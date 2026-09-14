//! **ferromotion-core — the model-based substrate for physical-AI motion, in pure Rust.**
//!
//! 150 modules over one shared [`Robot`]: load a real robot, differentiate its dynamics, resolve its
//! contacts, plan and optimise its motion, estimate its state, and check that the answer obeys the
//! physics. Pure `nalgebra` and `urdf-rs`, no BLAS, no Python, no GPU required — and it compiles clean
//! to `wasm32-unknown-unknown`, so the same code runs on the robot, on a laptop, and in a browser tab.
//!
//! ⛔ **This doc block read "robot kinematic optimization in Rust — a kinematic chain, forward
//! kinematics, a Jacobian, and a Levenberg–Marquardt solver" for 150 modules**, and closed by
//! describing the crate as a mirror of another project's spine. That was written when the crate had
//! about six modules and never grew with it. It is the first thing every docs.rs visitor reads, so it
//! was the single largest gap between what this crate does and what a reader could discover. The map
//! below is organised by what you are trying to do.
//!
//! # Load a robot
//!
//! [`from_urdf_str`] and [`from_urdf_full`] (URDF, parsed from a string so it works in the browser),
//! [`from_mjcf_str`] (MuJoCo XML), [`from_sdf`] (Gazebo SDFormat), [`parse_usda`] / [`robot_from_usda`]
//! / [`usda_from_robot`] (OpenUSD `UsdPhysics`, read *and* write), [`Robot::from_dh`] (a
//! Denavit–Hartenberg table straight off a datasheet), [`Ets`] (a robot as a string of elementary
//! transforms), [`tree_from_urdf`] for a branched tree such as a hand, and `closed_loop` for linkages
//! that are not serial chains.
//!
//! Every one of those formats points at geometry it does not carry. [`geometry_from_urdf`],
//! [`geometry_from_mjcf`] and [`geometry_from_sdf`] surface it — every shape with its origin, scale
//! and [`GeomRole`], with MJCF's half-extents converted to URDF's full ones and its `fromto` form
//! resolved to a pose. Each also reports **which links state no usable mass**, because all three
//! loaders substitute a silent zero for one they could have computed. And
//! [`resolve_uri`] expands a `package://`, `model://` or `file://` URI against a caller-supplied
//! package table. [`from_obj`] and [`from_stl`] (both encodings, dispatched on content rather than
//! extension) read the bytes; [`scale_mesh`] applies the description's scale; [`solid_inertia`]
//! integrates a mesh and a density into the [`LinkInertia`] the dynamics already take; and
//! [`inertia_of_parts`] composes several shapes in one link frame by parallel-axis about the
//! *combined* centre of mass, which is not the same as adding their tensors.
//!
//! [`primitive_mesh`] and [`primitive_link_inertia`] close the loop for a description made of boxes,
//! cylinders, capsules and spheres: those need **no asset at all**, so such a URDF yields a full
//! inertia vector from the XML alone. Byte slices, not paths: no asset is vendored and nothing here
//! reads the filesystem.
//!
//! ⛔ A URDF is not an actuator model. Reflected rotor inertia, viscous damping and Coulomb friction
//! are read from the model and applied *inside* RNEA — see [`identify_actuator`] and
//! [`actuator_plausibility`], and the section below `## A URDF is not an actuator model` in this
//! crate's README for why a model without them silently misreports torque.
//!
//! # Kinematics and inverse kinematics
//!
//! Forward kinematics and analytic geometric Jacobians on SE(3) live on [`Robot`]; `screw` carries the
//! Lie-group machinery (product of exponentials, adjoints, twists), `paden_kahan` the closed-form
//! geometric IK subproblems, [`yoshikawa`] and [`manipulability_gradient_analytic`] the manipulability
//! measures, and `reciprocal` the wrench/twist duality.
//!
//! IK comes in four shapes, because they fail differently: [`solve_ik`] (Levenberg–Marquardt over a
//! composable [`Cost`] stack), [`solve_ik_robust`] (restarts, for when LM stalls in a local minimum),
//! [`solve_diffik`] (a per-step velocity QP over a task stack), and [`tree_ik`] for a branched tree
//! with several tips. [`Retargeter`], [`PositionRetargeter`], [`VectorRetargeter`] and
//! [`DexPilotRetargeter`] map an observed keypoint stream — human demo, teleop, mocap — onto a robot.
//!
//! # Dynamics
//!
//! [`inverse_dynamics`] (RNEA) and [`mass_matrix`]; [`forward_dynamics`] via the mass-matrix solve and
//! [`forward_dynamics_aba`] via the O(n) Articulated-Body Algorithm; [`crba`] for the joint-space
//! inertia directly; [`floating_base_forward_dynamics`] and [`tree_floating_forward_dynamics`] for a
//! floating base, serial or branched; **analytical** derivatives `∂/∂q, ∂/∂q̇, ∂/∂τ` rather than finite
//! differences; [`gendyn`] over a generic scalar so the whole pipeline differentiates under an AD tape;
//! and [`forward_dynamics_in`] for a control loop with a deadline, allocation-free at steady state.
//!
//! Beyond rigid bodies: `lgvi` (a Lie-group variational integrator that needs no re-normalisation and
//! has no gimbal singularity), `rigidbody` (symplectic free rigid body), [`ModalModel`] (reduced-order
//! deformables), `cosserat` (variable-strain soft rods), `tensegrity` (force-density form-finding).
//!
//! # Contact
//!
//! The part that decides whether a physical-AI result transfers. Six solvers, because the right one
//! depends on what you need from it: [`solve_contacts_pgs`] (robust projected Gauss–Seidel),
//! [`solve_frictional_ipm`] (interior-point, fully differentiable), [`IpcFloor`] (barrier-based, guaranteed
//! intersection-free), [`HydroContact`] (pressure-field, smooth distributed forces), [`XpbdSolver`]
//! (position-based, small-steps), and [`AffineContact`] (the penalty contact in closed form, exact to
//! round-off, which is what the integrators are checked against).
//!
//! Applied at the level you need it: [`RobotContactSim`] for an articulated body,
//! [`floating_contact_step`] for a floating base, [`whole_body_contact_step`] for one hard frictional
//! non-penetrating solve over a whole body, and `hand_object` for a hand and an object in one solve.
//!
//! ⛔ **Gradients through contact are where differentiable simulators go wrong, and this crate measures
//! it rather than assuming it.** [`ContactLawResidual`] checks the answer against Signorini,
//! Coulomb and maximum-dissipation — a *solver* residual is not a *law* residual, and this crate has
//! shipped a solver reporting 9.8e-6 while violating Signorini by 9.8e-3. [`ContactModel`],
//! [`hybrid_jacobian`] and [`CfdContact`] carry the gradient story, including where it stops being
//! valid; `adaptive_contact` carries the error decomposition.
//!
//! # Collision geometry
//!
//! [`gjk`] and [`epa`] (narrowphase distance and penetration depth, with conservative-advancement
//! CCD) over the [`gjk::Support`] trait, so a ball, an oriented box or an arbitrary
//! [`gjk::ConvexPoints`] set all answer the same query. A link is not convex,
//! so [`CompoundHull`] carries it as a SET of convex parts: [`compound_distance`] gives the closest
//! pair a planner wants and [`compound_contacts`] gives **every** pair within a margin, which is the
//! constraint set a solver needs — one closest pair loses a simultaneous contact and the body sinks
//! through the other. [`compound_pgs_contacts`] carries that set the rest of the way to
//! [`solve_contacts_pgs`], building the contact frame and differencing the two bodies' Jacobians
//! through the [`ContactJacobian`] trait ([`StaticBody`] for a fixed world, [`SerialLinkParts`] for a
//! chain). [`try_convex_hull_3d`] hulls a loaded mesh into a part, refusing a degenerate
//! one rather than panicking. Also [`Bvh`] (AABB broadphase), [`SdfScene`] and [`Esdf`] and [`CspaceField`] (signed-distance
//! representations, including a composite configuration-space field), [`FociPlan`] (field-overlap
//! collision integral), `dcol` (differentiable collision between convex primitives), [`OccupancyGrid`]
//! grids from range sensors, and [`SphereCollisionCost`] for the sphere-model robot representation.
//!
//! # Planning
//!
//! Sampling: [`RrtStar`], [`PrmStar`], [`BitStar`]. Optimisation-based: [`Chomp`], [`Gpmp2`],
//! [`TrajectoryProblem`] (block-tridiagonal trajectory optimisation), [`solve_factor_graph`] (a general
//! factor graph, for topologies beyond a chain). Convex decomposition: [`Iris`] and [`Gcs`]
//! (shortest paths through graphs of convex sets). Grids and lattices: [`astar_grid`],
//! [`hybrid_astar`], [`lattice_astar`], [`LatticeDStarLite`] (incremental replanning when cells flip),
//! [`bug2`], [`distance_transform_plan`]. Car-like: [`dubins_shortest`], [`reeds_shepp`].
//! Nonholonomic: `chained_form`, `fourier_steering`, `hall_basis`. And [`plan_arm_reach`] as the bridge
//! that puts the sampling planners on a [`Robot`].
//!
//! Curves and integration: [`BSpline`], [`CatmullRom`], [`SplineSE3`] (continuous-time SE(3)),
//! [`dopri5_step`] (adaptive Dormand–Prince), [`gauss_legendre`].
//!
//! # Estimation and perception
//!
//! Factor graphs and smoothing: [`IncrementalLeastSquares`] (incremental QR), [`PoseGraph2D`],
//! `marginalize` (Schur complement, fixed-lag), [`LegSmoother`]. Robustness: [`gnc_solve`]
//! (graduated non-convexity), [`ransac`], the `mestimator` kernels. Geometry from images:
//! [`pnp`], [`decompose_essential`], [`homography_dlt`], [`BundleAdjustment`], [`PinholeCamera`],
//! `orb`, `sgm`, [`tag_pose`]. Point clouds: [`Icp`], [`KissIcp`],
//! `teaser`. Averaging on manifolds: [`average_quaternions`], [`rotation_averaging`],
//! [`translation_averaging`]. Ranging and navigation: [`trilaterate`], [`tdoa_localize`],
//! `radar_velocity`, [`lla_to_ecef`], `great_circle`. Synthetic sensing:
//! `sensor_render` gives depth cameras and lidar by sphere tracing over the analytic scene.
//!
//! Signals: [`fft`], [`cross_correlation`], [`Biquad`], [`SavGol`], [`hampel_filter`],
//! [`real_roots`], [`Pca`], [`kmeans`], `running_stats`.
//!
//! # Grasping
//!
//! [`force_closure_q1`] (a differentiable Ferrari–Canny metric), [`force_closure_q1_spatial`] in the
//! full six-dimensional wrench space, [`grasp_matrix`], and `grasp_bounds` for how many fingers a hand
//! actually needs (Carathéodory, Steinitz, and the exceptional surfaces).
//!
//! # Hybrid systems, and making a claim checkable
//!
//! This is the part that separates a demo from a result, and it is why the crate exists in this shape.
//!
//! - **Impacts and orbital stability**: [`plastic_impact`], [`plastic_impact_jacobian`],
//!   [`saltation_matrix`] (which returns `None` on a grazing contact rather than a wrong number),
//!   [`poincare_stability`], [`impact_expansion`], [`hybrid_certificate`].
//! - **Certificates**: [`lyapunov`] and the LMI substrate, [`solve_sdp`].
//! - **Specification**: [`Stl`] — Signal Temporal Logic with quantitative robustness, so "the task
//!   succeeded" is a number with a sign rather than an opinion.
//! - **Partial observability**: [`Belief`], [`expected_information_gain`], [`best_sensing_action`] —
//!   deciding what to *look at*. A filter answers "where am I"; this answers "which measurement next".
//! - **Causal sufficiency**: [`Scm`] — why a model that predicts observational data perfectly can be
//!   wrong about what happens when you *act*.
//! - **Optimal transport**: [`w2_gaussian`], [`sinkhorn`], [`w1_empirical_1d`], [`kantorovich_dual`],
//!   [`gromov_wasserstein`], [`jko_step`], [`schrodinger_bridge`], [`distributional_bellman`] — the
//!   distance a closed loop actually charges for a policy's error, which is not total variation.
//! - **Thermodynamic floors**: [`landauer_energy`], [`entropy_production`],
//!   [`bode_sensitivity_integral`], [`max_extractable_work`], [`bits_affordable`] — what a physical
//!   agent cannot go below, in joules.
//! - **Score-to-distance bounds**: [`score_to_tv`] and [`flow_matching_w2`], which convert a network's
//!   training error into a distance on the distribution it induces.
//!
//! # Identification and sim-to-real
//!
//! [`identify`] (inertial parameters with reported uncertainty and a pseudo-inertia consistency
//! check), [`identify_actuator`] and [`identify_actuator_with_gain`], [`actuator_plausibility`] (spot
//! an impossible declared limit from the model alone), [`confounding`] (screen an excitation *before*
//! running it), and `randomization` for domain randomisation with ranges the data chose.
//!
//! # Throughput
//!
//! `gpu` (behind the `gpu` feature, which is why this is not a link) is the `wgpu` path: batched
//! collision checking and six batched articulated-dynamics
//! kernels, each stepping thousands of environments in one dispatch. `ArticulatedGpu` carries
//! **per-environment mass properties AND per-environment ground** — `set_env_inertia`,
//! `set_all_inertia` and `set_env_contact` — so the two axes a sim-to-real transfer actually
//! randomises, link mass and contact stiffness/friction, both vary across the batch without leaving
//! the GPU. Both are read back by `env_inertia_raw` and `env_contact` rather than assumed to land.
//! [`forward_dynamics_in`] is the allocation-free CPU path. Both are optional to the rest.
//!
//! # What this crate does not do
//!
//! It does not render, it does not own a scene graph, and it does not train — training lives in
//! `ferromotion-learn`, controllers in `ferromotion-control`, and the deformable and fluid domains in
//! their own crates. Nothing here reads a file path: every loader takes a string, which is what keeps
//! the wasm target honest.

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
mod contact_laws;
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
pub mod tree_diffik;
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
mod compound;
mod compound_pgs;
pub mod voxel;
pub mod acd;
mod link_geometry;
mod mesh_io;
mod modal;
mod occupancy;
mod numerics;
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
mod dh;
mod bug2;
mod distance_transform;
mod dstar_lite;
mod ets;
mod state_lattice;
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
mod mjcf_tree;
mod mujoco_contact;
mod usda;
mod urdf;
mod xcorr;
mod xpbd;
pub use aba::{floating_base_forward_dynamics, floating_base_forward_dynamics_ext, forward_dynamics_aba};
pub use tree_dynamics::{tree_floating_forward_dynamics, tree_floating_mass_matrix, tree_forward_dynamics, tree_inverse_dynamics, tree_mass_matrix};
pub use crba::crba;
pub use apriltag::{decode_payload, tag_pose};
pub use cfd_contact::{rollout_impulse, CfdContact};
pub use bit_star::BitStar;
pub use bvh::{Aabb, Bvh};
pub use mesh3::{convex_hull_3d, try_convex_hull_3d, TriMesh3};
pub use mesh_io::{from_obj, from_stl, from_stl_ascii, from_stl_binary, scale_mesh, second_moment, solid_inertia};
pub use link_geometry::{geometry_from_urdf, inertia_of_parts, primitive_link_inertia, primitive_mesh, resolve_uri, transform_mesh, GeomRole, GeometryRef, LinkGeometry};
pub use acd::{convex_decompose, AcdOptions, AcdReport};
pub use tree_diffik::{solve_tree_diffik, TreeDiffIkOptions, TreeDiffIkResult, TreeFrameTask};
pub use compound::{compound_contacts, compound_distance, CompoundContact, CompoundHull};
pub use voxel::SolidVoxels;
pub use compound_pgs::{compound_pgs_contacts, CompoundPgs, ContactJacobian, SerialLinkParts, StaticBody};
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
pub use contact_ipm::{central_path_scale, solve_frictional_ipm, FrictionalStep, StFrictionContact};
pub use sampler_bounds::{consistency_error, flow_matching_w2, minimax_rate, score_to_tv, SamplerError};
pub use transport_geometry::{cramer_distance, distributional_bellman, distributional_bellman_contraction, free_energy, gromov_cost, gromov_wasserstein, gromov_wasserstein_from, jko_step, kantorovich_dual, maximal_wasserstein_1, schrodinger_bridge, total_variation, GromovPlan};
pub use contact_pgs::{solve_contacts_pgs, solve_contacts_pgs_with, PgsContact, PgsResult, PgsStabilization};
pub use contact::{
    solve_contacts, solve_contacts_diff, solve_contacts_friction, Contact, ContactSolve,
    ContactSolveDiff, FrictionContact,
};
pub use diffik::{solve_diffik, DiffIkOptions, DiffIkResult, FrameTaskDef};
pub use dcol::{proximity, proximity_grad_spheres, Primitive};
pub use contact_laws::{contact_law_residuals, worst_contact_law_residual, ContactLawResidual};
pub use dynamics::{
    actuator_plausibility, forward_dynamics, gravity_vector, inverse_dynamics, mass_matrix, ActuatorReport,
    LinkInertia, COULOMB_SMOOTHING,
};
pub use grasp_spatial::{force_closure_q1_planar_subspace, force_closure_q1_spatial, force_closure_soft_spatial, grasp_matrix, grasp_split, net_wrench, primitive_wrenches_spatial, wrench_rank, GraspContact3, GraspSplit};
pub use grasp::{force_closure_q1, force_closure_soft, primitive_wrenches, GraspContact};
pub use great_circle::{cross_track_distance, destination, haversine_distance, initial_bearing, EARTH_RADIUS};
pub use grid_astar::{astar_grid, astar_grid_conn, can_step, manhattan, octile, path_length as grid_path_length, Connectivity};
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
pub use occupancy::{OccupancyGrid, UnknownCells};
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
pub use identification::{
    confounding, identify_actuator, identify_actuator_with_gain, identify_consistent,
    identify_with_covariance, params_from_pseudo_inertia, ActuatorFit, ActuatorGainFit, Confounding,
    ConsistentFit, CurrentSample, IdentifiedParams, PlannedMotion, ACTUATOR_PARAMETERS,
};
pub use hybrid_gradient::{hybrid_jacobian, HybridGradientOptions, probe_stable_jacobian, split_residual, HybridGradientError, HybridLinearisation, HybridSystem};
pub use hybrid::{compose_monodromy, find_limit_cycle, flow_jacobian, hybrid_certificate, impact_expansion, return_map_jacobian, plastic_impact, plastic_impact_jacobian, poincare_stability, saltation_matrix, transverse_basis, transverse_metric, transverse_restriction, HybridCertificate, HybridEvent};
pub use hybrid_astar::{hybrid_astar, HybridConfig};
pub use reeds_shepp::{path_length as rs_path_length, reeds_shepp, RsSegment};
pub use rotation_avg::{rotation_averaging, spanning_tree_init, RotEdge};
pub use running_stats::RunningStats;
pub use translation_avg::{translation_averaging, TransEdge};
pub use trilateration::trilaterate;
pub use radar_velocity::{ego_velocity_ls, ego_velocity_ransac};
pub use randomization::{check_actuator_support, ActuatorSupportCheck, AdrSchedule, Lcg, ParameterDistribution, ACTUATOR_RESIDUAL_TOLERANCE};
pub use ransac::{ransac, RansacResult};
pub use savgol::SavGol;
pub use lgvi::LgviBody;
pub use numerics::{finite_singular_values, finite_svd};
pub use manipulability::{condition_number, force_ellipsoid_axes, isotropy, manipulability_gradient, manipulability_gradient_analytic, singular_values, yoshikawa};
pub use modal::{modal_analysis, ModalModel};
pub use mjcf::{from_mjcf_constrained, from_mjcf_full, from_mjcf_str, geometry_from_mjcf, to_mjcf};
pub use mjcf_tree::{tree_from_mjcf, tree_from_mjcf_str, MjcfJoint, MjcfJointKind, MjcfTree};
pub use mujoco_contact::{mujoco_diag_approx, mujoco_impedance, mujoco_kbip, solve_contacts_mujoco, InvWeight, MjContact, MjContactSolve, SolImp, SolRef};
pub use usda::{axis_token, parse_usda, robot_from_usda, usda_from_robot, ParseError, Prim, UsdaStage, Value};
pub use urdf::from_urdf_full;
pub use robust::solve_ik_robust;
pub use bug2::{bug2, bug2_turn, default_step_cap, m_line, Turn};
pub use dh::{DhConvention, DhRow};
pub use distance_transform::{descend as distance_transform_descend, distance_transform, plan as distance_transform_plan};
pub use dstar_lite::DStarLite;
pub use ets::{Et, Eta, Ets};
pub use state_lattice::{arc_endpoint, edge_free as lattice_edge_free, lattice_astar, Lattice, LatticeDStarLite, LatticeNode, LatticePath, LatticeWeights, Primitive as LatticePrimitive, PrimitiveKind as LatticePrimitiveKind};
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
// The planar rank gate. Exported under a qualified name because the bare `wrench_rank` at this root is the
// SPATIAL variant (from `grasp_spatial`), which shipped first; renaming that would break published API. The
// planar function was `pub` inside a private module and therefore unreachable from outside the crate until
// now — a defect the rustdoc gate surfaced and the test suite could not.
pub use grasp::wrench_rank as planar_wrench_rank;
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
pub use sdformat::{from_sdf, geometry_from_sdf};
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
    /// Optional torque (or force, for a prismatic joint) the actuator can produce, from the URDF's
    /// `<limit effort=...>`.
    ///
    /// **This was parsed and thrown away until 2026-08-20.** `urdf_rs` reads it, both loaders read `lower` and
    /// `upper` from the same element, and nothing carried `effort` across — so a controller or a reinforcement
    /// learning action space had no way to ask what the servo can actually deliver and had to be given a number
    /// by hand. For a stack whose stated purpose is a 1:1 sim of target hardware, the hardware's own declared
    /// capability is the wrong thing to guess at.
    ///
    /// `None` means the model did not state one (or stated zero, which URDF uses for "unlimited"), and a caller
    /// that needs a bound must then supply it explicitly rather than receive a silent default.
    pub effort: Option<f64>,
    /// Optional maximum joint rate, from the URDF's `<limit velocity=...>`. Same history as [`Joint::effort`].
    pub max_velocity: Option<f64>,
    /// **The drive's own inertia reflected through its transmission**, added to this joint's diagonal of the
    /// joint-space inertia matrix. MuJoCo's `armature`; MJCF states it, URDF has no field for it.
    ///
    /// **The units follow the joint kind**, and the quantity is not the same thing in both. For a revolute
    /// joint it is a rotor inertia, `N²·J_rotor`, in kg·m². For a **prismatic** joint it is a reflected
    /// **mass** in kg — a leadscrew of lead `L` reflects `J_rotor·(2π/L)²`, so a `1e-5` kg·m² rotor on a 5 mm
    /// lead presents **15.8 kg**. Against a 2 kg carriage that is 7.9:1, the same "the drive dominates the
    /// load" story as the SO-101's wrist, in different units. The recursion applies the term after the
    /// revolute/prismatic branch, so both work; only the units differ.
    ///
    /// This is not a numerical fudge, it is a term of the plant. A geared servo's rotor accelerates with the
    /// joint and its apparent inertia scales as the *square* of the ratio, so on a small distal link it does
    /// not merely contribute to the joint-space inertia — it is the larger term. Measured on the SO-101,
    /// whose wrist link inertia is `3.45e-5` kg·m² against a reflected `1.19e-2`: a factor of **345**.
    ///
    /// What omitting it costs, over a 4×4 grid of PD gains and substep counts (`so101_reach_rl --sweep`):
    ///
    /// | | reaches a 1 cm target | best settle | electrical | control rate |
    /// |---|---|---|---|---|
    /// | without | **1 of 16** configurations | 0.0177 m | 13.8 J | 10 kHz |
    /// | with | **16 of 16** | 0.0001 m | 4.0 J | 200 Hz |
    ///
    /// So the plant is not unsolvable without it — it is **stiff**: 50x the integration rate, and at identical
    /// gains 4.2x the settling error and 3.5x the energy, with a working region that shrinks to one corner of
    /// the grid. [`crate::actuator_plausibility`] finds this from the model alone,
    /// without running a simulation step. An earlier
    /// version of this note claimed "unsolvable, 0 of 32", which was wrong; that sweep still contained a hard
    /// velocity clamp that was itself injecting energy, and it never tried a low gain at a high substep
    /// count.
    ///
    /// `None` behaves exactly as zero, so a model that does not state one is unaffected.
    pub armature: Option<f64>,
    /// Optional joint viscous damping, the passive resistance `b·q̇` opposing motion. URDF's
    /// `<dynamics damping=...>`, MJCF's `damping`, SDF's `<axis><dynamics><damping>`.
    ///
    /// N·m·s/rad on a revolute joint, **N·s/m on a prismatic one**.
    ///
    /// For a DC servo the dominant part of this is not friction but **back-EMF speed droop**: available
    /// torque falls linearly to zero at no-load speed, so `b = τ_stall / ω_0` follows from two catalogue
    /// numbers rather than being fitted. Modelling it this way also removes the need for a hard velocity
    /// clamp, which injects energy at every clamp event.
    ///
    /// `None` behaves exactly as zero.
    pub damping: Option<f64>,
    /// Optional **Coulomb friction** magnitude, the load-independent resistance a joint loses to its own
    /// bearings and gear teeth. URDF's `<dynamics friction=...>`, MJCF's `frictionloss`.
    ///
    /// N·m on a revolute joint, **N on a prismatic one**.
    ///
    /// On a high-reduction drive this is not a small correction. A 345:1 gear train is where friction is
    /// largest, and omitting it makes an energy figure optimistic in the one place the loss concentrates —
    /// which is why this was worth going back for after [`Joint::armature`] and [`Joint::damping`].
    ///
    /// Applied **smoothed**: `f·tanh(q̇/ε)` with `ε` = [`crate::COULOMB_SMOOTHING`], not `f·sign(q̇)`. True
    /// Coulomb friction is discontinuous at zero velocity, and a discontinuity in the RNEA output makes any
    /// integrator chatter and any derivative meaningless. The cost of smoothing is stated where the constant
    /// is: a joint dwelling below `ε` has its friction *underestimated*, approaching zero at rest, so this
    /// models a joint in motion and not stiction holding a pose.
    ///
    /// `None` behaves exactly as zero.
    pub friction: Option<f64>,
}

impl Joint {
    pub fn revolute(origin: Iso, axis: Vector3<f64>) -> Self {
        Self { origin, axis: Unit::new_normalize(axis), kind: JointKind::Revolute, limits: None, effort: None, max_velocity: None, armature: None, damping: None, friction: None }
    }

    pub fn prismatic(origin: Iso, axis: Vector3<f64>) -> Self {
        Self { origin, axis: Unit::new_normalize(axis), kind: JointKind::Prismatic, limits: None, effort: None, max_velocity: None, armature: None, damping: None, friction: None }
    }

    /// Attach the actuator's torque/force capability. A non-positive value is treated as "unstated", matching
    /// the URDF convention where `effort="0"` means unlimited rather than immovable.
    pub fn with_effort(mut self, effort: f64) -> Self {
        self.effort = if effort.is_finite() && effort > 0.0 { Some(effort) } else { None };
        self
    }

    /// Attach the joint's maximum rate. Same non-positive convention as [`with_effort`](Joint::with_effort).
    pub fn with_max_velocity(mut self, v: f64) -> Self {
        self.max_velocity = if v.is_finite() && v > 0.0 { Some(v) } else { None };
        self
    }

    /// Attach the reflected drive inertia — `N²·J_rotor` in kg·m² for a revolute joint, a reflected mass in
    /// kg for a prismatic one. Same non-positive convention as [`with_effort`](Joint::with_effort): a zero or
    /// negative value is "unstated", not "weightless rotor".
    pub fn with_armature(mut self, armature: f64) -> Self {
        self.armature = if armature.is_finite() && armature > 0.0 { Some(armature) } else { None };
        self
    }

    /// Attach the joint's viscous damping. Same non-positive convention as
    /// [`with_effort`](Joint::with_effort).
    pub fn with_damping(mut self, damping: f64) -> Self {
        self.damping = if damping.is_finite() && damping > 0.0 { Some(damping) } else { None };
        self
    }

    /// Attach the joint's Coulomb friction magnitude. Same non-positive convention as
    /// [`with_effort`](Joint::with_effort) — a friction of zero is "unstated", which is the honest reading:
    /// no real joint has exactly none.
    pub fn with_friction(mut self, friction: f64) -> Self {
        self.friction = if friction.is_finite() && friction > 0.0 { Some(friction) } else { None };
        self
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
    ///
    /// `q.len()` must equal [`Self::dof`]. The chain is composed by zipping joints with `q`, so a short
    /// `q` used to be truncated silently: a 7-joint arm evaluated with six values reported a pose with
    /// the last joint frozen and no error anywhere. That is checked in debug builds now, which is where
    /// tests run; release builds keep the zip so a hot path pays nothing.
    pub fn fk(&self, q: &[f64]) -> Iso {
        debug_assert_eq!(q.len(), self.dof(), "fk: q has {} entries for a {}-joint chain; a short q silently freezes the tail of the arm", q.len(), self.dof());
        let mut t = Iso::identity();
        for (j, &qi) in self.joints.iter().zip(q) {
            t *= j.transform(qi);
        }
        t * self.ee_offset
    }

    /// 6×N world-frame geometric Jacobian: dq → [linear; angular] end-effector velocity.
    pub fn jacobian(&self, q: &[f64]) -> DMatrix<f64> {
        let n = self.dof();
        debug_assert_eq!(q.len(), n, "jacobian: q has {} entries for a {}-joint chain", q.len(), n);
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

    /// The **kinematic Hessian** `∂J/∂q`: `dof` matrices, each `6 x dof`, where entry `j` is the
    /// derivative of the geometric Jacobian with respect to `q[j]`.
    ///
    /// This is the second-order kinematics, and it is what [`Self::jacobian_dot`],
    /// [`crate::manipulability_gradient_analytic`] and any Gauss-Newton or Newton IK need. Without it
    /// those all fall back to differencing [`Self::jacobian`], which costs `2·dof` Jacobians per
    /// gradient and loses roughly half the available digits.
    ///
    /// Derived from the same geometric construction the Jacobian uses. Moving `q[j]` transforms every
    /// frame downstream of joint `j` and always moves the end effector, so with `z` the joint axes and
    /// `p` the joint origins in world:
    ///
    /// - a **revolute** `j` gives `∂z_i = z_j × z_i` and `∂p_i = z_j × (p_i − p_j)` for `i > j`, zero
    ///   otherwise, and `∂p_e = z_j × (p_e − p_j)` always;
    /// - a **prismatic** `j` rotates nothing, so `∂z_i = 0`, with `∂p_i = z_j` for `i > j` and
    ///   `∂p_e = z_j`.
    ///
    /// Those propagate through `J_v,i = z_i × (p_e − p_i)`, `J_ω,i = z_i` for a revolute column and
    /// `J_v,i = z_i`, `J_ω,i = 0` for a prismatic one. Verified against central differences of
    /// [`Self::jacobian`], which is the only reason to trust the algebra above.
    pub fn kinematic_hessian(&self, q: &[f64]) -> Vec<DMatrix<f64>> {
        let n = self.dof();
        // one forward pass for the axes and origins, matching `jacobian`
        let p_e = self.fk(q).translation.vector;
        let mut z = Vec::with_capacity(n);
        let mut p = Vec::with_capacity(n);
        let mut t = Iso::identity();
        for (j, &qi) in self.joints.iter().zip(q) {
            let pre = t * j.origin;
            z.push(pre.rotation * j.axis.into_inner());
            p.push(pre.translation.vector);
            t = pre * j.motion(qi);
        }

        let mut out = Vec::with_capacity(n);
        for jj in 0..n {
            let mut h = DMatrix::zeros(6, n);
            let revolute_j = matches!(self.joints[jj].kind, JointKind::Revolute);
            let dp_e = if revolute_j { z[jj].cross(&(p_e - p[jj])) } else { z[jj] };
            for i in 0..n {
                let downstream = i > jj;
                let (dz_i, dp_i) = if !downstream {
                    (Vector3::zeros(), Vector3::zeros())
                } else if revolute_j {
                    (z[jj].cross(&z[i]), z[jj].cross(&(p[i] - p[jj])))
                } else {
                    (Vector3::zeros(), z[jj])
                };
                match self.joints[i].kind {
                    JointKind::Revolute => {
                        let lin = dz_i.cross(&(p_e - p[i])) + z[i].cross(&(dp_e - dp_i));
                        h.fixed_view_mut::<3, 1>(0, i).copy_from(&lin);
                        h.fixed_view_mut::<3, 1>(3, i).copy_from(&dz_i);
                    }
                    JointKind::Prismatic => {
                        h.fixed_view_mut::<3, 1>(0, i).copy_from(&dz_i);
                    }
                }
            }
            out.push(h);
        }
        out
    }

    /// `J̇ = Σ_j (∂J/∂q_j)·q̇_j`, the Jacobian rate along a joint velocity.
    ///
    /// The term operational-space control needs for its `J̇ q̇` bias, and second-order IK for its
    /// feed-forward. Exact, from [`Self::kinematic_hessian`], rather than differenced over a timestep.
    pub fn jacobian_dot(&self, q: &[f64], qd: &[f64]) -> Option<DMatrix<f64>> {
        let n = self.dof();
        if q.len() != n || qd.len() != n {
            return None;
        }
        let h = self.kinematic_hessian(q);
        let mut out = DMatrix::zeros(6, n);
        for (j, hj) in h.iter().enumerate() {
            out += hj * qd[j];
        }
        Some(out)
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
