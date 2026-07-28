//! Load the real, official SO-101 (LeRobot / TheRobotStudio, CAD-derived, STS3215 servos) into
//! ferromotion — the trusted, verify-first 1:1 sim of the target hardware — and validate its dynamics:
//! FK, IK, mass matrix (SPD), gravity torques, and a torque-free gravity rollout (finite/stable).
//! URDF: TheRobotStudio/SO-ARM100 Simulation/SO101/so101_new_calib.urdf. This is step one of the
//! digital twin: the model must live in our engine and pass physics checks before any arbiter runs on it.
use ferromotion_core::{forward_dynamics, from_urdf_full, gravity_vector, mass_matrix, solve_ik, IkOptions};
use nalgebra::Vector3;

const URDF: &str = include_str!("so101.urdf");
const G: f64 = -9.81;

fn main() {
    let (robot, inertia) = from_urdf_full(URDF, "base_link", "gripper_link").expect("load SO-101 URDF");
    let n = robot.dof();
    println!("SO-101 loaded into ferromotion: {n} actuated DoF (base_link → gripper_link)");

    let home = vec![0.0; n];
    let tip = robot.fk(&home).translation.vector;
    println!("home tool position: [{:.3}, {:.3}, {:.3}] m", tip.x, tip.y, tip.z);

    // IK: reach a self-generated reachable pose from home.
    let qref: Vec<f64> = (0..n).map(|i| 0.4 * ((i as f64 + 1.0).sin())).collect();
    let target = robot.fk(&qref);
    let res = solve_ik(&robot, &target, &home, &IkOptions { max_iters: 300, ..IkOptions::default() });
    println!("IK to a reachable pose: converged={} residual={:.2e} iters={}", res.converged, res.error, res.iters);

    // Dynamics from the URDF inertials.
    let total_mass: f64 = inertia.iter().map(|l| l.mass).sum();
    println!("total moving mass (from URDF inertials): {total_mass:.3} kg");
    let gv = gravity_vector(&robot, &inertia, &home, Vector3::new(0.0, 0.0, G));
    let gr: Vec<f64> = gv.iter().map(|t| (t * 1000.0).round() / 1000.0).collect();
    println!("gravity-compensation torques at home (N·m): {gr:?}");
    let m = mass_matrix(&robot, &inertia, &home);
    let sym = (m.clone() - m.transpose()).norm() < 1e-9;
    let pd = m.clone().cholesky().is_some();
    println!("mass matrix {n}×{n}: symmetric={sym} PD={pd}");

    // Torque-free gravity rollout — the arm should swing under gravity and stay finite (no NaN/blow-up).
    let (mut q, mut qd) = (vec![0.3; n], vec![0.0; n]);
    let dt = 1e-3;
    for _ in 0..1000 {
        let tau = vec![0.0; n];
        let qdd = forward_dynamics(&robot, &inertia, &q, &qd, &tau, Vector3::new(0.0, 0.0, G));
        for i in 0..n {
            qd[i] += dt * qdd[i];
            q[i] += dt * qd[i];
        }
    }
    let finite = q.iter().chain(qd.iter()).all(|x| x.is_finite());
    println!("1000-step torque-free gravity rollout finite/stable: {finite}");

    println!("\n{}", if n == 5 && sym && pd && res.converged && finite {
        "PASS — the real SO-101 is now in the trusted sim, dynamics validated ✓ (the twin's foundation)"
    } else {
        "CHECK — one of {dof=5, SPD, IK, finite-rollout} did not hold; see above"
    });
}
