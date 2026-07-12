//! Load a real, external, open-source robot — NormaCore's fully-3D-printed ElRobot (7-DoF,
//! Feetech ST3215 servos) — straight from its URDF and run ferromotion on it: FK, IK, and RNEA dynamics.
//! Usage: cargo run -p ferromotion-core --example elrobot -- path/to/elrobot_follower.urdf
use nalgebra::Vector3;
use ferromotion_core::{from_urdf_full, gravity_vector, mass_matrix, solve_ik, IkOptions};

fn main() {
    let path = std::env::args().nth(1).expect("pass the URDF path");
    let xml = std::fs::read_to_string(&path).expect("read URDF");

    let (robot, inertia) = from_urdf_full(&xml, "base_link", "Gripper_Base_v1_1").expect("load URDF");
    let n = robot.dof();
    println!("ElRobot loaded: {n} actuated DoF (base_link → Gripper_Base)");

    let home = vec![0.0; n];
    let tip = robot.fk(&home).translation.vector;
    println!("home tool position: [{:.3}, {:.3}, {:.3}] m", tip.x, tip.y, tip.z);

    // IK: reach a pose from a reference configuration.
    let qref: Vec<f64> = (0..n).map(|i| 0.3 * ((i as f64 + 1.0).sin())).collect();
    let target = robot.fk(&qref);
    let res = solve_ik(&robot, &target, &home, &IkOptions { max_iters: 300, ..IkOptions::default() });
    println!("IK to a reachable pose: converged={} residual={:.2e} iters={}", res.converged, res.error, res.iters);

    // Dynamics from the URDF inertials.
    let total_mass: f64 = inertia.iter().map(|l| l.mass).sum();
    println!("total moving mass (from URDF inertials): {total_mass:.3} kg");
    let g = gravity_vector(&robot, &inertia, &home, Vector3::new(0.0, 0.0, -9.81));
    let g_round: Vec<f64> = g.iter().map(|t| (t * 1000.0).round() / 1000.0).collect();
    println!("gravity-compensation torques at home (N·m): {g_round:?}");
    let m = mass_matrix(&robot, &inertia, &home);
    println!("mass matrix {}×{}, symmetric={}, PD={}", n, n,
        (m.clone() - m.transpose()).norm() < 1e-9, m.clone().cholesky().is_some());
}
