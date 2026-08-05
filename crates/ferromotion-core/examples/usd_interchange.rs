//! **URDF to OpenUSD and back, checked on the dynamics rather than the structure.**
//!
//! A format converter that round-trips its own field names proves very little. The question that matters is whether a
//! robot handed through the format computes the same physics, so this runs the articulated-body algorithm on both sides
//! and compares joint accelerations.
//!
//! It then prices each of the three conventions the reader handles, by dropping one at a time and measuring what the
//! dynamics answer does. A convention that costs nothing does not need handling; these are not those.

use ferromotion_core::{forward_dynamics_aba, from_urdf_full, parse_usda, robot_from_usda, usda_from_robot, LinkInertia, Robot};
use nalgebra::Vector3;

const URDF: &str = r#"<robot name="arm">
  <link name="l0"/>
  <link name="l1"><inertial><mass value="2.0"/><origin xyz="0 0 0.15"/>
    <inertia ixx="0.03" iyy="0.03" izz="0.008" ixy="0" ixz="0" iyz="0"/></inertial></link>
  <link name="l2"><inertial><mass value="1.3"/><origin xyz="0 0 0.1"/>
    <inertia ixx="0.02" iyy="0.02" izz="0.005" ixy="0" ixz="0" iyz="0"/></inertial></link>
  <link name="l3"><inertial><mass value="0.7"/><origin xyz="0 0 0.05"/>
    <inertia ixx="0.01" iyy="0.01" izz="0.003" ixy="0" ixz="0" iyz="0"/></inertial></link>
  <joint name="j1" type="revolute"><parent link="l0"/><child link="l1"/><origin xyz="0 0 0.05"/>
    <axis xyz="0 0 1"/><limit lower="-1.5708" upper="1.5708" effort="20" velocity="3"/></joint>
  <joint name="j2" type="revolute"><parent link="l1"/><child link="l2"/><origin xyz="0 0 0.3"/>
    <axis xyz="0 1 0"/><limit lower="-2.0" upper="2.0" effort="20" velocity="3"/></joint>
  <joint name="j3" type="prismatic"><parent link="l2"/><child link="l3"/><origin xyz="0 0 0.2"/>
    <axis xyz="1 0 0"/><limit lower="-0.4" upper="0.4" effort="20" velocity="3"/></joint>
</robot>"#;

const Q: [f64; 3] = [0.31, -0.62, 0.08];
const QD: [f64; 3] = [0.12, 0.44, -0.21];
const TAU: [f64; 3] = [0.8, -0.5, 0.3];

fn accel(robot: &Robot, inertia: &[LinkInertia]) -> Vec<f64> {
    forward_dynamics_aba(robot, inertia, &Q, &QD, &TAU, Vector3::new(0.0, 0.0, -9.81))
}

fn worst(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
}

fn main() {
    let (urdf_robot, urdf_inertia) = from_urdf_full(URDF, "l0", "l3").expect("urdf parses");
    let reference = accel(&urdf_robot, &urdf_inertia);

    println!("URDF -> OpenUSD -> ferromotion, compared on joint accelerations");
    println!("  3-dof arm (revolute, revolute, prismatic), q = {Q:?}, qd = {QD:?}, tau = {TAU:?}");
    println!("  reference qdd from the URDF path: {reference:?}\n");

    // --- the round trip
    let stage_text = usda_from_robot(&urdf_robot, &urdf_inertia, "arm");
    let stage = parse_usda(&stage_text).expect("what we emit, we can read");
    let (usd_robot, usd_inertia) = robot_from_usda(&stage, "link0", "link3").expect("the chain resolves");
    let through_usd = accel(&usd_robot, &usd_inertia);

    println!("  qdd through the USD path:         {through_usd:?}");
    println!("  worst difference: {:.3e} rad/s^2\n", worst(&reference, &through_usd));
    println!("  stage: {} bytes, {} prims, upAxis {}, metersPerUnit {}", stage_text.len(), stage.walk().len(), stage.up_axis, stage.meters_per_unit);
    println!("  joint limits survive the radians -> degrees -> radians trip:");
    for (i, (a, b)) in urdf_robot.joints.iter().zip(&usd_robot.joints).enumerate() {
        match (a.limits, b.limits) {
            (Some(x), Some(y)) => println!("    j{}: ({:+.4}, {:+.4}) -> ({:+.4}, {:+.4})  delta {:.2e}", i + 1, x.0, x.1, y.0, y.1, (x.1 - y.1).abs()),
            _ => println!("    j{}: no limits", i + 1),
        }
    }

    // --- what each convention is worth, priced in the dynamics answer
    println!("\n  what dropping each convention costs, measured on the same qdd:");
    println!("    {:<44} {:>16}  {:>14}", "convention dropped", "worst qdd error", "relative");
    let scale_of = |r: &[f64]| r.iter().fold(0.0f64, |a, b| a.max(b.abs())).max(1e-12);

    // 1. metersPerUnit: read centimetre content as if it were metres
    let cm_text = stage_text.replace("metersPerUnit = 1", "metersPerUnit = 0.01");
    let cm_stage = parse_usda(&cm_text).unwrap();
    let (cm_robot, cm_inertia) = robot_from_usda(&cm_stage, "link0", "link3").unwrap();
    let honest = accel(&cm_robot, &cm_inertia);
    // and the same stage read by a reader that ignores the field
    let mut naive_stage = cm_stage.clone();
    naive_stage.meters_per_unit = 1.0;
    let (naive_robot, naive_inertia) = robot_from_usda(&naive_stage, "link0", "link3").unwrap();
    let naive = accel(&naive_robot, &naive_inertia);
    let e = worst(&honest, &naive);
    println!("    {:<44} {e:>16.4e}  {:>13.2}", "metersPerUnit (cm content read as m)", e / scale_of(&honest));
    println!("      {e:.1} rad/s^2 of error against a correct answer whose largest component is {:.1} - the relative", scale_of(&honest));
    println!("      figure near 1.0 means the error is the whole signal, not a perturbation of it.");

    // 2. revolute limits in degrees, read as radians
    let deg = usd_robot.joints[0].limits.unwrap().1;
    let raw = stage.prim("/arm/joint1").and_then(|p| p.attr("physics:upperLimit")).and_then(|v| v.as_number()).unwrap();
    println!("    {:<44} {:>16.4} {:>14}", "revolute limit (degrees read as radians)", raw - deg, "rad of slack");
    println!("      the stage says {raw} and the joint's real limit is {deg:.4} rad; a reader that skips the");
    println!("      conversion produces a {raw} rad limit, which is {:.0}x the true range and stops being a limit", raw / deg);

    // 3. upAxis: a Y-up stage read as Z-up
    let y_up = parse_usda(&stage_text.replace("upAxis = \"Z\"", "upAxis = \"Y\"")).unwrap();
    let g_in_stage = y_up.to_z_up().inverse() * Vector3::new(0.0, 0.0, -9.81);
    println!("    {:<44}", "upAxis (Y-up stage read as Z-up)");
    println!("      gravity in a Y-up stage points along ({:+.2}, {:+.2}, {:+.2}); read as Z-up the robot", g_in_stage.x, g_in_stage.y, g_in_stage.z);
    println!("      is rotated 90 degrees and every gravity torque is wrong. Nothing about it fails loudly.");

    println!("\n  The round trip agrees to {:.1e}, so the format carries the physics. The three convention", worst(&reference, &through_usd));
    println!("  errors do not fail loudly, which is why each one has a test that fails if the conversion is dropped.");

    println!("\n--- emitted stage ---\n{stage_text}");
}
