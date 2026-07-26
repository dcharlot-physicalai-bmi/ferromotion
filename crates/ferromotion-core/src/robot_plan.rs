//! **Collision-free reach planning for a [`Robot`]** — the bridge that puts the sampling planners to
//! work on an actual arm. The arm is approximated at each configuration by spheres swept along its
//! links (from forward kinematics); [`arm_clearance`] measures their signed distance to an
//! [`SdfScene`] of obstacles, and [`plan_arm_reach`] hands that collision test to [`RrtStar`] to find
//! a joint-space path from a start configuration to a goal that never drives the arm through an
//! obstacle. This is what turns "reach straight to the IK solution" into "reach *around* the wall".

use crate::{Robot, RrtStar, SdfScene};
use nalgebra::Vector3;

/// Collision spheres approximating the arm at configuration `q`: `per_link` centres spread along each
/// link (between consecutive frame origins), plus the tool point. Check each against an obstacle field.
pub fn arm_spheres(robot: &Robot, q: &[f64], per_link: usize) -> Vec<Vector3<f64>> {
    let dof = robot.dof();
    let frames: Vec<Vector3<f64>> = (0..=dof).map(|i| robot.frame_pose(q, i).translation.vector).collect();
    let mut pts = Vec::with_capacity(dof * per_link + 1);
    for i in 0..dof {
        let (a, b) = (frames[i], frames[i + 1]);
        for s in 0..per_link {
            let t = (s as f64 + 0.5) / per_link as f64;
            pts.push(a + t * (b - a));
        }
    }
    pts.push(robot.fk(q).translation.vector);
    pts
}

/// Minimum clearance of the arm (spheres of radius `r`) to `scene` at config `q`. Negative = the arm
/// is in collision.
pub fn arm_clearance(robot: &Robot, q: &[f64], scene: &SdfScene, r: f64, per_link: usize) -> f64 {
    arm_spheres(robot, q, per_link).iter().map(|c| scene.distance(c) - r).fold(f64::INFINITY, f64::min)
}

/// Options for [`plan_arm_reach`].
#[derive(Clone, Debug)]
pub struct ReachPlanOptions {
    /// Sphere radius approximating each link's thickness.
    pub link_radius: f64,
    /// Required clearance beyond the link radius (safety margin).
    pub margin: f64,
    /// Spheres per link for the swept-collision check.
    pub per_link: usize,
    /// Configuration-space resolution for edge checking.
    pub edge_res: f64,
    pub max_iters: usize,
    pub seed: u64,
}

impl Default for ReachPlanOptions {
    fn default() -> Self {
        ReachPlanOptions { link_radius: 0.03, margin: 0.005, per_link: 3, edge_res: 0.06, max_iters: 8000, seed: 1 }
    }
}

/// Plan a collision-free joint path from `q_start` to `q_goal` (within box joint bounds `lo`/`hi`)
/// that keeps the whole arm clear of `scene`, via [`RrtStar`]. Returns the path (a sequence of joint
/// configurations, `q_start` … `q_goal`) or `None` if no path was found within the iteration budget.
#[allow(clippy::too_many_arguments)]
pub fn plan_arm_reach(
    robot: &Robot,
    q_start: &[f64],
    q_goal: &[f64],
    scene: &SdfScene,
    lo: &[f64],
    hi: &[f64],
    opts: &ReachPlanOptions,
) -> Option<Vec<Vec<f64>>> {
    let edge_free = |a: &[f64], b: &[f64]| {
        let d = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f64>().sqrt();
        let n = (d / opts.edge_res).ceil().max(1.0) as usize;
        for k in 0..=n {
            let t = k as f64 / n as f64;
            let q: Vec<f64> = (0..a.len()).map(|i| a[i] + t * (b[i] - a[i])).collect();
            if arm_clearance(robot, &q, scene, opts.link_radius, opts.per_link) <= opts.margin {
                return false;
            }
        }
        true
    };
    let planner = RrtStar { dim: robot.dof(), step: 0.4, goal_bias: 0.2, gamma: 2.5, max_iters: opts.max_iters, seed: opts.seed };
    planner.plan(q_start, q_goal, lo, hi, edge_free).map(|r| r.path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{from_urdf_str, solve_diffik, DiffIkOptions, FrameTaskDef};
    use nalgebra::Point3;

    const ARM: &str = r#"<robot name="a"><link name="world"/><link name="base"/>
      <link name="l1"/><link name="l2"/><link name="l3"/><link name="l4"/><link name="l5"/><link name="l6"/><link name="tool"/>
      <joint name="j0" type="fixed"><parent link="world"/><child link="base"/><origin xyz="0 0 0.05" rpy="0 0 0"/></joint>
      <joint name="j1" type="revolute"><parent link="base"/><child link="l1"/><origin xyz="0 0 0.1" rpy="0 0 0"/><axis xyz="0 0 1"/><limit lower="-3.14" upper="3.14" effort="10" velocity="3"/></joint>
      <joint name="j2" type="revolute"><parent link="l1"/><child link="l2"/><origin xyz="0 0 0.2" rpy="0 0 0"/><axis xyz="0 1 0"/><limit lower="-3.14" upper="3.14" effort="10" velocity="3"/></joint>
      <joint name="j3" type="revolute"><parent link="l2"/><child link="l3"/><origin xyz="0 0 0.2" rpy="0 0 0"/><axis xyz="0 1 0"/><limit lower="-3.14" upper="3.14" effort="10" velocity="3"/></joint>
      <joint name="j4" type="revolute"><parent link="l3"/><child link="l4"/><origin xyz="0 0 0.1" rpy="0 0 0"/><axis xyz="0 1 0"/><limit lower="-3.14" upper="3.14" effort="10" velocity="3"/></joint>
      <joint name="j5" type="revolute"><parent link="l4"/><child link="l5"/><origin xyz="0 0 0.1" rpy="0 0 0"/><axis xyz="0 0 1"/><limit lower="-3.14" upper="3.14" effort="10" velocity="3"/></joint>
      <joint name="j6" type="revolute"><parent link="l5"/><child link="l6"/><origin xyz="0 0 0.05" rpy="0 0 0"/><axis xyz="0 1 0"/><limit lower="-3.14" upper="3.14" effort="10" velocity="3"/></joint>
      <joint name="jt" type="fixed"><parent link="l6"/><child link="tool"/><origin xyz="0 0 0.05" rpy="0 0 0"/></joint></robot>"#;

    fn path_min_clearance(robot: &Robot, path: &[Vec<f64>], scene: &SdfScene, o: &ReachPlanOptions) -> f64 {
        let mut worst = f64::INFINITY;
        for w in path.windows(2) {
            let d = w[0].iter().zip(&w[1]).map(|(x, y)| (x - y) * (x - y)).sum::<f64>().sqrt();
            let n = (d / o.edge_res).ceil().max(1.0) as usize;
            for k in 0..=n {
                let t = k as f64 / n as f64;
                let q: Vec<f64> = (0..w[0].len()).map(|i| w[0][i] + t * (w[1][i] - w[0][i])).collect();
                worst = worst.min(arm_clearance(robot, &q, scene, o.link_radius, o.per_link));
            }
        }
        worst
    }

    /// **Obstacle-avoidance oracle.** A wall stands between the arm and a target behind it. The naive
    /// straight-line joint interpolation drives the arm *through* the wall; the planned path keeps the
    /// whole arm clear and still arrives at the goal configuration.
    #[test]
    fn plans_a_collision_free_path_around_a_wall() {
        let robot = from_urdf_str(ARM, "world", "tool").unwrap();
        let dof = robot.dof();
        let q0 = vec![0.0; dof];

        // target in front and high enough to drape over a low wall; solve IK to it
        let target = Vector3::new(0.42, 0.0, 0.42);
        let tasks = [FrameTaskDef::new(dof, Vector3::new(0.0, 0.0, 0.05), target, 2.0, 1.0)];
        let q_goal = solve_diffik(&robot, &tasks, &q0, &DiffIkOptions::default()).q;
        let tip = (robot.fk(&q_goal)).translation.vector;
        assert!((tip - target).norm() < 0.03, "IK did not reach the target: {:?}", tip);

        // place the wall where the naive straight-line joint interpolation passes: at the tool's
        // midpoint config. The straight path then drives the tool through the wall, while the
        // endpoints (tool far from there) stay clear — a problem that genuinely needs a detour.
        let q_mid: Vec<f64> = (0..dof).map(|i| 0.5 * (q0[i] + q_goal[i])).collect();
        let tool_mid = robot.fk(&q_mid).translation.vector;
        let wall = SdfScene { prims: vec![crate::Sdf::Box { center: tool_mid, half: Vector3::new(0.06, 0.16, 0.11) }] };
        let opts = ReachPlanOptions { max_iters: 12000, ..Default::default() };
        // bound the search to the region around start and goal (RRT is far more sample-efficient
        // than over the full joint range)
        let pad = 0.7;
        let lo: Vec<f64> = (0..dof).map(|i| q0[i].min(q_goal[i]) - pad).collect();
        let hi: Vec<f64> = (0..dof).map(|i| q0[i].max(q_goal[i]) + pad).collect();

        let c0 = arm_clearance(&robot, &q0, &wall, opts.link_radius, opts.per_link);
        let cg = arm_clearance(&robot, &q_goal, &wall, opts.link_radius, opts.per_link);
        eprintln!("reach-plan: start clearance {c0:.3} m, goal clearance {cg:.3} m");
        // both endpoints must be collision-free for planning to make sense
        assert!(c0 > opts.margin, "start config is in collision: {c0:.3}");
        assert!(cg > opts.margin, "goal config is in collision: {cg:.3}");

        // naive straight interpolation collides with the wall
        let naive: Vec<Vec<f64>> = vec![q0.clone(), q_goal.clone()];
        let naive_clear = path_min_clearance(&robot, &naive, &wall, &opts);
        eprintln!("reach-plan: naive straight-path min clearance {naive_clear:.3} m (negative ⇒ hits the wall)");
        assert!(naive_clear < 0.0, "the wall should block the straight path (min clearance {naive_clear:.3})");

        // the planner finds a path that clears the wall and reaches the goal
        let path = plan_arm_reach(&robot, &q0, &q_goal, &wall, &lo, &hi, &opts).expect("no plan found");
        let planned_clear = path_min_clearance(&robot, &path, &wall, &opts);
        let goal_err: f64 = path.last().unwrap().iter().zip(&q_goal).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        eprintln!("reach-plan: {} waypoints, planned min clearance {planned_clear:.3} m, goal err {goal_err:.3}", path.len());
        assert!(planned_clear > 0.0, "planned path is not collision-free: {planned_clear:.3}");
        assert!(goal_err < 0.5, "planned path does not reach the goal config: {goal_err:.3}");
    }
}
