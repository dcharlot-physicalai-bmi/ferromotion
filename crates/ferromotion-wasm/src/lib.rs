//! ferromotion-wasm — universal WebAssembly bindings for ferromotion.
//!
//! Build a kinematic chain, then solve IK to a target pose — entirely in the browser,
//! no install, no server. This is the "universal" half of the initiative: the same Rust
//! kinematics core the native tools use, compiled to WASM for the Institute's on-device labs.
//!
//! JS usage (after `wasm-pack build`):
//! ```js
//! const r = new Chain();
//! r.add_revolute(0,0,0,  1,0,0,0,  0,0,1);   // origin xyz, origin quat wxyz, axis xyz
//! r.add_revolute(1,0,0,  1,0,0,0,  0,0,1);
//! r.add_revolute(1,0,0,  1,0,0,0,  0,0,1);
//! r.set_tool(1,0,0,  1,0,0,0);
//! const q = r.solve_ik([1.5,1.0,0, 1,0,0,0], [0,0,0]); // target pos+quat, seed
//! ```

use nalgebra::{Isometry3, Translation3, UnitQuaternion, Vector3};
use ferromotion_core::{
    solve, solve_ik, Cost, FrameTask, IkOptions, Iso, Joint, PointCost, PostureCost, Retargeter,
    Robot, SolveOptions, SphereCollisionCost, TrajectoryProblem,
};
use wasm_bindgen::prelude::*;

mod muscle_lab;
pub use muscle_lab::{DelayedRig, MuscleRig};

mod consensus_lab;
pub use consensus_lab::ConsensusLab;

mod cbf_lab;
pub use cbf_lab::CbfLab;

mod grasp_lab;
pub use grasp_lab::GraspLab;

mod dmp_lab;
pub use dmp_lab::DmpLab;

mod inekf_lab;
pub use inekf_lab::InekfLab;

mod topp_lab;
pub use topp_lab::ToppLab;

mod walk_lab;
pub use walk_lab::WalkLab;

mod koopman_lab;
pub use koopman_lab::KoopmanLab;

mod rmp_lab;
pub use rmp_lab::RmpLab;

fn iso(px: f64, py: f64, pz: f64, qw: f64, qx: f64, qy: f64, qz: f64) -> Iso {
    let q = UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(qw, qx, qy, qz));
    Isometry3::from_parts(Translation3::new(px, py, pz), q)
}

/// A serial kinematic chain you build joint-by-joint from JavaScript.
#[wasm_bindgen]
pub struct Chain {
    joints: Vec<Joint>,
    tool: Iso,
}

#[wasm_bindgen]
impl Chain {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Chain {
        Chain { joints: Vec::new(), tool: Iso::identity() }
    }

    /// Load a real robot from URDF text (fetch it over HTTP, pass the string), following the
    /// tree path from `base` link to `tip` link.
    pub fn from_urdf(xml: &str, base: &str, tip: &str) -> Result<Chain, JsError> {
        let robot = ferromotion_core::from_urdf_str(xml, base, tip).map_err(|e| JsError::new(&e))?;
        Ok(Chain { joints: robot.joints, tool: robot.ee_offset })
    }

    /// Append a revolute joint. `origin` = parent→joint transform (xyz + quaternion wxyz); `axis` = xyz.
    pub fn add_revolute(
        &mut self,
        px: f64, py: f64, pz: f64,
        qw: f64, qx: f64, qy: f64, qz: f64,
        ax: f64, ay: f64, az: f64,
    ) {
        self.joints.push(Joint::revolute(iso(px, py, pz, qw, qx, qy, qz), Vector3::new(ax, ay, az)));
    }

    /// Append a prismatic joint.
    pub fn add_prismatic(
        &mut self,
        px: f64, py: f64, pz: f64,
        qw: f64, qx: f64, qy: f64, qz: f64,
        ax: f64, ay: f64, az: f64,
    ) {
        self.joints.push(Joint::prismatic(iso(px, py, pz, qw, qx, qy, qz), Vector3::new(ax, ay, az)));
    }

    /// Fixed tool (end-effector) transform relative to the last joint.
    pub fn set_tool(&mut self, px: f64, py: f64, pz: f64, qw: f64, qx: f64, qy: f64, qz: f64) {
        self.tool = iso(px, py, pz, qw, qx, qy, qz);
    }

    pub fn dof(&self) -> usize {
        self.joints.len()
    }

    fn robot(&self) -> Robot {
        Robot { joints: self.joints.clone(), ee_offset: self.tool }
    }

    /// Forward kinematics: returns the tool pose as `[x,y,z, qw,qx,qy,qz]`.
    pub fn fk(&self, q: &[f64]) -> Vec<f64> {
        let t = self.robot().fk(q);
        let p = t.translation.vector;
        let r = t.rotation;
        vec![p.x, p.y, p.z, r.w, r.i, r.j, r.k]
    }

    /// Solve IK to `target` (`[x,y,z, qw,qx,qy,qz]`) from `seed`; returns the joint vector.
    /// The last returned element is the final residual norm (so callers can check the solve).
    pub fn solve_ik(&self, target: &[f64], seed: &[f64]) -> Vec<f64> {
        let tgt = iso(target[0], target[1], target[2], target[3], target[4], target[5], target[6]);
        let res = solve_ik(&self.robot(), &tgt, seed, &IkOptions::default());
        let mut out = res.q;
        out.push(res.error);
        out
    }

    /// One motion-retargeting step (call once per observed frame; warm-start `seed` with the
    /// previous result for a live teleop/mocap loop). `frames[i]` picks the robot frame,
    /// `offsets` is `3·k` (point in that frame), `targets` is `3·k` (observed world keypoints).
    /// Returns `[q…, residual]`.
    pub fn retarget_step(
        &self,
        frames: &[u32],
        offsets: &[f64],
        targets: &[f64],
        seed: &[f64],
        smoothness: f64,
        limit_weight: f64,
    ) -> Vec<f64> {
        let k = frames.len();
        let tasks: Vec<FrameTask> = (0..k)
            .map(|i| {
                FrameTask::new(
                    frames[i] as usize,
                    Vector3::new(offsets[3 * i], offsets[3 * i + 1], offsets[3 * i + 2]),
                    1.0,
                )
            })
            .collect();
        let tgts: Vec<Vector3<f64>> = (0..k)
            .map(|i| Vector3::new(targets[3 * i], targets[3 * i + 1], targets[3 * i + 2]))
            .collect();
        let mut rt = Retargeter::new(tasks);
        rt.smoothness = smoothness;
        rt.limit_weight = limit_weight;
        let res = rt.solve_frame(&self.robot(), &tgts, seed);
        let mut out = res.q;
        out.push(res.error);
        out
    }

    /// Plan a smooth trajectory from `seed` so the tool tip reaches `goal` (xyz), optionally routing
    /// around a sphere obstacle `[x,y,z,radius]` (radius ≤ 0 disables it). Returns `steps · dof`
    /// joint values, row-major by timestep — the whole motion, ready to animate.
    pub fn plan_reach(&self, seed: &[f64], goal: &[f64], obstacle: &[f64], steps: usize) -> Vec<f64> {
        let robot = self.robot();
        let n = robot.dof();
        let dof = n; // PointCost frame index = end of chain (tool offset applied via `tip`)
        let tip = self.tool.translation.vector;
        let goal_v = Vector3::new(goal[0], goal[1], goal[2]);

        // Warm start: solve position IK for a goal configuration, interpolate from the seed.
        let goal_costs: Vec<Box<dyn Cost>> = vec![Box::new(PointCost::new(dof, tip, goal_v, 1.0))];
        let goal_cfg = solve(&robot, &goal_costs, seed, &SolveOptions::default()).q;

        let mut costs: Vec<Vec<Box<dyn Cost>>> = (0..steps).map(|_| Vec::new()).collect();
        costs[0].push(Box::new(PostureCost::new(seed.to_vec(), 100.0)));
        costs[steps - 1].push(Box::new(PointCost::new(dof, tip, goal_v, 50.0)));
        if obstacle.len() >= 4 && obstacle[3] > 0.0 {
            let oc = Vector3::new(obstacle[0], obstacle[1], obstacle[2]);
            for c in costs.iter_mut() {
                c.push(Box::new(SphereCollisionCost::new(dof, tip, 0.0, oc, obstacle[3], 0.02, 50.0)));
            }
        }

        let prob = TrajectoryProblem {
            robot: &robot,
            costs,
            vel_weight: 0.3,
            opts: SolveOptions { max_iters: 400, ..SolveOptions::default() },
        };
        let init: Vec<Vec<f64>> = (0..steps)
            .map(|k| {
                let s = if steps > 1 { k as f64 / (steps - 1) as f64 } else { 0.0 };
                (0..n).map(|i| seed[i] * (1.0 - s) + goal_cfg[i] * s).collect()
            })
            .collect();

        let res = prob.solve(&init);
        let mut out = Vec::with_capacity(steps * n);
        for q in res.qs {
            out.extend_from_slice(&q);
        }
        out
    }
}

impl Default for Chain {
    fn default() -> Self {
        Self::new()
    }
}
