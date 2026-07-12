// End-to-end proof that the compiled WASM runs and computes correct IK in a JS runtime.
const ferromotion = require("../crates/ferromotion-wasm/pkg-node/ferromotion_wasm.js");

let failures = 0;
const near = (a, b, tol, msg) => {
  if (Math.abs(a - b) > tol) { console.error(`FAIL ${msg}: ${a} vs ${b}`); failures++; }
};

// 1) Programmatic planar 3R chain → IK.
const c = new ferromotion.Chain();
c.add_revolute(0, 0, 0, 1, 0, 0, 0, 0, 0, 1);
c.add_revolute(1, 0, 0, 1, 0, 0, 0, 0, 0, 1);
c.add_revolute(1, 0, 0, 1, 0, 0, 0, 0, 0, 1);
c.set_tool(1, 0, 0, 1, 0, 0, 0);
const qt = [0.3, -0.5, 0.4];
const target = c.fk(qt);
const out = c.solve_ik(target, [0, 0, 0]);
const err = out[out.length - 1];
console.log(`planar-3R: dof=${c.dof()} target_xyz=[${target.slice(0, 3).map(v => v.toFixed(3))}] ik_err=${err.toExponential(2)}`);
if (err > 1e-4) { console.error("FAIL: planar IK did not converge"); failures++; }

// 2) Load a real 6-DoF arm from URDF text, in-runtime.
const URDF = `<robot name="arm6"><link name="world"/><link name="base"/>
<link name="l1"/><link name="l2"/><link name="l3"/><link name="l4"/><link name="l5"/><link name="l6"/><link name="tool"/>
<joint name="j0" type="fixed"><parent link="world"/><child link="base"/><origin xyz="0 0 0.05" rpy="0 0 0"/></joint>
<joint name="j1" type="revolute"><parent link="base"/><child link="l1"/><origin xyz="0 0 0.1" rpy="0 0 0"/><axis xyz="0 0 1"/><limit lower="-3.14" upper="3.14" effort="10" velocity="3"/></joint>
<joint name="j2" type="revolute"><parent link="l1"/><child link="l2"/><origin xyz="0 0 0.2" rpy="0 0 0"/><axis xyz="0 1 0"/><limit lower="-3.14" upper="3.14" effort="10" velocity="3"/></joint>
<joint name="j3" type="revolute"><parent link="l2"/><child link="l3"/><origin xyz="0 0 0.2" rpy="0 0 0"/><axis xyz="0 1 0"/><limit lower="-3.14" upper="3.14" effort="10" velocity="3"/></joint>
<joint name="j4" type="revolute"><parent link="l3"/><child link="l4"/><origin xyz="0 0 0.1" rpy="0 0 0"/><axis xyz="0 1 0"/><limit lower="-3.14" upper="3.14" effort="10" velocity="3"/></joint>
<joint name="j5" type="revolute"><parent link="l4"/><child link="l5"/><origin xyz="0 0 0.1" rpy="0 0 0"/><axis xyz="0 0 1"/><limit lower="-3.14" upper="3.14" effort="10" velocity="3"/></joint>
<joint name="j6" type="revolute"><parent link="l5"/><child link="l6"/><origin xyz="0 0 0.05" rpy="0 0 0"/><axis xyz="0 1 0"/><limit lower="-3.14" upper="3.14" effort="10" velocity="3"/></joint>
<joint name="jtool" type="fixed"><parent link="l6"/><child link="tool"/><origin xyz="0 0 0.05" rpy="0 0 0"/></joint></robot>`;

const arm = ferromotion.Chain.from_urdf(URDF, "world", "tool");
const home = arm.fk([0, 0, 0, 0, 0, 0]);
console.log(`arm6: dof=${arm.dof()} home_xyz=[${home.slice(0, 3).map(v => v.toFixed(3))}]`);
near(arm.dof(), 6, 0, "arm dof");
near(home[2], 0.85, 1e-9, "arm home z");

const qTrue = [0.2, -0.6, 0.9, 0.4, 0.7, -0.3];
const tgt = arm.fk(qTrue);
const seed = qTrue.map(v => v + 0.15);
const sol = arm.solve_ik(tgt, seed);
const armErr = sol[sol.length - 1];
console.log(`arm6 IK: err=${armErr.toExponential(2)}`);
if (armErr > 1e-4) { console.error("FAIL: arm IK did not converge"); failures++; }

// 3) Motion retargeting: stream tool-tip keypoints from a known arm motion and retarget
//    frame-by-frame, warm-starting each step from the last (a live teleop loop). The tool point
//    (frame 6 + 5cm offset) comes straight from fk; the core's multi-frame math is unit-tested.
let prev = [0, 0, 0, 0, 0, 0];
let worst = 0;
for (let k = 0; k < 30; k++) {
  const s = k / 30;
  const qTru = [0, 1, 2, 3, 4, 5].map((i) => 0.5 * Math.sin(2 * Math.PI * s + i));
  const pose = arm.fk(qTru);
  const keypoint = new Float64Array([pose[0], pose[1], pose[2]]);
  const out = arm.retarget_step(
    new Uint32Array([6]), new Float64Array([0, 0, 0.05]), keypoint, prev, 0.005, 0.0);
  prev = out.slice(0, 6);
  const got = arm.fk(prev);
  worst = Math.max(worst, Math.hypot(got[0] - keypoint[0], got[1] - keypoint[1], got[2] - keypoint[2]));
}
console.log(`retarget stream: worst tool reproduction = ${worst.toExponential(2)} m`);
if (worst > 3e-3) { console.error("FAIL: retarget stream drifted"); failures++; }

// 4) plan_reach: a smooth trajectory to a goal, then the same reach routed around an obstacle.
{
  const dof = 3;
  const seed = [0.2, 0.3, 0.1];
  const goal = [1.2, 1.6, 0.0];
  const clear = c.plan_reach(new Float64Array(seed), new Float64Array(goal), new Float64Array([0, 0, 0, 0]), 20);
  const last = clear.slice((20 - 1) * dof, 20 * dof);
  const tip = c.fk(last);
  const reachErr = Math.hypot(tip[0] - goal[0], tip[1] - goal[1], tip[2] - goal[2]);
  console.log(`plan_reach: end-tool error = ${reachErr.toExponential(2)} m`);
  if (reachErr > 1e-2) { console.error("FAIL: plan_reach missed goal"); failures++; }

  const obs = [0.8, 0.9, 0.0, 0.3];
  const routed = c.plan_reach(new Float64Array(seed), new Float64Array(goal), new Float64Array(obs), 24);
  let minClear = Infinity;
  for (let k = 0; k < 24; k++) {
    const p = c.fk(routed.slice(k * dof, (k + 1) * dof));
    minClear = Math.min(minClear, Math.hypot(p[0] - obs[0], p[1] - obs[1], p[2] - obs[2]) - obs[3]);
  }
  console.log(`plan_reach + obstacle: min tool clearance = ${minClear.toFixed(3)} m`);
  if (minClear < -0.03) { console.error("FAIL: plan_reach penetrated obstacle"); failures++; }
}

console.log(failures === 0 ? "\n✅ WASM runtime: all checks passed" : `\n❌ ${failures} failure(s)`);
process.exit(failures === 0 ? 0 : 1);
