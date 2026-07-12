//! Robust IK, TRAC-IK style. Plain Levenberg–Marquardt IK can stall in a local minimum or return a
//! solution that violates joint limits. TRAC-IK's fix is to run several attempts from different
//! seeds and take the first valid one. We reimplement that method in pure Rust (the `optik` crate
//! ports TRAC-IK but on an NLopt C backend that can't compile to WASM — this keeps ferromotion universal).

use crate::{solve_ik, IkOptions, IkResult, Iso, Robot};

fn within_limits(robot: &Robot, q: &[f64]) -> bool {
    robot.joints.iter().zip(q).all(|(j, &qi)| match j.limits {
        Some((lo, hi)) => qi >= lo - 1e-6 && qi <= hi + 1e-6,
        None => true,
    })
}

/// Solve IK to `target` with up to `restarts` seeded attempts. The first attempt uses the zero
/// configuration; the rest are quasi-random within joint limits (deterministic LCG, so results are
/// reproducible). Returns the first attempt that converges *and* respects joint limits, else the
/// lowest-error attempt seen.
pub fn solve_ik_robust(robot: &Robot, target: &Iso, opts: &IkOptions, restarts: usize) -> IkResult {
    let n = robot.dof();
    let mut best: Option<IkResult> = None;
    let mut lcg: u64 = 0x2545_F491_4F6C_DD1D;

    for r in 0..restarts.max(1) {
        let seed: Vec<f64> = (0..n)
            .map(|i| {
                if r == 0 {
                    return 0.0;
                }
                lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let u = ((lcg >> 11) as f64) / ((1u64 << 53) as f64); // [0, 1)
                match robot.joints[i].limits {
                    Some((lo, hi)) => lo + u * (hi - lo),
                    None => (u - 0.5) * std::f64::consts::TAU,
                }
            })
            .collect();

        let res = solve_ik(robot, target, &seed, opts);
        if res.converged && within_limits(robot, &res.q) {
            return res;
        }
        if best.as_ref().map_or(true, |b| res.error < b.error) {
            best = Some(res);
        }
    }
    best.expect("at least one attempt runs")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{from_urdf_str, pose_error};

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

    #[test]
    fn robust_ik_reaches_target_within_limits() {
        let robot = from_urdf_str(ARM, "world", "tool").unwrap();
        let q_true = [2.9, -0.5, 0.7, 0.3, 0.4, -0.2]; // near a joint limit
        let target = robot.fk(&q_true);
        let res = solve_ik_robust(&robot, &target, &IkOptions::default(), 12);
        assert!(res.converged, "robust IK did not converge: {}", res.error);
        assert!(within_limits(&robot, &res.q), "solution violates joint limits");
        assert!(pose_error(&robot.fk(&res.q), &target).norm() < 1e-4);
    }
}
