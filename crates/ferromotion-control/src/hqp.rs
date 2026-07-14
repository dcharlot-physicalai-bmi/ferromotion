//! **Hierarchical Quadratic Programming (HQP)** — strict task-priority control (Escande, Mansard,
//! Kanoun, IJRR 2014; Kanoun et al.). Where our weighted whole-body controllers ([`crate::WholeBody`],
//! [`crate::OperationalSpace`]) trade tasks off against each other, HQP enforces a *strict* hierarchy:
//! each priority level is solved in the **null space of every higher level**, so a high-priority task
//! is achieved optimally and can never be degraded by a lower one.
//!
//! For equality tasks `Aₖ x = bₖ` this is the classic cascaded null-space recursion:
//! `x_k = x_{k-1} + (Aₖ Nₖ₋₁)⁺ (bₖ − Aₖ x_{k-1})`, `Nₖ = Nₖ₋₁ − (Aₖ Nₖ₋₁)⁺(Aₖ Nₖ₋₁)`, with `N₀ = I`.
//! The prototypical use is prioritized inverse kinematics / whole-body control: keep the foot planted
//! and the CoM balanced (top priority) while a reaching or posture task uses only the leftover
//! freedom. Pure `nalgebra` → WASM-clean.

use nalgebra::{DMatrix, DVector};

/// Solve a strictly-prioritized stack of equality tasks `[(A₁,b₁), (A₂,b₂), …]` (highest priority
/// first) for a solution `x ∈ ℝⁿ`. Each level is realized only within the null space of the ones above.
pub fn solve_hqp(tasks: &[(DMatrix<f64>, DVector<f64>)], n: usize) -> DVector<f64> {
    let mut x = DVector::zeros(n);
    let mut nmat = DMatrix::identity(n, n); // null-space projector of all higher-priority tasks
    for (a, b) in tasks {
        let an = a * &nmat; // task Jacobian restricted to the remaining freedom
        if an.norm() < 1e-12 {
            continue; // no freedom left for this task
        }
        let anp = an.clone().pseudo_inverse(1e-12).expect("pseudo-inverse");
        let dx = &anp * (b - a * &x);
        x += dx;
        nmat -= &anp * &an; // project the remaining freedom into this task's null space
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferromotion_core::{from_urdf_str, Robot};
    use nalgebra::Vector2;

    #[test]
    fn strict_priority_is_exact_and_lower_tasks_defer() {
        // Priority 1: x0=1, x1=2. Priority 2: x0=5 (conflicts!), x2=3.
        let n = 4;
        let a1 = DMatrix::from_row_slice(2, 4, &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        let b1 = DVector::from_row_slice(&[1.0, 2.0]);
        let a2 = DMatrix::from_row_slice(2, 4, &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        let b2 = DVector::from_row_slice(&[5.0, 3.0]);
        let x = solve_hqp(&[(a1.clone(), b1.clone()), (a2.clone(), b2.clone())], n);

        // Priority 1 met exactly.
        assert!((&a1 * &x - &b1).norm() < 1e-9, "top priority not exact: {}", (&a1 * &x - &b1).norm());
        // Its non-conflicting part (x2=3) is realized in the null space; the conflicting x0=5 defers.
        assert!((x[2] - 3.0).abs() < 1e-9, "free part of task 2 not met: x2={}", x[2]);
        assert!((x[0] - 1.0).abs() < 1e-9, "top priority overridden: x0={}", x[0]);
        assert!((&a2 * &x - &b2).norm() > 3.9, "task 2 should be unable to meet the conflict");
    }

    #[test]
    fn lower_priority_change_never_perturbs_higher() {
        let a1 = DMatrix::from_row_slice(1, 3, &[1.0, 1.0, 0.0]);
        let b1 = DVector::from_row_slice(&[2.0]);
        let a2 = DMatrix::from_row_slice(1, 3, &[1.0, 0.0, 0.0]);
        let solve = |b2v: f64| solve_hqp(&[(a1.clone(), b1.clone()), (a2.clone(), DVector::from_row_slice(&[b2v]))], 3);
        let (xa, xb) = (solve(0.0), solve(100.0));
        // Wildly different priority-2 targets, but priority 1 stays exactly satisfied in both.
        assert!((&a1 * &xa - &b1).norm() < 1e-9 && (&a1 * &xb - &b1).norm() < 1e-9);
    }

    const ARM3: &str = r#"<robot name="a3">
      <link name="base"/><link name="l1"/><link name="l2"/><link name="l3"/><link name="tool"/>
      <joint name="j1" type="revolute"><parent link="base"/><child link="l1"/><origin xyz="0 0 0"/><axis xyz="0 0 1"/></joint>
      <joint name="j2" type="revolute"><parent link="l1"/><child link="l2"/><origin xyz="1 0 0"/><axis xyz="0 0 1"/></joint>
      <joint name="j3" type="revolute"><parent link="l2"/><child link="l3"/><origin xyz="1 0 0"/><axis xyz="0 0 1"/></joint>
      <joint name="jt" type="fixed"><parent link="l3"/><child link="tool"/><origin xyz="1 0 0"/></joint>
    </robot>"#;

    // Prioritized IK on a redundant 3R arm: EE position (priority 1) + posture (priority 2, null space).
    fn ik(robot: &Robot, target: Vector2<f64>, q_rest: &[f64], use_posture: bool) -> ([f64; 3], f64) {
        let mut q = [0.3, 0.3, 0.3];
        for _ in 0..300 {
            let p = robot.fk(&q).translation.vector;
            let jee = robot.point_jacobian(&q, 3, &p).rows(0, 2).into_owned(); // 2×3
            let b1 = DVector::from_row_slice(&[target.x - p.x, target.y - p.y]);
            let mut tasks = vec![(jee, b1)];
            if use_posture {
                let a2 = DMatrix::identity(3, 3);
                let b2 = DVector::from_row_slice(&[q_rest[0] - q[0], q_rest[1] - q[1], q_rest[2] - q[2]]);
                tasks.push((a2, b2));
            }
            let dq = solve_hqp(&tasks, 3);
            for i in 0..3 {
                q[i] += 0.5 * dq[i];
            }
        }
        let p = robot.fk(&q).translation.vector;
        let err = (Vector2::new(p.x, p.y) - target).norm();
        (q, err)
    }

    #[test]
    fn prioritized_ik_reaches_exactly_and_posture_uses_the_null_space() {
        let robot: Robot = from_urdf_str(ARM3, "base", "tool").unwrap();
        let target = Vector2::new(1.5, 1.0);
        let q_rest = [0.1, 0.1, 0.1];
        let (q_no, err_no) = ik(&robot, target, &q_rest, false);
        let (q_yes, err_yes) = ik(&robot, target, &q_rest, true);

        // Both reach the EE target exactly (priority 1) …
        assert!(err_no < 1e-3 && err_yes < 1e-3, "EE not reached: {err_no}, {err_yes}");
        // … but the posture task (priority 2) pulls the redundant DoF toward q_rest, strictly closer
        // than the min-norm solution — using the 1-D null space the EE task leaves free.
        let dist = |q: &[f64; 3]| ((q[0] - q_rest[0]).powi(2) + (q[1] - q_rest[1]).powi(2) + (q[2] - q_rest[2]).powi(2)).sqrt();
        assert!(dist(&q_yes) < dist(&q_no) - 2e-3, "posture did not use the null space: {} vs {}", dist(&q_yes), dist(&q_no));
        assert!((q_yes != q_no), "posture produced no null-space motion");
    }
}
