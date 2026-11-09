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

/// Scalar product. Dynamic `nalgebra` products above 5×5 go through `matrixmultiply`, whose
/// accumulation order follows the host CPU. These tasks are small, and a host-dependent order
/// changes which redundant IK fixed point the posture task reaches.
fn gemm(a: &DMatrix<f64>, b: &DMatrix<f64>) -> DMatrix<f64> {
    let (m, k) = (a.nrows(), a.ncols());
    let n = b.ncols();
    let mut c = DMatrix::zeros(m, n);
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0;
            for t in 0..k {
                s += a[(i, t)] * b[(t, j)];
            }
            c[(i, j)] = s;
        }
    }
    c
}

fn gemv(a: &DMatrix<f64>, x: &DVector<f64>) -> DVector<f64> {
    let mut y = DVector::zeros(a.nrows());
    for i in 0..a.nrows() {
        let mut s = 0.0;
        for t in 0..a.ncols() {
            s += a[(i, t)] * x[t];
        }
        y[i] = s;
    }
    y
}

/// Eigenpairs of a symmetric matrix. Jacobi sweeps in index order, so the rotations do not
/// depend on which SIMD width `matrixmultiply` would have picked.
fn jacobi_sym(mut a: DMatrix<f64>) -> (Vec<f64>, DMatrix<f64>) {
    let n = a.nrows();
    let mut v = DMatrix::identity(n, n);
    for _ in 0..80 {
        let mut off = 0.0;
        for p in 0..n {
            for q in (p + 1)..n {
                off += a[(p, q)] * a[(p, q)];
            }
        }
        if off <= 1e-30 {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = a[(p, q)];
                if apq.abs() <= 1e-18 {
                    continue;
                }
                let app = a[(p, p)];
                let aqq = a[(q, q)];
                let tau = (aqq - app) / (2.0 * apq);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                a[(p, p)] = app - t * apq;
                a[(q, q)] = aqq + t * apq;
                a[(p, q)] = 0.0;
                a[(q, p)] = 0.0;
                for r in 0..n {
                    if r == p || r == q {
                        continue;
                    }
                    let arp = a[(r, p)];
                    let arq = a[(r, q)];
                    let rp = c * arp - s * arq;
                    let rq = s * arp + c * arq;
                    a[(r, p)] = rp;
                    a[(p, r)] = rp;
                    a[(r, q)] = rq;
                    a[(q, r)] = rq;
                }
                for r in 0..n {
                    let vrp = v[(r, p)];
                    let vrq = v[(r, q)];
                    v[(r, p)] = c * vrp - s * vrq;
                    v[(r, q)] = s * vrp + c * vrq;
                }
            }
        }
    }
    ((0..n).map(|i| a[(i, i)]).collect(), v)
}

/// Moore-Penrose inverse with singular values below `eps` dropped. `A⁺ = Aᵀ U Λ⁻¹ Uᵀ`
/// where `U` diagonalizes `AAᵀ` (or the transpose, when there are more rows than columns).
fn pseudo_inverse(a: &DMatrix<f64>, eps: f64) -> DMatrix<f64> {
    let (m, n) = a.shape();
    if m == 0 || n == 0 {
        return DMatrix::zeros(n, m);
    }
    if m > n {
        return pseudo_inverse(&a.transpose(), eps).transpose();
    }
    let at = a.transpose();
    let (evals, u) = jacobi_sym(gemm(a, &at));
    let mut inv = DMatrix::zeros(m, m);
    let scale = evals.iter().copied().fold(0.0, f64::max);
    // Absolute `eps` is what callers pass. A Jacobi sweep also leaves cancelled eigenvalues a few
    // ulps above zero; those are not task directions, and inverting them throws the higher
    // priority out of its null space.
    let floor = (eps * eps).max(scale * 1e-14);
    for i in 0..m {
        if evals[i] > floor {
            inv[(i, i)] = 1.0 / evals[i];
        }
    }
    gemm(&at, &gemm(&u, &gemm(&inv, &u.transpose())))
}

/// Solve a strictly-prioritized stack of equality tasks `[(A₁,b₁), (A₂,b₂), …]` (highest priority
/// first) for a solution `x ∈ ℝⁿ`. Each level is realized only within the null space of the ones above.
pub fn solve_hqp(tasks: &[(DMatrix<f64>, DVector<f64>)], n: usize) -> DVector<f64> {
    let mut x = DVector::zeros(n);
    let mut nmat = DMatrix::identity(n, n); // null-space projector of all higher-priority tasks
    for (a, b) in tasks {
        let an = gemm(a, &nmat); // task Jacobian restricted to the remaining freedom
        let mut an_norm = 0.0;
        for v in an.iter() {
            an_norm += v * v;
        }
        if an_norm.sqrt() < 1e-12 {
            continue; // no freedom left for this task
        }
        let anp = pseudo_inverse(&an, 1e-12);
        let residual = b - gemv(a, &x);
        let dx = gemv(&nmat, &gemv(&anp, &residual));
        x += dx;
        nmat -= gemm(&anp, &an); // project the remaining freedom into this task's null space
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
