//! General factor-graph optimization — for topologies beyond a serial chain or a trajectory:
//! coupled multi-robot problems, loop closures, kinematic trees. Each [`SparseFactor`] touches an
//! arbitrary subset of the global variables; the Gauss-Newton normal equations are therefore
//! *generally* sparse (not block-tridiagonal). We assemble only the touched blocks and solve each
//! LM step. Assembly emits only the touched blocks as sparse triplets and the normal equations are
//! factored with **`faer`'s sparse Cholesky** — so cost is proportional to the fill, not `n³`, and
//! large branched/loop graphs scale. Pure Rust → WASM-clean.

use crate::SolveOptions;
use faer::linalg::solvers::Solve;
use faer::sparse::{SparseColMat, Triplet};
use faer::{Mat, Side};
use nalgebra::{DMatrix, DVector};
use std::collections::BTreeMap;

/// A factor: a residual block that depends on an arbitrary subset of the global scalar variables.
pub trait SparseFactor {
    fn dim(&self) -> usize;
    /// Global variable indices this factor depends on.
    fn vars(&self) -> Vec<usize>;
    /// Residual (`dim`) and its Jacobian (`dim × vars().len()`), evaluated at the full `x`.
    fn eval(&self, x: &[f64]) -> (DVector<f64>, DMatrix<f64>);
}

/// Result of a factor-graph solve.
#[derive(Clone, Debug)]
pub struct SparseResult {
    pub x: Vec<f64>,
    pub error: f64,
    pub iters: usize,
    pub converged: bool,
}

/// Solve `(H + λI)·Δ = g` where `H` is given by its (pre-summed) nonzero entries. Emits the lower
/// triangle as faer triplets and factors with sparse Cholesky. Returns `None` if not PD (the caller
/// then raises λ). This is the single sparse-linear-algebra hook for the whole factor-graph solver.
fn solve_normal_equations(
    n: usize,
    h: &BTreeMap<(usize, usize), f64>,
    lambda: f64,
    g: &DVector<f64>,
) -> Option<DVector<f64>> {
    let mut trips: Vec<Triplet<usize, usize, f64>> = Vec::with_capacity(h.len() + n);
    for (&(i, j), &v) in h {
        if i < j {
            continue; // lower triangle only (Side::Lower); H is symmetric
        }
        let val = if i == j { v + lambda } else { v };
        trips.push(Triplet::new(i, j, val));
    }
    for d in 0..n {
        if !h.contains_key(&(d, d)) {
            trips.push(Triplet::new(d, d, lambda)); // ensure a full diagonal
        }
    }
    let mat = SparseColMat::<usize, f64>::try_new_from_triplets(n, n, &trips).ok()?;
    let llt = mat.sp_cholesky(Side::Lower).ok()?;
    let mut rhs = Mat::<f64>::zeros(n, 1);
    for i in 0..n {
        rhs[(i, 0)] = g[i];
    }
    llt.solve_in_place(&mut rhs);
    Some(DVector::from_fn(n, |i, _| rhs[(i, 0)]))
}

/// Levenberg–Marquardt over a general factor graph on `nvars` scalar variables.
pub fn solve_factor_graph(
    nvars: usize,
    factors: &[Box<dyn SparseFactor + '_>],
    x0: &[f64],
    opts: &SolveOptions,
) -> SparseResult {
    let total_cost = |x: &[f64]| -> f64 { factors.iter().map(|f| f.eval(x).0.norm_squared()).sum() };

    let mut x = DVector::from_row_slice(x0);
    let mut lambda = opts.lambda0;
    let mut cost = total_cost(x.as_slice());
    let mut gnorm = f64::INFINITY;
    let mut iters = 0;

    'outer: for it in 0..opts.max_iters {
        iters = it + 1;

        // Sparse-aware assembly: scatter each factor's local JᵀJ / Jᵀr into the touched indices,
        // pre-summed into a map so there are no duplicate entries.
        let mut h: BTreeMap<(usize, usize), f64> = BTreeMap::new();
        let mut g = DVector::<f64>::zeros(nvars);
        for f in factors {
            let idx = f.vars();
            let (r, j) = f.eval(x.as_slice());
            let m = idx.len();
            for a in 0..m {
                let ja = j.column(a);
                g[idx[a]] += ja.dot(&r);
                for b in 0..m {
                    *h.entry((idx[a], idx[b])).or_insert(0.0) += ja.dot(&j.column(b));
                }
            }
        }

        gnorm = g.norm();
        if gnorm < opts.tol {
            break;
        }

        loop {
            match solve_normal_equations(nvars, &h, lambda, &g) {
                Some(dq) => {
                    let x_new = &x - &dq;
                    let cost_new = total_cost(x_new.as_slice());
                    if cost_new < cost {
                        x = x_new;
                        cost = cost_new;
                        lambda = (lambda * 0.5).max(1e-12);
                        break;
                    }
                    lambda *= 3.0;
                }
                None => lambda *= 3.0, // not positive-definite at this λ — raise it
            }
            if lambda > 1e12 {
                break 'outer;
            }
        }
    }

    SparseResult { x: x.as_slice().to_vec(), error: cost.sqrt(), iters, converged: gnorm < 1e-4 }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{from_urdf_str, Robot};
    use nalgebra::{Point3, Vector3};

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

    // Tool tip of an arm (frame 6 + 5cm) pulled to a fixed target — touches one arm's 6 vars.
    struct TipTarget<'a> {
        robot: &'a Robot,
        base: usize,
        target: Vector3<f64>,
    }
    impl SparseFactor for TipTarget<'_> {
        fn dim(&self) -> usize {
            3
        }
        fn vars(&self) -> Vec<usize> {
            (self.base..self.base + 6).collect()
        }
        fn eval(&self, x: &[f64]) -> (DVector<f64>, DMatrix<f64>) {
            let q = &x[self.base..self.base + 6];
            let p = (self.robot.frame_pose(q, 6) * Point3::from(Vector3::new(0.0, 0.0, 0.05))).coords;
            let r = p - self.target;
            let j = self.robot.point_jacobian(q, 6, &p);
            (DVector::from_row_slice(&[r.x, r.y, r.z]), j)
        }
    }

    // Loop closure: the two arms' tips must coincide — touches BOTH arms' vars (off-diagonal block).
    struct TipsCoincide<'a> {
        r1: &'a Robot,
        r2: &'a Robot,
    }
    impl SparseFactor for TipsCoincide<'_> {
        fn dim(&self) -> usize {
            3
        }
        fn vars(&self) -> Vec<usize> {
            (0..12).collect()
        }
        fn eval(&self, x: &[f64]) -> (DVector<f64>, DMatrix<f64>) {
            let off = Vector3::new(0.0, 0.0, 0.05);
            let p1 = (self.r1.frame_pose(&x[0..6], 6) * Point3::from(off)).coords;
            let p2 = (self.r2.frame_pose(&x[6..12], 6) * Point3::from(off)).coords;
            let j1 = self.r1.point_jacobian(&x[0..6], 6, &p1);
            let j2 = self.r2.point_jacobian(&x[6..12], 6, &p2);
            let mut j = DMatrix::zeros(3, 12);
            j.view_mut((0, 0), (3, 6)).copy_from(&j1);
            j.view_mut((0, 6), (3, 6)).copy_from(&(-j2));
            let r = p1 - p2;
            (DVector::from_row_slice(&[r.x, r.y, r.z]), j)
        }
    }

    #[test]
    fn two_arms_loop_closure() {
        // A non-chain topology: arm1's tip → a target, and arm2's tip must meet arm1's tip.
        let robot = from_urdf_str(ARM, "world", "tool").unwrap();
        // Target = a definitely-reachable tip pose (generated from a real config).
        let q_ref = [0.3, -0.4, 0.6, 0.2, 0.3, -0.2];
        let target = (robot.frame_pose(&q_ref, 6) * Point3::from(Vector3::new(0.0, 0.0, 0.05))).coords;
        let factors: Vec<Box<dyn SparseFactor + '_>> = vec![
            Box::new(TipTarget { robot: &robot, base: 0, target }),
            Box::new(TipsCoincide { r1: &robot, r2: &robot }),
        ];
        let x0 = [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, -0.1, 0.3, -0.2, 0.0, 0.0, 0.0];
        let res = solve_factor_graph(12, &factors, &x0, &SolveOptions { max_iters: 400, ..SolveOptions::default() });
        assert!(res.converged, "did not converge: err={} iters={}", res.error, res.iters);

        let off = Vector3::new(0.0, 0.0, 0.05);
        let tip1 = (robot.frame_pose(&res.x[0..6], 6) * Point3::from(off)).coords;
        let tip2 = (robot.frame_pose(&res.x[6..12], 6) * Point3::from(off)).coords;
        assert!((tip1 - target).norm() < 1e-3, "arm1 missed target: {}", (tip1 - target).norm());
        assert!((tip1 - tip2).norm() < 1e-3, "tips did not coincide: {}", (tip1 - tip2).norm());
    }
}
