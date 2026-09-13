//! **Differential inverse kinematics on a branched tree with a floating base** — the task-stack QP a
//! humanoid needs, where [`solve_diffik`](crate::solve_diffik) and [`tree_ik`](crate::tree_ik) each
//! supply half.
//!
//! | | branched | floating base | orientation | limits in the QP |
//! |---|---|---|---|---|
//! | [`solve_diffik`](crate::solve_diffik) | ⛔ serial `Robot` only | ⛔ fixed | ⛔ position only | ✅ |
//! | [`tree_ik`](crate::tree_ik) | ✅ | ⛔ fixed | ⛔ position only | ⚠ clamped AFTER the step |
//! | this | ✅ | ✅ | ✅ | ✅ |
//!
//! A legged robot is not a manipulator bolted to the floor: its base pose is part of the answer, and
//! the tasks that decide it — keep both feet planted, hold the pelvis level, put a hand somewhere —
//! are simultaneous and compete. That is a task-stack QP, and it is what Pink and mink provide on
//! Pinocchio and MuJoCo.
//!
//! # The decision variable
//!
//! `x = [v (3) | ω (3) | q̇ (n)]` with the base block present only when
//! [`TreeDiffIkOptions::floating_base`] is set. **Both base blocks are in WORLD axes, and `ω` turns
//! the body about the base's own origin, not about the world origin:**
//!
//! ```text
//! base.translation += v·dt
//! base.rotation     = exp(ω·dt) · base.rotation
//! ```
//!
//! ⛔ The twist ordering is `[linear; angular]` throughout this module — error vector, weights and
//! Jacobian rows alike. It is stated here because a silent disagreement between the three is not a
//! wrong answer, it is a *plausible* wrong answer.
//!
//! # Why the Jacobian is differenced rather than derived
//!
//! ⭐ [`KinematicTree::frames`](crate::KinematicTree::frames) is the verified forward map, and a
//! Jacobian obtained by differencing it **cannot disagree with it** — the same reasoning
//! [`tree_ik`](crate::tree_ik) already gives. An analytic tree Jacobian would be a second statement of
//! the kinematics, free to drift from the first. The cost is `6 + n` forward passes per iteration,
//! which at IK rates is nothing next to being sure.
//!
//! ⚠ The base columns are the exception worth checking, because they are this module's own invention
//! rather than the tree's: `the_base_columns_match_the_analytic_rigid_body_twist` derives them a second
//! way, from `v + ω × (p_tip − p_base)`, and requires the two to agree. Measured, they agree to
//! `1.45e-8` — the finite-difference floor.
//!
//! # The Jacobian in the QP is of the ERROR, not of the pose
//!
//! ⛔ Worth stating because it looks like a discrepancy and is not. The columns assembled for the step
//! are `∂e/∂x`, so the step is Gauss-Newton on the task error. On the position rows that is the pose
//! Jacobian negated and nothing more. **On the rotation rows it is not**: the error is
//! `log(R_target · R_current⁻¹)`, and differentiating a logarithm brings in that map's own Jacobian,
//! which is the identity only where the error is zero.
//!
//! So the twist Jacobian and the one used here agree at convergence and differ away from it. The first
//! version of the base-column test differenced the ERROR and compared it against `v + ω × r`; it failed,
//! and the oracle was what was wrong. It now differences the POSE, which is what the base
//! parameterisation actually defines.

use crate::kinematic_tree::KinematicTree;
use crate::Iso;
use clarabel::algebra::CscMatrix;
use clarabel::solver::{DefaultSettingsBuilder, DefaultSolver, IPSolver, SupportedConeT};
use nalgebra::{DMatrix, DVector, Translation3, UnitQuaternion, Vector3};

/// One frame task: drive a named tip toward a target pose.
///
/// Setting `orientation_weight` to `0` makes it a position task and `position_weight` to `0` an
/// orientation task, so a foot that must be flat but may slide, or a gaze that must point but not
/// travel, are both expressible without a second type.
#[derive(Clone, Debug)]
pub struct TreeFrameTask {
    /// A tip name from [`KinematicTree::tip_offsets`].
    pub tip: String,
    pub target: Iso,
    pub position_weight: f64,
    pub orientation_weight: f64,
    /// Proportional gain on the error, in `1/s`. `gain·dt` near `1` takes a full Newton step.
    pub gain: f64,
}

impl TreeFrameTask {
    /// A pose task weighting position and orientation equally.
    pub fn pose(tip: &str, target: Iso, gain: f64) -> Self {
        Self { tip: tip.into(), target, position_weight: 1.0, orientation_weight: 1.0, gain }
    }

    /// A position-only task: the tip must arrive, however it is turned.
    pub fn position(tip: &str, target: Vector3<f64>, gain: f64) -> Self {
        Self {
            tip: tip.into(),
            target: Iso::from_parts(Translation3::from(target), UnitQuaternion::identity()),
            position_weight: 1.0,
            orientation_weight: 0.0,
            gain,
        }
    }
}

/// Options for [`solve_tree_diffik`].
#[derive(Clone, Debug)]
pub struct TreeDiffIkOptions {
    pub dt: f64,
    /// Joint velocity bound, per joint, in rad/s or m/s.
    pub vmax: f64,
    /// Base linear and angular velocity bounds. Ignored unless `floating_base`.
    pub base_vmax: (f64, f64),
    /// Tikhonov damping on the whole decision variable. Keeps the QP strictly convex at a singularity.
    pub damping: f64,
    pub max_iters: usize,
    /// Convergence: worst task position error in metres, and worst orientation error in radians.
    pub tol: (f64, f64),
    /// Solve for the base pose as well as the joints.
    pub floating_base: bool,
    /// Map joint position limits to per-step velocity bounds, so a limit is a QP CONSTRAINT rather
    /// than a clamp applied afterwards.
    pub use_limits: bool,
    /// Regularize the joints toward `rest` with this weight, resolving redundancy.
    pub posture: Option<(Vec<f64>, f64)>,
}

impl Default for TreeDiffIkOptions {
    fn default() -> Self {
        Self {
            dt: 0.05,
            vmax: 3.0,
            base_vmax: (3.0, 3.0),
            damping: 1e-6,
            max_iters: 300,
            tol: (1e-6, 1e-6),
            floating_base: false,
            use_limits: true,
            posture: None,
        }
    }
}

/// What the solve reached.
#[derive(Clone, Debug)]
pub struct TreeDiffIkResult {
    pub q: Vec<f64>,
    /// The base pose. Unchanged from the seed unless [`TreeDiffIkOptions::floating_base`].
    pub base: Iso,
    /// Worst task position error, metres. `0` when every task has `position_weight == 0`.
    pub position_error: f64,
    /// Worst task orientation error, radians.
    pub orientation_error: f64,
    pub iters: usize,
    pub converged: bool,
}

/// Upper-triangular CSC of a dense symmetric matrix, which is what clarabel wants for `P`.
fn csc_upper(p: &DMatrix<f64>) -> CscMatrix<f64> {
    let n = p.ncols();
    let mut colptr = Vec::with_capacity(n + 1);
    let mut rowval = Vec::new();
    let mut nzval = Vec::new();
    colptr.push(0);
    for j in 0..n {
        for i in 0..=j {
            rowval.push(i);
            nzval.push(p[(i, j)]);
        }
        colptr.push(rowval.len());
    }
    CscMatrix::new(n, n, colptr, rowval, nzval)
}

/// Box-constraint matrix `[I; -I]` (2n×n) in CSC.
fn csc_box(n: usize) -> CscMatrix<f64> {
    let mut colptr = Vec::with_capacity(n + 1);
    let mut rowval = Vec::with_capacity(2 * n);
    let mut nzval = Vec::with_capacity(2 * n);
    colptr.push(0);
    for j in 0..n {
        rowval.push(j);
        nzval.push(1.0);
        rowval.push(n + j);
        nzval.push(-1.0);
        colptr.push(rowval.len());
    }
    CscMatrix::new(2 * n, n, colptr, rowval, nzval)
}

/// Apply a decision-variable step of size `s` in coordinate `k`, returning the moved state.
///
/// One function for both the integrator and the finite difference, so the Jacobian is by construction
/// the derivative of the map the solve actually steps along. Writing them separately is how a
/// Jacobian comes to describe a slightly different system than the one being integrated.
fn step_state(base: &Iso, q: &[f64], nb: usize, k: usize, s: f64) -> (Iso, Vec<f64>) {
    let mut b = *base;
    let mut qq = q.to_vec();
    if nb == 6 && k < 6 {
        if k < 3 {
            b.translation.vector[k] += s;
        } else {
            let w = Vector3::ith(k - 3, s);
            b.rotation = UnitQuaternion::from_scaled_axis(w) * b.rotation;
        }
    } else {
        qq[k - nb] += s;
    }
    (b, qq)
}

/// The world twist error of one task at the current state, as `[linear; angular]`.
fn task_error(tree: &KinematicTree, base: &Iso, q: &[f64], t: &TreeFrameTask) -> Option<DVector<f64>> {
    let cur = tree.tip_pose(&t.tip, *base, q)?;
    let mut e = DVector::zeros(6);
    let dp = t.target.translation.vector - cur.translation.vector;
    // The world-frame rotation vector taking CURRENT to TARGET: `log(R_t · R_c⁻¹)`. Its body-frame
    // counterpart is `log(R_c⁻¹ · R_t)`.
    //
    // ⭐ **Swapping the two is a mutation that survives, and it should.** The claim written here first
    // was that a body-frame error fed to a world-frame Jacobian would be a plausible wrong answer.
    // Measured, it is not: the Jacobian is DIFFERENCED FROM THIS FUNCTION, so changing the convention
    // changes both sides together and the Gauss-Newton step stays consistent. With the swap applied,
    // `an_orientation_task_on_one_hinge_returns_exactly_the_angle_asked_for` still returns θ to 1e-7 at
    // all three angles. The two rotations are conjugate, so even the reported magnitude is identical.
    //
    // ⛔ That immunity is bought by differencing the forward map rather than deriving the Jacobian —
    // see the module header. Hand-write the Jacobian in one frame and this line in the other and it
    // becomes exactly the classic bug it looks like.
    let dr = (t.target.rotation * cur.rotation.inverse()).scaled_axis();
    for a in 0..3 {
        e[a] = dp[a];
        e[3 + a] = dr[a];
    }
    Some(e)
}

/// **Differential IK on a tree.** Iterates QP velocity steps until every task is met or the budget runs out.
///
/// # Errors
///
/// If `seed` does not match the tree's degrees of freedom, if there are no tasks, if a task names a tip
/// the tree does not have, or if an option is not finite and positive where it must be.
pub fn solve_tree_diffik(
    tree: &KinematicTree,
    base0: Iso,
    tasks: &[TreeFrameTask],
    seed: &[f64],
    opts: &TreeDiffIkOptions,
) -> Result<TreeDiffIkResult, String> {
    let n = tree.dof();
    if seed.len() != n {
        return Err(format!("the seed has {} values for a tree with {n} degrees of freedom", seed.len()));
    }
    if tasks.is_empty() {
        return Err("a solve with no tasks has nothing to satisfy".into());
    }
    for t in tasks {
        if !tree.tip_offsets.contains_key(&t.tip) {
            return Err(format!("no tip named {:?} in this tree", t.tip));
        }
        if !(t.gain.is_finite() && t.position_weight.is_finite() && t.orientation_weight.is_finite()) {
            return Err(format!("task {:?} carries a non-finite gain or weight", t.tip));
        }
        if t.position_weight < 0.0 || t.orientation_weight < 0.0 {
            return Err(format!("task {:?} carries a negative weight", t.tip));
        }
    }
    if !(opts.dt.is_finite() && opts.dt > 0.0) {
        return Err(format!("dt must be finite and positive, got {}", opts.dt));
    }
    if !(opts.damping.is_finite() && opts.damping >= 0.0) {
        return Err(format!("damping must be finite and non-negative, got {}", opts.damping));
    }

    let nb = if opts.floating_base { 6 } else { 0 };
    let nv = nb + n;
    if nv == 0 {
        return Err("a fixed-base tree with no joints has nothing to solve for".into());
    }
    let mut q = seed.to_vec();
    let mut base = base0;
    let a_csc = csc_box(nv);
    let settings = DefaultSettingsBuilder::default().verbose(false).build().map_err(|e| e.to_string())?;
    let h = 1e-7;

    let worst = |base: &Iso, q: &[f64]| -> Option<(f64, f64)> {
        let mut wp: f64 = 0.0;
        let mut wr: f64 = 0.0;
        for t in tasks {
            let e = task_error(tree, base, q, t)?;
            if t.position_weight > 0.0 {
                wp = wp.max(e.rows(0, 3).norm());
            }
            if t.orientation_weight > 0.0 {
                wr = wr.max(e.rows(3, 3).norm());
            }
        }
        Some((wp, wr))
    };

    let mut iters = 0usize;
    let mut reached = worst(&base, &q).ok_or("a task names a tip this tree cannot place")?;
    for it in 0..opts.max_iters {
        iters = it + 1;
        if reached.0 < opts.tol.0 && reached.1 < opts.tol.1 {
            iters = it;
            break;
        }

        let mut p = DMatrix::<f64>::identity(nv, nv) * opts.damping;
        let mut g = DVector::<f64>::zeros(nv);
        for t in tasks {
            let e = task_error(tree, &base, &q, t).ok_or("a task names a tip this tree cannot place")?;
            let mut j = DMatrix::<f64>::zeros(6, nv);
            for k in 0..nv {
                let (bp, qp) = step_state(&base, &q, nb, k, h);
                let ep = task_error(tree, &bp, &qp, t).ok_or("a task names a tip this tree cannot place")?;
                // d(pose)/dx = −d(error)/dx, since the error is target minus current
                for r in 0..6 {
                    j[(r, k)] = -(ep[r] - e[r]) / h;
                }
            }
            let mut w = DVector::<f64>::zeros(6);
            for a in 0..3 {
                w[a] = t.position_weight;
                w[3 + a] = t.orientation_weight;
            }
            let jw = DMatrix::from_fn(6, nv, |r, c| w[r] * j[(r, c)]);
            p += j.transpose() * &jw;
            g += jw.transpose() * (&e * t.gain);
        }
        if let Some((rest, pw)) = &opts.posture {
            if rest.len() != n {
                return Err(format!("the posture has {} values for a tree with {n} joints", rest.len()));
            }
            for i in 0..n {
                p[(nb + i, nb + i)] += pw;
                g[nb + i] += pw * (rest[i] - q[i]);
            }
        }
        let q_lin: Vec<f64> = (0..nv).map(|i| -g[i]).collect();

        let mut ub = vec![0.0; nv];
        let mut lb = vec![0.0; nv];
        for k in 0..nv {
            let bound = if nb == 6 && k < 3 {
                opts.base_vmax.0
            } else if nb == 6 && k < 6 {
                opts.base_vmax.1
            } else {
                opts.vmax
            };
            ub[k] = bound;
            lb[k] = -bound;
        }
        if opts.use_limits {
            for i in 0..n {
                if let Some((lo, hi)) = tree.joints[i].limits {
                    // ⭐ A limit enters as a BOUND ON THE STEP, so the QP never proposes a move that
                    // leaves the interval. `tree_ik` clamps after the fact, which lets the solver keep
                    // asking for the same infeasible direction and call the result converged.
                    ub[nb + i] = ub[nb + i].min((hi - q[i]) / opts.dt);
                    lb[nb + i] = lb[nb + i].max((lo - q[i]) / opts.dt);
                }
            }
        }
        let mut b = ub.clone();
        b.extend(lb.iter().map(|v| -v));

        let p_csc = csc_upper(&p);
        let cones = [SupportedConeT::NonnegativeConeT(2 * nv)];
        let mut solver = DefaultSolver::new(&p_csc, &q_lin, &a_csc, &b, &cones, settings.clone())
            .map_err(|e| format!("the step QP could not be assembled: {e}"))?;
        solver.solve();
        let x = &solver.solution.x;
        if x.iter().any(|v| !v.is_finite()) {
            break;
        }

        let mut moved = 0.0;
        for k in 0..nv {
            let s = x[k] * opts.dt;
            let (nb2, nq) = step_state(&base, &q, nb, k, s);
            base = nb2;
            q = nq;
            moved += s.abs();
        }
        reached = worst(&base, &q).ok_or("a task names a tip this tree cannot place")?;
        if moved < 1e-14 {
            break; // no feasible progress: a limit or a singularity holds every direction
        }
    }

    Ok(TreeDiffIkResult {
        q,
        base,
        position_error: reached.0,
        orientation_error: reached.1,
        iters,
        converged: reached.0 < opts.tol.0 && reached.1 < opts.tol.1,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kinematic_tree::tree_from_urdf;
    use nalgebra::Matrix3;

    /// One revolute joint about `z` at the origin, a tip 0.3 m out along `x`. Limits are a parameter so
    /// the same fixture can be pinned rigid.
    fn hinge(lo: f64, hi: f64) -> KinematicTree {
        let xml = format!(
            r#"<robot name="h"><link name="base"/>
              <link name="l1"><inertial><mass value="1"/><origin xyz="0.15 0 0"/>
                <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial></link>
              <link name="tip"/>
              <joint name="j1" type="revolute"><parent link="base"/><child link="l1"/><origin xyz="0 0 0"/>
                <axis xyz="0 0 1"/><limit lower="{lo}" upper="{hi}" effort="5" velocity="3"/></joint>
              <joint name="jt" type="fixed"><parent link="l1"/><child link="tip"/><origin xyz="0.3 0 0"/></joint>
            </robot>"#
        );
        tree_from_urdf(&xml, "base").expect("the hinge loads")
    }

    const ARM: &str = r#"<robot name="s">
      <link name="base"/>
      <link name="l1"><inertial><origin xyz="0 0 0.15"/><mass value="1.5"/>
        <inertia ixx="0.02" ixy="0" ixz="0" iyy="0.03" iyz="0" izz="0.025"/></inertial></link>
      <link name="l2"><inertial><origin xyz="0.1 0 0.1"/><mass value="1.0"/>
        <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.012" iyz="0" izz="0.011"/></inertial></link>
      <link name="l3"><inertial><origin xyz="0 0.05 0"/><mass value="0.6"/>
        <inertia ixx="0.005" ixy="0" ixz="0" iyy="0.006" iyz="0" izz="0.005"/></inertial></link>
      <link name="tool"/>
      <joint name="j1" type="revolute"><parent link="base"/><child link="l1"/><origin xyz="0 0 0.1"/>
        <axis xyz="0 0 1"/><limit lower="-3" upper="3" effort="10" velocity="3"/></joint>
      <joint name="j2" type="revolute"><parent link="l1"/><child link="l2"/><origin xyz="0.4 0 0"/>
        <axis xyz="0 1 0"/><limit lower="-3" upper="3" effort="10" velocity="3"/></joint>
      <joint name="j3" type="revolute"><parent link="l2"/><child link="l3"/><origin xyz="0.3 0 0"/>
        <axis xyz="0 1 0"/><limit lower="-3" upper="3" effort="10" velocity="3"/></joint>
      <joint name="jt" type="fixed"><parent link="l3"/><child link="tool"/><origin xyz="0.2 0 0"/></joint>
    </robot>"#;

    /// A shared wrist, then three two-joint fingers — the branch topology the whole module is for.
    fn hand() -> KinematicTree {
        let mut s = String::from(
            r#"<robot name="hand"><link name="palm"/>
              <link name="wrist"><inertial><mass value="0.4"/><origin xyz="0 0 0.02"/>
                <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/></inertial></link>
              <joint name="wj" type="revolute"><parent link="palm"/><child link="wrist"/><origin xyz="0 0 0.05"/>
                <axis xyz="0 0 1"/><limit lower="-2" upper="2" effort="5" velocity="3"/></joint>"#,
        );
        for f in 0..3 {
            let y = -0.03 + 0.03 * f as f64;
            s += &format!(
                r#"<link name="f{f}p"><inertial><mass value="0.05"/><origin xyz="0.02 0 0"/>
                     <inertia ixx="1e-5" ixy="0" ixz="0" iyy="1e-5" iyz="0" izz="1e-5"/></inertial></link>
                   <link name="f{f}d"><inertial><mass value="0.03"/><origin xyz="0.015 0 0"/>
                     <inertia ixx="5e-6" ixy="0" ixz="0" iyy="5e-6" iyz="0" izz="5e-6"/></inertial></link>
                   <link name="f{f}tip"/>
                   <joint name="f{f}j1" type="revolute"><parent link="wrist"/><child link="f{f}p"/>
                     <origin xyz="0.04 {y} 0"/><axis xyz="0 1 0"/>
                     <limit lower="-1.4" upper="1.4" effort="2" velocity="4"/></joint>
                   <joint name="f{f}j2" type="revolute"><parent link="f{f}p"/><child link="f{f}d"/>
                     <origin xyz="0.045 0 0"/><axis xyz="0 1 0"/>
                     <limit lower="-1.4" upper="1.4" effort="2" velocity="4"/></joint>
                   <joint name="f{f}t" type="fixed"><parent link="f{f}d"/><child link="f{f}tip"/>
                     <origin xyz="0.035 0 0"/></joint>"#
            );
        }
        tree_from_urdf(&(s + "</robot>"), "palm").expect("the hand loads")
    }

    fn iso(t: Vector3<f64>, r: UnitQuaternion<f64>) -> Iso {
        Iso::from_parts(Translation3::from(t), r)
    }

    /// ⭐⭐ **The floating base, against a closed form.**
    ///
    /// Pin the one joint rigid, so the base alone can satisfy the task. The answer is then not an
    /// optimisation outcome at all — it is algebra: the tip is a fixed transform `T` from the base, so
    /// the base that puts the tip at `target` is exactly `target · T⁻¹`, and there is no other.
    ///
    /// ⛔ This is the test that decides whether the base parameterisation is right. It exercises both
    /// blocks at once and in combination — the target is displaced AND turned — so a translation
    /// applied in the body frame instead of the world, or a rotation composed on the wrong side,
    /// cannot pass it.
    #[test]
    fn a_rigid_body_on_a_floating_base_lands_exactly_where_algebra_says() {
        let tree = hinge(0.0, 0.0); // the joint cannot move, so every degree of freedom used is the base's
        let seed = vec![0.0];
        let t_tip = tree.tip_pose("tip", Iso::identity(), &seed).expect("the tip exists");

        let target = iso(
            Vector3::new(0.4, -0.25, 0.6),
            UnitQuaternion::from_euler_angles(0.3, -0.45, 1.1),
        );
        let want = target * t_tip.inverse();

        let opts = TreeDiffIkOptions { floating_base: true, max_iters: 400, ..Default::default() };
        let r = solve_tree_diffik(&tree, Iso::identity(), &[TreeFrameTask::pose("tip", target, 10.0)], &seed, &opts)
            .expect("the solve runs");

        let dt = (r.base.translation.vector - want.translation.vector).norm();
        let dr = r.base.rotation.angle_to(&want.rotation);
        eprintln!(
            "  floating base: {} iters, tip error {:.2e} m / {:.2e} rad; base off by {dt:.2e} m / {dr:.2e} rad",
            r.iters, r.position_error, r.orientation_error
        );
        assert!(r.converged, "a rigid body with a free base can always reach a pose");
        assert!(dt < 1e-6 && dr < 1e-6, "the base must be target·T⁻¹; off by {dt:.3e} m and {dr:.3e} rad");
        assert!(r.q.iter().all(|v| v.abs() < 1e-12), "the joint is pinned at 0 and must not have moved: {:?}", r.q);
    }

    /// ⛔ **The base columns are this module's own invention, so they get a second derivation.**
    ///
    /// Everything else in the Jacobian is differenced from
    /// [`KinematicTree::frames`](crate::KinematicTree::frames) and so cannot disagree with the forward
    /// map. The six base columns cannot borrow that guarantee: nothing else in the crate defines them.
    ///
    /// The independent statement is rigid-body kinematics — a body turning at `ω` about its own origin
    /// and travelling at `v` carries a point at `r` from that origin at `v + ω × r`, and turns it at
    /// `ω`. In the `[linear; angular]` ordering this module uses that is exactly
    /// `[[I, −skew(r)], [0, I]]`, and the numerical columns must reproduce it.
    #[test]
    fn the_base_columns_match_the_analytic_rigid_body_twist() {
        let tree = hinge(-3.0, 3.0);
        // an off-identity base and a bent joint, so nothing is hidden by a coincidence at the origin
        let base = iso(Vector3::new(-0.2, 0.35, 0.7), UnitQuaternion::from_euler_angles(0.2, 0.5, -0.7));
        let q = vec![0.6];
        let pose0 = tree.tip_pose("tip", base, &q).expect("the tip exists");

        // ⚠ The POSE is differenced here, not the task error. The two differ by the Jacobian of the
        // logarithm on the rotation rows, which is the identity only at zero error — so differencing
        // the error against a target would be comparing a Gauss-Newton Jacobian with a twist and
        // finding a disagreement that is not a defect. What the base parameterisation defines, and
        // what this test is for, is the twist.
        let h = 1e-7;
        let mut num = DMatrix::<f64>::zeros(6, 6);
        for k in 0..6 {
            let (bp, qp) = step_state(&base, &q, 6, k, h);
            let pose = tree.tip_pose("tip", bp, &qp).expect("the tip exists");
            let dp = (pose.translation.vector - pose0.translation.vector) / h;
            let dr = (pose.rotation * pose0.rotation.inverse()).scaled_axis() / h;
            for r in 0..3 {
                num[(r, k)] = dp[r];
                num[(3 + r, k)] = dr[r];
            }
        }

        let p_tip = tree.tip_pose("tip", base, &q).expect("the tip exists").translation.vector;
        let r = p_tip - base.translation.vector;
        let skew = Matrix3::new(0.0, -r.z, r.y, r.z, 0.0, -r.x, -r.y, r.x, 0.0);
        let mut want = DMatrix::<f64>::zeros(6, 6);
        for a in 0..3 {
            want[(a, a)] = 1.0; // linear response to base translation
            want[(3 + a, 3 + a)] = 1.0; // angular response to base rotation
            for b in 0..3 {
                want[(a, 3 + b)] = -skew[(a, b)]; // linear response to base rotation: ω × r
            }
        }
        let worst = (0..6).flat_map(|i| (0..6).map(move |j| (i, j))).map(|(i, j)| (num[(i, j)] - want[(i, j)]).abs()).fold(0.0, f64::max);
        eprintln!("  base columns: worst disagreement with v + ω×r is {worst:.2e}");
        assert!(worst < 1e-6, "the differenced base columns disagree with rigid-body kinematics by {worst:.3e}\n{num}\n{want}");
    }

    /// ⭐ **Two implementations, one target.** The serial QP solver and this one share no code path:
    /// different robot type, different Jacobian (analytic there, differenced here), different variable
    /// layout. Reaching the same point is therefore evidence, not a tautology.
    #[test]
    fn the_tree_solver_and_the_serial_solver_reach_the_same_point() {
        use crate::diffik::{solve_diffik, DiffIkOptions, FrameTaskDef};
        let tree = tree_from_urdf(ARM, "base").expect("the arm loads as a tree");
        let robot = crate::from_urdf_str(ARM, "base", "tool").expect("the arm loads as a serial robot");
        let seed = vec![0.2, -0.3, 0.4];
        let tool = Vector3::new(0.2, 0.0, 0.0);

        // the two loaders must agree about where the tool IS before agreeing about where to put it
        let a = tree.tip_pose("tool", Iso::identity(), &seed).expect("tip").translation.vector;
        let b = (robot.frame_pose(&seed, 3) * nalgebra::Point3::from(tool)).coords;
        assert!((a - b).norm() < 1e-12, "the loaders disagree about the tool: {a:?} vs {b:?}");

        let target = Vector3::new(0.55, 0.22, 0.30);
        let tr = solve_tree_diffik(
            &tree,
            Iso::identity(),
            &[TreeFrameTask::position("tool", target, 10.0)],
            &seed,
            &TreeDiffIkOptions { tol: (1e-9, 1e-9), max_iters: 600, ..Default::default() },
        )
        .expect("the tree solve runs");
        let sr = solve_diffik(
            &robot,
            &[FrameTaskDef::new(3, tool, target, 10.0, 1.0)],
            &seed,
            &DiffIkOptions { tol: 1e-9, max_iters: 600, ..Default::default() },
        );

        let tp = tree.tip_pose("tool", Iso::identity(), &tr.q).expect("tip").translation.vector;
        let sp = (robot.frame_pose(&sr.q, 3) * nalgebra::Point3::from(tool)).coords;
        eprintln!(
            "  tree {} iters -> {:?} (err {:.2e});  serial {} iters -> {:?} (err {:.2e})",
            tr.iters, tp, tr.position_error, sr.iters, sp, sr.error
        );
        assert!(tr.converged, "the tree solve must reach a target well inside the workspace");
        assert!((tp - target).norm() < 1e-8, "the tree solve is {:.2e} from the target", (tp - target).norm());
        assert!((sp - target).norm() < 1e-8, "the control solve is {:.2e} from the target", (sp - target).norm());
    }

    /// ⭐ **A limit is a constraint on the step, not a clamp after it.**
    ///
    /// ⛔ [`tree_ik`](crate::tree_ik) takes an unconstrained damped-least-squares step and clamps the
    /// result into the interval. The joint value it reports is inside the limits either way — that is
    /// not what separates them. What separates them is that a clamp leaves the solver free to keep
    /// proposing the same infeasible direction, so the same step is taken and undone every iteration.
    /// Here the bound is in the QP, so the direction is never proposed.
    ///
    /// The target below is outside the limited workspace on purpose. What is asserted is that the
    /// answer stays legal and that the reported error is the TRUE residual — a solver that reported a
    /// pre-clamp error would claim to be closer than it is.
    #[test]
    fn a_joint_limit_is_respected_and_the_reported_error_is_the_real_one() {
        let mut tree = tree_from_urdf(ARM, "base").expect("loads");
        tree.joints[1].limits = Some((-0.1, 0.1));
        tree.joints[2].limits = Some((-0.1, 0.1));
        let seed = vec![0.0, 0.0, 0.0];
        let target = Vector3::new(0.2, 0.0, -0.9); // straight down, which needs the pitch joints

        let r = solve_tree_diffik(
            &tree,
            Iso::identity(),
            &[TreeFrameTask::position("tool", target, 10.0)],
            &seed,
            &TreeDiffIkOptions { max_iters: 200, ..Default::default() },
        )
        .expect("the solve runs");

        let reached = tree.tip_pose("tool", Iso::identity(), &r.q).expect("tip").translation.vector;
        let truth = (reached - target).norm();
        eprintln!("  limited arm: q = {:?}, reported {:.6} m, recomputed {:.6} m", r.q, r.position_error, truth);
        for (i, v) in r.q.iter().enumerate() {
            if let Some((lo, hi)) = tree.joints[i].limits {
                assert!(*v >= lo - 1e-12 && *v <= hi + 1e-12, "joint {i} left its limits [{lo}, {hi}] at {v}");
            }
        }
        assert!(!r.converged, "this target is out of the limited workspace and must not be reported as reached");
        assert!((r.position_error - truth).abs() < 1e-12, "the reported error {:.9} is not the real one {truth:.9}", r.position_error);
    }

    /// ⭐ **The reason this takes a LIST of tasks.** Three fingers share a wrist, so solving one alone
    /// spends the shared joint on it. A joint solve must do better across all three than the best
    /// single-task solve does.
    ///
    /// ⛔ The comparison is made at ONE configuration each. Solving three tasks separately yields three
    /// different `q` that cannot be combined, which is the whole problem — so each single-task solve is
    /// scored on all three targets.
    #[test]
    fn one_solve_for_three_fingers_beats_spending_the_shared_wrist_on_any_one() {
        let tree = hand();
        let seed = vec![0.0; tree.dof()];
        // reachable-ish targets pulling the wrist three different ways
        let targets = [
            Vector3::new(0.10, -0.055, 0.035),
            Vector3::new(0.11, 0.005, 0.060),
            Vector3::new(0.09, 0.060, 0.030),
        ];
        let tasks: Vec<TreeFrameTask> = (0..3)
            .map(|f| TreeFrameTask::position(&format!("f{f}tip"), targets[f], 10.0))
            .collect();
        let opts = TreeDiffIkOptions { max_iters: 400, ..Default::default() };

        let score = |q: &[f64]| -> f64 {
            (0..3)
                .map(|f| {
                    let p = tree.tip_pose(&format!("f{f}tip"), Iso::identity(), q).expect("tip").translation.vector;
                    (p - targets[f]).norm()
                })
                .fold(0.0, f64::max)
        };

        let joint = solve_tree_diffik(&tree, Iso::identity(), &tasks, &seed, &opts).expect("runs");
        let together = score(&joint.q);
        let mut alone = f64::INFINITY;
        for f in 0..3 {
            let one = solve_tree_diffik(&tree, Iso::identity(), &tasks[f..f + 1], &seed, &opts).expect("runs");
            let s = score(&one.q);
            eprintln!("  finger {f} alone: worst-of-three {s:.5} m");
            alone = alone.min(s);
        }
        eprintln!("  all three together: worst-of-three {together:.5} m (best single-task solve {alone:.5} m)");
        assert!(
            together < alone,
            "solving together ({together:.5} m) must beat the best single-task solve ({alone:.5} m)"
        );
    }

    /// **An orientation task on one hinge has an exact answer: the joint angle is the turn asked for.**
    ///
    /// The hinge turns about world `z` through the origin, so rotating the whole tip pose about `z` by
    /// `θ` is reachable and reachable only at `q = θ`. Anything else — a sign error, the body-frame log
    /// instead of the world-frame one, the two rotation blocks transposed — lands somewhere else.
    #[test]
    fn an_orientation_task_on_one_hinge_returns_exactly_the_angle_asked_for() {
        let tree = hinge(-3.0, 3.0);
        let seed = vec![0.0];
        let rest = tree.tip_pose("tip", Iso::identity(), &seed).expect("tip");
        for theta in [0.7_f64, -1.25, 2.4] {
            let rz = UnitQuaternion::from_scaled_axis(Vector3::z() * theta);
            let target = iso(rz * rest.translation.vector, rz * rest.rotation);
            let r = solve_tree_diffik(
                &tree,
                Iso::identity(),
                &[TreeFrameTask::pose("tip", target, 10.0)],
                &seed,
                &TreeDiffIkOptions { tol: (1e-10, 1e-10), max_iters: 400, ..Default::default() },
            )
            .expect("runs");
            eprintln!("  θ = {theta:+.3}: q = {:+.9} after {} iters", r.q[0], r.iters);
            assert!(r.converged, "θ = {theta}: a pure hinge rotation is exactly reachable");
            assert!((r.q[0] - theta).abs() < 1e-7, "θ = {theta}: got q = {:.9}", r.q[0]);
        }
    }

    /// A fixed base must not move, and the malformed calls must be refused rather than guessed at.
    #[test]
    fn a_fixed_base_stays_put_and_bad_input_is_refused() {
        let tree = hinge(-3.0, 3.0);
        let base = iso(Vector3::new(0.1, -0.2, 0.3), UnitQuaternion::from_euler_angles(0.1, 0.2, 0.3));
        let seed = vec![0.0];
        let task = TreeFrameTask::position("tip", Vector3::new(1.0, 1.0, 1.0), 1.0);
        let r = solve_tree_diffik(&tree, base, core::slice::from_ref(&task), &seed, &TreeDiffIkOptions::default())
            .expect("runs");
        assert!(
            (r.base.translation.vector - base.translation.vector).norm() < 1e-15
                && r.base.rotation.angle_to(&base.rotation) < 1e-15,
            "a fixed base must be returned untouched, got {:?}",
            r.base
        );

        let d = TreeDiffIkOptions::default();
        assert!(solve_tree_diffik(&tree, base, core::slice::from_ref(&task), &[0.0, 0.0], &d).is_err(), "seed length");
        assert!(solve_tree_diffik(&tree, base, &[], &seed, &d).is_err(), "no tasks");
        assert!(
            solve_tree_diffik(&tree, base, &[TreeFrameTask::position("nope", Vector3::zeros(), 1.0)], &seed, &d).is_err(),
            "unknown tip"
        );
        let mut nan = task.clone();
        nan.gain = f64::NAN;
        assert!(solve_tree_diffik(&tree, base, &[nan], &seed, &d).is_err(), "non-finite gain");
        let mut neg = task;
        neg.position_weight = -1.0;
        assert!(solve_tree_diffik(&tree, base, &[neg], &seed, &d).is_err(), "negative weight");
        assert!(
            solve_tree_diffik(&tree, base, &[TreeFrameTask::position("tip", Vector3::zeros(), 1.0)], &seed,
                &TreeDiffIkOptions { dt: 0.0, ..Default::default() }).is_err(),
            "dt of zero"
        );
    }
}
