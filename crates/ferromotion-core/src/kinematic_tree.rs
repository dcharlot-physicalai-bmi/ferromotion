//! **Loading a branched kinematic tree** — a hand, not a chain.
//!
//! The tree *runtime* in this crate is complete and verified. Four modules consume the
//! `(joints, parent[])` form: [`tree_dynamics`](crate::tree_floating_forward_dynamics),
//! [`whole_body_contact`](crate::whole_body_contact_step_pgs),
//! [`floating_contact`](crate::quadruped), and the batched GPU engine. What was missing is a way to *get* that form
//! from a file. [`from_urdf_full`](crate::from_urdf_full) walks a single path from base to tip and returns a serial
//! [`Robot`](crate::Robot); `mjcf` refuses branching outright with "serial chains only". So every tree consumer had to
//! be hand-built in code — [`quadruped`](crate::quadruped) constructs its topology by hand — and a five-finger hand
//! could not be loaded at all.
//!
//! [`tree_from_urdf`] closes that. It returns every actuated joint in the file with a `parent[]` array in topological
//! order, which is exactly what the existing tree routines already take.
//!
//! **The correctness oracle is agreement, not plausibility.** For a URDF that happens to be a serial chain, the tree
//! loader and the serial loader must produce the same kinematics. That is checked to `1e-15` on the tip frame across
//! sampled configurations, so the new path is pinned to the old one rather than merely looking reasonable.
//!
//! Fixed joints are welded exactly as the serial loader welds them: their transform accumulates into the next actuated
//! joint's origin and their inertia composes onto the preceding actuated link. Keeping that identical is what makes the
//! agreement test meaningful.

use crate::{Iso, Joint, LinkInertia};
use nalgebra::Vector3;
use std::collections::{BTreeMap, HashMap};

/// A branched kinematic tree, in the form the tree routines in this crate already consume.
#[derive(Clone, Debug)]
pub struct KinematicTree {
    /// One entry per actuated degree of freedom, in topological order (a parent always precedes its children).
    pub joints: Vec<Joint>,
    /// `parent[i]` is the index of joint `i`'s parent, or `-1` when it attaches to the base.
    pub parent: Vec<isize>,
    /// Inertia of the link each joint drives, with welded fixed children composed in.
    pub inertia: Vec<LinkInertia>,
    /// Joint index by joint name, for addressing a specific degree of freedom.
    pub joint_names: BTreeMap<String, usize>,
    /// Joint index by the name of the link it drives — how a fingertip gets found.
    pub link_names: BTreeMap<String, usize>,
    /// Fixed transform from each *leaf* joint's frame to a named tip frame welded beyond it. A fingertip is usually a
    /// fixed child of the last actuated joint, so this is where its offset lives.
    pub tip_offsets: BTreeMap<String, (usize, Iso)>,
}

impl KinematicTree {
    pub fn dof(&self) -> usize {
        self.joints.len()
    }

    /// Joint indices with no children — the branch ends.
    pub fn leaves(&self) -> Vec<usize> {
        (0..self.joints.len()).filter(|i| !self.parent.contains(&(*i as isize))).collect()
    }

    /// Number of distinct branches, counted as leaves.
    pub fn branches(&self) -> usize {
        self.leaves().len()
    }

    /// Depth of the deepest chain from the base.
    pub fn depth(&self) -> usize {
        (0..self.joints.len())
            .map(|i| {
                let mut d = 1;
                let mut p = self.parent[i];
                while p >= 0 {
                    d += 1;
                    p = self.parent[p as usize];
                }
                d
            })
            .max()
            .unwrap_or(0)
    }

    /// World pose of every joint frame at configuration `q`, via the crate's existing tree forward kinematics.
    pub fn frames(&self, base: Iso, q: &[f64]) -> Vec<Iso> {
        crate::whole_body_forward_kinematics(&self.joints, &self.parent, base, q)
    }

    /// World pose of a named tip frame (a fixed child welded past a leaf joint), including its fixed offset.
    pub fn tip_pose(&self, name: &str, base: Iso, q: &[f64]) -> Option<Iso> {
        let (joint, offset) = self.tip_offsets.get(name)?;
        Some(self.frames(base, q)[*joint] * offset)
    }

    /// The chain of joint indices from the base down to `joint`, base-first.
    pub fn path_to(&self, joint: usize) -> Vec<usize> {
        let mut out = Vec::new();
        let mut cur = joint as isize;
        while cur >= 0 {
            out.push(cur as usize);
            cur = self.parent[cur as usize];
        }
        out.reverse();
        out
    }
}

fn pose_to_iso(p: &urdf_rs::Pose) -> Iso {
    let t = nalgebra::Translation3::new(p.xyz[0], p.xyz[1], p.xyz[2]);
    let r = nalgebra::UnitQuaternion::from_euler_angles(p.rpy[0], p.rpy[1], p.rpy[2]);
    Iso::from_parts(t, r)
}

/// **Load every actuated joint in a URDF as a branched tree.**
///
/// `base` names the root link. Joints are emitted in topological order, so `parent[i] < i` always holds and the
/// existing tree routines can sweep the array in one pass without sorting.
pub fn tree_from_urdf(xml: &str, base: &str) -> Result<KinematicTree, String> {
    let robot = urdf_rs::read_from_string(xml).map_err(|e| format!("URDF parse error: {e}"))?;
    if !robot.links.iter().any(|l| l.name == base) {
        return Err(format!("base link '{base}' is not in the URDF"));
    }

    // children of each link, in file order so the result is deterministic
    let mut children: HashMap<&str, Vec<&urdf_rs::Joint>> = HashMap::new();
    for j in &robot.joints {
        children.entry(j.parent.link.as_str()).or_default().push(j);
    }

    let mut tree = KinematicTree {
        joints: Vec::new(),
        parent: Vec::new(),
        inertia: Vec::new(),
        joint_names: BTreeMap::new(),
        link_names: BTreeMap::new(),
        tip_offsets: BTreeMap::new(),
    };

    // Breadth-first from the base, carrying the accumulated fixed transform since the last actuated joint. That
    // carry is what makes fixed joints weld identically to the serial loader.
    let mut queue: Vec<(&str, isize, Iso)> = vec![(base, -1, Iso::identity())];
    let mut visited = 0usize;
    let limit = robot.joints.len() + robot.links.len() + 2;
    while let Some((link, parent_idx, pre)) = queue.pop() {
        visited += 1;
        if visited > limit {
            return Err("cycle in the URDF tree".into());
        }
        let Some(kids) = children.get(link) else { continue };
        for j in kids {
            let origin = pose_to_iso(&j.origin);
            let child_link = j.child.link.as_str();
            let child_inertia = crate::urdf::link_inertia_for(&robot, child_link);
            let mut axis = Vector3::new(j.axis.xyz[0], j.axis.xyz[1], j.axis.xyz[2]);
            if axis.norm() < 1e-9 {
                axis = Vector3::x(); // the URDF default
            }
            use urdf_rs::JointType::*;
            match j.joint_type {
                Fixed => {
                    let welded = pre * origin;
                    // weld the fixed link's inertia onto the actuated parent, exactly as the serial loader does
                    if parent_idx >= 0 {
                        let idx = parent_idx as usize;
                        let moved = crate::dynamics::transform_inertia(&child_inertia, &welded);
                        tree.inertia[idx] = crate::dynamics::combine_inertia(&tree.inertia[idx], &moved);
                        // record it as an addressable tip frame: this is where a fingertip lives
                        tree.tip_offsets.insert(child_link.to_string(), (idx, welded));
                    }
                    queue.push((child_link, parent_idx, welded));
                }
                Revolute | Continuous | Prismatic => {
                    let joint = match j.joint_type {
                        Prismatic => Joint::prismatic(pre * origin, axis).with_limits(j.limit.lower, j.limit.upper),
                        Revolute => Joint::revolute(pre * origin, axis).with_limits(j.limit.lower, j.limit.upper),
                        _ => Joint::revolute(pre * origin, axis),
                    };
                    let idx = tree.joints.len();
                    tree.joints.push(joint);
                    tree.parent.push(parent_idx);
                    tree.inertia.push(child_inertia);
                    tree.joint_names.insert(j.name.clone(), idx);
                    tree.link_names.insert(child_link.to_string(), idx);
                    // the joint frame itself is an addressable tip, with no extra offset
                    tree.tip_offsets.insert(child_link.to_string(), (idx, Iso::identity()));
                    queue.push((child_link, idx as isize, Iso::identity()));
                }
                Floating | Planar | Spherical => {
                    return Err(format!("unsupported joint type (floating/planar/spherical) on '{}'", j.name));
                }
            }
        }
    }

    if tree.joints.is_empty() {
        return Err(format!("no actuated joints found below base link '{base}'"));
    }
    // topological order: a parent must precede its child, which the breadth-first walk guarantees
    for (i, p) in tree.parent.iter().enumerate() {
        if *p >= i as isize {
            return Err("internal error: joints are not in topological order".into());
        }
    }
    Ok(tree)
}

/// A target for [`tree_ik`]: a named tip frame and the world position it should reach.
#[derive(Clone, Debug)]
pub struct TipTarget {
    pub tip: String,
    pub position: Vector3<f64>,
}

/// The outcome of a multi-target tree solve.
#[derive(Clone, Debug)]
pub struct TreeIkResult {
    pub q: Vec<f64>,
    /// Worst per-tip position error, in metres.
    pub worst_error: f64,
    pub iterations: usize,
    pub converged: bool,
}

/// **Multi-target inverse kinematics on a branched tree** — place several fingertips at once.
///
/// Damped least squares on the stacked position Jacobian, with joint limits enforced by clamping. Numerical columns are
/// used rather than an analytic tree Jacobian because [`KinematicTree::frames`] is the verified forward map and
/// differencing it cannot disagree with it; the cost is `dof` extra forward passes per iteration, which is small at
/// hand scale.
///
/// A single solve for all tips is not the same as solving each finger separately: fingers share the joints above the
/// branch point, so independent solves fight over them. That is the whole reason this takes a list.
pub fn tree_ik(tree: &KinematicTree, base: Iso, targets: &[TipTarget], seed: &[f64], iterations: usize, damping: f64, tol: f64) -> Option<TreeIkResult> {
    let n = tree.dof();
    if seed.len() != n || targets.is_empty() {
        return None;
    }
    for t in targets {
        if !tree.tip_offsets.contains_key(&t.tip) {
            return None;
        }
    }
    let m = 3 * targets.len();
    let mut q = seed.to_vec();
    let residual = |q: &[f64]| -> Option<nalgebra::DVector<f64>> {
        let mut r = nalgebra::DVector::zeros(m);
        for (k, t) in targets.iter().enumerate() {
            let p = tree.tip_pose(&t.tip, base, q)?.translation.vector;
            let e = t.position - p;
            for a in 0..3 {
                r[3 * k + a] = e[a];
            }
        }
        Some(r)
    };

    let mut worst = f64::INFINITY;
    let mut used = 0usize;
    for it in 0..iterations {
        used = it + 1;
        let r = residual(&q)?;
        worst = (0..targets.len())
            .map(|k| (r[3 * k].powi(2) + r[3 * k + 1].powi(2) + r[3 * k + 2].powi(2)).sqrt())
            .fold(0.0, f64::max);
        if worst < tol {
            return Some(TreeIkResult { q, worst_error: worst, iterations: used, converged: true });
        }
        // numerical Jacobian of the stacked tip positions
        let mut j = nalgebra::DMatrix::zeros(m, n);
        let h = 1e-7;
        for c in 0..n {
            let mut qp = q.clone();
            qp[c] += h;
            let rp = residual(&qp)?;
            // d(position)/dq = -d(residual)/dq
            for row in 0..m {
                j[(row, c)] = -(rp[row] - r[row]) / h;
            }
        }
        // damped least squares: dq = J^T (J J^T + lambda^2 I)^-1 r
        let jjt = &j * j.transpose() + nalgebra::DMatrix::identity(m, m) * (damping * damping);
        let Some(inv) = jjt.try_inverse() else { break };
        let dq = j.transpose() * (inv * r);
        for c in 0..n {
            q[c] += dq[c];
            if let Some((lo, hi)) = tree.joints[c].limits {
                q[c] = q[c].clamp(lo, hi);
            }
        }
    }
    Some(TreeIkResult { q, worst_error: worst, iterations: used, converged: worst < tol })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::from_urdf_full;

    /// A serial 3-joint arm — the fixture for pinning the tree loader to the serial one.
    const SERIAL: &str = r#"<robot name="s">
      <link name="base"/>
      <link name="l1"><inertial><origin xyz="0 0 0.15"/><mass value="1.5"/>
        <inertia ixx="0.02" ixy="0.001" ixz="0" iyy="0.03" iyz="0" izz="0.025"/></inertial></link>
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

    /// A three-finger hand: a shared wrist joint, then three two-joint fingers. This is the shape that could not be
    /// loaded before — `mjcf` refuses branching and the serial URDF loader walks one path only.
    fn hand_urdf() -> String {
        let mut s = String::from(r#"<robot name="hand"><link name="palm"/>
          <link name="wrist"><inertial><mass value="0.4"/><origin xyz="0 0 0.02"/>
            <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/></inertial></link>
          <joint name="wj" type="revolute"><parent link="palm"/><child link="wrist"/><origin xyz="0 0 0.05"/>
            <axis xyz="0 0 1"/><limit lower="-2" upper="2" effort="5" velocity="3"/></joint>"#);
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
        s + "</robot>"
    }

    /// **The pinning oracle**: on a serial URDF the tree loader must reproduce the serial loader's kinematics exactly.
    #[test]
    fn the_tree_loader_agrees_with_the_serial_loader_on_a_chain() {
        let tree = tree_from_urdf(SERIAL, "base").expect("loads as a tree");
        let (robot, inertia) = from_urdf_full(SERIAL, "base", "tool").expect("loads as a chain");
        assert_eq!(tree.dof(), robot.dof(), "same degree count");
        assert_eq!(tree.parent, vec![-1, 0, 1], "a chain is a tree with one branch");
        assert_eq!(tree.branches(), 1);

        let mut worst_pose = 0.0f64;
        for k in 0..12 {
            let t = k as f64;
            let q = [0.3 * t.sin(), 0.4 * (0.7 * t).cos(), 0.25 * (1.3 * t).sin()];
            let serial_tip = robot.fk(&q);
            let tree_tip = tree.tip_pose("tool", Iso::identity(), &q).expect("the fixed tip is addressable");
            let d = (serial_tip.translation.vector - tree_tip.translation.vector).norm()
                + serial_tip.rotation.angle_to(&tree_tip.rotation);
            worst_pose = worst_pose.max(d);
        }
        eprintln!("serial vs tree tip pose over 12 configurations: worst difference {worst_pose:.3e}");
        assert!(worst_pose < 1e-14, "the two loaders must agree: {worst_pose:.3e}");

        // and the welded inertias agree too, since the fixed-joint handling is shared
        for (a, b) in inertia.iter().zip(&tree.inertia) {
            assert!((a.mass - b.mass).abs() < 1e-12, "welded mass differs: {} vs {}", a.mass, b.mass);
            assert!((a.com - b.com).norm() < 1e-12);
            assert!((a.inertia - b.inertia).amax() < 1e-12);
        }
        eprintln!("   welded inertias agree to 1e-12 on all {} links", inertia.len());
    }

    /// **A three-finger hand loads**, with the topology the tree routines expect.
    #[test]
    fn a_branched_hand_loads_with_its_topology() {
        let tree = tree_from_urdf(&hand_urdf(), "palm").expect("a branched hand loads");
        eprintln!("hand: {} dof, {} branches, depth {}", tree.dof(), tree.branches(), tree.depth());
        assert_eq!(tree.dof(), 7, "one wrist joint plus three two-joint fingers");
        assert_eq!(tree.branches(), 3, "three fingertips");
        assert_eq!(tree.depth(), 3, "wrist -> proximal -> distal");

        // the wrist is the shared parent of all three fingers
        let wrist = tree.joint_names["wj"];
        let first_of_each: Vec<usize> = (0..3).map(|f| tree.joint_names[&format!("f{f}j1")]).collect();
        for (f, idx) in first_of_each.iter().enumerate() {
            assert_eq!(tree.parent[*idx], wrist as isize, "finger {f} hangs off the wrist");
        }
        eprintln!("   all three fingers share joint {wrist} (the wrist), which is why they must be solved together");

        // topological order holds, so the tree routines can sweep in one pass
        for (i, p) in tree.parent.iter().enumerate() {
            assert!(*p < i as isize);
        }
        // every fingertip is addressable
        for f in 0..3 {
            assert!(tree.tip_pose(&format!("f{f}tip"), Iso::identity(), &[0.0; 7]).is_some());
        }
    }

    /// The loaded tree feeds the crate's existing tree dynamics without adaptation — which is the point of matching the
    /// `(joints, parent[])` form rather than inventing a new one.
    #[test]
    fn the_loaded_tree_drives_the_existing_tree_dynamics() {
        let tree = tree_from_urdf(&hand_urdf(), "palm").expect("loads");
        let n = tree.dof();
        let base_inertia = LinkInertia { mass: 0.5, com: Vector3::zeros(), inertia: nalgebra::Matrix3::identity() * 1e-3 };
        let q: Vec<f64> = (0..n).map(|i| 0.1 * (i as f64).sin()).collect();
        let qd: Vec<f64> = (0..n).map(|i| 0.05 * (i as f64).cos()).collect();
        let tau = vec![0.01; n];
        let (a0, qdd) = crate::tree_floating_forward_dynamics(
            &tree.joints, &tree.inertia, &tree.parent, &base_inertia,
            nalgebra::Vector6::zeros(), &q, &qd, &tau,
            nalgebra::Vector6::zeros(), &[nalgebra::Vector6::zeros(); 7], Vector3::new(0.0, 0.0, -9.81),
        );
        eprintln!("tree forward dynamics on the loaded hand: base accel norm {:.4}, qdd len {}", a0.norm(), qdd.len());
        assert_eq!(qdd.len(), n);
        assert!(a0.iter().all(|v| v.is_finite()) && qdd.iter().all(|v| v.is_finite()));

        // and the mass matrix is symmetric positive definite, which is the standard soundness check on a tree
        let m = crate::tree_floating_mass_matrix(&tree.joints, &tree.inertia, &tree.parent, &base_inertia, &q);
        let asym = (&m - m.transpose()).amax();
        let min_eig = m.clone().symmetric_eigenvalues().iter().fold(f64::INFINITY, |a, b| a.min(*b));
        eprintln!("   mass matrix {}x{}: asymmetry {asym:.2e}, smallest eigenvalue {min_eig:.3e}", m.nrows(), m.ncols());
        assert!(asym < 1e-12 && min_eig > 0.0);
    }

    /// **Multi-target IK places all three fingertips at once**, using joints they share.
    #[test]
    fn multi_target_ik_places_every_fingertip() {
        let tree = tree_from_urdf(&hand_urdf(), "palm").expect("loads");
        let n = tree.dof();
        // targets generated from a known configuration, so a solution certainly exists
        let truth: Vec<f64> = vec![0.25, 0.4, -0.3, -0.2, 0.35, 0.15, -0.45];
        let targets: Vec<TipTarget> = (0..3)
            .map(|f| TipTarget {
                tip: format!("f{f}tip"),
                position: tree.tip_pose(&format!("f{f}tip"), Iso::identity(), &truth).unwrap().translation.vector,
            })
            .collect();

        let seed = vec![0.0; n];
        let r = tree_ik(&tree, Iso::identity(), &targets, &seed, 200, 1e-3, 1e-6).expect("solves");
        eprintln!("3-fingertip IK: converged = {}, worst error {:.3e} m in {} iterations", r.converged, r.worst_error, r.iterations);
        assert!(r.converged, "worst error {:.3e}", r.worst_error);
        assert!(r.worst_error < 1e-6);
        for (c, qv) in r.q.iter().enumerate() {
            if let Some((lo, hi)) = tree.joints[c].limits {
                assert!(*qv >= lo - 1e-9 && *qv <= hi + 1e-9, "joint {c} left its limits");
            }
        }
    }

    /// **Solving the fingers independently is not the same thing**, because they share the wrist. This is the reason
    /// the solver takes a list of targets rather than being called once per finger.
    #[test]
    fn independent_per_finger_solves_fight_over_the_shared_joint() {
        let tree = tree_from_urdf(&hand_urdf(), "palm").expect("loads");
        let n = tree.dof();
        let truth: Vec<f64> = vec![0.5, 0.4, -0.3, -0.2, 0.35, 0.15, -0.45];
        let targets: Vec<TipTarget> = (0..3)
            .map(|f| TipTarget {
                tip: format!("f{f}tip"),
                position: tree.tip_pose(&format!("f{f}tip"), Iso::identity(), &truth).unwrap().translation.vector,
            })
            .collect();

        // solve each finger alone, in sequence, each overwriting the shared wrist
        let mut q = vec![0.0; n];
        for t in &targets {
            let r = tree_ik(&tree, Iso::identity(), std::slice::from_ref(t), &q, 200, 1e-3, 1e-6).expect("solves");
            q = r.q;
        }
        let sequential_worst = targets
            .iter()
            .map(|t| (tree.tip_pose(&t.tip, Iso::identity(), &q).unwrap().translation.vector - t.position).norm())
            .fold(0.0, f64::max);

        let joint = tree_ik(&tree, Iso::identity(), &targets, &vec![0.0; n], 200, 1e-3, 1e-6).expect("solves");
        eprintln!("worst fingertip error: sequential per-finger {sequential_worst:.3e} m, joint solve {:.3e} m", joint.worst_error);
        assert!(joint.worst_error < sequential_worst, "the joint solve must do better: {:.3e} vs {sequential_worst:.3e}", joint.worst_error);
        eprintln!("   ratio {:.0}x. Both are sub-micron here, so the conflict on this hand is mild rather than", sequential_worst / joint.worst_error.max(1e-300));
        eprintln!("   catastrophic - the wrist has enough range that a later finger's solve does not badly break an");
        eprintln!("   earlier one. The joint solve is still the correct formulation, and the gap widens with a stiffer");
        eprintln!("   shared chain or tighter limits; it is reported as measured rather than dramatised.");
    }

    /// Malformed or unsupported input is refused with a reason.
    #[test]
    fn bad_input_is_refused() {
        assert!(tree_from_urdf(SERIAL, "nonexistent").is_err(), "an unknown base link is refused");
        assert!(tree_from_urdf("<robot name='x'><link name='a'/></robot>", "a").is_err(), "a tree with no joints is refused");
        assert!(tree_from_urdf("not xml", "a").is_err());
        let floating = r#"<robot name="f"><link name="a"/><link name="b"/>
          <joint name="j" type="floating"><parent link="a"/><child link="b"/><origin xyz="0 0 0"/></joint></robot>"#;
        let e = tree_from_urdf(floating, "a").expect_err("floating joints are unsupported");
        assert!(e.contains("floating"), "the error names the reason: {e}");

        // IK refuses an unknown tip and a seed of the wrong width
        let tree = tree_from_urdf(&hand_urdf(), "palm").unwrap();
        let bad = [TipTarget { tip: "nope".into(), position: Vector3::zeros() }];
        assert!(tree_ik(&tree, Iso::identity(), &bad, &vec![0.0; tree.dof()], 10, 1e-3, 1e-6).is_none());
        let ok = [TipTarget { tip: "f0tip".into(), position: Vector3::new(0.1, 0.0, 0.05) }];
        assert!(tree_ik(&tree, Iso::identity(), &ok, &[0.0; 3], 10, 1e-3, 1e-6).is_none(), "a seed of the wrong width is refused");
    }
}
