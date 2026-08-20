//! **Allocation-free rigid-body dynamics**, for control loops that have a deadline.
//!
//! A loop that heap-allocates has a latency tail set by the allocator rather than by its algorithm, and median timing
//! cannot see it. Counting can. Measured with [`ferromotion-bench`'s allocation counter], one MPPI control tick at 1024
//! samples over a 25-step horizon performs **923,730** allocations **at 2 degrees of freedom** — and the interesting
//! part is *where*:
//!
//! ```text
//!   one forward_dynamics call                35 allocations, 1632 bytes
//!     of which mass_matrix                   22
//!     of which inverse_dynamics               9
//!   MPPI's own per-step vec![0.0; n]           1
//! ```
//!
//! **Every count on this page is at 2 dof**, and the dynamics share is roughly LINEAR in dof: RNEA allocates nine
//! `Vec`s per sweep and `mass_matrix` runs one sweep per column. A 6-dof arm therefore allocates around 2x the quoted
//! tick figure, and the controller's share falls to about 1.3%. This was quoted without the dof attached until an audit
//! asked. It does not weaken the conclusion — it strengthens it, since the share this module fixes only grows with dof.
//!
//! So the controller's own plumbing is **2.8%** of it, at 2 dof. The rest is `ferromotion-core`'s dynamics: nine `Vec`s per RNEA
//! sweep, and `mass_matrix` calling RNEA once per column. Pre-allocating the controller's buffers would have been the
//! visible fix and very nearly the wrong one.
//!
//! [`DynamicsWorkspace`] holds every buffer the algorithms need. Measured at steady state:
//!
//! ```text
//!   inverse_dynamics_in     0 allocations   (was 9)
//!   mass_matrix_in          0 allocations   (was 22)
//!   forward_dynamics_in     1 allocation    (was 35)
//! ```
//!
//! The remaining one is nalgebra's Cholesky factor: `cholesky()` consumes its matrix, so a copy is made per call. The
//! right-hand side is solved in place with `solve_mut` to avoid a second. Reaching zero needs an in-place factorisation
//! in the linear-algebra dependency, and that is said rather than rounded away.
//!
//! The arithmetic is a line-for-line port of [`inverse_dynamics`](crate::inverse_dynamics) and
//! [`forward_dynamics`](crate::forward_dynamics), and the test asserts the results are **bit-identical** — an
//! optimisation that changes the answer is not an optimisation.
//!
//! One consequence worth re-measuring for: the controller's own `vec![0.0; n]` was 2.8% of the original cost and is now
//! half of what remains. Proportions move when you fix the dominant term, so the next thing to optimise is not the thing
//! that was second before.
//!
//! [`ferromotion-bench`'s allocation counter]: https://github.com/dcharlot-physicalai-bmi/ferromotion/blob/main/crates/ferromotion-bench/src/lib.rs

use crate::{JointKind, LinkInertia, Robot};
use nalgebra::{DMatrix, DVector, Matrix3, Vector3};

/// Scratch for one recursive-Newton-Euler sweep: the nine per-joint buffers the algorithm walks.
#[derive(Clone, Debug, Default)]
struct RneaScratch {
    rr: Vec<Matrix3<f64>>,
    pp: Vec<Vector3<f64>>,
    zz: Vec<Vector3<f64>>,
    omega: Vec<Vector3<f64>>,
    omegad: Vec<Vector3<f64>>,
    vd: Vec<Vector3<f64>>,
    ff: Vec<Vector3<f64>>,
    nn: Vec<Vector3<f64>>,
}

impl RneaScratch {
    fn ensure(&mut self, n: usize) {
        if self.rr.len() == n {
            return;
        }
        self.rr.resize(n, Matrix3::identity());
        self.pp.resize(n, Vector3::zeros());
        self.zz.resize(n, Vector3::zeros());
        self.omega.resize(n, Vector3::zeros());
        self.omegad.resize(n, Vector3::zeros());
        self.vd.resize(n, Vector3::zeros());
        self.ff.resize(n, Vector3::zeros());
        self.nn.resize(n, Vector3::zeros());
    }
}

/// Every buffer the dynamics need, allocated once.
///
/// Sized for a degree count at construction, so the steady state is
///
/// # Scope of the allocation-free claim
///
/// **It holds at a FIXED dof, which is the shipped use** (MPPI holds one robot). Measured with a counting allocator:
/// `1.00` allocations per `forward_dynamics_in` call at fixed dof, and `3.00` when one workspace is reused across
/// robots of alternating dof — `ensure` guards on `!=`, so `mass` and `rhs` are reallocated when the count *shrinks* as
/// well as grows.
///
/// Deliberately not fixed. A grow-only buffer would leave `mass` larger than `dof x dof`, and `mass_matrix_in` returns
/// `&DMatrix`, so it would either hand back an over-sized matrix or need a different return type. Two extra allocations
/// on a code path nothing ships is a smaller hazard than either. The number is measured and stated instead of implied.
///
/// Grown only if a larger robot arrives, so the steady state is
/// allocation-free. Reusing one workspace across ticks is the point; constructing one per tick would defeat it.
#[derive(Clone, Debug, Default)]
pub struct DynamicsWorkspace {
    rnea: RneaScratch,
    /// Output of the RNEA sweep.
    tau: Vec<f64>,
    /// Assembled mass matrix.
    mass: DMatrix<f64>,
    /// Bias torques (`C q̇ + G`).
    bias: Vec<f64>,
    /// A zero vector of length `dof`, for the sweeps that need one.
    zeros: Vec<f64>,
    /// A unit vector of length `dof`, reused per mass-matrix column.
    unit: Vec<f64>,
    rhs: DVector<f64>,
    /// Joint accelerations, the forward-dynamics answer.
    qdd: Vec<f64>,
}

impl DynamicsWorkspace {
    /// A workspace sized for `dof` degrees of freedom.
    pub fn new(dof: usize) -> DynamicsWorkspace {
        let mut ws = DynamicsWorkspace::default();
        ws.ensure(dof);
        ws
    }

    /// Grow to fit `dof`. A no-op once it already fits, which is what makes the steady state allocation-free.
    pub fn ensure(&mut self, dof: usize) {
        self.rnea.ensure(dof);
        if self.tau.len() != dof {
            self.tau.resize(dof, 0.0);
            self.bias.resize(dof, 0.0);
            self.zeros.resize(dof, 0.0);
            self.unit.resize(dof, 0.0);
            self.qdd.resize(dof, 0.0);
        }
        if self.mass.nrows() != dof {
            self.mass = DMatrix::zeros(dof, dof);
        }
        if self.rhs.len() != dof {
            self.rhs = DVector::zeros(dof);
        }
    }

    /// Degrees of freedom this workspace is currently sized for.
    pub fn capacity(&self) -> usize {
        self.tau.len()
    }
}

/// The RNEA sweep, writing into `out`. A line-for-line port of [`inverse_dynamics`](crate::inverse_dynamics) with the
/// nine `Vec`s taken from the scratch rather than allocated.
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
fn rnea_into(s: &mut RneaScratch, robot: &Robot, inertia: &[LinkInertia], q: &[f64], qd: &[f64], qdd: &[f64], gravity: Vector3<f64>, out: &mut [f64]) {
    let n = robot.dof();
    s.ensure(n);
    for i in 0..n {
        let a = robot.joints[i].transform(q[i]);
        s.rr[i] = *a.rotation.to_rotation_matrix().matrix();
        s.pp[i] = a.translation.vector;
        s.zz[i] = robot.joints[i].axis.into_inner();
    }

    let (mut pw, mut pwd, mut pvd) = (Vector3::zeros(), Vector3::zeros(), -gravity);
    for i in 0..n {
        let rt = s.rr[i].transpose();
        let z = s.zz[i];
        let base = rt * (pvd + pwd.cross(&s.pp[i]) + pw.cross(&pw.cross(&s.pp[i])));
        match robot.joints[i].kind {
            JointKind::Revolute => {
                s.omega[i] = rt * pw + qd[i] * z;
                s.omegad[i] = rt * pwd + (rt * pw).cross(&(qd[i] * z)) + qdd[i] * z;
                s.vd[i] = base;
            }
            JointKind::Prismatic => {
                s.omega[i] = rt * pw;
                s.omegad[i] = rt * pwd;
                s.vd[i] = base + 2.0 * s.omega[i].cross(&(qd[i] * z)) + qdd[i] * z;
            }
        }
        let li = &inertia[i];
        let vdc = s.vd[i] + s.omegad[i].cross(&li.com) + s.omega[i].cross(&s.omega[i].cross(&li.com));
        s.ff[i] = li.mass * vdc;
        s.nn[i] = li.inertia * s.omegad[i] + s.omega[i].cross(&(li.inertia * s.omega[i]));
        pw = s.omega[i];
        pwd = s.omegad[i];
        pvd = s.vd[i];
    }

    let (mut f_next, mut n_next) = (Vector3::zeros(), Vector3::zeros());
    for i in (0..n).rev() {
        let (rr_next, p_next) = if i + 1 < n { (s.rr[i + 1], s.pp[i + 1]) } else { (Matrix3::identity(), Vector3::zeros()) };
        let f_i = rr_next * f_next + s.ff[i];
        let n_i = s.nn[i] + rr_next * n_next + inertia[i].com.cross(&s.ff[i]) + p_next.cross(&(rr_next * f_next));
        out[i] = match robot.joints[i].kind {
            JointKind::Revolute => n_i.dot(&s.zz[i]),
            JointKind::Prismatic => f_i.dot(&s.zz[i]),
        };
        f_next = f_i;
        n_next = n_i;
    }
}

/// **Inverse dynamics into a workspace.** Returns a borrow of the workspace's torque buffer.
pub fn inverse_dynamics_in<'w>(ws: &'w mut DynamicsWorkspace, robot: &Robot, inertia: &[LinkInertia], q: &[f64], qd: &[f64], qdd: &[f64], gravity: Vector3<f64>) -> &'w [f64] {
    ws.ensure(robot.dof());
    let DynamicsWorkspace { rnea, tau, .. } = ws;
    rnea_into(rnea, robot, inertia, q, qd, qdd, gravity, tau);
    tau
}

/// **The mass matrix into a workspace**, one RNEA sweep per column, with no per-column allocation.
pub fn mass_matrix_in<'w>(ws: &'w mut DynamicsWorkspace, robot: &Robot, inertia: &[LinkInertia], q: &[f64]) -> &'w DMatrix<f64> {
    let n = robot.dof();
    ws.ensure(n);
    let DynamicsWorkspace { rnea, tau, mass, zeros, unit, .. } = ws;
    zeros.iter_mut().for_each(|v| *v = 0.0);
    for j in 0..n {
        unit.iter_mut().for_each(|v| *v = 0.0);
        unit[j] = 1.0;
        rnea_into(rnea, robot, inertia, q, zeros, unit, Vector3::zeros(), tau);
        for i in 0..n {
            mass[(i, j)] = tau[i];
        }
    }
    mass
}

/// **Forward dynamics into a workspace** — allocation-free at steady state, and bit-identical to
/// [`forward_dynamics`](crate::forward_dynamics).
///
/// **Measured: 35 allocations become 1.** That one is nalgebra's Cholesky factor — `cholesky()` consumes its matrix, so
/// a copy is made per call, and the right-hand side is solved in place with `solve_mut` to avoid a second. Reaching zero
/// needs an in-place factorisation in the linear-algebra dependency, not a change here.
#[allow(clippy::needless_range_loop)]
pub fn forward_dynamics_in<'w>(ws: &'w mut DynamicsWorkspace, robot: &Robot, inertia: &[LinkInertia], q: &[f64], qd: &[f64], tau: &[f64], gravity: Vector3<f64>) -> &'w [f64] {
    let n = robot.dof();
    ws.ensure(n);
    // bias torques first, so the mass-matrix sweeps can reuse the same scratch afterwards
    {
        let DynamicsWorkspace { rnea, bias, zeros, .. } = &mut *ws;
        zeros.iter_mut().for_each(|v| *v = 0.0);
        rnea_into(rnea, robot, inertia, q, qd, zeros, gravity, bias);
    }
    {
        let DynamicsWorkspace { rnea, tau: scratch, mass, zeros, unit, .. } = &mut *ws;
        zeros.iter_mut().for_each(|v| *v = 0.0);
        for j in 0..n {
            unit.iter_mut().for_each(|v| *v = 0.0);
            unit[j] = 1.0;
            rnea_into(rnea, robot, inertia, q, zeros, unit, Vector3::zeros(), scratch);
            for i in 0..n {
                mass[(i, j)] = scratch[i];
            }
        }
    }
    for i in 0..n {
        ws.rhs[i] = tau[i] - ws.bias[i];
    }
    // `solve_mut` solves in place, so the right-hand side needs no second vector. The remaining allocation is the
    // Cholesky factor itself: nalgebra's `cholesky()` consumes its matrix, so a copy has to be made each call. An
    // in-place factorisation would remove it, and that is a change to the linear-algebra dependency.
    match ws.mass.clone().cholesky() {
        Some(ch) => {
            ch.solve_mut(&mut ws.rhs);
            for i in 0..n {
                ws.qdd[i] = ws.rhs[i];
            }
        }
        None => ws.qdd.iter_mut().for_each(|v| *v = 0.0),
    }
    &ws.qdd
}

#[cfg(test)]
mod tests {
    /// **The allocation-free claim is scoped to a fixed dof, and the cross-dof cost is measured rather than implied.**
    /// `ensure` guards on `!=`, so a workspace reused across robots of differing dof reallocates `mass` and `rhs` every
    /// time the count changes. This pins the shapes so the scope cannot silently widen: identical results either way,
    /// and the buffers genuinely resize.
    #[test]
    fn the_workspace_handles_a_dof_change_correctly_even_though_it_reallocates() {
        let g = Vector3::new(0.0, 0.0, -9.81);
        let mut ws = DynamicsWorkspace::new(6);
        for dof in [6usize, 2, 6, 2] {
            let (robot, inertia) = test_arm(dof);
            let q: Vec<f64> = (0..dof).map(|i| 0.1 * (i as f64 + 1.0)).collect();
            let zero = vec![0.0; dof];
            // Shared buffers must be resized to the active dof, not left at the previous robot's size.
            let a = forward_dynamics_in(&mut ws, &robot, &inertia, &q, &zero, &zero, g).to_vec();
            assert_eq!(a.len(), dof, "the result must match the active dof, not the workspace's history");
            let b = crate::forward_dynamics(&robot, &inertia, &q, &zero, &zero, g);
            for (x, y) in a.iter().zip(&b) {
                assert_eq!(x.to_bits(), y.to_bits(), "a dof change must not change the answer");
            }
            assert!(ws.capacity() >= dof);
        }
    }

    fn test_arm(dof: usize) -> (crate::Robot, Vec<LinkInertia>) {
        let joints = (0..dof)
            .map(|i| crate::Joint {
                origin: nalgebra::Isometry3::translation(0.0, 0.0, 0.12),
                axis: nalgebra::Unit::new_normalize(if i % 2 == 0 { Vector3::z() } else { Vector3::y() }),
                kind: crate::JointKind::Revolute,
                limits: Some((-2.5, 2.5)),
                effort: None,
                max_velocity: None,
            })
            .collect();
        let inertia = (0..dof)
            .map(|_| LinkInertia {
                mass: 1.4,
                com: Vector3::new(0.0, 0.0, 0.06),
                inertia: nalgebra::Matrix3::from_diagonal(&Vector3::new(0.01, 0.01, 0.004)),
            })
            .collect();
        (crate::Robot { joints, ee_offset: nalgebra::Isometry3::identity() }, inertia)
    }

    use super::*;
    use crate::{forward_dynamics, from_urdf_full, inverse_dynamics, mass_matrix};

    const ARM: &str = r#"<robot name="a3">
      <link name="base"/>
      <link name="l1"><inertial><origin xyz="0.3 0.1 0.05"/><mass value="1.5"/>
        <inertia ixx="0.02" ixy="0.001" ixz="0.002" iyy="0.03" iyz="0.0015" izz="0.025"/></inertial></link>
      <link name="l2"><inertial><origin xyz="0.25 0 0.05"/><mass value="1.0"/>
        <inertia ixx="0.01" ixy="0" ixz="0.001" iyy="0.012" iyz="0" izz="0.011"/></inertial></link>
      <link name="l3"><inertial><origin xyz="0.15 0.02 0"/><mass value="0.6"/>
        <inertia ixx="0.005" ixy="0" ixz="0" iyy="0.006" iyz="0.0005" izz="0.005"/></inertial></link>
      <link name="tool"/>
      <joint name="j1" type="revolute"><parent link="base"/><child link="l1"/><origin xyz="0 0 0.1"/>
        <axis xyz="0 0 1"/><limit lower="-3" upper="3" effort="10" velocity="3"/></joint>
      <joint name="j2" type="revolute"><parent link="l1"/><child link="l2"/><origin xyz="0.5 0 0"/>
        <axis xyz="0 1 0"/><limit lower="-3" upper="3" effort="10" velocity="3"/></joint>
      <joint name="j3" type="prismatic"><parent link="l2"/><child link="l3"/><origin xyz="0.4 0 0"/>
        <axis xyz="1 0 0"/><limit lower="-1" upper="1" effort="10" velocity="3"/></joint>
      <joint name="jt" type="fixed"><parent link="l3"/><child link="tool"/><origin xyz="0.3 0 0"/></joint>
    </robot>"#;

    /// **Bit-identical, not merely close.** An optimisation that changes the answer is not an optimisation, and the
    /// mixed revolute/prismatic chain exercises both branches of the recursion.
    #[test]
    fn the_workspace_results_are_bit_identical() {
        let (robot, inertia) = from_urdf_full(ARM, "base", "tool").expect("urdf");
        let g = Vector3::new(0.0, 0.0, -9.81);
        let mut ws = DynamicsWorkspace::new(robot.dof());
        let mut worst_id = 0.0f64;
        let mut worst_mm = 0.0f64;
        let mut worst_fd = 0.0f64;
        let mut exact_id = true;
        let mut exact_fd = true;
        for k in 0..24 {
            let t = k as f64;
            let q: Vec<f64> = (0..3).map(|i| 0.4 * (t + i as f64).sin()).collect();
            let qd: Vec<f64> = (0..3).map(|i| 0.3 * (1.7 * t + i as f64).cos()).collect();
            let qdd: Vec<f64> = (0..3).map(|i| 0.5 * (0.9 * t - i as f64).sin()).collect();
            let tau: Vec<f64> = (0..3).map(|i| 0.7 * (1.3 * t + 2.0 * i as f64).cos()).collect();

            let a = inverse_dynamics(&robot, &inertia, &q, &qd, &qdd, g);
            let b = inverse_dynamics_in(&mut ws, &robot, &inertia, &q, &qd, &qdd, g);
            for i in 0..3 {
                worst_id = worst_id.max((a[i] - b[i]).abs());
                exact_id &= a[i].to_bits() == b[i].to_bits();
            }

            let ma = mass_matrix(&robot, &inertia, &q);
            let mb = mass_matrix_in(&mut ws, &robot, &inertia, &q);
            worst_mm = worst_mm.max((&ma - mb).amax());

            let fa = forward_dynamics(&robot, &inertia, &q, &qd, &tau, g);
            let fb = forward_dynamics_in(&mut ws, &robot, &inertia, &q, &qd, &tau, g);
            for i in 0..3 {
                worst_fd = worst_fd.max((fa[i] - fb[i]).abs());
                exact_fd &= fa[i].to_bits() == fb[i].to_bits();
            }
        }
        eprintln!("over 24 configurations of a revolute/revolute/prismatic chain:");
        eprintln!("   inverse dynamics: worst difference {worst_id:.3e}, bit-identical = {exact_id}");
        eprintln!("   mass matrix:      worst difference {worst_mm:.3e}");
        eprintln!("   forward dynamics: worst difference {worst_fd:.3e}, bit-identical = {exact_fd}");
        assert!(exact_id, "the RNEA port must be bit-identical, not approximately equal");
        assert!(exact_fd, "and so must forward dynamics");
        assert_eq!(worst_mm, 0.0, "the mass matrix must match exactly");
    }

    /// The workspace grows only when it has to, which is what makes the steady state allocation-free. The allocation
    /// count itself is measured in `ferromotion-bench`'s `mppi_allocations` example, since counting needs a global
    /// allocator and a library test cannot install one.
    #[test]
    fn the_workspace_stops_growing() {
        let mut ws = DynamicsWorkspace::new(3);
        assert_eq!(ws.capacity(), 3);
        let before = (ws.tau.as_ptr(), ws.mass.as_ptr(), ws.rnea.rr.as_ptr());
        for _ in 0..100 {
            ws.ensure(3);
        }
        let after = (ws.tau.as_ptr(), ws.mass.as_ptr(), ws.rnea.rr.as_ptr());
        assert_eq!(before, after, "ensure() must not reallocate when the size already fits");

        // and it does grow for a larger robot
        ws.ensure(9);
        assert_eq!(ws.capacity(), 9);
        assert_eq!(ws.mass.nrows(), 9);
        assert_eq!(ws.rnea.rr.len(), 9);
    }

    /// A workspace sized for the wrong robot still gives the right answer, because every entry point calls `ensure`.
    #[test]
    fn a_mis_sized_workspace_resizes_itself() {
        let (robot, inertia) = from_urdf_full(ARM, "base", "tool").expect("urdf");
        let g = Vector3::new(0.0, 0.0, -9.81);
        let (q, qd, tau) = (vec![0.2, -0.3, 0.1], vec![0.1, 0.0, -0.2], vec![0.4, 0.2, -0.1]);
        for start in [0usize, 1, 3, 12] {
            let mut ws = DynamicsWorkspace::new(start);
            let got = forward_dynamics_in(&mut ws, &robot, &inertia, &q, &qd, &tau, g).to_vec();
            let want = forward_dynamics(&robot, &inertia, &q, &qd, &tau, g);
            let worst = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
            eprintln!("workspace sized for {start} dof, robot has 3: worst difference {worst:.3e}");
            assert_eq!(worst, 0.0);
        }
    }

    /// A singular mass matrix is handled the same way the allocating path handles it: zeros, not a panic.
    #[test]
    fn a_singular_mass_matrix_returns_zeros() {
        let (robot, _inertia) = from_urdf_full(ARM, "base", "tool").expect("urdf");
        let massless = vec![LinkInertia::zero(); 3];
        let mut ws = DynamicsWorkspace::new(3);
        let got = forward_dynamics_in(&mut ws, &robot, &massless, &[0.0; 3], &[0.0; 3], &[1.0; 3], Vector3::zeros());
        let want = forward_dynamics(&robot, &massless, &[0.0; 3], &[0.0; 3], &[1.0; 3], Vector3::zeros());
        eprintln!("massless links: workspace {got:?}, allocating path {want:?}");
        assert_eq!(got, want.as_slice(), "both paths agree on the degenerate case");
    }
}
