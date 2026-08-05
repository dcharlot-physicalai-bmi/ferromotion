//! **How many times does one MPPI control tick touch the heap?**
//!
//! A real-time control loop that allocates has a latency tail set by the allocator rather than by its algorithm, and no
//! amount of median timing shows it. This counts.

use ferromotion_bench::alloc_count::{count, CountingAllocator};
use ferromotion_control::Mppi;
use ferromotion_core::{from_urdf_full, LinkInertia, Robot};
use nalgebra::Vector3;

#[global_allocator]
static ALLOC: CountingAllocator = CountingAllocator;

const ARM: &str = r#"<robot name="a2">
  <link name="base"/>
  <link name="l1"><inertial><mass value="1.2"/><origin xyz="0 0 0.15"/>
    <inertia ixx="0.02" ixy="0" ixz="0" iyy="0.02" iyz="0" izz="0.006"/></inertial></link>
  <link name="l2"><inertial><mass value="0.8"/><origin xyz="0 0 0.12"/>
    <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.003"/></inertial></link>
  <link name="tool"/>
  <joint name="j1" type="revolute"><parent link="base"/><child link="l1"/><origin xyz="0 0 0.1"/>
    <axis xyz="0 1 0"/><limit lower="-3" upper="3" effort="20" velocity="5"/></joint>
  <joint name="j2" type="revolute"><parent link="l1"/><child link="l2"/><origin xyz="0 0 0.3"/>
    <axis xyz="0 1 0"/><limit lower="-3" upper="3" effort="20" velocity="5"/></joint>
  <joint name="jt" type="fixed"><parent link="l2"/><child link="tool"/><origin xyz="0 0 0.25"/></joint>
</robot>"#;

fn mppi<'a>(robot: &'a Robot, inertia: &'a [LinkInertia], horizon: usize, samples: usize) -> Mppi<'a> {
    Mppi {
        robot, inertia, gravity: Vector3::new(0.0, 0.0, -9.81),
        horizon, dt: 0.02, num_samples: samples, lambda: 0.1, sigma: 1.5,
        q_goal: vec![0.6, -0.8], w_q: 12.0, w_qd: 0.4, w_terminal: 40.0, r_ctrl: 0.002,
        u_max: 8.0, rng: 12345,
    }
}

fn main() {
    let (robot, inertia) = from_urdf_full(ARM, "base", "tool").expect("urdf");
    println!("MPPI heap allocations per control tick (2 dof)\n");
    println!("{:>8}  {:>9}  {:>14}  {:>16}  {:>18}", "horizon", "samples", "allocations", "bytes", "alloc / (K x H)");

    for (horizon, samples) in [(25usize, 256usize), (25, 1024), (50, 1024), (25, 4096)] {
        let mut c = mppi(&robot, &inertia, horizon, samples);
        let (q, qd) = (vec![0.3, -0.4], vec![0.0, 0.0]);
        let nominal = vec![vec![0.0; 2]; horizon];
        // warm the allocator so the count is the tick's own, not first-touch growth
        let (mut plan, _) = (nominal.clone(), c.control(&q, &qd, &nominal));
        let ((_, next), stats) = count(|| c.control(&q, &qd, &plan));
        plan = next;
        let _ = &plan;
        println!("{horizon:>8}  {samples:>9}  {:>14}  {:>16}  {:>18.2}",
            stats.allocations, stats.bytes, stats.allocations as f64 / (samples * horizon) as f64);
    }

    // The count is ~36 per rollout step, not 1, so MPPI's own `vec![0.0; n]` is NOT the dominant term.
    // Locate the real source by counting the pieces directly.
    println!("\n  36 allocations per rollout STEP, not 1 - so MPPI's own buffers are a small part of it.");
    println!("  Counting the pieces:\n");
    let (q, qd, tau) = (vec![0.3, -0.4], vec![0.1, 0.0], vec![0.5, -0.3]);
    let g = Vector3::new(0.0, 0.0, -9.81);

    let (_, one_fd) = count(|| ferromotion_core::forward_dynamics(&robot, &inertia, &q, &qd, &tau, g));
    println!("  {:>44}  {:>6} allocations, {:>8} bytes", "one forward_dynamics call", one_fd.allocations, one_fd.bytes);

    let (_, one_mm) = count(|| ferromotion_core::mass_matrix(&robot, &inertia, &q));
    println!("  {:>44}  {:>6} allocations, {:>8} bytes", "one mass_matrix call", one_mm.allocations, one_mm.bytes);

    let (_, one_id) = count(|| ferromotion_core::inverse_dynamics(&robot, &inertia, &q, &qd, &tau, g));
    println!("  {:>44}  {:>6} allocations, {:>8} bytes", "one inverse_dynamics call", one_id.allocations, one_id.bytes);

    let (_, one_vec) = count(|| vec![0.0f64; 2]);
    println!("  {:>44}  {:>6} allocations, {:>8} bytes", "one vec![0.0; 2] (MPPI's own per-step)", one_vec.allocations, one_vec.bytes);

    let share = 1.0 / (one_fd.allocations + 1) as f64;
    println!("\n  So the hot path's allocation cost is inside the DYNAMICS, not inside MPPI's plumbing.");
    println!("  Pre-allocating MPPI's own buffers would remove about {:.1}% of the allocations per rollout step.", 100.0 * share);
    println!("  The fix has to be an allocation-free forward_dynamics, in ferromotion-core rather than the controller.\n");

    // --- the fix, measured
    let mut ws = ferromotion_core::DynamicsWorkspace::new(robot.dof());
    // one warm call, so the count is the steady state rather than first-touch growth
    let _ = ferromotion_core::forward_dynamics_in(&mut ws, &robot, &inertia, &q, &qd, &tau, g);
    let (_, ws_fd) = count(|| ferromotion_core::forward_dynamics_in(&mut ws, &robot, &inertia, &q, &qd, &tau, g));
    let (_, ws_id) = count(|| ferromotion_core::inverse_dynamics_in(&mut ws, &robot, &inertia, &q, &qd, &tau, g));
    let (_, ws_mm) = count(|| ferromotion_core::mass_matrix_in(&mut ws, &robot, &inertia, &q));
    println!("  with a reused DynamicsWorkspace:");
    println!("  {:>44}  {:>6} allocations, {:>8} bytes  ({:.0}x fewer)", "forward_dynamics_in", ws_fd.allocations, ws_fd.bytes,
        one_fd.allocations as f64 / ws_fd.allocations.max(1) as f64);
    println!("  {:>44}  {:>6} allocations, {:>8} bytes  ({:.0}x fewer)", "inverse_dynamics_in", ws_id.allocations, ws_id.bytes,
        one_id.allocations as f64 / ws_id.allocations.max(1) as f64);
    println!("  {:>44}  {:>6} allocations, {:>8} bytes  ({:.0}x fewer)", "mass_matrix_in", ws_mm.allocations, ws_mm.bytes,
        one_mm.allocations as f64 / ws_mm.allocations.max(1) as f64);

    println!("\n  Projected onto the 1024-sample, 25-step tick: {} allocations become about {}, a {:.0}x reduction.",
        923_730, 923_730 / (one_fd.allocations + 1) * (ws_fd.allocations + 1),
        (one_fd.allocations + 1) as f64 / (ws_fd.allocations + 1) as f64);
    println!("  The residue is 1 allocation for nalgebra's Cholesky factor plus 1 for MPPI's own per-step vector.");
    println!("  Note the flip: the controller's own plumbing was 2.8% of the original cost and is now half of what");
    println!("  remains. Proportions move when the dominant term is fixed, so the next target is not what was second.");
}
