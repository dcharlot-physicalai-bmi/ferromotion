//! **Learned floating-base locomotion on your GPU** — the whole assembly as a runnable demo.
//!
//! A free 6-DoF base on a 2-link leg starts in a deep crouch. Every generation, hundreds of candidate
//! linear controllers are each rolled out through the full floating-base contact simulator (forward
//! kinematics + foot contact + spatial articulated-body dynamics + SE(3) integration) — one GPU thread
//! per policy — and CEM moves toward the ones that stand the base up. Prints the learning curve and the
//! rollout throughput (the Brax/MJX/Newton parallel-env sweep, on whatever GPU wgpu exposes).
//!
//! Run: `cargo run -p ferromotion-core --example gpu_floating_gait_rl --release --features gpu`

use ferromotion_core::gpu::FloatingGaitGpu;
use ferromotion_core::{from_urdf_full, LinkInertia};
use nalgebra::{Matrix3, Vector3};
use std::time::Instant;

const LEG: &str = r#"<robot name="leg">
  <link name="base"/>
  <link name="thigh"><inertial><origin xyz="0 0 -0.15" rpy="0 0 0"/><mass value="1.0"/><inertia ixx="0.01" iyy="0.01" izz="0.004" ixy="0" ixz="0" iyz="0"/></inertial></link>
  <link name="shank"><inertial><origin xyz="0 0 -0.15" rpy="0 0 0"/><mass value="0.6"/><inertia ixx="0.006" iyy="0.006" izz="0.002" ixy="0" ixz="0" iyz="0"/></inertial></link>
  <link name="foot"/>
  <joint name="hip" type="revolute"><parent link="base"/><child link="thigh"/><origin xyz="0 0 0" rpy="0 0 0"/><axis xyz="0 1 0"/><limit lower="-2" upper="2" effort="30" velocity="8"/></joint>
  <joint name="knee" type="revolute"><parent link="thigh"/><child link="shank"/><origin xyz="0 0 -0.3" rpy="0 0 0"/><axis xyz="0 1 0"/><limit lower="-2" upper="2" effort="30" velocity="8"/></joint>
  <joint name="ankle" type="fixed"><parent link="shank"/><child link="foot"/><origin xyz="0 0 -0.3" rpy="0 0 0"/></joint></robot>"#;

fn main() {
    let (robot, inertia) = from_urdf_full(LEG, "base", "foot").unwrap();
    let n = robot.dof();
    let base_inertia = LinkInertia { mass: 6.0, com: Vector3::zeros(), inertia: Matrix3::from_diagonal(&Vector3::new(0.05, 0.05, 0.05)) };
    let g = Vector3::new(0.0, 0.0, -9.81);
    let (floor, kn, kd, dt) = (0.0, 6.0e3, 80.0, 1e-3);
    let contacts = vec![(n, Vector3::zeros(), 0.9)];
    let (effort_w, taumax, steps) = (2e-5, 40.0, 250usize);
    let (pop, gens) = (512usize, 80usize);

    let Some(gp) = FloatingGaitGpu::new(&robot, &inertia, &base_inertia, &contacts, floor, kn, kd, g, dt, pop) else {
        eprintln!("No usable GPU (wgpu found no adapter) — this demo needs a GPU.");
        return;
    };
    let dim = gp.policy_dim();

    // deep-crouch start (foot planted, base low) — the policy must stand it up
    let mut init = vec![0.0f64; 18 + 2 * n];
    for (k, v) in Matrix3::<f64>::identity().as_slice().iter().enumerate() {
        init[k] = *v;
    }
    init[11] = 0.40; // base height
    init[18] = 0.6; // hip
    init[19] = -1.2; // knee

    println!("learned floating-base stand-up · {n}-DoF leg + free base · {pop} policies/gen × {steps} steps × {gens} gens");
    println!("policy: linear, {dim} params · reward: base height × uprightness − effort\n");

    // do-nothing baseline
    let zero = vec![0.0f64; pop * dim];
    let base_reward = gp.rollout_rewards(&zero, &init, effort_w, taumax, steps)[0];

    let mut seed = 0x1234_5678u64;
    let mut rng = || {
        seed = seed.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = seed;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        (((z ^ (z >> 31)) as f64) / (u64::MAX as f64)) * 2.0 - 1.0
    };
    let mut mean = vec![0.0f64; dim];
    let mut sigma = vec![0.6f64; dim];
    let t0 = Instant::now();
    for gi in 0..gens {
        let mut batch = vec![0.0f64; pop * dim];
        for p in 0..pop {
            for d in 0..dim {
                batch[p * dim + d] = mean[d] + sigma[d] * rng();
            }
        }
        let rewards = gp.rollout_rewards(&batch, &init, effort_w, taumax, steps);
        let mut idx: Vec<usize> = (0..pop).collect();
        idx.sort_by(|&a, &b| rewards[b].partial_cmp(&rewards[a]).unwrap());
        let elite = (pop as f64 * 0.15) as usize;
        for d in 0..dim {
            let mut m = 0.0;
            for &e in idx.iter().take(elite) {
                m += batch[e * dim + d];
            }
            m /= elite as f64;
            let mut v = 0.0;
            for &e in idx.iter().take(elite) {
                v += (batch[e * dim + d] - m).powi(2);
            }
            mean[d] = m;
            sigma[d] = (v / elite as f64).sqrt().max(0.02);
        }
        if gi % 10 == 0 || gi == gens - 1 {
            let avg_h = rewards[idx[0]] / steps as f64;
            println!("  gen {gi:>3}: best reward {:>8.2}  (avg upright height ~{avg_h:.3} m)", rewards[idx[0]]);
        }
    }
    let elapsed = t0.elapsed().as_secs_f64();
    let total_rollouts = (pop * gens) as f64;
    let total_steps = total_rollouts * steps as f64;

    let final_reward = gp.rollout_rewards(&(0..pop).flat_map(|_| mean.clone()).collect::<Vec<_>>(), &init, effort_w, taumax, steps)[0];
    println!("\nlearned {final_reward:.2}  vs  do-nothing {base_reward:.2}  ({:.0}% higher)", 100.0 * (final_reward - base_reward) / base_reward.abs());
    println!(
        "throughput: {:.0} floating-base rollouts/s, {:.1}M sim-steps/s, over {elapsed:.2} s on the GPU",
        total_rollouts / elapsed,
        total_steps / elapsed / 1e6
    );
}
