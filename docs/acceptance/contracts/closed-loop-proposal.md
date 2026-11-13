# Proposal: a minimal closed loop

Not built. `JOINT_LOOP_EXECUTED` stays false. `EPISODE_ENVELOPE_INTEROPERABLE` stays false. `FERROMORPHIC_BIND_VERIFIED` stays false.

`bind_v2_timestamps` is a pure validator. It returns `Result<(), TimebaseError>` and does not write the snapshot, the `Sim`, or a receipt. Stamping happens only in `StepReceipt::from_step`. A loop that treats `Ok(())` as a control value is inventing an output the function does not have.

The step that is not yet possible is: ferromotion `robot_contact::step`, then ferromorphic bind and readout, then that readout back into the next `tau`. The calls below are the agreed design for that gap. They are not in ferromorphic `a6722b49ff7dc770b1cb062adac21aa5c7155401`.

## `robot_contact::step`

`RobotContactSim::step` in `crates/ferromotion-core/src/robot_contact.rs:31`:

```rust
pub fn step(&self, q: &[f64], qd: &[f64], tau: &[f64], dt: f64, g: Vector3<f64>) -> (Vec<f64>, Vec<f64>)
```

`tau.len()` is `self.robot.dof()`, and `dof` is `KinematicTree::joints.len()` (`kinematic_tree.rs:46`). That length is `nv`, the number of velocity coordinates of this tree. It is not `nu`. `nu` equals `nv` only when the model has one scalar torque actuator on each velocity coordinate and no extra actuator. The function indexes `tau[i]` for `i in 0..nv` and does not read an actuator list.

For a 1-DoF joint, `tau[i]` is the generalized force on that joint, added as `tau[i] - bias[i]` before `M⁻¹`. A hinge is in newton-metres. A slide would be in newtons. This function does not convert or saturate. It has no `ctrlrange`, no `forcerange`, and no `actuatorfrcrange`.

Saturation that does exist is elsewhere. `Actuator::force` clamps the scalar control to `ctrlrange` when `ctrllimited`, then clamps the scalar force to `forcerange` when `forcelimited` (`mujoco_actuator.rs:282` and `mujoco_actuator.rs:301`). `qfrc_actuator` then clamps the summed joint force to the joint `actuatorfrcrange` (`mujoco_actuator.rs:309`). `robot_contact::step` does not call either function. A loop that saturates must do it before `tau` is passed in, using those actuator and joint ranges, not a limit inside `step`.

## Reference model

`franka_emika_panda/panda_nohand.xml` at Menagerie `4d038b3feae26ec82b46a4d586379114012a8ac7`. That file is in the 212-model exact set. The home keyframe is `qpos="0 0 0 -1.57079 0 1.57079 -0.7853"`.

Seven hinges, depth-first, `joint1` through `joint7`. No free joint and no slide, so `nq = nv = 7`. Seven `<general>` actuators, one per joint, so `nu = 7` as well. `tau` for this model is length 7, in newton-metres, in joint order `joint1` … `joint7`.

Class `panda` sets `ctrlrange="-2.8973 2.8973"` and `forcerange="-87 87"` on `<general>`. A joint with no `range` inherits `range="-2.8973 2.8973"`. An actuator attribute written on the element replaces the class value. No joint sets a velocity limit, and none sets `actuatorfrcrange`.

| joint | qpos range (rad) | actuator `ctrlrange` | actuator `forcerange` (N·m) |
|---|---|---|---|
| joint1 | −2.8973 … 2.8973 | −2.8973 … 2.8973 | −87 … 87 |
| joint2 | −1.7628 … 1.7628 | −1.7628 … 1.7628 | −87 … 87 |
| joint3 | −2.8973 … 2.8973 | −2.8973 … 2.8973 | −87 … 87 |
| joint4 | −3.0718 … −0.0698 | −3.0718 … −0.0698 | −87 … 87 |
| joint5 | −2.8973 … 2.8973 | −2.8973 … 2.8973 | −12 … 12 |
| joint6 | −0.0175 … 3.7525 | −0.0175 … 3.7525 | −12 … 12 |
| joint7 | −2.8973 … 2.8973 | −2.8973 … 2.8973 | −12 … 12 |

The torque passed into `step` is saturated to that joint's `forcerange` before the call. `ctrlrange` is in radians here because these general actuators are position servos (`gainprm` and `biasprm` on the element). It is not a torque limit. `step` never sees it.

## Observation

Length `nq + nv`. For a tree of 1-DoF joints that is `2 * nv`. Order is every `qpos` entry, then every `qvel` entry, both in the same joint order as `tau`.

Units: hinge position in radians, hinge velocity in radians per second, slide position in metres, slide velocity in metres per second. A free joint is not in this reference model. If one appears, its `qpos` is the 7 MuJoCo free coordinates (3 metres, then a 4-component unit quaternion) and its `qvel` is 6 (3 metres per second, then 3 radians per second), and `nv` counts the 6, not the 7. `tau` still has length `nv`. This reference model has no free joint, so the observation length is 14.

Component `i` of `qpos` is clamped to that joint's `range`. Component `i` of `qvel` is clamped only when the joint states a finite velocity limit. `panda_nohand` states none, so the seven velocity components are not clamped.

Concrete observation bounds for `panda_nohand`, `qpos` then `qvel`:

| index | field | unit | clamp |
|---|---|---|---|
| 0 | joint1 position | rad | −2.8973 … 2.8973 |
| 1 | joint2 position | rad | −1.7628 … 1.7628 |
| 2 | joint3 position | rad | −2.8973 … 2.8973 |
| 3 | joint4 position | rad | −3.0718 … −0.0698 |
| 4 | joint5 position | rad | −2.8973 … 2.8973 |
| 5 | joint6 position | rad | −0.0175 … 3.7525 |
| 6 | joint7 position | rad | −2.8973 … 2.8973 |
| 7 | joint1 velocity | rad/s | none |
| 8 | joint2 velocity | rad/s | none |
| 9 | joint3 velocity | rad/s | none |
| 10 | joint4 velocity | rad/s | none |
| 11 | joint5 velocity | rad/s | none |
| 12 | joint6 velocity | rad/s | none |
| 13 | joint7 velocity | rad/s | none |

## Agreed ferromorphic API

bmi-2 will add these. They are not at `a6722b49`. Inputs and outputs are plain `f64` slices. The window is a count of ticks.

```rust
impl EpisodeEnvelope {
    /// Line-oriented text. `from_text(to_text())` returns the same `version`, `epoch`,
    /// `dt_bits`, `period_ns`, and `origin_ns`. `dt_bits` is 16 lowercase hex digits.
    pub fn to_text(&self) -> String;
    pub fn from_text(text: &str) -> Result<Self, TimebaseError>;
}

/// `counts` has length `window * n_neurons`, tick-major: tick 0 neuron 0, tick 0 neuron 1, …
/// `weights` has length `tau_len * window * n_neurons`, row-major over
/// `(tau index, tick in the window, neuron)`.
/// `bias` has length `tau_len`.
/// `tau_i = bias_i + Σ counts[t, n] * weights[i, t, n]`.
/// For `robot_contact::step`, `tau_len` is `nv`.
pub struct SpikeReadout {
    pub window: u64,
    pub n_neurons: usize,
    pub tau_len: usize,
    pub weights: Vec<f64>,
    pub bias: Vec<f64>,
}

impl SpikeReadout {
    pub fn tau(&self, counts: &[f64]) -> Vec<f64>;
}

/// `weight` is row-major `n_out * obs_len`. `bias`, `lo`, and `hi` have length `n_out`.
/// `out_i = clamp(bias_i + Σ_j weight[i, j] * observation[j], lo_i, hi_i)`.
/// `out` is amperes and has length `Sim`'s neuron count. It is the `external` slice of `try_step`.
pub struct CurrentEncoder {
    pub obs_len: usize,
    pub n_out: usize,
    pub weight: Vec<f64>,
    pub bias: Vec<f64>,
    pub lo: Vec<f64>,
    pub hi: Vec<f64>,
}

impl CurrentEncoder {
    pub fn amperes(&self, observation: &[f64]) -> Vec<f64>;
}
```

Both maps are pure. The same input bits produce the same output bits. There is no RNG. Products are `f64` multiplies and sums in the index order above. `clamp` is inclusive.

`SpikeReadout::tau` is the readout-to-control map into `robot_contact::step`'s `tau`. For `panda_nohand`, `tau_len` is 7. After `tau` returns, ferromotion saturates each entry to that joint's `forcerange` from the table above. The readout itself does not know those ranges.

`CurrentEncoder::amperes` is the control-to-current map. Its `observation` is the length-14 vector in the table. Position components are already clamped to the joint ranges. Velocity components are not clamped for this model. `lo` and `hi` are the ampere clamp, per neuron, and they are parameters of the encoder, not fields of `panda_nohand`.

## What ferromotion provides

`RobotContactSim::step` consumes `tau` and returns the next `(q, qd)`. `ChunkV2`, `encode`, `check_horizon`, and `action_timestamp_ns` build the action-chunk bytes. `parse_step_receipt` reads a `StepReceipt` text. The local `bind_v2_timestamps` takes a `BoundaryClock`, not a `ferromorphic::sim::ClockSnapshot`. A call to the local function is not a call into ferromorphic.

Ferromotion does not yet provide `SpikeReadout` or `CurrentEncoder`. Putting either only on the ferromotion side is the same pair of functions with the same signatures and the same determinism rule. The loop is unbuilt either way until those functions exist and the envelope has `to_text` / `from_text`.
