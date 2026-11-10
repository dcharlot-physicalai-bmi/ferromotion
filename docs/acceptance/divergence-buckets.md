# First-divergence buckets

Oracle profile `mujoco-3.13.0-x86_64-contract-off`, library `64156abe`,
Menagerie pin `4d038b3`. Compared: 212 models, three `mj_forward` states,
and 20 `mj_step`s. Equality is `f64::to_bits`.

A model is placed by the earliest mismatch in this order: compiled
constants (`mj_setConst`), kinematics (`mj_fwdKinematics`), smooth
dynamics (`mj_makeM` through `mj_fwdAcceleration`), collision
(`mj_collision` as it lands in `efc_pos`), constraint rows, the solver
(`mj_fwdConstraint`), then integration. States 0–2 and the 20 steps are
one pool. 64 models are exact for all 20 steps; 28 of those still miss
a forward field on a state the trajectory does not hit. The 148 that
fail a step are the table below. 175 of 546 states with exact `efc_pos`
and `efc_J` still miss `qacc`.

| models | stage | first field | MuJoCo site |
| ---: | --- | --- | --- |
| 42 | solver | `qacc` | `mj_fwdConstraint` `engine_forward.c:1223` |
| 35 | constants | `dof_M0` | `mj_setM0` `engine_setconst.c:60` |
| 29 | collision | `efc_pos` | `mj_instantiateContact` `engine_core_constraint.c:1605` |
| 14 | solver | `qacc_warmstart` | `mj_fwdConstraint` `engine_forward.c:1223` |
| 7 | constraint rows | `efc_J` | `mj_instantiateContact` `engine_core_constraint.c:1605` |
| 4 | constraint rows | `efc_D` | `mj_makeImpedance` `engine_core_constraint.c:2286` |
| 4 | kinematics | `ten_length` | `mj_tendon` `engine_core_smooth.c:931` |
| 4 | integration | `qpos` | `mj_Euler` / `mj_implicit` `engine_forward.c:1448` |
| 3 | constants | `tendon_invweight0` | tendon block `engine_setconst.c:1040` |
| 2 | smooth | `ten_velocity` | `mj_fwdVelocity` `engine_forward.c:201` |
| 1 | smooth | `qfrc_fluid` | `mj_passive` `engine_passive.c:1209` |
| 1 | solver | `qfrc_constraint` | `mj_fwdConstraint` `engine_forward.c:1223` |
| 1 | constraint rows | `efc_AR` | `mj_makeARNumeric` `engine_core_constraint.c:3219` |
| 1 | integration | `qvel` | `mj_Euler` / `mj_implicit` `engine_forward.c:1473` |

The solver rows (`qacc`, `qacc_warmstart`, `qfrc_constraint`) are one
site: `mj_fwdConstraint` calls `mj_solNewton` or `mj_solCG`. That is the
largest bucket, 57 of the 148. `dof_M0` is next, then contact distance.

## `dof_M0` — 35

dof_M0 = armature + mju_dot(cdof, crb*cdof, 6). `mj_setM0`, `engine_setconst.c:60`.

Where it first appears: 35 at model.

- `apptronik_apollo/scene.xml` (model)
- `hello_robot_stretch/scene.xml` (model)
- `hello_robot_stretch/stretch.xml` (model)
- `leap_hand/left_hand.xml` (model)
- `leap_hand/scene_left.xml` (model)
- `ms_human_700/MS-Human-700-Locomotion.xml` (model)
- `ms_human_700/MS-Human-700-Manipulation.xml` (model)
- `ms_human_700/MS-Human-700.xml` (model)
- `ms_human_700/scene.xml` (model)
- `ms_human_700/scene_locomotion.xml` (model)
- `ms_human_700/scene_manipulation.xml` (model)
- `pal_talos/scene_motor.xml` (model)
- `pal_talos/scene_position.xml` (model)
- `pal_talos/talos.xml` (model)
- `pal_talos/talos_motor.xml` (model)
- `pal_talos/talos_position.xml` (model)
- `pndbotics_adam_lite/adam_lite.xml` (model)
- `pndbotics_adam_lite/scene.xml` (model)
- `rainbow_robotics_rby1/rby1a_1.2.xml` (model)
- `rainbow_robotics_rby1/rby1m_1.2.xml` (model)
- `rainbow_robotics_rby1/scene_rby1a_1.2.xml` (model)
- `rainbow_robotics_rby1/scene_rby1m_1.2.xml` (model)
- `rainbow_robotics_rby1/scene_rby1m_1.3.xml` (model)
- `shadow_dexee/scene.xml` (model)
- `shadow_dexee/shadow_dexee.xml` (model)
- `toddlerbot_2xc/scene.xml` (model)
- `toddlerbot_2xc/scene_mjx.xml` (model)
- `toddlerbot_2xc/scene_pos.xml` (model)
- `toddlerbot_2xc/toddlerbot_2xc.xml` (model)
- `toddlerbot_2xc/toddlerbot_2xc_mjx.xml` (model)
- `toddlerbot_2xc/toddlerbot_2xc_pos.xml` (model)
- `unitree_g1/g1_mjx.xml` (model)
- `unitree_g1/g1_with_hands.xml` (model)
- `unitree_g1/scene_mjx.xml` (model)
- `unitree_g1/scene_with_hands.xml` (model)

## `qacc` — 42

Newton or CG; qacc after the constraint solve. `mj_fwdConstraint`, `engine_forward.c:1223`.

Where it first appears: 17 at state 0, 16 at state 1, 9 at state 2.

- `agility_cassie/scene.xml` (state 1)
- `anybotics_anymal_b/anymal_b.xml` (state 2)
- `berkeley_humanoid/berkeley_humanoid.xml` (state 1)
- `boston_dynamics_spot/scene.xml` (state 1)
- `boston_dynamics_spot/scene_arm.xml` (state 1)
- `boston_dynamics_spot/spot_arm.xml` (state 2)
- `flybody/scene.xml` (state 0)
- `fourier_n1/scene.xml` (state 2)
- `franka_emika_panda/scene.xml` (state 0)
- `franka_fr3/fr3.xml` (state 0)
- `google_barkour_vb/scene.xml` (state 0)
- `low_cost_robot_arm/low_cost_robot_arm.xml` (state 2)
- `low_cost_robot_arm/scene.xml` (state 1)
- `pal_tiago/scene_motor.xml` (state 1)
- `pal_tiago/tiago.xml` (state 2)
- `pal_tiago/tiago_motor.xml` (state 1)
- `pal_tiago_dual/scene_position.xml` (state 0)
- `pal_tiago_dual/scene_velocity.xml` (state 2)
- `pal_tiago_dual/tiago_dual.xml` (state 1)
- `pal_tiago_dual/tiago_dual_motor.xml` (state 1)
- `pal_tiago_dual/tiago_dual_position.xml` (state 0)
- `pal_tiago_dual/tiago_dual_velocity.xml` (state 1)
- `rainbow_robotics_rby1/scene_rby1m_1.2_no_gripper.xml` (state 1)
- `robot_soccer_kit/robot_soccer_kit.xml` (state 0)
- `robot_soccer_kit/scene.xml` (state 0)
- `robotiq_2f85/2f85.xml` (state 1)
- `robotiq_2f85_v4/2f85.xml` (state 1)
- `robotstudio_so101/scene_box.xml` (state 2)
- `seeed_rebot_devarm/scene.xml` (state 2)
- `stanford_tidybot/tidybot.xml` (state 2)
- `trossen_vx300s/scene.xml` (state 0)
- `trossen_vx300s/vx300s.xml` (state 0)
- `trossen_wxai/scene.xml` (state 0)
- `trossen_wxai/wxai_follower_biarm.xml` (state 0)
- `ufactory_xarm7/scene.xml` (state 0)
- `ufactory_xarm7/xarm7.xml` (state 0)
- `unitree_g1/g1.xml` (state 1)
- `unitree_g1/scene.xml` (state 1)
- `unitree_go1/scene.xml` (state 0)
- `unitree_go2/go2.xml` (state 0)
- `unitree_z1/z1.xml` (state 0)
- `universal_robots_ur10e/scene.xml` (state 1)

## `efc_pos` — 29

contact distance copied into efc_pos; the witness is mj_collision. `mj_instantiateContact`, `engine_core_constraint.c:1605`.

Where it first appears: 18 at state 1, 9 at state 2, 2 at state 0.

- `anybotics_anymal_b/scene.xml` (state 1)
- `anybotics_anymal_c/scene.xml` (state 1)
- `berkeley_humanoid/scene.xml` (state 2)
- `booster_t1/scene.xml` (state 1)
- `booster_t1/t1.xml` (state 2)
- `google_barkour_vb/scene_hfield_mjx.xml` (state 0)
- `hello_robot_stretch_3/scene.xml` (state 2)
- `leap_hand/right_hand.xml` (state 2)
- `leap_hand/scene_right.xml` (state 1)
- `pal_tiago/scene_position.xml` (state 1)
- `pal_tiago/scene_velocity.xml` (state 1)
- `pal_tiago_dual/scene_motor.xml` (state 1)
- `rainbow_robotics_rby1/scene_rby1a_1.2_no_gripper.xml` (state 2)
- `rethink_robotics_sawyer/scene.xml` (state 1)
- `robotis_omy_3m/scene.xml` (state 2)
- `shadow_hand/left_hand.xml` (state 1)
- `shadow_hand/scene_left.xml` (state 1)
- `shadow_hand/scene_right.xml` (state 2)
- `sharpa_wave/left_hand.xml` (state 1)
- `sharpa_wave/right_hand.xml` (state 1)
- `sharpa_wave/scene_left.xml` (state 1)
- `stanford_tidybot/scene.xml` (state 2)
- `toddlerbot_2xm/scene.xml` (state 1)
- `ufactory_lite6/lite6_gripper_narrow.xml` (state 0)
- `unitree_a1/scene.xml` (state 1)
- `unitree_go2/scene.xml` (state 1)
- `wonik_allegro/right_hand.xml` (state 1)
- `wonik_allegro/scene_left.xml` (state 2)
- `wonik_allegro/scene_right.xml` (state 1)

## `qacc_warmstart` — 14

the solved qacc carried into the next step. `mj_fwdConstraint`, `engine_forward.c:1223`.

Where it first appears: 4 at step 1, 2 at step 17, 1 at step 19, 1 at step 3, 1 at step 6, 1 at step 7, 1 at step 16, 1 at step 12, 1 at step 9, 1 at step 20.

- `agility_cassie/cassie.xml` (step 19)
- `anybotics_anymal_c/anymal_c.xml` (step 17)
- `arx_l5/arx_l5.xml` (step 1)
- `bitcraze_crazyflie_2/cf2.xml` (step 1)
- `bitcraze_crazyflie_2/scene.xml` (step 3)
- `fourier_n1/n1.xml` (step 6)
- `kinova_gen3/scene.xml` (step 1)
- `pal_tiago/tiago_position.xml` (step 7)
- `seeed_rebot_devarm/seeed_rebot_devarm.xml` (step 16)
- `ufactory_lite6/scene.xml` (step 12)
- `unitree_a1/a1.xml` (step 9)
- `unitree_go1/go1.xml` (step 17)
- `unitree_h1/h1.xml` (step 20)
- `universal_robots_ur5e/scene.xml` (step 1)

## `efc_J` — 7

contact Jacobian rows. `mj_instantiateContact`, `engine_core_constraint.c:1605`.

Where it first appears: 3 at state 1, 3 at state 2, 1 at state 0.

- `franka_emika_panda/mjx_single_cube.xml` (state 1)
- `robotiq_2f85_v4/scene.xml` (state 0)
- `robotstudio_so101/scene.xml` (state 2)
- `shadow_hand/right_hand.xml` (state 2)
- `sharpa_wave/scene_right.xml` (state 2)
- `toddlerbot_2xm/toddlerbot_2xm.xml` (state 1)
- `toddlerbot_2xm/toddlerbot_2xm_pos.xml` (state 1)

## `efc_D` — 4

efc_D = 1/R. `mj_makeImpedance`, `engine_core_constraint.c:2286`.

Where it first appears: 3 at state 0, 1 at state 2.

- `aloha/scene.xml` (state 0)
- `google_barkour_v0/scene.xml` (state 0)
- `google_barkour_v0/scene_barkour.xml` (state 0)
- `ufactory_lite6/lite6_gripper_wide.xml` (state 2)

## `ten_length` — 4

called from mj_fwdKinematics (engine_forward.c:125). `mj_tendon`, `engine_core_smooth.c:931`.

Where it first appears: 3 at state 1, 1 at state 2.

- `robotiq_2f85/scene.xml` (state 1)
- `toddlerbot_2xm/scene_mjx.xml` (state 2)
- `toddlerbot_2xm/scene_pos.xml` (state 1)
- `toddlerbot_2xm/toddlerbot_2xm_mjx.xml` (state 1)

## `qpos` — 4

mj_integratePos. `mj_Euler / mj_implicit`, `engine_forward.c:1448`.

Where it first appears: 1 at step 10, 1 at step 13, 1 at step 19, 1 at step 7.

- `google_barkour_v0/barkour_v0.xml` (step 10)
- `google_barkour_v0/barkour_v0_mjx.xml` (step 13)
- `google_barkour_vb/scene_mjx.xml` (step 19)
- `unitree_go2/scene_mjx.xml` (step 7)

## `tendon_invweight0` — 3

tendon_invweight0 = mju_dot(tmp, tmp+nv, nv). `mj_setConst tendon block`, `engine_setconst.c:1040`.

Where it first appears: 3 at model.

- `tetheria_aero_hand_open/left_hand.xml` (model)
- `tetheria_aero_hand_open/right_hand.xml` (model)
- `tetheria_aero_hand_open/scene_right.xml` (model)

## `ten_velocity` — 2

ten_velocity = ten_J * qvel. `mj_fwdVelocity`, `engine_forward.c:201`.

Where it first appears: 2 at state 1.

- `iit_softfoot/scene.xml` (state 1)
- `iit_softfoot/softfoot.xml` (state 1)

## `qfrc_fluid` — 1

mj_applyFT into qfrc_fluid from mj_inertiaBoxFluidModel / mj_ellipsoidFluidModel. `mj_passive`, `engine_passive.c:1209`.

Where it first appears: 1 at state 1.

- `flybody/fruitfly.xml` (state 1)

## `qfrc_constraint` — 1

J^T * efc_force. `mj_fwdConstraint`, `engine_forward.c:1223`.

Where it first appears: 1 at state 2.

- `google_robot/scene.xml` (state 2)

## `efc_AR` — 1

noslip inverse-inertia matrix. `mj_makeARNumeric`, `engine_core_constraint.c:3219`.

Where it first appears: 1 at state 0.

- `hello_robot_stretch_3/stretch.xml` (state 0)

## `qvel` — 1

velocity update inside mj_EulerSkip / mj_implicitSkip. `mj_Euler / mj_implicit`, `engine_forward.c:1473`.

Where it first appears: 1 at step 1.

- `wonik_allegro/left_hand.xml` (step 1)

## Step-exact models that still miss a forward state

These 28 pass all 20 steps and still differ on a state from the
three-state forward set.

- `agilex_piper/scene.xml` `efc_J` (state 2)
- `aloha/aloha.xml` `efc_D` (state 0)
- `anybotics_anymal_c/scene_mjx.xml` `efc_D` (state 0)
- `apptronik_apollo/apptronik_apollo.xml` `dof_M0` (model)
- `boston_dynamics_spot/spot.xml` `qacc` (state 2)
- `dynamixel_2r/scene.xml` `efc_AR` (state 2)
- `franka_emika_panda/panda.xml` `qacc` (state 0)
- `franka_fr3/scene.xml` `qacc` (state 0)
- `franka_fr3_v2/fr3v2.xml` `qacc` (state 0)
- `franka_fr3_v2/scene.xml` `qacc` (state 0)
- `google_barkour_v0/scene_mjx.xml` `efc_D` (state 0)
- `google_robot/robot.xml` `qacc` (state 1)
- `rainbow_robotics_rby1/rby1a_1.2_no_gripper.xml` `efc_pos` (state 1)
- `rainbow_robotics_rby1/rby1m_1.3.xml` `dof_M0` (model)
- `robotis_op3/op3.xml` `qacc` (state 1)
- `robotis_op3/scene.xml` `efc_J` (state 2)
- `robotstudio_so101/so101.xml` `efc_pos` (state 2)
- `skydio_x2/scene.xml` `qfrc_fluid` (state 1)
- `skydio_x2/x2.xml` `qfrc_fluid` (state 2)
- `trossen_wx250s/scene.xml` `qacc` (state 0)
- `trossen_wx250s/wx250s.xml` `qacc` (state 0)
- `trossen_wxai/wxai_follower.xml` `qacc` (state 0)
- `trs_so_arm100/so_arm100.xml` `qacc` (state 2)
- `umi_gripper/scene.xml` `efc_AR` (state 0)
- `umi_gripper/umi_gripper.xml` `efc_AR` (state 0)
- `unitree_h1/scene.xml` `efc_pos` (state 2)
- `unitree_z1/scene.xml` `qacc` (state 0)
- `unitree_z1/z1_gripper.xml` `qacc` (state 0)
