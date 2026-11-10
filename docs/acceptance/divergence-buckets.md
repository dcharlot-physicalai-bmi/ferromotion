# First-divergence buckets

Oracle profile `mujoco-3.13.0-x86_64-contract-off`, library `64156abe`,
Menagerie pin `4d038b3`. Compared: 212 models, three `mj_forward` states,
and 20 `mj_step`s. Equality is `f64::to_bits`.

A model is placed by the earliest mismatch in this order: compiled
constants (`mj_setConst`), kinematics (`mj_fwdKinematics`), smooth
dynamics (`mj_makeM` through `mj_fwdAcceleration`), collision
(`mj_collision` as it lands in `efc_pos`), constraint rows, the solver
(`mj_fwdConstraint`), then integration. States 0–2 and the 20 steps are
one pool. 106 models are exact for all 20 steps; 27 of those still miss
a forward field on a state the trajectory does not hit. The 106 that
fail a step are the table below. 47 of 546 states with exact `efc_pos`
and `efc_J` still miss `qacc`.

`mju_cholUpdate` on the dense Hessian, and `mju_cholUpdateSparse` on a
model with 60 or more dofs, now follow the pinned library. On
`unitree_z1` the cold factor already matched and the first downdate
did not. `qacc` as a first miss fell from 50 models to 3. The two
largest remaining sites are integration (`qpos`, 29) and contact
distance (`efc_pos`, 29).

| models | stage | first field | MuJoCo site |
| ---: | --- | --- | --- |
| 29 | integration | `qpos` | `mj_Euler / mj_implicit` `engine_forward.c:1448` |
| 29 | collision | `efc_pos` | `mj_instantiateContact` `engine_core_constraint.c:1605` |
| 9 | solver | `qacc_warmstart` | `mj_fwdConstraint` `engine_forward.c:1223` |
| 9 | constants | `tendon_invweight0` | `mj_setConst tendon block` `engine_setconst.c:1040` |
| 8 | kinematics | `ten_length` | `mj_tendon` `engine_core_smooth.c:931` |
| 6 | constraint rows | `efc_J` | `mj_instantiateContact` `engine_core_constraint.c:1605` |
| 5 | integration | `qvel` | `mj_Euler / mj_implicit` `engine_forward.c:1473` |
| 4 | constraint rows | `efc_AR` | `mj_makeARNumeric` `engine_core_constraint.c:3219` |
| 3 | solver | `qacc` | `mj_fwdConstraint` `engine_forward.c:1223` |
| 2 | smooth | `ten_velocity` | `mj_fwdVelocity` `engine_forward.c:201` |
| 1 | constraint rows | `efc_D` | `mj_makeImpedance` `engine_core_constraint.c:2286` |
| 1 | smooth | `qfrc_fluid` | `mj_passive` `engine_passive.c:1209` |

## `qpos` — 29

mj_integratePos. `mj_Euler / mj_implicit`, `engine_forward.c:1448`.

Where it first appears: 6 at step 1, 5 at step 5, 4 at step 2, 2 at step 8, 2 at step 13, 2 at step 3, 2 at step 7, 1 at step 10, 1 at step 17, 1 at step 19, 1 at step 4, 1 at step 16, 1 at step 9.

- `agility_cassie/cassie.xml` (step 1)
- `agility_cassie/scene.xml` (step 2)
- `anybotics_anymal_b/anymal_b.xml` (step 1)
- `anybotics_anymal_c/anymal_c.xml` (step 8)
- `bitcraze_crazyflie_2/scene.xml` (step 1)
- `boston_dynamics_spot/scene_arm.xml` (step 1)
- `google_barkour_v0/barkour_v0.xml` (step 10)
- `google_barkour_v0/barkour_v0_mjx.xml` (step 13)
- `google_barkour_vb/scene.xml` (step 17)
- `google_barkour_vb/scene_mjx.xml` (step 19)
- `pal_talos/scene_position.xml` (step 2)
- `pal_talos/talos_motor.xml` (step 2)
- `pal_tiago/scene_motor.xml` (step 4)
- `pal_tiago/tiago_position.xml` (step 8)
- `pal_tiago_dual/tiago_dual.xml` (step 5)
- `pal_tiago_dual/tiago_dual_motor.xml` (step 3)
- `pal_tiago_dual/tiago_dual_position.xml` (step 13)
- `pal_tiago_dual/tiago_dual_velocity.xml` (step 3)
- `pndbotics_adam_lite/adam_lite.xml` (step 16)
- `rainbow_robotics_rby1/scene_rby1m_1.2.xml` (step 5)
- `rainbow_robotics_rby1/scene_rby1m_1.2_no_gripper.xml` (step 5)
- `trossen_wxai/wxai_follower_biarm.xml` (step 1)
- `unitree_a1/a1.xml` (step 1)
- `unitree_g1/g1_with_hands.xml` (step 5)
- `unitree_g1/scene.xml` (step 5)
- `unitree_go1/go1.xml` (step 2)
- `unitree_go1/scene.xml` (step 9)
- `unitree_go2/scene_mjx.xml` (step 7)
- `unitree_h1/h1.xml` (step 7)

## `efc_pos` — 29

contact distance copied into efc_pos; the witness is mj_collision. `mj_instantiateContact`, `engine_core_constraint.c:1605`.

Where it first appears: 16 at state 1, 11 at state 2, 2 at state 0.

- `anybotics_anymal_b/scene.xml` (state 1)
- `anybotics_anymal_c/scene.xml` (state 1)
- `apptronik_apollo/scene.xml` (state 2)
- `berkeley_humanoid/scene.xml` (state 2)
- `booster_t1/scene.xml` (state 1)
- `booster_t1/t1.xml` (state 2)
- `google_barkour_vb/scene_hfield_mjx.xml` (state 0)
- `hello_robot_stretch/scene.xml` (state 2)
- `hello_robot_stretch_3/scene.xml` (state 2)
- `leap_hand/left_hand.xml` (state 1)
- `leap_hand/right_hand.xml` (state 2)
- `leap_hand/scene_left.xml` (state 1)
- `leap_hand/scene_right.xml` (state 1)
- `rainbow_robotics_rby1/scene_rby1a_1.2.xml` (state 2)
- `rainbow_robotics_rby1/scene_rby1a_1.2_no_gripper.xml` (state 2)
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
- `unitree_g1/scene_mjx.xml` (state 1)
- `wonik_allegro/right_hand.xml` (state 1)
- `wonik_allegro/scene_left.xml` (state 2)
- `wonik_allegro/scene_right.xml` (state 1)

## `qacc_warmstart` — 9

the solved qacc carried into the next step. `mj_fwdConstraint`, `engine_forward.c:1223`.

Where it first appears: 5 at step 1, 1 at step 10, 1 at step 9, 1 at step 18, 1 at step 5.

- `arx_l5/arx_l5.xml` (step 10)
- `bitcraze_crazyflie_2/cf2.xml` (step 1)
- `franka_emika_panda/scene.xml` (step 1)
- `rainbow_robotics_rby1/rby1m_1.2.xml` (step 1)
- `robotiq_2f85/2f85.xml` (step 9)
- `robotstudio_so101/scene_box.xml` (step 18)
- `toddlerbot_2xc/toddlerbot_2xc.xml` (step 1)
- `toddlerbot_2xc/toddlerbot_2xc_mjx.xml` (step 1)
- `universal_robots_ur10e/scene.xml` (step 5)

## `tendon_invweight0` — 9

tendon_invweight0 = mju_dot(tmp, tmp+nv, nv). `mj_setConst tendon block`, `engine_setconst.c:1040`.

Where it first appears: 9 at model.

- `ms_human_700/MS-Human-700-Locomotion.xml` (model)
- `ms_human_700/MS-Human-700-Manipulation.xml` (model)
- `ms_human_700/MS-Human-700.xml` (model)
- `ms_human_700/scene.xml` (model)
- `ms_human_700/scene_locomotion.xml` (model)
- `ms_human_700/scene_manipulation.xml` (model)
- `tetheria_aero_hand_open/left_hand.xml` (model)
- `tetheria_aero_hand_open/right_hand.xml` (model)
- `tetheria_aero_hand_open/scene_right.xml` (model)

## `ten_length` — 8

called from mj_fwdKinematics (engine_forward.c:125). `mj_tendon`, `engine_core_smooth.c:931`.

Where it first appears: 5 at state 1, 3 at state 2.

- `robotiq_2f85/scene.xml` (state 1)
- `toddlerbot_2xc/scene.xml` (state 1)
- `toddlerbot_2xc/scene_mjx.xml` (state 1)
- `toddlerbot_2xc/scene_pos.xml` (state 2)
- `toddlerbot_2xc/toddlerbot_2xc_pos.xml` (state 2)
- `toddlerbot_2xm/scene_mjx.xml` (state 2)
- `toddlerbot_2xm/scene_pos.xml` (state 1)
- `toddlerbot_2xm/toddlerbot_2xm_mjx.xml` (state 1)

## `efc_J` — 6

contact Jacobian rows. `mj_instantiateContact`, `engine_core_constraint.c:1605`.

Where it first appears: 4 at state 1, 1 at state 0, 1 at state 2.

- `franka_emika_panda/mjx_single_cube.xml` (state 1)
- `pndbotics_adam_lite/scene.xml` (state 1)
- `robotiq_2f85_v4/scene.xml` (state 0)
- `sharpa_wave/scene_right.xml` (state 2)
- `toddlerbot_2xm/toddlerbot_2xm.xml` (state 1)
- `toddlerbot_2xm/toddlerbot_2xm_pos.xml` (state 1)

## `qvel` — 5

velocity update inside mj_EulerSkip / mj_implicitSkip. `mj_Euler / mj_implicit`, `engine_forward.c:1473`.

Where it first appears: 3 at step 1, 1 at step 8, 1 at step 5.

- `robotiq_2f85_v4/2f85.xml` (step 8)
- `trossen_vx300s/scene.xml` (step 1)
- `trossen_vx300s/vx300s.xml` (step 1)
- `trossen_wxai/scene.xml` (step 5)
- `wonik_allegro/left_hand.xml` (step 1)

## `efc_AR` — 4

noslip inverse-inertia matrix. `mj_makeARNumeric`, `engine_core_constraint.c:3219`.

Where it first appears: 4 at state 0.

- `hello_robot_stretch/stretch.xml` (state 0)
- `hello_robot_stretch_3/stretch.xml` (state 0)
- `shadow_dexee/scene.xml` (state 0)
- `shadow_dexee/shadow_dexee.xml` (state 0)

## `qacc` — 3

Newton or CG; qacc after the constraint solve. `mj_fwdConstraint`, `engine_forward.c:1223`.

Where it first appears: 3 at state 0.

- `flybody/scene.xml` (state 0)
- `robot_soccer_kit/robot_soccer_kit.xml` (state 0)
- `robot_soccer_kit/scene.xml` (state 0)

## `ten_velocity` — 2

ten_velocity = ten_J * qvel. `mj_fwdVelocity`, `engine_forward.c:201`.

Where it first appears: 2 at state 1.

- `iit_softfoot/scene.xml` (state 1)
- `iit_softfoot/softfoot.xml` (state 1)

## `efc_D` — 1

efc_D = 1/R. `mj_makeImpedance`, `engine_core_constraint.c:2286`.

Where it first appears: 1 at state 0.

- `aloha/scene.xml` (state 0)

## `qfrc_fluid` — 1

mj_applyFT into qfrc_fluid from the box and ellipsoid fluid models. `mj_passive`, `engine_passive.c:1209`.

Where it first appears: 1 at state 1.

- `flybody/fruitfly.xml` (state 1)

## Step-exact models that still miss a forward state

These 27 pass all 20 steps and still differ on a state from the
three-state forward set.

- `agilex_piper/scene.xml` `efc_J` (state 2)
- `aloha/aloha.xml` `efc_D` (state 0)
- `anybotics_anymal_c/scene_mjx.xml` `efc_D` (state 0)
- `dynamixel_2r/scene.xml` `efc_AR` (state 2)
- `google_barkour_v0/scene.xml` `efc_D` (state 0)
- `google_barkour_v0/scene_barkour.xml` `efc_D` (state 0)
- `google_barkour_v0/scene_mjx.xml` `efc_D` (state 0)
- `pal_talos/scene_motor.xml` `efc_J` (state 1)
- `pal_tiago/scene_position.xml` `efc_pos` (state 1)
- `pal_tiago/scene_velocity.xml` `efc_pos` (state 1)
- `pal_tiago_dual/scene_motor.xml` `efc_pos` (state 1)
- `rainbow_robotics_rby1/rby1a_1.2_no_gripper.xml` `efc_pos` (state 1)
- `rainbow_robotics_rby1/rby1m_1.3.xml` `efc_pos` (state 2)
- `rethink_robotics_sawyer/scene.xml` `efc_pos` (state 1)
- `robotis_omy_3m/scene.xml` `efc_pos` (state 2)
- `robotis_op3/scene.xml` `efc_J` (state 2)
- `robotstudio_so101/scene.xml` `efc_J` (state 2)
- `robotstudio_so101/so101.xml` `efc_pos` (state 2)
- `shadow_hand/right_hand.xml` `efc_J` (state 2)
- `skydio_x2/scene.xml` `qfrc_fluid` (state 1)
- `skydio_x2/x2.xml` `qfrc_fluid` (state 2)
- `ufactory_lite6/lite6_gripper_wide.xml` `efc_D` (state 2)
- `umi_gripper/scene.xml` `efc_AR` (state 0)
- `umi_gripper/umi_gripper.xml` `efc_AR` (state 0)
- `unitree_g1/scene_with_hands.xml` `efc_pos` (state 2)
- `unitree_go2/scene.xml` `efc_pos` (state 1)
- `unitree_h1/scene.xml` `efc_pos` (state 2)
