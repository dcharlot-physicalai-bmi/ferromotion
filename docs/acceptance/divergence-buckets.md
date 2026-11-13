# First-divergence buckets

Oracle profile `mujoco-3.13.0-x86_64-contract-off`, library `64156abe`, Menagerie pin `4d038b3`. 212 models, three forward states, 20 steps. Equality is `f64::to_bits`.

165 models are exact for all 20 steps. 47 fail a step. A fixed tendon's length is `(prm * qpos) + L` in `mj_tendon`. `mju_dot3` in the primitive colliders is `(az bz) + (ax bx + ay by)`. Sphere-sphere distance is `(sqrt - r1) - r2`, and the witness offset is `(dist * 0.5) + r1`. Capsule-capsule numerators use the raw segment dots: `(dot12 * v - mc * u) / det`.

Box-box edge witnesses use `denom = 1 - b*b`, `s = (d1e - b*d2e) / denom`, `t = b*s - d2e`, then `s = d1e + b*t`. The edge corner `mji_mulMatVec3` is `(m2 v2) + (m1 v1 + m0 v0)`, then `pos +` that. Leap hand `efc_pos` is exact on all three forward states. Those four models still miss a step; the next forward field is `efc_J`.

`mj_advance` copies the step's final `qacc` into `qacc_warmstart`. A step miss that shows up there is that step's `qacc`. Comparing `qfrc_passive`, `qfrc_actuator`, `qacc_smooth`, `efc_force`, and `qacc` at the failing step's input splits the old 13-model `qacc_warmstart` bucket: 12 first miss `efc_force` (smooth acceleration already matches), and `bitcraze_crazyflie_2/cf2.xml` first missed `qfrc_passive`. That model has no constraints, so `qacc` is `qacc_smooth`. The site-transmission actuator matches. The 1 ulp was the inertia-box linear drag, `lfrc -= 0.5*rho*box[p]*box[q]*|v|*v`, multiplied from `0.5*rho` and subtracted. After that schedule, step-1 `qacc` matches. The remaining `qpos` miss was `mj_RungeKutta`: each tableau weight is one `mju_addToScl` (`nv` for velocity, then `nv+na` for acceleration and `act_dot` together), and `mj_advance` integrates position with that weighted velocity. Both Crazyflie files are now exact for 20 steps.

| models | stage | first field | site |
| ---: | --- | --- | --- |
| 11 | collision | `efc_pos` | `mjraw primitive / mesh` |
| 6 | solver | `efc_force` | `mj_fwdConstraint` |
| 9 | constants | `tendon_invweight0` | `set0 tendon block` |
| 5 | kinematics | `ten_length` | `mj_tendon` |
| 4 | constraint rows | `efc_J` | `mj_instantiateContact` |
| 4 | constraint rows | `efc_AR` | `mj_makeARNumeric` |
| 3 | solver | `qacc` | `mj_fwdConstraint` |
| 1 | integration | `qpos` | `mj_integratePos` |
| 2 | smooth | `ten_velocity` | `mj_fwdVelocity` |
| 1 | integration | `qvel` | `mj_advance` |
| 1 | smooth | `qfrc_fluid` | `mj_passive` |

## `efc_pos` — 11

Where it first appears: 5 at state 1, 4 at state 2, 2 at state 0.

- `apptronik_apollo/scene.xml` (state 2)
- `google_barkour_vb/scene_hfield_mjx.xml` (state 0)
- `leap_hand/left_hand.xml` (state 1)
- `leap_hand/right_hand.xml` (state 2)
- `leap_hand/scene_left.xml` (state 1)
- `leap_hand/scene_right.xml` (state 1)
- `stanford_tidybot/scene.xml` (state 2)
- `toddlerbot_2xm/scene.xml` (state 1)
- `ufactory_lite6/lite6_gripper_narrow.xml` (state 0)
- `wonik_allegro/scene_left.xml` (state 2)
- `wonik_allegro/scene_right.xml` (state 1)

## `efc_force` — 6

Where it first appears: 4 at step 1, 1 at step 9, 1 at step 18.

`qfrc_passive`, `qfrc_actuator`, and `qacc_smooth` match at that step. The step log still names `qacc_warmstart`, because that is the copy of this step's `qacc`.

All seven are Newton and dense (`nv` below 60). Spot's elliptic miss was not the cone force. `J` and `qacc` matched and `efc_aref` was one ulp off, from `getimpedance`: `imp = solimp[0] + y*(solimp[1]-solimp[0])`. With that product separated, Spot and eight other models are exact for 20 steps.

`toddlerbot_2xc_mjx` has one island, so the solver scale is `1/sum(diag M)`, bits `0x3fb7c7cbbacf81d1`, not `1/(meaninertia*nv)`. Iteration 1's improvement matches (`0x41b6b1b1798c8003`). The scaled gradient is three ulps off (`0x3fd3e6b0315133e4` vs `0x3fd3e6b0315133e7`) because `mju_norm(grad)` is four ulps off. The next linesearch then returns `alpha = 0`.

- `franka_emika_panda/scene.xml` (step 1)
- `rainbow_robotics_rby1/rby1m_1.2.xml` (step 1)
- `robotiq_2f85/2f85.xml` (step 9)
- `robotstudio_so101/scene_box.xml` (step 18)
- `shadow_hand/scene_right.xml` (step 1)
- `toddlerbot_2xc/toddlerbot_2xc.xml` (step 1)

## `tendon_invweight0` — 9

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

## `ten_length` — 5

Where it first appears: 2 at state 1, 3 at state 2.

- `toddlerbot_2xc/scene.xml` (state 1)
- `toddlerbot_2xc/scene_pos.xml` (state 2)
- `toddlerbot_2xc/toddlerbot_2xc_pos.xml` (state 2)
- `toddlerbot_2xm/scene_mjx.xml` (state 2)
- `toddlerbot_2xm/scene_pos.xml` (state 1)

## `efc_J` — 4

Where it first appears: 2 at state 1, 2 at state 0.

- `franka_emika_panda/mjx_single_cube.xml` (state 1)
- `robotiq_2f85/scene.xml` (state 0)
- `robotiq_2f85_v4/scene.xml` (state 0)
- `toddlerbot_2xm/toddlerbot_2xm_pos.xml` (state 1)

## `efc_AR` — 4

Where it first appears: 4 at state 0.

- `hello_robot_stretch/stretch.xml` (state 0)
- `hello_robot_stretch_3/scene.xml` (state 0)
- `shadow_dexee/scene.xml` (state 0)
- `shadow_dexee/shadow_dexee.xml` (state 0)

## `qacc` — 3

Where it first appears: 3 at state 0.

- `flybody/scene.xml` (state 0)
- `robot_soccer_kit/robot_soccer_kit.xml` (state 0)
- `robot_soccer_kit/scene.xml` (state 0)

## `qpos` — 1

Where it first appears: 1 at step 1.

- `trossen_wxai/wxai_follower_biarm.xml` (step 1)

## `ten_velocity` — 2

Where it first appears: 2 at state 1.

- `iit_softfoot/scene.xml` (state 1)
- `iit_softfoot/softfoot.xml` (state 1)

## `qvel` — 1

Where it first appears: 1 at step 5.

- `trossen_wxai/scene.xml` (step 5)

## `qfrc_fluid` — 1

Where it first appears: 1 at state 1.

- `flybody/fruitfly.xml` (state 1)

## Step-exact models that still miss a forward state

These 29 pass all 20 steps and still differ on a forward state.

- `agilex_piper/scene.xml` `efc_J` (state 2)
- `aloha/aloha.xml` `efc_D` (state 0)
- `aloha/scene.xml` `efc_D` (state 0)
- `anybotics_anymal_c/scene_mjx.xml` `efc_D` (state 0)
- `berkeley_humanoid/scene.xml` `efc_pos` (state 2)
- `booster_t1/scene.xml` `efc_pos` (state 1)
- `dynamixel_2r/scene.xml` `efc_AR` (state 2)
- `google_barkour_v0/scene.xml` `efc_D` (state 0)
- `google_barkour_v0/scene_barkour.xml` `efc_D` (state 0)
- `google_barkour_v0/scene_mjx.xml` `efc_D` (state 0)
- `hello_robot_stretch/scene.xml` `efc_pos` (state 2)
- `hello_robot_stretch_3/stretch.xml` `efc_AR` (state 0)
- `pal_talos/scene_motor.xml` `efc_J` (state 1)
- `pal_tiago/scene_position.xml` `efc_pos` (state 1)
- `pal_tiago/scene_velocity.xml` `efc_pos` (state 1)
- `pal_tiago_dual/scene_motor.xml` `efc_pos` (state 1)
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
- `wonik_allegro/right_hand.xml` `efc_pos` (state 1)
