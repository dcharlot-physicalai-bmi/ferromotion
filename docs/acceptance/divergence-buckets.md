# First-divergence buckets

Oracle profile `mujoco-3.13.0-x86_64-contract-off`, library `64156abe`, Menagerie pin `4d038b3`. 212 models, three forward states, 20 steps. Equality is `f64::to_bits`.

190 models are exact for all 20 steps. 22 fail a step. `mj_tendon` calls `mju_mulMatTVec` (`0x14edc0`). In `64156abe` that symbol vectorises at `n >= 4`: `res + scl*x` on the 4-wide prefix (`mulpd` then `addpd`) and `scl*x + res` on the tail (`mulsd` then `addsd`). `tendon_invweight0` is 212 of 212 after that schedule. Joined, and nothing left: `robotiq_2f85/scene.xml`, `robotiq_2f85_v4/scene.xml`, and the three `tetheria_aero_hand_open` models. A fixed tendon's length is `(prm * qpos) + L` in `mj_tendon`. `mju_dot3` in the primitive colliders is `(az bz) + (ax bx + ay by)`. `mju_mulMatVec3` (`0x14d790`) is `(m02 vz) + (m00 vx + m01 vy)` on row 0, `(m12 vz) + (m11 vy + m10 vx)` on row 1, and `(m21 vy + m20 vx) + (m22 vz)` on row 2. That corner transform is what `mjc_PlaneBox` feeds to `mju_dot3`. Franka `scene.xml` step 1 was a 1 ulp `efc_pos` on two plane–box contacts of `right_finger`, and the Newton factor was already identical. Sphere-sphere distance is `(sqrt - r1) - r2`, and the witness offset is `(dist * 0.5) + r1`. Capsule-capsule numerators use the raw segment dots: `(dot12 * v - mc * u) / det`.

Box-box edge witnesses use `denom = 1 - b*b`, `s = (d1e - b*d2e) / denom`, `t = b*s - d2e`, then `s = d1e + b*t`. The edge corner `mji_mulMatVec3` is `(m2 v2) + (m1 v1 + m0 v0)`, then `pos +` that. Leap hand `efc_pos` is exact on all three forward states. Those four models still miss a step; the next forward field is `efc_J`.

`mj_advance` copies the step's final `qacc` into `qacc_warmstart`. A step miss that shows up there is that step's `qacc`. Comparing `qfrc_passive`, `qfrc_actuator`, `qacc_smooth`, `efc_force`, and `qacc` at the failing step's input splits the old 13-model `qacc_warmstart` bucket: 12 first miss `efc_force` (smooth acceleration already matches), and `bitcraze_crazyflie_2/cf2.xml` first missed `qfrc_passive`. That model has no constraints, so `qacc` is `qacc_smooth`. The site-transmission actuator matches. The 1 ulp was the inertia-box linear drag, `lfrc -= 0.5*rho*box[p]*box[q]*|v|*v`, multiplied from `0.5*rho` and subtracted. After that schedule, step-1 `qacc` matches. The remaining `qpos` miss was `mj_RungeKutta`: each tableau weight is one `mju_addToScl` (`nv` for velocity, then `nv+na` for acceleration and `act_dot` together), and `mj_advance` integrates position with that weighted velocity. Both Crazyflie files are now exact for 20 steps.

| models | stage | first field | site |
| ---: | --- | --- | --- |
| 3 | collision | `efc_pos` | `mjraw primitive / mesh` |
| 3 | solver | `efc_force` | `mj_fwdConstraint` |
| 6 | smooth | `qacc` | `mj_step` |
| 4 | constraint rows | `efc_AR` | `mj_makeARNumeric` |
| 5 | solver | `qacc` | `mj_fwdConstraint` |
| 1 | smooth | `qfrc_fluid` | `mj_passive` |

## `efc_pos` — 3

Where it first appears: 1 at state 0, 2 at state 2.

`mjc_ConvexHField` (`0x6a700`) builds the cell as `(size + size) / (ncol - 1)`, then `(dx * c) - size` and `(dy * row) - size`. The height is `(sample * size2) + margin`. Barkour `scene_hfield_mjx.xml` is exact for 20 steps after that schedule. The three below still miss a step.

- `stanford_tidybot/scene.xml` (state 2)
- `ufactory_lite6/lite6_gripper_narrow.xml` (state 0)
- `wonik_allegro/scene_left.xml` (state 2)

## `efc_force` — 3

Where it first appears: 1 at step 1, 1 at step 9, 1 at step 18.

`qfrc_passive`, `qfrc_actuator`, and `qacc_smooth` match at that step. The step log still names `qacc_warmstart`, because that is the copy of this step's `qacc`.

All three are Newton and dense (`nv` below 60). Spot's elliptic miss was not the cone force. `J` and `qacc` matched and `efc_aref` was one ulp off, from `getimpedance`: `imp = solimp[0] + y*(solimp[1]-solimp[0])`. With that product separated, Spot and eight other models are exact for 20 steps.

`toddlerbot_2xc_mjx` has one island, so the solver scale is `1/sum(diag M)`, bits `0x3fb7c7cbbacf81d1`, not `1/(meaninertia*nv)`. Iteration 1's improvement matches (`0x41b6b1b1798c8003`). The scaled gradient is three ulps off (`0x3fd3e6b0315133e4` vs `0x3fd3e6b0315133e7`) because `mju_norm(grad)` is four ulps off. The next linesearch then returns `alpha = 0`.

- `rainbow_robotics_rby1/rby1m_1.2.xml` (step 1)
- `robotiq_2f85/2f85.xml` (step 9)
- `robotstudio_so101/scene_box.xml` (step 18)

## `ms_human_700` step 1 — 6

The index-35 ulp on `iliacus_r_tendon` was the cylinder correction in `mju_wrap`. `64156abe` takes each horizontal length as `sqrt(dy² + dx²)` (`mulpd`, `unpckhpd`, `addsd`, `sqrtsd`), the denominator as `L1 + (wlen + L0)`, and the corrected length as `sqrt(height² + wlen²)`. After that schedule, `ten_length` is 635 of 636. The one miss is `ms_human_700/scene.xml` state 1, index 186, one ulp (`0.4370790900602747` versus `0.43707909006027473`).

`mj_instantiateEquality` for a joint polynomial, in the same library, builds the position as successive `mulsd` by `dif` and `(term + acc)`, and the derivative as `data[1] + ((data[2]+data[2])*dif)`, then `((3*data[3])*dif)*dif` and `(((4*data[4])*dif)*dif)*dif`. `efc_J` on `MS-Human-700.xml` at the step-1 pose matches after that. The six models still miss step 1. On `MS-Human-700-Locomotion.xml` and `scene_locomotion.xml`, step-1 `qpos`, `qvel`, and `qacc_warmstart` match and `act` does not.

- `ms_human_700/MS-Human-700-Locomotion.xml` (step 1, `act`)
- `ms_human_700/MS-Human-700-Manipulation.xml` (step 1)
- `ms_human_700/MS-Human-700.xml` (step 1)
- `ms_human_700/scene.xml` (step 1)
- `ms_human_700/scene_locomotion.xml` (step 1, `act`)
- `ms_human_700/scene_manipulation.xml` (step 1)

## `efc_AR` — 4

Where it first appears: 4 at state 0.

- `hello_robot_stretch/stretch.xml` (state 0)
- `hello_robot_stretch_3/scene.xml` (state 0)
- `shadow_dexee/scene.xml` (state 0)
- `shadow_dexee/shadow_dexee.xml` (state 0)

## `qacc` — 5

Where it first appears: 5 at state 0.

`iit_softfoot` `ten_velocity` matches the three forward states after the `mju_mulMatTVec` schedule. The first listed miss is `qacc`, and the trajectory still misses step 1.

- `flybody/scene.xml` (state 0)
- `iit_softfoot/scene.xml` (state 0)
- `iit_softfoot/softfoot.xml` (state 0)
- `robot_soccer_kit/robot_soccer_kit.xml` (state 0)
- `robot_soccer_kit/scene.xml` (state 0)

## `qfrc_fluid` — 1

Where it first appears: 1 at state 1.

- `flybody/fruitfly.xml` (state 1)

## Step-exact models that still miss a forward state

These 13 pass all 20 steps and still differ on a forward state. The field is the first name in the native stage order.

- `anybotics_anymal_b/scene.xml` `efc_J`
- `anybotics_anymal_c/scene.xml` `efc_J`
- `anybotics_anymal_c/scene_mjx.xml` `qacc`
- `berkeley_humanoid/scene.xml` `efc_J`
- `dynamixel_2r/scene.xml` `efc_AR`
- `hello_robot_stretch/scene.xml` `qacc`
- `hello_robot_stretch_3/stretch.xml` `qacc`
- `pndbotics_adam_lite/scene.xml` `efc_J`
- `tetheria_aero_hand_open/left_hand.xml` `qfrc_spring`
- `tetheria_aero_hand_open/right_hand.xml` `qfrc_spring`
- `tetheria_aero_hand_open/scene_right.xml` `qfrc_spring`
- `umi_gripper/scene.xml` `qacc`
- `umi_gripper/umi_gripper.xml` `qacc`
