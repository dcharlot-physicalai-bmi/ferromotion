# First-divergence buckets

Oracle profile `mujoco-3.13.0-x86_64-contract-off`, library `64156abe`, Menagerie pin `4d038b3`. 212 models, three forward states, 20 steps. Equality is `f64::to_bits`.

212 models are exact for all 20 steps. None fail a step. `efc_state` is 636 of 636. `efc_AR` is 30 of 30. Joined: `hello_robot_stretch/stretch.xml`, `hello_robot_stretch_3/scene.xml`, and `flybody/scene.xml`. Every Menagerie collision `mesh_graph` matches `64156abe` (804 of 804). The nine that still differed after `qh_distplane` — both Stretch `link_aruco` bases, both `link_wrist_yaw`, `low_cost_robot_arm/elbow_to_wrist_extension_collision`, `robotiq_2f85_v4/follower`, and `seeed_rebot_devarm/base_link_col_5` — match once the plane offset dot, Gaussian elimination, and the merge tolerances are unfused the same way. Those nine do not move the 20-step set. `mjc_ccd` for these meshes is native GJK (`disableflags` 0, so `mjDSBL_NATIVECCD` is clear). Sphere-mesh reduces the sphere to a point and returns the inflated GJK witness; EPA does not run. `qh_distplane` in `64156abe` is `offset + p0·n0`, then `(p1·n1) + that`, then `(p2·n2) + that`. With that schedule, plus the unfused plane normal, the four mesh models' `mesh_graph` arrays match the oracle. `efc_pos` is 636 of 636. Joined, and nothing left: `rainbow_robotics_rby1/rby1m_1.2.xml`, `robotiq_2f85/2f85.xml`, `stanford_tidybot/scene.xml`, and `ufactory_lite6/lite6_gripper_narrow.xml`. `mju_sqrMatTDSparseNumeric` in `64156abe` forms each Hessian term as `(J · (vt · D)) + acc` (`mulsd` with `J` as the destination, then `addsd` of the accumulator). `efc_force` is 611 of 636, `qacc` 612 of 636, `qfrc_constraint` 611 of 636. Joined, and nothing left: `ms_human_700/MS-Human-700.xml`, `ms_human_700/scene.xml`, and `flybody/fruitfly.xml`. `mju_wrap` inside wrap (the side site's norm is below the radius) writes `cosG` as `((len1² + len0²) - dd) / ((len0 + len0)·len1)`, with `dd = (dif1²) + (dif0²)`. The Newton residual is `(asin(B·z) + asin(A·z))` minus `asin(z) + asin(z)`, and the point is `radius·((cos·vx) − (sin·vy))`, `radius·((cos·vy) + (sin·vx))`. `ten_length` and `ten_velocity` are 636 of 636. Joined, and nothing left: `ms_human_700/MS-Human-700-Manipulation.xml`. `mju_muscleGainLength` in `64156abe` writes the middle pieces as `1 + ((-0.5)*x)*x`. `mju_muscleDynamics` uses `s = 0.5 + (1.5*actclamp)` and `tau_deact + (tau_act-tau_deact)*sigmoid`, and `mju_sigmoid` is `(((x+x)-5)*(3*x) + 10) * ((x*x)*x)`. `mj_nextActivation` is `act + (h*act_dot)`. Joined, and nothing left: `ms_human_700/MS-Human-700-Locomotion.xml`, `ms_human_700/scene_locomotion.xml`, `ms_human_700/scene_manipulation.xml`, and `shadow_dexee/shadow_dexee.xml`. `mj_tendon` calls `mju_mulMatTVec` (`0x14edc0`). In `64156abe` that symbol vectorises at `n >= 4`: `res + scl*x` on the 4-wide prefix (`mulpd` then `addpd`) and `scl*x + res` on the tail (`mulsd` then `addsd`). `tendon_invweight0` is 212 of 212 after that schedule. Joined, and nothing left: `robotiq_2f85/scene.xml`, `robotiq_2f85_v4/scene.xml`, and the three `tetheria_aero_hand_open` models. A fixed tendon's length is `(prm * qpos) + L` in `mj_tendon`. `mju_dot3` in the primitive colliders is `(az bz) + (ax bx + ay by)`. `mju_mulMatVec3` (`0x14d790`) is `(m02 vz) + (m00 vx + m01 vy)` on row 0, `(m12 vz) + (m11 vy + m10 vx)` on row 1, and `(m21 vy + m20 vx) + (m22 vz)` on row 2. That corner transform is what `mjc_PlaneBox` feeds to `mju_dot3`. Franka `scene.xml` step 1 was a 1 ulp `efc_pos` on two plane–box contacts of `right_finger`, and the Newton factor was already identical. Sphere-sphere distance is `(sqrt - r1) - r2`, and the witness offset is `(dist * 0.5) + r1`. Capsule-capsule numerators use the raw segment dots: `(dot12 * v - mc * u) / det`.

Box-box edge witnesses use `denom = 1 - b*b`, `s = (d1e - b*d2e) / denom`, `t = b*s - d2e`, then `s = d1e + b*t`. The edge corner `mji_mulMatVec3` is `(m2 v2) + (m1 v1 + m0 v0)`, then `pos +` that. Leap hand `efc_pos` is exact on all three forward states. Those four models still miss a step; the next forward field is `efc_J`.

`mj_advance` copies the step's final `qacc` into `qacc_warmstart`. A step miss that shows up there is that step's `qacc`. Comparing `qfrc_passive`, `qfrc_actuator`, `qacc_smooth`, `efc_force`, and `qacc` at the failing step's input splits the old 13-model `qacc_warmstart` bucket: 12 first miss `efc_force` (smooth acceleration already matches), and `bitcraze_crazyflie_2/cf2.xml` first missed `qfrc_passive`. That model has no constraints, so `qacc` is `qacc_smooth`. The site-transmission actuator matches. The 1 ulp was the inertia-box linear drag, `lfrc -= 0.5*rho*box[p]*box[q]*|v|*v`, multiplied from `0.5*rho` and subtracted. After that schedule, step-1 `qacc` matches. The remaining `qpos` miss was `mj_RungeKutta`: each tableau weight is one `mju_addToScl` (`nv` for velocity, then `nv+na` for acceleration and `act_dot` together), and `mj_advance` integrates position with that weighted velocity. Both Crazyflie files are now exact for 20 steps.

| models | stage | first field | site |
| ---: | --- | --- | --- |
| 0 | step | — | — |

## `efc_pos`

`efc_pos` is 636 of 636. `mjc_ConvexHField` (`0x6a700`) builds the cell as `(size + size) / (ncol - 1)`, then `(dx * c) - size` and `(dy * row) - size`. The height is `(sample * size2) + margin`. Barkour `scene_hfield_mjx.xml` is exact for 20 steps after that schedule. The four mesh models above are exact for 20 steps after the contract-off `qh_distplane` schedule. `wonik_allegro/scene_left.xml` is exact for 20 steps. The capsule center `(halfaxis * t) + pos` did not move its step-6 miss. The witness is `mjraw_SphereBox`: outside the box, `deepest = (normal * r) + center` and `pos = (deepest * 0.5) + (clamped * 0.5)`.

## `so101`

`robotstudio_so101/scene_box.xml` is exact for 20 steps. The constraint `qacc` already matched. `qvel[11]`, the free box's last angular dof, was 8 ulps off. `mjd_freeMhat` builds the 6×6 block with unfused `mji_mulMatVec3` dots `(a0 b0 + a1 b1) + a2 b2`, `A += h_mass·lin`, `A += h·rot`, and `mju_factorLU6` / `mju_solveLU6` subtract the product.

## `efc_AR` — 3

`mju_addToScl` in `64156abe` uses `res + (scl·x)` on groups of four when the prefix is at least 6, and `(scl·x) + res` on the tail. The back-substitution is `x − L·xᵢ`. `mju_QCQP2` and the noslip residual `res − R·f` are unfused the same way. `efc_AR` is 30 of 30 and `efc_state` is 636 of 636. On Stretch 3 the cone at row 351 was `CONE` until the QCQP2 product was separated; the oracle force is on the satisfied side of `μN ≥ T`.

`shadow_dexee/scene.xml` is exact for 20 steps. The Newton `qacc` at step 19 already matched `64156abe` after five iterations. The remaining 6 ulps were `mju_QCQP3`, the elliptic 3-tangent noslip solve: separate products, sums left to right.

## `qacc`

Both `robot_soccer_kit` models are exact for 20 steps. The dry-friction noslip update is `force − res·ARinv`. On the kit, that was the 1 ulp in `qfrc_constraint[39]` (`0xbfb09932ded2aee1` versus `0xbfb09932ded2aee0`) which moved `qacc[39]` by 2 ulps. `qacc_smooth` already matched.

## Step-exact models that still miss a forward state

Both `iit_softfoot` models are exact for 20 steps. At step 3 the first miss was `efc_aref[47]`, 4 ulps (`0x4020df8a083d10e6` versus `0x4020df8a083d10e2`), on the third row of a connect. `mju_dotSparseX3` accumulates `(jac·qvel) + acc`. The forward field counts do not move.

These 14 pass all 20 steps and still differ on a forward state. The field is the first name in the native stage order.

- `anybotics_anymal_b/scene.xml` `efc_J`
- `anybotics_anymal_c/scene.xml` `efc_J`
- `anybotics_anymal_c/scene_mjx.xml` `qacc`
- `berkeley_humanoid/scene.xml` `efc_J`
- `dynamixel_2r/scene.xml` `efc_AR`
- `hello_robot_stretch/scene.xml` `qacc`
- `hello_robot_stretch_3/stretch.xml` `qacc`
- `pndbotics_adam_lite/scene.xml` `efc_J`
- `shadow_dexee/shadow_dexee.xml` `qacc`
- `tetheria_aero_hand_open/left_hand.xml` `qfrc_spring`
- `tetheria_aero_hand_open/right_hand.xml` `qfrc_spring`
- `tetheria_aero_hand_open/scene_right.xml` `qfrc_spring`
- `umi_gripper/scene.xml` `qacc`
- `umi_gripper/umi_gripper.xml` `qacc`

## 4.x sites gated by the 3.13-only `contract_off`

`contract_off()` is true only for `64156abe`. `5491d70f` and `4849b2dd` are compiled with `-ffp-contract=off`, so a product added into an accumulator is `mulsd`/`mulpd` then `addsd`/`addsd`, not `vfmadd`. Each row is that disassembly, or the 4.x source when the symbol is the same instruction sequence in both libraries.

| site | 4.x form | before | result |
| --- | --- | --- | --- |
| noslip `x − L·xᵢ` | unfused subtract | fused on 4.x | unfused with the rows below |
| noslip `mju_addToScl` | groups of four `res + (scl·x)`, tail `(scl·x) + res` | fused | unfused. `efc_AR` 30/30 |
| noslip `res − R·f` | unfused subtract | fused | unfused |
| `mju_QCQP2` | `0x1a9a00`: 28 `mulsd`, 8 `addsd`, 4 `subsd`, no `vfmadd` | fused | unfused. This was the site compensating for a fused `AR`: `flybody/scene.xml` state 0 and `hello_robot_stretch_3/stretch.xml` state 2 went `SATISFIED` to `CONE` until `det = (A11+λ)(A22+λ) − A12²` and `v = −Pb` were separated |
| noslip `force − res·ARinv` | unfused | already `contract_off() \|\| oracle_4_0_schedules()` | agrees |
| `mju_sqrMatTDSparseNumeric` | `scale = D·vt`, then `acc + (scale·J)` | already the 4.x branch | agrees |
| `mju_QCQP3` | `mulsd`/`addsd`/`mulpd`, no `vfmadd` | already `contract_off_line()` | agrees |
| `mj_nextActivation` | `0x191ba0`: `mulsd` of `h` or `tau·(1−exp)`, then `addsd` into `act` | fused `mul_add` | unfused. Steps 196 to 204 and 195 to 203 |
| `mjc_ConvexHField` cell | source `(2.0·size) / (ncol−1)`, then `dx·c − size` | 4.x path is `2.0·size`, not the 3.13 `size+size` | agrees. Barkour `efc_J` is not this gate |
| mesh `PairHash` | `h1 ^ (h2 + 0x9e3779b9 + (h1<<6) + (h1>>2))` | already `oracle_4_0_schedules()` | agrees |
| `mju_wrap` inside | no `vfmadd`; first residual is the same separate sum as the later ones | `contract_off() \|\| oracle_4_0_schedules()` | agrees |
| quat integrate / `mji_mulMatVec3` | no `vfmadd` | `off() \|\| oracle_4_0_schedules()` | agrees |

Local `contract_off()` helpers in the smooth, CCD, collision, and tendon modules call `contract_off_line()`, so they are already unfused on 4.x. They are not this flag.
