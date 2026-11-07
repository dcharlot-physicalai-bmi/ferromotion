# Compiled body mass and inertia, contract-off oracle

No fix landed. The published default is unchanged. Baseline ledger and golden arrays are unchanged. The three kinematics sites are unchanged, including the measured `qacc` regression from 24/636 with `mul_quat`+`rot_vec_quat` to 23/636 after `mul_mat_vec3` is also unfused. That regression was not re-measured, because neither the mass nor the contact-row defect was fixed.

## What was measured

Oracle: `mujoco-3.13.0-x86_64-contract-off`, artifact sha256 `64156abef7859ac3cedfa4f589c318ff642d1d5d8d0e078efabff21ab02040bc`. Worst `body_inertia` error in the 212-model native sweep is `hello_robot_stretch_3/stretch.xml`, absolute 3.533517429953547e-7, body 20 `link_SG3_gripper_body`.

A C probe linked to that library, and the same probe linked to the PyPI wheel, use the same `mjModel` offsets (`body_mass` 1944, `body_inertia` 1960). They do not agree:

| | mass | inertia |
| --- | --- | --- |
| contract-off source | 0.29 exactly | 0.00043548291967178881, 0.00038362346587136057, 0.00016903708934915525 |
| PyPI wheel and default Rust | 0.29 exactly | 0.00043552413190669587, 0.00038349071217081164, 0.00016939044109215057 |

Mass assignment matches. The body has one weighed geom, `mesh="link_SG3_gripper_body"` with explicit `mass="0.29"` (`class="visual"`). The collision geom of the same mesh has `mass="0"` from the `stretch` default and is not weighed. Rust and the oracle both see 14083 vertices and 23575 faces for that mesh. Equivalent-box half-sizes differ by about 1.9e-5, 4.0e-5 and 2.0e-5 m.

## Arithmetic branch that was tested and rejected

`mesh_inertia_mujoco` accumulates face area, volume and products of inertia with explicit `mul_add`. Rebuilding those accumulations as the contract-off C expressions (`mjuu_dot3`, `mjuu_crossvec`, and `P += volume/C * (2*(DD+EE+FF)+...)`) changes this mesh's volume by 1.084e-19 and leaves the body-inertia error at 3.533517429953e-7. That path was not committed.

NumPy `eigvalsh` of the Rust inertia tensor reproduces the Rust body inertia (0.00043552413190682044 and the two companions). Rust's `eig3` is consistent with its own tensor. The contract-off binary's tensor, or `mjuu_eig3` applied to that tensor inside the binary, is the unresolved branch. The tensor is not a field of `mjModel`, so it was not compared directly.

`body_mass` 192/212 and `body_inertia` 115/212 are therefore still the counts. The 21 op-clean models whose first native stage is `body_inertia`, and the 2 whose first stage is `body_mass`, are not closed. `body_mass` worst absolute error elsewhere is 4.44e-16 (`google_robot`), which is an ulp, unlike Stretch3's inertia.

## Contact rows, not started as a fix

`agilex_piper/piper.xml` `efc_J` length 56 versus 80 is `nv = 8` times `nefc` 7 versus 10. A C `mj_forward` at `qpos0` on both libraries reports `ncon = 1`, `condim = 3`, `nefc = 10`. The three missing Rust rows are the size of that one contact. Value tuning was not started.

`normalize4` remains the first covered kinematic op on 6 models when the three sites are unfused. It was not changed.
