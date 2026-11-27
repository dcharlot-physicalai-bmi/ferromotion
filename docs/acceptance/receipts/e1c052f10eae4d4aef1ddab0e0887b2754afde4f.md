# Contract-off oracle-profile receipt

Fresh measurement of `e1c052f10eae4d4aef1ddab0e0887b2754afde4f`. `cargo +1.99.1 clean`, then the receipt build, then both examples. The native summary and the field pin below were parsed from the native log this process just read. Equality is bitwise: each compared `f64` passes only when `to_bits` agrees. There is no tolerance. The profile is the oracle identity, not the published default.

The previous pin is `docs/acceptance/receipts/2aeb3611c9f3edca1961eb25ee374df56229d799.supersedes-c590d3a2.md`. That file is unchanged. Eleven of its field counts were short of the row total. This log raises all eleven, and none fall.

* ferromotion commit `e1c052f10eae4d4aef1ddab0e0887b2754afde4f`
* ferromotion tree `bf2071e96ce636b454b200d60d3f6088d45f0289`
* oracle id `mujoco-3.13.0-x86_64-contract-off`
* libmujoco sha256 pin `64156abef7859ac3cedfa4f589c318ff642d1d5d8d0e078efabff21ab02040bc`
* Menagerie `4d038b3feae26ec82b46a4d586379114012a8ac7`
* toolchain `rustc 1.99.1 (58cf8f9c3 2026-10-13)`, `cargo 1.99.1 (5f94df478 2026-08-27)`
* native log written `2026-11-27T13:42:22Z`
* step log written `2026-11-27T13:42:41Z`

The policy line for this measurement was:

```text
policy id=mujoco-3.13.0-x86_64-contract-off sha256=64156abef7859ac3cedfa4f589c318ff642d1d5d8d0e078efabff21ab02040bc mul_quat=Unfused rot_vec_quat=Unfused mul_mat_vec3=Contracted
```

`FERROMOTION_CONTRACT_SITES` was unset. Installing that id already selects unfused `mul_quat` and `rot_vec_quat`. `mul_mat_vec3` stays contracted.

`scripts/build_mujoco_3130_x86_contract_off.sh` produced `libmujoco.so.3.13.0`. `sha256sum` matched the pin above (4981136 bytes) before the comparison. The pinned oracle dumps were the files named below; they were not regenerated. The examples do not dlopen a plugin. This receipt does not re-pin one.

* oracle `x86-native.txt` sha256 `a3168d763bed63c9181275b6a908348c0c0609fde425be1ece840c75d110b5b7` (73420440 bytes)
* oracle `x86-step.txt` sha256 `a039f949d938cefc0cf3633edec1415497f8400a776b88809adf9bb06f5678be` (7582957 bytes)
* oracle `x86-mesh-graph.txt` sha256 `744105177f6ebfd14ffcd87602513f89af7bd38967c5c671cb3ba5314e50f51e` (14382748 bytes)

Commands, from a checkout of this commit, after `cargo +1.99.1 clean`:

```text
export FERROMOTION_BUILD_POLICY=mujoco-3.13.0-x86_64-contract-off
export FERROMOTION_ORACLE_SHA256=64156abef7859ac3cedfa4f589c318ff642d1d5d8d0e078efabff21ab02040bc
unset FERROMOTION_CONTRACT_SITES
cargo +1.99.1 build --release -p ferromotion-core --example menagerie_native --example menagerie_native_step
target/release/examples/menagerie_native <menagerie> <x86-native.txt>
MENAGERIE_LEDGER=<ledger> target/release/examples/menagerie_native_step <menagerie> <x86-step.txt>
```

Menagerie root was commit `4d038b3feae26ec82b46a4d586379114012a8ac7`. Both examples exited 0. That exit means they finished printing. It is not a claim that every field matched. A second `menagerie_native` run of this tree reproduced the native log hash below.

Comparison logs:

* native log sha256 `47956e7bdd28324a64f2afc81679c46094c7d8ceecdaf3a8cb6bc77b5c457db0`
* step log sha256 `577978e637fc4cbbd24368f202af17f74adf4536d5cd3c16cc430b1ede79ea5a`

## 20-step

20-step bit-exact models: 212 of 212. Loader refusals: 0.

The exact set is unchanged from the previous pin (`exact_set_sha256 caa71f582e61cf5eda1adcba7ff572da4ac4fb76596f8fd3a0f2232a1d5018bc`). Field counts that rose, and none fell: `efc_D` 635/636 to 636/636, `efc_J` 632/636 to 636/636, `efc_R` 634/636 to 636/636, `efc_aref` 633/636 to 636/636, `efc_b` 632/636 to 636/636, `efc_diagA` 635/636 to 636/636, `efc_force` 631/636 to 636/636, `efc_vel` 633/636 to 636/636, `qacc` 631/636 to 636/636, `qfrc_constraint` 631/636 to 636/636, `qfrc_spring` 630/636 to 636/636.

`efc_force`, `qacc`, and `qfrc_constraint` were 631 of 636. Each gained 5. The other eight short fields gained the rows that the same three contract-off sites had left unmatched. Every `mjData` field in this log is 636 of 636, and every `mjModel` field is 212 of 212.

Exact models (212):

* `agilex_piper/piper.xml`
* `agilex_piper/scene.xml`
* `agility_cassie/cassie.xml`
* `agility_cassie/scene.xml`
* `aloha/aloha.xml`
* `aloha/scene.xml`
* `anybotics_anymal_b/anymal_b.xml`
* `anybotics_anymal_b/scene.xml`
* `anybotics_anymal_c/anymal_c.xml`
* `anybotics_anymal_c/anymal_c_mjx.xml`
* `anybotics_anymal_c/scene.xml`
* `anybotics_anymal_c/scene_mjx.xml`
* `apptronik_apollo/apptronik_apollo.xml`
* `apptronik_apollo/scene.xml`
* `arx_l5/arx_l5.xml`
* `arx_l5/scene.xml`
* `berkeley_humanoid/berkeley_humanoid.xml`
* `berkeley_humanoid/scene.xml`
* `bitcraze_crazyflie_2/cf2.xml`
* `bitcraze_crazyflie_2/scene.xml`
* `booster_t1/scene.xml`
* `booster_t1/t1.xml`
* `boston_dynamics_spot/scene.xml`
* `boston_dynamics_spot/scene_arm.xml`
* `boston_dynamics_spot/spot.xml`
* `boston_dynamics_spot/spot_arm.xml`
* `dynamixel_2r/dynamixel_2r.xml`
* `dynamixel_2r/scene.xml`
* `flexiv_rizon4/flexiv_rizon4.xml`
* `flexiv_rizon4/scene.xml`
* `flexiv_rizon4s/flexiv_rizon4s.xml`
* `flexiv_rizon4s/scene.xml`
* `flybody/fruitfly.xml`
* `flybody/scene.xml`
* `fourier_n1/n1.xml`
* `fourier_n1/scene.xml`
* `franka_emika_panda/hand.xml`
* `franka_emika_panda/mjx_hand.xml`
* `franka_emika_panda/mjx_panda.xml`
* `franka_emika_panda/mjx_panda_nohand.xml`
* `franka_emika_panda/mjx_scene.xml`
* `franka_emika_panda/mjx_single_cube.xml`
* `franka_emika_panda/panda.xml`
* `franka_emika_panda/panda_nohand.xml`
* `franka_emika_panda/scene.xml`
* `franka_fr3/fr3.xml`
* `franka_fr3/scene.xml`
* `franka_fr3_v2/fr3v2.xml`
* `franka_fr3_v2/scene.xml`
* `google_barkour_v0/barkour_v0.xml`
* `google_barkour_v0/barkour_v0_mjx.xml`
* `google_barkour_v0/scene.xml`
* `google_barkour_v0/scene_barkour.xml`
* `google_barkour_v0/scene_mjx.xml`
* `google_barkour_vb/barkour_vb.xml`
* `google_barkour_vb/barkour_vb_mjx.xml`
* `google_barkour_vb/scene.xml`
* `google_barkour_vb/scene_hfield_mjx.xml`
* `google_barkour_vb/scene_mjx.xml`
* `google_robot/robot.xml`
* `google_robot/scene.xml`
* `hello_robot_stretch/scene.xml`
* `hello_robot_stretch/stretch.xml`
* `hello_robot_stretch_3/scene.xml`
* `hello_robot_stretch_3/stretch.xml`
* `i2rt_yam/scene.xml`
* `i2rt_yam/yam.xml`
* `iit_softfoot/scene.xml`
* `iit_softfoot/softfoot.xml`
* `kinova_gen3/gen3.xml`
* `kinova_gen3/scene.xml`
* `kuka_iiwa_14/iiwa14.xml`
* `kuka_iiwa_14/scene.xml`
* `leap_hand/left_hand.xml`
* `leap_hand/right_hand.xml`
* `leap_hand/scene_left.xml`
* `leap_hand/scene_right.xml`
* `low_cost_robot_arm/low_cost_robot_arm.xml`
* `low_cost_robot_arm/scene.xml`
* `ms_human_700/MS-Human-700-Locomotion.xml`
* `ms_human_700/MS-Human-700-Manipulation.xml`
* `ms_human_700/MS-Human-700.xml`
* `ms_human_700/scene.xml`
* `ms_human_700/scene_locomotion.xml`
* `ms_human_700/scene_manipulation.xml`
* `pal_talos/scene_motor.xml`
* `pal_talos/scene_position.xml`
* `pal_talos/talos.xml`
* `pal_talos/talos_motor.xml`
* `pal_talos/talos_position.xml`
* `pal_tiago/scene_motor.xml`
* `pal_tiago/scene_position.xml`
* `pal_tiago/scene_velocity.xml`
* `pal_tiago/tiago.xml`
* `pal_tiago/tiago_motor.xml`
* `pal_tiago/tiago_position.xml`
* `pal_tiago/tiago_velocity.xml`
* `pal_tiago_dual/scene_motor.xml`
* `pal_tiago_dual/scene_position.xml`
* `pal_tiago_dual/scene_velocity.xml`
* `pal_tiago_dual/tiago_dual.xml`
* `pal_tiago_dual/tiago_dual_motor.xml`
* `pal_tiago_dual/tiago_dual_position.xml`
* `pal_tiago_dual/tiago_dual_velocity.xml`
* `pndbotics_adam_lite/adam_lite.xml`
* `pndbotics_adam_lite/scene.xml`
* `rainbow_robotics_rby1/rby1a_1.2.xml`
* `rainbow_robotics_rby1/rby1a_1.2_no_gripper.xml`
* `rainbow_robotics_rby1/rby1m_1.2.xml`
* `rainbow_robotics_rby1/rby1m_1.2_no_gripper.xml`
* `rainbow_robotics_rby1/rby1m_1.3.xml`
* `rainbow_robotics_rby1/scene_rby1a_1.2.xml`
* `rainbow_robotics_rby1/scene_rby1a_1.2_no_gripper.xml`
* `rainbow_robotics_rby1/scene_rby1m_1.2.xml`
* `rainbow_robotics_rby1/scene_rby1m_1.2_no_gripper.xml`
* `rainbow_robotics_rby1/scene_rby1m_1.3.xml`
* `realsense_d435i/d435i.xml`
* `rethink_robotics_sawyer/sawyer.xml`
* `rethink_robotics_sawyer/scene.xml`
* `robot_soccer_kit/robot_soccer_kit.xml`
* `robot_soccer_kit/scene.xml`
* `robotiq_2f85/2f85.xml`
* `robotiq_2f85/scene.xml`
* `robotiq_2f85_v4/2f85.xml`
* `robotiq_2f85_v4/mjx_2f85.xml`
* `robotiq_2f85_v4/scene.xml`
* `robotis_omy_3m/omy_3m.xml`
* `robotis_omy_3m/scene.xml`
* `robotis_op3/op3.xml`
* `robotis_op3/scene.xml`
* `robotstudio_so101/scene.xml`
* `robotstudio_so101/scene_box.xml`
* `robotstudio_so101/so101.xml`
* `seeed_rebot_devarm/scene.xml`
* `seeed_rebot_devarm/seeed_rebot_devarm.xml`
* `shadow_dexee/scene.xml`
* `shadow_dexee/shadow_dexee.xml`
* `shadow_hand/left_hand.xml`
* `shadow_hand/right_hand.xml`
* `shadow_hand/scene_left.xml`
* `shadow_hand/scene_right.xml`
* `sharpa_wave/left_hand.xml`
* `sharpa_wave/right_hand.xml`
* `sharpa_wave/scene_left.xml`
* `sharpa_wave/scene_right.xml`
* `skydio_x2/scene.xml`
* `skydio_x2/x2.xml`
* `stanford_tidybot/base.xml`
* `stanford_tidybot/scene.xml`
* `stanford_tidybot/scene_base.xml`
* `stanford_tidybot/tidybot.xml`
* `tetheria_aero_hand_open/left_hand.xml`
* `tetheria_aero_hand_open/right_hand.xml`
* `tetheria_aero_hand_open/scene_right.xml`
* `toddlerbot_2xc/scene.xml`
* `toddlerbot_2xc/scene_mjx.xml`
* `toddlerbot_2xc/scene_pos.xml`
* `toddlerbot_2xc/toddlerbot_2xc.xml`
* `toddlerbot_2xc/toddlerbot_2xc_mjx.xml`
* `toddlerbot_2xc/toddlerbot_2xc_pos.xml`
* `toddlerbot_2xm/scene.xml`
* `toddlerbot_2xm/scene_mjx.xml`
* `toddlerbot_2xm/scene_pos.xml`
* `toddlerbot_2xm/toddlerbot_2xm.xml`
* `toddlerbot_2xm/toddlerbot_2xm_mjx.xml`
* `toddlerbot_2xm/toddlerbot_2xm_pos.xml`
* `trossen_vx300s/scene.xml`
* `trossen_vx300s/vx300s.xml`
* `trossen_wx250s/scene.xml`
* `trossen_wx250s/wx250s.xml`
* `trossen_wxai/scene.xml`
* `trossen_wxai/wxai_follower.xml`
* `trossen_wxai/wxai_follower_biarm.xml`
* `trs_so_arm100/scene.xml`
* `trs_so_arm100/so_arm100.xml`
* `ufactory_lite6/lite6.xml`
* `ufactory_lite6/lite6_gripper_narrow.xml`
* `ufactory_lite6/lite6_gripper_wide.xml`
* `ufactory_lite6/scene.xml`
* `ufactory_xarm7/hand.xml`
* `ufactory_xarm7/scene.xml`
* `ufactory_xarm7/xarm7.xml`
* `ufactory_xarm7/xarm7_nohand.xml`
* `umi_gripper/scene.xml`
* `umi_gripper/umi_gripper.xml`
* `unitree_a1/a1.xml`
* `unitree_a1/scene.xml`
* `unitree_g1/g1.xml`
* `unitree_g1/g1_mjx.xml`
* `unitree_g1/g1_with_hands.xml`
* `unitree_g1/scene.xml`
* `unitree_g1/scene_mjx.xml`
* `unitree_g1/scene_with_hands.xml`
* `unitree_go1/go1.xml`
* `unitree_go1/scene.xml`
* `unitree_go2/go2.xml`
* `unitree_go2/go2_mjx.xml`
* `unitree_go2/scene.xml`
* `unitree_go2/scene_mjx.xml`
* `unitree_h1/h1.xml`
* `unitree_h1/scene.xml`
* `unitree_z1/scene.xml`
* `unitree_z1/z1.xml`
* `unitree_z1/z1_gripper.xml`
* `universal_robots_ur10e/scene.xml`
* `universal_robots_ur10e/ur10e.xml`
* `universal_robots_ur5e/scene.xml`
* `universal_robots_ur5e/ur5e.xml`
* `wonik_allegro/left_hand.xml`
* `wonik_allegro/right_hand.xml`
* `wonik_allegro/scene_left.xml`
* `wonik_allegro/scene_right.xml`

```text
exact_set_sha256 caa71f582e61cf5eda1adcba7ff572da4ac4fb76596f8fd3a0f2232a1d5018bc
model agilex_piper/piper.xml
model agilex_piper/scene.xml
model agility_cassie/cassie.xml
model agility_cassie/scene.xml
model aloha/aloha.xml
model aloha/scene.xml
model anybotics_anymal_b/anymal_b.xml
model anybotics_anymal_b/scene.xml
model anybotics_anymal_c/anymal_c.xml
model anybotics_anymal_c/anymal_c_mjx.xml
model anybotics_anymal_c/scene.xml
model anybotics_anymal_c/scene_mjx.xml
model apptronik_apollo/apptronik_apollo.xml
model apptronik_apollo/scene.xml
model arx_l5/arx_l5.xml
model arx_l5/scene.xml
model berkeley_humanoid/berkeley_humanoid.xml
model berkeley_humanoid/scene.xml
model bitcraze_crazyflie_2/cf2.xml
model bitcraze_crazyflie_2/scene.xml
model booster_t1/scene.xml
model booster_t1/t1.xml
model boston_dynamics_spot/scene.xml
model boston_dynamics_spot/scene_arm.xml
model boston_dynamics_spot/spot.xml
model boston_dynamics_spot/spot_arm.xml
model dynamixel_2r/dynamixel_2r.xml
model dynamixel_2r/scene.xml
model flexiv_rizon4/flexiv_rizon4.xml
model flexiv_rizon4/scene.xml
model flexiv_rizon4s/flexiv_rizon4s.xml
model flexiv_rizon4s/scene.xml
model flybody/fruitfly.xml
model flybody/scene.xml
model fourier_n1/n1.xml
model fourier_n1/scene.xml
model franka_emika_panda/hand.xml
model franka_emika_panda/mjx_hand.xml
model franka_emika_panda/mjx_panda.xml
model franka_emika_panda/mjx_panda_nohand.xml
model franka_emika_panda/mjx_scene.xml
model franka_emika_panda/mjx_single_cube.xml
model franka_emika_panda/panda.xml
model franka_emika_panda/panda_nohand.xml
model franka_emika_panda/scene.xml
model franka_fr3/fr3.xml
model franka_fr3/scene.xml
model franka_fr3_v2/fr3v2.xml
model franka_fr3_v2/scene.xml
model google_barkour_v0/barkour_v0.xml
model google_barkour_v0/barkour_v0_mjx.xml
model google_barkour_v0/scene.xml
model google_barkour_v0/scene_barkour.xml
model google_barkour_v0/scene_mjx.xml
model google_barkour_vb/barkour_vb.xml
model google_barkour_vb/barkour_vb_mjx.xml
model google_barkour_vb/scene.xml
model google_barkour_vb/scene_hfield_mjx.xml
model google_barkour_vb/scene_mjx.xml
model google_robot/robot.xml
model google_robot/scene.xml
model hello_robot_stretch/scene.xml
model hello_robot_stretch/stretch.xml
model hello_robot_stretch_3/scene.xml
model hello_robot_stretch_3/stretch.xml
model i2rt_yam/scene.xml
model i2rt_yam/yam.xml
model iit_softfoot/scene.xml
model iit_softfoot/softfoot.xml
model kinova_gen3/gen3.xml
model kinova_gen3/scene.xml
model kuka_iiwa_14/iiwa14.xml
model kuka_iiwa_14/scene.xml
model leap_hand/left_hand.xml
model leap_hand/right_hand.xml
model leap_hand/scene_left.xml
model leap_hand/scene_right.xml
model low_cost_robot_arm/low_cost_robot_arm.xml
model low_cost_robot_arm/scene.xml
model ms_human_700/MS-Human-700-Locomotion.xml
model ms_human_700/MS-Human-700-Manipulation.xml
model ms_human_700/MS-Human-700.xml
model ms_human_700/scene.xml
model ms_human_700/scene_locomotion.xml
model ms_human_700/scene_manipulation.xml
model pal_talos/scene_motor.xml
model pal_talos/scene_position.xml
model pal_talos/talos.xml
model pal_talos/talos_motor.xml
model pal_talos/talos_position.xml
model pal_tiago/scene_motor.xml
model pal_tiago/scene_position.xml
model pal_tiago/scene_velocity.xml
model pal_tiago/tiago.xml
model pal_tiago/tiago_motor.xml
model pal_tiago/tiago_position.xml
model pal_tiago/tiago_velocity.xml
model pal_tiago_dual/scene_motor.xml
model pal_tiago_dual/scene_position.xml
model pal_tiago_dual/scene_velocity.xml
model pal_tiago_dual/tiago_dual.xml
model pal_tiago_dual/tiago_dual_motor.xml
model pal_tiago_dual/tiago_dual_position.xml
model pal_tiago_dual/tiago_dual_velocity.xml
model pndbotics_adam_lite/adam_lite.xml
model pndbotics_adam_lite/scene.xml
model rainbow_robotics_rby1/rby1a_1.2.xml
model rainbow_robotics_rby1/rby1a_1.2_no_gripper.xml
model rainbow_robotics_rby1/rby1m_1.2.xml
model rainbow_robotics_rby1/rby1m_1.2_no_gripper.xml
model rainbow_robotics_rby1/rby1m_1.3.xml
model rainbow_robotics_rby1/scene_rby1a_1.2.xml
model rainbow_robotics_rby1/scene_rby1a_1.2_no_gripper.xml
model rainbow_robotics_rby1/scene_rby1m_1.2.xml
model rainbow_robotics_rby1/scene_rby1m_1.2_no_gripper.xml
model rainbow_robotics_rby1/scene_rby1m_1.3.xml
model realsense_d435i/d435i.xml
model rethink_robotics_sawyer/sawyer.xml
model rethink_robotics_sawyer/scene.xml
model robot_soccer_kit/robot_soccer_kit.xml
model robot_soccer_kit/scene.xml
model robotiq_2f85/2f85.xml
model robotiq_2f85/scene.xml
model robotiq_2f85_v4/2f85.xml
model robotiq_2f85_v4/mjx_2f85.xml
model robotiq_2f85_v4/scene.xml
model robotis_omy_3m/omy_3m.xml
model robotis_omy_3m/scene.xml
model robotis_op3/op3.xml
model robotis_op3/scene.xml
model robotstudio_so101/scene.xml
model robotstudio_so101/scene_box.xml
model robotstudio_so101/so101.xml
model seeed_rebot_devarm/scene.xml
model seeed_rebot_devarm/seeed_rebot_devarm.xml
model shadow_dexee/scene.xml
model shadow_dexee/shadow_dexee.xml
model shadow_hand/left_hand.xml
model shadow_hand/right_hand.xml
model shadow_hand/scene_left.xml
model shadow_hand/scene_right.xml
model sharpa_wave/left_hand.xml
model sharpa_wave/right_hand.xml
model sharpa_wave/scene_left.xml
model sharpa_wave/scene_right.xml
model skydio_x2/scene.xml
model skydio_x2/x2.xml
model stanford_tidybot/base.xml
model stanford_tidybot/scene.xml
model stanford_tidybot/scene_base.xml
model stanford_tidybot/tidybot.xml
model tetheria_aero_hand_open/left_hand.xml
model tetheria_aero_hand_open/right_hand.xml
model tetheria_aero_hand_open/scene_right.xml
model toddlerbot_2xc/scene.xml
model toddlerbot_2xc/scene_mjx.xml
model toddlerbot_2xc/scene_pos.xml
model toddlerbot_2xc/toddlerbot_2xc.xml
model toddlerbot_2xc/toddlerbot_2xc_mjx.xml
model toddlerbot_2xc/toddlerbot_2xc_pos.xml
model toddlerbot_2xm/scene.xml
model toddlerbot_2xm/scene_mjx.xml
model toddlerbot_2xm/scene_pos.xml
model toddlerbot_2xm/toddlerbot_2xm.xml
model toddlerbot_2xm/toddlerbot_2xm_mjx.xml
model toddlerbot_2xm/toddlerbot_2xm_pos.xml
model trossen_vx300s/scene.xml
model trossen_vx300s/vx300s.xml
model trossen_wx250s/scene.xml
model trossen_wx250s/wx250s.xml
model trossen_wxai/scene.xml
model trossen_wxai/wxai_follower.xml
model trossen_wxai/wxai_follower_biarm.xml
model trs_so_arm100/scene.xml
model trs_so_arm100/so_arm100.xml
model ufactory_lite6/lite6.xml
model ufactory_lite6/lite6_gripper_narrow.xml
model ufactory_lite6/lite6_gripper_wide.xml
model ufactory_lite6/scene.xml
model ufactory_xarm7/hand.xml
model ufactory_xarm7/scene.xml
model ufactory_xarm7/xarm7.xml
model ufactory_xarm7/xarm7_nohand.xml
model umi_gripper/scene.xml
model umi_gripper/umi_gripper.xml
model unitree_a1/a1.xml
model unitree_a1/scene.xml
model unitree_g1/g1.xml
model unitree_g1/g1_mjx.xml
model unitree_g1/g1_with_hands.xml
model unitree_g1/scene.xml
model unitree_g1/scene_mjx.xml
model unitree_g1/scene_with_hands.xml
model unitree_go1/go1.xml
model unitree_go1/scene.xml
model unitree_go2/go2.xml
model unitree_go2/go2_mjx.xml
model unitree_go2/scene.xml
model unitree_go2/scene_mjx.xml
model unitree_h1/h1.xml
model unitree_h1/scene.xml
model unitree_z1/scene.xml
model unitree_z1/z1.xml
model unitree_z1/z1_gripper.xml
model universal_robots_ur10e/scene.xml
model universal_robots_ur10e/ur10e.xml
model universal_robots_ur5e/scene.xml
model universal_robots_ur5e/ur5e.xml
model wonik_allegro/left_hand.xml
model wonik_allegro/right_hand.xml
model wonik_allegro/scene_left.xml
model wonik_allegro/scene_right.xml
field mjData M 636 636
field mjData act_dot 636 636
field mjData actuator_force 636 636
field mjData actuator_length 636 636
field mjData actuator_velocity 636 636
field mjData cdof 636 636
field mjData cdof_dot 636 636
field mjData cinert 636 636
field mjData crb 636 636
field mjData cvel 636 636
field mjData efc_AR 30 30
field mjData efc_D 636 636
field mjData efc_J 636 636
field mjData efc_KBIP 636 636
field mjData efc_R 636 636
field mjData efc_aref 636 636
field mjData efc_b 636 636
field mjData efc_diagA 636 636
field mjData efc_force 636 636
field mjData efc_frictionloss 636 636
field mjData efc_id 636 636
field mjData efc_margin 636 636
field mjData efc_pos 636 636
field mjData efc_state 636 636
field mjData efc_type 636 636
field mjData efc_vel 636 636
field mjData nisland 636 636
field mjData qLD 636 636
field mjData qLDiagInv 636 636
field mjData qacc 636 636
field mjData qacc_smooth 636 636
field mjData qfrc_actuator 636 636
field mjData qfrc_bias 636 636
field mjData qfrc_constraint 636 636
field mjData qfrc_damper 636 636
field mjData qfrc_fluid 636 636
field mjData qfrc_gravcomp 636 636
field mjData qfrc_passive 636 636
field mjData qfrc_smooth 636 636
field mjData qfrc_spring 636 636
field mjData solver_niter 636 636
field mjData subtree_com 636 636
field mjData ten_length 636 636
field mjData ten_velocity 636 636
field mjData xanchor 636 636
field mjData xaxis 636 636
field mjData ximat 636 636
field mjData xipos 636 636
field mjData xmat 636 636
field mjData xpos 636 636
field mjData xquat 636 636
field mjModel M_colind 212 212
field mjModel M_rowadr 212 212
field mjModel M_rownnz 212 212
field mjModel body_dofadr 212 212
field mjModel body_dofnum 212 212
field mjModel body_inertia 212 212
field mjModel body_invweight0 212 212
field mjModel body_jntadr 212 212
field mjModel body_mass 212 212
field mjModel body_parentid 212 212
field mjModel body_rootid 212 212
field mjModel body_simple 212 212
field mjModel body_subtreemass 212 212
field mjModel dof_M0 212 212
field mjModel dof_armature 212 212
field mjModel dof_bodyid 212 212
field mjModel dof_invweight0 212 212
field mjModel dof_jntid 212 212
field mjModel dof_parentid 212 212
field mjModel dof_simplenum 212 212
field mjModel gravity 212 212
field mjModel jnt_bodyid 212 212
field mjModel jnt_dofadr 212 212
field mjModel tendon_invweight0 212 212
```

## Native per-field tally

Parsed from the native log named above. This text is the log's tally lines, not a summary copied from an earlier receipt.

```text
mjData  M                    636 of 636
mjData  act_dot              636 of 636
mjData  actuator_force       636 of 636
mjData  actuator_length      636 of 636
mjData  actuator_velocity    636 of 636
mjData  cdof                 636 of 636
mjData  cdof_dot             636 of 636
mjData  cinert               636 of 636
mjData  crb                  636 of 636
mjData  cvel                 636 of 636
mjData  efc_AR                30 of 30
mjData  efc_D                636 of 636
mjData  efc_J                636 of 636
mjData  efc_KBIP             636 of 636
mjData  efc_R                636 of 636
mjData  efc_aref             636 of 636
mjData  efc_b                636 of 636
mjData  efc_diagA            636 of 636
mjData  efc_force            636 of 636
mjData  efc_frictionloss     636 of 636
mjData  efc_id               636 of 636
mjData  efc_margin           636 of 636
mjData  efc_pos              636 of 636
mjData  efc_state            636 of 636
mjData  efc_type             636 of 636
mjData  efc_vel              636 of 636
mjData  nisland              636 of 636
mjData  qLD                  636 of 636
mjData  qLDiagInv            636 of 636
mjData  qacc                 636 of 636
mjData  qacc_smooth          636 of 636
mjData  qfrc_actuator        636 of 636
mjData  qfrc_bias            636 of 636
mjData  qfrc_constraint      636 of 636
mjData  qfrc_damper          636 of 636
mjData  qfrc_fluid           636 of 636
mjData  qfrc_gravcomp        636 of 636
mjData  qfrc_passive         636 of 636
mjData  qfrc_smooth          636 of 636
mjData  qfrc_spring          636 of 636
mjData  solver_niter         636 of 636
mjData  subtree_com          636 of 636
mjData  ten_length           636 of 636
mjData  ten_velocity         636 of 636
mjData  xanchor              636 of 636
mjData  xaxis                636 of 636
mjData  ximat                636 of 636
mjData  xipos                636 of 636
mjData  xmat                 636 of 636
mjData  xpos                 636 of 636
mjData  xquat                636 of 636
mjModel M_colind             212 of 212
mjModel M_rowadr             212 of 212
mjModel M_rownnz             212 of 212
mjModel body_dofadr          212 of 212
mjModel body_dofnum          212 of 212
mjModel body_inertia         212 of 212
mjModel body_invweight0      212 of 212
mjModel body_jntadr          212 of 212
mjModel body_mass            212 of 212
mjModel body_parentid        212 of 212
mjModel body_rootid          212 of 212
mjModel body_simple          212 of 212
mjModel body_subtreemass     212 of 212
mjModel dof_M0               212 of 212
mjModel dof_armature         212 of 212
mjModel dof_bodyid           212 of 212
mjModel dof_invweight0       212 of 212
mjModel dof_jntid            212 of 212
mjModel dof_parentid         212 of 212
mjModel dof_simplenum        212 of 212
mjModel gravity              212 of 212
mjModel jnt_bodyid           212 of 212
mjModel jnt_dofadr           212 of 212
mjModel tendon_invweight0    212 of 212
```
