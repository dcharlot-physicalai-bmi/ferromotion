# Direct-pull Thor2 plan

David approved a CPU-only download and build on thor2, with no credentials copied and no system changes. These scripts are the commands for that approval. Run one stage, read its receipt, then approve the next. Do not pipe a download into a shell.

The code under test is ferromotion `e0557dd109151a02702dfb262ea0a8c2b9dd07db`. This directory is the procedure that checks that commit out. It is not a substitute for that commit.

## What is public

Unauthenticated HTTPS returned 200 for all ten repositories, and `git fetch --depth 1` of each pinned commit works with `credential.helper` cleared, `GIT_TERMINAL_PROMPT=0`, and an askpass program that exits 1. No `Authorization` header is sent. Pins are in `pins.sh`: commit SHA and `COMMIT^{tree}` SHA.

| source | commit | tree |
| --- | --- | --- |
| ferromotion | `e0557dd109151a02702dfb262ea0a8c2b9dd07db` | `b30b3d5a5600110e1b871d931aceecfb284b5b67` |
| mujoco | `123347c0eeab7e13c8da0828ab593bbd95bcf335` | `c0408a44f164299fb200258a9226de55f508c59d` |
| mujoco_menagerie | `4d038b3feae26ec82b46a4d586379114012a8ac7` | `1fb749524d966e3ab6c52d94c7864e2c19a01de7` |
| libccd | `7931e764a19ef6b21b443376c699bbc9c6d4fba8` | `24104c36f6f32113dffaddd9e339f3cca64f9939` |
| qhull | `d1c2fc0caa5f644f3a0f220290d4a868c68ed4f6` | `d628b21c2a676f71cf13208dc49d7336fe698104` |
| tinyxml2 | `e6caeae85799003f4ca74ff26ee16a789bc2af48` | `b0095c6d2d899222047e22b4b164fc3979393c7e` |
| tinyobjloader | `2945a967c5303b2c8c14174117c45f3302591150` | `bade1d1afeefc9c96bef58af4445a06485d2e1b4` |
| lodepng | `17d08dd26cac4d63f43af217ebd70318bfb8189c` | `bfe03b87a19df25bf9c39df84f95f7b90551c87e` |
| miniz | `d10b03cc73475af673df40f06e5cefd1d5f940d9` | `33cd1ab4b9966f08dab259d63929fc001b69aab5` |
| MarchingCubeCpp | `f03a1b3ec29b1d7d865691ca8aea4f1eb2c2873d` | `a367ee1b40674af441f3984159dc17ba84c45a29` |

GitHub's `archive/<commit>.tar.gz` URLs also return 200. Those gzip files are not an identity: GitHub can change the compression wrapper. The scripts use `git fetch` of the commit so menagerie has a `.git` directory, which `run-212.sh` requires, and so the check is `HEAD` plus `HEAD^{tree}`.

`Cargo.lock` sha256 `a52b4c86fa3655b1c3f8dd1aafeb981031219b443152249a4cd9744142caf376` has no git dependencies. Stage 1 runs `cargo +1.99.1 vendor --locked` into the run directory. Stage 2 repeats resolution with `--offline --locked`. The build scripts then stay offline.

## What the 35385-file archive manifest is not

The V2 manifest, after the script overlay, has 35385 entries. A public checkout does not recreate those bytes, and these scripts do not claim that it does.

- `ferromotion/vendor/` is 28224 of those entries. It is `cargo vendor` output. Stage 1 vendors again from crates.io and records that vendor. It is not byte-compared to the archive vendor.
- Menagerie `.git` pack files are in the archive. A new clone's pack is a different byte string with the same commit and tree.
- `schedules/x86-native.txt` sha256 `a3168d763bed63c9181275b6a908348c0c0609fde425be1ece840c75d110b5b7` and `schedules/x86-step.txt` sha256 `a039f949d938cefc0cf3633edec1415497f8400a776b88809adf9bb06f5678be` are not in commit `e0557dd` and are not an upstream file. Stage 4 stops until those exact bytes are placed in the run directory. It does not download or synthesize them.
- Provenance receipts, `MANIFEST.sha256`, and `OVERLAY.txt` were produced while packing the archive.

ccd and qhull in the archive have the emscripten CMake edits. The patch files in `patches/` are those diffs. On Linux the `EMSCRIPTEN` branch is inactive. Stage 1 still applies them and stage 2 checks the patched `CMakeLists.txt` hashes `38275ef36861f46a833f193dd258e511325c33bc9f4b28c95262dad2958743a6` and `763fd80f648747858e8672d85bb3cb30820247201e6391a4cfd03af36dcc21e8`. lodepng's `CMakeLists.txt` is copied from the MuJoCo tree (`e3229a02ab1c3d239f73c6dff0ff321fc7dd2a59a0a07b35bccbeeee11761971`), which is where MuJoCo's own fetch puts it.

The mulquat fixture is the 338-byte file `fixtures/x86-contract-off-mulquat.txt`, sha256 `358875076e2be19ff4ee24d154a1189dfe5f3928cabf6da5da45f35930ce070f`.

## Stages

Each stage is one command. The run directory is `~/ferromotion-oracle/<ferromotion commit>-<UTC>/`. Set `FERROMOTION_ORACLE_HOME` only to move that parent; thor2 uses the default.

```text
bash 01-materialize.sh
bash 02-verify.sh <run>
bash 03-cbuild.sh <run>
bash 04-corpus.sh <run>
bash 05-rust.sh <run>
```

Stage 1 is the only network stage. It refuses a non-200 API response before cloning. `RUSTUP_AUTO_INSTALL=0`. It uses `~/.cargo/bin` and `cargo +1.99.1`. It does not modify `~/.gitconfig`, the shell profile, or the system toolchain. Registry data goes to `<run>/cargo-home`.

Stage 2 turns `CARGO_NET_OFFLINE=true` and checks every commit, tree, patched file, `Cargo.lock`, and `cargo metadata --offline --locked`.

Stage 3 calls `build-thor2.sh` after copying it into the run. That script allows only Linux aarch64 hostname `thor2`, clang 18.1.3, cmake 3.28.x, and ninja. Two compile jobs. `ulimit -t 14400` is per-process CPU seconds. `ulimit -v` is per-process virtual address space. `timeout -k 30 14400` is the wall-clock cap. No sudo, no apt, no GPU flag, no rustup. Compile flags stay `-O3 -DNDEBUG -fno-fast-math -ffp-contract=off -fno-math-errno -march=armv8-a`. The shared and module link adds `-rtlib=compiler-rt -unwindlib=libgcc` so `__muloti4` is defined inside `libmujoco` and the plugins. Each run uses a new `build-$STAMP` directory. `build-20261107T131346Z` is not reused.

Stage 4 stops with exit 2 while the two schedule files are absent or hashed differently, and it stops if `build-dir.txt` still names `build-20261107T131346Z`. When the new stage-3 directory is in place, it installs `run-probes.sh` and `run-212.sh` that link the self-contained library and do not append the builtins archive. A mulquat mismatch is exit 1 from the probe and is kept; the 212 dump still runs. `run-212.sh` exit 0 means the ARM dumps were written and the load set was 217 XML, 212 loads, and the five pinned failures. It does not mean the floats equal the x86 file. `compare-arm-x86.sh` then compares those dumps, bitwise, with `schedules/x86-native.txt` and `schedules/x86-step.txt` from oracle `mujoco-3.13.0-x86_64-contract-off` (libmujoco `64156abef7859ac3cedfa4f589c318ff642d1d5d8d0e078efabff21ab02040bc`). The verdict is `full_parity` or `first_divergence`, with per-field `models n/212` and native `states n/636` (step fields use `steps n/4240`). `stage4-ok` is written before a divergence exit so stage 5 can still read the ARM dumps. Exit 2 with `verdict not_run` means an x86 dump is missing or is not the pinned bytes.

Stage 5 runs `run-goldens.sh` and `run-rust-212.sh` with `--locked`, `--jobs 2`, offline mode, and a target directory outside the source tree. The Darwin tests are expected to fail on Linux; exit 1 there does not cancel the 212 comparison. `menagerie_native` and `menagerie_native_step` do not link `libmujoco`, so they do not take the compiler-rt archive. `run-rust-212.sh` exits 0 only for Rust-versus-ARM-dump parity: 212 models, 20 steps, every model stepped, and every `N of M` tally equal. That is not ARM-versus-x86 parity. A finished mismatch is exit 1. The receipt hashes the library, four plugins, schedules, executables, outputs, and toolchain.

## Schedule bytes

`x86-native.txt` is 73420440 bytes. `x86-step.txt` is 7582957 bytes. Of the native file, the `d` records are the recorded x86 field text. The loader ignores them when it reads `model`, `qpos`, `qvel`, `ctrl`, and `act`, but they are part of the pinned sha256. Recomputing them would be a different file. The fixtures `schedules/x86-native.txt.xz` (12611308 bytes, sha256 `1ed398c90a318fd9a514c7f6c8e14b966427c577f31410c872e90f84df537c4e`) and `schedules/x86-step.txt.xz` (2742944 bytes, sha256 `237fdfb8ea17eafe3a01f73c3418ecdc03af65b86e711f80d137353a1ccd55ed`) decompress with `xz -dc` to the original bytes. `materialize-schedules.sh` checks both the xz sha256 and the plaintext sha256.

## This Thor2 build

Stages 1 and 2 passed, and the compile linked. `write-receipt.sh` then exited 2 because `schedules/x86-native.txt` was not in the run directory, so `03-cbuild.sh` from `ecfccc6` wrote neither `stage3-ok` nor `build-dir.txt`. The libraries stayed at `build-20261107T131346Z`. `recover-stage3.sh` checks those files, the flags in `identity.txt` and `build.ninja`, and the stage3 logs. It does not run cmake. It writes `stage3-ok` only after the on-host `write-receipt.sh` succeeds.

## Approval boundary

Stages 1 and 2 are ready to approve. Stage 3 is ready to approve on thor2 after stage 2's receipt. Stages 4 and 5 are written and will refuse until the two schedule files are supplied by a separate decision. They are not fetched from the agent store.
