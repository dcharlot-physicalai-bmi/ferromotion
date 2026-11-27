# Optional isolated Rust toolchain

Do not run this until David has approved the host modification on a parent approval card.
Nothing in `scripts/` installs Rust. Do not change a shell profile. Do not set the default
toolchain for any other user. Do not overwrite `~/.rustup` or `~/.cargo` if they already exist.
If `~/.cargo/bin/rustc` and `~/.cargo/bin/cargo` already exist and `cargo -vV` reports a
release of at least 1.89, use those absolute paths and skip this install.
Project `rust-version` is 1.89. The x86 compiler that checked these scripts was 1.99.1.

Official bootstrap binary, aarch64-unknown-linux-gnu:
`https://static.rust-lang.org/rustup/dist/aarch64-unknown-linux-gnu/rustup-init`
sha256 `9f608c142e0e38a78957d0661ba1dfaf0b6fb37184cfaa4b9a0aa57af6184282`.
Re-download the sibling `rustup-init.sha256` from that directory and refuse the file if the digest differs.
Toolchain to install: `1.99.1`.
Isolated homes, under the oracle work directory, not the user default:

```text
REV=123347c0eeab7e13c8da0828ab593bbd95bcf335
BASE="$HOME/ferromotion-oracle/$REV"
export RUSTUP_HOME="$BASE/rustup"
export CARGO_HOME="$BASE/cargo"
curl -fsSL -o "$BASE/rustup-init" \
  https://static.rust-lang.org/rustup/dist/aarch64-unknown-linux-gnu/rustup-init
echo "9f608c142e0e38a78957d0661ba1dfaf0b6fb37184cfaa4b9a0aa57af6184282  $BASE/rustup-init" | sha256sum -c
chmod +x "$BASE/rustup-init"
"$BASE/rustup-init" -y --no-modify-path --default-host aarch64-unknown-linux-gnu --default-toolchain 1.99.1
```

`--no-modify-path` does not edit `~/.profile` or `~/.bashrc`.

Ferromotion source is already in this bundle at `ferromotion/`, archived commit
`7e500d4530a460303ec20e3e3a05294dd31b1700`, plus the script overlay recorded in
`provenance/OVERLAY.txt`. `vendor/` and `.cargo/config.toml` are already there.
That config replaces crates.io with `vendored-sources` and sets `net.offline = true`.
It does not pass `--locked`, and it does not set `CARGO_BUILD_JOBS` or `CARGO_TARGET_DIR`.
`scripts/run-goldens.sh` and `scripts/run-rust-212.sh` do.

Do not clone over this tree. Do not copy it onto a Mac. Do not run `cargo fetch`.
There is no aarch64 ferromotion binary in this bundle. This x86 cloud host did not
cross-link one. An unexecuted cross binary would be a fabricated ARM run.

After the toolchain exists:

```text
export PATH="$CARGO_HOME/bin:$PATH"
export CARGO_HOME="$BASE/cargo"
export RUSTUP_HOME="$BASE/rustup"
./scripts/run-goldens.sh
./scripts/run-rust-212.sh <run-dir>
```

Those scripts set `CARGO_NET_OFFLINE=true`, `CARGO_BUILD_JOBS=2`, `--locked`, `--jobs 2`,
and a fresh `CARGO_TARGET_DIR` under `$HOME/ferromotion-oracle/$REV/`, outside the source tree.
`--test-threads=2` only limits how many tests run at once. It is not the compile job cap.

`run-goldens.sh` runs the three embedded Darwin tests. A failure means those arrays
did not match. It is not a Linux oracle, and the script does not rewrite the tests.

`run-rust-212.sh` compares `menagerie_native` and `menagerie_native_step` to
`<run-dir>/native.txt` and `<run-dir>/step.txt` from `run-212.sh`.
Exit 0 only when all 212 models are compared, all 20 steps are present, every model
stepped, and every `N of M` tally has N = M. Exit 1 means that comparison finished
and did not match. Exit 3 means cargo failed, the summary is missing, or the counts
are not 212 models and 20 steps. The example programs themselves still return 0
after printing mismatches. This wrapper does not treat that as acceptance.

Do not run `menagerie_cpu_dump`. Do not run a Python oracle. This bundle does not
install a MuJoCo Python package. `schedules/x86-native.txt` and `schedules/x86-step.txt`
are the replay inputs. The ARM dumps are what `run-212.sh` writes.
