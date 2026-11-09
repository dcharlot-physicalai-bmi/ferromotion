#!/usr/bin/env bash
# Probe fixture laid out like Thor2: the real mulquat file, a stage-3 directory,
# five libraries, compiler-rt, and a probe output directory. Runs the real
# fixture parser and the real direct-pull receipt.
set -euo pipefail
export LC_ALL=C
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
pull="$(cd "$here/.." && pwd)"
parser="$root/read-mulquat-fixture.sh"
fixture="$pull/fixtures/x86-contract-off-mulquat.txt"
receipt_src="$pull/direct-pull-receipt.sh"
data='3fe8000000000000 bfc5c28f5c28f5c2 3fc3333333333333 3fdc28f5c28f5c29'
comment_hit="$(awk 'NF==4 {print; exit}' "$fixture")"
[[ "$comment_hit" == "# oracle id mujoco-3.13.0-x86_64-contract-off" ]]
got="$("$parser" "$fixture")"
[[ "$got" == "$data" ]]
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
printf '%s\n' '# only a comment' >"$tmp/comments.txt"
if "$parser" "$tmp/comments.txt" >"$tmp/out" 2>"$tmp/err"; then
  echo "comments-only fixture was accepted" >&2
  exit 1
fi
grep -q '0 data lines' "$tmp/err"
printf '%s\n%s\n' "$data" "$data" >"$tmp/two.txt"
if "$parser" "$tmp/two.txt" >"$tmp/out" 2>"$tmp/err"; then
  echo "two data lines were accepted" >&2
  exit 1
fi
grep -q '2 data lines' "$tmp/err"
printf '%s\n' 'not hex fields here' >"$tmp/bad.txt"
if "$parser" "$tmp/bad.txt" >"$tmp/out" 2>"$tmp/err"; then
  echo "non-hex data line was accepted" >&2
  exit 1
fi
grep -q 'not four 16-digit hex fields' "$tmp/err"

# shellcheck source=../lib.sh
source "$pull/lib.sh"
direct_pull_accept_stage3_build \
  /home/cbio/ferromotion-oracle/123347c0eeab7e13c8da0828ab593bbd95bcf335/build-20261109T213225Z
set +e
direct_pull_accept_stage3_build \
  /home/cbio/ferromotion-oracle/123347c0eeab7e13c8da0828ab593bbd95bcf335/build-20261107T131346Z
old_rc=$?
set -e
[[ "$old_rc" -eq 2 ]]
grep -q 'build-20261109T213225Z' "$pull/lib.sh"
grep -q 'read-mulquat-fixture.sh' "$pull/04-corpus.sh"
grep -q 'direct_pull_accept_stage3_build' "$pull/04-corpus.sh"
grep -q "NF==4" "$root/run-probes.sh" && {
  echo "run-probes.sh still uses NF==4" >&2
  exit 1
}
grep -q 'NF==4' "$pull/compare-arm-x86.sh" && {
  echo "compare-arm-x86.sh parses with NF==4" >&2
  exit 1
}
grep -F -q 'native\.txt and' "$pull/05-rust.sh"

log="$tmp/stage4-212.log"
cat >"$log" <<'EOF'
# wrote /tmp/comment/native.txt and /tmp/comment/step.txt
wrote /tmp/probe/RECEIPT.txt
wrote /home/cbio/ferromotion-oracle/123347c0eeab7e13c8da0828ab593bbd95bcf335/run-20261109T213225Z/native.txt and /home/cbio/ferromotion-oracle/123347c0eeab7e13c8da0828ab593bbd95bcf335/run-20261109T213225Z/step.txt
EOF
corpus_run="$(grep -E '^wrote .+\/native\.txt and .+\/step\.txt$' "$log" | tail -n 1)"
corpus_run="${corpus_run#wrote }"
corpus_run="${corpus_run%% and *}"
corpus_run="$(dirname "$corpus_run")"
[[ "$corpus_run" == /home/cbio/ferromotion-oracle/123347c0eeab7e13c8da0828ab593bbd95bcf335/run-20261109T213225Z ]]

run="$tmp/run"
stage3="$tmp/build-20261109T213225Z"
libdir="$stage3/cmake/lib"
rt="$tmp/libclang_rt.builtins-aarch64.a"
probe="$tmp/probe-20261109T213225Z"
mkdir -p "$run/scripts" "$run/probes" "$run/fixtures" "$run/schedules" \
  "$run/provenance" "$run/ferromotion/scripts/arm_oracle" "$run/ferromotion/.cargo" \
  "$libdir" "$probe"
printf 'builtins\n' >"$rt"
rt_sha="$(sha256sum "$rt" | awk '{print $1}')"
cat >"$stage3/identity.txt" <<'EOF'
flags -O3 -DNDEBUG -fno-fast-math -ffp-contract=off -fno-math-errno -march=armv8-a
shared_link -rtlib=compiler-rt -unwindlib=libgcc
EOF
cat >"$stage3/compiler-rt.txt" <<EOF
compiler_rt_query clang -print-libgcc-file-name -rtlib=compiler-rt
compiler_rt_path $rt
compiler_rt_sha256 $rt_sha
EOF
for lib in libmujoco.so.3.13.0 libmujoco.so libactuator.so libelasticity.so libsensor.so libsdf_plugin.so; do
  printf 'so %s\n' "$lib" >"$libdir/$lib"
done
lib_sha="$(sha256sum "$libdir/libmujoco.so.3.13.0" | awk '{print $1}')"
# shellcheck source=../../compiler-rt.sh
source "$root/compiler-rt.sh"
arm_oracle_write_role_identity "$probe/identity.txt" probe "$stage3"
grep -F -x -q 'identity_role probe' "$probe/identity.txt"
printf 'probe\n' >"$probe/mulquat"
printf '%s\n' "$data" >"$probe/mulquat-arm64.txt"
printf '%s\n' "$data" >"$probe/mulquat-x86.txt"
echo 'mulquat_bits match_x86_contract_off' >"$probe/result.txt"
cp -f "$fixture" "$run/fixtures/x86-contract-off-mulquat.txt"
cp -f "$pull/probes/mulquat.c" "$run/probes/mulquat.c"
cp -f "$root/compiler-rt.sh" "$run/scripts/compiler-rt.sh"
cp -f "$parser" "$run/scripts/read-mulquat-fixture.sh"
cp -f "$receipt_src" "$run/scripts/write-receipt.sh"
chmod +x "$run/scripts/write-receipt.sh" "$run/scripts/read-mulquat-fixture.sh"
for rel in \
  probes/oracle_dump.c probes/pyhex.h probes/hexfloat_selftest.c \
  probes/expected-load-failures.txt scripts/limits.sh scripts/apply-overlay.sh \
  scripts/accept-rust-212.sh scripts/build-thor2.sh scripts/run-212.sh \
  scripts/run-goldens.sh scripts/run-probes.sh scripts/run-rust-212.sh \
  ferromotion/scripts/arm_oracle/README.md \
  ferromotion/scripts/arm_oracle/OPTIONAL-RUSTUP.md \
  ferromotion/Cargo.lock ferromotion/.cargo/config.toml \
  provenance/stage2-ok provenance/cargo-vv.txt \
  schedules/x86-native.txt schedules/x86-step.txt
do
  mkdir -p "$run/$(dirname "$rel")"
  [[ -f "$run/$rel" ]] || printf 'fixture %s\n' "$rel" >"$run/$rel"
done
for rel in src/mujoco deps/ccd deps/qhull deps/tinyxml2 deps/tinyobjloader \
  deps/lodepng deps/miniz deps/marchingcubecpp menagerie ferromotion
do
  mkdir -p "$run/$rel"
  git -C "$run/$rel" init -q
  git -C "$run/$rel" config user.email test@example.com
  git -C "$run/$rel" config user.name test
  printf 'x\n' >"$run/$rel/README"
  git -C "$run/$rel" add README
  git -C "$run/$rel" commit -q -m init
done

run_receipt() {
  bash "$run/scripts/write-receipt.sh" "$@"
}

if run_receipt "$probe" "$libdir" probe >"$tmp/pin.out" 2>"$tmp/pin.err"; then
  echo "receipt accepted unpinned git heads" >&2
  exit 1
fi
grep -q 'HEAD' "$tmp/pin.err"
grep -q 'is not' "$tmp/pin.err"

rm -f "$probe/identity.txt"
if FERROMOTION_RECEIPT_LAYOUT_TEST=1 run_receipt "$probe" "$libdir" probe >"$tmp/no-id.out" 2>"$tmp/no-id.err"; then
  echo "probe receipt succeeded without identity.txt" >&2
  exit 1
fi
grep -q 'identity.txt' "$tmp/no-id.out" "$tmp/no-id.err"
arm_oracle_write_role_identity "$probe/identity.txt" probe "$stage3"
FERROMOTION_RECEIPT_LAYOUT_TEST=1 run_receipt "$probe" "$libdir" probe >"$tmp/ok.out" 2>"$tmp/ok.err"
grep -q 'identity_role probe' "$probe/RECEIPT.txt"
grep -q "compiler_rt_sha256 $rt_sha" "$probe/RECEIPT.txt"
grep -q "$lib_sha" "$probe/RECEIPT.txt"
grep -q 'mulquat-arm64.txt' "$probe/RECEIPT.txt"
grep -q "$data" "$probe/mulquat-arm64.txt"

bare="$tmp/build-no-identity"
mkdir -p "$bare"
if FERROMOTION_RECEIPT_LAYOUT_TEST=1 run_receipt "$bare" "$libdir" build >"$tmp/build.out" 2>"$tmp/build.err"; then
  echo "build receipt succeeded without identity.txt" >&2
  exit 1
fi
grep -q 'identity.txt' "$tmp/build.out" "$tmp/build.err"
FERROMOTION_RECEIPT_LAYOUT_TEST=1 run_receipt "$stage3" "$libdir" build >"$tmp/build-ok.out" 2>"$tmp/build-ok.err"
grep -q 'shared_link -rtlib=compiler-rt -unwindlib=libgcc' "$stage3/RECEIPT.txt"
grep -q "compiler_rt_sha256 $rt_sha" "$stage3/RECEIPT.txt"

corpus="$tmp/run-stamp"
mkdir -p "$corpus"
arm_oracle_write_role_identity "$corpus/identity.txt" corpus "$stage3"
for f in oracle_dump classify.txt failures.txt native.txt step.txt libdir.txt; do
  printf 'corpus\n' >"$corpus/$f"
done
printf '%s\n' "$libdir" >"$corpus/libdir.txt"
FERROMOTION_RECEIPT_LAYOUT_TEST=1 run_receipt "$corpus" "$libdir" corpus >"$tmp/corpus.out" 2>"$tmp/corpus.err"
grep -q 'identity_role corpus' "$corpus/RECEIPT.txt"
grep -q "compiler_rt_sha256 $rt_sha" "$corpus/RECEIPT.txt"

rust="$tmp/rust212-stamp"
mkdir -p "$rust"
arm_oracle_write_role_identity "$rust/identity.txt" rust "$stage3"
for f in native.txt step.txt result.txt accept.txt; do
  printf 'rust\n' >"$rust/$f"
done
FERROMOTION_RECEIPT_LAYOUT_TEST=1 run_receipt "$rust" "$libdir" rust >"$tmp/rust.out" 2>"$tmp/rust.err"
grep -q 'identity_role rust' "$rust/RECEIPT.txt"
grep -q 'rust_links_libmujoco no' "$rust/RECEIPT.txt"
grep -q "compiler_rt_sha256 $rt_sha" "$rust/RECEIPT.txt"

goldens="$tmp/goldens-stamp"
mkdir -p "$goldens"
printf 'identity_role goldens\n' >"$goldens/identity.txt"
printf 'cargo\n' >"$goldens/cargo-test.txt"
printf 'result\n' >"$goldens/result.txt"
if FERROMOTION_RECEIPT_LAYOUT_TEST=1 run_receipt "$goldens" /nonexistent goldens >"$tmp/g.out" 2>"$tmp/g.err"; then
  :
else
  echo "goldens receipt failed" >&2
  cat "$tmp/g.err" >&2
  exit 1
fi
grep -q 'does not bind libmujoco' "$goldens/RECEIPT.txt"
rm -f "$goldens/identity.txt"
if FERROMOTION_RECEIPT_LAYOUT_TEST=1 run_receipt "$goldens" /nonexistent goldens >"$tmp/g2.out" 2>"$tmp/g2.err"; then
  echo "goldens receipt succeeded without identity.txt" >&2
  exit 1
fi
grep -q 'identity.txt' "$tmp/g2.out" "$tmp/g2.err"
echo "probe-receipt-fixture ok"
