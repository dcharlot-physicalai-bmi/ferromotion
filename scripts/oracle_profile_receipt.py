#!/usr/bin/env python3
"""Write a dated profile receipt from a finished comparison.

Every hash, count, model id, and timestamp is read from the files and
commands named on the command line. The model list is the ledger's
bit_exact rows, sorted. Its sha256 is the digest defined in
oracle_profile_gate.model_set_sha256.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

from oracle_profile_gate import exact_models, model_set_sha256, parse_fields, parse_step_summary, sha256_file

ROOT = Path(__file__).resolve().parents[1]
LIB_CONST = {
    "mujoco-4.0.0-x86_64-contract-off": "ORACLE_4_0_SHA256",
    "mujoco-4.0.1-x86_64-contract-off": "ORACLE_4_0_1_SHA256",
    "mujoco-3.13.0-x86_64-contract-off": "ORACLE_X86_CONTRACT_OFF_SHA256",
}


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def tool_version(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True).splitlines()[0]


def const_sha(name: str) -> str:
    text = (ROOT / "crates/ferromotion-core/src/fp_contract.rs").read_text()
    m = re.search(rf'pub const {name}: &str = "([0-9a-f]{{64}})";', text)
    if not m:
        raise SystemExit(f"{name} is not in fp_contract.rs")
    return m.group(1)


def plugin_lines(lib: Path, plugin_dir: Path | None) -> list[str]:
    directory = plugin_dir if plugin_dir is not None else lib.parent
    plugins = sorted(path for path in directory.glob("*.so") if path.is_file())
    if not plugins:
        raise SystemExit(f"no plugins in {directory}")
    lines = ["* plugins:"]
    for path in plugins:
        lines.append(f"  * `{path.name}` sha256 `{sha256_file(path)}`")
    lines.append("  These examples do not dlopen them.")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-id", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--lib", type=Path, required=True)
    parser.add_argument("--plugin-dir", type=Path)
    parser.add_argument("--menagerie", type=Path, required=True)
    parser.add_argument("--native-dump", type=Path, required=True)
    parser.add_argument("--step-dump", type=Path, required=True)
    parser.add_argument("--native-log", type=Path, required=True)
    parser.add_argument("--step-log", type=Path, required=True)
    parser.add_argument("--stderr", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--started", required=True)
    parser.add_argument("--finished", required=True)
    args = parser.parse_args()

    const = LIB_CONST[args.oracle_id]
    lib_sha = sha256_file(args.lib)
    if lib_sha != const_sha(const):
        raise SystemExit(f"libmujoco {lib_sha} does not match {const}")
    menagerie = git("-C", str(args.menagerie), "rev-parse", "HEAD")
    pin = json.loads((ROOT / "conformance/menagerie_pin.json").read_text())["menagerie_sha"]
    if menagerie != pin:
        raise SystemExit(f"menagerie {menagerie} does not match pin {pin}")
    commit = args.commit
    if git("rev-parse", f"{commit}^{{commit}}") != commit:
        raise SystemExit(f"{commit} is not a full commit id")
    tree = git("rev-parse", f"{commit}^{{tree}}")
    policy = next(line for line in args.stderr.read_text().splitlines() if line.startswith("policy id="))
    if args.oracle_id not in policy or lib_sha not in policy:
        raise SystemExit("stderr policy line does not name this oracle")
    steps, exact, compared, refusals = parse_step_summary(args.step_log.read_text())
    if steps != 20:
        raise SystemExit(f"step log ran {steps} steps, not 20")
    models = sorted(exact_models(args.ledger))
    if len(models) != exact:
        raise SystemExit(f"ledger {len(models)} != step exact {exact}")
    fields = parse_fields(args.native_log.read_text())
    if not fields:
        raise SystemExit("native log has no per-field counts")
    set_sha = model_set_sha256(models)
    native_sha = sha256_file(args.native_log)
    step_sha = sha256_file(args.step_log)
    native_dump_sha = sha256_file(args.native_dump)
    step_dump_sha = sha256_file(args.step_dump)
    rustc = tool_version(["rustc", "+1.99.1", "-vV"])
    cargo = tool_version(["cargo", "+1.99.1", "-vV"])
    rel = f"docs/acceptance/receipts/{commit}.md"
    out = ROOT / rel
    body = "\n".join(
        [
            "# Contract-off oracle-profile receipt",
            "",
            "Regenerated on this machine. Equality is bitwise: each compared `f64` passes only when `to_bits` agrees. There is no tolerance. The profile is the oracle identity, not the published default.",
            "",
            f"* ferromotion commit `{commit}`",
            f"* ferromotion tree `{tree}`",
            f"* oracle id `{args.oracle_id}`",
            f"* libmujoco sha256 pin `{lib_sha}`",
            *plugin_lines(args.lib, args.plugin_dir),
            f"* Menagerie `{menagerie}`",
            f"* toolchain `{rustc}`, `{cargo}`",
            f"* UTC `{args.started}` through `{args.finished}`",
            "",
            "The policy line printed by the examples was:",
            "",
            "```text",
            policy,
            "```",
            "",
            "`FERROMOTION_CONTRACT_SITES` was unset. Installing that id already selects unfused `mul_quat` and `rot_vec_quat`. `mul_mat_vec3` stays contracted.",
            "",
            "The comparison read the pinned library and the pinned oracle dumps after `sha256sum` matched the pin and the dump hashes below. No second rebuild of libmujoco was used.",
            "",
            f"* oracle `x86-native.txt` sha256 `{native_dump_sha}` ({args.native_dump.stat().st_size} bytes)",
            f"* oracle `x86-step.txt` sha256 `{step_dump_sha}` ({args.step_dump.stat().st_size} bytes)",
            "",
            "Commands, from a checkout of this commit:",
            "",
            "```text",
            f"export FERROMOTION_BUILD_POLICY={args.oracle_id}",
            f"export FERROMOTION_ORACLE_SHA256={lib_sha}",
            "unset FERROMOTION_CONTRACT_SITES",
            "cargo +1.99.1 build --release -p ferromotion-core --example menagerie_native --example menagerie_native_step",
            "target/release/examples/menagerie_native <menagerie> <x86-native.txt>",
            "MENAGERIE_LEDGER=<ledger> target/release/examples/menagerie_native_step <menagerie> <x86-step.txt>",
            "```",
            "",
            f"Menagerie root was commit `{menagerie}`. Both examples exited 0. That exit means they finished printing. It is not a claim that every field matched.",
            "",
            "Comparison logs:",
            "",
            f"* native log sha256 `{native_sha}`",
            f"* step log sha256 `{step_sha}`",
            "",
            "## 20-step",
            "",
            f"20-step bit-exact models: {exact} of {compared}. Loader refusals: {refusals}.",
            "",
            f"* exact model set sha256 `{set_sha}`",
            "",
            "The exact model set is the ledger rows whose status is `bit_exact`, sorted. The sha256 covers those ids, one UTF-8 line each, with a trailing newline.",
            "",
            f"Exact models ({exact}):",
            "",
            *[f"* `{model}`" for model in models],
            "",
            "## Native per-field tally",
            "",
            "```text",
            args.native_log.read_text().rstrip("\n"),
            "```",
            "",
        ]
    )
    # The sentence above is the definition implemented by model_set_sha256.
    if model_set_sha256(models) != set_sha:
        raise SystemExit("model set sha256 changed while writing")
    if len(parse_fields(body)) != len(fields):
        raise SystemExit("embedded native log lost per-field lines")
    out.write_text(body)
    digest = hashlib.sha256(body.encode()).hexdigest()
    sidecar = Path(str(out) + ".sha256")
    sidecar.write_text(f"{digest}  {rel}\n")
    print(f"wrote {rel}")
    print(f"receipt sha256 {digest}")
    print(f"exact-model-set sha256 {set_sha}")
    print(f"native log sha256 {native_sha}")
    print(f"step log sha256 {step_sha}")
    print(f"20-step {exact} of {compared}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
