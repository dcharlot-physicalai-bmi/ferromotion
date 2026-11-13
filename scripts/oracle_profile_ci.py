#!/usr/bin/env python3
"""Hash-check the contract-off oracle and run one named comparison.

The published Menagerie job is a different oracle (the PyPI wheel). This script
does not write that ledger and does not change its pass rule.

Exit 0 only when the library and the requested dump match the pin and the
comparison is bit-exact on every required row. A hash mismatch exits 2.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIN_PATH = ROOT / "conformance" / "oracle_x86_contract_off.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_file(path: Path, digest: str, nbytes: int | None, label: str) -> None:
    if not path.is_file():
        print(f"{label} missing: {path}", file=sys.stderr)
        sys.exit(2)
    got = sha256_file(path)
    size = path.stat().st_size
    if got != digest or (nbytes is not None and size != nbytes):
        print(f"{label} sha256 {got} size {size}; pin {digest} size {nbytes}", file=sys.stderr)
        sys.exit(2)
    print(f"{label} sha256 {got} ({size} bytes)")


def parse_native(text: str) -> dict[str, tuple[int, int]]:
    out: dict[str, tuple[int, int]] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 5 and parts[3] == "of" and parts[2].isdigit() and parts[4].isdigit():
            out[parts[1]] = (int(parts[2]), int(parts[4]))
    return out


def parse_step(text: str) -> tuple[int, int]:
    m = re.search(r"BIT FOR BIT at every one of \d+ steps: (\d+) of (\d+) models", text)
    if not m:
        print("step summary line missing", file=sys.stderr)
        sys.exit(1)
    return int(m.group(1)), int(m.group(2))


def run(cmd: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    merged = os.environ.copy()
    merged.pop("FERROMOTION_CONTRACT_SITES", None)
    merged.update(env)
    return subprocess.run(cmd, cwd=ROOT, env=merged, text=True, capture_output=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("native", "step"), required=True)
    parser.add_argument("--lib", type=Path, required=True)
    parser.add_argument("--menagerie", type=Path, required=True)
    parser.add_argument("--dump", type=Path, required=True)
    args = parser.parse_args()
    pin = json.loads(PIN_PATH.read_text())
    check_file(args.lib, pin["lib_sha256"], None, "libmujoco")
    if args.phase == "native":
        check_file(args.dump, pin["native_dump_sha256"], pin["native_dump_bytes"], "x86-native.txt")
        example = "menagerie_native"
    else:
        check_file(args.dump, pin["step_dump_sha256"], pin["step_dump_bytes"], "x86-step.txt")
        example = "menagerie_native_step"
    env = {
        "FERROMOTION_BUILD_POLICY": pin["oracle_id"],
        "FERROMOTION_ORACLE_SHA256": pin["lib_sha256"],
    }
    cmd = ["cargo", "+stable", "run", "--release", "-p", "ferromotion-core", "--example", example, "--", str(args.menagerie), str(args.dump)]
    if args.phase == "step":
        ledger = args.dump.parent / "oracle-step-ledger.jsonl"
        env["MENAGERIE_LEDGER"] = str(ledger)
    proc = run(cmd, env)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        return proc.returncode
    policy = "policy id=mujoco-3.13.0-x86_64-contract-off sha256=" + pin["lib_sha256"]
    if policy not in proc.stderr:
        print("policy line missing from stderr", file=sys.stderr)
        return 1
    if args.phase == "step":
        exact, compared = parse_step(proc.stdout)
        print(f"step exact {exact} of {compared}")
        if compared != pin["expected_step_exact"] or exact != pin["expected_step_exact"]:
            return 1
        return 0
    tally = parse_native(proc.stdout)
    states = pin["expected_native_states"]
    for name in pin["native_required"]:
        got = tally.get(name)
        print(f"native {name} {got}")
        if got != (states, states):
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
