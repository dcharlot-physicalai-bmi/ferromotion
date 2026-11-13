#!/usr/bin/env python3
"""Hash-check the contract-off oracle and compare it to the latest receipt.

The published Menagerie job is a different oracle (the PyPI wheel). This script
does not write that ledger and does not change its pass rule.

The count pin is the newest receipt under ``docs/acceptance/receipts`` whose
named commit is an ancestor of HEAD. That receipt carries the sorted 20-step
exact model ids, ``exact_set_sha256`` of those ids (one per line, trailing
newline), and every ``mjData`` / ``mjModel`` tally line.

A pinned model that is no longer exact, or a per-field count that drops, fails.
A larger exact set or a higher per-field count fails until a receipt records it.
The library and dump hashes stay in ``conformance/oracle_x86_contract_off.json``.
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
RECEIPTS = ROOT / "docs" / "acceptance" / "receipts"
COMMIT_RE = re.compile(r"ferromotion commit `([0-9a-f]{40})`")
BULLET_RE = re.compile(r"^\* `([^`]+)`$")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def exact_set_sha256(models: list[str]) -> str:
    payload = "".join(model + "\n" for model in sorted(models))
    return hashlib.sha256(payload.encode()).hexdigest()


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
    """Map ``mjData name`` / ``mjModel name`` to ``(same, of)`` from tally lines."""
    out: dict[str, tuple[int, int]] = {}
    for line in text.splitlines():
        parts = line.split()
        if (
            len(parts) >= 5
            and parts[0] in ("mjData", "mjModel")
            and parts[3] == "of"
            and parts[2].isdigit()
            and parts[4].isdigit()
            and ":" not in parts[1]
        ):
            out[f"{parts[0]} {parts[1]}"] = (int(parts[2]), int(parts[4]))
    return out


def parse_step_summary(text: str) -> tuple[int, int]:
    match = re.search(r"BIT FOR BIT at every one of \d+ steps: (\d+) of (\d+) models", text)
    if not match:
        raise ValueError("step summary line missing")
    return int(match.group(1)), int(match.group(2))


def bullet_models(text: str) -> list[str]:
    return [match.group(1) for line in text.splitlines() if (match := BULLET_RE.match(line.strip()))]


def parse_pin_block(text: str) -> tuple[str, list[str], dict[str, tuple[int, int]]]:
    start = text.find("```text\nexact_set_sha256 ")
    if start < 0:
        raise ValueError("receipt has no exact_set_sha256 block")
    body = text[start + len("```text\n") :].split("```", 1)[0]
    lines = [line for line in body.splitlines() if line.strip()]
    recorded = lines[0].split()
    if len(recorded) != 2 or recorded[0] != "exact_set_sha256":
        raise ValueError("exact_set_sha256 line is malformed")
    models: list[str] = []
    fields: dict[str, tuple[int, int]] = {}
    for line in lines[1:]:
        parts = line.split()
        if parts[0] == "model" and len(parts) == 2:
            models.append(parts[1])
        elif parts[0] == "field" and len(parts) == 5 and parts[1] in ("mjData", "mjModel"):
            fields[f"{parts[1]} {parts[2]}"] = (int(parts[3]), int(parts[4]))
        else:
            raise ValueError(f"unreadable pin line: {line}")
    digest = exact_set_sha256(models)
    if digest != recorded[1]:
        raise ValueError(f"exact_set_sha256 {recorded[1]} does not hash the model list ({digest})")
    if models != sorted(models):
        raise ValueError("exact model ids are not sorted")
    bullets = bullet_models(text)
    if bullets and bullets != models:
        raise ValueError("Exact models list does not match the pin block")
    return recorded[1], models, fields


def receipt_commit(text: str) -> str | None:
    match = COMMIT_RE.search(text)
    return match.group(1) if match else None


def commit_time(sha: str) -> int | None:
    proc = subprocess.run(
        ["git", "-C", str(ROOT), "log", "-1", "--format=%ct", sha],
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0 or not proc.stdout.strip().isdigit():
        return None
    return int(proc.stdout.strip())


def is_ancestor(sha: str) -> bool:
    proc = subprocess.run(
        ["git", "-C", str(ROOT), "merge-base", "--is-ancestor", sha, "HEAD"],
        capture_output=True,
    )
    return proc.returncode == 0


def latest_receipt(directory: Path | None = None) -> Path:
    directory = directory or RECEIPTS
    ranked: list[tuple[int, Path]] = []
    for path in sorted(directory.glob("*.md")):
        sha = receipt_commit(path.read_text())
        if sha is None:
            continue
        when = commit_time(sha)
        if when is None or not is_ancestor(sha):
            continue
        ranked.append((when, path))
    if not ranked:
        raise ValueError(f"no receipt under {directory} names an ancestor of HEAD")
    return max(ranked)[1]


def judge(
    pinned_models: list[str],
    pinned_fields: dict[str, tuple[int, int]],
    observed_models: list[str] | None,
    observed_fields: dict[str, tuple[int, int]] | None,
) -> list[str]:
    errors: list[str] = []
    if observed_models is not None:
        pinned = set(pinned_models)
        observed = set(observed_models)
        left = sorted(pinned - observed)
        joined = sorted(observed - pinned)
        if left:
            errors.append("regression: left " + ", ".join(left))
        if joined:
            errors.append("unreceipted gain: joined " + ", ".join(joined))
    if observed_fields is not None:
        for key, (same, total) in pinned_fields.items():
            got = observed_fields.get(key)
            if got is None:
                errors.append(f"regression: missing {key}")
                continue
            got_same, got_total = got
            if got_same < same or got_total < total:
                errors.append(f"regression: {key} {got_same} of {got_total} below {same} of {total}")
            elif got_same > same or got_total > total:
                errors.append(f"unreceipted gain: {key} {got_same} of {got_total} above {same} of {total}")
        for key in sorted(set(observed_fields) - set(pinned_fields)):
            got_same, got_total = observed_fields[key]
            errors.append(f"unreceipted gain: new {key} {got_same} of {got_total}")
    return errors


def ledger_exact(path: Path) -> list[str]:
    models = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("status") == "bit_exact":
            models.append(row["model"])
    return sorted(models)


def run(cmd: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    merged = os.environ.copy()
    merged.pop("FERROMOTION_CONTRACT_SITES", None)
    merged.update(env)
    return subprocess.run(cmd, cwd=ROOT, env=merged, text=True, capture_output=True)


def self_test() -> int:
    models = ["b/x.xml", "a/y.xml"]
    digest = exact_set_sha256(models)
    fields = {"mjData efc_J": (10, 636), "mjModel body_mass": (212, 212)}
    block = "\n".join(
        ["```text", f"exact_set_sha256 {digest}", "model a/y.xml", "model b/x.xml", "field mjData efc_J 10 636", "field mjModel body_mass 212 212", "```"]
    )
    text = "* `a/y.xml`\n* `b/x.xml`\n" + block + "\n"
    got_digest, got_models, got_fields = parse_pin_block(text)
    if got_digest != digest or got_models != sorted(models) or got_fields != fields:
        print("self-test did not parse the pin", file=sys.stderr)
        return 1
    if judge(got_models, got_fields, sorted(models), fields):
        print("self-test rejected a matching receipt", file=sys.stderr)
        return 1
    regressed = judge(got_models, got_fields, ["a/y.xml"], {"mjData efc_J": (9, 636), "mjModel body_mass": (212, 212)})
    if not any(item.startswith("regression: left") for item in regressed) or not any("efc_J" in item and item.startswith("regression:") for item in regressed):
        print("self-test did not fail a regression", regressed, file=sys.stderr)
        return 1
    gained = judge(got_models, got_fields, sorted(models + ["c/z.xml"]), {"mjData efc_J": (11, 636), "mjModel body_mass": (212, 212)})
    if not any(item.startswith("unreceipted gain: joined") for item in gained) or not any("efc_J" in item and item.startswith("unreceipted gain:") for item in gained):
        print("self-test did not fail an unreceipted gain", gained, file=sys.stderr)
        return 1
    raised_models = sorted(models + ["c/z.xml"])
    raised_fields = {"mjData efc_J": (11, 636), "mjModel body_mass": (212, 212)}
    if judge(raised_models, raised_fields, raised_models, raised_fields):
        print("self-test rejected a receipt that records the gain", file=sys.stderr)
        return 1
    bad = text.replace(digest, "0" * 64)
    try:
        parse_pin_block(bad)
    except ValueError:
        pass
    else:
        print("self-test accepted a pin whose hash does not match the set", file=sys.stderr)
        return 1
    print("self-test ok: a regression fails, an unreceipted gain fails, a matching receipt passes")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("native", "step"))
    parser.add_argument("--lib", type=Path)
    parser.add_argument("--menagerie", type=Path)
    parser.add_argument("--dump", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if args.phase is None or args.lib is None or args.menagerie is None or args.dump is None:
        parser.error("--phase, --lib, --menagerie, and --dump are required")
    artifact = json.loads(PIN_PATH.read_text())
    check_file(args.lib, artifact["lib_sha256"], None, "libmujoco")
    if args.phase == "native":
        check_file(args.dump, artifact["native_dump_sha256"], artifact["native_dump_bytes"], "x86-native.txt")
        example = "menagerie_native"
    else:
        check_file(args.dump, artifact["step_dump_sha256"], artifact["step_dump_bytes"], "x86-step.txt")
        example = "menagerie_native_step"
    receipt_path = latest_receipt()
    _digest, pinned_models, pinned_fields = parse_pin_block(receipt_path.read_text())
    print(f"receipt {receipt_path.name} exact {len(pinned_models)} fields {len(pinned_fields)}")
    env = {
        "FERROMOTION_BUILD_POLICY": artifact["oracle_id"],
        "FERROMOTION_ORACLE_SHA256": artifact["lib_sha256"],
    }
    cmd = [
        "cargo",
        "+stable",
        "run",
        "--release",
        "-p",
        "ferromotion-core",
        "--example",
        example,
        "--",
        str(args.menagerie),
        str(args.dump),
    ]
    ledger = args.dump.parent / "oracle-step-ledger.jsonl"
    if args.phase == "step":
        env["MENAGERIE_LEDGER"] = str(ledger)
    proc = run(cmd, env)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        return proc.returncode
    policy = "policy id=mujoco-3.13.0-x86_64-contract-off sha256=" + artifact["lib_sha256"]
    if policy not in proc.stderr:
        print("policy line missing from stderr", file=sys.stderr)
        return 1
    if args.phase == "step":
        exact, compared = parse_step_summary(proc.stdout)
        observed = ledger_exact(ledger)
        print(f"step exact {exact} of {compared}; ledger {len(observed)}")
        if exact != len(observed):
            print("step summary count does not match the ledger", file=sys.stderr)
            return 1
        errors = judge(pinned_models, pinned_fields, observed, None)
    else:
        errors = judge(pinned_models, pinned_fields, None, parse_native(proc.stdout))
    for error in errors:
        print(error, file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
