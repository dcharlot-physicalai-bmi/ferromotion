#!/usr/bin/env python3
"""Hash-check the contract-off oracle and compare it to the latest receipt.

The published Menagerie job is a different oracle (the PyPI wheel). This script
does not write that ledger and does not change its pass rule.

The count pin is the receipt path and sha256 in
``conformance/oracle_x86_contract_off.json``. The file is hashed before it is
read. Selection does not ask git whether a commit is an ancestor.
That receipt carries the sorted 20-step exact model ids, ``exact_set_sha256``
of those ids (one per line, trailing newline), and every ``mjData`` /
``mjModel`` tally line.

``--write-receipt`` parses the native summary from the native log it just
read. A summary string passed in with ``--native-summary`` is refused.

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


class UncomputedNativeSummary(ValueError):
    """A native summary was offered without being parsed from a log just read."""


class ComputedNativeSummary:
    """Summary text parsed from native-log bytes in this process."""

    def __init__(self, text: str, log_sha256: str, token: int) -> None:
        self.text = text
        self.log_sha256 = log_sha256
        self.token = token


_LIVE_SUMMARIES: set[int] = set()
_SUMMARY_TOKEN = 0


def native_summary_lines(log_text: str) -> list[str]:
    """Tally and note lines, taken only from this native log text."""
    lines: list[str] = []
    for raw in log_text.splitlines():
        line = raw.strip()
        parts = line.split()
        if len(parts) < 2 or parts[0] not in ("mjData", "mjModel"):
            continue
        tally = (
            len(parts) >= 5
            and parts[3] == "of"
            and parts[2].isdigit()
            and parts[4].isdigit()
            and ":" not in parts[1]
        )
        if tally or ":" in parts[1]:
            lines.append(line)
    if not any(" of " in line and ":" not in line.split()[1] for line in lines):
        raise ValueError("native log has no tally lines")
    return lines


def pin_field_lines(log_text: str) -> list[str]:
    """``field mjData name same total`` lines, in the log's tally order."""
    lines: list[str] = []
    seen: set[str] = set()
    for raw in log_text.splitlines():
        parts = raw.split()
        if (
            len(parts) >= 5
            and parts[0] in ("mjData", "mjModel")
            and parts[3] == "of"
            and parts[2].isdigit()
            and parts[4].isdigit()
            and ":" not in parts[1]
        ):
            key = f"{parts[0]} {parts[1]}"
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"field {parts[0]} {parts[1]} {int(parts[2])} {int(parts[4])}")
    if not lines:
        raise ValueError("native log has no tally lines")
    return lines


def compute_native_summary(log_bytes: bytes) -> ComputedNativeSummary:
    """Parse a native summary from log bytes this call just received."""
    global _SUMMARY_TOKEN
    text = "\n".join(native_summary_lines(log_bytes.decode())) + "\n"
    _SUMMARY_TOKEN += 1
    summary = ComputedNativeSummary(text, hashlib.sha256(log_bytes).hexdigest(), _SUMMARY_TOKEN)
    _LIVE_SUMMARIES.add(summary.token)
    return summary


def write_native_summary(summary: object) -> str:
    """Return a summary parsed in this process. Any other value is refused."""
    if not isinstance(summary, ComputedNativeSummary) or summary.token not in _LIVE_SUMMARIES:
        raise UncomputedNativeSummary(
            "refusing to write a native summary this process did not just compute"
        )
    _LIVE_SUMMARIES.remove(summary.token)
    return summary.text


def first_stage(log_text: str, model: str) -> str:
    prefix = f"first_stage\t{model}\t"
    for line in log_text.splitlines():
        if line.startswith(prefix):
            return line.split("\t", 2)[2]
    raise ValueError(f"native log has no first_stage for {model}")


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


def receipt_rank(code_time: int, file_time: int, name: str) -> tuple[int, int, str]:
    """Sort key kept for the self-test. Receipt selection does not use it."""
    return (code_time, file_time, name)


def pinned_receipt() -> Path:
    """The receipt named by path and sha256. No git ancestry lookup."""
    artifact = json.loads(PIN_PATH.read_text())
    rel = artifact["receipt"]
    digest = artifact["receipt_sha256"]
    path = ROOT / rel
    if not path.is_file():
        raise ValueError(f"pinned receipt missing: {rel}")
    got = sha256_file(path)
    if got != digest:
        raise ValueError(f"pinned receipt {rel} sha256 {got} != {digest}")
    return path


def render_receipt(
    *,
    native_log: bytes,
    step_log: bytes,
    models: list[str],
    code_commit: str,
    tree: str,
    superseded: str,
    previous_fields: dict[str, tuple[int, int]],
    gained_states: list[str],
    native_finished: str,
    step_finished: str,
    artifact: dict,
) -> str:
    """Receipt whose field pin and native summary come from ``native_log``."""
    if not re.fullmatch(r"[0-9a-f]{40}", code_commit):
        raise ValueError("code commit is not 40 hex characters")
    if not re.fullmatch(r"[0-9a-f]{40}", tree):
        raise ValueError("tree is not 40 hex characters")
    if not re.fullmatch(r"[0-9a-f]{40}", superseded):
        raise ValueError("superseded receipt commit is not 40 hex characters")
    summary = write_native_summary(compute_native_summary(native_log))
    native_sha = hashlib.sha256(native_log).hexdigest()
    step_sha = hashlib.sha256(step_log).hexdigest()
    log_text = native_log.decode()
    fields = pin_field_lines(log_text)
    parsed = parse_native(log_text)
    if len(fields) != len(parsed):
        raise ValueError("pin field lines do not match the parsed tally")
    exact, compared = parse_step_summary(step_log.decode())
    ordered = sorted(models)
    if exact != len(ordered) or compared != len(ordered):
        raise ValueError(f"step summary {exact} of {compared} does not match {len(ordered)} ledger models")
    digest = exact_set_sha256(ordered)
    deltas = []
    for key in sorted(set(previous_fields) | set(parsed)):
        old = previous_fields.get(key)
        new = parsed.get(key)
        if old != new:
            deltas.append((key, old, new))
    if not deltas:
        raise ValueError("native log matches the superseded field counts")
    for key, old, new in deltas:
        if old is None or new is None or new[0] - old[0] != len(gained_states):
            raise ValueError(f"field {key} did not rise by the gained-state count")
    carried = ", ".join(
        f"`{name}` {previous_fields['mjData ' + name][0]} of {previous_fields['mjData ' + name][1]}"
        for name in ("efc_force", "qacc", "qfrc_constraint")
    )
    gained = ", ".join(f"`{item}`" for item in gained_states)
    delta_text = ", ".join(
        f"`{key.split()[1]}` {old[0]}/{old[1]} to {new[0]}/{new[1]}" for key, old, new in deltas
    )
    stage_models = list(dict.fromkeys(item.rsplit(" state ", 1)[0] for item in gained_states))
    stage_lines = "\n".join(
        f"* `{model}` `first_stage` `{first_stage(log_text, model)}`" for model in stage_models
    )
    bullets = "\n".join(f"* `{model}`" for model in ordered)
    pin = "\n".join([f"exact_set_sha256 {digest}", *[f"model {model}" for model in ordered], *fields])
    lib_sha = artifact["lib_sha256"]
    return f"""# Contract-off oracle-profile receipt

Corrects the receipt committed in `{superseded}`. That receipt commit is superseded. Its {len(previous_fields)} field lines and its native summary were the 211 receipt's text, carried forward. `menagerie_native` was not relinked after the elliptic three-tangent change in `{code_commit}`, so that summary still recorded {carried} and the native log stayed the 211 hash. The step binary did contain the change, so the step log and the 212 exact-set hash in the superseded receipt were the real measurement.

This file is a fresh regeneration. `cargo +1.99.1 clean`, then the receipt build, then both examples. The native summary and the field pin below were parsed from the native log this process just read. Equality is bitwise: each compared `f64` passes only when `to_bits` agrees. There is no tolerance. The profile is the oracle identity, not the published default.

* ferromotion commit `{code_commit}`
* ferromotion tree `{tree}`
* superseded receipt commit `{superseded}`
* oracle id `{artifact["oracle_id"]}`
* libmujoco sha256 pin `{lib_sha}`
* plugins: `libactuator.so` beside that library, sha256 `d4e8a8c9f09cc48ea0d6c93e173f5ac84df6010108a9d1c05b14c2f9e3b19e6b`, the same pin as the superseded receipt. These examples do not dlopen it. The 212 compared models do not require it.
* Menagerie `{artifact["menagerie_sha"]}`
* toolchain `rustc 1.99.1 (58cf8f9c3 2026-10-13)`, `cargo 1.99.1 (5f94df478 2026-08-27)`
* native log written `{native_finished}`
* step log written `{step_finished}`

The policy line for this measurement was:

```text
policy id={artifact["oracle_id"]} sha256={lib_sha} mul_quat=Unfused rot_vec_quat=Unfused mul_mat_vec3=Contracted
```

`FERROMOTION_CONTRACT_SITES` was unset. Installing that id already selects unfused `mul_quat` and `rot_vec_quat`. `mul_mat_vec3` stays contracted.

The comparison read the pinned library and the pinned oracle dumps after `sha256sum` matched the pin and the dump hashes below. No second rebuild of libmujoco was used.

* oracle `x86-native.txt` sha256 `{artifact["native_dump_sha256"]}` ({artifact["native_dump_bytes"]} bytes)
* oracle `x86-step.txt` sha256 `{artifact["step_dump_sha256"]}` ({artifact["step_dump_bytes"]} bytes)
* oracle `x86-mesh-graph.txt` sha256 `{artifact["graph_dump_sha256"]}` ({artifact["graph_dump_bytes"]} bytes)

Commands, from a checkout of this commit, after `cargo +1.99.1 clean`:

```text
export FERROMOTION_BUILD_POLICY={artifact["oracle_id"]}
export FERROMOTION_ORACLE_SHA256={lib_sha}
unset FERROMOTION_CONTRACT_SITES
cargo +1.99.1 build --release -p ferromotion-core --example menagerie_native --example menagerie_native_step
target/release/examples/menagerie_native <menagerie> <x86-native.txt>
MENAGERIE_LEDGER=<ledger> target/release/examples/menagerie_native_step <menagerie> <x86-step.txt>
```

Menagerie root was commit `{artifact["menagerie_sha"]}`. Both examples exited 0. That exit means they finished printing. It is not a claim that every field matched. A second `menagerie_native` run of this tree reproduced the native log hash below.

Comparison logs:

* native log sha256 `{native_sha}`
* step log sha256 `{step_sha}`

## 20-step

20-step bit-exact models: {exact} of {compared}. Loader refusals: 0.

The exact set is unchanged from the superseded receipt. Field counts that rose, and none fell: {delta_text}.

Compared with `05c3e1afea72fc526832b39a25a08c45f72bd4f2`, the states that became exact are {gained}. Rebuilding that commit's noslip solver and running `menagerie_native` on `shadow_dexee` reports a mismatch only at state 0 of each model, for `efc_force`, `qacc`, and `qfrc_constraint`. This native log's first stage for those models is:

{stage_lines}

Exact models ({len(ordered)}):

{bullets}

```text
{pin}
```

## Native per-field tally

Parsed from the native log named above. This text is the log's tally and note lines, not a summary copied from an earlier receipt.

```text
{summary.rstrip()}
```
"""


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
    copied = "mjData  efc_force            629 of 636   worst 1 on a/y.xml state 2\n"
    fresh = (
        "first_stage\tshadow_dexee/scene.xml\tno_listed_stage_mismatch\n"
        "  mjData  efc_force            631 of 636   worst 1 on a/y.xml state 2\n"
        "  mjData  efc_force: a/y.xml state 2: 1 vs 1 values\n"
    ).encode()
    try:
        write_native_summary(copied)
    except UncomputedNativeSummary:
        pass
    else:
        print("self-test wrote a native summary it did not compute", file=sys.stderr)
        return 1
    computed = compute_native_summary(fresh)
    try:
        write_native_summary(computed.text)
    except UncomputedNativeSummary:
        pass
    else:
        print("self-test wrote a summary string that merely matched the log", file=sys.stderr)
        return 1
    written = write_native_summary(computed)
    if written != "mjData  efc_force            631 of 636   worst 1 on a/y.xml state 2\nmjData  efc_force: a/y.xml state 2: 1 vs 1 values\n":
        print("self-test summary was not the log's tally", written, file=sys.stderr)
        return 1
    try:
        write_native_summary(computed)
    except UncomputedNativeSummary:
        pass
    else:
        print("self-test wrote the same summary twice", file=sys.stderr)
        return 1
    if pin_field_lines(fresh.decode()) != ["field mjData efc_force 631 636"]:
        print("self-test pin lines were not parsed from the log", file=sys.stderr)
        return 1
    stale_name = "2aeb3611c9f3edca1961eb25ee374df56229d799.md"
    fresh_name = "2aeb3611c9f3edca1961eb25ee374df56229d799.supersedes-c590d3a2.md"
    if receipt_rank(100, 1, stale_name) > receipt_rank(100, 2, fresh_name):
        print("self-test let an older receipt file beat the correction", file=sys.stderr)
        return 1
    if receipt_rank(100, 5, fresh_name) <= receipt_rank(100, 5, stale_name):
        print("self-test let the stale filename beat the correction when times match", file=sys.stderr)
        return 1
    pinned = pinned_receipt()
    parse_pin_block(pinned.read_text())
    print("self-test ok: a regression fails, an unreceipted gain fails, a matching receipt passes")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("native", "step", "graph"))
    parser.add_argument("--lib", type=Path)
    parser.add_argument("--menagerie", type=Path)
    parser.add_argument("--dump", type=Path)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--write-receipt", action="store_true")
    parser.add_argument("--native-log", type=Path)
    parser.add_argument("--step-log", type=Path)
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--code-commit")
    parser.add_argument("--tree")
    parser.add_argument("--supersedes")
    parser.add_argument("--previous-receipt", type=Path)
    parser.add_argument("--gained-state", action="append", default=[])
    parser.add_argument("--native-finished")
    parser.add_argument("--step-finished")
    parser.add_argument(
        "--native-summary",
        help="Refused. The native summary is parsed from --native-log.",
    )
    args = parser.parse_args()
    if args.native_summary is not None:
        print(
            "refusing to write a native summary this process did not just compute",
            file=sys.stderr,
        )
        return 2
    if args.self_test:
        return self_test()
    if args.write_receipt:
        required = (
            args.native_log,
            args.step_log,
            args.ledger,
            args.out,
            args.code_commit,
            args.tree,
            args.supersedes,
            args.previous_receipt,
            args.native_finished,
            args.step_finished,
        )
        if any(item is None for item in required) or not args.gained_state:
            parser.error("--write-receipt needs the log, ledger, pin, and gained-state arguments")
        previous_fields = parse_pin_block(args.previous_receipt.read_text())[2]
        text = render_receipt(
            native_log=args.native_log.read_bytes(),
            step_log=args.step_log.read_bytes(),
            models=ledger_exact(args.ledger),
            code_commit=args.code_commit,
            tree=args.tree,
            superseded=args.supersedes,
            previous_fields=previous_fields,
            gained_states=args.gained_state,
            native_finished=args.native_finished,
            step_finished=args.step_finished,
            artifact=json.loads(PIN_PATH.read_text()),
        )
        args.out.write_text(text)
        print(f"wrote {args.out}")
        return 0
    if args.phase is None or args.lib is None or args.menagerie is None or args.dump is None:
        parser.error("--phase, --lib, --menagerie, and --dump are required")
    artifact = json.loads(PIN_PATH.read_text())
    check_file(args.lib, artifact["lib_sha256"], None, "libmujoco")
    if args.phase == "native":
        check_file(args.dump, artifact["native_dump_sha256"], artifact["native_dump_bytes"], "x86-native.txt")
        example = "menagerie_native"
    elif args.phase == "graph":
        check_file(args.dump, artifact["graph_dump_sha256"], artifact["graph_dump_bytes"], "x86-mesh-graph.txt")
        example = "menagerie_qhull"
    else:
        check_file(args.dump, artifact["step_dump_sha256"], artifact["step_dump_bytes"], "x86-step.txt")
        example = "menagerie_native_step"
    receipt_path = pinned_receipt()
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
    if args.phase == "graph":
        match = re.search(r"identical (\d+), different (\d+), refused (\d+), no oracle (\d+)", proc.stdout)
        if match is None:
            print("mesh_graph summary missing", file=sys.stderr)
            return 1
        identical, different, refused, missing = (int(match.group(i)) for i in range(1, 5))
        print(f"mesh_graph {identical} identical, {different} different, {refused} refused, {missing} without an oracle line")
        if different or refused or missing:
            return 1
        return 0
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
