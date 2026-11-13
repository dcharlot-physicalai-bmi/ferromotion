#!/usr/bin/env python3
"""Pin a profile run to the latest receipt for its oracle id.

The latest receipt in docs/acceptance/receipts for that id is the pin.
Its sidecar must match the file bytes. A run passes when the sorted
20-step-exact model set and every per-field count equal that receipt.
A model leaving the set, or a per-field count dropping, fails. A larger
set or a higher count fails until a new receipt records that set's sha256
and those counts. Field names are compared as written; a rename is a loss
of the old name. Nothing in this script is a numeric pin.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECEIPTS = ROOT / "docs" / "acceptance" / "receipts"

STEP_RE = re.compile(
    r"BIT FOR BIT at every one of (\d+) steps: (\d+) of (\d+) models stepped \((\d+) the loader refuses\)"
)
FIELD_RE = re.compile(r"^\s*(mjData|mjModel)\s+(\S+)\s+(\d+) of (\d+)\s*(?:worst\b.*)?$")
ORACLE_RE = re.compile(r"^\* oracle id `([^`]+)`$")
UTC_RE = re.compile(r"^\* UTC `([^`]+)` through `([^`]+)`$")
SET_RE = re.compile(r"^\* exact model set sha256 `([0-9a-f]{64})`$")
COUNT_RE = re.compile(r"^20-step bit-exact models: (\d+) of (\d+)\. Loader refusals: (\d+)\.$")
LIB_RE = re.compile(r"^\* libmujoco sha256 pin `([0-9a-f]{64})`$")
DUMP_RE = re.compile(r"^\* oracle `([^`]+)` sha256 `([0-9a-f]{64})` \((\d+) bytes\)$")
MODEL_RE = re.compile(r"^\* `([^`]+)`$")


@dataclass
class Receipt:
    path: Path
    oracle_id: str
    utc_end: str
    exact: int
    compared: int
    refusals: int
    models: tuple[str, ...]
    model_set_sha256: str
    stated_model_set_sha256: str | None
    fields: dict[str, tuple[int, int]]
    lib_sha256: str
    dumps: dict[str, tuple[str, int]]


@dataclass
class Run:
    steps: int
    exact: int
    compared: int
    refusals: int
    models: tuple[str, ...]
    fields: dict[str, tuple[int, int]]
    model_set_sha256: str = ""

    def __post_init__(self) -> None:
        self.model_set_sha256 = model_set_sha256(self.models)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def model_set_sha256(models: list[str] | tuple[str, ...]) -> str:
    """sha256 of the sorted ids, one UTF-8 line each, including a trailing newline."""
    body = "".join(f"{model}\n" for model in sorted(models))
    return hashlib.sha256(body.encode()).hexdigest()


def parse_fields(text: str) -> dict[str, tuple[int, int]]:
    fields: dict[str, tuple[int, int]] = {}
    for line in text.splitlines():
        m = FIELD_RE.match(line)
        if not m:
            continue
        key = f"{m.group(1)} {m.group(2)}"
        fields[key] = (int(m.group(3)), int(m.group(4)))
    return fields


def parse_step_summary(text: str) -> tuple[int, int, int, int]:
    m = STEP_RE.search(text)
    if not m:
        raise SystemExit("step summary line is missing")
    return tuple(int(m.group(i)) for i in range(1, 5))  # type: ignore[return-value]


def exact_models(ledger: Path) -> list[str]:
    models: list[str] = []
    for line in ledger.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("status") == "bit_exact":
            models.append(row["model"])
    return models


def load_receipt(path: Path) -> Receipt:
    sidecar = Path(str(path) + ".sha256")
    if not sidecar.is_file():
        raise SystemExit(f"receipt sidecar missing: {sidecar}")
    digest = sha256_file(path)
    side_text = sidecar.read_text()
    side_digest = side_text.split()[0]
    side_name = side_text.split(None, 1)[1].strip() if len(side_text.split()) > 1 else ""
    rel = path.relative_to(ROOT).as_posix()
    if side_digest != digest or side_name != rel:
        raise SystemExit(f"receipt sidecar does not hash {rel}")
    text = path.read_text()
    oracle = UTC = stated = None
    count = None
    lib = None
    dumps: dict[str, tuple[str, int]] = {}
    in_models = False
    models: list[str] = []
    for line in text.splitlines():
        if m := ORACLE_RE.match(line):
            oracle = m.group(1)
        elif m := UTC_RE.match(line):
            UTC = m.group(2)
        elif m := SET_RE.match(line):
            stated = m.group(1)
        elif m := COUNT_RE.match(line):
            count = tuple(int(m.group(i)) for i in range(1, 4))
        elif m := LIB_RE.match(line):
            lib = m.group(1)
        elif m := DUMP_RE.match(line):
            dumps[m.group(1)] = (m.group(2), int(m.group(3)))
        elif line.startswith("Exact models"):
            in_models = True
        elif line.startswith("## ") and in_models:
            in_models = False
        elif in_models and (m := MODEL_RE.match(line)):
            models.append(m.group(1))
    if oracle is None or UTC is None or count is None or lib is None:
        raise SystemExit(f"receipt {path} is missing oracle id, UTC, count, or lib hash")
    if len(models) != count[0]:
        raise SystemExit(f"receipt {path} lists {len(models)} models and states {count[0]}")
    computed = model_set_sha256(models)
    if stated is not None and stated != computed:
        raise SystemExit(f"receipt {path} exact-model-set sha256 does not hash its list")
    fields = parse_fields(text)
    if not fields:
        raise SystemExit(f"receipt {path} has no per-field counts")
    return Receipt(
        path=path,
        oracle_id=oracle,
        utc_end=UTC,
        exact=count[0],
        compared=count[1],
        refusals=count[2],
        models=tuple(sorted(models)),
        model_set_sha256=computed,
        stated_model_set_sha256=stated,
        fields=fields,
        lib_sha256=lib,
        dumps=dumps,
    )


def latest_receipt(oracle_id: str) -> Receipt:
    found = [load_receipt(path) for path in sorted(RECEIPTS.glob("*.md"))]
    matched = [rec for rec in found if rec.oracle_id == oracle_id]
    if not matched:
        raise SystemExit(f"no receipt for {oracle_id}")
    return max(matched, key=lambda rec: (rec.utc_end, rec.path.name))


def load_run(native: Path, step: Path, ledger: Path) -> Run:
    steps, exact, compared, refusals = parse_step_summary(step.read_text())
    models = tuple(sorted(exact_models(ledger)))
    if len(models) != exact:
        raise SystemExit(f"ledger has {len(models)} bit-exact models; step log says {exact}")
    fields = parse_fields(native.read_text())
    if not fields:
        raise SystemExit("native log has no per-field counts")
    return Run(steps, exact, compared, refusals, models, fields)


def check_blob(path: Path, digest: str, nbytes: int | None, label: str) -> None:
    got = sha256_file(path)
    size = path.stat().st_size
    if got != digest or (nbytes is not None and size != nbytes):
        raise SystemExit(f"{label} sha256 {got} size {size}; receipt {digest} size {nbytes}")
    print(f"{label} sha256 {got} ({size} bytes)")


def classify(pin: Receipt, run: Run) -> tuple[list[str], list[str]]:
    loss: list[str] = []
    gain: list[str] = []
    pin_models = set(pin.models)
    run_models = set(run.models)
    left = sorted(pin_models - run_models)
    added = sorted(run_models - pin_models)
    if left:
        loss.append("models left the set: " + ", ".join(left))
    if added:
        gain.append("models joined the set: " + ", ".join(added))
    if run.compared < pin.compared or run.refusals > pin.refusals or run.steps < 20:
        loss.append(
            f"step summary {run.exact} of {run.compared} over {run.steps}, refusals {run.refusals}"
        )
    elif run.compared != pin.compared or run.refusals != pin.refusals or run.steps != 20:
        gain.append(
            f"step summary {run.exact} of {run.compared} over {run.steps}, refusals {run.refusals}"
        )
    for name, (same, total) in pin.fields.items():
        if name not in run.fields:
            loss.append(f"field missing: {name}")
            continue
        got_same, got_total = run.fields[name]
        if got_total != total:
            loss.append(f"{name} denominator {got_total} != {total}")
        elif got_same < same:
            loss.append(f"{name} {got_same} < {same}")
        elif got_same > same:
            gain.append(f"{name} {got_same} > {same}")
    for name in sorted(set(run.fields) - set(pin.fields)):
        gain.append(f"new field {name} {run.fields[name][0]} of {run.fields[name][1]}")
    return loss, gain


def check(oracle_id: str, native: Path, step: Path, ledger: Path, lib: Path | None, native_dump: Path | None, step_dump: Path | None) -> int:
    pin = latest_receipt(oracle_id)
    if lib is not None:
        check_blob(lib, pin.lib_sha256, None, "libmujoco")
    if native_dump is not None:
        digest, nbytes = pin.dumps["x86-native.txt"]
        check_blob(native_dump, digest, nbytes, "x86-native.txt")
    if step_dump is not None:
        digest, nbytes = pin.dumps["x86-step.txt"]
        check_blob(step_dump, digest, nbytes, "x86-step.txt")
    run = load_run(native, step, ledger)
    print(f"pin {pin.path.relative_to(ROOT)} utc {pin.utc_end}")
    print(f"pin exact-model-set sha256 {pin.model_set_sha256}")
    print(f"observed exact-model-set sha256 {run.model_set_sha256}")
    loss, gain = classify(pin, run)
    if loss:
        print("loss:", file=sys.stderr)
        for item in loss:
            print(f"  {item}", file=sys.stderr)
        return 1
    if gain:
        print("gain without a receipt that hashes the new set:", file=sys.stderr)
        for item in gain:
            print(f"  {item}", file=sys.stderr)
        print(f"observed exact-model-set sha256 {run.model_set_sha256}", file=sys.stderr)
        print(f"latest receipt hashes {pin.model_set_sha256}", file=sys.stderr)
        return 1
    print(f"profile matches {pin.path.name}: {run.exact} of {run.compared} over {run.steps} steps")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-id", required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--step", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--lib", type=Path)
    parser.add_argument("--native-dump", type=Path)
    parser.add_argument("--step-dump", type=Path)
    args = parser.parse_args()
    return check(args.oracle_id, args.native, args.step, args.ledger, args.lib, args.native_dump, args.step_dump)


if __name__ == "__main__":
    sys.exit(main())
