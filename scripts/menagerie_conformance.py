#!/usr/bin/env python3
"""Re-execute Menagerie conformance. A previously written ledger is not a pass.

The process exits 0 only when, at the pinned corpus and MuJoCo 3.13.0:

* every ``*/*.xml`` file is classified exactly once;
* the loadable count is the pinned count (212 at this corpus);
* every loadable model is bit-exact with ``mj_step`` for all 20 steps
  (``qpos``, ``qvel``, ``act``, ``qacc_warmstart``);
* ``efc_J`` and ``qacc`` are bit-exact on every native-forward state
  (212 models × 3 states = 636).

Load failures are written into the ledger. They are not dropped. A missing
file, a count change, a rust refusal, or a bit mismatch is a non-zero exit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIN_PATH = ROOT / "conformance" / "menagerie_pin.json"
STEP_ARRAYS = ("qpos", "qvel", "act", "qacc_warmstart")
NATIVE_REQUIRED = ("efc_J", "qacc")


def load_pin() -> dict:
    return json.loads(PIN_PATH.read_text())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head(path: Path) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def enumerate_xml(menagerie: Path) -> list[str]:
    return sorted(p.relative_to(menagerie).as_posix() for p in menagerie.glob("*/*.xml"))


def classify_loads(menagerie: Path, xmls: list[str]) -> list[dict]:
    import mujoco

    rows = []
    for rel in xmls:
        try:
            mujoco.MjModel.from_xml_path(str(menagerie / rel))
            rows.append({"model": rel, "mujoco_load": "ok"})
        except Exception as exc:  # noqa: BLE001 — the ledger must keep the loader's own error
            rows.append({
                "model": rel,
                "mujoco_load": "error",
                "mujoco_error_type": type(exc).__name__,
                "mujoco_error": str(exc).splitlines()[0][:500],
            })
    return rows


def run(cmd: list[str], cwd: Path, env: dict | None = None) -> subprocess.CompletedProcess[str]:
    merged = os.environ.copy()
    if env:
        merged.update(env)
    return subprocess.run(cmd, cwd=cwd, env=merged, text=True, capture_output=True)


def parse_native_summary(text: str) -> dict[str, tuple[int, int]]:
    """Map array name to (same, n) from menagerie_native's tally lines."""
    out: dict[str, tuple[int, int]] = {}
    for line in text.splitlines():
        parts = line.split()
        # "mjData  efc_J  359 of 636" or "mjModel body_mass 212 of 212"
        if len(parts) >= 5 and parts[3] == "of" and parts[2].isdigit() and parts[4].isdigit():
            out[parts[1]] = (int(parts[2]), int(parts[4]))
    return out


def validate(pin: dict, xmls: list[str], loads: list[dict], step_rows: list[dict], native: dict[str, tuple[int, int]] | None) -> list[str]:
    errors: list[str] = []
    if len(xmls) != pin["expected_xml"]:
        errors.append(f"enumerated {len(xmls)} xml files, pin expects {pin['expected_xml']}")
    by_model = {row["model"]: row for row in loads}
    if set(by_model) != set(xmls) or len(loads) != len(xmls):
        errors.append("load classification does not cover every xml exactly once")
    loadable = [r["model"] for r in loads if r["mujoco_load"] == "ok"]
    if len(loadable) != pin["expected_mujoco_loadable"]:
        errors.append(f"{len(loadable)} models load, pin expects {pin['expected_mujoco_loadable']}")
    stepped = {row["model"]: row for row in step_rows}
    for rel in loadable:
        row = stepped.get(rel)
        if row is None:
            errors.append(f"no step result for loadable model {rel}")
            continue
        if row.get("status") != "bit_exact":
            errors.append(f"{rel} step status {row.get('status')} {row.get('detail', '')}")
        if row.get("steps") != pin["step_count"]:
            errors.append(f"{rel} compared {row.get('steps')} steps, pin requires {pin['step_count']}")
    extras = set(stepped) - set(loadable)
    if extras:
        errors.append(f"step ledger names models MuJoCo did not load: {sorted(extras)[:5]}")
    if native is not None:
        states = pin["expected_native_state_rows"]
        for name in NATIVE_REQUIRED:
            got = native.get(name)
            if got is None:
                errors.append(f"native tally has no {name} line")
            elif got != (states, states):
                errors.append(f"{name} {got[0]} of {got[1]}, required {states} of {states}")
    return errors


def self_test() -> int:
    pin = load_pin()
    xmls = [f"m{i}/a.xml" for i in range(pin["expected_xml"])]
    loads = [{"model": rel, "mujoco_load": "ok"} for rel in xmls[: pin["expected_mujoco_loadable"]]]
    loads += [{"model": rel, "mujoco_load": "error", "mujoco_error": "fixture"} for rel in xmls[pin["expected_mujoco_loadable"] :]]
    steps = [{"model": rel, "status": "bit_exact", "steps": pin["step_count"], "detail": ""} for rel in xmls[: pin["expected_mujoco_loadable"]]]
    native = {name: (pin["expected_native_state_rows"], pin["expected_native_state_rows"]) for name in NATIVE_REQUIRED}
    ok = validate(pin, xmls, loads, steps, native)
    if ok:
        print("self-test expected a clean fixture to pass:", ok)
        return 1
    steps[0]["status"] = "mismatch"
    bad = validate(pin, xmls, loads, steps, native)
    if not any("mismatch" in e for e in bad):
        print("self-test did not fail a mismatch:", bad)
        return 1
    missing = validate(pin, xmls, loads, steps[1:], native)
    if not any("no step result" in e for e in missing):
        print("self-test did not fail a missing model:", missing)
        return 1
    print("self-test ok: a complete bit-exact ledger passes, a mismatch or a missing model fails")
    return 0


def provenance(pin: dict, menagerie: Path) -> dict:
    import mujoco

    rustc = subprocess.check_output(["rustc", "+stable", "--version"], text=True).strip()
    os_release = {}
    rel = Path("/etc/os-release")
    if rel.exists():
        for line in rel.read_text().splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                os_release[k] = v.strip().strip('"')
    return {
        "evidence_kind": "independently_executed",
        "prior_claim": pin["prior_claim"],
        "menagerie_repo": pin["menagerie_repo"],
        "menagerie_sha": git_head(menagerie),
        "mujoco_version": mujoco.__version__,
        "mujoco_file": mujoco.__file__,
        "precision": pin["precision"],
        "equality": pin["equality"],
        "step_count": pin["step_count"],
        "step_seed": pin["step_seed"],
        "step_initialization": pin["step_initialization"],
        "solver": pin["solver"],
        "step_arrays": list(STEP_ARRAYS),
        "native_states": pin["native_states"],
        "native_seed": pin["native_seed"],
        "native_state_0": pin["native_state_0"],
        "native_arrays_required": list(NATIVE_REQUIRED),
        "ferromotion_sha": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "rustc": rustc,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "os_release": os_release,
        "platform_contract": (
            "Oracle: Linux x86_64 MuJoCo 3.13.0 wheel, glibc libm, that wheel's "
            "floating-point contraction. Unit-test goldens are MuJoCo 3.13.0 arm64 "
            "dylib bits (Apple clang contraction and __sincos_stret). Non-Apple "
            "sincos in ferromotion is host libm, not __sincos_stret. A match to "
            "host libm is not a match to the arm64 reference."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("menagerie", nargs="?", type=Path)
    parser.add_argument("--out", type=Path, default=Path("conformance/menagerie_ledger.json"))
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--skip-native", action="store_true", help="run only the 20-step comparison; the exit status stays non-zero")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if args.menagerie is None:
        parser.error("menagerie checkout is required")

    pin = load_pin()
    menagerie = args.menagerie.resolve()
    head = git_head(menagerie)
    if head != pin["menagerie_sha"]:
        print(f"menagerie HEAD {head} is not the pin {pin['menagerie_sha']}", file=sys.stderr)
        return 2
    import mujoco

    if mujoco.__version__ != pin["mujoco_version"]:
        print(f"mujoco {mujoco.__version__} is not {pin['mujoco_version']}", file=sys.stderr)
        return 2

    xmls = enumerate_xml(menagerie)
    loads = classify_loads(menagerie, xmls)
    work = Path(os.environ.get("MENAGERIE_WORK", "/tmp/menagerie-conformance")).resolve()
    work.mkdir(parents=True, exist_ok=True)
    step_oracle = work / "step_oracle.txt"
    native_oracle = work / "native_oracle.txt"
    step_ledger = work / "step_ledger.jsonl"
    py = sys.executable
    step_run = run([py, str(ROOT / "scripts" / "mujoco_native_step_oracle.py"), str(menagerie), str(step_oracle), str(pin["step_count"])], ROOT)
    if step_run.returncode != 0:
        print(step_run.stderr, file=sys.stderr)
        return step_run.returncode
    print(step_run.stdout.strip())
    cargo = shutil.which("cargo") or "cargo"
    step_cmp = run(
        [cargo, "+stable", "run", "--release", "-p", "ferromotion-core", "--example", "menagerie_native_step", "--", str(menagerie), str(step_oracle)],
        ROOT,
        {"MENAGERIE_LEDGER": str(step_ledger)},
    )
    print(step_cmp.stdout)
    if step_cmp.returncode != 0:
        print(step_cmp.stderr, file=sys.stderr)
        return step_cmp.returncode
    step_rows = [json.loads(line) for line in step_ledger.read_text().splitlines() if line.strip()]

    native = None
    native_stdout = ""
    if not args.skip_native:
        native_run = run([py, str(ROOT / "scripts" / "mujoco_native_oracle.py"), str(menagerie), str(native_oracle), str(pin["native_states"])], ROOT)
        if native_run.returncode != 0:
            print(native_run.stderr, file=sys.stderr)
            return native_run.returncode
        print(native_run.stdout.strip())
        native_cmp = run(
            [cargo, "+stable", "run", "--release", "-p", "ferromotion-core", "--example", "menagerie_native", "--", str(menagerie), str(native_oracle)],
            ROOT,
        )
        native_stdout = native_cmp.stdout
        print(native_stdout)
        if native_cmp.returncode != 0:
            print(native_cmp.stderr, file=sys.stderr)
            return native_cmp.returncode
        native = parse_native_summary(native_stdout)

    gpu = shutil.which("nvidia-smi") is not None
    def importable(mod: str) -> bool:
        try:
            __import__(mod)
            return True
        except Exception:
            return False

    warp_ok, newton_ok = importable("warp"), importable("newton")
    errors = validate(pin, xmls, loads, step_rows, native if not args.skip_native else None)
    if args.skip_native:
        errors.append("native efc_J/qacc phase was not executed")
    exact = sum(1 for r in step_rows if r.get("status") == "bit_exact")
    tally = native or {}

    def cell(feature: str, key: str) -> dict:
        same, n = tally.get(key, (None, None))
        return {
            "feature": feature,
            "solver": "mujoco_3.13.0_cpu",
            "backend": "cpu",
            "status": "executed" if key in tally else "not_run",
            "matched": same == n and n not in (None, 0),
            "same": same,
            "n": n,
        }

    blocked_reason = "no NVIDIA device and the package is not installed; not a pass"
    feature_matrix = [
        {
            "feature": "trajectory_qpos_qvel_act_warmstart",
            "solver": "mujoco_3.13.0_cpu",
            "backend": "cpu",
            "status": "executed",
            "matched": exact == pin["expected_mujoco_loadable"] and bool(step_rows),
            "same": exact,
            "n": pin["expected_mujoco_loadable"],
        },
        cell("mass_matrix", "M"),
        cell("constraint_jacobian", "efc_J"),
        cell("constraint_force", "qfrc_constraint"),
        cell("actuator_force", "actuator_force"),
        cell("solver_acceleration", "qacc"),
        cell("solver_iterations", "solver_niter"),
        {
            "feature": "matched_fidelity_rollout",
            "solver": "mujoco_warp",
            "backend": "gpu",
            "revision": "4d1c64bc077d725247e9bf5e2ac3f357749ff804",
            "status": "blocked" if not (gpu and warp_ok) else "not_measured",
            "matched": False,
            "reason": blocked_reason if not (gpu and warp_ok) else "import succeeded; no rollout was run",
        },
        {
            "feature": "gradient",
            "solver": "mujoco_warp",
            "backend": "gpu",
            "revision": "4d1c64bc077d725247e9bf5e2ac3f357749ff804",
            "status": "blocked",
            "matched": False,
            "reason": "no gradient was evaluated; Warp's README at this revision says differentiability is not yet available",
        },
        {
            "feature": "matched_fidelity_rollout",
            "solver": "newton",
            "backend": "gpu",
            "revision": "1a497c00ed255db343e39eec4d614864c9aa9b95",
            "status": "blocked" if not (gpu and newton_ok) else "not_measured",
            "matched": False,
            "reason": blocked_reason if not (gpu and newton_ok) else "import succeeded; no rollout was run",
        },
        {
            "feature": "gradient",
            "solver": "newton",
            "backend": "gpu",
            "revision": "1a497c00ed255db343e39eec4d614864c9aa9b95",
            "status": "blocked",
            "matched": False,
            "reason": "no gradient was evaluated on this host",
        },
    ]
    record = {
        **provenance(pin, menagerie),
        "step_oracle_sha256": sha256_file(step_oracle),
        "native_oracle_sha256": sha256_file(native_oracle) if native_oracle.exists() else None,
        "xml_count": len(xmls),
        "mujoco_loadable": sum(1 for r in loads if r["mujoco_load"] == "ok"),
        "mujoco_load_errors": sum(1 for r in loads if r["mujoco_load"] != "ok"),
        "models": [],
        "native_tally": {k: {"same": v[0], "n": v[1]} for k, v in (native or {}).items()},
        "native_stdout_sha256": hashlib.sha256(native_stdout.encode()).hexdigest() if native_stdout else None,
        "errors": errors,
        "passed": not errors,
        "step_bit_exact": exact,
        "step_mismatch": sum(1 for r in step_rows if r.get("status") == "mismatch"),
        "feature_matrix": feature_matrix,
        "warp_newton": {
            "gpu_device": gpu,
            "warp_importable": warp_ok,
            "newton_importable": newton_ok,
            "executed": False,
        },
        "smolvla": {
            "checkpoint": "5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6",
            "weights_downloaded": False,
            "integrated": False,
            "physical_actuation": False,
        },
    }
    load_by = {r["model"]: r for r in loads}
    step_by = {r["model"]: r for r in step_rows}
    for rel in xmls:
        row = {"model": rel, **load_by[rel]}
        if rel in step_by:
            row.update({k: v for k, v in step_by[rel].items() if k != "model"})
        record["models"].append(row)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2) + "\n")
    print(f"wrote {args.out} passed={record['passed']} errors={len(errors)}")
    for err in errors[:30]:
        print(" ", err)
    return 0 if record["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
