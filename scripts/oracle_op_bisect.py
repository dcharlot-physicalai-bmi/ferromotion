#!/usr/bin/env python3
"""Compare instrumented kinematics ops to a named MuJoCo shared library.

Covered ops are those with an exported C function in that library. Any other
recorded op is uncovered. Uncovered is not a pass. A model whose covered ops
all match is not a 212/212 trajectory pass.

Reads JSON lines from oracle_op_trace. Writes a summary JSON.
"""

from __future__ import annotations

import ctypes
import json
import struct
import sys
from pathlib import Path


COVERED = {
    "mul_quat": "mju_mulQuat",
    "normalize4": "mju_normalize4",
    "rot_vec_quat": "mju_rotVecQuat",
    "axis_angle2quat": "mju_axisAngle2Quat",
}


def load_so(path: str) -> ctypes.CDLL:
    lib = ctypes.CDLL(path)
    c_double = ctypes.c_double
    p = ctypes.POINTER(c_double)
    lib.mju_mulQuat.argtypes = [p, p, p]
    lib.mju_normalize4.argtypes = [p]
    lib.mju_rotVecQuat.argtypes = [p, p, p]
    lib.mju_axisAngle2Quat.argtypes = [p, p, ctypes.c_double]
    return lib


def bits_to_doubles(bits: list[int]) -> ctypes.Array:
    raw = b"".join(int(b).to_bytes(8, "little") for b in bits)
    buf = (ctypes.c_double * len(bits)).from_buffer_copy(raw)
    return buf


def doubles_to_bits(buf) -> list[int]:
    return [int.from_bytes(bytes(buf)[i : i + 8], "little") for i in range(0, len(bytes(buf)), 8)]


def oracle_out(lib: ctypes.CDLL, op: str, inputs: list[int]) -> list[int] | None:
    if op not in COVERED:
        return None
    if op == "mul_quat":
        res = (ctypes.c_double * 4)()
        a = bits_to_doubles(inputs[:4])
        b = bits_to_doubles(inputs[4:8])
        lib.mju_mulQuat(res, a, b)
        return doubles_to_bits(res)
    if op == "normalize4":
        q = bits_to_doubles(inputs)
        lib.mju_normalize4(q)
        return doubles_to_bits(q)
    if op == "rot_vec_quat":
        res = (ctypes.c_double * 3)()
        v = bits_to_doubles(inputs[:3])
        q = bits_to_doubles(inputs[3:7])
        lib.mju_rotVecQuat(res, v, q)
        return doubles_to_bits(res)
    if op == "axis_angle2quat":
        res = (ctypes.c_double * 4)()
        axis = bits_to_doubles(inputs[:3])
        angle = struct.unpack("<d", int(inputs[3]).to_bytes(8, "little"))[0]
        lib.mju_axisAngle2Quat(res, axis, ctypes.c_double(angle))
        return doubles_to_bits(res)
    return None


def main() -> int:
    if len(sys.argv) != 4:
        print("usage: oracle_op_bisect.py <libmujoco.so> <trace.jsonl> <summary.json>", file=sys.stderr)
        return 2
    lib = load_so(sys.argv[1])
    rows = [json.loads(line) for line in Path(sys.argv[2]).read_text().splitlines() if line.strip()]
    first_counts: dict[str, int] = {}
    loaded = 0
    refused = 0
    covered_mismatches = 0
    models = []
    for row in rows:
        if not row.get("loaded"):
            refused += 1
            models.append({"model": row["model"], "loaded": False})
            continue
        loaded += 1
        uncovered_before = 0
        first = None
        covered_seen = 0
        covered_match = 0
        for sample in row["ops"]:
            op = sample["op"]
            if op not in COVERED:
                if first is None:
                    uncovered_before += 1
                continue
            covered_seen += 1
            got = oracle_out(lib, op, sample["in"])
            if got == sample["out"]:
                covered_match += 1
                continue
            if first is None:
                first = op
                covered_mismatches += 1
        label = first or "no_divergence_in_covered_ops"
        first_counts[label] = first_counts.get(label, 0) + 1
        models.append({
            "model": row["model"],
            "loaded": True,
            "first_divergent_covered_op": first,
            "uncovered_ops_before_that": uncovered_before,
            "covered_ops": covered_seen,
            "covered_ops_matching": covered_match if first is None else None,
            "pass_212": False,
        })
    summary = {
        "oracle_so": sys.argv[1],
        "covered_ops": sorted(COVERED),
        "uncovered_ops_are_not_passes": ["mul_mat_vec3", "sincos", "solver", "mass_matrix", "constraints", "tendons", "20_step_trajectory"],
        "loaded_models": loaded,
        "refused_models": refused,
        "models_with_a_covered_mismatch": covered_mismatches,
        "first_divergent_covered_op_counts": first_counts,
        "this_is_not_a_trajectory_pass": True,
        "models": models,
    }
    Path(sys.argv[3]).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in summary if k != "models"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
