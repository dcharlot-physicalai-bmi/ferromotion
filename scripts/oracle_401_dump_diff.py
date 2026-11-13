#!/usr/bin/env python3
"""Compare two oracle dumps and print model and field counts.

The native and step files are the schema dumper's text. A model counts once
for a field when any of that field's lines differ. Sparse Jacobian models are
read from a `nv jacobian sparse rel` table produced by loading each model.
"""
from __future__ import annotations

import argparse
import collections
from pathlib import Path


def models(path: Path) -> list[tuple[str, dict[str, list[str]]]]:
    out: list[tuple[str, dict[str, list[str]]]] = []
    rel = None
    fields: dict[str, list[str]] = collections.defaultdict(list)
    for line in path.read_text().splitlines():
        if line.startswith("model\t"):
            if rel is not None:
                out.append((rel, fields))
            rel = line.split("\t", 1)[1]
            fields = collections.defaultdict(list)
        elif rel is not None and "\t" in line:
            kind, rest = line.split("\t", 1)
            name = rest.split("\t", 1)[0] if kind in ("m", "d") else kind
            fields[f"{kind}:{name}" if kind in ("m", "d") else kind].append(line)
    if rel is not None:
        out.append((rel, fields))
    return out


def changed(a: Path, b: Path) -> tuple[list[str], dict[str, list[str]]]:
    left, right = models(a), models(b)
    if [rel for rel, _ in left] != [rel for rel, _ in right]:
        raise SystemExit("model order differs")
    by_field: dict[str, list[str]] = collections.defaultdict(list)
    names: list[str] = []
    for (rel, fa), (_, fb) in zip(left, right):
        local = []
        for key in set(fa) | set(fb):
            if fa.get(key) != fb.get(key):
                local.append(key)
                by_field[key].append(rel)
        if local:
            names.append(rel)
    return names, by_field


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-a", type=Path, required=True)
    parser.add_argument("--native-b", type=Path, required=True)
    parser.add_argument("--step-a", type=Path, required=True)
    parser.add_argument("--step-b", type=Path, required=True)
    parser.add_argument("--sparse", type=Path, required=True)
    args = parser.parse_args()
    sparse = set()
    for line in args.sparse.read_text().splitlines():
        nv, _jac, flag, rel = line.split("\t", 3)
        if flag == "1":
            sparse.add(rel)
            print(f"sparse\t{nv}\t{rel}")
    for label, a, b in (
        ("native", args.native_a, args.native_b),
        ("step", args.step_a, args.step_b),
    ):
        names, by_field = changed(a, b)
        print(f"{label} models {len(names)}")
        for rel in names:
            mark = "sparse" if rel in sparse else "dense"
            print(f"{label} model {mark} {rel}")
        for key, rels in sorted(by_field.items(), key=lambda item: (-len(item[1]), item[0])):
            print(f"{label} field {len(rels)} {key}")


if __name__ == "__main__":
    main()
