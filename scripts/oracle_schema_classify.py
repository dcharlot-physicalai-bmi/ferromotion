#!/usr/bin/env python3
"""Classify the 3.13.0-loadable Menagerie models under a later compiler.

A model is `compiles` when the later compiler loads it and nbody, njnt, ngeom,
nq, nv, nu, neq, ntendon, and nsensor equal the 3.13.0 record. `rejected` keeps
the later compiler's error bytes. `topology` names every count that moved.
Models the 3.13.0 compiler itself rejected stay in the table as `rejected_by_3.13`.
"""
import json
import sys

COUNTS = ("nbody", "njnt", "ngeom", "nq", "nv", "nu", "neq", "ntendon", "nsensor")


def load(path):
    rows = {}
    order = []
    meta = {}
    lines = open(path, "rb").read().split(b"\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        i += 1
        if not line:
            continue
        parts = line.decode("utf-8", "surrogateescape").split("\t")
        if parts[0] in ("schema", "dump_schema", "version", "header", "neqdata"):
            meta[parts[0]] = parts[1]
            continue
        if parts[0] == "summary":
            meta["summary"] = parts[1:]
            continue
        if parts[0] == "ok":
            rel = parts[1]
            counts = {}
            # ok rel nbody N njnt N ...
            it = iter(parts[2:])
            for key, val in zip(it, it):
                counts[key] = int(val)
            rows[rel] = {"status": "ok", "counts": counts}
            order.append(rel)
            continue
        if parts[0] == "fail":
            rel = parts[1]
            elen = int(parts[2])
            # The error is the next physical line's raw bytes, length-prefixed.
            # getline kept the payload as the following record, which may itself
            # contain no extra newline because we wrote elen bytes plus one newline.
            raw = lines[i] if i < len(lines) else b""
            i += 1
            if len(raw) != elen:
                # The error itself contained newlines and split. Reassemble.
                buf = raw
                while len(buf) < elen and i < len(lines):
                    buf += b"\n" + lines[i]
                    i += 1
                raw = buf[:elen]
            rows[rel] = {"status": "fail", "error": raw}
            order.append(rel)
            continue
        raise SystemExit(f"bad classify line {parts[0]!r} in {path}")
    return meta, order, rows


def main():
    if len(sys.argv) != 4:
        raise SystemExit("usage: oracle_schema_classify.py <3.13 classify> <later classify> <out.json>")
    base_meta, order, base = load(sys.argv[1])
    later_meta, _, later = load(sys.argv[2])
    table = []
    tally = {"compiles": 0, "rejected": 0, "topology": 0, "rejected_by_3.13": 0, "missing": 0}
    for rel in order:
        b = base[rel]
        if b["status"] != "ok":
            tally["rejected_by_3.13"] += 1
            table.append({
                "model": rel,
                "class": "rejected_by_3.13",
                "error_3.13": b["error"].decode("utf-8", "surrogateescape"),
            })
            continue
        L = later.get(rel)
        if L is None:
            tally["missing"] += 1
            table.append({"model": rel, "class": "missing"})
            continue
        if L["status"] != "ok":
            tally["rejected"] += 1
            table.append({
                "model": rel,
                "class": "rejected",
                "error": L["error"].decode("utf-8", "surrogateescape"),
            })
            continue
        changed = {k: [b["counts"][k], L["counts"][k]] for k in COUNTS if b["counts"][k] != L["counts"][k]}
        if changed:
            tally["topology"] += 1
            table.append({"model": rel, "class": "topology", "counts": changed})
        else:
            tally["compiles"] += 1
            table.append({"model": rel, "class": "compiles"})
    doc = {
        "base": base_meta,
        "later": later_meta,
        "counts_compared": list(COUNTS),
        "tally": tally,
        "loadable_on_3.13": tally["compiles"] + tally["rejected"] + tally["topology"] + tally["missing"],
        "models": table,
    }
    with open(sys.argv[3], "w") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(json.dumps({"later": later_meta.get("version"), **tally}, sort_keys=True))


if __name__ == "__main__":
    main()
