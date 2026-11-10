#!/usr/bin/env python3
"""Field diff of two versioned dumps. Equality is the raw text of each field line.

A field present on only one side is reported as added or removed. A 3.13 dump has
no eq_data, tendon_type, xiquat, geom_xquat, site_xquat, or contact_surfacevel lines.
"""
import collections
import hashlib
import sys


def index(path):
    """model -> list of (kind, name, state, text) in file order."""
    models = collections.OrderedDict()
    cur = None
    state = -1
    for line in open(path):
        line = line.rstrip("\n")
        if not line:
            continue
        parts = line.split("\t")
        key = parts[0]
        if key == "model":
            cur = parts[1]
            models[cur] = []
            state = -1
            continue
        if cur is None:
            continue
        if key == "load_error":
            models[cur].append(("load_error", "", state, parts[1] if len(parts) > 1 else ""))
            continue
        if key == "topology":
            models[cur].append(("topology", "", state, line))
            continue
        if key == "s":
            state = int(parts[1])
            continue
        if key in ("qpos", "qvel", "ctrl", "act", "warm"):
            name = "qacc_warmstart" if key == "warm" else key
            kind = "step" if state >= 0 else "input"
            models[cur].append((kind, f"{name}@{state}", state, parts[1] if len(parts) > 1 else ""))
            continue
        if key in ("m", "d"):
            name = parts[1]
            text = parts[2] if len(parts) > 2 else ""
            slot = "model" if key == "m" else f"state{state if state >= 0 else models[cur] and _state_of(models[cur])}"
            models[cur].append((key, name, slot, text))
            continue
    return models


def _state_of(rows):
    n = sum(1 for k, name, slot, _ in rows if k == "qpos" or (k == "input" and name.startswith("qpos")))
    # native dumps: each state begins with a qpos input line. Count qpos lines already stored.
    return sum(1 for k, name, _, _ in rows if k == "input" and name.startswith("qpos@")) - 1


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: oracle_schema_diff.py <before.dump> <after.dump>")
    a = index(sys.argv[1])
    b = index(sys.argv[2])
    # Re-index native state numbers from qpos input lines, which _state_of only approximates
    # while scanning. Rebuild state tags from a second pass that counts qpos.
    def retag(models, step):
        out = collections.OrderedDict()
        for rel, rows in models.items():
            tagged = []
            st = -1
            seen_q = 0
            for kind, name, slot, text in rows:
                if kind == "input" and name.startswith("qpos@"):
                    seen_q += 1
                    st = seen_q - 1
                    tagged.append((kind, f"qpos@{st}", st, text))
                elif kind == "input":
                    base = name.split("@")[0]
                    tagged.append((kind, f"{base}@{st}", st, text))
                elif kind == "d":
                    tagged.append((kind, name, st, text))
                else:
                    tagged.append((kind, name, slot, text))
            out[rel] = tagged
        return out

    # Detect step dumps by an "s\t" having produced step kind.
    def is_step(models):
        return any(k == "step" for rows in models.values() for k, *_ in rows)

    step = is_step(a) or is_step(b)
    if not step:
        a = retag(a, False)
        b = retag(b, False)
    changed = collections.OrderedDict()
    added = collections.Counter()
    removed = collections.Counter()
    load_changed = []
    for rel in list(dict.fromkeys([*a.keys(), *b.keys()])):
        ra = a.get(rel, [])
        rb = b.get(rel, [])
        if any(k == "load_error" for k, *_ in ra) or any(k == "load_error" for k, *_ in rb):
            ea = next((t for k, _, _, t in ra if k == "load_error"), None)
            eb = next((t for k, _, _, t in rb if k == "load_error"), None)
            if ea != eb:
                load_changed.append(rel)
            continue
        if any(k == "topology" for k, *_ in ra + rb):
            load_changed.append(rel)
            continue
        ma = {(k, n, s): t for k, n, s, t in ra}
        mb = {(k, n, s): t for k, n, s, t in rb}
        keys = list(dict.fromkeys([*ma.keys(), *mb.keys()]))
        for key in keys:
            if key not in ma:
                added[f"{key[0]} {key[1]}"] += 1
                changed.setdefault(f"added {key[0]} {key[1]}", []).append(rel)
                continue
            if key not in mb:
                removed[f"{key[0]} {key[1]}"] += 1
                changed.setdefault(f"removed {key[0]} {key[1]}", []).append(rel)
                continue
            if ma[key] != mb[key]:
                label = f"{key[0]} {key[1]}"
                changed.setdefault(label, []).append(rel)
    print(f"models_before {len(a)} models_after {len(b)} load_or_topology {len(load_changed)}")
    for label, rels in changed.items():
        uniq = list(dict.fromkeys(rels))
        print(f"field\t{label}\tmodels\t{len(uniq)}")
    if load_changed:
        print("load_or_topology\t" + " ".join(load_changed[:30]))


if __name__ == "__main__":
    main()
