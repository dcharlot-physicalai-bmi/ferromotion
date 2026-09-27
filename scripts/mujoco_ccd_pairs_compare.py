"""Compare ferromotion's CCD contacts with MuJoCo's own engine_collision_gjk.c, pair by pair, bit for bit.

    MENAGERIE_CCD_DUMP=dump.txt cargo run --release --example menagerie_contacts -- <menagerie> <flat oracle>
    <work dir>/mjsrc/pairs_on < dump.txt > compare.txt      # scripts/mujoco_ccd_harness_build.sh builds it
    python scripts/mujoco_ccd_pairs_compare.py compare.txt

Each pair in `compare.txt` carries our contacts (Rust's shortest round-trip decimal, which parses back to the
same double) and MuJoCo's (C99 hex floats). A pair is bit-identical when every number is the same double.
"""
import collections
import sys


def parse(path):
    out = []
    lines = open(path).read().split("\n")
    i = 0
    while i < len(lines):
        if lines[i].startswith("pair\t"):
            hdr = lines[i]
            ours = lines[i + 1].split("\t")
            mj = lines[i + 2].split("\t")
            o = [tuple(float(x) for x in c.split()) for c in ours[2:] if c.strip()]
            m = [tuple(float.fromhex(x) for x in c.split()) for c in mj[2:] if c.strip()]
            out.append((hdr, o, m))
            i += 3
        else:
            i += 1
    return out


rows = parse(sys.argv[1])
exact = close = diff = 0
bad = []
by_model = collections.Counter()
for hdr, o, m in rows:
    if o == m:
        exact += 1
        continue
    by_model[hdr.split("\t")[1]] += 1
    if len(o) == len(m) and all(max(abs(a - b) for a, b in zip(x, y)) < 1e-9 for x, y in zip(o, m)):
        close += 1
        bad.append((hdr, "close"))
        continue
    diff += 1
    bad.append((hdr, "diff"))
print(f"{sys.argv[1]}: {len(rows)} pairs; bit-identical {exact}, within 1e-9 {close}, different {diff}")
for h, w in bad[:8]:
    print("  ", w, h)
sys.exit(0 if not bad else 1)
