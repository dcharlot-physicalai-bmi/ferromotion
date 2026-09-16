"""Flatten `menagerie_contacts_oracle.py`'s JSON files into one tab-separated text file per model, which
`examples/menagerie_contacts.rs` reads (ferromotion-core has no JSON dependency)."""
import json, os, sys, glob
src, dst = sys.argv[1], sys.argv[2]
os.makedirs(dst, exist_ok=True)
def f(x): return repr(float(x))
n = 0
for path in sorted(glob.glob(os.path.join(src, "*.json"))):
    if path.endswith("_summary.json"): continue
    d = json.load(open(path))
    lines = ["\t".join(["model", d["file"], str(d["cone"]), str(int(d["filterparent"])), str(int(d["nativeccd"])), str(int(d["multiccd"])),
                        f(d["ccd_tolerance"]), str(d["ccd_iterations"]), str(d["npair"]), str(d["nexclude"]), str(int(d["override"])), f(d["o_margin"])])]
    for i, g in enumerate(d["geoms"]):
        lines.append("\t".join(["geom", str(i), g["name"], g["type"], g["body"], *map(f, g["size"]), str(g["contype"]), str(g["conaffinity"]),
                                str(g["condim"]), f(g["margin"]), f(g["gap"]), str(g["priority"]), f(g["solmix"]), *map(f, g["friction"]),
                                *map(f, g["solref"]), *map(f, g["solimp"]), f(g["rbound"]), g["mesh"] or "-", f(g.get("adhesion", 0.0)), *map(f, g.get("aabb", [0, 0, 0, 0, 0, 0]))]))
    for name, mm in d.get("meshes", {}).items():
        lines.append("\t".join(["mesh", name, *map(f, mm["pos"]), *map(f, mm["quat"]), str(mm["nvert"])]))
    for k, s in enumerate(d["samples"]):
        lines.append("sample\t%d" % k)
        for i, (p, m) in enumerate(zip(s["geom_xpos"], s["geom_xmat"])):
            lines.append("\t".join(["gpose", str(i), *map(f, p), *map(f, m)]))
        for c in s["contacts"]:
            lines.append("\t".join(["contact", str(c["g1"]), str(c["g2"]), f(c["dist"]), *map(f, c["pos"]), *map(f, c["frame"]), str(c["dim"]),
                                    f(c["includemargin"]), *map(f, c["friction"]), *map(f, c["solref"]), *map(f, c["solimp"]), str(c["exclude"])]))
    with open(os.path.join(dst, d["file"].replace("/", "__") + ".txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    n += 1
print("flattened", n, "models into", dst)
