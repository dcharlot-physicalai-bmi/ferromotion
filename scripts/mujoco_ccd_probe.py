"""MuJoCo's native-CCD contacts for the pairs the primitive routines do not cover — pinned oracle for a
GJK/EPA/multicontact port. A convex mesh (a box-shaped OBJ and an octahedron) is written to a temp dir."""
import mujoco, numpy as np, os, tempfile
np.set_printoptions(precision=17)
d = tempfile.mkdtemp()
# box mesh 0.2 x 0.3 x 0.15 full extents (half 0.1, 0.15, 0.075), as 12 triangles
V = [(x, y, z) for x in (-0.1, 0.1) for y in (-0.15, 0.15) for z in (-0.075, 0.075)]
F = [(0,1,3),(0,3,2),(4,6,7),(4,7,5),(0,4,5),(0,5,1),(2,3,7),(2,7,6),(0,2,6),(0,6,4),(1,5,7),(1,7,3)]
with open(os.path.join(d, "boxmesh.obj"), "w") as f:
    for v in V: f.write("v %r %r %r\n" % v)
    for a,b,c in F: f.write("f %d %d %d\n" % (a+1,b+1,c+1))
# octahedron radius 0.12
O = [(0.12,0,0),(-0.12,0,0),(0,0.12,0),(0,-0.12,0),(0,0,0.12),(0,0,-0.12)]
OF = [(0,2,4),(2,1,4),(1,3,4),(3,0,4),(2,0,5),(1,2,5),(3,1,5),(0,3,5)]
with open(os.path.join(d, "octa.obj"), "w") as f:
    for v in O: f.write("v %r %r %r\n" % v)
    for a,b,c in OF: f.write("f %d %d %d\n" % (a+1,b+1,c+1))

def run(label, worldbody, option=""):
    xml = f'''<mujoco><compiler meshdir="{d}"/><option>{option}</option>
<asset><mesh name="boxmesh" file="boxmesh.obj"/><mesh name="octa" file="octa.obj"/></asset>
<worldbody>{worldbody}</worldbody></mujoco>'''
    m = mujoco.MjModel.from_xml_string(xml); dd = mujoco.MjData(m)
    mujoco.mj_forward(m, dd)
    print(f"=== {label}: ncon={dd.ncon} ccd_tol={m.opt.ccd_tolerance} ccd_iter={m.opt.ccd_iterations}")
    for i in range(dd.ncon):
        c = dd.contact[i]
        g1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom[0]); g2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom[1])
        print(f"  {g1}/{g2} dist={c.dist!r} pos={[repr(float(x)) for x in c.pos]} normal={[repr(float(x)) for x in c.frame[:3]]} dim={c.dim}")
def body(name, geom, pos, quat="1 0 0 0"):
    return '<body name="%s" pos="%s" quat="%s"><freejoint/>%s</body>' % (name, pos, quat, geom)
def geom(name, ty, extra):
    return '<geom name="%s" type="%s" %s/>' % (name, ty, extra)

run("boxmesh on box (face, multiccd)", body("a", geom("bx", "box", 'size="0.2 0.3 0.15"'), "0 0 1") + body("b", geom("m", "mesh", 'mesh="boxmesh"'), "0.05 0.1 1.215", "0.9961947 0 0 0.0871557"))
run("boxmesh on box, multiccd disabled", body("a", geom("bx", "box", 'size="0.2 0.3 0.15"'), "0 0 1") + body("b", geom("m", "mesh", 'mesh="boxmesh"'), "0.05 0.1 1.215", "0.9961947 0 0 0.0871557"), option='<flag multiccd="disable"/>')
run("octa on box (vertex)", body("a", geom("bx", "box", 'size="0.2 0.3 0.15"'), "0 0 1") + body("b", geom("m", "mesh", 'mesh="octa"'), "0.03 0.02 1.26"))
run("octa vs boxmesh (mesh-mesh)", body("a", geom("m1", "mesh", 'mesh="boxmesh"'), "0 0 1") + body("b", geom("m2", "mesh", 'mesh="octa"'), "0.02 0.03 1.15", "0.9238795 0.3826834 0 0"))
run("cylinder on box (face)", body("a", geom("bx", "box", 'size="0.2 0.3 0.15"'), "0 0 1") + body("b", geom("cy", "cylinder", 'size="0.05 0.1"'), "0.04 0.02 1.24"))
run("capsule on box", body("a", geom("bx", "box", 'size="0.2 0.3 0.15"'), "0 0 1") + body("b", geom("cp", "capsule", 'size="0.04 0.1"'), "0.04 0.02 1.18", "0.7071068 0 0.7071068 0"))
run("cylinder-cylinder crossed", body("a", geom("c1", "cylinder", 'size="0.05 0.2"'), "0 0 1") + body("b", geom("c2", "cylinder", 'size="0.05 0.2"'), "0.02 0.01 1.095", "0.7071068 0 0.7071068 0"))
run("ellipsoid on box", body("a", geom("bx", "box", 'size="0.2 0.3 0.15"'), "0 0 1") + body("b", geom("el", "ellipsoid", 'size="0.06 0.04 0.03"'), "0.03 0.02 1.175", "0.9238795 0 0.3826834 0"))
run("sphere vs boxmesh (sphere shortcut)", body("a", geom("m1", "mesh", 'mesh="boxmesh"'), "0 0 1") + body("b", geom("s", "sphere", 'size="0.05"'), "0.03 0.04 1.12"))
run("boxmesh on plane (plane-convex multi)", '<geom name="floor" type="plane" size="1 1 0.1"/>' + body("b", geom("m", "mesh", 'mesh="boxmesh"'), "0 0 0.07", "0.9961947 0.0871557 0 0"))
