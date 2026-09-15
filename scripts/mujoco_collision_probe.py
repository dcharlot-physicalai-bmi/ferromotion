"""MuJoCo's contacts for each primitive pair at fixed poses — the pinned oracle for `mujoco_collision.rs`.

    python scripts/mujoco_collision_probe.py      (needs the `mujoco` wheel; numbers in the tests are 3.13.0's)
"""
import mujoco, numpy as np

def run(label, xml):
    m = mujoco.MjModel.from_xml_string(xml); d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    print(f"=== {label}: ncon={d.ncon}")
    for i in range(d.ncon):
        c = d.contact[i]
        g1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom[0]); g2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom[1])
        print(f"  {g1}/{g2} dist={c.dist!r} pos={[repr(float(x)) for x in c.pos]} normal={[repr(float(x)) for x in c.frame[:3]]} t1={[repr(float(x)) for x in c.frame[3:6]]} dim={c.dim} incl={c.includemargin!r} fri={c.friction.tolist()} solref={c.solref.tolist()} solimp={c.solimp.tolist()}")

def body(name, geom, pos, quat="1 0 0 0"):
    return '<body name="%s" pos="%s" quat="%s"><freejoint/>%s</body>' % (name, pos, quat, geom)
def geom(name, ty, size, extra=""):
    return '<geom name="%s" type="%s" size="%s" %s/>' % (name, ty, size, extra)
def world(*inner):
    return '<mujoco><worldbody>' + "".join(inner) + '</worldbody></mujoco>'
floor = '<geom name="floor" type="plane" size="1 1 0.1"/>'

run("plane-sphere", world(floor, body("b", geom("s", "sphere", "0.1"), "0.2 0.3 0.095")))
run("plane-capsule tilted", world(floor, body("b", geom("c", "capsule", "0.05 0.2"), "0 0 0.12", "0.9659258 0 0.2588190 0")))
run("plane-cylinder tilted", world(floor, body("b", geom("c", "cylinder", "0.1 0.15"), "0.1 0 0.14", "0.9848078 0.1736482 0 0")))
run("plane-box tilted", world(floor, body("b", geom("bx", "box", "0.1 0.15 0.05"), "0 0 0.06", "0.9961947 0.0871557 0 0")))
run("sphere-sphere", world(body("a", geom("s1", "sphere", "0.1"), "0 0 1"), body("b", geom("s2", "sphere", "0.15"), "0.2 0.1 1.05")))
run("sphere-capsule touching", world(body("a", geom("s", "sphere", "0.1"), "0.1 0.03 1.12"), body("b", geom("c", "capsule", "0.05 0.3"), "0 0 1", "0.7071068 0 0.7071068 0")))
run("sphere-capsule beyond the end", world(body("a", geom("s", "sphere", "0.1"), "0.36 0.02 1.1"), body("b", geom("c", "capsule", "0.05 0.3"), "0 0 1", "0.7071068 0 0.7071068 0")))
run("capsule-capsule skew", world(body("a", geom("c1", "capsule", "0.05 0.3"), "0 0 1"), body("b", geom("c2", "capsule", "0.04 0.25"), "0.05 0.02 1.1", "0.7071068 0.7071068 0 0")))
run("capsule-capsule parallel", world(body("a", geom("c1", "capsule", "0.05 0.3"), "0 0 1"), body("b", geom("c2", "capsule", "0.04 0.25"), "0.08 0 1.1")))
run("sphere-cylinder side", world(body("a", geom("s", "sphere", "0.1"), "0.18 0.02 1.05"), body("b", geom("c", "cylinder", "0.1 0.15"), "0 0 1")))
run("sphere-cylinder cap", world(body("a", geom("s", "sphere", "0.1"), "0.02 0.03 1.24"), body("b", geom("c", "cylinder", "0.1 0.15"), "0 0 1")))
run("sphere-box face", world(body("a", geom("s", "sphere", "0.1"), "0.05 0.02 1.24"), body("b", geom("bx", "box", "0.2 0.3 0.15"), "0 0 1")))
run("sphere-box edge", world(body("a", geom("s", "sphere", "0.1"), "0.26 0.36 1.0"), body("b", geom("bx", "box", "0.2 0.3 0.15"), "0 0 1")))
run("box-box face", world(body("a", geom("b1", "box", "0.2 0.3 0.15"), "0 0 1"), body("b", geom("b2", "box", "0.1 0.1 0.1"), "0.05 0.1 1.24", "0.9961947 0 0 0.0871557")))
run("box-box edge", world(body("a", geom("b1", "box", "0.2 0.3 0.15"), "0 0 1"), body("b", geom("b2", "box", "0.1 0.1 0.1"), "0.25 0.0 1.2", "0.9238795 0 0.3826834 0")))
run("param mixing: solmix + friction max + condim max + margin/gap", world('<geom name="floor" type="plane" size="1 1 0.1" friction="0.3" solref="0.05 0.8" solimp="0.8 0.9 0.002 0.4 3" solmix="2" condim="3"/>', body("b", geom("s", "sphere", "0.1", 'friction="0.7" solref="0.01 1.2" solimp="0.95 0.99 0.005 0.6 1" solmix="0.5" condim="4" margin="0.01" gap="0.002"'), "0 0 0.105")))
run("param mixing: priority wins", world('<geom name="floor" type="plane" size="1 1 0.1" friction="0.3" priority="1" solref="0.05 0.8" condim="1"/>', body("b", geom("s", "sphere", "0.1", 'friction="0.7" solref="0.01 1.2" condim="6"'), "0 0 0.095")))
