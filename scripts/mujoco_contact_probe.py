"""MuJoCo's soft-constraint contact, exposed: for one frictionless sphere-on-plane contact at set penetrations
and normal velocities, the impedance/stiffness/damping it computes (efc_KBIP), its reference acceleration
(efc_aref), regulariser (efc_R), inverse (efc_D), the force it solves for, and the resulting qacc."""
import mujoco, numpy as np
np.set_printoptions(precision=15)

def scene(condim=1, solref="0.02 1", solimp="0.9 0.95 0.001 0.5 2", cone="pyramidal", solver="Newton", mass_r=(1.0, 0.1), friction="1 0.005 0.0001", extra=""):
    return f'''<mujoco><option gravity="0 0 -9.81" cone="{cone}" solver="{solver}" iterations="200" tolerance="1e-15" timestep="0.001"><flag warmstart="disable"/></option>
<worldbody>
<geom name="floor" type="plane" size="1 1 0.1" condim="{condim}" solref="{solref}" solimp="{solimp}" friction="{friction}"/>
<body name="ball" pos="0 0 {mass_r[1]}"><freejoint/><geom type="sphere" size="{mass_r[1]}" mass="{mass_r[0]}" condim="{condim}" solref="{solref}" solimp="{solimp}" friction="{friction}"/></body>
{extra}
</worldbody></mujoco>'''

def probe(xml, z, vz, label, vx=0.0, wy=0.0):
    m = mujoco.MjModel.from_xml_string(xml); d = mujoco.MjData(m)
    d.qpos[2] = z; d.qvel[2] = vz; d.qvel[0] = vx; d.qvel[4] = wy
    mujoco.mj_forward(m, d)
    print(f"=== {label}: z={z} vz={vz} vx={vx} ncon={d.ncon} nefc={d.nefc}")
    for i in range(d.nefc):
        print(f"  row{i} type={d.efc_type[i]} pos={d.efc_pos[i]:.12e} margin={d.efc_margin[i]} vel={d.efc_vel[i]:.12e} KBIP={d.efc_KBIP[i].tolist()} R={d.efc_R[i]:.12e} D={d.efc_D[i]:.12e} aref={d.efc_aref[i]:.12e} force={d.efc_force[i]:.12e}")
    print(f"  qacc={d.qacc.tolist()}")
    if d.ncon: 
        c = d.contact[0]; print(f"  contact dist={c.dist:.12e} frame={c.frame.tolist()} dim={c.dim} solref={c.solref.tolist()} solimp={c.solimp.tolist()} friction={c.friction.tolist()} efc_address={c.efc_address}")
    return m, d

# 1. frictionless, resting penetration ladder (default solref/solimp of MuJoCo: 0.02 1 / 0.9 0.95 0.001 0.5 2)
for z in [0.1, 0.0999, 0.0995, 0.099, 0.095]:
    probe(scene(condim=1), z, 0.0, "condim1 rest")
# 2. frictionless, moving
probe(scene(condim=1), 0.0995, -0.3, "condim1 closing")
probe(scene(condim=1), 0.0995, +0.3, "condim1 opening")
# 3. non-default solref/solimp
probe(scene(condim=1, solref="0.05 0.5", solimp="0.8 0.99 0.01 0.3 3"), 0.0995, -0.1, "condim1 custom solref/solimp")
# negative solref = direct stiffness/damping form
probe(scene(condim=1, solref="-1000 -50"), 0.0995, -0.1, "condim1 negative solref (k,b direct)")
# 4. condim 3 pyramidal, sliding
probe(scene(condim=3, cone="pyramidal"), 0.0995, 0.0, "condim3 pyramidal sliding", vx=0.5)
probe(scene(condim=3, cone="elliptic"), 0.0995, 0.0, "condim3 elliptic sliding", vx=0.5)
probe(scene(condim=3, cone="pyramidal"), 0.0995, 0.0, "condim3 pyramidal at rest")
