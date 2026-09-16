#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <float.h>
#include <mujoco/mujoco.h>
#include "engine/engine_collision_convex.h"
#include "engine/engine_collision_gjk.h"
#include "engine/engine_util_blas.h"
#include "engine/engine_util_spatial.h"
#include "engine/engine_inline.h"
#define mjMINVAL2 (mjMINVAL * mjMINVAL)
typedef struct { mjtNum dist; mjtNum pos[3]; mjtNum normal[3]; mjtNum tangent[3]; } PreContact;

// center function for convex collision algorithms
void mjc_center(mjtNum res[3], const mjCCDObj *obj) {
  int g = obj->geom;
  int f = obj->flex;
  int e = obj->elem;
  int v = obj->vert;

  if (obj->geom_type == mjGEOM_HFIELD) {
    mju_zero3(res);
    for (int i=0; i < 6; i++) {
      mji_addTo3(res, obj->data.hfield.prism[i]);
     }
    mju_scl3(res, res, 1.0/6.0);
    return;
  }

  // return geom position
  if (g >= 0) {
    mji_copy3(res, obj->pos);
    return;
  }

  // return flex element position
  if (e >= 0) {
    mji_copy3(res, obj->data.flex.aabb + 6*(obj->data.flex.elemadr[f]+e));
    return;
  }

  // return flex vertex position
  if (f >= 0) {
    mji_copy3(res, obj->data.flex.vert_xpos + 3*(obj->data.flex.vertadr[f]+v));
    return;
  }
}

// ------------------------------------ Support functions -----------------------------------------

// transform a vector from global to local frame
static inline void mulMatTVec3(mjtNum res[3], const mjtNum mat[9], const mjtNum dir[3]) {
  // perform matT * dir
  res[0] = mat[0]*dir[0] + mat[3]*dir[1] + mat[6]*dir[2];
  res[1] = mat[1]*dir[0] + mat[4]*dir[1] + mat[7]*dir[2];
  res[2] = mat[2]*dir[0] + mat[5]*dir[1] + mat[8]*dir[2];
}


// transform a vector from local to global frame
static inline void localToGlobal(mjtNum res[3], const mjtNum mat[9], const mjtNum dir[3],
                                 const mjtNum pos[3]) {
  // perform mat * dir + pos
  res[0] = mat[0]*dir[0] + mat[1]*dir[1] + mat[2]*dir[2];
  res[1] = mat[3]*dir[0] + mat[4]*dir[1] + mat[5]*dir[2];
  res[2] = mat[6]*dir[0] + mat[7]*dir[1] + mat[8]*dir[2];
  res[0] += pos[0];
  res[1] += pos[1];
  res[2] += pos[2];
}


// point support function
void mjc_pointSupport(mjtNum res[3], mjCCDObj* obj, const mjtNum dir[3]) {
  mji_copy3(res, obj->pos);
}


// sphere support function
static void mjc_sphereSupport(mjtNum res[3], mjCCDObj* obj, const mjtNum dir[3]) {
  // sphere data
  const mjtNum* pos = obj->pos;
  mjtNum radius = obj->size[0];

  res[0] = radius*dir[0] + pos[0];
  res[1] = radius*dir[1] + pos[1];
  res[2] = radius*dir[2] + pos[2];
}


// line support function (capsule)
void mjc_lineSupport(mjtNum res[3], mjCCDObj* obj, const mjtNum dir[3]) {
  // capsule data
  const mjtNum* mat = obj->mat;
  const mjtNum* pos = obj->pos;
  mjtNum length = obj->size[1];

  mjtNum dot = mat[2]*dir[0] + mat[5]*dir[1] + mat[8]*dir[2];
  mjtNum scl = dot >= 0 ? length : -length;

  // transform result to global frame
  res[0] = mat[2]*scl + pos[0];
  res[1] = mat[5]*scl + pos[1];
  res[2] = mat[8]*scl + pos[2];
}


// capsule support function
static void mjc_capsuleSupport(mjtNum res[3], mjCCDObj* obj, const mjtNum dir[3]) {
  // capsule data
  const mjtNum* mat = obj->mat;
  const mjtNum* pos = obj->pos;
  mjtNum radius = obj->size[0];
  mjtNum length = obj->size[1];

  // rotate dir to geom local frame
  mjtNum local_dir[3], local_supp[3];
  mulMatTVec3(local_dir, mat, dir);

  // start with sphere
  local_supp[0] = local_dir[0] * radius;
  local_supp[1] = local_dir[1] * radius;
  local_supp[2] = local_dir[2] * radius;

  // add cylinder contribution
  local_supp[2] += (local_dir[2] >= 0 ? length : -length);

  // transform result to global frame
  localToGlobal(res, mat, local_supp, pos);
}


// ellipsoid support function
static void mjc_ellipsoidSupport(mjtNum res[3], mjCCDObj* obj, const mjtNum dir[3]) {
  // ellipsoid data
  const mjtNum* mat = obj->mat;
  const mjtNum* pos = obj->pos;
  const mjtNum* size = obj->size;

  // rotate dir to geom local frame
  mjtNum local_dir[3], local_supp[3];
  mulMatTVec3(local_dir, mat, dir);

  // find support point on unit sphere: scale dir by ellipsoid sizes
  local_supp[0] = local_dir[0] * size[0];
  local_supp[1] = local_dir[1] * size[1];
  local_supp[2] = local_dir[2] * size[2];

  mjtNum norm2 = local_supp[0]*local_supp[0] + local_supp[1]*local_supp[1] + local_supp[2]*local_supp[2];

  // too small to normalize
  if (norm2 < mjMINVAL2) {
    res[0] = mat[0]*size[0] + pos[0];
    res[1] = mat[3]*size[0] + pos[1];
    res[2] = mat[6]*size[0] + pos[2];
    return;
  }

  // normalize and transform to ellipsoid
  mjtNum norm_inv = 1/mju_sqrt(norm2);
  local_supp[0] *= norm_inv * size[0];
  local_supp[1] *= norm_inv * size[1];
  local_supp[2] *= norm_inv * size[2];

  // transform result to global frame
  localToGlobal(res, mat, local_supp, pos);
}


// cylinder support function
static void mjc_cylinderSupport(mjtNum res[3], mjCCDObj* obj, const mjtNum dir[3]) {
  // cylinder data
  const mjtNum* mat = obj->mat;
  const mjtNum* pos = obj->pos;
  const mjtNum* size = obj->size;

  // rotate dir to geom local frame
  mjtNum local_dir[3], local_supp[3];
  mulMatTVec3(local_dir, mat, dir);

  mjtNum n2 = local_dir[0]*local_dir[0] + local_dir[1]*local_dir[1];
  mjtNum scl = n2 >= mjMINVAL2 ? size[0] / mju_sqrt(n2) : 0;
  local_supp[0] = scl * local_dir[0];
  local_supp[1] = scl * local_dir[1];

  // set result in Z direction
  local_supp[2] = local_dir[2] >= 0 ? size[1] : -size[1];
  obj->vertindex = local_dir[2] >= 0 ? 0 : 1;

  // transform result to global frame
  localToGlobal(res, mat, local_supp, pos);
}


// box support function
static void mjc_boxSupport(mjtNum res[3], mjCCDObj* obj, const mjtNum dir[3]) {
  // box data
  const mjtNum* mat = obj->mat;
  const mjtNum* pos = obj->pos;
  const mjtNum* size = obj->size;

  // rotate dir to geom local frame
  mjtNum local_dir[3], local_supp[3];
  mulMatTVec3(local_dir, mat, dir);

  // find support point in local frame
  local_supp[0] = local_dir[0] >= 0 ? size[0] : -size[0];
  local_supp[1] = local_dir[1] >= 0 ? size[1] : -size[1];
  local_supp[2] = local_dir[2] >= 0 ? size[2] : -size[2];

  // mark the index of the corner of the box for fast lookup
  obj->vertindex  = (local_supp[0] > 0) ? 1 : 0;
  obj->vertindex |= (local_supp[1] > 0) ? 2 : 0;
  obj->vertindex |= (local_supp[2] > 0) ? 4 : 0;

  // transform support point to global frame
  localToGlobal(res, mat, local_supp, pos);
}

static int mjc_isDistinctContact(const PreContact* con, int ncon, mjtNum tolerance) {
  const mjtNum* last_pos = con[ncon - 1].pos;
  for (int i=0; i < ncon-1; i++) {
    if (mju_dist3(con[i].pos, last_pos) <= tolerance) {
      return 0;
    }
  }
  return 1;
}


// in-place rotation of spatial frame around given point of origin
static void mju_rotateFrame(const mjtNum origin[3], const mjtNum rot[9],
                            mjtNum xmat[9], mjtNum xpos[3]) {
  mjtNum mat[9], vec[3], rel[3];

  // rotate frame: xmat = rot*xmat
  mju_mulMatMat3(mat, rot, xmat);
  mju_copy(xmat, mat, 9);

  // vector to rotation origin: rel = origin - xpos
  mji_sub3(rel, origin, xpos);

  // displacement of origin due to rotation: vec = rot*rel - rel
  mju_mulMatVec3(vec, rot, rel);
  mju_subFrom3(vec, rel);

  // correct xpos by subtracting displacement: xpos = xpos - vec
  mji_subFrom3(xpos, vec);
}
static void init(mjCCDObj* o, int type, const double size[3], const double pos[3], const double quat_in[4], double margin) {
  memset(o, 0, sizeof *o);
  o->geom = 0; o->geom_type = type; memcpy(o->size, size, 3*sizeof(double)); memcpy(o->pos, pos, 3*sizeof(double));
  double q[4] = {quat_in[0], quat_in[1], quat_in[2], quat_in[3]}; mju_normalize4(q); mju_quat2Mat(o->mat, q);
  o->vertindex = -1; o->meshindex = -1; o->flex = o->elem = o->vert = -1; o->margin = margin; o->rotate[0] = 1;
  o->center = mjc_center;
  switch (type) {
    case mjGEOM_SPHERE: o->support = mjc_sphereSupport; break;
    case mjGEOM_CAPSULE: o->support = mjc_capsuleSupport; break;
    case mjGEOM_ELLIPSOID: o->support = mjc_ellipsoidSupport; break;
    case mjGEOM_CYLINDER: o->support = mjc_cylinderSupport; break;
    case mjGEOM_BOX: o->support = mjc_boxSupport; break;
  }
}
static double rbound(const mjCCDObj* o) {
  const double* s = o->size;
  switch (o->geom_type) {
    case mjGEOM_SPHERE: return s[0];
    case mjGEOM_CAPSULE: return s[0] + s[1];
    case mjGEOM_CYLINDER: return sqrt(s[0]*s[0] + s[1]*s[1]);
    case mjGEOM_ELLIPSOID: return fmax(fmax(s[0], s[1]), s[2]);
    case mjGEOM_BOX: return sqrt(s[0]*s[0] + s[1]*s[1] + s[2]*s[2]);
  }
  return 0;
}
static int penetration(mjCCDObj* o1, mjCCDObj* o2, PreContact* con, int nconmax, double margin) {
  mjCCDConfig config; mjCCDStatus status;
  config.max_iterations = 35; config.tolerance = 1e-6; config.max_contacts = nconmax; config.dist_cutoff = 0;
  config.npolygonmax = 0; config.nmeshdegmax = 0;
  static char buffer[1 << 20]; config.buffer = buffer;
  int ncon = 0; double dist;
  if ((dist = mjc_ccd(&config, &status, o1, o2)) < 0) {
    for (int i = 0; i < status.nx; i++) {
      con[i].dist = margin + status.dist[i];
      for (int k = 0; k < 3; k++) { con[i].pos[k] = 0.5*(status.x1[3*i+k] + status.x2[3*i+k]); con[i].normal[k] = status.x1[3*i+k] - status.x2[3*i+k]; con[i].tangent[k] = 0; }
      mju_normalize3(con[i].normal);
    }
    ncon = status.nx;
  }
  return ncon;
}
static int max_contacts(const mjCCDObj* o1, const mjCCDObj* o2, int multiccd) {
  if (o1->margin > 0 || o2->margin > 0) return 1;
  int t1 = o1->geom_type, t2 = o2->geom_type;
  if (t1 == mjGEOM_BOX && t2 == mjGEOM_BOX) return 8;
  if ((t1 == mjGEOM_BOX || t1 == mjGEOM_MESH || t1 == mjGEOM_CYLINDER) && (t2 == mjGEOM_BOX || t2 == mjGEOM_MESH || t2 == mjGEOM_CYLINDER)) return multiccd ? 4 : 1;
  return 1;
}
static int convex(mjCCDObj* o1, mjCCDObj* o2, PreContact* con, double margin, int multiccd) {
  int maxc = max_contacts(o1, o2, multiccd);
  int ncon = penetration(o1, o2, con, maxc, margin);
  if (maxc > 1) return ncon;
  if (ncon == 1 && multiccd && o1->geom_type != mjGEOM_ELLIPSOID && o1->geom_type != mjGEOM_SPHERE && o2->geom_type != mjGEOM_ELLIPSOID && o2->geom_type != mjGEOM_SPHERE) {
    const mjtNum relative_tolerance = 1e-3, perturbation_angle = 1e-3;
    mjtNum frame[9]; mju_copy3(frame, con[0].normal); mju_zero(frame+3, 6); mju_makeFrame(frame);
    const mjtNum tolerance = relative_tolerance * mju_min(rbound(o1), rbound(o2));
    mjtNum* axes[2] = {frame+3, frame+6}; mjtNum angles[2] = {-perturbation_angle, perturbation_angle};
    mjtNum pos1[3], mat1[9], pos2[3], mat2[9]; mju_copy3(pos1, o1->pos); mju_copy(mat1, o1->mat, 9); mju_copy3(pos2, o2->pos); mju_copy(mat2, o2->mat, 9);
    for (int a = 0; a < 2; a++) for (int g = 0; g < 2; g++) {
      mjtNum quat[4], rot[9]; mji_axisAngle2Quat(quat, axes[a], angles[g]); mju_quat2Mat(rot, quat);
      mju_rotateFrame(con[0].pos, rot, o1->mat, o1->pos);
      mjtNum invrot[9]; mju_transpose(invrot, rot, 3, 3);
      mju_rotateFrame(con[0].pos, invrot, o2->mat, o2->pos);
      int n = penetration(o1, o2, con + ncon, 1, margin);
      if (n && mjc_isDistinctContact(con, ncon + 1, tolerance)) { con[ncon].dist = con[0].dist; ncon += 1; }
      mju_copy3(o1->pos, pos1); mju_copy(o1->mat, mat1, 9); mju_copy3(o2->pos, pos2); mju_copy(o2->mat, mat2, 9);
    }
  }
  return ncon;
}
static void run(const char* label, mjCCDObj* o1, mjCCDObj* o2, int multiccd) {
  PreContact con[64]; int n = convex(o1, o2, con, 0.0, multiccd);
  printf("=== %s: ncon=%d\n", label, n);
  for (int i = 0; i < n; i++) printf("  dist=%.17g pos=[%.17g, %.17g, %.17g] normal=[%.17g, %.17g, %.17g]\n", con[i].dist, con[i].pos[0], con[i].pos[1], con[i].pos[2], con[i].normal[0], con[i].normal[1], con[i].normal[2]);
}
// same, but taking the rotation matrix directly — Menagerie poses come out of the oracle as geom_xmat
static void init_mat(mjCCDObj* o, int type, const double size[3], const double pos[3], const double mat[9], double margin) {
  double q[4] = {1, 0, 0, 0};
  init(o, type, size, pos, q, margin);
  memcpy(o->mat, mat, 9 * sizeof(double));
}

int main(void) {
  const double id[4] = {1,0,0,0};
  mjCCDObj a, b;
  double bx[3] = {0.2, 0.3, 0.15}, bp[3] = {0, 0, 1};
  { double s[3] = {0.05, 0.1, 0}, p[3] = {0.04, 0.02, 1.24}; init(&a, mjGEOM_CYLINDER, s, p, id, 0); init(&b, mjGEOM_BOX, bx, bp, id, 0); run("cylinder on box", &a, &b, 1); }
  { double s[3] = {0.04, 0.1, 0}, p[3] = {0.04, 0.02, 1.18}, q[4] = {0.7071068, 0, 0.7071068, 0}; init(&a, mjGEOM_CAPSULE, s, p, q, 0); init(&b, mjGEOM_BOX, bx, bp, id, 0); run("capsule on box", &a, &b, 1); }
  { double s[3] = {0.05, 0.2, 0}, p1[3] = {0, 0, 1}, p2[3] = {0.02, 0.01, 1.095}, q[4] = {0.7071068, 0, 0.7071068, 0}; init(&a, mjGEOM_CYLINDER, s, p1, id, 0); init(&b, mjGEOM_CYLINDER, s, p2, q, 0); run("cylinder-cylinder crossed", &a, &b, 1); }
  { double s[3] = {0.06, 0.04, 0.03}, p[3] = {0.03, 0.02, 1.175}, q[4] = {0.9238795, 0, 0.3826834, 0}; init(&a, mjGEOM_ELLIPSOID, s, p, q, 0); init(&b, mjGEOM_BOX, bx, bp, id, 0); run("ellipsoid on box", &a, &b, 1); }

  // fourier_n1/n1.xml geoms 1 and 16 at MuJoCo's own geom_xpos/geom_xmat: two cylinders overlapping by
  // 31 micrometres, where this port and the arm64 wheel report opposite normals at identical depth
  {
    double s1[3] = {0.065, 0.0675, 0}, p1[3] = {0.08539340246023025, 0.032896559837938814, 0.6641187211978501};
    double m1[9] = {-0.49418139395483024, -0.8109762031412986, 0.31321294323093446,
                     0.6602653884717504, -0.11574723429074663, 0.7420594279033802,
                    -0.5655390054232421, 0.5735156281153694, 0.592659647396652};
    double s2[3] = {0.06, 0.095, 0}, p2[3] = {0.04788949966356997, 0.23417202888574384, 0.6552473273724194};
    double m2[9] = {-0.6018919636550226, -0.46927452071099596, 0.6461481937597335,
                    -0.1511126975750603, -0.7275770566496756, -0.6691760450498682,
                     0.7841498687872679, -0.5004128803782532, 0.3670094446093328};
    init_mat(&a, mjGEOM_CYLINDER, s1, p1, m1, 0);
    init_mat(&b, mjGEOM_CYLINDER, s2, p2, m2, 0);
    run("fourier_n1 geom1 vs geom16 (31 um overlap)", &a, &b, 1);
  }
  return 0;
}
