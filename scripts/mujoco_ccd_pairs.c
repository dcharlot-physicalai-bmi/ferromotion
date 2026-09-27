// Rerun Menagerie CCD pairs through MuJoCo 3.13.0's OWN engine_collision_gjk.c. Reads the pairs
// `menagerie_contacts` writes under MENAGERIE_CCD_DUMP=<file> (each object's type, size, pose, and a mesh's
// stored vertices, qhull graph, extrema and polygons in MuJoCo's order), runs mjc_Convex's logic on them, and
// prints MuJoCo's contacts beside ours:
//
//     scripts/mujoco_ccd_harness_build.sh <work dir>
//     <work dir>/mjsrc/pairs_on < dump.txt > compare.txt
//     python scripts/mujoco_ccd_pairs_compare.py compare.txt
//
// Built with floating-point contraction ON it is the Apple-silicon wheel; OFF, a build without FMA.
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

// ---- mesh support, from engine_collision_convex.c 3.13.0 ----
static inline mjtNum dot3f(const mjtNum a[3], const float b[3]) {
  return a[0]*(mjtNum)b[0] + a[1]*(mjtNum)b[1] + a[2]*(mjtNum)b[2];
}
static void mjc_meshSupport(mjtNum res[3], mjCCDObj* obj, const mjtNum dir[3]) {
  const mjtNum* mat = obj->mat;
  const mjtNum* pos = obj->pos;
  const float* verts = obj->data.mesh.vert;
  int nverts = obj->data.mesh.nvert;
  mjtNum local_dir[3];
  mulMatTVec3(local_dir, mat, dir);
  mjtNum max = -FLT_MAX;
  int imax = 0;
  if (obj->vertindex >= 0) {
    imax = obj->vertindex;
    max = dot3f(local_dir, verts + 3*imax);
  }
  for (int i=0; i < nverts; i++) {
    mjtNum vdot = dot3f(local_dir, verts + 3*i);
    if (vdot > max) {
      max = vdot;
      imax = i;
    }
  }
  obj->vertindex = imax;
  local_dir[0] = (mjtNum)verts[3*imax + 0];
  local_dir[1] = (mjtNum)verts[3*imax + 1];
  local_dir[2] = (mjtNum)verts[3*imax + 2];
  localToGlobal(res, mat, local_dir, pos);
}
static void mjc_hillclimbSupport(mjtNum res[3], mjCCDObj* obj, const mjtNum dir[3]) {
  int numvert = obj->data.mesh.graph[0];
  const int* vert_edgeadr = obj->data.mesh.graph + 2;
  const int* vert_globalid = obj->data.mesh.graph  + 2 + numvert;
  const int* edge_localid = obj->data.mesh.graph + 2 + 2*numvert;
  const float* verts = obj->data.mesh.vert;
  const mjtNum* pos = obj->pos;
  const mjtNum* mat = obj->mat;
  mjtNum local_dir[3];
  mulMatTVec3(local_dir, mat, dir);
  int prev = -1;
  int imax;
  int cx = (local_dir[0] > 0.4) - (local_dir[0] < -0.4) + 1;
  int cy = (local_dir[1] > 0.4) - (local_dir[1] < -0.4) + 1;
  int cz = (local_dir[2] > 0.4) - (local_dir[2] < -0.4) + 1;
  int grid_idx = obj->data.mesh.extrema[cx*9 + cy*3 + cz];
  if (obj->meshindex >= 0) {
    mjtNum cached_dot = dot3f(local_dir, verts + 3*vert_globalid[obj->meshindex]);
    mjtNum seed_dot = dot3f(local_dir, verts + 3*vert_globalid[grid_idx]);
    imax = (seed_dot > cached_dot) ? grid_idx : obj->meshindex;
  } else {
    imax = grid_idx;
  }
  mjtNum max = dot3f(local_dir, verts + 3*vert_globalid[imax]);
  while (imax != prev) {
    prev = imax;
    int subidx;
    for (int i = vert_edgeadr[imax]; (subidx = edge_localid[i]) >= 0; i++) {
      mjtNum vdot = dot3f(local_dir, verts + 3*vert_globalid[subidx]);
      if (vdot > max) {
        max = vdot;
        imax = subidx;
      }
    }
  }
  obj->meshindex = imax;
  obj->vertindex = imax = vert_globalid[imax];
  local_dir[0] = (mjtNum)verts[3*imax + 0];
  local_dir[1] = (mjtNum)verts[3*imax + 1];
  local_dir[2] = (mjtNum)verts[3*imax + 2];
  localToGlobal(res, mat, local_dir, pos);
}

// ---- reading the dump ----
static char* line = NULL; static size_t linecap = 0;
static char* field(char** s) { char* t = strsep(s, "\t"); return t ? t : ""; }
typedef struct { float* vert; int* graph; int extrema[27]; int* polyvertadr; int* polyvertnum; int* polyvert; mjtNum* polynormal;
                 int* polymapadr; int* polymapnum; int* polymap; int npoly; int nvert; } MeshData;
static int read_obj(mjCCDObj* o, MeshData* md, double margin) {
  if (getline(&line, &linecap, stdin) <= 0) return 0;
  char* s = line; field(&s);
  int type = atoi(field(&s));
  double size[3], pos[3], mat[9];
  char* f = field(&s); sscanf(f, "%lf %lf %lf", size, size+1, size+2);
  f = field(&s); sscanf(f, "%lf %lf %lf", pos, pos+1, pos+2);
  f = field(&s); for (int k = 0; k < 9; k++) mat[k] = strtod(f, &f);
  memset(o, 0, sizeof *o);
  o->geom = 0; o->geom_type = type; memcpy(o->size, size, 3*sizeof(double)); memcpy(o->pos, pos, 3*sizeof(double)); memcpy(o->mat, mat, 9*sizeof(double));
  o->vertindex = -1; o->meshindex = -1; o->flex = o->elem = o->vert = -1; o->margin = margin; o->rotate[0] = 1;
  o->center = mjc_center;
  switch (type) {
    case mjGEOM_SPHERE: o->support = mjc_sphereSupport; break;
    case mjGEOM_CAPSULE: o->support = mjc_capsuleSupport; break;
    case mjGEOM_ELLIPSOID: o->support = mjc_ellipsoidSupport; break;
    case mjGEOM_CYLINDER: o->support = mjc_cylinderSupport; break;
    case mjGEOM_BOX: o->support = mjc_boxSupport; break;
  }
  if (type == mjGEOM_MESH) {
    if (getline(&line, &linecap, stdin) <= 0) return 0;
    s = line; field(&s);
    int nvert = atoi(field(&s));
    md->nvert = nvert;
    md->vert = malloc(sizeof(float) * 3 * nvert);
    f = field(&s); for (int k = 0; k < 3*nvert; k++) md->vert[k] = (float)strtod(f, &f);
    f = field(&s);
    int has_graph = f[0] != '-';
    if (has_graph) {
      int cap = 1024, n = 0; md->graph = malloc(sizeof(int) * cap);
      char* e;
      for (;;) { long v = strtol(f, &e, 10); if (e == f) break; if (n == cap) { cap *= 2; md->graph = realloc(md->graph, sizeof(int) * cap); } md->graph[n++] = (int)v; f = e; }
    } else md->graph = NULL;
    f = field(&s);
    if (has_graph) for (int k = 0; k < 27; k++) md->extrema[k] = (int)strtol(f, &f, 10);
    md->npoly = atoi(field(&s));
    f = field(&s);
    md->polyvertadr = malloc(sizeof(int) * (md->npoly + 1)); md->polyvertnum = malloc(sizeof(int) * (md->npoly + 1));
    md->polynormal = malloc(sizeof(mjtNum) * 3 * (md->npoly + 1));
    int cap = 1024, nv = 0; md->polyvert = malloc(sizeof(int) * cap);
    for (int p = 0; p < md->npoly; p++) {
      int n = (int)strtol(f, &f, 10);
      md->polyvertadr[p] = nv; md->polyvertnum[p] = n;
      for (int k = 0; k < n; k++) { if (nv == cap) { cap *= 2; md->polyvert = realloc(md->polyvert, sizeof(int) * cap); } md->polyvert[nv++] = (int)strtol(f, &f, 10); }
      for (int k = 0; k < 3; k++) md->polynormal[3*p + k] = strtod(f, &f);
    }
    // polymap: per vertex, its polygons in index order (as mjCMesh::MakePolygons fills polygon_map_)
    md->polymapnum = calloc(nvert, sizeof(int)); md->polymapadr = calloc(nvert, sizeof(int));
    for (int k = 0; k < nv; k++) md->polymapnum[md->polyvert[k]]++;
    int adr = 0; for (int v = 0; v < nvert; v++) { md->polymapadr[v] = adr; adr += md->polymapnum[v]; }
    md->polymap = malloc(sizeof(int) * (adr + 1)); int* fill = calloc(nvert, sizeof(int));
    for (int p = 0; p < md->npoly; p++) for (int k = 0; k < md->polyvertnum[p]; k++) { int v = md->polyvert[md->polyvertadr[p] + k]; md->polymap[md->polymapadr[v] + fill[v]++] = p; }
    free(fill);
    o->data.mesh.nvert = nvert; o->data.mesh.vert = md->vert; o->data.mesh.mesh_polynum = md->npoly;
    o->data.mesh.mpolymapadr = md->polymapadr; o->data.mesh.mpolymapnum = md->polymapnum; o->data.mesh.polymap = md->polymap;
    o->data.mesh.polyvertadr = md->polyvertadr; o->data.mesh.polyvertnum = md->polyvertnum; o->data.mesh.polyvert = md->polyvert;
    o->data.mesh.polynormal = md->polynormal;
    if (md->graph && nvert >= mjMESH_HILLCLIMB_MIN) { o->data.mesh.graph = md->graph; o->data.mesh.extrema = md->extrema; o->support = mjc_hillclimbSupport; }
    else { o->data.mesh.graph = NULL; o->data.mesh.extrema = NULL; o->support = mjc_meshSupport; }
  }
  return 1;
}
static double rbound_of(const mjCCDObj* o) {
  const double* s = o->size;
  switch (o->geom_type) {
    case mjGEOM_SPHERE: return s[0];
    case mjGEOM_CAPSULE: return s[0] + s[1];
    case mjGEOM_CYLINDER: return sqrt(s[0]*s[0] + s[1]*s[1]);
    case mjGEOM_ELLIPSOID: return fmax(fmax(s[0], s[1]), s[2]);
    case mjGEOM_BOX: return sqrt(s[0]*s[0] + s[1]*s[1] + s[2]*s[2]);
    case mjGEOM_MESH: {
      double h[3] = {0, 0, 0};
      for (int i = 0; i < o->data.mesh.nvert; i++) for (int k = 0; k < 3; k++) { double a = fabs((double)o->data.mesh.vert[3*i + k]); if (a > h[k]) h[k] = a; }
      return sqrt(h[0]*h[0] + h[1]*h[1] + h[2]*h[2]);
    }
  }
  return 0;
}
static double g_tol; static int g_iter;
static int penetration2(mjCCDObj* o1, mjCCDObj* o2, PreContact* con, int nconmax, double margin) {
  mjCCDConfig config; mjCCDStatus status;
  config.max_iterations = g_iter; config.tolerance = g_tol; config.max_contacts = nconmax; config.dist_cutoff = 0;
  config.npolygonmax = 4096; config.nmeshdegmax = 4096;
  static uint8_t* buffer = NULL; static size_t bufsize = 0;
  size_t need = mjc_ccdSize(4096, 4096, g_iter);
  if (need > bufsize) { buffer = realloc(buffer, need); bufsize = need; }
  config.buffer = buffer;
  int ncon = 0; double dist;
  if ((dist = mjc_ccd(&config, &status, o1, o2)) < 0) {
    for (int i = 0; i < status.nx; i++) {
      con[i].dist = margin + status.dist[i];
      mji_add3(con[i].pos, status.x1 + 3*i, status.x2 + 3*i);
      mju_scl3(con[i].pos, con[i].pos, 0.5);
      mji_sub3(con[i].normal, status.x1 + 3*i, status.x2 + 3*i);
      mju_normalize3(con[i].normal);
      mji_zero3(con[i].tangent);
    }
    ncon = status.nx;
  }
  return ncon;
}
static int max_contacts2(const mjCCDObj* o1, const mjCCDObj* o2, int multiccd) {
  if (o1->margin > 0 || o2->margin > 0) return 1;
  int t1 = o1->geom_type, t2 = o2->geom_type;
  if (t1 == mjGEOM_BOX && t2 == mjGEOM_BOX) return 8;
  if ((t1 == mjGEOM_BOX || t1 == mjGEOM_MESH || t1 == mjGEOM_CYLINDER) && (t2 == mjGEOM_BOX || t2 == mjGEOM_MESH || t2 == mjGEOM_CYLINDER)) return multiccd ? 4 : 1;
  return 1;
}
static int convex2(mjCCDObj* o1, mjCCDObj* o2, PreContact* con, double margin, int multiccd) {
  int maxc = max_contacts2(o1, o2, multiccd);
  int ncon = penetration2(o1, o2, con, maxc, margin);
  if (maxc > 1) return ncon;
  if (ncon == 1 && multiccd && o1->geom_type != mjGEOM_ELLIPSOID && o1->geom_type != mjGEOM_SPHERE && o2->geom_type != mjGEOM_ELLIPSOID && o2->geom_type != mjGEOM_SPHERE) {
    const mjtNum relative_tolerance = 1e-3, perturbation_angle = 1e-3;
    mjtNum frame[9]; mji_copy3(frame, con[0].normal); mju_zero(frame+3, 6); mju_makeFrame(frame);
    const mjtNum tolerance = relative_tolerance * mju_min(rbound_of(o1), rbound_of(o2));
    mjtNum* axes[2] = {frame+3, frame+6}; mjtNum angles[2] = {-perturbation_angle, perturbation_angle};
    mjtNum pos1[3], mat1[9], pos2[3], mat2[9]; mji_copy3(pos1, o1->pos); mju_copy(mat1, o1->mat, 9); mji_copy3(pos2, o2->pos); mju_copy(mat2, o2->mat, 9);
    for (int a = 0; a < 2; a++) for (int g = 0; g < 2; g++) {
      mjtNum quat[4], rot[9]; mji_axisAngle2Quat(quat, axes[a], angles[g]); mju_quat2Mat(rot, quat);
      mju_rotateFrame(con[0].pos, rot, o1->mat, o1->pos);
      mjtNum invrot[9]; mju_transpose(invrot, rot, 3, 3);
      mju_rotateFrame(con[0].pos, invrot, o2->mat, o2->pos);
      int n = penetration2(o1, o2, con + ncon, 1, margin);
      if (n && mjc_isDistinctContact(con, ncon + 1, tolerance)) { con[ncon].dist = con[0].dist; ncon += 1; }
      mji_copy3(o1->pos, pos1); mju_copy(o1->mat, mat1, 9); mji_copy3(o2->pos, pos2); mju_copy(o2->mat, mat2, 9);
    }
  }
  return ncon;
}
int main(void) {
  while (getline(&line, &linecap, stdin) > 0) {
    if (strncmp(line, "pair\t", 5)) continue;
    char hdr[4096]; strncpy(hdr, line, sizeof hdr - 1); hdr[sizeof hdr - 1] = 0; hdr[strcspn(hdr, "\n")] = 0;
    char* s = line; field(&s); char* rel = field(&s); (void)rel; field(&s); field(&s); field(&s);
    double margin = strtod(field(&s), NULL); g_tol = strtod(field(&s), NULL); g_iter = atoi(field(&s)); int multiccd = atoi(field(&s));
    mjCCDObj a, b; MeshData ma = {0}, mb = {0};
    if (!read_obj(&a, &ma, margin) || !read_obj(&b, &mb, margin)) break;
    if (getline(&line, &linecap, stdin) <= 0) break;
    char ours[65536]; strncpy(ours, line, sizeof ours - 1); ours[sizeof ours - 1] = 0; ours[strcspn(ours, "\n")] = 0;
    PreContact con[64];
    int n = convex2(&a, &b, con, margin, multiccd);
    printf("%s\n%s\nmujoco\t%d", hdr, ours, n);
    for (int i = 0; i < n; i++) printf("\t%a %a %a %a %a %a %a", con[i].dist, con[i].pos[0], con[i].pos[1], con[i].pos[2], con[i].normal[0], con[i].normal[1], con[i].normal[2]);
    printf("\n");
    free(ma.vert); free(ma.graph); free(ma.polyvertadr); free(ma.polyvertnum); free(ma.polyvert); free(ma.polynormal); free(ma.polymapadr); free(ma.polymapnum); free(ma.polymap);
    free(mb.vert); free(mb.graph); free(mb.polyvertadr); free(mb.polyvertnum); free(mb.polyvert); free(mb.polynormal); free(mb.polymapadr); free(mb.polymapnum); free(mb.polymap);
  }
  return 0;
}
