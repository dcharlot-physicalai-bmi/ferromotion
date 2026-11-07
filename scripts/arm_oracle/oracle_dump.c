/* Full-field Menagerie oracle in the text format of scripts/mujoco_native_oracle.py
   and scripts/mujoco_native_step_oracle.py. States are replayed from an existing
   dump so the random stream is the numpy seed already used on x86. Physics is
   recomputed by the libmujoco this binary is linked to. */
#include <mujoco/mujoco.h>
#include <ctype.h>
#include <dirent.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

static void pyhex(FILE* o, double x) {
  uint64_t u;
  memcpy(&u, &x, 8);
  if (u >> 63) fputc('-', o);
  int be = (int)((u >> 52) & 0x7ff);
  uint64_t frac = u & 0xfffffffffffffULL;
  if (be == 0 && frac == 0) {
    fputs("0x0.0p+0", o);
    return;
  }
  if (be == 0x7ff) {
    fputs(frac ? "nan" : "inf", o);
    return;
  }
  if (be == 0) {
    int shift = 0;
    uint64_t v = frac;
    while ((v & (1ULL << 51)) == 0) {
      v <<= 1;
      shift++;
    }
    uint64_t mant = (v << 1) & 0xfffffffffffffULL;
    fprintf(o, "0x1.%013llxp%+d", (unsigned long long)mant, -1022 - shift);
    return;
  }
  fprintf(o, "0x1.%013llxp%+d", (unsigned long long)frac, be - 1023);
}

static void space_d(FILE* o, const double* p, int n) {
  for (int i = 0; i < n; i++) {
    if (i) fputc(' ', o);
    pyhex(o, p[i]);
  }
}
static void space_i(FILE* o, const int* p, int n) {
  for (int i = 0; i < n; i++) fprintf(o, "%s%d", i ? " " : "", p[i]);
}
static void space_b(FILE* o, const mjtByte* p, int n) {
  for (int i = 0; i < n; i++) fprintf(o, "%s%d", i ? " " : "", (int)p[i]);
}

static void line_d(FILE* o, const char* k, const char* name, const double* p, int n) {
  fprintf(o, "%s\t%s\t", k, name);
  space_d(o, p, n);
  fputc('\n', o);
}
static void line_i(FILE* o, const char* k, const char* name, const int* p, int n) {
  fprintf(o, "%s\t%s\t", k, name);
  space_i(o, p, n);
  fputc('\n', o);
}

typedef struct {
  char* rel;
  int nstate;
  double** qpos;
  double** qvel;
  double** ctrl;
  double** act;
  int* nq;
  int* nv;
  int* nu;
  int* na;
} Sched;

static char* xstrdup(const char* s) {
  size_t n = strlen(s);
  char* p = malloc(n + 1);
  memcpy(p, s, n + 1);
  return p;
}

static double* parse_nums(const char* s, int* n_out) {
  int cap = 16, n = 0;
  double* a = malloc((size_t)cap * sizeof(double));
  while (*s) {
    while (*s == ' ' || *s == '\t') s++;
    if (!*s) break;
    char* end = NULL;
    /* Python hex floats are not strtod. Parse 0xH.HHHp±E ourselves via bits. */
    int neg = 0;
    if (*s == '-') { neg = 1; s++; }
    uint64_t bits = 0;
    if (s[0] == '0' && s[1] == 'x') {
      s += 2;
      const char* pexp = strchr(s, 'p');
      if (!pexp) { free(a); return NULL; }
      char mant[128];
      size_t ml = (size_t)(pexp - s);
      if (ml >= sizeof mant) { free(a); return NULL; }
      memcpy(mant, s, ml);
      mant[ml] = 0;
      char* dot = strchr(mant, '.');
      char digits[32] = {0};
      int nd = 0;
      const char* frac = "";
      const char* ip = mant;
      if (dot) { *dot = 0; frac = dot + 1; }
      /* copy up to 13 hex fraction digits */
      for (int i = 0; frac[i] && nd < 13; i++) digits[nd++] = frac[i];
      while (nd < 13) digits[nd++] = '0';
      uint64_t fbits = strtoull(digits, NULL, 16);
      int exp = atoi(pexp + 1);
      if (mant[0] == '0' && fbits == 0) bits = 0;
      else if (mant[0] == '0') bits = fbits;
      else bits = ((uint64_t)(exp + 1023) << 52) | fbits;
      if (neg) bits |= 1ULL << 63;
      s = pexp + 1;
      while (*s && *s != ' ' && *s != '\t') s++;
    } else {
      double v = strtod(neg ? s - 1 : s, &end);
      if (end == s) break;
      memcpy(&bits, &v, 8);
      s = end;
      (void)v;
    }
    double v;
    memcpy(&v, &bits, 8);
    if (n == cap) { cap *= 2; a = realloc(a, (size_t)cap * sizeof(double)); }
    a[n++] = v;
  }
  *n_out = n;
  return a;
}

static Sched* load_native_schedule(const char* path, int* nmodels) {
  FILE* f = fopen(path, "r");
  if (!f) { perror(path); exit(2); }
  int cap = 16, n = 0;
  Sched* ms = calloc((size_t)cap, sizeof(Sched));
  char* line = NULL;
  size_t len = 0;
  Sched* cur = NULL;
  while (getline(&line, &len, f) != -1) {
    char* tab = strchr(line, '\t');
    if (!tab) continue;
    *tab = 0;
    char* rest = tab + 1;
    char* nl = strchr(rest, '\n');
    if (nl) *nl = 0;
    if (strcmp(line, "model") == 0) {
      if (n == cap) { cap *= 2; ms = realloc(ms, (size_t)cap * sizeof(Sched)); }
      cur = &ms[n++];
      memset(cur, 0, sizeof(Sched));
      cur->rel = xstrdup(rest);
    } else if (strcmp(line, "qpos") == 0 && cur) {
      int k = cur->nstate++;
      cur->qpos = realloc(cur->qpos, (size_t)cur->nstate * sizeof(double*));
      cur->qvel = realloc(cur->qvel, (size_t)cur->nstate * sizeof(double*));
      cur->ctrl = realloc(cur->ctrl, (size_t)cur->nstate * sizeof(double*));
      cur->act = realloc(cur->act, (size_t)cur->nstate * sizeof(double*));
      cur->nq = realloc(cur->nq, (size_t)cur->nstate * sizeof(int));
      cur->nv = realloc(cur->nv, (size_t)cur->nstate * sizeof(int));
      cur->nu = realloc(cur->nu, (size_t)cur->nstate * sizeof(int));
      cur->na = realloc(cur->na, (size_t)cur->nstate * sizeof(int));
      cur->qpos[k] = parse_nums(rest, &cur->nq[k]);
      cur->qvel[k] = NULL; cur->ctrl[k] = NULL; cur->act[k] = NULL;
      cur->nv[k] = cur->nu[k] = cur->na[k] = 0;
    } else if (cur && cur->nstate) {
      int k = cur->nstate - 1;
      int nn = 0;
      if (strcmp(line, "qvel") == 0) cur->qvel[k] = parse_nums(rest, &cur->nv[k]);
      else if (strcmp(line, "ctrl") == 0) cur->ctrl[k] = parse_nums(rest, &cur->nu[k]);
      else if (strcmp(line, "act") == 0) cur->act[k] = parse_nums(rest, &cur->na[k]);
      (void)nn;
    }
  }
  free(line);
  fclose(f);
  *nmodels = n;
  return ms;
}

static void emit_model(FILE* o, mjModel* m) {
  int nb = (int)m->nbody, nj = (int)m->njnt, nv = (int)m->nv, nt = (int)m->ntendon;
  line_i(o, "m", "body_parentid", m->body_parentid, nb);
  line_i(o, "m", "body_rootid", m->body_rootid, nb);
  line_i(o, "m", "body_weldid", m->body_weldid, nb);
  line_i(o, "m", "body_jntnum", m->body_jntnum, nb);
  line_i(o, "m", "body_jntadr", m->body_jntadr, nb);
  line_i(o, "m", "body_dofnum", m->body_dofnum, nb);
  line_i(o, "m", "body_dofadr", m->body_dofadr, nb);
  fprintf(o, "m\tbody_simple\t"); space_b(o, m->body_simple, nb); fputc('\n', o);
  line_d(o, "m", "body_mass", m->body_mass, nb);
  line_d(o, "m", "body_subtreemass", m->body_subtreemass, nb);
  line_d(o, "m", "body_inertia", m->body_inertia, 3 * nb);
  line_d(o, "m", "body_ipos", m->body_ipos, 3 * nb);
  line_d(o, "m", "body_iquat", m->body_iquat, 4 * nb);
  line_d(o, "m", "body_pos", m->body_pos, 3 * nb);
  line_d(o, "m", "body_quat", m->body_quat, 4 * nb);
  line_i(o, "m", "jnt_type", m->jnt_type, nj);
  line_i(o, "m", "jnt_qposadr", m->jnt_qposadr, nj);
  line_i(o, "m", "jnt_dofadr", m->jnt_dofadr, nj);
  line_i(o, "m", "jnt_bodyid", m->jnt_bodyid, nj);
  line_d(o, "m", "jnt_pos", m->jnt_pos, 3 * nj);
  line_d(o, "m", "jnt_axis", m->jnt_axis, 3 * nj);
  line_i(o, "m", "dof_bodyid", m->dof_bodyid, nv);
  line_i(o, "m", "dof_jntid", m->dof_jntid, nv);
  line_i(o, "m", "dof_parentid", m->dof_parentid, nv);
  line_i(o, "m", "dof_simplenum", m->dof_simplenum, nv);
  line_d(o, "m", "dof_M0", m->dof_M0, nv);
  line_d(o, "m", "dof_armature", m->dof_armature, nv);
  line_d(o, "m", "dof_damping", m->dof_damping, nv);
  line_i(o, "m", "M_rownnz", m->M_rownnz, nv);
  line_i(o, "m", "M_rowadr", m->M_rowadr, nv);
  line_i(o, "m", "M_colind", m->M_colind, (int)m->nC);
  line_d(o, "m", "qpos0", m->qpos0, (int)m->nq);
  line_d(o, "m", "body_invweight0", m->body_invweight0, 2 * nb);
  line_d(o, "m", "dof_invweight0", m->dof_invweight0, nv);
  line_d(o, "m", "tendon_invweight0", m->tendon_invweight0, nt);
  line_d(o, "m", "gravity", m->opt.gravity, 3);
}

static void emit_data(FILE* o, mjModel* m, mjData* d) {
  int nb = (int)m->nbody, nj = (int)m->njnt, nv = (int)m->nv, nu = (int)m->nu, na = (int)m->na;
  int ne = d->nefc;
  line_d(o, "d", "xpos", d->xpos, 3 * nb);
  line_d(o, "d", "xquat", d->xquat, 4 * nb);
  line_d(o, "d", "xmat", d->xmat, 9 * nb);
  line_d(o, "d", "xipos", d->xipos, 3 * nb);
  line_d(o, "d", "ximat", d->ximat, 9 * nb);
  line_d(o, "d", "xanchor", d->xanchor, 3 * nj);
  line_d(o, "d", "xaxis", d->xaxis, 3 * nj);
  line_d(o, "d", "subtree_com", d->subtree_com, 3 * nb);
  line_d(o, "d", "cinert", d->cinert, 10 * nb);
  line_d(o, "d", "cdof", d->cdof, 6 * nv);
  line_d(o, "d", "crb", d->crb, 10 * nb);
  /* Python exposes the factorised length nC, not the allocated nM. */
  line_d(o, "d", "M", d->M, (int)m->nC);
  line_d(o, "d", "qLD", d->qLD, (int)m->nC);
  line_d(o, "d", "qLDiagInv", d->qLDiagInv, nv);
  line_d(o, "d", "cvel", d->cvel, 6 * nb);
  line_d(o, "d", "cdof_dot", d->cdof_dot, 6 * nv);
  line_d(o, "d", "qfrc_bias", d->qfrc_bias, nv);
  line_d(o, "d", "qfrc_spring", d->qfrc_spring, nv);
  line_d(o, "d", "qfrc_damper", d->qfrc_damper, nv);
  line_d(o, "d", "qfrc_gravcomp", d->qfrc_gravcomp, nv);
  line_d(o, "d", "qfrc_fluid", d->qfrc_fluid, nv);
  line_d(o, "d", "qfrc_passive", d->qfrc_passive, nv);
  line_d(o, "d", "ten_length", d->ten_length, (int)m->ntendon);
  line_d(o, "d", "ten_velocity", d->ten_velocity, (int)m->ntendon);
  line_d(o, "d", "actuator_length", d->actuator_length, nu);
  line_d(o, "d", "actuator_velocity", d->actuator_velocity, nu);
  line_d(o, "d", "actuator_force", d->actuator_force, nu);
  line_d(o, "d", "act_dot", d->act_dot, na);
  line_d(o, "d", "qfrc_actuator", d->qfrc_actuator, nv);
  line_d(o, "d", "qfrc_smooth", d->qfrc_smooth, nv);
  line_d(o, "d", "qacc_smooth", d->qacc_smooth, nv);
  line_i(o, "d", "efc_type", d->efc_type, ne);
  line_i(o, "d", "efc_id", d->efc_id, ne);
  line_d(o, "d", "efc_pos", d->efc_pos, ne);
  line_d(o, "d", "efc_margin", d->efc_margin, ne);
  line_d(o, "d", "efc_frictionloss", d->efc_frictionloss, ne);
  line_d(o, "d", "efc_diagA", d->efc_diagA, ne);
  line_d(o, "d", "efc_R", d->efc_R, ne);
  line_d(o, "d", "efc_D", d->efc_D, ne);
  line_d(o, "d", "efc_KBIP", d->efc_KBIP, 4 * ne);
  line_d(o, "d", "efc_vel", d->efc_vel, ne);
  line_d(o, "d", "efc_aref", d->efc_aref, ne);
  line_d(o, "d", "efc_b", d->efc_b, ne);
  line_d(o, "d", "efc_force", d->efc_force, ne);
  line_i(o, "d", "efc_state", d->efc_state, ne);
  line_d(o, "d", "qfrc_constraint", d->qfrc_constraint, nv);
  line_d(o, "d", "qacc", d->qacc, nv);
  int nvj = nv;
  double* J = calloc((size_t)(ne > 0 ? ne : 1) * (size_t)(nvj > 0 ? nvj : 1), sizeof(double));
  if (ne && nvj) {
    if (mj_isSparse(m))
      mju_sparse2dense(J, d->efc_J, ne, nvj, d->efc_J_rownnz, d->efc_J_rowadr, d->efc_J_colind);
    else
      memcpy(J, d->efc_J, (size_t)ne * (size_t)nvj * sizeof(double));
  }
  line_d(o, "d", "efc_J", J, ne * nvj);
  free(J);
  if (m->opt.noslip_iterations > 0 && !mj_isSparse(m))
    line_d(o, "d", "efc_AR", d->efc_AR, ne * ne);
  fprintf(o, "d\tnisland\t%d\n", d->nisland);
  int ns = d->nisland > 0 ? d->nisland : 1;
  line_i(o, "d", "solver_niter", d->solver_niter, ns);
}

static int replay_native(const char* root, const char* sched_path, const char* out_path) {
  int n = 0;
  Sched* ms = load_native_schedule(sched_path, &n);
  FILE* o = fopen(out_path, "w");
  if (!o) { perror(out_path); return 2; }
  char err[2048];
  for (int i = 0; i < n; i++) {
    char path[4096];
    snprintf(path, sizeof path, "%s/%s", root, ms[i].rel);
    mjModel* m = mj_loadXML(path, 0, err, sizeof err);
    if (!m) {
      fprintf(stderr, "load_fail %s %s\n", ms[i].rel, err);
      fclose(o);
      return 3;
    }
    fprintf(o, "model\t%s\n", ms[i].rel);
    emit_model(o, m);
    mjData* d = mj_makeData(m);
    for (int s = 0; s < ms[i].nstate; s++) {
      mj_resetData(m, d);
      if (ms[i].nq[s] != (int)m->nq || !ms[i].qvel[s] || ms[i].nv[s] != (int)m->nv) {
        fprintf(stderr, "state shape %s state %d nq %d/%d nv %d/%d\n", ms[i].rel, s, ms[i].nq[s], (int)m->nq, ms[i].nv[s], (int)m->nv);
        return 4;
      }
      memcpy(d->qpos, ms[i].qpos[s], (size_t)m->nq * sizeof(double));
      memcpy(d->qvel, ms[i].qvel[s], (size_t)m->nv * sizeof(double));
      if (m->nu && ms[i].ctrl[s]) memcpy(d->ctrl, ms[i].ctrl[s], (size_t)m->nu * sizeof(double));
      if (m->na && ms[i].act[s]) memcpy(d->act, ms[i].act[s], (size_t)m->na * sizeof(double));
      mj_forward(m, d);
      fprintf(o, "qpos\t"); space_d(o, d->qpos, (int)m->nq); fputc('\n', o);
      fprintf(o, "qvel\t"); space_d(o, d->qvel, (int)m->nv); fputc('\n', o);
      fprintf(o, "ctrl\t"); space_d(o, d->ctrl, (int)m->nu); fputc('\n', o);
      fprintf(o, "act\t"); space_d(o, d->act, (int)m->na); fputc('\n', o);
      emit_data(o, m, d);
    }
    mj_deleteData(d);
    mj_deleteModel(m);
  }
  fclose(o);
  fprintf(stderr, "native replay %d models -> %s\n", n, out_path);
  return 0;
}

static int replay_step(const char* root, const char* sched_path, const char* out_path, int nstep) {
  int n = 0;
  Sched* ms = load_native_schedule(sched_path, &n);
  FILE* o = fopen(out_path, "w");
  if (!o) { perror(out_path); return 2; }
  char err[2048];
  for (int i = 0; i < n; i++) {
    if (ms[i].nstate < 1) continue;
    char path[4096];
    snprintf(path, sizeof path, "%s/%s", root, ms[i].rel);
    mjModel* m = mj_loadXML(path, 0, err, sizeof err);
    if (!m) {
      fprintf(stderr, "load_fail %s %s\n", ms[i].rel, err);
      fclose(o);
      return 3;
    }
    mjData* d = mj_makeData(m);
    mj_resetData(m, d);
    memcpy(d->qpos, ms[i].qpos[0], (size_t)m->nq * sizeof(double));
    memcpy(d->qvel, ms[i].qvel[0], (size_t)m->nv * sizeof(double));
    if (m->nu && ms[i].ctrl[0]) memcpy(d->ctrl, ms[i].ctrl[0], (size_t)m->nu * sizeof(double));
    if (m->na && ms[i].act[0]) memcpy(d->act, ms[i].act[0], (size_t)m->na * sizeof(double));
    fprintf(o, "model\t%s\n", ms[i].rel);
    fprintf(o, "qpos\t"); space_d(o, d->qpos, (int)m->nq); fputc('\n', o);
    fprintf(o, "qvel\t"); space_d(o, d->qvel, (int)m->nv); fputc('\n', o);
    fprintf(o, "act\t"); space_d(o, d->act, (int)m->na); fputc('\n', o);
    fprintf(o, "ctrl\t"); space_d(o, d->ctrl, (int)m->nu); fputc('\n', o);
    for (int k = 1; k <= nstep; k++) {
      mj_step(m, d);
      fprintf(o, "s\t%d\n", k);
      fprintf(o, "qpos\t"); space_d(o, d->qpos, (int)m->nq); fputc('\n', o);
      fprintf(o, "qvel\t"); space_d(o, d->qvel, (int)m->nv); fputc('\n', o);
      fprintf(o, "act\t"); space_d(o, d->act, (int)m->na); fputc('\n', o);
      fprintf(o, "warm\t"); space_d(o, d->qacc_warmstart, (int)m->nv); fputc('\n', o);
    }
    mj_deleteData(d);
    mj_deleteModel(m);
  }
  fclose(o);
  fprintf(stderr, "step replay %d models -> %s\n", n, out_path);
  return 0;
}

static int ends_xml(const char* s) {
  size_t n = strlen(s);
  return n > 4 && strcmp(s + n - 4, ".xml") == 0;
}
static int cmpstr(const void* a, const void* b) {
  return strcmp(*(char* const*)a, *(char* const*)b);
}
static int classify(const char* root, const char* out_path) {
  DIR* top = opendir(root);
  if (!top) { perror(root); return 2; }
  char** rels = NULL;
  int n = 0, cap = 0;
  struct dirent* ent;
  while ((ent = readdir(top))) {
    if (ent->d_name[0] == '.') continue;
    char dirpath[4096];
    snprintf(dirpath, sizeof dirpath, "%s/%s", root, ent->d_name);
    struct stat st;
    if (stat(dirpath, &st) != 0 || !S_ISDIR(st.st_mode)) continue;
    DIR* sub = opendir(dirpath);
    if (!sub) continue;
    struct dirent* xml;
    while ((xml = readdir(sub))) {
      if (!ends_xml(xml->d_name)) continue;
      if (n == cap) { cap = cap ? cap * 2 : 64; rels = realloc(rels, (size_t)cap * sizeof(char*)); }
      char rel[4096];
      snprintf(rel, sizeof rel, "%s/%s", ent->d_name, xml->d_name);
      rels[n++] = xstrdup(rel);
    }
    closedir(sub);
  }
  closedir(top);
  qsort(rels, (size_t)n, sizeof(char*), cmpstr);
  FILE* o = fopen(out_path, "w");
  if (!o) { perror(out_path); return 2; }
  int ok = 0, bad = 0;
  char err[2048];
  for (int i = 0; i < n; i++) {
    char path[4096];
    snprintf(path, sizeof path, "%s/%s", root, rels[i]);
    mjModel* m = mj_loadXML(path, 0, err, sizeof err);
    if (!m) {
      /* one line, first error line only */
      char* nl = strchr(err, '\n');
      if (nl) *nl = 0;
      fprintf(o, "fail\t%s\t%s\n", rels[i], err);
      bad++;
    } else {
      fprintf(o, "ok\t%s\n", rels[i]);
      ok++;
      mj_deleteModel(m);
    }
    free(rels[i]);
  }
  free(rels);
  fclose(o);
  fprintf(stderr, "classify xml %d ok %d fail %d\n", n, ok, bad);
  return 0;
}

int main(int argc, char** argv) {
  if (argc >= 2 && strcmp(argv[1], "classify") == 0) {
    if (argc < 4) {
      fprintf(stderr, "usage: oracle_dump classify <menagerie> <out>\n");
      return 2;
    }
    const char* plug = getenv("MUJOCO_PLUGIN_DIR");
    if (plug && plug[0]) mj_loadAllPluginLibraries(plug, NULL);
    return classify(argv[2], argv[3]);
  }
  if (argc < 5) {
    fprintf(stderr, "usage: oracle_dump native|step <menagerie> <schedule> <out> [steps]\n");
    return 2;
  }
  const char* plug = getenv("MUJOCO_PLUGIN_DIR");
  if (plug && plug[0]) mj_loadAllPluginLibraries(plug, NULL);
  if (strcmp(argv[1], "native") == 0) return replay_native(argv[2], argv[3], argv[4]);
  if (strcmp(argv[1], "step") == 0) {
    int nstep = argc > 5 ? atoi(argv[5]) : 20;
    return replay_step(argv[2], argv[3], argv[4], nstep);
  }
  fprintf(stderr, "unknown mode %s\n", argv[1]);
  return 2;
}
