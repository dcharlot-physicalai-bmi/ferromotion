/* Versioned Menagerie dump for one libmujoco.
 *
 * Schema ferromotion.mujoco-dump.v1 is the line layout of
 * scripts/mujoco_native_oracle.py and scripts/mujoco_native_step_oracle.py.
 * That is the 3.13.0 contract-off dump. Compiling this file against a
 * 3.13.0 (or any mjNEQDATA == 11, pre-4.0) header does not emit the later
 * fields, so a 3.13 dump stays byte-identical.
 *
 * v2 (mjNEQDATA > 11, MuJoCo 3.17) appends m eq_data at width mjNEQDATA.
 * v3 (mjVERSION_HEADER >= 4000000) also appends m tendon_type and, on each
 * state, d xiquat, d geom_xquat, d site_xquat, d contact_surfacevel.
 * Appended lines are ignored by the 3.13 ferromotion readers.
 */
#include <mujoco/mujoco.h>

#include <ctype.h>
#include <dirent.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#if mjNEQDATA > 11
#define ORACLE_DUMP_SCHEMA "ferromotion.mujoco-dump.v2"
#define ORACLE_DUMP_EQ_DATA 1
#else
#define ORACLE_DUMP_SCHEMA "ferromotion.mujoco-dump.v1"
#define ORACLE_DUMP_EQ_DATA 0
#endif
#if mjVERSION_HEADER >= 4000000
#undef ORACLE_DUMP_SCHEMA
#define ORACLE_DUMP_SCHEMA "ferromotion.mujoco-dump.v3"
#define ORACLE_DUMP_V3 1
#else
#define ORACLE_DUMP_V3 0
#endif

static FILE* OUT;

static void pyhex(FILE* o, double x) {
  uint64_t u;
  memcpy(&u, &x, 8);
  int neg = (int)(u >> 63);
  int be = (int)((u >> 52) & 0x7ff);
  uint64_t frac = u & 0xfffffffffffffULL;
  if (be == 0x7ff) {
    fprintf(stderr, "refusing: NaN or Inf in an oracle field\n");
    exit(3);
  }
  if (neg) fputc('-', o);
  if (be == 0 && frac == 0) {
    fputs("0x0.0p+0", o);
    return;
  }
  if (be == 0) {
    fprintf(o, "0x0.%013llxp-1022", (unsigned long long)frac);
    return;
  }
  fprintf(o, "0x1.%013llxp%+d", (unsigned long long)frac, be - 1023);
}

static void line_d(const char* k, const char* name, const mjtNum* p, int n) {
  fprintf(OUT, "%s\t%s\t", k, name);
  for (int i = 0; i < n; i++) {
    if (i) fputc(' ', OUT);
    pyhex(OUT, (double)p[i]);
  }
  fputc('\n', OUT);
}

static void line_i(const char* k, const char* name, const int* p, int n) {
  fprintf(OUT, "%s\t%s\t", k, name);
  for (int i = 0; i < n; i++) fprintf(OUT, "%s%d", i ? " " : "", p[i]);
  fputc('\n', OUT);
}

static void line_b(const char* k, const char* name, const mjtByte* p, int n) {
  fprintf(OUT, "%s\t%s\t", k, name);
  for (int i = 0; i < n; i++) fprintf(OUT, "%s%d", i ? " " : "", (int)p[i]);
  fputc('\n', OUT);
}

static int fit_size(mjtSize n, const char* what) {
  if (n < 0 || n > 2000000000) {
    fprintf(stderr, "refusing: %s size %lld\n", what, (long long)n);
    exit(3);
  }
  return (int)n;
}

static int parse_one(const char** ps, double* out) {
  const char* s = *ps;
  while (*s == ' ' || *s == '\t') s++;
  if (!*s) return 0;
  int neg = 0;
  if (*s == '+') s++;
  else if (*s == '-') {
    neg = 1;
    s++;
  }
  if (!(s[0] == '0' && (s[1] == 'x' || s[1] == 'X'))) return -1;
  s += 2;
  const char* pexp = strchr(s, 'p');
  if (!pexp) pexp = strchr(s, 'P');
  if (!pexp) return -1;
  const char* dot = NULL;
  for (const char* q = s; q < pexp; q++)
    if (*q == '.') dot = q;
  if (!dot) return -1;
  int lead = 0;
  if (dot == s || dot - s != 1 || !isxdigit((unsigned char)*s)) return -1;
  lead = (*s <= '9') ? (*s - '0') : (tolower((unsigned char)*s) - 'a' + 10);
  char frac[16];
  int flen = 0;
  for (const char* q = dot + 1; q < pexp; q++) {
    if (!isxdigit((unsigned char)*q) || flen >= 13) return -1;
    frac[flen++] = *q;
  }
  while (flen < 13) frac[flen++] = '0';
  frac[13] = 0;
  uint64_t fbits = strtoull(frac, NULL, 16);
  char* end = NULL;
  long exp = strtol(pexp + 1, &end, 10);
  if (end == pexp + 1) return -1;
  uint64_t bits;
  if (lead == 0 && fbits == 0) bits = 0;
  else if (lead == 0) bits = fbits;
  else bits = ((uint64_t)(exp + 1023) << 52) | fbits;
  if (neg) bits |= 1ULL << 63;
  memcpy(out, &bits, 8);
  *ps = end;
  return 1;
}

static double* parse_row(const char* s, int* n_out) {
  int cap = 16, n = 0;
  double* a = malloc((size_t)cap * sizeof(double));
  if (!a) exit(3);
  while (*s) {
    double x;
    int rc = parse_one(&s, &x);
    if (rc == 0) break;
    if (rc < 0) {
      free(a);
      return NULL;
    }
    if (n == cap) {
      cap *= 2;
      a = realloc(a, (size_t)cap * sizeof(double));
      if (!a) exit(3);
    }
    a[n++] = x;
  }
  *n_out = n;
  return a;
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
  if (!p) exit(3);
  memcpy(p, s, n + 1);
  return p;
}

static void sched_add_state(Sched* sc, double* q, int nq, double* v, int nv, double* u, int nu, double* a, int na) {
  int i = sc->nstate++;
  sc->qpos = realloc(sc->qpos, (size_t)sc->nstate * sizeof(double*));
  sc->qvel = realloc(sc->qvel, (size_t)sc->nstate * sizeof(double*));
  sc->ctrl = realloc(sc->ctrl, (size_t)sc->nstate * sizeof(double*));
  sc->act = realloc(sc->act, (size_t)sc->nstate * sizeof(double*));
  sc->nq = realloc(sc->nq, (size_t)sc->nstate * sizeof(int));
  sc->nv = realloc(sc->nv, (size_t)sc->nstate * sizeof(int));
  sc->nu = realloc(sc->nu, (size_t)sc->nstate * sizeof(int));
  sc->na = realloc(sc->na, (size_t)sc->nstate * sizeof(int));
  sc->qpos[i] = q;
  sc->qvel[i] = v;
  sc->ctrl[i] = u;
  sc->act[i] = a;
  sc->nq[i] = nq;
  sc->nv[i] = nv;
  sc->nu[i] = nu;
  sc->na[i] = na;
}

/* Schedule lines, in order, per model: qpos, qvel, ctrl, act. Repeated per state. */
static Sched* load_schedule(const char* path, int* n_out) {
  FILE* f = fopen(path, "r");
  if (!f) {
    perror(path);
    exit(2);
  }
  int cap = 16, n = 0;
  Sched* sc = calloc((size_t)cap, sizeof(Sched));
  char* line = NULL;
  size_t len = 0;
  double* q = NULL;
  double* v = NULL;
  double* u = NULL;
  double* a = NULL;
  int nq = 0, nv = 0, nu = 0, na = 0;
  int have_q = 0, have_v = 0, have_u = 0, have_a = 0;
  while (getline(&line, &len, f) != -1) {
    char* tab = strchr(line, '\t');
    if (!tab) continue;
    *tab = 0;
    char* rest = tab + 1;
    size_t L = strlen(rest);
    while (L && (rest[L - 1] == '\n' || rest[L - 1] == '\r')) rest[--L] = 0;
    if (strcmp(line, "model") == 0) {
      if (n == cap) {
        cap *= 2;
        sc = realloc(sc, (size_t)cap * sizeof(Sched));
      }
      memset(&sc[n], 0, sizeof(Sched));
      sc[n].rel = xstrdup(rest);
      n++;
      have_q = have_v = have_u = have_a = 0;
      continue;
    }
    if (n == 0) continue;
    double* row = NULL;
    int nr = 0;
    if (strcmp(line, "qpos") == 0 || strcmp(line, "qvel") == 0 || strcmp(line, "ctrl") == 0 || strcmp(line, "act") == 0) {
      row = parse_row(rest, &nr);
      if (!row) {
        fprintf(stderr, "bad hex on %s in %s\n", sc[n - 1].rel, path);
        exit(3);
      }
    } else {
      continue;
    }
    Sched* cur = &sc[n - 1];
    if (strcmp(line, "qpos") == 0) {
      q = row;
      nq = nr;
      have_q = 1;
    } else if (strcmp(line, "qvel") == 0) {
      v = row;
      nv = nr;
      have_v = 1;
    } else if (strcmp(line, "ctrl") == 0) {
      u = row;
      nu = nr;
      have_u = 1;
    } else if (strcmp(line, "act") == 0) {
      a = row;
      na = nr;
      have_a = 1;
    }
    if (have_q && have_v && have_u && have_a) {
      sched_add_state(cur, q, nq, v, nv, u, nu, a, na);
      have_q = have_v = have_u = have_a = 0;
    }
  }
  free(line);
  fclose(f);
  *n_out = n;
  return sc;
}

typedef struct {
  char* rel;
  char* path;
} Item;

static int cmp_item(const void* a, const void* b) {
  return strcmp(((const Item*)a)->rel, ((const Item*)b)->rel);
}

static Item* list_models(const char* root, int* n_out) {
  DIR* top = opendir(root);
  if (!top) {
    perror(root);
    exit(2);
  }
  int cap = 64, n = 0;
  Item* items = malloc((size_t)cap * sizeof(Item));
  struct dirent* ent;
  while ((ent = readdir(top))) {
    if (ent->d_name[0] == '.') continue;
    char dirpath[4096];
    snprintf(dirpath, sizeof(dirpath), "%s/%s", root, ent->d_name);
    struct stat st;
    if (stat(dirpath, &st) != 0 || !S_ISDIR(st.st_mode)) continue;
    DIR* sub = opendir(dirpath);
    if (!sub) continue;
    struct dirent* xml;
    while ((xml = readdir(sub))) {
      size_t ln = strlen(xml->d_name);
      if (ln < 5 || strcmp(xml->d_name + ln - 4, ".xml") != 0) continue;
      if (xml->d_name[0] == '.') continue;
      if (n == cap) {
        cap *= 2;
        items = realloc(items, (size_t)cap * sizeof(Item));
      }
      char rel[4096], path[4096];
      snprintf(rel, sizeof(rel), "%s/%s", ent->d_name, xml->d_name);
      snprintf(path, sizeof(path), "%s/%s", dirpath, xml->d_name);
      items[n].rel = xstrdup(rel);
      items[n].path = xstrdup(path);
      n++;
    }
    closedir(sub);
  }
  closedir(top);
  qsort(items, (size_t)n, sizeof(Item), cmp_item);
  *n_out = n;
  return items;
}

static void load_plugins(const char* dir) {
  if (!dir || !dir[0] || strcmp(dir, "-") == 0) return;
  const char* names[] = {"libactuator.so", "libelasticity.so", "libsensor.so", "libsdf_plugin.so", NULL};
  for (int i = 0; names[i]; i++) {
    char path[4096];
    snprintf(path, sizeof(path), "%s/%s", dir, names[i]);
    struct stat st;
    if (stat(path, &st) != 0) continue;
    mj_loadPluginLibrary(path);
  }
}

static void dump_model_v1(const mjModel* m) {
  line_i("m", "body_parentid", m->body_parentid, m->nbody);
  line_i("m", "body_rootid", m->body_rootid, m->nbody);
  line_i("m", "body_weldid", m->body_weldid, m->nbody);
  line_i("m", "body_jntnum", m->body_jntnum, m->nbody);
  line_i("m", "body_jntadr", m->body_jntadr, m->nbody);
  line_i("m", "body_dofnum", m->body_dofnum, m->nbody);
  line_i("m", "body_dofadr", m->body_dofadr, m->nbody);
  line_b("m", "body_simple", m->body_simple, m->nbody);
  line_d("m", "body_mass", m->body_mass, m->nbody);
  line_d("m", "body_subtreemass", m->body_subtreemass, m->nbody);
  line_d("m", "body_inertia", m->body_inertia, m->nbody * 3);
  line_d("m", "body_ipos", m->body_ipos, m->nbody * 3);
  line_d("m", "body_iquat", m->body_iquat, m->nbody * 4);
  line_d("m", "body_pos", m->body_pos, m->nbody * 3);
  line_d("m", "body_quat", m->body_quat, m->nbody * 4);
  line_i("m", "jnt_type", m->jnt_type, m->njnt);
  line_i("m", "jnt_qposadr", m->jnt_qposadr, m->njnt);
  line_i("m", "jnt_dofadr", m->jnt_dofadr, m->njnt);
  line_i("m", "jnt_bodyid", m->jnt_bodyid, m->njnt);
  line_d("m", "jnt_pos", m->jnt_pos, m->njnt * 3);
  line_d("m", "jnt_axis", m->jnt_axis, m->njnt * 3);
  line_i("m", "dof_bodyid", m->dof_bodyid, m->nv);
  line_i("m", "dof_jntid", m->dof_jntid, m->nv);
  line_i("m", "dof_parentid", m->dof_parentid, m->nv);
  line_i("m", "dof_simplenum", m->dof_simplenum, m->nv);
  line_d("m", "dof_M0", m->dof_M0, m->nv);
  line_d("m", "dof_armature", m->dof_armature, m->nv);
  line_d("m", "dof_damping", m->dof_damping, m->nv);
  line_i("m", "M_rownnz", m->M_rownnz, m->nv);
  line_i("m", "M_rowadr", m->M_rowadr, m->nv);
  line_i("m", "M_colind", m->M_colind, fit_size(m->nC, "nC"));
  line_d("m", "qpos0", m->qpos0, m->nq);
  line_d("m", "body_invweight0", m->body_invweight0, m->nbody * 2);
  line_d("m", "dof_invweight0", m->dof_invweight0, m->nv);
  line_d("m", "tendon_invweight0", m->tendon_invweight0, m->ntendon);
  line_d("m", "gravity", m->opt.gravity, 3);
#if ORACLE_DUMP_EQ_DATA
  line_d("m", "eq_data", m->eq_data, m->neq * mjNEQDATA);
#endif
#if ORACLE_DUMP_V3
  line_i("m", "tendon_type", m->tendon_type, m->ntendon);
#endif
}

static void dump_data_v1(const mjModel* m, mjData* d) {
  line_d("d", "xpos", d->xpos, m->nbody * 3);
  line_d("d", "xquat", d->xquat, m->nbody * 4);
  line_d("d", "xmat", d->xmat, m->nbody * 9);
  line_d("d", "xipos", d->xipos, m->nbody * 3);
  line_d("d", "ximat", d->ximat, m->nbody * 9);
  line_d("d", "xanchor", d->xanchor, m->njnt * 3);
  line_d("d", "xaxis", d->xaxis, m->njnt * 3);
  line_d("d", "subtree_com", d->subtree_com, m->nbody * 3);
  line_d("d", "cinert", d->cinert, m->nbody * 10);
  line_d("d", "cdof", d->cdof, m->nv * 6);
  line_d("d", "crb", d->crb, m->nbody * 10);
  line_d("d", "M", d->M, fit_size(m->nC, "M"));
  line_d("d", "qLD", d->qLD, fit_size(m->nC, "qLD"));
  line_d("d", "qLDiagInv", d->qLDiagInv, m->nv);
  line_d("d", "cvel", d->cvel, m->nbody * 6);
  line_d("d", "cdof_dot", d->cdof_dot, m->nv * 6);
  line_d("d", "qfrc_bias", d->qfrc_bias, m->nv);
  line_d("d", "qfrc_spring", d->qfrc_spring, m->nv);
  line_d("d", "qfrc_damper", d->qfrc_damper, m->nv);
  line_d("d", "qfrc_gravcomp", d->qfrc_gravcomp, m->nv);
  line_d("d", "qfrc_fluid", d->qfrc_fluid, m->nv);
  line_d("d", "qfrc_passive", d->qfrc_passive, m->nv);
  line_d("d", "ten_length", d->ten_length, m->ntendon);
  line_d("d", "ten_velocity", d->ten_velocity, m->ntendon);
  line_d("d", "actuator_length", d->actuator_length, fit_size(m->nout, "nout"));
  line_d("d", "actuator_velocity", d->actuator_velocity, fit_size(m->nout, "nout"));
  line_d("d", "actuator_force", d->actuator_force, fit_size(m->nout, "nout"));
  line_d("d", "act_dot", d->act_dot, m->na);
  line_d("d", "qfrc_actuator", d->qfrc_actuator, m->nv);
  line_d("d", "qfrc_smooth", d->qfrc_smooth, m->nv);
  line_d("d", "qacc_smooth", d->qacc_smooth, m->nv);
  line_i("d", "efc_type", d->efc_type, d->nefc);
  line_i("d", "efc_id", d->efc_id, d->nefc);
  line_d("d", "efc_pos", d->efc_pos, d->nefc);
  line_d("d", "efc_margin", d->efc_margin, d->nefc);
  line_d("d", "efc_frictionloss", d->efc_frictionloss, d->nefc);
  line_d("d", "efc_diagA", d->efc_diagA, d->nefc);
  line_d("d", "efc_R", d->efc_R, d->nefc);
  line_d("d", "efc_D", d->efc_D, d->nefc);
  line_d("d", "efc_KBIP", d->efc_KBIP, d->nefc * 4);
  line_d("d", "efc_vel", d->efc_vel, d->nefc);
  line_d("d", "efc_aref", d->efc_aref, d->nefc);
  line_d("d", "efc_b", d->efc_b, d->nefc);
  line_d("d", "efc_force", d->efc_force, d->nefc);
  line_i("d", "efc_state", d->efc_state, d->nefc);
  line_d("d", "qfrc_constraint", d->qfrc_constraint, m->nv);
  line_d("d", "qacc", d->qacc, m->nv);
  /* Dense efc_J, the same conversion as scripts/mujoco_native_oracle.py. */
  {
    int rows = d->nefc;
    int cols = m->nv;
    long cells = (long)rows * (long)cols;
    mjtNum* J = NULL;
    if (cells > 0) {
      J = calloc((size_t)cells, sizeof(mjtNum));
      if (!J) exit(3);
      if (mj_isSparse(m)) {
        mju_sparse2dense(J, d->efc_J, rows, cols, d->efc_J_rownnz, d->efc_J_rowadr, d->efc_J_colind);
      } else {
        memcpy(J, d->efc_J, (size_t)cells * sizeof(mjtNum));
      }
    }
    line_d("d", "efc_J", J, (int)cells);
    free(J);
  }
  if (m->opt.noslip_iterations > 0 && !mj_isSparse(m)) {
    line_d("d", "efc_AR", d->efc_AR, d->nefc * d->nefc);
  }
  fprintf(OUT, "d\tnisland\t%d\n", d->nisland);
  {
    int niter = d->nisland > 0 ? d->nisland : 1;
    line_i("d", "solver_niter", d->solver_niter, niter);
  }
#if ORACLE_DUMP_V3
  line_d("d", "xiquat", d->xiquat, m->nbody * 4);
  line_d("d", "geom_xquat", d->geom_xquat, m->ngeom * 4);
  line_d("d", "site_xquat", d->site_xquat, m->nsite * 4);
  {
    int n = d->ncon * 3;
    mjtNum* sv = NULL;
    if (n > 0) {
      sv = malloc((size_t)n * sizeof(mjtNum));
      if (!sv) exit(3);
      for (int i = 0; i < d->ncon; i++) {
        sv[3 * i] = d->contact[i].surfacevel[0];
        sv[3 * i + 1] = d->contact[i].surfacevel[1];
        sv[3 * i + 2] = d->contact[i].surfacevel[2];
      }
    }
    line_d("d", "contact_surfacevel", sv, n);
    free(sv);
  }
#endif
}

static void emit_vec(const char* key, const double* p, int n) {
  fprintf(OUT, "%s\t", key);
  for (int i = 0; i < n; i++) {
    if (i) fputc(' ', OUT);
    pyhex(OUT, p[i]);
  }
  fputc('\n', OUT);
}

static void apply_state(const mjModel* m, mjData* d, const double* q, int nq, const double* v, int nv, const double* u, int nu, const double* a, int na) {
  if (nq != m->nq || nv != m->nv || nu != m->nu || na != m->na) {
    fprintf(stderr, "state size q %d/%ld v %d/%ld u %d/%ld a %d/%ld\n", nq, (long)m->nq, nv, (long)m->nv, nu, (long)m->nu, na, (long)m->na);
    exit(3);
  }
  mj_resetData(m, d);
  if (nq) memcpy(d->qpos, q, (size_t)nq * sizeof(mjtNum));
  if (nv) memcpy(d->qvel, v, (size_t)nv * sizeof(mjtNum));
  if (nu) memcpy(d->ctrl, u, (size_t)nu * sizeof(mjtNum));
  if (na) memcpy(d->act, a, (size_t)na * sizeof(mjtNum));
}

static void cmd_classify(const char* root, const char* plugins, const char* out_path) {
  load_plugins(plugins);
  OUT = fopen(out_path, "w");
  if (!OUT) {
    perror(out_path);
    exit(2);
  }
  fprintf(OUT, "schema\tferromotion.mujoco-compile.v1\n");
  fprintf(OUT, "dump_schema\t%s\n", ORACLE_DUMP_SCHEMA);
  fprintf(OUT, "version\t%s\n", mj_versionString());
  fprintf(OUT, "header\t%d\n", mjVERSION_HEADER);
  fprintf(OUT, "neqdata\t%d\n", mjNEQDATA);
  int n = 0;
  Item* items = list_models(root, &n);
  int loads = 0, fails = 0;
  for (int i = 0; i < n; i++) {
    char err[65536];
    err[0] = 0;
    mjModel* m = mj_loadXML(items[i].path, NULL, err, (int)sizeof(err));
    if (!m) {
      fails++;
      /* Length-prefixed verbatim compiler error. Bytes after the newline are exact. */
      size_t elen = strlen(err);
      fprintf(OUT, "fail\t%s\t%zu\n", items[i].rel, elen);
      fwrite(err, 1, elen, OUT);
      fputc('\n', OUT);
      continue;
    }
    loads++;
    fprintf(OUT, "ok\t%s\tnbody\t%ld\tnjnt\t%ld\tngeom\t%ld\tnq\t%ld\tnv\t%ld\tnu\t%ld\tneq\t%ld\tntendon\t%ld\tnsensor\t%ld\n",
            items[i].rel, (long)m->nbody, (long)m->njnt, (long)m->ngeom, (long)m->nq, (long)m->nv, (long)m->nu, (long)m->neq, (long)m->ntendon, (long)m->nsensor);
    mj_deleteModel(m);
  }
  fprintf(OUT, "summary\txml\t%d\tloads\t%d\tfails\t%d\n", n, loads, fails);
  fclose(OUT);
  fprintf(stderr, "classify %s xml %d loads %d fails %d\n", ORACLE_DUMP_SCHEMA, n, loads, fails);
}

static void cmd_props(const char* root, const char* plugins, const char* out_path) {
  load_plugins(plugins);
  OUT = fopen(out_path, "w");
  if (!OUT) {
    perror(out_path);
    exit(2);
  }
  int n = 0;
  Item* items = list_models(root, &n);
  int loads = 0;
  for (int i = 0; i < n; i++) {
    char err[65536];
    err[0] = 0;
    mjModel* m = mj_loadXML(items[i].path, NULL, err, (int)sizeof(err));
    if (!m) continue;
    loads++;
    fprintf(OUT, "model\t%s\n", items[i].rel);
    fprintf(OUT, "nq\t%ld\n", (long)m->nq);
    fprintf(OUT, "nv\t%ld\n", (long)m->nv);
    fprintf(OUT, "nu\t%ld\n", (long)m->nu);
    fprintf(OUT, "na\t%ld\n", (long)m->na);
    fprintf(OUT, "njnt\t%ld\n", (long)m->njnt);
    emit_vec("qpos0", m->qpos0, m->nq);
    for (int j = 0; j < m->njnt; j++) {
      fprintf(OUT, "jnt\t%d\t%d\t%d\t", m->jnt_type[j], m->jnt_qposadr[j], (int)m->jnt_limited[j]);
      pyhex(OUT, m->jnt_range[2 * j]);
      fputc(' ', OUT);
      pyhex(OUT, m->jnt_range[2 * j + 1]);
      fputc('\n', OUT);
    }
    for (int a = 0; a < m->nu; a++) {
      fprintf(OUT, "actlim\t%d\t", (int)m->actuator_ctrllimited[a]);
      pyhex(OUT, m->actuator_ctrlrange[2 * a]);
      fputc(' ', OUT);
      pyhex(OUT, m->actuator_ctrlrange[2 * a + 1]);
      fputc('\n', OUT);
    }
    mj_deleteModel(m);
  }
  fclose(OUT);
  fprintf(stderr, "props loads %d of %d\n", loads, n);
}

static void cmd_native(const char* root, const char* plugins, const char* schedule, const char* out_path) {
  load_plugins(plugins);
  int ns = 0;
  Sched* sc = load_schedule(schedule, &ns);
  OUT = fopen(out_path, "w");
  if (!OUT) {
    perror(out_path);
    exit(2);
  }
  for (int i = 0; i < ns; i++) {
    char path[4096];
    snprintf(path, sizeof(path), "%s/%s", root, sc[i].rel);
    char err[65536];
    err[0] = 0;
    mjModel* m = mj_loadXML(path, NULL, err, (int)sizeof(err));
    fprintf(OUT, "model\t%s\n", sc[i].rel);
    if (!m) {
      fprintf(OUT, "load_error\t%s\n", err);
      continue;
    }
    dump_model_v1(m);
    mjData* d = mj_makeData(m);
    for (int s = 0; s < sc[i].nstate; s++) {
      if (sc[i].nq[s] != m->nq || sc[i].nv[s] != m->nv || sc[i].nu[s] != m->nu || sc[i].na[s] != m->na) {
        fprintf(OUT, "topology\tnq\t%d\t%ld\tnv\t%d\t%ld\tnu\t%d\t%ld\tna\t%d\t%ld\n",
                sc[i].nq[s], (long)m->nq, sc[i].nv[s], (long)m->nv, sc[i].nu[s], (long)m->nu, sc[i].na[s], (long)m->na);
        continue;
      }
      apply_state(m, d, sc[i].qpos[s], sc[i].nq[s], sc[i].qvel[s], sc[i].nv[s], sc[i].ctrl[s], sc[i].nu[s], sc[i].act[s], sc[i].na[s]);
      mj_forward(m, d);
      emit_vec("qpos", sc[i].qpos[s], sc[i].nq[s]);
      emit_vec("qvel", sc[i].qvel[s], sc[i].nv[s]);
      emit_vec("ctrl", sc[i].ctrl[s], sc[i].nu[s]);
      emit_vec("act", sc[i].act[s], sc[i].na[s]);
      dump_data_v1(m, d);
    }
    mj_deleteData(d);
    mj_deleteModel(m);
  }
  fclose(OUT);
  fprintf(stderr, "native %s models %d\n", ORACLE_DUMP_SCHEMA, ns);
}

static void cmd_step(const char* root, const char* plugins, const char* schedule, const char* out_path) {
  load_plugins(plugins);
  int ns = 0;
  Sched* sc = load_schedule(schedule, &ns);
  OUT = fopen(out_path, "w");
  if (!OUT) {
    perror(out_path);
    exit(2);
  }
  const int N = 20;
  for (int i = 0; i < ns; i++) {
    char path[4096];
    snprintf(path, sizeof(path), "%s/%s", root, sc[i].rel);
    char err[65536];
    err[0] = 0;
    mjModel* m = mj_loadXML(path, NULL, err, (int)sizeof(err));
    fprintf(OUT, "model\t%s\n", sc[i].rel);
    if (!m) {
      fprintf(OUT, "load_error\t%s\n", err);
      continue;
    }
    if (sc[i].nstate != 1) {
      fprintf(stderr, "step schedule %s has %d states, want 1\n", sc[i].rel, sc[i].nstate);
      exit(3);
    }
    if (sc[i].nq[0] != m->nq || sc[i].nv[0] != m->nv || sc[i].nu[0] != m->nu || sc[i].na[0] != m->na) {
      fprintf(OUT, "topology\tnq\t%d\t%ld\tnv\t%d\t%ld\tnu\t%d\t%ld\tna\t%d\t%ld\n",
              sc[i].nq[0], (long)m->nq, sc[i].nv[0], (long)m->nv, sc[i].nu[0], (long)m->nu, sc[i].na[0], (long)m->na);
      mj_deleteModel(m);
      continue;
    }
    mjData* d = mj_makeData(m);
    apply_state(m, d, sc[i].qpos[0], sc[i].nq[0], sc[i].qvel[0], sc[i].nv[0], sc[i].ctrl[0], sc[i].nu[0], sc[i].act[0], sc[i].na[0]);
    emit_vec("qpos", d->qpos, m->nq);
    emit_vec("qvel", d->qvel, m->nv);
    emit_vec("act", d->act, m->na);
    emit_vec("ctrl", d->ctrl, m->nu);
    for (int k = 1; k <= N; k++) {
      mj_step(m, d);
      fprintf(OUT, "s\t%d\n", k);
      emit_vec("qpos", d->qpos, m->nq);
      emit_vec("qvel", d->qvel, m->nv);
      emit_vec("act", d->act, m->na);
      emit_vec("warm", d->qacc_warmstart, m->nv);
    }
    mj_deleteData(d);
    mj_deleteModel(m);
  }
  fclose(OUT);
  fprintf(stderr, "step %s models %d steps %d\n", ORACLE_DUMP_SCHEMA, ns, N);
}

static void usage(void) {
  fprintf(stderr,
          "usage: oracle_schema_dump schema\n"
          "       oracle_schema_dump classify <menagerie> <plugin-dir|-> <out>\n"
          "       oracle_schema_dump props <menagerie> <plugin-dir|-> <out>\n"
          "       oracle_schema_dump native <menagerie> <plugin-dir|-> <schedule> <out>\n"
          "       oracle_schema_dump step <menagerie> <plugin-dir|-> <schedule> <out>\n");
}

int main(int argc, char** argv) {
  if (argc < 2) {
    usage();
    return 2;
  }
  if (strcmp(argv[1], "schema") == 0) {
    printf("dump_schema\t%s\nversion\t%s\nheader\t%d\nneqdata\t%d\n", ORACLE_DUMP_SCHEMA, mj_versionString(), mjVERSION_HEADER, mjNEQDATA);
    return 0;
  }
  if (strcmp(argv[1], "classify") == 0 && argc == 5) {
    cmd_classify(argv[2], argv[3], argv[4]);
    return 0;
  }
  if (strcmp(argv[1], "props") == 0 && argc == 5) {
    cmd_props(argv[2], argv[3], argv[4]);
    return 0;
  }
  if (strcmp(argv[1], "native") == 0 && argc == 6) {
    cmd_native(argv[2], argv[3], argv[4], argv[5]);
    return 0;
  }
  if (strcmp(argv[1], "step") == 0 && argc == 6) {
    cmd_step(argv[2], argv[3], argv[4], argv[5]);
    return 0;
  }
  usage();
  return 2;
}
