/* Step the ferromotion pid fixtures against one libmujoco and its libactuator.
 * Prints bit patterns. Does not decide a pass.
 *
 * argv: plugin <libactuator.so>
 *       builtin <libactuator.so>   (no plugin; <pid> with no kp)
 */
#include <mujoco/mujoco.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static void bits(const char* name, const mjtNum* p, int n) {
  printf("%s", name);
  for (int i = 0; i < n; i++) {
    uint64_t u;
    double x = (double)p[i];
    memcpy(&u, &x, 8);
    printf(" %016llx", (unsigned long long)u);
  }
  printf("\n");
}

static const char* PLUGIN_XML =
    "<mujoco><option timestep=\"0.002\" integrator=\"implicitfast\"/>"
    "<extension><plugin plugin=\"mujoco.pid\"><instance name=\"pid1\">"
    "<config key=\"kp\" value=\"7.3\"/><config key=\"ki\" value=\"15\"/>"
    "<config key=\"kd\" value=\"1.7\"/><config key=\"imax\" value=\"3\"/>"
    "<config key=\"slewmax\" value=\"4\"/></instance></plugin></extension>"
    "<worldbody><body name=\"a\" pos=\"0 0 1\"><joint name=\"h1\" type=\"hinge\" axis=\"0 1 0\" damping=\"0.05\"/>"
    "<geom type=\"capsule\" fromto=\"0 0 0 0.3 0 0\" size=\"0.03\"/>"
    "<body name=\"b\" pos=\"0.3 0 0\"><joint name=\"h2\" type=\"hinge\" axis=\"0 1 0\"/>"
    "<geom type=\"capsule\" fromto=\"0 0 0 0.3 0 0\" size=\"0.03\"/></body></body></worldbody>"
    "<actuator><plugin joint=\"h1\" plugin=\"mujoco.pid\" instance=\"pid1\" actdim=\"2\" ctrlrange=\"-1 1\" forcerange=\"-20 20\"/>"
    "<position joint=\"h2\" kp=\"10\" kv=\"0.5\"/></actuator></mujoco>";

static const char* BARE_XML =
    "<mujoco><option timestep=\"0.002\" integrator=\"implicitfast\"/>"
    "<worldbody><body name=\"a\" pos=\"0 0 1\"><joint name=\"h1\" type=\"hinge\" axis=\"0 1 0\"/>"
    "<geom type=\"capsule\" fromto=\"0 0 0 0.3 0 0\" size=\"0.03\"/></body></worldbody>"
    "<actuator><pid joint=\"h1\"/></actuator></mujoco>";

int main(int argc, char** argv) {
  if (argc != 3) {
    fprintf(stderr, "usage: oracle_pid_probe plugin|builtin <libactuator.so>\n");
    return 2;
  }
  printf("version\t%s\nheader\t%d\n", mj_versionString(), mjVERSION_HEADER);
  int plugin = strcmp(argv[1], "plugin") == 0;
  if (plugin) mj_loadPluginLibrary(argv[2]);
  char err[8192];
  err[0] = 0;
  mjModel* m = NULL;
  char path[] = "/tmp/oracle-pid-fixture.xml";
  FILE* f = fopen(path, "w");
  if (!f) return 2;
  fputs(plugin ? PLUGIN_XML : BARE_XML, f);
  fclose(f);
  err[0] = 0;
  m = mj_loadXML(path, NULL, err, (int)sizeof(err));
  if (!m) {
    printf("load_error\t%s\n", err);
    return 1;
  }
  printf("nu\t%ld\nna\t%ld\nnactuator\t%ld\n", (long)m->nu, (long)m->na, (long)m->nactuator);
  if (!plugin) {
    /* mjs_setToPID stores kp as -biasprm[1] and ki as gainprm[0]. */
    bits("biasprm", m->actuator_biasprm, (int)(m->nactuator * mjNBIAS));
    bits("gainprm", m->actuator_gainprm, (int)(m->nactuator * mjNGAIN));
    printf("gaintype %d biastype %d dyntype %d\n", m->actuator_gaintype[0], m->actuator_biastype[0], m->actuator_dyntype[0]);
    mj_deleteModel(m);
    return 0;
  }
  mjData* d = mj_makeData(m);
  mj_resetData(m, d);
  d->qpos[0] = 0.14;
  d->qpos[1] = -0.3;
  d->qvel[0] = 0.17;
  d->qvel[1] = -0.4;
  d->act[0] = 0.1;
  d->act[1] = 0.05;
  for (int k = 0; k < 10; k++) {
    d->ctrl[0] = k < 5 ? 0.9 : -0.5;
    d->ctrl[1] = 0.4;
    mj_step(m, d);
  }
  bits("qpos", d->qpos, m->nq);
  bits("qvel", d->qvel, m->nv);
  bits("act", d->act, m->na);
  mj_deleteData(d);
  mj_deleteModel(m);
  return 0;
}
