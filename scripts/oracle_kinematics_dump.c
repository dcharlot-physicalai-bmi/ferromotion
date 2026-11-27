/* Dump mj_kinematics geom frames at qpos0 for each model XML under argv[1].
 * Link this against the named oracle shared library, not the PyPI wheel.
 */
#include <mujoco/mujoco.h>
#include <dirent.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

static int ends_xml(const char* name) {
  size_t n = strlen(name);
  return n > 4 && strcmp(name + n - 4, ".xml") == 0;
}

static void dump_model(const char* root, const char* rel, const char* path) {
  char err[1024] = {0};
  mjModel* m = mj_loadXML(path, NULL, err, sizeof(err));
  if (!m) {
    printf("model\t%s\tload_error\t%s\n", rel, err);
    return;
  }
  mjData* d = mj_makeData(m);
  mj_kinematics(m, d);
  printf("model\t%s\tnq\t%d\tngeom\t%d\n", rel, m->nq, m->ngeom);
  printf("qpos0");
  for (int i = 0; i < m->nq; i++) {
    uint64_t bits;
    memcpy(&bits, &d->qpos[i], 8);
    printf("\t%016llx", (unsigned long long)bits);
  }
  printf("\n");
  for (int g = 0; g < m->ngeom; g++) {
    printf("geom\t%d", g);
    for (int k = 0; k < 3; k++) {
      uint64_t bits;
      memcpy(&bits, &d->geom_xpos[g * 3 + k], 8);
      printf("\t%016llx", (unsigned long long)bits);
    }
    for (int k = 0; k < 9; k++) {
      uint64_t bits;
      memcpy(&bits, &d->geom_xmat[g * 9 + k], 8);
      printf("\t%016llx", (unsigned long long)bits);
    }
    printf("\n");
  }
  mj_deleteData(d);
  mj_deleteModel(m);
}

int main(int argc, char** argv) {
  if (argc != 2) {
    fprintf(stderr, "usage: oracle_kinematics_dump <menagerie root>\n");
    return 2;
  }
  DIR* top = opendir(argv[1]);
  if (!top) return 2;
  struct dirent* ent;
  while ((ent = readdir(top))) {
    if (ent->d_name[0] == '.') continue;
    char dirpath[4096];
    snprintf(dirpath, sizeof(dirpath), "%s/%s", argv[1], ent->d_name);
    DIR* sub = opendir(dirpath);
    if (!sub) continue;
    struct dirent* xml;
    while ((xml = readdir(sub))) {
      if (!ends_xml(xml->d_name)) continue;
      char path[4096], rel[4096];
      snprintf(path, sizeof(path), "%s/%s", dirpath, xml->d_name);
      snprintf(rel, sizeof(rel), "%s/%s", ent->d_name, xml->d_name);
      dump_model(argv[1], rel, path);
    }
    closedir(sub);
  }
  closedir(top);
  return 0;
}
