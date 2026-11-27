#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef double mjtNum;
void mju_mulQuat(mjtNum res[4], const mjtNum qa[4], const mjtNum qb[4]);
void mju_rotVecQuat(mjtNum res[3], const mjtNum vec[3], const mjtNum quat[4]);
void mju_mulMatVec3(mjtNum res[3], const mjtNum mat[9], const mjtNum vec[3]);

static int parse_u64s(const char *s, uint64_t *out, int cap) {
  int n = 0;
  const char *p = s;
  while (*p != '\0' && n < cap) {
    char *end = NULL;
    unsigned long long v = strtoull(p, &end, 10);
    if (end == p) {
      return -1;
    }
    out[n++] = (uint64_t)v;
    if (*end == ',') {
      p = end + 1;
    } else if (*end == '\0') {
      break;
    } else {
      return -1;
    }
  }
  return n;
}

static void bits_to_f64(const uint64_t *u, double *d, int n) {
  for (int i = 0; i < n; i++) {
    memcpy(&d[i], &u[i], 8);
  }
}

static void print_outs(const double *d, int n) {
  for (int i = 0; i < n; i++) {
    uint64_t u;
    memcpy(&u, &d[i], 8);
    printf("%s%llu", i ? "," : "", (unsigned long long)u);
  }
}

int main(int argc, char **argv) {
  char line[16384];
  if (argc != 2) {
    fprintf(stderr, "usage: op-vectors <vectors.txt>\n");
    return 2;
  }
  FILE *f = fopen(argv[1], "r");
  if (!f) {
    perror(argv[1]);
    return 2;
  }
  while (fgets(line, sizeof line, f)) {
    size_t len = strlen(line);
    while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r')) {
      line[--len] = '\0';
    }
    if (len == 0) {
      continue;
    }
    char *tab1 = strchr(line, '\t');
    char *tab2;
    char *op;
    char *inputs;
    uint64_t in[16];
    double a[16];
    double o[4];
    int n;
    int on = 0;
    if (!tab1) {
      fprintf(stderr, "refusing: vector line has no tab\n");
      return 2;
    }
    *tab1 = '\0';
    tab2 = strchr(tab1 + 1, '\t');
    if (!tab2) {
      fprintf(stderr, "refusing: vector line has no output column\n");
      return 2;
    }
    *tab2 = '\0';
    op = line;
    inputs = tab1 + 1;
    n = parse_u64s(inputs, in, 16);
    if (n < 0) {
      fprintf(stderr, "refusing: input bits are not unsigned integers\n");
      return 2;
    }
    bits_to_f64(in, a, n);
    if (strcmp(op, "mul_quat") == 0 && n == 8) {
      mju_mulQuat(o, a, a + 4);
      on = 4;
    } else if (strcmp(op, "rot_vec_quat") == 0 && n == 7) {
      mju_rotVecQuat(o, a, a + 3);
      on = 3;
    } else if (strcmp(op, "mul_mat_vec3") == 0 && n == 12) {
      mju_mulMatVec3(o, a, a + 9);
      on = 3;
    } else {
      fprintf(stderr, "refusing: %s with %d inputs\n", op, n);
      return 2;
    }
    fputs(op, stdout);
    fputc('\t', stdout);
    fputs(inputs, stdout);
    fputc('\t', stdout);
    print_outs(o, on);
    fputc('\n', stdout);
  }
  return 0;
}
