#include <mujoco/mujoco.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
/* x86 contract-off mju_mulQuat bits for this vector. Not a Darwin golden. */
int main(void) {
  double a[4] = {0.9, 0.1, -0.2, 0.3};
  double b[4] = {0.8, -0.1, 0.4, 0.2};
  double o[4];
  mju_mulQuat(o, a, b);
  for (int i = 0; i < 4; i++) {
    uint64_t u;
    memcpy(&u, &o[i], 8);
    printf("%016llx%c", (unsigned long long)u, i == 3 ? '\n' : ' ');
  }
  return 0;
}
