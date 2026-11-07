/* Python float.hex for finite doubles. NaN and Inf are refused (return -1).
   Subnormals are 0x0.<52-bit fraction>p-1022, which is Python's spelling and the
   spelling the Rust oracle parser turns back into the raw fraction bits. */
#ifndef ARM_ORACLE_PYHEX_H
#define ARM_ORACLE_PYHEX_H
#include <stdint.h>
#include <stdio.h>
#include <string.h>
static int pyhex_write(FILE* o, double x) {
  uint64_t u;
  memcpy(&u, &x, 8);
  int neg = (int)(u >> 63);
  int be = (int)((u >> 52) & 0x7ff);
  uint64_t frac = u & 0xfffffffffffffULL;
  if (be == 0x7ff) return -1;
  if (neg) fputc('-', o);
  if (be == 0 && frac == 0) {
    fputs("0x0.0p+0", o);
    return 0;
  }
  if (be == 0) {
    fprintf(o, "0x0.%013llxp-1022", (unsigned long long)frac);
    return 0;
  }
  fprintf(o, "0x1.%013llxp%+d", (unsigned long long)frac, be - 1023);
  return 0;
}
#endif
