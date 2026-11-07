#include "pyhex.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
/* Round-trip with the same rules as menagerie_native_step.rs `hex`. */
static int parse_pyhex(const char* s, uint64_t* out) {
  int neg = 0;
  if (*s == '-') { neg = 1; s++; }
  if (s[0] != '0' || s[1] != 'x') return -1;
  s += 2;
  const char* p = strchr(s, 'p');
  if (!p || p - s >= 64) return -1;
  char mant[64];
  memcpy(mant, s, (size_t)(p - s));
  mant[p - s] = 0;
  char* dot = strchr(mant, '.');
  const char* ip = mant;
  const char* frac = "";
  if (dot) { *dot = 0; frac = dot + 1; }
  char digits[16];
  int nd = 0;
  for (int i = 0; frac[i] && nd < 13; i++) digits[nd++] = frac[i];
  while (nd < 13) digits[nd++] = '0';
  digits[nd] = 0;
  uint64_t fbits = strtoull(digits, NULL, 16);
  int exp = atoi(p + 1);
  uint64_t bits;
  if (mant[0] == '0' && fbits == 0) bits = 0;
  else if (mant[0] == '0') bits = fbits;
  else bits = ((uint64_t)(exp + 1023) << 52) | fbits;
  if (neg) bits |= 1ULL << 63;
  *out = bits;
  (void)ip;
  return 0;
}
static int expect_hex(uint64_t bits, const char* want) {
  double x;
  memcpy(&x, &bits, 8);
  char buf[128];
  FILE* mem = fmemopen(buf, sizeof buf, "w");
  if (!mem) return 1;
  int rc = pyhex_write(mem, x);
  fclose(mem);
  if (rc != 0) {
    fprintf(stderr, "finite refused %016llx\n", (unsigned long long)bits);
    return 1;
  }
  if (strcmp(buf, want) != 0) {
    fprintf(stderr, "spelling %016llx\n got %s\n want %s\n", (unsigned long long)bits, buf, want);
    return 1;
  }
  uint64_t back = 0;
  if (parse_pyhex(buf, &back) != 0 || back != bits) {
    fprintf(stderr, "roundtrip %016llx -> %s -> %016llx\n", (unsigned long long)bits, buf, (unsigned long long)back);
    return 1;
  }
  return 0;
}
static int expect_refuse(uint64_t bits) {
  double x;
  memcpy(&x, &bits, 8);
  char buf[128];
  memset(buf, 0, sizeof buf);
  FILE* mem = fmemopen(buf, sizeof buf, "w");
  int rc = pyhex_write(mem, x);
  fclose(mem);
  if (rc == 0) {
    fprintf(stderr, "non-finite accepted %016llx as %s\n", (unsigned long long)bits, buf);
    return 1;
  }
  return 0;
}
int main(void) {
  int bad = 0;
  bad += expect_hex(0x0000000000000000ULL, "0x0.0p+0");
  bad += expect_hex(0x8000000000000000ULL, "-0x0.0p+0");
  bad += expect_hex(0x0000000000000001ULL, "0x0.0000000000001p-1022");
  bad += expect_hex(0x0000000000000002ULL, "0x0.0000000000002p-1022");
  bad += expect_hex(0x000fffffffffffffULL, "0x0.fffffffffffffp-1022");
  bad += expect_hex(0x800fffffffffffffULL, "-0x0.fffffffffffffp-1022");
  bad += expect_hex(0x0010000000000000ULL, "0x1.0000000000000p-1022");
  bad += expect_hex(0x8010000000000000ULL, "-0x1.0000000000000p-1022");
  bad += expect_hex(0x3ff0000000000000ULL, "0x1.0000000000000p+0");
  bad += expect_hex(0xbff0000000000000ULL, "-0x1.0000000000000p+0");
  bad += expect_hex(0x7fefffffffffffffULL, "0x1.fffffffffffffp+1023");
  bad += expect_refuse(0x7ff0000000000000ULL);
  bad += expect_refuse(0xfff0000000000000ULL);
  bad += expect_refuse(0x7ff8000000000001ULL);
  bad += expect_refuse(0xfff8000000000001ULL);
  if (bad) {
    fprintf(stderr, "hexfloat selftest failed %d\n", bad);
    return 1;
  }
  printf("hexfloat selftest ok\n");
  return 0;
}
