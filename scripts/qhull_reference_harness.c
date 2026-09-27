// MuJoCo 3.13.0's mjCMesh::MakeGraph over qhull d1c2fc0 (the version MuJoCo pins), as a reference for the
// Rust port in crates/ferromotion-core/src/qhull.rs. Reads `menagerie_qhull --dump`'s format on stdin and
// prints each mesh's graph in mesh_graph's layout, one `graph <model> <mesh> <ints>` line per mesh.
//
// Built WITH floating-point contraction (-ffp-contract=on, clang's default on arm64) it reproduces the
// Apple-silicon MuJoCo wheel's mesh_graph on all 794 Menagerie collision meshes; without it, 120 differ.
// See scripts/qhull_reference_build.sh.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <setjmp.h>
#include "qhull_ra.h"

static int graph(const double* dvert, int nvert, int maxhullvert, int** out) {
  int adr, ok, curlong, totlong, exitcode;
  facetT *facet, **facetp;
  vertexT *vertex, *vertex1, **vertex1p;
  char qhopt[64] = "qhull Qt";
  if (maxhullvert > -1) sprintf(qhopt + strlen(qhopt), " Q9 TA%d", maxhullvert - 4);
  qhT qh_qh; qhT* qh = &qh_qh;
  qh_zero(qh, stderr);
  qh_init_A(qh, stdin, stdout, stderr, 0, NULL);
  int* g = NULL; int sz = 0;
  exitcode = setjmp(qh->errexit);
  qh->NOerrexit = False;
  if (!exitcode) {
    qh_initflags(qh, qhopt);
    qh_init_B(qh, (double*)dvert, nvert, 3, False);
    qh_qhull(qh);
    qh_triangulate(qh);
    qh_vertexneighbors(qh);
    int numvert = qh->num_vertices, numface = qh->num_facets;
    sz = 2 + 3 * numvert + 6 * numface;
    g = malloc(sz * sizeof(int));
    g[0] = numvert; g[1] = numface;
    int* vert_edgeadr = g + 2; int* vert_globalid = g + 2 + numvert;
    int* edge_localid = g + 2 + 2 * numvert; int* face_globalid = g + 2 + 3 * numvert + 3 * numface;
    int i = adr = 0; ok = 1;
    FORALLvertices {
      int pid = qh_pointid(qh, vertex->point);
      vert_edgeadr[i] = adr; vert_globalid[i] = pid;
      int start = adr;
      FOREACHsetelement_(facetT, vertex->neighbors, facet) {
        FOREACHsetelement_(vertexT, facet->vertices, vertex1) {
          int pid1 = qh_pointid(qh, vertex1->point);
          if (pid != pid1) {
            int j;
            for (j = start; j < adr; j++) if (pid1 == edge_localid[j]) break;
            if (j >= adr) edge_localid[adr++] = pid1;
          }
        }
      }
      edge_localid[adr++] = -1;
      i++;
    }
    if (adr != numvert + 3 * numface) { fprintf(stderr, "wrong size\n"); ok = 0; }
    adr = 0;
    FORALLfacets {
      int ii = 0; int ind[3] = {0, 1, 2};
      if (facet->toporient) { ind[0] = 1; ind[1] = 0; }
      FOREACHsetelement_(vertexT, facet->vertices, vertex1) face_globalid[adr + ind[ii++]] = qh_pointid(qh, vertex1->point);
      adr += 3;
    }
    qh_freeqhull(qh, !qh_ALL);
    qh_memfreeshort(qh, &curlong, &totlong);
    int* hullid = malloc(nvert * sizeof(int));
    for (int k = 0; k < nvert; k++) hullid[k] = -1;
    for (int k = 0; k < numvert; k++) hullid[vert_globalid[k]] = k;
    for (int k = 0; k < numvert + 3 * numface; k++) if (edge_localid[k] >= 0) edge_localid[k] = hullid[edge_localid[k]];
    free(hullid);
    if (!ok) { free(g); g = NULL; sz = 0; }
  } else {
    qh_freeqhull(qh, !qh_ALL);
    qh_memfreeshort(qh, &curlong, &totlong);
    if (g) free(g);
    g = NULL; sz = -1;
  }
  *out = g;
  return sz;
}

int main(int argc, char** argv) {
  const char* only = argc > 1 ? argv[1] : NULL;
  char line[4096], model[1024], name[1024];
  int cap, nvert;
  while (fgets(line, sizeof line, stdin)) {
    if (sscanf(line, "mesh\t%1023[^\t]\t%1023[^\t]\t%d\t%d", model, name, &cap, &nvert) != 4) continue;
    double* v = malloc(3 * nvert * sizeof(double));
    for (int i = 0; i < nvert; i++) {
      uint64_t b[3];
      if (!fgets(line, sizeof line, stdin) || sscanf(line, "%llx %llx %llx", &b[0], &b[1], &b[2]) != 3) { fprintf(stderr, "bad vertex\n"); return 1; }
      memcpy(v + 3 * i, b, 3 * sizeof(double));
    }
    if (only && strcmp(only, name)) { free(v); continue; }
    int* g; int sz = graph(v, nvert, cap, &g);
    printf("graph\t%s\t%s\t", model, name);
    if (sz <= 0) printf("-1\n");
    else { for (int k = 0; k < sz; k++) printf(k ? " %d" : "%d", g[k]); printf("\n"); free(g); }
    fflush(stdout);
    free(v);
  }
  return 0;
}
