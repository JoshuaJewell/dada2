#include "dada.h"

// [[Rcpp::interfaces(cpp)]]

/************* DP WORKSPACE *****************
 * Workspace pooling for dynamic programming alignment matrices
 * Eliminates malloc/free overhead by reusing pre-allocated buffers
 */

DPWorkspace* dp_workspace_new(size_t capacity) {
  DPWorkspace *ws = (DPWorkspace*) malloc(sizeof(DPWorkspace));
  if (ws == NULL) {
    Rcpp::stop("Memory allocation failed for DPWorkspace structure.");
  }

  ws->d = (int16_t*) aligned_malloc(capacity * sizeof(int16_t), AVX512_ALIGNMENT);
  ws->p = (int16_t*) aligned_malloc(capacity * sizeof(int16_t), AVX512_ALIGNMENT);

  if (ws->d == NULL || ws->p == NULL) {
    if (ws->d) aligned_free(ws->d);
    if (ws->p) aligned_free(ws->p);
    free(ws);
    Rcpp::stop("Memory allocation failed for DPWorkspace buffers.");
  }

  ws->capacity = capacity;
  return ws;
}

void dp_workspace_ensure_capacity(DPWorkspace *ws, size_t needed) {
  if (ws == NULL) {
    Rcpp::stop("DPWorkspace is NULL.");
  }

  if (needed > ws->capacity) {
    // Reallocate with larger capacity
    aligned_free(ws->d);
    aligned_free(ws->p);

    ws->d = (int16_t*) aligned_malloc(needed * sizeof(int16_t), AVX512_ALIGNMENT);
    ws->p = (int16_t*) aligned_malloc(needed * sizeof(int16_t), AVX512_ALIGNMENT);

    if (ws->d == NULL || ws->p == NULL) {
      Rcpp::stop("Memory reallocation failed for DPWorkspace buffers.");
    }

    ws->capacity = needed;
  }
}

void dp_workspace_free(DPWorkspace *ws) {
  if (ws != NULL) {
    if (ws->d) aligned_free(ws->d);
    if (ws->p) aligned_free(ws->p);
    free(ws);
  }
}
