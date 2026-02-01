#ifndef TAXONOMY_CUDA_WRAPPER_H
#define TAXONOMY_CUDA_WRAPPER_H

#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Check if CUDA is available
bool cuda_check_available(int *cuda_cores, size_t *vram_mb);

// CUDA taxonomy assignment (batched for memory efficiency)
bool cuda_assign_taxonomy_batch(
    int *h_karrays,
    int *h_arraylen,
    float *h_lgk_probability,
    int *h_best_genus,
    float *h_best_logp,
    int nseq,
    int ngenus,
    int n_kmers,
    int max_arraylen,
    size_t max_vram_mb
);

// CUDA bootstrap evaluation (batched for memory efficiency)
bool cuda_assign_taxonomy_bootstrap(
    int *h_bootarrays,
    int *h_boot_arraylen,
    float *h_lgk_probability,
    int *h_boot_genus,
    int nseq,
    int nboot,
    int ngenus,
    int n_kmers,
    int max_boot_arraylen,
    size_t max_vram_mb
);

#ifdef __cplusplus
}
#endif

#endif // TAXONOMY_CUDA_WRAPPER_H
