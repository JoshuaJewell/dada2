// CUDA implementation for taxonomy assignment
// Designed for memory-efficient processing with batching for GPUs with limited VRAM

#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <stddef.h>
#include <stdbool.h>

#define CUDA_CHECK(call) \
  do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
      fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(err)); \
      return false; \
    } \
  } while(0)

// CUDA kernel: Compute genus scores for all sequence-genus pairs
// Each thread processes one sequence-genus pair
__global__ void compute_genus_scores_kernel(
    int *d_karrays,           // [nseq][max_arraylen] - query kmers
    int *d_arraylen,          // [nseq] - number of valid kmers per sequence
    float *d_lgk_probability, // [ngenus_batch][n_kmers] - reference probabilities
    float *d_scores,          // [nseq][ngenus_batch] - output scores
    int nseq,
    int ngenus_batch,
    int n_kmers,
    int max_arraylen
) {
    int seq_idx = blockIdx.x;
    int genus_idx = threadIdx.x + blockIdx.y * blockDim.x;

    if (seq_idx >= nseq || genus_idx >= ngenus_batch) return;

    int arraylen = d_arraylen[seq_idx];
    int *karray = &d_karrays[seq_idx * max_arraylen];
    float *lgk_v = &d_lgk_probability[genus_idx * n_kmers];

    float logp = 0.0f;

    // Sum log probabilities (inner loop - highly parallel on GPU)
    #pragma unroll 4
    for(int pos = 0; pos < arraylen; pos++) {
        int kmer = karray[pos];
        logp += lgk_v[kmer];
    }

    d_scores[seq_idx * ngenus_batch + genus_idx] = logp;
}

// CUDA kernel: Find best genus per sequence using reduction
__global__ void find_best_genus_kernel(
    float *d_scores,      // [nseq][ngenus_batch]
    int *d_best_genus,    // [nseq] - output best genus index
    float *d_best_logp,   // [nseq] - output best log probability
    int nseq,
    int ngenus_batch,
    int genus_offset      // Offset for this batch in global genus space
) {
    int seq_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (seq_idx >= nseq) return;

    float *scores = &d_scores[seq_idx * ngenus_batch];
    float max_logp = -1e30f;  // Very negative
    int max_g = -1;

    // Find max score for this sequence across batch
    for(int g = 0; g < ngenus_batch; g++) {
        if (scores[g] > max_logp) {
            max_logp = scores[g];
            max_g = g + genus_offset;
        }
    }

    // Update global best if this batch's best is better
    // Note: We'll handle this on CPU side for first batch vs subsequent batches
    d_best_genus[seq_idx] = max_g;
    d_best_logp[seq_idx] = max_logp;
}

// CUDA kernel: Bootstrap genus scoring
// Process all bootstrap iterations for all sequences on GPU
__global__ void compute_bootstrap_scores_kernel(
    int *d_bootarrays,        // [nseq][NBOOT][boot_arraylen] - bootstrap kmers
    int *d_boot_arraylen,     // [nseq] - bootstrap array lengths (arraylen/8)
    float *d_lgk_probability, // [ngenus_batch][n_kmers] - reference probabilities
    float *d_scores,          // [nseq][NBOOT][ngenus_batch] - output scores
    int nseq,
    int nboot,
    int ngenus_batch,
    int n_kmers,
    int max_boot_arraylen
) {
    int seq_idx = blockIdx.x;
    int boot_idx = blockIdx.y;
    int genus_idx = threadIdx.x + blockIdx.z * blockDim.x;

    if (seq_idx >= nseq || boot_idx >= nboot || genus_idx >= ngenus_batch) return;

    int arraylen = d_boot_arraylen[seq_idx];
    int *bootarray = &d_bootarrays[(seq_idx * nboot + boot_idx) * max_boot_arraylen];
    float *lgk_v = &d_lgk_probability[genus_idx * n_kmers];

    float logp = 0.0f;

    #pragma unroll 4
    for(int pos = 0; pos < arraylen; pos++) {
        int kmer = bootarray[pos];
        logp += lgk_v[kmer];
    }

    d_scores[(seq_idx * nboot + boot_idx) * ngenus_batch + genus_idx] = logp;
}

// CUDA kernel: Find best genus for each bootstrap iteration
__global__ void find_best_bootstrap_genus_kernel(
    float *d_scores,          // [nseq][NBOOT][ngenus_batch]
    int *d_boot_genus,        // [nseq][NBOOT] - output bootstrap genus indices
    float *d_boot_logp,       // [nseq][NBOOT] - output bootstrap log probabilities
    int nseq,
    int nboot,
    int ngenus_batch,
    int genus_offset
) {
    int seq_idx = blockIdx.x;
    int boot_idx = threadIdx.x + blockIdx.y * blockDim.x;

    if (seq_idx >= nseq || boot_idx >= nboot) return;

    float *scores = &d_scores[(seq_idx * nboot + boot_idx) * ngenus_batch];
    float max_logp = -1e30f;
    int max_g = -1;

    for(int g = 0; g < ngenus_batch; g++) {
        if (scores[g] > max_logp) {
            max_logp = scores[g];
            max_g = g + genus_offset;
        }
    }

    d_boot_genus[seq_idx * nboot + boot_idx] = max_g;
    d_boot_logp[seq_idx * nboot + boot_idx] = max_logp;
}

// Host function: Check if CUDA is available and get device info
extern "C" bool cuda_check_available(int *cuda_cores, size_t *vram_mb) {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);

    if (err != cudaSuccess || device_count == 0) {
        return false;
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    *cuda_cores = prop.multiProcessorCount * 128; // Rough estimate
    *vram_mb = prop.totalGlobalMem / (1024 * 1024);

    return true;
}

// Host function: Process taxonomy assignment on GPU with memory-aware batching
extern "C" bool cuda_assign_taxonomy_batch(
    int *h_karrays,           // Host: [nseq][max_arraylen]
    int *h_arraylen,          // Host: [nseq]
    float *h_lgk_probability, // Host: [ngenus][n_kmers]
    int *h_best_genus,        // Host: [nseq] - output
    float *h_best_logp,       // Host: [nseq] - output
    int nseq,
    int ngenus,
    int n_kmers,
    int max_arraylen,
    size_t max_vram_mb
) {
    // Calculate memory requirements
    size_t query_mem = (size_t)nseq * max_arraylen * sizeof(int) + nseq * sizeof(int);
    size_t result_mem = (size_t)nseq * sizeof(int) + nseq * sizeof(float);
    size_t genus_per_batch_mem = (size_t)n_kmers * sizeof(float);

    // Reserve 500MB for CUDA overhead
    size_t available_vram = (max_vram_mb - 500) * 1024 * 1024;
    size_t used_fixed = query_mem + result_mem;

    if (used_fixed >= available_vram) {
        fprintf(stderr, "Not enough VRAM for query data (%zu MB needed, %zu MB available)\n",
                used_fixed / (1024*1024), available_vram / (1024*1024));
        return false;
    }

    // Calculate how many genera we can fit in one batch
    // Need space for: genus data (ngenus * n_kmers * 4) + score matrix (nseq * ngenus * 4)
    size_t available_for_batch = available_vram - used_fixed;

    // Memory per genus: reference data + score column
    size_t mem_per_genus = (size_t)n_kmers * sizeof(float) + (size_t)nseq * sizeof(float);
    int ngenus_per_batch = (int)(available_for_batch / mem_per_genus);

    // Be conservative - use only 90% of calculated batch size to leave safety margin
    ngenus_per_batch = (ngenus_per_batch * 9) / 10;

    // Ensure minimum batch size
    if (ngenus_per_batch < 100) {
        fprintf(stderr, "VRAM too small for efficient batching (can only fit %d genera per batch)\n",
                ngenus_per_batch);
        if (ngenus_per_batch < 10) {
            fprintf(stderr, "VRAM insufficient for GPU acceleration\n");
            return false;
        }
    }

    size_t score_mem_per_batch = (size_t)nseq * ngenus_per_batch * sizeof(float);

    int nbatches = (ngenus + ngenus_per_batch - 1) / ngenus_per_batch;

    printf("CUDA taxonomy: Processing %d genera in %d batches (%d genera/batch)\n",
           ngenus, nbatches, ngenus_per_batch);
    printf("VRAM usage: Query data %.1f MB, Batch data %.1f MB, Total %.1f MB / %.1f MB available\n",
           (float)query_mem / (1024*1024),
           (float)(genus_per_batch_mem * ngenus_per_batch + score_mem_per_batch) / (1024*1024),
           (float)(used_fixed + genus_per_batch_mem * ngenus_per_batch + score_mem_per_batch) / (1024*1024),
           (float)available_vram / (1024*1024));

    // Allocate device memory for query data (stays on GPU for all batches)
    int *d_karrays, *d_arraylen;
    CUDA_CHECK(cudaMalloc(&d_karrays, nseq * max_arraylen * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_arraylen, nseq * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_karrays, h_karrays, nseq * max_arraylen * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_arraylen, h_arraylen, nseq * sizeof(int), cudaMemcpyHostToDevice));

    // Allocate device memory for batch processing
    float *d_lgk_batch;
    float *d_scores;
    int *d_best_genus;
    float *d_best_logp;

    CUDA_CHECK(cudaMalloc(&d_lgk_batch, (size_t)ngenus_per_batch * n_kmers * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scores, (size_t)nseq * ngenus_per_batch * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_best_genus, nseq * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_best_logp, nseq * sizeof(float)));

    // Initialize best results with very negative values
    float init_logp = -1e30f;
    int init_genus = -1;
    float *h_batch_logp = (float*)malloc(nseq * sizeof(float));
    int *h_batch_genus = (int*)malloc(nseq * sizeof(int));
    for(int i = 0; i < nseq; i++) {
        h_best_logp[i] = init_logp;
        h_best_genus[i] = init_genus;
    }

    // Process each batch
    for(int batch = 0; batch < nbatches; batch++) {
        int genus_start = batch * ngenus_per_batch;
        int genus_end = (batch + 1) * ngenus_per_batch;
        if (genus_end > ngenus) genus_end = ngenus;
        int ngenus_this_batch = genus_end - genus_start;

        // Copy this batch of genera to GPU
        size_t batch_bytes = (size_t)ngenus_this_batch * n_kmers * sizeof(float);
        CUDA_CHECK(cudaMemcpy(d_lgk_batch,
                             &h_lgk_probability[genus_start * n_kmers],
                             batch_bytes,
                             cudaMemcpyHostToDevice));

        // Launch kernel: compute scores
        dim3 blocks(nseq, (ngenus_this_batch + 255) / 256);
        dim3 threads(256, 1);

        compute_genus_scores_kernel<<<blocks, threads>>>(
            d_karrays, d_arraylen, d_lgk_batch, d_scores,
            nseq, ngenus_this_batch, n_kmers, max_arraylen
        );
        CUDA_CHECK(cudaGetLastError());

        // Launch kernel: find best in this batch
        int threads_per_block = 256;
        int blocks_needed = (nseq + threads_per_block - 1) / threads_per_block;

        find_best_genus_kernel<<<blocks_needed, threads_per_block>>>(
            d_scores, d_best_genus, d_best_logp,
            nseq, ngenus_this_batch, genus_start
        );
        CUDA_CHECK(cudaGetLastError());

        // Copy batch results back
        CUDA_CHECK(cudaMemcpy(h_batch_genus, d_best_genus, nseq * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_batch_logp, d_best_logp, nseq * sizeof(float), cudaMemcpyDeviceToHost));

        // Update global best
        for(int i = 0; i < nseq; i++) {
            if (h_batch_logp[i] > h_best_logp[i]) {
                h_best_logp[i] = h_batch_logp[i];
                h_best_genus[i] = h_batch_genus[i];
            }
        }

        if ((batch + 1) % 5 == 0 || batch == nbatches - 1) {
            printf("  Processed batch %d/%d (%.1f%% complete)\n",
                   batch + 1, nbatches, 100.0f * (batch + 1) / nbatches);
        }
    }

    // Cleanup
    free(h_batch_logp);
    free(h_batch_genus);
    cudaFree(d_karrays);
    cudaFree(d_arraylen);
    cudaFree(d_lgk_batch);
    cudaFree(d_scores);
    cudaFree(d_best_genus);
    cudaFree(d_best_logp);

    return true;
}

// Host function: GPU-accelerated bootstrap evaluation
extern "C" bool cuda_assign_taxonomy_bootstrap(
    int *h_bootarrays,        // Host: [nseq][NBOOT][max_boot_arraylen]
    int *h_boot_arraylen,     // Host: [nseq]
    float *h_lgk_probability, // Host: [ngenus][n_kmers]
    int *h_boot_genus,        // Host: [nseq][NBOOT] - output
    int nseq,
    int nboot,
    int ngenus,
    int n_kmers,
    int max_boot_arraylen,
    size_t max_vram_mb
) {
    // Calculate memory requirements
    size_t boot_data_mem = (size_t)nseq * nboot * max_boot_arraylen * sizeof(int);
    size_t boot_len_mem = (size_t)nseq * sizeof(int);
    size_t result_mem = (size_t)nseq * nboot * sizeof(int);

    // Reserve 500MB for CUDA overhead
    size_t available_vram = (max_vram_mb - 500) * 1024 * 1024;
    size_t used_fixed = boot_data_mem + boot_len_mem + result_mem;

    if (used_fixed >= available_vram) {
        fprintf(stderr, "Not enough VRAM for bootstrap data\n");
        return false;
    }

    // Calculate batch size
    size_t available_for_batch = available_vram - used_fixed;
    size_t mem_per_genus = (size_t)n_kmers * sizeof(float) +
                           (size_t)nseq * nboot * sizeof(float); // score matrix
    int ngenus_per_batch = (int)(available_for_batch / mem_per_genus);
    ngenus_per_batch = (ngenus_per_batch * 9) / 10; // 90% safety margin

    if (ngenus_per_batch < 100) {
        fprintf(stderr, "VRAM too small for bootstrap batching\n");
        return false;
    }

    int nbatches = (ngenus + ngenus_per_batch - 1) / ngenus_per_batch;

    printf("CUDA bootstrap: Processing %d iterations across %d genera in %d batches\n",
           nboot, ngenus, nbatches);

    // Allocate device memory for bootstrap data (stays on GPU for all batches)
    int *d_bootarrays, *d_boot_arraylen;
    CUDA_CHECK(cudaMalloc(&d_bootarrays, boot_data_mem));
    CUDA_CHECK(cudaMalloc(&d_boot_arraylen, boot_len_mem));
    CUDA_CHECK(cudaMemcpy(d_bootarrays, h_bootarrays, boot_data_mem, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_boot_arraylen, h_boot_arraylen, boot_len_mem, cudaMemcpyHostToDevice));

    // Allocate device memory for batch processing
    float *d_lgk_batch;
    float *d_scores;
    int *d_boot_genus;
    int *d_batch_genus;
    float *d_batch_logp;

    CUDA_CHECK(cudaMalloc(&d_lgk_batch, (size_t)ngenus_per_batch * n_kmers * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scores, (size_t)nseq * nboot * ngenus_per_batch * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_boot_genus, nseq * nboot * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_batch_genus, nseq * nboot * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_batch_logp, nseq * nboot * sizeof(float)));

    // Allocate and initialize result tracking arrays
    int *h_batch_genus = (int*)malloc(nseq * nboot * sizeof(int));
    float *h_batch_logp = (float*)malloc(nseq * nboot * sizeof(float));
    float *h_best_logp = (float*)malloc(nseq * nboot * sizeof(float));

    // Initialize best results with very negative values
    for(int i = 0; i < nseq * nboot; i++) {
        h_boot_genus[i] = -1;
        h_best_logp[i] = -1e30f;
    }

    // Process each batch
    for(int batch = 0; batch < nbatches; batch++) {
        int genus_start = batch * ngenus_per_batch;
        int genus_end = (batch + 1) * ngenus_per_batch;
        if (genus_end > ngenus) genus_end = ngenus;
        int ngenus_this_batch = genus_end - genus_start;

        // Copy this batch of genera to GPU
        size_t batch_bytes = (size_t)ngenus_this_batch * n_kmers * sizeof(float);
        CUDA_CHECK(cudaMemcpy(d_lgk_batch,
                             &h_lgk_probability[genus_start * n_kmers],
                             batch_bytes,
                             cudaMemcpyHostToDevice));

        // Launch bootstrap scoring kernel
        dim3 blocks_score(nseq, nboot, (ngenus_this_batch + 255) / 256);
        dim3 threads_score(256, 1, 1);

        compute_bootstrap_scores_kernel<<<blocks_score, threads_score>>>(
            d_bootarrays, d_boot_arraylen, d_lgk_batch, d_scores,
            nseq, nboot, ngenus_this_batch, n_kmers, max_boot_arraylen
        );
        CUDA_CHECK(cudaGetLastError());

        // Find best genus for each bootstrap iteration
        dim3 blocks_best(nseq, (nboot + 255) / 256);
        dim3 threads_best(256);

        find_best_bootstrap_genus_kernel<<<blocks_best, threads_best>>>(
            d_scores, d_batch_genus, d_batch_logp,
            nseq, nboot, ngenus_this_batch, genus_start
        );
        CUDA_CHECK(cudaGetLastError());

        // Copy batch results back
        CUDA_CHECK(cudaMemcpy(h_batch_genus, d_batch_genus,
                             nseq * nboot * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_batch_logp, d_batch_logp,
                             nseq * nboot * sizeof(float), cudaMemcpyDeviceToHost));

        // Update global best (compare across batches)
        int updates = 0;
        for(int i = 0; i < nseq * nboot; i++) {
            if (h_batch_logp[i] > h_best_logp[i]) {
                h_best_logp[i] = h_batch_logp[i];
                h_boot_genus[i] = h_batch_genus[i];
                updates++;
            }
        }

        if ((batch + 1) % 5 == 0 || batch == nbatches - 1) {
            printf("  Bootstrap batch %d/%d (%.1f%% complete)\n",
                   batch + 1, nbatches, 100.0f * (batch + 1) / nbatches);
        }
    }

    // Cleanup
    free(h_batch_genus);
    free(h_batch_logp);
    free(h_best_logp);
    cudaFree(d_bootarrays);
    cudaFree(d_boot_arraylen);
    cudaFree(d_lgk_batch);
    cudaFree(d_scores);
    cudaFree(d_boot_genus);
    cudaFree(d_batch_genus);
    cudaFree(d_batch_logp);

    return true;
}
