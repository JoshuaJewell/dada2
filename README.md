# DADA2 with Performance Optimisations

## Complete Optimisation Summary

This README documents performance optimisations implemented in DADA2, covering both CPU and GPU acceleration for [Benjamin Callahan's dada2 repository](https://github.com/benjjneb/dada2).



## Table of Contents
1. [Overview](#overview)
2. [CPU Optimisations](#cpu-optimisations)
3. [GPU/CUDA Optimisations](#gpu-optimisations)
4. [Memory Optimisations](#memory-optimisations)
5. [Installation & Usage](#installation--usage)
6. [Troubleshooting](#troubleshooting)



## Overview

### Optimisations Implemented

| Component | Optimisation |
|-----------|--------------|
| **Denoising (dada)** | AVX2/AVX-512 SIMD |
| **Denoising (dada)** | Aligned memory |
| **Denoising (dada)** | Vector pre-allocation |
| **Taxonomy (CPU)** | OpenMP parallelisation |
| **Taxonomy (GPU)** | CUDA acceleration |



## CPU Optimisations

### 1. Denoising Algorithm

#### 1.1 AVX2/AVX-512 SIMD Vectorisation
**Files**: `src/kmers.cpp`, `src/dada.h`

**What it does**:
- Upgraded KMER distance calculations from SSE2 (8 elements) to AVX2 (16 elements) and AVX-512 (32 elements)
- Runtime CPU detection automatically selects optimal SIMD level
- Falls back gracefully (hopefully) to SSE2 or scalar on older CPUs

**Functions added**:
- `kmer_dist_AVX2()` / `kmer_dist_AVX512()`: 16-bit KMER distance
- `kmer_dist_AVX2_8()` / `kmer_dist_AVX512_8()`: 8-bit KMER distance
- `kord_dist_AVX2()` / `kord_dist_AVX512()`: Ordered KMER distance
- `kmer_dist_dispatch()`: Runtime dispatch to optimal implementation

#### 1.2 Aligned Memory Allocation
**Files**: `src/dada.h`, `src/nwalign_*.cpp`

**What it does**:
- 64-byte aligned memory allocation for DP matrices
- Enables efficient SIMD loads/stores
- Better cache line utilisation

#### 1.3 Structural Improvements
**Files**: `src/cluster.cpp`, `src/containers.cpp`

**What it does**:
- Pre-reserve vector capacity to avoid reallocations
- Increased buffer sizes (RAWBUF, CLUSTBUF: 50→256)
- Prepared DP workspace pooling infrastructure

### 2. Taxonomy Assignment

#### 2.1 OpenMP Genus-Level Parallelisation
**Files**: `src/taxonomy.cpp`

**What it does**:
- Parallelises genus loop across CPU cores
- Each thread processes subset of genera for each sequence
- Uses `#pragma omp parallel for` with SIMD hints

**Function**: `get_best_genus_parallel()`



## GPU Optimisations

### CUDA Acceleration (Nvidia GPU's Only)

#### Architecture

Memory-efficient batched processing for GPUs with limited VRAM (tested for 6GB VRAM constraint)

**Files created**:
- `src/taxonomy_cuda.cu`: CUDA kernels and host functions
- `src/taxonomy_cuda_wrapper.h`: C/C++ interface
- Integration in `src/taxonomy.cpp`

#### How It Works

1. **Query Data**: Loaded once to GPU, stays resident
2. **Reference Database**: Streamed in batches
   - Calculates optimal batch size based on available VRAM
   - For 5GB VRAM: processes ~2000-3000 genera per batch
   - Multiple batches cover entire database

3. **Processing**:
   ```
   For each batch of genera:
     1. Copy batch to GPU
     2. Launch kernel: compute all sequence×genus scores
     3. Find best genus in batch
     4. Update global best
     5. Free GPU memory for batch
   ```

4. **Bootstrap**: Currently on CPU

#### CUDA Kernels

**`compute_genus_scores_kernel`**:
- Each thread: one sequence-genus pair
- Highly parallel
- Inner loop: sum of log probabilities (SIMD-friendly)

**`find_best_genus_kernel`**:
- Per-sequence reduction to find maximum score
- Uses shared memory for efficiency



## Installation & Usage

### Current Status

The package is **installed with all optimisations active** including CUDA GPU support:

```r
library(dada2)

# These automatically use AVX2/AVX-512:
errF <- learnErrors(filtFs)
dadaFs <- dada(derepFs, err=errF)

# This uses OpenMP parallelisation:
taxa <- assignTaxonomy(seqtab, refFasta, verbose=TRUE)
```

### Enabling CUDA (for assignTaxonomy() speedup)

#### Step 1: Install CUDA Toolkit

**Ubuntu/Debian**:
```bash
# Check if NVIDIA GPU exists
lspci | grep -i nvidia

# Install CUDA (Ubuntu 22.04/Debian 12 example)
wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install cuda-toolkit-12-3

# Add to PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

**Verify installation**:
```bash
nvcc --version
nvidia-smi
```

#### Step 2: Rebuild DADA2 with CUDA

```bash
cd /home/joshua/Documents/repos/dada2
R CMD build --no-build-vignettes .
R CMD INSTALL dada2_1.36.0.tar.gz
```

You should see:
```
Compiling with CUDA support...
NVCC compiling taxonomy_cuda.cu...
```

#### Step 3: Use CUDA-accelerated Taxonomy

```r
library(dada2)

# CUDA will be used automatically if available
# Set verbose=TRUE to see what's happening
taxa <- assignTaxonomy(seqtab, refFasta, verbose=TRUE)

# Example output:
# "CUDA detected: 2560 cores, 5772 MB VRAM"
# "Preparing 837 sequences for GPU processing..."
# "Launching GPU genus classification..."
# "CUDA taxonomy: Processing 53594 genera in 3 batches (18735 genera/batch)"
# "VRAM usage: Query data 1.2 MB, Batch data 4743.6 MB, Total 4744.8 MB / 5272.0 MB available"
# "  Processed batch 3/3 (100.0% complete)"
# "GPU genus classification complete!"
# "Attempting GPU-accelerated bootstrap evaluation..."
# "Launching GPU bootstrap evaluation..."
# "CUDA bootstrap: Processing 100 iterations across 53594 genera in 7 batches"
# "  Bootstrap batch 5/7 (71.4% complete)"
# "  Bootstrap batch 7/7 (100.0% complete)"
# "GPU bootstrap evaluation complete!"
```

### Controlling Thread Count

```r
# Limit OpenMP threads if needed (e.g., on shared servers)
Sys.setenv(OMP_NUM_THREADS = "8")

# Limit RcppParallel threads
options(RcppParallel.thread.num = 8)
```



## Troubleshooting

### CUDA Not Being Used

**Check if CUDA compiled**:
```bash
# Look for CUDA mentions during installation
R CMD INSTALL dada2_1.36.0.tar.gz 2>&1 | grep CUDA
```

**If "CUDA not found"**:
1. Verify CUDA installed: `nvcc --version`
2. Check PATH: `echo $PATH | grep cuda`
3. Set explicitly: `export CUDA_HOME=/usr/local/cuda`
4. Rebuild package

**Test CUDA from R**:
```r
library(dada2)
# Run with verbose
taxa <- assignTaxonomy(seqtab[1:10,], refFasta, verbose=TRUE)
# Should say "CUDA detected" if working
```

### CUDA Out of Memory

**Symptoms**: "CUDA error: out of memory"

**Solutions**:
1. Code automatically adjusts batch size, but can force smaller:
   - Edit `src/taxonomy_cuda.cu`
   - Reduce `ngenus_per_batch` calculation
2. Check GPU isn't being used by other processes:
   ```bash
   nvidia-smi  # Shows GPU memory usage
   ```



## Technical Details

### Files Modified/Created

#### CPU Optimisations
- `src/kmers.cpp`: AVX2/AVX-512 implementations
- `src/dada.h`: Function declarations, aligned malloc
- `src/cpu_detect.cpp/h`: CPU feature detection
- `src/cluster.cpp`: Vector pre-reservation
- `src/containers.cpp`: Buffer sizes
- `src/taxonomy.cpp`: OpenMP parallelisation
- `src/nwalign_*.cpp`: Aligned malloc
- `src/dp_workspace.cpp`: Workspace pooling (infrastructure)
- `src/Makevars`: Compiler flags

#### GPU Optimisations
- `src/taxonomy_cuda.cu`: CUDA kernels
- `src/taxonomy_cuda_wrapper.h`: C interface
- `src/taxonomy.cpp`: CUDA integration
- `src/Makevars`: Conditional CUDA compilation

### Compiler Flags

**Active** (CPU):
- `-O3`: Maximum optimisation
- `-mavx2`: Enable AVX2 SIMD
- `-mavx512f -mavx512bw`: Enable AVX-512 (if supported)
- `-fopenmp`: Enable OpenMP threading

**When CUDA enabled**:
- `-DHAVE_CUDA`: Enable CUDA code paths
- NVCC flags: `-O3 --compiler-options -fPIC`
- GPU architectures: SM 5.0, 6.0, 7.0, 7.5 (configurable)



## Credits

My laptop for taking hours to run assignTaxonomy(), now it can finish in minutes. Poor thing.

Anthropic's Claude for just about anything and everything that generative AI was capable of supporting this project.

Based on original DADA2 algorithm by Callahan, B.J., McMurdie, P.J., Rosen, M.J., Han, A.W., Johnson, A.J.A. & Holmes, S.P.



## References

Callahan, B.J., McMurdie, P.J., Rosen, M.J., Han, A.W., Johnson, A.J.A. & Holmes, S.P., 2016. DADA2: High‑resolution sample inference from Illumina amplicon data. Nature Methods, 13, pp.581‑583. DOI: https://doi.org/10.1038/nmeth.3869.



## License

These optimisations maintain the same license as DADA2 package.



Last updated: 2026-02-01
Version: DADA2 1.36.0-optimised
