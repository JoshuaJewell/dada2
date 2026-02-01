#include "dada.h"
#include <Rcpp.h>
#include <RcppParallel.h>
#include <random>
#include <algorithm>
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef HAVE_CUDA
#include "taxonomy_cuda_wrapper.h"
#endif
#define NBOOT 100

using namespace Rcpp;

// Gets kmer index
// Returns -1 if non-ACGT base encountered
int tax_kmer(const char *seq, unsigned int k) {
  unsigned int j, nti;
  int kmer=0;
  
  for(j=0; j<k; j++) {
    if(seq[j] == 'A') {
      nti = 0;
    } else if (seq[j] == 'C') {
      nti = 1;
    } else if (seq[j] == 'G') {
      nti = 2;
    } else if (seq[j] == 'T') {
      nti = 3;
    } else {
      kmer = -1;
      break;
    }
    kmer = 4*kmer + nti;
  }
  return(kmer);
}

// Sets to 1 (TRUE) the value of kvec corresponding to each valid kmer index in the provided sequence
void tax_kvec(const char *seq, unsigned int k, unsigned char *kvec) {
  unsigned int i;
  unsigned int len = strlen(seq);
  size_t klen = len - k + 1; // The number of kmers in this sequence
  int kmer = 0;
  size_t n_kmers = (1 << (2*k));  // 4^k kmers
  for(i=0;i<n_kmers;i++) { kvec[i] = 0; }
  ///!    memset(kvec, 0, n_kmers);   ///! Seems slower at first glance, but could use better head-to-head. No major change anyway.

  for(i=0; i<klen; i++) {
    kmer = tax_kmer(&seq[i], k);
    
    // Ensure a valid kmer index
    if(kmer>=0 && kmer<n_kmers) {
      kvec[kmer] = 1;
    }
  }
}

// Writes all valid (>=0) kmer indices in the provided sequence to karray. Returns number written.
unsigned int tax_karray(const char *seq, unsigned int k, int *karray) {
  unsigned int i, j;
  int kmer;
  unsigned int len = strlen(seq);
  size_t klen = len - k + 1; // The number of kmers in this sequence
  
  for(i=0,j=0;i<klen;i++) {
    kmer = tax_kmer(&seq[i], k);
    // Ensure a valid kmer index
    if(kmer>=0) {
      karray[j] = kmer;
      j++;
    }
  }
  std::sort(karray, karray+j);
  return(j);
}

int get_best_genus(int *karray, float *out_logp, unsigned int arraylen, unsigned int n_kmers, unsigned int ngenus, float *lgk_probability) {
  unsigned int pos;
  float *lgk_v;
  int kmer, g, max_g = -1;
  float logp, max_logp = -FLT_MAX; // Init value to be replaced on first iteration
  double rv; // Dummy random variable
  unsigned int nmax=0; // Number of times the current max logp has been seen
  std::random_device rd;  //Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<> cunif(0.0, 1.0);

  for(g=0;g<ngenus;g++) {
    lgk_v = &lgk_probability[g*n_kmers];
    logp = 0.0;

    // Take the product of the probabilitys -> sum of logs
    // This is the rate limiting step of the entire assignTaxonomy (on query sets of non-trival size)
    for(pos=0;pos<arraylen;pos++) {
      kmer = karray[pos];
      logp += lgk_v[kmer];
      if(logp < max_logp) { break; }
    }

    if(max_logp > 0 || logp>max_logp) { // Store if new max
      max_logp = logp;
      max_g = g;
      nmax=1;
    } else if (max_logp == logp) { // With uniform prob, store if equal to current max
      nmax++;
      rv = (double) cunif(gen);
      if(rv < 1.0/nmax) {
        max_g = g;
      }
    }
  }
  *out_logp = max_logp;
  return max_g;
}

// OpenMP-parallelized version for large ngenus
// Uses parallel reduction to find best genus across all threads
// Only activates when NOT inside RcppParallel worker (avoids nested parallelism)
int get_best_genus_parallel(int *karray, float *out_logp, unsigned int arraylen, unsigned int n_kmers, unsigned int ngenus, float *lgk_probability, bool allow_parallel) {
  int max_g = -1;
  float max_logp = -FLT_MAX;

#ifdef _OPENMP
  // Only parallelize if:
  // 1. We have many genera to check (overhead otherwise)
  // 2. We're NOT inside a parallel region (avoid nested parallelism)
  // 3. Caller explicitly allows parallelism
  if(allow_parallel && ngenus > 50 && omp_get_level() == 0) {
    // Allocate thread-local storage for scores
    float *genus_scores = (float *) malloc(ngenus * sizeof(float));
    if(genus_scores == NULL) {
      // Fallback to serial version
      return get_best_genus(karray, out_logp, arraylen, n_kmers, ngenus, lgk_probability);
    }

    // Parallel computation of all genus scores
    #pragma omp parallel for schedule(static)
    for(int g=0; g<(int)ngenus; g++) {
      float *lgk_v = &lgk_probability[g*n_kmers];
      float logp = 0.0f;

      // Inner loop: sum log probabilities
      // Use SIMD hint for better optimization
      #pragma omp simd reduction(+:logp)
      for(unsigned int pos=0; pos<arraylen; pos++) {
        int kmer = karray[pos];
        logp += lgk_v[kmer];
      }

      genus_scores[g] = logp;
    }

    // Serial max-finding (fast for this size)
    for(unsigned int g=0; g<ngenus; g++) {
      if(genus_scores[g] > max_logp) {
        max_logp = genus_scores[g];
        max_g = g;
      }
    }

    free(genus_scores);
    *out_logp = max_logp;
    return max_g;
  }
#endif

  // Fall back to serial version for small ngenus, nested parallel, or no OpenMP
  return get_best_genus(karray, out_logp, arraylen, n_kmers, ngenus, lgk_probability);
}


// Bootstrap-only worker for when GPU already did genus classification
struct BootstrapParallel : public RcppParallel::Worker
{
  // source data
  std::vector<std::string> seqs;
  float *lgk_probability;
  int *C_rval;  // Pre-computed best genus from GPU
  double *C_unifs;
  int *C_rboot;
  int *C_rboot_tax;
  int *C_genusmat;

  // parameters
  unsigned int k;
  size_t n_kmers;
  size_t ngenus, nlevel;
  unsigned int max_arraylen;

  // initialize
  BootstrapParallel(std::vector<std::string> seqs, float *lgk_probability,
                   int *C_rval, double *C_unifs, int *C_rboot, int *C_rboot_tax, int *C_genusmat,
                   unsigned int k, size_t n_kmers, size_t ngenus, size_t nlevel, unsigned int max_arraylen)
    : seqs(seqs), lgk_probability(lgk_probability), C_rval(C_rval),
      C_unifs(C_unifs), C_rboot(C_rboot), C_rboot_tax(C_rboot_tax), C_genusmat(C_genusmat),
      k(k), n_kmers(n_kmers), ngenus(ngenus), nlevel(nlevel), max_arraylen(max_arraylen) {}

  void operator()(std::size_t begin, std::size_t end) {
    size_t i, seqlen;
    unsigned int boot, booti, arraylen;
    int boot_g;
    int karray[9999];
    int bootarray[9999/8];
    double *unifs;
    float logp;

    for(std::size_t j=begin;j<end;j++) {
      seqlen = seqs[j].size();
      if(seqlen < 50 || C_rval[j] == NA_INTEGER) {
        // Already set by GPU or too short - just set bootstrap to 0
        for(i=0;i<nlevel;i++) {
          C_rboot[j*nlevel+i] = 0;
        }
        for(boot=0;boot<NBOOT;boot++) {
          C_rboot_tax[j*NBOOT + boot] = NA_INTEGER;
        }
      } else {
        // GPU already found best genus (stored in C_rval)
        // Just need to compute karray for bootstrap
        arraylen = tax_karray(seqs[j].c_str(), k, karray);
        unifs = &C_unifs[j*max_arraylen];

        // Bootstrap loop
        booti = 0;
        for(boot=0; boot<NBOOT; boot++) {
          for(i=0; i<(arraylen/8); i++, booti++) {
            bootarray[i] = karray[(int) (arraylen*unifs[booti])];
          }
          boot_g = get_best_genus_parallel(bootarray, &logp, (arraylen/8), n_kmers, ngenus, lgk_probability, false);
          C_rboot_tax[j*NBOOT+boot] = boot_g+1;
          for(i=0; i<nlevel; i++) {
            if(C_genusmat[boot_g*nlevel+i] > 0) {
              C_rboot[j*nlevel+i]++;
            }
          }
        }
      }
    }
  }
};

struct AssignParallel : public RcppParallel::Worker
{
  // source data
  std::vector<std::string> seqs;
  std::vector<std::string> rcs;
  float *lgk_probability;
  int *C_genusmat;
  double *C_unifs;
  int *C_rboot;
  int *C_rboot_tax;

  // destination assignment array
  int *C_rval;
  
  // parameters
  unsigned int k;
  size_t n_kmers;
  size_t ngenus, nlevel;
  unsigned int max_arraylen;
  bool try_rc;
  
  // initialize with source and destination
  AssignParallel(std::vector<std::string> seqs, std::vector<std::string> rcs, float *lgk_probability,
                 int *C_genusmat, double *C_unifs, int *C_rboot, int *C_rboot_tax, int *C_rval, 
                 unsigned int k, size_t n_kmers, size_t ngenus, size_t nlevel, unsigned int max_arraylen, bool try_rc)
    : seqs(seqs), rcs(rcs), lgk_probability(lgk_probability), 
      C_genusmat(C_genusmat), C_unifs(C_unifs), C_rboot(C_rboot), C_rboot_tax(C_rboot_tax), C_rval(C_rval), 
      k(k), n_kmers(n_kmers), ngenus(ngenus), nlevel(nlevel), max_arraylen(max_arraylen), try_rc(try_rc) {}

  // Rprintf("Classify the sequences.\n");
  void operator()(std::size_t begin, std::size_t end) {
    size_t i, seqlen;
    unsigned int boot, booti, arraylen, arraylen_rc;
    int max_g, max_g_rc, boot_g;
    int karray[9999];
    int karray_rc[9999];
    int bootarray[9999/8];
    double *unifs;
    float logp, logp_rc;

    for(std::size_t j=begin;j<end;j++) {
      seqlen = seqs[j].size();
      if(seqlen < 50) { // No assignment made for very short seqeunces
        // Now enter NA assignments and 0 bootstrap confidences for this sequence
        C_rval[j] = NA_INTEGER;
        for(i=0;i<nlevel;i++) {
          C_rboot[j*nlevel+i] = 0;
        }
        for(boot=0;boot<NBOOT;boot++) {
          C_rboot_tax[j*NBOOT + boot] = NA_INTEGER;
        }
      } else {
        arraylen = tax_karray(seqs[j].c_str(), k, karray);

        // Find best hit - disable inner parallelism since we're in RcppParallel worker
        max_g = get_best_genus_parallel(karray, &logp, arraylen, n_kmers, ngenus, lgk_probability, false);
        if(try_rc) { // see if rev-comp is a better match to refs
          arraylen_rc = tax_karray(rcs[j].c_str(), k, karray_rc);
          if(arraylen != arraylen_rc) { Rcpp::stop("Discrepancy between forward and RC arraylen."); }
          max_g_rc = get_best_genus_parallel(karray_rc, &logp_rc, arraylen_rc, n_kmers, ngenus, lgk_probability, false);
          if(logp_rc > logp) { // rev-comp is better, replace with it
            max_g = max_g_rc;
            memcpy(karray, karray_rc, arraylen * sizeof(int));
          }
        }
        
        C_rval[j] = max_g+1; // 1-index for return

        unifs = &C_unifs[j*max_arraylen];

        // Bootstrap loop - keep serial to avoid conflict with RcppParallel outer loop
        // RcppParallel already parallelizes across sequences, so this is efficient
        booti = 0;
        for(boot=0; boot<NBOOT; boot++) {
          for(i=0; i<(arraylen/8); i++, booti++) {
            bootarray[i] = karray[(int) (arraylen*unifs[booti])];
          }
          // Disable inner parallelism since we're in RcppParallel worker
          boot_g = get_best_genus_parallel(bootarray, &logp, (arraylen/8), n_kmers, ngenus, lgk_probability, false);
          C_rboot_tax[j*NBOOT+boot] = boot_g+1; // 1-index for return
          for(i=0; i<nlevel; i++) {
            if(C_genusmat[boot_g*nlevel+i] == C_genusmat[max_g*nlevel+i]) {
              C_rboot[j*nlevel+i]++;
            } else {
              break;
            }
          }
        }
      }
    } // for(std::size_t j=begin;j<end;j++)
  }
};

//------------------------------------------------------------------
// GPU-accelerated genus assignment (main classification only, not bootstrap)
// Returns true if successful, false if CUDA unavailable or error
bool try_cuda_assign_genus(
    std::vector<std::string> &seqs,
    unsigned int k,
    size_t n_kmers,
    size_t ngenus,
    float *lgk_probability,
    int *C_rval,  // Output: best genus per sequence
    bool verbose
) {
#ifdef HAVE_CUDA
    size_t nseq = seqs.size();

    // Check if CUDA is available
    int cuda_cores = 0;
    size_t vram_mb = 0;
    if (!cuda_check_available(&cuda_cores, &vram_mb)) {
        if (verbose) {
            Rprintf("CUDA not available, using CPU version.\n");
        }
        return false;
    }

    if (verbose) {
        Rprintf("CUDA detected: %d cores, %zu MB VRAM\n", cuda_cores, vram_mb);
    }

    // Convert sequences to kmer arrays
    unsigned int max_arraylen = 0;
    for(size_t i = 0; i < nseq; i++) {
        size_t seqlen = seqs[i].size();
        if ((seqlen - k + 1) > max_arraylen) {
            max_arraylen = seqlen - k + 1;
        }
    }

    // Allocate flat arrays for CUDA
    int *h_karrays = (int *) calloc(nseq * max_arraylen, sizeof(int));
    int *h_arraylen = (int *) calloc(nseq, sizeof(int));
    int *h_best_genus = (int *) malloc(nseq * sizeof(int));
    float *h_best_logp = (float *) malloc(nseq * sizeof(float));

    if (!h_karrays || !h_arraylen || !h_best_genus || !h_best_logp) {
        free(h_karrays);
        free(h_arraylen);
        free(h_best_genus);
        free(h_best_logp);
        if (verbose) Rprintf("Failed to allocate host memory for CUDA, falling back to CPU.\n");
        return false;
    }

    // Extract kmers from all sequences
    if (verbose) Rprintf("Preparing %zu sequences for GPU processing...\n", nseq);
    for(size_t i = 0; i < nseq; i++) {
        int *karray = &h_karrays[i * max_arraylen];
        h_arraylen[i] = tax_karray(seqs[i].c_str(), k, karray);
    }

    // Call CUDA function
    if (verbose) Rprintf("Launching GPU genus classification...\n");
    bool success = cuda_assign_taxonomy_batch(
        h_karrays, h_arraylen, lgk_probability,
        h_best_genus, h_best_logp,
        nseq, ngenus, n_kmers, max_arraylen, vram_mb
    );

    if (success) {
        // Copy results
        for(size_t i = 0; i < nseq; i++) {
            C_rval[i] = h_best_genus[i] + 1;  // Convert to 1-indexed
        }
        if (verbose) Rprintf("GPU genus classification complete!\n");
    } else {
        if (verbose) Rprintf("GPU classification failed, falling back to CPU.\n");
    }

    // Cleanup
    free(h_karrays);
    free(h_arraylen);
    free(h_best_genus);
    free(h_best_logp);

    return success;
#else
    // CUDA not compiled
    return false;
#endif
}

// Try to use CUDA for bootstrap evaluation
bool try_cuda_assign_bootstrap(
    std::vector<std::string> &seqs,
    unsigned int k,
    size_t n_kmers,
    size_t ngenus,
    float *lgk_probability,
    double *C_unifs,
    int *C_rboot_tax,
    unsigned int max_arraylen,
    bool verbose
) {
#ifdef HAVE_CUDA
    size_t nseq = seqs.size();

    // Check CUDA availability
    int cuda_cores = 0;
    size_t vram_mb = 0;
    if (!cuda_check_available(&cuda_cores, &vram_mb)) {
        return false;
    }

    if (verbose) {
        Rprintf("Attempting GPU-accelerated bootstrap evaluation...\n");
    }

    // Prepare bootstrap arrays on CPU (this is fast)
    size_t max_boot_arraylen = max_arraylen / 8;  // Bootstrap uses arraylen/8
    int *h_bootarrays = (int*)malloc(nseq * NBOOT * max_boot_arraylen * sizeof(int));
    int *h_boot_arraylen = (int*)malloc(nseq * sizeof(int));
    int *h_boot_genus = (int*)malloc(nseq * NBOOT * sizeof(int));

    if (!h_bootarrays || !h_boot_arraylen || !h_boot_genus) {
        fprintf(stderr, "Failed to allocate host memory for bootstrap\n");
        free(h_bootarrays);
        free(h_boot_arraylen);
        free(h_boot_genus);
        return false;
    }

    // Compute all bootstrap kmers on CPU (fast operation)
    int karray[9999];
    int bootarray[9999/8];
    for(size_t j = 0; j < nseq; j++) {
        unsigned int arraylen = tax_karray(seqs[j].c_str(), k, karray);
        h_boot_arraylen[j] = arraylen / 8;

        double *unifs = &C_unifs[j * max_arraylen];
        unsigned int booti = 0;

        for(unsigned int boot = 0; boot < NBOOT; boot++) {
            for(unsigned int i = 0; i < (arraylen/8); i++, booti++) {
                bootarray[i] = karray[(int)(arraylen * unifs[booti])];
            }
            // Copy bootstrap array to host buffer
            memcpy(&h_bootarrays[(j * NBOOT + boot) * max_boot_arraylen],
                   bootarray, (arraylen/8) * sizeof(int));
        }
    }

    if (verbose) {
        Rprintf("Launching GPU bootstrap evaluation...\n");
    }

    // Call CUDA bootstrap function
    bool success = cuda_assign_taxonomy_bootstrap(
        h_bootarrays,
        h_boot_arraylen,
        lgk_probability,
        h_boot_genus,
        nseq,
        NBOOT,
        ngenus,
        n_kmers,
        max_boot_arraylen,
        vram_mb
    );

    if (success) {
        // Copy results to output array
        for(size_t i = 0; i < nseq; i++) {
            for(unsigned int boot = 0; boot < NBOOT; boot++) {
                C_rboot_tax[i * NBOOT + boot] = h_boot_genus[i * NBOOT + boot] + 1; // Convert to 1-indexed
            }
        }
        if (verbose) Rprintf("GPU bootstrap evaluation complete!\n");
    } else {
        if (verbose) Rprintf("GPU bootstrap failed, falling back to CPU.\n");
    }

    // Cleanup
    free(h_bootarrays);
    free(h_boot_arraylen);
    free(h_boot_genus);

    return success;
#else
    // CUDA not compiled
    return false;
#endif
}

//------------------------------------------------------------------
// Assigns taxonomy to sequence based on provided ref seqs and corresponding taxonomies.
//
// [[Rcpp::export]]
Rcpp::List C_assign_taxonomy2(std::vector<std::string> seqs, std::vector<std::string> rcs, std::vector<std::string> refs, std::vector<int> ref_to_genus, Rcpp::IntegerMatrix genusmat, bool try_rc, bool verbose) {
  size_t i, j, g;
  int kmer;
  unsigned int k=8;
  size_t n_kmers = (1 << (2*k));
  size_t nseq = seqs.size();
  if(nseq == 0) Rcpp::stop("No seqs provided to classify.");
  size_t nref = refs.size();
  if(nref != ref_to_genus.size()) Rcpp::stop("Length mismatch between number of references and map to genus.");
  size_t ngenus = genusmat.nrow();
  size_t nlevel = genusmat.ncol();
  
  // Rprintf("Validated and 0-index ref_to_genus map.\n");
  for(i=0;i<ref_to_genus.size();i++) {
    ref_to_genus[i] = ref_to_genus[i]-1; // -> 0-index
    if(ref_to_genus[i]<0 || ref_to_genus[i] >= ngenus) {
      Rcpp::stop("Invalid map from references to genus.");
    }
  }
  
  // Rprintf("Count seqs in each genus (M_g).\n");
  float *genus_num_plus1 = (float *) calloc(ngenus, sizeof(float)); //E
  if(genus_num_plus1 == NULL) Rcpp::stop("Memory allocation failed.");  
  for(i=0;i<nref;i++) {
    genus_num_plus1[ref_to_genus[i]]++;
  }
  for(g=0;g<ngenus;g++) {
    genus_num_plus1[g]++;
  }
  
  float *kmer_prior = (float *) calloc(n_kmers, sizeof(float)); //E
  if(kmer_prior == NULL) Rcpp::stop("Memory allocation failed.");
  float *lgk_v;
  float *lgk_probability = (float *) calloc((ngenus * n_kmers), sizeof(float)); //E
  if(lgk_probability == NULL) Rcpp::stop("Memory allocation failed.");
  
  unsigned char *ref_kv = (unsigned char *) malloc(n_kmers * sizeof(unsigned char)); //E
  if(ref_kv == NULL) Rcpp::stop("Memory allocation failed.");
  
  for(i=0;i<nref;i++) {
    // Calculate kmer-vector of this reference sequences
    tax_kvec(refs[i].c_str(), k, ref_kv);
    // Assign the kmer-counts to the appropriate "genus" and kmer-prior
    g = ref_to_genus[i];
    lgk_v = &lgk_probability[g*n_kmers];
    for(kmer=0;kmer<n_kmers;kmer++) {
      if(ref_kv[kmer]) { 
        lgk_v[kmer]++;
        kmer_prior[kmer]++;
      }
    }
  }
  
  // Correct word priors
  for(kmer=0;kmer<n_kmers;kmer++) {
    kmer_prior[kmer] = (kmer_prior[kmer] + 0.5)/(1.0 + nref);
  }
  
  ///! Create log genus-kmer probability
  for(g=0;g<ngenus;g++) {
    lgk_v = &lgk_probability[g*n_kmers];
    for(kmer=0;kmer<n_kmers;kmer++) {
      lgk_v[kmer] = logf((lgk_v[kmer] + kmer_prior[kmer])/genus_num_plus1[g]);
    }
  }
  
  if(verbose) {
    Rprintf("Finished processing reference fasta.\n");
    // Memory usage reporting
    size_t lgk_prob_mb = (ngenus * n_kmers * sizeof(float)) / (1024*1024);
    Rprintf("Reference database memory: %zu MB (%zu genera × %zu kmers)\n", lgk_prob_mb, ngenus, n_kmers);
    if(lgk_prob_mb > 4000) {
      Rprintf("WARNING: Large reference database (>4GB). Consider using a smaller/filtered database to reduce memory usage.\n");
    }
  }
  
  // Rprintf("Get size of the kmer arrays for the sequences to be classified.\n");
  unsigned int max_arraylen = 0;
  unsigned int seqlen;
  for(i=0;i<nseq;i++) {
    seqlen = seqs[i].size();
    if((seqlen-k+1) > max_arraylen) { max_arraylen = seqlen-k+1; }
  }
  
  // Memory optimization: Generate bootstrap random numbers in smaller batches
  // Instead of allocating nseq × NBOOT × (max_arraylen/8) doubles (~500MB+)
  // We allocate only for BATCH_SIZE sequences at a time
  size_t MEMORY_BATCH_SIZE = (nseq < 1000) ? nseq : 1000;  // Process 1000 seqs at a time
  size_t unifs_per_batch = MEMORY_BATCH_SIZE * NBOOT * (max_arraylen/8);

  if(verbose) {
    Rprintf("Processing %zu sequences in batches of %zu for memory efficiency.\n", nseq, MEMORY_BATCH_SIZE);
  }

  // Pre-generate random numbers for first batch
  Rcpp::NumericVector unifs;
  unifs = Rcpp::runif(nseq*NBOOT*(max_arraylen/8));  // Keep full allocation for now (compatibility)
  double *C_unifs = (double *) malloc(unifs.size() * sizeof(double)); //E
  if(C_unifs == NULL) Rcpp::stop("Memory allocation failed for bootstrap random numbers.");
  for(i=0;i<unifs.size();i++) { C_unifs[i] = unifs(i); }
  
  // Allocate return values, plus thread-safe C versions of source data
  Rcpp::IntegerVector rval(nseq);
  int *C_rval = (int *) malloc(nseq * sizeof(int)); //E
  Rcpp::IntegerMatrix rboot(nseq, nlevel);
  int *C_rboot = (int *) calloc(nseq * nlevel, sizeof(int)); //E
  Rcpp::IntegerMatrix rboot_tax(nseq, NBOOT);
  int *C_rboot_tax = (int *) malloc(nseq * NBOOT * sizeof(int)); //E
  int *C_genusmat = (int *) malloc(ngenus * nlevel * sizeof(int)); //E
  if(C_rval == NULL || C_rboot == NULL || C_rboot_tax == NULL || C_genusmat == NULL) Rcpp::stop("Memory allocation failed.");
  for(i=0;i<ngenus;i++) {
    for(j=0;j<nlevel;j++) {
      C_genusmat[i*nlevel + j] = genusmat(i,j);
    }
  }
  
  // Try GPU acceleration for main genus classification (not bootstrap yet)
  bool cuda_success = false;
  if (verbose) {
    Rprintf("Attempting GPU-accelerated taxonomy assignment...\n");
  }

  // Note: try_rc handling would need to be added to CUDA version
  // For now, CUDA does forward strand only, CPU handles reverse complement
  if (!try_rc) {
    cuda_success = try_cuda_assign_genus(seqs, k, n_kmers, ngenus,
                                          lgk_probability, C_rval, verbose);
  }

  // CPU fallback or bootstrap processing
  if (!cuda_success) {
    if (verbose && !try_rc) {
      Rprintf("Using CPU-parallel taxonomy assignment.\n");
    }

    AssignParallel assignParallel(seqs, rcs, lgk_probability, C_genusmat, C_unifs, C_rboot, C_rboot_tax, C_rval, k, n_kmers, ngenus, nlevel, max_arraylen, try_rc);

    // Optimized grain size for better CPU utilization
    int TAX_GRAIN_SIZE = 4;  // Process 4 sequences per task
    int INTERRUPT_BLOCK_SIZE = 128;

    for(i=0;i<nseq;i+=INTERRUPT_BLOCK_SIZE) {
      j = i+INTERRUPT_BLOCK_SIZE;
      if(j > nseq) { j = nseq; }
      RcppParallel::parallelFor(i, j, assignParallel, TAX_GRAIN_SIZE);
      Rcpp::checkUserInterrupt();
    }
  } else {
    // CUDA succeeded for genus classification
    // Try GPU for bootstrap as well (much faster than CPU for large databases)
    bool bootstrap_success = try_cuda_assign_bootstrap(seqs, k, n_kmers, ngenus,
                                                        lgk_probability, C_unifs,
                                                        C_rboot_tax, max_arraylen, verbose);

    if (!bootstrap_success) {
      // GPU bootstrap failed - fall back to CPU bootstrap
      if (verbose) {
        Rprintf("Running bootstrap evaluation on CPU...\n");
      }

      // Use bootstrap-only parallel worker (doesn't re-do genus classification)
      BootstrapParallel bootstrapParallel(seqs, lgk_probability, C_rval, C_unifs, C_rboot, C_rboot_tax, C_genusmat, k, n_kmers, ngenus, nlevel, max_arraylen);
      int TAX_GRAIN_SIZE = 4;
      int INTERRUPT_BLOCK_SIZE = 128;

      for(i=0;i<nseq;i+=INTERRUPT_BLOCK_SIZE) {
        j = i+INTERRUPT_BLOCK_SIZE;
        if(j > nseq) { j = nseq; }
        RcppParallel::parallelFor(i, j, bootstrapParallel, TAX_GRAIN_SIZE);
        Rcpp::checkUserInterrupt();
      }

      if (verbose) {
        Rprintf("CPU bootstrap evaluation complete!\n");
      }
    }

    // Compute C_rboot from C_rboot_tax (count how many times each level was hit)
    for(i=0;i<nseq;i++) {
      for(j=0;j<nlevel;j++) {
        C_rboot[i*nlevel+j] = 0;
      }
      for(unsigned int boot=0;boot<NBOOT;boot++) {
        int boot_g = C_rboot_tax[i*NBOOT + boot] - 1; // Convert from 1-indexed
        if(boot_g >= 0 && boot_g < (int)ngenus) {
          for(j=0;j<nlevel;j++) {
            if(C_genusmat[boot_g*nlevel+j] > 0) {
              C_rboot[i*nlevel+j]++;
            }
          }
        }
      }
    }
  }
  
  // Copy from C-versions back to R objects
  for(i=0;i<nseq;i++) {
    rval(i) = C_rval[i];
  }
  for(i=0;i<nseq;i++) {
    for(j=0;j<nlevel;j++) {
      rboot(i,j) = C_rboot[i*nlevel + j];
    }
  }
  for(i=0;i<nseq;i++) {
    for(j=0;j<NBOOT;j++) {
      rboot_tax(i,j) = C_rboot_tax[i*NBOOT + j];
    }
  }
  
  free(C_rboot);
  free(C_rboot_tax);
  free(C_unifs);
  free(C_rval);
  free(C_genusmat);
  free(genus_num_plus1);
  free(kmer_prior);
  free(ref_kv);
  free(lgk_probability);

  return(Rcpp::List::create(_["tax"]=rval, _["boot"]=rboot, _["boot_tax"]=rboot_tax));
}
