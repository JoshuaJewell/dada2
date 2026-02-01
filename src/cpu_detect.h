#ifndef _CPU_DETECT_H_
#define _CPU_DETECT_H_

struct CPUFeatures {
  bool sse2;
  bool avx2;
  bool avx512f;
  bool avx512bw;
};

void initialize_cpu_detection();
int get_optimal_simd_level();  // Returns: 0=none, 2=SSE2, 3=AVX2, 4=AVX512

#endif
