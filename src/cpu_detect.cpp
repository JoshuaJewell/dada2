#include "dada.h"
#include "cpu_detect.h"

#ifdef __x86_64
#include <cpuid.h>
#endif

static CPUFeatures g_cpu_features = {false, false, false, false};
static bool g_cpu_detected = false;

CPUFeatures detect_cpu_features() {
  CPUFeatures feat = {false, false, false, false};

#ifdef __x86_64
  unsigned int eax, ebx, ecx, edx;

  // Check for SSE2 (CPUID function 1, EDX bit 26)
  if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
    feat.sse2 = (edx & (1 << 26)) != 0;
  }

  // Check for AVX2 and AVX-512 (CPUID function 7, subleaf 0)
  if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
    feat.avx2 = (ebx & (1 << 5)) != 0;
    feat.avx512f = (ebx & (1 << 16)) != 0;
    feat.avx512bw = (ebx & (1 << 30)) != 0;
  }
#endif

  return feat;
}

void initialize_cpu_detection() {
  if (!g_cpu_detected) {
    g_cpu_features = detect_cpu_features();
    g_cpu_detected = true;
  }
}

int get_optimal_simd_level() {
  initialize_cpu_detection();

  // Return the highest available SIMD level
  if (g_cpu_features.avx512f && g_cpu_features.avx512bw) return 4;
  if (g_cpu_features.avx2) return 3;
  if (g_cpu_features.sse2) return 2;
  return 0;
}
