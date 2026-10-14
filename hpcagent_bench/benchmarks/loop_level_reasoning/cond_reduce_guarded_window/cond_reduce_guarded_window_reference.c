/* Hand-written C23 reference of ``cond_reduce_guarded_window`` under the v2 C-ABI, fp64 variant.
 *
 * Written by SPCL (ETH Zurich) for HPCAgent-Bench, a sibling of the TSVC-2.5 extension loop
 * cond_reduce_sum.
 *
 * DELIBERATELY CARRIES NO ``hpcagent_bench-autogen`` MARKER. emit_io treats an unmarked
 * reference as a hand-written override and never regenerates it, which is the point: this
 * corpus exists to ask whether compilers vectorize and parallelize human-written C where they
 * fail on translator-generated C. Edit it by hand, never through the emitter.
 *
 * The numpy reference remains the correctness oracle. */

#include <stdint.h>

void cond_reduce_guarded_window_fp64(const double *restrict a, const double *restrict b, double *restrict out,
                                     const int64_t K, const int64_t LEN_1D) {
  out[0] = 0.0;
  for (int64_t i = 0; i < LEN_1D; ++i) {
    if (i + K < LEN_1D) {
      out[0] = out[0] + a[i + K] * b[i];
    }
  }
}
