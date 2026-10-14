/* C++ baseline reference for HPCAgent-Bench kernel bout_arakawa, emitted by HPCAgent-Bench's NumpyToX C++ translator (numpyto_cpp) from the numpy reference. The v2 C-ABI carries no timer. Not the scoring oracle -- the numpy reference remains the correctness oracle. */

// hpcagent_bench-autogen -- generated from bout_arakawa_numpy.py; edit the numpy reference and regenerate, or delete this line to keep local edits as a hand override.
#include <cstdint>
#include <cmath>
#include <type_traits>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#ifndef NPB_HD
#if defined(__HIPCC__) || defined(__CUDACC__)
#define NPB_HD __host__ __device__
#else
#define NPB_HD
#endif
#endif
/* A refused ``malloc`` (an armed ``RLIMIT_DATA`` cap, an exhausted node) aborts naming the
 * array and byte count instead of surfacing as a SIGSEGV at the first write. A function, so
 * the kernel body stays free of control flow pet / pluto would have to model. [[maybe_unused]]:
 * clang's -Wunused-function flags an uncalled static inline in C++ (not in C), and a kernel
 * without a heap allocation never calls it. */
[[maybe_unused]] static inline void __npb_alloc_check(const void *p, std::size_t nbytes, const char *name) {
    if (p == nullptr && nbytes != 0) {
        std::fprintf(stderr, "out of memory: %zu bytes for %s\n", nbytes, name);
        std::abort();
    }
}
// Math constants as typed constexpr values. ``<cmath>`` may
// predefine M_PI / M_E as macros (glibc __USE_MISC); undefine
// them so the names rebind to our constexpr values -- we emit no
// macro DEFINITION, only remove the platform ones.
// [[maybe_unused]]: namespace-scope constexpr has internal linkage, so a
// kernel that references neither draws -Wunused-const-variable from clang
// (the C prelude spells these as macros and never does). They are prelude
// vocabulary offered to every kernel, which is exactly this attribute.
#ifdef M_PI
#undef M_PI
#endif
#ifdef M_E
#undef M_E
#endif
[[maybe_unused]] constexpr double M_PI = 3.14159265358979323846;
[[maybe_unused]] constexpr double M_E  = 2.71828182845904523536;
// Complex support via the GCC/Clang ``double _Complex`` extension
// (no <complex.h>, so no name clashes). The imaginary unit and
// the C99-named helpers are constexpr/inline FUNCTIONS, not macros.
constexpr NPB_HD double creal(double _Complex z) { return __real__ z; }
constexpr NPB_HD double cimag(double _Complex z) { return __imag__ z; }
inline NPB_HD double _Complex __npb_make_complex(double re, double im) {
    double _Complex z; __real__ z = re; __imag__ z = im; return z;
}
static const double _Complex _Complex_I = __npb_make_complex(0.0, 1.0);
inline NPB_HD double cabs(double _Complex z) {
    return sqrt(creal(z)*creal(z) + cimag(z)*cimag(z));
}
inline NPB_HD double carg(double _Complex z) { return atan2(cimag(z), creal(z)); }
/* ``cexp(z) = exp(re) * (cos(im) + i*sin(im))``. */
inline NPB_HD double _Complex cexp(double _Complex z) {
    return __npb_make_complex(exp(creal(z))*cos(cimag(z)),
                             exp(creal(z))*sin(cimag(z)));
}
/* ``clog(z) = log(|z|) + i*arg(z)``. */
inline NPB_HD double _Complex clog(double _Complex z) {
    return __npb_make_complex(log(cabs(z)), carg(z));
}
/* ``csqrt(z) = exp((1/2) * log(z))`` -- principal branch. */
inline NPB_HD double _Complex csqrt(double _Complex z) {
    double _Complex l = clog(z);
    return cexp(__npb_make_complex(0.5*creal(l), 0.5*cimag(l)));
}
/* ``cpow(z, w) = exp(w * log(z))`` -- general complex pow. */
inline NPB_HD double _Complex cpow(double _Complex z, double _Complex w) {
    double _Complex l = clog(z);
    return cexp(__npb_make_complex(
        creal(w)*creal(l) - cimag(w)*cimag(l),
        creal(w)*cimag(l) + cimag(w)*creal(l)));
}
/* ``z.conjugate()`` -- complex-conjugate scalar helper. */
inline NPB_HD double _Complex __npb_conj(double _Complex z) {
    return __npb_make_complex(creal(z), -cimag(z));
}
/* Integer power for VLA shape bounds. */
constexpr NPB_HD int64_t __npb_int_pow(int64_t base, int64_t exp) {
    int64_t result = 1;
    while (exp > 0) {
        if (exp & 1) result *= base;
        base *= base;
        exp >>= 1;
    }
    return result;
}
/* Ternary-form ``max`` / ``min`` as constexpr function templates
 * so a mixed call like ``max(double, int)`` promotes the int
 * operand via the usual arithmetic conversions (``std::max``
 * would require both args to share a type). They PROPAGATE NaN (a
 * NaN in EITHER operand yields NaN): these serve the elementwise
 * ``np.maximum``/``np.minimum`` broadcast and the ``np.maximum.at`` /
 * ``np.minimum.at`` scatter folds, which follow numpy (propagate),
 * not Python builtin max. For finite operands the result is the
 * larger/smaller -- so the 3-way builtin max (needleman_wunsch,
 * always finite) is unchanged; integer NaN tests are dead. */
template <class A, class B>
constexpr NPB_HD auto max(A a, B b) { return a != a ? a : (b != b ? b : (b > a ? b : a)); }
template <class A, class B>
constexpr NPB_HD auto min(A a, B b) { return a != a ? a : (b != b ? b : (b < a ? b : a)); }
/* Elementwise ``np.maximum``/``np.minimum`` lower to ``fmax``/``fmin``;
 * libm ``fmax``/``fmin`` SUPPRESS NaN but numpy PROPAGATES it. These
 * single-evaluation helpers return NaN when either operand is NaN.
 * Integral operands take the exact integer compare (the same INTEGRAL/floating
 * split int_floor makes): converting them to double rounds anything above 2**53,
 * so min(2**53 + 1, 2**53 + 2) came back 2**53 -- a value neither operand had. */
template <class A, class B>
constexpr NPB_HD auto __npb_fmax(A a, B b) {
    if constexpr (std::is_integral_v<A> && std::is_integral_v<B>) {
        return a > b ? a : b;
    } else {
        return a != a ? a : (b != b ? b : (a > b ? a : b));
    }
}
template <class A, class B>
constexpr NPB_HD auto __npb_fmin(A a, B b) {
    if constexpr (std::is_integral_v<A> && std::is_integral_v<B>) {
        return a < b ? a : b;
    } else {
        return a != a ? a : (b != b ? b : (a < b ? a : b));
    }
}
/* ``np.sign``: numpy ``sign(nan) == nan`` and ``sign(0) == 0``. The
 * naive ``(x>0)-(x<0)`` gives 0 for NaN and evaluates ``x`` twice. */
inline NPB_HD double __npb_sign(double x) {
    return x != x ? x : (double)((x > 0) - (x < 0));
}
/* Python ``//`` floors toward -inf; C++ ``/`` truncates toward zero.
 * C++ has no built-in floor-division, so it is always this helper. The
 * INTEGRAL/floating split is decided by the operand TYPE here rather than
 * inferred from the source AST -- guessing it wrong emitted a no-op floor
 * over an already-truncated integer quotient. */
template <class A, class B>
constexpr NPB_HD auto int_floor(A a, B b) {
    if constexpr (std::is_integral_v<A> && std::is_integral_v<B>) {
        return a / b - ((a % b != 0) && ((a < 0) ^ (b < 0)));
    } else {
        return std::floor(static_cast<double>(a) / static_cast<double>(b));
    }
}
/* Ceil-division counterpart (toward +inf), exact for both signs -- unlike
 * the ``(a + b - 1) / b`` idiom, which holds only for a positive divisor
 * and overflows near the integer maximum. */
template <class A, class B>
constexpr NPB_HD auto int_ceil(A a, B b) {
    if constexpr (std::is_integral_v<A> && std::is_integral_v<B>) {
        return a / b + ((a % b != 0) && ((a < 0) == (b < 0)));
    } else {
        return std::ceil(static_cast<double>(a) / static_cast<double>(b));
    }
}
/* Python ``%`` returns the sign of the divisor; C/C++ the dividend.
 * Same type-dispatch as int_floor (floating operands need npy_remainder,
 * which the integer form cannot express on doubles). */
template <class A, class B>
constexpr NPB_HD auto python_mod(A a, B b) {
    if constexpr (std::is_integral_v<A> && std::is_integral_v<B>) {
        return (a % b + b) % b;
    } else {
        double m = std::fmod(static_cast<double>(a), static_cast<double>(b));
        if (m != 0.0 && ((b < 0.0) != (m < 0.0))) m += static_cast<double>(b);
        return m;
    }
}
/* Floating-point ``%``: numpy floored modulo (sign of the divisor),
 * which integer ``python_mod`` cannot express on doubles. Mirrors
 * numpy ``npy_remainder`` (fmod + sign-of-divisor fixup). */
inline NPB_HD double python_fmod(double a, double b) {
    double m = std::fmod(a, b);
    if (m != 0.0 && ((b < 0.0) != (m < 0.0))) m += b;
    return m;
}

extern "C" {

static void arakawa_bracket(const double *__restrict__ dx, const double *__restrict__ dz, const double *__restrict__ f, const double *__restrict__ g, double *__restrict__ result, const int64_t NX, const int64_t NY, const int64_t NZ);

static void arakawa_bracket(const double *__restrict__ dx, const double *__restrict__ dz, const double *__restrict__ f, const double *__restrict__ g, double *__restrict__ result, const int64_t NX, const int64_t NY, const int64_t NZ) {
    double jpp_lo;
    double jpx_lo;
    double jxp_lo;
    double jpp_hi;
    double jpx_hi;
    double jxp_hi;
    double *jpp_mid = (double *)malloc((size_t)(((NZ - 2))) * sizeof(double));
    __npb_alloc_check(jpp_mid, (size_t)(((NZ - 2))) * sizeof(double), "jpp_mid");
    double *jpx_mid = (double *)malloc((size_t)((((NZ - 1) - 1))) * sizeof(double));
    __npb_alloc_check(jpx_mid, (size_t)((((NZ - 1) - 1))) * sizeof(double), "jpx_mid");
    double *jxp_mid = (double *)malloc((size_t)(((NZ - 2))) * sizeof(double));
    __npb_alloc_check(jxp_mid, (size_t)(((NZ - 2))) * sizeof(double), "jxp_mid");
    for (int64_t jx = 1; jx < (NX - 1); ++jx) {
      for (int64_t jy = 0; jy < NY; ++jy) {
        jpp_lo = (((f[((jx)*(NY) + (jy))*(NZ) + (1)] - f[((jx)*(NY) + (jy))*(NZ) + ((NZ - 1))]) * (g[(((jx + 1))*(NY) + (jy))*(NZ) + (0)] - g[(((jx - 1))*(NY) + (jy))*(NZ) + (0)])) - ((f[(((jx + 1))*(NY) + (jy))*(NZ) + (0)] - f[(((jx - 1))*(NY) + (jy))*(NZ) + (0)]) * (g[((jx)*(NY) + (jy))*(NZ) + (1)] - g[((jx)*(NY) + (jy))*(NZ) + ((NZ - 1))])));
        jpx_lo = ((((g[(((jx + 1))*(NY) + (jy))*(NZ) + (0)] * (f[(((jx + 1))*(NY) + (jy))*(NZ) + (1)] - f[(((jx + 1))*(NY) + (jy))*(NZ) + ((NZ - 1))])) - (g[(((jx - 1))*(NY) + (jy))*(NZ) + (0)] * (f[(((jx - 1))*(NY) + (jy))*(NZ) + (1)] - f[(((jx - 1))*(NY) + (jy))*(NZ) + ((NZ - 1))]))) - (g[((jx)*(NY) + (jy))*(NZ) + (1)] * (f[(((jx + 1))*(NY) + (jy))*(NZ) + (1)] - f[(((jx - 1))*(NY) + (jy))*(NZ) + (1)]))) + (g[((jx)*(NY) + (jy))*(NZ) + ((NZ - 1))] * (f[(((jx + 1))*(NY) + (jy))*(NZ) + ((NZ - 1))] - f[(((jx - 1))*(NY) + (jy))*(NZ) + ((NZ - 1))])));
        jxp_lo = ((((g[(((jx + 1))*(NY) + (jy))*(NZ) + (1)] * (f[((jx)*(NY) + (jy))*(NZ) + (1)] - f[(((jx + 1))*(NY) + (jy))*(NZ) + (0)])) - (g[(((jx - 1))*(NY) + (jy))*(NZ) + ((NZ - 1))] * (f[(((jx - 1))*(NY) + (jy))*(NZ) + (0)] - f[((jx)*(NY) + (jy))*(NZ) + ((NZ - 1))]))) - (g[(((jx - 1))*(NY) + (jy))*(NZ) + (1)] * (f[((jx)*(NY) + (jy))*(NZ) + (1)] - f[(((jx - 1))*(NY) + (jy))*(NZ) + (0)]))) + (g[(((jx + 1))*(NY) + (jy))*(NZ) + ((NZ - 1))] * (f[(((jx + 1))*(NY) + (jy))*(NZ) + (0)] - f[((jx)*(NY) + (jy))*(NZ) + ((NZ - 1))])));
        result[((jx)*(NY) + (jy))*(NZ) + (0)] = (((jpp_lo + jpx_lo) + jxp_lo) * (1.0 / ((12.0 * dz[(jx)*(NY) + (jy)]) * dx[(jx)*(NY) + (jy)])));
        for (int64_t __w0 = 0; __w0 < (NZ - 2); ++__w0) {
          jpp_mid[__w0] = (((f[((jx)*(NY) + (jy))*(NZ) + ((__w0 + 2))] - f[((jx)*(NY) + (jy))*(NZ) + (__w0)]) * (g[(((jx + 1))*(NY) + (jy))*(NZ) + ((__w0 + 1))] - g[(((jx - 1))*(NY) + (jy))*(NZ) + ((__w0 + 1))])) - ((f[(((jx + 1))*(NY) + (jy))*(NZ) + ((__w0 + 1))] - f[(((jx - 1))*(NY) + (jy))*(NZ) + ((__w0 + 1))]) * (g[((jx)*(NY) + (jy))*(NZ) + ((__w0 + 2))] - g[((jx)*(NY) + (jy))*(NZ) + (__w0)])));
        }
        for (int64_t __w0 = 0; __w0 < ((NZ - 1) - 1); ++__w0) {
          jpx_mid[__w0] = ((((g[(((jx + 1))*(NY) + (jy))*(NZ) + ((__w0 + 1))] * (f[(((jx + 1))*(NY) + (jy))*(NZ) + ((__w0 + 2))] - f[(((jx + 1))*(NY) + (jy))*(NZ) + (__w0)])) - (g[(((jx - 1))*(NY) + (jy))*(NZ) + ((__w0 + 1))] * (f[(((jx - 1))*(NY) + (jy))*(NZ) + ((__w0 + 2))] - f[(((jx - 1))*(NY) + (jy))*(NZ) + (__w0)]))) - (g[((jx)*(NY) + (jy))*(NZ) + ((__w0 + 2))] * (f[(((jx + 1))*(NY) + (jy))*(NZ) + ((__w0 + 2))] - f[(((jx - 1))*(NY) + (jy))*(NZ) + ((__w0 + 2))]))) + (g[((jx)*(NY) + (jy))*(NZ) + (__w0)] * (f[(((jx + 1))*(NY) + (jy))*(NZ) + (__w0)] - f[(((jx - 1))*(NY) + (jy))*(NZ) + (__w0)])));
        }
        for (int64_t __w0 = 0; __w0 < (NZ - 2); ++__w0) {
          jxp_mid[__w0] = ((((g[(((jx + 1))*(NY) + (jy))*(NZ) + ((__w0 + 2))] * (f[((jx)*(NY) + (jy))*(NZ) + ((__w0 + 2))] - f[(((jx + 1))*(NY) + (jy))*(NZ) + ((__w0 + 1))])) - (g[(((jx - 1))*(NY) + (jy))*(NZ) + (__w0)] * (f[(((jx - 1))*(NY) + (jy))*(NZ) + ((__w0 + 1))] - f[((jx)*(NY) + (jy))*(NZ) + (__w0)]))) - (g[(((jx - 1))*(NY) + (jy))*(NZ) + ((__w0 + 2))] * (f[((jx)*(NY) + (jy))*(NZ) + ((__w0 + 2))] - f[(((jx - 1))*(NY) + (jy))*(NZ) + ((__w0 + 1))]))) + (g[(((jx + 1))*(NY) + (jy))*(NZ) + (__w0)] * (f[(((jx + 1))*(NY) + (jy))*(NZ) + ((__w0 + 1))] - f[((jx)*(NY) + (jy))*(NZ) + (__w0)])));
        }
        for (int64_t si2 = 1; si2 < (NZ - 1); ++si2) {
          result[((jx)*(NY) + (jy))*(NZ) + (si2)] = (((jpp_mid[(si2 - 1)] + jpx_mid[(si2 - 1)]) + jxp_mid[(si2 - 1)]) * (1.0 / ((12.0 * dz[(jx)*(NY) + (jy)]) * dx[(jx)*(NY) + (jy)])));
        }
        jpp_hi = (((f[((jx)*(NY) + (jy))*(NZ) + (0)] - f[((jx)*(NY) + (jy))*(NZ) + ((NZ - 2))]) * (g[(((jx + 1))*(NY) + (jy))*(NZ) + ((NZ - 1))] - g[(((jx - 1))*(NY) + (jy))*(NZ) + ((NZ - 1))])) - ((f[(((jx + 1))*(NY) + (jy))*(NZ) + ((NZ - 1))] - f[(((jx - 1))*(NY) + (jy))*(NZ) + ((NZ - 1))]) * (g[((jx)*(NY) + (jy))*(NZ) + (0)] - g[((jx)*(NY) + (jy))*(NZ) + ((NZ - 2))])));
        jpx_hi = ((((g[(((jx + 1))*(NY) + (jy))*(NZ) + ((NZ - 1))] * (f[(((jx + 1))*(NY) + (jy))*(NZ) + (0)] - f[(((jx + 1))*(NY) + (jy))*(NZ) + ((NZ - 2))])) - (g[(((jx - 1))*(NY) + (jy))*(NZ) + ((NZ - 1))] * (f[(((jx - 1))*(NY) + (jy))*(NZ) + (0)] - f[(((jx - 1))*(NY) + (jy))*(NZ) + ((NZ - 2))]))) - (g[((jx)*(NY) + (jy))*(NZ) + (0)] * (f[(((jx + 1))*(NY) + (jy))*(NZ) + (0)] - f[(((jx - 1))*(NY) + (jy))*(NZ) + (0)]))) + (g[((jx)*(NY) + (jy))*(NZ) + ((NZ - 2))] * (f[(((jx + 1))*(NY) + (jy))*(NZ) + ((NZ - 2))] - f[(((jx - 1))*(NY) + (jy))*(NZ) + ((NZ - 2))])));
        jxp_hi = ((((g[(((jx + 1))*(NY) + (jy))*(NZ) + (0)] * (f[((jx)*(NY) + (jy))*(NZ) + (0)] - f[(((jx + 1))*(NY) + (jy))*(NZ) + ((NZ - 1))])) - (g[(((jx - 1))*(NY) + (jy))*(NZ) + ((NZ - 2))] * (f[(((jx - 1))*(NY) + (jy))*(NZ) + ((NZ - 1))] - f[((jx)*(NY) + (jy))*(NZ) + ((NZ - 2))]))) - (g[(((jx - 1))*(NY) + (jy))*(NZ) + (0)] * (f[((jx)*(NY) + (jy))*(NZ) + (0)] - f[(((jx - 1))*(NY) + (jy))*(NZ) + ((NZ - 1))]))) + (g[(((jx + 1))*(NY) + (jy))*(NZ) + ((NZ - 2))] * (f[(((jx + 1))*(NY) + (jy))*(NZ) + ((NZ - 1))] - f[((jx)*(NY) + (jy))*(NZ) + ((NZ - 2))])));
        result[((jx)*(NY) + (jy))*(NZ) + ((NZ - 1))] = (((jpp_hi + jpx_hi) + jxp_hi) * (1.0 / ((12.0 * dz[(jx)*(NY) + (jy)]) * dx[(jx)*(NY) + (jy)])));
      }
    }
    free(jpp_mid);
    free(jpx_mid);
    free(jxp_mid);
}

void bout_arakawa_fp64(const double *__restrict__ dx, const double *__restrict__ dz, double *__restrict__ f, const double *__restrict__ g, double *__restrict__ result, const int64_t NX, const int64_t NY, const int64_t NZ, const int64_t nsteps) {
        for (int64_t _step = 0; _step < nsteps; ++_step) {
          arakawa_bracket(dx, dz, f, g, result, NX, NY, NZ);
          for (int64_t si0 = 1; si0 < (NX - 1); ++si0) {
            for (int64_t si1 = 0; si1 < NY; ++si1) {
              for (int64_t si2 = 0; si2 < NZ; ++si2) {
                f[((si0)*(NY) + (si1))*(NZ) + (si2)] += (0.001 * result[((si0)*(NY) + (si1))*(NZ) + (si2)]);
              }
            }
          }
        }
}
} // extern "C"
