/*
 * Exception Theory — Pattern Engine Test Harness
 * ═══════════════════════════════════════════════════════════════════════
 *
 * Executable test harness for et_pattern_engine.c.
 *
 * Exercises ALL exported C functions with known inputs, verifies outputs,
 * and prints results. This file serves two purposes:
 *
 *   1. CLion project target: add_executable gives CLion full code insight
 *      for the .c file (resolves stdlib, enables navigation, etc.)
 *
 *   2. Verification: confirms the C engine produces correct results
 *      before the Python ctypes layer is involved.
 *
 * ET Derivation (Three Tools):
 *   Identification Principle: This file identifies ALL exported functions
 *   in et_pattern_engine.c and their correct behavior.
 *
 *   Descriptor Gap Principle: The gap was the missing executable target —
 *   CLion's T (code insight) could not traverse the project because the
 *   D-set (CMakeLists.txt) lacked add_executable. This file IS that
 *   missing Descriptor.
 *
 *   Subsumption Law: Every exported function is tested. Every test either
 *   passes or fails with a clear message. No function is left unverified.
 *   The test harness subsumes the entire public API without remainder.
 *
 * constexpr Convention (Descriptor Gap Principle):
 *   Test fixture variables that were originally declared `const` have been
 *   promoted to `constexpr` where all values are compile-time deterministic.
 *   The original `const` signified design-intent immutability — these are
 *   fixed test inputs whose values define the D-set of each test scenario.
 *   `constexpr` subsumes `const` without remainder (Subsumption Law): every
 *   constexpr variable is implicitly const, while additionally providing the
 *   compiler's T (optimization traversal) with the full compile-time D-set.
 *   Each promoted variable is annotated with its ET role in the test.
 *
 * P ∘ D ∘ T = E
 * Author: Michael James Muller — Aevum_Defluo
 */

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>     // IWYU pragma: keep — required for std::log2/round/fabs/isfinite portability

/* ── Import the pattern engine functions ────────────────────────────── */

#ifdef _WIN32
#define IMPORT __declspec(dllimport)
#else
#define IMPORT
#endif

#ifdef __cplusplus
extern "C" {
#endif

IMPORT int32_t *find_repeated_patterns(
    const int32_t *stream, int n,
    int min_len, int max_len, int min_count, int min_net_savings,
    int *out_n_patterns, int *out_buf_size);

IMPORT void build_k_stream(const uint8_t *data, int n,
                            const int32_t *byte_k_table, int32_t *k_out);

IMPORT void build_dk_stream(const int32_t *k_stream, int n, int32_t *dk_out);

IMPORT void gate_archetype_batch(const int32_t *patterns_buf, int n_patterns,
                                  int n_res, double incoherence_cents,
                                  uint8_t *out_mask);

IMPORT int subsume_greedy(int n, int n_archetypes,
                           const int32_t *arch_lengths, const int32_t *arch_n_pos,
                           const int32_t *arch_positions,
                           int32_t *placements, int32_t *used_mask);

IMPORT void free_buffer(int32_t *buf);

/* ── Tier 1: Curvature analysis (added) ──
 * Mirror of the C struct layout in et_pattern_engine.c. Field order MUST match
 * the C declaration so the C++ test harness reads the same bytes the C engine
 * writes. */
typedef struct {
    double curvature_mean;
    double curvature_variance;
    int32_t curvature_class;     /* 0=flat 1=elliptic 2=hyperbolic 3=variable 4=singular */
    double euler_characteristic;
    int32_t max_abs_curvature;
} CurvatureStats;

IMPORT void build_ddk_stream(const int32_t *dk_stream, int n,
                              int32_t *ddk_out);

IMPORT void compute_curvature_stats(const int32_t *ddk_stream, int n,
                                     int n_res, CurvatureStats *out);

IMPORT void compute_pattern_curvature(const int32_t *pattern_dk, int pat_len,
                                       double *out_curvature_mean,
                                       double *out_curvature_variance,
                                       double *out_geodesic_factor);

/* ── Tier 2: Geodesic residual coding (Mode 3) ── */
IMPORT void build_geodesic_residual(const int32_t *dk_stream, int n,
                                     int connection_order, int window_size,
                                     int32_t *residual_out,
                                     int32_t *gamma_out);

#ifdef __cplusplus
}
#endif

/* ── Test infrastructure ───────────────────────────────────────────── */

static int g_tests_run    = 0;
static int g_tests_passed = 0;

#define TEST(name) do { \
    g_tests_run++; \
    printf("  %-50s ", name); \
} while(0)

#define PASS() do { g_tests_passed++; printf("[PASS]\n"); } while(0)
#define FAIL(msg) printf("[FAIL] %s\n", msg)

/* ═══════════════════════════════════════════════════════════════════════
 * Test 1: find_repeated_patterns — known repeating stream
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_find_repeated_patterns()
{
    /* Stream: [1,2,3,1,2,3,4,5,1,2,3,4,5]
     * Pattern (1,2,3) occurs 3 times at positions 0, 3, 8
     * Pattern (1,2,3,4,5) occurs 2 times at positions 3, 8  */
    const int32_t stream[] = {1,2,3, 1,2,3, 4,5, 1,2,3, 4,5};
    constexpr int n = sizeof(stream) / sizeof(stream[0]);  /* was const int — stream D-count (element count of test fixture) */
    int n_patterns = 0, buf_size = 0;

    TEST("find_repeated_patterns: basic repeats");
    int32_t *result = find_repeated_patterns(stream, n, 2, n/2, 2, 1,
                                              &n_patterns, &buf_size);
    if (result && n_patterns > 0) {
        PASS();
    } else {
        FAIL("no patterns found in repeating stream");
    }
    printf("    Found %d patterns, buf_size=%d\n", n_patterns, buf_size);

    /* Parse and print first few */
    if (result && n_patterns > 0) {
        int pos = 1;
        for (int i = 0; i < n_patterns && i < 5; i++) {
            const int pat_len = result[pos++];
            const int occ_cnt = result[pos++];
            printf("    pat_len=%d, occ=%d, pattern=(", pat_len, occ_cnt);
            for (int j = 0; j < pat_len && j < 8; j++) {
                if (j > 0) printf(",");
                printf("%d", result[pos + j]);
            }
            pos += pat_len;
            printf("), positions=[");
            for (int j = 0; j < occ_cnt && j < 8; j++) {
                if (j > 0) printf(",");
                printf("%d", result[pos + j]);
            }
            pos += occ_cnt;
            printf("]\n");
        }
    }

    if (result) free_buffer(result);

    /* Trivial case: stream too small */
    TEST("find_repeated_patterns: trivial stream");
    constexpr int32_t tiny[] = {1, 2};  /* was const int32_t[] — minimal Δk stream (trivial D-set, verifies empty-result path) */
    result = find_repeated_patterns(tiny, 2, 2, 1, 2, 1,
                                     &n_patterns, &buf_size);
    if (n_patterns == 0) {
        PASS();
    } else {
        FAIL("should find 0 patterns in 2-element stream");
    }
    if (result) free_buffer(result);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Test 2: build_k_stream — vectorized byte→k lookup
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_build_k_stream()
{
    /* Simple identity-like table: k = byte * 10 */
    int32_t table[256];
    for (int i = 0; i < 256; i++) table[i] = i * 10;

    const uint8_t data[] = {0, 1, 2, 127, 255};
    constexpr int n = sizeof(data) / sizeof(data[0]);  /* was const int — byte sample D-count (element count of lookup test fixture) */
    int32_t k_out[5];
    memset(k_out, 0xFF, sizeof(k_out));   /* Poison output buffer (cstring) */

    TEST("build_k_stream: vectorized lookup");
    build_k_stream(data, n, table, k_out);

    const int ok = k_out[0] == 0 && k_out[1] == 10 && k_out[2] == 20
                   && k_out[3] == 1270 && k_out[4] == 2550;
    if (ok) {
        PASS();
    } else {
        FAIL("k values don't match expected");
        for (int i = 0; i < n; i++)
            printf("    k_out[%d] = %d (expected %d)\n", i, k_out[i], data[i] * 10);
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * Test 3: build_dk_stream — first differences
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_build_dk_stream()
{
    const int32_t k_stream[] = {100, 250, 200, 500, 300};
    constexpr int n = 5;  /* was const int — k-stream D-count (5 lattice coordinates in the first-differences test) */
    int32_t dk_out[4];
    memset(dk_out, 0xFF, sizeof(dk_out));  /* Poison output buffer (cstring) */

    TEST("build_dk_stream: first differences");
    build_dk_stream(k_stream, n, dk_out);

    const int ok = dk_out[0] == 150 && dk_out[1] == -50
                   && dk_out[2] == 300 && dk_out[3] == -200;
    if (ok) {
        PASS();
    } else {
        FAIL("dk values don't match");
        for (int i = 0; i < 4; i++)
            printf("    dk_out[%d] = %d\n", i, dk_out[i]);
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * Test 4: gate_archetype_batch — IncoherenceFilter L1-L4
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_gate_archetype_batch()
{
    /* Build a tiny pattern buffer with one pattern: (0, 0)
     * Δk = 0 → ratio = 1.0 → ε = 0 → fully coherent, should pass */
    const int32_t buf[] = {
        1,         /* n_patterns = 1 */
        2, 3,      /* pat_len=2, occ_cnt=3 */
        0, 0,      /* symbols: Δk = 0, 0 */
        0, 5, 10   /* positions (ignored by gate) */
    };
    uint8_t mask[1] = {};

    TEST("gate_archetype_batch: coherent pattern (dk=0)");
    gate_archetype_batch(buf, 1, 27720, 50.0, mask);
    if (mask[0] == 1) {
        PASS();
    } else {
        FAIL("Δk=0 pattern should be coherent");
    }

    /* Verify L1 math independently using ET lattice computation (cmath):
     * k_exact = N·log₂(ratio), ε = |k_exact − round(k_exact)| × (1200/N) cents
     * For Δk=0: ratio=1 → log₂(1)=0 → k_exact=0 → ε=0 */
    const double k_exact = 27720.0 * std::log2(1.0);
    const double eps_verify = std::fabs(k_exact - std::round(k_exact)) * (1200.0 / 27720.0);
    printf("    L1 ε for Δk=0: %.6f cents (finite=%d, must be < %.1f)\n",
           eps_verify, std::isfinite(eps_verify), 50.0);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Test 5: subsume_greedy — non-overlapping placement
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_subsume_greedy()
{
    /* Stream of 20 symbols.
     * Archetype 0: len=3, positions [0, 5, 10, 15]
     * Archetype 1: len=2, positions [3, 8, 13]
     * Non-overlapping: all should be placed (no conflicts) */
    constexpr int n = 20;  /* was const int — P-extent (simulated stream length defining the subsumption domain) */
    constexpr int n_arch = 2;  /* was const int — archetype D-count (number of distinct archetypes in the test) */
    constexpr int32_t lengths[] = {3, 2};  /* was const int32_t[] — per-archetype pattern length D-set */
    constexpr int32_t n_pos[]   = {4, 3};  /* was const int32_t[] — per-archetype occurrence D-count */
    const int32_t positions[] = {0, 5, 10, 15, 3, 8, 13};  /* flat concatenated */

    int32_t placements[40];  /* max 20 × 2 */
    int32_t used_mask[2];

    TEST("subsume_greedy: non-overlapping placement");
    const int n_placed = subsume_greedy(n, n_arch, lengths, n_pos, positions,
                                        placements, used_mask);

    if (n_placed == 7 && used_mask[0] == 1 && used_mask[1] == 1) {
        PASS();
    } else {
        FAIL("expected 7 placements, both archetypes used");
        printf("    n_placed=%d, used=[%d,%d]\n",
               n_placed, used_mask[0], used_mask[1]);
    }

    /* Verify overlap blocking */
    TEST("subsume_greedy: overlap blocking");
    /* Archetype 0: len=4, positions [0, 2]
     * Position 2 overlaps with placement at 0 (0+4=4 > 2) */
    constexpr int32_t lengths2[] = {4};  /* was const int32_t[] — single-archetype pattern length D for overlap test */
    constexpr int32_t n_pos2[]   = {2};  /* was const int32_t[] — position count D for overlap conflict verification */
    constexpr int32_t positions2[] = {0, 2};  /* was const int32_t[] — overlapping position pair for subsumption conflict test */
    int32_t placements2[4];
    int32_t used2[1];

    const int n_placed2 = subsume_greedy(10, 1, lengths2, n_pos2, positions2,
                                          placements2, used2);
    if (n_placed2 == 1 && placements2[1] == 0) {
        PASS();
    } else {
        FAIL("second position should be blocked by overlap");
        printf("    n_placed=%d\n", n_placed2);
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * Test 6: build_ddk_stream — second-order finite difference (Tier 1.B.2)
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_build_ddk_stream()
{
    /* dk = [10, 12, 14, 13, 16] → ddk = [12-10, 14-12, 13-14, 16-13]
     *                                  = [2,    2,     -1,    3] */
    constexpr int32_t dk[] = {10, 12, 14, 13, 16};      /* was const int32_t[] — Δk test fixture (D-set of the second-difference test) */
    constexpr int n = sizeof(dk) / sizeof(dk[0]);       /* was const int — element count */
    int32_t ddk_out[4];
    memset(ddk_out, 0xFF, sizeof(ddk_out));             /* Poison output buffer */

    TEST("build_ddk_stream: second differences");
    build_ddk_stream(dk, n, ddk_out);

    const int ok = ddk_out[0] == 2 && ddk_out[1] == 2
                   && ddk_out[2] == -1 && ddk_out[3] == 3;
    if (ok) {
        PASS();
    } else {
        FAIL("ΔΔk values do not match expected [2,2,-1,3]");
        for (int i = 0; i < 4; i++)
            printf("    ddk_out[%d] = %d\n", i, ddk_out[i]);
    }

    /* Trivial guard: n < 2 → no output written, no crash */
    TEST("build_ddk_stream: trivial n=1 guard");
    int32_t guard_out[1] = {0};
    build_ddk_stream(dk, 1, guard_out);
    if (guard_out[0] == 0) {
        PASS();
    } else {
        FAIL("n<2 should leave output untouched");
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * Test 7: compute_curvature_stats — 5-class block classification (Tier 1.B.3)
 * Verifies all five curvature classes derived from the design doc §3.3.
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_compute_curvature_stats()
{
    CurvatureStats stats;

    /* ── Case 1: FLAT (class 0) — Exception state ──
     * Constant Δk → all ΔΔk = 0 → K̄=0, σ²=0, max=0 */
    constexpr int32_t flat[] = {0, 0, 0, 0, 0, 0, 0, 0};   /* was const int32_t[] — flat ΔΔk D-set */
    TEST("compute_curvature_stats: FLAT (class 0)");
    compute_curvature_stats(flat, 8, 27720, &stats);
    if (stats.curvature_class == 0
        && stats.curvature_mean == 0.0
        && stats.curvature_variance == 0.0
        && stats.max_abs_curvature == 0
        && stats.euler_characteristic == 0.0) {
        PASS();
    } else {
        FAIL("flat case misclassified");
        printf("    K̄=%.4f σ²=%.4f class=%d χ=%.4f max=%d\n",
               stats.curvature_mean, stats.curvature_variance,
               stats.curvature_class, stats.euler_characteristic,
               stats.max_abs_curvature);
    }

    /* ── Case 2: ELLIPTIC (class 1) — Unsubstantiated {P,D} ──
     * Constant +1 ΔΔk → K̄=1, σ²=0, mean ≥ π/12 ≈ 0.262 → class 1 */
    constexpr int32_t ellip[] = {1, 1, 1, 1, 1, 1, 1, 1}; /* was const int32_t[] — elliptic K>0 D-set */
    TEST("compute_curvature_stats: ELLIPTIC (class 1)");
    compute_curvature_stats(ellip, 8, 27720, &stats);
    if (stats.curvature_class == 1) {
        PASS();
    } else {
        FAIL("elliptic case misclassified");
        printf("    K̄=%.4f σ²=%.4f class=%d\n",
               stats.curvature_mean, stats.curvature_variance, stats.curvature_class);
    }

    /* ── Case 3: HYPERBOLIC (class 2) — Mediation {D,T} ──
     * Constant -1 ΔΔk → K̄=-1, σ²=0, mean ≤ -π/12 → class 2 */
    constexpr int32_t hyper[] = {-1, -1, -1, -1, -1, -1, -1, -1};   /* was const int32_t[] — hyperbolic K<0 D-set */
    TEST("compute_curvature_stats: HYPERBOLIC (class 2)");
    compute_curvature_stats(hyper, 8, 27720, &stats);
    if (stats.curvature_class == 2) {
        PASS();
    } else {
        FAIL("hyperbolic case misclassified");
        printf("    K̄=%.4f σ²=%.4f class=%d\n",
               stats.curvature_mean, stats.curvature_variance, stats.curvature_class);
    }

    /* ── Case 4: VARIABLE (class 3) — mixed regions ──
     * ΔΔk alternates [3,-3,3,-3,...] → K̄=0, σ²=9 (≥ V=1/12), max=3 (<12) → class 3 */
    constexpr int32_t var[] = {3, -3, 3, -3, 3, -3, 3, -3};         /* was const int32_t[] — variable curvature D-set */
    TEST("compute_curvature_stats: VARIABLE (class 3)");
    compute_curvature_stats(var, 8, 27720, &stats);
    if (stats.curvature_class == 3 && stats.max_abs_curvature == 3) {
        PASS();
    } else {
        FAIL("variable case misclassified");
        printf("    K̄=%.4f σ²=%.4f class=%d max=%d\n",
               stats.curvature_mean, stats.curvature_variance,
               stats.curvature_class, stats.max_abs_curvature);
    }

    /* ── Case 5: SINGULAR (class 4) — Incoherence {P,T}, D-bridge broken ──
     * Isolated spike with |K_i| ≥ N=12 → class 4 */
    constexpr int32_t sing[] = {0, 0, 0, 50, 0, 0, 0, 0};           /* was const int32_t[] — singular curvature D-set */
    TEST("compute_curvature_stats: SINGULAR (class 4)");
    compute_curvature_stats(sing, 8, 27720, &stats);
    if (stats.curvature_class == 4 && stats.max_abs_curvature == 50) {
        PASS();
    } else {
        FAIL("singular case misclassified");
        printf("    K̄=%.4f σ²=%.4f class=%d max=%d\n",
               stats.curvature_mean, stats.curvature_variance,
               stats.curvature_class, stats.max_abs_curvature);
    }

    /* ── Case 6: empty stream guard ──
     * n=0 must not crash, must return zeros + class 0 */
    TEST("compute_curvature_stats: empty stream guard");
    memset(&stats, 0xFF, sizeof(stats));
    compute_curvature_stats(nullptr, 0, 27720, &stats);
    if (stats.curvature_class == 0
        && stats.curvature_mean == 0.0
        && stats.curvature_variance == 0.0
        && stats.max_abs_curvature == 0) {
        PASS();
    } else {
        FAIL("empty stream guard failed to zero-init");
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * Test 8: compute_pattern_curvature — per-pattern stats for DB (Tier 1.B.4)
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_compute_pattern_curvature()
{
    double mean, variance, geodesic_factor;

    /* ── Flat pattern: constant Δk ──
     * pat = [5,5,5,5,5] → ΔΔk all 0 → K̄=0, σ²=0, F_K=1.0 */
    constexpr int32_t flat_pat[] = {5, 5, 5, 5, 5};   /* was const int32_t[] — flat-pattern D-set */
    TEST("compute_pattern_curvature: flat pattern");
    compute_pattern_curvature(flat_pat, 5, &mean, &variance, &geodesic_factor);
    if (mean == 0.0 && variance == 0.0 && geodesic_factor == 1.0) {
        PASS();
    } else {
        FAIL("flat pattern: expected K̄=0 σ²=0 F_K=1");
        printf("    K̄=%.6f σ²=%.6f F_K=%.6f\n", mean, variance, geodesic_factor);
    }

    /* ── Quadratic pattern: pat[i] = i² ──
     * pat = [0,1,4,9,16] → ΔΔk = [2,2,2] → K̄=2, σ²=0, F_K=1.0 */
    constexpr int32_t quad_pat[] = {0, 1, 4, 9, 16};  /* was const int32_t[] — quadratic-pattern D-set */
    TEST("compute_pattern_curvature: quadratic pattern");
    compute_pattern_curvature(quad_pat, 5, &mean, &variance, &geodesic_factor);
    if (mean == 2.0 && variance == 0.0 && geodesic_factor == 1.0) {
        PASS();
    } else {
        FAIL("quadratic pattern: expected K̄=2 σ²=0 F_K=1");
        printf("    K̄=%.6f σ²=%.6f F_K=%.6f\n", mean, variance, geodesic_factor);
    }

    /* ── Variable pattern: ΔΔk alternates ──
     * pat = [0,1,3,4,6,7] → Δk = [1,2,1,2,1] → ΔΔk = [1,-1,1,-1] → K̄=0, σ²=1, F_K=1/(1+1)=0.5 */
    constexpr int32_t variable_pat[] = {0, 1, 3, 4, 6, 7};  /* was const int32_t[] — variable-curvature pattern D-set */
    TEST("compute_pattern_curvature: variable pattern");
    compute_pattern_curvature(variable_pat, 6, &mean, &variance, &geodesic_factor);
    /* Expected: ΔΔk_0 = 3-2*1+0 = 1
     *           ΔΔk_1 = 4-2*3+1 = -1
     *           ΔΔk_2 = 6-2*4+3 = 1
     *           ΔΔk_3 = 7-2*6+4 = -1
     *           K̄ = 0, σ² = 1, F_K = 0.5 */
    if (mean == 0.0 && variance == 1.0 && geodesic_factor == 0.5) {
        PASS();
    } else {
        FAIL("variable pattern: expected K̄=0 σ²=1 F_K=0.5");
        printf("    K̄=%.6f σ²=%.6f F_K=%.6f\n", mean, variance, geodesic_factor);
    }

    /* ── Short-pattern guard: pat_len < 3 ──
     * No ΔΔk values exist → must return F_K=1.0 (geodesic by default) */
    constexpr int32_t short_pat[] = {7, 9};   /* was const int32_t[] — sub-3-length D-set, exercises guard branch */
    TEST("compute_pattern_curvature: short pattern (len<3)");
    compute_pattern_curvature(short_pat, 2, &mean, &variance, &geodesic_factor);
    if (mean == 0.0 && variance == 0.0 && geodesic_factor == 1.0) {
        PASS();
    } else {
        FAIL("short pattern guard failed");
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * Test 9: build_geodesic_residual — Mode 3 (Tier 2.A.8)
 *
 * Verifies all three connection orders (0/1/2) compute correctly and
 * the encoder→decoder reconstruction is bit-exact via C-truncating
 * integer arithmetic.
 *
 * Roundtrip strategy: compute residuals via build_geodesic_residual,
 * then reconstruct Δk causally using the SAME connection formula and
 * the SAME truncating division. If reconstruction equals the original
 * Δk stream, the encoder and decoder agree — Mode 3 is lossless.
 * ═══════════════════════════════════════════════════════════════════════ */

/* C-style truncating integer division — mirrors et_c_trunc_div in the
 * pattern engine. Used by the test harness to reconstruct Δk from
 * residuals + connection so that encoder/decoder parity can be verified
 * end-to-end inside this single test binary. */
static inline int32_t test_c_trunc_div(int64_t numerator, int32_t denominator)
{
    if (denominator == 0) return 0;
    return (int32_t)(numerator / (int64_t)denominator);
}

/* Reconstruct Δk causally from residuals + connection (mirrors the
 * Python decoder in et_cdf_compressor.py). Lengths: residual_n = dk_n - 1.
 * Returns true if reconstructed equals the original Δk; false otherwise. */
static bool test_reconstruct_dk(
    const int32_t *dk_original, int dk_n,
    const int32_t *residuals,   int residual_n,
    int connection_order, int window_size)
{
    if (dk_n < 2 || residual_n != dk_n - 1) return false;
    int32_t *dk_rec = (int32_t *)malloc(sizeof(int32_t) * dk_n);
    if (!dk_rec) return false;
    dk_rec[0] = dk_original[0];   /* dk0_saved */
    for (int i = 0; i < residual_n; i++) {
        int32_t gamma = 0;
        if (connection_order >= 1 && i > 0) {
            int w_start = i - window_size + 1;
            if (w_start < 0) w_start = 0;
            int64_t ddk_sum = 0;
            int count = 0;
            for (int j = w_start; j < i; j++) {
                ddk_sum += (int64_t)(dk_rec[j + 1] - dk_rec[j]);
                count++;
            }
            if (count > 0) gamma = test_c_trunc_div(ddk_sum, count);
        }
        if (connection_order >= 2 && i > 1) {
            int w_start = i - window_size + 1;
            if (w_start < 1) w_start = 1;
            int64_t dddk_sum = 0;
            int count = 0;
            for (int j = w_start; j < i - 1; j++) {
                int32_t ddk_j  = dk_rec[j + 1] - dk_rec[j];
                int32_t ddk_j1 = dk_rec[j + 2] - dk_rec[j + 1];
                dddk_sum += (int64_t)(ddk_j1 - ddk_j);
                count++;
            }
            if (count > 0) gamma += test_c_trunc_div(dddk_sum, 2 * count);
        }
        dk_rec[i + 1] = residuals[i] + dk_rec[i] + gamma;
    }
    bool match = true;
    for (int i = 0; i < dk_n; i++) {
        if (dk_rec[i] != dk_original[i]) { match = false; break; }
    }
    free(dk_rec);
    return match;
}

static void test_build_geodesic_residual()
{
    /* ── Test A: connection_order = 0 (zeroth-order, Γ = 0) ──
     * Predicts Δk_{i+1} = Δk_i. Residual = ΔΔk.
     * For dk = [10, 12, 14, 13, 16] → ΔΔk = [2, 2, -1, 3]
     * Order 0 residuals MUST equal ΔΔk exactly. */
    constexpr int32_t dk_a[] = {10, 12, 14, 13, 16};        /* was const int32_t[] — order-0 test D-set */
    constexpr int n_a = 5;                                  /* was const int — element count */
    int32_t res_a[4], gam_a[4];
    memset(res_a, 0xFF, sizeof(res_a));
    memset(gam_a, 0xFF, sizeof(gam_a));
    TEST("build_geodesic_residual: order=0 → residual == ΔΔk");
    build_geodesic_residual(dk_a, n_a, 0, 4, res_a, gam_a);
    if (res_a[0] == 2 && res_a[1] == 2 && res_a[2] == -1 && res_a[3] == 3
        && gam_a[0] == 0 && gam_a[1] == 0 && gam_a[2] == 0 && gam_a[3] == 0) {
        PASS();
    } else {
        FAIL("order=0 residual or gamma incorrect");
        for (int i = 0; i < 4; i++)
            printf("    res[%d]=%d gamma[%d]=%d\n", i, res_a[i], i, gam_a[i]);
    }

    /* Roundtrip — order 0 with Γ=0 is trivially exact (just sum residuals). */
    TEST("build_geodesic_residual: order=0 roundtrip");
    if (test_reconstruct_dk(dk_a, n_a, res_a, 4, 0, 4)) {
        PASS();
    } else {
        FAIL("order=0 reconstruction did not equal original Δk");
    }

    /* ── Test B: connection_order = 1 (first-order, Γ = mean ΔΔk) ──
     * For dk = [0, 1, 2, 3, 4, 5] → ΔΔk = [0, 0, 0, 0]
     * Γ at every i ≥ 1 = 0 (mean of zeros). Residuals = 1 - 1 - 0 = 0.
     * The pattern is purely linear, the order-1 predictor is exact. */
    constexpr int32_t dk_b[] = {0, 1, 2, 3, 4, 5};          /* was const int32_t[] — linear-progression D-set */
    constexpr int n_b = 6;
    int32_t res_b[5], gam_b[5];
    TEST("build_geodesic_residual: order=1 on linear data → warmup then zero");
    build_geodesic_residual(dk_b, n_b, 1, 4, res_b, gam_b);
    /* Expected residuals at order 1 on linear Δk:
     *   i=0: no prior ΔΔk samples, so Γ_0 = 0 by construction.
     *        predicted = Δk_0 + 0 = 0; actual = Δk_1 = 1; residual = 1 (warmup)
     *   i≥1: Γ_i = mean of preceding ΔΔk samples = 1 exactly.
     *        predicted = Δk_i + 1 = Δk_{i+1}; residual = 0.
     * The "warmup then zero" pattern is correct and confirms the connection
     * locks on after the first sample. */
    bool warmup_then_zero = (res_b[0] == 1)
                            && (res_b[1] == 0)
                            && (res_b[2] == 0)
                            && (res_b[3] == 0)
                            && (res_b[4] == 0);
    if (warmup_then_zero) {
        PASS();
    } else {
        FAIL("order=1 on linear data should give res = [1, 0, 0, 0, 0] (warmup then locked)");
        for (int i = 0; i < 5; i++)
            printf("    res[%d]=%d gamma[%d]=%d\n", i, res_b[i], i, gam_b[i]);
    }
    TEST("build_geodesic_residual: order=1 linear roundtrip");
    if (test_reconstruct_dk(dk_b, n_b, res_b, 5, 1, 4)) {
        PASS();
    } else {
        FAIL("order=1 reconstruction did not equal original Δk");
    }

    /* ── Test C: connection_order = 1 with non-zero ΔΔk ──
     * For dk = [10, 13, 16, 19, 22] → ΔΔk = [3, 3, 3, 3]
     * Γ_1 = ΔΔk_0 = 3.  ρ_1 = Δk_2 - Δk_1 - Γ_1 = 16 - 13 - 3 = 0
     * Residuals after warmup MUST cluster near zero — the connection
     * captures the constant ΔΔk perfectly. */
    constexpr int32_t dk_c[] = {10, 13, 16, 19, 22};        /* was const int32_t[] — constant-ΔΔk D-set */
    constexpr int n_c = 5;
    int32_t res_c[4], gam_c[4];
    TEST("build_geodesic_residual: order=1 captures constant trend");
    build_geodesic_residual(dk_c, n_c, 1, 4, res_c, gam_c);
    /* res_c[0] = 13 - 10 - 0 = 3 (no preceding ΔΔk → gamma = 0)
     * res_c[1] = 16 - 13 - 3 = 0
     * res_c[2] = 19 - 16 - 3 = 0
     * res_c[3] = 22 - 19 - 3 = 0 */
    if (res_c[0] == 3 && res_c[1] == 0 && res_c[2] == 0 && res_c[3] == 0) {
        PASS();
    } else {
        FAIL("order=1 should drive residuals to zero on constant ΔΔk");
        for (int i = 0; i < 4; i++)
            printf("    res[%d]=%d gamma[%d]=%d\n", i, res_c[i], i, gam_c[i]);
    }
    TEST("build_geodesic_residual: order=1 constant-trend roundtrip");
    if (test_reconstruct_dk(dk_c, n_c, res_c, 4, 1, 4)) {
        PASS();
    } else {
        FAIL("constant-trend reconstruction failed");
    }

    /* ── Test D: connection_order = 2 (second-order, quadratic predictor) ──
     * For dk = [0, 1, 4, 9, 16, 25, 36] (i² Δk values)
     * → ΔΔk = [3, 5, 7, 9, 11]   linear in i
     * → ΔΔΔk = [2, 2, 2, 2]       constant
     * Order 2 predictor uses ½·mean(ΔΔΔk) which captures the constant
     * third derivative. The roundtrip must hold despite the bigger
     * window arithmetic. */
    constexpr int32_t dk_d[] = {0, 1, 4, 9, 16, 25, 36};    /* was const int32_t[] — quadratic D-set */
    constexpr int n_d = 7;
    int32_t res_d[6], gam_d[6];
    TEST("build_geodesic_residual: order=2 on quadratic data (no-crash check)");
    /* The no-crash check is meaningful: order=2 walks two nested windows
     * over the dk_stream and could trip integer overflow or invalid index
     * if the bounds were wrong. The roundtrip below is the bit-exactness
     * check; this PASS just confirms the function returned cleanly. */
    build_geodesic_residual(dk_d, n_d, 2, 4, res_d, gam_d);
    PASS();
    /* The roundtrip is the meaningful test — exact residual values
     * depend on integer truncation in the running mean. */
    TEST("build_geodesic_residual: order=2 quadratic roundtrip");
    if (test_reconstruct_dk(dk_d, n_d, res_d, 6, 2, 4)) {
        PASS();
    } else {
        FAIL("order=2 quadratic reconstruction failed");
        for (int i = 0; i < 6; i++)
            printf("    res[%d]=%d gamma[%d]=%d\n", i, res_d[i], i, gam_d[i]);
    }

    /* ── Test E: NEGATIVE Δk values exercise C-truncation parity ──
     * Encoder uses C truncation toward zero; the test reconstructor uses
     * the SAME truncation. Decoder in production uses Python helper that
     * mirrors this. If C-truncation were silently floor-dividing somewhere,
     * roundtrip on negative-mean windows would fail. */
    constexpr int32_t dk_e[] = {10, -5, -3, -8, -10, -7, -4, -6, -9};   /* was const int32_t[] — negative-mean D-set */
    constexpr int n_e = 9;
    int32_t res_e[8], gam_e[8];
    TEST("build_geodesic_residual: order=1 negative-Δk roundtrip");
    build_geodesic_residual(dk_e, n_e, 1, 4, res_e, gam_e);
    if (test_reconstruct_dk(dk_e, n_e, res_e, 8, 1, 4)) {
        PASS();
    } else {
        FAIL("negative-Δk order=1 reconstruction failed");
    }
    TEST("build_geodesic_residual: order=2 negative-Δk roundtrip");
    build_geodesic_residual(dk_e, n_e, 2, 4, res_e, gam_e);
    if (test_reconstruct_dk(dk_e, n_e, res_e, 8, 2, 4)) {
        PASS();
    } else {
        FAIL("negative-Δk order=2 reconstruction failed");
    }

    /* ── Test F: Trivial guards — n < 2, order out of range, window=0 ── */
    TEST("build_geodesic_residual: n=1 guard");
    int32_t guard_res[1] = {0xDEAD};
    int32_t guard_gam[1] = {0xDEAD};
    build_geodesic_residual(dk_a, 1, 0, 4, guard_res, guard_gam);
    /* No output should be written; values remain at sentinel */
    if (guard_res[0] == (int32_t)0xDEAD && guard_gam[0] == (int32_t)0xDEAD) {
        PASS();
    } else {
        FAIL("n=1 guard wrote outputs");
    }

    TEST("build_geodesic_residual: order=-1 clamped to 0");
    int32_t res_clamp[4], gam_clamp[4];
    build_geodesic_residual(dk_a, n_a, -1, 4, res_clamp, gam_clamp);
    /* With clamped order=0, gamma must be all zeros */
    if (gam_clamp[0] == 0 && gam_clamp[1] == 0 && gam_clamp[2] == 0 && gam_clamp[3] == 0) {
        PASS();
    } else {
        FAIL("order=-1 should clamp to 0 (Γ=0 for all i)");
    }

    TEST("build_geodesic_residual: order=99 clamped to 2");
    /* High order should clamp to 2; behavior should not crash. */
    build_geodesic_residual(dk_d, n_d, 99, 4, res_d, gam_d);
    /* Roundtrip with clamped order=2 must succeed. */
    if (test_reconstruct_dk(dk_d, n_d, res_d, 6, 2, 4)) {
        PASS();
    } else {
        FAIL("order=99 clamp+roundtrip failed");
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * Main — run all tests
 * ═══════════════════════════════════════════════════════════════════════ */

int main()
{
    printf("═══════════════════════════════════════════════════════════\n");
    printf(" ET Pattern Engine — Test Harness\n");
    printf(" P ∘ D ∘ T = E\n");
    printf("═══════════════════════════════════════════════════════════\n\n");

    test_find_repeated_patterns();
    printf("\n");
    test_build_k_stream();
    printf("\n");
    test_build_dk_stream();
    printf("\n");
    test_gate_archetype_batch();
    printf("\n");
    test_subsume_greedy();
    printf("\n");
    test_build_ddk_stream();
    printf("\n");
    test_compute_curvature_stats();
    printf("\n");
    test_compute_pattern_curvature();
    printf("\n");
    test_build_geodesic_residual();

    printf("\n═══════════════════════════════════════════════════════════\n");
    printf(" Results: %d / %d tests passed", g_tests_passed, g_tests_run);
    if (g_tests_passed == g_tests_run) {
        printf(" — ALL PASS\n");
    } else {
        printf(" — %d FAILED\n", g_tests_run - g_tests_passed);
    }
    printf("═══════════════════════════════════════════════════════════\n");

    return g_tests_passed == g_tests_run ? EXIT_SUCCESS : EXIT_FAILURE;
}