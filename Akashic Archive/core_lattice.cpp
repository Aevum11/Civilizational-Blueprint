// ============================================================================
// core_lattice.cpp — Module 2: Core Lattice Engine (Level 1) — Implementation
//
// P ∘ D ∘ T = E
// Every computation at 1200-bit (361-dps). Zero IEEE 754. Zero ad hoc.
//
// The projection formula IS the master equation at single-ratio scope:
//   r is the P-content (a featureless positive real)
//   N and the lattice structure are the D-content (finite constraints)
//   round() is the T-act (resolution of continuous to discrete)
// ============================================================================

#include "core_lattice.h"

#include <cstring>
#include <algorithm>
#include <sstream>
#include <cassert>

namespace et::lattice {

// ============================================================================
// Section 1: String conversions for enums
// ============================================================================

const char* manifold_state_string(ManifoldState state) {
    switch (state) {
        case ManifoldState::EXCEPTION:       return "PDT";
        case ManifoldState::UNSUBSTANTIATED: return "PD";
        case ManifoldState::MEDIATION:       return "DT";
        case ManifoldState::INCOHERENCE:     return "PT";
    }
    // Structurally unreachable — all enum values covered.
    // Descriptor Gap: if reached, a new manifold state was added without
    // updating this function. The gap IS the missing descriptor.
    throw ETError(ETError::Code::INVALID_INPUT,
                  "manifold_state_string",
                  "Unknown ManifoldState value");
}

const char* fqg_quadrant_string(FQGQuadrant q) {
    switch (q) {
        case FQGQuadrant::SR: return "SR";
        case FQGQuadrant::CR: return "CR";
        case FQGQuadrant::SI: return "SI";
        case FQGQuadrant::CI: return "CI";
    }
    throw ETError(ETError::Code::INVALID_INPUT,
                  "fqg_quadrant_string",
                  "Unknown FQGQuadrant value");
}

// ============================================================================
// Section 2: Gaussian Prime Classification
//
// Paper §6: the three Gaussian classes parallel the three Cardinals.
//   p = 2:         Ramified — P-class (substrate doubling)
//   p ≡ 3 (mod 4): Inert    — D-class (stays on real axis)
//   p ≡ 1 (mod 4): Split    — D+T-class (factors across axes)
// ============================================================================

GaussianClass classify_gaussian_prime(const ETInteger& p) {
    // p = 2 is the unique ramified prime in ℤ[i]
    if (p == 2) {
        return GaussianClass::RAMIFIED;
    }

    // For odd primes: compute p mod 4
    ETInteger four(int64_t(4));
    ETInteger remainder = p % four;

    // p ≡ 1 (mod 4): Split — Fermat's theorem on sums of two squares
    if (remainder == ETInteger(int64_t(1))) {
        return GaussianClass::SPLIT;
    }

    // p ≡ 3 (mod 4): Inert — remains prime in ℤ[i]
    // This is the only remaining case for odd primes
    return GaussianClass::INERT;
}

GaussianSignature compute_gaussian_signature(const ETInteger& d) {
    GaussianSignature sig;
    sig.is_all_inert = true;
    sig.is_all_split = true;
    sig.is_ramified_present = false;

    // d = 1 has no prime factors — trivial signature
    if (d <= ETInteger(int64_t(1))) {
        sig.signature_string = "1";
        return sig;
    }

    // Factorize d using Module 1's arbitrary-precision factorization
    auto factors = intmath::factorize(d);

    // Classify each prime power factor
    for (const auto& [prime, exponent] : factors) {
        GaussianClass gc = classify_gaussian_prime(prime);
        sig.factors.push_back({prime, exponent, gc});

        switch (gc) {
            case GaussianClass::RAMIFIED:
                sig.is_all_inert = false;
                sig.is_all_split = false;
                sig.is_ramified_present = true;
                break;
            case GaussianClass::INERT:
                sig.is_all_split = false;
                break;
            case GaussianClass::SPLIT:
                sig.is_all_inert = false;
                break;
        }
    }

    // Build signature string: "R^e1·I^e2·S^e3" or compact form
    std::ostringstream oss;
    bool first = true;
    for (const auto& gf : sig.factors) {
        if (!first) oss << "\xC2\xB7"; // UTF-8 middle dot ·
        first = false;

        switch (gf.gclass) {
            case GaussianClass::RAMIFIED: oss << "R"; break;
            case GaussianClass::INERT:    oss << "I"; break;
            case GaussianClass::SPLIT:    oss << "S"; break;
        }
        if (gf.exponent > 1) {
            oss << "^" << gf.exponent;
        }
    }
    sig.signature_string = oss.str();

    return sig;
}

// ============================================================================
// Section 3: Factorization String
//
// Produces human-readable factorization of d: "2³·3·5·7"
// Uses Unicode superscript digits for exponents > 1.
// ============================================================================

// Helper: convert an integer exponent to Unicode superscript string
static std::string exponent_to_superscript(int exp) {
    if (exp <= 1) return "";
    // Unicode superscript digits: ⁰¹²³⁴⁵⁶⁷⁸⁹
    static const char* sup_digits[] = {
        "\xE2\x81\xB0", // ⁰
        "\xC2\xB9",     // ¹
        "\xC2\xB2",     // ²
        "\xC2\xB3",     // ³
        "\xE2\x81\xB4", // ⁴
        "\xE2\x81\xB5", // ⁵
        "\xE2\x81\xB6", // ⁶
        "\xE2\x81\xB7", // ⁷
        "\xE2\x81\xB8", // ⁸
        "\xE2\x81\xB9"  // ⁹
    };

    std::string result;
    std::string exp_str = std::to_string(exp);
    for (char c : exp_str) {
        int digit = c - '0';
        if (digit >= 0 && digit <= 9) {
            result += sup_digits[digit];
        }
    }
    return result;
}

std::string factorization_to_string(
    const std::vector<std::pair<ETInteger, int>>& factors) {

    if (factors.empty()) return "1";

    std::ostringstream oss;
    bool first = true;
    for (const auto& [prime, exp] : factors) {
        if (!first) oss << "\xC2\xB7"; // UTF-8 middle dot ·
        first = false;
        oss << prime.to_string();
        if (exp > 1) {
            oss << exponent_to_superscript(exp);
        }
    }
    return oss.str();
}

// ============================================================================
// Section 4: Derived Property Computation Functions
// ============================================================================

ETValue compute_coupling_xi(const ETInteger& d) {
    // ξ(d) = 137 / ((d-1)² + S²)  where S = ET_S = 4
    // A₀(d) = (d-1)² + S² = (d-1)² + 16
    ETValue d_val = ETValue::from_integer(d);
    ETValue one(int64_t(1));
    ETValue d_minus_1 = d_val - one;
    ETValue s_squared(int64_t(ET_S * ET_S)); // 16
    ETValue a0 = d_minus_1 * d_minus_1 + s_squared;
    ETValue one_three_seven(int64_t(137));
    return one_three_seven / a0;
}

ETValue compute_impedance(const ETInteger& d) {
    // A₀(d) = (d-1)² + S² where S = 4
    ETValue d_val = ETValue::from_integer(d);
    ETValue one(int64_t(1));
    ETValue d_minus_1 = d_val - one;
    ETValue s_squared(int64_t(ET_S * ET_S)); // 16
    return d_minus_1 * d_minus_1 + s_squared;
}

ETValue compute_tightness(const ETValue& eps) {
    // t = 100/(100+|ε|)
    // Range: [2/3, 1] for |ε| ∈ [0, 50]
    ETValue hundred(int64_t(100));
    ETValue abs_eps = math::abs(eps);
    return hundred / (hundred + abs_eps);
}

ETValue compute_di_distance(const ETValue& eps) {
    // di_dist = |ε|/50
    // 0 at lattice point, 1 at ∂I boundary
    ETValue fifty(int64_t(50));
    return math::abs(eps) / fifty;
}

std::pair<ETValue, bool> compute_variance(
    const ETInteger& n, const ETInteger& k) {
    // V(n,k) = (n²-1)/(12·2^k)
    // n = sublattice family d (cardinality), k = fold depth
    //
    // V(n,k) = (n²-1) / (12·2^k)
    // All computation via MPFR. No int64. No caps. No limits.
    // MPFR's exp2 handles arbitrary-precision exponents natively.

    ETValue n_val = ETValue::from_integer(n);
    ETValue one(int64_t(1));
    ETValue twelve(int64_t(12));
    ETValue numerator = n_val * n_val - one;

    // 2^k via MPFR — works for any k (positive, negative, huge)
    ETValue k_val = ETValue::from_integer(k);
    ETValue two_pow_k = math::exp2(k_val);

    ETValue denominator = twelve * two_pow_k;
    ETValue result = numerator / denominator;
    return {result, true};
}

ETValue compute_elegance_symmetry(const ETInteger& N, const ETInteger& d) {
    // symmetry = N/d
    ETValue N_val = ETValue::from_integer(N);
    ETValue d_val = ETValue::from_integer(d);
    return N_val / d_val;
}

std::pair<ETValue, bool> compute_elegance_universal(
    const ETValue& symmetry, const ETValue& tightness,
    const ETInteger& p_plus_q) {
    // E = symmetry × tightness × simplicity
    // simplicity = 100/max(1, p+q)
    ETValue hundred(int64_t(100));
    ETValue pq_val = ETValue::from_integer(p_plus_q);
    ETValue one(int64_t(1));

    // max(1, p+q)
    ETValue denom = (pq_val < one) ? one : pq_val;
    ETValue simplicity = hundred / denom;
    ETValue universal = symmetry * tightness * simplicity;
    return {universal, true};
}

FQGQuadrant compute_fqg_quadrant(
    const ETInteger& d, const ETInteger& N, char perspective) {
    // Simple: d divides N (i.e., N mod d == 0)
    // Complex: d does not divide N
    ETInteger remainder = N % d;
    bool is_simple = remainder.is_zero();

    bool is_real = (perspective == 'r' || perspective == 'R');

    if (is_simple && is_real)  return FQGQuadrant::SR;
    if (!is_simple && is_real) return FQGQuadrant::CR;
    if (is_simple && !is_real) return FQGQuadrant::SI;
    return FQGQuadrant::CI;
}

ETInteger compute_palindromic_partner(const ETInteger& d, const ETInteger& N) {
    // At N=12: partner = 12-d for d∈{1..11}, self for d∈{6,12}
    // Generalized: partner = N-d
    // Self-partners: d = N/2 (if N is even) and d = N
    ETInteger partner = N - d;

    // Check for self-partner conditions
    if (partner == d) {
        return d; // d = N/2 — self-partner
    }
    if (d == N) {
        return d; // d = N — self-partner (full resolution)
    }
    if (partner.is_zero() || partner.is_negative()) {
        // d >= N: only possible when d = N (handled above)
        return d;
    }

    return partner;
}

ManifoldState compute_manifold_state(
    const ETValue& eps, const ETInteger& d, const ETInteger& N) {
    // Manifold state determination from projection properties:
    //
    // {P,D,T} Exception: ε = 0 exactly — the value sits precisely on a
    //   lattice point. All three primitives are present and bound.
    //
    // {P,T} Incoherence: |ε| ≥ 50 cents — at or beyond the ∂I boundary.
    //   This is the self-defeating state where D is insufficient.
    //   (The ∂I boundary at ε = ±50 cents is the structural ceiling.)
    //
    // {P,D} Unsubstantiated: Default for non-zero, non-boundary ε.
    //   The value has substrate (P) and constraint (D) but T's rounding
    //   introduced a gap, indicating potential without full substantiation.
    //
    // {D,T} Mediation: Requires external context to determine.
    //   At projection level, we cannot distinguish PD from DT without
    //   knowing the structural role of the value. Default to PD.
    //
    // Note: "Exactly zero" means mpfr_zero_p(ε) — we use the MPFR
    // comparison, not a tolerance check. At 1200-bit precision, if ε
    // is zero, it IS zero (e.g., log₂(2) = 1 exactly → ε = 0).

    if (eps.is_zero()) {
        return ManifoldState::EXCEPTION;
    }

    // Check ∂I boundary: |ε| ≥ 50 cents
    ETValue abs_eps = math::abs(eps);
    ETValue boundary(int64_t(50));
    if (abs_eps >= boundary) {
        return ManifoldState::INCOHERENCE;
    }

    // Default: Unsubstantiated (PD)
    // The distinction between PD and DT requires external context
    // (what the value represents structurally) — not determinable
    // from the projection triple alone.
    return ManifoldState::UNSUBSTANTIATED;
}

int32_t eps_to_microcents(const ETValue& eps) {
    // ε_micros = round(ε × 1000)
    // ε is in cents; micro-cents = cents × 1000
    // Computation: MPFR multiply → MPFR round → GMP extract → int32_t storage
    ETValue thousand(int64_t(1000));
    ETValue scaled = eps * thousand;
    ETValue rounded = math::round(scaled);

    // Convert to ETInteger via GMP (no int64 intermediate)
    ETInteger micros = ETInteger::from_etvalue(rounded);

    // The schema stores eps_micros as int32_t. For valid projections
    // (|ε| ≤ 50 cents), |micros| ≤ 50000, well within int32_t.
    // For values outside this range: saturate at ∂I boundary.
    // This is the ONLY narrowing point, and only for storage format.
    ETInteger bound{static_cast<int64_t>(ET_DI_BOUNDARY_MICROS)};
    ETInteger neg_bound = -bound;
    if (micros > bound) {
        return ET_DI_BOUNDARY_MICROS;
    }
    if (micros < neg_bound) {
        return -ET_DI_BOUNDARY_MICROS;
    }

    // Safe to extract: value is within [-50000, 50000]
    // Use mpz_get_si which handles this range on all platforms
    return static_cast<int32_t>(mpz_get_si(micros.raw()));
}

std::pair<ETValue, bool> compute_quintic_tension(
    const ETInteger& d, const ETValue& eps, const ETInteger& N) {
    // Quintic tension τ₅ — specific to d=5 sublattice interactions
    // Computed when d has a factor of 5
    ETInteger five(int64_t(5));
    ETInteger remainder = d % five;
    if (!remainder.is_zero()) {
        return {ETValue(int64_t(0)), false};
    }

    // τ₅ = ε × (d/5) — the quintic-scaled residual
    // This captures how the quintic sublattice's residual scales with d
    ETValue d_val = ETValue::from_integer(d);
    ETValue five_val(int64_t(5));
    ETValue scale = d_val / five_val;
    ETValue tension = eps * scale;
    return {tension, true};
}

// ============================================================================
// Section 5: Rational Approximation via Continued Fractions
// ============================================================================

RationalApprox best_rational_approx(
    const ETValue& r, const ETInteger& max_q) {
    // Continued fraction expansion to find best rational p/q
    // with q ≤ max_q.
    //
    // Algorithm: compute continued fraction convergents of |r|.
    // Track best approximation as p_n/q_n.
    // Stop when q exceeds max_q.

    ETValue abs_r = math::abs(r);

    // Initialize convergents: p_{-1}/q_{-1} = 1/0, p_0/q_0 = a_0/1
    ETValue x = abs_r;
    ETValue a = math::floor(x);

    // Use ETInteger for p, q to maintain exact integer arithmetic
    ETInteger p_prev(int64_t(1));
    ETInteger q_prev(int64_t(0));
    ETInteger p_curr = ETInteger::from_etvalue(a);  // GMP direct, no int64
    ETInteger q_curr(int64_t(1));

    // Best approximation so far
    ETInteger best_p = p_curr;
    ETInteger best_q = q_curr;

    // If r is already an integer or very close, we're done
    ETValue frac_part = x - a;
    if (frac_part.is_zero()) {
        ETValue error = abs_r - ETValue::from_integer(best_p) / ETValue::from_integer(best_q);
        return {best_p, best_q, math::abs(error)};
    }

    // Iterate continued fraction
    // Limit iterations to prevent infinite loops for irrationals
    // (at 361 dps, we have at most ~1200 bits of information)
    for (int iter = 0; iter < 2000; iter++) {
        // x = 1/(x - a)
        x = ETValue(int64_t(1)) / frac_part;
        a = math::floor(x);
        frac_part = x - a;

        // Compute next convergent: p_n = a_n·p_{n-1} + p_{n-2}
        ETInteger a_etint = ETInteger::from_etvalue(a);  // GMP direct, no int64
        ETInteger p_next = a_etint * p_curr + p_prev;
        ETInteger q_next = a_etint * q_curr + q_prev;

        // Check if q exceeds max_q
        if (q_next > max_q) {
            break;
        }

        // Update
        p_prev = p_curr;
        q_prev = q_curr;
        p_curr = p_next;
        q_curr = q_next;
        best_p = p_curr;
        best_q = q_curr;

        // If fractional part is zero, we have exact representation
        if (frac_part.is_zero()) {
            break;
        }
    }

    // Compute error: |r| - p/q
    ETValue p_val = ETValue::from_integer(best_p);
    ETValue q_val = ETValue::from_integer(best_q);
    ETValue approx_val = p_val / q_val;
    ETValue error = math::abs(abs_r - approx_val);

    return {best_p, best_q, error};
}

void apply_simplicity(ProjectionResult& proj, const RationalApprox& approx) {
    // p_plus_q = |p| + |q|
    ETInteger abs_p = approx.p; // already absolute from best_rational_approx
    proj.p_plus_q = abs_p + approx.q;

    // simplicity = 100/max(1, p+q)
    ETValue hundred(int64_t(100));
    ETValue pq_val = ETValue::from_integer(proj.p_plus_q);
    ETValue one(int64_t(1));
    ETValue denom = (pq_val < one) ? one : pq_val;
    proj.elegance_simplicity = hundred / denom;

    // universal = symmetry × tightness × simplicity
    proj.elegance_universal = proj.elegance_symmetry * proj.tightness * proj.elegance_simplicity;
    proj.has_simplicity = true;
}

// ============================================================================
// Section 6: The Projection Formula — Π_N(r) = (k, d, ε)
//
// Paper §5, Definition 5.1:
//   k = round(N·log₂(r))
//   g = gcd(|k|, N)     [with convention g = N when k = 0]
//   d = N/g
//   ε = (N·log₂(r) − k) × 1200/N   [cents]
//
// This IS the master equation P ∘ D ∘ T = E at single-ratio scope:
//   r = P-content (featureless positive real)
//   N, lattice structure = D-content (finite constraints)
//   round() = T-act (resolution of continuous to discrete)
// ============================================================================

ProjectionResult project(const ETValue& r, const ETInteger& N) {
    // ── Validate inputs ─────────────────────────────────────────────────
    if (!r.is_positive()) {
        if (r.is_zero()) {
            throw ETError(ETError::Code::ANNIHILATION_APPROACH,
                          "lattice::project",
                          "r = 0 — annihilation boundary approached. "
                          "The locus r=0 is not in the domain of Π_N: "
                          "log₂(0) = −∞, the projection diverges.");
        }
        throw ETError(ETError::Code::INVALID_INPUT,
                      "lattice::project",
                      "r must be positive (r ∈ ℝ⁺). "
                      "The lattice is a structure on (ℝ⁺, ×), not on ℝ.",
                      r.to_string(30));
    }
    if (!N.is_positive()) {
        throw ETError(ETError::Code::INVALID_INPUT,
                      "lattice::project",
                      "N must be a positive integer",
                      N.to_string());
    }

    ProjectionResult result;
    result.N = N;

    // ── Steps 1–6: Core projection using the bijection as teleporter ──
    //
    // The bijection Π_N⁻¹(k,ε) = 2^((k + ε·N/1200)/N) is algebraically
    // exact: Π_N⁻¹(Π_N(r)) = r. We USE this identity in the computation
    // rather than fighting its numerical shadow.
    //
    // Problem with naive formula: ε = (N·log₂(r) − k) × 1200/N
    //   → subtracts two nearly-equal values (catastrophic cancellation)
    //   → for on-lattice values like √2, the composition sqrt→log₂
    //     produces ULP noise that appears as nonzero ε
    //
    // Solution: compute the lattice point L_k = 2^(k/N) via pullback,
    // then compare r to L_k DIRECTLY. If equal at storage precision,
    // ε = 0 by the bijection identity. If not, compute via the stable
    // formula ε = 1200·log₂(r/L_k) — the ratio r/L_k ≈ 1 + small,
    // so log₂(1+x) is perfectly conditioned. Zero cancellation.
    //
    // Why the direct comparison works: MPFR correctly rounds both
    // mpfr_sqrt(2) and mpfr_exp2(0.5) to the same 1200-bit float.
    // Any value that IS 2^(k/N) will match its lattice point exactly
    // at storage precision. The bijection teleports the comparison.

    // ── Step 1: Determine k at 2× precision ─────────────────────────
    // Higher precision ensures correct rounding of k even when
    // N·log₂(r) is near a half-integer (rounding boundary).
    constexpr mpfr_prec_t PROJ_PREC = 2 * ET_PRECISION_BITS;  // 2400

    mpfr_t hp_log2r, hp_N, hp_Nlog2r, hp_k;
    mpfr_init2(hp_log2r,  PROJ_PREC);
    mpfr_init2(hp_N,      PROJ_PREC);
    mpfr_init2(hp_Nlog2r, PROJ_PREC);
    mpfr_init2(hp_k,      PROJ_PREC);

    mpfr_log2(hp_log2r, r.raw(), MPFR_RNDN);
    mpfr_set_z(hp_N, N.raw(), MPFR_RNDN);
    mpfr_mul(hp_Nlog2r, hp_N, hp_log2r, MPFR_RNDN);
    mpfr_round(hp_k, hp_Nlog2r);

    // Extract k as ETInteger via GMP (no int64 bottleneck)
    {
        mpz_t k_mpz;
        mpz_init(k_mpz);
        mpfr_get_z(k_mpz, hp_k, MPFR_RNDN);
        result.k = ETInteger::from_mpz(k_mpz);
        mpz_clear(k_mpz);
    }

    mpfr_clear(hp_log2r);
    mpfr_clear(hp_N);
    mpfr_clear(hp_Nlog2r);
    mpfr_clear(hp_k);

    // ── Step 2: Sign ─────────────────────────────────────────────────
    result.sign = 1;  // r ∈ ℝ⁺ always

    // ── Step 3: g = gcd(|k|, N), d = N/g ────────────────────────────
    if (result.k.is_zero()) {
        result.g = N;
    } else {
        ETInteger abs_k = result.k;
        if (abs_k.is_negative()) {
            abs_k = -abs_k;
        }
        result.g = intmath::gcd(abs_k, N);
    }
    result.d = N / result.g;

    // ── Step 4: Compute the lattice point L_k = 2^(k/N) ─────────────
    // This IS the pullback of (k, 0, N) — the bijection teleporter.
    // At storage precision, L_k is the correctly-rounded 2^(k/N).
    ETValue k_val = ETValue::from_integer(result.k);
    ETValue N_val = ETValue::from_integer(N);
    ETValue k_over_N = k_val / N_val;
    ETValue L_k = math::exp2(k_over_N);

    // ── Step 5: ε via bijection comparison ───────────────────────────
    // The algebraic identity: if r = 2^(k/N) then ε = 0.
    // At storage precision: if r == L_k, then r IS on the lattice.
    // MPFR correct rounding guarantees: any r that truly equals 2^(k/N)
    // will produce the same 1200-bit float as L_k. The comparison is exact.
    if (r == L_k) {
        // ON THE LATTICE — ε = 0 by the bijection identity.
        result.eps = ETValue();  // exact zero
        // Exact rational form: 0/1
        result.eps_rational_num = ETInteger(int64_t(0));
        result.eps_rational_den = ETInteger(int64_t(1));
        result.has_eps_rational = true;
    } else {
        // OFF THE LATTICE — compute ε via the stable ratio formula.
        // ε = 1200 · log₂(r / L_k)
        //
        // Why this is stable: r/L_k ≈ 1 + δ where |δ| is small.
        // log₂(1 + δ) ≈ δ/ln(2) — perfectly conditioned near 1.
        // No catastrophic cancellation (vs the naive N·log₂(r) − k).
        ETValue ratio = r / L_k;
        ETValue log2_ratio = math::log2(ratio);
        ETValue twelve_hundred(int64_t(1200));
        result.eps = twelve_hundred * log2_ratio;
        // Rational form not determined at projection time — Module 5 (CF method) will populate
        result.eps_rational_num = ETInteger(int64_t(0));
        result.eps_rational_den = ETInteger(int64_t(0));
        result.has_eps_rational = false;
    }

    // ε in micro-cents (lossless integer for |ε| < 50)
    result.eps_micros = eps_to_microcents(result.eps);

    // ── Step 7: Derived properties ───────────────────────────────────

    // Factorization of d
    auto d_factors = intmath::factorize(result.d);
    result.d_factorization = factorization_to_string(d_factors);

    // Gaussian signature
    result.gaussian_sig = compute_gaussian_signature(result.d);

    // Coprime skeleton: gcd(|k|, N) == 1 ⟹ d == N
    result.coprime_skeleton = (result.g == ETInteger(int64_t(1)));

    // Tightness: 100/(100+|ε|)
    result.tightness = compute_tightness(result.eps);

    // ∂I distance: |ε|/50
    result.di_distance = compute_di_distance(result.eps);

    // Manifold state
    result.manifold_state = compute_manifold_state(result.eps, result.d, N);

    // Elegance symmetry: N/d
    result.elegance_symmetry = compute_elegance_symmetry(N, result.d);

    // Simplicity not yet computed (requires rational approximation)
    result.has_simplicity = false;
    result.elegance_simplicity = ETValue(int64_t(0));
    result.elegance_universal = ETValue(int64_t(0));
    result.p_plus_q = ETInteger(int64_t(0));

    // Coupling ξ(d) = 137/((d-1)²+16)
    result.coupling_xi = compute_coupling_xi(result.d);

    // Impedance A₀(d) = (d-1)²+16
    result.impedance_a0 = compute_impedance(result.d);

    // Variance V(n,k) — n = d, k = fold depth
    // The fold depth in this context is the power of 2 in g = gcd(|k|,N)
    // For a simpler computation, use d as n and the lattice k as fold
    // But the canonical definition is V(n,k) = (n²-1)/(12·2^k) where
    // n is the sublattice cardinality (d) and k is the fold/observation depth.
    // For a first projection, use the lattice k coordinate as the fold depth.
    auto [var_val, var_valid] = compute_variance(result.d, result.k);
    result.variance_vnk = var_val;
    result.has_variance = var_valid;

    // FQG quadrant (default: real-axis perspective)
    result.fqg_quadrant = compute_fqg_quadrant(result.d, N, 'r');

    // Palindromic partner: d ↔ (N-d)
    result.palindromic_partner_d = compute_palindromic_partner(result.d, N);

    // Quintic tension τ₅
    auto [qt_val, qt_valid] = compute_quintic_tension(result.d, result.eps, N);
    result.quintic_tension_cents = qt_val;
    result.has_quintic = qt_valid;

    return result;
}

// Convenience overload
ProjectionResult project(const ETValue& r, int64_t N) {
    return project(r, ETInteger(N));
}

// ============================================================================
// Section 7: Bijection Pullback — Π_N⁻¹(k, ε, N) = r
//
// Paper §12, Theorem 12.1 (Losslessness):
//   Π_N⁻¹(k, d, ε) = 2^((k + ε·N/1200)/N)
//
// The composition Π_N⁻¹ ∘ Π_N is the identity on ℝ⁺.
// The d component is redundant (determined by k and N) but provides
// the structural classification.
// ============================================================================

ETValue pullback(const ETInteger& k, const ETValue& eps, const ETInteger& N) {
    // r = 2^((k + ε·N/1200)/N)
    ETValue k_val = ETValue::from_integer(k);
    ETValue N_val = ETValue::from_integer(N);
    ETValue twelve_hundred(int64_t(1200));

    // exponent = (k + ε·N/1200) / N
    ETValue eps_contribution = eps * N_val / twelve_hundred;
    ETValue numerator = k_val + eps_contribution;
    ETValue exponent = numerator / N_val;

    // r = 2^exponent
    return math::exp2(exponent);
}

ETValue pullback(const ProjectionResult& proj) {
    return pullback(proj.k, proj.eps, proj.N);
}

// ============================================================================
// Section 8: k-Arithmetic
//
// The Sempaevum's native arithmetic on lattice coordinates:
//   Multiplication: r₁ × r₂ → k₁ + k₂ (log₂ additivity)
//   Reciprocation:  1/r → -k (log₂ negation)
//   Powers:         r^n → n·k (log₂ scaling)
//
// The k-coordinates combine exactly (integer arithmetic).
// The ε residuals require value-space computation + reprojection:
//   combined_r = pullback(k₁,ε₁,N) OP pullback(k₂,ε₂,N)
//   Then reproject combined_r to get the exact result.
//
// This two-step process (k-arithmetic on integers, ε in value space)
// is the structural reality: k captures the lattice-exact part,
// ε captures the sub-lattice residual that must be resolved in ℝ⁺.
// ============================================================================

KArithResult k_add(
    const ETInteger& k1, const ETValue& eps1,
    const ETInteger& k2, const ETValue& eps2,
    const ETInteger& N) {
    // Multiplication: r₁ × r₂
    // In value space: recover both values, multiply, reproject
    ETValue r1 = pullback(k1, eps1, N);
    ETValue r2 = pullback(k2, eps2, N);
    ETValue product = r1 * r2;

    // Reproject to get the exact result on the lattice
    // The k-coordinate of the product is k₁ + k₂ (exact integer sum)
    ETInteger k_result = k1 + k2;

    // The ε of the product must be computed by reprojection
    // ε_result = (N·log₂(product) − k_result) × 1200/N
    ETValue N_val = ETValue::from_integer(N);
    ETValue log2_prod = math::log2(product);
    ETValue N_log2_prod = N_val * log2_prod;
    ETValue k_result_val = ETValue::from_integer(k_result);
    ETValue twelve_hundred(int64_t(1200));
    ETValue eps_result = (N_log2_prod - k_result_val) * twelve_hundred / N_val;

    return {k_result, eps_result, product};
}

KArithResult k_negate(
    const ETInteger& k, const ETValue& eps,
    const ETInteger& N) {
    // Reciprocation: 1/r → -k, and ε needs value-space computation
    ETValue r = pullback(k, eps, N);
    ETValue reciprocal = ETValue(int64_t(1)) / r;

    ETInteger k_result = -k;

    // Reproject for ε
    ETValue N_val = ETValue::from_integer(N);
    ETValue log2_recip = math::log2(reciprocal);
    ETValue N_log2_recip = N_val * log2_recip;
    ETValue k_result_val = ETValue::from_integer(k_result);
    ETValue twelve_hundred(int64_t(1200));
    ETValue eps_result = (N_log2_recip - k_result_val) * twelve_hundred / N_val;

    return {k_result, eps_result, reciprocal};
}

KArithResult k_scale(
    const ETInteger& k, const ETValue& eps,
    const ETInteger& n,
    const ETInteger& N) {
    // Power: r^n → n·k, and ε needs value-space computation
    ETValue r = pullback(k, eps, N);
    ETValue n_val = ETValue::from_integer(n);
    ETValue powered = math::pow(r, n_val);

    ETInteger k_result = n * k;

    // Reproject for ε
    ETValue N_val = ETValue::from_integer(N);
    ETValue log2_pow = math::log2(powered);
    ETValue N_log2_pow = N_val * log2_pow;
    ETValue k_result_val = ETValue::from_integer(k_result);
    ETValue twelve_hundred(int64_t(1200));
    ETValue eps_result = (N_log2_pow - k_result_val) * twelve_hundred / N_val;

    return {k_result, eps_result, powered};
}

} // namespace et::lattice
