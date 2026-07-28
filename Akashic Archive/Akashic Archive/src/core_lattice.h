// ============================================================================
// core_lattice.h — Module 2: Core Lattice Engine (Level 1)
//
// The projection formula Π_N(r) = (k, d, ε). The bijection pullback
// Π_N⁻¹(k, d, ε) = r. k-arithmetic: k-addition (multiplication),
// k-negation (reciprocation), k-scaling (powers). All derived property
// materialization.
//
// Pure math on lattice coordinates. No I/O, no files, no threads.
// Every module that works with lattice addresses calls this.
//
// Dependencies: Module 1 (Precision Stack) only.
//
// ET Derivation Standard:
//   Projection formula from Paper §5 Definition 5.1:
//     k = round(N·log₂(r))
//     g = gcd(|k|, N)
//     d = N/g
//     ε = (N·log₂(r) − k) × 1200/N  [cents]
//
//   Bijection pullback from Paper §12 Theorem 12.1:
//     Π_N⁻¹(k, d, ε) = 2^((k + ε·N/1200)/N)
//
//   Gaussian signature from Paper §6:
//     p=2: Ramified (R)
//     p≡1 (mod 4): Split (S)
//     p≡3 (mod 4): Inert (I)
//
// Identification Principle applied to this module:
//   P = the lattice L_N = { 2^(k/N) : k ∈ ℤ } ⊂ (ℝ⁺, ×)
//   D = the projection formula, k-arithmetic rules, derived properties
//   T = the rounding operator (the T-act of projection)
//
// P ∘ D ∘ T = E
// ============================================================================

#pragma once

#include "precision_stack.h"

#include <string>
#include <vector>
#include <cstdint>

namespace et::lattice {

// ============================================================================
// Gaussian Prime Classification
//
// For each prime p in the factorization of d, classify its behavior
// over ℤ[i] (Gaussian integers):
//   p = 2:           Ramified (R) — 2 = -i(1+i)², splits with multiplicity
//   p ≡ 1 (mod 4):   Split (S)   — p = π·π̄, factors into conjugate primes
//   p ≡ 3 (mod 4):   Inert (I)   — p remains prime in ℤ[i]
//
// The three classes parallel the three Cardinals P, D, T:
//   Ramified → P-class (substrate doubling)
//   Inert    → D-class (constraint preservation on real axis)
//   Split    → D+T-class (factoring across real/imaginary split)
// ============================================================================

enum class GaussianClass : uint8_t {
    RAMIFIED = 0,   // p = 2
    INERT    = 1,   // p ≡ 3 (mod 4)
    SPLIT    = 2    // p ≡ 1 (mod 4)
};

// Classification of a single prime power factor
struct GaussianFactor {
    ETInteger prime;
    int       exponent;
    GaussianClass gclass;
};

// Full Gaussian signature of a d-value
struct GaussianSignature {
    std::vector<GaussianFactor> factors;

    // Pre-computed summary flags
    bool is_all_inert;       // true iff every prime factor is Inert
    bool is_all_split;       // true iff every prime factor is Split
    bool is_ramified_present; // true iff at least one factor is Ramified

    // Human-readable signature string: e.g., "R^1·I^2·S^1" or "I^3"
    std::string signature_string;
};

// ============================================================================
// Manifold State
//
// The four states from power set of {P,D,T} with cardinality ≥ 2:
//   {P,D,T} = Exception      — fully substantiated, zero variance
//   {P,D}   = Unsubstantiated — potential without agency
//   {D,T}   = Mediation       — T navigating D without fixed substrate
//   {P,T}   = Incoherence     — self-defeating, forbidden (open set)
// ============================================================================

enum class ManifoldState : uint8_t {
    EXCEPTION       = 0,   // {P,D,T} — ε = 0 and d divides N
    UNSUBSTANTIATED = 1,   // {P,D}   — ε ≠ 0, not at ∂I boundary
    MEDIATION       = 2,   // {D,T}   — at structural intermediary
    INCOHERENCE     = 3    // {P,T}   — near ∂I boundary (|ε| ≥ 50 cents)
};

// Convert manifold state to the canonical string used in schema
[[nodiscard]] const char* manifold_state_string(ManifoldState state);

// ============================================================================
// FQG Quadrant — Force Quadrant Grid classification
//
// From the 144-cell 12×12 grid (Paper §10):
//   SR = Simple Real    (d divides 12, real-axis projection)
//   CR = Complex Real   (d does NOT divide 12, real-axis projection)
//   SI = Simple Imaginary (d divides 12, imaginary-axis projection)
//   CI = Complex Imaginary (d does NOT divide 12, imaginary-axis projection)
//
// At N=12, divisors of 12 are {1,2,3,4,6,12} — the "simple" families.
// Non-divisors at higher N (d=5,7,8,9,10,11,...) are "complex" families.
// ============================================================================

enum class FQGQuadrant : uint8_t {
    SR = 0,   // Simple Real
    CR = 1,   // Complex Real
    SI = 2,   // Simple Imaginary
    CI = 3    // Complex Imaginary
};

// Convert FQG quadrant to canonical string
[[nodiscard]] const char* fqg_quadrant_string(FQGQuadrant q);

// ============================================================================
// ProjectionResult — The complete output of Π_N(r)
//
// Contains the core triple (k, d, ε) plus the sign, plus all derived
// properties that are materialized at projection time for O(1) query.
//
// Every field is computed from the core triple + N. Nothing is optional
// that can be computed (NULL-ability is only for properties requiring
// external context like detection_status or curvature_class).
// ============================================================================

struct ProjectionResult {
    // ── Core triple ────────────────────────────────────────────────────
    int           sign;            // +1 or -1 (sign of r relative to 1)
    ETInteger     k;               // lattice coordinate (arbitrary precision)
    ETInteger     d;               // sublattice family = N/gcd(|k|,N)
    ETValue       eps;             // descriptor gap in cents (full 361-dps)
    // ── ε in micro-cents ─────────────────────────────────────────────
    int32_t       eps_micros;      // ε in micro-cents (signed integer; lossless for |ε|<50)

    // ── ε as exact rational (when known) ───────────────────────────────
    // Populated when the exact rational form of ε is determinable:
    //   - ε = 0 (on-lattice via bijection teleporter): num = 0, den = 1
    //   - Future: Module 5 (CF method) populates when CF identifies exact form
    // When has_eps_rational = false, these are unset (not yet determined).
    ETInteger     eps_rational_num;   // exact ε numerator
    ETInteger     eps_rational_den;   // exact ε denominator (positive)
    bool          has_eps_rational;   // whether the rational form is known
    ETInteger     N;               // resolution at which projection was computed

    // ── GCD value ──────────────────────────────────────────────────────
    ETInteger     g;               // gcd(|k|, N) — the bridge between k and d

    // ── Factorization of d ─────────────────────────────────────────────
    std::string   d_factorization; // e.g., "2^3·3·5·7"

    // ── Gaussian signature ─────────────────────────────────────────────
    GaussianSignature gaussian_sig;

    // ── Coprime skeleton ───────────────────────────────────────────────
    bool          coprime_skeleton; // gcd(|k|,N) == 1 ⟹ d == N

    // ── Tightness (how close to the lattice point) ─────────────────────
    // t(r) = 100/(100+|ε|)  where ε is in cents
    // Range: [100/150, 1] = [2/3, 1] for |ε| ∈ [0, 50]
    ETValue       tightness;

    // ── ∂I distance (normalized distance to incoherence boundary) ──────
    // di_dist = |ε|/50  — 0 at lattice point, 1 at ∂I boundary
    ETValue       di_distance;

    // ── Manifold state ─────────────────────────────────────────────────
    ManifoldState manifold_state;

    // ── Elegance score factors ─────────────────────────────────────────
    // E(r) = (N/d) × (100/(100+|ε|)) × (100/(p+q))
    //       = symmetry × tightness × simplicity
    ETValue       elegance_symmetry;   // N/d
    // Note: elegance_simplicity and elegance_universal require p/q
    // rational approximation, which is computed separately when available.
    // These are set to valid values only when p_plus_q is known.
    ETValue       elegance_simplicity; // 100/max(1, p+q); set when p_plus_q known
    ETValue       elegance_universal;  // product of all three; set when p_plus_q known
    ETInteger     p_plus_q;            // |p|+|q| from lowest-terms rational form
    bool          has_simplicity;      // whether simplicity/universal are computed

    // ── Coupling ξ(d) = 137/((d-1)²+S²) where S=4 ────────────────────
    // Dynamically computed for any d, not capped at d∈{1..12}
    ETValue       coupling_xi;

    // ── Impedance A₀(d) = (d-1)²+S² where S=4 ────────────────────────
    ETValue       impedance_a0;

    // ── Variance V(n,k) = (n²-1)/(12·2^k) ─────────────────────────────
    // n = d (sublattice family), k = fold depth
    // Valid when k >= 0; for extreme k, may be near-zero
    ETValue       variance_vnk;
    bool          has_variance;        // false if k is extreme

    // ── FQG quadrant ───────────────────────────────────────────────────
    // Requires knowing the perspective (real_axis vs imaginary_axis)
    // Default: real_axis for standard projections
    FQGQuadrant   fqg_quadrant;

    // ── Palindromic partner ────────────────────────────────────────────
    // d_partner = 12-d for d∈{1..11}, self for d∈{6,12}
    // At N=12: well-defined for d∈{1..12}
    // At arbitrary N: computed as N-d for d∈{1..N-1}, self for d∈{N/2, N}
    ETInteger     palindromic_partner_d;

    // ── Quintic tension τ₅ ─────────────────────────────────────────────
    // The quintic residual; specific to d=5 sublattice interactions
    // NULL (has_quintic = false) when not applicable
    ETValue       quintic_tension_cents;
    bool          has_quintic;
};

// ============================================================================
// Core Projection Function
//
// Π_N(r) = (k, d, ε)
//
// Given a positive real r and resolution N, compute the full projection
// including all derived properties.
//
// Preconditions:
//   - r must be positive (r > 0); r = 0 hits annihilation boundary
//   - N must be a positive integer (N > 0)
//
// The function computes all fields of ProjectionResult except:
//   - elegance_simplicity, elegance_universal (require rational approx)
//   - detection_status, curvature_class (require external context)
//
// Throws:
//   ETError::ANNIHILATION_APPROACH if r <= 0
//   ETError::INVALID_INPUT if N <= 0
// ============================================================================

[[nodiscard]] ProjectionResult project(const ETValue& r, const ETInteger& N);

// Convenience overload with int64_t N
[[nodiscard]] ProjectionResult project(const ETValue& r, int64_t N);

// ============================================================================
// Bijection Pullback
//
// Π_N⁻¹(k, ε, N) = 2^((k + ε·N/1200)/N)
//
// Reconstruct the original value r from the projection triple.
// The Losslessness Theorem (Paper §12 Theorem 12.1) guarantees
// this is the exact inverse of project().
//
// Preconditions:
//   - N must be positive
//   - ε should be in [-50, +50] cents (but the formula is valid for any ε)
// ============================================================================

[[nodiscard]] ETValue pullback(const ETInteger& k, const ETValue& eps, const ETInteger& N);

// Convenience: pullback from a ProjectionResult
[[nodiscard]] ETValue pullback(const ProjectionResult& proj);

// ============================================================================
// k-Arithmetic
//
// The Sempaevum's native arithmetic on lattice coordinates:
//   Multiplication: k₁ + k₂ (log₂ additivity)
//   Reciprocation:  -k       (log₂ negation)
//   Powers:         n·k      (log₂ scaling)
//
// These are structurally exact (integer operations on k).
// The ε residuals combine in value-space, requiring reprojection.
// ============================================================================

// Multiply two lattice values: product_k = k₁ + k₂
// Also computes combined ε (requires value-space computation + reprojection)
struct KArithResult {
    ETInteger k_result;           // resulting k coordinate
    ETValue   eps_result;         // resulting ε (from value-space)
    ETValue   r_result;           // the actual product value at 361 dps
};

// k-addition: multiplication of lattice values
// r₁ × r₂ → k₁ + k₂ with ε reprojection
[[nodiscard]] KArithResult k_add(
    const ETInteger& k1, const ETValue& eps1,
    const ETInteger& k2, const ETValue& eps2,
    const ETInteger& N);

// k-negation: reciprocation
// 1/r → -k, -ε
[[nodiscard]] KArithResult k_negate(
    const ETInteger& k, const ETValue& eps,
    const ETInteger& N);

// k-scaling: integer power
// r^n → n·k with ε reprojection
[[nodiscard]] KArithResult k_scale(
    const ETInteger& k, const ETValue& eps,
    const ETInteger& n,
    const ETInteger& N);

// ============================================================================
// Derived Property Functions
//
// Each function computes one category of derived properties.
// The project() function calls all of these internally, but they
// are exposed individually for cases where only specific properties
// are needed (e.g., computing coupling for a known d without projection).
// ============================================================================

// Compute Gaussian signature from d's prime factorization
[[nodiscard]] GaussianSignature compute_gaussian_signature(const ETInteger& d);

// Classify a single prime by its Gaussian class
[[nodiscard]] GaussianClass classify_gaussian_prime(const ETInteger& p);

// Compute d_factorization string from factors
// e.g., [(2,3),(3,1),(5,1),(7,1)] → "2³·3·5·7"
[[nodiscard]] std::string factorization_to_string(
    const std::vector<std::pair<ETInteger, int>>& factors);

// Compute coupling ξ(d) = 137/((d-1)² + S²) where S = ET_S = 4
// Works for any positive d, not capped at 12
[[nodiscard]] ETValue compute_coupling_xi(const ETInteger& d);

// Compute impedance A₀(d) = (d-1)² + S² where S = ET_S = 4
[[nodiscard]] ETValue compute_impedance(const ETInteger& d);

// Compute tightness: t = 100/(100+|ε|)
[[nodiscard]] ETValue compute_tightness(const ETValue& eps);

// Compute ∂I distance: |ε|/50
[[nodiscard]] ETValue compute_di_distance(const ETValue& eps);

// Compute variance V(n,k) = (n²-1)/(12·2^k)
// n = sublattice family d (cardinality), k = fold depth
// Returns (value, is_valid) where is_valid = false if k causes underflow
[[nodiscard]] std::pair<ETValue, bool> compute_variance(
    const ETInteger& n, const ETInteger& k);

// Compute elegance symmetry factor: N/d
[[nodiscard]] ETValue compute_elegance_symmetry(
    const ETInteger& N, const ETInteger& d);

// Compute full elegance score: symmetry × tightness × simplicity
// Returns (score, true) if p_plus_q is available, (zero, false) otherwise
[[nodiscard]] std::pair<ETValue, bool> compute_elegance_universal(
    const ETValue& symmetry, const ETValue& tightness,
    const ETInteger& p_plus_q);

// Determine FQG quadrant for a given d at resolution N
// perspective: 'r' for real axis, 'i' for imaginary axis
[[nodiscard]] FQGQuadrant compute_fqg_quadrant(
    const ETInteger& d, const ETInteger& N, char perspective);

// Compute palindromic partner of d at resolution N
// At N=12: d↔(12-d) for d∈{1..11}, self for d∈{6,12}
// Generalized: d↔(N-d) for d∈{1..N-1}, self for d∈{N/2, N}
[[nodiscard]] ETInteger compute_palindromic_partner(
    const ETInteger& d, const ETInteger& N);

// Determine manifold state from projection properties
// {P,D,T} Exception: ε = 0 exactly
// {P,T} Incoherence: |ε| ≥ 50 cents (at ∂I boundary)
// {D,T} Mediation: specific structural criteria
// {P,D} Unsubstantiated: default otherwise
[[nodiscard]] ManifoldState compute_manifold_state(
    const ETValue& eps, const ETInteger& d, const ETInteger& N);

// Compute ε in micro-cents (signed integer) — lossless for |ε| < 50 cents
// ε_micros = round(ε × 1000) where ε is in cents
// Range: [-50000, +50000] for valid projections
[[nodiscard]] int32_t eps_to_microcents(const ETValue& eps);

// Compute quintic tension τ₅ (d=5 sublattice specific)
// Returns (value, true) if applicable, (zero, false) otherwise
[[nodiscard]] std::pair<ETValue, bool> compute_quintic_tension(
    const ETInteger& d, const ETValue& eps, const ETInteger& N);

// ============================================================================
// Rational Approximation (for elegance simplicity factor)
//
// Find the best rational p/q approximation of r (the original value)
// for computing the simplicity factor 100/(p+q).
// Uses continued fraction expansion.
// ============================================================================

struct RationalApprox {
    ETInteger p;         // numerator (absolute value)
    ETInteger q;         // denominator (positive)
    ETValue   error;     // |r - p/q|
};

// Find best rational approximation with denominator ≤ max_q
// Uses continued fraction convergents
[[nodiscard]] RationalApprox best_rational_approx(
    const ETValue& r, const ETInteger& max_q);

// Update a ProjectionResult with the simplicity factor from rational approx
void apply_simplicity(ProjectionResult& proj, const RationalApprox& approx);

} // namespace et::lattice
