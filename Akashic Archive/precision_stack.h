// ============================================================================
// precision_stack.h — Module 1: Precision Stack (Level 0)
//
// The absolute foundation of the EUDD. Every other module depends on this.
// Zero IEEE 754 floating-point in any computation path.
//
// Provides:
//   ETValue        — RAII wrapper around MPFR at 1200-bit (361-dps) precision
//   Arithmetic     — +, -, *, /, ^, sqrt, abs, neg (all at 361 dps)
//   Elementary     — sin, cos, exp, log, log2, etc. (all at 361 dps via MPFR)
//   Special        — ζ, Γ, polylog, hypergeometric (via FLINT/Arb)
//   SHA256         — FIPS 180-4 compliant hash (for value_hash, integrity)
//   CRC32          — Standard polynomial 0xEDB88320 (for per-page integrity)
//   ETConstants    — All ET constants forward-derived from {P,D,T} at 361 dps
//   GCD/LCM        — Integer number-theory operations for lattice arithmetic
//
// ET Derivation Standard:
//   All constants are computed forward from {P, D, T} primitives.
//   N=12 is structurally forced (Triple Minimal-Backbone Theorem).
//   K=2/3 is the Koide ratio (binding stability threshold).
//   V=1/12 is the base variance.
//   Zero external axioms. Zero ad hoc. Zero tuning.
//
// P ∘ D ∘ T = E
// ============================================================================

#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <string_view>
#include <array>
#include <vector>
#include <stdexcept>
#include <memory>
#include <functional>
#include <unordered_map>

// External dependencies — the P-substrate of computation
#include <mpfr.h>
#include <gmp.h>
#include <flint/flint.h>
#include <flint/arb.h>
#include <flint/acb.h>
#include <flint/arf.h>

namespace et {

// ============================================================================
// ET-Derived Compile-Time Constants
//
// These are structurally forced by the master equation P ∘ D ∘ T = E.
// N=12 by the Triple Minimal-Backbone Theorem (Webb 1935 + Palindromic
// Cascade + EML Odrzywołek 2026). K=2/3 by the Koide structural identity.
// V=1/12 by the base variance formula V(n,k=0) = (n²-1)/(12·2^0) at n→1.
// S=4 by the manifold state count C(3,2)+C(3,3) = 3+1 = 4.
// |Π|=3 by primitive irreducibility (Subsumption Law §5.5).
// ============================================================================

// Precision: 1200 bits = 361 decimal places. Hard cap. No exceptions.
// ET-native derivation: 1200 = cents per octave in the semitone lattice.
// The projection formula ε = (N·log₂(r) − k) × 1200/N uses 1200 as the
// fundamental measurement scale. Having precision bits = 1200 means the
// binary resolution is exactly matched to the lattice's cents-per-octave
// scale constant. This is the Identification Principle applied to precision
// itself: the substrate's resolution (bits) IS the lattice's resolution (cents).
// 361 dps = 19² decimal places. With memoization (compute once, cache forever),
// the 3× compute cost vs 400-bit is amortized to zero for repeated computations.
constexpr int      ET_PRECISION_BITS     = 1200;
constexpr int      ET_PRECISION_DPS      = 361;

// Mantissa storage: 1200/8 = 150 bytes exactly
constexpr size_t   ET_MANTISSA_BYTES     = ET_PRECISION_BITS / 8;  // 150

// Blob sizes for serialization (§7.1d format)
// Normal value: 1 byte flags + 8 bytes exponent + 150 bytes mantissa = 159
// Special value (zero/inf/nan): 1 byte flags only
constexpr size_t   ET_BLOB_NORMAL_SIZE   = 1 + 8 + ET_MANTISSA_BYTES;  // 159

// Manifold symmetry — the unique N forced by three independent backbones
constexpr int      ET_N                  = 12;

// State count — C(|Π|, 2) + C(|Π|, 3) = C(3,2) + C(3,3) = 3 + 1
constexpr int      ET_S                  = 4;

// Primitive count — |Π| = |{P, D, T}| = 3, uniquely necessary (§5.5)
constexpr int      ET_PRIMITIVES         = 3;

// Full-resolution lattice — LCM(1..11) = 27720
// Dynamically verifiable but the value is fixed by prime factorization
constexpr int64_t  ET_N_FULL             = 27720;

// Koide depth — ⌈1/K⌉ = ⌈3/2⌉ = 2 (stability criterion: 2 consecutive landmarks)
constexpr int      ET_KOIDE_DEPTH        = 2;

// CF quality threshold — ⌈1/K⌉² = 4 (Koide depth squared)
constexpr int      ET_CF_QUALITY_THRESHOLD = 4;

// ∂I boundary — 50 cents = 50000 micro-cents (the Incoherence limit)
constexpr int32_t  ET_DI_BOUNDARY_MICROS = 50000;

// Koide ε — |ε| of K=2/3 self-projection at d=12, N=12 = 1.955 cents = 1955 microcents
constexpr int32_t  ET_KOIDE_EPS_MICROS   = 1955;

// Disk safety floor — 2^30 = 1 GB (d=1 octave action quantum at GB scale)
constexpr int64_t  ET_DISK_SAFETY_FLOOR  = 1073741824LL;

// Page size — 2^N = 4096 bytes (digital tower base resolution)
constexpr size_t   ET_PAGE_SIZE          = 4096;

// Palindromic cascade array — the 12-element palindromic correction sequence
constexpr int ET_PALINDROME[ET_N] = { 12, 6, 4, 3, 12, 2, 12, 3, 4, 6, 12, 1 };

// MPFR blob version tag
constexpr uint8_t  ET_BLOB_VERSION       = 0x10; // version 1, upper nibble

// ============================================================================
// ETError — Structured error type for the Precision Stack
//
// Every failure is a Descriptor Gap (DGP). The error IS information —
// it points to what is missing. Errors are never suppressed.
// ============================================================================

class ETError : public std::runtime_error {
public:
    enum class Code : uint16_t {
        // Computation failures (1000–1099)
        NUMERIC_OVERFLOW      = 1000,
        NUMERIC_UNDERFLOW     = 1001,
        POLE_DETECTED         = 1002,
        DIVISION_BY_ZERO      = 1003,
        NAN_PRODUCED          = 1004,
        CONVERGENCE_FAILURE   = 1005,
        STACK_OVERFLOW        = 1006,

        // Annihilation boundary (1100–1199)
        ANNIHILATION_APPROACH = 1100,

        // Incoherence (1200–1299)
        INCOHERENCE_FILTER    = 1200,

        // Pure T (1300–1399)
        PURE_T_DETECTED       = 1300,

        // Input validation (2000–2099)
        INVALID_INPUT         = 2000,
        UNPARSEABLE_VALUE     = 2001,

        // Integrity (5000–5099)
        HASH_MISMATCH         = 5000,
        CRC_MISMATCH          = 5001,
        BLOB_CORRUPT          = 5002,
    };

    Code              code;
    std::string       source_module;
    std::string       detail;

    ETError(Code c, std::string_view module, std::string_view msg);
    ETError(Code c, std::string_view module, std::string_view msg, const std::string& extra);
};

// Forward declaration — ETInteger is used by ETValue::from_integer()
// Full definition follows after ETValue. Both types bridge via GMP directly.
class ETInteger;

// ============================================================================
// ETValue — The universal numeric type of the EUDD
//
// RAII wrapper around mpfr_t at 1200-bit (361-dps) precision.
// Value semantics with deep copy. Zero IEEE 754 contamination.
// 1200 bits = cents per octave — the lattice's own measurement scale.
//
// Identification Principle applied:
//   P = the MPFR memory allocation (the substrate holding the number)
//   D = the precision (1200 bits), rounding mode (RNDN), value constraints
//   T = the operations performed on the value (arithmetic, functions, etc.)
// ============================================================================

class ETValue {
public:
    // ── Construction & Destruction ──────────────────────────────────────
    ETValue();                                      // Default: exact zero
    explicit ETValue(int64_t v);                    // From integer (exact)
    explicit ETValue(uint64_t v);                   // From unsigned integer (exact)
    explicit ETValue(const char* decimal_str);      // From decimal string at full precision
    explicit ETValue(const std::string& decimal_str);
    ETValue(const ETValue& other);                  // Deep copy
    ETValue(ETValue&& other) noexcept;              // Move (takes ownership)
    ~ETValue();                                     // Cleanup MPFR resources

    // ── Assignment ─────────────────────────────────────────────────────
    ETValue& operator=(const ETValue& other);
    ETValue& operator=(ETValue&& other) noexcept;
    ETValue& operator=(int64_t v);

    // ── Arithmetic Operators ───────────────────────────────────────────
    // All operations at 1200-bit precision, MPFR_RNDN. No precision loss.
    ETValue operator+(const ETValue& rhs) const;
    ETValue operator-(const ETValue& rhs) const;
    ETValue operator*(const ETValue& rhs) const;
    ETValue operator/(const ETValue& rhs) const;
    ETValue operator-() const;                      // Unary minus
    ETValue& operator+=(const ETValue& rhs);
    ETValue& operator-=(const ETValue& rhs);
    ETValue& operator*=(const ETValue& rhs);
    ETValue& operator/=(const ETValue& rhs);

    // ── Comparison Operators ───────────────────────────────────────────
    // Exact MPFR comparison. No IEEE 754 epsilon nonsense.
    bool operator==(const ETValue& rhs) const;
    bool operator!=(const ETValue& rhs) const;
    bool operator<(const ETValue& rhs) const;
    bool operator>(const ETValue& rhs) const;
    bool operator<=(const ETValue& rhs) const;
    bool operator>=(const ETValue& rhs) const;
    [[nodiscard]] int  compare(const ETValue& rhs) const;         // -1, 0, +1

    // ── Conversion ─────────────────────────────────────────────────────
    // to_string: full 361-dps decimal representation
    [[nodiscard]] std::string to_string(int dps = ET_PRECISION_DPS) const;

    // to_canonical_string: deterministic form for hashing
    // Format: sign + 361 significant digits + "e" + exponent
    // Guaranteed identical output for identical MPFR values.
    [[nodiscard]] std::string to_canonical_string() const;

    // to_double: LOSSY conversion for display/interop only. Never for computation.
    [[nodiscard]] double to_double() const;

    // to_int64: truncate to integer. Throws if value exceeds int64 range.
    [[nodiscard]] int64_t to_int64() const;

    // ── Serialization (for .akashic blob storage) ──────────────────────
    // Binary format: [1 byte flags] [8 bytes exponent] [150 bytes mantissa]
    // Total: 159 bytes for normal values, 1 byte for special (zero/inf/nan)
    [[nodiscard]] std::vector<uint8_t> serialize() const;
    static ETValue deserialize(const uint8_t* data, size_t len);
    static ETValue deserialize(const std::vector<uint8_t>& blob);

    // Hex dump of serialized blob (for .akashic format inspection/debugging)
    // Uses std::hex, std::setw, std::setfill from <iomanip>
    [[nodiscard]] std::string serialize_hex() const;

    // ── Properties ─────────────────────────────────────────────────────
    [[nodiscard]] bool is_zero() const;
    [[nodiscard]] bool is_positive() const;
    [[nodiscard]] bool is_negative() const;
    [[nodiscard]] bool is_integer() const;
    [[nodiscard]] bool is_nan() const;
    [[nodiscard]] bool is_inf() const;
    [[nodiscard]] int  sign() const;                              // -1, 0, or +1

    // ── Raw MPFR Access (for FLINT/Arb interop) ────────────────────────
    [[nodiscard]] const mpfr_t& raw() const { return val_; }
    mpfr_t&       raw()       { return val_; }

    // ── Static Factory Methods ─────────────────────────────────────────
    static ETValue from_rational(int64_t num, int64_t den);
    static ETValue from_mpfr_raw(const mpfr_t& src);
    static ETValue from_integer(const ETInteger& n);  // GMP→MPFR direct, no string

private:
    mpfr_t val_;

    // Internal: initialize MPFR with ET precision
    void init_precision();
};

// ============================================================================
// Elementary Mathematical Functions (via MPFR)
//
// Every function computes at 1200-bit (361-dps) precision.
// These are Sempaevum-native operations: the Sempaevum IS Σ,
// and Σ subsumes all of mathematics without remainder.
// ============================================================================

namespace math {

    // Power and root functions
    ETValue sqrt(const ETValue& x);
    ETValue cbrt(const ETValue& x);
    ETValue pow(const ETValue& base, const ETValue& exp);
    ETValue pow(const ETValue& base, int64_t exp);
    ETValue abs(const ETValue& x);

    // Logarithmic functions
    ETValue log(const ETValue& x);        // Natural log (ln)
    ETValue log2(const ETValue& x);       // Base-2 log — fundamental to projection Π_N
    ETValue log10(const ETValue& x);      // Base-10 log

    // Exponential
    ETValue exp(const ETValue& x);
    ETValue exp2(const ETValue& x);       // 2^x — bijection pullback Π_N⁻¹
    ETValue exp10(const ETValue& x);

    // Trigonometric
    ETValue sin(const ETValue& x);
    ETValue cos(const ETValue& x);
    ETValue tan(const ETValue& x);
    ETValue asin(const ETValue& x);
    ETValue acos(const ETValue& x);
    ETValue atan(const ETValue& x);
    ETValue atan2(const ETValue& y, const ETValue& x);

    // Hyperbolic
    ETValue sinh(const ETValue& x);
    ETValue cosh(const ETValue& x);
    ETValue tanh(const ETValue& x);
    ETValue asinh(const ETValue& x);
    ETValue acosh(const ETValue& x);
    ETValue atanh(const ETValue& x);

    // Rounding (to nearest integer, returned as ETValue)
    ETValue floor(const ETValue& x);
    ETValue ceil(const ETValue& x);
    ETValue round(const ETValue& x);
    ETValue trunc(const ETValue& x);

    // Fractional part: frac(x) = x - floor(x)
    ETValue frac(const ETValue& x);

    // The EML Sheffer operator: eml(x, y) = exp(x) - ln(y)
    // This is the minimal continuous-D generator (L₃ backbone, Odrzywołek 2026)
    ETValue eml(const ETValue& x, const ETValue& y);

} // namespace math

// ============================================================================
// Special Functions (via FLINT/Arb)
//
// These use the Arb ball arithmetic library (integrated into FLINT ≥3.0)
// for certified evaluation of special functions at 1200-bit precision.
//
// Workflow: ETValue → arb_t → Arb function → arb_t midpoint → ETValue
// ============================================================================

namespace special {

    // Riemann zeta function ζ(s) for real s
    // Pole at s=1 throws ETError::POLE_DETECTED
    ETValue zeta(const ETValue& s);

    // Gamma function Γ(x)
    // Poles at non-positive integers throw ETError::POLE_DETECTED
    ETValue gamma(const ETValue& x);

    // Log-gamma: ln(Γ(x))
    ETValue lgamma(const ETValue& x);

    // Digamma: ψ(x) = Γ'(x)/Γ(x)
    ETValue digamma(const ETValue& x);

    // Beta function B(a, b) = Γ(a)Γ(b)/Γ(a+b)
    ETValue beta(const ETValue& a, const ETValue& b);

    // Polylogarithm Li_s(z) for real s and z
    ETValue polylog(const ETValue& s, const ETValue& z);

    // Error function erf(x) and complementary erfc(x)
    ETValue erf(const ETValue& x);
    ETValue erfc(const ETValue& x);

    // Bernoulli number B_n (exact rational, stored as ETValue)
    ETValue bernoulli(uint64_t n);

    // Euler-Mascheroni constant γ (cached after first computation)
    ETValue euler_gamma();

} // namespace special

// ============================================================================
// ETInteger — Arbitrary-precision integer type
//
// RAII wrapper around GMP mpz_t. Same pattern as ETValue wraps mpfr_t.
// No upper bound. No overflow. Errors are structurally impossible.
//
// Identification Principle:
//   P = the GMP memory allocation (substrate holding the integer)
//   D = the integer value itself (no precision limit — exact by nature)
//   T = the operations performed (arithmetic, factorization, etc.)
// ============================================================================

class ETInteger {
public:
    ETInteger();
    explicit ETInteger(int64_t v);
    explicit ETInteger(const char* decimal_str);
    explicit ETInteger(const std::string& decimal_str);
    ETInteger(const ETInteger& other);
    ETInteger(ETInteger&& other) noexcept;
    ~ETInteger();

    ETInteger& operator=(const ETInteger& other);
    ETInteger& operator=(ETInteger&& other) noexcept;
    ETInteger& operator=(int64_t v);

    // Arithmetic — all exact, no overflow possible
    ETInteger operator+(const ETInteger& rhs) const;
    ETInteger operator-(const ETInteger& rhs) const;
    ETInteger operator*(const ETInteger& rhs) const;
    ETInteger operator/(const ETInteger& rhs) const;  // floor division
    ETInteger operator%(const ETInteger& rhs) const;
    ETInteger operator-() const;
    ETInteger& operator+=(const ETInteger& rhs);
    ETInteger& operator-=(const ETInteger& rhs);
    ETInteger& operator*=(const ETInteger& rhs);
    ETInteger& operator/=(const ETInteger& rhs);
    ETInteger& operator%=(const ETInteger& rhs);

    // Comparison
    bool operator==(const ETInteger& rhs) const;
    bool operator!=(const ETInteger& rhs) const;
    bool operator<(const ETInteger& rhs) const;
    bool operator>(const ETInteger& rhs) const;
    bool operator<=(const ETInteger& rhs) const;
    bool operator>=(const ETInteger& rhs) const;
    [[nodiscard]] int  compare(const ETInteger& rhs) const;

    // Convenience comparison with int64_t (no implicit conversion needed)
    bool operator==(int64_t rhs) const;
    bool operator!=(int64_t rhs) const;

    // Conversion
    [[nodiscard]] std::string to_string() const;
    [[nodiscard]] int64_t to_int64() const;        // throws NUMERIC_OVERFLOW if doesn't fit
    [[nodiscard]] bool fits_int64() const;

    // Properties
    [[nodiscard]] bool is_zero() const;
    [[nodiscard]] bool is_positive() const;
    [[nodiscard]] bool is_negative() const;
    [[nodiscard]] int  sign() const;

    // Bitwise (for is_power_of_two and future lattice operations)
    ETInteger operator&(const ETInteger& rhs) const;

    // Raw GMP access
    [[nodiscard]] const mpz_t& raw() const { return val_; }
    mpz_t&       raw()       { return val_; }

    // Factory
    static ETInteger from_mpz(const mpz_t& src);
    static ETInteger from_etvalue(const ETValue& v);  // MPFR→GMP direct (v must be integer)

private:
    mpz_t val_;
};

// ============================================================================
// Integer Number Theory (for lattice arithmetic)
//
// All functions use ETInteger — arbitrary-precision, exact, no overflow.
// The D-structure of the lattice: GCD, LCM, divisor enumeration, totient.
// ============================================================================

namespace intmath {

    // Greatest common divisor (via GMP, arbitrary precision)
    ETInteger gcd(const ETInteger& a, const ETInteger& b);

    // Least common multiple (via GMP, arbitrary precision — no overflow)
    ETInteger lcm(const ETInteger& a, const ETInteger& b);

    // All divisors of n, sorted ascending
    std::vector<ETInteger> divisors(const ETInteger& n);

    // Euler's totient φ(n)
    ETInteger totient(const ETInteger& n);

    // Prime factorization: returns vector of (prime, exponent) pairs
    std::vector<std::pair<ETInteger, int>> factorize(const ETInteger& n);

    // Is n a power of 2?
    bool is_power_of_two(const ETInteger& n);

    // LCM landmark generator: yields lcm(1..k) for successive k
    // where the lcm actually changes (new prime or prime power enters)
    // Returns: sorted vector of (k, lcm_value) pairs — no overflow, ever
    std::vector<std::pair<int, ETInteger>> lcm_landmarks(int max_k);

    // ── Convenience overloads for int64_t callers ──────────────────────
    // Return ETInteger (unbounded) even from int64_t inputs
    ETInteger gcd(int64_t a, int64_t b);
    ETInteger lcm(int64_t a, int64_t b);
    std::vector<ETInteger> divisors(int64_t n);
    ETInteger totient(int64_t n);
    std::vector<std::pair<ETInteger, int>> factorize(int64_t n);
    bool is_power_of_two(int64_t n);

} // namespace intmath

// ============================================================================
// SHA-256 Hash — FIPS 180-4 Compliant
//
// Used for: value_hash, equation_hash, header integrity, exe tamper detection.
// Self-contained implementation — no external crypto dependency.
// ============================================================================

class SHA256 {
public:
    SHA256();

    // Feed data incrementally
    void update(const void* data, size_t len);
    void update(const std::string& s);
    void update(const std::vector<uint8_t>& v);

    // Finalize and return 32-byte digest
    std::array<uint8_t, 32> finalize();

    // Convenience: hash in one call
    static std::array<uint8_t, 32> hash(const void* data, size_t len);
    static std::array<uint8_t, 32> hash(const std::string& s);

    // Return hex string (64 chars)
    static std::string hash_hex(const void* data, size_t len);
    static std::string hash_hex(const std::string& s);

private:
    uint32_t state_[8];
    uint8_t  buffer_[64];
    size_t   buffer_len_;
    uint64_t total_len_;

    void process_block(const uint8_t block[64]);
};

// ============================================================================
// CRC-32 Checksum — Standard Polynomial 0xEDB88320
//
// Used for: per-page integrity in .akashic format, WAL entry verification.
// ============================================================================

class CRC32 {
public:
    CRC32();

    void update(const void* data, size_t len);
    void update(const std::string& s);

    [[nodiscard]] uint32_t finalize() const;

    // Convenience: compute in one call
    static uint32_t compute(const void* data, size_t len);
    static uint32_t compute(const std::string& s);

private:
    uint32_t crc_;
};

// ============================================================================
// ETConstants — All ET constants at 361-dps precision
//
// Computed once on initialization, cached for the lifetime of the process.
// Every constant is forward-derived from {P, D, T} primitives.
//
// Descriptor Gap Principle: if a constant is missing, add it here.
// Subsumption Law: this set must subsume every constant any module needs.
// ============================================================================

class ETConstants {
public:
    // Initialize all constants (called by et::initialize())
    static void initialize();

    // Whether constants have been initialized
    static bool is_initialized();

    // ── Mathematical Constants ─────────────────────────────────────────
    static const ETValue& pi();               // π = 3.14159265358979...
    static const ETValue& e();                // e = 2.71828182845904...
    static const ETValue& euler_gamma();      // γ = 0.57721566490153...
    static const ETValue& phi();              // φ = (1+√5)/2 = 1.61803398874989...
    static const ETValue& ln2();              // ln(2) = 0.69314718055994...
    static const ETValue& sqrt2();            // √2 = 1.41421356237309...
    static const ETValue& sqrt3();            // √3 = 1.73205080756887...
    static const ETValue& sqrt5();            // √5 = 2.23606797749978...

    // ── ET Structural Constants ────────────────────────────────────────
    // These are the D-content of the master equation P ∘ D ∘ T = E

    static const ETValue& K();               // Koide ratio = 2/3
    static const ETValue& V();               // Base variance = 1/12
    static const ETValue& N();               // Manifold symmetry = 12
    static const ETValue& S_val();           // State count = 4
    static const ETValue& sigma();           // Shimmer amplitude = √(1/12)
    static const ETValue& life_threshold();  // 13/12 — subliminal→detected boundary
    static const ETValue& k_em();            // Active EM channels = 8 (= N × κ where κ = K)
    static const ETValue& p_eff();           // Effective palindromic degree = 10/3

    // ── Fine Structure Constant Decomposition (§3.18.2) ────────────────
    // α⁻¹(ET) = A₀ + A₁ − A_cross − Σ_geometric
    // = 137 + √3/48 − √3/(93312π²) − 1/(216(18π−1))
    static const ETValue& alpha_inv();       // α⁻¹(ET) = 137.03599916744...
    static const ETValue& alpha_A0();        // 137 (base impedance)
    static const ETValue& alpha_A1();        // √3/48 (open shimmer)
    static const ETValue& alpha_Across();    // √3/(93312π²) (cross-term)
    static const ETValue& alpha_Sigma();     // 1/(216(18π−1)) (geometric sum)

    // ── Cascade Residuals (§3.18.3) ────────────────────────────────────
    static const ETValue& delta_r();         // |12·log₂(12) − 43|
    static const ETValue& delta_theta();     // |24π/ln(2) − 109|

    // ── Gaze Thresholds (§3.18.9) — Just-Intonation intervals ─────────
    static const ETValue& gaze_subliminal(); // 13/12 (augmented unison)
    static const ETValue& gaze_detected();   // 6/5 (quintic minor third) = Γ
    static const ETValue& gaze_locked();     // 3/2 (perfect fifth)
    static const ETValue& gaze_lock_con();   // 5/4 (major third, quintic comma carrier)

    // ── Zeta values ζ(2) through ζ(13) ────────────────────────────────
    static const ETValue& zeta(int s);       // s ∈ {2..13}, cached

    // ── Impedance coupling ξ(d) = 137/((d-1)²+S²) for d ∈ {1..12} ────
    static const ETValue& coupling_xi(int d); // d ∈ {1..12}, cached

    // ── Impedance A₀_magic(d) = (d-1)²+S² for d ∈ {1..12} ────────────
    static const ETValue& impedance(int d);  // d ∈ {1..12}, cached

private:
    static bool initialized_;

    // Storage for cached constants
    static std::unique_ptr<ETValue> pi_, e_, euler_gamma_, phi_, ln2_;
    static std::unique_ptr<ETValue> sqrt2_, sqrt3_, sqrt5_;
    static std::unique_ptr<ETValue> K_, V_, N_, S_val_, sigma_;
    static std::unique_ptr<ETValue> life_threshold_, k_em_, p_eff_;
    static std::unique_ptr<ETValue> alpha_inv_, alpha_A0_, alpha_A1_;
    static std::unique_ptr<ETValue> alpha_Across_, alpha_Sigma_;
    static std::unique_ptr<ETValue> delta_r_, delta_theta_;
    static std::unique_ptr<ETValue> gaze_subliminal_, gaze_detected_;
    static std::unique_ptr<ETValue> gaze_locked_, gaze_lock_con_;
    static std::array<std::unique_ptr<ETValue>, 14> zeta_cache_;    // indices 2..13
    static std::array<std::unique_ptr<ETValue>, 13> xi_cache_;      // indices 1..12
    static std::array<std::unique_ptr<ETValue>, 13> impedance_cache_; // indices 1..12

    // Internal computation helpers
    static void compute_mathematical_constants();
    static void compute_et_structural_constants();
    static void compute_fine_structure();
    static void compute_cascade_residuals();
    static void compute_gaze_thresholds();
    static void compute_zeta_values();
    static void compute_impedance_coupling();
};

// ============================================================================
// Forward type for Module 3 (Akashic Format) — memoization equation cache
// Maps canonical equation strings → memoized results at 361-dps
// Uses <unordered_map> include
// ============================================================================

using MemoCache = std::unordered_map<std::string, ETValue>;

// ============================================================================
// Global Initialization
//
// MUST be called before any ET computation. Sets MPFR default precision
// to 1200 bits (cents per octave), initializes FLINT, computes all ET constants.
//
// Idempotent: safe to call multiple times (only runs once).
// ============================================================================

void initialize();
bool is_initialized();

} // namespace et