// ============================================================================
// precision_stack.cpp — Module 1: Precision Stack (Level 0) — Implementation
//
// P ∘ D ∘ T = E
// Every computation at 1200-bit (361-dps). Zero IEEE 754. Zero ad hoc.
// 1200 bits = cents per octave — the lattice's own measurement scale.
// ============================================================================

#include "precision_stack.h"

#include <cstring>
#include <cmath>
#include <cassert>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <climits>
#include <numeric>
#include <mutex>

namespace et {

// ============================================================================
// Section 1: Global State and Initialization
// ============================================================================

static bool g_initialized = false;
static std::once_flag g_init_flag;

void initialize() {
    std::call_once(g_init_flag, []() {
        // Set MPFR default precision to 1200 bits (361 dps) — the ET hard cap
        // 1200 = cents per octave: binary precision matched to lattice scale
        mpfr_set_default_prec(ET_PRECISION_BITS);

        // Initialize FLINT (idempotent in FLINT 3.x)
        flint_set_num_threads(1); // single-threaded for FLINT; we manage our own threads

        // Compute all ET constants at 361 dps
        ETConstants::initialize();

        g_initialized = true;
    });
}

bool is_initialized() {
    return g_initialized;
}

// ============================================================================
// Section 2: ETError
// ============================================================================

ETError::ETError(Code c, std::string_view module, std::string_view msg)
    : std::runtime_error(std::string(msg))
    , code(c)
    , source_module(module)
    , detail(msg)
{}

ETError::ETError(Code c, std::string_view module, std::string_view msg, const std::string& extra)
    : std::runtime_error(std::string(msg) + ": " + extra)
    , code(c)
    , source_module(module)
    , detail(std::string(msg) + ": " + extra)
{}

// ============================================================================
// Section 2b: Portable int64 ↔ GMP helpers
//
// On Windows x64, long is 32-bit but int64_t is 64-bit. GMP's mpz_set_si
// takes long, which causes narrowing. These helpers handle all values.
// ============================================================================

namespace {

void mpz_set_int64(mpz_t z, int64_t v) {
    if (v >= LONG_MIN && v <= LONG_MAX) {
        mpz_set_si(z, static_cast<long>(v));
    } else if (v >= 0) {
        auto uv = static_cast<uint64_t>(v);
        mpz_import(z, 1, 1, sizeof(uint64_t), 0, 0, &uv);
    } else {
        // Handle negative values without UB on INT64_MIN
        auto uv = static_cast<uint64_t>(-(v + 1)) + 1;
        mpz_import(z, 1, 1, sizeof(uint64_t), 0, 0, &uv);
        mpz_neg(z, z);
    }
}

} // anonymous namespace

// ============================================================================
// Section 3: ETValue — Construction, Destruction, Assignment
// ============================================================================

void ETValue::init_precision() {
    mpfr_init2(val_, ET_PRECISION_BITS);
}

ETValue::ETValue() {
    init_precision();
    mpfr_set_zero(val_, 1); // +0
}

ETValue::ETValue(int64_t v) {
    init_precision();
    mpfr_set_sj(val_, v, MPFR_RNDN);
}

ETValue::ETValue(uint64_t v) {
    init_precision();
    mpfr_set_ui(val_, static_cast<unsigned long>(v), MPFR_RNDN);
}

ETValue::ETValue(const char* decimal_str) {
    init_precision();
    if (mpfr_set_str(val_, decimal_str, 10, MPFR_RNDN) != 0) {
        mpfr_clear(val_);
        throw ETError(ETError::Code::UNPARSEABLE_VALUE,
                      "ETValue::ETValue(const char*)",
                      "Failed to parse decimal string",
                      std::string(decimal_str));
    }
}

ETValue::ETValue(const std::string& decimal_str)
    : ETValue(decimal_str.c_str())
{}

ETValue::ETValue(const ETValue& other) {
    init_precision();
    mpfr_set(val_, other.val_, MPFR_RNDN);
}

ETValue::ETValue(ETValue&& other) noexcept {
    // Steal the MPFR internals
    val_[0] = other.val_[0];
    // Reinitialize other to a valid but empty state
    mpfr_init2(other.val_, ET_PRECISION_BITS);
    mpfr_set_zero(other.val_, 1);
}

ETValue::~ETValue() {
    mpfr_clear(val_);
}

ETValue& ETValue::operator=(const ETValue& other) {
    if (this != &other) {
        mpfr_set(val_, other.val_, MPFR_RNDN);
    }
    return *this;
}

ETValue& ETValue::operator=(ETValue&& other) noexcept {
    if (this != &other) {
        mpfr_clear(val_);
        val_[0] = other.val_[0];
        mpfr_init2(other.val_, ET_PRECISION_BITS);
        mpfr_set_zero(other.val_, 1);
    }
    return *this;
}

ETValue& ETValue::operator=(int64_t v) {
    mpfr_set_sj(val_, v, MPFR_RNDN);
    return *this;
}

// ── Static Factory Methods ─────────────────────────────────────────────

ETValue ETValue::from_rational(int64_t num, int64_t den) {
    if (den == 0) {
        throw ETError(ETError::Code::DIVISION_BY_ZERO,
                      "ETValue::from_rational",
                      "Denominator is zero");
    }
    ETValue result;
    // Use GMP rationals for exact intermediate computation
    mpq_t q;
    mpq_init(q);
    // Set numerator and denominator separately to avoid int64→long narrowing
    mpz_set_int64(mpq_numref(q), num);
    int64_t abs_den = (den < 0) ? -den : den;
    mpz_set_int64(mpq_denref(q), abs_den);
    if (den < 0) {
        mpq_neg(q, q);
    }
    mpq_canonicalize(q);
    mpfr_set_q(result.val_, q, MPFR_RNDN);
    mpq_clear(q);
    return result;
}

ETValue ETValue::from_mpfr_raw(const mpfr_t& src) {
    ETValue result;
    mpfr_set(result.val_, src, MPFR_RNDN);
    return result;
}

ETValue ETValue::from_integer(const ETInteger& n) {
    ETValue result;
    mpfr_set_z(result.val_, n.raw(), MPFR_RNDN);
    return result;
}

// ============================================================================
// Section 4: ETValue — Arithmetic Operators
// ============================================================================

ETValue ETValue::operator+(const ETValue& rhs) const {
    ETValue result;
    mpfr_add(result.val_, val_, rhs.val_, MPFR_RNDN);
    return result;
}

ETValue ETValue::operator-(const ETValue& rhs) const {
    ETValue result;
    mpfr_sub(result.val_, val_, rhs.val_, MPFR_RNDN);
    return result;
}

ETValue ETValue::operator*(const ETValue& rhs) const {
    ETValue result;
    mpfr_mul(result.val_, val_, rhs.val_, MPFR_RNDN);
    return result;
}

ETValue ETValue::operator/(const ETValue& rhs) const {
    if (mpfr_zero_p(rhs.val_)) {
        throw ETError(ETError::Code::DIVISION_BY_ZERO,
                      "ETValue::operator/",
                      "Division by zero — annihilation boundary approached");
    }
    ETValue result;
    mpfr_div(result.val_, val_, rhs.val_, MPFR_RNDN);
    return result;
}

ETValue ETValue::operator-() const {
    ETValue result;
    mpfr_neg(result.val_, val_, MPFR_RNDN);
    return result;
}

ETValue& ETValue::operator+=(const ETValue& rhs) {
    mpfr_add(val_, val_, rhs.val_, MPFR_RNDN);
    return *this;
}

ETValue& ETValue::operator-=(const ETValue& rhs) {
    mpfr_sub(val_, val_, rhs.val_, MPFR_RNDN);
    return *this;
}

ETValue& ETValue::operator*=(const ETValue& rhs) {
    mpfr_mul(val_, val_, rhs.val_, MPFR_RNDN);
    return *this;
}

ETValue& ETValue::operator/=(const ETValue& rhs) {
    if (mpfr_zero_p(rhs.val_)) {
        throw ETError(ETError::Code::DIVISION_BY_ZERO,
                      "ETValue::operator/=",
                      "Division by zero — annihilation boundary approached");
    }
    mpfr_div(val_, val_, rhs.val_, MPFR_RNDN);
    return *this;
}

// ============================================================================
// Section 5: ETValue — Comparison Operators
// ============================================================================

int ETValue::compare(const ETValue& rhs) const {
    return mpfr_cmp(val_, rhs.val_);
}

bool ETValue::operator==(const ETValue& rhs) const { return mpfr_equal_p(val_, rhs.val_) != 0; }
bool ETValue::operator!=(const ETValue& rhs) const { return mpfr_equal_p(val_, rhs.val_) == 0; }
bool ETValue::operator<(const ETValue& rhs)  const { return mpfr_less_p(val_, rhs.val_) != 0; }
bool ETValue::operator>(const ETValue& rhs)  const { return mpfr_greater_p(val_, rhs.val_) != 0; }
bool ETValue::operator<=(const ETValue& rhs) const { return mpfr_lessequal_p(val_, rhs.val_) != 0; }
bool ETValue::operator>=(const ETValue& rhs) const { return mpfr_greaterequal_p(val_, rhs.val_) != 0; }

// ============================================================================
// Section 6: ETValue — Conversion
// ============================================================================

std::string ETValue::to_string(int dps) const {
    if (mpfr_nan_p(val_)) return "NaN";
    if (mpfr_inf_p(val_)) return (mpfr_sgn(val_) < 0) ? "-Inf" : "Inf";
    if (mpfr_zero_p(val_)) return "0";

    // Use mpfr_sprintf for controlled decimal output
    // Format: %.<dps>Rf gives fixed-point with <dps> decimal places
    // But for very large/small numbers, we need scientific notation
    char* buf = nullptr;
    int len = mpfr_asprintf(&buf, "%.*Re", dps, val_);
    if (len < 0 || buf == nullptr) {
        throw ETError(ETError::Code::NAN_PRODUCED,
                      "ETValue::to_string",
                      "mpfr_asprintf failed");
    }
    std::string result(buf);
    mpfr_free_str(buf);
    return result;
}

std::string ETValue::to_canonical_string() const {
    if (mpfr_nan_p(val_)) return "NaN";
    if (mpfr_inf_p(val_)) return (mpfr_sgn(val_) < 0) ? "-Inf" : "+Inf";
    if (mpfr_zero_p(val_)) return "+0e0";

    // Get the mantissa digits and exponent for deterministic hashing
    mpfr_exp_t exp_val;
    // Request exactly ET_PRECISION_DPS+1 significant digits (sign + digits)
    char* digits = mpfr_get_str(nullptr, &exp_val, 10,
                                ET_PRECISION_DPS + 1, val_, MPFR_RNDN);
    if (digits == nullptr) {
        throw ETError(ETError::Code::NAN_PRODUCED,
                      "ETValue::to_canonical_string",
                      "mpfr_get_str failed");
    }

    // Build canonical form: [sign][digits]e[exponent]
    std::string result;
    const char* p = digits;
    if (*p == '-') {
        result += '-';
        p++;
    } else {
        result += '+';
    }
    result += p;
    result += 'e';
    result += std::to_string(exp_val);

    mpfr_free_str(digits);
    return result;
}

double ETValue::to_double() const {
    double d = mpfr_get_d(val_, MPFR_RNDN);
    // Validate: finite MPFR values must produce finite doubles.
    // mpfr_get_d silently returns ±Inf on range overflow — detect and report.
    // Uses std::isfinite from <cmath>.
    if (!mpfr_nan_p(val_) && !mpfr_inf_p(val_) && !mpfr_zero_p(val_)
        && !std::isfinite(d)) {
        throw ETError(ETError::Code::NUMERIC_OVERFLOW,
                      "ETValue::to_double",
                      "Value exceeds IEEE 754 double range",
                      to_string(20));
    }
    return d;
}

int64_t ETValue::to_int64() const {
    if (!mpfr_fits_intmax_p(val_, MPFR_RNDN)) {
        throw ETError(ETError::Code::NUMERIC_OVERFLOW,
                      "ETValue::to_int64",
                      "Value exceeds int64 range",
                      to_string(20));
    }
    return static_cast<int64_t>(mpfr_get_sj(val_, MPFR_RNDN));
}

// ============================================================================
// Section 7: ETValue — Serialization
//
// Binary blob format for .akashic storage:
//   Byte 0: flags
//     bits 7-4: version (0001 = v1)
//     bit  3:   sign (0=positive, 1=negative)
//     bits 2-0: special (000=normal, 001=zero, 010=+inf, 011=-inf, 100=nan)
//   For NORMAL values (special == 000):
//     Bytes 1-8:   exponent (int64_t, little-endian)
//     Bytes 9-158: mantissa (150 bytes = 1200 bits, big-endian, MSB-aligned)
//     Total: 159 bytes (= ET_BLOB_NORMAL_SIZE)
//   For SPECIAL values:
//     Total: 1 byte
// ============================================================================

std::vector<uint8_t> ETValue::serialize() const {
    uint8_t flags = ET_BLOB_VERSION; // version 1 in upper nibble

    if (mpfr_nan_p(val_)) {
        flags |= 0x04; // special = nan
        return { flags };
    }
    if (mpfr_inf_p(val_)) {
        flags |= (mpfr_sgn(val_) < 0) ? 0x0B : 0x02; // sign + special=inf
        return { flags };
    }
    if (mpfr_zero_p(val_)) {
        flags |= (mpfr_signbit(val_) ? 0x09 : 0x01); // sign + special=zero
        return { flags };
    }

    // Normal value
    if (mpfr_sgn(val_) < 0) {
        flags |= 0x08; // sign bit
    }

    std::vector<uint8_t> blob(ET_BLOB_NORMAL_SIZE);
    blob[0] = flags;

    // Export exponent (little-endian int64)
    mpfr_exp_t exp_val = mpfr_get_exp(val_);
    int64_t exp64 = static_cast<int64_t>(exp_val);
    for (int i = 0; i < 8; i++) {
        blob[1 + i] = static_cast<uint8_t>((exp64 >> (i * 8)) & 0xFF);
    }

    // Export mantissa via GMP
    // Get the significand as an mpz_t, then export to bytes
    mpz_t mantissa;
    mpz_init(mantissa);

    // mpfr stores the significand as an mpz; we extract it
    // The significand satisfies: val = sign * mantissa * 2^(exp - prec)
    mpfr_t abs_val;
    mpfr_init2(abs_val, ET_PRECISION_BITS);
    mpfr_abs(abs_val, val_, MPFR_RNDN);

    // Scale: mantissa_int = abs_val * 2^(prec - exp)
    mpfr_t scaled;
    mpfr_init2(scaled, ET_PRECISION_BITS);
    mpfr_mul_2si(scaled, abs_val, ET_PRECISION_BITS - exp_val, MPFR_RNDN);
    mpfr_get_z(mantissa, scaled, MPFR_RNDN);

    // Export mantissa to ET_MANTISSA_BYTES bytes big-endian
    size_t count = 0;
    uint8_t mantissa_bytes[ET_MANTISSA_BYTES + 16] = {}; // extra space for safety
    mpz_export(mantissa_bytes, &count, 1 /* big-endian */, 1, 1, 0, mantissa);

    // Copy into blob, right-aligned (MSB-first, padded with leading zeros)
    size_t offset = ET_MANTISSA_BYTES - std::min(count, ET_MANTISSA_BYTES);
    std::memset(&blob[9], 0, ET_MANTISSA_BYTES);
    std::memcpy(&blob[9 + offset], mantissa_bytes, std::min(count, ET_MANTISSA_BYTES));

    mpz_clear(mantissa);
    mpfr_clear(abs_val);
    mpfr_clear(scaled);

    return blob;
}

ETValue ETValue::deserialize(const uint8_t* data, size_t len) {
    if (len == 0) {
        throw ETError(ETError::Code::BLOB_CORRUPT,
                      "ETValue::deserialize", "Empty blob");
    }

    uint8_t flags = data[0];
    uint8_t version = (flags >> 4) & 0x0F;
    if (version != 1) {
        throw ETError(ETError::Code::BLOB_CORRUPT,
                      "ETValue::deserialize", "Unknown blob version",
                      std::to_string(version));
    }

    bool negative = (flags & 0x08) != 0;
    uint8_t special = flags & 0x07;

    ETValue result;

    if (special == 0x04) { // NaN
        mpfr_set_nan(result.val_);
        return result;
    }
    if (special == 0x02 || special == 0x03) { // +/-Inf
        mpfr_set_inf(result.val_, negative ? -1 : 1);
        return result;
    }
    if (special == 0x01) { // Zero
        mpfr_set_zero(result.val_, negative ? -1 : 1);
        return result;
    }

    // Normal value — need ET_BLOB_NORMAL_SIZE bytes
    if (len < ET_BLOB_NORMAL_SIZE) {
        throw ETError(ETError::Code::BLOB_CORRUPT,
                      "ETValue::deserialize", "Normal value blob too short",
                      std::to_string(len));
    }

    // Read exponent (little-endian int64)
    int64_t exp64 = 0;
    for (int i = 0; i < 8; i++) {
        exp64 |= static_cast<int64_t>(data[1 + i]) << (i * 8);
    }

    // Read mantissa (ET_MANTISSA_BYTES bytes big-endian)
    mpz_t mantissa;
    mpz_init(mantissa);
    mpz_import(mantissa, ET_MANTISSA_BYTES, 1 /* big-endian */, 1, 1, 0, &data[9]);

    // Reconstruct: val = sign * mantissa * 2^(exp - prec)
    mpfr_set_z(result.val_, mantissa, MPFR_RNDN);
    mpfr_div_2si(result.val_, result.val_,
                 ET_PRECISION_BITS - static_cast<mpfr_exp_t>(exp64), MPFR_RNDN);
    if (negative) {
        mpfr_neg(result.val_, result.val_, MPFR_RNDN);
    }

    mpz_clear(mantissa);
    return result;
}

ETValue ETValue::deserialize(const std::vector<uint8_t>& blob) {
    return deserialize(blob.data(), blob.size());
}

std::string ETValue::serialize_hex() const {
    auto blob = serialize();
    std::ostringstream oss;
    oss << std::hex << std::setfill('0');
    for (size_t i = 0; i < blob.size(); i++) {
        if (i > 0 && i % 16 == 0) oss << '\n';
        else if (i > 0) oss << ' ';
        oss << std::setw(2) << static_cast<int>(blob[i]);
    }
    return oss.str();
}

// ── Properties ─────────────────────────────────────────────────────────

bool ETValue::is_zero()     const { return mpfr_zero_p(val_) != 0; }
bool ETValue::is_positive() const { return mpfr_sgn(val_) > 0; }
bool ETValue::is_negative() const { return mpfr_sgn(val_) < 0; }
bool ETValue::is_nan()      const { return mpfr_nan_p(val_) != 0; }
bool ETValue::is_inf()      const { return mpfr_inf_p(val_) != 0; }

bool ETValue::is_integer() const {
    if (mpfr_nan_p(val_) || mpfr_inf_p(val_)) return false;
    return mpfr_integer_p(val_) != 0;
}

int ETValue::sign() const {
    return mpfr_sgn(val_);
}

// ============================================================================
// Section 8: Elementary Mathematical Functions (via MPFR)
// ============================================================================

namespace math {

#define ET_UNARY_MPFR(name, mpfr_func)                       \
    ETValue name(const ETValue& x) {                         \
        ETValue result;                                      \
        mpfr_func(result.raw(), x.raw(), MPFR_RNDN);        \
        if (mpfr_nan_p(result.raw())) {                      \
            throw ETError(ETError::Code::NAN_PRODUCED,       \
                          "math::" #name,                    \
                          "Function produced NaN",           \
                          x.to_string(30));                  \
        }                                                    \
        return result;                                       \
    }

ET_UNARY_MPFR(sqrt,  mpfr_sqrt)
ET_UNARY_MPFR(cbrt,  mpfr_cbrt)
ET_UNARY_MPFR(log,   mpfr_log)
ET_UNARY_MPFR(log2,  mpfr_log2)
ET_UNARY_MPFR(log10, mpfr_log10)
ET_UNARY_MPFR(exp,   mpfr_exp)
ET_UNARY_MPFR(exp2,  mpfr_exp2)
ET_UNARY_MPFR(exp10, mpfr_exp10)
ET_UNARY_MPFR(sin,   mpfr_sin)
ET_UNARY_MPFR(cos,   mpfr_cos)
ET_UNARY_MPFR(tan,   mpfr_tan)
ET_UNARY_MPFR(asin,  mpfr_asin)
ET_UNARY_MPFR(acos,  mpfr_acos)
ET_UNARY_MPFR(atan,  mpfr_atan)
ET_UNARY_MPFR(sinh,  mpfr_sinh)
ET_UNARY_MPFR(cosh,  mpfr_cosh)
ET_UNARY_MPFR(tanh,  mpfr_tanh)
ET_UNARY_MPFR(asinh, mpfr_asinh)
ET_UNARY_MPFR(acosh, mpfr_acosh)
ET_UNARY_MPFR(atanh, mpfr_atanh)

#undef ET_UNARY_MPFR

ETValue abs(const ETValue& x) {
    ETValue result;
    mpfr_abs(result.raw(), x.raw(), MPFR_RNDN);
    return result;
}

ETValue pow(const ETValue& base, const ETValue& exp) {
    ETValue result;
    mpfr_pow(result.raw(), base.raw(), exp.raw(), MPFR_RNDN);
    if (mpfr_nan_p(result.raw())) {
        throw ETError(ETError::Code::NAN_PRODUCED,
                      "math::pow",
                      "pow produced NaN",
                      base.to_string(30) + " ^ " + exp.to_string(30));
    }
    return result;
}

ETValue pow(const ETValue& base, int64_t exp) {
    ETValue result;
    if (exp >= LONG_MIN && exp <= LONG_MAX) {
        mpfr_pow_si(result.raw(), base.raw(), static_cast<long>(exp), MPFR_RNDN);
    } else {
        // Exponent exceeds long range — convert to ETValue and use mpfr_pow
        ETValue exp_val(exp);
        mpfr_pow(result.raw(), base.raw(), exp_val.raw(), MPFR_RNDN);
    }
    return result;
}

ETValue atan2(const ETValue& y, const ETValue& x) {
    ETValue result;
    mpfr_atan2(result.raw(), y.raw(), x.raw(), MPFR_RNDN);
    return result;
}

ETValue floor(const ETValue& x) {
    ETValue result;
    mpfr_floor(result.raw(), x.raw());
    return result;
}

ETValue ceil(const ETValue& x) {
    ETValue result;
    mpfr_ceil(result.raw(), x.raw());
    return result;
}

ETValue round(const ETValue& x) {
    ETValue result;
    mpfr_round(result.raw(), x.raw());
    return result;
}

ETValue trunc(const ETValue& x) {
    ETValue result;
    mpfr_trunc(result.raw(), x.raw());
    return result;
}

ETValue frac(const ETValue& x) {
    ETValue result;
    mpfr_frac(result.raw(), x.raw(), MPFR_RNDN);
    return result;
}

// EML Sheffer operator: eml(x, y) = exp(x) - ln(y)
// The minimal continuous-D generator (L₃ backbone)
ETValue eml(const ETValue& x, const ETValue& y) {
    ETValue exp_x = exp(x);
    ETValue ln_y = log(y);
    return exp_x - ln_y;
}

} // namespace math

// ============================================================================
// Section 9: Special Functions (via FLINT/Arb)
//
// Workflow for each function:
//   1. Convert ETValue (mpfr_t) to arb_t via arf_t intermediate
//   2. Call the Arb function at 1200-bit precision
//   3. Extract the midpoint back to mpfr_t
//
// RAII wrappers ensure no resource leaks.
// ============================================================================

namespace {

// RAII wrapper for arb_t
struct ArbGuard {
    arb_t val;
    ArbGuard() { arb_init(val); }
    ~ArbGuard() { arb_clear(val); }
    ArbGuard(const ArbGuard&) = delete;
    ArbGuard& operator=(const ArbGuard&) = delete;
};

// RAII wrapper for acb_t
struct AcbGuard {
    acb_t val;
    AcbGuard() { acb_init(val); }
    ~AcbGuard() { acb_clear(val); }
    AcbGuard(const AcbGuard&) = delete;
    AcbGuard& operator=(const AcbGuard&) = delete;
};

// RAII wrapper for arf_t
struct ArfGuard {
    arf_t val;
    ArfGuard() { arf_init(val); }
    ~ArfGuard() { arf_clear(val); }
    ArfGuard(const ArfGuard&) = delete;
    ArfGuard& operator=(const ArfGuard&) = delete;
};

// Convert ETValue → arb_t
void etvalue_to_arb(arb_t out, const ETValue& v) {
    ArfGuard mid;
    arf_set_mpfr(mid.val, v.raw());
    arb_set_arf(out, mid.val);
}

// Convert arb_t midpoint → ETValue
ETValue arb_to_etvalue(const arb_t in) {
    ETValue result;
    arf_get_mpfr(result.raw(), arb_midref(in), MPFR_RNDN);
    return result;
}

} // anonymous namespace

namespace special {

ETValue zeta(const ETValue& s) {
    // Check for pole at s=1
    ETValue one(int64_t(1));
    if (s == one) {
        throw ETError(ETError::Code::POLE_DETECTED,
                      "special::zeta",
                      "Riemann zeta has a pole at s=1");
    }

    ArbGuard arb_s, arb_result;
    etvalue_to_arb(arb_s.val, s);
    arb_zeta(arb_result.val, arb_s.val, ET_PRECISION_BITS);
    return arb_to_etvalue(arb_result.val);
}

ETValue gamma(const ETValue& x) {
    // Check for poles at non-positive integers
    if (x.is_integer() && !x.is_positive() && !x.is_zero()) {
        throw ETError(ETError::Code::POLE_DETECTED,
                      "special::gamma",
                      "Gamma function has a pole at non-positive integer",
                      x.to_string(30));
    }
    // Also pole at x=0
    if (x.is_zero()) {
        throw ETError(ETError::Code::POLE_DETECTED,
                      "special::gamma",
                      "Gamma function has a pole at x=0");
    }

    ArbGuard arb_x, arb_result;
    etvalue_to_arb(arb_x.val, x);
    arb_gamma(arb_result.val, arb_x.val, ET_PRECISION_BITS);
    return arb_to_etvalue(arb_result.val);
}

ETValue lgamma(const ETValue& x) {
    ArbGuard arb_x, arb_result;
    etvalue_to_arb(arb_x.val, x);
    arb_lgamma(arb_result.val, arb_x.val, ET_PRECISION_BITS);
    return arb_to_etvalue(arb_result.val);
}

ETValue digamma(const ETValue& x) {
    ArbGuard arb_x, arb_result;
    etvalue_to_arb(arb_x.val, x);
    arb_digamma(arb_result.val, arb_x.val, ET_PRECISION_BITS);
    return arb_to_etvalue(arb_result.val);
}

ETValue beta(const ETValue& a, const ETValue& b) {
    // B(a,b) = Γ(a)Γ(b)/Γ(a+b)
    ETValue ga = gamma(a);
    ETValue gb = gamma(b);
    ETValue gab = gamma(a + b);
    return (ga * gb) / gab;
}

ETValue polylog(const ETValue& s, const ETValue& z) {
    AcbGuard acb_s, acb_z, acb_result;

    // Set real parts from ETValues, imaginary parts to zero
    ArbGuard arb_s, arb_z;
    etvalue_to_arb(arb_s.val, s);
    etvalue_to_arb(arb_z.val, z);

    acb_set_arb(acb_s.val, arb_s.val);
    acb_set_arb(acb_z.val, arb_z.val);

    acb_polylog(acb_result.val, acb_s.val, acb_z.val, ET_PRECISION_BITS);

    // Extract real part (polylog of real args should be real for |z|<1)
    ETValue result;
    arf_get_mpfr(result.raw(), arb_midref(acb_realref(acb_result.val)), MPFR_RNDN);
    return result;
}

ETValue erf(const ETValue& x) {
    // MPFR has erf built-in
    ETValue result;
    mpfr_erf(result.raw(), x.raw(), MPFR_RNDN);
    return result;
}

ETValue erfc(const ETValue& x) {
    ETValue result;
    mpfr_erfc(result.raw(), x.raw(), MPFR_RNDN);
    return result;
}

ETValue bernoulli(uint64_t n) {
    // Odd Bernoulli numbers (except B_1) are zero
    if (n > 1 && n % 2 != 0) {
        return ETValue(int64_t(0));
    }

    ArbGuard arb_result;
    arb_bernoulli_ui(arb_result.val, static_cast<ulong>(n), ET_PRECISION_BITS);
    return arb_to_etvalue(arb_result.val);
}

ETValue euler_gamma() {
    ETValue result;
    mpfr_const_euler(result.raw(), MPFR_RNDN);
    return result;
}

} // namespace special

// ============================================================================
// Section 10: ETInteger — Arbitrary-Precision Integer Type
// ============================================================================

ETInteger::ETInteger()                          { mpz_init(val_); }
ETInteger::ETInteger(int64_t v)                 { mpz_init(val_); mpz_set_int64(val_, v); }
ETInteger::ETInteger(const char* decimal_str)   { mpz_init(val_); if (mpz_set_str(val_, decimal_str, 10) != 0) { mpz_clear(val_); throw ETError(ETError::Code::UNPARSEABLE_VALUE, "ETInteger", "Bad integer string", std::string(decimal_str)); } }
ETInteger::ETInteger(const std::string& decimal_str) : ETInteger(decimal_str.c_str()) {}
ETInteger::ETInteger(const ETInteger& other)    { mpz_init_set(val_, other.val_); }
ETInteger::ETInteger(ETInteger&& other) noexcept { val_[0] = other.val_[0]; mpz_init(other.val_); }
ETInteger::~ETInteger()                         { mpz_clear(val_); }

ETInteger& ETInteger::operator=(const ETInteger& other) { if (this != &other) mpz_set(val_, other.val_); return *this; }
ETInteger& ETInteger::operator=(ETInteger&& other) noexcept { if (this != &other) { mpz_clear(val_); val_[0] = other.val_[0]; mpz_init(other.val_); } return *this; }
ETInteger& ETInteger::operator=(int64_t v) { mpz_set_int64(val_, v); return *this; }

ETInteger ETInteger::operator+(const ETInteger& rhs) const { ETInteger res; mpz_add(res.val_, val_, rhs.val_); return res; }
ETInteger ETInteger::operator-(const ETInteger& rhs) const { ETInteger res; mpz_sub(res.val_, val_, rhs.val_); return res; }
ETInteger ETInteger::operator*(const ETInteger& rhs) const { ETInteger res; mpz_mul(res.val_, val_, rhs.val_); return res; }
ETInteger ETInteger::operator/(const ETInteger& rhs) const { if (mpz_sgn(rhs.val_) == 0) throw ETError(ETError::Code::DIVISION_BY_ZERO, "ETInteger::operator/", "Division by zero"); ETInteger res; mpz_fdiv_q(res.val_, val_, rhs.val_); return res; }
ETInteger ETInteger::operator%(const ETInteger& rhs) const { if (mpz_sgn(rhs.val_) == 0) throw ETError(ETError::Code::DIVISION_BY_ZERO, "ETInteger::operator%", "Modulo by zero"); ETInteger res; mpz_fdiv_r(res.val_, val_, rhs.val_); return res; }
ETInteger ETInteger::operator-() const { ETInteger res; mpz_neg(res.val_, val_); return res; }

ETInteger& ETInteger::operator+=(const ETInteger& rhs) { mpz_add(val_, val_, rhs.val_); return *this; }
ETInteger& ETInteger::operator-=(const ETInteger& rhs) { mpz_sub(val_, val_, rhs.val_); return *this; }
ETInteger& ETInteger::operator*=(const ETInteger& rhs) { mpz_mul(val_, val_, rhs.val_); return *this; }
ETInteger& ETInteger::operator/=(const ETInteger& rhs) { if (mpz_sgn(rhs.val_) == 0) throw ETError(ETError::Code::DIVISION_BY_ZERO, "ETInteger::operator/=", "Division by zero"); mpz_fdiv_q(val_, val_, rhs.val_); return *this; }
ETInteger& ETInteger::operator%=(const ETInteger& rhs) { if (mpz_sgn(rhs.val_) == 0) throw ETError(ETError::Code::DIVISION_BY_ZERO, "ETInteger::operator%=", "Modulo by zero"); mpz_fdiv_r(val_, val_, rhs.val_); return *this; }

bool ETInteger::operator==(const ETInteger& rhs) const { return mpz_cmp(val_, rhs.val_) == 0; }
bool ETInteger::operator!=(const ETInteger& rhs) const { return mpz_cmp(val_, rhs.val_) != 0; }
bool ETInteger::operator<(const ETInteger& rhs)  const { return mpz_cmp(val_, rhs.val_) < 0; }
bool ETInteger::operator>(const ETInteger& rhs)  const { return mpz_cmp(val_, rhs.val_) > 0; }
bool ETInteger::operator<=(const ETInteger& rhs) const { return mpz_cmp(val_, rhs.val_) <= 0; }
bool ETInteger::operator>=(const ETInteger& rhs) const { return mpz_cmp(val_, rhs.val_) >= 0; }
int  ETInteger::compare(const ETInteger& rhs)  const { return mpz_cmp(val_, rhs.val_); }

bool ETInteger::operator==(int64_t rhs) const { ETInteger tmp(rhs); return *this == tmp; }
bool ETInteger::operator!=(int64_t rhs) const { ETInteger tmp(rhs); return *this != tmp; }

ETInteger ETInteger::operator&(const ETInteger& rhs) const { ETInteger res; mpz_and(res.val_, val_, rhs.val_); return res; }

std::string ETInteger::to_string() const {
    char* s = mpz_get_str(nullptr, 10, val_);
    std::string result(s);
    free(s);
    return result;
}

int64_t ETInteger::to_int64() const {
    if (!fits_int64()) {
        throw ETError(ETError::Code::NUMERIC_OVERFLOW,
                      "ETInteger::to_int64",
                      "Value exceeds int64 range",
                      to_string());
    }
    // mpz_get_si returns long (32-bit on Win64) — use string for full int64 range
    if (mpz_fits_slong_p(val_)) {
        return static_cast<int64_t>(mpz_get_si(val_));
    }
    return std::strtoll(to_string().c_str(), nullptr, 10);
}

bool ETInteger::fits_int64() const {
    static ETInteger min_val(INT64_MIN);
    static ETInteger max_val(INT64_MAX);
    return mpz_cmp(val_, min_val.val_) >= 0 && mpz_cmp(val_, max_val.val_) <= 0;
}

bool ETInteger::is_zero()     const { return mpz_sgn(val_) == 0; }
bool ETInteger::is_positive() const { return mpz_sgn(val_) > 0; }
bool ETInteger::is_negative() const { return mpz_sgn(val_) < 0; }
int  ETInteger::sign()        const { return mpz_sgn(val_); }

ETInteger ETInteger::from_mpz(const mpz_t& src) {
    ETInteger result;
    mpz_set(result.val_, src);
    return result;
}

ETInteger ETInteger::from_etvalue(const ETValue& v) {
    // v must be an integer-valued ETValue. Uses mpfr_get_z for direct conversion.
    // No int64 intermediate. No string intermediate. Pure GMP.
    ETInteger result;
    mpfr_get_z(result.val_, v.raw(), MPFR_RNDN);
    return result;
}

// ============================================================================
// Section 10b: Integer Number Theory — All Arbitrary Precision
// ============================================================================

namespace intmath {

ETInteger gcd(const ETInteger& a, const ETInteger& b) {
    ETInteger result;
    mpz_gcd(result.raw(), a.raw(), b.raw());
    return result;
}

ETInteger lcm(const ETInteger& a, const ETInteger& b) {
    if (a.is_zero() || b.is_zero()) return ETInteger(int64_t(0));
    ETInteger result;
    mpz_lcm(result.raw(), a.raw(), b.raw());
    return result;
}

std::vector<ETInteger> divisors(const ETInteger& n) {
    if (!n.is_positive()) return {};
    std::vector<ETInteger> result;
    ETInteger i(int64_t(1));
    while (true) {
        ETInteger sq = i * i;
        if (sq > n) break;
        ETInteger rem = n % i;
        if (rem.is_zero()) {
            result.push_back(i);
            ETInteger quot = n / i;
            if (quot != i) {
                result.push_back(quot);
            }
        }
        i += ETInteger(int64_t(1));
    }
    // Sort using GMP comparison
    std::sort(result.begin(), result.end(),
              [](const ETInteger& a, const ETInteger& b) { return a < b; });
    return result;
}

ETInteger totient(const ETInteger& n) {
    if (!n.is_positive()) return ETInteger(int64_t(0));
    ETInteger result = n;
    ETInteger temp = n;
    ETInteger one(int64_t(1));
    ETInteger p(int64_t(2));

    while (true) {
        ETInteger pp = p * p;
        if (pp > temp) break;
        ETInteger rem = temp % p;
        if (rem.is_zero()) {
            while (true) {
                rem = temp % p;
                if (!rem.is_zero()) break;
                temp /= p;
            }
            result -= result / p;
        }
        p += one;
    }
    if (temp > one) {
        result -= result / temp;
    }
    return result;
}

std::vector<std::pair<ETInteger, int>> factorize(const ETInteger& n) {
    if (n <= ETInteger(int64_t(1))) return {};
    ETInteger remaining = n;
    ETInteger one(int64_t(1));
    std::vector<std::pair<ETInteger, int>> factors;

    ETInteger p(int64_t(2));
    while (true) {
        ETInteger pp = p * p;
        if (pp > remaining) break;
        ETInteger rem = remaining % p;
        if (rem.is_zero()) {
            int exp = 0;
            while (true) {
                rem = remaining % p;
                if (!rem.is_zero()) break;
                remaining /= p;
                exp++;
            }
            factors.emplace_back(p, exp);
        }
        p += one;
    }
    if (remaining > one) {
        factors.emplace_back(remaining, 1);
    }
    return factors;
}

bool is_power_of_two(const ETInteger& n) {
    if (!n.is_positive()) return false;
    ETInteger n_minus_1 = n - ETInteger(int64_t(1));
    ETInteger bitand_result = n & n_minus_1;
    return bitand_result.is_zero();
}

std::vector<std::pair<int, ETInteger>> lcm_landmarks(int max_k) {
    std::vector<std::pair<int, ETInteger>> result;
    ETInteger current_lcm(int64_t(1));

    for (int k = 1; k <= max_k; k++) {
        ETInteger gk{static_cast<int64_t>(k)};
        ETInteger new_lcm = lcm(current_lcm, gk);
        if (new_lcm != current_lcm) {
            current_lcm = new_lcm;
            result.emplace_back(k, current_lcm);
        }
    }
    return result;
}

// ── Convenience overloads ──────────────────────────────────────────────

ETInteger gcd(int64_t a, int64_t b)      { return gcd(ETInteger(a), ETInteger(b)); }
ETInteger lcm(int64_t a, int64_t b)      { return lcm(ETInteger(a), ETInteger(b)); }
std::vector<ETInteger> divisors(int64_t n) { return divisors(ETInteger(n)); }
ETInteger totient(int64_t n)             { return totient(ETInteger(n)); }
std::vector<std::pair<ETInteger, int>> factorize(int64_t n) { return factorize(ETInteger(n)); }
bool is_power_of_two(int64_t n)          { return is_power_of_two(ETInteger(n)); }

} // namespace intmath

// ============================================================================
// Section 11: SHA-256 — FIPS 180-4 Compliant Implementation
//
// Self-contained. No external crypto dependency.
// Used for: value_hash, equation_hash, header_checksum, exe tamper detection.
// ============================================================================

namespace {

// SHA-256 round constants: first 32 bits of fractional parts of cube roots of first 64 primes
constexpr uint32_t SHA256_K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

inline uint32_t sha_rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
inline uint32_t sha_ch(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (~x & z); }
inline uint32_t sha_maj(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (x & z) ^ (y & z); }
inline uint32_t sha_sigma0(uint32_t x) { return sha_rotr(x, 2) ^ sha_rotr(x, 13) ^ sha_rotr(x, 22); }
inline uint32_t sha_sigma1(uint32_t x) { return sha_rotr(x, 6) ^ sha_rotr(x, 11) ^ sha_rotr(x, 25); }
inline uint32_t sha_gamma0(uint32_t x) { return sha_rotr(x, 7) ^ sha_rotr(x, 18) ^ (x >> 3); }
inline uint32_t sha_gamma1(uint32_t x) { return sha_rotr(x, 17) ^ sha_rotr(x, 19) ^ (x >> 10); }

} // anonymous namespace

SHA256::SHA256()
    : state_{}
    , buffer_{}
    , buffer_len_(0)
    , total_len_(0) {
    // Initial hash values: first 32 bits of fractional parts of square roots of first 8 primes
    state_[0] = 0x6a09e667;
    state_[1] = 0xbb67ae85;
    state_[2] = 0x3c6ef372;
    state_[3] = 0xa54ff53a;
    state_[4] = 0x510e527f;
    state_[5] = 0x9b05688c;
    state_[6] = 0x1f83d9ab;
    state_[7] = 0x5be0cd19;
}

void SHA256::process_block(const uint8_t block[64]) {
    uint32_t W[64];

    // Message schedule
    for (int i = 0; i < 16; i++) {
        W[i] = (static_cast<uint32_t>(block[i * 4]) << 24)
             | (static_cast<uint32_t>(block[i * 4 + 1]) << 16)
             | (static_cast<uint32_t>(block[i * 4 + 2]) << 8)
             | (static_cast<uint32_t>(block[i * 4 + 3]));
    }
    for (int i = 16; i < 64; i++) {
        W[i] = sha_gamma1(W[i - 2]) + W[i - 7] + sha_gamma0(W[i - 15]) + W[i - 16];
    }

    // Working variables
    uint32_t a = state_[0], b = state_[1], c = state_[2], d = state_[3];
    uint32_t e = state_[4], f = state_[5], g = state_[6], h = state_[7];

    // Compression
    for (int i = 0; i < 64; i++) {
        uint32_t T1 = h + sha_sigma1(e) + sha_ch(e, f, g) + SHA256_K[i] + W[i];
        uint32_t T2 = sha_sigma0(a) + sha_maj(a, b, c);
        h = g; g = f; f = e; e = d + T1;
        d = c; c = b; b = a; a = T1 + T2;
    }

    state_[0] += a; state_[1] += b; state_[2] += c; state_[3] += d;
    state_[4] += e; state_[5] += f; state_[6] += g; state_[7] += h;
}

void SHA256::update(const void* data, size_t len) {
    const auto* bytes = static_cast<const uint8_t*>(data);
    total_len_ += len;

    // Fill buffer first
    while (len > 0) {
        size_t space = 64 - buffer_len_;
        size_t to_copy = std::min(len, space);
        std::memcpy(buffer_ + buffer_len_, bytes, to_copy);
        buffer_len_ += to_copy;
        bytes += to_copy;
        len -= to_copy;

        if (buffer_len_ == 64) {
            process_block(buffer_);
            buffer_len_ = 0;
        }
    }
}

void SHA256::update(const std::string& s) {
    update(s.data(), s.size());
}

void SHA256::update(const std::vector<uint8_t>& v) {
    update(v.data(), v.size());
}

std::array<uint8_t, 32> SHA256::finalize() {
    // Pad: append 1 bit, then zeros, then 64-bit big-endian length
    uint64_t bit_len = total_len_ * 8;

    uint8_t pad = 0x80;
    update(&pad, 1);

    // Pad with zeros until buffer_len_ ≡ 56 (mod 64)
    uint8_t zero = 0x00;
    while (buffer_len_ != 56) {
        update(&zero, 1);
    }

    // Append length as 64-bit big-endian
    uint8_t len_bytes[8];
    for (int i = 7; i >= 0; i--) {
        len_bytes[7 - i] = static_cast<uint8_t>((bit_len >> (i * 8)) & 0xFF);
    }
    update(len_bytes, 8);

    // Extract digest (big-endian)
    std::array<uint8_t, 32> digest{};
    for (int i = 0; i < 8; i++) {
        digest[i * 4]     = static_cast<uint8_t>((state_[i] >> 24) & 0xFF);
        digest[i * 4 + 1] = static_cast<uint8_t>((state_[i] >> 16) & 0xFF);
        digest[i * 4 + 2] = static_cast<uint8_t>((state_[i] >> 8) & 0xFF);
        digest[i * 4 + 3] = static_cast<uint8_t>(state_[i] & 0xFF);
    }
    return digest;
}

std::array<uint8_t, 32> SHA256::hash(const void* data, size_t len) {
    SHA256 ctx;
    ctx.update(data, len);
    return ctx.finalize();
}

std::array<uint8_t, 32> SHA256::hash(const std::string& s) {
    return hash(s.data(), s.size());
}

std::string SHA256::hash_hex(const void* data, size_t len) {
    auto digest = hash(data, len);
    std::string hex;
    hex.reserve(64);
    static const char hextab[] = "0123456789abcdef";
    for (uint8_t b : digest) {
        hex += hextab[b >> 4];
        hex += hextab[b & 0x0F];
    }
    return hex;
}

std::string SHA256::hash_hex(const std::string& s) {
    return hash_hex(s.data(), s.size());
}

// ============================================================================
// Section 12: CRC-32 — Standard Polynomial 0xEDB88320
// ============================================================================

// CRC-32 lookup table (generated once, read-only thereafter)
static uint32_t g_crc32_table[256];
static bool g_crc32_table_ready = false;

static void ensure_crc32_table() {
    if (g_crc32_table_ready) return;
    for (uint32_t i = 0; i < 256; i++) {
        uint32_t crc = i;
        for (int j = 0; j < 8; j++) {
            if (crc & 1)
                crc = (crc >> 1) ^ 0xEDB88320u;
            else
                crc >>= 1;
        }
        g_crc32_table[i] = crc;
    }
    g_crc32_table_ready = true;
}

CRC32::CRC32() : crc_(0xFFFFFFFF) {
    ensure_crc32_table();
}

void CRC32::update(const void* data, size_t len) {
    const auto* bytes = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < len; i++) {
        crc_ = g_crc32_table[(crc_ ^ bytes[i]) & 0xFF] ^ (crc_ >> 8);
    }
}

void CRC32::update(const std::string& s) {
    update(s.data(), s.size());
}

uint32_t CRC32::finalize() const {
    return crc_ ^ 0xFFFFFFFF;
}

uint32_t CRC32::compute(const void* data, size_t len) {
    CRC32 ctx;
    ctx.update(data, len);
    return ctx.finalize();
}

uint32_t CRC32::compute(const std::string& s) {
    return compute(s.data(), s.size());
}

// ============================================================================
// Section 13: ETConstants — All ET constants at 361-dps precision
//
// Every constant is forward-derived from {P, D, T} primitives.
// Computed once, cached for process lifetime.
//
// Identification Principle applied to the constants themselves:
//   P = the number line (the substrate these values live on)
//   D = the ET axioms that determine each value's magnitude
//   T = the computation that substantiates each value at 361 dps
// ============================================================================

bool ETConstants::initialized_ = false;

// Static storage — all initialized to nullptr
std::unique_ptr<ETValue> ETConstants::pi_;
std::unique_ptr<ETValue> ETConstants::e_;
std::unique_ptr<ETValue> ETConstants::euler_gamma_;
std::unique_ptr<ETValue> ETConstants::phi_;
std::unique_ptr<ETValue> ETConstants::ln2_;
std::unique_ptr<ETValue> ETConstants::sqrt2_;
std::unique_ptr<ETValue> ETConstants::sqrt3_;
std::unique_ptr<ETValue> ETConstants::sqrt5_;
std::unique_ptr<ETValue> ETConstants::K_;
std::unique_ptr<ETValue> ETConstants::V_;
std::unique_ptr<ETValue> ETConstants::N_;
std::unique_ptr<ETValue> ETConstants::S_val_;
std::unique_ptr<ETValue> ETConstants::sigma_;
std::unique_ptr<ETValue> ETConstants::life_threshold_;
std::unique_ptr<ETValue> ETConstants::k_em_;
std::unique_ptr<ETValue> ETConstants::p_eff_;
std::unique_ptr<ETValue> ETConstants::alpha_inv_;
std::unique_ptr<ETValue> ETConstants::alpha_A0_;
std::unique_ptr<ETValue> ETConstants::alpha_A1_;
std::unique_ptr<ETValue> ETConstants::alpha_Across_;
std::unique_ptr<ETValue> ETConstants::alpha_Sigma_;
std::unique_ptr<ETValue> ETConstants::delta_r_;
std::unique_ptr<ETValue> ETConstants::delta_theta_;
std::unique_ptr<ETValue> ETConstants::gaze_subliminal_;
std::unique_ptr<ETValue> ETConstants::gaze_detected_;
std::unique_ptr<ETValue> ETConstants::gaze_locked_;
std::unique_ptr<ETValue> ETConstants::gaze_lock_con_;
std::array<std::unique_ptr<ETValue>, 14> ETConstants::zeta_cache_;
std::array<std::unique_ptr<ETValue>, 13> ETConstants::xi_cache_;
std::array<std::unique_ptr<ETValue>, 13> ETConstants::impedance_cache_;

void ETConstants::initialize() {
    if (initialized_) return;

    compute_mathematical_constants();
    compute_et_structural_constants();
    compute_fine_structure();
    compute_cascade_residuals();
    compute_gaze_thresholds();
    compute_zeta_values();
    compute_impedance_coupling();

    initialized_ = true;
}

bool ETConstants::is_initialized() { return initialized_; }

void ETConstants::compute_mathematical_constants() {
    // π — via MPFR constant
    pi_ = std::make_unique<ETValue>();
    mpfr_const_pi(pi_->raw(), MPFR_RNDN);

    // e — exp(1)
    e_ = std::make_unique<ETValue>(math::exp(ETValue(int64_t(1))));

    // γ (Euler-Mascheroni) — via MPFR constant
    euler_gamma_ = std::make_unique<ETValue>();
    mpfr_const_euler(euler_gamma_->raw(), MPFR_RNDN);

    // ln(2) — via MPFR constant
    ln2_ = std::make_unique<ETValue>();
    mpfr_const_log2(ln2_->raw(), MPFR_RNDN);

    // √2, √3, √5
    sqrt2_ = std::make_unique<ETValue>(math::sqrt(ETValue(int64_t(2))));
    sqrt3_ = std::make_unique<ETValue>(math::sqrt(ETValue(int64_t(3))));
    sqrt5_ = std::make_unique<ETValue>(math::sqrt(ETValue(int64_t(5))));

    // φ = (1 + √5) / 2 — the golden ratio
    phi_ = std::make_unique<ETValue>((ETValue(int64_t(1)) + *sqrt5_) / ETValue(int64_t(2)));
}

void ETConstants::compute_et_structural_constants() {
    // K = 2/3 — Koide ratio (exact rational)
    K_ = std::make_unique<ETValue>(ETValue::from_rational(2, 3));

    // V = 1/12 — Base variance (exact rational)
    V_ = std::make_unique<ETValue>(ETValue::from_rational(1, 12));

    // N = 12 — Manifold symmetry (exact integer)
    N_ = std::make_unique<ETValue>(int64_t(ET_N));

    // S = 4 — State count (exact integer)
    S_val_ = std::make_unique<ETValue>(int64_t(ET_S));

    // σ = √(1/12) = √(V) — Shimmer amplitude
    sigma_ = std::make_unique<ETValue>(math::sqrt(*V_));

    // LIFE_THRESHOLD = 13/12 — subliminal boundary (exact rational)
    life_threshold_ = std::make_unique<ETValue>(ETValue::from_rational(13, 12));

    // K_EM = 8 = N × κ where κ = K = 2/3; K_EM = 12 × 2/3 = 8
    k_em_ = std::make_unique<ETValue>(int64_t(8));

    // p_eff = 10/3 — effective palindromic degree
    // (1/12) × Σ_{n=0..11} (12/PALINDROME[n])
    // PALINDROME = [12,6,4,3,12,2,12,3,4,6,12,1]
    // Sum = 12/12 + 12/6 + 12/4 + 12/3 + 12/12 + 12/2 + 12/12 + 12/3 + 12/4 + 12/6 + 12/12 + 12/1
    //     = 1 + 2 + 3 + 4 + 1 + 6 + 1 + 4 + 3 + 2 + 1 + 12 = 40
    // p_eff = 40/12 = 10/3
    p_eff_ = std::make_unique<ETValue>(ETValue::from_rational(10, 3));
}

void ETConstants::compute_fine_structure() {
    // α⁻¹(ET) = 137 + √3/48 − √3/(93312π²) − 1/(216(18π−1))
    //
    // Structural decomposition (§3.18.2):
    //   A₀ = (N-1)² + S² = 11² + 4² = 121 + 16 = 137  (base impedance at d=12)
    //   A₁ = √3/48 = σ/K_EM                              (open shimmer)
    //   A_cross = √3/(93312π²) = (2/π)·A₁·A₂             (cross-term)
    //   Σ_geometric = 1/(216(18π−1)) = κ²/[N²(Nπ−κ)]     (closed Mediation loops)

    // A₀ = 137 exactly
    alpha_A0_ = std::make_unique<ETValue>(int64_t(137));

    // A₁ = √3 / 48
    ETValue forty_eight(int64_t(48));
    alpha_A1_ = std::make_unique<ETValue>(*sqrt3_ / forty_eight);

    // A_cross = √3 / (93312 π²)
    // 93312 = 48 × 1944 = 48 × K_EM × N² / (2/π) ... structural derivation
    ETValue pi_sq = *pi_ * *pi_;
    ETValue denom_cross = ETValue(int64_t(93312)) * pi_sq;
    alpha_Across_ = std::make_unique<ETValue>(*sqrt3_ / denom_cross);

    // Σ_geometric = 1 / (216 × (18π − 1))
    // 216 = 6³ = (N/2)³ — the hexadic cube
    ETValue eighteen_pi = ETValue(int64_t(18)) * *pi_;
    ETValue denom_sigma = ETValue(int64_t(216)) * (eighteen_pi - ETValue(int64_t(1)));
    alpha_Sigma_ = std::make_unique<ETValue>(ETValue(int64_t(1)) / denom_sigma);

    // α⁻¹(ET) = A₀ + A₁ − A_cross − Σ_geometric
    alpha_inv_ = std::make_unique<ETValue>(
        *alpha_A0_ + *alpha_A1_ - *alpha_Across_ - *alpha_Sigma_
    );
}

void ETConstants::compute_cascade_residuals() {
    // |δ_r| = |12·log₂(12) − 43|
    // 12·log₂(12) = 12 × (2 + log₂(3)) = 24 + 12·log₂(3)
    ETValue twelve_log2_twelve = ETValue(int64_t(12)) * math::log2(ETValue(int64_t(12)));
    ETValue delta_r_raw = twelve_log2_twelve - ETValue(int64_t(43));
    delta_r_ = std::make_unique<ETValue>(math::abs(delta_r_raw));

    // |δ_θ| = |24π/ln(2) − 109|
    ETValue twenty_four_pi_over_ln2 = (ETValue(int64_t(24)) * *pi_) / *ln2_;
    ETValue delta_theta_raw = twenty_four_pi_over_ln2 - ETValue(int64_t(109));
    delta_theta_ = std::make_unique<ETValue>(math::abs(delta_theta_raw));
}

void ETConstants::compute_gaze_thresholds() {
    // All four gaze thresholds are just-intonation intervals (§3.18.9)
    gaze_subliminal_ = std::make_unique<ETValue>(ETValue::from_rational(13, 12)); // augmented unison
    gaze_detected_   = std::make_unique<ETValue>(ETValue::from_rational(6, 5));   // quintic minor third = Γ
    gaze_locked_     = std::make_unique<ETValue>(ETValue::from_rational(3, 2));   // perfect fifth
    gaze_lock_con_   = std::make_unique<ETValue>(ETValue::from_rational(5, 4));   // major third
}

void ETConstants::compute_zeta_values() {
    // ζ(s) for s ∈ {2..13} via FLINT/Arb
    for (int si = 2; si <= 13; si++) {
        ETValue s_val{static_cast<int64_t>(si)};
        zeta_cache_[si] = std::make_unique<ETValue>(special::zeta(s_val));
    }
}

void ETConstants::compute_impedance_coupling() {
    // A₀_magic(d) = (d-1)² + S² where S=4
    // ξ(d) = 137 / A₀_magic(d)
    // For d ∈ {1..12}
    ETValue s_squared(int64_t(ET_S * ET_S)); // S² = 16
    ETValue one_three_seven(int64_t(137));

    for (int d = 1; d <= 12; d++) {
        int64_t d_minus_1 = d - 1;
        ETValue a0 = ETValue(d_minus_1 * d_minus_1) + s_squared;
        impedance_cache_[d] = std::make_unique<ETValue>(a0);
        xi_cache_[d] = std::make_unique<ETValue>(one_three_seven / a0);
    }
}

// ── Accessors (return cached constants) ────────────────────────────────────

const ETValue& ETConstants::pi()               { return *pi_; }
const ETValue& ETConstants::e()                { return *e_; }
const ETValue& ETConstants::euler_gamma()      { return *euler_gamma_; }
const ETValue& ETConstants::phi()              { return *phi_; }
const ETValue& ETConstants::ln2()              { return *ln2_; }
const ETValue& ETConstants::sqrt2()            { return *sqrt2_; }
const ETValue& ETConstants::sqrt3()            { return *sqrt3_; }
const ETValue& ETConstants::sqrt5()            { return *sqrt5_; }
const ETValue& ETConstants::K()               { return *K_; }
const ETValue& ETConstants::V()               { return *V_; }
const ETValue& ETConstants::N()               { return *N_; }
const ETValue& ETConstants::S_val()           { return *S_val_; }
const ETValue& ETConstants::sigma()           { return *sigma_; }
const ETValue& ETConstants::life_threshold()  { return *life_threshold_; }
const ETValue& ETConstants::k_em()            { return *k_em_; }
const ETValue& ETConstants::p_eff()           { return *p_eff_; }
const ETValue& ETConstants::alpha_inv()       { return *alpha_inv_; }
const ETValue& ETConstants::alpha_A0()        { return *alpha_A0_; }
const ETValue& ETConstants::alpha_A1()        { return *alpha_A1_; }
const ETValue& ETConstants::alpha_Across()    { return *alpha_Across_; }
const ETValue& ETConstants::alpha_Sigma()     { return *alpha_Sigma_; }
const ETValue& ETConstants::delta_r()         { return *delta_r_; }
const ETValue& ETConstants::delta_theta()     { return *delta_theta_; }
const ETValue& ETConstants::gaze_subliminal() { return *gaze_subliminal_; }
const ETValue& ETConstants::gaze_detected()   { return *gaze_detected_; }
const ETValue& ETConstants::gaze_locked()     { return *gaze_locked_; }
const ETValue& ETConstants::gaze_lock_con()   { return *gaze_lock_con_; }

const ETValue& ETConstants::zeta(int s) {
    if (s < 2 || s > 13) {
        throw ETError(ETError::Code::INVALID_INPUT,
                      "ETConstants::zeta",
                      "Cached zeta values only for s in {2..13}",
                      std::to_string(s));
    }
    return *zeta_cache_[s];
}

const ETValue& ETConstants::coupling_xi(int d) {
    if (d < 1 || d > 12) {
        throw ETError(ETError::Code::INVALID_INPUT,
                      "ETConstants::coupling_xi",
                      "Coupling ξ(d) only for d in {1..12}",
                      std::to_string(d));
    }
    return *xi_cache_[d];
}

const ETValue& ETConstants::impedance(int d) {
    if (d < 1 || d > 12) {
        throw ETError(ETError::Code::INVALID_INPUT,
                      "ETConstants::impedance",
                      "Impedance A₀(d) only for d in {1..12}",
                      std::to_string(d));
    }
    return *impedance_cache_[d];
}

} // namespace et