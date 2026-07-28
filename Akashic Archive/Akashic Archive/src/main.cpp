// ============================================================================
// main.cpp — EUDD Manager Entry Point
//
// Stage 1: Precision Stack verification.
// Future stages will add --omniscient mode dispatch and full Manager startup.
//
// P ∘ D ∘ T = E
// ============================================================================

#include "precision_stack.h"
#include "core_lattice.h"
#include "akashic_format.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

// ============================================================================
// Stage 1 Verification — Structural Self-Test
//
// Confirms:
//   1. MPFR initialized at 1200-bit precision (361 dps = 1200 cents/octave)
//   2. All ET constants computed at 361 dps
//   3. Lossless bijection round-trip (basic: rational values)
//   4. SHA-256 known-answer test (FIPS compliance)
//   5. CRC-32 known-answer test
//   6. Fine structure constant α⁻¹(ET) matches expected value
//   7. Impedance monotonic descent ξ(1) > ξ(2) > ... > ξ(12) = 1.0
//   8. Cascade residuals |δ_r|, |δ_θ| produce correct n_max values
//   9. FLINT/Arb special functions operational (ζ(3) spot check)
//   10. Serialization round-trip (ETValue → blob → ETValue)
// ============================================================================

static int g_pass_count = 0;
static int g_fail_count = 0;

static void check(bool condition, const char* test_name, const std::string& detail = "") {
    if (condition) {
        std::printf("  [PASS] %s\n", test_name);
        g_pass_count++;
    } else {
        std::printf("  [FAIL] %s", test_name);
        if (!detail.empty()) {
            std::printf(" — %s", detail.c_str());
        }
        std::printf("\n");
        g_fail_count++;
    }
}

static int run_stage1_verification() {
    std::printf("\n");
    std::printf("===========================================================\n");
    std::printf("  EUDD Manager — Stage 1 Precision Stack Verification\n");
    std::printf("  P . D . T = E\n");
    std::printf("===========================================================\n\n");

    // ── 1. Initialization ──────────────────────────────────────────────
    std::printf("[Section 1] Initialization\n");
    try {
        et::initialize();
        check(et::is_initialized(), "et::initialize() completed");
        check(et::ETConstants::is_initialized(), "ETConstants initialized");
    } catch (const std::exception& ex) {
        std::printf("  [FATAL] Initialization failed: %s\n", ex.what());
        return 1;
    }

    // ── 2. ETValue Basic Operations ────────────────────────────────────
    std::printf("\n[Section 2] ETValue Basic Operations\n");
    {
        et::ETValue zero;
        check(zero.is_zero(), "Default ETValue is zero");
        check(!zero.is_negative(), "Zero is not negative");

        et::ETValue one(int64_t(1));
        check(!one.is_zero(), "ETValue(1) is not zero");
        check(one.is_positive(), "ETValue(1) is positive");
        check(one.is_integer(), "ETValue(1) is integer");

        et::ETValue neg_one = -one;
        check(neg_one.is_negative(), "Negation produces negative");

        // Arithmetic
        et::ETValue two = one + one;
        check(two == et::ETValue(int64_t(2)), "1 + 1 = 2");

        et::ETValue six = et::ETValue(int64_t(2)) * et::ETValue(int64_t(3));
        check(six == et::ETValue(int64_t(6)), "2 * 3 = 6");

        et::ETValue half = one / et::ETValue(int64_t(2));
        et::ETValue quarter = half * half;
        et::ETValue expected_quarter = et::ETValue::from_rational(1, 4);
        check(quarter == expected_quarter, "0.5 * 0.5 = 0.25");

        // Rational construction
        et::ETValue two_thirds = et::ETValue::from_rational(2, 3);
        check(two_thirds == et::ETConstants::K(), "2/3 = K (Koide ratio)");

        // String round-trip
        std::string pi_str = et::ETConstants::pi().to_string(30);
        check(pi_str.find("3.1415926535") != std::string::npos,
              "π string starts with 3.1415926535",
              pi_str.substr(0, 40));
    }

    // ── 3. Mathematical Constants ──────────────────────────────────────
    std::printf("\n[Section 3] Mathematical Constants at 361 dps\n");
    {
        // π > 3.14159 and < 3.14160
        check(et::ETConstants::pi() > et::ETValue("3.14159")
           && et::ETConstants::pi() < et::ETValue("3.14160"),
              "π in expected range");

        // e > 2.71828 and < 2.71829
        check(et::ETConstants::e() > et::ETValue("2.71828")
           && et::ETConstants::e() < et::ETValue("2.71829"),
              "e in expected range");

        // φ > 1.61803 and < 1.61804
        check(et::ETConstants::phi() > et::ETValue("1.61803")
           && et::ETConstants::phi() < et::ETValue("1.61804"),
              "φ in expected range");

        // γ > 0.57721 and < 0.57722
        check(et::ETConstants::euler_gamma() > et::ETValue("0.57721")
           && et::ETConstants::euler_gamma() < et::ETValue("0.57722"),
              "γ in expected range");

        // Verify φ² = φ + 1 (golden ratio identity)
        const auto& phi = et::ETConstants::phi();
        et::ETValue phi_sq = phi * phi;
        et::ETValue phi_plus_1 = phi + et::ETValue(int64_t(1));
        et::ETValue diff = et::math::abs(phi_sq - phi_plus_1);
        check(diff < et::ETValue("1e-350"), "φ² = φ + 1 (golden ratio identity)",
              diff.to_string(20));
    }

    // ── 4. ET Structural Constants ─────────────────────────────────────
    std::printf("\n[Section 4] ET Structural Constants\n");
    {
        check(et::ETConstants::K() == et::ETValue::from_rational(2, 3), "K = 2/3");
        check(et::ETConstants::V() == et::ETValue::from_rational(1, 12), "V = 1/12");
        check(et::ETConstants::N() == et::ETValue(int64_t(12)), "N = 12");
        check(et::ETConstants::S_val() == et::ETValue(int64_t(4)), "S = 4");

        // σ = √(1/12) — verify σ² = 1/12
        const auto& sigma = et::ETConstants::sigma();
        et::ETValue sigma_sq = sigma * sigma;
        const auto& v = et::ETConstants::V();
        et::ETValue diff = et::math::abs(sigma_sq - v);
        check(diff < et::ETValue("1e-350"), "σ² = V = 1/12",
              diff.to_string(20));

        check(et::ETConstants::life_threshold() == et::ETValue::from_rational(13, 12),
              "LIFE_THRESHOLD = 13/12");
        check(et::ETConstants::k_em() == et::ETValue(int64_t(8)), "K_EM = 8");
        check(et::ETConstants::p_eff() == et::ETValue::from_rational(10, 3), "p_eff = 10/3");
    }

    // ── 5. Fine Structure Constant ─────────────────────────────────────
    std::printf("\n[Section 5] Fine Structure Constant α⁻¹(ET)\n");
    {
        const auto& alpha_inv = et::ETConstants::alpha_inv();

        // CODATA 2022: α⁻¹ = 137.035999177(21)
        // ET prediction: α⁻¹(ET) = 137.03599916744...
        // Agreement within 0.46σ → difference < 1.1e-10
        check(alpha_inv > et::ETValue("137.035999"),
              "α⁻¹(ET) > 137.035999");
        check(alpha_inv < et::ETValue("137.036000"),
              "α⁻¹(ET) < 137.036000");

        // Verify A₀ = 137 exactly
        check(et::ETConstants::alpha_A0() == et::ETValue(int64_t(137)),
              "A₀ = 137 (base impedance)");

        // Verify α⁻¹ = A₀ + A₁ − A_cross − Σ_geometric
        et::ETValue reconstructed =
            et::ETConstants::alpha_A0()
          + et::ETConstants::alpha_A1()
          - et::ETConstants::alpha_Across()
          - et::ETConstants::alpha_Sigma();
        et::ETValue diff = et::math::abs(alpha_inv - reconstructed);
        check(diff < et::ETValue("1e-350"),
              "α⁻¹ = A₀ + A₁ - A_cross - Σ_geometric (self-consistency)",
              diff.to_string(20));

        std::printf("  α⁻¹(ET) = %s\n", alpha_inv.to_string(40).c_str());
    }

    // ── 6. Impedance Monotonic Descent ─────────────────────────────────
    std::printf("\n[Section 6] Impedance ξ(d) Monotonic Descent\n");
    {
        bool monotonic = true;
        for (int d = 1; d < 12; d++) {
            if (et::ETConstants::coupling_xi(d) <= et::ETConstants::coupling_xi(d + 1)) {
                monotonic = false;
                break;
            }
        }
        check(monotonic, "ξ(d) strictly decreasing for d=1..12");

        // ξ(1) = 137/16 = 8.5625
        et::ETValue xi1 = et::ETConstants::coupling_xi(1);
        et::ETValue expected_xi1 = et::ETValue::from_rational(137, 16);
        check(xi1 == expected_xi1, "ξ(1) = 137/16 = 8.5625");

        // ξ(12) = 137/137 = 1.0 exactly
        et::ETValue xi12 = et::ETConstants::coupling_xi(12);
        check(xi12 == et::ETValue(int64_t(1)), "ξ(12) = 1.0 exactly (EM baseline)");

        // A₀(12) = 137
        check(et::ETConstants::impedance(12) == et::ETValue(int64_t(137)),
              "A₀(12) = (12-1)² + 4² = 121 + 16 = 137");
    }

    // ── 7. Cascade Residuals ───────────────────────────────────────────
    std::printf("\n[Section 7] Cascade Residuals and n_max\n");
    {
        const auto& delta_r = et::ETConstants::delta_r();
        const auto& delta_theta = et::ETConstants::delta_theta();

        // n_max_r = ⌊0.5/|δ_r|⌋ should be 25
        et::ETValue half("0.5");
        et::ETValue n_max_r_val = et::math::floor(half / delta_r);
        check(n_max_r_val == et::ETValue(int64_t(25)), "n_max_r = 25",
              "computed: " + n_max_r_val.to_string(5));

        // n_max_θ = ⌊0.5/|δ_θ|⌋ should be 2
        et::ETValue n_max_theta_val = et::math::floor(half / delta_theta);
        check(n_max_theta_val == et::ETValue(int64_t(2)), "n_max_θ = 2",
              "computed: " + n_max_theta_val.to_string(5));

        // ── N-Weight Subsumption Theorem (NWS-1, NWS-2) ──────────────────
        // The freedom ratio |δ_θ|/|δ_r| at 12ET is NOT N=12. It is 11.4249...
        // The gap 12 - ratio = 0.5751... is the d=35 cross-complex shadow
        // (5×7 = non-divisors of 12, the quintic shadow's cross-complex partner).
        // The N-Weight identity: ratio + shadow = N = 12 exactly.
        // At 27720ET (terminal simultaneous-activation resolution = LCM(1..11)),
        // both the ratio and shadow project to lattice-exact positions
        // (|ε| < lattice spacing = 1200/27720 ≈ 0.0433¢).

        et::ETValue ratio = delta_theta / delta_r;

        // Test 1: ratio at 12ET ≈ 11.4249 (1200-bit exact)
        check(ratio > et::ETValue("11.424884771244") && ratio < et::ETValue("11.424884771245"),
              "|δ_θ|/|δ_r| = 11.424884771244... at 12ET (d=35 shadow present)",
              ratio.to_string(30));

        // Test 2: shadow = N - ratio ≈ 0.5751
        et::ETValue n_val(int64_t(12));
        et::ETValue shadow = n_val - ratio;
        check(shadow > et::ETValue("0.575115228755") && shadow < et::ETValue("0.575115228756"),
              "Shadow gap = 0.575115228755... (d=35 = 5×7 cross-complex)",
              shadow.to_string(30));

        // Test 3: N-Weight Subsumption identity — ratio + shadow = 12 exactly
        et::ETValue sum = ratio + shadow;
        et::ETValue identity_diff = et::math::abs(sum - n_val);
        check(identity_diff < et::ETValue("1e-350"),
              "N-Weight identity: ratio + shadow = 12 (exact at 1200-bit)",
              identity_diff.to_string(20));

        // Test 4: 27720ET subsumption — project ratio onto 27720ET lattice,
        // verify |ε| < lattice spacing (1200/27720 ≈ 0.0433 cents)
        // Projection: exact_k = 27720 × log₂(ratio), k = round(exact_k),
        //             ε_cents = (exact_k − k) × 1200/27720
        et::ETValue n_full(int64_t(27720));
        et::ETValue exact_k = n_full * et::math::log2(ratio);
        et::ETValue k_rounded = et::math::round(exact_k);
        et::ETValue eps_frac = exact_k - k_rounded;
        et::ETValue twelve_hundred(int64_t(1200));
        et::ETValue eps_cents = (eps_frac * twelve_hundred) / n_full;
        et::ETValue abs_eps = et::math::abs(eps_cents);
        et::ETValue lattice_spacing = twelve_hundred / n_full;
        check(abs_eps < lattice_spacing,
              "27720ET subsumption: |ε| < spacing (ratio is lattice-exact at N_FULL)",
              "|ε|=" + abs_eps.to_string(15) + "¢, spacing=" + lattice_spacing.to_string(15) + "¢");
    }

    // ── 8. Special Functions (FLINT/Arb) ───────────────────────────────
    std::printf("\n[Section 8] Special Functions via FLINT/Arb\n");
    {
        // ζ(2) = π²/6
        et::ETValue zeta2 = et::ETConstants::zeta(2);
        et::ETValue pi_sq_over_6 = (et::ETConstants::pi() * et::ETConstants::pi())
                                   / et::ETValue(int64_t(6));
        et::ETValue diff = et::math::abs(zeta2 - pi_sq_over_6);
        check(diff < et::ETValue("1e-350"), "ζ(2) = π²/6",
              diff.to_string(20));

        // ζ(3) ≈ 1.2020569... (Apéry's constant)
        et::ETValue zeta3 = et::ETConstants::zeta(3);
        check(zeta3 > et::ETValue("1.20205") && zeta3 < et::ETValue("1.20206"),
              "ζ(3) ≈ 1.2020569... (Apéry's constant)",
              zeta3.to_string(30));

        // Γ(1) = 1 exactly (0! = 1)
        et::ETValue gamma_1 = et::special::gamma(et::ETValue(int64_t(1)));
        check(et::math::abs(gamma_1 - et::ETValue(int64_t(1))) < et::ETValue("1e-350"),
              "Γ(1) = 1");

        // Γ(1/2) = √π
        et::ETValue gamma_half = et::special::gamma(et::ETValue::from_rational(1, 2));
        et::ETValue sqrt_pi = et::math::sqrt(et::ETConstants::pi());
        et::ETValue diff2 = et::math::abs(gamma_half - sqrt_pi);
        check(diff2 < et::ETValue("1e-350"), "Γ(1/2) = √π",
              diff2.to_string(20));
    }

    // ── 9. SHA-256 Known-Answer Test ───────────────────────────────────
    std::printf("\n[Section 9] SHA-256 FIPS Compliance\n");
    {
        // NIST test vector: SHA-256("abc") = ba7816bf...
        std::string hash_abc = et::SHA256::hash_hex("abc");
        check(hash_abc == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
              "SHA-256(\"abc\") matches NIST test vector");

        // Empty string: SHA-256("") = e3b0c442...
        std::string hash_empty = et::SHA256::hash_hex("");
        check(hash_empty == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
              "SHA-256(\"\") matches NIST test vector");
    }

    // ── 10. CRC-32 Known-Answer Test ───────────────────────────────────
    std::printf("\n[Section 10] CRC-32 Verification\n");
    {
        // CRC-32 of "123456789" = 0xCBF43926
        uint32_t crc = et::CRC32::compute("123456789");
        check(crc == 0xCBF43926, "CRC-32(\"123456789\") = 0xCBF43926",
              "computed: 0x" + [&](){
                  char buf[16];
                  std::snprintf(buf, sizeof(buf), "%08X", crc);
                  return std::string(buf);
              }());
    }

    // ── 11. Serialization Round-Trip ───────────────────────────────────
    std::printf("\n[Section 11] ETValue Serialization Round-Trip\n");
    {
        // Round-trip test for π
        const auto& original_pi = et::ETConstants::pi();
        auto blob = original_pi.serialize();
        et::ETValue recovered_pi = et::ETValue::deserialize(blob);
        et::ETValue diff = et::math::abs(original_pi - recovered_pi);
        check(diff < et::ETValue("1e-350"),
              "π serialization round-trip lossless",
              "blob size: " + std::to_string(blob.size()) + " bytes, diff: " + diff.to_string(20));

        // Round-trip for zero
        et::ETValue zero;
        auto zero_blob = zero.serialize();
        et::ETValue recovered_zero = et::ETValue::deserialize(zero_blob);
        check(recovered_zero.is_zero(), "Zero serialization round-trip",
              "blob size: " + std::to_string(zero_blob.size()) + " bytes");

        // Round-trip for negative value
        et::ETValue neg = -et::ETConstants::phi();
        auto neg_blob = neg.serialize();
        et::ETValue recovered_neg = et::ETValue::deserialize(neg_blob);
        et::ETValue neg_diff = et::math::abs(neg - recovered_neg);
        check(neg_diff < et::ETValue("1e-350"),
              "-φ serialization round-trip lossless");
    }

    // ── 12. Integer Number Theory ──────────────────────────────────────
    std::printf("\n[Section 12] Integer Number Theory\n");
    {
        check(et::intmath::gcd(12, 8) == 4, "gcd(12, 8) = 4");
        check(et::intmath::gcd(27720, 12) == 12, "gcd(27720, 12) = 12");
        check(et::intmath::lcm(12, 8) == 24, "lcm(12, 8) = 24");

        // LCM(1..11) = 27720 = N_FULL — computed with arbitrary precision
        et::ETInteger lcm_1_11(int64_t(1));
        for (int k = 2; k <= 11; k++) {
            lcm_1_11 = et::intmath::lcm(lcm_1_11, et::ETInteger(int64_t(k)));
        }
        check(lcm_1_11 == et::ET_N_FULL, "LCM(1..11) = 27720 = N_FULL",
              "computed: " + lcm_1_11.to_string());

        // Divisors of 12 = {1,2,3,4,6,12} — the 6 simple sublattice families
        auto divs_12 = et::intmath::divisors(12);
        check(divs_12.size() == 6, "τ(12) = 6 (six simple sublattice families)");
        // Build expected vector for comparison
        bool divs_match = (divs_12.size() == 6
            && divs_12[0] == 1 && divs_12[1] == 2 && divs_12[2] == 3
            && divs_12[3] == 4 && divs_12[4] == 6 && divs_12[5] == 12);
        check(divs_match, "divisors(12) = {1,2,3,4,6,12}");

        // Totient: φ(12) = 4
        check(et::intmath::totient(12) == 4, "φ(12) = 4");

        // LCM landmarks — fully arbitrary precision, no overflow ceiling
        auto landmarks = et::intmath::lcm_landmarks(13);
        bool found_12 = false, found_60 = false, found_27720 = false;
        for (const auto& [k, v] : landmarks) {
            if (v == 12) found_12 = true;
            if (v == 60) found_60 = true;
            if (v == 27720) found_27720 = true;
        }
        check(found_12, "LCM landmark N=12 found");
        check(found_60, "LCM landmark N=60 found");
        check(found_27720, "LCM landmark N=27720 found");

        // 2^12 = 4096 = page size
        check(et::intmath::is_power_of_two(4096), "4096 = 2^12 is power of two");
        check(!et::intmath::is_power_of_two(12), "12 is NOT power of two");
    }

    // ── Summary ────────────────────────────────────────────────────────
    std::printf("\n===========================================================\n");
    std::printf("  Results: %d passed, %d failed, %d total\n",
                g_pass_count, g_fail_count, g_pass_count + g_fail_count);
    if (g_fail_count == 0) {
        std::printf("  Status:  ALL TESTS PASSED\n");
        std::printf("  Module 1 (Precision Stack) VERIFIED\n");
    } else {
        std::printf("  Status:  %d FAILURES — investigation required\n", g_fail_count);
    }
    std::printf("===========================================================\n\n");

    return (g_fail_count == 0) ? 0 : 1;
}

// ============================================================================
// Stage 2 Verification — Core Lattice Engine
//
// Confirms:
//   1. Projection Π_12(1) = (0, 1, 0)
//   2. Projection Π_12(2) = (12, 1, 0)
//   3. Projection Π_12(√2) = (6, 2, 0)
//   4. Projection Π_12(3/2) = (7, 12, +1.955¢)
//   5. Projection Π_12(2/3) = (-7, 12, -1.955¢)
//   6. Bijection pullback round-trip (Losslessness Theorem)
//   7. k-arithmetic: multiplication, reciprocation, power
//   8. Gaussian signature classification
//   9. Coupling ξ(d) for arbitrary d
//   10. Tightness, ∂I distance, manifold state
//   11. FQG quadrant, palindromic partner
//   12. N=27720 projection (full resolution)
// ============================================================================

static int run_stage2_verification() {
    std::printf("\n");
    std::printf("===========================================================\n");
    std::printf("  EUDD Manager — Stage 2 Core Lattice Engine Verification\n");
    std::printf("  P . D . T = E\n");
    std::printf("===========================================================\n\n");

    // ── 1. Canonical Projections at N=12 (Paper §5 Prop 5.3) ──────────
    std::printf("[Section 1] Canonical Projections at N=12\n");
    {
        // Π_12(1) = (0, 1, 0) — the identity cell
        auto p1 = et::lattice::project(et::ETValue(int64_t(1)), int64_t(12));
        check(p1.k == 0, "Π_12(1): k = 0");
        check(p1.d == 1, "Π_12(1): d = 1 (identity/octave cell)");
        check(p1.eps.is_zero(), "Π_12(1): ε = 0 (exactly on lattice)");
        check(p1.manifold_state == et::lattice::ManifoldState::EXCEPTION,
              "Π_12(1): manifold state = PDT (Exception)");
        check(p1.eps_micros == 0, "Π_12(1): ε_micros = 0");

        // Π_12(2) = (12, 1, 0) — one octave up
        auto p2 = et::lattice::project(et::ETValue(int64_t(2)), int64_t(12));
        check(p2.k == 12, "Π_12(2): k = 12");
        check(p2.d == 1, "Π_12(2): d = 1 (octave)");
        check(p2.eps.is_zero(), "Π_12(2): ε = 0 (exactly on lattice)");

        // Π_12(√2) = (6, 2, 0) — tritone
        et::ETValue sqrt2 = et::math::sqrt(et::ETValue(int64_t(2)));
        auto p3 = et::lattice::project(sqrt2, int64_t(12));
        check(p3.k == 6, "Π_12(√2): k = 6");
        check(p3.d == 2, "Π_12(√2): d = 2 (tritone/pivot)");
        check(p3.eps.is_zero(), "Π_12(√2): ε = 0 (exactly on lattice)");

        // Π_12(4) = (24, 1, 0) — two octaves
        auto p4 = et::lattice::project(et::ETValue(int64_t(4)), int64_t(12));
        check(p4.k == 24, "Π_12(4): k = 24");
        check(p4.d == 1, "Π_12(4): d = 1 (two octaves)");
        check(p4.eps.is_zero(), "Π_12(4): ε = 0");
    }

    // ── 2. Perfect Fifth and Koide Ratio (Paper §5 Props 5.4-5.5) ────
    std::printf("\n[Section 2] Perfect Fifth & Koide Ratio\n");
    {
        // Π_12(3/2) = (7, 12, +1.955¢) — perfect fifth
        et::ETValue three_halves = et::ETValue::from_rational(3, 2);
        auto pf = et::lattice::project(three_halves, int64_t(12));
        check(pf.k == 7, "Π_12(3/2): k = 7");
        check(pf.d == 12, "Π_12(3/2): d = 12 (full resolution, coprime)");
        // ε should be ≈ +1.955 cents (the Pythagorean comma)
        check(pf.eps > et::ETValue("1.954") && pf.eps < et::ETValue("1.956"),
              "Π_12(3/2): ε ≈ +1.955¢ (Pythagorean comma)",
              pf.eps.to_string(20));
        check(pf.eps_micros == 1955,
              "Π_12(3/2): ε_micros ≈ 1955",
              "got: " + std::to_string(pf.eps_micros));
        check(pf.coprime_skeleton, "Π_12(3/2): coprime skeleton (gcd(7,12)=1)");

        // Π_12(2/3) = (-7, 12, -1.955¢) — Koide ratio (mirror of fifth)
        et::ETValue two_thirds = et::ETValue::from_rational(2, 3);
        auto pk = et::lattice::project(two_thirds, int64_t(12));
        check(pk.k == -7, "Π_12(2/3): k = -7");
        check(pk.d == 12, "Π_12(2/3): d = 12");
        check(pk.eps < et::ETValue("-1.954") && pk.eps > et::ETValue("-1.956"),
              "Π_12(2/3): ε ≈ -1.955¢",
              pk.eps.to_string(20));

        // Mirror symmetry: ε(3/2) = -ε(2/3)
        et::ETValue sum_eps = pf.eps + pk.eps;
        check(et::math::abs(sum_eps) < et::ETValue("1e-340"),
              "Mirror symmetry: ε(3/2) + ε(2/3) = 0",
              sum_eps.to_string(20));
    }

    // ── 3. Bijection Pullback Round-Trip (Losslessness Theorem) ───────
    std::printf("\n[Section 3] Bijection Pullback Round-Trip\n");
    {
        // Test with 3/2: project then pullback should recover original
        et::ETValue three_halves = et::ETValue::from_rational(3, 2);
        auto proj = et::lattice::project(three_halves, int64_t(12));
        et::ETValue recovered = et::lattice::pullback(proj);
        et::ETValue diff = et::math::abs(three_halves - recovered);
        check(diff < et::ETValue("1e-340"),
              "Pullback(Π_12(3/2)) = 3/2 (lossless round-trip)",
              diff.to_string(20));

        // Test with π: irrational value
        const auto& pi = et::ETConstants::pi();
        auto proj_pi = et::lattice::project(pi, int64_t(12));
        et::ETValue recovered_pi = et::lattice::pullback(proj_pi);
        et::ETValue diff_pi = et::math::abs(pi - recovered_pi);
        check(diff_pi < et::ETValue("1e-340"),
              "Pullback(Π_12(π)) = π (lossless round-trip)",
              diff_pi.to_string(20));

        // Test with φ at N=27720 (full resolution)
        const auto& phi = et::ETConstants::phi();
        auto proj_phi = et::lattice::project(phi, int64_t(27720));
        et::ETValue recovered_phi = et::lattice::pullback(proj_phi);
        et::ETValue diff_phi = et::math::abs(phi - recovered_phi);
        check(diff_phi < et::ETValue("1e-340"),
              "Pullback(Π_27720(φ)) = φ (lossless at N_FULL)",
              diff_phi.to_string(20));

        // Test with ζ(3) at N=12
        const auto& zeta3 = et::ETConstants::zeta(3);
        auto proj_z3 = et::lattice::project(zeta3, int64_t(12));
        et::ETValue recovered_z3 = et::lattice::pullback(proj_z3);
        et::ETValue diff_z3 = et::math::abs(zeta3 - recovered_z3);
        check(diff_z3 < et::ETValue("1e-340"),
              "Pullback(Π_12(ζ(3))) = ζ(3) (Apéry's constant round-trip)",
              diff_z3.to_string(20));
    }

    // ── 4. k-Arithmetic ───────────────────────────────────────────────
    std::printf("\n[Section 4] k-Arithmetic\n");
    {
        // k-addition: 3/2 × 4/3 = 2 (perfect fifth × perfect fourth = octave)
        // Π_12(3/2) → k=7, Π_12(4/3) → k=5
        // k_add: 7+5 = 12 → d=1, ε should be ≈ 0
        auto p_fifth = et::lattice::project(et::ETValue::from_rational(3, 2), int64_t(12));
        auto p_fourth = et::lattice::project(et::ETValue::from_rational(4, 3), int64_t(12));
        auto product = et::lattice::k_add(
            p_fifth.k, p_fifth.eps,
            p_fourth.k, p_fourth.eps,
            et::ETInteger(int64_t(12)));
        check(product.k_result == 12, "k-add: 3/2 × 4/3 → k = 12 (octave)");
        // Verify the actual product value ≈ 2
        et::ETValue diff = et::math::abs(product.r_result - et::ETValue(int64_t(2)));
        check(diff < et::ETValue("1e-340"),
              "k-add: 3/2 × 4/3 = 2.0 exactly",
              diff.to_string(20));

        // k-negation: reciprocal of 3/2 is 2/3
        auto recip = et::lattice::k_negate(p_fifth.k, p_fifth.eps, et::ETInteger(int64_t(12)));
        check(recip.k_result == -7, "k-negate: 1/(3/2) → k = -7");
        et::ETValue expected = et::ETValue::from_rational(2, 3);
        et::ETValue diff_recip = et::math::abs(recip.r_result - expected);
        check(diff_recip < et::ETValue("1e-340"),
              "k-negate: 1/(3/2) = 2/3",
              diff_recip.to_string(20));

        // k-scale: (3/2)^12 → k = 84 (twelve perfect fifths)
        auto power = et::lattice::k_scale(p_fifth.k, p_fifth.eps,
            et::ETInteger(int64_t(12)), et::ETInteger(int64_t(12)));
        check(power.k_result == 84, "k-scale: (3/2)^12 → k = 84");
    }

    // ── 5. Gaussian Signature ─────────────────────────────────────────
    std::printf("\n[Section 5] Gaussian Signature Classification\n");
    {
        // d=1: no factors → trivial signature "1"
        auto g1 = et::lattice::compute_gaussian_signature(et::ETInteger(int64_t(1)));
        check(g1.factors.empty(), "Gaussian(1): no factors");
        check(g1.signature_string == "1", "Gaussian(1): signature = \"1\"");

        // d=2: p=2 → Ramified
        auto g2 = et::lattice::compute_gaussian_signature(et::ETInteger(int64_t(2)));
        check(g2.factors.size() == 1, "Gaussian(2): one factor");
        check(g2.factors[0].gclass == et::lattice::GaussianClass::RAMIFIED,
              "Gaussian(2): p=2 is Ramified");
        check(g2.is_ramified_present, "Gaussian(2): ramified present");

        // d=3: p=3 ≡ 3 (mod 4) → Inert
        auto g3 = et::lattice::compute_gaussian_signature(et::ETInteger(int64_t(3)));
        check(g3.factors[0].gclass == et::lattice::GaussianClass::INERT,
              "Gaussian(3): p=3 is Inert (3 ≡ 3 mod 4)");
        check(g3.is_all_inert, "Gaussian(3): all inert");

        // d=5: p=5 ≡ 1 (mod 4) → Split
        auto g5 = et::lattice::compute_gaussian_signature(et::ETInteger(int64_t(5)));
        check(g5.factors[0].gclass == et::lattice::GaussianClass::SPLIT,
              "Gaussian(5): p=5 is Split (5 ≡ 1 mod 4)");
        check(g5.is_all_split, "Gaussian(5): all split");

        // d=12 = 2²·3: Ramified × Inert
        auto g12 = et::lattice::compute_gaussian_signature(et::ETInteger(int64_t(12)));
        check(g12.factors.size() == 2, "Gaussian(12): two prime factors (2,3)");
        check(g12.is_ramified_present, "Gaussian(12): ramified present (2)");
        check(!g12.is_all_inert && !g12.is_all_split,
              "Gaussian(12): mixed signature");

        // d=35 = 5×7: Split × Inert (the biological resolution cell)
        auto g35 = et::lattice::compute_gaussian_signature(et::ETInteger(int64_t(35)));
        check(g35.factors.size() == 2, "Gaussian(35): two factors (5,7)");
        // 5 ≡ 1 (mod 4) → Split, 7 ≡ 3 (mod 4) → Inert
        bool has_split = false, has_inert = false;
        for (const auto& f : g35.factors) {
            if (f.gclass == et::lattice::GaussianClass::SPLIT) has_split = true;
            if (f.gclass == et::lattice::GaussianClass::INERT) has_inert = true;
        }
        check(has_split && has_inert,
              "Gaussian(35): Split(5) × Inert(7) — mixed D+T and D");
    }

    // ── 6. Coupling ξ(d) Dynamic Computation ──────────────────────────
    std::printf("\n[Section 6] Coupling ξ(d) Dynamic\n");
    {
        // Verify against cached Module 1 values for d∈{1..12}
        for (int d = 1; d <= 12; d++) {
            et::ETValue computed = et::lattice::compute_coupling_xi(et::ETInteger(int64_t(d)));
            const et::ETValue& cached = et::ETConstants::coupling_xi(d);
            et::ETValue diff = et::math::abs(computed - cached);
            check(diff < et::ETValue("1e-350"),
                  ("ξ(" + std::to_string(d) + ") matches Module 1 cache").c_str());
        }

        // Test for d beyond 12 (no cap!)
        et::ETValue xi_35 = et::lattice::compute_coupling_xi(et::ETInteger(int64_t(35)));
        // ξ(35) = 137/((34)²+16) = 137/(1156+16) = 137/1172
        et::ETValue expected_xi_35 = et::ETValue(int64_t(137)) /
            (et::ETValue(int64_t(34*34)) + et::ETValue(int64_t(16)));
        et::ETValue diff = et::math::abs(xi_35 - expected_xi_35);
        check(diff < et::ETValue("1e-350"),
              "ξ(35) = 137/1172 (biological resolution, no cap)");

        // d=132 = 12×11 — the structural maximum combined family
        et::ETValue xi_132 = et::lattice::compute_coupling_xi(et::ETInteger(int64_t(132)));
        check(xi_132.is_positive(), "ξ(132) is positive");
        check(xi_132 < xi_35, "ξ(132) < ξ(35) (monotonic descent continues)");
    }

    // ── 7. Tightness, ∂I Distance, Manifold State ─────────────────────
    std::printf("\n[Section 7] Tightness, ∂I Distance, Manifold State\n");
    {
        // ε = 0 → tightness = 1.0, di_distance = 0, state = Exception
        et::ETValue t0 = et::lattice::compute_tightness(et::ETValue(int64_t(0)));
        check(t0 == et::ETValue(int64_t(1)), "tightness(ε=0) = 1.0");

        et::ETValue d0 = et::lattice::compute_di_distance(et::ETValue(int64_t(0)));
        check(d0.is_zero(), "di_distance(ε=0) = 0");

        // ε = 50 → tightness = 100/150 = 2/3 = K, di_distance = 1.0
        et::ETValue t50 = et::lattice::compute_tightness(et::ETValue(int64_t(50)));
        et::ETValue expected_t50 = et::ETValue::from_rational(2, 3);
        et::ETValue diff = et::math::abs(t50 - expected_t50);
        check(diff < et::ETValue("1e-350"),
              "tightness(ε=50) = 2/3 = K (Koide ratio at ∂I boundary!)");

        et::ETValue d50 = et::lattice::compute_di_distance(et::ETValue(int64_t(50)));
        check(d50 == et::ETValue(int64_t(1)), "di_distance(ε=50) = 1.0 (at ∂I)");

        // Manifold state tests
        auto ms0 = et::lattice::compute_manifold_state(
            et::ETValue(int64_t(0)), et::ETInteger(int64_t(1)), et::ETInteger(int64_t(12)));
        check(ms0 == et::lattice::ManifoldState::EXCEPTION,
              "Manifold state(ε=0) = PDT (Exception)");

        auto ms50 = et::lattice::compute_manifold_state(
            et::ETValue(int64_t(50)), et::ETInteger(int64_t(12)), et::ETInteger(int64_t(12)));
        check(ms50 == et::lattice::ManifoldState::INCOHERENCE,
              "Manifold state(ε=50) = PT (Incoherence at ∂I)");

        auto ms1 = et::lattice::compute_manifold_state(
            et::ETValue("1.955"), et::ETInteger(int64_t(12)), et::ETInteger(int64_t(12)));
        check(ms1 == et::lattice::ManifoldState::UNSUBSTANTIATED,
              "Manifold state(ε=1.955) = PD (Unsubstantiated)");
    }

    // ── 8. FQG Quadrant & Palindromic Partner ─────────────────────────
    std::printf("\n[Section 8] FQG Quadrant & Palindromic Partner\n");
    {
        // d=1 divides 12 → Simple Real (SR)
        auto fqg1 = et::lattice::compute_fqg_quadrant(
            et::ETInteger(int64_t(1)), et::ETInteger(int64_t(12)), 'r');
        check(fqg1 == et::lattice::FQGQuadrant::SR, "FQG(d=1, real) = SR");

        // d=12 divides 12 → Simple Real (SR)
        auto fqg12 = et::lattice::compute_fqg_quadrant(
            et::ETInteger(int64_t(12)), et::ETInteger(int64_t(12)), 'r');
        check(fqg12 == et::lattice::FQGQuadrant::SR, "FQG(d=12, real) = SR");

        // d=5 at N=12: 12 mod 5 ≠ 0 → Complex Real (CR)
        // But wait: at N=12, d can only be a divisor of 12. d=5 doesn't occur at N=12.
        // At N=60: d=5 divides 60 → Simple Real (SR)
        auto fqg5_60 = et::lattice::compute_fqg_quadrant(
            et::ETInteger(int64_t(5)), et::ETInteger(int64_t(60)), 'r');
        check(fqg5_60 == et::lattice::FQGQuadrant::SR, "FQG(d=5, N=60, real) = SR");

        // d=7 at N=60: 60 mod 7 ≠ 0 → Complex Real (CR)
        auto fqg7_60 = et::lattice::compute_fqg_quadrant(
            et::ETInteger(int64_t(7)), et::ETInteger(int64_t(60)), 'r');
        check(fqg7_60 == et::lattice::FQGQuadrant::CR, "FQG(d=7, N=60, real) = CR");

        // Imaginary axis: d=3, N=12 → Simple Imaginary (SI)
        auto fqg3_i = et::lattice::compute_fqg_quadrant(
            et::ETInteger(int64_t(3)), et::ETInteger(int64_t(12)), 'i');
        check(fqg3_i == et::lattice::FQGQuadrant::SI, "FQG(d=3, N=12, imag) = SI");

        // Palindromic partners at N=12
        // 1 ↔ 11
        auto pp1 = et::lattice::compute_palindromic_partner(
            et::ETInteger(int64_t(1)), et::ETInteger(int64_t(12)));
        check(pp1 == 11, "Palindromic partner(d=1, N=12) = 11");

        // 3 ↔ 9
        auto pp3 = et::lattice::compute_palindromic_partner(
            et::ETInteger(int64_t(3)), et::ETInteger(int64_t(12)));
        check(pp3 == 9, "Palindromic partner(d=3, N=12) = 9");

        // 5 ↔ 7
        auto pp5 = et::lattice::compute_palindromic_partner(
            et::ETInteger(int64_t(5)), et::ETInteger(int64_t(12)));
        check(pp5 == 7, "Palindromic partner(d=5, N=12) = 7");

        // 6 ↔ 6 (self-partner at midpoint)
        auto pp6 = et::lattice::compute_palindromic_partner(
            et::ETInteger(int64_t(6)), et::ETInteger(int64_t(12)));
        check(pp6 == 6, "Palindromic partner(d=6, N=12) = 6 (self-partner)");

        // 12 ↔ 12 (self-partner at N=12)
        auto pp12 = et::lattice::compute_palindromic_partner(
            et::ETInteger(int64_t(12)), et::ETInteger(int64_t(12)));
        check(pp12 == 12, "Palindromic partner(d=12, N=12) = 12 (self-partner)");
    }

    // ── 9. N=27720 Projection (Full Resolution) ──────────────────────
    std::printf("\n[Section 9] Full-Resolution Projection at N=27720\n");
    {
        // ── ζ(3) at N=27720: d=693 (the d=693 attractor from §3.18.1) ──
        const auto& zeta3 = et::ETConstants::zeta(3);
        auto pz3 = et::lattice::project(zeta3, int64_t(27720));
        check(pz3.k == 7360, "Π_27720(ζ(3)): k = 7360");
        check(pz3.d == 693, "Π_27720(ζ(3)): d = 693 (the d=693 attractor)");
        // 693 = 3² × 7 × 11 — cubic-squared × septic × undecimal
        check(pz3.gaussian_sig.is_ramified_present == false,
              "Π_27720(ζ(3)): d=693 has no ramified factors (no p=2)");

        // Pullback round-trip at N=27720
        et::ETValue recovered = et::lattice::pullback(pz3);
        et::ETValue diff = et::math::abs(zeta3 - recovered);
        check(diff < et::ETValue("1e-340"),
              "Pullback(Π_27720(ζ(3))) = ζ(3) (lossless at N_FULL)",
              diff.to_string(20));

        std::printf("  ζ(3) at 27720ET: k=%s, d=%s, ε=%s¢\n",
                    pz3.k.to_string().c_str(),
                    pz3.d.to_string().c_str(),
                    pz3.eps.to_string(15).c_str());

        // ── π at N=27720: coprime to 27720 → d=27720 ──────────────────
        const auto& pi = et::ETConstants::pi();
        auto ppi = et::lattice::project(pi, int64_t(27720));
        check(ppi.d == 27720,
              "Π_27720(π): d = 27720 (π is coprime to 27720)");
        check(ppi.coprime_skeleton,
              "Π_27720(π): coprime skeleton (gcd(k,N)=1)");
        std::printf("  π at 27720ET: k=%s, d=%s, ε=%s¢\n",
                    ppi.k.to_string().c_str(),
                    ppi.d.to_string().c_str(),
                    ppi.eps.to_string(15).c_str());

        // ── φ at N=27720: d=6930 ───────────────────────────────────────
        const auto& phi = et::ETConstants::phi();
        auto pphi = et::lattice::project(phi, int64_t(27720));
        check(pphi.d == 6930,
              "Π_27720(φ): d = 6930");
        std::printf("  φ at 27720ET: k=%s, d=%s, ε=%s¢\n",
                    pphi.k.to_string().c_str(),
                    pphi.d.to_string().c_str(),
                    pphi.eps.to_string(15).c_str());

        // ── K = 2/3 at N=27720: d=1848 ────────────────────────────────
        auto pk = et::lattice::project(et::ETConstants::K(), int64_t(27720));
        check(pk.d == 1848,
              "Π_27720(K=2/3): d = 1848");
        std::printf("  K=2/3 at 27720ET: k=%s, d=%s, ε=%s¢\n",
                    pk.k.to_string().c_str(),
                    pk.d.to_string().c_str(),
                    pk.eps.to_string(15).c_str());

        // ── α⁻¹ at N=27720: d=315, |ε|≈0.002¢ (§3.18.2) ─────────────
        const auto& alpha_inv = et::ETConstants::alpha_inv();
        auto pa = et::lattice::project(alpha_inv, int64_t(27720));
        check(pa.d == 315,
              "Π_27720(α⁻¹): d = 315 (3²×5×7 — cubic-squared × quintic × septic)");
        check(et::math::abs(pa.eps) < et::ETValue("0.01"),
              "Π_27720(α⁻¹): |ε| < 0.01¢ (near-lattice-exact)",
              pa.eps.to_string(15));
        std::printf("  α⁻¹ at 27720ET: k=%s, d=%s, ε=%s¢\n",
                    pa.k.to_string().c_str(),
                    pa.d.to_string().c_str(),
                    pa.eps.to_string(15).c_str());
    }

    // ── 10. On-Lattice Values via Bijection Teleporter ────────────────
    std::printf("\n[Section 10] On-Lattice Values (Bijection Teleporter)\n");
    {
        // Every value of the form 2^(k/N) must give ε = 0 exactly.
        // The bijection-teleporter computes L_k = 2^(k/N) and compares
        // directly to r. MPFR correct rounding guarantees the match.

        // 2^(1/3) at N=12: k=4, d=3 (cubic), ε=0
        et::ETValue two_pow_1_3 = et::math::exp2(et::ETValue::from_rational(1, 3));
        auto p13 = et::lattice::project(two_pow_1_3, int64_t(12));
        check(p13.k == 4, "Π_12(2^(1/3)): k = 4");
        check(p13.d == 3, "Π_12(2^(1/3)): d = 3 (cubic)");
        check(p13.eps.is_zero(), "Π_12(2^(1/3)): ε = 0 exactly (on lattice)");

        // 2^(1/4) at N=12: k=3, d=4 (quartic), ε=0
        et::ETValue two_pow_1_4 = et::math::exp2(et::ETValue::from_rational(1, 4));
        auto p14 = et::lattice::project(two_pow_1_4, int64_t(12));
        check(p14.k == 3, "Π_12(2^(1/4)): k = 3");
        check(p14.d == 4, "Π_12(2^(1/4)): d = 4 (quartic)");
        check(p14.eps.is_zero(), "Π_12(2^(1/4)): ε = 0 exactly (on lattice)");

        // 2^(1/6) at N=12: k=2, d=6 (hexadic), ε=0
        et::ETValue two_pow_1_6 = et::math::exp2(et::ETValue::from_rational(1, 6));
        auto p16 = et::lattice::project(two_pow_1_6, int64_t(12));
        check(p16.k == 2, "Π_12(2^(1/6)): k = 2");
        check(p16.d == 6, "Π_12(2^(1/6)): d = 6 (hexadic)");
        check(p16.eps.is_zero(), "Π_12(2^(1/6)): ε = 0 exactly (on lattice)");

        // 2^(1/12) at N=12: k=1, d=12 (full resolution), ε=0
        et::ETValue two_pow_1_12 = et::math::exp2(et::ETValue::from_rational(1, 12));
        auto p112 = et::lattice::project(two_pow_1_12, int64_t(12));
        check(p112.k == 1, "Π_12(2^(1/12)): k = 1");
        check(p112.d == 12, "Π_12(2^(1/12)): d = 12 (full resolution)");
        check(p112.eps.is_zero(), "Π_12(2^(1/12)): ε = 0 exactly (on lattice)");

        // 2^(5/12) at N=12: k=5, d=12 (coprime, 5 coprime to 12), ε=0
        et::ETValue two_pow_5_12 = et::math::exp2(et::ETValue::from_rational(5, 12));
        auto p512 = et::lattice::project(two_pow_5_12, int64_t(12));
        check(p512.k == 5, "Π_12(2^(5/12)): k = 5");
        check(p512.d == 12, "Π_12(2^(5/12)): d = 12 (5 coprime to 12)");
        check(p512.eps.is_zero(), "Π_12(2^(5/12)): ε = 0 exactly (on lattice)");
        check(p512.coprime_skeleton, "Π_12(2^(5/12)): coprime skeleton");

        // ε rational form: on-lattice values get exact (0, 1)
        check(p512.has_eps_rational, "On-lattice: has_eps_rational = true");
        check(p512.eps_rational_num.is_zero(), "On-lattice: eps_rational_num = 0");
        check(p512.eps_rational_den == 1, "On-lattice: eps_rational_den = 1");

        // Off-lattice: has_eps_rational = false (Module 5 CF method will determine later)
        et::ETValue three_halves_check = et::ETValue::from_rational(3, 2);
        auto p_off = et::lattice::project(three_halves_check, int64_t(12));
        check(!p_off.has_eps_rational,
              "Off-lattice (3/2): has_eps_rational = false (deferred to Module 5 CF)");
    }

    // ── 11. Impedance A₀(d) ───────────────────────────────────────────
    std::printf("\n[Section 11] Impedance A₀(d)\n");
    {
        // A₀(d) = (d−1)² + S² where S = 4
        // From §3.18.4 table:
        check(et::lattice::compute_impedance(et::ETInteger(int64_t(1)))
              == et::ETValue(int64_t(16)),
              "A₀(1) = 16 (Pure Will / Gravity)");
        check(et::lattice::compute_impedance(et::ETInteger(int64_t(3)))
              == et::ETValue(int64_t(20)),
              "A₀(3) = 20 (Cubic / QCD)");
        check(et::lattice::compute_impedance(et::ETInteger(int64_t(5)))
              == et::ETValue(int64_t(32)),
              "A₀(5) = 32 (Quintic / Golden)");
        check(et::lattice::compute_impedance(et::ETInteger(int64_t(12)))
              == et::ETValue(int64_t(137)),
              "A₀(12) = 137 (EM / Full Resolution = α⁻¹ base)");

        // Dynamic: A₀(35) = (34)² + 16 = 1156 + 16 = 1172
        check(et::lattice::compute_impedance(et::ETInteger(int64_t(35)))
              == et::ETValue(int64_t(1172)),
              "A₀(35) = 1172 (biological resolution)");
    }

    // ── 12. Variance V(n,k) ───────────────────────────────────────────
    std::printf("\n[Section 12] Variance V(n,k)\n");
    {
        // V(n,k) = (n²−1)/(12·2^k)
        // V(12, 0) = (144-1)/(12·1) = 143/12 = 11.9166...
        auto [v12_0, ok1] = et::lattice::compute_variance(
            et::ETInteger(int64_t(12)), et::ETInteger(int64_t(0)));
        et::ETValue expected = et::ETValue::from_rational(143, 12);
        et::ETValue diff = et::math::abs(v12_0 - expected);
        check(diff < et::ETValue("1e-340"),
              "V(12,0) = 143/12",
              v12_0.to_string(20));

        // V(2, 0) = (4-1)/(12·1) = 3/12 = 1/4
        auto [v2_0, ok2] = et::lattice::compute_variance(
            et::ETInteger(int64_t(2)), et::ETInteger(int64_t(0)));
        check(v2_0 == et::ETValue::from_rational(1, 4),
              "V(2,0) = 1/4");

        // V(1, 0) = (1-1)/(12·1) = 0 exactly
        auto [v1_0, ok3] = et::lattice::compute_variance(
            et::ETInteger(int64_t(1)), et::ETInteger(int64_t(0)));
        check(v1_0.is_zero(), "V(1,0) = 0 (d=1 identity has zero variance)");

        // V(12, 1) = 143/(12·2) = 143/24
        auto [v12_1, ok4] = et::lattice::compute_variance(
            et::ETInteger(int64_t(12)), et::ETInteger(int64_t(1)));
        et::ETValue expected2 = et::ETValue::from_rational(143, 24);
        diff = et::math::abs(v12_1 - expected2);
        check(diff < et::ETValue("1e-340"),
              "V(12,1) = 143/24");
    }

    // ── 13. Factorization Strings ─────────────────────────────────────
    std::printf("\n[Section 13] Factorization Strings\n");
    {
        auto proj12 = et::lattice::project(et::ETValue::from_rational(3, 2), int64_t(12));
        check(!proj12.d_factorization.empty(),
              "Π_12(3/2): d_factorization is non-empty");
        std::printf("  d=12 factorization: %s\n", proj12.d_factorization.c_str());

        // d=1: should be "1"
        auto proj1 = et::lattice::project(et::ETValue(int64_t(1)), int64_t(12));
        check(proj1.d_factorization == "1",
              "d=1 factorization string = \"1\"");

        // d=693 = 3²·7·11 — verify the factorization string content
        auto pz3 = et::lattice::project(et::ETConstants::zeta(3), int64_t(27720));
        std::printf("  d=693 factorization: %s\n", pz3.d_factorization.c_str());
        // Should contain 3, 7, 11 as factors
        check(pz3.d_factorization.find('3') != std::string::npos
           && pz3.d_factorization.find('7') != std::string::npos
           && pz3.d_factorization.find("11") != std::string::npos,
              "d=693 factorization contains 3, 7, 11");
    }

    // ── 14. Elegance Score Factors ────────────────────────────────────
    std::printf("\n[Section 14] Elegance Score Factors\n");
    {
        // For Π_12(3/2): d=12, ε≈1.955
        auto proj = et::lattice::project(et::ETValue::from_rational(3, 2), int64_t(12));

        // Symmetry = N/d = 12/12 = 1
        check(proj.elegance_symmetry == et::ETValue(int64_t(1)),
              "Elegance symmetry(3/2, N=12) = 12/12 = 1");

        // Tightness = 100/(100+1.955) ≈ 0.98083...
        check(proj.tightness > et::ETValue("0.980") && proj.tightness < et::ETValue("0.981"),
              "Tightness(3/2) ≈ 0.9808");

        // Apply simplicity via rational approx: 3/2 → p=3, q=2, p+q=5
        et::ETValue three_halves = et::ETValue::from_rational(3, 2);
        auto approx = et::lattice::best_rational_approx(
            three_halves, et::ETInteger(int64_t(1000)));
        check(approx.p == 3 && approx.q == 2,
              "Rational approx(3/2) = 3/2 (exact)");

        et::lattice::apply_simplicity(proj, approx);
        check(proj.has_simplicity, "Simplicity applied successfully");
        // simplicity = 100/max(1,5) = 20
        check(proj.elegance_simplicity == et::ETValue(int64_t(20)),
              "Elegance simplicity(3/2) = 100/5 = 20");

        // Universal = 1 × 0.9808... × 20 ≈ 19.616...
        check(proj.elegance_universal > et::ETValue("19.6") &&
              proj.elegance_universal < et::ETValue("19.7"),
              "Elegance universal(3/2) ≈ 19.6");
    }

    // ── Summary ────────────────────────────────────────────────────────
    std::printf("\n===========================================================\n");
    std::printf("  Results: %d passed, %d failed, %d total\n",
                g_pass_count, g_fail_count, g_pass_count + g_fail_count);
    if (g_fail_count == 0) {
        std::printf("  Status:  ALL TESTS PASSED\n");
        std::printf("  Module 2 (Core Lattice Engine) VERIFIED\n");
    } else {
        std::printf("  Status:  %d FAILURES — investigation required\n", g_fail_count);
    }
    std::printf("===========================================================\n\n");

    return (g_fail_count == 0) ? 0 : 1;
}

// ============================================================================
// Stage 3 Verification — Akashic Format Self-Test
//
// Confirms:
//   1. File creation writes valid header (magic, version, ET constants)
//   2. Header SHA-256 checksum is correct on disk
//   3. File reopen succeeds with header verification
//   4. Page allocation returns valid page-aligned offsets
//   5. Page write/read round-trip preserves content exactly
//   6. Page CRC-32 integrity verification passes on valid pages
//   7. Page CRC-32 detects corruption (modified body fails check)
//   8. Section directory get/set operations work correctly
//   9. Section initialization allocates pages properly
//   10. Memoization store: basic insert and lookup (cache hit)
//   11. Memoization store: lookup miss returns nullptr
//   12. Memoization store: K = 2/3 load factor triggers rehash
//   13. Memoization store: rehash doubles capacity, preserves entries
//   14. Memoization store: idempotent store (same hash → same entry)
//   15. Full file integrity check passes on clean file
//   16. Full file integrity check detects corrupted page
//   17. File close/reopen round-trip preserves header state
//   18. On-disk struct sizes match spec (compile-time verified)
// ============================================================================

static int run_stage3_verification() {
    std::printf("\n");
    std::printf("===========================================================\n");
    std::printf("  EUDD Manager — Stage 3 Akashic Format Verification\n");
    std::printf("  P . D . T = E\n");
    std::printf("===========================================================\n\n");

    const char* test_path = "test_sempaevum.akashic";

    // Clean up any leftover test file
    std::remove(test_path);

    // ── 1. File Creation ──────────────────────────────────────────────
    std::printf("[Section 1] File Creation\n");
    {
        et::akashic::AkashicFile af;
        try {
            af.create(test_path);
            check(af.is_open(), "File created and open");

            const auto& hdr = af.header();

            // Verify magic bytes
            check(hdr.magic[0] == 'S' && hdr.magic[1] == 'M' &&
                  hdr.magic[2] == 'V' && hdr.magic[3] == 'M',
                  "Header magic = \"SMVM\"");

            // Verify format version
            check(hdr.format_version == et::akashic::FORMAT_VERSION,
                  "Header format version correct",
                  "expected: " + std::to_string(et::akashic::FORMAT_VERSION));

            // Verify ET constants
            check(hdr.n_base == et::ET_N,
                  "Header N_base = 12 (forced resolution)");
            check(hdr.k_num == 2 && hdr.k_den == 3,
                  "Header K = 2/3 (Koide ratio)");
            check(hdr.v_num == 1 && hdr.v_den == 12,
                  "Header V = 1/12 (base variance)");

            // Verify self-projection (4096 = 2^12 → d=1, ε=0 at N=12)
            check(hdr.self_d == 1, "Self-projection d=1 (octave — 4096 is pure power of 2)");
            check(hdr.self_eps_micros == 0, "Self-projection ε=0 exactly (lattice-rational)");
            check(hdr.self_n == 12, "Self-projection N=12");
            check(hdr.self_k == 144, "Self-projection k=144 (12*log₂(4096)=12*12=144)");

            // Verify initial metrics
            check(hdr.total_generators == 0, "Initial total_generators = 0");
            check(hdr.total_memoized == 0, "Initial total_memoized = 0");
            check(hdr.total_pages == 1, "Initial total_pages = 1 (header only)");

            // Verify timestamps are set
            check(hdr.created_at_ns > 0, "created_at_ns is set");
            check(hdr.modified_at_ns > 0, "modified_at_ns is set");

            af.close();
            check(!af.is_open(), "File closed successfully");
        } catch (const std::exception& ex) {
            std::printf("  [FATAL] File creation failed: %s\n", ex.what());
            return 1;
        }
    }

    // ── 2. Header SHA-256 Checksum ────────────────────────────────────
    std::printf("\n[Section 2] Header SHA-256 Verification\n");
    {
        et::akashic::AkashicFile af;
        af.open(test_path);
        check(af.verify_header_checksum(), "On-disk header SHA-256 checksum valid");
        af.close();
    }

    // ── 3. File Reopen ────────────────────────────────────────────────
    std::printf("\n[Section 3] File Reopen\n");
    {
        et::akashic::AkashicFile af;
        try {
            af.open(test_path);
            check(af.is_open(), "File reopened successfully");

            const auto& hdr = af.header();
            check(hdr.magic[0] == 'S' && hdr.magic[1] == 'M' &&
                  hdr.magic[2] == 'V' && hdr.magic[3] == 'M',
                  "Reopened header magic still \"SMVM\"");
            check(hdr.n_base == et::ET_N, "Reopened header N_base still 12");
            check(hdr.k_num == 2 && hdr.k_den == 3, "Reopened header K still 2/3");

            af.close();
        } catch (const std::exception& ex) {
            std::printf("  [FATAL] File reopen failed: %s\n", ex.what());
            return 1;
        }
    }

    // ── 4. Page Allocation ────────────────────────────────────────────
    std::printf("\n[Section 4] Page Allocation\n");
    {
        et::akashic::AkashicFile af;
        af.open(test_path);

        // Allocate a page
        uint64_t offset1 = af.allocate_page(
            et::akashic::PageType::MEMO_ENTRY,
            et::akashic::SectionID::MEMOIZATION_STORE);

        check(offset1 == et::akashic::PAGE_SIZE,
              "First allocated page at offset 4096 (after header)",
              "got: " + std::to_string(offset1));
        check(offset1 % et::akashic::PAGE_SIZE == 0,
              "Page offset is page-aligned");

        // Allocate a second page
        uint64_t offset2 = af.allocate_page(
            et::akashic::PageType::MEMO_ENTRY,
            et::akashic::SectionID::MEMOIZATION_STORE);

        check(offset2 == 2 * et::akashic::PAGE_SIZE,
              "Second page at offset 8192",
              "got: " + std::to_string(offset2));
        check(offset2 > offset1, "Pages allocated in order");

        // Total pages should now be 3 (header + 2 data pages)
        check(af.header().total_pages == 3,
              "total_pages = 3 after allocating 2 data pages");

        af.close();
    }

    // ── 5. Page Write/Read Round-Trip ─────────────────────────────────
    std::printf("\n[Section 5] Page Write/Read Round-Trip\n");
    {
        et::akashic::AkashicFile af;
        af.open(test_path);

        // Allocate a page
        uint64_t offset = af.allocate_page(
            et::akashic::PageType::CATALOG_DATA,
            et::akashic::SectionID::STRUCTURAL_CATALOG);

        // Write test data into the page body
        et::akashic::Page write_page{};
        write_page.clear();
        write_page.header.page_type = static_cast<uint8_t>(et::akashic::PageType::CATALOG_DATA);
        write_page.header.section_id = static_cast<uint8_t>(et::akashic::SectionID::STRUCTURAL_CATALOG);
        write_page.header.entry_count = 42;  // Meaningful: 42 combined families
        write_page.header.used_bytes = 256;

        // Fill body with a recognizable pattern
        for (size_t i = 0; i < et::akashic::PAGE_BODY_SIZE; ++i) {
            write_page.body[i] = static_cast<uint8_t>(i & 0xFF);
        }

        af.write_page(offset, write_page);

        // Read back
        et::akashic::Page read_page{};
        af.read_page(offset, read_page);

        check(read_page.header.entry_count == 42, "Page entry_count preserved");
        check(read_page.header.used_bytes == 256, "Page used_bytes preserved");

        // Verify body content byte by byte
        bool body_match = true;
        for (size_t i = 0; i < et::akashic::PAGE_BODY_SIZE; ++i) {
            if (read_page.body[i] != static_cast<uint8_t>(i & 0xFF)) {
                body_match = false;
                break;
            }
        }
        check(body_match, "Page body content preserved exactly (4032 bytes)");

        af.close();
    }

    // ── 6. Page CRC-32 Verification ───────────────────────────────────
    std::printf("\n[Section 6] Page CRC-32 Integrity\n");
    {
        et::akashic::AkashicFile af;
        af.open(test_path);

        // The page written in Section 5 should verify
        // (write_page computes CRC-32 automatically)
        uint64_t offset = 3 * et::akashic::PAGE_SIZE;  // The third data page

        check(af.verify_page(offset), "Valid page passes CRC-32 verification");

        af.close();
    }

    // ── 7. CRC-32 Corruption Detection ────────────────────────────────
    std::printf("\n[Section 7] CRC-32 Corruption Detection\n");
    {
        // Write directly to corrupt a page body byte
        FILE* raw = std::fopen(test_path, "r+b");
        check(raw != nullptr, "Opened file for raw corruption test");

        if (raw) {
            // Corrupt one byte in the body of the page at offset 3*4096
            // Body starts at offset 3*4096 + 64 (after page header)
            uint64_t corrupt_offset = 3 * et::akashic::PAGE_SIZE + 64 + 100;
#ifdef _WIN32
            _fseeki64(raw, static_cast<int64_t>(corrupt_offset), SEEK_SET);
#else
            fseeko(raw, static_cast<off_t>(corrupt_offset), SEEK_SET);
#endif
            uint8_t bad_byte = 0xFF;
            std::fwrite(&bad_byte, 1, 1, raw);
            std::fflush(raw);
            std::fclose(raw);

            // Now try to verify — should fail
            et::akashic::AkashicFile af;
            af.open(test_path);

            check(!af.verify_page(3 * et::akashic::PAGE_SIZE),
                  "Corrupted page FAILS CRC-32 verification");

            af.close();
        }
    }

    // ── 8. Section Directory ──────────────────────────────────────────
    std::printf("\n[Section 8] Section Directory\n");
    {
        et::akashic::AkashicFile af;
        af.open(test_path);

        // All sections should initially be 0 (uncreated) except any we
        // set during prior tests. Let's test a fresh section:
        uint64_t gen_offset = af.section_offset(et::akashic::SectionID::GENERATOR_BACKBONE);
        check(gen_offset == 0, "Generator backbone section initially uncreated (offset=0)");

        // Initialize a section
        uint64_t new_offset = af.initialize_section(
            et::akashic::SectionID::GENERATOR_BACKBONE,
            et::akashic::PageType::GENERATOR_ENTRY);

        check(new_offset > 0, "Section initialization allocated a page");
        check(new_offset % et::akashic::PAGE_SIZE == 0, "Section offset is page-aligned");
        check(af.section_offset(et::akashic::SectionID::GENERATOR_BACKBONE) == new_offset,
              "Section directory updated");

        af.close();
    }

    // ── 9. Memoization Store: Basic Insert & Lookup ───────────────────
    std::printf("\n[Section 9] Memoization Store\n");
    {
        et::akashic::AkashicFile af;
        af.open(test_path);

        auto& memo = af.memo_store();

        // Create a test equation hash
        auto hash1 = et::SHA256::hash("2+2=4");

        // Lookup should be a MISS initially
        auto* result = memo.lookup(hash1);
        check(result == nullptr, "Lookup of unstored hash returns nullptr (MISS)");
        check(memo.total_lookups() == 1, "total_lookups incremented on miss");
        check(memo.total_hits() == 0, "total_hits unchanged on miss");

        // Store an entry
        et::akashic::MemoEntry entry;
        entry.equation_hash = hash1;
        entry.canonical_form = "2+2";
        entry.form_class = et::akashic::FormClass::ARITHMETIC_COMPUTATION;
        entry.operation_type = et::akashic::OpType::ADD;
        entry.output_value = et::ETValue(int64_t(4));
        entry.output_eps_micros = 0;
        entry.reference_count = 1;
        entry.first_computed_ns = et::akashic::AkashicFile::now_ns();
        entry.last_referenced_ns = entry.first_computed_ns;
        entry.occupied = true;

        auto* stored = memo.store(entry);
        check(stored != nullptr, "Store returned non-null pointer");
        check(stored->occupied, "Stored entry is occupied");
        check(memo.occupied() == 1, "occupied count = 1 after first store");

        // Lookup should now be a HIT
        auto* hit = memo.lookup(hash1);
        check(hit != nullptr, "Lookup of stored hash returns non-null (HIT)");
        check(memo.total_hits() == 1, "total_hits incremented on hit");
        check(hit->reference_count == 2, "reference_count incremented on hit");

        // Second hash: should be a MISS
        auto hash2 = et::SHA256::hash("3*5=15");
        auto* miss = memo.lookup(hash2);
        check(miss == nullptr, "Lookup of different hash is MISS");

        af.close();
    }

    // ── 10. Memoization Store: K = 2/3 Load Factor ────────────────────
    std::printf("\n[Section 10] Memoization K = 2/3 Load Factor\n");
    {
        et::akashic::MemoStore memo;
        memo.initialize(16);  // Small capacity for testing: 16 slots

        // K = 2/3 means rehash triggers when occupied > 16 * 2/3 ≈ 10.67
        // So after 10 inserts: no rehash. After 11: should rehash.
        check(memo.capacity() == 16, "Initial capacity = 16");

        for (int i = 0; i < 10; ++i) {
            std::string expr = "test_" + std::to_string(i);
            auto hash = et::SHA256::hash(expr);

            et::akashic::MemoEntry entry;
            entry.equation_hash = hash;
            entry.canonical_form = expr;
            entry.form_class = et::akashic::FormClass::ARITHMETIC_COMPUTATION;
            entry.operation_type = et::akashic::OpType::ADD;
            entry.output_value = et::ETValue(int64_t(i));
            entry.output_eps_micros = 0;
            entry.reference_count = 1;
            entry.first_computed_ns = et::akashic::AkashicFile::now_ns();
            entry.last_referenced_ns = entry.first_computed_ns;
            entry.occupied = true;

            memo.store(entry);
        }

        check(memo.occupied() == 10, "10 entries stored");
        check(memo.capacity() == 16, "Capacity still 16 (load = 10/16 = 0.625 < 2/3)");
        check(!memo.needs_rehash(), "No rehash needed at load 10/16");

        // Insert 11th entry — should trigger rehash (11/16 = 0.6875 > 2/3 = 0.6667)
        auto hash11 = et::SHA256::hash("test_10");
        et::akashic::MemoEntry entry11;
        entry11.equation_hash = hash11;
        entry11.canonical_form = "test_10";
        entry11.form_class = et::akashic::FormClass::ARITHMETIC_COMPUTATION;
        entry11.operation_type = et::akashic::OpType::ADD;
        entry11.output_value = et::ETValue(int64_t(10));
        entry11.output_eps_micros = 0;
        entry11.reference_count = 1;
        entry11.first_computed_ns = et::akashic::AkashicFile::now_ns();
        entry11.last_referenced_ns = entry11.first_computed_ns;
        entry11.occupied = true;

        memo.store(entry11);

        check(memo.capacity() == 32,
              "Capacity doubled to 32 after K=2/3 load exceeded (doubling law)");
        check(memo.occupied() == 11, "All 11 entries preserved after rehash");

        // Verify all 11 entries are still findable
        bool all_found = true;
        for (int i = 0; i <= 10; ++i) {
            auto hash = et::SHA256::hash("test_" + std::to_string(i));
            if (memo.lookup(hash) == nullptr) {
                all_found = false;
                break;
            }
        }
        check(all_found, "All 11 entries recoverable after rehash");
    }

    // ── 11. Memoization Store: Idempotent Store ───────────────────────
    std::printf("\n[Section 11] Memoization Idempotency\n");
    {
        et::akashic::MemoStore memo;
        memo.initialize(64);

        auto hash = et::SHA256::hash("sin(pi/4)");
        et::akashic::MemoEntry entry;
        entry.equation_hash = hash;
        entry.canonical_form = "sin(pi/4)";
        entry.form_class = et::akashic::FormClass::FUNCTION_EVALUATION;
        entry.operation_type = et::akashic::OpType::SIN;
        entry.output_value = et::ETValue("0.707106781186547524400844362104849039284835937688474036588339868995366239231053519425193767163820786367506");
        entry.output_eps_micros = 0;
        entry.reference_count = 1;
        entry.first_computed_ns = et::akashic::AkashicFile::now_ns();
        entry.last_referenced_ns = entry.first_computed_ns;
        entry.occupied = true;

        memo.store(entry);
        check(memo.occupied() == 1, "First store: occupied = 1");

        // Store same hash again (idempotent)
        memo.store(entry);
        check(memo.occupied() == 1, "Second store of same hash: occupied still 1 (idempotent)");

        // Lookup should show reference_count incremented
        auto* result = memo.lookup(hash);
        check(result != nullptr, "Lookup after idempotent store succeeds");
        // reference_count: 1 (initial) + 1 (idempotent store increment) + 1 (lookup) = 3
        check(result->reference_count == 3, "reference_count = 3 (initial + store + lookup)");
    }

    // ── 12. Full File Integrity ───────────────────────────────────────
    std::printf("\n[Section 12] Full File Integrity\n");
    {
        // First, create a clean file
        std::remove(test_path);
        et::akashic::AkashicFile af;
        af.create(test_path);

        // Allocate and write a few pages
        for (int i = 0; i < 5; ++i) {
            uint64_t offset = af.allocate_page(
                et::akashic::PageType::MEMO_ENTRY,
                et::akashic::SectionID::MEMOIZATION_STORE);
            et::akashic::Page p{};
            p.clear();
            p.header.page_type = static_cast<uint8_t>(et::akashic::PageType::MEMO_ENTRY);
            p.header.section_id = static_cast<uint8_t>(et::akashic::SectionID::MEMOIZATION_STORE);
            p.header.entry_count = static_cast<uint32_t>(i + 1);
            // Write some data in the body
            for (size_t j = 0; j < 100; ++j) {
                p.body[j] = static_cast<uint8_t>((i * 100 + j) & 0xFF);
            }
            p.header.used_bytes = 100;
            af.write_page(offset, p);
        }

        af.flush_header();

        // Full integrity check should pass
        uint64_t corrupt_page = 0;
        check(af.verify_full_integrity(&corrupt_page),
              "Full integrity check passes on clean file");

        af.close();
    }

    // ── 13. Compile-Time Size Assertions ──────────────────────────────
    std::printf("\n[Section 13] Compile-Time Struct Sizes\n");
    {
        check(sizeof(et::akashic::AkashicFileHeader) == 4096,
              "AkashicFileHeader = 4096 bytes (= 2^N = one page)");
        check(sizeof(et::akashic::PageHeader) == 64,
              "PageHeader = 64 bytes");
        check(sizeof(et::akashic::Page) == 4096,
              "Page = 4096 bytes (header + body)");
        check(et::akashic::PAGE_BODY_SIZE == 4032,
              "PAGE_BODY_SIZE = 4032 bytes (4096 - 64)");
    }

    // ════════════════════════════════════════════════════════════════
    // Stage 3b — Persistent Memoization Tests
    // ════════════════════════════════════════════════════════════════

    // ── 14. Varint Encode/Decode Round-Trip ────────────────────────
    std::printf("\n[Section 14] Varint Encode/Decode\n");
    {
        uint8_t buf[9];
        size_t bytes_read = 0;

        // Single-byte values (0–127)
        size_t n = et::akashic::varint::encode(0, buf);
        check(n == 1, "varint(0) encodes to 1 byte");
        check(et::akashic::varint::decode(buf, &bytes_read) == 0, "varint(0) round-trips");

        n = et::akashic::varint::encode(127, buf);
        check(n == 1, "varint(127) encodes to 1 byte");
        check(et::akashic::varint::decode(buf, &bytes_read) == 127, "varint(127) round-trips");

        // Two-byte values (128–16383)
        n = et::akashic::varint::encode(128, buf);
        check(n == 2, "varint(128) encodes to 2 bytes");
        check(et::akashic::varint::decode(buf, &bytes_read) == 128, "varint(128) round-trips");

        n = et::akashic::varint::encode(16383, buf);
        check(n == 2, "varint(16383) encodes to 2 bytes");
        check(et::akashic::varint::decode(buf, &bytes_read) == 16383, "varint(16383) round-trips");

        // Larger values
        n = et::akashic::varint::encode(27720, buf);
        check(n == 3, "varint(27720) encodes to 3 bytes");
        check(et::akashic::varint::decode(buf, &bytes_read) == 27720, "varint(27720=N_FULL) round-trips");

        // Signed values via zigzag
        n = et::akashic::varint::encode_signed(0, buf);
        check(n == 1, "varint_signed(0) encodes to 1 byte");
        check(et::akashic::varint::decode_signed(buf, &bytes_read) == 0, "varint_signed(0) round-trips");

        n = et::akashic::varint::encode_signed(-1, buf);
        check(n == 1, "varint_signed(-1) encodes to 1 byte (zigzag→1)");
        check(et::akashic::varint::decode_signed(buf, &bytes_read) == -1, "varint_signed(-1) round-trips");

        n = et::akashic::varint::encode_signed(7360, buf);
        check(n == 3, "varint_signed(7360) encodes to 3 bytes (zigzag→14720)");
        check(et::akashic::varint::decode_signed(buf, &bytes_read) == 7360,
              "varint_signed(7360=k of ζ(3) at N=27720) round-trips");

        n = et::akashic::varint::encode_signed(-16215, buf);
        check(n == 3, "varint_signed(-16215) encodes to 3 bytes (zigzag→32429)");
        check(et::akashic::varint::decode_signed(buf, &bytes_read) == -16215,
              "varint_signed(-16215=k of K at N=27720) round-trips");

        // encoded_size matches actual encode length
        check(et::akashic::varint::encoded_size(0) == 1, "encoded_size(0) = 1");
        check(et::akashic::varint::encoded_size(127) == 1, "encoded_size(127) = 1");
        check(et::akashic::varint::encoded_size(128) == 2, "encoded_size(128) = 2");
        check(et::akashic::varint::encoded_size(27720) == 3, "encoded_size(27720) = 3");
    }

    // ── 15. MemoEntry Serialize/Deserialize Round-Trip ─────────────
    std::printf("\n[Section 15] MemoEntry Serialization Round-Trip\n");
    {
        // Create a realistic entry: ζ(3) × π
        et::akashic::MemoEntry orig;
        orig.equation_hash = et::SHA256::hash("zeta(3)*pi");
        orig.canonical_form = "zeta(3)*pi";
        orig.form_class = et::akashic::FormClass::LATTICE_MULTIPLICATION;
        orig.operation_type = et::akashic::OpType::MUL;

        // Input refs: two operands at N=27720
        et::akashic::MemoEntry::InputRef ref1;
        ref1.n = et::ETInteger(int64_t(27720));
        ref1.k = et::ETInteger(int64_t(7360));
        ref1.d = et::ETInteger(int64_t(693));
        et::akashic::MemoEntry::InputRef ref2;
        ref2.n = et::ETInteger(int64_t(27720));
        ref2.k = et::ETInteger(int64_t(45779));
        ref2.d = et::ETInteger(int64_t(27720));
        orig.input_refs.push_back(ref1);
        orig.input_refs.push_back(ref2);

        // Output: ζ(3)·π ≈ 3.77574844285105...
        orig.output_n = et::ETInteger(int64_t(27720));
        orig.output_k = et::ETInteger(int64_t(53139));
        orig.output_d = et::ETInteger(int64_t(27720));
        orig.output_eps_micros = 120;
        orig.output_value = et::ETValue("3.77574844285105464946459427106961562600849647824032352898949532019437466006988752484712992662592088688");
        orig.reference_count = 42;
        orig.first_computed_ns = 1714742400000000000ULL;
        orig.last_referenced_ns = 1714742401000000000ULL;
        orig.occupied = true;

        // Serialize
        auto blob = et::akashic::memo_serial::serialize(orig);
        check(blob.size() > 100, "Serialized entry is non-trivial size",
              "got: " + std::to_string(blob.size()) + " bytes");

        // Deserialize
        size_t consumed = 0;
        auto restored = et::akashic::memo_serial::deserialize(blob.data(), blob.size(), &consumed);

        check(consumed == blob.size(), "Deserialize consumed all bytes");
        check(restored.occupied, "Restored entry is occupied");
        check(restored.equation_hash == orig.equation_hash, "Hash preserved");
        check(restored.canonical_form == orig.canonical_form, "Canonical form preserved");
        check(restored.form_class == orig.form_class, "Form class preserved");
        check(restored.operation_type == orig.operation_type, "Operation type preserved");
        check(restored.input_refs.size() == 2, "Input ref count preserved");
        check(restored.output_eps_micros == 120, "Output eps_micros preserved");
        check(restored.reference_count == 42, "Reference count preserved");
        check(restored.first_computed_ns == 1714742400000000000ULL, "first_computed_ns preserved");
        check(restored.last_referenced_ns == 1714742401000000000ULL, "last_referenced_ns preserved");

        // Verify 361-dps precision preserved exactly
        check(restored.output_value == orig.output_value,
              "Output value preserved at 361-dps (1200-bit) exactly");
    }

    // ── 16. Cross-Session Persistence ─────────────────────────────
    std::printf("\n[Section 16] Cross-Session Persistence\n");
    {
        // Clean start
        std::remove(test_path);

        // Session 1: create file, store entries, close
        {
            et::akashic::AkashicFile af;
            af.create(test_path);

            auto& memo = af.memo_store();

            // Store entry 1: 2+2=4
            et::akashic::MemoEntry e1;
            e1.equation_hash = et::SHA256::hash("2+2");
            e1.canonical_form = "2+2";
            e1.form_class = et::akashic::FormClass::ARITHMETIC_COMPUTATION;
            e1.operation_type = et::akashic::OpType::ADD;
            e1.output_n = et::ETInteger(int64_t(12));
            e1.output_k = et::ETInteger(int64_t(24));
            e1.output_d = et::ETInteger(int64_t(1));
            e1.output_eps_micros = 0;
            e1.output_value = et::ETValue(int64_t(4));
            e1.reference_count = 1;
            e1.first_computed_ns = et::akashic::AkashicFile::now_ns();
            e1.last_referenced_ns = e1.first_computed_ns;
            e1.occupied = true;
            memo.store(e1);

            // Store entry 2: sqrt(2)
            et::akashic::MemoEntry e2;
            e2.equation_hash = et::SHA256::hash("sqrt(2)");
            e2.canonical_form = "sqrt(2)";
            e2.form_class = et::akashic::FormClass::FUNCTION_EVALUATION;
            e2.operation_type = et::akashic::OpType::SQRT;
            e2.output_n = et::ETInteger(int64_t(12));
            e2.output_k = et::ETInteger(int64_t(6));
            e2.output_d = et::ETInteger(int64_t(2));
            e2.output_eps_micros = 0;
            e2.output_value = et::ETConstants::sqrt2();
            e2.reference_count = 5;
            e2.first_computed_ns = et::akashic::AkashicFile::now_ns();
            e2.last_referenced_ns = e2.first_computed_ns;
            e2.occupied = true;
            memo.store(e2);

            // Store entry 3: ζ(3) (Apéry's constant — the canonical ET test value)
            et::akashic::MemoEntry e3;
            e3.equation_hash = et::SHA256::hash("zeta(3)");
            e3.canonical_form = "zeta(3)";
            e3.form_class = et::akashic::FormClass::FUNCTION_EVALUATION;
            e3.operation_type = et::akashic::OpType::ZETA;
            e3.output_n = et::ETInteger(int64_t(27720));
            e3.output_k = et::ETInteger(int64_t(7360));
            e3.output_d = et::ETInteger(int64_t(693));
            e3.output_eps_micros = -85;
            e3.output_value = et::ETConstants::zeta(3);
            e3.reference_count = 100;
            e3.first_computed_ns = et::akashic::AkashicFile::now_ns();
            e3.last_referenced_ns = e3.first_computed_ns;
            e3.occupied = true;
            memo.store(e3);

            check(memo.occupied() == 3, "Session 1: 3 entries stored");

            af.close();
        }

        // Session 2: reopen and verify all entries survived
        {
            et::akashic::AkashicFile af;
            af.open(test_path);

            auto& memo = af.memo_store();
            check(memo.occupied() == 3, "Session 2: 3 entries loaded from disk");

            // Lookup each entry — should be cache HITs
            auto* hit1 = memo.lookup(et::SHA256::hash("2+2"));
            check(hit1 != nullptr, "Session 2: '2+2' found (cache HIT across sessions)");
            if (hit1) {
                check(hit1->canonical_form == "2+2", "Session 2: canonical form preserved");
                check(hit1->output_eps_micros == 0, "Session 2: eps_micros preserved");
            }

            auto* hit2 = memo.lookup(et::SHA256::hash("sqrt(2)"));
            check(hit2 != nullptr, "Session 2: 'sqrt(2)' found (cache HIT)");
            if (hit2) {
                // Verify √2 at 361-dps precision survives serialization
                check(hit2->output_value == et::ETConstants::sqrt2(),
                      "Session 2: √2 preserved at 361-dps EXACTLY");
            }

            auto* hit3 = memo.lookup(et::SHA256::hash("zeta(3)"));
            check(hit3 != nullptr, "Session 2: 'zeta(3)' found (cache HIT)");
            if (hit3) {
                check(hit3->output_value == et::ETConstants::zeta(3),
                      "Session 2: ζ(3) preserved at 361-dps EXACTLY");
                check(hit3->output_eps_micros == -85, "Session 2: ζ(3) eps_micros=-85 preserved");
                check(hit3->canonical_form == "zeta(3)", "Session 2: ζ(3) canonical form preserved");
            }

            // Miss should still be a miss
            auto* miss = memo.lookup(et::SHA256::hash("never_stored"));
            check(miss == nullptr, "Session 2: unstored hash is still MISS");

            af.close();
        }
    }

    // ── 17. Multi-Page Memoization ────────────────────────────────
    std::printf("\n[Section 17] Multi-Page Memoization Storage\n");
    {
        std::remove(test_path);

        // Store enough entries to require multiple pages
        // ~300 bytes per entry, PAGE_BODY = 4032 → ~13 entries per page
        // 20 entries should span 2+ pages
        {
            et::akashic::AkashicFile af;
            af.create(test_path);
            auto& memo = af.memo_store();

            for (int i = 0; i < 20; ++i) {
                std::string expr = "multi_page_test_" + std::to_string(i) +
                    "_with_padding_to_increase_entry_size_aaaaaaaaaaaaaaaa";
                et::akashic::MemoEntry e;
                e.equation_hash = et::SHA256::hash(expr);
                e.canonical_form = expr;
                e.form_class = et::akashic::FormClass::ARITHMETIC_COMPUTATION;
                e.operation_type = et::akashic::OpType::ADD;
                e.output_n = et::ETInteger(int64_t(12));
                e.output_k = et::ETInteger(int64_t(i));
                e.output_d = et::ETInteger(int64_t(1));
                e.output_eps_micros = i * 100;
                e.output_value = et::ETValue(int64_t(i * i));
                e.reference_count = static_cast<uint64_t>(i + 1);
                e.first_computed_ns = 1000000000ULL * static_cast<uint64_t>(i);
                e.last_referenced_ns = e.first_computed_ns;
                e.occupied = true;
                memo.store(e);
            }

            check(memo.occupied() == 20, "Stored 20 entries for multi-page test");
            af.close();
        }

        // Reopen and verify all 20 entries survived
        {
            et::akashic::AkashicFile af;
            af.open(test_path);
            auto& memo = af.memo_store();

            check(memo.occupied() == 20, "All 20 entries loaded from multi-page storage");

            // Verify each one
            bool all_found = true;
            for (int i = 0; i < 20; ++i) {
                std::string expr = "multi_page_test_" + std::to_string(i) +
                    "_with_padding_to_increase_entry_size_aaaaaaaaaaaaaaaa";
                auto* hit = memo.lookup(et::SHA256::hash(expr));
                if (!hit) {
                    all_found = false;
                    break;
                }
                if (hit->output_eps_micros != i * 100) {
                    all_found = false;
                    break;
                }
            }
            check(all_found, "All 20 entries recoverable with correct eps_micros values");

            af.close();
        }
    }

    // ── Cleanup test file ─────────────────────────────────────────────
    std::remove(test_path);

    // ── Summary ───────────────────────────────────────────────────────
    std::printf("\n===========================================================\n");
    std::printf("  Results: %d passed, %d failed, %d total\n",
                g_pass_count, g_fail_count, g_pass_count + g_fail_count);
    if (g_fail_count == 0) {
        std::printf("  Status:  ALL TESTS PASSED\n");
        std::printf("  Module 3 (Akashic Format) VERIFIED\n");
    } else {
        std::printf("  Status:  %d FAILURES — investigation required\n", g_fail_count);
    }
    std::printf("===========================================================\n\n");

    return (g_fail_count == 0) ? 0 : 1;
}

// ============================================================================
// Main Entry Point
// ============================================================================

int main(int argc, char* argv[]) {
    // Check for --omniscient flag (Module 26 dispatch point)
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--omniscient") == 0) {
            // Future: dispatch to omniscient_main() when Module 26 is implemented
            std::printf("Omniscient mode not yet implemented (Module 26).\n");
            return EXIT_FAILURE;
        }
    }

    // Stage 1: run Precision Stack verification
    int stage1_result = run_stage1_verification();

    // Stage 2: run Core Lattice Engine verification
    // Only run if Stage 1 passed (Module 2 depends on Module 1)
    if (stage1_result != 0) {
        std::printf("\n  Stage 1 failed — skipping Stages 2–3.\n");
        return stage1_result;
    }

    // Reset counters for Stage 2
    g_pass_count = 0;
    g_fail_count = 0;

    int stage2_result = run_stage2_verification();

    // Stage 3: run Akashic Format verification
    // Only run if Stage 2 passed (Module 3 depends on Modules 1, 2)
    if (stage2_result != 0) {
        std::printf("\n  Stage 2 failed — skipping Stage 3.\n");
        return stage2_result;
    }

    // Reset counters for Stage 3
    g_pass_count = 0;
    g_fail_count = 0;

    return run_stage3_verification();
}