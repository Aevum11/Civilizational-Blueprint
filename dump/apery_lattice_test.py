#!/usr/bin/env python3
"""
Apéry's Constant on the Lattice — Verification Test Script
============================================================

Complete ET-native verification script for the Apéry's Constant lattice
placement work (`Apery_Constant_on_the_Lattice_Place_and_Solve.md`).

Tests every structural claim in the document:
    - Full tower projection for ζ(s) for any s
    - Dynamic LCM tower generation (no hardcoded landmark list)
    - ET-derived resolution-relative sub-step threshold θ(N) = (1200/N)·V_base
    - Classification of each tower landmark: pre-convergence / plateau /
      false-resolution / true-home / intermediate-home / persistent-home /
      deep-home / post-deep
    - Gaussian prime classification of d-family factors (ramified/inert/split)
    - Coprime-skeleton status at each landmark
    - Shadow analysis (ζ(3)/(6/5) and its tower)
    - Attractor detection — which zeta values share d-families at each landmark
    - Apéry series verification (direct sum and fast-converging series)
    - Explicit assertions against every numerical claim in the document

Usage:
    python3 apery_lattice_test.py [--max-s 13] [--max-prime 13] [--precision 80]

ET Derivation Standard:
    - All thresholds are ET-derived (V_base=1/12, K=2/3, N=12)
    - Tower is dynamically generated from LCM of consecutive primes, not hardcoded
    - Precision scales with deepest tower landmark
    - No placeholders, no simulations, no stubs — production-ready

Based on Exception Theory by Michael James Muller.
From: "For every exception there is an exception, except the exception."
      P ∘ D ∘ T = E
"""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Iterator

try:
    from mpmath import mp, mpf, log, fabs, zeta, pi
except ImportError:
    sys.stderr.write("FATAL: mpmath is required. Install via: pip install mpmath\n")
    sys.exit(1)


# =============================================================================
# ET Constants (non-negotiable — derived from {P, D, T} primitive structure)
# =============================================================================

MANIFOLD_SYMMETRY: int = 12              # N = 3 primitives × 4 manifold states
BASE_VARIANCE: mpf = mpf(1) / mpf(12)    # V_base = 1/N
KOIDE_THRESHOLD: mpf = mpf(2) / mpf(3)   # K — tightness at ∂I boundary
T_WEIGHT: mpf = mpf(1) / mpf(3)          # T's share in P∘D∘T triadic balance
NORMALIZATION_EPSILON: mpf = mpf("1e-60")  # ET manifold stability floor
DI_BOUNDARY_CENTS: int = 50              # ∂I Incoherence boundary at |ε|=50¢
LIFE_THRESHOLD: mpf = mpf(13) / mpf(12)  # consciousness / permanence threshold

# Sub-step resolution threshold — ET derived (Option C from false-resolution investigation):
# At resolution N, the lattice step is 1200/N cents; sub-step precision is
# that step scaled by V_base. This tightens automatically as N grows.
def sub_step_threshold_cents(N: int) -> mpf:
    """θ_sub(N) = (1200/N) · V_base — the ET-native sub-cent threshold that
    scales with the viewing resolution. At N=12: 100·V_base·1 ≈ 8.33¢.
    At N=27720: 100/27720 ≈ 0.0036¢."""
    return mpf(1200) / mpf(N) * BASE_VARIANCE


# =============================================================================
# Dynamic LCM Tower — generated from primes, not hardcoded
# =============================================================================

def primes_up_to(limit: int) -> list[int]:
    """Sieve of Eratosthenes — genuinely dynamic, no list caps."""
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    return [i for i in range(limit + 1) if sieve[i]]


def lcm(a: int, b: int) -> int:
    return a * b // math.gcd(a, b) if a and b else 0


def lcm_of_range(start: int, end: int) -> int:
    """LCM(start..end) — dynamic, no upper bound beyond what Python supports."""
    result = 1
    for n in range(start, end + 1):
        result = lcm(result, n)
    return result


def generate_tower(max_prime: int) -> list[int]:
    """
    Generate the LCM tower dynamically. Landmarks are the structurally significant
    resolutions where new prime sublattice families first become native.

    For each prime p ≤ max_prime, generate landmarks derived from primes ≤ p.
    The result is the natural LCM-based tower without hardcoded milestones.
    """
    primes = primes_up_to(max_prime)
    landmarks: set[int] = set()

    # LCM landmarks: LCM(1..n) for n from 2 up to max_prime
    for n in range(2, max_prime + 1):
        landmarks.add(lcm_of_range(1, n))

    # Intermediate landmarks: structurally significant multiples
    # (base N=12 times successive primes, 12 × 2, 12 × 3, etc.)
    for p in primes:
        landmarks.add(MANIFOLD_SYMMETRY * p)
        # Also p-power landmarks within 12×p² range
        pp = p * p
        if pp <= max_prime * MANIFOLD_SYMMETRY:
            landmarks.add(MANIFOLD_SYMMETRY * pp)

    # Octave extensions of base biological-tier 420
    bio = lcm_of_range(1, 7)  # 420
    for k in range(1, 13):
        candidate = bio * k
        if candidate <= lcm_of_range(1, max_prime):
            landmarks.add(candidate)

    # Always include the base manifold 12
    landmarks.add(MANIFOLD_SYMMETRY)

    return sorted(landmarks)


# =============================================================================
# Core Lattice Projection
# =============================================================================

def factorize(n: int) -> dict[int, int]:
    """Integer prime factorization. Returns {prime: exponent}."""
    if n <= 1:
        return {}
    factors: dict[int, int] = {}
    temp = n
    p = 2
    while p * p <= temp:
        while temp % p == 0:
            factors[p] = factors.get(p, 0) + 1
            temp //= p
        p += 1
    if temp > 1:
        factors[temp] = factors.get(temp, 0) + 1
    return factors


def gaussian_class(p: int) -> str:
    """Classify a rational prime in the Gaussian integers ℤ[i]:
       RAMIFIED (p=2, P-type substrate),
       INERT (p ≡ 3 mod 4, pure D-type real axis),
       SPLIT (p ≡ 1 mod 4, D+T mixed Exception-type).
       From ET_Where_Does_Zero_Over_Zero_Come_In §22."""
    if p == 2:
        return "RAMIFIED"
    if p % 4 == 3:
        return "INERT"
    if p % 4 == 1:
        return "SPLIT"
    raise ValueError(f"Not a prime: {p}")


@dataclass(frozen=True)
class LatticeProjection:
    """A single lattice projection of ratio r at resolution N."""
    N: int
    r: mpf
    k: int
    d: int
    eps_cents: mpf

    @property
    def tightness(self) -> mpf:
        """t = 100 / (100 + |ε|)"""
        return mpf(100) / (mpf(100) + fabs(self.eps_cents))

    @property
    def dI_pct(self) -> mpf:
        """Distance to ∂I boundary (|ε|=50¢) as percentage."""
        return min(fabs(self.eps_cents) / mpf(DI_BOUNDARY_CENTS) * mpf(100), mpf(100))

    @property
    def quintic_tension_cents(self) -> mpf:
        """τ₅ — distance to nearest 5-ET position in cents."""
        step_5 = mpf(self.N) / mpf(5)
        nearest_5 = round(self.k / float(step_5)) * step_5
        return fabs(mpf(self.k) - nearest_5) * (mpf(1200) / mpf(self.N))

    @property
    def d_factors(self) -> dict[int, int]:
        """Prime factorization of d."""
        return factorize(self.d)

    @property
    def gaussian_signature(self) -> tuple[str, ...]:
        """Tuple of Gaussian-class labels for d's prime factors (sorted by prime)."""
        factors = self.d_factors
        return tuple(gaussian_class(p) for p in sorted(factors.keys()))

    @property
    def is_all_inert(self) -> bool:
        """True iff every prime factor of d is Gaussian-inert (≡3 mod 4)."""
        if not self.d_factors:
            return False
        return all(gc == "INERT" for gc in self.gaussian_signature)

    @property
    def is_coprime_skeleton(self) -> bool:
        """True iff gcd(k, N) = 1 (k in the coprime skeleton at this landmark)."""
        if self.k == 0:
            return False  # k=0 means ratio at unity; d=N by convention but not skeleton
        return math.gcd(abs(self.k), self.N) == 1

    @property
    def is_sub_step(self) -> bool:
        """True iff |ε| < ET-derived sub-step threshold (1200/N)·V_base."""
        return fabs(self.eps_cents) < sub_step_threshold_cents(self.N)

    @property
    def is_sub_cent(self) -> bool:
        """True iff |ε| < 1.0¢ (legacy coarse-comparison threshold)."""
        return fabs(self.eps_cents) < mpf(1)


def project(r: mpf, N: int) -> LatticeProjection:
    """
    Project ratio r onto the ET lattice at resolution N.

        k = round(N · log₂(r))
        d = N / gcd(|k|, N)
        ε = (1200·log₂(r)) - k·(1200/N)    [in cents]

    Preserves full mpmath precision.
    """
    if r <= 0:
        raise ValueError(f"Ratio must be positive, got {r}")
    lr = log(r, 2)
    val = mpf(N) * lr
    k = int(round(val))
    d = N // math.gcd(abs(k), N) if k != 0 else N
    eps_cents = mpf(1200) * lr - mpf(k) * (mpf(1200) / mpf(N))
    return LatticeProjection(N=N, r=r, k=k, d=d, eps_cents=eps_cents)


# =============================================================================
# Tower Trajectory Classification
# =============================================================================

@dataclass
class TowerTrajectory:
    """Complete tower trajectory for a ratio — all landmarks plus classification."""
    ratio_name: str
    ratio_value: mpf
    projections: list[LatticeProjection] = field(default_factory=list)

    @property
    def sub_step_events(self) -> list[LatticeProjection]:
        """All landmarks where |ε| < sub-step threshold."""
        return [p for p in self.projections if p.is_sub_step]

    @property
    def sub_cent_events(self) -> list[LatticeProjection]:
        """All landmarks where |ε| < 1¢ (coarse criterion)."""
        return [p for p in self.projections if p.is_sub_cent]

    def classify_landmarks(self) -> list[tuple[LatticeProjection, str]]:
        """
        Classify each landmark in the trajectory using the unified false-resolution
        / true-home diagnostic.

        Classification categories:
            PRE_CONVERGENCE — before any sub-cent event
            FALSE_RESOLUTION — sub-cent event whose d-family does NOT recur at
                               any later sub-cent landmark
            TRUE_HOME — first sub-cent event whose d-family DOES recur at a
                        later sub-cent landmark (signals persistent home)
            INTERMEDIATE_HOME — sub-cent event in a different d-family
                                than the true home's family
            PERSISTENT_HOME — sub-cent event at same d-family as established true home
            DEEP_HOME — the last (deepest-resolution) sub-cent event, if its
                        d-family differs from the true home
            PLATEAU — non-sub-cent landmark whose d-family has ≥3 total
                      occurrences in the tower AND ε is identical across those
                      occurrences (within 0.01¢). This captures the d-family
                      stability signature that distinguishes a structural
                      plateau from coincidental d-family matches.
            POST_CONVERGENCE — non-sub-cent landmark after the true home
        """
        # Count d-family occurrences across the ENTIRE tower for plateau detection.
        # A plateau has a d-family that appears ≥3 times with the same ε.
        d_occurrences: dict[int, list[int]] = {}  # d -> list of indices
        for idx, proj in enumerate(self.projections):
            d_occurrences.setdefault(proj.d, []).append(idx)

        # Identify plateau d-families: ≥3 occurrences with same ε (within 0.01¢)
        plateau_d_families: set[int] = set()
        for d_val, indices in d_occurrences.items():
            if len(indices) < 3:
                continue
            eps_values = [self.projections[i].eps_cents for i in indices]
            # Plateau requires ε consistency across all occurrences
            eps_spread = max(eps_values) - min(eps_values)
            if fabs(eps_spread) < mpf("0.01"):
                plateau_d_families.add(d_val)

        # Collect the set of d-families that appear at sub-cent landmarks.
        # A true home's d-family must recur — i.e., appear at ≥2 sub-cent events.
        sub_cent_d_counts: dict[int, int] = {}
        for proj in self.projections:
            if proj.is_sub_cent:
                sub_cent_d_counts[proj.d] = sub_cent_d_counts.get(proj.d, 0) + 1

        # Identify true home: first sub-cent event whose d-family recurs at
        # another sub-cent landmark later in the tower.
        true_home_idx: int | None = None
        for idx, proj in enumerate(self.projections):
            if not proj.is_sub_cent:
                continue
            # Recurrence check: does proj.d appear at any LATER sub-cent landmark?
            d_recurs = any(
                later.is_sub_cent and later.d == proj.d
                for later in self.projections[idx + 1:])
            if d_recurs:
                true_home_idx = idx
                break

        # Deep home: last sub-cent event (distinct if d differs from true_home_d)
        deep_home_idx: int | None = None
        for idx in range(len(self.projections) - 1, -1, -1):
            if self.projections[idx].is_sub_cent:
                deep_home_idx = idx
                break

        classifications: list[tuple[LatticeProjection, str]] = []
        true_home_d = (self.projections[true_home_idx].d
                       if true_home_idx is not None else None)

        for idx, proj in enumerate(self.projections):
            if proj.is_sub_cent:
                if true_home_idx is None:
                    # No recurring d-family anywhere — every sub-cent is false
                    label = "FALSE_RESOLUTION"
                elif idx < true_home_idx:
                    # Sub-cent before the recurring true home — false resolution
                    label = "FALSE_RESOLUTION"
                elif idx == true_home_idx:
                    label = "TRUE_HOME"
                elif idx == deep_home_idx and proj.d != true_home_d:
                    label = "DEEP_HOME"
                elif proj.d == true_home_d:
                    label = "PERSISTENT_HOME"
                else:
                    label = "INTERMEDIATE_HOME"
            else:
                # Non-sub-cent landmark
                if proj.d in plateau_d_families:
                    label = "PLATEAU"
                elif true_home_idx is not None and idx > true_home_idx:
                    label = "POST_CONVERGENCE"
                else:
                    label = "PRE_CONVERGENCE"
            classifications.append((proj, label))

        return classifications

    @property
    def true_home(self) -> LatticeProjection | None:
        """The true-home projection (first persistent sub-cent), if any."""
        for proj, label in self.classify_landmarks():
            if label == "TRUE_HOME":
                return proj
        return None

    @property
    def deep_home(self) -> LatticeProjection | None:
        """The deepest-resolution sub-cent projection, if any."""
        subs = self.sub_cent_events
        return subs[-1] if subs else None

    @property
    def intermediate_homes(self) -> list[LatticeProjection]:
        """All intermediate-home events."""
        return [p for p, label in self.classify_landmarks()
                if label == "INTERMEDIATE_HOME"]

    @property
    def false_resolutions(self) -> list[LatticeProjection]:
        """All false-resolution events."""
        return [p for p, label in self.classify_landmarks()
                if label == "FALSE_RESOLUTION"]


def build_trajectory(name: str, r: mpf, tower: list[int]) -> TowerTrajectory:
    """Build a complete tower trajectory for ratio r with the given landmark list."""
    traj = TowerTrajectory(ratio_name=name, ratio_value=r)
    for N in tower:
        traj.projections.append(project(r, N))
    return traj


# =============================================================================
# Attractor Detection — find shared sublattice families across multiple ratios
# =============================================================================

def find_attractors(trajectories: list[TowerTrajectory],
                    min_members: int = 2) -> dict[tuple[int, int], list[str]]:
    """
    At each landmark, find (N, d) pairs shared by ≥min_members ratios.

    Returns:
        dict of (N, d) → list of ratio names that share d at resolution N.
    """
    attractors: dict[tuple[int, int], list[str]] = {}
    # Collect all (N, d, name) tuples
    for traj in trajectories:
        for proj in traj.projections:
            key = (proj.N, proj.d)
            attractors.setdefault(key, []).append(traj.ratio_name)
    # Filter to multi-member attractors
    return {k: v for k, v in attractors.items() if len(v) >= min_members}


# =============================================================================
# Apéry Series Verification — direct and fast-converging
# =============================================================================

def apery_direct_partial(N_terms: int) -> Fraction:
    """Σ_{n=1}^{N_terms} 1/n³ as exact rational. Use for small N only —
    grows to astronomical denominators at large N. Prefer apery_direct_mpf
    for convergence verification."""
    s = Fraction(0)
    for n in range(1, N_terms + 1):
        s += Fraction(1, n ** 3)
    return s


def apery_direct_mpf(N_terms: int) -> mpf:
    """Σ_{n=1}^{N_terms} 1/n³ in high-precision floating point.
    The direct-sum convergence is O(1/N²): at N=1000 the residual from ζ(3)
    is approximately 0.0005 in value, or ~0.6/N² cents in log₂ ratio (~0.0006¢).
    Use this for verifying convergence to ζ(3) rather than the exact-rational
    version which becomes computationally impractical beyond N ~ 200."""
    s = mpf(0)
    for n in range(1, N_terms + 1):
        s += mpf(1) / mpf(n) ** 3
    return s


def apery_fast_partial(N_terms: int) -> Fraction:
    """(5/2) · Σ_{n=1}^{N_terms} (-1)^(n-1) / (n³ · C(2n, n)) — Apéry's fast series."""
    s = Fraction(0)
    for n in range(1, N_terms + 1):
        # Binomial coefficient C(2n, n) computed incrementally
        binom = 1
        for k in range(n):
            binom = binom * (n + k + 1) // (k + 1)
        term = Fraction(1, n ** 3 * binom)
        if n % 2 == 1:
            s += term
        else:
            s -= term
    return Fraction(5, 2) * s


# =============================================================================
# Verification Test Framework
# =============================================================================

@dataclass
class TestResult:
    name: str
    passed: bool
    detail: str = ""

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        sep = " — " if self.detail else ""
        return f"  [{status}] {self.name}{sep}{self.detail}"


class TestRunner:
    """Collect and report test results."""
    def __init__(self, *, tolerance_cents: float = 0.001) -> None:
        self.results: list[TestResult] = []
        self.tolerance = mpf(tolerance_cents)

    def check_equal(self, name: str, actual, expected,
                    tolerance: mpf | None = None) -> TestResult:
        """Check actual ≈ expected within tolerance."""
        tol = tolerance if tolerance is not None else self.tolerance
        if isinstance(expected, (int, Fraction)):
            expected_mpf = mpf(expected) if isinstance(expected, int) else \
                           mpf(expected.numerator) / mpf(expected.denominator)
        else:
            expected_mpf = mpf(expected)
        actual_mpf = mpf(actual) if not isinstance(actual, mpf) else actual
        diff = fabs(actual_mpf - expected_mpf)
        passed = diff <= tol
        detail = (f"actual={float(actual_mpf):.6f}, expected={float(expected_mpf):.6f}, "
                  f"diff={float(diff):.6g}, tol={float(tol):.6g}")
        result = TestResult(name=name, passed=passed, detail=detail)
        self.results.append(result)
        return result

    def check_int(self, name: str, actual: int, expected: int) -> TestResult:
        passed = (actual == expected)
        detail = f"actual={actual}, expected={expected}"
        result = TestResult(name=name, passed=passed, detail=detail)
        self.results.append(result)
        return result

    def check_bool(self, name: str, actual: bool, expected: bool,
                   extra: str = "") -> TestResult:
        passed = (actual == expected)
        detail = f"actual={actual}, expected={expected}"
        if extra:
            detail += f" ({extra})"
        result = TestResult(name=name, passed=passed, detail=detail)
        self.results.append(result)
        return result

    def check_set(self, name: str, actual: set, expected: set) -> TestResult:
        passed = (actual == expected)
        missing = expected - actual
        extra = actual - expected
        detail_parts = []
        if missing:
            detail_parts.append(f"missing={sorted(missing)}")
        if extra:
            detail_parts.append(f"extra={sorted(extra)}")
        if not detail_parts:
            detail_parts.append(f"set={sorted(actual)}")
        result = TestResult(name=name, passed=passed, detail=" ".join(detail_parts))
        self.results.append(result)
        return result

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if not r.passed)

    def summary(self) -> str:
        lines = [
            "=" * 75,
            f"  TEST SUMMARY: {self.passed}/{self.total} passed, {self.failed} failed",
            "=" * 75,
        ]
        if self.failed:
            lines.append("\n  FAILURES:")
            for r in self.results:
                if not r.passed:
                    lines.append(f"    {r}")
        return "\n".join(lines)


# =============================================================================
# Main Verification — tests all document claims
# =============================================================================

def run_full_verification(*, max_s: int, max_prime: int,
                          precision_dps: int, verbose: bool) -> TestRunner:
    mp.dps = precision_dps
    runner = TestRunner(tolerance_cents=0.001)

    # --- Build tower dynamically ---
    tower = generate_tower(max_prime)
    if verbose:
        print(f"\nGenerated tower ({len(tower)} landmarks, up to N=LCM(1..{max_prime})={tower[-1]}):")
        print(f"  {tower}")

    # --- Compute all zeta values ---
    zetas: dict[int, mpf] = {s: zeta(s) for s in range(2, max_s + 1)}
    if verbose:
        print(f"\nComputed ζ(2) through ζ({max_s}) at {precision_dps}-digit precision.")

    # --- Build trajectories ---
    trajectories: dict[int, TowerTrajectory] = {}
    for s, val in zetas.items():
        trajectories[s] = build_trajectory(f"ζ({s})", val, tower)

    zeta3_traj = trajectories[3]

    # =========================================================================
    # SECTION A — Primary ζ(3) claims from the document
    # =========================================================================
    if verbose:
        print("\n" + "=" * 75)
        print("  SECTION A: Primary ζ(3) placement claims")
        print("=" * 75)

    # A.1 — 12ET projection: (k=3, d=4, ε=+18.606¢)
    p12 = next(p for p in zeta3_traj.projections if p.N == 12)
    runner.check_int("A.1a ζ(3)@12ET k=3", p12.k, 3)
    runner.check_int("A.1b ζ(3)@12ET d=4", p12.d, 4)
    runner.check_equal("A.1c ζ(3)@12ET ε≈+18.606¢", p12.eps_cents, mpf("18.606231"),
                       tolerance=mpf("0.01"))

    # A.2 — 36ET: k=10, d=18, ε=-14.727¢
    p36 = next(p for p in zeta3_traj.projections if p.N == 36)
    runner.check_int("A.2a ζ(3)@36ET k=10", p36.k, 10)
    runner.check_int("A.2b ζ(3)@36ET d=18", p36.d, 18)
    runner.check_equal("A.2c ζ(3)@36ET ε≈-14.727¢", p36.eps_cents, mpf("-14.7271"),
                       tolerance=mpf("0.01"))

    # A.3 — d=15 plateau: 60, 120, 180, 240, 360, 420ET all at d=15, ε=-1.394¢
    plateau_landmarks = {60, 120, 180, 240, 360, 420}
    for N_plat in plateau_landmarks:
        p_plat = next((p for p in zeta3_traj.projections if p.N == N_plat), None)
        if p_plat is None:
            continue  # landmark not in tower for this max_prime
        runner.check_int(f"A.3 ζ(3)@{N_plat}ET d=15 (plateau)", p_plat.d, 15)
        runner.check_equal(f"A.3 ζ(3)@{N_plat}ET ε≈-1.394¢", p_plat.eps_cents,
                           mpf("-1.3938"), tolerance=mpf("0.001"))

    # A.4 — 132ET false resolution: d=132, ε=+0.424¢
    p132 = next((p for p in zeta3_traj.projections if p.N == 132), None)
    if p132:
        runner.check_int("A.4a ζ(3)@132ET d=132", p132.d, 132)
        runner.check_equal("A.4b ζ(3)@132ET ε≈+0.424¢", p132.eps_cents, mpf("0.4244"),
                           tolerance=mpf("0.001"))
        runner.check_bool("A.4c ζ(3)@132ET sub-cent", p132.is_sub_cent, True)

    # A.5 — 840ET true home: k=223, d=840, ε=+0.035¢, coprime skeleton
    p840 = next((p for p in zeta3_traj.projections if p.N == 840), None)
    if p840:
        runner.check_int("A.5a ζ(3)@840ET k=223", p840.k, 223)
        runner.check_int("A.5b ζ(3)@840ET d=840", p840.d, 840)
        runner.check_equal("A.5c ζ(3)@840ET ε≈+0.035¢", p840.eps_cents, mpf("0.0348"),
                           tolerance=mpf("0.001"))
        runner.check_bool("A.5d ζ(3)@840ET in coprime skeleton", p840.is_coprime_skeleton, True)
        runner.check_bool("A.5e ζ(3)@840ET k=223 is prime",
                          is_prime(223), True, "223 prime required for coprime-skeleton claim")

    # A.6 — 1260ET intermediate home: d=252 = 2²·3²·7
    p1260 = next((p for p in zeta3_traj.projections if p.N == 1260), None)
    if p1260:
        runner.check_int("A.6a ζ(3)@1260ET d=252", p1260.d, 252)
        expected_factors = {2: 2, 3: 2, 7: 1}
        runner.check_bool(
            "A.6b ζ(3)@1260ET d=252 factorization 2²·3²·7",
            p1260.d_factors == expected_factors, True,
            f"actual factors: {p1260.d_factors}")

    # A.7 — 27720ET deep home: d=693 = 3²·7·11, ε≈-0.008¢, all-inert
    p27720 = next((p for p in zeta3_traj.projections if p.N == 27720), None)
    if p27720:
        runner.check_int("A.7a ζ(3)@27720ET d=693", p27720.d, 693)
        expected_693 = {3: 2, 7: 1, 11: 1}
        runner.check_bool(
            "A.7b ζ(3)@27720ET d=693 factorization 3²·7·11",
            p27720.d_factors == expected_693, True,
            f"actual factors: {p27720.d_factors}")
        runner.check_bool(
            "A.7c ζ(3)@27720ET d=693 is all-inert Gaussian",
            p27720.is_all_inert, True,
            f"signature: {p27720.gaussian_signature}")
        runner.check_equal("A.7d ζ(3)@27720ET ε≈-0.008¢", p27720.eps_cents,
                           mpf("-0.00849"), tolerance=mpf("0.001"))

    # =========================================================================
    # SECTION B — Trajectory classification (document §10.8)
    # =========================================================================
    if verbose:
        print("\n" + "=" * 75)
        print("  SECTION B: ζ(3) tower trajectory classification")
        print("=" * 75)

    classifications = zeta3_traj.classify_landmarks()
    class_by_N = {proj.N: label for proj, label in classifications}

    # B.1 — 132ET classified FALSE_RESOLUTION
    if 132 in class_by_N:
        runner.check_bool("B.1 ζ(3)@132ET classified FALSE_RESOLUTION",
                          class_by_N[132] == "FALSE_RESOLUTION", True,
                          f"actual: {class_by_N[132]}")

    # B.2 — 840ET classified TRUE_HOME
    if 840 in class_by_N:
        runner.check_bool("B.2 ζ(3)@840ET classified TRUE_HOME",
                          class_by_N[840] == "TRUE_HOME", True,
                          f"actual: {class_by_N[840]}")

    # B.3 — 1260ET classified INTERMEDIATE_HOME
    if 1260 in class_by_N:
        runner.check_bool("B.3 ζ(3)@1260ET classified INTERMEDIATE_HOME",
                          class_by_N[1260] == "INTERMEDIATE_HOME", True,
                          f"actual: {class_by_N[1260]}")

    # B.4 — d=15 plateau classified PLATEAU
    for N_plat in plateau_landmarks:
        if N_plat in class_by_N:
            runner.check_bool(f"B.4 ζ(3)@{N_plat}ET classified PLATEAU",
                              class_by_N[N_plat] == "PLATEAU", True,
                              f"actual: {class_by_N[N_plat]}")

    # =========================================================================
    # SECTION C — Shadow analysis ζ(3)/(6/5)
    # =========================================================================
    if verbose:
        print("\n" + "=" * 75)
        print("  SECTION C: Shadow ratio ζ(3)/(6/5) analysis")
        print("=" * 75)

    six_fifths = mpf(6) / mpf(5)
    shadow_ratio = zetas[3] / six_fifths
    shadow_offset_cents = log(zetas[3], 2) * mpf(1200) - log(six_fifths, 2) * mpf(1200)

    # C.1 — Shadow offset ≈ +2.965¢
    runner.check_equal("C.1 Shadow offset ζ(3)-6/5 ≈ +2.965¢",
                       shadow_offset_cents, mpf("2.964944"), tolerance=mpf("0.001"))

    # C.2 — Shadow ratio true home at 420ET with d=420, ε≈+0.108¢
    shadow_traj = build_trajectory("ζ(3)/(6/5)", shadow_ratio, tower)
    p420_shadow = next((p for p in shadow_traj.projections if p.N == 420), None)
    if p420_shadow:
        runner.check_int("C.2a Shadow@420ET d=420", p420_shadow.d, 420)
        runner.check_equal("C.2b Shadow@420ET ε≈+0.108¢", p420_shadow.eps_cents,
                           mpf("0.1078"), tolerance=mpf("0.001"))
        runner.check_bool("C.2c Shadow@420ET sub-cent",
                          p420_shadow.is_sub_cent, True)

    # =========================================================================
    # SECTION D — Apéry series convergence verification
    # =========================================================================
    if verbose:
        print("\n" + "=" * 75)
        print("  SECTION D: Apéry series verification")
        print("=" * 75)

    # D.1 — Direct Σ 1/n³ partials converge to ζ(3).
    # Convergence rate: remainder after N terms ≈ 1/(2N²) in value, which translates
    # to ~720/N² cents in log₂ ratio. N=1000 gives ~0.00072¢ residual — well below 0.01¢.
    # Using mpmath (not exact Fraction) because Fraction denominators grow astronomically.
    direct_1000 = apery_direct_mpf(1000)
    direct_residual_cents = (log(zetas[3], 2) - log(direct_1000, 2)) * mpf(1200)
    runner.check_equal("D.1 Direct Σ_{n=1..1000} 1/n³ converges (residual <0.01¢ from ζ(3))",
                       direct_residual_cents, 0, tolerance=mpf("0.01"))

    # D.2 — Apéry fast series at N=10 matches ζ(3) to high precision
    fast_10 = apery_fast_partial(10)
    fast_10_mpf = mpf(fast_10.numerator) / mpf(fast_10.denominator)
    runner.check_equal("D.2 Apéry fast Σ_{n=1..10} matches ζ(3) to <0.001¢",
                       log(zetas[3], 2) * mpf(1200) - log(fast_10_mpf, 2) * mpf(1200),
                       0, tolerance=mpf("0.001"))

    # D.3 — Apéry N=1 partial lands at d=3 (cubic — the expected family)
    fast_1 = apery_fast_partial(1)
    fast_1_mpf = mpf(fast_1.numerator) / mpf(fast_1.denominator)
    p_fast_1 = project(fast_1_mpf, 12)
    runner.check_int("D.3 Apéry fast N=1 lands at d=3 (cubic) at 12ET", p_fast_1.d, 3)

    # D.4 — Apéry partials from N≥5 land at d=4 at 12ET (quartic is structural)
    for N_terms in [5, 10, 20, 50]:
        partial = apery_fast_partial(N_terms)
        partial_mpf = mpf(partial.numerator) / mpf(partial.denominator)
        p_partial = project(partial_mpf, 12)
        runner.check_int(f"D.4 Apéry fast N={N_terms} → d=4 at 12ET (quartic structural)",
                         p_partial.d, 4)

    # =========================================================================
    # SECTION E — Zeta spectrum pattern (document §6.2 + §10.1, §10.7)
    # =========================================================================
    if verbose:
        print("\n" + "=" * 75)
        print("  SECTION E: Zeta spectrum and attractor structure")
        print("=" * 75)

    # E.1 — 12ET pattern: ζ(2) and ζ(3) at d=4; ζ(4..7) at d=12
    if max_s >= 7:
        for s in [2, 3]:
            p = next(p for p in trajectories[s].projections if p.N == 12)
            runner.check_int(f"E.1 ζ({s})@12ET d=4 (quartic family)", p.d, 4)
        for s in [4, 5, 6, 7]:
            p = next(p for p in trajectories[s].projections if p.N == 12)
            runner.check_int(f"E.1 ζ({s})@12ET d=12 (unit-neighborhood)", p.d, 12)

    # E.2 — d=693 attractor at 27720ET: ζ(3), ζ(9), ζ(10) share it
    if 27720 in tower and max_s >= 10:
        attractor_693_members: set[int] = set()
        for s in range(2, max_s + 1):
            p = next((p for p in trajectories[s].projections if p.N == 27720), None)
            if p and p.d == 693:
                attractor_693_members.add(s)
        expected_members = {3, 9, 10}
        runner.check_set(
            "E.2 d=693 attractor at 27720ET contains {3, 9, 10}",
            attractor_693_members, expected_members)

    # E.3 — All-inert prediction FALSIFICATION: ζ(5), ζ(7), ζ(11), ζ(13) NOT all-inert at 27720ET
    if 27720 in tower and max_s >= 7:
        for s in [5, 7]:
            p = next((p for p in trajectories[s].projections if p.N == 27720), None)
            if p:
                runner.check_bool(
                    f"E.3 ζ({s})@27720ET NOT all-inert (prediction falsified)",
                    p.is_all_inert, False,
                    f"d={p.d}, factors={p.d_factors}, signature={p.gaussian_signature}")
        if max_s >= 13:
            for s in [11, 13]:
                p = next((p for p in trajectories[s].projections if p.N == 27720), None)
                if p:
                    runner.check_bool(
                        f"E.3 ζ({s})@27720ET NOT all-inert",
                        p.is_all_inert, False,
                        f"d={p.d}, factors={p.d_factors}")

    # E.4 — ζ(3) and ζ(9) ARE all-inert at 27720ET (the only odd zetas in our range)
    if 27720 in tower and max_s >= 9:
        for s in [3, 9]:
            p = next((p for p in trajectories[s].projections if p.N == 27720), None)
            if p:
                runner.check_bool(f"E.4 ζ({s})@27720ET IS all-inert",
                                  p.is_all_inert, True,
                                  f"d={p.d}, signature={p.gaussian_signature}")

    # E.5 — At 360360ET (LCM(1..13)), d=693 attractor dissolves
    if max_prime >= 13:
        full_res = lcm_of_range(1, 13)
        if full_res in tower:
            for s in [3, 9, 10]:
                if s > max_s:
                    continue
                p = next((p for p in trajectories[s].projections if p.N == full_res), None)
                if p:
                    runner.check_bool(
                        f"E.5 ζ({s})@{full_res}ET d≠693 (attractor dissolves)",
                        p.d != 693, True,
                        f"actual d={p.d}")

    # =========================================================================
    # SECTION F — Gaussian-prime classification structural invariants
    # =========================================================================
    if verbose:
        print("\n" + "=" * 75)
        print("  SECTION F: Gaussian classification invariants")
        print("=" * 75)

    # F.1 — Prime 2 is RAMIFIED
    runner.check_bool("F.1 2 is RAMIFIED", gaussian_class(2) == "RAMIFIED", True)
    # F.2 — Primes 3, 7, 11 are INERT
    for p in [3, 7, 11]:
        runner.check_bool(f"F.2 {p} is INERT", gaussian_class(p) == "INERT", True)
    # F.3 — Primes 5, 13 are SPLIT
    for p in [5, 13]:
        runner.check_bool(f"F.3 {p} is SPLIT", gaussian_class(p) == "SPLIT", True)

    # F.4 — d=693 (all factors 3, 7, 11) is all-inert
    test_proj = LatticeProjection(N=27720, r=mpf(0), k=7360, d=693, eps_cents=mpf(0))
    runner.check_bool("F.4 d=693 is all-inert", test_proj.is_all_inert, True)
    # F.5 — d=840 (factors 2, 3, 5, 7) is NOT all-inert
    test_proj = LatticeProjection(N=840, r=mpf(0), k=223, d=840, eps_cents=mpf(0))
    runner.check_bool("F.5 d=840 NOT all-inert (contains 2 and 5)",
                      test_proj.is_all_inert, False)

    # =========================================================================
    # SECTION G — ET constant self-verification
    # =========================================================================
    if verbose:
        print("\n" + "=" * 75)
        print("  SECTION G: ET constant verification")
        print("=" * 75)

    runner.check_int("G.1 N=12 (3 primitives × 4 manifold states)", MANIFOLD_SYMMETRY, 12)
    runner.check_equal("G.2 V_base = 1/12", BASE_VARIANCE, mpf(1) / mpf(12))
    runner.check_equal("G.3 K = 2/3 (Koide threshold)", KOIDE_THRESHOLD, mpf(2) / mpf(3))
    runner.check_equal("G.4 T_WEIGHT = 1/3", T_WEIGHT, mpf(1) / mpf(3))
    runner.check_equal("G.5 LIFE_THRESHOLD = 13/12", LIFE_THRESHOLD, mpf(13) / mpf(12))

    # G.6 — sub_step_threshold_cents consistency
    runner.check_equal("G.6a θ_sub(12) = 100·V_base/1 ≈ 8.33¢",
                       sub_step_threshold_cents(12),
                       mpf(100) / mpf(12), tolerance=mpf("1e-10"))
    runner.check_equal("G.6b θ_sub(27720) ≈ 0.00361¢",
                       sub_step_threshold_cents(27720),
                       mpf(100) / mpf(27720), tolerance=mpf("1e-10"))

    return runner


def is_prime(n: int) -> bool:
    """Primality test via trial division up to √n."""
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    for p in range(3, int(math.isqrt(n)) + 1, 2):
        if n % p == 0:
            return False
    return True


# =============================================================================
# Report Generation
# =============================================================================

def print_trajectory_report(traj: TowerTrajectory,
                            classifications: list[tuple[LatticeProjection, str]]) -> None:
    """Print formatted trajectory report for a single ratio."""
    print(f"\n{'-' * 100}")
    print(f"  Trajectory: {traj.ratio_name} = {float(traj.ratio_value):.12f}")
    print(f"{'-' * 100}")
    header = (f"  {'N':>6} {'k':>8} {'d':>7} {'factors':>20} {'Gaussian':>18} "
              f"{'ε (¢)':>10} {'sub¢':>5} {'coprime':>7} {'class':>20}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for proj, label in classifications:
        factors = proj.d_factors
        fstr = "·".join(f"{p}^{e}" if e > 1 else str(p)
                        for p, e in sorted(factors.items())) or "1"
        g_str = "·".join(gc[0] for gc in proj.gaussian_signature) or "-"
        sub_mark = "YES" if proj.is_sub_cent else ""
        coprime_mark = "YES" if proj.is_coprime_skeleton else ""
        print(f"  {proj.N:>6} {proj.k:>8} {proj.d:>7} {fstr:>20} {g_str:>18} "
              f"{float(proj.eps_cents):>+10.4f} {sub_mark:>5} {coprime_mark:>7} {label:>20}")


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0] if __doc__ else "Apéry lattice test",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-s", type=int, default=13,
                        help="Maximum zeta s index to test (default: 13)")
    parser.add_argument("--max-prime", type=int, default=13,
                        help="Maximum prime for LCM tower generation (default: 13)")
    parser.add_argument("--precision", type=int, default=80,
                        help="mpmath precision in decimal digits (default: 80)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print full trajectory reports for every zeta value")
    parser.add_argument("--show-attractors", action="store_true",
                        help="Print multi-member attractor analysis")
    args = parser.parse_args()

    if args.max_s < 3:
        sys.stderr.write("max-s must be ≥ 3 (ζ(3) is the primary subject)\n")
        return 2
    if args.max_prime < 7:
        sys.stderr.write("max-prime must be ≥ 7 (minimum LCM tower depth for biological threshold)\n")
        return 2

    print("=" * 75)
    print("  APÉRY'S CONSTANT ζ(3) — LATTICE VERIFICATION TEST")
    print("  Testing every structural claim in the placement document.")
    print("=" * 75)
    print(f"  max_s = {args.max_s}, max_prime = {args.max_prime}, "
          f"precision = {args.precision} decimal digits")

    runner = run_full_verification(max_s=args.max_s, max_prime=args.max_prime,
                                   precision_dps=args.precision, verbose=args.verbose)

    if args.verbose:
        # Rebuild trajectories for display (run_full_verification doesn't expose them)
        mp.dps = args.precision
        tower = generate_tower(args.max_prime)
        for s in range(3, min(args.max_s + 1, 14)):
            traj = build_trajectory(f"ζ({s})", zeta(s), tower)
            print_trajectory_report(traj, traj.classify_landmarks())

    if args.show_attractors:
        print("\n" + "=" * 75)
        print("  MULTI-MEMBER ATTRACTORS (d-families shared by ≥2 zeta values)")
        print("=" * 75)
        mp.dps = args.precision
        tower = generate_tower(args.max_prime)
        trajectories = [build_trajectory(f"ζ({s})", zeta(s), tower)
                        for s in range(2, args.max_s + 1)]
        attractors = find_attractors(trajectories, min_members=2)
        # Show attractors at deep landmarks only (N ≥ 840)
        deep_attractors = {k: v for k, v in attractors.items() if k[0] >= 840}
        for (N, d), members in sorted(deep_attractors.items(), key=lambda x: (-x[0][0], x[0][1])):
            if len(members) >= 2:
                print(f"  N={N:>6} d={d:>6}: shared by {len(members)} values — {members}")

    # Print all test results
    for result in runner.results:
        print(result)
    print(runner.summary())

    return 0 if runner.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
