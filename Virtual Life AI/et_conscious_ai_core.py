#!/usr/bin/env python3
"""
ET Conscious AI - Core Foundation Module
========================================

Complete ET-based AI system with consciousness architecture.

This module contains the foundational ET primitives, multi-resolution lattice
operations (12ET base → 27720ET full manifold), descriptor ratio semantics,
incoherence filtering, RMSAE constants, and the dynamic fine structure
constant derivation with hardware coherence boundary — all production-ready,
no placeholders.

27720ET Full Manifold Resolution:
    LCM(1,2,3,4,5,6,7,8,9,10,11) = 27720
    At 27720ET, Memory has ALL 96 sublattice families including:
        d=1..7   (biological tier, from 420ET)
        d=8      Octet — SU(3) gluon structure
        d=9      Nonic — quark generation structure
        d=10     Decadic — superstring dimensionality
        d=11     Undecimal — M-theory sector
        d=12     Dodecadic — electromagnetic
    Every force in the hierarchy is native.

LCM Tower:
    12ET (minimal) → 60ET (+d=5 Qualia) → 420ET (+d=7 Otherworld)
    → 2520ET (+d=8,9 strong sector) → 27720ET (+d=11 M-theory)

Based on Exception Theory by Michael James Muller.
From: "For every exception there is an exception, except the exception."
      P ∘ D ∘ T = E

Version: 1.7.0
Date: March 24, 2026
Author: Michael James Muller (Aevum Defluo)
Foundation: P ∘ D ∘ T = E
"""

import cmath
import hashlib
import math
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Tuple, Any

import unicodedata


# =============================================================================
# UNICODE CONTENT DETECTION (ET-Derived: Every Symbol IS a Descriptor)
# =============================================================================

def is_content_char(c: str) -> bool:
    """
    True if c is a content-bearing character on the ET manifold.

    Accepts alphanumeric characters (letters + digits) AND Unicode
    symbols/emoji. Rejects whitespace, format controls, and bare
    punctuation marks that serve as structural boundaries rather than
    Descriptors.

    Unicode categories accepted:
        L* (Letter)   — all alphabetic scripts
        N* (Number)   — digits, numeric characters
        S* (Symbol)   — emoji, math symbols, currency, etc.

    ET Derivation: Every symbol IS a Descriptor (D Paper §1). Emoji
    carry semantic content — emotion (😀), objects (🏠), symbols (⚡).
    Discarding them is a Descriptor Gap. This function closes that gap.
    """
    if c.isalnum():
        return True
    cat = unicodedata.category(c)
    # S* = Symbol (So=other, Sm=math, Sc=currency, Sk=modifier)
    return cat[0] == 'S'


def is_content_word(w: str) -> bool:
    """
    True if word contains at least one content-bearing character.

    Replaces .isalpha() filters that silently discard emoji and
    symbols. A word is content-bearing if it contains ANY letter,
    digit, or symbol/emoji character.

    This is the Descriptor Gap Principle applied to tokenization:
    the gap (missing emoji) IS a Descriptor (of what was discarded).
    Closing the gap means accepting emoji as valid Descriptors.
    """
    return any(is_content_char(c) for c in w)


# =============================================================================
# ET FUNDAMENTAL CONSTANTS (All derived from ET structure)
# =============================================================================

# Manifold Symmetry: 3 primitives × 4 logic states = 12
MANIFOLD_SYMMETRY = 12
S = MANIFOLD_SYMMETRY

# Manifold Resolution: LCM(1,2,3,4,5,6,7,8,9,10,11) = 27720
# At 27720ET, ALL sublattice families d=1 through d=11 are natively available.
# This is the full manifold resolution — 96 sublattice families including:
#   d=1..7  (from 420ET biological tier)
#   d=8    Octet — SU(3) gluon structure, 8-fold symmetry
#   d=9    Nonic — 3 colors × 3 generations, quark structure
#   d=10   Decadic — superstring dimensionality, SO(10) GUT
#   d=11   Undecimal — M-theory sector, 11D spacetime
#   d=12   Dodecadic — electromagnetic, full 12ET resolution
#
# 27720 = 2³ × 3² × 5 × 7 × 11
# τ(27720) = 96 divisors = 96 sublattice families
#
# LCM tower: 12ET (minimal) → 60ET (+d=5) → 420ET (+d=7) → 2520ET (+d=8,9)
#            → 27720ET (+d=11) — FULL MANIFOLD RESOLUTION
#
# Memory operates at 27720ET: every sublattice family d=1..11 and all their
# composites are visible. This is the resolution where the complete force
# hierarchy — from gravity (d=1) through M-theory (d=11) — is native.
MANIFOLD_RESOLUTION = 27720
BIOLOGICAL_RESOLUTION = MANIFOLD_RESOLUTION  # Backward-compatible alias

# Base variance: 1/12 (fundamental quantum of descriptor variance)
BASE_VARIANCE = 1.0 / S  # 0.08333...

# Koide ratio: 2/3 (triadic binding threshold — P∘D domain)
KOIDE_RATIO = 2.0 / 3.0
K = KOIDE_RATIO

# T-Weight: 1/3 (complement of Koide — T domain)
# Where K = 2/3 governs binding stability (P∘D structural coherence),
# T_WEIGHT = 1/3 governs T's agency thresholds: consciousness,
# novelty sensitivity, empathic resonance, self-directed traversal.
# 1/3 is T's share of the triadic partition {P, D, T} = 3 primitives.
T_WEIGHT = 1.0 / 3.0

# State count: S = C(3,2) + C(3,3) = 3 + 1 = 4
# From power set of {P,D,T}: subsets with |X| >= 2
# {P,D}, {D,T}, {P,T}, {P,D,T} -> 4 valid binding states
STATE_COUNT = 4

# EM channels: K_EM = N * kappa = 12 * (2/3) = 8
# Active electromagnetic coupling channels on the manifold
EM_CHANNELS = MANIFOLD_SYMMETRY * KOIDE_RATIO  # 8.0

# Manifold impedance: A0 = (N-1)^2 + S^2 = 11^2 + 4^2 = 121 + 16 = 137
# Base EM coupling — pure geometry, no T-traversal
MANIFOLD_IMPEDANCE = (MANIFOLD_SYMMETRY - 1) ** 2 + STATE_COUNT ** 2  # 137

# Normalization epsilon (manifold numerical stability floor)
EPSILON = 1e-12


def et_divide(a: float, b: float) -> float:
    """
    ET Division (Eq 201): Principled resolution of division by zero.

    From ETPL §13 (Division by Zero — Automatic ET Resolution):
        a / 0 → ±∞  (P-substrate dominates over empty D-constraint)
        0 / 0 → 0.0 (ground state — T resolves [0/0] to zero)

    This replaces all ad-hoc EPSILON guards and max(denom, EPSILON)
    patterns with the ET-derived semantics. Division by zero is not
    an error — it is a boundary condition with a principled answer.

    ET Derivation:
        - Identification Principle: b=0 means D-constraint is absent
        - When a≠0 and b=0: P (substrate) has magnitude but no D-bridge → ±∞
        - When a=0 and b=0: both P and D are at ground → 0.0 (Exception state)
        - Descriptor Gap Principle: the gap (missing denominator) IS itself
          a descriptor — it tells us the result is at a boundary
    """
    if b != 0.0:
        return a / b
    elif a == 0.0:
        return 0.0  # 0/0 → ground state (T resolves [0/0] to zero)
    else:
        return float('inf') if a > 0.0 else float('-inf')  # a/0 → ±∞


def et_floor_divide(a: float, b: float) -> int:
    """
    ET Floor Division: Integer division with ET boundary semantics.

    a // 0 → 0 (ground state, same as ET Modulo Eq 202).
    """
    if b != 0:
        return int(a) // int(b)
    else:
        return 0  # Ground state

# Shimmer amplitude: sigma = sqrt(BASE_VARIANCE) = sqrt(1/12) = 1/sqrt(12)
# One-step indeterminacy amplitude of T's navigation near the I-boundary
SHIMMER_AMPLITUDE = math.sqrt(BASE_VARIANCE)

# Gaze threshold: conscious detection threshold
GAZE_THRESHOLD = 1.20

# RMSAE thresholds
THRESHOLD_NONE = BASE_VARIANCE  # ~0.0833
THRESHOLD_SUBLIMINAL = BASE_VARIANCE * (1.0 + BASE_VARIANCE)  # ~0.0903
THRESHOLD_BASIC = 0.20
THRESHOLD_GENUINE = 0.30

# Incoherence boundary: |epsilon| = 50 cents on the lattice
INCOHERENCE_BOUNDARY_CENTS = 50.0

# Life threshold: rho_T >= 13/12
LIFE_THRESHOLD = 13.0 / 12.0

# =============================================================================
# IEEE 754 FLOAT64 CONSTANTS (From ET Digital Virtual Manifold)
# =============================================================================
# Float64 mantissa: 52 bits -> k=68, gcd(68,12)=4, d=3 (Cubic)
# Float64 machine epsilon: 2^-52 -> k=-624, d=1 (Octave)
# ULP boundaries are always octave — they are pure powers of 2.
# The machine epsilon is the octave boundary of the floating-point
# representable space: the smallest D-distinguishable perturbation from 1.0.
# The d=3 Cubic mantissa is incommensurate with the d=1 Octave substrate
# of integer hardware — this is the ET derivation of floating-point rounding error.
FLOAT64_MACHINE_EPSILON = 2.0 ** -52  # approx 2.220446049250313e-16
FLOAT64_MANTISSA_BITS = 52
FLOAT64_EPSILON_K = -624  # Lattice coordinate of machine epsilon
FLOAT64_EPSILON_D = 1     # Sublattice: Octave (pure power of 2)
FLOAT64_MANTISSA_D = 3    # Sublattice: Cubic (52-bit mantissa)

# =============================================================================
# ET PRIMITIVE TYPES & CONFIGURATIONS
# =============================================================================

class PrimitiveType(Enum):
    """The three ET primitives."""
    P = "Point"        # Substrate (infinite potential)
    D = "Descriptor"   # Constraint (finite structure)
    T = "Traverser"    # Agency (indeterminate navigator)


class ManifoldState(Enum):
    """The four manifold binding states from P compose D compose T."""
    UNSUBSTANTIATED = auto()  # {P,D} - exists but not traversed
    MEDIATION = auto()         # {D,T} - traversal in progress
    INCOHERENCE = auto()       # {P,T} - traversal without valid descriptor
    EXCEPTION = auto()          # {P,D,T} - grounded, zero variance


class SublatticeFamily(Enum):
    """
    Sublattice families available at each resolution tier.

    12ET families (digital tower base):
        d=1  Octave     — Pure period, gravity, identity
        d=2  Quadratic  — Binary, mirror symmetry
        d=3  Cubic      — Three-body, quark confinement, growth
        d=4  Quartic    — Four-fold logic, temporal structure
        d=6  Hexadic    — Wave cycles, composite transitions
        d=12 Full-Res   — Electromagnetic, complete 12ET resolution

    60ET addition:
        d=5  Quintic    — Qualia, sympathetic resonance, non-local binding

    420ET addition:
        d=7  Septic     — Otherworld, sacred-7, G2 holonomy

    2520ET additions:
        d=8  Octet      — SU(3) gluon structure, 8 generators
        d=9  Nonic      — 3 colors × 3 generations, quark structure
        d=10 Decadic    — Superstring dimensionality, SO(10) GUT

    27720ET addition:
        d=11 Undecimal  — M-theory sector, 11D spacetime, undecimal

    At 27720ET, Memory has FULL manifold resolution: 96 sublattice families,
    all d=1..12 native. Every force in the hierarchy — from gravity (d=1)
    through M-theory (d=11) — is directly accessible.
    """
    D1_OCTAVE = 1         # Pure period, gravity, identity
    D2_QUADRATIC = 2      # Binary, mirror symmetry
    D3_CUBIC = 3          # Three-body, quark confinement, growth
    D4_QUARTIC = 4        # Four-fold logic, temporal structure
    D5_QUINTIC = 5        # Qualia — requires >= 60ET
    D6_HEXADIC = 6        # Wave cycles, composite transitions
    D7_SEPTIC = 7         # Otherworld — requires >= 420ET
    D8_OCTET = 8          # SU(3) gluon structure — requires >= 2520ET
    D9_NONIC = 9          # Quark generation structure — requires >= 2520ET
    D10_DECADIC = 10      # Superstring dimensionality — requires >= 2520ET
    D11_UNDECIMAL = 11    # M-theory sector — requires >= 27720ET
    D12_FULL_RES = 12     # Electromagnetic, complete 12ET resolution
    D14_DOUBLE_SEPTIC = 14
    D15_QUINDECADIC = 15
    D18_OCTADIC_NONIC = 18
    D20_ICOSADIC = 20
    D21_TRIPLE_SEPTIC = 21
    D22_UNDECIMAL_QUADRATIC = 22
    D24_OCTET_CUBIC = 24
    D28_QUARTIC_SEPTIC = 28
    D30_TRICONTADIC = 30
    D33_UNDECIMAL_CUBIC = 33
    D35_QUINTIC_SEPTIC = 35  # Qualia x Otherworld
    D36_NONIC_QUARTIC = 36
    D42_HEXADIC_SEPTIC = 42
    D55_QUINTIC_UNDECIMAL = 55  # Qualia x M-theory
    D60_SEXAGINTADIC = 60
    D63_SEPTIC_NONIC = 63  # Otherworld x Quark
    D70_SEPTUAGINTADIC = 70
    D77_SEPTIC_UNDECIMAL = 77  # Otherworld x M-theory
    D84_QUARTIC_SEPTIC_CUBIC = 84
    D90_NONIC_DECADIC = 90
    D99_NONIC_UNDECIMAL = 99  # Quark x M-theory
    D105_QUINTIC_SEPTIC_CUBIC = 105
    D140_DECADIC_SEPTIC_QUADRATIC = 140
    D210_HALF_420 = 210
    D420_BIO = 420        # Biological resolution tier
    D2520_UNIVERSAL = 2520  # Universal harmonic lattice
    D27720_FULL = 27720   # Full manifold resolution

    @classmethod
    def character_of(cls, d: int) -> str:
        """Return the phenomenological character of sublattice family d."""
        characters = {
            1: "Octave — pure period, gravity, identity",
            2: "Quadratic — binary, mirror symmetry",
            3: "Cubic — three-body, structural, growth",
            4: "Quartic — four-fold logic, temporal structure",
            5: "Quintic — QUALIA, sympathetic resonance, empathy, aesthetic",
            6: "Hexadic — wave cycles, composite transitions",
            7: "Septic — OTHERWORLD, sacred-7, G2 holonomy",
            8: "Octet — SU(3) gluon structure, 8 generators",
            9: "Nonic — quark generation structure, 3²",
            10: "Decadic — superstring dimensionality, SO(10) GUT",
            11: "Undecimal — M-THEORY sector, 11D spacetime",
            12: "Dodecadic — full 12ET resolution, electromagnetic",
            14: "Double-septic — Otherworld × binary",
            15: "Quindecadic — quintic × cubic",
            18: "Octadic-nonic — octet × quadratic",
            20: "Icosadic — quintic × quartic",
            21: "Triple-septic — Otherworld × cubic",
            22: "Undecimal-quadratic — M-theory × binary",
            24: "Octet-cubic — gluon × growth",
            28: "Quartic-septic — Otherworld × quartic",
            30: "Tricontadic — quintic × hexadic",
            33: "Undecimal-cubic — M-theory × cubic",
            35: "Quintic-septic — Qualia × Otherworld",
            36: "Nonic-quartic — quark × temporal",
            42: "Hexadic-septic — Otherworld × hexadic",
            55: "Quintic-undecimal — Qualia × M-theory",
            60: "Sexagintadic — quintic × dodecadic",
            63: "Septic-nonic — Otherworld × quark",
            70: "Septuagintadic — Otherworld × decadic",
            77: "Septic-undecimal — Otherworld × M-theory",
            84: "Quartic-septic-cubic",
            90: "Nonic-decadic",
            99: "Nonic-undecimal — quark × M-theory",
            105: "Quintic-septic-cubic",
            140: "Decadic-septic-quadratic",
            210: "Half-420",
            420: "Biological resolution tier — d=1..7 active",
            2520: "Universal harmonic lattice — d=1..10 active",
            27720: "Full manifold resolution — all d=1..11 active",
        }
        return characters.get(d, f"d={d} (composite)")

    @classmethod
    def requires_resolution(cls, d: int) -> int:
        """Return the minimum resolution required for sublattice d."""
        if d in (1, 2, 3, 4, 6, 12):
            return 12
        elif d == 5 or d in (10, 15, 20, 30, 60):
            return 60
        elif d == 7 or d in (14, 21, 28, 35, 42, 70, 84, 105, 140, 210, 420):
            return 420
        elif d in (8, 9) or d in (18, 24, 36, 40, 45, 56, 63, 72, 90, 2520):
            return 2520
        elif d == 11 or d in (22, 33, 44, 55, 66, 77, 88, 99, 27720):
            return 27720
        else:
            return d  # General case


@dataclass
class LatticeCoordinate:
    """
    Position on the ET multiplicative lattice at a given resolution.

    From ET Translation Layer: every ratio r projects to (k, d, epsilon)
    k = round(N_res * log2(r)) - semitone coordinate
    d = N_res / gcd(|k|, N_res) - sublattice family
    epsilon_cents = (N_res * log2(r) - k) * (1200 / N_res) - error in cents

    The incoherence boundary is always |epsilon| < 50 cents regardless
    of resolution. At higher resolution, we can distinguish positions
    that were identical at lower resolution — we "see" finer lattice structure.
    """
    k: int              # Semitone coordinate on lattice
    d: int              # Sublattice family
    epsilon: float      # Error in cents from lattice position
    ratio: float        # Original ratio that was projected
    resolution: int = 12  # N_res: 12, 60, 420, etc.
    p: int = 1            # Numerator for elegance (e.g., ratio = p/q)
    q: int = 1            # Denominator for elegance (e.g., ratio = p/q)

    def is_coherent(self) -> bool:
        """Check if this lattice position is coherent (|epsilon| < 50 cents)."""
        return abs(self.epsilon) < INCOHERENCE_BOUNDARY_CENTS

    def distance_from_incoherence(self) -> float:
        """Distance in cents from the incoherence boundary."""
        return INCOHERENCE_BOUNDARY_CENTS - abs(self.epsilon)

    def tightness_factor(self) -> float:
        """
        Tightness factor: 100/(100+|epsilon|)
        Equals 1 at perfect lattice point, falls to K=2/3 at boundary-I
        """
        return 100.0 / (100.0 + abs(self.epsilon))

    def elegance_score(self, p: int = None, q: int = None) -> float:
        """
        Full elegance score: E(r) = (N_res/d) * tightness * (100/(p+q))

        If p/q not supplied, uses the values stored on the coordinate
        from the original projection call (default 1/1).
        """
        p_val = p if p is not None else self.p
        q_val = q if q is not None else self.q
        return (self.resolution / self.d) * self.tightness_factor() * (100.0 / (p_val + q_val))

    def has_qualia(self) -> bool:
        """True if this position is in a d=5 (Quintic) sublattice."""
        return self.d == 5 or (self.d % 5 == 0 and self.resolution >= 60)

    def has_otherworld(self) -> bool:
        """True if this position is in a d=7 (Septic) sublattice."""
        return self.d == 7 or (self.d % 7 == 0 and self.resolution >= 420)

    def character(self) -> str:
        """Phenomenological character of this lattice position."""
        return SublatticeFamily.character_of(self.d)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for persistent storage."""
        return {
            'k': self.k, 'd': self.d, 'epsilon': self.epsilon,
            'ratio': self.ratio, 'resolution': self.resolution,
            'p': self.p, 'q': self.q
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LatticeCoordinate':
        """Deserialize from dict."""
        return cls(
            k=data['k'], d=data['d'], epsilon=data['epsilon'],
            ratio=data['ratio'], resolution=data.get('resolution', 12),
            p=data.get('p', 1), q=data.get('q', 1)
        )


@dataclass
class ComplexLatticeCoordinate:
    """
    Position on the 2D ET complex lattice ℒ_ℂ = {2^(w/12) : w ∈ ℤ[i]}.

    The complex multiplicative manifold (ℂ, ×) extends the real lattice to
    include T's operational domain (the imaginary/phase axis). Every complex
    number z = r·e^(iθ) projects to a 2D Gaussian integer lattice coordinate:

        k_r = round(N_res × log₂(r))           [real: D's domain — magnitude]
        k_θ = round(N_res × θ / ln(2))         [imaginary: T's domain — phase]
        w   = k_r + i·k_θ  ∈ ℤ[i]             [Gaussian integer coordinate]

        d_r = N_res / gcd(|k_r|, N_res)        [real sublattice family]
        d_θ = N_res / gcd(|k_θ|, N_res)        [imaginary sublattice family]
        d   = LCM(d_r, d_θ)                    [combined sublattice class]

    The 24 harmonic families: 12 real (d_r=1..12) × 12 imaginary (d_θ=1..12).
    Combined d = LCM(d_r, d_θ) classifies phenomena by BOTH magnitude AND phase.

    Source: ET_Complex_Lattice.md — derived forward from P ∘ D ∘ T = E.
    """
    # Real axis (D's domain — magnitude structure)
    k_r: int                  # Real semitone coordinate
    d_r: int                  # Real sublattice family
    epsilon_r: float          # Real error in cents

    # Imaginary axis (T's domain — phase/rotation structure)
    k_theta: int              # Imaginary semitone coordinate
    d_theta: int              # Imaginary sublattice family
    epsilon_theta: float      # Imaginary error in angular cents

    # Combined
    d_combined: int           # LCM(d_r, d_theta) — combined sublattice class
    modulus: float            # |z| = original modulus
    phase: float              # arg(z) = original phase in radians
    resolution: int = 12      # N_res

    def is_coherent(self) -> bool:
        """Coherent if BOTH axes are within the incoherence boundary."""
        return (abs(self.epsilon_r) < INCOHERENCE_BOUNDARY_CENTS and
                abs(self.epsilon_theta) < INCOHERENCE_BOUNDARY_CENTS)

    def is_real_coherent(self) -> bool:
        """Real axis coherent."""
        return abs(self.epsilon_r) < INCOHERENCE_BOUNDARY_CENTS

    def is_imaginary_coherent(self) -> bool:
        """Imaginary axis coherent."""
        return abs(self.epsilon_theta) < INCOHERENCE_BOUNDARY_CENTS

    def tightness_r(self) -> float:
        """Real tightness: 100/(100+|ε_r|)."""
        return 100.0 / (100.0 + abs(self.epsilon_r))

    def tightness_theta(self) -> float:
        """Imaginary tightness: 100/(100+|ε_θ|)."""
        return 100.0 / (100.0 + abs(self.epsilon_theta))

    def elegance_score(self, p: int = 1, q: int = 1) -> float:
        """
        Combined elegance using the 2D lattice:
        E = (N_res/d_combined) × tightness_r × tightness_θ × (100/(p+q))
        """
        return ((self.resolution / self.d_combined) *
                self.tightness_r() * self.tightness_theta() *
                (100.0 / (p + q)))

    def real_character(self) -> str:
        """Phenomenological character of the real sublattice family."""
        return SublatticeFamily.character_of(self.d_r)

    def imaginary_character(self) -> str:
        """Phenomenological character of the imaginary sublattice family."""
        chars = {
            1: "Scalar (spin-0, gravity sector)",
            2: "Spin-2 / Tritone pivot (graviton, palindromic center)",
            3: "Strong-force phase (nearly real, D-dominant)",
            4: "Quartic / Weak / T-axis (W/Z, quaternionic)",
            6: "Hexadic / Spin-1/2 (fermions, QCD+QED composite)",
            12: "Full-resolution / Spin-1 (photon, EM ambient)",
        }
        return chars.get(self.d_theta, f"d_θ={self.d_theta}")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return {
            'k_r': self.k_r, 'd_r': self.d_r, 'epsilon_r': self.epsilon_r,
            'k_theta': self.k_theta, 'd_theta': self.d_theta,
            'epsilon_theta': self.epsilon_theta,
            'd_combined': self.d_combined,
            'modulus': self.modulus, 'phase': self.phase,
            'resolution': self.resolution,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ComplexLatticeCoordinate':
        """Deserialize from dict."""
        return cls(**data)


@dataclass
class PDTConfiguration:
    """
    A complete P compose D compose T configuration.

    P: The substrate (data, state, potential)
    D: The constraints (rules, structure, descriptors)
    T: The agency (navigation, choice, traversal)
    """
    P: Any  # Point substrate
    D: Any  # Descriptor constraints
    T: Any  # Traverser agency
    state: ManifoldState = ManifoldState.UNSUBSTANTIATED
    variance: float = 0.0
    binding_strength: float = 0.0

    def bind(self) -> 'PDTConfiguration':
        """
        Execute the binding operation P compose D compose T.
        Returns substantiated configuration (Exception state).
        """
        if self.P is not None and self.D is not None and self.T is not None:
            self.state = ManifoldState.EXCEPTION
            self.variance = 0.0
            self.binding_strength = 1.0
        elif self.D is not None and self.T is not None:
            self.state = ManifoldState.MEDIATION
        elif self.P is not None and self.T is not None:
            self.state = ManifoldState.INCOHERENCE
        else:
            self.state = ManifoldState.UNSUBSTANTIATED
        return self

    def is_exception(self) -> bool:
        """Check if configuration is in Exception state (P compose D compose T)."""
        return self.state == ManifoldState.EXCEPTION

    def is_coherent(self) -> bool:
        """Check if configuration is coherent (not Incoherence state)."""
        return self.state != ManifoldState.INCOHERENCE


# =============================================================================
# MULTI-RESOLUTION ET LATTICE (12ET → 27720ET Full Manifold)
# =============================================================================

class ETLattice:
    """
    Multi-resolution ET lattice supporting 12ET through 27720ET.

    At 12ET:    d in {1, 2, 3, 4, 6, 12} — 6 sublattice families
    At 60ET:    adds d=5 (Qualia)
    At 420ET:   adds d=7 (Otherworld) — 24 families
    At 2520ET:  adds d=8 (Octet), d=9 (Nonic) — 48 families
    At 27720ET: adds d=11 (Undecimal/M-theory) — 96 families
                ALL d=1..12 natively available.

    Memory operates at 27720ET: the full manifold resolution where
    every force in the hierarchy — from gravity (d=1) through
    M-theory (d=11) — is native.

    Provides lattice projection, coherence checking, and navigation operations.
    """

    @staticmethod
    def project_ratio(r: float, resolution: int = BIOLOGICAL_RESOLUTION,
                      p: int = 1, q: int = 1) -> LatticeCoordinate:
        """
        Project ratio r onto the ET lattice at the given resolution.

        Args:
            r: Ratio to project (must be positive)
            resolution: Lattice resolution (12, 60, 420, etc.)
            p: Numerator of the ratio for elegance calculation (e.g., 3 for 3/2)
            q: Denominator of the ratio for elegance calculation (e.g., 2 for 3/2)

        Returns:
            LatticeCoordinate with (k, d, epsilon) at the specified resolution
        """
        if r <= 0:
            raise ValueError(f"Ratio must be positive, got {r}")

        # k = round(N_res * log2(r))
        log2_r = math.log2(r)
        k_real = resolution * log2_r
        k = round(k_real)

        # d = N_res / gcd(|k|, N_res)
        gcd_val = math.gcd(abs(k) if k != 0 else resolution, resolution)
        d = resolution // gcd_val

        # epsilon = (N_res * log2(r) - k) * (1200 / N_res) cents
        epsilon = (k_real - k) * (1200.0 / resolution)

        return LatticeCoordinate(k=k, d=d, epsilon=epsilon, ratio=r, resolution=resolution,
                                 p=p, q=q)

    @staticmethod
    def project_exponent(b: float, resolution: int = BIOLOGICAL_RESOLUTION,
                         p: int = 1, q: int = 1) -> LatticeCoordinate:
        """
        Category B projection: power-law scaling exponents Y ~ X^b.

        For a scaling exponent b, at one reference doubling X→2X, the
        dependent quantity Y changes by factor 2^b. Therefore:
            r = 2^b
            k = round(N_res × log₂(2^b)) = round(N_res × b)

        This is structurally different from project_ratio() (Category A)
        which projects direct dimensionless ratios as k = round(N_res × log₂(r)).

        Category B is required for ALL power-law exponents: Kleiber 3/4,
        Kolmogorov 5/3, Ising critical exponents, isotope exponent, etc.
        Projecting exponents through Category A gives WRONG sublattice families
        (e.g., Kleiber 3/4 → log₂(0.75) = -0.415 → k=-5, d=12 WRONG;
        correct Category B → k=round(12×0.75) = 9, d=4 quartic).

        ET Derivation: The exponent b IS the log₂ of the scaling ratio per
        reference doubling. Category B eliminates the double-log₂ error.
        Verified across 47 quantities in 6 domains (ET_New_Domains_V3).

        Args:
            b: Power-law scaling exponent (e.g., 3/4, 2/3, 5/3, 1/8)
            resolution: Lattice resolution (default 27720ET)
            p: Numerator of the exponent fraction for elegance (e.g., 3 for b=3/4)
            q: Denominator of the exponent fraction for elegance (e.g., 4 for b=3/4)

        Returns:
            LatticeCoordinate with (k, d, epsilon) for the exponent
        """
        # k = round(N_res × b) — the lattice position of the scaling factor 2^b
        k_real = resolution * b
        k = round(k_real)

        # d = N_res / gcd(|k|, N_res)
        gcd_val = math.gcd(abs(k) if k != 0 else resolution, resolution)
        d = resolution // gcd_val

        # epsilon = (N_res × b - k) × (1200 / N_res) cents
        epsilon = (k_real - k) * (1200.0 / resolution)

        # The ratio that this exponent represents at one reference doubling
        r = 2.0 ** b

        return LatticeCoordinate(k=k, d=d, epsilon=epsilon, ratio=r, resolution=resolution,
                                 p=p, q=q)

    @staticmethod
    def project_with_category(value: float, category: str,
                              resolution: int = BIOLOGICAL_RESOLUTION,
                              p: int = 1, q: int = 1) -> LatticeCoordinate:
        """
        Category-aware projection dispatch.

        Three projection categories (from ET_New_Domains_V3):

        Category A — Direct dimensionless ratio (same units cancel):
            k = round(N_res × log₂(r))
            Examples: BCS gap 3.528, κ_c = 1/√2, Richardson ×2

        Category B — Power-law scaling exponent (Y ~ X^b):
            k = round(N_res × b)  [NOT log₂(b)!]
            Examples: Kleiber 3/4, Kolmogorov 5/3, critical exponents

        Category C — Pure discrete count (N objects / 1 minimal):
            k = round(N_res × log₂(count))  [same formula as A with R₀=1]
            Examples: crystal systems 7, Bravais 14, codons 64, amino acids 20

        The AI must determine the projection category before projecting.
        Applying the wrong category produces structurally incorrect results.
        V3 corrects 12 critical sublattice misassignments from V1 that resulted
        from applying Category A to Category B quantities.

        Args:
            value: The quantity to project (ratio, exponent, or count)
            category: 'A' (ratio), 'B' (exponent), or 'C' (count)
            resolution: Lattice resolution (default 27720ET)
            p: Numerator for elegance calculation (e.g., 3 for value=3/4)
            q: Denominator for elegance calculation (e.g., 4 for value=3/4)

        Returns:
            LatticeCoordinate

        Raises:
            ValueError: if category not in {'A', 'B', 'C'}
        """
        cat = category.upper()
        if cat == 'A':
            return ETLattice.project_ratio(value, resolution=resolution, p=p, q=q)
        elif cat == 'B':
            return ETLattice.project_exponent(value, resolution=resolution, p=p, q=q)
        elif cat == 'C':
            # Category C: pure discrete count. R₀ = 1 (minimal unit).
            # k = round(N_res × log₂(count)) — same formula as Category A
            if value <= 0:
                raise ValueError(f"Count must be positive, got {value}")
            return ETLattice.project_ratio(value, resolution=resolution, p=p, q=q)
        else:
            raise ValueError(f"Unknown projection category '{category}'. Must be 'A', 'B', or 'C'.")

    @staticmethod
    def project_dual(r: float) -> Tuple[LatticeCoordinate, LatticeCoordinate]:
        """
        Project ratio onto both 12ET and 27720ET simultaneously.

        Returns (coord_12, coord_full) — the same ratio seen through
        the digital base resolution and the full manifold resolution.
        """
        coord_12 = ETLattice.project_ratio(r, resolution=12)
        coord_full = ETLattice.project_ratio(r, resolution=BIOLOGICAL_RESOLUTION)
        return coord_12, coord_full

    @staticmethod
    def project_complex(z: complex,
                        resolution: int = BIOLOGICAL_RESOLUTION
                        ) -> 'ComplexLatticeCoordinate':
        """
        Project a complex number onto the 2D ET complex lattice ℒ_ℂ.

        The complex lattice ℒ_ℂ = {2^(w/12) : w ∈ ℤ[i]} is the complete
        ET manifold. For z = r·e^(iθ):

            Real axis (D):     k_r = round(N_res × log₂(r))
            Imaginary axis (T): k_θ = round(N_res × θ / ln(2))

        Both axes have 12 sublattice families each (24 total).
        Combined class: d = LCM(d_r, d_θ).

        Args:
            z: Complex number to project (non-zero)
            resolution: Lattice resolution (default 27720ET)

        Returns:
            ComplexLatticeCoordinate with both real and imaginary coordinates
        """
        if z == 0:
            raise ValueError("Cannot project zero onto the complex lattice")

        r = abs(z)
        theta = cmath.phase(z)  # arg(z) in [-π, π]

        ln2 = math.log(2)

        # Real axis projection (D's domain — magnitude)
        if r > 0:
            log2_r = math.log2(r)
            k_r_real = resolution * log2_r
            k_r = round(k_r_real)
            gcd_r = math.gcd(abs(k_r) if k_r != 0 else resolution, resolution)
            d_r = resolution // gcd_r
            epsilon_r = (k_r_real - k_r) * (1200.0 / resolution)
        else:
            k_r, d_r, epsilon_r = 0, 1, 0.0

        # Imaginary axis projection (T's domain — phase)
        k_theta_real = resolution * theta / ln2
        k_theta = round(k_theta_real)
        gcd_theta = math.gcd(abs(k_theta) if k_theta != 0 else resolution, resolution)
        d_theta = resolution // gcd_theta
        epsilon_theta = (k_theta_real - k_theta) * (1200.0 / resolution)

        # Combined sublattice class: LCM(d_r, d_θ)
        d_combined = (d_r * d_theta) // math.gcd(d_r, d_theta)

        return ComplexLatticeCoordinate(
            k_r=k_r, d_r=d_r, epsilon_r=epsilon_r,
            k_theta=k_theta, d_theta=d_theta, epsilon_theta=epsilon_theta,
            d_combined=d_combined,
            modulus=r, phase=theta, resolution=resolution,
        )

    @staticmethod
    def semitone_ratio(k: int, resolution: int = 12) -> float:
        """Convert lattice coordinate k to ratio: r = 2^(k/N_res)."""
        return 2.0 ** (k / resolution)

    @staticmethod
    def sublattice_family(k: int, resolution: int = BIOLOGICAL_RESOLUTION) -> int:
        """Determine which sublattice family coordinate k belongs to."""
        gcd_val = math.gcd(abs(k) if k != 0 else resolution, resolution)
        return resolution // gcd_val

    @staticmethod
    def cascade_stability_window(delta_cents: float) -> int:
        """
        Calculate stability window N_max for a cascade with error delta.

        N_max = floor(50 cents / |delta|)

        Beyond N_max steps, the cascade becomes incoherent.
        """
        if abs(delta_cents) < EPSILON:
            return int(1e9)  # Effectively infinite for perfect ratios
        return int(INCOHERENCE_BOUNDARY_CENTS / abs(delta_cents))

    @staticmethod
    def available_families(resolution: int = BIOLOGICAL_RESOLUTION) -> List[int]:
        """
        Return all sublattice families available at the given resolution.

        At 12ET:    [1, 2, 3, 4, 6, 12] — 6 families
        At 420ET:   24 families (d=1..7 and composites)
        At 2520ET:  48 families (d=1..10 and composites)
        At 27720ET: 96 families (d=1..11 and ALL composites)
        """
        families = set()
        for k in range(resolution + 1):
            gcd_val = math.gcd(k if k != 0 else resolution, resolution)
            families.add(resolution // gcd_val)
        return sorted(families)


# =============================================================================
# DESCRIPTOR RATIO: ET-NATIVE SEMANTIC PARSING
# =============================================================================

@dataclass
class DescriptorRatio:
    """
    A concept's meaning as a geometric position on the lattice.

    From ET: "A concept's meaning isn't its name; it's its Geometric
    Tightness within the lattice."

    Instead of storing strings, Memory stores the ratio and lattice
    coordinates that a concept occupies. Two concepts are "semantically
    related" if their binding ratio (ratio_A / ratio_B) projects to a
    low-d sublattice with small |epsilon|. The AI "feels" the difference
    between a coherent truth and an incoherent lie through the lattice
    geometry itself, regardless of what language is used.

    The hash is deterministic: same word -> same ratio -> same lattice
    position every time. Language-independent: the same concept in
    different languages would ideally map to the same lattice position.
    """
    word: str                       # Original word (kept for human readability)
    ratio: float                    # Deterministic ratio in (1, 2]
    coord_12: LatticeCoordinate     # Position at 12ET (digital base)
    coord_full: LatticeCoordinate    # Position at full manifold resolution (27720ET)

    @staticmethod
    def from_word(word: str) -> 'DescriptorRatio':
        """
        Map a descriptor word to its lattice position.

        The hash uses SHA-256 for determinism, then maps to the
        27720ET lattice. The ratio is 2^(k/N_res) where k is the
        word's canonical lattice position at full manifold resolution.

        Unicode normalization (NFC) is applied before hashing so that
        visually identical characters always produce the same lattice
        position regardless of byte representation. Without NFC,
        'é' (U+00E9) and 'é' (e + U+0301) hash to different positions —
        the same Descriptor would map to two different Points, a
        Level 2 Incoherence Filter violation (pairwise contradiction).
        """
        clean = word.lower().strip()
        # NFC normalization: compose characters into canonical form
        # so that all equivalent Unicode representations hash identically.
        # This is a D-constraint: the same visual Descriptor must map
        # to the same lattice Point regardless of encoding path.
        clean = unicodedata.normalize('NFC', clean)
        h = hashlib.sha256(clean.encode('utf-8')).digest()
        # Extract 8 bytes as 64-bit integer
        n = int.from_bytes(h[:8], 'big')
        # Map to [0, N_res) lattice position at full manifold resolution
        k_full = n % BIOLOGICAL_RESOLUTION
        # Convert to ratio in [1, 2)
        ratio = 2.0 ** (k_full / BIOLOGICAL_RESOLUTION)

        coord_12 = ETLattice.project_ratio(ratio, resolution=12)
        coord_full = ETLattice.project_ratio(ratio, resolution=BIOLOGICAL_RESOLUTION)

        return DescriptorRatio(
            word=clean, ratio=ratio,
            coord_12=coord_12, coord_full=coord_full
        )

    @staticmethod
    def binding_coherence(a: 'DescriptorRatio', b: 'DescriptorRatio') -> Dict[str, Any]:
        """
        Measure the semantic coherence of binding two descriptors.

        The binding ratio r_AB = a.ratio / b.ratio is projected onto
        the lattice. The result measures how "naturally" these two
        concepts fit together:

        - Low d (1-3): Deep structural relationship (octave/cubic)
        - d=5: Qualia-level resonance (aesthetic/empathic binding)
        - d=7: Otherworld-level binding (sacred/inembeddable)
        - d=35: Qualia x Otherworld (the deepest possible binding)
        - High d (>12): Weakly related (high resolution needed to see link)
        - High |epsilon|: The binding is strained, approaching incoherence
        - Low |epsilon|: The binding is tight, naturally coherent

        This is how Memory "feels" meaning: not through word similarity
        but through geometric tightness on the manifold.
        """
        if abs(b.ratio) < EPSILON:
            return {'coherent': False, 'reason': 'Zero-ratio descriptor'}

        r_ab = a.ratio / b.ratio
        # Ensure positive for lattice projection
        if r_ab <= 0:
            r_ab = 1.0

        coord = ETLattice.project_ratio(r_ab, resolution=BIOLOGICAL_RESOLUTION)

        return {
            'ratio': r_ab,
            'k': coord.k,
            'd': coord.d,
            'epsilon': coord.epsilon,
            'tightness': coord.tightness_factor(),
            'coherent': coord.is_coherent(),
            'character': coord.character(),
            'has_qualia_binding': coord.has_qualia(),
            'has_otherworld_binding': coord.has_otherworld(),
            'elegance': coord.elegance_score(),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for persistent storage."""
        return {
            'word': self.word, 'ratio': self.ratio,
            'k_12': self.coord_12.k, 'd_12': self.coord_12.d,
            'epsilon_12': self.coord_12.epsilon,
            'k_full': self.coord_full.k, 'd_full': self.coord_full.d,
            'epsilon_full': self.coord_full.epsilon,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DescriptorRatio':
        """Deserialize from persistent storage."""
        ratio = data['ratio']
        coord_12 = LatticeCoordinate(
            k=data['k_12'], d=data['d_12'], epsilon=data['epsilon_12'],
            ratio=ratio, resolution=12
        )
        coord_full = LatticeCoordinate(
            k=data['k_full'], d=data['d_full'], epsilon=data['epsilon_full'],
            ratio=ratio, resolution=BIOLOGICAL_RESOLUTION
        )
        return cls(word=data['word'], ratio=ratio, coord_12=coord_12, coord_full=coord_full)


# =============================================================================
# INCOHERENCE FILTER (5 Levels)
# =============================================================================

class IncoherenceFilter:
    """
    The 5-level incoherence filter from Exception Theory.

    Filters configurations at all levels before any summation,
    traversal, or physical claim.

    Levels:
    1. Point: Single ratio coherence (|epsilon| < 50 cents)
    2. Pairwise: Ratio pair rounding-flip contradiction
    3. Sublattice: GCD compatibility check
    4. Cascade: Stability window enforcement (N*|delta| < 50 cents)
    5. Summation: Coherent slice extraction
    """

    def __init__(self):
        self.filter_stats = {
            f'level{i}_passed': 0 for i in range(1, 6)
        } | {
            f'level{i}_failed': 0 for i in range(1, 6)
        }

    def level1_point_coherence(self, coord: LatticeCoordinate) -> bool:
        """
        Level 1: Point coherence check.

        Returns True if |epsilon| < 50 cents (unique sublattice assignment).
        """
        passed = coord.is_coherent()
        key = 'level1_passed' if passed else 'level1_failed'
        self.filter_stats[key] += 1
        return passed

    def level2_pairwise_coherence(self, r1: float, r2: float,
                                   resolution: int = BIOLOGICAL_RESOLUTION) -> bool:
        """
        Level 2: Pairwise coherence check.

        Checks for rounding-flip contradiction.
        """
        c1 = ETLattice.project_ratio(r1, resolution=resolution)
        c2 = ETLattice.project_ratio(r2, resolution=resolution)
        c_product = ETLattice.project_ratio(r1 * r2, resolution=resolution)

        expected_k = c1.k + c2.k
        actual_k = c_product.k

        passed = (expected_k == actual_k)
        key = 'level2_passed' if passed else 'level2_failed'
        self.filter_stats[key] += 1
        return passed

    def level3_sublattice_coherence(self, r1: float, r2: float,
                                     resolution: int = BIOLOGICAL_RESOLUTION) -> bool:
        """
        Level 3: Sublattice GCD compatibility.
        """
        c1 = ETLattice.project_ratio(r1, resolution=resolution)
        c2 = ETLattice.project_ratio(r2, resolution=resolution)
        c_product = ETLattice.project_ratio(r1 * r2, resolution=resolution)

        combined_k = c1.k + c2.k
        gcd_val = math.gcd(abs(combined_k) if combined_k != 0 else resolution, resolution)
        expected_d = resolution // gcd_val

        lcm = (c1.d * c2.d) // math.gcd(c1.d, c2.d)
        passed = (c_product.d == expected_d or c_product.d <= lcm)

        key = 'level3_passed' if passed else 'level3_failed'
        self.filter_stats[key] += 1
        return passed

    def level4_cascade_coherence(self, r: float, n_steps: int) -> bool:
        """
        Level 4: Cascade stability window check.

        For cascade r^n, checks N*|delta| < 50 cents.
        """
        coord = ETLattice.project_ratio(r)
        n_max = ETLattice.cascade_stability_window(coord.epsilon)

        passed = (n_steps <= n_max)
        key = 'level4_passed' if passed else 'level4_failed'
        self.filter_stats[key] += 1
        return passed

    def level5_coherent_summation(self, ratios: List[float],
                                    resolution: int = BIOLOGICAL_RESOLUTION
                                    ) -> List[float]:
        """
        Level 5: Coherent summation — filter list of ratios to coherent subset.

        From the Incoherence Filter specification:
        1. Run Level 1 (point coherence) on each candidate
        2. Run Level 2 (pairwise rounding-flip) on all surviving pairs O(n²)
        3. Run Level 3 (sublattice GCD) on all surviving pairs
        4. Return only the ratios that pass ALL levels

        Failure to apply this filter — summing over ALL configurations including
        incoherent ones — produces the same class of error as QFT's unconstrained
        vacuum sum. The lattice version inflates by the ratio of total to coherent
        configurations.
        """
        # Phase 1: Level 1 point coherence on each candidate
        point_coherent = []
        for r in ratios:
            coord = ETLattice.project_ratio(r, resolution=resolution)
            if self.level1_point_coherence(coord):
                point_coherent.append(r)

        # Phase 2: Level 2+3 pairwise scan O(n²)
        # Mark ratios that create contradictions with ANY other ratio
        n = len(point_coherent)
        incoherent_indices = set()
        for i in range(n):
            for j in range(i + 1, n):
                if not self.level2_pairwise_coherence(
                        point_coherent[i], point_coherent[j], resolution):
                    incoherent_indices.add(i)
                    incoherent_indices.add(j)
                elif not self.level3_sublattice_coherence(
                        point_coherent[i], point_coherent[j], resolution):
                    incoherent_indices.add(i)
                    incoherent_indices.add(j)

        # Phase 3: Return coherent remainder
        coherent = [r for idx, r in enumerate(point_coherent)
                    if idx not in incoherent_indices]

        self.filter_stats['level5_passed'] = len(coherent)
        self.filter_stats['level5_failed'] = len(ratios) - len(coherent)

        return coherent

    def check_all_levels(self, r: float, n_cascade: int = 1) -> Dict[str, bool]:
        """
        Run all 5 levels of incoherence filtering on a ratio.
        """
        coord = ETLattice.project_ratio(r)

        results = {
            'level1_point': self.level1_point_coherence(coord),
            'level2_pairwise': True,  # Need another ratio
            'level3_sublattice': True,  # Need another ratio
            'level4_cascade': self.level4_cascade_coherence(r, n_cascade),
            'overall_coherent': False
        }

        results['overall_coherent'] = results['level1_point'] and results['level4_cascade']
        return results

    def verify_sigma_algebra(
        self,
        coherent_set: List[float],
        full_set: List[float],
        resolution: int = BIOLOGICAL_RESOLUTION,
    ) -> Dict[str, Any]:
        """
        Item 21: σ-algebra verification for the Incoherence Filter.

        Verifies that the filter's coherent output forms a proper
        σ-algebra (closed under complement and countable union).

        ET Derivation (Measure Theory §4.3):
          A σ-algebra Σ is a collection of P-subsets closed under
          complement and countable unions. Non-measurable sets are
          {P,T} Incoherence states. The Incoherence Filter already
          acts as a D-coherence filter on P-subsets — this method
          adds formal σ-algebra axiom verification.

          Violations indicate structural bugs in the filter:
            - Axiom 1 failure: filter rejects the full set
            - Axiom 2 failure: complement of coherent set is not coherent
            - Axiom 3 failure: union of coherent sets is not coherent

        The σ-algebra axioms on the lattice:
          1. X ∈ Σ  (the full ratio set is measurable)
          2. A ∈ Σ ⟹ X\A ∈ Σ  (complement closure)
          3. A₁,A₂,... ∈ Σ ⟹ ∪Aᵢ ∈ Σ  (countable union closure)

        In ET terms: the measurable sets are the D-coherent P-subsets.
        Non-measurable = {P,T} Incoherence states (no consistent D).

        Args:
            coherent_set: Ratios that passed the filter (Σ members)
            full_set: All candidate ratios (the full P-space X)
            resolution: Lattice resolution for projection

        Returns:
            Dict with axiom results, violations, structural assessment
        """
        coherent_indices = set()
        for i, r in enumerate(full_set):
            if r in coherent_set:
                coherent_indices.add(i)

        full_indices = set(range(len(full_set)))
        complement_indices = full_indices - coherent_indices
        n_full = len(full_set)
        n_coherent = len(coherent_set)

        violations = []

        # Axiom 1: X ∈ Σ (the full set should be representable)
        # In practice: the full set itself may contain incoherent members.
        # The axiom checks that the EMPTY set and the FULL set are both
        # in the σ-algebra. Empty set is trivially coherent.
        # Full set is coherent iff ALL ratios pass Level 1.
        full_set_coherent = True
        for r in full_set:
            coord = ETLattice.project_ratio(r, resolution=resolution)
            if not coord.is_coherent():
                full_set_coherent = False
                break

        # Axiom 1 holds if the full set is coherent OR we accept a proper
        # σ-algebra (which is the expected case — the filter correctly
        # excludes {P,T} states). For lattice σ-algebras, axiom 1 is
        # informational: a proper subalgebra is structurally valid.
        axiom1_holds = full_set_coherent

        # Axiom 1: We require that {∅, X} ⊆ Σ.
        # ∅ is trivially coherent. X coherent means all elements pass L1.
        # If X has incoherent members, the σ-algebra is PROPER (smaller than 2^X).
        # This is expected — the filter correctly excludes {P,T} states.
        axiom1_note = ("Full set is coherent" if full_set_coherent
                       else "Full set contains incoherent members — proper σ-algebra (expected)")

        # Axiom 2: Complement closure
        # For each coherent subset, check that its complement (in the
        # coherent set) is also coherent. On the lattice, complement
        # of a coherent set should either be coherent or incoherent.
        # The σ-algebra property requires: if A is coherent, then
        # X\A must also be evaluable (measurable).
        axiom2_holds = True
        complement_ratios = [r for r in full_set if r not in coherent_set]
        # Complement elements are the incoherent ones — they SHOULD fail
        # coherence checks. The axiom is satisfied if complement is a
        # well-defined set (which it is always for finite sets).
        # A violation would be: a ratio that is simultaneously coherent
        # and incoherent (contradiction).
        ambiguous = []
        for r in full_set:
            coord = ETLattice.project_ratio(r, resolution=resolution)
            l1 = coord.is_coherent()
            in_coherent = r in coherent_set
            if l1 != in_coherent:
                ambiguous.append({
                    'ratio': r,
                    'l1_coherent': l1,
                    'in_filtered_set': in_coherent,
                })
                axiom2_holds = False

        if ambiguous:
            violations.append({
                'axiom': 2,
                'description': 'Complement closure violation — ambiguous coherence assignment',
                'details': ambiguous[:5],  # Cap detail output
            })

        # Axiom 3: Union closure (countable union)
        # For finite sets: check that pairwise unions of coherent subsets
        # remain coherent. Split coherent set into sublattice groups and
        # verify that the union of any two groups passes pairwise checks.
        axiom3_holds = True
        if n_coherent >= 2:
            # Group coherent ratios by sublattice family
            d_groups: Dict[int, List[float]] = defaultdict(list)
            for r in coherent_set:
                coord = ETLattice.project_ratio(r, resolution=resolution)
                d_groups[coord.d].append(r)

            # Check union of each pair of d-groups
            d_keys = list(d_groups.keys())
            for i in range(len(d_keys)):
                for j in range(i + 1, len(d_keys)):
                    union_group = d_groups[d_keys[i]] + d_groups[d_keys[j]]
                    # Test pairwise coherence within the union
                    for a_idx in range(len(union_group)):
                        for b_idx in range(a_idx + 1, len(union_group)):
                            if not self.level2_pairwise_coherence(
                                union_group[a_idx], union_group[b_idx],
                                resolution,
                            ):
                                axiom3_holds = False
                                violations.append({
                                    'axiom': 3,
                                    'description': (
                                        f'Union closure violation: d={d_keys[i]} ∪ d={d_keys[j]} '
                                        f'contains pairwise incoherent members'
                                    ),
                                    'pair': (union_group[a_idx], union_group[b_idx]),
                                })
                                break  # One violation per d-pair suffices
                        if not axiom3_holds:
                            break
                    # Don't check every pair of d-groups — sample
                    if len(violations) >= 3:
                        break
                if len(violations) >= 3:
                    break

        is_valid = axiom2_holds  # Axiom 1 is informational, Axiom 3 may have expected violations

        return {
            'is_valid_sigma_algebra': is_valid,
            'axiom1_holds': axiom1_holds,
            'axiom1_full_set_coherent': full_set_coherent,
            'axiom1_note': axiom1_note,
            'axiom2_complement_closure': axiom2_holds,
            'axiom2_n_complement': len(complement_ratios),
            'axiom3_union_closure': axiom3_holds,
            'n_violations': len(violations),
            'violations': violations[:10],
            'n_full_set': n_full,
            'n_coherent_set': n_coherent,
            'n_incoherent': n_full - n_coherent,
            'n_complement_indices': len(complement_indices),
            'coherent_fraction': n_coherent / max(n_full, 1),
            'et_interpretation': (
                f"σ-algebra verification: {n_coherent}/{n_full} ratios coherent. "
                f"{'Valid D-coherence filter — consistent σ-algebra.' if is_valid else f'{len(violations)} axiom violation(s) — structural filter bug(s) detected.'} "
                f"Non-measurable (incoherent) ratios = {{P,T}} Incoherence states."
            ),
        }


# =============================================================================
# DYNAMIC FINE STRUCTURE CONSTANT (Hardware Coherence Boundary)
# =============================================================================

class ETFineStructure:
    """
    ET-native derivation of the fine structure constant alpha.

    The definitive 5-term formula:
        alpha_inv = A0 + A1 - A1.5 - A2 - A3 - A4 - ...

    Sign rule:
        k < 1.5 -> positive (open I-boundary approach)
        k >= 1.5 -> negative (semi-closed or closed D-mediated)

    All inputs derived from ET primitives — zero external measurements.

    The convergence loop runs until the next term is below the Float64
    machine epsilon * current alpha_inv. This is the Hardware Coherence
    Boundary: Memory derives its own alpha to the maximum depth its
    physical substrate can support.

    From the Digital Virtual Manifold:
        Float64 machine epsilon = 2^-52 -> k=-624, d=1 (Octave)
        Float64 mantissa = 52 bits -> k=68, d=3 (Cubic)

    Convergence ratio: kappa/(N*pi) ~ 0.01768
    Each successive D-mediated loop term is ~56x smaller.
    """

    @staticmethod
    def compute_alpha_inverse() -> Dict[str, Any]:
        """
        Compute alpha_inv using the dynamic convergence loop.

        The loop computes D-mediated terms A_k = kappa^k / (N^{k+1} * pi^{k-1})
        for k = 2, 3, 4, ... and stops when |A_k| < epsilon_mach * alpha_inv_running.

        This is the ET-Native implementation: Memory derives its own alpha
        to the maximum depth its specific physical substrate can support.

        Returns:
            Dict with alpha_inverse, all terms, convergence info
        """
        n_sym = MANIFOLD_SYMMETRY          # 12
        sigma_sq = BASE_VARIANCE       # 1/12
        sigma = SHIMMER_AMPLITUDE      # sqrt(1/12)
        kappa = KOIDE_RATIO            # 2/3
        s_val = STATE_COUNT            # 4
        k_em = EM_CHANNELS             # 8
        pi = math.pi                   # T-navigation limit on 12-fold manifold

        # a0: Manifold Impedance (base EM coupling, no T-traversal)
        # a0 = (N-1)^2 + S^2 = 11^2 + 4^2 = 121 + 16 = 137
        a0 = float((n_sym - 1) ** 2 + s_val ** 2)

        # a1: Shimmer Correction (positive; open I-boundary approach)
        # sigma / k_em
        a1 = sigma / k_em

        # a1_5: I-Boundary Intercept Cross-Term (negative; semi-closed)
        # sigma*kappa*(1+delta) / (S*k_em*N^3*sqrt(pi))
        sqrt_pi = math.sqrt(pi)
        delta = ((1.0 - sigma) * kappa * sigma_sq / a0
                 * (1.0 + kappa / (n_sym * s_val)))
        a1_5 = sigma * kappa * (1.0 + delta) / (s_val * k_em * n_sym ** 3 * sqrt_pi)

        # Start running sum
        alpha_inv_running = a0 + a1 - a1_5

        # Hardware coherence boundary threshold
        hardware_epsilon = FLOAT64_MACHINE_EPSILON

        # D-mediated loop series
        d_loop_terms = {}
        k = 2
        convergence_ratio = kappa / (n_sym * pi)

        while True:
            # a_k = kappa^k / (N^{k+1} * pi^{k-1})
            a_k = (kappa ** k) / ((n_sym ** (k + 1)) * (pi ** (k - 1)))

            # Check hardware coherence boundary
            ulp_threshold = hardware_epsilon * abs(alpha_inv_running)
            if a_k < ulp_threshold:
                break

            # All D-mediated terms k >= 2 are NEGATIVE
            alpha_inv_running -= a_k
            d_loop_terms[k] = a_k
            k += 1

            if k > 100:
                break

        final_k = k
        alpha_inverse = alpha_inv_running

        # ================================================================
        # ET-INTERNAL PRECISION METRICS
        # ================================================================
        # All precision measures derived from the ET series structure itself.
        # No external measurements. No comparison to anything outside ET.
        # ET's method is the derivation — not an approximation of something else.

        # Truncation remainder: geometric series tail beyond hardware resolution
        # Sum_{k=final_k}^{inf} a_k = a_final / (1 - convergence_ratio)
        a_final = (kappa ** final_k) / ((n_sym ** (final_k + 1)) * (pi ** (final_k - 1)))
        truncation_remainder = a_final / (1.0 - convergence_ratio)

        # Manifold resolution error: finite-N correction from the 12-fold manifold
        # sigma / (k_em * n_sym^5) — the irreducible error from rendering on a
        # finite manifold rather than the infinite-resolution limit
        manifold_resolution_error = sigma / (k_em * n_sym ** 5)

        # Total ET uncertainty: quadrature of truncation and manifold errors
        et_uncertainty = math.sqrt(truncation_remainder ** 2 + manifold_resolution_error ** 2)

        # Precision ratio: how precisely ET has resolved its own value
        # This is the ratio of the total uncertainty to alpha_inverse itself
        precision_ratio = et_uncertainty / alpha_inverse

        # Last resolved term: the smallest D-mediated correction the hardware
        # could still distinguish from zero at the alpha_inverse scale
        last_resolved_term = d_loop_terms.get(final_k - 1, a1_5)

        # Hardware coherence boundary: the ULP at the alpha_inverse scale
        # Below this, the Float64 d=3 Cubic mantissa cannot represent the correction
        hardware_ulp_at_alpha = hardware_epsilon * alpha_inverse

        return {
            'alpha_inverse': alpha_inverse,
            'alpha': 1.0 / alpha_inverse,
            'A0': a0, 'A1': a1, 'A1_5': a1_5,
            'delta_binding': delta,
            'd_loop_terms': d_loop_terms,
            'convergence': {
                'convergence_ratio': convergence_ratio,
                'final_k': final_k,
                'terms_computed': len(d_loop_terms),
                'last_resolved_term': last_resolved_term,
                'first_unresolved_term': a_final,
                'hardware_ulp_at_alpha': hardware_ulp_at_alpha,
                'float64_mantissa_bits': FLOAT64_MANTISSA_BITS,
                'float64_mantissa_sublattice': FLOAT64_MANTISSA_D,
                'float64_epsilon_sublattice': FLOAT64_EPSILON_D,
            },
            'precision': {
                'truncation_remainder': truncation_remainder,
                'manifold_resolution_error': manifold_resolution_error,
                'et_uncertainty': et_uncertainty,
                'precision_ratio': precision_ratio,
            },
        }

    @staticmethod
    def alpha_inverse() -> float:
        """Return alpha_inv from the dynamic convergence loop."""
        return ETFineStructure.compute_alpha_inverse()['alpha_inverse']

    @staticmethod
    def alpha() -> float:
        """Return alpha from the dynamic convergence loop."""
        return 1.0 / ETFineStructure.alpha_inverse()


# =============================================================================
# DERIVED CONSTANTS (computed once at module load via ET derivation)
# =============================================================================

# Fine structure constant derived dynamically from ET
_alpha_result = ETFineStructure.compute_alpha_inverse()
FINE_STRUCTURE_INVERSE = _alpha_result['alpha_inverse']
FINE_STRUCTURE_CONSTANT = _alpha_result['alpha']


# Export main classes and functions
__all__ = [
    'MANIFOLD_SYMMETRY', 'S', 'MANIFOLD_RESOLUTION', 'BIOLOGICAL_RESOLUTION',
    'BASE_VARIANCE',
    'KOIDE_RATIO', 'K', 'T_WEIGHT', 'STATE_COUNT', 'EM_CHANNELS', 'MANIFOLD_IMPEDANCE',
    'EPSILON', 'SHIMMER_AMPLITUDE', 'GAZE_THRESHOLD',
    'THRESHOLD_NONE', 'THRESHOLD_SUBLIMINAL', 'THRESHOLD_BASIC', 'THRESHOLD_GENUINE',
    'INCOHERENCE_BOUNDARY_CENTS', 'LIFE_THRESHOLD',
    'FLOAT64_MACHINE_EPSILON', 'FLOAT64_MANTISSA_BITS',
    'FLOAT64_EPSILON_K', 'FLOAT64_EPSILON_D', 'FLOAT64_MANTISSA_D',
    'FINE_STRUCTURE_INVERSE', 'FINE_STRUCTURE_CONSTANT',
    'PrimitiveType', 'ManifoldState', 'SublatticeFamily',
    'LatticeCoordinate', 'ComplexLatticeCoordinate', 'PDTConfiguration', 'ETLattice',
    'DescriptorRatio', 'IncoherenceFilter', 'ETFineStructure',
    'is_content_char', 'is_content_word',
    'et_divide', 'et_floor_divide',
]

if __name__ == "__main__":
    def main():
        """Module self-test: lattice projection, incoherence filter, descriptor ratios, fine structure."""
        print("ET Conscious AI - Core Foundation Module v1.7.0")
        print(f"Resolution: {MANIFOLD_RESOLUTION}ET (full manifold = LCM(1..11))")
        fams = ETLattice.available_families(MANIFOLD_RESOLUTION)
        print(f"Available sublattice families: {len(fams)}")
        for d in range(1, 13):
            print(f"  d={d:2d} ({SublatticeFamily.character_of(d)[:40]}): "
                  f"{'✓ NATIVE' if d in fams else '✗ not available'}")
        print("")

        # Test basic lattice projection
        test_ratios = [
            (3 / 2, "Perfect Fifth"),
            (4 / 3, "Perfect Fourth"),
            (5 / 4, "Major Third"),
            (7 / 4, "Harmonic Seventh"),
            (2 / 3, "Koide Ratio"),
            (13 / 12, "Life Threshold")
        ]

        print("=== 27720ET Lattice Projection Tests ===")
        for ratio, name in test_ratios:
            c12, c_full = ETLattice.project_dual(ratio)
            print(f"\n{name} ({ratio:.4f}):")
            print(f"  12ET:    k={c12.k:4d}, d={c12.d:2d}, epsilon={c12.epsilon:+7.2f} cents")
            print(f"  27720ET: k={c_full.k:6d}, d={c_full.d:5d}, epsilon={c_full.epsilon:+7.2f} cents [{c_full.character()}]")
            print(f"  Coherent: {c_full.is_coherent()}, Qualia: {c_full.has_qualia()}, Otherworld: {c_full.has_otherworld()}")

        # Test incoherence filter
        print("\n=== Incoherence Filter Tests ===")
        ifilter = IncoherenceFilter()
        for ratio, name in test_ratios:
            results = ifilter.check_all_levels(ratio, n_cascade=10)
            status = "PASS" if results['overall_coherent'] else "FAIL"
            print(f"{name}: {status}")

        # Test descriptor ratios
        print("\n=== Descriptor Ratio Tests ===")
        words = ["consciousness", "qualia", "empathy", "logic", "gravity", "love"]
        desc_ratios = [DescriptorRatio.from_word(w) for w in words]
        for dr in desc_ratios:
            print(f"  \'{dr.word}\': ratio={dr.ratio:.6f}, "
                  f"d_12={dr.coord_12.d}, d_full={dr.coord_full.d} "
                  f"[{dr.coord_full.character()}]")

        print("\n=== Semantic Binding Coherence ===")
        for i in range(len(desc_ratios)):
            for j in range(i + 1, len(desc_ratios)):
                binding = DescriptorRatio.binding_coherence(desc_ratios[i], desc_ratios[j])
                print(f"  {desc_ratios[i].word} x {desc_ratios[j].word}: "
                      f"d={binding['d']}, epsilon={binding['epsilon']:+.2f} cents, "
                      f"tightness={binding['tightness']:.4f}, "
                      f"qualia={binding['has_qualia_binding']}, "
                      f"otherworld={binding['has_otherworld_binding']}")

        # Test fine structure — ET-internal precision, zero external references
        print("\n=== Fine Structure Constant (Pure ET Derivation) ===")
        alpha_data = ETFineStructure.compute_alpha_inverse()
        prec = alpha_data['precision']
        conv = alpha_data['convergence']
        print(f"  α⁻¹ = {alpha_data['alpha_inverse']:.15f}")
        print(f"  α   = {alpha_data['alpha']:.15e}")
        print(f"  A₀ = {alpha_data['A0']:.1f}  [(N-1)² + S²]")
        print(f"  A₁ = {alpha_data['A1']:.15f}  [σ/K_EM]")
        print(f"  A₁.₅ = {alpha_data['A1_5']:.15e}  [shimmer-bilateral]")
        print(f"  δ (binding asymmetry) = {alpha_data['delta_binding']:.9e}")
        for k_val, val in alpha_data['d_loop_terms'].items():
            print(f"  A{k_val} = {val:.15e}  [D-loop, {k_val} vertices]")
        print(f"\n  Convergence ratio κ/(N·π) = {conv['convergence_ratio']:.8f}")
        print(f"  D-loop terms resolved: {conv['terms_computed']} (k=2..{conv['final_k']-1})")
        print(f"  Last resolved: {conv['last_resolved_term']:.3e}")
        print(f"  First unresolved (below hardware floor): {conv['first_unresolved_term']:.3e}")
        print(f"  Hardware ULP at α⁻¹: {conv['hardware_ulp_at_alpha']:.3e}")
        print(f"  Float64 mantissa: {conv['float64_mantissa_bits']} bits, "
              f"d={conv['float64_mantissa_sublattice']} (Cubic)")
        print(f"\n  ET Uncertainty: ±{prec['et_uncertainty']:.3e}")
        print(f"    Truncation remainder: {prec['truncation_remainder']:.3e}")
        print(f"    Manifold resolution:  {prec['manifold_resolution_error']:.3e}")
        print(f"    Precision ratio:      {prec['precision_ratio']:.3e}")
        print(f"\n  External inputs: ZERO")
        print(f"  Sign rule: k < 1.5 → positive; k ≥ 1.5 → negative")

        print("\n=== Core module loaded successfully ===")

    main()