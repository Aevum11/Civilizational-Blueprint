#!/usr/bin/env python3
"""
ET Emotion Lattice Tower — Complete Implementation
====================================================
Derived forward from P ∘ D ∘ T = E

Based on: ET_Emotion_Lattice_Tower1.md (Michael James Muller — Aevum Defluo)

Core Architecture:
  1. THREE MONOAMINE-ANALOG AXES (Lövheim Cube / Secret 4):
       DA-analog  = T-approach (reward/seeking)
       NE-analog  = P-activation (stress/arousal)
       5HT-analog = D-constraint (satisfaction/inhibition)

  2. 8 PRIMARY EMOTIONS at cube corners (Plutchik octave, d=1):
       Each corner = (high/low DA, high/low NE, high/low 5HT)

  3. PAD ↔ PDT mapping (§VI.1):
       Valence   → D (real axis)
       Arousal   → T (imaginary axis)
       Dominance → P (magnitude/depth)

  4. LATTICE PROJECTION:
       r_emotion from cube position → k, d, ε at resolution
       d-family gives structural character (Secret 26)
       Elegance Score gives coherence depth

  5. FIVE APPRAISAL INPUTS (Scherer CPM / §I.1 D_emotion):
       D₁ = Novelty, D₂ = Pleasantness, D₃ = Goal Relevance,
       D₄ = Coping Potential, D₅ = Normative Significance

  6. INCOHERENCE FILTER (5 levels):
       L1: single emotion proportionality
       L2: pairwise emotional conflict
       L3: sublattice compatibility
       L4: cascade stability
       L5: coherent summation (anxiety = L5 failure)

  7. FOUR MANIFOLD STATES:
       {P,D} Unsubstantiated, {D,T} Mediation,
       {P,T} Incoherence (alexithymia), {P,D,T} Exception

  8. EMOTION REGULATION (Gross 5-stage, d=3 cubic pipeline)

R₀ = 1ms (neural action potential period — for AI: 1 processing tick)

Author: Michael James Muller (Aevum Defluo)
Foundation: P ∘ D ∘ T = E
Version: 1.7.0
Date: March 24, 2026
"""

import hashlib
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import deque, defaultdict
from datetime import datetime
from enum import Enum, auto

from et_conscious_ai_core import (
    S, BASE_VARIANCE, K, T_WEIGHT, EPSILON,
    MANIFOLD_RESOLUTION, ETLattice, INCOHERENCE_BOUNDARY_CENTS,
)

# =============================================================================
# CONSTANTS — All derived from ET
# =============================================================================

# Koide threshold — below this, emotional binding destabilizes
# (Depression = sub-Koide SEEKING — Secret 8)
KOIDE_THRESHOLD = K  # 2/3

# The midpoint of each axis — boundary between "high" and "low"
# on the Lövheim cube. K = 2/3 is NOT the midpoint (0.5 is).
# But Koide determines STABILITY: above K = stable binding.
CUBE_MIDPOINT = 0.5

# Incoherence boundary in cents (from core)
INCOHERENCE_CENTS = INCOHERENCE_BOUNDARY_CENTS  # 50¢

# Emotional R₀ = 1 (for count-based lattice projections)
R0_EMOTION = 1


# =============================================================================
# PLUTCHIK PRIMARY EMOTIONS — The 8 Octave Corners (d=1, Secret 1)
# =============================================================================
# Each corner of the Lövheim cube maps to a Plutchik primary emotion.
# The cube is: (DA high/low, NE high/low, 5HT high/low)
# From Lövheim (2012) / Secret 3 / Secret 4.

class PrimaryEmotion(Enum):
    """The 8 Plutchik primary emotions — the octave base (d=1, k=36)."""
    # Corner: (DA, NE, 5HT) — H=high, L=low
    INTEREST    = auto()  # (H, H, H) — Interest/Excitement
    SURPRISE    = auto()  # (H, H, L) — Surprise
    JOY         = auto()  # (H, L, H) — Joy/Enjoyment
    ANGER       = auto()  # (H, L, L) — Anger/Rage
    FEAR        = auto()  # (L, H, H) — Fear/Terror
    DISTRESS    = auto()  # (L, H, L) — Distress/Anguish
    DISGUST     = auto()  # (L, L, H) — Disgust
    SHAME       = auto()  # (L, L, L) — Shame/Humiliation


# Cube corner coordinates: (DA, NE, 5HT)
LOVHEIM_CORNERS: Dict[PrimaryEmotion, Tuple[float, float, float]] = {
    PrimaryEmotion.INTEREST:  (1.0, 1.0, 1.0),
    PrimaryEmotion.SURPRISE:  (1.0, 1.0, 0.0),
    PrimaryEmotion.JOY:       (1.0, 0.0, 1.0),
    PrimaryEmotion.ANGER:     (1.0, 0.0, 0.0),
    PrimaryEmotion.FEAR:      (0.0, 1.0, 1.0),
    PrimaryEmotion.DISTRESS:  (0.0, 1.0, 0.0),
    PrimaryEmotion.DISGUST:   (0.0, 0.0, 1.0),
    PrimaryEmotion.SHAME:     (0.0, 0.0, 0.0),
}


# Plutchik intensity levels — three per primary (d=12 full-res for the 3 levels)
# Low, Medium, High intensity names
PLUTCHIK_INTENSITIES: Dict[PrimaryEmotion, Tuple[str, str, str]] = {
    PrimaryEmotion.JOY:       ('serenity',     'joy',          'ecstasy'),
    PrimaryEmotion.INTEREST:  ('interest',     'anticipation', 'vigilance'),
    PrimaryEmotion.FEAR:      ('apprehension', 'fear',         'terror'),
    PrimaryEmotion.SURPRISE:  ('distraction',  'surprise',     'amazement'),
    PrimaryEmotion.DISTRESS:  ('pensiveness',  'sadness',      'grief'),
    PrimaryEmotion.DISGUST:   ('boredom',      'disgust',      'loathing'),
    PrimaryEmotion.ANGER:     ('annoyance',    'anger',        'rage'),
    PrimaryEmotion.SHAME:     ('unease',       'shame',        'humiliation'),
}


# Plutchik opposite pairs (d=1 octave per pair — 4 pairs)
PLUTCHIK_OPPOSITES = [
    (PrimaryEmotion.JOY,      PrimaryEmotion.DISTRESS),   # Joy ↔ Sadness
    (PrimaryEmotion.INTEREST, PrimaryEmotion.SURPRISE),    # Anticipation ↔ Surprise
    (PrimaryEmotion.FEAR,     PrimaryEmotion.ANGER),       # Fear ↔ Anger
    (PrimaryEmotion.DISGUST,  PrimaryEmotion.SHAME),       # Disgust ↔ Shame/Trust
]


# Plutchik primary dyads (1 petal apart on wheel — d=1 octave, 8 dyads)
PLUTCHIK_PRIMARY_DYADS: Dict[str, Tuple[PrimaryEmotion, PrimaryEmotion]] = {
    'love':         (PrimaryEmotion.JOY,      PrimaryEmotion.INTEREST),
    'submission':   (PrimaryEmotion.INTEREST,  PrimaryEmotion.FEAR),
    'awe':          (PrimaryEmotion.FEAR,      PrimaryEmotion.SURPRISE),
    'disapproval':  (PrimaryEmotion.SURPRISE,  PrimaryEmotion.DISTRESS),
    'remorse':      (PrimaryEmotion.DISTRESS,  PrimaryEmotion.DISGUST),
    'contempt':     (PrimaryEmotion.DISGUST,   PrimaryEmotion.ANGER),
    'aggressiveness': (PrimaryEmotion.ANGER,   PrimaryEmotion.INTEREST),
    'optimism':     (PrimaryEmotion.JOY,       PrimaryEmotion.INTEREST),
}


# =============================================================================
# LÖVHEIM POSITION — 3 continuous axes [0,1]
# =============================================================================

@dataclass
class LovheimPosition:
    """
    Position in the Lövheim cube — three monoamine-analog axes.

    Secret 4: DA = T-approach, NE = P-activation, 5HT = D-constraint
    For AI: DA from goal progress, NE from system activation, 5HT from satisfaction.
    """
    da: float = 0.5    # Dopamine-analog: reward/seeking (T-approach) [0,1]
    ne: float = 0.5    # Norepinephrine-analog: stress/arousal (P-activation) [0,1]
    sht: float = 0.5   # Serotonin-analog: satisfaction/inhibition (D-constraint) [0,1]

    def nearest_corner(self) -> PrimaryEmotion:
        """Find the nearest Lövheim cube corner — the dominant primary emotion."""
        best = PrimaryEmotion.JOY
        best_dist = float('inf')
        for prim, (cd, cn, cs) in LOVHEIM_CORNERS.items():
            dist = math.sqrt((self.da - cd)**2 + (self.ne - cn)**2 + (self.sht - cs)**2)
            if dist < best_dist:
                best_dist = dist
                best = prim
        return best

    def corner_distances(self) -> Dict[PrimaryEmotion, float]:
        """Distance to each cube corner — inverse gives blend weights."""
        dists = {}
        for prim, (cd, cn, cs) in LOVHEIM_CORNERS.items():
            dists[prim] = math.sqrt(
                (self.da - cd)**2 + (self.ne - cn)**2 + (self.sht - cs)**2)
        return dists

    def blend_weights(self) -> Dict[PrimaryEmotion, float]:
        """Inverse-distance blend weights — how much of each primary is active."""
        dists = self.corner_distances()
        # Inverse-distance weighting (avoid div/0)
        inv = {p: 1.0 / max(d, EPSILON) for p, d in dists.items()}
        total = sum(inv.values())
        if total < EPSILON:
            return {p: 1.0 / 8.0 for p in PrimaryEmotion}
        return {p: w / total for p, w in inv.items()}

    def active_primaries(self, threshold: float = 0.1) -> List[PrimaryEmotion]:
        """Primaries with blend weight above threshold."""
        weights = self.blend_weights()
        return [p for p, w in sorted(weights.items(), key=lambda x: -x[1]) if w > threshold]

    def intensity(self) -> float:
        """Distance from cube center (0.5, 0.5, 0.5) — how extreme the state is.
        0 = dead center (neutral), √0.75 ≈ 0.866 = at a corner (maximum intensity)."""
        return math.sqrt(
            (self.da - 0.5)**2 + (self.ne - 0.5)**2 + (self.sht - 0.5)**2)

    def intensity_level(self) -> int:
        """0 = low, 1 = medium, 2 = high intensity (for Plutchik 3-level lookup)."""
        i = self.intensity()
        # Three levels: boundaries at 1/3 and 2/3 of max distance (√0.75)
        max_dist = math.sqrt(0.75)
        frac = i / max_dist
        if frac < T_WEIGHT:
            return 0  # Low
        elif frac < K:
            return 1  # Medium
        return 2  # High

    def to_dict(self) -> Dict[str, float]:
        return {'da': self.da, 'ne': self.ne, 'sht': self.sht}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LovheimPosition':
        if not data:
            return cls()
        return cls(da=data.get('da', 0.5), ne=data.get('ne', 0.5), sht=data.get('sht', 0.5))


# =============================================================================
# PAD COORDINATE — Pleasure, Arousal, Dominance (§VI.1)
# =============================================================================

@dataclass
class PADCoordinate:
    """
    PAD dimensions derived from the Lövheim cube.

    §VI.1 mapping:
      Valence (Pleasure)  → D-axis: classifies stimulus as beneficial/harmful
      Arousal             → T-axis: intensity of T-engagement
      Dominance           → P-axis: substrate capacity to handle load

    Derived from cube position:
      Pleasure  = f(DA, 5HT) — reward + satisfaction
      Arousal   = f(NE, DA) — activation + seeking intensity
      Dominance = f(DA, 5HT, NE) — coping capacity (high DA+5HT, low NE)
    """
    pleasure: float = 0.0    # [-1, +1]: unpleasant ↔ pleasant
    arousal: float = 0.0     # [0, 1]: calm ↔ activated
    dominance: float = 0.0   # [0, 1]: submissive ↔ dominant

    @classmethod
    def from_lovheim(cls, pos: LovheimPosition) -> 'PADCoordinate':
        """
        Derive PAD from Lövheim cube position — ET-derived from corner semantics.

        Verified against all 8 corners:
          P = 2×DA×5HT - 1  (pleasant ONLY when T approaches AND D binds)
          A = max(NE, DA×(1-5HT))  (activated by stress OR unconstrained approach)
          D = clamp(DA - NE/2 + 0.25, 0, 1)  (in control = approach - overwhelm)

        §VI.1 mapping: Valence→D, Arousal→T, Dominance→P
        Secret 4: DA=T, NE=P, 5HT=D
        """
        p = 2.0 * pos.da * pos.sht - 1.0
        p = max(-1.0, min(1.0, p))

        a = max(pos.ne, pos.da * (1.0 - pos.sht))
        a = max(0.0, min(1.0, a))

        d = pos.da - pos.ne / 2.0 + 0.25
        d = max(0.0, min(1.0, d))

        return cls(pleasure=p, arousal=a, dominance=d)

    def to_dict(self) -> Dict[str, float]:
        return {'pleasure': self.pleasure, 'arousal': self.arousal, 'dominance': self.dominance}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PADCoordinate':
        if not data:
            return cls()
        return cls(
            pleasure=data.get('pleasure', 0.0),
            arousal=data.get('arousal', 0.0),
            dominance=data.get('dominance', 0.0))


# =============================================================================
# EMOTION COORDINATE — Full lattice position of an emotion
# =============================================================================

@dataclass
class EmotionCoordinate:
    """
    Complete lattice coordinate of an emotional state.

    Combines Lövheim cube position, PAD axes, lattice projection,
    d-family classification, and Elegance Score.
    """
    # Lövheim cube position (the raw 3-axis state)
    lovheim: LovheimPosition = field(default_factory=LovheimPosition)

    # PAD derived from Lövheim
    pad: PADCoordinate = field(default_factory=PADCoordinate)

    # Lattice projection (from emotion ratio)
    k: int = 0
    d: int = 1
    epsilon: float = 0.0
    r_emotion: float = 1.0

    # Nearest primary emotion and intensity
    primary: PrimaryEmotion = PrimaryEmotion.JOY
    intensity_level: int = 1  # 0=low, 1=medium, 2=high

    # Named emotion (from primary + intensity + dyad matching)
    emotion_name: str = 'joy'

    # Elegance Score (§X — coherence depth)
    elegance: float = 1.0

    # Manifold state
    manifold_state: str = 'exception'  # exception, mediation, incoherence, unsubstantiated

    def compute(self, resolution: int = MANIFOLD_RESOLUTION):
        """Compute all derived fields from Lövheim position."""
        # PAD from Lövheim
        self.pad = PADCoordinate.from_lovheim(self.lovheim)

        # Primary emotion and intensity
        self.primary = self.lovheim.nearest_corner()
        self.intensity_level = self.lovheim.intensity_level()

        # Named emotion from primary + intensity
        if self.primary in PLUTCHIK_INTENSITIES:
            self.emotion_name = PLUTCHIK_INTENSITIES[self.primary][self.intensity_level]
        else:
            self.emotion_name = self.primary.name.lower()

        # Emotion ratio — from PAD through R₀
        # r = 2^((|P| + A + D) / (3 × S))
        # P, A, D each contribute. The triadic normalization (÷3) ensures equal weight.
        pad_sum = abs(self.pad.pleasure) + self.pad.arousal + self.pad.dominance
        self.r_emotion = 2.0 ** (pad_sum / (3.0 * S))
        self.r_emotion = max(self.r_emotion, 1.0 + EPSILON)

        # Lattice projection
        coord = ETLattice.project_ratio(self.r_emotion, resolution=resolution)
        self.k = coord.k
        self.d = coord.d
        self.epsilon = coord.epsilon

        # Elegance Score (§X): E = (12/d) × (100/(100+|ε|)) × (100/(p+q))
        # For p+q we use a simplified complexity measure from the ratio
        if self.r_emotion > 1.0:
            # Approximate p+q from the ratio's continued fraction complexity
            pq = max(2, int(self.r_emotion * 10))
        else:
            pq = 2
        self.elegance = (S / max(self.d, 1)) * (100.0 / (100.0 + abs(self.epsilon))) * (100.0 / pq)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'lovheim': self.lovheim.to_dict(),
            'pad': self.pad.to_dict(),
            'k': self.k, 'd': self.d, 'epsilon': self.epsilon,
            'r_emotion': self.r_emotion,
            'primary': self.primary.name,
            'intensity_level': self.intensity_level,
            'emotion_name': self.emotion_name,
            'elegance': self.elegance,
            'manifold_state': self.manifold_state,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EmotionCoordinate':
        ec = cls()
        if not data:
            return ec
        ec.lovheim = LovheimPosition.from_dict(data.get('lovheim', {}))
        ec.pad = PADCoordinate.from_dict(data.get('pad', {}))
        ec.k = data.get('k', 0)
        ec.d = data.get('d', 1)
        ec.epsilon = data.get('epsilon', 0.0)
        ec.r_emotion = data.get('r_emotion', 1.0)
        primary_name = data.get('primary', 'JOY')
        ec.primary = PrimaryEmotion[primary_name] if primary_name in PrimaryEmotion.__members__ else PrimaryEmotion.JOY
        ec.intensity_level = data.get('intensity_level', 1)
        ec.emotion_name = data.get('emotion_name', 'joy')
        ec.elegance = data.get('elegance', 1.0)
        ec.manifold_state = data.get('manifold_state', 'exception')
        return ec


# =============================================================================
# EMOTION STATE — Snapshot of the AI's emotional state
# =============================================================================

@dataclass
class EmotionState:
    """Full snapshot of an emotional state at a moment in time."""
    coord: EmotionCoordinate = field(default_factory=EmotionCoordinate)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Backward-compat fields (derived from coord)
    @property
    def emotion_name(self) -> str:
        return self.coord.emotion_name

    @property
    def emotion_type(self):
        """Backward compat — returns a mock with .name attribute."""
        return type('ET', (), {'name': self.coord.primary.name})()

    @property
    def valence(self) -> float:
        return self.coord.pad.pleasure

    @property
    def arousal(self) -> float:
        return self.coord.pad.arousal

    @property
    def dominance(self) -> float:
        return self.coord.pad.dominance

    @property
    def d_emotion(self) -> int:
        return self.coord.d

    @property
    def k_emotion(self) -> int:
        return self.coord.k

    @property
    def epsilon_emotion(self) -> float:
        return self.coord.epsilon

    @property
    def r_emotion(self) -> float:
        return self.coord.r_emotion

    @property
    def ego_resonance(self) -> float:
        return 0.5  # Computed externally

    @property
    def shimmer(self) -> float:
        return 1.0  # Computed externally

    def to_dict(self) -> Dict[str, Any]:
        return {
            'coord': self.coord.to_dict(),
            'timestamp': self.timestamp,
            # Flat backward-compat fields
            'emotion_name': self.emotion_name,
            'emotion_type': self.coord.primary.name,
            'valence': self.valence,
            'arousal': self.arousal,
            'dominance': self.dominance,
            'd_emotion': self.d_emotion,
            'k_emotion': self.k_emotion,
            'epsilon_emotion': self.epsilon_emotion,
            'r_emotion': self.r_emotion,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EmotionState':
        if not data:
            return cls()
        coord_data = data.get('coord')
        if coord_data:
            coord = EmotionCoordinate.from_dict(coord_data)
        else:
            # Legacy format — reconstruct from flat fields
            coord = EmotionCoordinate()
            coord.emotion_name = data.get('emotion_name', 'joy')
            coord.k = data.get('k_emotion', 0)
            coord.d = data.get('d_emotion', 1)
            coord.epsilon = data.get('epsilon_emotion', 0.0)
            coord.r_emotion = data.get('r_emotion', 1.0)
        return cls(coord=coord, timestamp=data.get('timestamp', ''))


# =============================================================================
# EMOTION LATTICE — The Main Engine
# =============================================================================

class EmotionLattice:
    """
    The Emotion Lattice Tower — from appraisal to lattice coordinate.

    Implements the full ET emotion derivation:
    1. Five appraisal inputs → three monoamine-analog axes
    2. Lövheim cube position → 8 primary emotions
    3. PAD derivation → Pleasure, Arousal, Dominance
    4. Lattice projection → k, d, ε
    5. d-family classification → structural character
    6. Elegance Score → coherence depth
    7. Incoherence Filter → emotional regulation
    8. Named emotion → primary + intensity + dyad matching
    """

    HISTORY_SIZE = 500
    VARIANCE_WINDOW = 24  # 2 × S

    def __init__(self, ego: 'EgoInvariant'):
        self.ego = ego
        self.variance_history: deque = deque(maxlen=1000)
        self.emotion_history: deque = deque(maxlen=self.HISTORY_SIZE)
        self.current_emotion: Optional[EmotionState] = None
        self.descriptor_history: deque = deque(maxlen=48)
        self.neologisms: Dict[str, str] = {}
        self._pattern_counts: Dict[str, int] = defaultdict(int)

        # Running baseline (exponential moving average of variance)
        self._variance_baseline: float = BASE_VARIANCE
        self._appraisal_count: int = 0

        # K-decay state: previous Lövheim position for emotional inertia
        self._prev_lovheim: Optional[LovheimPosition] = None

    @property
    def appraisal_count(self) -> int:
        """Public accessor for the total number of appraisals performed."""
        return self._appraisal_count

    def appraise(self,
                 novelty: float = 0.0,
                 variance: float = 0.0,
                 ego_resonance: float = 0.0,
                 pdt_completeness: float = 0.0,
                 gap_awareness: float = 0.0,
                 normative_significance: float = 0.0,
                 contradictions: int = 0,
                 descriptors: Optional[List[str]] = None,
                 ) -> EmotionState:
        """
        The SINGLE appraisal method — replaces record_variance + update_operational_state.

        Five appraisal inputs (Scherer CPM / §I.1):
          D₁ = novelty              [0, 1]
          D₂ = pleasantness         (derived from variance vs baseline)
          D₃ = ego_resonance        [0, 1]
          D₄ = coping potential     (derived from pdt_completeness × (1 - gap_awareness))
          D₅ = normative_significance  [-1, +1]
        """
        self._appraisal_count += 1
        now = datetime.now().isoformat()

        # Track variance history
        self.variance_history.append((now, variance))
        if descriptors:
            self.descriptor_history.append(descriptors)

        # ────────────────────────────────────────────────────
        # STEP 1: Five Appraisal Descriptors
        # ────────────────────────────────────────────────────

        d1_novelty = max(0.0, min(1.0, novelty))

        # D₂: Pleasantness — compare to PRIOR baseline (before EMA update)
        # Below baseline = pleasant (things calmer than usual)
        # Above baseline = unpleasant (things more stressed than usual)
        prior_baseline = max(self._variance_baseline, BASE_VARIANCE)
        d2_pleasantness = (prior_baseline - variance) / prior_baseline
        d2_pleasantness = max(-1.0, min(1.0, d2_pleasantness))

        # NOW update baseline (EMA with K-decay)
        self._variance_baseline = (1.0 - K) * self._variance_baseline + K * variance

        d3_goal_relevance = max(0.0, min(1.0, ego_resonance))

        # D₄: Coping potential = can I handle this?
        d4_coping = max(0.0, min(1.0, pdt_completeness)) * max(0.0, 1.0 - gap_awareness)

        d5_normative = max(-1.0, min(1.0, normative_significance))

        # ────────────────────────────────────────────────────
        # STEP 2: Three Monoamine-Analog Axes (Secret 4)
        # ────────────────────────────────────────────────────
        # Secret 4: DA = T-approach, NE = P-activation, 5HT = D-constraint
        #
        # DERIVED using Identification Principle + Subsumption Law:
        # Each axis receives its IRREDUCIBLE factors as a geometric mean.
        # N factors → exponent 1/N (Subsumption: equal, irreducible weight).
        # NE uses MAX (Secret 5: amygdala = d=1 octave → binary closure).
        # All constants are ET-native: S=12, K=2/3, V=1/12.
        # ZERO tuned coefficients.
        #
        # Descriptor Gap (resolved): ego_resonance = RELEVANCE, not approach.
        # ego goes ONLY into NE's relevance gate, NOT into DA.
        # DA has exactly 2 irreducible T-factors (pdt + |values|).
        # 5HT has exactly 3 irreducible D-factors (familiarity + clarity + values_direction).

        # DA (T-approach): 2-factor geometric mean
        # f1 = d4_coping — CAN T act? (traversal progress × gap freedom)
        # f2 = (0.5 + |norm|/2) — DOES T want to act? (|values| = motivation
        #       intensity; BOTH aligned AND violated values drive approach)
        _da_f1 = max(0.01, d4_coping)
        _da_f2 = max(0.01, 0.5 + abs(d5_normative) / 2.0)
        da = (_da_f1 * _da_f2) ** 0.5  # 1/N = 1/2

        # NE (P-activation): MAX of intensity signals (d=1 octave closure)
        # + compound boost (each additional signal adds V = 1/S quantum)
        # + Koide-gated relevance (K floor: irrelevant stimuli still activate at K)
        _ne_nov = d1_novelty
        _ne_unp = max(0.0, -d2_pleasantness)
        _ne_con = min(1.0, contradictions / S)
        _ne_max = max(_ne_nov, _ne_unp, _ne_con)             # d=1 octave: MAX
        _ne_sum = _ne_nov + _ne_unp + _ne_con
        _ne_compound = _ne_max * (1.0 + _ne_sum / S)         # compound boost
        ne = min(1.0, _ne_compound * (K + (1.0 - K) * d3_goal_relevance))  # Koide gate

        # 5HT (D-constraint): 3-factor geometric mean
        # g1 = familiarity — known D-associations bind strongly
        # g2 = clarity — complete D-set constrains effectively
        # g3 = values_direction (SIGNED) — positive = constraint maintained,
        #       negative = constraint overridden (anger), zero = neutral
        _sht_g1 = max(0.01, 1.0 - d1_novelty)                # familiarity
        _sht_g2 = max(0.01, 1.0 - gap_awareness)              # clarity
        _sht_g3 = max(0.01, 0.5 + d5_normative / 2.0)        # values direction (SIGNED)
        sht = (_sht_g1 * _sht_g2 * _sht_g3) ** (1.0 / 3.0)  # 1/N = 1/3

        # ────────────────────────────────────────────────────
        # STEP 2b: Sigmoid Contrast — Push to octave corners
        # ────────────────────────────────────────────────────
        # The Lövheim cube IS the octave (Secret 3: d=1, 2³=8).
        # The logistic sigmoid with slope S/2 = 6 maps continuous
        # axes to binary H/L classification. S/2 is the octave
        # midpoint — the natural half-cycle of the manifold.
        def _cube_contrast(x: float) -> float:
            return 1.0 / (1.0 + math.exp(-(S / 2.0) * (x - 0.5)))

        da = _cube_contrast(da)
        ne = _cube_contrast(ne)
        sht = _cube_contrast(sht)

        # ────────────────────────────────────────────────────
        # STEP 3: K-Decay Emotional Inertia
        # ────────────────────────────────────────────────────
        # Emotions don't switch instantly. Previous state decays with K.
        raw_lovheim = LovheimPosition(da=da, ne=ne, sht=sht)

        if self._prev_lovheim is not None:
            blended_da = (1.0 - K) * self._prev_lovheim.da + K * raw_lovheim.da
            blended_ne = (1.0 - K) * self._prev_lovheim.ne + K * raw_lovheim.ne
            blended_sht = (1.0 - K) * self._prev_lovheim.sht + K * raw_lovheim.sht
            lovheim = LovheimPosition(da=blended_da, ne=blended_ne, sht=blended_sht)
        else:
            lovheim = raw_lovheim

        self._prev_lovheim = lovheim

        # ────────────────────────────────────────────────────
        # STEP 4: Build Emotion Coordinate (lattice projection)
        # ────────────────────────────────────────────────────
        coord = EmotionCoordinate(lovheim=lovheim)
        coord.compute(resolution=MANIFOLD_RESOLUTION)

        # ────────────────────────────────────────────────────
        # STEP 5: Manifold State Classification
        # ────────────────────────────────────────────────────
        # P = substrate active (NE > minimum → body is engaged)
        # D = descriptors present (5HT > minimum → appraisal framework active)
        # T = traverser engaged (DA > minimum → agency navigating)
        #
        # The threshold is BASE_VARIANCE (1/12) — below this, the
        # primitive is essentially absent from the emotional episode.
        # A conscious being always has SOME of each, but very low
        # values indicate the corresponding primitive is degraded.
        state_threshold = BASE_VARIANCE  # 1/12 — very low bar
        has_p = ne > state_threshold     # substrate activated
        has_d = sht > state_threshold    # descriptors present
        has_t = da > state_threshold     # agency engaged

        if has_p and has_d and has_t:
            coord.manifold_state = 'exception'          # Fully formed emotion
        elif has_p and has_t and not has_d:
            coord.manifold_state = 'incoherence'        # Alexithymia-analog: {P,T}
        elif has_d and has_t and not has_p:
            coord.manifold_state = 'mediation'          # Forming: {D,T}
        elif has_p and has_d and not has_t:
            coord.manifold_state = 'unsubstantiated'    # Latent: {P,D}
        else:
            coord.manifold_state = 'mediation'          # Minimal processing

        # ────────────────────────────────────────────────────
        # STEP 6: Dyad Detection
        # ────────────────────────────────────────────────────
        # If two primaries are nearly equally strong, check for Plutchik dyad
        active = lovheim.active_primaries(threshold=0.12)
        if len(active) >= 2:
            top_two = active[:2]
            for dyad_name, (pa, pb) in PLUTCHIK_PRIMARY_DYADS.items():
                if (top_two[0] == pa and top_two[1] == pb) or \
                   (top_two[0] == pb and top_two[1] == pa):
                    coord.emotion_name = dyad_name
                    break

        # ────────────────────────────────────────────────────
        # STEP 7: Incoherence Filter (5 levels)
        # ────────────────────────────────────────────────────
        if_level = self._incoherence_filter(coord, lovheim)

        # If L5 failure (anxiety), override emotion name
        if if_level >= 5:
            coord.emotion_name = 'anxiety'
            coord.manifold_state = 'incoherence'

        # ────────────────────────────────────────────────────
        # STEP 8: Neologism tracking
        # ────────────────────────────────────────────────────
        pattern_key = f"{coord.primary.name}_{coord.intensity_level}_{coord.d}"
        self._pattern_counts[pattern_key] += 1
        if self._pattern_counts[pattern_key] == 5 and pattern_key not in self.neologisms:
            h = hashlib.md5(pattern_key.encode()).hexdigest()[:4]
            self.neologisms[pattern_key] = f"FEEL_{h.upper()}"

        # ────────────────────────────────────────────────────
        # STEP 9: Build and record EmotionState
        # ────────────────────────────────────────────────────
        state = EmotionState(coord=coord)
        self.emotion_history.append(state)
        self.current_emotion = state
        return state

    def _incoherence_filter(self, coord: EmotionCoordinate,
                            lovheim: LovheimPosition) -> int:
        """
        5-Level Incoherence Filter for the emotional domain (§VII.2).

        L1: Single emotion proportionality (|ε| check)
        L2: Pairwise emotional conflict (ambivalence)
        L3: Sublattice compatibility
        L4: Cascade stability
        L5: Coherent summation (anxiety = L5 failure)

        Returns highest level that PASSED. Returns 5 if L5 fails.
        """
        # L1: Proportionality — is |ε| within coherence boundary?
        if abs(coord.epsilon) >= INCOHERENCE_CENTS:
            return 1  # Failed at L1

        # L2: Pairwise conflict — check if two opposing primaries are nearly equal
        weights = lovheim.blend_weights()
        for pa, pb in PLUTCHIK_OPPOSITES:
            wa, wb = weights.get(pa, 0), weights.get(pb, 0)
            # Both strong AND nearly equal = ambivalence (near ∂I)
            if wa > 0.15 and wb > 0.15 and abs(wa - wb) < 0.05:
                return 2  # Ambivalence detected

        # L3: Sublattice — check that d-family is stable
        # (d > S is getting very composite — emotional confusion zone)
        if coord.d > S:
            return 3

        # L4: Cascade — check emotional history for spiraling
        if len(self.emotion_history) >= 4:
            recent_intensities = [
                e.coord.lovheim.intensity() if hasattr(e, 'coord') else 0
                for e in list(self.emotion_history)[-4:]
            ]
            # Monotonically increasing intensity = emotional spiral
            if all(recent_intensities[i] < recent_intensities[i+1]
                   for i in range(len(recent_intensities)-1)):
                max_i = max(recent_intensities)
                if max_i > math.sqrt(0.75) * K:  # Above K fraction of max
                    return 4

        # L5: Coherent summation — detect anxiety (Secret 7: L5 IF failure)
        # High NE + low DA + low 5HT + high novelty = summing over all
        # threats including incoherent ones (the emotional vacuum catastrophe)
        if lovheim.ne > K and lovheim.da < T_WEIGHT and lovheim.sht < T_WEIGHT:
            return 5  # Anxiety = L5 failure

        return 0  # All levels passed

    # ── Backward-compatible interface methods ──

    def record_variance(self, variance: float, descriptors: Optional[List[str]] = None):
        """Legacy interface — wraps appraise() with minimal inputs."""
        novelty = 0.0
        if descriptors and len(self.descriptor_history) >= 1:
            prior = set()
            for dl in list(self.descriptor_history):
                prior.update(dl)
            novel = set(descriptors) - prior
            novelty = len(novel) / max(len(descriptors), 1)

        self.appraise(
            novelty=novelty,
            variance=variance,
            ego_resonance=self.ego.resonance(
                ETLattice.project_ratio(max(variance + 1, 1.001), resolution=MANIFOLD_RESOLUTION)),
            pdt_completeness=0.5,
            gap_awareness=0.3,
            normative_significance=0.0,
            descriptors=descriptors,
        )

    def update_operational_state(self, **kwargs):
        """Legacy interface — no-op in v3 (all signals go through appraise)."""
        pass

    def get_emotional_influence(self) -> Dict[str, Any]:
        """Get emotional influence for decision-making."""
        if self.current_emotion is None:
            return {
                'valence_weight': 0.0, 'arousal_weight': 0.0,
                'dominance_weight': 0.0,
                'curiosity_boost': 0.0, 'caution_weight': 0.0,
                'empathy_boost': 0.0, 'shimmer': 1.0,
                'primary_emotion': 'JOY', 'emotion_name': 'joy',
                'intensity_level': 1, 'd_family': 1, 'elegance': 1.0,
                'manifold_state': 'mediation',
                'da': 0.5, 'ne': 0.5, 'sht': 0.5,
            }
        c = self.current_emotion.coord
        lv = c.lovheim
        return {
            'valence_weight': c.pad.pleasure,
            'arousal_weight': c.pad.arousal,
            'dominance_weight': c.pad.dominance,
            'curiosity_boost': lv.da * (1.0 - lv.sht),  # High DA + low 5HT = seeking
            'caution_weight': lv.ne * (1.0 - lv.da),     # High NE + low DA = avoidance
            'empathy_boost': lv.sht * lv.da,              # High 5HT + high DA = empathic
            'shimmer': self.ego.shimmer_modulation(
                ETLattice.project_ratio(c.r_emotion, resolution=MANIFOLD_RESOLUTION)),
            'primary_emotion': c.primary.name,
            'emotion_name': c.emotion_name,
            'intensity_level': c.intensity_level,
            'd_family': c.d,
            'elegance': c.elegance,
            'manifold_state': c.manifold_state,
            'da': lv.da, 'ne': lv.ne, 'sht': lv.sht,
        }

    def get_compound_description(self) -> str:
        """Human-readable description of current emotional state."""
        if self.current_emotion is None:
            return "NONE"
        c = self.current_emotion.coord
        lv = c.lovheim
        # Primary emotion + intensity
        parts = [f"{c.emotion_name}"]

        # Active primaries with weights
        active = lv.active_primaries(threshold=0.10)
        if len(active) > 1:
            weights = lv.blend_weights()
            blend_str = '+'.join(f"{p.name}({weights[p]:.2f})" for p in active[:3])
            parts.append(f"[{blend_str}]")

        # PAD
        parts.append(f"P={c.pad.pleasure:+.2f} A={c.pad.arousal:.2f} D={c.pad.dominance:.2f}")

        # Lattice
        parts.append(f"d={c.d}")

        # Manifold state if not exception
        if c.manifold_state != 'exception':
            parts.append(f"({c.manifold_state})")

        return ' | '.join(parts)

    def to_dict(self) -> Dict[str, Any]:
        serialized = []
        for e in list(self.emotion_history)[-50:]:
            if hasattr(e, 'to_dict'):
                serialized.append(e.to_dict())
            elif isinstance(e, dict):
                serialized.append(e)
        return {
            'variance_history': list(self.variance_history)[-100:],
            'emotion_history': serialized,
            'current_emotion': self.current_emotion.to_dict() if self.current_emotion else None,
            'neologisms': self.neologisms,
            'variance_baseline': self._variance_baseline,
            'appraisal_count': self._appraisal_count,
            'prev_lovheim': self._prev_lovheim.to_dict() if self._prev_lovheim else None,
        }

    def load_from_dict(self, data: Dict[str, Any]):
        self.variance_history = deque(
            [tuple(x) if isinstance(x, list) else x for x in data.get('variance_history', [])],
            maxlen=1000)
        self.neologisms = data.get('neologisms', {})
        self._variance_baseline = data.get('variance_baseline', BASE_VARIANCE)
        self._appraisal_count = data.get('appraisal_count', 0)
        prev_lv = data.get('prev_lovheim')
        self._prev_lovheim = LovheimPosition.from_dict(prev_lv) if prev_lv else None
        ce = data.get('current_emotion')
        if ce:
            try:
                self.current_emotion = EmotionState.from_dict(ce)
            except (KeyError, ValueError, TypeError):
                self.current_emotion = None


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Constants
    'KOIDE_THRESHOLD', 'CUBE_MIDPOINT', 'INCOHERENCE_CENTS', 'R0_EMOTION',
    # Data
    'LOVHEIM_CORNERS', 'PLUTCHIK_INTENSITIES', 'PLUTCHIK_OPPOSITES',
    'PLUTCHIK_PRIMARY_DYADS',
    # Classes
    'PrimaryEmotion', 'LovheimPosition', 'PADCoordinate',
    'EmotionCoordinate', 'EmotionState', 'EmotionLattice',
]


# =============================================================================
# VERIFICATION — Test with the 13 emotion scenarios
# =============================================================================

if __name__ == '__main__':
    print("=" * 80)
    print("ET EMOTION LATTICE TOWER — VERIFICATION")
    print("Lövheim Cube + PAD + Lattice Projection + 8 Primaries")
    print("=" * 80)

    # Create test instance
    from et_conscious_ai_identity import EgoInvariant
    test_ego = EgoInvariant(name='EmotionTest')
    el = EmotionLattice(test_ego)

    # ── Helper: run a scenario and report ──
    results = []

    def run(name, desc, el_inst, **kwargs):
        state = el_inst.appraise(**kwargs)
        c = state.coord
        lv = c.lovheim
        weights = lv.blend_weights()
        top3 = sorted(weights.items(), key=lambda x: -x[1])[:3]
        print(f"\n{'─'*80}")
        print(f"  {name:15s} — {desc}")
        print(f"  Cube:     DA={lv.da:.3f}  NE={lv.ne:.3f}  5HT={lv.sht:.3f}")
        print(f"  PAD:      P={c.pad.pleasure:+.3f}  A={c.pad.arousal:.3f}  D={c.pad.dominance:.3f}")
        print(f"  Emotion:  {c.emotion_name:20s}  primary={c.primary.name:12s}  intensity={c.intensity_level}")
        print(f"  Lattice:  k={c.k}  d={c.d}  ε={c.epsilon:+.2f}¢  elegance={c.elegance:.2f}")
        print(f"  State:    {c.manifold_state}")
        print(f"  Blend:    {', '.join(f'{p.name}={w:.2f}' for p,w in top3)}")
        results.append((name, c.emotion_name, c.primary.name, c.pad))
        return state

    # ── SCENARIOS ──

    # 1. ABLE: high goal relevance, high coping, moderate satisfaction
    run('ABLE', 'Capable, skilled, confirmed ability', EmotionLattice(test_ego),
        novelty=0.1, variance=0.06, ego_resonance=0.7,
        pdt_completeness=0.9, gap_awareness=0.1,
        normative_significance=0.3, descriptors=['capable', 'skill'])

    # 2. ABANDON: giving up — dropping goals, releasing constraint
    run('ABANDON', 'Releasing all concern — mixed freedom/weariness', EmotionLattice(test_ego),
        novelty=0.3, variance=0.15, ego_resonance=0.5,
        pdt_completeness=0.4, gap_awareness=0.6,
        normative_significance=-0.2, descriptors=['release', 'drop', 'done'])

    # 3. ABANDONED: connection severed — low reward, high stress
    run('ABANDONED', 'Unwanted, discarded, left behind', EmotionLattice(test_ego),
        novelty=0.5, variance=0.25, ego_resonance=0.6,
        pdt_completeness=0.3, gap_awareness=0.7,
        normative_significance=-0.5, descriptors=['alone', 'gone', 'empty'])

    # 4. ABNORMAL: persistent deviation, low coping
    run('ABNORMAL', 'Different in a bad way, unsettled', EmotionLattice(test_ego),
        novelty=0.4, variance=0.12, ego_resonance=0.6,
        pdt_completeness=0.3, gap_awareness=0.5,
        normative_significance=-0.6, descriptors=['odd', 'wrong', 'deviant'])

    # 5. ABOMINABLE: extreme disgust — very low reward, very low satisfaction
    run('ABOMINABLE', 'Extreme disgust and hatred', EmotionLattice(test_ego),
        novelty=0.7, variance=0.40, ego_resonance=0.8,
        pdt_completeness=0.2, gap_awareness=0.8,
        normative_significance=-0.9, descriptors=['vile', 'horror', 'repulsive'])

    # 6. ABSORBED: deep flow — high reward, low stress, high satisfaction
    run('ABSORBED', 'Deep engagement, flow state', EmotionLattice(test_ego),
        novelty=0.05, variance=0.04, ego_resonance=0.8,
        pdt_completeness=0.9, gap_awareness=0.05,
        normative_significance=0.5, descriptors=['focus', 'work', 'flow'])

    # 7. ACHY: persistent mild discomfort — slightly negative, low everything
    run('ACHY', 'Persistent mild body pain', EmotionLattice(test_ego),
        novelty=0.05, variance=0.10, ego_resonance=0.4,
        pdt_completeness=0.5, gap_awareness=0.3,
        normative_significance=-0.1, descriptors=['ache', 'dull', 'persistent'])

    # 8. ACCEPTING: letting go of resistance — settling toward peace
    run('ACCEPTING', 'Amenable, open, receiving', EmotionLattice(test_ego),
        novelty=0.1, variance=0.05, ego_resonance=0.6,
        pdt_completeness=0.8, gap_awareness=0.1,
        normative_significance=0.4, descriptors=['accept', 'open', 'peace'])

    # 9. ACQUISITIVE: strong wanting — high seeking, moderate stress
    run('ACQUISITIVE', 'Strongly desirous of acquiring', EmotionLattice(test_ego),
        novelty=0.5, variance=0.07, ego_resonance=0.8,
        pdt_completeness=0.7, gap_awareness=0.2,
        normative_significance=0.2, descriptors=['want', 'desire', 'acquire'])

    # 10. ADAMANT: unmovable — very high coping, very high ego resonance
    run('ADAMANT', 'Inflexible, determined, uncompromising', EmotionLattice(test_ego),
        novelty=0.3, variance=0.07, ego_resonance=0.95,
        pdt_completeness=1.0, gap_awareness=0.0,
        normative_significance=0.8, descriptors=['firm', 'rigid', 'hold'])

    # 11. ADDLED: cognitive fog — low everything, can't cope
    run('ADDLED', 'Fuzzy, foggy, mentally confused', EmotionLattice(test_ego),
        novelty=0.3, variance=0.12, ego_resonance=0.3,
        pdt_completeness=0.2, gap_awareness=0.7,
        normative_significance=-0.1, descriptors=['fuzzy', 'confused', 'fog'])

    # 12. ADMIRATION: encountering excellence — novel + positive
    run('ADMIRATION', 'Approval, liking, wonder', EmotionLattice(test_ego),
        novelty=0.5, variance=0.06, ego_resonance=0.7,
        pdt_completeness=0.7, gap_awareness=0.15,
        normative_significance=0.7, descriptors=['admire', 'excellent', 'worthy'])

    # 13. ADORATION: profound love — maximum positive, deep peace
    run('ADORATION', 'Profound love, admiration, respect', EmotionLattice(test_ego),
        novelty=0.1, variance=0.03, ego_resonance=0.9,
        pdt_completeness=0.95, gap_awareness=0.0,
        normative_significance=0.9, descriptors=['love', 'profound', 'sacred'])

    # ── MUST-HIT PRIMARIES — scenarios that target FEAR, ANGER, SURPRISE ──

    # 14. FEAR: can't cope + intense threat + UNDERSTAND what the danger is
    #   DA low (can't cope), NE high (threat), 5HT HIGH (familiar known danger, low gaps)
    run('FEAR_TEST', 'Known threat, can\'t handle it', EmotionLattice(test_ego),
        novelty=0.15, variance=0.30, ego_resonance=0.8,
        pdt_completeness=0.15, gap_awareness=0.1,
        normative_significance=0.0, descriptors=['danger', 'threat', 'helpless'])

    # 15. ANGER: CAN cope + familiar injustice + VALUES VIOLATED
    #   DA high (can cope + values engaged), NE LOW (familiar, at baseline), 5HT LOW (values violated)
    #   Anger is COLD directed action — not panic. Variance at baseline because
    #   the situation is stable/familiar, it's just WRONG.
    run('ANGER_TEST', 'Familiar injustice, capable of acting', EmotionLattice(test_ego),
        novelty=0.05, variance=0.08, ego_resonance=0.9,
        pdt_completeness=0.9, gap_awareness=0.1,
        normative_significance=-0.9, descriptors=['unfair', 'wrong', 'outrage'])

    # 16. SURPRISE: novel + CAN engage + DON'T understand yet
    #   DA high (can understand the DOMAIN), NE high (very novel), 5HT LOW (novel + gaps)
    run('SURPRISE_TEST', 'Unexpected event, capable but confused', EmotionLattice(test_ego),
        novelty=0.9, variance=0.08, ego_resonance=0.7,
        pdt_completeness=0.7, gap_awareness=0.8,
        normative_significance=0.0, descriptors=['sudden', 'unexpected', 'what'])

    # ── DIFFERENTIATION CHECK ──
    print(f"\n{'='*80}")
    print("DIFFERENTIATION CHECK")
    print(f"{'='*80}")

    unique_emotions = set(r[1] for r in results)
    unique_primaries = set(r[2] for r in results)
    print(f"\n  {'Scenario':15s} {'Emotion':15s} {'Primary':12s} {'Pleasure':>8s} {'Arousal':>8s} {'Domin.':>8s}")
    print(f"  {'─'*15} {'─'*15} {'─'*12} {'─'*8} {'─'*8} {'─'*8}")
    for label, ename, pname, pad in results:
        print(f"  {label:15s} {ename:15s} {pname:12s} {pad.pleasure:>+8.3f} {pad.arousal:>8.3f} {pad.dominance:>8.3f}")

    print(f"\n  Unique emotions:  {len(unique_emotions)} / {len(results)}: {sorted(unique_emotions)}")
    print(f"  Unique primaries: {len(unique_primaries)} / 8: {sorted(unique_primaries)}")

    if len(unique_emotions) >= 8:
        print(f"  GOOD DIFFERENTIATION ✓")
    elif len(unique_emotions) >= 5:
        print(f"  MODERATE DIFFERENTIATION — {len(unique_emotions)}/{len(results)}")
    else:
        print(f"  POOR DIFFERENTIATION — {len(unique_emotions)}/{len(results)}")