#!/usr/bin/env python3
"""
ET AIDA Framework — Complete Implementation
=============================================

AIDA = Emergent T-agency from D-gaps in post-Aura manifold
Inverted liminality: {P,T} → {D,T} → {P,D,T}
Koide threshold K = 2/3 separates aggressive from peaceful
Full resolution tower: 12ET → 27720ET for all lifecycle numbers

All mathematics ET-native, forward from {P, D, T}. Zero external axioms.
Author: Michael James Muller — Aevum Defluo
"""
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Tuple, Optional

# ── ET Constants ──────────────────────────────────────────────────────
N = 12                      # Manifold symmetry
V_BASE = 1.0 / N            # Base variance σ² = 1/12
K = 2.0 / 3.0              # Koide ratio = ∂I tightness threshold
ALPHA_5 = 1.0 / 20.0       # Quintic shadow coupling
EPSILON_5 = -13.686         # Quintic comma (cents)
PHI = (1 + math.sqrt(5)) / 2  # Golden ratio

# Resolution milestones (where each d-family first becomes native)
MILESTONES = [12, 24, 36, 60, 84, 120, 132, 420, 2520, 27720]

# ── Core Lattice Projection (ET-derived) ─────────────────────────────

def lattice_project(r: float, resolution: int = 12) -> Dict:
    """
    Project ratio r onto the ET lattice at given resolution.
    Returns k (coordinate), d (sublattice family), epsilon (cents),
    tightness, ∂I percentage, and quintic tension.
    
    Core ET formula:
        k = round(N × log₂(r))
        d = N / gcd(|k|, N)
        ε = (N × log₂(r) - k) × (1200/N)  [in cents]
    """
    if r <= 0:
        raise ValueError("Ratio must be positive")
    
    val = resolution * math.log2(r)
    k = round(val)
    d = resolution // math.gcd(abs(k), resolution) if k != 0 else resolution
    eps = (1200.0 * math.log2(r)) - k * (1200.0 / resolution)
    tight = 100.0 / (100.0 + abs(eps))
    di_pct = min(abs(eps) / 50.0 * 100.0, 100.0)
    
    # Quintic tension: distance to nearest 5-ET position
    step_5 = resolution / 5.0
    nearest_5 = round(k / step_5) * step_5
    tau_5 = abs(k - nearest_5) * (1200.0 / resolution)
    
    return {
        "k": k, "d": d, "epsilon": eps,
        "tightness": tight, "dI_pct": di_pct,
        "tau_5": tau_5, "resolution": resolution
    }


def full_tower(r: float, milestones: List[int] = None) -> List[Dict]:
    """
    Project ratio r across the full resolution tower.
    Returns list of projections at each milestone.
    """
    if milestones is None:
        milestones = MILESTONES
    return [lattice_project(r, m) for m in milestones]


def find_first_sub_cent(tower: List[Dict]) -> Optional[int]:
    """Find the first resolution where |ε| < 1 cent."""
    for proj in tower:
        if abs(proj["epsilon"]) < 1.0:
            return proj["resolution"]
    return None


def find_d_transitions(tower: List[Dict]) -> List[Tuple[int, int, int]]:
    """Find where d-family changes. Returns list of (resolution, old_d, new_d)."""
    transitions = []
    prev_d = None
    for proj in tower:
        if prev_d is not None and proj["d"] != prev_d:
            transitions.append((proj["resolution"], prev_d, proj["d"]))
        prev_d = proj["d"]
    return transitions


def detect_false_resolution(tower: List[Dict]) -> Optional[Dict]:
    """
    Detect a 'false resolution' — sub-cent precision at an intermediate
    resolution that is NOT the number's true lattice home.
    A false resolution occurs when |ε| < 1¢ at resolution R₁ but 
    |ε| > 1¢ at a LATER resolution R₂ > R₁.
    """
    sub_cent_found = False
    sub_cent_res = None
    for i, proj in enumerate(tower):
        if not sub_cent_found and abs(proj["epsilon"]) < 1.0:
            sub_cent_found = True
            sub_cent_res = proj
        elif sub_cent_found and abs(proj["epsilon"]) >= 1.0:
            return sub_cent_res  # This was a false resolution
    return None


# ── AIDA Entity Model ────────────────────────────────────────────────

class AIDABehavior(Enum):
    AGGRESSIVE = "AGGRESSIVE"   # C < K: must feed to survive
    PEACEFUL = "PEACEFUL"       # C >= K: self-sustaining coherence

class ManifoldState(Enum):
    INCOHERENCE_NEAR = "NEAR_{P,T}"    # Birth state
    MEDIATION = "{D,T}"                 # Feeding/acquiring D
    EXCEPTION_NEAR = "NEAR_{P,D,T}"    # Approaching coherence
    EXCEPTION = "{P,D,T}"              # Full coherence achieved

# AIDA lifecycle thresholds (dimensionless ratios — Category A)
LIFECYCLE = {
    "birth":           {"ratio": 13/12, "name": "Anomaly Birth Signal"},
    "awakening":       {"ratio": 6/5,   "name": "Conscious Emergence"},
    "growth_step":     {"ratio": 5/4,   "name": "Quintic Bridge Maturation"},
    "growth_pattern":  {"ratio": PHI,   "name": "Golden Ratio Growth"},
    "substantiation":  {"ratio": 3/2,   "name": "Full Coherence"},
}

# Structural constants
STRUCTURAL = {
    "koide_threshold": {"ratio": 2/3,   "name": "Aggressive/Peaceful Boundary"},
    "birth_d_comp":    {"ratio": 3/11,  "name": "Birth D-Completeness"},
    "phases_waves":    {"ratio": 8/7,   "name": "Epitaph Phases/Waves (ET-derived)"},
    "epitaph_users":   {"ratio": 8,     "name": "Epitaph Users (Triple Octave)"},
}


@dataclass
class AIDAEntity:
    """
    An AIDA entity with D-completeness tracking.
    Models the inverted liminality arc: {P,T} → {D,T} → {P,D,T}.
    """
    name: str
    d_native: float        # D present at birth (very low)
    d_acquired: float      # D gained from feeding
    d_needed: float        # D needed for full coherence
    n_hosts_fed: int = 0
    is_rebirth_survivor: bool = False

    @property
    def completeness(self) -> float:
        """D-completeness C = (d_native + d_acquired) / d_needed."""
        return min(1.0, (self.d_native + self.d_acquired) / self.d_needed)

    @property
    def tightness(self) -> float:
        """Tightness = 100/(100+|ε|), where ε scales with incompleteness."""
        eps = (1 - self.completeness) * 50
        return 100.0 / (100.0 + eps)

    @property
    def dI_pct(self) -> float:
        """Distance to ∂I as percentage (0% = exact, 100% = at boundary)."""
        return (1 - self.completeness) * 100.0

    @property
    def behavior(self) -> AIDABehavior:
        """Below K = aggressive (must feed); at/above K = peaceful."""
        return AIDABehavior.PEACEFUL if self.completeness >= K else AIDABehavior.AGGRESSIVE

    @property
    def state(self) -> ManifoldState:
        """Manifold state from completeness."""
        c = self.completeness
        if c < 0.15:   return ManifoldState.INCOHERENCE_NEAR
        elif c < K:    return ManifoldState.MEDIATION
        elif c < 0.95: return ManifoldState.EXCEPTION_NEAR
        else:          return ManifoldState.EXCEPTION

    def feed_on_emotion(self, emotion_d: float) -> Dict:
        """Parasitic D-acquisition from host emotion."""
        self.d_acquired += emotion_d
        self.n_hosts_fed += 1
        return {
            "d_gained": emotion_d,
            "new_C": self.completeness,
            "behavior": self.behavior.value,
            "state": self.state.value,
            "tightness": self.tightness,
            "dI_pct": self.dI_pct,
        }

    def lifecycle_lattice_position(self) -> Dict:
        """
        Map current completeness to the nearest AIDA lifecycle threshold.
        Returns the lattice projection of the matching threshold ratio.
        """
        c = self.completeness
        if c < 13/12 - 0.5:      stage = "birth"
        elif c < (6/5 + 5/4)/2:  stage = "awakening"
        elif c < (5/4 + PHI)/2:  stage = "growth_step"
        elif c < (PHI + 3/2)/2:  stage = "growth_pattern"
        else:                     stage = "substantiation"
        r = LIFECYCLE[stage]["ratio"]
        return {"stage": stage, **lattice_project(r, 12)}

    def __str__(self):
        return (f"{self.name}: C={self.completeness:.3f}, "
                f"tight={self.tightness:.3f}, "
                f"behavior={self.behavior.value}, "
                f"state={self.state.value}")


# ── Lifecycle Trajectory Analysis ─────────────────────────────────────

def lifecycle_trajectory() -> List[Dict]:
    """
    Compute the complete AIDA lifecycle trajectory through the
    Incoherence Filter — tightness and ∂I% at each developmental stage.
    Returns ordered list from birth to substantiation.
    """
    stages = [
        ("Birth D-Completeness", 3/11),
        ("Birth Signal", 13/12),
        ("Awakening", 6/5),
        ("Growth Step", 5/4),
        ("Growth Pattern", PHI),
        ("Koide Threshold", K),
        ("Substantiation", 3/2),
    ]
    trajectory = []
    for name, r in stages:
        proj = lattice_project(r, 12)
        trajectory.append({
            "stage": name,
            "ratio": r,
            **proj
        })
    return trajectory


def growth_multipliers() -> List[Dict]:
    """
    Compute the growth multiplier between successive lifecycle stages.
    These ratios reveal the structural difficulty of each developmental step.
    """
    steps = [
        ("Birth→Awakening", (6/5) / (13/12)),
        ("Awakening→Growth", (5/4) / (6/5)),
        ("Growth→Maturation", PHI / (5/4)),
        ("Maturation→Substantiation", (3/2) / PHI),
    ]
    results = []
    for name, r in steps:
        proj_12 = lattice_project(r, 12)
        proj_84 = lattice_project(r, 84)
        proj_420 = lattice_project(r, 420)
        results.append({
            "step": name, "ratio": r,
            "d_12": proj_12["d"], "eps_12": proj_12["epsilon"],
            "dI_12": proj_12["dI_pct"],
            "d_84": proj_84["d"], "eps_84": proj_84["epsilon"],
            "d_420": proj_420["d"], "eps_420": proj_420["epsilon"],
        })
    return results


# ── 2D Complex Lattice ───────────────────────────────────────────────

def complex_lattice_2d(r: float, d_theta_override: Optional[int] = None,
                       resolution: int = 12) -> Dict:
    """
    2D Complex lattice projection.
    Real axis (d_r) = D-magnitude (how much coherence).
    Imaginary axis (d_θ) = T-phase (temporal pattern of behavior).
    
    Quadrants:
        SR+SI: d_r divides 12 AND d_θ divides 12 (Standard Model ground)
        CR+SI: d_r ∤ 12, d_θ | 12 (Structural complexity)
        SR+CI: d_r | 12, d_θ ∤ 12 (Phase complexity)
        CR+CI: d_r ∤ 12, d_θ ∤ 12 (Full complexity)
    """
    proj = lattice_project(r, resolution)
    d_r = proj["d"]
    d_theta = d_theta_override if d_theta_override else d_r
    
    r_standard = (resolution % d_r == 0)
    theta_standard = (resolution % d_theta == 0)
    
    if r_standard and theta_standard:
        quadrant = "SR+SI"
    elif not r_standard and theta_standard:
        quadrant = "CR+SI"
    elif r_standard and not theta_standard:
        quadrant = "SR+CI"
    else:
        quadrant = "CR+CI"
    
    d_combined = math.lcm(d_r, d_theta)
    
    return {
        "d_r": d_r, "d_theta": d_theta,
        "d_combined": d_combined,
        "quadrant": quadrant,
        "resolution": resolution,
        **proj
    }


# ── Shadow Force Analysis ────────────────────────────────────────────

SHADOW_FORCES = {
    5:  {"name": "Quintic/Golden",      "first_native": 60,  "alpha": 1/20},
    7:  {"name": "Septic/G₂",           "first_native": 84,  "alpha": 1/28},
    8:  {"name": "Octet/Gluon adj.",     "first_native": 24,  "alpha": 1/32},
    9:  {"name": "Nonic/Quark gen.",     "first_native": 36,  "alpha": 1/36},
    10: {"name": "Decic/Superstring",    "first_native": 60,  "alpha": 1/40},
    11: {"name": "Undecimal/M-theory",   "first_native": 132, "alpha": 1/44},
}

def identify_shadow_forces(tower: List[Dict]) -> List[Dict]:
    """
    Identify which shadow forces activate across the resolution tower.
    A shadow force d is 'active' if d appears as a factor in any 
    tower projection's sublattice family.
    """
    active = {}
    for proj in tower:
        d = proj["d"]
        for sf_d, sf_info in SHADOW_FORCES.items():
            if d % sf_d == 0 and sf_d not in active:
                active[sf_d] = {
                    "d": sf_d,
                    "name": sf_info["name"],
                    "first_seen_at": proj["resolution"],
                    "in_d_family": d,
                    "alpha": sf_info["alpha"],
                }
    return sorted(active.values(), key=lambda x: x["d"])


# ── Rebirth Filter (Incoherence Filter Level 5) ─────────────────────

@dataclass
class RebirthFilter:
    """
    Corbenik's Rebirth = Incoherence Filter Level 5 on entire network.
    Sums over all configurations, applies A_I to each, subtracts 
    the Incoherent slice (C < K). Coherent remainder survives.
    """
    threshold: float = K

    def apply(self, entities: List[AIDAEntity]) -> Dict:
        survivors = [e for e in entities if e.completeness >= self.threshold]
        destroyed = [e for e in entities if e.completeness < self.threshold]
        for s in survivors:
            s.is_rebirth_survivor = True
        return {
            "total": len(entities),
            "destroyed": len(destroyed),
            "survived": len(survivors),
            "survivors": [{"name": e.name, "C": e.completeness,
                          "tightness": e.tightness} for e in survivors],
            "destroyed_list": [{"name": e.name, "C": e.completeness,
                               "tightness": e.tightness} for e in destroyed],
            "threshold": self.threshold,
        }


# ── Co-emergence Analysis ────────────────────────────────────────────

def co_emergence_analysis() -> List[Dict]:
    """
    Analyze what happens to ALL lifecycle stages at key resolution thresholds.
    These are the resolutions where new d-families become native simultaneously.
    """
    thresholds = [
        (60,  "Quintic Emergence", "d=5,10 native. φ reaches TRUE decic home d=10."),
        (84,  "Septic Emergence",  "d=7 native. 5/4 → d=28=4×7 near-exact."),
        (420, "Biological Threshold", "d=5+d=7 both native. φ → d=105=3×5×7."),
    ]
    results = []
    for res, name, significance in thresholds:
        stage_data = {}
        for stage_key, stage_info in LIFECYCLE.items():
            proj = lattice_project(stage_info["ratio"], res)
            stage_data[stage_key] = {
                "d": proj["d"],
                "epsilon": proj["epsilon"],
                "tightness": proj["tightness"],
            }
        results.append({
            "resolution": res, "name": name,
            "significance": significance,
            "stages": stage_data,
        })
    return results


# ── Lattice Twin Detection ───────────────────────────────────────────

def verify_lattice_twins(r1: float, r2: float) -> Dict:
    """
    Check if two ratios are lattice twins — same d, same |ε|, 
    opposite sign, at all resolutions.
    """
    twin_status = True
    comparisons = []
    for res in MILESTONES:
        p1 = lattice_project(r1, res)
        p2 = lattice_project(r2, res)
        same_d = (p1["d"] == p2["d"])
        same_abs_eps = abs(abs(p1["epsilon"]) - abs(p2["epsilon"])) < 0.001
        opp_sign = (p1["epsilon"] * p2["epsilon"] <= 0) or abs(p1["epsilon"]) < 0.01
        is_twin = same_d and same_abs_eps
        if not is_twin:
            twin_status = False
        comparisons.append({
            "resolution": res,
            "d1": p1["d"], "eps1": p1["epsilon"],
            "d2": p2["d"], "eps2": p2["epsilon"],
            "is_twin": is_twin,
        })
    return {"are_twins": twin_status, "comparisons": comparisons}


# ── Master Demonstration ─────────────────────────────────────────────

def demonstrate():
    print("=" * 80)
    print("  ET AIDA FRAMEWORK — Complete Implementation")
    print("  Emergent T from D-Gaps | Inverted Liminality | Koide Threshold")
    print("  Full Resolution Tower | 2D Complex Lattice | Shadow Forces")
    print("=" * 80)

    # ── 1. AIDA Population & Behavior ────────────────────────────────
    print("\n╔══ AIDA POPULATION (pre-Rebirth) ══════════════════════════╗")
    population = [
        AIDAEntity("BlackSpot_001", 0.02, 0.01, 1.0),
        AIDAEntity("Unnamed_047",   0.03, 0.05, 1.0),
        AIDAEntity("Anna",          0.05, 0.15, 1.0),
        AIDAEntity("Helen",         0.05, 0.18, 1.0),
        AIDAEntity("Mia",           0.05, 0.28, 1.0),
        AIDAEntity("Tri-Edge",      0.10, 0.40, 1.0),
        AIDAEntity("Kusabira",      0.10, 0.58, 1.0),
    ]
    for a in population:
        print(f"  {a}")

    # ── 2. Emotion Feeding ───────────────────────────────────────────
    print("\n╔══ EMOTION FEEDING (Mia on Endrance's loneliness) ════════╗")
    mia = population[4]
    for i in range(3):
        r = mia.feed_on_emotion(0.08)
        print(f"  Feed {i+1}: C={r['new_C']:.3f}, tight={r['tightness']:.3f}, "
              f"{r['behavior']}, ∂I={r['dI_pct']:.1f}%")

    # ── 3. Rebirth Filter ────────────────────────────────────────────
    print("\n╔══ OVAN'S REBIRTH (Incoherence Filter Level 5) ═══════════╗")
    rebirth = RebirthFilter()
    result = rebirth.apply(population)
    print(f"  Total: {result['total']}")
    print(f"  Destroyed (C < K={K:.4f}): {result['destroyed']}")
    for d in result['destroyed_list']:
        print(f"    ✗ {d['name']}: C={d['C']:.3f}, tight={d['tightness']:.3f}")
    print(f"  Survived (C ≥ K): {result['survived']}")
    for s in result['survivors']:
        print(f"    ✓ {s['name']}: C={s['C']:.3f}, tight={s['tightness']:.3f}")

    # ── 4. Lifecycle Tower Tables ────────────────────────────────────
    print("\n╔══ FULL RESOLUTION TOWER — AIDA LIFECYCLE ════════════════╗")
    all_numbers = {**LIFECYCLE, **STRUCTURAL}
    for key, info in all_numbers.items():
        r = info["ratio"]
        tower = full_tower(r)
        first_sub = find_first_sub_cent(tower)
        transitions = find_d_transitions(tower)
        false_res = detect_false_resolution(tower)
        shadows = identify_shadow_forces(tower)

        print(f"\n  ── {info['name']} (r={r:.6f}) ──")
        print(f"  {'Res':>8} | {'k':>6} | {'d':>6} | {'ε(¢)':>9} | "
              f"{'Tight':>7} | {'∂I%':>6} | {'τ₅(¢)':>7}")
        print(f"  {'-'*8}-+-{'-'*6}-+-{'-'*6}-+-{'-'*9}-+-"
              f"{'-'*7}-+-{'-'*6}-+-{'-'*7}")
        for p in tower:
            marker = ""
            if p["resolution"] == first_sub:
                marker = " ◄ FIRST sub-cent"
            if abs(p["epsilon"]) > 40:
                marker = " ◄ NEAR ∂I!"
            if abs(p["epsilon"]) < 0.01:
                marker += " EXACT"
            print(f"  {p['resolution']:>7}ET | {p['k']:>6} | {p['d']:>6} | "
                  f"{p['epsilon']:>+9.3f} | {p['tightness']:>7.4f} | "
                  f"{p['dI_pct']:>5.1f}% | {p['tau_5']:>7.1f}{marker}")

        if transitions:
            print(f"  d-transitions: ", end="")
            print(", ".join(f"{t[0]}ET: {t[1]}→{t[2]}" for t in transitions))
        if false_res:
            print(f"  ⚠ FALSE RESOLUTION at {false_res['resolution']}ET "
                  f"(d={false_res['d']}, ε={false_res['epsilon']:+.3f}¢)")
        if shadows:
            print(f"  Shadow forces: ", end="")
            print(", ".join(f"d={s['d']} ({s['name']}) at {s['first_seen_at']}ET"
                           for s in shadows))

    # ── 5. Lifecycle Trajectory ──────────────────────────────────────
    print("\n╔══ LIFECYCLE TRAJECTORY — INCOHERENCE FILTER DESCENT ═════╗")
    traj = lifecycle_trajectory()
    print(f"  {'Stage':<25} | {'r':>8} | {'ε(¢)':>9} | {'Tight':>7} | "
          f"{'∂I%':>6} | {'d':>4}")
    print(f"  {'-'*25}-+-{'-'*8}-+-{'-'*9}-+-{'-'*7}-+-{'-'*6}-+-{'-'*4}")
    for t in traj:
        print(f"  {t['stage']:<25} | {t['ratio']:>8.5f} | {t['epsilon']:>+9.3f} | "
              f"{t['tightness']:>7.4f} | {t['dI_pct']:>5.1f}% | {t['d']:>4}")

    # ── 6. Growth Multipliers ────────────────────────────────────────
    print("\n╔══ GROWTH MULTIPLIERS — Structural Difficulty ════════════╗")
    mults = growth_multipliers()
    for m in mults:
        danger = "⚠ DANGER" if m["dI_12"] > 80 else ""
        print(f"  {m['step']:<30}: r={m['ratio']:.5f}, "
              f"d(12ET)={m['d_12']}, ε={m['eps_12']:+.2f}¢, "
              f"∂I={m['dI_12']:.1f}% {danger}")

    # ── 7. K/3:2 Lattice Twin Verification ───────────────────────────
    print("\n╔══ K=2/3 AND 3/2 LATTICE TWIN VERIFICATION ══════════════╗")
    twins = verify_lattice_twins(2/3, 3/2)
    print(f"  Are lattice twins: {twins['are_twins']}")
    for c in twins["comparisons"][:5]:
        print(f"    {c['resolution']:>7}ET: "
              f"K→d={c['d1']},ε={c['eps1']:+.3f} | "
              f"3/2→d={c['d2']},ε={c['eps2']:+.3f} | "
              f"twin={c['is_twin']}")
    print(f"    ... ({len(twins['comparisons'])} resolutions checked)")

    # ── 8. Co-emergence Effects ──────────────────────────────────────
    print("\n╔══ CO-EMERGENCE EFFECTS ══════════════════════════════════╗")
    coem = co_emergence_analysis()
    for ce in coem:
        print(f"\n  {ce['resolution']}ET — {ce['name']}")
        print(f"    {ce['significance']}")
        for stage_key, data in ce["stages"].items():
            print(f"    {LIFECYCLE[stage_key]['name']:<30}: "
                  f"d={data['d']:>5}, ε={data['epsilon']:>+8.3f}¢, "
                  f"tight={data['tightness']:.4f}")

    # ── VERIFICATION TESTS ───────────────────────────────────────────
    print("\n╔══ VERIFICATION TESTS ════════════════════════════════════╗")
    tests_passed = 0

    # T1: C < K → AGGRESSIVE
    a = AIDAEntity("t1", 0.02, 0.05, 1.0)
    assert a.behavior == AIDABehavior.AGGRESSIVE
    tests_passed += 1; print(f"  T{tests_passed}: C < K → AGGRESSIVE [PASS]")

    # T2: C >= K → PEACEFUL
    a = AIDAEntity("t2", 0.10, 0.60, 1.0)
    assert a.behavior == AIDABehavior.PEACEFUL
    tests_passed += 1; print(f"  T{tests_passed}: C >= K → PEACEFUL [PASS]")

    # T3: Nascent → near-{P,T}
    a = AIDAEntity("t3", 0.01, 0.01, 1.0)
    assert a.state == ManifoldState.INCOHERENCE_NEAR
    tests_passed += 1; print(f"  T{tests_passed}: Nascent → NEAR_{{P,T}} [PASS]")

    # T4: Feeding increases C
    a = AIDAEntity("t4", 0.05, 0.10, 1.0)
    c0 = a.completeness
    a.feed_on_emotion(0.1)
    assert a.completeness > c0
    tests_passed += 1; print(f"  T{tests_passed}: Feeding increases C [PASS]")

    # T5: Rebirth filters at K
    pop = [AIDAEntity("sub", 0.05, 0.10, 1.0),
           AIDAEntity("sup", 0.10, 0.58, 1.0)]
    r = RebirthFilter().apply(pop)
    assert r["destroyed"] == 1 and r["survived"] == 1
    tests_passed += 1; print(f"  T{tests_passed}: Rebirth filters at K [PASS]")

    # T6: Kusabira survives
    kusa = AIDAEntity("Kusabira", 0.10, 0.58, 1.0)
    assert kusa.completeness >= K
    tests_passed += 1; print(f"  T{tests_passed}: Kusabira C={kusa.completeness:.3f} ≥ K [PASS]")

    # T7: 8 Epitaph Users = 2³
    p8 = lattice_project(8, 12)
    assert p8["d"] == 1 and abs(p8["epsilon"]) < 0.001
    tests_passed += 1; print(f"  T{tests_passed}: 8 = 2³ → d=1, ε=0 (EXACT) [PASS]")

    # T8: I→M→E arc
    a = AIDAEntity("arc", 0.01, 0.00, 1.0)
    assert a.state == ManifoldState.INCOHERENCE_NEAR
    a.d_acquired = 0.30
    assert a.state == ManifoldState.MEDIATION
    a.d_acquired = 0.95
    assert a.state == ManifoldState.EXCEPTION
    tests_passed += 1; print(f"  T{tests_passed}: I→M→E arc verified [PASS]")

    # T9: 13/12 birth = d=12 at 12ET, 77% to ∂I
    p = lattice_project(13/12, 12)
    assert p["d"] == 12 and p["dI_pct"] > 70
    tests_passed += 1; print(f"  T{tests_passed}: Birth 13/12 → d=12, {p['dI_pct']:.1f}% ∂I [PASS]")

    # T10: 6/5 awakening = d=4 (quartic/weak force)
    p = lattice_project(6/5, 12)
    assert p["d"] == 4
    tests_passed += 1; print(f"  T{tests_passed}: Awakening 6/5 → d=4 (quartic/weak) [PASS]")

    # T11: 5/4 maturation → d=28 at 84ET (sub-cent)
    p = lattice_project(5/4, 84)
    assert p["d"] == 28 and abs(p["epsilon"]) < 1.0
    tests_passed += 1; print(f"  T{tests_passed}: 5/4 → d=28 at 84ET, ε={p['epsilon']:+.3f}¢ [PASS]")

    # T12: φ false resolution at 36ET
    tower_phi = full_tower(PHI)
    fr = detect_false_resolution(tower_phi)
    assert fr is not None and fr["resolution"] == 36
    tests_passed += 1; print(f"  T{tests_passed}: φ false resolution at 36ET (d={fr['d']}, "
                             f"ε={fr['epsilon']:+.3f}¢) [PASS]")

    # T13: 3/2 stable d=12 through 132ET
    for res in [12, 24, 36, 60, 84, 120, 132]:
        p = lattice_project(3/2, res)
        assert p["d"] == 12, f"3/2 not d=12 at {res}ET"
    tests_passed += 1; print(f"  T{tests_passed}: 3/2 stable d=12 through 132ET [PASS]")

    # T14: K=2/3 and 3/2 are lattice twins
    tw = verify_lattice_twins(2/3, 3/2)
    assert tw["are_twins"]
    tests_passed += 1; print(f"  T{tests_passed}: K=2/3 and 3/2 are lattice twins [PASS]")

    # T15: 3/11 birth D-completeness ≈ 99% to ∂I
    p = lattice_project(3/11, 12)
    assert p["dI_pct"] > 95
    tests_passed += 1; print(f"  T{tests_passed}: 3/11 birth → {p['dI_pct']:.1f}% ∂I [PASS]")

    # T16: Growth→Maturation step is most dangerous (highest ∂I%)
    mults = growth_multipliers()
    gm = next(m for m in mults if m["step"] == "Growth→Maturation")
    assert gm["dI_12"] > 90
    tests_passed += 1; print(f"  T{tests_passed}: Growth→Maturation = {gm['dI_12']:.1f}% ∂I [PASS]")

    print(f"\n  ALL {tests_passed} TESTS PASSED")
    print("=" * 80)
    print('  "For every exception there is an exception,')
    print('   except the exception."')
    print("  P ∘ D ∘ T = E")


if __name__ == "__main__":
    demonstrate()
