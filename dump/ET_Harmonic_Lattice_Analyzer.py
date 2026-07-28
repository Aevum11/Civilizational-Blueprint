#!/usr/bin/env python3
"""
ET HARMONIC LATTICE ANALYZER v2
Sempaevum Bijection — Music Industry Tool
1200-Digit Precision · All 12 Families · Scala Export · MIDI · Bidirectional

All mathematics ET-native, derived from {P, D, T}. Zero float. Zero ad hoc.
Author: Michael James Muller — Aevum Defluo (Exception Theory)

USAGE:
  python3 ET_Harmonic_Lattice_Analyzer.py               # Full demo
  python3 ET_Harmonic_Lattice_Analyzer.py note 440       # Analyze Hz
  python3 ET_Harmonic_Lattice_Analyzer.py ratio 3/2      # Analyze ratio
  python3 ET_Harmonic_Lattice_Analyzer.py midi 60         # MIDI → lattice
  python3 ET_Harmonic_Lattice_Analyzer.py harmonics 110 16
  python3 ET_Harmonic_Lattice_Analyzer.py scala ji5       # Export .scl
  python3 ET_Harmonic_Lattice_Analyzer.py scale shruti    # Analyze scale
  python3 ET_Harmonic_Lattice_Analyzer.py lattice2freq 7 1.955  # (k,eps)->Hz
  python3 ET_Harmonic_Lattice_Analyzer.py families        # All 12 families
"""

import mpmath
import sys
import os
from math import gcd
from fractions import Fraction

# ═══════════════════════════════════════════════════════════════
# PRECISION — 1200 target + 15 guard. NO FLOAT ANYWHERE.
# ═══════════════════════════════════════════════════════════════
TARGET_DIGITS = 1200
mpmath.mp.dps = TARGET_DIGITS + 15

# ═══════════════════════════════════════════════════════════════
# ET CONSTANTS — derived from P∘D∘T = E. All forced.
# ═══════════════════════════════════════════════════════════════
N = 12
S = 4
V_BASE = mpmath.mpf(1) / mpmath.mpf(N)
K_KOIDE = mpmath.mpf(2) / mpmath.mpf(3)
A0_LOCAL = (N - 1)**2 + S**2  # = 137
N_FULL = 27720
CONCERT_A4 = mpmath.mpf("440")
MIDI_A4 = 69

def compute_lcm_tower():
    tower = []
    current_lcm = 1
    for n in range(1, 50):
        current_lcm = current_lcm * n // gcd(current_lcm, n)
        if n >= 4:
            tower.append(current_lcm)
    return tower

LCM_TOWER = compute_lcm_tower()

# ═══════════════════════════════════════════════════════════════
# CORE BIJECTION — Theorem 12.1 (Losslessness). Pure mpmath.
# ═══════════════════════════════════════════════════════════════

def et_project(r_input, n_res=N):
    """Π_N(r) = (k, d, ε)  at 1200-digit precision."""
    if isinstance(r_input, Fraction):
        r_mp = mpmath.mpf(r_input.numerator) / mpmath.mpf(r_input.denominator)
    elif isinstance(r_input, mpmath.mpf):
        r_mp = r_input
    elif isinstance(r_input, int):
        r_mp = mpmath.mpf(r_input)
    else:
        r_mp = mpmath.mpf(str(r_input))
    if r_mp <= 0:
        return None
    n_mp = mpmath.mpf(n_res)
    log2_r = mpmath.log(r_mp, 2)
    continuous_k = n_mp * log2_r
    k_mp = mpmath.nint(continuous_k)
    k = int(k_mp)
    abs_k = abs(k)
    g = gcd(abs_k, n_res) if k != 0 else n_res
    d = n_res // g
    eps_mp = (continuous_k - k_mp) * (mpmath.mpf(1200) / n_mp)
    return (k, d, eps_mp)

def et_pullback(k, eps_mp, n_res=N):
    """Π_N⁻¹(k,d,ε) = 2^((k+εN/1200)/N). Exact inverse."""
    n_mp = mpmath.mpf(n_res)
    k_mp = mpmath.mpf(k)
    continuous_k = k_mp + eps_mp * n_mp / mpmath.mpf(1200)
    return mpmath.power(2, continuous_k / n_mp)

# ═══════════════════════════════════════════════════════════════
# ALL 12 HARMONIC FAMILIES + Magical Impedance
# ═══════════════════════════════════════════════════════════════

def euler_totient(n):
    result = n; p = 2; temp = n
    while p * p <= temp:
        if temp % p == 0:
            while temp % p == 0: temp //= p
            result -= result // p
        p += 1
    if temp > 1: result -= result // temp
    return result

def family_info(d):
    """Complete characterization of family d (all 12)."""
    is_simple = (N % d == 0)
    native_N = N if is_simple else (N * d // gcd(N, d))
    phi_d = euler_totient(d)
    a0 = (d - 1)**2 + S**2
    xi = mpmath.mpf(A0_LOCAL) / mpmath.mpf(a0)
    DATA = {
        1:  ("Octave / Unison", "Gravity / Identity",
             "Period closure — the most fundamental structure",
             "This note is on the octave axis. Maximum stability, like a tonic."),
        2:  ("Tritone", "Binary opposition / Pivot",
             "Half-period pivot — maximum tension",
             "A tritone-family interval. Maximum structural tension."),
        3:  ("Major 3rd / Minor 6th", "Strong force / Cubic",
             "Three-fold symmetry — augmented triad structure",
             "Major-third family. Think augmented triads, three-fold symmetry."),
        4:  ("Minor 3rd / Major 6th", "Weak force / Quartic",
             "Four-fold symmetry — diminished seventh structure",
             "Minor-third family. Think diminished chords, four-fold symmetry."),
        5:  ("Pentatonic (shadow)", "Quintic / Golden",
             "Five-fold / golden-ratio symmetry",
             "Pentatonic family. Backbone of blues, rock, folk, world music."),
        6:  ("Whole tone", "Electroweak / Hexagonal",
             "Hexagonal symmetry — whole-tone scale",
             "Whole-tone family. Think Debussy — shimmering, dream-like."),
        7:  ("Septimal / Blue note (shadow)", "Septic / G2",
             "Seven-fold — the 'blue' intervals",
             "Septimal interval — the elusive 'blue note' 12-TET can't reach."),
        8:  ("Octatonic (shadow)", "Gluon octet / SU(3)",
             "Eight-fold — diminished scale structure",
             "Octatonic family. Messiaen, Stravinsky, Bartok territory."),
        9:  ("Third-of-third (shadow)", "Nonic / Quark 3x3",
             "Nine-fold — trisected third",
             "Nonic — a refined subdivision of the third."),
        10: ("Decatonic (shadow)", "Decic / Superstring",
             "Ten-fold — extended chromatic",
             "Decatonic — microtonal territory beyond standard chromatic."),
        11: ("Undecimal / Neutral (shadow)", "Undecimal / M-theory",
             "Eleven-fold — neutral intervals of Arabic maqam",
             "Undecimal — the characteristic 'neutral' sound of Arabic music."),
        12: ("Chromatic / Semitone", "Electromagnetic / Full res",
             "Coprime to N — max complexity, includes P5 and P4",
             "Chromatic family. Full complexity. Perfect fifth and fourth live here."),
    }
    music, physics, character, friendly = DATA.get(d, (f"d={d}", f"d={d}", f"{d}-fold", f"{d}-fold interval"))
    return {
        "d": d, "music_name": music, "physics_name": physics,
        "character": character, "friendly": friendly,
        "is_simple": is_simple, "native_N": native_N, "totient": phi_d,
        "impedance_A0": a0, "coupling_xi": xi,
        "coupling_xi_str": mpmath.nstr(xi, 8),
        "status": "SIMPLE" if is_simple else f"SHADOW (N={native_N})",
    }

# Koide attractor/shadow — the Pythagorean comma
_pyth = et_project(Fraction(3, 2), N)
KOIDE_ATTRACTOR_EPS = mpmath.fabs(_pyth[2])

# ═══════════════════════════════════════════════════════════════
# FREQUENCY ↔ LATTICE — bidirectional industry bridge
# ═══════════════════════════════════════════════════════════════

NOTE_NAMES = ["A","A#/Bb","B","C","C#/Db","D","D#/Eb","E","F","F#/Gb","G","G#/Ab"]

def freq_to_lattice(hz_str, ref_str="440", n_res=N):
    hz_mp = mpmath.mpf(str(hz_str))
    ref_mp = mpmath.mpf(str(ref_str))
    r_mp = hz_mp / ref_mp
    proj = et_project(r_mp, n_res)
    if proj is None: return None
    k, d, eps_mp = proj
    info = family_info(d)
    semitone = k % N; octave = 4 + (k + 9) // N
    note = NOTE_NAMES[semitone % len(NOTE_NAMES)]
    return {"hz_mp": hz_mp, "hz_str": mpmath.nstr(hz_mp, 12), "ratio_mp": r_mp,
            "k": k, "d": d, "eps_mp": eps_mp, "note": f"{note}{octave}",
            "midi": MIDI_A4 + k, "family": info, "n_res": n_res}

def lattice_to_freq(k, eps_mp, n_res=N, ref_str="440"):
    ref_mp = mpmath.mpf(str(ref_str))
    return et_pullback(k, eps_mp, n_res) * ref_mp

def midi_to_lattice(midi_note, n_res=N):
    k = midi_note - MIDI_A4
    g = gcd(abs(k), n_res) if k != 0 else n_res
    d = n_res // g
    return (k, d, mpmath.mpf(0))

# ═══════════════════════════════════════════════════════════════
# SCALA .SCL EXPORT — industry-standard tuning file format
# ═══════════════════════════════════════════════════════════════

def generate_scala_file(name, ratios_dict, filepath=None):
    """Generate a .scl file. Loads into any Scala-compatible synth."""
    sorted_deg = []
    for dname, ratio in ratios_dict.items():
        if isinstance(ratio, Fraction):
            r_mp = mpmath.mpf(ratio.numerator) / mpmath.mpf(ratio.denominator)
        elif isinstance(ratio, mpmath.mpf):
            r_mp = ratio
        else:
            r_mp = mpmath.mpf(str(ratio))
        sorted_deg.append((dname, r_mp))
    sorted_deg.sort(key=lambda x: x[1])
    # Remove unison
    if sorted_deg and mpmath.fabs(sorted_deg[0][1] - mpmath.mpf(1)) < mpmath.mpf("1e-100"):
        sorted_deg = sorted_deg[1:]
    lines = [f"! {name}.scl", f"! Generated by ET Harmonic Lattice Analyzer (Sempaevum, {TARGET_DIGITS}-digit)",
             "!", name, f" {len(sorted_deg)}"]
    for dname, r_mp in sorted_deg:
        cents_mp = mpmath.mpf(1200) * mpmath.log(r_mp, 2)
        lines.append(f" {mpmath.nstr(cents_mp, 10)}  ! {dname}")
    content = "\n".join(lines) + "\n"
    if filepath:
        with open(filepath, 'w') as f: f.write(content)
    return content

# ═══════════════════════════════════════════════════════════════
# TUNING SYSTEM GENERATORS — dynamic, from generating principles
# ═══════════════════════════════════════════════════════════════

def gen_equal_temperament(n_div):
    ratios = {}
    for k in range(n_div + 1):
        r_mp = mpmath.power(2, mpmath.mpf(k) / mpmath.mpf(n_div))
        nm = "unison" if k == 0 else ("octave" if k == n_div else f"degree {k}")
        ratios[nm] = r_mp
    return ratios

def gen_pythagorean(n_fifths=12):
    fifth = mpmath.mpf(3) / mpmath.mpf(2)
    ratios = {"unison": mpmath.mpf(1)}
    raw = []
    for i in range(1, n_fifths):
        r = mpmath.power(fifth, mpmath.mpf(i))
        while r >= 2: r /= 2
        while r < 1: r *= 2
        raw.append((r, f"5th^{i}"))
    raw.sort(key=lambda x: x[0])
    for r, nm in raw: ratios[nm] = r
    ratios["octave"] = mpmath.mpf(2)
    return ratios

def gen_just_5limit():
    return {"unison": Fraction(1,1), "16/15": Fraction(16,15), "9/8": Fraction(9,8),
            "6/5": Fraction(6,5), "5/4": Fraction(5,4), "4/3": Fraction(4,3),
            "45/32": Fraction(45,32), "3/2": Fraction(3,2), "8/5": Fraction(8,5),
            "5/3": Fraction(5,3), "9/5": Fraction(9,5), "15/8": Fraction(15,8),
            "octave": Fraction(2,1)}

def gen_just_7limit():
    b = gen_just_5limit()
    b["7/6"] = Fraction(7,6); b["7/4"] = Fraction(7,4)
    b["7/5"] = Fraction(7,5); b["8/7"] = Fraction(8,7)
    return b

def gen_just_11limit():
    b = gen_just_7limit()
    b["11/8"] = Fraction(11,8); b["11/9"] = Fraction(11,9)
    b["12/11"] = Fraction(12,11); b["11/6"] = Fraction(11,6)
    return b

def gen_maqam_rast():
    return {"tonic": Fraction(1,1), "dugah": Fraction(9,8), "sikah": Fraction(5,4),
            "jaharkah": Fraction(4,3), "nawa": Fraction(3,2), "husayni": Fraction(27,16),
            "awj": Fraction(15,8), "kirdan": Fraction(2,1)}

def gen_maqam_bayati():
    return {"tonic": Fraction(1,1), "2nd (12/11)": Fraction(12,11), "3rd (32/27)": Fraction(32,27),
            "4th (4/3)": Fraction(4,3), "5th (3/2)": Fraction(3,2), "6th (18/11)": Fraction(18,11),
            "7th (16/9)": Fraction(16,9), "octave": Fraction(2,1)}

def gen_indian_shruti():
    return {"Sa": Fraction(1,1), "r1": Fraction(256,243), "r2": Fraction(16,15),
            "R1": Fraction(10,9), "R2": Fraction(9,8), "g1": Fraction(32,27),
            "g2": Fraction(6,5), "G1": Fraction(5,4), "G2": Fraction(81,64),
            "m1": Fraction(4,3), "m2": Fraction(27,20), "M1": Fraction(45,32),
            "M2": Fraction(729,512), "Pa": Fraction(3,2), "d1": Fraction(128,81),
            "d2": Fraction(8,5), "D1": Fraction(5,3), "D2": Fraction(27,16),
            "n1": Fraction(16,9), "n2": Fraction(9,5), "N1": Fraction(15,8),
            "N2": Fraction(243,128), "Sa'": Fraction(2,1)}

def gen_gamelan_slendro():
    return {"nem": mpmath.mpf(1), "barang": mpmath.mpf("1.2330"),
            "gulu": mpmath.mpf("1.5132"), "dhadha": mpmath.mpf("1.7706"),
            "lima": mpmath.mpf("1.8750"), "nem'": mpmath.mpf(2)}

def gen_gamelan_pelog():
    return {"bem": mpmath.mpf(1), "gulu": mpmath.mpf("1.0714"),
            "dhadha": mpmath.mpf("1.2018"), "pelog": mpmath.mpf("1.2857"),
            "lima": mpmath.mpf("1.5086"), "nem": mpmath.mpf("1.6257"),
            "barang": mpmath.mpf("1.8116"), "bem'": mpmath.mpf(2)}

SCALE_REGISTRY = {
    "12tet": ("12-TET", lambda: gen_equal_temperament(12)),
    "pythagorean": ("Pythagorean", gen_pythagorean),
    "ji5": ("5-limit JI", gen_just_5limit),
    "ji7": ("7-limit JI", gen_just_7limit),
    "ji11": ("11-limit JI", gen_just_11limit),
    "rast": ("Maqam Rast", gen_maqam_rast),
    "bayati": ("Maqam Bayati", gen_maqam_bayati),
    "shruti": ("22-shruti", gen_indian_shruti),
    "slendro": ("Slendro", gen_gamelan_slendro),
    "pelog": ("Pelog", gen_gamelan_pelog),
}

# ═══════════════════════════════════════════════════════════════
# ANALYSIS ENGINE
# ═══════════════════════════════════════════════════════════════

def analyze_ratio_full(r_input, label=""):
    results = []
    for n_val in [N, 60, 420, 2520, N_FULL]:
        proj = et_project(r_input, n_val)
        if proj is None: continue
        k, d, eps_mp = proj
        if isinstance(r_input, Fraction):
            r_mp = mpmath.mpf(r_input.numerator) / mpmath.mpf(r_input.denominator)
        elif isinstance(r_input, mpmath.mpf): r_mp = r_input
        else: r_mp = mpmath.mpf(str(r_input))
        r_back = et_pullback(k, eps_mp, n_val)
        rt_err = mpmath.fabs(r_back - r_mp)
        results.append({"N": n_val, "k": k, "d": d, "eps_mp": eps_mp,
                        "family": family_info(d), "roundtrip_err": rt_err})
    return results

def analyze_scale(name, ratios_dict, n_res=N):
    degrees = []; fam_count = {}
    for dname, ratio in ratios_dict.items():
        proj = et_project(ratio, n_res)
        if proj is None: continue
        k, d, eps_mp = proj
        info = family_info(d)
        fam_count[d] = fam_count.get(d, 0) + 1
        degrees.append({"name": dname, "k": k, "d": d, "eps_mp": eps_mp, "family": info})
    return {"name": name, "degrees": degrees, "family_count": fam_count}

def analyze_harmonics(hz_str, n_harmonics, n_res=N):
    f0 = mpmath.mpf(str(hz_str)); results = []
    for n in range(1, n_harmonics + 1):
        hz = f0 * mpmath.mpf(n); r = mpmath.mpf(n)
        proj = et_project(r, n_res)
        if proj is None: continue
        k, d, eps_mp = proj
        results.append({"harmonic": n, "hz_str": mpmath.nstr(hz, 10),
                        "k": k, "d": d, "eps_mp": eps_mp, "family": family_info(d)})
    return results

# ═══════════════════════════════════════════════════════════════
# OUTPUT — Musician-friendly. Simple. Clear.
# ═══════════════════════════════════════════════════════════════

def print_note(hz_str, ref_str="440"):
    r = freq_to_lattice(hz_str, ref_str)
    if r is None: print("  Invalid."); return
    f = r["family"]
    tag = "●" if f["is_simple"] else "○"
    print()
    print(f"  ┌─── YOUR NOTE {'─'*52}─┐")
    print(f"  │  Frequency:    {r['hz_str']:>12s} Hz{' '*38}│")
    print(f"  │  Nearest Note: {r['note']:<8s}  (MIDI {r['midi']}){' '*30}│")
    print(f"  │  Cents Off:    {mpmath.nstr(r['eps_mp'], 6):>12s}¢{' '*37}│")
    print(f"  │{'─'*67}│")
    print(f"  │  {tag} Family d={f['d']:>2d}: {f['music_name']:<42s}│")
    print(f"  │    Coupling ξ = {f['coupling_xi_str']:<8s}  A₀ = {f['impedance_A0']:<5d}{' '*21}│")
    print(f"  │    {f['status']:<63s}│")
    print(f"  │{'─'*67}│")
    print(f"  │  {f['friendly'][:65]:<65s}│")
    print(f"  └{'─'*67}┘")
    if mpmath.fabs(mpmath.fabs(r['eps_mp']) - KOIDE_ATTRACTOR_EPS) < mpmath.mpf("0.1"):
        print(f"  ★ KOIDE ATTRACTOR: |ε| ≈ {mpmath.nstr(KOIDE_ATTRACTOR_EPS, 6)}¢")

def print_ratio(r_input, label=""):
    results = analyze_ratio_full(r_input, label)
    if not results: print("  Invalid."); return
    lbl = label if label else str(r_input)
    base = results[0]; f = base["family"]
    tag = "●" if f["is_simple"] else "○"
    print(f"\n  ┌─── RATIO: {lbl} {'─'*(55-len(lbl))}─┐")
    print(f"  │  {tag} Family d={f['d']}: {f['music_name']:<20s} / {f['physics_name']:<22s}│")
    print(f"  │  ξ = {f['coupling_xi_str']:<8s} A₀ = {f['impedance_A0']:<5d}  {f['status']:>20s}{' '*9}│")
    print(f"  │  {f['friendly'][:65]:<65s}│")
    print(f"  └{'─'*67}┘")
    print(f"\n  {'N':>7s}  {'k':>8s}  {'d':>5s}  {'ε (cents)':>14s}  {'Status':6s}  {'Family':28s}  {'Lossless'}")
    print(f"  {'─'*7}  {'─'*8}  {'─'*5}  {'─'*14}  {'─'*6}  {'─'*28}  {'─'*12}")
    for res in results:
        fi = res["family"]; t = "●" if fi["is_simple"] else "○"
        mus = fi["music_name"] if res["N"] == N else f"d={res['d']}"
        rt = mpmath.nstr(res["roundtrip_err"], 4)
        print(f"  {res['N']:>7d}  {res['k']:>8d}  {res['d']:>5d}  {mpmath.nstr(res['eps_mp'], 8):>14s}  {t:6s}  {mus:28s}  {rt}")
    eps_12 = results[0]["eps_mp"]
    if mpmath.fabs(mpmath.fabs(eps_12) - KOIDE_ATTRACTOR_EPS) < mpmath.mpf("0.1"):
        print(f"\n  ★ KOIDE ATTRACTOR SHADOW: |ε| ≈ {mpmath.nstr(KOIDE_ATTRACTOR_EPS, 6)}¢")
        print(f"    The Sempaevum self-projection point. N, 1/N, K, 1/K all land here.")

def print_scale(name, ratios_dict, n_res=N, export_scl=False):
    a = analyze_scale(name, ratios_dict, n_res)
    print(f"\n  {'═'*75}")
    print(f"  {name}")
    print(f"  {'═'*75}")
    print(f"  {'Degree':<22s} {'k':>5s}  {'d':>3s}  {'ε':>12s}  {'●/○':3s}  {'Family'}")
    print(f"  {'─'*22} {'─'*5}  {'─'*3}  {'─'*12}  {'─'*3}  {'─'*30}")
    for deg in a["degrees"]:
        fi = deg["family"]; t = "●" if fi["is_simple"] else "○"
        print(f"  {deg['name']:<22s} {deg['k']:>5d}  {deg['d']:>3d}  {mpmath.nstr(deg['eps_mp'],6):>12s}  {t:3s}  {fi['music_name']}")
    print(f"\n  Family distribution (● SIMPLE, ○ SHADOW):")
    for d in sorted(a["family_count"].keys()):
        fi = family_info(d); t = "●" if fi["is_simple"] else "○"
        c = a["family_count"][d]; bar = "█" * c
        print(f"    {t} d={d:>2d} {fi['music_name']:<28s}: {c:>2d} {bar}")
    if export_scl:
        p = f"/mnt/user-data/outputs/{name.replace(' ','_').replace('/','_')}.scl"
        generate_scala_file(name, ratios_dict, p)
        print(f"\n  Scala file: {p}")

def print_harmonics(hz_str, n_h):
    results = analyze_harmonics(hz_str, n_h)
    print(f"\n  {'═'*75}")
    print(f"  HARMONIC SERIES of {hz_str} Hz — {n_h} partials")
    print(f"  {'═'*75}")
    print(f"  {'#':>3s}  {'Hz':>12s}  {'k':>5s}  {'d':>3s}  {'ε':>12s}  {'●/○':3s}  {'Family'}")
    print(f"  {'─'*3}  {'─'*12}  {'─'*5}  {'─'*3}  {'─'*12}  {'─'*3}  {'─'*28}")
    for r in results:
        fi = r["family"]; t = "●" if fi["is_simple"] else "○"
        print(f"  {r['harmonic']:>3d}  {r['hz_str']:>12s}  {r['k']:>5d}  {r['d']:>3d}  {mpmath.nstr(r['eps_mp'],6):>12s}  {t:3s}  {fi['music_name']}")

def print_families():
    print(f"\n  {'═'*78}")
    print(f"  THE 12 HARMONIC FAMILIES — Every interval belongs to one")
    print(f"  ● = SIMPLE (native at N=12)   ○ = SHADOW (needs higher resolution)")
    print(f"  ξ = coupling strength (higher = more fundamental)")
    print(f"  {'═'*78}")
    print(f"  {'d':>3s}  {'●/○':3s}  {'A₀':>5s}  {'ξ':>8s}  {'φ(d)':>4s}  {'Music':28s}  {'Physics'}")
    print(f"  {'─'*3}  {'─'*3}  {'─'*5}  {'─'*8}  {'─'*4}  {'─'*28}  {'─'*30}")
    for d in range(1, 13):
        f = family_info(d); t = "●" if f["is_simple"] else "○"
        print(f"  {d:>3d}  {t:3s}  {f['impedance_A0']:>5d}  {f['coupling_xi_str']:>8s}  {f['totient']:>4d}  {f['music_name']:<28s}  {f['physics_name']}")
    print(f"\n  Koide attractor: d=12, |ε| = {mpmath.nstr(KOIDE_ATTRACTOR_EPS, 10)}¢")
    print(f"  = Pythagorean comma = Sempaevum self-projection point")

# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("═" * 78)
    print("  ET HARMONIC LATTICE ANALYZER v2")
    print("  Every interval has a structural identity. This tool reveals it.")
    print(f"  {TARGET_DIGITS}-digit precision · All 12 families · Scala export · MIDI")
    print("═" * 78)
    print_families()
    print(f"\n\n{'─'*78}")
    print(f"  PART 1: THE PERFECT FIFTH — Koide Attractor")
    print(f"{'─'*78}")
    print_ratio(Fraction(3, 2), "3/2 (perfect fifth)")
    print(f"\n\n{'─'*78}")
    print(f"  PART 2: FREQUENCY ANALYSIS")
    print(f"{'─'*78}")
    for hz in ["261.6255653005986", "440", "432", "329.6275569128699"]:
        print_note(hz)
    print(f"\n\n{'─'*78}")
    print(f"  PART 3: ALL 12 FAMILIES IN ACTION")
    print(f"{'─'*78}")
    for r, lbl in [(Fraction(2,1),"2/1 octave"), (Fraction(7,5),"7/5 sept-TT"),
                    (Fraction(5,4),"5/4 just M3"), (Fraction(6,5),"6/5 just m3"),
                    (Fraction(11,8),"11/8 neutral 4th"), (Fraction(9,8),"9/8 M2"),
                    (Fraction(7,4),"7/4 blue 7th"), (Fraction(3,2),"3/2 P5")]:
        print_ratio(r, lbl)
    print(f"\n\n{'─'*78}")
    print(f"  PART 4: HARMONIC SERIES")
    print(f"{'─'*78}")
    print_harmonics("110", 16)
    print(f"\n\n{'─'*78}")
    print(f"  PART 5: WORLD TUNING SYSTEMS")
    print(f"{'─'*78}")
    for key in ["ji5", "rast", "shruti", "slendro"]:
        nm, fn = SCALE_REGISTRY[key]; print_scale(nm, fn())
    print(f"\n\n{'─'*78}")
    print(f"  PART 6: SCALA EXPORT")
    print(f"{'─'*78}")
    scl = generate_scala_file("5-limit JI", gen_just_5limit(),
                               "/mnt/user-data/outputs/just_intonation_5limit.scl")
    for line in scl.split('\n')[:8]: print(f"    {line}")
    print(f"\n  P ∘ D ∘ T = E")
    print("═" * 78)

def cli():
    if len(sys.argv) < 2: main(); return
    cmd = sys.argv[1].lower()
    if cmd == "note" and len(sys.argv) >= 3:
        for hz in sys.argv[2:]: print_note(hz)
    elif cmd == "ratio" and len(sys.argv) >= 3:
        for rs in sys.argv[2:]:
            if '/' in rs: print_ratio(Fraction(rs), rs)
            else: print_ratio(rs, rs)
    elif cmd == "midi" and len(sys.argv) >= 3:
        for ms in sys.argv[2:]:
            m = int(ms); k, d, eps = midi_to_lattice(m)
            hz = lattice_to_freq(k, eps)
            s = k % 12; o = 4 + (k+9)//12
            print(f"  MIDI {m} → k={k} d={d} ε=0¢ | {NOTE_NAMES[s%12]}{o} | {mpmath.nstr(hz,12)} Hz | {family_info(d)['music_name']}")
    elif cmd == "harmonics" and len(sys.argv) >= 4:
        print_harmonics(sys.argv[2], int(sys.argv[3]))
    elif cmd == "families":
        print_families()
    elif cmd == "scala" and len(sys.argv) >= 3:
        key = sys.argv[2].lower()
        if key in SCALE_REGISTRY:
            nm, fn = SCALE_REGISTRY[key]; r = fn()
            p = f"/mnt/user-data/outputs/{key}.scl"
            generate_scala_file(nm, r, p); print(f"  Written: {p}")
        else: print(f"  Unknown: {key}. Available: {', '.join(sorted(SCALE_REGISTRY.keys()))}")
    elif cmd == "scale" and len(sys.argv) >= 3:
        key = sys.argv[2].lower()
        if key.endswith("tet"):
            try: n = int(key.replace("tet","")); print_scale(f"{n}-TET", gen_equal_temperament(n)); return
            except ValueError: pass
        if key in SCALE_REGISTRY:
            nm, fn = SCALE_REGISTRY[key]; print_scale(nm, fn())
        else: print(f"  Unknown: {key}. Available: {', '.join(sorted(SCALE_REGISTRY.keys()))}, <n>tet")
    elif cmd == "lattice2freq" and len(sys.argv) >= 3:
        k = int(sys.argv[2]); eps = mpmath.mpf(sys.argv[3]) if len(sys.argv) > 3 else mpmath.mpf(0)
        hz = lattice_to_freq(k, eps)
        print(f"  (k={k}, ε={mpmath.nstr(eps,8)}¢) → {mpmath.nstr(hz, 15)} Hz")
    else: print(f"  Commands: note ratio midi harmonics families scala scale lattice2freq")

if __name__ == "__main__":
    cli()
