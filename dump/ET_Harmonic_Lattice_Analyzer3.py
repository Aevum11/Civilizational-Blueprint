#!/usr/bin/env python3
"""
ET HARMONIC LATTICE ANALYZER v2
Lossless frequency↔note converter. Sempaevum bijection, 1200-digit precision.

Just give it input:
  tool.py 440              → 440.0 Hz = A4  MIDI 69  ε=0.0¢  d=1
  tool.py 3/2              → 3/2 = 701.955¢ → Perfect 5th  k=7 d=12 ε=1.955¢
  tool.py C4               → C4 = 261.626 Hz  MIDI 60  k=-9
  tool.py 261.63 440 880   → batch convert
  tool.py scala ji5        → export Scala .scl file (load into any synth)
  tool.py scale shruti     → analyze a tuning system
  tool.py compare rast ji5 → cross-cultural structural comparison

Author: Michael James Muller — Aevum Defluo (Exception Theory)
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

def interval_name_12tet(k):
    """Standard 12-TET interval name for lattice coordinate k."""
    NAMES = {0:"Unison (P1)", 1:"Minor 2nd (m2)", 2:"Major 2nd (M2)",
             3:"Minor 3rd (m3)", 4:"Major 3rd (M3)", 5:"Perfect 4th (P4)",
             6:"Tritone (TT)", 7:"Perfect 5th (P5)", 8:"Minor 6th (m6)",
             9:"Major 6th (M6)", 10:"Minor 7th (m7)", 11:"Major 7th (M7)"}
    octaves = k // 12; remainder = k % 12
    name = NAMES.get(remainder, f"k={remainder}")
    if octaves == 0: return name
    elif octaves == 1 and remainder == 0: return "Octave (P8)"
    elif octaves > 0: return f"{name} + {octaves} oct"
    else: return f"{name} - {abs(octaves)} oct"

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

def gen_thai_7tet():
    """Thai classical: 7 equal divisions. Embedded in N=84 = lcm(12,7)."""
    return gen_equal_temperament(7)

def gen_werkmeister_III():
    """Werkmeister III (1691): baroque well-temperament. 4 fifths tempered by 1/4 Pythagorean comma."""
    pyth_comma = mpmath.power(mpmath.mpf(3), 12) / mpmath.power(mpmath.mpf(2), 19)
    qc = mpmath.power(pyth_comma, mpmath.mpf(1) / mpmath.mpf(4))
    pure5 = mpmath.mpf(3) / mpmath.mpf(2)
    temp5 = pure5 / qc
    # C-G-D-A tempered, A-E-B pure, B-F# tempered, rest pure
    seq = [("C","G",True),("G","D",True),("D","A",True),("A","E",False),
           ("E","B",False),("B","F#",True),("F#","C#",False),("C#","Ab",False),
           ("Ab","Eb",False),("Eb","Bb",False),("Bb","F",False),("F","C",False)]
    nr = {"C": mpmath.mpf(1)}; cur = mpmath.mpf(1)
    for fn, tn, tempered in seq:
        cur = cur * (temp5 if tempered else pure5)
        while cur >= 2: cur /= 2
        while cur < 1: cur *= 2
        nr[tn] = cur
    chrom = sorted(nr.items(), key=lambda x: x[1])
    ratios = {}
    for nm, r in chrom: ratios[nm] = r
    ratios["C (oct)"] = mpmath.mpf(2)
    return ratios

def gen_meantone_quarter_comma():
    """Quarter-comma meantone: fifth = 5^(1/4), major thirds pure."""
    fifth = mpmath.power(mpmath.mpf(5), mpmath.mpf(1) / mpmath.mpf(4))
    names = ["C#","D","Eb","E","F","F#","G","Ab","A","Bb","B"]
    ratios = {"C": mpmath.mpf(1)}; cur = mpmath.mpf(1); raw = []
    for i in range(1, 12):
        cur *= fifth
        while cur >= 2: cur /= 2
        while cur < 1: cur *= 2
        raw.append((cur, i))
    raw.sort(key=lambda x: x[0])
    for idx, (r, _) in enumerate(raw): ratios[names[idx]] = r
    ratios["C (oct)"] = mpmath.mpf(2)
    return ratios

def generate_just_intonation(prime_limit=5, max_complexity=25):
    """Dynamic JI: all ratios p/q with primes <= limit, p+q <= max_complexity."""
    primes = []
    for c in range(2, prime_limit + 1):
        if all(c % p != 0 for p in primes): primes.append(c)
    def uses_only(n, allowed):
        if n <= 0: return False
        for p in allowed:
            while n % p == 0: n //= p
        return n == 1
    rset = set(); rset.add(Fraction(1,1)); rset.add(Fraction(2,1))
    for p in range(1, max_complexity):
        for q in range(1, max_complexity):
            if p + q > max_complexity or gcd(p, q) != 1: continue
            if not uses_only(p, primes) or not uses_only(q, primes): continue
            f = Fraction(p, q)
            while f >= 2: f = f / 2
            while f < 1: f = f * 2
            rset.add(f)
    ratios = {}
    for f in sorted(rset):
        ratios[f"{f.numerator}/{f.denominator}"] = f
    return ratios

# ─── MAQAM GENERATOR (jins-based) ────────────────────────────
def gen_maqam_from_jins(jins_lower, jins_upper, name=""):
    """General maqam generator from two jins (tetrachords) plus connector."""
    ratios = {"tonic": Fraction(1,1)}
    pos = mpmath.mpf(1)
    for i, interval in enumerate(jins_lower):
        if isinstance(interval, Fraction):
            pos = pos * mpmath.mpf(interval.numerator) / mpmath.mpf(interval.denominator)
        else:
            pos = pos * mpmath.mpf(str(interval))
        ratios[f"deg{i+2}"] = pos
    for i, interval in enumerate(jins_upper):
        if isinstance(interval, Fraction):
            pos = pos * mpmath.mpf(interval.numerator) / mpmath.mpf(interval.denominator)
        else:
            pos = pos * mpmath.mpf(str(interval))
        ratios[f"deg{len(jins_lower)+i+2}"] = pos
    ratios["octave"] = mpmath.mpf(2)
    return ratios

def gen_maqam_hijaz():
    """Maqam Hijaz — augmented 2nd character, flamenco-like."""
    return {"tonic": Fraction(1,1), "2nd (16/15)": Fraction(16,15),
            "3rd (5/4)": Fraction(5,4), "4th (4/3)": Fraction(4,3),
            "5th (3/2)": Fraction(3,2), "6th (8/5)": Fraction(8,5),
            "7th (15/8)": Fraction(15,8), "octave": Fraction(2,1)}

def gen_maqam_saba():
    """Maqam Saba — unstable, characteristic descending movement."""
    return {"tonic": Fraction(1,1), "2nd (12/11)": Fraction(12,11),
            "3rd (32/27)": Fraction(32,27), "4th (18/13)": Fraction(18,13),
            "5th (3/2)": Fraction(3,2), "6th (8/5)": Fraction(8,5),
            "7th (16/9)": Fraction(16,9), "octave": Fraction(2,1)}

def gen_maqam_nahawand():
    """Maqam Nahawand — Arabic natural minor."""
    return {"tonic": Fraction(1,1), "2nd (9/8)": Fraction(9,8),
            "3rd (6/5)": Fraction(6,5), "4th (4/3)": Fraction(4,3),
            "5th (3/2)": Fraction(3,2), "6th (8/5)": Fraction(8,5),
            "7th (9/5)": Fraction(9,5), "octave": Fraction(2,1)}

def gen_maqam_kurd():
    """Maqam Kurd — Phrygian-like, starts with semitone."""
    return {"tonic": Fraction(1,1), "2nd (16/15)": Fraction(16,15),
            "3rd (6/5)": Fraction(6,5), "4th (4/3)": Fraction(4,3),
            "5th (3/2)": Fraction(3,2), "6th (8/5)": Fraction(8,5),
            "7th (9/5)": Fraction(9,5), "octave": Fraction(2,1)}

def gen_maqam_ajam():
    """Maqam Ajam — Arabic major, close to Western major."""
    return {"tonic": Fraction(1,1), "2nd (9/8)": Fraction(9,8),
            "3rd (5/4)": Fraction(5,4), "4th (4/3)": Fraction(4,3),
            "5th (3/2)": Fraction(3,2), "6th (27/16)": Fraction(27,16),
            "7th (15/8)": Fraction(15,8), "octave": Fraction(2,1)}

def gen_maqam_sikah():
    """Maqam Sikah — neutral third tonic, characteristic of Eastern music."""
    return {"tonic": Fraction(1,1), "2nd (12/11)": Fraction(12,11),
            "3rd (32/27)": Fraction(32,27), "4th (4/3)": Fraction(4,3),
            "5th (16/11)": Fraction(16,11), "6th (3/2)": Fraction(3,2),
            "7th (18/11)": Fraction(18,11), "octave": Fraction(2,1)}

# ─── PERSIAN DASTGAH ─────────────────────────────────────────
def gen_dastgah_shur():
    """Dastgah-e Shur — the most important Persian dastgah."""
    return {"tonic": Fraction(1,1), "2nd (10/9)": Fraction(10,9),
            "koron (11/9)": Fraction(11,9), "4th (4/3)": Fraction(4,3),
            "5th (3/2)": Fraction(3,2), "6th (5/3)": Fraction(5,3),
            "koron7 (11/6)": Fraction(11,6), "octave": Fraction(2,1)}

def gen_dastgah_mahur():
    """Dastgah-e Mahur — Persian major, similar to Western."""
    return {"tonic": Fraction(1,1), "2nd (9/8)": Fraction(9,8),
            "3rd (5/4)": Fraction(5,4), "4th (4/3)": Fraction(4,3),
            "5th (3/2)": Fraction(3,2), "6th (5/3)": Fraction(5,3),
            "7th (15/8)": Fraction(15,8), "octave": Fraction(2,1)}

def gen_dastgah_segah():
    """Dastgah-e Segah — starts on neutral third."""
    return {"tonic": Fraction(1,1), "2nd (12/11)": Fraction(12,11),
            "3rd (6/5)": Fraction(6,5), "4th (4/3)": Fraction(4,3),
            "5th (3/2)": Fraction(3,2), "6th (18/11)": Fraction(18,11),
            "7th (9/5)": Fraction(9,5), "octave": Fraction(2,1)}

def gen_dastgah_chahargah():
    """Dastgah-e Chahargah — augmented second character."""
    return {"tonic": Fraction(1,1), "2nd (9/8)": Fraction(9,8),
            "3rd (11/9)": Fraction(11,9), "4th (4/3)": Fraction(4,3),
            "5th (3/2)": Fraction(3,2), "6th (13/8)": Fraction(13,8),
            "7th (11/6)": Fraction(11,6), "octave": Fraction(2,1)}

# ─── INDIAN RAGAS (selections from 22-shruti) ────────────────
def gen_raga_selection(shruti_indices, name=""):
    """Generate a raga from shruti indices (1-based into the 22-shruti list)."""
    all_shruti = gen_indian_shruti()
    keys = list(all_shruti.keys())
    ratios = {}
    for idx in shruti_indices:
        if 0 <= idx < len(keys):
            ratios[keys[idx]] = all_shruti[keys[idx]]
    return ratios

def gen_raga_bhairav():
    """Raga Bhairav — morning raga, komal re and dha."""
    return {"Sa": Fraction(1,1), "re (16/15)": Fraction(16,15),
            "Ga (5/4)": Fraction(5,4), "Ma (4/3)": Fraction(4,3),
            "Pa (3/2)": Fraction(3,2), "dha (8/5)": Fraction(8,5),
            "Ni (15/8)": Fraction(15,8), "Sa'": Fraction(2,1)}

def gen_raga_yaman():
    """Raga Yaman/Kalyan — evening, tivra Ma."""
    return {"Sa": Fraction(1,1), "Re (9/8)": Fraction(9,8),
            "Ga (5/4)": Fraction(5,4), "Ma# (45/32)": Fraction(45,32),
            "Pa (3/2)": Fraction(3,2), "Dha (27/16)": Fraction(27,16),
            "Ni (15/8)": Fraction(15,8), "Sa'": Fraction(2,1)}

def gen_raga_todi():
    """Raga Todi — complex chromatic, all komal except tivra Ma."""
    return {"Sa": Fraction(1,1), "re (256/243)": Fraction(256,243),
            "ga (32/27)": Fraction(32,27), "Ma# (45/32)": Fraction(45,32),
            "Pa (3/2)": Fraction(3,2), "dha (128/81)": Fraction(128,81),
            "Ni (15/8)": Fraction(15,8), "Sa'": Fraction(2,1)}

def gen_raga_bhairavi():
    """Raga Bhairavi — all komal, devotional."""
    return {"Sa": Fraction(1,1), "re (16/15)": Fraction(16,15),
            "ga (6/5)": Fraction(6,5), "Ma (4/3)": Fraction(4,3),
            "Pa (3/2)": Fraction(3,2), "dha (8/5)": Fraction(8,5),
            "ni (9/5)": Fraction(9,5), "Sa'": Fraction(2,1)}

def gen_raga_kafi():
    """Raga Kafi — monsoon, komal ga and ni."""
    return {"Sa": Fraction(1,1), "Re (9/8)": Fraction(9,8),
            "ga (6/5)": Fraction(6,5), "Ma (4/3)": Fraction(4,3),
            "Pa (3/2)": Fraction(3,2), "Dha (5/3)": Fraction(5,3),
            "ni (9/5)": Fraction(9,5), "Sa'": Fraction(2,1)}

def gen_raga_bilawal():
    """Raga Bilawal — natural major, shuddh swaras."""
    return {"Sa": Fraction(1,1), "Re (9/8)": Fraction(9,8),
            "Ga (5/4)": Fraction(5,4), "Ma (4/3)": Fraction(4,3),
            "Pa (3/2)": Fraction(3,2), "Dha (5/3)": Fraction(5,3),
            "Ni (15/8)": Fraction(15,8), "Sa'": Fraction(2,1)}

def gen_raga_marwa():
    """Raga Marwa — twilight, komal re + tivra Ma, no Pa."""
    return {"Sa": Fraction(1,1), "re (16/15)": Fraction(16,15),
            "Ga (5/4)": Fraction(5,4), "Ma# (45/32)": Fraction(45,32),
            "Dha (27/16)": Fraction(27,16), "Ni (15/8)": Fraction(15,8),
            "Sa'": Fraction(2,1)}

# ─── EAST ASIAN ──────────────────────────────────────────────
def gen_chinese_12lu():
    """Chinese 12-lü — ancient Pythagorean-based system (c. 600 BCE)."""
    return gen_pythagorean(12)  # Same generating principle

def gen_japanese_hirajoshi():
    """Hirajoshi — Japanese pentatonic, Yamada koto tuning."""
    return {"1 (tonic)": Fraction(1,1), "2 (9/8)": Fraction(9,8),
            "3 (6/5)": Fraction(6,5), "4 (3/2)": Fraction(3,2),
            "5 (8/5)": Fraction(8,5), "octave": Fraction(2,1)}

def gen_japanese_miyakobushi():
    """Miyako-bushi — Japanese urban mode, In scale."""
    return {"1": Fraction(1,1), "2 (16/15)": Fraction(16,15),
            "3 (4/3)": Fraction(4,3), "4 (3/2)": Fraction(3,2),
            "5 (8/5)": Fraction(8,5), "octave": Fraction(2,1)}

def gen_japanese_insen():
    """In-sen — Japanese pentatonic, Kusakabe mode."""
    return {"1": Fraction(1,1), "2 (16/15)": Fraction(16,15),
            "3 (4/3)": Fraction(4,3), "4 (3/2)": Fraction(3,2),
            "5 (9/5)": Fraction(9,5), "octave": Fraction(2,1)}

def gen_balinese_pelog():
    """Balinese pelog — distinct from Javanese, brighter spacing."""
    return {"ding": mpmath.mpf(1), "dong": mpmath.mpf("1.0667"),
            "deng": mpmath.mpf("1.2222"), "dung": mpmath.mpf("1.5"),
            "dang": mpmath.mpf("1.6296"), "ding'": mpmath.mpf(2)}

# ─── MORE WELL-TEMPERAMENTS ─────────────────────────────────
def gen_well_temperament(tempered_fifths_map):
    """General well-temperament generator. tempered_fifths_map: {step_index: fraction_of_comma}."""
    pyth_comma = mpmath.power(mpmath.mpf(3), 12) / mpmath.power(mpmath.mpf(2), 19)
    pure5 = mpmath.mpf(3) / mpmath.mpf(2)
    note_names_circle = ["C","G","D","A","E","B","F#","C#","Ab","Eb","Bb","F"]
    nr = {"C": mpmath.mpf(1)}; cur = mpmath.mpf(1)
    for i in range(12):
        frac = tempered_fifths_map.get(i, mpmath.mpf(0))
        if frac != 0:
            temp = pure5 / mpmath.power(pyth_comma, frac)
        else:
            temp = pure5
        cur = cur * temp
        while cur >= 2: cur /= 2
        while cur < 1: cur *= 2
        if i < 11: nr[note_names_circle[i+1]] = cur
    chrom = sorted(nr.items(), key=lambda x: x[1])
    ratios = {}
    for nm, r in chrom: ratios[nm] = r
    ratios["C (oct)"] = mpmath.mpf(2)
    return ratios

def gen_kirnberger_III():
    """Kirnberger III — 4 fifths tempered by 1/4 syntonic comma."""
    sc = mpmath.mpf(81) / mpmath.mpf(80)  # syntonic comma
    pyth = mpmath.power(mpmath.mpf(3), 12) / mpmath.power(mpmath.mpf(2), 19)
    # Kirnberger III: C-G, G-D, D-A, A-E tempered by 1/4 syntonic comma
    # schisma absorbed in E-B fifth
    pure5 = mpmath.mpf(3) / mpmath.mpf(2)
    qsc = mpmath.power(sc, mpmath.mpf(1)/mpmath.mpf(4))
    circle = ["C","G","D","A","E","B","F#","C#","Ab","Eb","Bb","F"]
    tempered = {0,1,2,3}  # C-G, G-D, D-A, A-E
    nr = {"C": mpmath.mpf(1)}; cur = mpmath.mpf(1)
    for i in range(12):
        fifth = pure5 / qsc if i in tempered else pure5
        # For E-B, absorb the schisma (pyth_comma / syntonic_comma^4)
        if i == 4:
            schisma = pyth / mpmath.power(sc, 4)
            fifth = pure5 / schisma
        cur = cur * fifth
        while cur >= 2: cur /= 2
        while cur < 1: cur *= 2
        if i < 11: nr[circle[i+1]] = cur
    chrom = sorted(nr.items(), key=lambda x: x[1])
    ratios = {}
    for nm, r in chrom: ratios[nm] = r
    ratios["C (oct)"] = mpmath.mpf(2)
    return ratios

def gen_vallotti():
    """Vallotti temperament — 6 fifths tempered by 1/6 Pythagorean comma."""
    frac = mpmath.mpf(1)/mpmath.mpf(6)
    return gen_well_temperament({0: frac, 1: frac, 2: frac, 3: frac, 4: frac, 5: frac})

def gen_young_2():
    """Young Well Temperament #2 (Thomas Young, 1799)."""
    sixth = mpmath.mpf(1)/mpmath.mpf(6)
    return gen_well_temperament({0: sixth, 1: sixth, 2: sixth, 3: sixth, 4: sixth, 5: sixth})

def gen_neidhardt_III():
    """Neidhardt III — 1/12 comma spread across all fifths (approaches 12-TET)."""
    twelfth = mpmath.mpf(1)/mpmath.mpf(12)
    return gen_well_temperament({i: twelfth for i in range(12)})

# ─── MEANTONE VARIANTS ──────────────────────────────────────
def gen_meantone_sixth_comma():
    """Sixth-comma meantone — compromise between meantone and 12-TET."""
    sc = mpmath.mpf(81) / mpmath.mpf(80)
    fifth = (mpmath.mpf(3)/mpmath.mpf(2)) / mpmath.power(sc, mpmath.mpf(1)/mpmath.mpf(6))
    names = ["C#","D","Eb","E","F","F#","G","Ab","A","Bb","B"]
    ratios = {"C": mpmath.mpf(1)}; cur = mpmath.mpf(1); raw = []
    for i in range(1, 12):
        cur *= fifth
        while cur >= 2: cur /= 2
        while cur < 1: cur *= 2
        raw.append((cur, i))
    raw.sort(key=lambda x: x[0])
    for idx, (r, _) in enumerate(raw): ratios[names[idx]] = r
    ratios["C (oct)"] = mpmath.mpf(2)
    return ratios

def gen_meantone_third_comma():
    """Third-comma meantone (Salinas) — pure minor thirds."""
    sc = mpmath.mpf(81) / mpmath.mpf(80)
    fifth = (mpmath.mpf(3)/mpmath.mpf(2)) / mpmath.power(sc, mpmath.mpf(1)/mpmath.mpf(3))
    names = ["C#","D","Eb","E","F","F#","G","Ab","A","Bb","B"]
    ratios = {"C": mpmath.mpf(1)}; cur = mpmath.mpf(1); raw = []
    for i in range(1, 12):
        cur *= fifth
        while cur >= 2: cur /= 2
        while cur < 1: cur *= 2
        raw.append((cur, i))
    raw.sort(key=lambda x: x[0])
    for idx, (r, _) in enumerate(raw): ratios[names[idx]] = r
    ratios["C (oct)"] = mpmath.mpf(2)
    return ratios

# ─── EXTENDED JI / EXPERIMENTAL ──────────────────────────────
def gen_partch_43():
    """Harry Partch 43-tone scale — 11-limit JI, full chromatic."""
    return generate_just_intonation(11, 30)

def gen_blues_ji():
    """Blues scale in 7-limit JI — septimal blue notes."""
    return {"tonic": Fraction(1,1), "m3 (6/5)": Fraction(6,5),
            "P4 (4/3)": Fraction(4,3), "blue (7/5)": Fraction(7,5),
            "P5 (3/2)": Fraction(3,2), "blue7 (7/4)": Fraction(7,4),
            "octave": Fraction(2,1)}

def gen_bohlen_pierce():
    """Bohlen-Pierce — non-octave scale based on 3:1 tritave, 13 steps."""
    ratios = {}
    for k in range(14):
        r_mp = mpmath.power(3, mpmath.mpf(k) / mpmath.mpf(13))
        nm = "unison" if k == 0 else ("tritave" if k == 13 else f"BP-{k}")
        ratios[nm] = r_mp
    return ratios

def gen_carlos_alpha():
    """Wendy Carlos Alpha — step ≈ 78 cents, non-octave, pure m3 and M3."""
    step = mpmath.mpf("78") / mpmath.mpf(1200)
    ratios = {}
    for k in range(16):
        ratios[f"α-{k}" if k > 0 else "unison"] = mpmath.power(2, step * mpmath.mpf(k))
    return ratios

def gen_carlos_beta():
    """Wendy Carlos Beta — step ≈ 63.8 cents, pure P4."""
    step = mpmath.mpf("63.8") / mpmath.mpf(1200)
    ratios = {}
    for k in range(20):
        ratios[f"β-{k}" if k > 0 else "unison"] = mpmath.power(2, step * mpmath.mpf(k))
    return ratios

# ─── AFRICAN ─────────────────────────────────────────────────
def gen_chopi_xylophone():
    """Chopi timbila xylophone (Mozambique) — 7-note near-equidistant."""
    return {"1": mpmath.mpf(1), "2": mpmath.mpf("1.148"), "3": mpmath.mpf("1.316"),
            "4": mpmath.mpf("1.468"), "5": mpmath.mpf("1.622"),
            "6": mpmath.mpf("1.774"), "7": mpmath.mpf("1.905"), "8": mpmath.mpf(2)}

def gen_ethiopian_kiñit():
    """Ethiopian kiñit — pentatonic framework, anchihoye mode."""
    return {"1": Fraction(1,1), "2 (9/8)": Fraction(9,8),
            "3 (5/4)": Fraction(5,4), "4 (3/2)": Fraction(3,2),
            "5 (5/3)": Fraction(5,3), "octave": Fraction(2,1)}

# ═══════════════════════════════════════════════════════════════
# SCALE REGISTRY — ALL major world tuning systems
# ═══════════════════════════════════════════════════════════════
SCALE_REGISTRY = {
    # Western — Equal Temperaments
    "12tet": ("12-TET (Western standard)", lambda: gen_equal_temperament(12)),
    "17tet": ("17-TET", lambda: gen_equal_temperament(17)),
    "19tet": ("19-TET (Salinas/Costeley)", lambda: gen_equal_temperament(19)),
    "22tet": ("22-TET (Indian-adjacent)", lambda: gen_equal_temperament(22)),
    "24tet": ("24-TET (Arabic quarter-tone)", lambda: gen_equal_temperament(24)),
    "31tet": ("31-TET (Huygens/Fokker)", lambda: gen_equal_temperament(31)),
    "34tet": ("34-TET", lambda: gen_equal_temperament(34)),
    "41tet": ("41-TET", lambda: gen_equal_temperament(41)),
    "43tet": ("43-TET", lambda: gen_equal_temperament(43)),
    "53tet": ("53-TET (Turkish/Holdrian)", lambda: gen_equal_temperament(53)),
    "72tet": ("72-TET (Ekmelic)", lambda: gen_equal_temperament(72)),
    "7tet": ("Thai 7-TET", gen_thai_7tet),
    # Western — Historical Temperaments
    "pythagorean": ("Pythagorean", gen_pythagorean),
    "meantone": ("Quarter-comma Meantone", gen_meantone_quarter_comma),
    "meantone6": ("Sixth-comma Meantone", gen_meantone_sixth_comma),
    "meantone3": ("Third-comma Meantone (Salinas)", gen_meantone_third_comma),
    "werkmeister": ("Werkmeister III", gen_werkmeister_III),
    "kirnberger": ("Kirnberger III", gen_kirnberger_III),
    "vallotti": ("Vallotti", gen_vallotti),
    "young": ("Young #2", gen_young_2),
    "neidhardt": ("Neidhardt III", gen_neidhardt_III),
    # Just Intonation
    "ji5": ("5-limit JI", gen_just_5limit),
    "ji7": ("7-limit JI", gen_just_7limit),
    "ji11": ("11-limit JI", gen_just_11limit),
    "ji-dynamic": ("Dynamic JI (5-limit)", lambda: generate_just_intonation(5, 25)),
    "partch": ("Partch 43-tone (11-limit)", gen_partch_43),
    # Arabic Maqamat
    "rast": ("Maqam Rast", gen_maqam_rast),
    "bayati": ("Maqam Bayati", gen_maqam_bayati),
    "hijaz": ("Maqam Hijaz", gen_maqam_hijaz),
    "saba": ("Maqam Saba", gen_maqam_saba),
    "nahawand": ("Maqam Nahawand", gen_maqam_nahawand),
    "kurd": ("Maqam Kurd", gen_maqam_kurd),
    "ajam": ("Maqam Ajam", gen_maqam_ajam),
    "sikah": ("Maqam Sikah", gen_maqam_sikah),
    # Persian Dastgah
    "shur": ("Dastgah Shur", gen_dastgah_shur),
    "mahur": ("Dastgah Mahur", gen_dastgah_mahur),
    "segah": ("Dastgah Segah", gen_dastgah_segah),
    "chahargah": ("Dastgah Chahargah", gen_dastgah_chahargah),
    # Indian
    "shruti": ("22-shruti", gen_indian_shruti),
    "bhairav": ("Raga Bhairav", gen_raga_bhairav),
    "yaman": ("Raga Yaman/Kalyan", gen_raga_yaman),
    "todi": ("Raga Todi", gen_raga_todi),
    "bhairavi": ("Raga Bhairavi", gen_raga_bhairavi),
    "kafi": ("Raga Kafi", gen_raga_kafi),
    "bilawal": ("Raga Bilawal", gen_raga_bilawal),
    "marwa": ("Raga Marwa", gen_raga_marwa),
    # East Asian
    "12lu": ("Chinese 12-lü", gen_chinese_12lu),
    "hirajoshi": ("Japanese Hirajoshi", gen_japanese_hirajoshi),
    "miyakobushi": ("Japanese Miyako-bushi", gen_japanese_miyakobushi),
    "insen": ("Japanese In-sen", gen_japanese_insen),
    "slendro": ("Javanese Slendro", gen_gamelan_slendro),
    "pelog": ("Javanese Pelog", gen_gamelan_pelog),
    "bali-pelog": ("Balinese Pelog", gen_balinese_pelog),
    # African
    "chopi": ("Chopi Timbila (Mozambique)", gen_chopi_xylophone),
    "kinit": ("Ethiopian Kiñit", gen_ethiopian_kiñit),
    # Blues / Jazz
    "blues": ("Blues Scale (7-limit JI)", gen_blues_ji),
    # Non-octave / Experimental
    "bohlen-pierce": ("Bohlen-Pierce (3:1 tritave)", gen_bohlen_pierce),
    "carlos-alpha": ("Wendy Carlos Alpha", gen_carlos_alpha),
    "carlos-beta": ("Wendy Carlos Beta", gen_carlos_beta),
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

def compare_tuning_systems(systems, n_res=N):
    """Structural comparison of multiple tuning systems on the lattice."""
    comparisons = {}
    for sys_name, sys_ratios in systems.items():
        a = analyze_scale(sys_name, sys_ratios, n_res)
        fam_usage = {}
        for deg in a["degrees"]:
            d = deg["d"]
            if d not in fam_usage: fam_usage[d] = []
            fam_usage[d].append(deg)
        comparisons[sys_name] = {"analysis": a, "family_usage": fam_usage,
                                  "families_used": sorted(fam_usage.keys())}
    return comparisons

def find_structural_connections(comparisons):
    """Find cross-cultural connections: same (k,d), different ε."""
    position_map = {}
    for sys_name, comp in comparisons.items():
        for deg in comp["analysis"]["degrees"]:
            key = (deg["k"], deg["d"])
            if key not in position_map: position_map[key] = []
            position_map[key].append({"system": sys_name, "degree": deg["name"],
                "ratio": deg.get("ratio_str", ""), "eps_mp": deg["eps_mp"]})
    shared = {}
    for key, entries in position_map.items():
        systems_present = set(e["system"] for e in entries)
        if len(systems_present) >= 2: shared[key] = entries
    return shared

def analyze_frequencies(hz_list, ref_str="440", n_res=N):
    """Batch frequency analysis."""
    results = []
    for hz_str in hz_list:
        r = freq_to_lattice(str(hz_str), ref_str, n_res)
        if r: results.append(r)
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

def print_cross_cultural(shared_positions):
    """Print cross-cultural structural connections."""
    if not shared_positions:
        print("\n  No shared lattice positions found."); return
    print(f"\n  {'═'*75}")
    print(f"  CROSS-CULTURAL STRUCTURAL CONNECTIONS")
    print(f"  Same lattice position (k, d) — differ ONLY in ε")
    print(f"  {'═'*75}")
    for (k, d), entries in sorted(shared_positions.items()):
        f = family_info(d); tag = "●" if f["is_simple"] else "○"
        print(f"\n  {tag} (k={k}, d={d}) — {f['music_name']} / {f['physics_name']}:")
        for e in entries:
            print(f"    {e['system']:<25s} {e['degree']:<20s} ε={mpmath.nstr(e['eps_mp'],6):>12s}¢")
        eps_vals = [e['eps_mp'] for e in entries]
        spread = mpmath.fabs(max(eps_vals) - min(eps_vals))
        if spread > mpmath.mpf("0.001"):
            print(f"    ── ε-spread: {mpmath.nstr(spread,3)}¢")
        else:
            print(f"    ── EXACT MATCH across all traditions")

def print_elegance(name, ratios_dict, n_res=N):
    """Elegance score: ε = (N/d) · 100/(100+|ε|) · 100/(p+q)."""
    print(f"\n  Elegance Analysis: {name}")
    print(f"  {'Degree':<22s} {'k':>5s}  {'d':>3s}  {'|ε|':>8s}  {'N/d':>5s}  {'Score'}")
    print(f"  {'─'*22} {'─'*5}  {'─'*3}  {'─'*8}  {'─'*5}  {'─'*10}")
    for dname, ratio in ratios_dict.items():
        proj = et_project(ratio, n_res)
        if proj is None: continue
        k, d, eps_mp = proj
        depth = n_res // d
        eps_fac = mpmath.mpf(100) / (mpmath.mpf(100) + mpmath.fabs(eps_mp))
        if isinstance(ratio, Fraction):
            p, q = ratio.numerator, ratio.denominator
        else:
            frac = Fraction(str(mpmath.nstr(ratio, 15))).limit_denominator(1000)
            p, q = frac.numerator, frac.denominator
        cpx_fac = mpmath.mpf(100) / mpmath.mpf(p + q)
        elegance = mpmath.mpf(depth) * eps_fac * cpx_fac
        print(f"  {dname:<22s} {k:>5d}  {d:>3d}  {mpmath.nstr(mpmath.fabs(eps_mp),4):>8s}  {depth:>5d}  {mpmath.nstr(elegance,6)}")

def print_frequencies(hz_list, ref_str="440"):
    """Batch frequency analysis — musician-friendly table."""
    results = analyze_frequencies(hz_list, ref_str)
    print(f"\n  {'═'*75}")
    print(f"  FREQUENCY ANALYSIS (ref: {ref_str} Hz)")
    print(f"  {'═'*75}")
    print(f"  {'Hz':>12s}  {'k':>5s}  {'d':>3s}  {'ε':>12s}  {'Note':>8s}  {'MIDI':>4s}  {'Family'}")
    print(f"  {'─'*12}  {'─'*5}  {'─'*3}  {'─'*12}  {'─'*8}  {'─'*4}  {'─'*28}")
    for r in results:
        f = r["family"]; tag = "●" if f["is_simple"] else "○"
        print(f"  {r['hz_str']:>12s}  {r['k']:>5d}  {r['d']:>3d}  {mpmath.nstr(r['eps_mp'],6):>12s}  {r['note']:>8s}  {r['midi']:>4d}  {tag} {f['music_name']}")

# ═══════════════════════════════════════════════════════════════
# FILE I/O — File in, converted file out. The bijection as a pipe.
# ═══════════════════════════════════════════════════════════════

def read_midi_file(filepath):
    """Pure-Python MIDI parser. Reads .mid → list of (tick, channel, note, velocity, on/off)."""
    events = []; tempo = 500000  # default 120 BPM
    with open(filepath, 'rb') as f:
        data = f.read()
    pos = 0
    def read_bytes(n):
        nonlocal pos; chunk = data[pos:pos+n]; pos += n; return chunk
    def read_int(n):
        return int.from_bytes(read_bytes(n), 'big')
    def read_varlen():
        nonlocal pos; val = 0
        while True:
            b = data[pos]; pos += 1; val = (val << 7) | (b & 0x7F)
            if not (b & 0x80): return val
    # Header
    hdr = read_bytes(4)
    if hdr != b'MThd': return [], 480, 500000
    read_int(4)  # header length
    fmt = read_int(2); ntracks = read_int(2); division = read_int(2)
    ticks_per_beat = division if not (division & 0x8000) else 480
    # Tracks
    for _ in range(ntracks):
        trk_hdr = read_bytes(4)
        if trk_hdr != b'MTrk': break
        trk_len = read_int(4); trk_end = pos + trk_len
        abs_tick = 0; running_status = 0
        while pos < trk_end:
            delta = read_varlen(); abs_tick += delta
            byte = data[pos]
            if byte & 0x80: pos += 1; status = byte
            else: status = running_status
            if status == 0xFF:  # meta event
                meta_type = data[pos]; pos += 1; meta_len = read_varlen()
                meta_data = read_bytes(meta_len)
                if meta_type == 0x51 and meta_len == 3:  # tempo
                    tempo = (meta_data[0] << 16) | (meta_data[1] << 8) | meta_data[2]
                if meta_type == 0x2F: break  # end of track
            elif status >= 0xF0:  # sysex
                slen = read_varlen(); read_bytes(slen)
            elif (status & 0xF0) in (0x80, 0x90):  # note off/on
                running_status = status
                note = data[pos]; vel = data[pos+1]; pos += 2
                ch = status & 0x0F; on = (status & 0xF0) == 0x90 and vel > 0
                events.append((abs_tick, ch, note, vel, on))
            elif (status & 0xF0) in (0xA0, 0xB0, 0xE0):
                running_status = status; pos += 2
            elif (status & 0xF0) in (0xC0, 0xD0):
                running_status = status; pos += 1
    return events, ticks_per_beat, tempo

def process_midi_file(filepath):
    """MIDI file → CSV with lattice coordinates for every note."""
    events, tpb, tempo = read_midi_file(filepath)
    us_per_tick = tempo / tpb
    print("time_ms,midi,note,hz,k,d,epsilon_cents,velocity")
    for tick, ch, note, vel, on in events:
        if not on: continue
        time_ms = mpmath.mpf(str(tick)) * mpmath.mpf(str(us_per_tick)) / mpmath.mpf(1000)
        k = note - MIDI_A4
        g = gcd(abs(k), N) if k != 0 else N; d = N // g
        hz_mp = lattice_to_freq(k, mpmath.mpf(0))
        sm = k % 12; oc = 4 + (k + 9) // 12
        nname = NOTE_NAMES[sm % len(NOTE_NAMES)]
        print(f"{mpmath.nstr(time_ms,6)},{note},{nname}{oc},{mpmath.nstr(hz_mp,12)},{k},{d},0.000,{vel}")

def write_midi_file(notes, filepath, bpm=120):
    """Write a MIDI file from list of (hz_or_midi, duration_ms, velocity).
    hz values are projected through the bijection → nearest MIDI note + pitch bend."""
    tpb = 480; us_per_beat = 60000000 // bpm
    ticks_per_ms = mpmath.mpf(tpb) * mpmath.mpf(1000) / mpmath.mpf(str(us_per_beat))
    def varlen_bytes(val):
        if val < 0: val = 0
        result = [val & 0x7F]
        val >>= 7
        while val:
            result.append((val & 0x7F) | 0x80); val >>= 7
        return bytes(reversed(result))
    track_data = bytearray()
    # Tempo meta event
    track_data += varlen_bytes(0) + b'\xFF\x51\x03'
    track_data += us_per_beat.to_bytes(3, 'big')
    abs_tick = 0
    for hz_str, dur_ms, vel in notes:
        # Project Hz through bijection
        proj = et_project(mpmath.mpf(str(hz_str)) / CONCERT_A4, N)
        if proj is None: continue
        k, d, eps_mp = proj
        midi_note = MIDI_A4 + k
        if midi_note < 0 or midi_note > 127: continue
        # Pitch bend for ε (±50¢ maps to ±8192, MIDI pitch bend range)
        bend_raw = int(mpmath.nint(mpmath.mpf(8192) * eps_mp / mpmath.mpf(50)))
        bend = max(-8192, min(8191, bend_raw)) + 8192
        bend_lsb = bend & 0x7F; bend_msb = (bend >> 7) & 0x7F
        dur_ticks = max(1, int(mpmath.mpf(str(dur_ms)) * ticks_per_ms))
        # Pitch bend
        track_data += varlen_bytes(0) + bytes([0xE0, bend_lsb, bend_msb])
        # Note on
        track_data += varlen_bytes(0) + bytes([0x90, midi_note, vel])
        # Note off after duration
        track_data += varlen_bytes(dur_ticks) + bytes([0x80, midi_note, 0])
    # End of track
    track_data += varlen_bytes(0) + b'\xFF\x2F\x00'
    # Build file
    header = b'MThd' + (6).to_bytes(4,'big') + (0).to_bytes(2,'big')
    header += (1).to_bytes(2,'big') + tpb.to_bytes(2,'big')
    track = b'MTrk' + len(track_data).to_bytes(4,'big') + bytes(track_data)
    with open(filepath, 'wb') as f: f.write(header + track)
    print(f"  Written: {filepath} ({len(notes)} notes, {bpm} BPM)")

def read_scala_file(filepath):
    """Read a .scl file → dict of {degree_name: ratio_mpf}."""
    ratios = {"unison": mpmath.mpf(1)}
    with open(filepath) as f:
        lines = [l.strip() for l in f if l.strip() and not l.strip().startswith('!')]
    if len(lines) < 2: return ratios
    name = lines[0]; count = int(lines[1].strip())
    for i, line in enumerate(lines[2:2+count]):
        parts = line.split('!'); val_str = parts[0].strip()
        deg_name = parts[1].strip() if len(parts) > 1 else f"degree_{i+1}"
        if '/' in val_str:
            frac = Fraction(val_str)
            r_mp = mpmath.mpf(frac.numerator) / mpmath.mpf(frac.denominator)
        elif '.' in val_str:
            cents_mp = mpmath.mpf(val_str)
            r_mp = mpmath.power(2, cents_mp / mpmath.mpf(1200))
        else:
            cents_mp = mpmath.mpf(val_str)
            r_mp = mpmath.power(2, cents_mp / mpmath.mpf(1200))
        ratios[deg_name] = r_mp
    return ratios

def process_scala_file(filepath):
    """Scala .scl file → frequency table with lattice coordinates."""
    ratios = read_scala_file(filepath)
    base_hz = CONCERT_A4  # default reference
    print(f"# Scala file: {filepath}")
    print(f"# Reference: {mpmath.nstr(base_hz, 6)} Hz")
    print("degree,ratio,hz,cents,k,d,epsilon_cents")
    for dname, r_mp in ratios.items():
        hz_mp = r_mp * base_hz
        cents_mp = mpmath.mpf(1200) * mpmath.log(r_mp, 2) if r_mp > 0 else mpmath.mpf(0)
        proj = et_project(r_mp, N)
        if proj:
            k, d, eps = proj
            print(f"{dname},{mpmath.nstr(r_mp,10)},{mpmath.nstr(hz_mp,10)},{mpmath.nstr(cents_mp,6)},{k},{d},{mpmath.nstr(eps,6)}")

def process_text_file(filepath):
    """Text file (one value per line: Hz, ratio, or note name) → converted output."""
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                convert(line)

def process_file(filepath, output_path=None):
    """Auto-detect file type and process."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext in ('.mid', '.midi'):
        process_midi_file(filepath)
    elif ext == '.scl':
        process_scala_file(filepath)
    elif ext == '.wav':
        print("  WAV pitch detection requires scipy. Use an external pitch detector and pipe:")
        print(f"    aubiopitch {filepath} | python3 {sys.argv[0]}")
        print(f"    crepe {filepath} --output pitch.csv && python3 {sys.argv[0]} pitch.csv")
    else:
        process_text_file(filepath)

# ═══════════════════════════════════════════════════════════════
# CONVERTER — Input in, answer out. The bijection does the work.
# ═══════════════════════════════════════════════════════════════

_NOTE_OFFSETS = {"C":0,"D":2,"E":4,"F":5,"G":7,"A":9,"B":11}

def parse_note_name(s):
    """Parse 'C4', 'A#3', 'Bb5' → k (lattice coordinate). None if not a note."""
    s = s.strip()
    if not s or s[0].upper() not in _NOTE_OFFSETS: return None
    offset = _NOTE_OFFSETS[s[0].upper()]; pos = 1
    if pos < len(s) and s[pos] == '#': offset += 1; pos += 1
    elif pos < len(s) and s[pos].lower() == 'b' and (pos+1 >= len(s) or s[pos+1].isdigit()): offset -= 1; pos += 1
    try: octave = int(s[pos:])
    except (ValueError, IndexError): return None
    return (octave - 4) * 12 + (offset - 9)

def convert(s):
    """Auto-detect input, convert, print one line."""
    s = s.strip()
    # Fraction → ratio
    if '/' in s and s[0].isdigit():
        try:
            frac = Fraction(s)
            cents_mp = mpmath.mpf(1200) * mpmath.log(mpmath.mpf(frac.numerator)/mpmath.mpf(frac.denominator), 2)
            proj = et_project(frac, N)
            k, d, eps_mp = proj
            print(f"{s} = {mpmath.nstr(cents_mp, 6)}¢ → {interval_name_12tet(k)}  k={k} d={d} ε={mpmath.nstr(eps_mp, 6)}¢")
            return
        except: pass
    # Note name → Hz
    k_parsed = parse_note_name(s)
    if k_parsed is not None:
        hz_mp = lattice_to_freq(k_parsed, mpmath.mpf(0))
        print(f"{s} = {mpmath.nstr(hz_mp, 12)} Hz  MIDI {MIDI_A4 + k_parsed}  k={k_parsed}")
        return
    # Number → Hz to note
    try:
        test = mpmath.mpf(s)
        if test > 0:
            r = freq_to_lattice(s)
            if r:
                print(f"{r['hz_str']} Hz = {r['note']}  MIDI {r['midi']}  ε={mpmath.nstr(r['eps_mp'], 4)}¢  d={r['d']}")
                return
    except: pass
    print(f"  Cannot parse: {s}")

# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

_CMDS = {"note","ratio","midi","harmonics","families","scale","scala",
         "compare","elegance","selfproject","lattice2freq","freqs","scales","to-midi"}

def cli():
    # Stdin pipe: no args + piped input → convert each line
    if len(sys.argv) < 2:
        if not sys.stdin.isatty():
            for line in sys.stdin:
                line = line.strip()
                if line and not line.startswith('#'): convert(line)
            return
        print("  Lossless frequency↔note converter (Sempaevum bijection, 1200-digit)")
        print("  File/stream in → converted out:\n")
        print("    tool.py 440 261.63 3/2 C4    convert values (Hz, ratios, notes)")
        print("    cat freqs.txt | tool.py       pipe: stdin → stdout")
        print("    tool.py freqs.txt             file: auto-detect, convert")
        print("    tool.py song.mid              MIDI → CSV of Hz + lattice coords")
        print("    tool.py tuning.scl             Scala → frequency table")
        print("    tool.py to-midi in.txt out.mid Hz list → MIDI file\n")
        print("  Advanced: scale scala compare harmonics families selfproject elegance scales")
        print(f"  {len(SCALE_REGISTRY)} tuning systems. 'scales' to list.")
        return

    first = sys.argv[1]

    # File input: if first arg is an existing file, process it
    if os.path.isfile(first) and first.lower() not in _CMDS:
        process_file(first)
        return

    # to-midi: Hz list file → MIDI file
    if first.lower() == "to-midi" and len(sys.argv) >= 4:
        infile = sys.argv[2]; outfile = sys.argv[3]
        bpm = int(sys.argv[4]) if len(sys.argv) > 4 else 120
        notes = []
        with open(infile) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'): continue
                parts = line.split(',') if ',' in line else line.split()
                hz = parts[0].strip()
                dur = parts[1].strip() if len(parts) > 1 else "500"
                vel = int(parts[2].strip()) if len(parts) > 2 else 100
                notes.append((hz, dur, vel))
        write_midi_file(notes, outfile, bpm)
        return

    # Bare args (not a command keyword, not a file) → auto-convert
    if first.lower() not in _CMDS:
        for arg in sys.argv[1:]: convert(arg)
        return

    cmd = first.lower()

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
            f = family_info(d)
            print(f"  MIDI {m} → k={k} d={d} ε=0¢ | {NOTE_NAMES[s%12]}{o} | {mpmath.nstr(hz,12)} Hz | {f['music_name']}")

    elif cmd == "harmonics" and len(sys.argv) >= 4:
        print_harmonics(sys.argv[2], int(sys.argv[3]))

    elif cmd == "families":
        print_families()

    elif cmd == "selfproject":
        print(f"\n  KOIDE ATTRACTOR — Self-projection identity")
        print(f"  The Sempaevum's four defining constants all land on d=12:\n")
        for name, val in [("N = 12", mpmath.mpf(N)), ("V = 1/12", V_BASE),
                           ("K = 2/3", K_KOIDE), ("1/K = 3/2", mpmath.mpf(1)/K_KOIDE)]:
            proj = et_project(val, N)
            if proj:
                k, d, eps = proj
                print(f"    Π₁₂({name:12s}) = (k={k:>3d}, d={d:>2d}, ε={mpmath.nstr(eps,10):>16s}¢)")
        print(f"\n  All → d=12, |ε| = {mpmath.nstr(KOIDE_ATTRACTOR_EPS,10)}¢ = Pythagorean comma")

    elif cmd == "compare" and len(sys.argv) >= 4:
        systems = {}
        for key in sys.argv[2:]:
            k_low = key.lower()
            if k_low in SCALE_REGISTRY:
                nm, fn = SCALE_REGISTRY[k_low]; systems[nm] = fn()
            elif k_low.endswith("tet"):
                try:
                    n = int(k_low.replace("tet",""))
                    systems[f"{n}-TET"] = gen_equal_temperament(n)
                except ValueError: pass
        if len(systems) >= 2:
            comps = compare_tuning_systems(systems)
            shared = find_structural_connections(comps)
            print_cross_cultural(shared)
        else:
            print("  Need at least 2 valid scale names to compare.")

    elif cmd == "elegance" and len(sys.argv) >= 3:
        key = sys.argv[2].lower()
        if key in SCALE_REGISTRY:
            nm, fn = SCALE_REGISTRY[key]; print_elegance(nm, fn())
        else: print(f"  Unknown: {key}")

    elif cmd == "scala" and len(sys.argv) >= 3:
        key = sys.argv[2].lower()
        if key in SCALE_REGISTRY:
            nm, fn = SCALE_REGISTRY[key]; r = fn()
            p = sys.argv[3] if len(sys.argv) > 3 else f"/mnt/user-data/outputs/{key}.scl"
            content = generate_scala_file(nm, r, p)
            print(f"  Written: {p}")
            print(content)
        else: print(f"  Unknown: {key}. Run with no args to see all systems.")

    elif cmd == "scale" and len(sys.argv) >= 3:
        key = sys.argv[2].lower()
        if key.endswith("tet"):
            try:
                n = int(key.replace("tet",""))
                print_scale(f"{n}-TET", gen_equal_temperament(n))
                return
            except ValueError: pass
        if key in SCALE_REGISTRY:
            nm, fn = SCALE_REGISTRY[key]; print_scale(nm, fn())
        else: print(f"  Unknown: {key}. Run with no args to see all systems.")

    elif cmd == "scales":
        for key in sorted(SCALE_REGISTRY.keys()):
            nm, _ = SCALE_REGISTRY[key]
            print(f"  {key:<20s}  {nm}")

    elif cmd == "lattice2freq" and len(sys.argv) >= 3:
        k = int(sys.argv[2]); eps = mpmath.mpf(sys.argv[3]) if len(sys.argv) > 3 else mpmath.mpf(0)
        hz = lattice_to_freq(k, eps)
        print(f"  (k={k}, ε={mpmath.nstr(eps,8)}¢) → {mpmath.nstr(hz, 15)} Hz")

    elif cmd == "freqs" and len(sys.argv) >= 3:
        print_frequencies(sys.argv[2:])

    else:
        print(f"  Unknown command: {cmd}. Run with no args for help.")

if __name__ == "__main__":
    cli()
