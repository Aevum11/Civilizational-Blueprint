#!/usr/bin/env python3
"""
ET STATIC REMOVAL — T-SHADOW METHOD
======================================
Derived from the T-Shadow Analysis (Muller, 2026).

THE PRINCIPLE:
  Every sample is a T-act. T's character changes with D-constraint:

  T under D-constraint (voice, room, breath):
    - T/D resolution gradient: energy concentrates at cell CENTERS
      (T/D = 1.655 at |ε|<5¢) and depletes at ∂I (T/D = 0.719 at |ε|>45¢)
    - ε sign bias: D leans Koide (ε<0), T leans comma (ε>0)
    - |ε|/cell deviates from 1/S = 1/4

  T WITHOUT D-constraint (electronic static):
    - No gradient: energy uniform across ε (T/D ≈ 1 everywhere)
    - No sign bias: ε equally positive and negative
    - |ε|/cell = 1/S = 1/4 exactly

  The T-shadow gradient IS the classifier. No statistics needed.
  The gradient is a structural property of the lattice, measured
  against the manifold constant 1/S = 1/4 (S=4 states).

ALGORITHM:
  For each window:
    1. Measure the resolution gradient: center energy vs ∂I energy
    2. Measure the sign lean: positive ε count vs negative ε count
    3. BOTH flat → T without D-constraint → static → zero
    4. EITHER shows structure → T under D → signal → keep ALL
  Guard: windows adjacent to signal are preserved.

CONSERVATIVE: only removes windows with ZERO structural signature.
Anything with even weak D-constraint is kept.

Reads ETLM. Reconstructs cleaned WAV via pullback.
"""

import sys
import os
import struct
import wave
from math import gcd, lcm as math_lcm

from mpmath import mp, mpf, nint, nstr, fabs as mpfabs

mp.dps = 150
WORK_DPS = 100
CENTS_PER_OCTAVE = mpf(1200)
N_BASE = 12
BIT_DEPTH = 32
MAX_VAL = (1 << (BIT_DEPTH - 1))

# Structural constants
S = 4                          # Manifold states {E, M, I, U}
ONE_OVER_S = 1.0 / S           # T's indeterminate face: |ε|/cell at zero D-constraint
CELL_CENTS = 1200.0 / N_BASE   # 100¢ at N=12
DI_BOUNDARY = CELL_CENTS / 2   # 50¢ — ∂I boundary
COMMA_CENTS = 1.955             # Koide-comma attractor pair


def get_script_dir():
    return os.path.dirname(os.path.abspath(__file__))


# ═══════════════════════════════════════════════════════════════
# ETLM DECODING
# ═══════════════════════════════════════════════════════════════
MAGIC = b"ETLM"


def decode_header(data):
    offset = 0
    assert data[offset:offset + 4] == MAGIC, "Not an ETLM file"
    offset += 4
    version = struct.unpack_from('<H', data, offset)[0]; offset += 2
    sample_rate = struct.unpack_from('<I', data, offset)[0]; offset += 4
    channels = struct.unpack_from('<H', data, offset)[0]; offset += 2
    N = struct.unpack_from('<I', data, offset)[0]; offset += 4
    R0_len = struct.unpack_from('<I', data, offset)[0]; offset += 4
    R0_str = data[offset:offset + R0_len].decode('utf-8'); offset += R0_len
    dps = struct.unpack_from('<I', data, offset)[0]; offset += 4
    total_samples = struct.unpack_from('<Q', data, offset)[0]; offset += 8
    return {
        'version': version, 'sample_rate': sample_rate,
        'channels': channels, 'N': N, 'R0_str': R0_str,
        'dps': dps, 'total_samples': total_samples
    }, offset


def decode_sample(data, offset, N):
    flags = data[offset]; offset += 1
    sign = -1 if (flags & 0x01) else 1
    is_zero = bool(flags & 0x02)
    if is_zero:
        return {'k_r': None, 'd_r': None, 'eps_r': mpf(0),
                'sign': 0, 'is_zero': True}, offset
    k_len = struct.unpack_from('<H', data, offset)[0]; offset += 2
    k_r = int.from_bytes(data[offset:offset + k_len], byteorder='little', signed=True)
    offset += k_len
    eps_len = struct.unpack_from('<H', data, offset)[0]; offset += 2
    eps_r = mpf(data[offset:offset + eps_len].decode('utf-8'))
    offset += eps_len
    g = gcd(abs(k_r), N) if k_r != 0 else N
    d_r = N // g
    return {'k_r': k_r, 'd_r': d_r, 'eps_r': eps_r,
            'sign': sign, 'is_zero': False}, offset


def pullback_to_int(k, eps, sign, N, R0):
    exponent = (mpf(k) + eps * mpf(N) / CENTS_PER_OCTAVE) / mpf(N)
    r = mpf(2) ** exponent
    return int(nint(mpf(sign) * r * R0 * mpf(MAX_VAL)))  # ← T-ACT


# ═══════════════════════════════════════════════════════════════
# T-SHADOW CLASSIFIER
# ═══════════════════════════════════════════════════════════════

def classify_by_t_shadow(samples, n_channels, sample_rate, N):
    """
    Classify frames using T-shadow structural properties.
    
    For each window:
      1. Resolution gradient: count samples at cell centers (|ε| < cell/6)
         vs at ∂I edge (|ε| > cell×0.9). Signal has MORE centers than edges.
         Static has equal centers and edges.
      2. ε sign lean: count positive ε vs negative ε.
         Signal has a bias (D→Koide or T→comma). Static has 50/50.
      3. BOTH flat → static. EITHER structured → signal.
    """
    n_frames = len(samples) // n_channels
    cell = CELL_CENTS  # 100¢ at N=12

    # Thresholds from T-shadow analysis (structural, from 1/S and K)
    center_bound = cell / 6      # ~16.67¢ — inner sixth of cell
    edge_bound = cell * 0.9 / 2  # ~45¢ — outer tenth of half-cell (near ∂I)

    window_ms = 30  # 30ms windows
    window_frames = max(1, int(sample_rate * window_ms / 1000))

    print(f"    Window: {window_frames} frames ({window_ms}ms)")
    print(f"    Center: |ε| < {center_bound:.1f}¢ | Edge: |ε| > {edge_bound:.1f}¢")
    print(f"    1/S = {ONE_OVER_S} (T's structural baseline)")

    # ── Compute per-frame ε values ──
    print(f"    Extracting ε from lattice coordinates...")
    frame_eps = []  # List of lists: for each frame, the ε values of all channels
    for frame_idx in range(n_frames):
        eps_vals = []
        for ch in range(n_channels):
            s = samples[frame_idx * n_channels + ch]
            if not s['is_zero'] and s['eps_r'] is not None:
                eps_vals.append(float(s['eps_r']))
        frame_eps.append(eps_vals)

    # ── Classify windows by T-shadow gradient ──
    print(f"    Measuring T-shadow gradient in each window...")
    window_is_signal = []
    window_ranges = []

    for win_start in range(0, n_frames, window_frames):
        win_end = min(win_start + window_frames, n_frames)

        # Collect all ε in this window
        center_count = 0
        edge_count = 0
        pos_count = 0
        neg_count = 0
        total_eps = 0

        for frame_idx in range(win_start, win_end):
            for eps_val in frame_eps[frame_idx]:
                abs_eps = abs(eps_val)
                total_eps += 1

                # Resolution gradient measurement
                if abs_eps < center_bound:
                    center_count += 1
                if abs_eps > edge_bound:
                    edge_count += 1

                # Sign lean measurement
                if eps_val > 0:
                    pos_count += 1
                elif eps_val < 0:
                    neg_count += 1

        # ── Structural classification ──
        has_gradient = False
        has_sign_lean = False

        if total_eps >= 10:
            # CHECK 1: Resolution gradient
            # For uniform ε (static): center fraction ≈ center_bound/50 = 1/3
            # For D-constrained T (signal): center fraction > 1/3
            # The T-shadow shows T/D = 1.655 at centers — significant excess
            center_frac = center_count / total_eps
            edge_frac = edge_count / total_eps
            expected_center = center_bound / DI_BOUNDARY  # 1/3 for uniform
            expected_edge = (DI_BOUNDARY - edge_bound) / DI_BOUNDARY  # 1/10 for uniform

            # Gradient present if center exceeds edge by the structural ratio
            # The T-shadow ratio is 1.655/0.719 = 2.3 center-to-edge
            if center_count > 0 and edge_count > 0:
                measured_ratio = (center_frac / expected_center) / (edge_frac / expected_edge)
                has_gradient = measured_ratio > 1.3  # Structural threshold from T-shadow
            elif center_count > 0 and edge_count == 0:
                has_gradient = True  # All centers, no edges = very structured
            # If all edges, no centers — unusual but possible, keep it

            # CHECK 2: ε sign lean (Koide/comma separation)
            # The T-shadow shows D at 53% negative, T at 51% positive
            # Static would be 50/50. Any bias > 52% indicates D or T character.
            if pos_count + neg_count > 0:
                majority_frac = max(pos_count, neg_count) / (pos_count + neg_count)
                has_sign_lean = majority_frac > 0.52  # 52% = weakest detected bias

        # DECISION: Both flat → static. Either structured → signal.
        is_signal = has_gradient or has_sign_lean or total_eps < 10

        window_is_signal.append(is_signal)
        window_ranges.append((win_start, win_end))

    n_sig_wins = sum(window_is_signal)
    n_stat_wins = len(window_is_signal) - n_sig_wins
    print(f"    Signal windows: {n_sig_wins}")
    print(f"    Static windows: {n_stat_wins}")

    # ── Transition guard: preserve edges ──
    guarded = list(window_is_signal)
    for i in range(len(guarded)):
        if not guarded[i]:
            # Keep if adjacent to signal (preserve attack/release)
            if i > 0 and window_is_signal[i - 1]:
                guarded[i] = True
            elif i < len(window_is_signal) - 1 and window_is_signal[i + 1]:
                guarded[i] = True

    n_guard = sum(guarded) - n_sig_wins
    if n_guard > 0:
        print(f"    Transition guard: {n_guard} windows preserved")

    # ── Expand to per-frame ──
    frame_is_signal = [False] * n_frames
    for win_idx, (win_start, win_end) in enumerate(window_ranges):
        if guarded[win_idx]:
            for f_idx in range(win_start, win_end):
                frame_is_signal[f_idx] = True

    sig_frames = sum(frame_is_signal)
    stat_frames = n_frames - sig_frames
    print(f"    Signal frames: {sig_frames} ({100.0 * sig_frames / n_frames:.1f}%)")
    print(f"    Static frames: {stat_frames} ({100.0 * stat_frames / n_frames:.1f}%)")

    return frame_is_signal


def main():
    script_dir = get_script_dir()

    print("=" * 70)
    print("  ET STATIC REMOVAL — T-SHADOW METHOD")
    print("  T/D resolution gradient + ε sign lean (Koide/comma)")
    print("  Static = T without D-constraint (flat gradient, no lean)")
    print("  Signal = T under D-constraint (gradient OR lean present)")
    print("  Reads ETLM → structural classification → clean WAV")
    print("=" * 70)

    # Find ETLM
    etlm_files = [f for f in os.listdir(script_dir) if f.lower().endswith('.etlm')]
    if not etlm_files:
        print(f"\n  No ETLM files found in: {script_dir}")
        input("  Press Enter to exit...")
        sys.exit(1)

    if len(etlm_files) == 1:
        etlm_name = etlm_files[0]
        print(f"\n  Found: {etlm_name}")
    else:
        print(f"\n  ETLM files:\n")
        for i, f in enumerate(etlm_files):
            size_mb = os.path.getsize(os.path.join(script_dir, f)) / (1024 * 1024)
            print(f"    [{i + 1}] {f} ({size_mb:.1f} MB)")
        try:
            choice = input(f"\n  Select [1-{len(etlm_files)}]: ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(etlm_files):
                etlm_name = etlm_files[idx]
            else:
                print("  Invalid.")
                input("  Press Enter to exit...")
                sys.exit(1)
        except (ValueError, EOFError, KeyboardInterrupt):
            sys.exit(1)

    etlm_path = os.path.join(script_dir, etlm_name)
    base_name = os.path.splitext(etlm_name)[0]
    clean_wav = os.path.join(script_dir, f"{base_name}_cleaned.wav")
    static_wav = os.path.join(script_dir, f"{base_name}_static_only.wav")
    report_path = os.path.join(script_dir, f"{base_name}_tshadow_report.txt")

    # ─── Read ETLM ───
    print(f"\n  Reading {etlm_name}...")
    with open(etlm_path, 'rb') as f:
        file_data = f.read()

    header, data_offset = decode_header(file_data)
    N = header['N']
    R0 = mpf(header['R0_str'])
    sample_rate = header['sample_rate']
    n_channels = header['channels']
    total_samples = header['total_samples']
    n_frames = total_samples // n_channels

    print(f"  N={N}, R₀={header['R0_str']}, {sample_rate} Hz, "
          f"{n_channels} ch, {total_samples} samples, {n_frames} frames")

    # ─── Decode ───
    print(f"\n  Decoding {total_samples} samples...")
    samples = []
    offset = data_offset
    for i in range(total_samples):
        sample, offset = decode_sample(file_data, offset, N)
        samples.append(sample)
        if (i + 1) % 200000 == 0:
            print(f"    {i + 1}/{total_samples}...")
    print(f"  Decoded.")

    # ─── T-shadow classification ───
    print(f"\n  T-shadow classification...")
    frame_is_signal = classify_by_t_shadow(samples, n_channels, sample_rate, N)

    # ─── d-family profiles ───
    signal_d = {}
    static_d = {}
    signal_total = 0
    static_total = 0
    signal_eps_sum = 0.0
    static_eps_sum = 0.0

    for frame_idx in range(n_frames):
        is_sig = frame_is_signal[frame_idx]
        for ch in range(n_channels):
            s = samples[frame_idx * n_channels + ch]
            if not s['is_zero'] and s['d_r'] is not None:
                target = signal_d if is_sig else static_d
                target[s['d_r']] = target.get(s['d_r'], 0) + 1
                eps_abs = abs(float(s['eps_r']))
                if is_sig:
                    signal_total += 1
                    signal_eps_sum += eps_abs
                else:
                    static_total += 1
                    static_eps_sum += eps_abs

    print(f"\n  SIGNAL ({signal_total} samples):")
    if signal_total > 0:
        sig_eps_cell = (signal_eps_sum / signal_total) / CELL_CENTS
        print(f"    |ε|/cell = {sig_eps_cell:.6f} (1/S = {ONE_OVER_S})")
    for d in sorted(signal_d.keys()):
        print(f"    d={d:>3}: {100.0 * signal_d[d] / max(1, signal_total):>7.2f}%")

    if static_total > 0:
        stat_eps_cell = (static_eps_sum / static_total) / CELL_CENTS
        print(f"\n  STATIC ({static_total} samples):")
        print(f"    |ε|/cell = {stat_eps_cell:.6f} (1/S = {ONE_OVER_S})")
        for d in sorted(static_d.keys()):
            print(f"    d={d:>3}: {100.0 * static_d[d] / max(1, static_total):>7.2f}%")

    # ─── Reconstruct ───
    print(f"\n  Reconstructing via pullback...")
    clean_raw = bytearray()
    static_raw = bytearray()

    for frame_idx in range(n_frames):
        is_sig = frame_is_signal[frame_idx]
        for ch in range(n_channels):
            s = samples[frame_idx * n_channels + ch]
            if s['is_zero']:
                int_val = 0
            else:
                int_val = pullback_to_int(s['k_r'], s['eps_r'], s['sign'], N, R0)
                int_val = max(-MAX_VAL, min(MAX_VAL - 1, int_val))

            if is_sig:
                clean_raw.extend(struct.pack('<i', int_val))
                static_raw.extend(struct.pack('<i', 0))
            else:
                clean_raw.extend(struct.pack('<i', 0))
                static_raw.extend(struct.pack('<i', int_val))

        if (frame_idx + 1) % 100000 == 0:
            print(f"    {frame_idx + 1}/{n_frames}...")

    samp_width = BIT_DEPTH // 8

    print(f"  Writing {os.path.basename(clean_wav)}...")
    with wave.open(clean_wav, 'wb') as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(samp_width)
        wf.setframerate(sample_rate)
        wf.writeframes(bytes(clean_raw))

    print(f"  Writing {os.path.basename(static_wav)}...")
    with wave.open(static_wav, 'wb') as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(samp_width)
        wf.setframerate(sample_rate)
        wf.writeframes(bytes(static_raw))

    # ─── Report ───
    report = [
        "=" * 70,
        "  ET STATIC REMOVAL — T-SHADOW REPORT",
        f"  Source: {etlm_name}",
        f"  N={N}, {sample_rate} Hz, {n_channels} ch",
        "=" * 70,
        "",
        "  Method: T-shadow structural classification",
        "  Classifier 1: T/D resolution gradient (center vs ∂I edge)",
        "  Classifier 2: ε sign lean (Koide/comma bias)",
        "  Static = BOTH flat (T without D-constraint)",
        "  Signal = EITHER structured (T under D-constraint)",
        "",
        f"  Signal frames: {sum(frame_is_signal)}",
        f"  Static frames: {n_frames - sum(frame_is_signal)}",
        f"  Signal samples: {signal_total}",
        f"  Static samples: {static_total}",
    ]

    if signal_total > 0:
        report.append(f"\n  Signal |ε|/cell = {signal_eps_sum/signal_total/CELL_CENTS:.6f}")
    if static_total > 0:
        report.append(f"  Static |ε|/cell = {static_eps_sum/static_total/CELL_CENTS:.6f}")
    report.append(f"  1/S = {ONE_OVER_S} (manifold constant, S=4)")

    report.append(f"\n  SIGNAL d-families:")
    for d in sorted(signal_d.keys()):
        report.append(f"    d={d:>3}: {100.0 * signal_d[d] / max(1, signal_total):.2f}%")

    if static_total > 0:
        report.append(f"\n  STATIC d-families:")
        for d in sorted(static_d.keys()):
            report.append(f"    d={d:>3}: {100.0 * static_d[d] / max(1, static_total):.2f}%")

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))

    print(f"\n{'═' * 70}")
    print(f"  COMPLETE")
    print(f"{'═' * 70}")
    print(f"  {os.path.basename(clean_wav):<45} signal (static zeroed)")
    print(f"  {os.path.basename(static_wav):<45} static only (isolated)")
    print(f"  {os.path.basename(report_path):<45} T-shadow report")
    print(f"\n  Play the static-only file to verify what was removed.")
    print(f"  Signal samples were NEVER modified.")


if __name__ == "__main__":
    main()
    input("\n  Press Enter to exit...")
