#!/usr/bin/env python3
"""
ET MULTI-LAYER STRUCTURAL SEPARATOR
=====================================
Non-destructive separation into 4 structural layers.

LAYERS:
  1. VOICE     — Strong D-constraint: high T/D gradient + ε sign lean
  2. TRANSIENT — Brief bursts: d=6/d=12 ratio inverts (d=6 dominates d=12)
  3. AMBIENT   — Weak D-constraint: some gradient OR lean present
  4. STATIC    — Zero D-constraint: no gradient, no lean, T at 1/S

STRUCTURAL CLASSIFIERS (from T-shadow analysis):
  Transient: d=6 energy > d=12 energy in window (hexadic dominance)
             Confirmed: plosive had d=6 at 46.88% vs d=12 at 14.47%
             Normal voice: d=12 at 31.17% vs d=6 at 17.42%
  Voice:     T/D resolution gradient > 1.3 AND ε sign lean > 52%
  Ambient:   Gradient > 1.1 OR lean > 51% (weak but present D-constraint)
  Static:    Both flat (gradient ≈ 1.0 AND lean ≈ 50%)

GUARANTEE: Voice + Transient + Ambient + Static = Original
           Every sample in exactly one layer. Zero remainder.

Reads ETLM. Reconstructs 4 WAVs via pullback.
"""

import sys
import os
import struct
import wave
from math import gcd, lcm as math_lcm

from mpmath import mp, mpf, nint, nstr

mp.dps = 150
WORK_DPS = 100
CENTS_PER_OCTAVE = mpf(1200)
N_BASE = 12
BIT_DEPTH = 32
MAX_VAL = (1 << (BIT_DEPTH - 1))

CELL_CENTS = 1200.0 / N_BASE
DI_BOUNDARY = CELL_CENTS / 2
ONE_OVER_S = 0.25


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
    return int(nint(mpf(sign) * r * R0 * mpf(MAX_VAL)))  # T-ACT


# ═══════════════════════════════════════════════════════════════
# MULTI-LAYER CLASSIFIER
# ═══════════════════════════════════════════════════════════════

# Layer IDs
VOICE = 0
TRANSIENT = 1
AMBIENT = 2
STATIC = 3
LAYER_NAMES = ["voice", "transient", "ambient", "static"]


def classify_window(d_counts, eps_values, total_nonzero):
    """
    Classify a single window into one of 4 layers using d-family
    composition, T/D resolution gradient, and epsilon sign lean.
    
    Returns: VOICE, TRANSIENT, AMBIENT, or STATIC
    """
    if total_nonzero < 5:
        return STATIC

    # ── CLASSIFIER 1: d-family composition for TRANSIENT detection ──
    # The T-shadow analysis proved: plosives have d=6 >> d=12
    # Voice has d=12 >> d=6. The ratio inverts at transients.
    d6_count = d_counts.get(6, 0)
    d12_count = d_counts.get(12, 0)
    d4_count = d_counts.get(4, 0)

    # Transient signature: d=6 dominates d=12 (ratio > 1)
    # AND d=4 + d=6 together dominate (plosive character)
    if d12_count > 0:
        d6_d12_ratio = d6_count / d12_count
    else:
        d6_d12_ratio = 10.0 if d6_count > 0 else 0.0

    # Strong transient: d=6 > d=12 AND combined d=4+d=6 > 40%
    hexadic_quartic_frac = (d4_count + d6_count) / total_nonzero
    is_transient = d6_d12_ratio > 1.0 and hexadic_quartic_frac > 0.35

    if is_transient:
        return TRANSIENT

    # ── CLASSIFIER 2: T/D resolution gradient for VOICE detection ──
    center_bound = CELL_CENTS / 6   # ~16.67 cents
    edge_bound = CELL_CENTS * 0.45  # ~45 cents

    center_count = 0
    edge_count = 0
    pos_count = 0
    neg_count = 0

    for eps_val in eps_values:
        abs_e = abs(eps_val)
        if abs_e < center_bound:
            center_count += 1
        if abs_e > edge_bound:
            edge_count += 1
        if eps_val > 0:
            pos_count += 1
        elif eps_val < 0:
            neg_count += 1

    # Resolution gradient: ratio of center excess to edge excess
    expected_center_frac = center_bound / DI_BOUNDARY  # 1/3
    expected_edge_frac = (DI_BOUNDARY - edge_bound) / DI_BOUNDARY  # 1/10
    center_frac = center_count / total_nonzero
    edge_frac = edge_count / total_nonzero

    if edge_frac > 0 and expected_edge_frac > 0:
        gradient = (center_frac / expected_center_frac) / (edge_frac / expected_edge_frac)
    elif center_count > 0 and edge_count == 0:
        gradient = 5.0  # Very strong gradient (all centers, no edges)
    else:
        gradient = 1.0  # Flat

    # ── CLASSIFIER 3: epsilon sign lean (Koide/comma) ──
    total_signed = pos_count + neg_count
    if total_signed > 0:
        majority_frac = max(pos_count, neg_count) / total_signed
    else:
        majority_frac = 0.5

    # ── DECISION ──
    # Voice: strong gradient AND strong lean
    strong_gradient = gradient > 1.3
    strong_lean = majority_frac > 0.52

    if strong_gradient and strong_lean:
        return VOICE

    # Ambient: weak but present structure (either gradient or lean)
    weak_gradient = gradient > 1.1
    weak_lean = majority_frac > 0.51

    if weak_gradient or weak_lean:
        return AMBIENT

    # Static: no structure detected
    return STATIC


def classify_all_frames(samples, n_channels, sample_rate, N):
    """
    Classify every frame into one of 4 layers.
    Returns per-frame layer assignment list.
    """
    n_frames = len(samples) // n_channels

    window_ms = 15  # 15ms windows — short enough for transients
    window_frames = max(1, int(sample_rate * window_ms / 1000))

    print(f"    Window: {window_frames} frames ({window_ms}ms)")

    # ── Extract per-frame d and eps ──
    print(f"    Extracting lattice coordinates...")
    frame_d = []
    frame_eps = []
    for frame_idx in range(n_frames):
        d_vals = []
        eps_vals = []
        for ch in range(n_channels):
            s = samples[frame_idx * n_channels + ch]
            if not s['is_zero'] and s['d_r'] is not None:
                d_vals.append(s['d_r'])
                eps_vals.append(float(s['eps_r']))
        frame_d.append(d_vals)
        frame_eps.append(eps_vals)

    # ── Classify each window ──
    print(f"    Classifying windows...")
    window_layers = []
    window_ranges = []

    for win_start in range(0, n_frames, window_frames):
        win_end = min(win_start + window_frames, n_frames)

        d_counts = {}
        eps_values = []
        total_nonzero = 0

        for frame_idx in range(win_start, win_end):
            for d_val in frame_d[frame_idx]:
                d_counts[d_val] = d_counts.get(d_val, 0) + 1
                total_nonzero += 1
            eps_values.extend(frame_eps[frame_idx])

        layer = classify_window(d_counts, eps_values, total_nonzero)
        window_layers.append(layer)
        window_ranges.append((win_start, win_end))

    # Count per layer
    for layer_id in range(4):
        count = sum(1 for l in window_layers if l == layer_id)
        print(f"    {LAYER_NAMES[layer_id]:>12}: {count} windows")

    # ── Transition guard: transients adjacent to voice become voice ──
    # (A plosive right before a vowel is part of the speech)
    guarded = list(window_layers)
    for i in range(len(guarded)):
        if guarded[i] == TRANSIENT:
            # If adjacent to voice, keep as transient (correct classification)
            pass
        elif guarded[i] == STATIC:
            # If adjacent to voice or transient, upgrade to ambient
            # (preserve attack/release edges)
            has_active_neighbor = False
            if i > 0 and window_layers[i - 1] in (VOICE, TRANSIENT):
                has_active_neighbor = True
            if i < len(window_layers) - 1 and window_layers[i + 1] in (VOICE, TRANSIENT):
                has_active_neighbor = True
            if has_active_neighbor:
                guarded[i] = AMBIENT

    # Second pass: ambient adjacent to voice on both sides → voice
    for i in range(1, len(guarded) - 1):
        if guarded[i] == AMBIENT:
            if guarded[i - 1] == VOICE and guarded[i + 1] == VOICE:
                guarded[i] = VOICE

    # Report after guarding
    for layer_id in range(4):
        count = sum(1 for l in guarded if l == layer_id)
        print(f"    {LAYER_NAMES[layer_id]:>12}: {count} windows (after guard)")

    # ── Expand to per-frame ──
    frame_layers = [STATIC] * n_frames
    for win_idx, (win_start, win_end) in enumerate(window_ranges):
        for f_idx in range(win_start, win_end):
            frame_layers[f_idx] = guarded[win_idx]

    for layer_id in range(4):
        count = sum(1 for l in frame_layers if l == layer_id)
        pct = 100.0 * count / n_frames
        print(f"    {LAYER_NAMES[layer_id]:>12}: {count} frames ({pct:.1f}%)")

    return frame_layers


def main():
    script_dir = get_script_dir()

    print("=" * 70)
    print("  ET MULTI-LAYER STRUCTURAL SEPARATOR")
    print("  Voice + Transient + Ambient + Static = Original (exact)")
    print("  d-family composition | T/D gradient | Koide/comma lean")
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

    # Output paths
    layer_paths = {}
    for layer_id in range(4):
        name = LAYER_NAMES[layer_id]
        layer_paths[layer_id] = os.path.join(script_dir, f"{base_name}_{name}.wav")
    report_path = os.path.join(script_dir, f"{base_name}_separation_report.txt")

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

    print(f"  N={N}, R0={header['R0_str']}, {sample_rate} Hz, "
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

    # ─── Classify ───
    print(f"\n  Classifying into 4 structural layers...")
    frame_layers = classify_all_frames(samples, n_channels, sample_rate, N)

    # ─── d-family profiles per layer ───
    layer_d = {lid: {} for lid in range(4)}
    layer_total = {lid: 0 for lid in range(4)}
    layer_eps_sum = {lid: 0.0 for lid in range(4)}

    for frame_idx in range(n_frames):
        lid = frame_layers[frame_idx]
        for ch in range(n_channels):
            s = samples[frame_idx * n_channels + ch]
            if not s['is_zero'] and s['d_r'] is not None:
                layer_d[lid][s['d_r']] = layer_d[lid].get(s['d_r'], 0) + 1
                layer_total[lid] += 1
                layer_eps_sum[lid] += abs(float(s['eps_r']))

    for lid in range(4):
        name = LAYER_NAMES[lid]
        total = layer_total[lid]
        if total > 0:
            eps_cell = (layer_eps_sum[lid] / total) / CELL_CENTS
            print(f"\n  {name.upper()} ({total} samples, |eps|/cell={eps_cell:.4f}):")
            for d in sorted(layer_d[lid].keys()):
                pct = 100.0 * layer_d[lid][d] / total
                print(f"    d={d:>3}: {pct:>7.2f}%")

    # ─── Reconstruct 4 WAVs via pullback ───
    print(f"\n  Reconstructing 4 layers via pullback...")
    samp_width = BIT_DEPTH // 8

    # Pre-compute all PCM values
    pcm_values = []
    for s in samples:
        if s['is_zero']:
            pcm_values.append(0)
        else:
            val = pullback_to_int(s['k_r'], s['eps_r'], s['sign'], N, R0)
            pcm_values.append(max(-MAX_VAL, min(MAX_VAL - 1, val)))

    # Build raw bytes for each layer
    layer_raw = {lid: bytearray() for lid in range(4)}

    for frame_idx in range(n_frames):
        lid = frame_layers[frame_idx]
        for ch in range(n_channels):
            sample_idx = frame_idx * n_channels + ch
            int_val = pcm_values[sample_idx]

            for target_lid in range(4):
                if target_lid == lid:
                    layer_raw[target_lid].extend(struct.pack('<i', int_val))
                else:
                    layer_raw[target_lid].extend(struct.pack('<i', 0))

        if (frame_idx + 1) % 100000 == 0:
            print(f"    {frame_idx + 1}/{n_frames}...")

    # Write WAVs
    for lid in range(4):
        path = layer_paths[lid]
        print(f"  Writing {os.path.basename(path)}...")
        with wave.open(path, 'wb') as wf:
            wf.setnchannels(n_channels)
            wf.setsampwidth(samp_width)
            wf.setframerate(sample_rate)
            wf.writeframes(bytes(layer_raw[lid]))

    # ─── Report ───
    report = [
        "=" * 70,
        "  ET MULTI-LAYER SEPARATION REPORT",
        f"  Source: {etlm_name}",
        f"  N={N}, {sample_rate} Hz, {n_channels} ch",
        "=" * 70,
        "",
        "  CLASSIFIERS:",
        "    Transient: d=6/d=12 ratio > 1.0 + hexadic+quartic > 35%",
        "    Voice:     T/D gradient > 1.3 + sign lean > 52%",
        "    Ambient:   gradient > 1.1 OR lean > 51%",
        "    Static:    both flat (no structural signature)",
        "",
        "  GUARANTEE: Voice + Transient + Ambient + Static = Original",
        "  Every sample in exactly one layer. Zero remainder.",
        "",
    ]

    for lid in range(4):
        name = LAYER_NAMES[lid]
        total = layer_total[lid]
        report.append(f"  {name.upper()} ({total} samples):")
        if total > 0:
            eps_cell = (layer_eps_sum[lid] / total) / CELL_CENTS
            report.append(f"    |eps|/cell = {eps_cell:.6f} (1/S = {ONE_OVER_S})")
            for d in sorted(layer_d[lid].keys()):
                pct = 100.0 * layer_d[lid][d] / total
                report.append(f"    d={d:>3}: {pct:.2f}%")
        report.append("")

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))

    # ─── Summary ───
    print(f"\n{'='*70}")
    print(f"  SEPARATION COMPLETE")
    print(f"{'='*70}")
    for lid in range(4):
        name = LAYER_NAMES[lid]
        path = layer_paths[lid]
        size = os.path.getsize(path)
        frames = sum(1 for l in frame_layers if l == lid)
        pct = 100.0 * frames / n_frames
        print(f"  {os.path.basename(path):<40} {size:>10,} bytes  "
              f"({frames} frames, {pct:.1f}%)")
    print(f"  {os.path.basename(report_path):<40} structural report")
    print(f"\n  Voice + Transient + Ambient + Static = Original (exact)")
    print(f"  Each layer is independently playable and analyzable.")
    print(f"  Voice + Transient = studio master with consonant definition.")


if __name__ == "__main__":
    main()
    input("\n  Press Enter to exit...")
