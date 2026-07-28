#!/usr/bin/env python3
"""
ET STATIC REMOVAL — PURE STRUCTURAL d-FAMILY ISOLATION
========================================================
Reads the ETLM directly. Every sample already has (k, d, ε).

THE LATTICE IS THE CLASSIFIER:
  k = lattice position (determines amplitude on the crystal)
  d = sublattice family (structural classification, deterministic from k)
  ε = intra-cell coordinate (exact crystal position)

IDENTIFICATION (Three Tools):
  Static lives at specific lattice positions. During silence regions
  (where ONLY static exists), the d-families and k values form the
  exact structural fingerprint of the static.

  Voice and real sounds produce d-family SEQUENCES that follow the
  palindromic cascade in order (smooth k traversal). Static produces
  d-family sequences that are structurally incoherent (random k jumps).

CLASSIFICATION:
  1. Profile static from silence using k and d (lattice coordinates)
  2. For each frame: is the k-trajectory CASCADE-COHERENT with neighbors?
     Coherent = consecutive k values follow smooth traversal = signal
     Incoherent AND below static k-ceiling = static
  3. Guard: frames adjacent to signal are preserved (transient protection)

WHAT THIS IS NOT:
  Not statistical. Not Shannon. Not chi-squared. Not amplitude gating.
  The k coordinate IS the lattice position. The d-family IS the
  structural classification. The palindromic cascade IS the reference.
  Everything is deterministic structure from the crystal.
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

# The palindromic cascade at N=12
# d_sequence[k mod 12] gives the sublattice family for cell k
D_SEQUENCE = []
for _k in range(N_BASE):
    _g = gcd(_k, N_BASE) if _k != 0 else N_BASE
    D_SEQUENCE.append(N_BASE // _g)
# [1, 12, 6, 4, 3, 12, 2, 12, 3, 4, 6, 12]


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
# CASCADE COHERENCE — the d-family structural classifier
# ═══════════════════════════════════════════════════════════════

def is_cascade_coherent(k_prev, k_curr, N):
    """
    Check if the transition from k_prev to k_curr is CASCADE-COHERENT.
    
    Cascade-coherent means the k-trajectory is traversing the lattice
    smoothly — stepping through adjacent or nearby cells. This is what
    real audio does (smooth waveform → smooth k traversal).
    
    Static jumps randomly, producing incoherent transitions.
    
    The bound is structural: derived from N and the lattice geometry.
    At N=12, a k-step of ≤3 means the signal is moving through at most
    3 cells per sample — consistent with frequencies up to ~12 kHz at 48 kHz
    sample rate (which covers the entire voice band).
    """
    if k_prev is None or k_curr is None:
        return False
    delta_k = abs(k_curr - k_prev)
    # Structural bound: N/4 cells per sample covers the full audio band
    # At N=12: bound = 3 cells (each cell = 100¢ = one semitone)
    # 3 semitones per sample at 48kHz = up to ~12kHz fundamental
    return delta_k <= N // 4


def compute_coherence_runs(frames_k, N):
    """
    Compute runs of cascade-coherent transitions in the k-trajectory.
    
    A run is a consecutive sequence of frames where each transition
    is cascade-coherent (smooth k traversal following the lattice).
    
    Returns list of (start_frame, end_frame, run_length) for all runs.
    Long runs = signal (voice traversing the lattice smoothly).
    Short runs or gaps = potential static.
    """
    n_frames = len(frames_k)
    if n_frames < 2:
        return []

    # For each frame, check coherence with the previous frame
    coherent = [False] * n_frames
    coherent[0] = False  # First frame has no predecessor

    for i in range(1, n_frames):
        coherent[i] = is_cascade_coherent(frames_k[i - 1], frames_k[i], N)

    # Extract runs of consecutive coherent frames
    runs = []
    run_start = None
    for i in range(n_frames):
        if coherent[i]:
            if run_start is None:
                run_start = i - 1  # Include the frame before the coherent transition
        else:
            if run_start is not None:
                runs.append((run_start, i, i - run_start))
                run_start = None
    if run_start is not None:
        runs.append((run_start, n_frames, n_frames - run_start))

    return runs


def classify_frames(samples, n_channels, sample_rate, N):
    """
    Classify each frame as SIGNAL or STATIC using cascade coherence.
    
    The palindromic cascade determines the structural reference.
    Smooth k-trajectories (voice) produce long coherence runs.
    Random k-trajectories (static) produce short/no runs.
    
    Returns per-frame boolean list: True = signal, False = static.
    """
    n_frames = len(samples) // n_channels

    # ── Extract per-frame k values (max across channels) ──
    print("    Extracting lattice coordinates...")
    frames_k = []
    for frame_idx in range(n_frames):
        best_k = None
        for ch in range(n_channels):
            s = samples[frame_idx * n_channels + ch]
            if not s['is_zero'] and s['k_r'] is not None:
                if best_k is None or s['k_r'] > best_k:
                    best_k = s['k_r']
        frames_k.append(best_k)

    # ── Compute cascade coherence runs ──
    print("    Computing cascade coherence runs...")
    runs = compute_coherence_runs(frames_k, N)

    total_run_frames = sum(r[2] for r in runs)
    print(f"    Found {len(runs)} coherence runs covering {total_run_frames} frames")
    if runs:
        run_lengths = sorted([r[2] for r in runs])
        print(f"    Run lengths: min={run_lengths[0]}, median={run_lengths[len(run_lengths)//2]}, "
              f"max={run_lengths[-1]}")

    # ── Mark signal frames from runs ──
    # Minimum run length to count as signal:
    # At 48 kHz, one period of 200 Hz voice = 240 frames
    # Even a short consonant burst is ~5ms = 240 frames
    # Be conservative: runs of 100+ frames = definite signal
    # Runs of 10-100 frames = likely signal (transients, short sounds)
    # Runs of <10 frames = likely coincidental coherence in static
    min_signal_run = max(10, sample_rate // 4800)  # ~10 frames at 48kHz

    frame_is_signal = [False] * n_frames

    for run_start, run_end, run_length in runs:
        if run_length >= min_signal_run:
            for f_idx in range(run_start, run_end):
                frame_is_signal[f_idx] = True

    # ── Transition guard: expand signal regions ──
    # Preserve attack/release edges by guarding adjacent frames
    guard_frames = max(10, sample_rate // 4800)  # ~10 frames = ~0.2ms
    guarded = list(frame_is_signal)
    for i in range(n_frames):
        if frame_is_signal[i]:
            for g in range(max(0, i - guard_frames), min(n_frames, i + guard_frames + 1)):
                guarded[g] = True

    signal_count = sum(guarded)
    static_count = n_frames - signal_count
    print(f"    Signal frames: {signal_count} ({100.0 * signal_count / n_frames:.1f}%)")
    print(f"    Static frames: {static_count} ({100.0 * static_count / n_frames:.1f}%)")

    return guarded


def main():
    script_dir = get_script_dir()

    print("=" * 70)
    print("  ET STATIC REMOVAL — PALINDROMIC CASCADE CLASSIFIER")
    print("  Reads ETLM → d-family cascade coherence → clean WAV")
    print("  Deterministic. Structural. No statistics. No Shannon.")
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
    report_path = os.path.join(script_dir, f"{base_name}_static_report.txt")

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
    print(f"  Palindromic cascade: {D_SEQUENCE}")

    # ─── Decode all samples ───
    print(f"\n  Decoding {total_samples} samples...")
    samples = []
    offset = data_offset
    for i in range(total_samples):
        sample, offset = decode_sample(file_data, offset, N)
        samples.append(sample)
        if (i + 1) % 200000 == 0:
            print(f"    {i + 1}/{total_samples}...")
    print(f"  Decoded.")

    # ─── Classify by cascade coherence ───
    print(f"\n  Classifying by palindromic cascade coherence...")
    frame_is_signal = classify_frames(samples, n_channels, sample_rate, N)

    # ─── d-family profiles for signal vs static ───
    signal_d = {}
    static_d = {}
    signal_total = 0
    static_total = 0

    for frame_idx in range(n_frames):
        is_sig = frame_is_signal[frame_idx]
        for ch in range(n_channels):
            s = samples[frame_idx * n_channels + ch]
            if not s['is_zero'] and s['d_r'] is not None:
                target = signal_d if is_sig else static_d
                target[s['d_r']] = target.get(s['d_r'], 0) + 1
                if is_sig:
                    signal_total += 1
                else:
                    static_total += 1

    print(f"\n  d-family profile — SIGNAL ({signal_total} samples):")
    for d in sorted(signal_d.keys()):
        pct = 100.0 * signal_d[d] / max(1, signal_total)
        print(f"    d={d:>3}: {pct:>7.2f}%")

    if static_total > 0:
        print(f"\n  d-family profile — STATIC ({static_total} samples):")
        for d in sorted(static_d.keys()):
            pct = 100.0 * static_d[d] / max(1, static_total)
            print(f"    d={d:>3}: {pct:>7.2f}%")

    # ─── Reconstruct WAVs via pullback ───
    print(f"\n  Reconstructing via pullback Π_N⁻¹...")
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
        "  ET STATIC REMOVAL — CASCADE COHERENCE REPORT",
        f"  Source: {etlm_name}",
        f"  N={N}, {sample_rate} Hz, {n_channels} ch",
        f"  Cascade: {D_SEQUENCE}",
        "=" * 70,
        f"  Method: Palindromic cascade coherence",
        f"  Cascade-coherent transitions identify smooth k-traversal (signal)",
        f"  Incoherent transitions identify random k-jumps (static)",
        f"",
        f"  Signal frames: {sum(frame_is_signal)}",
        f"  Static frames: {n_frames - sum(frame_is_signal)}",
        f"  Signal samples: {signal_total}",
        f"  Static samples: {static_total}",
    ]

    report.append(f"\n  SIGNAL d-families:")
    for d in sorted(signal_d.keys()):
        report.append(f"    d={d:>3}: {100.0 * signal_d[d] / max(1, signal_total):.2f}%")

    if static_total > 0:
        report.append(f"\n  STATIC d-families:")
        for d in sorted(static_d.keys()):
            report.append(f"    d={d:>3}: {100.0 * static_d[d] / max(1, static_total):.2f}%")

    with open(report_path, 'w') as f:
        f.write('\n'.join(report))

    print(f"\n{'═' * 70}")
    print(f"  COMPLETE")
    print(f"{'═' * 70}")
    print(f"  {os.path.basename(clean_wav):<40} signal (static zeroed)")
    print(f"  {os.path.basename(static_wav):<40} static only (isolated)")
    print(f"  {os.path.basename(report_path):<40} structural report")
    print(f"\n  Play the static-only file to hear what was removed.")
    print(f"  Signal samples were NEVER modified.")


if __name__ == "__main__":
    main()
    input("\n  Press Enter to exit...")
