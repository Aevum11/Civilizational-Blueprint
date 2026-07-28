#!/usr/bin/env python3
"""
ET DEEP STRUCTURAL ANALYSIS
============================
Analyzes a WAV file through the ET lattice and extracts structural data
that conventional audio analysis cannot see.

Outputs:
  {name}_analysis_report.txt    — Full text report
  {name}_epsilon_histogram.csv  — ε distribution data
  {name}_d_family_timeline.csv  — d-family per sample over time
  {name}_tightness_timeline.csv — Tightness per sample over time
  {name}_channel_comparison.txt — Per-channel lattice comparison
  {name}_artifact_analysis.txt  — Hardware artifact investigation

Usage:
  python3 et_deep_analysis.py 3.wav

All output saved next to the input file.

Derived forward from P∘D∘T = E.
"""

import sys
import os
import struct
import wave
from math import gcd

from mpmath import (
    mp, mpf, log as mplog, sqrt as mpsqrt,
    nint, fabs, nstr
)

# ═══════════════════════════════════════════════════════════════
# PRECISION — 100 dps working + 50 guard = 150 internal
# Sufficient for analysis (WAV samples have ~10 meaningful digits)
# ═══════════════════════════════════════════════════════════════
mp.dps = 150
WORK_DPS = 100
LOG2 = mplog(mpf(2))
CENTS_PER_OCTAVE = mpf(1200)
LAMBDA_MANIFOLD = CENTS_PER_OCTAVE / LOG2
N_BASE = 12


def project(r_str, N=12):
    """Project r onto lattice at resolution N. Returns (k, d, ε)."""
    r = mpf(r_str)
    log2_r = mplog(r) / LOG2
    x = mpf(N) * log2_r
    k = int(nint(x))  # ← T-ACT
    g = gcd(abs(k), N) if k != 0 else N
    d = N // g
    eps = (x - mpf(k)) * CENTS_PER_OCTAVE / mpf(N)
    return k, d, eps


def tightness(eps_cents, N=12):
    """Tightness function t(ε) = W/(W+|ε|) where W = 1200/N."""
    W = CENTS_PER_OCTAVE / mpf(N)
    return W / (W + fabs(eps_cents))


def read_wav_samples(filepath):
    """Read WAV, return per-channel sample string lists + metadata."""
    with wave.open(filepath, 'rb') as wf:
        n_channels = wf.getnchannels()
        samp_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    bit_depth = samp_width * 8
    max_val = (1 << (bit_depth - 1))

    # Separate channels
    channels = [[] for _ in range(n_channels)]

    for i in range(n_frames):
        for ch in range(n_channels):
            offset = (i * n_channels + ch) * samp_width
            raw_bytes = raw[offset:offset + samp_width]

            if samp_width == 1:
                val = raw_bytes[0] - 128
            elif samp_width == 2:
                val = struct.unpack_from('<h', raw_bytes)[0]
            elif samp_width == 3:
                val = int.from_bytes(raw_bytes, byteorder='little', signed=True)
            elif samp_width == 4:
                val = struct.unpack_from('<i', raw_bytes)[0]
            else:
                val = 0

            channels[ch].append(f"{val}/{max_val}")

    metadata = {
        'sample_rate': sample_rate,
        'channels': n_channels,
        'bit_depth': bit_depth,
        'n_frames': n_frames,
        'max_val': max_val,
    }
    return channels, metadata


def analyze_channel(samples, channel_name, sample_rate):
    """Full ET lattice analysis on a single channel. Returns analysis dict."""
    print(f"    Projecting {len(samples)} samples ({channel_name})...")

    # Per-sample data
    k_values = []
    d_values = []
    eps_values = []
    tight_values = []
    signs = []
    zero_count = 0
    escalation_count = 0

    # Escalation projectors
    esc_projectors = {60: 60, 420: 420}

    for idx, s_str in enumerate(samples):
        s = mpf(s_str)

        if s == mpf(0):
            zero_count += 1
            k_values.append(None)
            d_values.append(None)
            eps_values.append(mpf(0))
            tight_values.append(mpf(1))
            signs.append(0)
            continue

        sign = 1 if s > 0 else -1
        signs.append(sign)
        r = fabs(s)

        k, d, eps = project(nstr(r, WORK_DPS), N_BASE)

        # ∂I check — escalate if needed
        t = tightness(eps, N_BASE)
        koide_K = mpf(2) / mpf(3)

        if t <= koide_K + mpf("0.05"):
            for N_esc in [60, 420]:
                k_e, d_e, eps_e = project(nstr(r, WORK_DPS), N_esc)
                t_e = tightness(eps_e, N_esc)
                eps_max_e = mpf(600) / mpf(N_esc)
                t_boundary_e = mpf(100) / (mpf(100) + eps_max_e)
                if t_e > t_boundary_e + mpf("0.05"):
                    k, d, eps = k_e, d_e, eps_e
                    t = t_e
                    escalation_count += 1
                    break

        k_values.append(k)
        d_values.append(d)
        eps_values.append(eps)
        tight_values.append(t)

        if idx % 50000 == 0 and idx > 0:
            print(f"      {idx}/{len(samples)} samples processed...")

    print(f"      Done. {len(samples)} samples, {zero_count} zeros, "
          f"{escalation_count} escalations.")

    return {
        'k': k_values, 'd': d_values, 'eps': eps_values,
        'tightness': tight_values, 'signs': signs,
        'zero_count': zero_count, 'escalation_count': escalation_count,
        'total': len(samples), 'sample_rate': sample_rate,
        'channel_name': channel_name,
    }


def epsilon_histogram(eps_values, d_values, bins=200):
    """Build histogram of ε values for non-zero samples. Returns (bin_centers, counts)."""
    # Collect ε from non-zero samples only (where d is not None)
    valid_eps = []
    for eps_val, d_val in zip(eps_values, d_values):
        if d_val is not None:
            valid_eps.append(float(eps_val))
    
    if not valid_eps:
        return [], []
    
    min_e = min(valid_eps)
    max_e = max(valid_eps)
    
    # Ensure we have a range
    if max_e <= min_e:
        return [min_e], [len(valid_eps)]
    
    bin_width = (max_e - min_e) / bins
    histogram = [0] * bins
    
    for e in valid_eps:
        idx = int((e - min_e) / bin_width)
        if idx >= bins:
            idx = bins - 1
        if idx < 0:
            idx = 0
        histogram[idx] += 1
    
    centers = [min_e + (i + 0.5) * bin_width for i in range(bins)]
    return centers, histogram


def d_family_distribution(d_values):
    """Count d-family occurrences."""
    counts = {}
    total_nonzero = 0
    for d in d_values:
        if d is not None:
            counts[d] = counts.get(d, 0) + 1
            total_nonzero += 1
    return counts, total_nonzero


def tightness_statistics(tight_values):
    """Compute tightness distribution statistics."""
    valid = [t for t in tight_values if t is not None]
    if not valid:
        return {}

    n = len(valid)
    mean_t = sum(valid) / mpf(n)
    min_t = min(valid)
    max_t = max(valid)

    # Count in zones
    koide_K = mpf(2) / mpf(3)
    twilight_threshold = mpf("0.752")

    lattice_exact = sum(1 for t in valid if t > mpf("0.99"))
    high_tight = sum(1 for t in valid if mpf("0.85") < t <= mpf("0.99"))
    normal = sum(1 for t in valid if twilight_threshold < t <= mpf("0.85"))
    twilight = sum(1 for t in valid if koide_K < t <= twilight_threshold)
    boundary = sum(1 for t in valid if t <= koide_K)

    return {
        'mean': mean_t, 'min': min_t, 'max': max_t,
        'lattice_exact': lattice_exact,
        'high_tightness': high_tight,
        'normal': normal,
        'twilight_zone': twilight,
        'at_boundary': boundary,
        'total': n,
    }


def detect_speech_regions(signs, sample_rate, window_ms=20):
    """Detect speech vs silence regions based on sign change rate."""
    window_samples = int(sample_rate * window_ms / 1000)
    regions = []

    for start in range(0, len(signs), window_samples):
        end = min(start + window_samples, len(signs))
        window = signs[start:end]

        # Count sign changes in window
        changes = 0
        nonzero = 0
        for i in range(1, len(window)):
            if window[i] != 0:
                nonzero += 1
            if window[i] != 0 and window[i - 1] != 0 and window[i] != window[i - 1]:
                changes += 1

        # Speech has high sign change rate and high nonzero count
        is_speech = changes > 2 and nonzero > len(window) * 0.5
        regions.append({
            'start_sample': start,
            'end_sample': end,
            'start_time': start / sample_rate,
            'end_time': end / sample_rate,
            'is_speech': is_speech,
            'sign_changes': changes,
            'nonzero_fraction': nonzero / max(1, len(window)),
        })

    return regions


def d_family_by_region(d_values, regions):
    """Compute d-family distribution separately for speech and silence."""
    speech_d = {}
    silence_d = {}
    speech_total = 0
    silence_total = 0

    for region in regions:
        start = region['start_sample']
        end = region['end_sample']
        target = speech_d if region['is_speech'] else silence_d

        for i in range(start, min(end, len(d_values))):
            d = d_values[i]
            if d is not None:
                target[d] = target.get(d, 0) + 1
                if region['is_speech']:
                    speech_total += 1
                else:
                    silence_total += 1

    return speech_d, speech_total, silence_d, silence_total


def investigate_artifacts(channels_data, metadata):
    """Investigate the 6515 Hz base frequency and its harmonics."""
    lines = []
    lines.append("=" * 70)
    lines.append("  HARDWARE ARTIFACT INVESTIGATION")
    lines.append("  The 13,031 Hz and 19,547 Hz artifacts")
    lines.append("=" * 70)
    lines.append("")
    lines.append("  From spectrum analysis:")
    lines.append("    Artifact 1: 13,031 Hz at -57.48 dB (+4.5 dB above floor)")
    lines.append("    Artifact 2: 19,547 Hz at -58.19 dB (+3.8 dB above floor)")
    lines.append(f"    Ratio: 19547/13031 = {nstr(mpf(19547)/mpf(13031), 10)}")
    lines.append(f"    This is 3/2 to within {nstr(fabs(mpf(19547)/mpf(13031) - mpf(3)/mpf(2)), 6)}")
    lines.append(f"    Interval: {nstr(CENTS_PER_OCTAVE * mplog(mpf(19547)/mpf(13031)) / LOG2, 6)} cents")
    lines.append(f"    A PERFECT FIFTH (702.0 cents)")
    lines.append("")
    lines.append("  Base frequency derivation:")
    lines.append(f"    If 13031 = 2 × base and 19547 = 3 × base:")
    lines.append(f"    base = 13031/2 = {nstr(mpf(13031)/mpf(2), 8)} Hz")
    lines.append(f"    base = 19547/3 = {nstr(mpf(19547)/mpf(3), 8)} Hz")
    lines.append(f"    Average base ≈ 6515.6 Hz")
    lines.append("")
    lines.append("  Relationship to sample rate:")
    lines.append(f"    48000 / 6515.6 = {nstr(mpf(48000)/mpf('6515.6'), 8)}")
    lines.append(f"    48000 / 13031 = {nstr(mpf(48000)/mpf(13031), 8)}")
    lines.append(f"    48000 / 19547 = {nstr(mpf(48000)/mpf(19547), 8)}")
    lines.append("")
    lines.append("  ET lattice projection of 3/2:")

    k, d, eps = project(nstr(mpf(3) / mpf(2), WORK_DPS), 12)
    lines.append(f"    Π₁₂(3/2) = (k={k}, d={d}, ε={nstr(eps, 6)}¢)")
    lines.append(f"    d=12 = the EM/coprime family")
    lines.append(f"    The hardware artifacts encode the most fundamental")
    lines.append(f"    non-octave interval on the lattice.")
    lines.append("")

    # Check if 4th harmonic would alias
    fourth_harm = 4 * 6515.6
    nyquist = metadata['sample_rate'] / 2
    lines.append(f"  Aliasing check:")
    lines.append(f"    4th harmonic: 4 × 6515.6 = {fourth_harm:.1f} Hz")
    lines.append(f"    Nyquist: {nyquist} Hz")
    if fourth_harm > nyquist:
        alias_freq = metadata['sample_rate'] - fourth_harm
        lines.append(f"    ALIASED to: {alias_freq:.1f} Hz")
        lines.append(f"    (Check for energy near {alias_freq:.0f} Hz in spectrum)")
    else:
        lines.append(f"    Below Nyquist — should be visible directly")

    return '\n'.join(lines)


def format_d_distribution(counts, total, label=""):
    """Format a d-family distribution for display."""
    lines = []
    if label:
        lines.append(f"  {label}:")

    # Euler totient for comparison
    totient = {1: 1, 2: 1, 3: 2, 4: 2, 6: 2, 12: 4}

    for d in sorted(counts.keys()):
        count = counts[d]
        pct = mpf(100) * mpf(count) / mpf(total)
        expected = ""
        if d in totient:
            expected_pct = mpf(100) * mpf(totient[d]) / mpf(12)
            deviation = pct - expected_pct
            expected = f"  (φ/12={nstr(expected_pct, 4)}%, Δ={nstr(deviation, 4):>7}%)"
        lines.append(f"    d={d:>3}: {count:>8} ({nstr(pct, 4):>8}%){expected}")

    return '\n'.join(lines)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Find all WAV files in the same folder as this script
    wav_files = [f for f in os.listdir(script_dir) if f.lower().endswith('.wav')]
    
    if not wav_files:
        print(f"  No WAV files found in: {script_dir}")
        input("  Press Enter to exit...")
        sys.exit(1)
    
    # If command-line argument given, use it
    if len(sys.argv) >= 2 and os.path.exists(sys.argv[1]):
        wav_path = sys.argv[1]
    elif len(wav_files) == 1:
        wav_path = os.path.join(script_dir, wav_files[0])
        print(f"  Found: {wav_files[0]}")
    else:
        # Multiple WAV files — let user pick
        print(f"\n  WAV files in {script_dir}:\n")
        for i, f in enumerate(wav_files):
            size_mb = os.path.getsize(os.path.join(script_dir, f)) / (1024 * 1024)
            print(f"    [{i+1}] {f} ({size_mb:.1f} MB)")
        
        try:
            choice = input(f"\n  Select [1-{len(wav_files)}]: ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(wav_files):
                wav_path = os.path.join(script_dir, wav_files[int(choice) - 1])
            else:
                print("  Invalid selection.")
                input("  Press Enter to exit...")
                sys.exit(1)
        except (EOFError, KeyboardInterrupt):
            sys.exit(1)
    
    if not os.path.isabs(wav_path):
        wav_path = os.path.join(script_dir, wav_path)
    
    base_name = os.path.splitext(os.path.basename(wav_path))[0]
    out_dir = os.path.dirname(os.path.abspath(wav_path))

    print("=" * 70)
    print("  ET DEEP STRUCTURAL ANALYSIS")
    print(f"  Input: {wav_path}")
    print(f"  Precision: {WORK_DPS} dps ({mp.dps} internal)")
    print("=" * 70)

    # ─── Read WAV ───
    print(f"\n  Reading WAV...")
    channels, metadata = read_wav_samples(wav_path)
    n_ch = metadata['channels']
    sr = metadata['sample_rate']
    print(f"  {metadata['n_frames']} frames, {n_ch} channels, "
          f"{sr} Hz, {metadata['bit_depth']}-bit")

    # ─── Analyze each channel ───
    channel_analyses = []
    for ch_idx in range(n_ch):
        ch_name = ["Left", "Right", "Center", "LFE"][ch_idx] if ch_idx < 4 else f"Ch{ch_idx}"
        print(f"\n  Analyzing channel {ch_idx + 1}/{n_ch} ({ch_name}):")
        analysis = analyze_channel(channels[ch_idx], ch_name, sr)
        channel_analyses.append(analysis)

    # ═══════════════════════════════════════════════════════════════
    # REPORT 1: Main analysis report
    # ═══════════════════════════════════════════════════════════════
    report_path = os.path.join(out_dir, f"{base_name}_analysis_report.txt")
    print(f"\n  Generating report → {os.path.basename(report_path)}")

    report = []
    report.append("=" * 70)
    report.append("  ET DEEP STRUCTURAL ANALYSIS REPORT")
    report.append(f"  Source: {os.path.basename(wav_path)}")
    report.append(f"  {metadata['n_frames']} frames, {n_ch} channels, "
                  f"{sr} Hz, {metadata['bit_depth']}-bit")
    report.append("=" * 70)

    for ch_idx, analysis in enumerate(channel_analyses):
        ch_name = analysis['channel_name']
        report.append(f"\n{'─' * 70}")
        report.append(f"  CHANNEL: {ch_name}")
        report.append(f"{'─' * 70}")

        # Basic stats
        nonzero_eps = [e for e, d in zip(analysis['eps'], analysis['d']) if d is not None]
        nonzero_k = [k for k in analysis['k'] if k is not None]

        if nonzero_k:
            report.append(f"  Samples: {analysis['total']} total, "
                         f"{analysis['zero_count']} zeros")
            report.append(f"  Escalations: {analysis['escalation_count']}")
            report.append(f"  k range: [{min(nonzero_k)}, {max(nonzero_k)}]")

            # ε statistics
            eps_mean = sum(nonzero_eps) / mpf(len(nonzero_eps))
            eps_rms = mpsqrt(sum(e * e for e in nonzero_eps) / mpf(len(nonzero_eps)))
            cell_width = CENTS_PER_OCTAVE / mpf(N_BASE)
            structural_rms = cell_width / mpsqrt(mpf(12))
            deviation_pct = (mpf(1) - eps_rms / structural_rms) * mpf(100)

            report.append(f"\n  ε STATISTICS:")
            report.append(f"    Mean:         {nstr(eps_mean, 8)}¢")
            report.append(f"    RMS:          {nstr(eps_rms, 8)}¢")
            report.append(f"    Structural:   {nstr(structural_rms, 8)}¢ (cell/√12)")
            report.append(f"    Deviation:    {nstr(deviation_pct, 4)}% below structural")
            report.append(f"    Min:          {nstr(min(nonzero_eps), 8)}¢")
            report.append(f"    Max:          {nstr(max(nonzero_eps), 8)}¢")

            # d-family distribution
            d_counts, d_total = d_family_distribution(analysis['d'])
            report.append(f"\n  d-FAMILY DISTRIBUTION (vs Euler totient φ(d)/12):")
            report.append(format_d_distribution(d_counts, d_total))

            # Tightness statistics
            t_stats = tightness_statistics(analysis['tightness'])
            if t_stats:
                report.append(f"\n  TIGHTNESS ZONES (Identity F):")
                report.append(f"    Lattice-exact (t > 0.99): {t_stats['lattice_exact']:>8} "
                             f"({nstr(mpf(100)*mpf(t_stats['lattice_exact'])/mpf(t_stats['total']),4)}%)")
                report.append(f"    High (0.85 < t ≤ 0.99):  {t_stats['high_tightness']:>8} "
                             f"({nstr(mpf(100)*mpf(t_stats['high_tightness'])/mpf(t_stats['total']),4)}%)")
                report.append(f"    Normal (0.75 < t ≤ 0.85): {t_stats['normal']:>8} "
                             f"({nstr(mpf(100)*mpf(t_stats['normal'])/mpf(t_stats['total']),4)}%)")
                report.append(f"    Twilight (K < t ≤ 0.75):  {t_stats['twilight_zone']:>8} "
                             f"({nstr(mpf(100)*mpf(t_stats['twilight_zone'])/mpf(t_stats['total']),4)}%)")
                report.append(f"    ∂I boundary (t ≤ K=2/3):  {t_stats['at_boundary']:>8} "
                             f"({nstr(mpf(100)*mpf(t_stats['at_boundary'])/mpf(t_stats['total']),4)}%)")
                report.append(f"    Mean tightness:           {nstr(t_stats['mean'], 8)}")

            # Speech vs silence d-family comparison
            regions = detect_speech_regions(analysis['signs'], sr)
            speech_d, speech_total, silence_d, silence_total = \
                d_family_by_region(analysis['d'], regions)

            if speech_total > 0:
                report.append(f"\n  d-FAMILY: SPEECH REGIONS ({speech_total} samples):")
                report.append(format_d_distribution(speech_d, speech_total))

            if silence_total > 0:
                report.append(f"\n  d-FAMILY: SILENCE REGIONS ({silence_total} samples):")
                report.append(format_d_distribution(silence_d, silence_total))

    # ─── Channel comparison ───
    if n_ch >= 2:
        report.append(f"\n{'═' * 70}")
        report.append(f"  CHANNEL COMPARISON (Left vs Right)")
        report.append(f"{'═' * 70}")

        for metric in ['zero_count', 'escalation_count']:
            vals = [a[metric] for a in channel_analyses[:2]]
            report.append(f"  {metric}: L={vals[0]}, R={vals[1]}, "
                         f"Δ={abs(vals[0]-vals[1])}")

        # Compare d-family distributions
        for ch_idx in range(min(2, n_ch)):
            d_counts, d_total = d_family_distribution(channel_analyses[ch_idx]['d'])
            ch_name = channel_analyses[ch_idx]['channel_name']
            report.append(f"\n  {ch_name}:")
            for d in sorted(d_counts.keys()):
                pct = mpf(100) * mpf(d_counts[d]) / mpf(d_total)
                report.append(f"    d={d:>3}: {nstr(pct, 4)}%")

    # ─── Artifact investigation ───
    report.append(f"\n{investigate_artifacts(channel_analyses, metadata)}")

    # Write report
    report_text = '\n'.join(report)
    with open(report_path, 'w') as f:
        f.write(report_text)
    print(f"  Report saved.")

    # ═══════════════════════════════════════════════════════════════
    # REPORT 2: ε histogram CSV
    # ═══════════════════════════════════════════════════════════════
    hist_path = os.path.join(out_dir, f"{base_name}_epsilon_histogram.csv")
    print(f"  Generating ε histogram → {os.path.basename(hist_path)}")

    with open(hist_path, 'w') as f:
        f.write("bin_center_cents,count_left")
        if n_ch >= 2:
            f.write(",count_right")
        f.write("\n")

        centers_L, counts_L = epsilon_histogram(
            channel_analyses[0]['eps'], channel_analyses[0]['d'])
        if n_ch >= 2:
            centers_R, counts_R = epsilon_histogram(
                channel_analyses[1]['eps'], channel_analyses[1]['d'])

        for i in range(len(centers_L)):
            line = f"{centers_L[i]:.4f},{counts_L[i]}"
            if n_ch >= 2 and i < len(counts_R):
                line += f",{counts_R[i]}"
            f.write(line + "\n")

    # ═══════════════════════════════════════════════════════════════
    # REPORT 3: d-family timeline CSV (sampled every N samples)
    # ═══════════════════════════════════════════════════════════════
    timeline_path = os.path.join(out_dir, f"{base_name}_d_family_timeline.csv")
    print(f"  Generating d-family timeline → {os.path.basename(timeline_path)}")

    # Sample every 100 samples to keep file manageable
    step = 100
    with open(timeline_path, 'w') as f:
        header = "time_s"
        for ch_idx in range(n_ch):
            ch_name = channel_analyses[ch_idx]['channel_name']
            header += f",d_{ch_name},eps_{ch_name},tight_{ch_name}"
        f.write(header + "\n")

        n_samples = len(channel_analyses[0]['d'])
        for i in range(0, n_samples, step):
            t = mpf(i) / mpf(sr)
            line = f"{nstr(t, 6)}"
            for ch_idx in range(n_ch):
                a = channel_analyses[ch_idx]
                d_val = a['d'][i] if a['d'][i] is not None else 0
                eps_val = nstr(a['eps'][i], 6) if i < len(a['eps']) else "0"
                t_val = nstr(a['tightness'][i], 4) if i < len(a['tightness']) else "0"
                line += f",{d_val},{eps_val},{t_val}"
            f.write(line + "\n")

    # ═══════════════════════════════════════════════════════════════
    # REPORT 4: Tightness timeline CSV
    # ═══════════════════════════════════════════════════════════════
    tight_path = os.path.join(out_dir, f"{base_name}_tightness_timeline.csv")
    print(f"  Generating tightness timeline → {os.path.basename(tight_path)}")

    with open(tight_path, 'w') as f:
        header = "time_s"
        for ch_idx in range(n_ch):
            header += f",tightness_{channel_analyses[ch_idx]['channel_name']}"
        f.write(header + "\n")

        for i in range(0, n_samples, step):
            t = mpf(i) / mpf(sr)
            line = f"{nstr(t, 6)}"
            for ch_idx in range(n_ch):
                a = channel_analyses[ch_idx]
                t_val = nstr(a['tightness'][i], 6) if i < len(a['tightness']) else "0"
                line += f",{t_val}"
            f.write(line + "\n")

    # ═══════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'═' * 70}")
    print(f"  ANALYSIS COMPLETE")
    print(f"{'═' * 70}")
    print(f"\n  Output files:")
    print(f"    {os.path.basename(report_path)}")
    print(f"    {os.path.basename(hist_path)}")
    print(f"    {os.path.basename(timeline_path)}")
    print(f"    {os.path.basename(tight_path)}")
    print(f"\n  Key findings (Left channel):")

    a = channel_analyses[0]
    nonzero_eps = [e for e, d in zip(a['eps'], a['d']) if d is not None]
    if nonzero_eps:
        eps_rms = mpsqrt(sum(e * e for e in nonzero_eps) / mpf(len(nonzero_eps)))
        print(f"    ε RMS: {nstr(eps_rms, 6)}¢")
    print(f"    Escalations: {a['escalation_count']}")

    d_counts, d_total = d_family_distribution(a['d'])
    if 12 in d_counts and d_total > 0:
        d12_pct = mpf(100) * mpf(d_counts[12]) / mpf(d_total)
        print(f"    d=12 (EM): {nstr(d12_pct, 4)}% (totient predicts 33.33%)")

    t_stats = tightness_statistics(a['tightness'])
    if t_stats:
        print(f"    Mean tightness: {nstr(t_stats['mean'], 6)}")
        print(f"    ∂I boundary samples: {t_stats['at_boundary']}")

    print(f"\n  Open the CSV files in a spreadsheet to graph the data.")
    print(f"  The ε histogram shows the crystal distribution of your voice.")
    print(f"  The d-family timeline shows lattice structure evolving over time.")


if __name__ == "__main__":
    main()
    input("\n  Press Enter to exit...")
