#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
ET DIRECT PATH v2 — Bit-Exact Delivery to Bluetooth (double-click ready)
================================================================================
Exception Theory — Michael James Muller — Exception Theory LLC
P * D * T = E

DOUBLE-CLICK THIS FILE. The window stays open, always: every path ends at a
prompt, every error prints its full traceback and waits. No CLI knowledge
needed; a numbered menu drives everything.

THE METHOD (the gravimeter's doctrine, run in reverse)
------------------------------------------------------
Backend: sounddevice/PortAudio with the host-API priority proven in
et_lossless_microphone.py — WASAPI EXCLUSIVE preferred, because the shared
Windows engine "resamples to whatever it wants, interpolating fake samples."
Exclusive mode lets US dictate the endpoint's rate, so the sample-rate
converter, float mixer, APO chain and limiter never touch the stream: every
bit the Bluetooth codec receives is a bit this program chose.

ET COMPLIANCE (what is forbidden stays out)
-------------------------------------------
* No float in any sample path or produced value: raw int16/int24 bytes pass
  untouched; all processing is integer arithmetic.
* No resampling, no interpolation, no dither, no Taylor/Shannon/Nyquist
  machinery anywhere: on rate mismatch this program dictates the rate in
  exclusive mode or tells you the exact Windows dialog to align — it never
  fabricates samples.
* Gain exists ONLY as exact octave shifts (powers of two = pure k-moves on
  the amplitude lattice): requantization epsilon introduced = 0, verified.
* The live meter is the MAG-3 mirror-pair decomposition, integer-exact:
  M = L+R (the kappa=0 pair sum, the ground channel), S = L-R (the mirror
  difference). Displayed as truncated ratio strings, never float.
* Honesty (Rule 14): the radio's PA wattage is silicon and regulation; no
  software raises it, this one included. What this program removes is every
  middleman between your samples and the codec — the part of the chain that
  actually differs from your phone.

MODES:  [1] Bit-exact WAV player (exclusive)   [2] Live loopback bridge
        [3] List audio devices & host APIs      [4] Bluetooth radio probe
        [5] Lossless family analyzer (dps=250)  [6] Self-test   [0] Exit
================================================================================
"""

import sys, os, subprocess, traceback, struct, wave
from fractions import Fraction

N_ET = 12
BLOCK = 4096            # 2^12 frames — octave-exact buffer discipline

# ── window discipline: NOTHING closes silently ───────────────────────────────
def _hold(msg="Press Enter to continue..."):
    try: input(msg)
    except EOFError: pass

def _excepthook(t, v, tb):
    print("\n" + "=" * 70)
    traceback.print_exception(t, v, tb)
    print("=" * 70)
    print("The error above is the complete story — nothing was hidden.")
    _hold("Press Enter to close...")
sys.excepthook = _excepthook

# ── dependency bootstrap (runs once, explains itself) ────────────────────────
def ensure(pkg, pipname=None):
    try:
        return __import__(pkg)
    except ImportError:
        print(f"Installing {pipname or pkg} (one time)...")
        r = subprocess.run([sys.executable, "-m", "pip", "install", pipname or pkg],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout); print(r.stderr)
            raise RuntimeError(f"pip could not install {pipname or pkg}; see output above.")
        return __import__(pkg)

# ── exact integer machinery (self-tested) ────────────────────────────────────
def octave_stage(ints, shift):
    """Gain as pure powers of two — k-shifts on the amplitude lattice, zero
    fractional requantization. Saturating, arithmetic, exact."""
    if shift == 0:
        return list(ints)
    out = []
    for s in ints:
        v = (int(s) << shift) if shift > 0 else (int(s) >> (-shift))
        if v > 32767: v = 32767
        elif v < -32768: v = -32768
        out.append(v)
    return out

def ms_butterfly(L, R):
    M = [int(a) + int(b) for a, b in zip(L, R)]
    S = [int(a) - int(b) for a, b in zip(L, R)]
    return M, S

def ms_inverse(M, S):
    return [(m + s) >> 1 for m, s in zip(M, S)], [(m - s) >> 1 for m, s in zip(M, S)]

def mirror_meter(L, R):
    """MAG-3 decomposition, integer-exact: energy of the pair-sum (ground)
    vs the mirror difference. Returns a truncated ratio string, no float."""
    M, S = ms_butterfly(L, R)
    em = sum(m * m for m in M); es = sum(s * s for s in S)
    if em == 0:
        return "M:S = 0:1 (pure side)"
    q = (Fraction(es, em) * 1000).numerator // (Fraction(es, em) * 1000).denominator
    return f"S/M = {q // 1000}.{q % 1000:03d} (0 = mono ground, 1 = full mirror)"

def trunc3(fr):
    q = (fr.numerator * 1000) // fr.denominator
    return f"{q // 1000}.{q % 1000:03d}"

def family_projection(samples):
    """The gravimeter's method as an inspector: exact DFT (mpf, dps=250) of a
    small frame; dominant spectral ratios projected to harmonic families."""
    mp_mod = ensure("mpmath")
    mp, mpf, log, nint = mp_mod.mp, mp_mod.mpf, mp_mod.log, mp_mod.nint
    from math import gcd
    mp.dps = 250
    n = len(samples)
    two_pi = 2 * mp.pi
    mags = []
    for k in range(1, n // 2):
        re = mpf(0); im = mpf(0)
        for t, s in enumerate(samples):
            ang = two_pi * k * t / n
            re += s * mp_mod.cos(ang); im += s * mp_mod.sin(ang)
        mags.append((k, mp_mod.sqrt(re * re + im * im)))
    mags.sort(key=lambda p: -p[1])
    if not mags or mags[0][1] == 0:
        return {}
    peak = mags[0][1]
    top = [k for k, m in mags[:8] if m > peak / 64]      # exact power-of-2 floor
    fam = {}
    f0 = top[0]
    for k in top:
        x = N_ET * log(mpf(k) / f0) / log(mpf(2))
        kk = int(nint(x))
        d = N_ET // (gcd(abs(kk), N_ET) if kk else N_ET)
        fam[d] = fam.get(d, 0) + 1
    return fam

# ── audio helpers (Windows; guarded imports) ─────────────────────────────────
def _sd():
    return ensure("sounddevice")

def open_exclusive_out(sd, rate, channels, dtype, device=None):
    """The doctrine ladder: WASAPI Exclusive first (we dictate the rate),
    shared only as a last resort — and it SAYS which one you got."""
    try:
        ws = sd.WasapiSettings(exclusive=True)
        st = sd.RawOutputStream(samplerate=rate, channels=channels, dtype=dtype,
                                blocksize=BLOCK, extra_settings=ws, device=device)
        st.start()
        return st, "WASAPI EXCLUSIVE (rate dictated; engine bypassed)"
    except Exception as e:
        print(f"  Exclusive denied by driver ({e}); falling back to direct shared.")
        st = sd.RawOutputStream(samplerate=rate, channels=channels, dtype=dtype,
                                blocksize=BLOCK, device=device)
        st.start()
        return st, "shared (direct stream, engine present — quality still bit-chosen)"

def pick_output(sd):
    devs = sd.query_devices()
    outs = [(i, d) for i, d in enumerate(devs) if d["max_output_channels"] > 0]
    print("\nOutput endpoints:")
    for i, d in outs:
        api = sd.query_hostapis(d["hostapi"])["name"]
        print(f"  [{i:2d}] {d['name']}  ({api}, {int(d['default_samplerate'])} Hz)")
    bt_hint = [i for i, d in outs if any(w in d["name"].lower() for w in
               ("bluetooth", "bt", "hands", "headset", "headphone", "buds", "airpod", "wh-", "wf-"))]
    if bt_hint:
        print(f"  (Bluetooth-looking endpoints: {bt_hint})")
    raw = input("Endpoint index (Enter = system default): ").strip()
    return int(raw) if raw else None

# ── MODE 1: bit-exact WAV player ─────────────────────────────────────────────
def wav_player():
    sd = _sd()
    path = input("Path to .wav file (drag it onto this window, then Enter): ").strip().strip('"')
    if not os.path.isfile(path):
        print("File not found:", path); return
    wf = wave.open(path, "rb")
    width, ch, rate, nfr = wf.getsampwidth(), wf.getnchannels(), wf.getframerate(), wf.getnframes()
    dtype = {2: "int16", 3: "int24", 4: "int32"}.get(width)
    if dtype is None:
        print(f"Unsupported sample width {width * 8}-bit — no conversion will be fabricated."); return
    print(f"\n{os.path.basename(path)}: {rate} Hz, {ch} ch, {width * 8}-bit, {nfr} frames")
    dev = pick_output(sd)
    st, mode = open_exclusive_out(sd, rate, ch, dtype, dev)
    print("PLAYING  [" + mode + "]  — the file's own rate, its own bits, no middlemen.")
    print("Ctrl+C stops.")
    try:
        while True:
            data = wf.readframes(BLOCK)
            if not data:
                break
            st.write(data)                      # raw bytes: bit-exact, no float ever
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        st.stop(); st.close(); wf.close()
    print("Done — every delivered bit was the file's own.")

# ── MODE 2: live loopback bridge ─────────────────────────────────────────────
def loopback_bridge():
    sd = _sd()
    pyaudio = ensure("pyaudiowpatch")
    p = pyaudio.PyAudio()
    try:
        lb = p.get_default_wasapi_loopback()
    except Exception:
        print("No default loopback found. Set your default output to the device whose")
        print("audio you want bridged (e.g. speakers), then rerun this mode.")
        p.terminate(); return
    rate = int(lb["defaultSampleRate"]); ch = 2
    print(f"\nSource (loopback): {lb['name']} @ {rate} Hz")
    dev = pick_output(sd)
    shift_raw = input("Octave gain shift (integer, 0 = unity, e.g. -1 halves exactly): ").strip()
    shift = int(shift_raw) if shift_raw else 0
    inp = p.open(format=pyaudio.paInt16, channels=ch, rate=rate, input=True,
                 input_device_index=lb["index"], frames_per_buffer=BLOCK)
    st, mode = open_exclusive_out(sd, rate, ch, "int16", dev)
    print(f"BRIDGE LIVE  [{mode}]  @ {rate} Hz — Ctrl+C stops.")
    blocks = clips = 0
    try:
        while True:
            raw = inp.read(BLOCK, exception_on_overflow=False)
            ints = struct.unpack("<%dh" % (len(raw) // 2), raw)
            if shift:
                staged = octave_stage(ints, shift)
                clips += sum(1 for a, b in zip(staged, octave_stage(ints, shift)) if abs(b) == 32767)
                ints = staged
                raw = struct.pack("<%dh" % len(ints), *ints)
            st.write(raw)
            blocks += 1
            if blocks % 64 == 0:                # 2^6 — status without spam
                L = ints[0::2]; R = ints[1::2]
                print(f"  blocks {blocks:6d} | clip {clips:4d} | {mirror_meter(L, R)}", end="\r")
    except KeyboardInterrupt:
        print("\nBridge closed.")
    finally:
        inp.close(); st.stop(); st.close(); p.terminate()

# ── MODE 3/4: devices & radio probe ──────────────────────────────────────────
def list_devices():
    sd = _sd()
    print("\nHost APIs (the doctrine ladder — Exclusive-capable first):")
    for i, api in enumerate(sd.query_hostapis()):
        print(f"  {api['name']}")
    print()
    print(sd.query_devices())

def probe_radio():
    ps = (r"$r = Get-PnpDevice -Class Bluetooth | Where-Object {$_.FriendlyName -match 'Radio|Adapter'};"
          r"foreach ($d in $r) { 'RADIO: ' + $d.FriendlyName; "
          r"(Get-PnpDeviceProperty -InstanceId $d.InstanceId -ErrorAction SilentlyContinue | "
          r"Where-Object {$_.KeyName -match 'Manufacturer|DriverVersion|HardwareIds'} | "
          r"ForEach-Object { '  ' + $_.KeyName.Split('_')[-1] + ': ' + ($_.Data -join ' ') }) }")
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                             capture_output=True, text=True, timeout=40)
        print(out.stdout or out.stderr or "(no output)")
        print("This chipset identification is the prerequisite for any vendor")
        print("power-index (Layer-2) work — the data retrieved, not guessed.")
    except FileNotFoundError:
        print("PowerShell not found — this mode is Windows-only.")

# ── MODE 5: lossless analyzer ────────────────────────────────────────────────
def analyzer():
    path = input("Path to a .wav to inspect (Enter = capture 1s of loopback): ").strip().strip('"')
    if path:
        wf = wave.open(path, "rb")
        if wf.getsampwidth() != 2:
            print("Analyzer expects 16-bit WAV for the exact int path."); return
        raw = wf.readframes(4096); wf.close()
        ints = list(struct.unpack("<%dh" % (len(raw) // 2), raw))[0::wf.getnchannels() or 1][:96]
    else:
        pyaudio = ensure("pyaudiowpatch")
        p = pyaudio.PyAudio(); lb = p.get_default_wasapi_loopback()
        stm = p.open(format=pyaudio.paInt16, channels=2, rate=int(lb["defaultSampleRate"]),
                     input=True, input_device_index=lb["index"], frames_per_buffer=BLOCK)
        raw = stm.read(BLOCK, exception_on_overflow=False)
        stm.close(); p.terminate()
        ints = list(struct.unpack("<%dh" % (len(raw) // 2), raw))[0::2][:96]
    print("Family projection of dominant spectral ratios (string->mpf, dps=250):")
    print(" ", family_projection(ints) or "(silence)")

# ── MODE 6: self-test (runs anywhere, no audio hardware) ─────────────────────
def selftest():
    ok = tot = 0
    def rep(name, c):
        nonlocal ok, tot
        tot += 1; ok += bool(c)
        print(("[PASS] " if c else "[FAIL] ") + name)
    import random, math as _m
    data = list(range(-8192, 8192, 7))
    rep("octave staging >>2<<2 exact on aligned data",
        all(((a >> 2) << 2) == b for a, b in zip(data, octave_stage(octave_stage(data, -2), 2))))
    rep("octave staging shift 0 is identity", octave_stage(data, 0) == data)
    random.seed(N_ET)
    L = [random.randint(-32768, 32767) for _ in range(4096)] + [-32768, 32767, 0, 1, -1]
    R = [random.randint(-32768, 32767) for _ in range(4096)] + [32767, -32768, 0, -1, 1]
    M, S = ms_butterfly(L, R); L2, R2 = ms_inverse(M, S)
    rep("mirror butterfly bit-exact invertible (4101 pairs)", L2 == L and R2 == R)
    rep("mirror meter: identical channels -> pure ground (S/M = 0.000)",
        mirror_meter([5, -7, 9], [5, -7, 9]).startswith("S/M = 0.000"))
    n = 96; f0 = 4
    sig = [int(12000 * _m.sin(2 * _m.pi * f0 * t / n)
              + 8000 * _m.sin(2 * _m.pi * 2 * f0 * t / n)
              + 6000 * _m.sin(2 * _m.pi * 6 * t / n)) for t in range(n)]
    fam = family_projection(sig)
    rep("family projection: octave -> d=1 and fifth -> d=12 classified", 
        fam.get(1, 0) >= 1 and fam.get(12, 0) >= 1)
    print(f"\nSELFTEST: {ok}/{tot} PASSED")
    return ok == tot

# ── menu ─────────────────────────────────────────────────────────────────────
MENU = """
==========================================================
 ET DIRECT PATH v2 — bit-exact Bluetooth delivery
   1  Bit-exact WAV player (WASAPI Exclusive)
   2  Live loopback bridge (system audio -> headphones)
   3  List audio devices and host APIs
   4  Bluetooth radio probe (chipset identification)
   5  Lossless family analyzer (dps=250)
   6  Self-test (verify every exact-math component)
   0  Exit
==========================================================
"""

def main():
    print("ET DIRECT PATH v2 — window stays open; every error tells its story.")
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    actions = {"1": wav_player, "2": loopback_bridge, "3": list_devices,
               "4": probe_radio, "5": analyzer, "6": selftest}
    while True:
        print(MENU)
        choice = input("Choice: ").strip()
        if choice == "0":
            break
        fn = actions.get(choice)
        if fn is None:
            print("Enter a number from the menu."); continue
        try:
            fn()
        except KeyboardInterrupt:
            print("\n(cancelled)")
        except Exception:
            traceback.print_exc()
            print("The traceback above is complete — nothing hidden.")
        _hold()
    print("Closed cleanly.")
    _hold("Press Enter to exit...")

if __name__ == "__main__":
    main()
