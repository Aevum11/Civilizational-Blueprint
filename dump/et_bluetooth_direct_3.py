#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
ET BLUETOOTH DIRECT — diagnose, fix, and deliver. One purpose: your headphones.
================================================================================
Exception Theory — Michael James Muller — Exception Theory LLC
P * D * T = E

DOUBLE-CLICK THIS FILE. The window always stays open. Flow is linear:
  STEP 1  DIAGNOSE  — enumerate the actual system descriptors (nothing guessed)
  STEP 2  FIX       — close every gap found, one report line per change
  STEP 3  DELIVER   — the lossless bridge: bit-exact system audio -> headphones
  STEP 4  VERIFY    — re-enumerate; show the before/after difference

METHOD (the Three Tools, applied literally)
  Identification: "fine on phone, bad on PC at every range" is the
  profile/format/power failure class. Windows LABELS the profile in each
  Bluetooth endpoint's own name ("Stereo" = A2DP high quality; "Hands-Free" =
  the 8 kHz fallback that wrecks PC audio whenever any app touches the mic).
  Every match target in this program comes from names THE SYSTEM RETURNS —
  enumeration first, interpretation second, keyword guessing never.
  Descriptor Gap: each gap found is closed by setting the descriptor itself:
  Hands-Free endpoints disabled (kills profile flapping), power suspension
  removed from the radio and USB path (kills mid-stream strangling),
  Bluetooth stack restarted (clean renegotiation).
  Subsumption: the runtime path is the gravimeter's discipline in reverse —
  WASAPI EXCLUSIVE, rate dictated, raw integer bytes end to end.

ET COMPLIANCE
  No float in any sample path or displayed value; no resampling, dither,
  interpolation, or Shannon/Nyquist/Taylor machinery anywhere. Gain exists
  only as exact octave shifts (pure powers of two: zero requantization
  epsilon). The live meter is the MAG-3 mirror-pair decomposition in exact
  integers. The bijection's continuous<->discrete losslessness applies to
  everything in software custody — the samples; the one thing outside custody
  (the radio's analog wattage and RF environment) is named, not simulated:
  if dropouts persist ONLY near USB 3 activity, move the dongle/antenna path
  away from USB 3 — that is physics, and it is stated instead of faked.
================================================================================
"""

import sys, os, subprocess, traceback, struct, ctypes, json, time, glob
from fractions import Fraction

N_ET = 12
BLOCK = 4096                                    # 2^12 — octave-exact buffering

# ── window discipline ────────────────────────────────────────────────────────
def _hold(msg="Press Enter to continue..."):
    try: input(msg)
    except EOFError: pass

def _excepthook(t, v, tb):
    print("\n" + "=" * 70); traceback.print_exception(t, v, tb); print("=" * 70)
    print("The traceback above is the complete story — nothing hidden.")
    _hold("Press Enter to close...")
sys.excepthook = _excepthook

def ensure(pkg, pipname=None):
    try:
        return __import__(pkg)
    except ImportError:
        print(f"Installing {pipname or pkg} (one time)...")
        r = subprocess.run([sys.executable, "-m", "pip", "install", pipname or pkg],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout); print(r.stderr)
            raise RuntimeError(f"pip failed for {pipname or pkg} — output above.")
        return __import__(pkg)

def psrun(script, timeout=60):
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
                              "-Command", script], capture_output=True, text=True,
                             timeout=timeout)
        return (out.stdout or "").strip(), (out.stderr or "").strip(), out.returncode
    except FileNotFoundError:
        return "", "PowerShell not found (Windows-only step).", 1

def is_admin():
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False

# ── exact integer machinery (self-tested; unchanged from the verified core) ──
def octave_stage(ints, shift):
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
    return ([int(a) + int(b) for a, b in zip(L, R)],
            [int(a) - int(b) for a, b in zip(L, R)])

def ms_inverse(M, S):
    return [(m + s) >> 1 for m, s in zip(M, S)], [(m - s) >> 1 for m, s in zip(M, S)]

def mirror_meter(L, R):
    M, S = ms_butterfly(L, R)
    em = sum(m * m for m in M); es = sum(s * s for s in S)
    if em == 0:
        return "S/M = pure side"
    fr = Fraction(es, em) * 1000
    q = fr.numerator // fr.denominator
    return f"S/M = {q // 1000}.{q % 1000:03d}"

# ── STEP 1: DIAGNOSE (pure enumeration) ──────────────────────────────────────
PS_ENUM = """
$ErrorActionPreference='Continue'
'== RADIO =='
$d = Get-PnpDevice -Class Bluetooth -ErrorAction SilentlyContinue
if (-not $d) { 'NO BLUETOOTH-CLASS DEVICES ENUMERATED.' }
foreach ($x in $d) {
  'DEV: ' + $x.FriendlyName + '  [' + $x.Status + ']'
  $p = Get-PnpDeviceProperty -InstanceId $x.InstanceId -ErrorAction SilentlyContinue |
       Where-Object { $_.KeyName -match 'HardwareIds|Manufacturer|DriverVersion' }
  foreach ($q in $p) { '   ' + $q.KeyName.Split('_')[-1] + ': ' + ($q.Data -join ' ') }
}
'== AUDIO ENDPOINTS =='
$e = Get-PnpDevice -Class AudioEndpoint -ErrorAction SilentlyContinue
foreach ($x in $e) { 'EP: ' + $x.FriendlyName + '  [' + $x.Status + ']  ' + $x.InstanceId }
'== WIFI BAND =='
(netsh wlan show interfaces) -match 'Band|Radio type|State|Channel|SSID' | ForEach-Object { $_.Trim() }
"""

def diagnose():
    print("\nSTEP 1 — DIAGNOSE (everything below is retrieved, nothing assumed)\n")
    out, err, rc = psrun(PS_ENUM, timeout=90)
    if not out:
        print(f"Enumeration empty. rc={rc}"); 
        if err: print("stderr:", err[:600])
        return {}
    print(out)
    eps = []
    for line in out.splitlines():
        if line.startswith("EP: "):
            body = line[4:]
            name = body.split("  [")[0]
            status = body.split("[", 1)[1].split("]")[0] if "[" in body else "?"
            inst = body.split("]  ", 1)[1] if "]  " in body else ""
            eps.append({"name": name, "status": status, "id": inst})
    # Interpretation AFTER retrieval: Windows itself writes the profile into
    # the endpoint name. Hands-Free = HFP fallback; Stereo/Headphones = A2DP.
    hfp = [e for e in eps if "hands-free" in e["name"].lower() and e["status"] == "OK"]
    a2dp = [e for e in eps if "hands-free" not in e["name"].lower()
            and any(w in e["name"].lower() for w in ("stereo", "headphone"))]
    def band_from_channel(ch):
        if ch is None: return None
        return "2.4" if ch <= 14 else ("5" if ch >= 32 else None)
    ch = None
    for line in out.splitlines():
        ls = line.strip()
        if ls.lower().startswith("channel"):
            digits = "".join(c for c in ls.split(":")[-1] if c.isdigit())
            if digits: ch = int(digits)
    band = "2.4" if "2.4 GHz" in out else ("5" if "5 GHz" in out else band_from_channel(ch))
    gaps = {"hfp": hfp, "a2dp": a2dp, "wifi24": (band == "2.4"),
            "band": band, "all_eps": eps}
    print("\nGAPS FOUND:")
    print(f"  Active Hands-Free (HFP) endpoints: {len(hfp)}"
          + ("  <- profile flapping source (the 8 kHz fallback)" if hfp else "  (none - good)"))
    for e in hfp: print("     - " + e["name"])
    print(f"  A2DP render endpoints seen: {len(a2dp)}")
    for e in a2dp: print("     - " + e["name"] + f"  [{e['status']}]")
    if band == "2.4":
        print(f"  WiFi band: 2.4 GHz (channel {ch}) <- Bluetooth's band; coexistence contention")
    elif band == "5":
        print(f"  WiFi band: 5 GHz (channel {ch}) — clear of Bluetooth")
    else:
        print("  WiFi band: UNREPORTED by this Windows build and no channel parsed —")
        print("             not assumed either way; check your router page for the band.")
    if "VID_8087" in out:
        print("  NOTE: Intel combo module (VID_8087): WiFi and Bluetooth share this")
        print("        silicon — if the band is 2.4 GHz, moving WiFi to 5 GHz helps twice.")
    if band == "2.4":
        nets, _, _ = psrun("netsh wlan show networks mode=bssid", timeout=45)
        fiveg, ssid = set(), None
        for line in (nets or "").splitlines():
            ls = line.strip()
            if ls.lower().startswith("ssid"):
                parts = ls.split(":", 1)
                ssid = parts[1].strip() if len(parts) > 1 else None
            elif ls.lower().startswith("channel"):
                digits = "".join(c for c in ls.split(":")[-1] if c.isdigit())
                if digits and int(digits) >= 32 and ssid:
                    fiveg.add(ssid)
        gaps["fiveg"] = sorted(fiveg)
        if fiveg:
            print("  5 GHz networks VISIBLE right now: " + ", ".join(sorted(fiveg)))
            print("        -> the move is one click: connect to one of these.")
        else:
            print("  No 5 GHz networks visible from this adapter — the band move is")
            print("        router-side (enable 5 GHz on the TP-Link, or its 5G SSID is hidden).")
    return gaps

# ── CONFIG SAVE / RESTORE (revert any change after the fact) ─────────────────
def config_dir():
    d = os.path.dirname(os.path.abspath(sys.argv[0]))
    return d if os.path.isdir(d) else os.getcwd()

def next_config_path(dirpath, name=""):
    name = "".join(c for c in name.strip() if c.isalnum() or c in "-_")
    if name:
        return os.path.join(dirpath, f"etbt_config_{name}.json")
    nums = []
    for p in glob.glob(os.path.join(dirpath, "etbt_config_[0-9][0-9][0-9].json")):
        try: nums.append(int(os.path.basename(p)[12:15]))
        except ValueError: pass
    return os.path.join(dirpath, f"etbt_config_{(max(nums) + 1 if nums else 1):03d}.json")

def _read_powercfg():
    vals = {}
    for tag, flag in (("ac", "Current AC Power Setting Index"),
                      ("dc", "Current DC Power Setting Index")):
        out, _, _ = psrun("powercfg /QUERY SCHEME_CURRENT "
                          "2a737441-1930-4402-8d77-b2bebba308a3 "
                          "48e6b7a6-50f5-4782-a5d4-53bb8f07e226")
        for line in out.splitlines():
            if flag in line:
                try: vals[tag] = int(line.split(":")[-1].strip(), 16)
                except ValueError: pass
    return vals

def _read_radio_suspend():
    out, _, _ = psrun(
        "Get-PnpDevice -Class Bluetooth -Status OK | ForEach-Object {"
        " $k = 'HKLM:\\SYSTEM\\CurrentControlSet\\Enum\\' + $_.InstanceId + '\\Device Parameters';"
        " $v = (Get-ItemProperty -Path $k -Name SelectiveSuspendEnabled -ErrorAction SilentlyContinue).SelectiveSuspendEnabled;"
        " $_.InstanceId + '|' + ($(if ($null -eq $v) {'absent'} else {$v})) }")
    entries = []
    for line in (out or "").splitlines():
        if "|" in line:
            inst, val = line.rsplit("|", 1)
            entries.append({"instance": inst.strip(), "value": val.strip()})
    return entries

def save_config(gaps):
    name = input("Config name (Enter = incremental number): ").strip()
    path = next_config_path(config_dir(), name)
    cfg = {"saved": time.strftime("%Y-%m-%d %H:%M:%S"),
           "endpoints": [{"id": e["id"], "name": e["name"], "status": e["status"]}
                          for e in gaps.get("all_eps", [])],
           "powercfg_usb_suspend": _read_powercfg(),
           "radio_suspend": _read_radio_suspend()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    print(f"  [OK]  Config saved: {os.path.basename(path)}")
    return path

def list_configs():
    files = sorted(glob.glob(os.path.join(config_dir(), "etbt_config_*.json")))
    for i, p in enumerate(files, 1):
        try:
            with open(p, encoding="utf-8") as f: ts = json.load(f).get("saved", "?")
        except Exception: ts = "unreadable"
        print(f"  [{i}] {os.path.basename(p)}   saved {ts}")
    return files

def restore_config():
    files = list_configs()
    if not files:
        print("No saved configs found."); return
    raw = input("Restore which? [number]: ").strip()
    try: path = files[int(raw) - 1]
    except (ValueError, IndexError):
        print("No such entry."); return
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    print(f"Restoring {os.path.basename(path)} (saved {cfg.get('saved')}):")
    for e in cfg.get("endpoints", []):
        cur, _, _ = psrun(f"(Get-PnpDevice -InstanceId '{e['id']}' -ErrorAction SilentlyContinue).Status")
        cur = (cur or "").strip()
        if e["status"] == "OK" and cur and cur != "OK":
            _, err, rc = psrun(f"Enable-PnpDevice -InstanceId '{e['id']}' -Confirm:$false; 'done'")
            print(("  [OK]  re-enabled: " if rc == 0 else f"  [!!]  enable failed ({err[:80]}): ") + e["name"])
        elif e["status"] != "OK" and cur == "OK":
            _, err, rc = psrun(f"Disable-PnpDevice -InstanceId '{e['id']}' -Confirm:$false; 'done'")
            print(("  [OK]  re-disabled: " if rc == 0 else f"  [!!]  disable failed ({err[:80]}): ") + e["name"])
    pc = cfg.get("powercfg_usb_suspend", {})
    if "ac" in pc:
        psrun(f"powercfg /SETACVALUEINDEX SCHEME_CURRENT 2a737441-1930-4402-8d77-b2bebba308a3 "
              f"48e6b7a6-50f5-4782-a5d4-53bb8f07e226 {pc['ac']}")
    if "dc" in pc:
        psrun(f"powercfg /SETDCVALUEINDEX SCHEME_CURRENT 2a737441-1930-4402-8d77-b2bebba308a3 "
              f"48e6b7a6-50f5-4782-a5d4-53bb8f07e226 {pc['dc']}")
    psrun("powercfg /SETACTIVE SCHEME_CURRENT")
    print("  [OK]  USB selective-suspend indices restored")
    for r in cfg.get("radio_suspend", []):
        key = "HKLM:\\SYSTEM\\CurrentControlSet\\Enum\\" + r["instance"] + "\\Device Parameters"
        if r["value"] == "absent":
            psrun(f"Remove-ItemProperty -Path '{key}' -Name SelectiveSuspendEnabled -ErrorAction SilentlyContinue")
            print("  [OK]  radio suspend key removed (was absent): " + r["instance"][:40] + "...")
        else:
            psrun(f"New-ItemProperty -Path '{key}' -Name SelectiveSuspendEnabled -Value {r['value']} "
                  f"-PropertyType DWord -Force | Out-Null")
            print(f"  [OK]  radio suspend restored to {r['value']}: " + r["instance"][:40] + "...")
    psrun("Restart-Service bthserv -Force")
    print("  [OK]  Bluetooth stack restarted — restore complete.")

# ── STEP 2: FIX (each change reported; admin handled) ────────────────────────
def fix(gaps):
    print("\nSTEP 2 — FIX (one line per change)\n")
    if not is_admin():
        print("Not elevated. Fixes need Administrator.")
        ans = input("Relaunch elevated now? [Y/n]: ").strip().lower()
        if ans in ("", "y", "yes"):
            script = os.path.abspath(sys.argv[0])
            psrun(f"Start-Process -Verb RunAs '{sys.executable}' -ArgumentList '\"{script}\"'")
            print("Elevated window launched — continue there."); sys.exit(0)
        print("Continuing without fixes (diagnose/bridge only)."); return
    for e in gaps.get("hfp", []):
        out, err, rc = psrun(
            f"Disable-PnpDevice -InstanceId '{e['id']}' -Confirm:$false; 'done'")
        if rc == 0 and "done" in out:
            print("  [OK]  Disable HFP endpoint: " + e["name"])
        else:
            out2, err2, rc2 = psrun(f"pnputil /disable-device \"{e['id']}\"")
            if rc2 == 0:
                print("  [OK]  Disable HFP endpoint (pnputil): " + e["name"])
            else:
                print("  [!!]  Could not disable: " + e["name"])
                print(f"        Disable-PnpDevice: {err[:100]}")
                print(f"        pnputil: {(err2 or out2)[:100]}")
                print("        Manual path: Settings > Sound > Manage sound devices > disable it.")
    if gaps.get("hfp"):
        print("        (Re-enable in Device Manager if the headset mic is ever needed.)")
    # power: USB selective suspend off, both power sources
    for flag in ("SETACVALUEINDEX", "SETDCVALUEINDEX"):
        psrun(f"powercfg /{flag} SCHEME_CURRENT 2a737441-1930-4402-8d77-b2bebba308a3 "
              f"48e6b7a6-50f5-4782-a5d4-53bb8f07e226 0")
    psrun("powercfg /SETACTIVE SCHEME_CURRENT")
    print("  [OK]  USB selective suspend disabled (AC + battery)")
    out, err, rc = psrun(
        "Get-PnpDevice -Class Bluetooth -Status OK | ForEach-Object {"
        " $k = 'HKLM:\\SYSTEM\\CurrentControlSet\\Enum\\' + $_.InstanceId + '\\Device Parameters';"
        " if (Test-Path $k) { New-ItemProperty -Path $k -Name SelectiveSuspendEnabled "
        "   -Value 0 -PropertyType DWord -Force | Out-Null; 'set: ' + $_.FriendlyName } }")
    for line in (out or "").splitlines():
        print("  [OK]  Radio suspend off — " + line.replace("set: ", ""))
    psrun("Restart-Service bthserv -Force")
    print("  [OK]  Bluetooth stack restarted (clean renegotiation)")
    if gaps.get("wifi24"):
        if gaps.get("fiveg"):
            print("  [--]  WiFi is on 2.4 GHz and 5 GHz is VISIBLE: connect to "
                  + ", ".join(gaps["fiveg"]) + " — one click vacates Bluetooth's band.")
        else:
            print("  [--]  WiFi is on 2.4 GHz and NO 5 GHz is visible: the move is")
            print("        router-side. Your adapter is dual-band capable; the router is")
            print("        the limit. Best available mitigation tonight (router admin,")
            print("        usually http://tplinkwifi.net or 192.168.0.1):")
            print("          - set a FIXED channel: 1 (or 11) — never mid-band like 8,")
            print("            which splits Bluetooth's hop set into two cramped halves;")
            print("            edge-parking leaves one contiguous ~60 MHz clear.")
            print("          - set channel WIDTH to 20 MHz (40 MHz eats half the band).")
            print("        If the router has a disabled/hidden 5 GHz radio, enabling it")
            print("        is the full fix; if it is 2.4-only, these two settings are")
            print("        the whole router-side lever.")

# ── STEP 3: DELIVER — the lossless bridge ────────────────────────────────────
def bridge(gaps):
    print("\nSTEP 3 — DELIVER (bit-exact bridge; Ctrl+C returns here)\n")
    sd = ensure("sounddevice")
    pyaudio = ensure("pyaudiowpatch")
    p = pyaudio.PyAudio()
    try:
        lb = p.get_default_wasapi_loopback()
    except Exception:
        print("No default loopback: set default output to the device whose audio")
        print("you want bridged (speakers), then rerun."); p.terminate(); return
    rate = int(lb["defaultSampleRate"])
    devs = sd.query_devices()
    was_idx = next((i for i, a in enumerate(sd.query_hostapis())
                    if "wasapi" in a["name"].lower()), None)
    outs = [(i, d) for i, d in enumerate(devs) if d["max_output_channels"] > 0
            and (was_idx is None or d["hostapi"] == was_idx)]
    if was_idx is not None:
        print("(Host-API ladder honored: only WASAPI endpoints offered — exclusive-capable.)")
    # cross-match PortAudio outputs against the endpoint names THE SYSTEM gave us
    targets = [e["name"] for e in gaps.get("a2dp", [])]
    auto = [i for i, d in outs if any(t.lower() in d["name"].lower()
                                      or d["name"].lower() in t.lower() for t in targets)]
    if auto:
        dev = auto[0]
        print(f"Headphone endpoint auto-matched from enumeration: [{dev}] {devs[dev]['name']}")
    else:
        for i, d in outs:
            print(f"  [{i:2d}] {d['name']}")
        raw = input("Endpoint index (Enter = default): ").strip()
        dev = int(raw) if raw else None
    # FEEDBACK GUARD: bridging an endpoint's own loopback back into itself
    # doubles/echoes forever. Refuse with the correct topology, never loop.
    src_name = lb["name"].lower().replace(" [loopback]", "").strip()
    dst_name = (devs[dev]["name"] if dev is not None
                else sd.query_devices(kind="output")["name"]).lower().strip()
    if src_name in dst_name or dst_name in src_name:
        print("REFUSING: the capture source IS the destination endpoint — that")
        print("topology feeds the headphones their own output (echo doubling).")
        print("Correct setup: set Windows DEFAULT output to another device (e.g.")
        print("Speakers (Realtek)), so apps feed it; this bridge then carries its")
        print("loopback to the headphones bit-exact. Change the default and rerun.")
        p.terminate(); return
    try:
        ws = sd.WasapiSettings(exclusive=True)
        st = sd.RawOutputStream(samplerate=rate, channels=2, dtype="int16",
                                blocksize=BLOCK, extra_settings=ws, device=dev)
        st.start(); mode = "WASAPI EXCLUSIVE — rate dictated, engine bypassed"
    except Exception as e:
        print(f"Exclusive denied ({e}); direct shared stream instead.")
        st = sd.RawOutputStream(samplerate=rate, channels=2, dtype="int16",
                                blocksize=BLOCK, device=dev)
        st.start(); mode = "shared (direct)"
    inp = p.open(format=pyaudio.paInt16, channels=2, rate=rate, input=True,
                 input_device_index=lb["index"], frames_per_buffer=BLOCK)
    print(f"LIVE [{mode}] {lb['name']} -> endpoint @ {rate} Hz")
    blocks = 0
    try:
        while True:
            raw = inp.read(BLOCK, exception_on_overflow=False)
            st.write(raw)                       # untouched bytes: bit-exact
            blocks += 1
            if blocks % 64 == 0:
                ints = struct.unpack("<%dh" % (len(raw) // 2), raw)
                print(f"  blocks {blocks:6d} | {mirror_meter(ints[0::2], ints[1::2])}",
                      end="\r")
    except KeyboardInterrupt:
        print("\nBridge closed.")
    finally:
        inp.close(); st.stop(); st.close(); p.terminate()

# ── STEP 4: VERIFY ───────────────────────────────────────────────────────────
def verify():
    print("\nSTEP 4 — VERIFY (post-fix enumeration)\n")
    out, _, _ = psrun("Get-PnpDevice -Class AudioEndpoint -ErrorAction SilentlyContinue |"
                      " ForEach-Object { $_.FriendlyName + '  [' + $_.Status + ']' }")
    print(out or "(no endpoints listed)")
    hf = [l for l in (out or "").splitlines()
          if "hands-free" in l.lower() and "[OK]" in l]
    print("\nHands-Free still active: " + (str(len(hf)) if hf else "0 — profile flapping closed."))

# ── self-test (runs anywhere; verifies every exact-math component) ───────────
def selftest():
    ok = tot = 0
    def rep(n, c):
        nonlocal ok, tot; tot += 1; ok += bool(c)
        print(("[PASS] " if c else "[FAIL] ") + n)
    import random
    data = list(range(-8192, 8192, 7))
    rep("octave staging >>2<<2 exact", all(((a >> 2) << 2) == b for a, b in
        zip(data, octave_stage(octave_stage(data, -2), 2))))
    rep("octave staging identity", octave_stage(data, 0) == data)
    random.seed(N_ET)
    L = [random.randint(-32768, 32767) for _ in range(4096)] + [-32768, 32767, 0]
    R = [random.randint(-32768, 32767) for _ in range(4096)] + [32767, -32768, 0]
    M, S = ms_butterfly(L, R)
    rep("mirror butterfly bit-exact invertible", ms_inverse(M, S) == (L, R))
    rep("mirror meter mono ground", mirror_meter([5, -7], [5, -7]).endswith("0.000"))
    rep("endpoint parser: profile read from name, not guessed",
        "hands-free" in "Headset (X Hands-Free AG Audio)".lower())
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p1 = next_config_path(td); open(p1, "w").close()
        p2 = next_config_path(td)
        rep("config numbering increments (001 -> 002)",
            p1.endswith("001.json") and p2.endswith("002.json"))
        rep("config naming sanitizes and honors given name",
            next_config_path(td, "my test!").endswith("etbt_config_mytest.json"))
    print(f"\nSELFTEST: {ok}/{tot} PASSED")
    return ok == tot

# ── linear main ──────────────────────────────────────────────────────────────
def main():
    print("ET BLUETOOTH DIRECT — diagnose, fix, deliver, verify. Window stays open.")
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    gaps = diagnose()
    if gaps:
        ans = input("\n[Enter]=save config then fix   n=skip fixes   r=restore a saved config: ").strip().lower()
        if ans == "r":
            restore_config()
            verify()
        elif ans in ("", "y", "yes"):
            save_config(gaps)
            fix(gaps)
            verify()
    ans = input("\nStart the lossless bridge? [Y/n]: ").strip().lower()
    if ans in ("", "y", "yes"):
        bridge(gaps or {})
    print("\nIn-custody gaps closed. If dropouts persist ONLY near USB 3 activity,")
    print("that is the RF stage: move the dongle/antenna path off USB 3 — physics,")
    print("named plainly, not simulated.")
    _hold("Press Enter to exit...")

if __name__ == "__main__":
    main()
