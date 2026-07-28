# www.ExceptionTheory.com — Local Run & Pre-Deployment Verification

**Exception Theory — Michael James Muller — Exception Theory LLC**
*P ∘ D ∘ T = E*

This document is the complete procedure for running the site locally on the
Windows 10 machine, verifying it (visually and computationally), testing it on
the Galaxy S22 Ultra and the Samsung TV over LAN, and clearing the gates before
deployment. The site is fully static — zero JavaScript, zero backend — so
"running" it means serving files; "verifying" it means (a) rebuilding it from
the forge and watching 176/176 pass, and (b) checking the four effects against
their laws with your own eyes.

---

## 0. READ THIS FIRST — Required folder structure (the white-page fix)

**If you opened the page and got a WHITE page — plain black text, links, and a
giant diagram — nothing is broken and nothing "failed to load" except one
file: the stylesheet.** The page loaded; `css/et.css` did not, because the
files were not arranged in the folder tree the page expects. A from-scratch
static site is not like Weebly: there is no platform resolving paths for you.
`index.html` contains the line

```
<link rel="stylesheet" href="css/et.css">
```

which is a **relative path**: it means "the file `et.css` inside a folder
named `css` sitting next to me." If that file is not exactly there, the
browser's built-in error handling is to render the page unstyled — silently,
no crash, no popup. White page = wrong file placement. Full stop.

### The required tree

Assemble the downloaded files EXACTLY like this before doing anything else:

```
et_site\                       ← the folder name and location are your choice
│                                 (e.g. C:\Users\Mike\Desktop\et_site\)
├── index.html                 ← at the TOP level of the folder
├── et_tokens.json             ← top level
├── VERIFICATION_REPORT.txt    ← top level
├── css\                       ← a folder named exactly:  css
│   └── et.css                 ← the stylesheet, inside it, named exactly:  et.css
└── assets\                    ← a folder named exactly:  assets
    └── et_wheel.svg           ← the wheel, inside it
```

The forge sources and documents (`et_site_forge.cpp`, `cie_observer_data.hpp`,
`et_site_forge.py`, `cie_observer_data.py`, `ET_Site_Design_Document.md`, this
file) may sit at the top level alongside `index.html` — they have no effect on
rendering. Only the tree above matters to the browser.

### Assembly steps (Windows)

1. Create the folder, e.g. `Desktop\et_site`.
2. Inside it create the two subfolders — Explorer ▸ New Folder twice, named
   `css` and `assets` — or from a terminal opened in the folder:
   `mkdir css assets`
3. Move the downloads into place per the tree: `index.html`,
   `et_tokens.json`, `VERIFICATION_REPORT.txt` at the top;
   `et.css` **into** `css\`; `et_wheel.svg` **into** `assets\`.
4. **File-name gotcha:** Windows hides extensions by default, and some
   browsers save text files with an extra `.txt`. In Explorer turn on
   View ▸ Show ▸ **File name extensions**, and confirm the names are exactly
   `index.html`, `et.css`, `et_wheel.svg` — lowercase, no `.txt` tacked on.

### 10-second self-check (do this before anything else in this document)

Double-click `index.html`.

- **Correct:** a near-black page; the equation **P ∘ D ∘ T = E** glowing in
  color; drifting, twinkling stars behind everything.
- **Wrong:** a white page with plain text and a huge diagram → the tree above
  is not in place. Fix file locations and names; change no code.

### If it is still white after the tree is in place

Press **F12** ▸ **Network** tab ▸ reload the page. Look at the `et.css` row:

- Status **200** → stylesheet loaded; the page cannot be white.
- Status **404** (or `net::ERR_FILE_NOT_FOUND` when opened via
  double-click) → the browser tells you the exact URL where it looked;
  compare that path against the tree and correct the placement or the name.

**Shortcut that skips manual assembly entirely:** the forge rebuild in §5
emits a ready-made `dist\` folder already in this exact structure — copy or
serve `dist\` whole and the layout is guaranteed.

---

## 1. What is in this folder

The paths in this table ARE the required tree from §0: `css/et.css` means
"the file `et.css` inside the folder `css`" — not a file named `css/et.css`
and not `et.css` sitting loose next to `index.html`.

| File / dir | Role |
|---|---|
| `index.html` | The site (single page) |
| `css/et.css` | Every value a lattice address; all four effects live here |
| `assets/et_wheel.svg` | The sigil wheel (also inlined in the page) |
| `et_tokens.json` | The full token ledger: every value + its derivation law |
| `VERIFICATION_REPORT.txt` | The 176/176 check log from the shipping build |
| `et_site_forge.cpp` | **The authoritative forge** (C++ MPFR) — rebuilds everything above |
| `cie_observer_data.hpp` | Frozen CIE/sRGB translation layer (compile-time include) |
| `et_site_forge.py`, `cie_observer_data.py` | Independent Python cross-check (superseded as authority; see §6) |
| `ET_Site_Design_Document.md` | Full design record, laws, work history |
| `LOCAL_RUN_INSTRUCTIONS.md` | This file |

**Deployment set** (what actually goes to the server, nothing else):
`index.html`, `css/`, `assets/`, `et_tokens.json`, `VERIFICATION_REPORT.txt`.
The tokens file and report are intentionally public — they are the site's
transparency layer.

---

## 2. Fastest look (30 seconds, no server)

**Prerequisite: the §0 tree is assembled and its 10-second self-check shows
the dark page.** If your page is white, go back to §0 — do not continue.

Because the site has **no JavaScript and no fetches**, `file://` works:

1. Double-click `index.html`.
2. It opens in your default browser and everything renders, including all four
   effects, because every resource is a relative-path static file.

This is fine for a first look. For real testing, use the HTTP server in §3 —
it mirrors deployment exactly (MIME types, paths, caching behavior) and lets
the phone and TV connect.

---

## 3. Proper local server (mirrors deployment)

**Prerequisite: the §0 tree is assembled and verified.** The server does not
fix placement — it serves whatever folder you point it at, exactly as-is. A
white page over `http://` means the same thing it means over `file://`: the
tree is wrong.

Python 3 is already on the machine. From a terminal (**cmd**, PowerShell, or
Windows Terminal), `cd` into the folder that **directly contains
`index.html`** — the `et_site` folder itself from §0, NOT the `css` folder,
NOT its parent. Example:

```
cd C:\Users\Mike\Desktop\et_site
```

Then run:

```
py -m http.server 8080 --bind 127.0.0.1
```

Then open: **http://localhost:8080/**

- The console will log each request (`index.html`, `css/et.css`, …). You
  should see **no request for any `.js` file** — there are none.
- Stop the server with `Ctrl+C`.
- If `py` isn't on PATH, use `python -m http.server 8080 --bind 127.0.0.1`.

Alternative (if you prefer the Node toolchain from the TLP work):
`npx serve . -l 8080` — equivalent result.

---

## 4. Phone and TV testing over LAN (S22 Ultra, Samsung TV)

1. Find the PC's LAN IPv4: run `ipconfig` and read the `IPv4 Address` of the
   active adapter (e.g. `192.168.1.23`).
2. Start the server bound to all interfaces:

```
py -m http.server 8080 --bind 0.0.0.0
```

3. If Windows Firewall prompts, allow access on **Private networks**. (If no
   prompt appears and the phone can't connect, add an inbound rule for TCP
   8080, Private profile, or temporarily use
   `netsh advfirewall firewall add rule name="et-site-test" dir=in action=allow protocol=TCP localport=8080`
   and delete the rule after testing.)
4. On the S22 Ultra (same Wi-Fi), open **http://192.168.1.23:8080/**
   (your IP from step 1). Same on the TV's browser.

**Expected differences on non-Chromium-desktop engines:**
- **Comet:** `offset-path: path(...)` on SVG children requires a current
  Chrome/Edge/Firefox. On engines without it (some TV browsers, older
  Samsung Internet), the comet simply does not appear — by design the comet
  circles carry `opacity: 0` outside the animation, so degradation is clean,
  not broken.
- Everything else (tunnel, glow, starfield, Engine panels) is plain CSS
  animation + gradients and runs anywhere modern.

---

## 5. Rebuild from the forge and verify (the real test)

This is the verification that matters: compile the authoritative forge,
run it, and watch it refuse to emit a site unless **176/176** checks pass.

### 5a. Recommended path — MSYS2 (g++, matches the build header exactly)

1. Install MSYS2 from msys2.org (default location `C:\msys64`).
2. Open the **"MSYS2 UCRT64"** shell and install the toolchain + libraries:

```
pacman -S --needed mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-mpfr mingw-w64-ucrt-x86_64-gmp
```

3. `cd` to this folder (Windows drives mount as `/c/...`, e.g.
   `cd "/c/Users/Mike/Downloads/et_site"`).
4. Compile — this is the exact command from the file header:

```
g++ -std=c++17 -O2 -Wall -o et_site_forge et_site_forge.cpp -lmpfr -lgmp
```

   Expected: **no output** (clean, zero warnings).
5. Run it:

```
./et_site_forge.exe
```

   Expected tail of output:

```
  PASSED: 176   FAILED: 0   TOTAL: 176
  dist/ written: index.html, css/et.css, assets/et_wheel.svg,
  et_tokens.json, VERIFICATION_REPORT.txt  (no JavaScript emitted)
```

   Exit code is `0` on success and `1` if **any** check fails (the build
   gates itself; check with `echo $?`).
6. **Reproducibility check** (byte-identical determinism):

```
cp -r dist dist_run1
./et_site_forge.exe
diff -r dist_run1 dist && echo BYTE-IDENTICAL
```

   Expected: `BYTE-IDENTICAL`.
7. Compare your fresh build against the shipped files (they should match
   byte-for-byte except the footer year if rebuilt in a different calendar
   year):

```
diff dist/css/et.css css/et.css
diff dist/index.html index.html
```

8. Serve the fresh build: `cd dist` then §3.

### 5b. Alternative path — vcpkg + MSVC/CLion

1. `vcpkg install mpfr gmp` (triplet `x64-windows`; the mpfr port pulls gmp).
2. Minimal `CMakeLists.txt` in this folder:

```cmake
cmake_minimum_required(VERSION 3.20)
project(et_site_forge CXX)
set(CMAKE_CXX_STANDARD 17)
add_executable(et_site_forge et_site_forge.cpp)
find_library(MPFR_LIB mpfr REQUIRED)
find_library(GMP_LIB  gmp  REQUIRED)
find_path(MPFR_INC mpfr.h REQUIRED)
target_include_directories(et_site_forge PRIVATE ${MPFR_INC})
target_link_libraries(et_site_forge PRIVATE ${MPFR_LIB} ${GMP_LIB})
```

3. Open in CLion with the vcpkg toolchain file
   (`-DCMAKE_TOOLCHAIN_FILE=<vcpkg>/scripts/buildsystems/vcpkg.cmake`),
   build, and run with the **working directory set to this folder** (the
   forge writes `dist/` relative to the working directory;
   `cie_observer_data.hpp` is needed at compile time only).
4. Same expectations as 5a steps 5–7.

Note on `mpfr_rint` ties-to-even, precision (250 digits = 200 working + 50
guard), and the zero-float guarantee: all enforced inside the source and its
own checks — nothing to configure.

---

## 6. The Python cross-check (optional)

`et_site_forge.py` is retained as the **independent cross-implementation**,
not for deployment. Its dist output is the superseded v1 architecture.

- If you want to reproduce the 48/48 hex agreement: copy `et_site_forge.py`
  and `cie_observer_data.py` into a **separate scratch folder** (it also
  writes `./dist` and would otherwise overwrite the C++ output), run
  `py et_site_forge.py` (needs `pip install mpmath`), then compare the
  48 `hex*` values under `spectral_classes` in the two `et_tokens.json`
  files. They must be identical, character for character.
- **Never deploy the Python `dist/`.**

---

## 7. Visual verification checklist — each effect against its law

Do this on desktop Chrome or Edge at http://localhost:8080/ with DevTools
open (F12).

**Global**
- [ ] Page renders **dark** with the colored, glowing equation and stars.
      A white unstyled page means the §0 tree is wrong — `et.css` must show
      status **200** in the Network tab. Fix placement, not code.
- [ ] **Network tab:** no `.js` requested anywhere. **Console:** empty.
- [ ] Disable JavaScript (DevTools ▸ Ctrl+Shift+P ▸ "Disable JavaScript"),
      reload: the site is **identical**. Re-enable after.
- [ ] View source: no `<script>` element exists.

**The deep field (starfield)**
- [ ] 137 stars, three drift layers (the drift is deliberately glacial —
      3072/6144/12288 s per wrap; confirm direction by watching one bright
      star against a heading for ~a minute).
- [ ] Twinkle periods 2–4 s; the large haloed stars (d ≤ 2) visibly breathe
      into full brightness (they touch the opacity ceiling — the Exception
      cap); the tiny dust caps near 0.117 (= 16/137).

**Octave Tunnel (hero backdrop)**
- [ ] Faint spectral rings expand behind the equation.
- [ ] **Seamlessness test:** watch across the 32 s mark (one full period) —
      there must be **no visible jump or seam**, because the loop is one
      exact octave. Watch two periods (64 s) to be sure.

**Ψ-glow (hero equation)**
- [ ] The aura of `P ∘ D ∘ T = E` breathes with an 8 s period, subtle,
      never dying to zero (Shimmer range [1−√V, 1+√V]).

**Cascade Comet (sigil wheel — scroll to The Lattice)**
- [ ] A gold comet with three fading ghosts traverses the star polygon.
- [ ] Full circuit = **12.99 s**. Time it.
- [ ] **Impedance law visible:** it crawls through the segments landing on
      gravity/tritone-class nodes and whips through the EM-class ones —
      the speed ratio between slowest and fastest segment is ξ(1)/ξ(12) =
      8.5625 : 1.

**The Engine (nav ▸ Engine)**
- [ ] Panels change every 8 s; full cycle 96 s (12 panels).
- [ ] The **3/2** panel shows ε = 1.955…¢ — the canonical delta.
- [ ] The **4/3** panel shows the *same* |ε| (IC-149) and k = 5, d = 12.
- [ ] The **2^(1/12)** and **√2** panels read **ε = 0 — lattice-exact**.
- [ ] On every panel, the `Π⁻¹(k,ε)` line's 60 digits match the `r` line's
      60 digits exactly. Pick one and compare by eye.

**Reduced motion**
- [ ] DevTools ▸ Ctrl+Shift+P ▸ "Show Rendering" ▸ *Emulate CSS
      prefers-reduced-motion: reduce* (or Windows Settings ▸ Accessibility ▸
      Visual effects ▸ Animation effects **Off**). Everything freezes: stars
      static, tunnel static, comet hidden, Engine shows the π panel; the
      page remains fully readable.

**Layout / responsive**
- [ ] DevTools device toolbar at 393 px width (S22 Ultra class): columns
      collapse to single at the 512 px breakpoint (bp-mid = 2^(108/12) =
      512 exactly); hero equation scales fluidly; nothing overflows.
- [ ] Real-device pass on the S22 Ultra via §4.

**Content gates**
- [ ] Footer + About email reads **exceptiontheory@gmail.com** (typo fix
      confirmed in place).
- [ ] DOI link opens `https://doi.org/10.5281/zenodo.19762311`.
- [ ] Products shows the "Forthcoming" state (list still pending).

---

## 8. Pre-deployment gate (all boxes required)

- [ ] §0 folder tree verified: page renders dark; `et.css` loads with 200.
- [ ] §5 rebuild: clean compile, **176/176 PASSED**, exit code 0.
- [ ] §5.6 double-run **BYTE-IDENTICAL**.
- [ ] §7 checklist complete on desktop.
- [ ] §4 pass on the S22 Ultra (and TV if desired).
- [ ] Deployment set copied (§1 list only — no forge sources unless you
      *want* to publish them).
- [ ] Server: any static host works; for the planned VPS, the nginx block
      in ET_Site_Design_Document §9 applies unchanged (static root + TLS;
      `gzip on;` recommended — `et.css` and `index.html` compress ~4:1).

When every box is checked, the site is verified end to end: the mathematics
by the forge's own 176 gates, the rendering by your eyes, on your hardware.

*Michael James Muller — Aevum Defluo — Exception Theory LLC — P ∘ D ∘ T = E*
