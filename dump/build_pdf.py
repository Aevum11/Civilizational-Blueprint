#!/usr/bin/env python3
"""
ET Algebraic Identities — PDF Compilation Builder
==================================================
Combines all 13 algebraic identity verification scripts into a single PDF
with proper separators, script titles, and identity tags.

Order is the chronological derivation order:
  Identity #0  → verify_lossless_bijection.py     (Foundation: r ↔ (k,d,ε))
  Finding 11   → cross_resolution_transition.py   (Inter-tower transition maps)
  Identity A   → lattice_arithmetic_identity1.py  (Multiplication, division, powers)
  Identity B   → differential_control_identity1.py (Λ = 1200/ln 2)
  Identity C   → d_family_composition_identity1.py (Set-valued d₁ ⊗ d₂)
  Identity D   → complex_lattice_arithmetic_identity.py (Two-axis L_N^C)
  Identity E1  → harmonic_fqg_composition1.py     (12×12, 42 closure)
  Identity E2  → sublattice_fqg_composition.py    (36·4^ℓ growth)
  Identity E3  → composite_bridge_identity.py     (Harmonic ↔ Sublattice bridge)
  Identity F   → incoherence_boundary_identity.py (t(50¢) = K = 2/3)
  Identity G   → triple_backbone_bridge_identity.py (Π_N = Disc∘T∘Cont)
  Identity H   → harmonic_transfer_tensor.py      (T_κ(d₁,d₂;d₃))
  Identity I   → substantiation_transition_identity.py (Birth Triad algebra)
"""

import os
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# -----------------------------------------------------------------------------
# FONT REGISTRATION — DejaVu Sans Mono supports the full Unicode range used by
# the scripts (Greek letters Π ε σ ∘ ∂, math symbols ∀ ⊗ ∈ ≠ ≤ ≥, arrows → ↔,
# subscripts ₀₁₂..., superscripts ⁰¹²..., box-drawing chars used in ASCII art).
# Without a Unicode TTF font, reportlab's built-in Courier would render these
# as solid black boxes — see the PDF skill warning on this exact point.
#
# FONT CHOICE: GNU FreeFont (FreeMono / FreeSans / FreeSerif) is used because
# DejaVu Sans Mono is missing four characters that appear in the scripts:
#   ℓ (U+2113, script small l),     ∤ (U+2224, does not divide),
#   ⟹ (U+27F9, long ⇒),             ⟺ (U+27FA, long ⇔).
# GNU FreeFont covers all 86 distinct non-ASCII glyphs used across the 13
# scripts with zero misses — audited via fontTools.ttLib.
# -----------------------------------------------------------------------------
FONT_REGULAR = "/usr/share/fonts/truetype/freefont/FreeSans.ttf"
FONT_BOLD    = "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf"
FONT_MONO    = "/usr/share/fonts/truetype/freefont/FreeMono.ttf"
FONT_MONO_B  = "/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf"

pdfmetrics.registerFont(TTFont("DejaVu",       FONT_REGULAR))
pdfmetrics.registerFont(TTFont("DejaVu-Bold",  FONT_BOLD))
pdfmetrics.registerFont(TTFont("DejaVuMono",   FONT_MONO))
pdfmetrics.registerFont(TTFont("DejaVuMono-B", FONT_MONO_B))


# -----------------------------------------------------------------------------
# PAGE LAYOUT CONSTANTS
# -----------------------------------------------------------------------------
PAGE_W, PAGE_H = letter                       # 612 × 792 pt
MARGIN_L       = 0.6 * inch
MARGIN_R       = 0.6 * inch
MARGIN_T       = 0.7 * inch
MARGIN_B       = 0.7 * inch
USABLE_W       = PAGE_W - MARGIN_L - MARGIN_R  # 472 pt
USABLE_H       = PAGE_H - MARGIN_T - MARGIN_B  # 644 pt

CODE_FONT_SIZE = 7.5
CODE_LEADING   = 9.0                          # line height for code
HEADER_FONT_SZ = 8.0

# At 7.5pt DejaVuSansMono, average glyph width ≈ 4.5pt → ~104 cols per line.
# We wrap explicitly at MAX_COLS to avoid right-margin overflow.
MAX_COLS       = 100

UPLOAD_DIR     = "/mnt/user-data/uploads"
OUT_PATH       = "/mnt/user-data/outputs/ET_Algebraic_Identities_Compilation.pdf"

# -----------------------------------------------------------------------------
# SCRIPT ORDER — chronological derivation order from the corpus history.
# The 'tag' is the identity tag, 'title' is the descriptive heading,
# 'subtitle' is the one-line summary of what the script proves.
# -----------------------------------------------------------------------------
SCRIPTS = [
    {
        "filename": "verify_lossless_bijection.py",
        "tag":      "Identity #0",
        "title":    "Lossless Bijection Verification",
        "subtitle": "Foundation: r ↔ (k, d, ε) is an algebraic identity, "
                    "proven symbolically (sympy) and numerically (precision-"
                    "scaling). The seed from which all other identities grow.",
    },
    {
        "filename": "cross_resolution_transition.py",
        "tag":      "Finding 11",
        "title":    "Cross-Resolution Transition Maps",
        "subtitle": "Inter-tower transition algebra: Π_N₂ ∘ Π_N₁⁻¹ as the "
                    "coordinate transition between charts. Cross-resolution, "
                    "cross-seed, full cross-tower, and commutativity proof.",
    },
    {
        "filename": "lattice_arithmetic_identity1.py",
        "tag":      "Identity A",
        "title":    "Lattice Arithmetic",
        "subtitle": "Multiplication, division, reciprocation, and powers in "
                    "(k, d, ε) coordinates without accessing r. The κ "
                    "(T-correction) term ∈ {−1, 0, +1}. Associativity proven.",
    },
    {
        "filename": "differential_control_identity1.py",
        "tag":      "Identity B",
        "title":    "Differential Control Law",
        "subtitle": "Continuous-time bijection evolution: dε = Λ · dr/r with "
                    "Λ = 1200/ln 2. Exact finite shift r_new = r · 2^(Δε/1200). "
                    "Cell transitions, restoration control (exponential ε-decay).",
    },
    {
        "filename": "d_family_composition_identity1.py",
        "tag":      "Identity C",
        "title":    "d-Family Composition Law",
        "subtitle": "Set-valued d₁ ⊗ d₂: which d-families can result from "
                    "configuration interaction. Residue set Res_N(d) and the "
                    "lcm bound. Tells the field which family dominates an "
                    "interaction.",
    },
    {
        "filename": "complex_lattice_arithmetic_identity.py",
        "tag":      "Identity D",
        "title":    "Complex Lattice Arithmetic",
        "subtitle": "Two-axis complex lattice L_N^C: real axis k_r, imaginary "
                    "axis k_θ. Phase projection with Λ_θ = 600/π. U(1) "
                    "compactness, complex multiply / reciprocal / power.",
    },
    {
        "filename": "harmonic_fqg_composition1.py",
        "tag":      "Identity E1",
        "title":    "Harmonic FQG Composition",
        "subtitle": "Fixed 144-cell (12×12) Force Quadrant Grid. The harmonic "
                    "d-composition closure: 42 elements (Bell-like number). "
                    "No primes greater than 12 appear in the closure set.",
    },
    {
        "filename": "sublattice_fqg_composition.py",
        "tag":      "Identity E2",
        "title":    "Sublattice FQG Composition",
        "subtitle": "Growing FQG via tower escalation: cell count 36 · 4^ℓ. "
                    "Lattice-exact invariance for ε = 0 values. d-bouncing "
                    "across tower levels for ε ≠ 0 values.",
    },
    {
        "filename": "composite_bridge_identity.py",
        "tag":      "Identity E3",
        "title":    "Composite Bridge Identity",
        "subtitle": "The bridge between the harmonic (fixed) and sublattice "
                    "(growing) families. Three-layer partition: Harmonic, "
                    "Harmonic Composite, Tower-Native. Shadow vs decomposition.",
    },
    {
        "filename": "incoherence_boundary_identity.py",
        "tag":      "Identity F",
        "title":    "∂I (Boundary of Incoherence)",
        "subtitle": "The Koide-tightness boundary: t(50¢) = K = 2/3 exactly. "
                    "Bifurcation at the ∂I edge. Pairwise / sublattice / "
                    "lattice incoherence filters from a single algebraic root.",
    },
    {
        "filename": "triple_backbone_bridge_identity.py",
        "tag":      "Identity G",
        "title":    "Triple Backbone Bridge Identity",
        "subtitle": "The bridge between L₁ (Webb stroke), L₂ (Palindromic "
                    "Cascade), and L₃ (EML) and the lattice. Π_N = "
                    "Disc_Webb ∘ T_round ∘ Cont_EML. Catalan C₆ = 132 = d_max.",
    },
    {
        "filename": "harmonic_transfer_tensor.py",
        "tag":      "Identity H",
        "title":    "Harmonic Transfer Tensor",
        "subtitle": "Inter-family energy transfer from lattice geometry: "
                    "T_κ(d₁, d₂; d₃) — 648 entries, partition of unity. "
                    "EM universality, gravitational accessibility, fusion as T-event.",
    },
    {
        "filename": "substantiation_transition_identity.py",
        "tag":      "Identity I",
        "title":    "Substantiation Transition (Birth Triad Algebra)",
        "subtitle": "The transition (P, D, T) → (P∘D∘T) → E in lattice "
                    "coordinates. Fixed point (0, 1, 0), canonical (−53, 12, 0). "
                    "Kolmogorov seed structure of the Birth Triad.",
    },
]


# -----------------------------------------------------------------------------
# CODE WRAPPING — preserves indentation, wraps long lines with a continuation
# marker so the original line structure remains visible.
# -----------------------------------------------------------------------------
def wrap_code_line(line, max_cols=MAX_COLS):
    """Wrap a single source line to <= max_cols characters, preserving
    indentation on continuations and marking continuations with '↪'.
    Tabs are expanded to 4 spaces. Returns a list of display lines."""
    line = line.replace("\t", "    ").rstrip("\n").rstrip("\r")

    if len(line) <= max_cols:
        return [line]

    # Compute leading whitespace to preserve indent on continuations
    stripped = line.lstrip(" ")
    indent = len(line) - len(stripped)
    indent_str = " " * indent

    # Continuation prefix: indent + continuation arrow + one space
    cont_prefix = indent_str + "↪ "
    cont_cols   = max_cols - len(cont_prefix)
    if cont_cols < 20:
        cont_cols = max_cols - 4
        cont_prefix = "  ↪ "

    out = [line[:max_cols]]
    rest = line[max_cols:]
    while rest:
        chunk = rest[:cont_cols]
        out.append(cont_prefix + chunk)
        rest = rest[cont_cols:]
    return out


def read_script_lines(filename):
    """Read a script and return a list of (line_number, wrapped_lines_list)."""
    path = os.path.join(UPLOAD_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        raw_lines = f.readlines()

    result = []
    for ln, raw in enumerate(raw_lines, start=1):
        display_lines = wrap_code_line(raw)
        result.append((ln, display_lines))
    return result


# -----------------------------------------------------------------------------
# DRAWING PRIMITIVES
# -----------------------------------------------------------------------------
def draw_text_centered(c, x_center, y, text, font, size, color=(0, 0, 0)):
    c.setFont(font, size)
    c.setFillColorRGB(*color)
    w = pdfmetrics.stringWidth(text, font, size)
    c.drawString(x_center - w / 2.0, y, text)


def draw_text_wrapped_centered(c, x_center, y_start, text, font, size,
                                 max_width, leading, color=(0, 0, 0)):
    """Word-wrap centered paragraph. Returns final y."""
    c.setFont(font, size)
    c.setFillColorRGB(*color)
    words = text.split()
    if not words:
        return y_start
    lines = []
    cur = words[0]
    for w in words[1:]:
        trial = cur + " " + w
        if pdfmetrics.stringWidth(trial, font, size) <= max_width:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    lines.append(cur)
    y = y_start
    for ln in lines:
        wpt = pdfmetrics.stringWidth(ln, font, size)
        c.drawString(x_center - wpt / 2.0, y, ln)
        y -= leading
    return y


# -----------------------------------------------------------------------------
# COVER PAGE
# -----------------------------------------------------------------------------
def draw_cover(c):
    # Top decorative rule
    c.setStrokeColorRGB(0.1, 0.1, 0.1)
    c.setLineWidth(1.5)
    c.line(MARGIN_L, PAGE_H - 1.0 * inch,
           PAGE_W - MARGIN_R, PAGE_H - 1.0 * inch)
    c.setLineWidth(0.6)
    c.line(MARGIN_L, PAGE_H - 1.05 * inch,
           PAGE_W - MARGIN_R, PAGE_H - 1.05 * inch)

    y = PAGE_H - 2.2 * inch
    draw_text_centered(c, PAGE_W / 2, y,
                       "EXCEPTION THEORY",
                       "DejaVu-Bold", 26)
    y -= 36
    draw_text_centered(c, PAGE_W / 2, y,
                       "Algebraic Identities",
                       "DejaVu-Bold", 20)
    y -= 28
    draw_text_centered(c, PAGE_W / 2, y,
                       "Complete Verification Script Compilation",
                       "DejaVu", 14)

    y -= 1.0 * inch
    draw_text_wrapped_centered(
        c, PAGE_W / 2, y,
        "Thirteen forward-derived algebraic identities verifying the "
        "Sempaevum bijection Π_N(r) = (k, d, ε) and its derived structures. "
        "Each script is an algebraic identity, not an approximation — "
        "verified at high precision via mpmath and symbolically via sympy.",
        "DejaVu", 11, USABLE_W - 60, 16)

    y -= 1.6 * inch
    draw_text_centered(c, PAGE_W / 2, y,
                       "Michael James Muller — Aevum Defluo",
                       "DejaVu-Bold", 12)
    y -= 18
    draw_text_centered(c, PAGE_W / 2, y,
                       "P ∘ D ∘ T = E",
                       "DejaVu-Bold", 14)
    y -= 18
    draw_text_centered(c, PAGE_W / 2, y,
                       "Derivation Standard: ET-native, zero external axioms",
                       "DejaVu", 10, color=(0.35, 0.35, 0.35))

    # Bottom decorative rule
    c.setLineWidth(0.6)
    c.line(MARGIN_L, 1.2 * inch, PAGE_W - MARGIN_R, 1.2 * inch)
    c.setLineWidth(1.5)
    c.line(MARGIN_L, 1.15 * inch, PAGE_W - MARGIN_R, 1.15 * inch)

    draw_text_centered(c, PAGE_W / 2, 0.85 * inch,
                       "For every exception there is an exception, "
                       "except the exception.",
                       "DejaVu", 9, color=(0.4, 0.4, 0.4))

    c.showPage()


# -----------------------------------------------------------------------------
# TABLE OF CONTENTS PAGE
# -----------------------------------------------------------------------------
def draw_toc(c, page_index):
    # Header
    y = PAGE_H - MARGIN_T
    draw_text_centered(c, PAGE_W / 2, y, "CONTENTS",
                       "DejaVu-Bold", 18)
    y -= 8
    c.setStrokeColorRGB(0, 0, 0)
    c.setLineWidth(0.8)
    c.line(MARGIN_L, y, PAGE_W - MARGIN_R, y)
    y -= 30

    c.setFillColorRGB(0, 0, 0)
    line_h = 22

    # Column layout: tag | title | filename | page
    col_tag_x      = MARGIN_L
    col_title_x    = MARGIN_L + 80
    col_page_x     = PAGE_W - MARGIN_R

    for i, s in enumerate(SCRIPTS):
        # Row top text (tag + title)
        c.setFont("DejaVu-Bold", 10)
        c.drawString(col_tag_x, y, s["tag"])
        c.setFont("DejaVu-Bold", 10)
        title_text = s["title"]
        c.drawString(col_title_x, y, title_text)
        # Page number (right-aligned)
        c.setFont("DejaVu", 10)
        page_text = str(page_index[i])
        pw = pdfmetrics.stringWidth(page_text, "DejaVu", 10)
        c.drawString(col_page_x - pw, y, page_text)

        # Row second line (filename)
        y2 = y - 12
        c.setFont("DejaVuMono", 8.5)
        c.setFillColorRGB(0.32, 0.32, 0.32)
        c.drawString(col_title_x, y2, s["filename"])
        c.setFillColorRGB(0, 0, 0)

        # Separator dots between title and page number
        # (cosmetic dotted leader)
        # Skipped for cleanliness — page number is right-aligned and clear.

        y -= line_h

        # If we somehow run out of space, page break (shouldn't with 13 rows)
        if y < MARGIN_B + 40:
            c.showPage()
            y = PAGE_H - MARGIN_T

    # Footer
    c.setFont("DejaVu", 8.5)
    c.setFillColorRGB(0.35, 0.35, 0.35)
    c.drawCentredString(PAGE_W / 2, MARGIN_B - 10,
                        "Exception Theory — Algebraic Identities Compilation")
    c.setFillColorRGB(0, 0, 0)
    c.showPage()


# -----------------------------------------------------------------------------
# SEPARATOR / TITLE PAGE FOR EACH SCRIPT
# -----------------------------------------------------------------------------
def draw_separator(c, script_meta, script_index):
    # Decorative top frame
    c.setStrokeColorRGB(0.1, 0.1, 0.1)
    c.setLineWidth(2.0)
    c.rect(MARGIN_L - 0.15 * inch,
           MARGIN_B - 0.15 * inch,
           USABLE_W + 0.3 * inch,
           USABLE_H + 0.3 * inch,
           stroke=1, fill=0)
    c.setLineWidth(0.5)
    c.rect(MARGIN_L - 0.05 * inch,
           MARGIN_B - 0.05 * inch,
           USABLE_W + 0.1 * inch,
           USABLE_H + 0.1 * inch,
           stroke=1, fill=0)

    # Identity tag (large, centered, upper third)
    y = PAGE_H - 2.6 * inch
    draw_text_centered(c, PAGE_W / 2, y,
                       script_meta["tag"],
                       "DejaVu-Bold", 32)

    y -= 12
    c.setStrokeColorRGB(0, 0, 0)
    c.setLineWidth(0.8)
    rule_half = 1.5 * inch
    c.line(PAGE_W / 2 - rule_half, y, PAGE_W / 2 + rule_half, y)

    # Descriptive title
    y -= 36
    draw_text_centered(c, PAGE_W / 2, y,
                       script_meta["title"],
                       "DejaVu-Bold", 18)

    # Filename in monospace
    y -= 36
    draw_text_centered(c, PAGE_W / 2, y,
                       script_meta["filename"],
                       "DejaVuMono-B", 13, color=(0.2, 0.2, 0.2))

    # One-line subtitle (wrapped)
    y -= 48
    draw_text_wrapped_centered(
        c, PAGE_W / 2, y,
        script_meta["subtitle"],
        "DejaVu", 11, USABLE_W - 80, 16,
        color=(0.25, 0.25, 0.25))

    # Position marker at the bottom: "Script N of 13"
    y_bottom = MARGIN_B + 0.4 * inch
    draw_text_centered(c, PAGE_W / 2, y_bottom,
                       f"Script {script_index} of {len(SCRIPTS)}",
                       "DejaVu", 10, color=(0.4, 0.4, 0.4))

    # ET tagline at very bottom
    draw_text_centered(c, PAGE_W / 2, y_bottom - 16,
                       "P ∘ D ∘ T = E",
                       "DejaVu-Bold", 11, color=(0.3, 0.3, 0.3))

    c.showPage()


# -----------------------------------------------------------------------------
# CODE PAGES — render the script source with line numbers, header, footer
# -----------------------------------------------------------------------------
def draw_code_header(c, script_meta, page_num, total_pages):
    """Header at top of each code page."""
    y = PAGE_H - MARGIN_T + 18

    # Left: tag + title
    c.setFont("DejaVu-Bold", HEADER_FONT_SZ)
    c.setFillColorRGB(0.1, 0.1, 0.1)
    left_text = f"{script_meta['tag']} — {script_meta['title']}"
    c.drawString(MARGIN_L, y, left_text)

    # Right: filename + page count within script
    c.setFont("DejaVuMono", HEADER_FONT_SZ)
    c.setFillColorRGB(0.35, 0.35, 0.35)
    right_text = f"{script_meta['filename']}   ·   p. {page_num}/{total_pages}"
    rw = pdfmetrics.stringWidth(right_text, "DejaVuMono", HEADER_FONT_SZ)
    c.drawString(PAGE_W - MARGIN_R - rw, y, right_text)

    # Underline rule
    c.setStrokeColorRGB(0.6, 0.6, 0.6)
    c.setLineWidth(0.4)
    c.line(MARGIN_L, y - 4, PAGE_W - MARGIN_R, y - 4)
    c.setFillColorRGB(0, 0, 0)


def estimate_pages_for_script(wrapped_lines):
    """Given the list of (lineno, [display_lines]), estimate how many code
    pages it occupies, based on USABLE_H and CODE_LEADING. Used so we can
    write a 'page X of Y' header on each code page."""
    # Total display lines (sum of len(display_lines) for each source line)
    total_display = sum(len(d) for _, d in wrapped_lines)
    lines_per_page = int((USABLE_H - 6) // CODE_LEADING)
    pages = (total_display + lines_per_page - 1) // lines_per_page
    return max(1, pages), lines_per_page


def draw_code_pages(c, script_meta, wrapped_lines):
    """Render the source code across as many pages as needed.
    Each source line gets a small line-number gutter on the left.
    Continuation lines (from wrapping) have no gutter number."""
    total_pages, lines_per_page = estimate_pages_for_script(wrapped_lines)

    # Determine gutter width: enough digits for the largest line number
    max_lineno = wrapped_lines[-1][0] if wrapped_lines else 1
    gutter_digits = max(3, len(str(max_lineno)))
    # gutter holds: line number (right-aligned, gutter_digits chars) + " │ "
    gutter_text_width = gutter_digits + 3  # "NNN │ "

    page_num   = 1
    line_count = 0
    y = PAGE_H - MARGIN_T

    draw_code_header(c, script_meta, page_num, total_pages)
    c.setFont("DejaVuMono", CODE_FONT_SIZE)
    c.setFillColorRGB(0, 0, 0)

    for src_ln, display_list in wrapped_lines:
        for i, dl in enumerate(display_list):
            if line_count >= lines_per_page:
                # Footer for current page
                draw_code_footer(c, page_num, total_pages)
                c.showPage()
                page_num += 1
                line_count = 0
                y = PAGE_H - MARGIN_T
                draw_code_header(c, script_meta, page_num, total_pages)
                c.setFont("DejaVuMono", CODE_FONT_SIZE)
                c.setFillColorRGB(0, 0, 0)

            # Gutter — line number on the first display line, blank on continuations
            if i == 0:
                gutter = f"{src_ln:>{gutter_digits}d} │ "
                c.setFillColorRGB(0.55, 0.55, 0.55)
                c.drawString(MARGIN_L, y - CODE_FONT_SIZE - 1, gutter)
                c.setFillColorRGB(0, 0, 0)
            else:
                gutter = " " * gutter_digits + " │ "
                c.setFillColorRGB(0.55, 0.55, 0.55)
                c.drawString(MARGIN_L, y - CODE_FONT_SIZE - 1, gutter)
                c.setFillColorRGB(0, 0, 0)

            # Code text (after gutter)
            gutter_pt_width = pdfmetrics.stringWidth(
                gutter, "DejaVuMono", CODE_FONT_SIZE)
            c.drawString(MARGIN_L + gutter_pt_width,
                         y - CODE_FONT_SIZE - 1, dl)

            y -= CODE_LEADING
            line_count += 1

    # Footer for the last page of this script
    draw_code_footer(c, page_num, total_pages)
    c.showPage()


def draw_code_footer(c, page_num, total_pages):
    """Bottom footer on each code page."""
    y = MARGIN_B - 18
    c.setStrokeColorRGB(0.6, 0.6, 0.6)
    c.setLineWidth(0.4)
    c.line(MARGIN_L, y + 10, PAGE_W - MARGIN_R, y + 10)
    c.setFont("DejaVu", 7.5)
    c.setFillColorRGB(0.4, 0.4, 0.4)
    c.drawString(MARGIN_L, y,
                 "ET Algebraic Identities Compilation")
    right = "P ∘ D ∘ T = E"
    rw = pdfmetrics.stringWidth(right, "DejaVu", 7.5)
    c.drawString(PAGE_W - MARGIN_R - rw, y, right)
    c.setFillColorRGB(0, 0, 0)


# -----------------------------------------------------------------------------
# MAIN BUILD
# -----------------------------------------------------------------------------
def build_pdf():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    c = canvas.Canvas(OUT_PATH, pagesize=letter)
    c.setTitle("Exception Theory — Algebraic Identities Compilation")
    c.setAuthor("Michael James Muller (Aevum Defluo)")
    c.setSubject("Forward-derived algebraic identities of the Sempaevum bijection")
    c.setCreator("ET PDF Builder")

    # ---- Pre-read all scripts so we can compute page numbers for the TOC ----
    print("Reading scripts...")
    cache = []
    for s in SCRIPTS:
        wrapped = read_script_lines(s["filename"])
        total_pages, _ = estimate_pages_for_script(wrapped)
        cache.append({
            "meta":     s,
            "wrapped":  wrapped,
            "n_pages":  total_pages,
        })
        print(f"  {s['filename']:>45}  →  {total_pages:>3d} code pages")

    # ---- Compute starting page index for each script's separator ----
    # Page 1 = cover, page 2 = TOC, page 3 = first separator
    # Each script occupies: 1 separator page + n_pages code pages
    page_index = []
    p = 3  # first separator
    for c_item in cache:
        page_index.append(p)            # the separator page (where the script "starts")
        p += 1                          # separator
        p += c_item["n_pages"]          # code pages

    # ---- Draw cover ----
    print("Drawing cover...")
    draw_cover(c)

    # ---- Draw TOC ----
    print("Drawing TOC...")
    draw_toc(c, page_index)

    # ---- Draw each script ----
    for i, c_item in enumerate(cache, start=1):
        print(f"Drawing script {i}/{len(cache)}: {c_item['meta']['filename']}")
        draw_separator(c, c_item["meta"], i)
        draw_code_pages(c, c_item["meta"], c_item["wrapped"])

    c.save()
    size_kb = os.path.getsize(OUT_PATH) / 1024
    print(f"\n✓ PDF created: {OUT_PATH}")
    print(f"  Size: {size_kb:.1f} KB")
    return OUT_PATH


if __name__ == "__main__":
    build_pdf()
