#!/usr/bin/env python3
"""
IMAGE RESIZER — Batch resize all images in this script's folder.
================================================================
Drop this script into any folder containing images and double-click.
All images are resized to fit within MAX_DIMENSION while preserving
aspect ratio. Originals are untouched; resized copies are saved
with a '_resized' suffix.

Usage:
  - Double-click this file in Windows Explorer
  - Or: python resize_images.py
  - Or: python resize_images.py 6000        (custom max dimension)
  - Or: python resize_images.py 4000 jpg    (custom max + output format)

Author: Utility for ET Audio Analysis output viewing
"""

import os
import sys
import time
import traceback

# ═══════════════════════════════════════════════════════════════════
#  CONFIGURATION — Edit these to taste
# ═══════════════════════════════════════════════════════════════════

MAX_DIMENSION = 7500      # Max width OR height in pixels (aspect preserved)
                          # 7500 maximizes detail within the 8000px viewing limit
OUTPUT_FORMAT = None       # None = keep original format; or 'png', 'jpg', etc.
JPEG_QUALITY  = 95         # Quality for JPEG output (1-100)
RESAMPLE      = 'LANCZOS' # Resampling filter: LANCZOS (best for downscaling)
SUFFIX        = '_resized' # Appended to filename before extension
SKIP_EXISTING = True       # Skip if resized file already exists
PNG_COMPRESS  = 6          # PNG compression level (0-9). Lossless; higher = smaller file, slower
IMAGE_EXTENSIONS = {
    '.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif',
    '.gif', '.webp', '.ico', '.ppm', '.pgm', '.pbm',
}

# ═══════════════════════════════════════════════════════════════════
#  METADATA PRESERVATION
# ═══════════════════════════════════════════════════════════════════

def _build_save_kwargs(out_ext, original_mode, icc_profile, dpi_info,
                       png_text_chunks, gamma, transparency):
    """
    Build save kwargs that preserve every piece of metadata the format supports.
    
    Preserves:
      - ICC color profile (PNG, JPEG, TIFF)
      - DPI / physical resolution (all formats)
      - PNG text chunks (Software, Author, Description, etc.)
      - Gamma (PNG)
      - Transparency (PNG with palette)
      - Bit depth via mode preservation (no silent conversion)
    """
    kwargs = {}
    ext_lower = out_ext.lower()

    if ext_lower in ('.jpg', '.jpeg'):
        kwargs['quality'] = JPEG_QUALITY
        kwargs['optimize'] = True
        kwargs['subsampling'] = 0  # 4:4:4 — no chroma subsampling for max quality
        if icc_profile:
            kwargs['icc_profile'] = icc_profile
        if dpi_info:
            kwargs['dpi'] = dpi_info

    elif ext_lower == '.png':
        kwargs['compress_level'] = PNG_COMPRESS
        # Build PngInfo with all text metadata
        from PIL.PngImagePlugin import PngInfo
        pnginfo = PngInfo()
        for key, val in png_text_chunks.items():
            try:
                pnginfo.add_text(key, str(val))
            except Exception:
                pass  # Skip unparseable chunks
        kwargs['pnginfo'] = pnginfo
        if icc_profile:
            kwargs['icc_profile'] = icc_profile
        if dpi_info:
            kwargs['dpi'] = dpi_info
        if gamma is not None:
            kwargs['gamma'] = gamma
        if transparency is not None:
            kwargs['transparency'] = transparency

    elif ext_lower in ('.tiff', '.tif'):
        kwargs['compression'] = 'tiff_lzw'  # Lossless compression
        if icc_profile:
            kwargs['icc_profile'] = icc_profile
        if dpi_info:
            kwargs['dpi'] = dpi_info

    elif ext_lower == '.webp':
        kwargs['quality'] = 100     # Lossless-equivalent
        kwargs['lossless'] = True
        if icc_profile:
            kwargs['icc_profile'] = icc_profile

    elif ext_lower == '.bmp':
        if dpi_info:
            kwargs['dpi'] = dpi_info

    return kwargs


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  IMAGE RESIZER — Batch resize all images in this folder")
    print("=" * 70)

    # ── Resolve script directory ──────────────────────────────────
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"\n  Folder: {script_dir}")

    # ── Parse optional CLI args ───────────────────────────────────
    max_dim = MAX_DIMENSION
    out_fmt = OUTPUT_FORMAT

    if len(sys.argv) >= 2:
        try:
            max_dim = int(sys.argv[1])
            print(f"  Max dimension override: {max_dim}")
        except ValueError:
            print(f"  Warning: '{sys.argv[1]}' is not a number, using default {max_dim}")

    if len(sys.argv) >= 3:
        out_fmt = sys.argv[2].lower().strip('.')
        print(f"  Output format override: {out_fmt}")

    # ── Import PIL (with decompression bomb disabled) ─────────────
    try:
        from PIL import Image, PngImagePlugin
        from PIL.PngImagePlugin import PngInfo
        try:
            from PIL import ImageCms
            has_cms = True
        except ImportError:
            has_cms = False
        Image.MAX_IMAGE_PIXELS = None  # Disable bomb check for known-safe files
        resample_filter = getattr(Image, RESAMPLE, Image.LANCZOS)
        print(f"  Pillow loaded — decompression bomb check disabled")
        print(f"  Resampling: {RESAMPLE}")
        print(f"  ICC profile support: {'YES' if has_cms else 'NO (ImageCms unavailable)'}")
        print(f"  Preservation: bit depth, ICC profile, DPI, PNG text metadata")
    except ImportError:
        print("\n  ERROR: Pillow is not installed.")
        print("  Install it with:  pip install Pillow")
        input("\n  Press Enter to exit...")
        return

    # ── Discover image files ──────────────────────────────────────
    all_files = []
    for entry in os.listdir(script_dir):
        full_path = os.path.join(script_dir, entry)
        if not os.path.isfile(full_path):
            continue
        name, ext = os.path.splitext(entry)
        if ext.lower() not in IMAGE_EXTENSIONS:
            continue
        # Skip files that are already resized
        if name.endswith(SUFFIX):
            continue
        all_files.append((full_path, name, ext))

    if not all_files:
        print(f"\n  No image files found in {script_dir}")
        input("\n  Press Enter to exit...")
        return

    all_files.sort(key=lambda x: x[1].lower())
    print(f"\n  Found {len(all_files)} image(s) to process:")
    for fp, name, ext in all_files:
        size_mb = os.path.getsize(fp) / (1024 * 1024)
        print(f"    {name}{ext}  ({size_mb:.1f} MB)")

    # ── Process each image ────────────────────────────────────────
    print(f"\n  Resizing to fit within {max_dim}×{max_dim} pixels...")
    print("-" * 70)

    success_count = 0
    skip_count = 0
    fail_count = 0

    for idx, (fp, name, ext) in enumerate(all_files, 1):
        # Determine output path
        if out_fmt is not None:
            out_ext = f".{out_fmt}"
        else:
            out_ext = ext
        out_name = f"{name}{SUFFIX}{out_ext}"
        out_path = os.path.join(script_dir, out_name)

        # Skip if already exists
        if SKIP_EXISTING and os.path.exists(out_path):
            print(f"  [{idx}/{len(all_files)}] SKIP (exists): {out_name}")
            skip_count += 1
            continue

        print(f"  [{idx}/{len(all_files)}] Processing: {name}{ext} ... ", end="", flush=True)
        t0 = time.time()

        try:
            img = Image.open(fp)
            w, h = img.size
            original_mode = img.mode

            # ── Extract all preservable metadata BEFORE any transforms ──

            # 1. ICC color profile
            icc_profile = img.info.get('icc_profile', None)

            # 2. DPI / physical resolution
            dpi_info = img.info.get('dpi', None)

            # 3. PNG text metadata (Software, Description, Author, etc.)
            png_text_chunks = {}
            if ext.lower() == '.png':
                for key, val in img.info.items():
                    if isinstance(val, str) and key not in ('icc_profile', 'dpi',
                                                            'gamma', 'transparency'):
                        png_text_chunks[key] = val
                # Also check for PngInfo text entries
                if hasattr(img, 'text'):
                    for key, val in img.text.items():
                        png_text_chunks[key] = val

            # 4. Transparency info (for palette/RGBA images)
            transparency = img.info.get('transparency', None)

            # 5. Gamma
            gamma = img.info.get('gamma', None)

            # Report what we found
            preserved = []
            if icc_profile:
                preserved.append('ICC')
            if dpi_info:
                preserved.append(f'DPI={dpi_info[0]:.0f}')
            if png_text_chunks:
                preserved.append(f'{len(png_text_chunks)} text chunks')
            if original_mode in ('I', 'I;16', 'F'):
                preserved.append(f'mode={original_mode}')
            preserve_str = f" [{', '.join(preserved)}]" if preserved else ""

            # Check if resize is needed
            if w <= max_dim and h <= max_dim:
                print(f"already {w}×{h} (within limit){preserve_str} — copying as-is")
                # Even for copy, preserve metadata
                save_kwargs = _build_save_kwargs(out_ext, original_mode, icc_profile,
                                                 dpi_info, png_text_chunks, gamma,
                                                 transparency)
                img.save(out_path, **save_kwargs)
                success_count += 1
                img.close()
                continue

            # Calculate new size preserving aspect ratio
            ratio = min(max_dim / w, max_dim / h)
            new_w = int(w * ratio)
            new_h = int(h * ratio)

            print(f"{w}×{h} → {new_w}×{new_h}{preserve_str} ... ", end="", flush=True)

            # ── Resize (preserving bit depth / mode) ─────────────────
            # PIL's resize works on the native mode — I;16 stays I;16,
            # RGBA stays RGBA, etc. No silent conversion.
            img_resized = img.resize((new_w, new_h), resample_filter)
            img.close()

            # ── Save with all preserved metadata ─────────────────────
            save_kwargs = _build_save_kwargs(out_ext, original_mode, icc_profile,
                                             dpi_info, png_text_chunks, gamma,
                                             transparency)

            # JPEG doesn't support alpha or 16-bit — convert only if forced to JPEG
            if out_ext.lower() in ('.jpg', '.jpeg'):
                if img_resized.mode in ('RGBA', 'LA', 'P'):
                    img_resized = img_resized.convert('RGB')
                elif img_resized.mode in ('I', 'I;16', 'F'):
                    img_resized = img_resized.convert('L')

            img_resized.save(out_path, **save_kwargs)
            img_resized.close()

            elapsed = time.time() - t0
            out_size_mb = os.path.getsize(out_path) / (1024 * 1024)
            print(f"done ({elapsed:.1f}s, {out_size_mb:.1f} MB)")
            success_count += 1

        except Exception as e:
            elapsed = time.time() - t0
            print(f"FAILED ({elapsed:.1f}s)")
            print(f"         Error: {e}")
            traceback.print_exc()
            fail_count += 1

    # ── Summary ───────────────────────────────────────────────────
    print("-" * 70)
    print(f"\n  DONE:")
    print(f"    Resized:  {success_count}")
    print(f"    Skipped:  {skip_count}")
    print(f"    Failed:   {fail_count}")
    print(f"    Output:   {script_dir}")

    input("\n  Press Enter to exit...")


if __name__ == '__main__':
    main()
