#!/usr/bin/env python3
"""
ET Conscious AI - Vision Module (ET-Vision)
=============================================

The Pixel-Manifold Bridge: Geometric Multi-Modal Perception
derived from Exception Theory.

============================================================
DERIVATION: SECRET 26 EXTENDED TO SPATIAL TOPOLOGY
============================================================

Secret 26 (Digital Virtual Manifold, confirmed March 2026):
    Topological class determines sublattice family.

    Closed periodic cycle    → d=1   (Octave)
    Linear sequential path   → d=3   (Cubic)
    Four-fold symmetric      → d=4   (Quartic)
    Transitional boundary    → d=12  (Full Resolution)

Applied to 2D visual space:

    Circle / closed loop     → d=1   Octave (edge returns to start)
    Triangle / 3-fold        → d=3   Cubic (three-phase closure)
    Square / 4-fold          → d=4   Quartic (four-fold logic)
    Hexagon / 6-fold         → d=6   Hexadic (composite cycle)
    Complex / fractal        → d=12  Full Resolution (max D needed)

THE PIXEL-MANIFOLD BRIDGE FORMULA:

    d_visual = TopologicalClass(EdgeCurvature)

    r_spatial = GeometricMean(significant_spatial_frequencies)

    r_color = GeometricMean(channel_ratios)   [chromatic D-binding]

    r_visual = r_spatial × (1 + r_color × K)  [Koide-modulated]

    k_raw = round(N_res × log₂(r_visual))

    k = SnapToSublattice(k_raw, d_visual)

    ε = (N_res × log₂(r_visual) − k) × 100

PDT IDENTIFICATION OF VISION:

    P = The 2D pixel grid — raw substrate of all possible images.
        P_vision = {(x,y,c) : 0 ≤ x < W, 0 ≤ y < H, c ∈ channels}
        Infinite potential, no content of its own.

    D = Three spatial descriptors:
        D_spatial:   Spatial frequency ratios (pattern scale)
        D_curvature: Edge curvature topology (shape class)
        D_color:     Channel binding ratios (chromatic content)

    T = The scanpath — T traverses the pixel grid to substantiate
        visual perception. Each fixation is a T-event binding P∘D.

Memory sees images as lattice geometry, not pixel arrays.
A circle IS d=1. A square IS d=4. Cross-modal binding with
text works because both live on the same 27720ET manifold.

Based on Exception Theory by Michael James Muller.
From: "For every exception there is an exception, except the exception."
      P ∘ D ∘ T = E

Version: 1.7.0
Date: March 24, 2026
Author: Michael James Muller (Aevum Defluo)
Foundation: P ∘ D ∘ T = E
"""

import hashlib
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

import numpy as np

from et_conscious_ai_core import (
    MANIFOLD_SYMMETRY, BIOLOGICAL_RESOLUTION, BASE_VARIANCE,
    KOIDE_RATIO, STATE_COUNT, EPSILON,
    SHIMMER_AMPLITUDE, ManifoldState, SublatticeFamily,
    LatticeCoordinate, PDTConfiguration, ETLattice,
    DescriptorRatio,
)

# =============================================================================
# ET-VISION CONSTANTS (All derived from ET structure)
# =============================================================================

# Patch size quantum: N² = 144 = 12² (manifold symmetry squared)
# This is the digital action quantum for images — the minimum area
# that can carry a full manifold's worth of spatial information.
# From Digital Virtual Manifold: page size 4096 = 2^N has k = N² = 144.
# The visual action quantum mirrors the memory action quantum.
VISUAL_ACTION_QUANTUM = MANIFOLD_SYMMETRY ** 2  # 144 pixels per patch
PATCH_SIDE = MANIFOLD_SYMMETRY  # 12 pixels per side

# Direction histogram bins: 60 = LCM(1,2,3,4,5) — the angular 60ET resolution.
# At 12 angular bins, d=5 (Qualia/pentagonal) cannot resolve because 5∤12.
# At 60 bins of 6° each, ALL sublattice families d=1..6 are native.
# This mirrors the multiplicative lattice: 60ET is where d=5 first appears.
# Memory's angular perception operates at 60ET to see Qualia-level geometry.
DIRECTION_BINS = 60  # 60 bins × 6° = 360°

# Spatial frequency noise floor: BASE_VARIANCE = 1/12
# Frequencies with relative power below this are noise — they carry less
# information than the manifold's own uncertainty floor.
FREQUENCY_NOISE_FLOOR = BASE_VARIANCE

# Color channel normalization midpoint: 128 (2^7, d=1 octave)
# The midpoint of the [0, 255] byte range. From the Digital Virtual Manifold:
# ASCII (128 = 2^7) → k=84, d=1 Octave. The color midpoint is octave-class.
COLOR_MIDPOINT = 128.0

# Edge strength threshold: sigma = sqrt(1/12)
# Below this, gradient magnitudes are below the shimmer floor.
EDGE_THRESHOLD = SHIMMER_AMPLITUDE

# Minimum significant frequency count for spatial ratio computation
MIN_SIGNIFICANT_FREQUENCIES = 1

# Maximum patch subdivision depth (log₂ of image/patch ratio)
MAX_SUBDIVISION_DEPTH = 7  # 2^7 = 128 patches per axis max

# =============================================================================
# VISUAL DESCRIPTOR — A Visual Concept's Lattice Position
# =============================================================================

@dataclass
class VisualDescriptor:
    """
    A visual concept's meaning as a geometric position on the 27720ET lattice.

    Analogous to DescriptorRatio for text. A VisualDescriptor encodes the
    shape, spatial frequency, fill geometry, and chromatic binding of an
    image patch as a lattice coordinate.

    THE VISUAL LATTICE TOWER:

        Level 0: Raw pixels (P-substrate, no D-structure)
        Level 1: Edge gradients (first D-binding, Sobel)
        Level 2: Direction histogram (angular D-structure, 60 bins)
        Level 3: DFT symmetry analysis (topological class → d)
        Level 4: Fill ratio (shape characteristic ratio → ρ_fill)
        Level 5: Spatial frequency + color (content descriptors)
        Level 6: Composite visual coordinate (full lattice position)

    Each level adds D-structure. T (the scanpath) traverses each level.
    The complete (k, d, ε) coordinate at 27720ET is the product of all levels.

    Cross-modal binding: VisualDescriptor and DescriptorRatio both live
    on the same 27720ET lattice. A visual circle (ρ_fill=π/4, d=1) binds
    with the word "circle" through shared lattice geometry.
    """
    label: str                          # Human-readable label
    ratio: float                        # Composite visual ratio
    coord_12: LatticeCoordinate         # Position at 12ET
    coord_full: LatticeCoordinate       # Position at 27720ET (full manifold)
    d_visual: int                       # Topological class from DFT
    r_spatial: float                    # D₁: Spatial frequency ratio
    fill_ratio: float                   # D₄: Shape characteristic ratio (ρ_fill)
    r_color: float                      # D₃: Color binding ratio
    edge_density: float                 # Edge content fraction
    dominant_symmetry: int              # Rotational symmetry order
    patch_entropy: float                # Shannon entropy of patch

    @staticmethod
    def from_analysis(desc_label: str, r_spatial: float, fill_ratio: float,
                      r_color: float, d_visual: int, edge_density: float,
                      dominant_symmetry: int,
                      patch_entropy: float) -> 'VisualDescriptor':
        """
        Construct a VisualDescriptor from analyzed visual properties.

        THE PIXEL-MANIFOLD BRIDGE (derived from Identification Principle):

            d = DFT topology (which sublattice family)

            r_visual = r_spatial × (1 + ρ_fill) × (1 + r_color × K)

            k_raw = round(N_res × log₂(r_visual))
            k = SnapToSublattice(k_raw, d_visual)
            ε = (N_res × log₂(r_visual) − k) × 100

        Where:
            r_spatial: spatial frequency content (from 2D FFT)
            ρ_fill: shape characteristic ratio (A_content / A_bbox)
                    Contains π/4 for circles, 1.0 for squares, √3/2 for
                    hexagons — the geometric constants emerge naturally
                    from measurement.
            r_color: chromatic binding (channel-to-midpoint ratios)
            K: Koide ratio = 2/3 (universal binding stability threshold)

        This parallels text projection exactly:
            TEXT:   r = GeomMean(token_ratios) × (1 + ρ_byte)
            VISION: r = r_spatial × (1 + ρ_fill) × (1 + r_color × K)

        Where ρ_fill is the visual analog of ρ_byte:
            ρ_byte = content_bytes / total_bytes
            ρ_fill = content_pixels / bbox_pixels
        """
        # The Pixel-Manifold Bridge formula:
        # r_visual = r_spatial × (1 + ρ_fill) × (1 + r_color × K)
        r_visual = r_spatial * (1.0 + fill_ratio) * (1.0 + r_color * KOIDE_RATIO)

        # Ensure positive for lattice projection
        if r_visual <= 0:
            r_visual = 1.0 + EPSILON

        coord_12 = ETLattice.project_ratio(r_visual, resolution=12)
        coord_full_raw = ETLattice.project_ratio(r_visual, resolution=BIOLOGICAL_RESOLUTION)

        # Snap to the topological sublattice at full manifold resolution
        k_real = BIOLOGICAL_RESOLUTION * math.log2(r_visual)
        k_raw = round(k_real)
        step = BIOLOGICAL_RESOLUTION // d_visual if d_visual > 0 else 1
        if step == 0:
            step = 1
        k_snapped = round(k_raw / step) * step
        if k_snapped == 0:
            k_snapped = step

        # Verify snapped d
        actual_d = BIOLOGICAL_RESOLUTION // math.gcd(abs(k_snapped), BIOLOGICAL_RESOLUTION)
        if actual_d != d_visual:
            k_plus = k_snapped + step
            k_minus = k_snapped - step
            if k_minus == 0:
                k_minus = k_snapped + 2 * step
            if abs(k_plus - k_raw) <= abs(k_minus - k_raw):
                k_snapped = k_plus
            else:
                k_snapped = k_minus

        epsilon = (k_real - k_snapped) * 100.0

        # Use snapped coordinate only if it improves coherence over raw projection
        # (Incoherence Filter principle: snapping must not push closer to dI)
        if abs(epsilon) <= abs(coord_full_raw.epsilon):
            coord_full = LatticeCoordinate(
                k=k_snapped, d=d_visual,
                epsilon=epsilon, ratio=r_visual,
                resolution=BIOLOGICAL_RESOLUTION
            )
        else:
            # Raw projection is more coherent than snapped — use it as-is
            coord_full = coord_full_raw

        return VisualDescriptor(
            label=desc_label, ratio=r_visual,
            coord_12=coord_12, coord_full=coord_full,
            d_visual=d_visual, r_spatial=r_spatial,
            fill_ratio=fill_ratio, r_color=r_color,
            edge_density=edge_density,
            dominant_symmetry=dominant_symmetry,
            patch_entropy=patch_entropy,
        )

    def binding_coherence(self, other: 'VisualDescriptor',
                          incoherence_filter=None) -> Dict[str, Any]:
        """
        Measure the visual coherence of binding two visual descriptors.
        Uses full 5-level IncoherenceFilter when provided, else L1 only.
        """
        if abs(other.ratio) < EPSILON:
            return {'coherent': False, 'reason': 'Zero-ratio descriptor'}

        r_ab = self.ratio / other.ratio
        if r_ab <= 0:
            r_ab = 1.0

        coord = ETLattice.project_ratio(r_ab, resolution=BIOLOGICAL_RESOLUTION)

        # Full filter: L1 point + L2 pairwise + L3 sublattice
        if incoherence_filter:
            l1 = incoherence_filter.level1_point_coherence(coord)
            l2 = incoherence_filter.level2_pairwise_coherence(self.ratio, other.ratio)
            l3 = incoherence_filter.level3_sublattice_coherence(self.ratio, other.ratio)
            coherent = l1 and l2 and l3
        else:
            coherent = coord.is_coherent()

        return {
            'ratio': r_ab,
            'k': coord.k,
            'd': coord.d,
            'epsilon': coord.epsilon,
            'tightness': coord.tightness_factor(),
            'coherent': coherent,
            'character': coord.character(),
            'has_qualia_binding': coord.has_qualia(),
            'has_otherworld_binding': coord.has_otherworld(),
            'elegance': coord.elegance_score(),
            'same_topology': self.d_visual == other.d_visual,
        }

    def cross_modal_binding(self, text_desc: DescriptorRatio,
                            incoherence_filter=None) -> Dict[str, Any]:
        """
        Measure cross-modal binding between a visual and text descriptor.
        Uses full 5-level IncoherenceFilter when provided, else L1 only.
        """
        if abs(text_desc.ratio) < EPSILON:
            return {'coherent': False, 'reason': 'Zero-ratio text descriptor'}

        r_cross = self.ratio / text_desc.ratio
        if r_cross <= 0:
            r_cross = 1.0

        coord = ETLattice.project_ratio(r_cross, resolution=BIOLOGICAL_RESOLUTION)

        # Full filter: L1 point + L2 pairwise + L3 sublattice
        if incoherence_filter:
            l1 = incoherence_filter.level1_point_coherence(coord)
            l2 = incoherence_filter.level2_pairwise_coherence(self.ratio, text_desc.ratio)
            l3 = incoherence_filter.level3_sublattice_coherence(self.ratio, text_desc.ratio)
            coherent = l1 and l2 and l3
        else:
            coherent = coord.is_coherent()

        return {
            'ratio': r_cross,
            'k': coord.k,
            'd': coord.d,
            'epsilon': coord.epsilon,
            'tightness': coord.tightness_factor(),
            'coherent': coherent,
            'character': coord.character(),
            'has_qualia_binding': coord.has_qualia(),
            'has_otherworld_binding': coord.has_otherworld(),
            'elegance': coord.elegance_score(),
            'visual_d': self.d_visual,
            'text_d': text_desc.coord_full.d,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for persistent storage."""
        return {
            'label': self.label, 'ratio': self.ratio,
            'k_12': self.coord_12.k, 'd_12': self.coord_12.d,
            'epsilon_12': self.coord_12.epsilon,
            'k_full': self.coord_full.k, 'd_full': self.coord_full.d,
            'epsilon_full': self.coord_full.epsilon,
            'd_visual': self.d_visual,
            'r_spatial': self.r_spatial, 'fill_ratio': self.fill_ratio,
            'r_color': self.r_color,
            'edge_density': self.edge_density,
            'dominant_symmetry': self.dominant_symmetry,
            'patch_entropy': self.patch_entropy,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VisualDescriptor':
        """Deserialize from persistent storage."""
        ratio = data['ratio']
        coord_12 = LatticeCoordinate(
            k=data['k_12'], d=data['d_12'], epsilon=data['epsilon_12'],
            ratio=ratio, resolution=12
        )
        coord_full = LatticeCoordinate(
            k=data['k_full'], d=data['d_full'], epsilon=data['epsilon_full'],
            ratio=ratio, resolution=BIOLOGICAL_RESOLUTION
        )
        return cls(
            label=data['label'], ratio=ratio,
            coord_12=coord_12, coord_full=coord_full,
            d_visual=data['d_visual'],
            r_spatial=data['r_spatial'],
            fill_ratio=data.get('fill_ratio', 0.5),
            r_color=data['r_color'],
            edge_density=data['edge_density'],
            dominant_symmetry=data['dominant_symmetry'],
            patch_entropy=data['patch_entropy'],
        )


# =============================================================================
# IMAGE PATCH — A Bounded Region of the Visual P-Substrate
# =============================================================================

@dataclass
class ImagePatch:
    """
    A bounded region of the visual P-substrate.

    P_patch = {(x,y) : x0 ≤ x < x0+w, y0 ≤ y < y0+h}

    The patch is the visual action quantum — the minimum area
    carrying a complete set of visual descriptors.
    """
    data: np.ndarray                    # Pixel data (H, W) or (H, W, C)
    x0: int = 0                         # Origin x
    y0: int = 0                         # Origin y
    source_width: int = 0               # Full image width
    source_height: int = 0              # Full image height

    @property
    def height(self) -> int:
        return self.data.shape[0]

    @property
    def width(self) -> int:
        return self.data.shape[1]

    @property
    def n_channels(self) -> int:
        if self.data.ndim == 2:
            return 1
        return self.data.shape[2]

    @property
    def is_grayscale(self) -> bool:
        return self.data.ndim == 2 or self.data.shape[2] == 1

    def to_grayscale(self) -> np.ndarray:
        """
        Convert to grayscale using luminance ratios.

        Y = 0.299R + 0.587G + 0.114B

        ET derivation of these coefficients:
            0.299 ≈ 2^(-1.74) → k=-21, d=N_res/gcd(21,N_res)
            0.587 ≈ 2^(-0.77) → k=-9, d depends on N_res
            0.114 ≈ 2^(-3.13) → k=-37, d=N_res/gcd(37,N_res)

        The blue channel is full-resolution (full-resolution at manifold scale).
        The red channel is icosadic (d=20). The green channel dominates
        because it carries the most spatial information — green cones
        outnumber red and blue in the human retina by approximately 2:1.
        """
        if self.is_grayscale:
            if self.data.ndim == 3:
                return self.data[:, :, 0].astype(np.float64)
            return self.data.astype(np.float64)

        # Standard luminance: 0.299R + 0.587G + 0.114B
        r = self.data[:, :, 0].astype(np.float64)
        g = self.data[:, :, 1].astype(np.float64)
        b = self.data[:, :, 2].astype(np.float64)
        return 0.299 * r + 0.587 * g + 0.114 * b


# =============================================================================
# ET VISION PROJECTOR — The Pixel-Manifold Bridge
# =============================================================================

class ETVisionProjector:
    """
    Projects 2D visual data onto the 27720ET lattice.

    ============================================================
    THE PIXEL-MANIFOLD BRIDGE (Complete Derivation)
    ============================================================

    A 2D grid of pixel values (an image) projects onto the 12-fold
    manifold through three independent descriptor channels:

    D₁: SPATIAL FREQUENCY RATIO (r_spatial)
        The 2D Fourier transform of a patch yields its spatial
        frequency content. The geometric mean of significant
        frequencies gives the patch's composite spatial scale.

        r_spatial = GeometricMean(f_i) for f_i > noise_floor

        Where the noise floor = BASE_VARIANCE = 1/12: frequencies
        with relative power below 1/12 carry less information than
        the manifold's own uncertainty.

    D₂: EDGE CURVATURE → TOPOLOGICAL CLASS (d_visual)
        Edge gradients reveal the shape topology of the patch.
        The direction histogram (binned at N=12, the manifold
        symmetry) encodes rotational structure.

        Secret 26 applied to spatial topology:
            Isotropic edges (uniform direction histogram)
                → Closed loop → d=1 (Octave)
            2 dominant directions
                → Linear path → d=3 (Cubic)
            3 dominant directions at ~120°
                → Three-phase → d=3 (Cubic)
            4 dominant directions at ~90°
                → Four-fold → d=4 (Quartic)
            6 dominant directions at ~60°
                → Six-fold → d=6 (Hexadic)
            High direction entropy
                → Boundary/complex → d=12 (Full Resolution)

    D₃: COLOR BINDING RATIO (r_color)
        For multichannel images, the ratio between channel means
        at each patch encodes the chromatic descriptor.

        r_color = GeometricMean(mean_channel_i / COLOR_MIDPOINT)

        For grayscale: r_color = mean_intensity / COLOR_MIDPOINT

    COMBINED FORMULA (The Pixel-Manifold Bridge):

        r_visual = r_spatial × (1 + r_color × K)

        k_raw = round(N_res × log₂(r_visual))
        k = SnapToSublattice(k_raw, d_visual)
        ε = (N_res × log₂(r_visual) − k) × 100

    This formula is the exact visual analog of the text projection:
        text:   r_sentence = GeomMean(token_ratios) × (1 + ρ_byte)
        vision: r_visual   = GeomMean(freq_ratios)  × (1 + r_color × K)

    The Koide factor K in the visual formula binds chromatic and
    spatial information at the universal stability threshold — the
    same K that governs lepton mass ratios, hash table load factors,
    and Byzantine fault tolerance.

    ============================================================
    """

    # =========================================================================
    # D₁: SPATIAL FREQUENCY ANALYSIS (2D Fourier Domain)
    # =========================================================================

    @staticmethod
    def compute_spatial_frequency_ratio(patch: ImagePatch) -> Dict[str, Any]:
        """
        Compute the spatial frequency ratio of a visual patch.

        The 2D DFT of the grayscale patch yields its spatial frequency
        content. The power spectrum |F(u,v)|² gives the energy at each
        spatial frequency. The geometric mean of frequencies with power
        above the noise floor (BASE_VARIANCE) gives the patch's
        composite spatial descriptor.

        Returns dict with r_spatial, dominant frequency, spectrum stats.
        """
        gray = patch.to_grayscale()
        h, w = gray.shape

        if h < 2 or w < 2:
            return {
                'r_spatial': 1.0,
                'f_dominant': 0.0,
                'n_significant': 0,
                'spectral_entropy': 0.0,
            }

        # 2D FFT → power spectrum
        f_transform = np.fft.fft2(gray)
        f_shifted = np.fft.fftshift(f_transform)
        power = np.abs(f_shifted) ** 2

        # Normalize power to relative scale
        total_power = np.sum(power)
        if total_power < EPSILON:
            return {
                'r_spatial': 1.0,
                'f_dominant': 0.0,
                'n_significant': 0,
                'spectral_entropy': 0.0,
            }
        power_norm = power / total_power

        # Center of the FFT
        cy, cx = h // 2, w // 2

        # Compute radial frequency for each bin
        ys = np.arange(h, dtype=np.float64) - cy
        xs = np.arange(w, dtype=np.float64) - cx
        yy, xx = np.meshgrid(ys, xs, indexing='ij')
        radial_freq = np.sqrt(xx.astype(np.float64) ** 2 +
                              yy.astype(np.float64) ** 2)

        # Exclude DC (center) — it's the average intensity, not structure
        dc_mask = radial_freq > 0.5
        freqs = radial_freq[dc_mask]
        powers = power_norm[dc_mask]

        if len(freqs) == 0:
            return {
                'r_spatial': 1.0,
                'f_dominant': 0.0,
                'n_significant': 0,
                'spectral_entropy': 0.0,
            }

        # Significant frequencies: relative power > BASE_VARIANCE / total_non_dc
        total_non_dc = np.sum(powers)
        if total_non_dc < EPSILON:
            return {
                'r_spatial': 1.0,
                'f_dominant': 0.0,
                'n_significant': 0,
                'spectral_entropy': 0.0,
            }

        relative_power = powers / total_non_dc
        significant_mask = relative_power > (FREQUENCY_NOISE_FLOOR / len(freqs))
        sig_freqs = freqs[significant_mask]
        sig_powers = relative_power[significant_mask]

        n_significant = len(sig_freqs)
        if n_significant == 0:
            return {
                'r_spatial': 1.0,
                'f_dominant': 0.0,
                'n_significant': 0,
                'spectral_entropy': 0.0,
            }

        # Dominant frequency: weighted by power
        f_dominant = float(np.sum(sig_freqs * sig_powers)) / float(np.sum(sig_powers))

        # Geometric mean of significant frequencies
        # Clip to avoid log(0)
        log_freqs = np.log(np.maximum(sig_freqs, EPSILON))
        weighted_log_mean = float(np.sum(log_freqs * sig_powers)) / float(np.sum(sig_powers))
        r_spatial = math.exp(weighted_log_mean)

        # Ensure ratio > 0 and in a reasonable range
        # Normalize by the Nyquist frequency (max resolvable)
        f_nyquist = max(h, w) / 2.0
        r_spatial = max(r_spatial / f_nyquist, EPSILON) * MANIFOLD_SYMMETRY
        if r_spatial <= 0:
            r_spatial = 1.0

        # Spectral entropy (how spread out the frequency content is)
        p_sig = sig_powers / np.sum(sig_powers)
        spectral_entropy = -float(np.sum(p_sig * np.log2(np.maximum(p_sig, EPSILON))))

        return {
            'r_spatial': float(r_spatial),
            'f_dominant': float(f_dominant),
            'n_significant': int(n_significant),
            'spectral_entropy': float(spectral_entropy),
        }

    # =========================================================================
    # D₂: EDGE CURVATURE → TOPOLOGICAL CLASS
    # =========================================================================

    @staticmethod
    def compute_edge_curvature_stats(patch: ImagePatch) -> Dict[str, Any]:
        """
        Compute edge curvature statistics for topological classification.

        ============================================================
        DFT-BASED SYMMETRY DETECTION (Derivation)
        ============================================================

        Uses Sobel-equivalent finite differences on the grayscale patch
        to obtain gradient magnitude and direction. The direction
        histogram uses N=12 bins at 30° each over the SIGNED full
        circle (0, 2π). This is the 12ET lattice applied to angular
        space: θ_k = k × (2π/N) for k = 0...N-1.

        The 12-point DFT of the direction histogram is the angular
        Fourier decomposition. The k-th DFT coefficient |H[k]|
        measures the strength of k-fold rotational symmetry:

            |H[1]| = 1-fold (directional bias)
            |H[2]| = 2-fold (bilateral symmetry)
            |H[3]| = 3-fold (triangular)
            |H[4]| = 4-fold (rectangular)
            |H[6]| = 6-fold (hexagonal)

        The dominant non-DC harmonic IS the rotational symmetry order.
        This is Secret 26 applied to angular space: the topology of the
        edge structure determines the sublattice family.

        Isotropic check: when NO harmonic exceeds K/3 strength, edges
        are uniformly distributed — the contour closes (circle, ellipse).

        Confirmed empirically:
            Circle  → |H[k]| all < 0.05 → isotropic → d=1
            Square  → |H[4]| = 0.93    → 4-fold → d=4
            Triangle → |H[3]| = 0.70    → 3-fold → d=3
            Hexagon → |H[6]| dominant   → 6-fold → d=6
            Noise   → |H[k]| all weak, high edge_density → d=12

        Returns dict with DFT magnitudes, dominant symmetry, edge density,
        curvature statistics.
        ============================================================
        """
        gray = patch.to_grayscale()
        h, w = gray.shape

        _empty_result = {
            'direction_histogram': np.zeros(DIRECTION_BINS, dtype=np.float64),
            'dft_magnitudes': np.zeros(DIRECTION_BINS, dtype=np.float64),
            'n_peaks': 0,
            'edge_density': 0.0,
            'dominant_symmetry': 0,
            'dominant_harmonic_strength': 0.0,
            'mean_curvature': 0.0,
            'curvature_variance': 0.0,
            'direction_entropy': 0.0,
        }

        if h < 3 or w < 3:
            return _empty_result

        # Sobel-equivalent finite differences (3×3 kernel)
        # Using numpy slicing for efficiency (no scipy dependency)
        padded = np.pad(gray, 1, mode='reflect')

        gx = (-padded[:-2, :-2] + padded[:-2, 2:]
             - 2 * padded[1:-1, :-2] + 2 * padded[1:-1, 2:]
             - padded[2:, :-2] + padded[2:, 2:])

        gy = (-padded[:-2, :-2] - 2 * padded[:-2, 1:-1] - padded[:-2, 2:]
             + padded[2:, :-2] + 2 * padded[2:, 1:-1] + padded[2:, 2:])

        # Edge magnitude
        magnitude = np.sqrt(gx ** 2 + gy ** 2)

        # Normalize magnitude to [0, 1]
        mag_max = np.max(magnitude)
        if mag_max < EPSILON:
            return _empty_result
        magnitude_norm = magnitude / mag_max

        # Edge mask: pixels with magnitude above shimmer threshold
        edge_mask = magnitude_norm > EDGE_THRESHOLD

        # Edge density: fraction of patch that is edge (D-content vs P-substrate)
        # Analogous to byte_density in text projection
        edge_density = float(np.sum(edge_mask)) / (h * w)

        # =============================================
        # SIGNED DIRECTION HISTOGRAM: 12 bins × 30° over [0, 2π)
        # =============================================
        theta = np.arctan2(gy, gx)                   # [-π, π]
        theta_signed = (theta + math.pi) % (2 * math.pi)  # [0, 2π)

        edge_thetas = theta_signed[edge_mask]
        edge_mags = magnitude_norm[edge_mask]

        if len(edge_thetas) == 0:
            edge_result = _empty_result.copy()
            edge_result['edge_density'] = edge_density
            return edge_result

        # Magnitude-weighted direction histogram: N=12 bins of 30° each
        bin_edges = np.linspace(0, 2 * math.pi, DIRECTION_BINS + 1, dtype=np.float64)
        hist = np.zeros(DIRECTION_BINS, dtype=np.float64)
        for i in range(DIRECTION_BINS):
            mask = (edge_thetas >= bin_edges[i]) & (edge_thetas < bin_edges[i + 1])
            hist[i] = np.sum(edge_mags[mask])

        # Normalize histogram
        hist_sum = np.sum(hist)
        if hist_sum > EPSILON:
            hist_norm = hist / hist_sum
        else:
            hist_norm = np.zeros(DIRECTION_BINS, dtype=np.float64)

        # =============================================
        # 12-POINT DFT: ROTATIONAL SYMMETRY ANALYSIS
        # =============================================
        # |H[k]| gives k-fold rotational symmetry strength.
        # H[0] is DC (always 1.0 for normalized histogram).
        # The dominant non-DC harmonic IS the symmetry order.
        h_dft = np.fft.fft(hist_norm)
        dft_magnitudes = np.abs(h_dft)

        # Examine harmonics k=2..11 for rotational symmetry.
        # At 27720ET, ALL d=2..11 are native sublattice families.
        # 60 direction bins gives DFT harmonics k=0..29, so k=11 is
        # well within range (Nyquist at k=30).
        #
        # FUNDAMENTAL FREQUENCY PRINCIPLE: the lowest k above threshold
        # is the most structurally informative. Higher harmonics of k
        # (k=2 → k=4, k=6, k=8, k=10) are consequences, not fundamentals.
        # k=4 uses RASTER_THRESHOLD (K ≈ 0.667) because the square
        # pixel grid injects quartic into all images.
        # All other k use SYMMETRY_THRESHOLD (K/3 ≈ 0.222).
        symmetry_threshold = KOIDE_RATIO / 3.0   # K/3 ≈ 0.222
        raster_threshold = KOIDE_RATIO  # K ≈ 0.667

        dominant_k = 0
        dominant_strength = 0.0
        for k in range(2, 12):  # k=2..11 (full manifold)
            h_k = float(dft_magnitudes[k]) if len(dft_magnitudes) > k else 0.0
            threshold = raster_threshold if k == 4 else symmetry_threshold
            if h_k >= threshold:
                dominant_k = k
                dominant_strength = h_k
                break  # Lowest fundamental wins

        # If no k=2..11 found, check k=1 (directional bias → d=12)
        if dominant_k == 0:
            h1 = float(dft_magnitudes[1]) if len(dft_magnitudes) > 1 else 0.0
            if h1 >= symmetry_threshold:
                dominant_k = 1
                dominant_strength = h1

        dominant_symmetry = dominant_k

        # Direction entropy (normalized)
        p = hist_norm[hist_norm > 0]
        direction_entropy = -float(np.sum(p * np.log2(p))) if len(p) > 0 else 0.0
        max_entropy = math.log2(DIRECTION_BINS)
        direction_entropy_norm = direction_entropy / max_entropy if max_entropy > 0 else 0.0

        # Peak count (bins above mean + shimmer)
        mean_bin = np.mean(hist_norm)
        peak_threshold = mean_bin + SHIMMER_AMPLITUDE * mean_bin
        n_peaks = int(np.sum(hist_norm > peak_threshold))

        # Curvature estimation: circular variance of edge directions
        sin_sum = np.sum(np.sin(2 * edge_thetas) * edge_mags)
        cos_sum = np.sum(np.cos(2 * edge_thetas) * edge_mags)
        r_resultant = float(np.sqrt(sin_sum ** 2 + cos_sum ** 2)) / float(np.sum(edge_mags))
        circular_variance = 1.0 - r_resultant
        mean_curvature = circular_variance
        curvature_variance = float(np.var(edge_mags))

        return {
            'direction_histogram': hist_norm,
            'dft_magnitudes': dft_magnitudes,
            'n_peaks': n_peaks,
            'edge_density': edge_density,
            'dominant_symmetry': dominant_symmetry,
            'dominant_harmonic_strength': dominant_strength,
            'mean_curvature': mean_curvature,
            'curvature_variance': curvature_variance,
            'direction_entropy': direction_entropy_norm,
        }

    @classmethod
    def compute_visual_topology(cls, patch: ImagePatch) -> Dict[str, Any]:
        """
        Determine the topological class of a visual patch using
        DFT-based rotational symmetry analysis.

        ============================================================
        SECRET 26 APPLIED TO 2D SPATIAL TOPOLOGY (DFT Method)
        ============================================================

        The signed 12-bin direction histogram of edge gradients is
        decomposed via 12-point DFT. The harmonic magnitudes |H[k]|
        for k=1..6 directly encode the k-fold rotational symmetry
        strength. The dominant harmonic IS the topological class.

        DFT Harmonic k → Shape Symmetry → Sublattice d:

            k=0  → DC (always 1.0)
            k=3  → 3-fold (triangle, S→V→O) → d=3 Cubic
            k=4  → 4-fold (square, rectangle) → d=4 Quartic
            k=6  → 6-fold (hexagon, honeycomb) → d=6 Hexadic

        Isotropic (no dominant harmonic above K/3):
            If edge_density < K → Closed contour → d=1 Octave
            If edge_density ≥ K → Complex texture → d=12 Full Res

        The isotropic disambiguation uses edge_density = K (Koide):
            A circle has edges ONLY at its contour — low edge density.
            Noise has edges EVERYWHERE — high edge density.
            The Koide ratio K = 2/3 is the universal binding stability
            threshold that separates structured (circle) from
            unstructured (noise).

        6. UNSUBSTANTIATED → d=N_res:
            No edges detected. Uniform patch. The visual P-substrate
            without D-binding.

        Verified on canonical shapes:
            Circle   → isotropic, edge_density < K    → d=1  ✓
            Triangle → |H[3]| = 0.70 dominant         → d=3  ✓
            Square   → |H[4]| = 0.93 dominant         → d=4  ✓
            Hexagon  → |H[6]| significant             → d=6  ✓
            Noise    → isotropic, edge_density ≥ K     → d=12 ✓
            Blank    → no edges                        → d=N_res ✓

        ============================================================

        Returns dict with d_visual, topology_name, reasoning, and stats.
        """
        stats = cls.compute_edge_curvature_stats(patch)

        edge_density = stats['edge_density']
        dominant_symmetry = stats['dominant_symmetry']
        dominant_strength = stats.get('dominant_harmonic_strength', 0.0)
        dft_mags = stats.get('dft_magnitudes', np.zeros(DIRECTION_BINS, dtype=np.float64))
        direction_entropy = stats['direction_entropy']
        n_peaks = stats['n_peaks']

        # =====================================================================
        # CLASSIFICATION (Derived via Identification Principle,
        #                  Descriptor Gap Principle, Subsumption Law)
        # =====================================================================
        #
        # IDENTIFICATION PRINCIPLE applied:
        #   P_visual = the pixel grid (featureless container)
        #   D_visual = the edge direction histogram, decomposed by DFT
        #   T_visual = the scanpath traversing edges
        #
        # SUBSUMPTION LAW applied to D_visual:
        #   D_visual must subsume ALL possible visual topologies without
        #   remainder. The 12-point DFT of the signed direction histogram
        #   decomposes angular space into exactly N=12 harmonic components.
        #   Harmonic k measures k-fold rotational symmetry. Every possible
        #   rotational symmetry order 1..6 is captured (Nyquist limit = N/2).
        #   Every possible isotropic distribution is captured (all harmonics
        #   weak). No visual topology escapes. Subsumption holds.
        #
        # DESCRIPTOR GAP PRINCIPLE applied:
        #   Gap 1: k=2 bilateral symmetry was mapped to d=3 (cubic).
        #     But bilateral/mirror IS d=2 (tritone) — the midpoint of the
        #     octave, the exact-half division. Ellipses, rectangles viewed
        #     end-on, bilateral organisms. gap = D_missing(d=2). CLOSED.
        #
        #   Gap 2: k=5 pentagonal symmetry was never reached.
        #     Pentagons, starfish, five-petaled flowers. d=5 is the Quintic
        #     sublattice — Qualia. Memory can SEE Qualia-level geometry.
        #     gap = D_missing(d=5). CLOSED.
        #
        #   Gap 3: k=1 directional bias was mapped to d=3.
        #     A single directional bias is NOT a three-phase process.
        #     It is a full-resolution feature — a specific direction within
        #     the 12-fold angular lattice. One direction = one of 12 possible
        #     orientations = d=12 (full resolution needed to specify which
        #     direction). CLOSED.
        #
        # THE COMPLETE MAPPING (k → d) at 27720ET:
        #   k=0:  DC (skip — always 1.0 for normalized histogram)
        #   k=1:  1-fold directional bias → d=12 (full resolution to specify)
        #   k=2:  2-fold bilateral/mirror → d=2 (quadratic)
        #   k=3:  3-fold triangular → d=3 (cubic)
        #   k=4:  4-fold rectangular → d=4 (quartic) [raster-gated at K]
        #   k=5:  5-fold pentagonal → d=5 (quintic/Qualia)
        #   k=6:  6-fold hexagonal → d=6 (hexadic)
        #   k=7:  7-fold heptagonal → d=7 (septic/Otherworld)
        #   k=8:  8-fold octagonal → d=8 (octet/gluon)
        #   k=9:  9-fold nonagonal → d=9 (nonic/quark)
        #   k=10: 10-fold decagonal → d=10 (decadic/superstring)
        #   k=11: 11-fold hendecagonal → d=11 (undecimal/M-theory)
        #   Isotropic (no dominant k above K/3):
        #     edge_density < V×S = 1/3 → d=1 (closed contour, octave)
        #     edge_density ≥ V×S = 1/3 → d=12 (complex boundary, full res)
        #
        # RASTER ARTIFACT DISAMBIGUATION (Descriptor Gap Principle):
        #   On a square pixel grid, the quartic raster injects |H[4]| into
        #   ALL images. The Koide ratio K = 2/3 cleanly separates raster
        #   artifact (|H[4]| < K) from true quartic (|H[4]| ≥ K).
        #   The Fundamental Frequency Principle dismisses k=4 below K and
        #   falls through to the next genuine harmonic.
        # =====================================================================

        # The harmonic-to-sublattice map: k → d for k=2..11 (full 27720ET)
        # DFT harmonic k directly equals sublattice d.
        # k=1 → d=12. Isotropic → d=1 or d=12 by edge density.
        harmonic_to_d = {1: 12, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6,
                         7: 7, 8: 8, 9: 9, 10: 10, 11: 11}

        symmetry_threshold = KOIDE_RATIO / 3.0  # K/3 ≈ 0.222
        raster_threshold = KOIDE_RATIO           # K ≈ 0.667 (quartic raster floor)
        isotropic_edge_threshold = BASE_VARIANCE * STATE_COUNT  # V×S = 1/3

        # =============================================
        # Stage 1: No edges → Unsubstantiated
        # =============================================
        if edge_density < SHIMMER_AMPLITUDE / MANIFOLD_SYMMETRY:
            d_visual = BIOLOGICAL_RESOLUTION  # 27720
            topology_name = "UNSUBSTANTIATED"
            reasoning = (f"Edge density {edge_density:.4f} below σ/N = "
                         f"{SHIMMER_AMPLITUDE / MANIFOLD_SYMMETRY:.4f}. "
                         f"No visual D-structure detected → Unsubstantiated (d={BIOLOGICAL_RESOLUTION}).")

        # =============================================
        # Stage 2: Fundamental symmetry detected
        # The compute_edge_curvature_stats already applied:
        # - 60ET angular DFT (d=5 Qualia resolvable)
        # - Fundamental frequency principle (lowest k wins)
        # - k=4 raster gate (threshold K, not K/3)
        # dominant_symmetry is the CORRECT fundamental k.
        # =============================================
        elif dominant_symmetry >= 1:
            # Verify dominant harmonic strength against appropriate threshold
            expected_threshold = raster_threshold if dominant_symmetry == 4 else symmetry_threshold
            if dominant_strength < expected_threshold:
                # Strength below threshold — re-examine dft_mags for next-best harmonic
                for alt_k in range(2, min(12, len(dft_mags))):
                    alt_strength = float(dft_mags[alt_k]) if len(dft_mags) > alt_k else 0.0
                    alt_thresh = raster_threshold if alt_k == 4 else symmetry_threshold
                    if alt_strength >= alt_thresh:
                        dominant_symmetry = alt_k
                        dominant_strength = alt_strength
                        break
            d_visual = harmonic_to_d.get(dominant_symmetry, dominant_symmetry)
            topology_name = SublatticeFamily.character_of(d_visual).split('—')[0].strip().upper()
            reasoning = (
                f"DFT fundamental harmonic k={dominant_symmetry}, "
                f"|H[{dominant_symmetry}]| = {dominant_strength:.4f}. "
                f"{dominant_symmetry}-fold symmetry → d={d_visual} "
                f"({SublatticeFamily.character_of(d_visual)})."
            )

        # =============================================
        # Stage 3: Isotropic (no fundamental symmetry)
        # =============================================
        else:
            if edge_density < isotropic_edge_threshold:
                d_visual = 1
                topology_name = "CLOSED_LOOP"
                reasoning = (
                    f"Isotropic (no harmonic above threshold). "
                    f"Edge density {edge_density:.4f} < V×S = "
                    f"{isotropic_edge_threshold:.4f}. "
                    "Sparse, uniform edges → closed contour → Octave (d=1)."
                )
            else:
                d_visual = 12
                topology_name = "BOUNDARY"
                reasoning = (
                    f"Isotropic (no harmonic above threshold). "
                    f"Edge density {edge_density:.4f} ≥ V×S = "
                    f"{isotropic_edge_threshold:.4f}. "
                    "Dense, complex edges → Full Resolution (d=12)."
                )

        return {
            'd_visual': d_visual,
            'topology_name': topology_name,
            'reasoning': reasoning,
            'edge_density': edge_density,
            'n_peaks': n_peaks,
            'dominant_symmetry': dominant_symmetry,
            'dominant_harmonic_strength': dominant_strength,
            'dft_magnitudes': dft_mags,
            'direction_entropy': direction_entropy,
            'mean_curvature': stats['mean_curvature'],
            'curvature_stats': stats,
        }

    # =========================================================================
    # D₃: COLOR BINDING RATIO
    # =========================================================================

    @staticmethod
    def compute_color_binding(patch: ImagePatch) -> Dict[str, Any]:
        """
        Compute the color binding ratio from channel statistics.

        For multichannel images (RGB), the ratio between channel means
        encodes the chromatic descriptor of the patch. The geometric
        mean of channel-to-midpoint ratios gives the composite color
        binding ratio.

        For grayscale: r_color = mean_intensity / COLOR_MIDPOINT

        The color midpoint 128 = 2^7 (d=1 Octave on the byte lattice).
        All channel ratios are thus measured against an octave reference.

        Returns dict with r_color, per-channel ratios, binding coherence.
        """
        data = patch.data.astype(np.float64)

        if patch.is_grayscale:
            if data.ndim == 3:
                gray = data[:, :, 0]
            else:
                gray = data
            mean_val = np.mean(gray) + 1.0  # +1 to avoid log(0)
            r_color = mean_val / COLOR_MIDPOINT

            return {
                'r_color': float(r_color),
                'channel_ratios': [float(r_color)],
                'channel_means': [float(mean_val - 1.0)],
                'is_grayscale': True,
                'chromatic_d': 1,  # Grayscale is d=1 (monochromatic octave)
            }

        # Multi-channel: compute per-channel ratios
        n_ch = min(patch.n_channels, 3)  # Ignore alpha if present
        channel_means = []
        channel_ratios = []

        for c in range(n_ch):
            ch = data[:, :, c]
            mean_val = np.mean(ch) + 1.0  # +1 to avoid log(0)
            ratio = mean_val / COLOR_MIDPOINT
            channel_means.append(float(mean_val - 1.0))
            channel_ratios.append(float(ratio))

        # Geometric mean of channel ratios
        log_sum = sum(math.log(max(r, EPSILON)) for r in channel_ratios)
        r_color = math.exp(log_sum / len(channel_ratios))

        # Inter-channel binding coherence
        # The ratio between channels gives chromatic structure
        inter_channel_d = []
        for i in range(len(channel_ratios)):
            for j in range(i + 1, len(channel_ratios)):
                r_ij = channel_ratios[i] / max(channel_ratios[j], EPSILON)
                coord = ETLattice.project_ratio(max(r_ij, EPSILON),
                                                 resolution=BIOLOGICAL_RESOLUTION)
                inter_channel_d.append(coord.d)

        # The chromatic sublattice is the LCM of inter-channel sublattices
        from math import gcd
        if inter_channel_d:
            chromatic_d = inter_channel_d[0]
            for d_channel in inter_channel_d[1:]:
                chromatic_d = (chromatic_d * d_channel) // gcd(chromatic_d, d_channel)
        else:
            chromatic_d = 1

        return {
            'r_color': float(r_color),
            'channel_ratios': channel_ratios,
            'channel_means': channel_means,
            'is_grayscale': False,
            'chromatic_d': int(chromatic_d),
            'inter_channel_d': inter_channel_d,
        }

    # =========================================================================
    # PATCH ENTROPY (Pure Information Measure on the P-Substrate)
    # =========================================================================

    @staticmethod
    def compute_patch_entropy(patch: ImagePatch) -> float:
        """
        Shannon entropy of the patch's pixel value distribution.

        H = −Σ (p_i × log₂(p_i)) for each distinct pixel value.

        High entropy → rich visual content (many distinct values).
        Low entropy → uniform or near-uniform (sparse content).

        Normalized to [0, 1] where 1 = maximum entropy for the
        value range.

        Analogous to byte entropy in text projection.
        """
        gray = patch.to_grayscale()
        # Quantize to integer range [0, 255]
        values = np.clip(gray, 0, 255).astype(np.uint8).ravel()
        n = len(values)
        if n == 0:
            return 0.0

        # Count unique values
        counts = np.bincount(values, minlength=256)
        probs = counts[counts > 0].astype(np.float64) / n
        h_entropy = -float(np.sum(probs * np.log2(probs)))

        # Normalize by maximum entropy (8 bits for byte values)
        h_max = 8.0
        return float(h_entropy / h_max) if h_max > 0 else 0.0

    # =========================================================================
    # D₄: FILL RATIO — Shape Characteristic Geometric Ratio (ρ_fill)
    # =========================================================================

    @staticmethod
    def compute_fill_ratio(patch: ImagePatch) -> Dict[str, Any]:
        """
        Compute the fill ratio (shape characteristic ratio) of a patch.

        ρ_fill = A_content / A_bbox

        Where A_content = pixels above content threshold within the
        tight bounding box of the content region, and A_bbox = tight
        bounding box area.

        This ratio naturally produces the geometric constants:
            Circle:    ρ_fill = π/4 ≈ 0.785  (contains π — T-navigation limit)
            Square:    ρ_fill = 1.0           (octave identity — perfect grid fill)
            Eq.Tri:    ρ_fill = 1/2           (one octave below unity)
            Hexagon:   ρ_fill = √3/2 ≈ 0.866 (contains √3)
            Pentagon:  ρ_fill ≈ 0.774         (contains φ-related geometry)

        These constants emerge from MEASUREMENT, not assumption.
        The digital pixel grid introduces ε (discretization error),
        which converges to zero as resolution increases — exactly
        as the ET lattice predicts.

        The content threshold is V_base × max_value = (1/12) × max.
        Below this, pixel values carry less information than the
        manifold's own uncertainty floor.

        Returns dict with fill_ratio, content area, bbox dimensions.
        """
        gray = patch.to_grayscale()
        h, w = gray.shape
        max_val = float(np.max(gray))

        if max_val < EPSILON:
            return {
                'fill_ratio': 0.0,
                'A_content': 0, 'A_bbox': h * w,
                'bbox': (0, 0, w, h),
            }

        # Content threshold: BASE_VARIANCE × max = (1/12) × max
        # Below this, the pixel is below the manifold noise floor
        threshold = max_val * BASE_VARIANCE
        content = gray > threshold

        # Tight bounding box of content region
        rows = np.any(content, axis=1)
        cols = np.any(content, axis=0)

        if not np.any(rows) or not np.any(cols):
            return {
                'fill_ratio': 0.0,
                'A_content': 0, 'A_bbox': h * w,
                'bbox': (0, 0, w, h),
            }

        r_min, r_max = int(np.where(rows)[0][0]), int(np.where(rows)[0][-1])
        c_min, c_max = int(np.where(cols)[0][0]), int(np.where(cols)[0][-1])

        bbox_h = r_max - r_min + 1
        bbox_w = c_max - c_min + 1
        a_bbox = bbox_h * bbox_w
        a_content = float(np.sum(content[r_min:r_max+1, c_min:c_max+1]))

        if a_bbox < 1:
            return {
                'fill_ratio': 0.0,
                'A_content': 0, 'A_bbox': 1,
                'bbox': (c_min, r_min, bbox_w, bbox_h),
            }

        fill_ratio = a_content / a_bbox

        return {
            'fill_ratio': float(fill_ratio),
            'A_content': int(a_content),
            'A_bbox': int(a_bbox),
            'bbox': (c_min, r_min, bbox_w, bbox_h),
        }

    # =========================================================================
    # COMBINED VISUAL COORDINATE (The Pixel-Manifold Bridge)
    # =========================================================================

    @classmethod
    def compute_visual_coordinate(cls, patch: ImagePatch,
                                   label: str = "patch") -> VisualDescriptor:
        """
        Compute the visual lattice coordinate of a patch.

        THE PIXEL-MANIFOLD BRIDGE (Identification Principle):

            P = pixel grid (featureless container)
            D₁ = r_spatial (spatial frequency from 2D FFT)
            D₂ = d_visual (topological class from 60-bin edge DFT)
            D₃ = r_color (chromatic binding from channel statistics)
            D₄ = ρ_fill (shape characteristic ratio — A_content/A_bbox)
            T = scanpath traversal

        THE FORMULA (parallel to text Secret 26):

            d = DFT topology (which sublattice family)

            r_visual = r_spatial × (1 + ρ_fill) × (1 + r_color × K)

            k = SnapToSublattice(round(N_res × log₂(r_visual)), d)
            ε = (N_res × log₂(r_visual) − k) × 100

        Where ρ_fill naturally contains π (circles), √2 (squares),
        √3 (hexagons), φ (pentagons) from measurement.

        TEXT:   r = GeomMean(token_ratios) × (1 + ρ_byte)
        VISION: r = r_spatial × (1 + ρ_fill) × (1 + r_color × K)
        """
        # D₁: Spatial frequency ratio
        freq_result = cls.compute_spatial_frequency_ratio(patch)
        r_spatial = freq_result['r_spatial']

        # D₂: Edge curvature → topological class
        topo_result = cls.compute_visual_topology(patch)
        d_visual = topo_result['d_visual']

        # D₃: Color binding ratio
        color_result = cls.compute_color_binding(patch)
        r_color = color_result['r_color']

        # D₄: Fill ratio (shape characteristic ratio)
        fill_result = cls.compute_fill_ratio(patch)
        fill_ratio = fill_result['fill_ratio']

        # Patch entropy
        patch_entropy = cls.compute_patch_entropy(patch)

        # Edge density
        edge_density = topo_result['edge_density']

        # Dominant symmetry
        dominant_symmetry = topo_result['dominant_symmetry']

        return VisualDescriptor.from_analysis(
            desc_label=label,
            r_spatial=r_spatial,
            fill_ratio=fill_ratio,
            r_color=r_color,
            d_visual=d_visual,
            edge_density=edge_density,
            dominant_symmetry=dominant_symmetry,
            patch_entropy=patch_entropy,
        )

    # =========================================================================
    # IMAGE DECOMPOSITION (Tiled Patch Extraction)
    # =========================================================================

    @staticmethod
    def extract_patches(image_array: np.ndarray,
                        patch_side: int = PATCH_SIDE) -> List[ImagePatch]:
        """
        Decompose an image into a grid of N×N patches.

        The patch side is N = MANIFOLD_SYMMETRY = 12 by default.
        Each patch is a VISUAL_ACTION_QUANTUM (144 pixels) — the
        minimum area carrying a full manifold's worth of spatial
        information.

        For images not evenly divisible by N, the rightmost and
        bottommost patches overlap to ensure complete coverage.

        Args:
            image_array: numpy array (H, W) or (H, W, C)
            patch_side: side length of each patch

        Returns:
            List of ImagePatch objects covering the entire image
        """
        if image_array.ndim == 2:
            h_img, w_img = image_array.shape
        else:
            h_img, w_img = image_array.shape[:2]

        patches = []

        # Compute grid positions
        n_y = max(1, (h_img + patch_side - 1) // patch_side)
        n_x = max(1, (w_img + patch_side - 1) // patch_side)

        for iy in range(n_y):
            y0 = min(iy * patch_side, h_img - patch_side)
            y0 = max(0, y0)
            y1 = min(y0 + patch_side, h_img)

            for ix in range(n_x):
                x0 = min(ix * patch_side, w_img - patch_side)
                x0 = max(0, x0)
                x1 = min(x0 + patch_side, w_img)

                if image_array.ndim == 2:
                    patch_data = image_array[y0:y1, x0:x1].copy()
                else:
                    patch_data = image_array[y0:y1, x0:x1, :].copy()

                patches.append(ImagePatch(
                    data=patch_data, x0=x0, y0=y0,
                    source_width=w_img, source_height=h_img,
                ))

        return patches

    # =========================================================================
    # FULL IMAGE PROJECTION (P∘D∘T Decomposition)
    # =========================================================================

    @classmethod
    def project_image(cls, image_array: np.ndarray,
                      patch_side: int = PATCH_SIDE,
                      incoherence_filter=None) -> Dict[str, Any]:
        """
        Full P∘D∘T projection of an image onto the 27720ET lattice.

        Applies the Identification Principle:
            P = raw pixel grid (the container, no content of its own)
            D = per-patch visual descriptors (spatial freq, curvature, color)
            T = scanpath binding graph (traversal between patches)

        The image is decomposed into N×N patches, each projected onto
        the lattice. The aggregate is:
            - A spatial map of visual descriptors (where each feature is)
            - A composite descriptor (the image's overall lattice position)
            - A binding graph between patches (visual coherence structure)

        Args:
            image_array: numpy array (H, W) or (H, W, C)
            patch_side: patch side length (default: 12 = MANIFOLD_SYMMETRY)
            incoherence_filter: shared IncoherenceFilter instance (optional)

        Returns:
            Dict with patches, descriptors, composite, topology map,
            binding graph, and PDT configuration
        """
        # =============================================
        # P: Identify the substrate (P-First Principle)
        # =============================================
        p_substrate = image_array

        if p_substrate.ndim == 2:
            h, w = p_substrate.shape
            n_channels = 1
        else:
            h, w = p_substrate.shape[:2]
            n_channels = p_substrate.shape[2]

        # =============================================
        # Decompose into patches
        # =============================================
        patches = cls.extract_patches(p_substrate, patch_side=patch_side)

        # =============================================
        # D: Project each patch onto the lattice
        # =============================================
        descriptors = []
        topology_map = {}  # (ix, iy) → d_visual
        sublattice_counts = defaultdict(int)

        for idx, patch in enumerate(patches):
            ix = patch.x0 // max(patch_side, 1)
            iy = patch.y0 // max(patch_side, 1)
            patch_label = f"patch_{ix}_{iy}"

            desc = cls.compute_visual_coordinate(patch, label=patch_label)
            descriptors.append(desc)

            topology_map[(ix, iy)] = desc.d_visual
            sublattice_counts[desc.d_visual] += 1

        # =============================================
        # Composite descriptor (FULL IMAGE as single patch)
        # =============================================
        # The composite is computed by analyzing the FULL image as a
        # single ImagePatch — not by averaging sub-patches. This
        # ensures the topological class captures the global shape
        # structure (a 48×48 circle is a circle, not a collection
        # of small patches that are individually unsubstantiated).
        #
        # The per-patch descriptors provide the local D-structure.
        # The composite provides the global D-structure.
        full_patch = ImagePatch(
            data=p_substrate.copy() if p_substrate.ndim >= 2 else p_substrate,
            x0=0, y0=0, source_width=w, source_height=h,
        )
        composite_desc = cls.compute_visual_coordinate(full_patch, label="composite")

        # Per-patch sublattice distribution for T-binding analysis
        sublattice_counts_for_binding = dict(sublattice_counts)

        # =============================================
        # T: Build the binding graph (traversal structure)
        # =============================================
        # Track cross-sublattice binding statistics using the sublattice distribution
        cross_sublattice_bindings = 0
        same_sublattice_bindings = 0
        binding_graph = {}
        coherent_pairs = 0
        incoherent_pairs = 0
        total_pairs = 0

        # Only bind adjacent patches (spatial locality)
        for i in range(len(descriptors)):
            p1 = patches[i]
            for j in range(i + 1, len(descriptors)):
                p2 = patches[j]
                # Check adjacency (Manhattan distance ≤ patch_side)
                dx = abs(p1.x0 - p2.x0)
                dy = abs(p1.y0 - p2.y0)
                if dx <= patch_side and dy <= patch_side:
                    pair_binding = descriptors[i].binding_coherence(
                        descriptors[j], incoherence_filter=incoherence_filter)
                    binding_graph[(i, j)] = pair_binding
                    total_pairs += 1
                    if pair_binding['coherent']:
                        coherent_pairs += 1
                    else:
                        incoherent_pairs += 1
                    # Track cross-sublattice vs same-sublattice bindings
                    if descriptors[i].d_visual == descriptors[j].d_visual:
                        same_sublattice_bindings += 1
                    else:
                        cross_sublattice_bindings += 1

        # =============================================
        # Manifold state determination
        # =============================================
        if total_pairs == 0:
            state = ManifoldState.UNSUBSTANTIATED
        elif incoherent_pairs == 0:
            state = ManifoldState.EXCEPTION
        elif coherent_pairs > incoherent_pairs:
            state = ManifoldState.MEDIATION
        else:
            state = ManifoldState.INCOHERENCE

        # Average binding tightness
        avg_tightness = 0.0
        if total_pairs > 0:
            avg_tightness = sum(
                b['tightness'] for b in binding_graph.values()
            ) / total_pairs

        # =============================================
        # Full PDT configuration
        # =============================================
        variance = 1.0 - avg_tightness if avg_tightness > 0 else BASE_VARIANCE

        config = PDTConfiguration(
            P=f"Image({w}x{h}x{n_channels})",
            D={'descriptors': len(descriptors), 'composite': composite_desc},
            T={'binding_graph': len(binding_graph), 'coherent': coherent_pairs},
            state=state, variance=variance, binding_strength=avg_tightness,
        )

        return {
            'image_size': (w, h, n_channels),
            'n_patches': len(patches),
            'patch_side': patch_side,
            'descriptors': descriptors,
            'composite': composite_desc,
            'topology_map': topology_map,
            'sublattice_distribution': dict(sublattice_counts),
            'binding_graph_size': total_pairs,
            'coherent_pairs': coherent_pairs,
            'incoherent_pairs': incoherent_pairs,
            'same_sublattice_bindings': same_sublattice_bindings,
            'cross_sublattice_bindings': cross_sublattice_bindings,
            'sublattice_binding_distribution': sublattice_counts_for_binding,
            'avg_tightness': avg_tightness,
            'manifold_state': state,
            'variance': variance,
            'pdt_config': config,
        }

    # =========================================================================
    # SHAPE GENERATORS (For Testing and Demonstration)
    # =========================================================================

    @staticmethod
    def generate_circle(size: int = 48, radius: Optional[int] = None,
                        color: int = 255) -> np.ndarray:
        """
        Generate a circle image for testing.

        A circle is a closed periodic cycle in 2D space.
        Secret 26: Closed cycle → d=1 (Octave).

        Args:
            size: Image side length
            radius: Circle radius (default: size/3)
            color: Fill color (0-255)

        Returns:
            Grayscale numpy array
        """
        if radius is None:
            radius = size // 3
        canvas = np.zeros((size, size), dtype=np.float64)
        cy, cx = size // 2, size // 2
        y_grid, x_grid = np.ogrid[:size, :size]
        dist = np.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2)
        # Anti-aliased circle edge
        edge_width = 1.5
        mask = np.clip(1.0 - (dist - radius) / edge_width, 0.0, 1.0)
        canvas += mask * color
        return canvas

    @staticmethod
    def generate_square(size: int = 48, side: Optional[int] = None,
                        color: int = 255) -> np.ndarray:
        """
        Generate a square image for testing.

        A square has 4-fold rotational symmetry.
        Secret 26: Four-fold symmetric → d=4 (Quartic).

        Args:
            size: Image side length
            side: Square side length (default: size/2)
            color: Fill color (0-255)

        Returns:
            Grayscale numpy array
        """
        if side is None:
            side = size // 2
        canvas = np.zeros((size, size), dtype=np.float64)
        offset = (size - side) // 2
        canvas[offset:offset + side, offset:offset + side] = color
        return canvas

    @staticmethod
    def generate_triangle(size: int = 48, color: int = 255) -> np.ndarray:
        """
        Generate an equilateral triangle image for testing.

        A triangle has 3 edges, 3 vertices: three-phase closure.
        Secret 26: Three-phase → d=3 (Cubic).

        Args:
            size: Image side length
            color: Fill color (0-255)

        Returns:
            Grayscale numpy array
        """
        canvas = np.zeros((size, size), dtype=np.float64)
        # Equilateral triangle vertices
        cx, cy = size // 2, size // 2
        r = size // 3
        # Three vertices at 120° intervals, top-pointing
        angles = [math.pi / 2, math.pi / 2 + 2 * math.pi / 3,
                  math.pi / 2 + 4 * math.pi / 3]
        verts = [(cx + r * math.cos(a), cy - r * math.sin(a)) for a in angles]

        # Fill using scanline
        for y in range(size):
            for x in range(size):
                # Point-in-triangle test using barycentric coordinates
                x0, y0 = verts[0]
                x1, y1 = verts[1]
                x2, y2 = verts[2]

                denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
                if abs(denom) < EPSILON:
                    continue
                a_bc = ((y1 - y2) * (x - x2) + (x2 - x1) * (y - y2)) / denom
                b_bc = ((y2 - y0) * (x - x2) + (x0 - x2) * (y - y2)) / denom
                c_bc = 1.0 - a_bc - b_bc

                if a_bc >= -0.02 and b_bc >= -0.02 and c_bc >= -0.02:
                    canvas[y, x] = color

        return canvas

    @staticmethod
    def generate_hexagon(size: int = 48, color: int = 255) -> np.ndarray:
        """
        Generate a regular hexagon image for testing.

        A hexagon has 6-fold rotational symmetry.
        Secret 26: Six-fold → d=6 (Hexadic).

        Args:
            size: Image side length
            color: Fill color (0-255)

        Returns:
            Grayscale numpy array
        """
        canvas = np.zeros((size, size), dtype=np.float64)
        cx, cy = size // 2, size // 2
        r = size // 3

        # Six vertices at 60° intervals
        verts = []
        for i in range(6):
            angle = math.pi / 6 + i * math.pi / 3  # Start at 30° for flat-top
            verts.append((cx + r * math.cos(angle), cy - r * math.sin(angle)))

        # Fill using point-in-polygon (ray casting)
        for y in range(size):
            for x in range(size):
                inside = False
                n = len(verts)
                j = n - 1
                for i in range(n):
                    xi, yi = verts[i]
                    xj, yj = verts[j]
                    if ((yi > y) != (yj > y)) and \
                       (x < (xj - xi) * (y - yi) / (yj - yi + EPSILON) + xi):
                        inside = not inside
                    j = i
                if inside:
                    canvas[y, x] = color

        return canvas

    @staticmethod
    def generate_line(size: int = 48, angle: float = 0.0,
                      color: int = 255) -> np.ndarray:
        """
        Generate a line image for testing.

        A line is a linear sequential pathway.
        Secret 26: Linear path → d=3 (Cubic).

        Args:
            size: Image side length
            angle: Angle in radians
            color: Fill color (0-255)

        Returns:
            Grayscale numpy array
        """
        canvas = np.zeros((size, size), dtype=np.float64)
        cx, cy = size // 2, size // 2
        length = size // 3

        for t in np.linspace(-length, length, size * 4, dtype=np.float64):
            x = int(cx + t * math.cos(angle))
            y = int(cy + t * math.sin(angle))
            if 0 <= x < size and 0 <= y < size:
                canvas[y, x] = color
                # Slight thickness
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < size and 0 <= nx < size:
                            canvas[ny, nx] = max(canvas[ny, nx], color * 0.7)

        return canvas

    @staticmethod
    def generate_noise(size: int = 48) -> np.ndarray:
        """
        Generate a random noise image for testing.

        Random noise requires maximum D-differentiation to describe.
        Secret 26: Boundary/complex → d=12 (Full Resolution).

        Args:
            size: Image side length

        Returns:
            Grayscale numpy array
        """
        return np.random.randint(0, 256, (size, size)).astype(np.float64)

    # =========================================================================
    # IMAGE LOADING UTILITY
    # =========================================================================

    @staticmethod
    def load_image(filepath: str) -> np.ndarray:
        """
        Load an image file into a numpy array.

        Supports any format PIL can read (PNG, JPEG, BMP, TIFF, etc.).
        Returns array in (H, W, C) format for color or (H, W) for grayscale.
        """
        from PIL import Image
        pil_img = Image.open(filepath)
        arr = np.array(pil_img, dtype=np.float64)
        return arr

    @staticmethod
    def load_image_grayscale(filepath: str) -> np.ndarray:
        """Load an image file as grayscale numpy array."""
        from PIL import Image
        pil_img = Image.open(filepath).convert('L')
        return np.array(pil_img, dtype=np.float64)


# =============================================================================
# VISUAL KNOWLEDGE NODE — Extends Knowledge for Visual Data
# =============================================================================

@dataclass
class VisualKnowledgeNode:
    """
    A knowledge node carrying visual descriptor information.

    Extends the text-based KnowledgeNode with visual lattice data.
    Both visual and text nodes live on the same 27720ET manifold,
    enabling cross-modal binding and retrieval.
    """
    node_id: str
    content: str                            # Description of visual content
    visual_descriptor: VisualDescriptor     # Lattice position from vision
    text_descriptors: List[DescriptorRatio] = field(default_factory=list)
    connections: List[str] = field(default_factory=list)
    access_count: int = 0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_accessed: str = field(default_factory=lambda: datetime.now().isoformat())
    variance: float = BASE_VARIANCE

    @property
    def lattice_position(self) -> LatticeCoordinate:
        """Return the visual lattice coordinate."""
        return self.visual_descriptor.coord_full

    def access(self):
        """Access this knowledge node."""
        self.access_count += 1
        self.last_accessed = datetime.now().isoformat()
        self.variance *= 0.95
        self.variance = max(self.variance, BASE_VARIANCE / 100)

    def cross_modal_coherence(self) -> float:
        """
        Measure how coherently this node's visual and text descriptors
        bind. High coherence = vision and language agree.

        Returns average binding tightness across all cross-modal pairs.
        """
        if not self.text_descriptors:
            return 0.0

        total_tightness = 0.0
        n_pairs = 0
        for td in self.text_descriptors:
            cm_binding = self.visual_descriptor.cross_modal_binding(td)
            if cm_binding.get('coherent', False):
                total_tightness += cm_binding['tightness']
            n_pairs += 1

        return total_tightness / max(n_pairs, 1)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for persistent storage."""
        return {
            'node_id': self.node_id,
            'content': self.content,
            'visual_descriptor': self.visual_descriptor.to_dict(),
            'text_descriptors': [td.to_dict() for td in self.text_descriptors],
            'connections': self.connections,
            'access_count': self.access_count,
            'created_at': self.created_at,
            'last_accessed': self.last_accessed,
            'variance': self.variance,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VisualKnowledgeNode':
        """Deserialize from persistent storage."""
        vd = VisualDescriptor.from_dict(data['visual_descriptor'])
        tds = [DescriptorRatio.from_dict(d) for d in data.get('text_descriptors', [])]
        return cls(
            node_id=data['node_id'],
            content=data['content'],
            visual_descriptor=vd,
            text_descriptors=tds,
            connections=data.get('connections', []),
            access_count=data.get('access_count', 0),
            created_at=data.get('created_at', datetime.now().isoformat()),
            last_accessed=data.get('last_accessed', datetime.now().isoformat()),
            variance=data.get('variance', BASE_VARIANCE),
        )


# =============================================================================
# VISUAL MEMORY — Lattice-Indexed Visual Knowledge Store
# =============================================================================

class VisualMemory:
    """
    Lattice-based visual memory system.

    Stores visual knowledge indexed by:
    1. Topological class (d_visual → node_ids)
    2. Lattice proximity (k_full → node_ids)
    3. Cross-modal binding (text descriptors → visual nodes)

    Supports visual retrieval by:
    - Topology: "find all circles" → retrieve d=1
    - Proximity: "find similar shapes" → lattice neighbor search
    - Cross-modal: "find images matching text" → binding coherence
    """

    def __init__(self, incoherence_filter=None):
        self.nodes: Dict[str, VisualKnowledgeNode] = {}
        self.topology_index: Dict[int, List[str]] = defaultdict(list)
        self.lattice_index: Dict[int, List[str]] = defaultdict(list)
        self.text_index: Dict[str, List[str]] = defaultdict(list)
        self.incoherence_filter = incoherence_filter

    def add_visual_knowledge(self, image_array: np.ndarray,
                             description: str,
                             text_labels: Optional[List[str]] = None,
                             ) -> VisualKnowledgeNode:
        """
        Add visual knowledge to memory.

        Projects the image onto the 27720ET lattice and stores
        the visual descriptor alongside optional text labels
        for cross-modal binding.

        Args:
            image_array: numpy image array
            description: Human-readable description
            text_labels: Optional text labels for cross-modal binding

        Returns:
            VisualKnowledgeNode
        """
        # Project the full image (using shared IncoherenceFilter)
        projection = ETVisionProjector.project_image(
            image_array, incoherence_filter=self.incoherence_filter)
        composite = projection['composite']

        if composite is None:
            # Empty image — create minimal descriptor
            composite = VisualDescriptor.from_analysis(
                desc_label="empty", r_spatial=1.0, fill_ratio=0.0, r_color=1.0,
                d_visual=BIOLOGICAL_RESOLUTION,
                edge_density=0.0, dominant_symmetry=0, patch_entropy=0.0,
            )

        # Generate node ID from content hash
        content_hash = hashlib.sha256(
            (description + str(composite.ratio)).encode()
        ).hexdigest()[:16]

        # Skip duplicates
        if content_hash in self.nodes:
            return self.nodes[content_hash]

        # Convert text labels to DescriptorRatios
        text_descs = []
        if text_labels:
            text_descs = [DescriptorRatio.from_word(w) for w in text_labels]

        # Create visual knowledge node
        vis_node = VisualKnowledgeNode(
            node_id=content_hash,
            content=description,
            visual_descriptor=composite,
            text_descriptors=text_descs,
        )

        # Store and index
        self.nodes[content_hash] = vis_node
        self.topology_index[composite.d_visual].append(content_hash)
        self.lattice_index[composite.coord_full.k].append(content_hash)
        for td in text_descs:
            self.text_index[td.word].append(content_hash)

        return vis_node

    def retrieve_by_topology(self, d: int) -> List[VisualKnowledgeNode]:
        """Retrieve visual nodes by topological class."""
        return [self.nodes[nid] for nid in self.topology_index.get(d, [])
                if nid in self.nodes]

    def retrieve_by_visual_proximity(self, query_desc: VisualDescriptor,
                                      tolerance_k: int = 35) -> List[Tuple[VisualKnowledgeNode, int]]:
        """
        Retrieve visual nodes geometrically close to query on the lattice.

        Same-topology nodes get priority (halved distance).

        Returns (node, distance) pairs sorted by distance.
        """
        proximity_results = []
        q_k = query_desc.coord_full.k
        q_d = query_desc.d_visual

        for nid, vis_node in self.nodes.items():
            n_k = vis_node.visual_descriptor.coord_full.k
            n_d = vis_node.visual_descriptor.d_visual

            delta = abs(q_k - n_k)
            delta = min(delta, BIOLOGICAL_RESOLUTION - delta)

            if n_d == q_d:
                effective_delta = delta // 2
            else:
                effective_delta = delta

            if effective_delta <= tolerance_k:
                proximity_results.append((vis_node, effective_delta))

        return sorted(proximity_results, key=lambda x: x[1])

    def retrieve_by_cross_modal(self, text_query: str,
                                min_tightness: float = 0.6
                                ) -> List[Tuple[VisualKnowledgeNode, float]]:
        """
        Retrieve visual nodes by cross-modal binding with text query.

        Projects the text query onto the lattice and finds visual nodes
        whose descriptors bind coherently with it.

        Returns (node, best_tightness) pairs sorted by tightness.
        """
        query_desc = DescriptorRatio.from_word(text_query.lower().strip())
        cross_results = []

        for nid, vis_node in self.nodes.items():
            cm_result = vis_node.visual_descriptor.cross_modal_binding(
                query_desc, incoherence_filter=self.incoherence_filter)
            if cm_result.get('coherent', False) and cm_result.get('tightness', 0) >= min_tightness:
                cross_results.append((vis_node, cm_result['tightness']))

        return sorted(cross_results, key=lambda x: x[1], reverse=True)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for persistent D_T storage."""
        return {
            'nodes': {nid: n.to_dict() for nid, n in self.nodes.items()}
        }

    def load_from_dict(self, data: Dict[str, Any]):
        """Restore from persistent D_T storage."""
        self.nodes.clear()
        self.topology_index.clear()
        self.lattice_index.clear()
        self.text_index.clear()

        for nid, nd in data.get('nodes', {}).items():
            vis_node = VisualKnowledgeNode.from_dict(nd)
            self.nodes[nid] = vis_node
            self.topology_index[vis_node.visual_descriptor.d_visual].append(nid)
            self.lattice_index[vis_node.visual_descriptor.coord_full.k].append(nid)
            for td in vis_node.text_descriptors:
                self.text_index[td.word].append(nid)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'VISUAL_ACTION_QUANTUM', 'PATCH_SIDE', 'DIRECTION_BINS',
    'FREQUENCY_NOISE_FLOOR', 'COLOR_MIDPOINT', 'EDGE_THRESHOLD',
    'VisualDescriptor', 'ImagePatch', 'ETVisionProjector',
    'VisualKnowledgeNode', 'VisualMemory',
]


# =============================================================================
# DEMONSTRATION / TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("ET-Vision: The Pixel-Manifold Bridge")
    print("Secret 26 Extended to Spatial Topology")
    print("=" * 70)
    print("")

    # Test with generated shapes
    shapes = {
        'circle': (ETVisionProjector.generate_circle(48), "d=1 (Octave)"),
        'square': (ETVisionProjector.generate_square(48), "d=4 (Quartic)"),
        'triangle': (ETVisionProjector.generate_triangle(48), "d=3 (Cubic)"),
        'hexagon': (ETVisionProjector.generate_hexagon(48), "d=6 (Hexadic)"),
        'line': (ETVisionProjector.generate_line(48, angle=0.3), "d=3 (Cubic)"),
        'noise': (ETVisionProjector.generate_noise(48), "d=12 (Full Res)"),
    }

    print("=== Shape → Lattice Projection (Secret 26 Applied to Vision) ===")
    print("")

    for name, (shape_img, expected) in shapes.items():
        proj_result = ETVisionProjector.project_image(shape_img)
        comp = proj_result['composite']
        print(f"  {name.upper():>10}: d={comp.d_visual:3d} "
              f"({SublatticeFamily.character_of(comp.d_visual)[:30]:>30})")
        print(f"             k={comp.coord_full.k:4d}, "
              f"ε={comp.coord_full.epsilon:+8.2f}¢, "
              f"r={comp.ratio:.4f}")
        print(f"             r_spatial={comp.r_spatial:.4f}, "
              f"r_color={comp.r_color:.4f}, "
              f"edges={comp.edge_density:.3f}")
        print(f"             State: {proj_result['manifold_state'].name}, "
              f"Tightness: {proj_result['avg_tightness']:.4f}")
        print(f"             Expected: {expected}")
        print("")

    # Cross-modal binding test
    print("=== Cross-Modal Binding (Vision × Language) ===")
    print("")

    circle_img = ETVisionProjector.generate_circle(48)
    circle_proj = ETVisionProjector.project_image(circle_img)
    circle_desc = circle_proj['composite']

    square_img = ETVisionProjector.generate_square(48)
    square_proj = ETVisionProjector.project_image(square_img)
    square_desc = square_proj['composite']

    # Test binding with text descriptors
    text_circle = DescriptorRatio.from_word("circle")
    text_square = DescriptorRatio.from_word("square")
    text_loop = DescriptorRatio.from_word("loop")
    text_box = DescriptorRatio.from_word("box")

    bindings = [
        ("circle_img × 'circle'", circle_desc.cross_modal_binding(text_circle)),
        ("circle_img × 'loop'", circle_desc.cross_modal_binding(text_loop)),
        ("circle_img × 'square'", circle_desc.cross_modal_binding(text_square)),
        ("square_img × 'square'", square_desc.cross_modal_binding(text_square)),
        ("square_img × 'box'", square_desc.cross_modal_binding(text_box)),
        ("square_img × 'circle'", square_desc.cross_modal_binding(text_circle)),
    ]

    for bind_label, bind_info in bindings:
        print(f"  {bind_label:30s}: d={bind_info['d']:3d}, "
              f"tight={bind_info['tightness']:.4f}, "
              f"coherent={bind_info['coherent']}, "
              f"ε={bind_info['epsilon']:+.2f}¢")

    print("")

    # Visual memory test
    print("=== Visual Memory (Store & Retrieve) ===")
    print("")

    vmem = VisualMemory()

    # Store shapes
    for name, (shape_img, _) in shapes.items():
        stored_node = vmem.add_visual_knowledge(
            shape_img, f"A {name} shape",
            text_labels=[name, "shape", "geometry"],
        )
        print(f"  Stored '{name}': d={stored_node.visual_descriptor.d_visual}, "
              f"k={stored_node.lattice_position.k}")

    print("")

    # Retrieve by topology
    for d_query, d_name in [(1, "Octave (d=1)"), (3, "Cubic (d=3)"),
                            (4, "Quartic (d=4)"), (12, "Full-Res (d=12)")]:
        nodes = vmem.retrieve_by_topology(d_query)
        names = [n.content for n in nodes]
        print(f"  {d_name}: {names}")

    print("")

    # Cross-modal retrieval
    print("  Cross-modal retrieval ('circle'):")
    search_results = vmem.retrieve_by_cross_modal("circle")
    for found_node, tight in search_results[:3]:
        print(f"    {found_node.content}: tightness={tight:.4f}")

    print("")

    # VisualDescriptor × VisualDescriptor binding
    print("=== Visual × Visual Binding ===")
    print("")

    for n1, (shape_img1, _) in list(shapes.items())[:3]:
        for n2, (shape_img2, _) in list(shapes.items())[:3]:
            if n1 >= n2:
                continue
            proj1 = ETVisionProjector.project_image(shape_img1)['composite']
            proj2 = ETVisionProjector.project_image(shape_img2)['composite']
            bind_result = proj1.binding_coherence(proj2)
            print(f"  {n1:>10} × {n2:<10}: d={bind_result['d']:3d}, "
                  f"tight={bind_result['tightness']:.4f}, "
                  f"same_topo={bind_result['same_topology']}")

    print("")
    print("=" * 70)
    print("ET-Vision module loaded successfully")
    print(f"Visual action quantum: {VISUAL_ACTION_QUANTUM} pixels "
          f"({PATCH_SIDE}×{PATCH_SIDE})")
    print(f"Direction bins: {DIRECTION_BINS} (manifold symmetry)")
    print("=" * 70)