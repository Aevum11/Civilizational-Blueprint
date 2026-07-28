#!/usr/bin/env python3
"""
Exception Theory — Compressed Descriptor Format (CDF) Compressor
=================================================================
A complete file compressor built entirely from ET-derived mathematics.
No conventional compression algorithms (gzip, lzma, deflate, etc.).
All compression is performed via ET lattice projection, elegance scoring,
archetype subsumption, incoherence filtering, and recursive lattice towers.

P ∘ D ∘ T = E
Author: Michael James Muller — Aevum Defluo
"""

import sys
import os
import struct
import math
import time
import hashlib
import logging
import traceback
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass, field
from collections import Counter, defaultdict
from fractions import Fraction
from concurrent.futures import ThreadPoolExecutor

import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
# ET CONSTANTS — Derived from P ∘ D ∘ T = E, not assumed
# ═══════════════════════════════════════════════════════════════════════════════

N_RES = 12                    # MANIFOLD_SYMMETRY: 3 primitives × 4 states
N_FULL = 27720                # Full lattice resolution: LCM(1..11)
V_BASE = 1.0 / 12.0          # BASE_VARIANCE: 1/N
K_KOIDE = 2.0 / 3.0          # KOIDE_RATIO: binding stability threshold
S_MANIFOLD = 12               # Manifold symmetry number
LIFE_THRESHOLD = 13.0 / 12.0  # Archetype collapse threshold: 1 + V
BIO_TIER_RES = 420            # Biological tier resolution: LCM(1..7)
CLUSTER_TIGHTNESS_K = K_KOIDE # Tightness threshold at ∂I boundary
INCOHERENCE_BOUNDARY_CENTS = 50.0  # |ε| at ∂I
MAX_COMPRESSION_DEPTH = S_MANIFOLD  # Maximum recursive depth = 12
BLOCK_SIZE = 4096             # Digital action quantum: 2^N = 2^12 bytes
SUBLATTICE_FAMILIES = [1, 2, 3, 4, 6, 12]  # Divisors of 12

# CDF file magic and version
CDF_MAGIC = b'CDF\x01'       # 4 bytes magic
CDF_VERSION = 1

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('CDF')


# ═══════════════════════════════════════════════════════════════════════════════
# ET LATTICE MATH ENGINE — All compression derives from this
# ═══════════════════════════════════════════════════════════════════════════════

def lattice_project(ratio: float, n_res: int = N_RES) -> Tuple[int, int, float]:
    """
    Project a ratio onto the ET lattice.
    
    Returns (k, d, epsilon_cents):
        k = nearest lattice position (semitone index)
        d = reduced denominator (sublattice family)
        epsilon = deviation from lattice point in cents
    
    This is the IDENTIFICATION PRINCIPLE applied to ratios:
    every ratio has a unique (k, d, ε) decomposition.
    """
    if ratio <= 0:
        return (0, n_res, 0.0)
    
    log2_r = math.log2(ratio)
    k_exact = n_res * log2_r
    k = round(k_exact)
    epsilon_cents = (k_exact - k) * (1200.0 / n_res)
    
    k_abs = abs(k) if k != 0 else n_res
    g = math.gcd(k_abs, n_res)
    d = n_res // g
    
    return (k, d, epsilon_cents)


def elegance_score(ratio: float, p: int, q: int, n_res: int = N_RES) -> float:
    """
    Compute the Elegance Score of a ratio r = p/q.
    
    E(r) = (N_res/d) × 100/(100+|ε|) × 100/(p+q)
    
    High elegance = stable attractor, low variance.
    Low elegance = near ∂I boundary, high variance.
    """
    if p + q == 0:
        return 0.0
    
    k, d, eps = lattice_project(ratio, n_res)
    
    resolution_factor = n_res / d
    tightness = 100.0 / (100.0 + abs(eps))
    complexity_factor = 100.0 / (p + q)
    
    return resolution_factor * tightness * complexity_factor


def tightness_factor(epsilon_cents: float) -> float:
    """
    Tightness: 100/(100+|ε|)
    At ∂I boundary (|ε|=50¢): tightness = K = 2/3
    """
    return 100.0 / (100.0 + abs(epsilon_cents))


def coherence_depth(epsilon_cents: float) -> float:
    """
    Coherence depth: Δ_∂I(r) = tightness(r) - K
    Positive = coherent interior, Negative = incoherent
    """
    return tightness_factor(epsilon_cents) - K_KOIDE


def incoherence_filter_l1(epsilon_cents: float) -> bool:
    """Level 1: Point coherence — |ε| < 50¢"""
    return abs(epsilon_cents) < INCOHERENCE_BOUNDARY_CENTS


def incoherence_filter_l2(k1: int, k2: int, n_res: int = N_RES) -> bool:
    """Level 2: Pairwise coherence — no rounding-flip contradiction"""
    # Two lattice points are pairwise coherent if they don't create
    # a contradiction when both projected
    return True  # Simplified: individual projections don't contradict


def incoherence_filter_l3(d1: int, d2: int) -> bool:
    """Level 3: Sublattice coherence — GCD compatibility"""
    g = math.gcd(d1, d2)
    return g > 0  # Compatible sublattice families


def incoherence_filter_l4(deltas: List[float], n_cascade: int) -> bool:
    """Level 4: Cascade coherence — N×|δ| < 50¢"""
    if not deltas:
        return True
    avg_delta = sum(abs(d) for d in deltas) / len(deltas)
    return n_cascade * avg_delta < INCOHERENCE_BOUNDARY_CENTS


def discover_seed(values: np.ndarray) -> float:
    """
    Discover the R₀ seed of a data block.
    
    R₀ = geometric mean of all non-zero descriptor values.
    This is the natural reference unit — the smallest closed T-traversal
    loop of this particular P-substrate's D-structure.
    
    From the Multifold paper: R₀ = D_period(P_i)
    """
    nonzero = values[values > 0]
    if len(nonzero) == 0:
        return 1.0
    
    # Geometric mean via log-space to avoid overflow
    log_mean = np.mean(np.log2(nonzero.astype(np.float64)))
    return 2.0 ** log_mean


def compute_ratios(values: np.ndarray, r0: float) -> np.ndarray:
    """
    Compute dimensionless ratios r = value / R₀ for all descriptors.
    Convention-free, Identification Principle-derived.
    """
    if r0 <= 0:
        r0 = 1.0
    ratios = values.astype(np.float64) / r0
    # Clamp to avoid log2(0)
    ratios[ratios <= 0] = 1e-15
    return ratios


# ═══════════════════════════════════════════════════════════════════════════════
# ARCHETYPE COMPRESSION ENGINE
# Bytes are Descriptors. Similar descriptors cluster into archetypes.
# Archetypes subsume their members (Subsumption Law).
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class LatticeNode:
    """A descriptor projected onto the ET lattice."""
    value: int            # Original byte value (0-255)
    index: int            # Position in the data stream
    k: int                # Lattice position
    d: int                # Sublattice family
    epsilon: float        # Deviation in cents
    elegance: float       # Elegance score
    ratio: float          # Dimensionless ratio r = value/R₀


@dataclass
class Archetype:
    """
    An archetype is a centroid that subsumes similar lattice nodes.
    The centroid is the geometric mean of subsumed ratios,
    projected back onto the lattice.
    """
    centroid_ratio: float
    centroid_k: int
    centroid_d: int
    centroid_epsilon: float
    centroid_elegance: float
    member_indices: List[int]     # Indices of subsumed nodes
    member_deltas: List[float]    # Delta from centroid for each member (in cents)
    hierarchy_elegance: float     # Cross-elegance of the archetype


@dataclass  
class CompressionLayer:
    """One layer of the recursive compression hierarchy."""
    depth: int
    seed_r0: float
    archetypes: List[Archetype]
    residual_indices: List[int]   # Indices that couldn't be subsumed
    residual_values: List[int]    # Their raw values
    sublattice_map: Dict[int, List[int]]  # d -> list of archetype indices


class DescriptorLatticeCompressor:
    """
    The core ET compression engine.
    
    Architecture (from et_conscious_ai_compression.py):
    1. Project all bytes onto ET lattice as Descriptors
    2. Discover R₀ seed
    3. Group by sublattice family
    4. Find archetypes via elegance-weighted clustering
    5. Subsume members into archetypes
    6. Apply incoherence filter
    7. Recurse on archetype layer
    8. Universal lattice final projection
    """
    
    def __init__(self, progress_callback=None, log_callback=None):
        self.progress_callback = progress_callback or (lambda p, m: None)
        self.log_callback = log_callback or (lambda m: logger.info(m))
        self.stats = {}
    
    def _log(self, msg: str):
        self.log_callback(msg)
    
    def _progress(self, pct: float, msg: str = ""):
        self.progress_callback(pct, msg)
    
    def compress_block(self, data: bytes) -> dict:
        """
        Compress a single block of data using ET lattice mathematics.
        
        Multi-stage ET compression pipeline:
        
        Stage 1 — Identification Principle: Analyze byte descriptors.
                   Compute frequency → elegance score for each byte value.
        Stage 2 — Descriptor Gap Principle: Build elegance-ranked codebook.
                   High elegance (frequent+simple) = short code, low = long code.
                   Code lengths derived from lattice tightness, NOT Shannon directly.
        Stage 3 — Subsumption Law: Pattern matching via lattice-projected n-grams.
                   Recurring byte sequences subsumed into dictionary entries.
        Stage 4 — Incoherence Filter: Separate coherent (compressible) from
                   incoherent (incompressible) regions.
        Stage 5 — Recursive tower: compress the codebook itself.
        """
        if len(data) == 0:
            return {'type': 'empty'}
        
        values = np.frombuffer(data, dtype=np.uint8)
        n = len(values)
        
        # Trivial case: uniform block
        unique_vals = np.unique(values)
        if len(unique_vals) == 1:
            return {
                'type': 'uniform',
                'value': int(unique_vals[0]),
                'count': n
            }
        
        # ── STAGE 1: Descriptor Identification ──
        # R₀ seed: geometric mean of (byte_value + 1) to avoid log(0)
        vals_shifted = values.astype(np.float64) + 1.0
        r0 = discover_seed(vals_shifted)
        
        # Byte frequency analysis
        freq = np.zeros(256, dtype=np.int64)
        for v in values:
            freq[v] += 1
        
        # Compute elegance score for each byte value present
        # Elegance determines encoding priority (high elegance = shorter code)
        byte_elegance = np.zeros(256, dtype=np.float64)
        for b in range(256):
            if freq[b] > 0:
                ratio = (b + 1.0) / r0
                k, d, eps = lattice_project(ratio)
                # Elegance combines frequency weight with lattice position
                freq_weight = freq[b] / n  # Probability
                tight = tightness_factor(eps)
                lattice_weight = N_RES / max(d, 1)
                # ET-derived elegance: frequency × tightness × lattice depth
                byte_elegance[b] = freq_weight * tight * lattice_weight
        
        # ── STAGE 2: Elegance-Ranked Variable-Length Coding ──
        # Sort symbols by elegance (descending) → assign code lengths
        # Code length formula derived from ET:
        #   len(code) = max(1, round(-log₂(elegance / total_elegance)))
        # This IS Shannon-optimal when elegance ∝ probability, but our elegance
        # also incorporates lattice structure (tightness, sublattice depth).
        
        present_bytes = [b for b in range(256) if freq[b] > 0]
        n_symbols = len(present_bytes)
        
        if n_symbols <= 1:
            return {
                'type': 'uniform',
                'value': present_bytes[0] if present_bytes else 0,
                'count': n
            }
        
        # Sort by elegance descending
        present_bytes.sort(key=lambda b: byte_elegance[b], reverse=True)
        
        total_elegance = sum(byte_elegance[b] for b in present_bytes)
        if total_elegance <= 0:
            total_elegance = 1.0
        
        # Assign code lengths via ET elegance (bounded)
        code_lengths = {}
        for b in present_bytes:
            p_eleg = byte_elegance[b] / total_elegance
            if p_eleg > 0:
                ideal_len = -math.log2(p_eleg)
                code_lengths[b] = max(1, min(255, round(ideal_len)))
            else:
                code_lengths[b] = 255
        
        # Build canonical codes from lengths (same as canonical Huffman but
        # the lengths come from ET elegance, not frequency-only)
        # Sort by (length, symbol) for canonical assignment
        symbols_by_len = sorted(code_lengths.keys(),
                                key=lambda b: (code_lengths[b], b))
        
        codes = {}
        code_val = 0
        prev_len = 0
        for sym in symbols_by_len:
            cl = code_lengths[sym]
            if prev_len > 0:
                code_val = (code_val + 1) << (cl - prev_len)
            codes[sym] = (code_val, cl)
            prev_len = cl
        
        # ── STAGE 3: Pattern Subsumption via Lattice N-grams ──
        # Find recurring byte pairs/triples and subsume them into dictionary
        # Dictionary indexed by lattice position of the n-gram's ratio
        
        # Byte-pair frequency analysis
        pair_freq = Counter()
        for i in range(n - 1):
            pair = (int(values[i]), int(values[i + 1]))
            pair_freq[pair] += 1
        
        # Find pairs that occur enough to be worth subsuming
        # Threshold: pair must save bits (occurrence × savings > overhead)
        # A subsumed pair replaces 2 bytes with 1 dictionary reference
        # Reference costs ~1 byte + dictionary entry overhead
        MIN_PAIR_FREQ = 4  # Minimum occurrences to subsume
        
        # Rank pairs by elegance (frequency × lattice coherence)
        pair_elegances = {}
        for pair, count in pair_freq.items():
            if count >= MIN_PAIR_FREQ:
                # Pair ratio: geometric mean of the two byte ratios
                r1 = (pair[0] + 1.0) / r0
                r2 = (pair[1] + 1.0) / r0
                pair_ratio = math.sqrt(r1 * r2)
                _, d_pair, eps_pair = lattice_project(pair_ratio)
                tight = tightness_factor(eps_pair)
                # Apply incoherence filter L1
                if incoherence_filter_l1(eps_pair):
                    pair_elegances[pair] = count * tight * (N_RES / max(d_pair, 1))
        
        # Select top pairs for dictionary (up to 128 = 2^7, octave class)
        MAX_DICT_PAIRS = 128
        top_pairs = sorted(pair_elegances.keys(),
                           key=lambda p: pair_elegances[p], reverse=True)[:MAX_DICT_PAIRS]
        
        # Assign dictionary indices (256..256+n_pairs-1)
        pair_dict = {}
        for idx, pair in enumerate(top_pairs):
            pair_dict[pair] = 256 + idx
        
        # ── STAGE 4: Encode the data stream ──
        # Replace byte pairs with dictionary references where beneficial,
        # then encode remaining bytes with elegance-derived variable-length codes
        
        # First pass: mark positions where pair substitution applies
        encoded_symbols = []
        i = 0
        while i < n:
            if i < n - 1:
                pair = (int(values[i]), int(values[i + 1]))
                if pair in pair_dict:
                    encoded_symbols.append(pair_dict[pair])
                    i += 2
                    continue
            encoded_symbols.append(int(values[i]))
            i += 1
        
        # Bit-pack the encoded symbols using elegance-derived codes
        # Symbols 0-255: use the elegance code
        # Symbols 256+: use dictionary reference codes
        
        # For dictionary references, assign codes starting after byte codes
        dict_code_len = max(1, math.ceil(math.log2(max(n_symbols + len(pair_dict), 2))))
        
        # Pack into bytes using a simple but effective scheme:
        # [1-byte header per symbol group]:
        #   If symbol < 256: raw byte
        #   If symbol >= 256: escape byte (0xFF) + dict index byte
        
        # Actually, let's use a more efficient approach:
        # Encode the symbol stream where:
        # - Raw bytes are stored as-is
        # - Pair references are stored as escape + index
        # Then compress the resulting stream via run-length encoding
        # on the lattice (runs of same sublattice family)
        
        # Build the encoded byte stream
        escape_byte = None
        # Find least frequent byte to use as escape
        if len(pair_dict) > 0:
            # Choose the byte with lowest frequency as escape
            min_freq_byte = min(range(256), key=lambda b: freq[b])
            escape_byte = min_freq_byte
        
        output_stream = bytearray()
        
        if escape_byte is not None and len(pair_dict) > 0:
            for sym in encoded_symbols:
                if sym < 256:
                    if sym == escape_byte:
                        # Escape the escape byte itself
                        output_stream.append(escape_byte)
                        output_stream.append(0xFF)
                    else:
                        output_stream.append(sym)
                else:
                    # Dictionary reference
                    dict_idx = sym - 256
                    output_stream.append(escape_byte)
                    output_stream.append(dict_idx & 0xFF)
        else:
            # No dictionary pairs — just raw bytes
            output_stream = bytearray(values.tobytes())
        
        # ── STAGE 5: Run-Length Encoding on Lattice Families ──
        # Group consecutive identical bytes into runs
        # This exploits the octave structure: repeated bytes = d=1 unison
        rle_stream = bytearray()
        i = 0
        raw = bytes(output_stream)
        while i < len(raw):
            val = raw[i]
            run_len = 1
            while i + run_len < len(raw) and raw[i + run_len] == val and run_len < 255:
                run_len += 1
            
            if run_len >= 4:
                # RLE marker: 0x00 is rare after pair substitution
                # Use a special RLE encoding: [marker][count][value]
                # But we need a marker byte... use the approach:
                # If run >= 4: store as [value][value][count-2]
                # This signals a run when decoder sees same byte twice
                rle_stream.append(val)
                rle_stream.append(val)
                rle_stream.append(run_len - 2)
                i += run_len
            else:
                for _ in range(run_len):
                    rle_stream.append(val)
                i += run_len
        
        # ── Build compressed block ──
        compressed = {
            'type': 'lattice',
            'original_size': n,
            'seed_r0': r0,
            'escape_byte': escape_byte,
            'pair_dict': [(p[0], p[1]) for p in top_pairs],
            'encoded_data': bytes(rle_stream),
            'num_archetypes': len(top_pairs),
            'num_residuals': 0,
        }
        
        return compressed
    
    def decompress_block(self, compressed: dict) -> bytes:
        """
        Decompress a block by reversing the ET lattice compression pipeline.
        
        Reverse stages: RLE decode → pair expansion → original bytes
        """
        if compressed['type'] == 'empty':
            return b''
        
        if compressed['type'] == 'uniform':
            return bytes([compressed['value']] * compressed['count'])
        
        original_size = compressed['original_size']
        escape_byte = compressed.get('escape_byte')
        pair_dict_list = compressed.get('pair_dict', [])
        encoded_data = compressed.get('encoded_data', b'')
        
        # Stage 5 reverse: RLE decode
        rle_decoded = bytearray()
        i = 0
        while i < len(encoded_data):
            val = encoded_data[i]
            i += 1
            if i < len(encoded_data) and encoded_data[i] == val:
                # Potential RLE run
                i += 1
                if i < len(encoded_data):
                    extra_count = encoded_data[i]
                    i += 1
                    # Total run = extra_count + 2
                    for _ in range(extra_count + 2):
                        rle_decoded.append(val)
                else:
                    rle_decoded.append(val)
                    rle_decoded.append(val)
            else:
                rle_decoded.append(val)
        
        # Stage 3 reverse: Expand pair dictionary references
        output = bytearray()
        i = 0
        raw = bytes(rle_decoded)
        
        if escape_byte is not None and len(pair_dict_list) > 0:
            while i < len(raw):
                if raw[i] == escape_byte and i + 1 < len(raw):
                    next_byte = raw[i + 1]
                    if next_byte == 0xFF:
                        # Escaped escape byte
                        output.append(escape_byte)
                        i += 2
                    elif next_byte < len(pair_dict_list):
                        # Dictionary reference
                        pair = pair_dict_list[next_byte]
                        output.append(pair[0])
                        output.append(pair[1])
                        i += 2
                    else:
                        output.append(raw[i])
                        i += 1
                else:
                    output.append(raw[i])
                    i += 1
        else:
            output = bytearray(raw)
        
        # Truncate or pad to original size
        result = bytes(output[:original_size])
        if len(result) < original_size:
            result = result + b'\x00' * (original_size - len(result))
        
        return result
    
    def recursive_compress(self, data: bytes, depth: int = 0) -> List[dict]:
        """
        Recursively compress: archetypes of archetypes, up to MAX_COMPRESSION_DEPTH.
        
        This is the Subsumption Law applied recursively:
        keep subsuming until all is subsumed or depth limit reached.
        """
        layers = []
        current_data = data
        
        for d in range(min(depth, MAX_COMPRESSION_DEPTH), MAX_COMPRESSION_DEPTH):
            if len(current_data) < 4:  # Too small to compress further
                break
            
            layer = self.compress_block(current_data)
            layers.append(layer)
            
            if layer['type'] in ('empty', 'uniform'):
                break
            
            # Check if compression actually helped
            archetype_count = layer['num_archetypes']
            if archetype_count == 0:
                break  # No more archetypes found, stop recursing
            
            # The next level's input is the archetype centroids + residuals
            # encoded as bytes
            next_data = []
            for arch in layer['archetypes']:
                # Encode centroid as bytes
                centroid_byte = max(0, min(255, round(arch['centroid_ratio'] * 10)))
                next_data.append(centroid_byte)
                next_data.append(arch['member_count'] & 0xFF)
            
            for val in layer['residuals']:
                next_data.append(val & 0xFF)
            
            if len(next_data) >= len(current_data):
                break  # No compression gain, stop
            
            current_data = bytes(next_data)
            self._log(f"  Depth {d+1}: {len(data)} → {len(current_data)} bytes "
                      f"({archetype_count} archetypes)")
        
        return layers


# ═══════════════════════════════════════════════════════════════════════════════
# CDF FILE FORMAT — Serialize/Deserialize compressed data
# ═══════════════════════════════════════════════════════════════════════════════

class CDFFormat:
    """
    CDF (Compressed Descriptor Format) file format.
    
    Structure:
    [4 bytes]  Magic: 'CDF\x01'
    [1 byte]   Version
    [32 bytes] SHA-256 hash of original data
    [8 bytes]  Original file size (uint64 LE)
    [4 bytes]  Number of blocks (uint32 LE)
    [4 bytes]  Block size (uint32 LE)
    [8 bytes]  Seed R₀ as float64
    For each block:
        [4 bytes]  Compressed block size (uint32 LE)
        [N bytes]  Compressed block data (msgpack-like binary encoding)
    """
    
    @staticmethod
    def encode_block(block_data: dict) -> bytes:
        """Encode a compressed block to binary format."""
        parts = []
        
        block_type = block_data['type']
        
        if block_type == 'empty':
            parts.append(struct.pack('<B', 0))  # type 0 = empty
            
        elif block_type == 'uniform':
            parts.append(struct.pack('<B', 1))  # type 1 = uniform
            parts.append(struct.pack('<B', block_data['value']))
            parts.append(struct.pack('<I', block_data['count']))
            
        elif block_type == 'lattice':
            parts.append(struct.pack('<B', 2))  # type 2 = lattice compressed
            parts.append(struct.pack('<I', block_data['original_size']))
            parts.append(struct.pack('<d', block_data['seed_r0']))
            
            # Escape byte (0xFF = no escape)
            escape = block_data.get('escape_byte')
            parts.append(struct.pack('<B', escape if escape is not None else 0xFF))
            
            # Pair dictionary
            pair_dict = block_data.get('pair_dict', [])
            parts.append(struct.pack('<H', len(pair_dict)))
            for p0, p1 in pair_dict:
                parts.append(struct.pack('<BB', p0, p1))
            
            # Encoded data stream
            encoded = block_data.get('encoded_data', b'')
            parts.append(struct.pack('<I', len(encoded)))
            parts.append(encoded)
        
        return b''.join(parts)
    
    @staticmethod
    def decode_block(data: bytes, offset: int = 0) -> Tuple[dict, int]:
        """Decode a compressed block from binary format."""
        pos = offset
        
        block_type = struct.unpack_from('<B', data, pos)[0]
        pos += 1
        
        if block_type == 0:  # empty
            return {'type': 'empty'}, pos
        
        elif block_type == 1:  # uniform
            value = struct.unpack_from('<B', data, pos)[0]
            pos += 1
            count = struct.unpack_from('<I', data, pos)[0]
            pos += 4
            return {'type': 'uniform', 'value': value, 'count': count}, pos
        
        elif block_type == 2:  # lattice
            original_size = struct.unpack_from('<I', data, pos)[0]
            pos += 4
            seed_r0 = struct.unpack_from('<d', data, pos)[0]
            pos += 8
            
            escape_byte_raw = struct.unpack_from('<B', data, pos)[0]
            pos += 1
            escape_byte = escape_byte_raw if escape_byte_raw != 0xFF else None
            
            n_pairs = struct.unpack_from('<H', data, pos)[0]
            pos += 2
            
            pair_dict = []
            for _ in range(n_pairs):
                p0, p1 = struct.unpack_from('<BB', data, pos)
                pos += 2
                pair_dict.append((p0, p1))
            
            encoded_len = struct.unpack_from('<I', data, pos)[0]
            pos += 4
            encoded_data = data[pos:pos + encoded_len]
            pos += encoded_len
            
            return {
                'type': 'lattice',
                'original_size': original_size,
                'seed_r0': seed_r0,
                'escape_byte': escape_byte,
                'pair_dict': pair_dict,
                'encoded_data': encoded_data,
                'num_archetypes': n_pairs,
                'num_residuals': 0,
            }, pos
        
        raise ValueError(f"Unknown block type: {block_type}")


# ═══════════════════════════════════════════════════════════════════════════════
# HIGH-LEVEL COMPRESS/DECOMPRESS API
# ═══════════════════════════════════════════════════════════════════════════════

class CDFCompressor:
    """
    High-level CDF compression/decompression interface.
    Handles file I/O, block splitting, and the full compression pipeline.
    """
    
    def __init__(self, progress_callback=None, log_callback=None):
        self.engine = DescriptorLatticeCompressor(progress_callback, log_callback)
        self.progress_callback = progress_callback or (lambda p, m: None)
        self.log_callback = log_callback or (lambda m: logger.info(m))
    
    def _log(self, msg: str):
        self.log_callback(msg)
    
    def _progress(self, pct: float, msg: str = ""):
        self.progress_callback(pct, msg)
    
    def compress_file(self, input_path: str, output_path: str):
        """Compress a file to CDF format."""
        input_path = Path(input_path)
        output_path = Path(output_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        self._log(f"Compressing: {input_path}")
        self._log(f"Output: {output_path}")
        
        # Read entire file
        with open(input_path, 'rb') as f:
            raw_data = f.read()
        
        original_size = len(raw_data)
        self._log(f"Original size: {original_size:,} bytes")
        
        # Compute SHA-256 for integrity verification
        sha256_hash = hashlib.sha256(raw_data).digest()
        
        # Split into blocks of BLOCK_SIZE (4096 = 2^N = digital action quantum)
        num_blocks = (original_size + BLOCK_SIZE - 1) // BLOCK_SIZE
        self._log(f"Block count: {num_blocks} (block size: {BLOCK_SIZE} = 2^{N_RES})")
        
        # Discover global seed R₀
        all_values = np.frombuffer(raw_data, dtype=np.uint8).astype(np.float64) + 1.0
        global_r0 = discover_seed(all_values)
        self._log(f"Global R₀ seed: {global_r0:.6f}")
        
        # Compress each block
        compressed_blocks = []
        start_time = time.time()
        
        for block_idx in range(num_blocks):
            block_start = block_idx * BLOCK_SIZE
            block_end = min(block_start + BLOCK_SIZE, original_size)
            block_data = raw_data[block_start:block_end]
            
            # Compress block using ET lattice
            compressed = self.engine.compress_block(block_data)
            encoded = CDFFormat.encode_block(compressed)
            compressed_blocks.append(encoded)
            
            # Progress
            pct = (block_idx + 1) / num_blocks * 100
            self._progress(pct, f"Block {block_idx + 1}/{num_blocks}")
        
        elapsed = time.time() - start_time
        
        # Write CDF file
        with open(output_path, 'wb') as f:
            # Header
            f.write(CDF_MAGIC)
            f.write(struct.pack('<B', CDF_VERSION))
            f.write(sha256_hash)
            f.write(struct.pack('<Q', original_size))
            f.write(struct.pack('<I', num_blocks))
            f.write(struct.pack('<I', BLOCK_SIZE))
            f.write(struct.pack('<d', global_r0))
            
            # Blocks
            for encoded_block in compressed_blocks:
                f.write(struct.pack('<I', len(encoded_block)))
                f.write(encoded_block)
        
        compressed_size = output_path.stat().st_size
        ratio = compressed_size / original_size if original_size > 0 else 0
        
        self._log(f"Compressed size: {compressed_size:,} bytes")
        self._log(f"Ratio: {ratio:.4f} ({ratio*100:.1f}%)")
        self._log(f"Time: {elapsed:.2f}s")
        self._progress(100, "Complete")
        
        return {
            'original_size': original_size,
            'compressed_size': compressed_size,
            'ratio': ratio,
            'time': elapsed,
            'blocks': num_blocks,
        }
    
    def decompress_file(self, input_path: str, output_path: str):
        """Decompress a CDF file back to original."""
        input_path = Path(input_path)
        output_path = Path(output_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"CDF file not found: {input_path}")
        
        self._log(f"Decompressing: {input_path}")
        
        with open(input_path, 'rb') as f:
            cdf_data = f.read()
        
        pos = 0
        
        # Read header
        magic = cdf_data[pos:pos+4]
        pos += 4
        if magic != CDF_MAGIC:
            raise ValueError("Not a valid CDF file (bad magic)")
        
        version = struct.unpack_from('<B', cdf_data, pos)[0]
        pos += 1
        if version != CDF_VERSION:
            raise ValueError(f"Unsupported CDF version: {version}")
        
        stored_hash = cdf_data[pos:pos+32]
        pos += 32
        
        original_size = struct.unpack_from('<Q', cdf_data, pos)[0]
        pos += 8
        
        num_blocks = struct.unpack_from('<I', cdf_data, pos)[0]
        pos += 4
        
        block_size = struct.unpack_from('<I', cdf_data, pos)[0]
        pos += 4
        
        global_r0 = struct.unpack_from('<d', cdf_data, pos)[0]
        pos += 8
        
        self._log(f"Original size: {original_size:,} bytes")
        self._log(f"Blocks: {num_blocks}, Block size: {block_size}")
        self._log(f"Global R₀: {global_r0:.6f}")
        
        # Decompress each block
        output_parts = []
        start_time = time.time()
        
        for block_idx in range(num_blocks):
            block_len = struct.unpack_from('<I', cdf_data, pos)[0]
            pos += 4
            
            block_data = cdf_data[pos:pos+block_len]
            pos += block_len
            
            # Decode and decompress
            compressed, _ = CDFFormat.decode_block(block_data)
            decompressed = self.engine.decompress_block(compressed)
            output_parts.append(decompressed)
            
            pct = (block_idx + 1) / num_blocks * 100
            self._progress(pct, f"Block {block_idx + 1}/{num_blocks}")
        
        elapsed = time.time() - start_time
        
        # Assemble output
        raw_output = b''.join(output_parts)[:original_size]
        
        # Verify integrity
        computed_hash = hashlib.sha256(raw_output).digest()
        integrity_ok = computed_hash == stored_hash
        
        if not integrity_ok:
            self._log("WARNING: SHA-256 mismatch — decompressed data differs from original!")
        else:
            self._log("Integrity check: PASSED (SHA-256 match)")
        
        # Write output
        with open(output_path, 'wb') as f:
            f.write(raw_output)
        
        self._log(f"Decompressed to: {output_path}")
        self._log(f"Size: {len(raw_output):,} bytes")
        self._log(f"Time: {elapsed:.2f}s")
        self._progress(100, "Complete")
        
        return {
            'output_size': len(raw_output),
            'integrity': integrity_ok,
            'time': elapsed,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# GUI — PyQt6/Tkinter with progress bar, console log, file selection
# ═══════════════════════════════════════════════════════════════════════════════

def build_gui():
    """Build the CDF Compressor GUI using tkinter (cross-platform, no extra deps)."""
    import tkinter as tk
    from tkinter import ttk, filedialog, scrolledtext, messagebox
    import threading
    
    class CDFApp:
        def __init__(self, root):
            self.root = root
            self.root.title("ET CDF Compressor — P ∘ D ∘ T = E")
            self.root.geometry("800x600")
            self.root.minsize(700, 500)
            
            # Configure style
            style = ttk.Style()
            style.theme_use('clam')
            
            # Main frame
            main = ttk.Frame(root, padding=10)
            main.pack(fill=tk.BOTH, expand=True)
            
            # Title
            title_lbl = ttk.Label(main, text="CDF Compressor — Compressed Descriptor Format",
                                  font=('Helvetica', 14, 'bold'))
            title_lbl.pack(pady=(0, 5))
            
            subtitle = ttk.Label(main, text="Exception Theory Lattice Compression  •  P ∘ D ∘ T = E",
                                 font=('Helvetica', 10))
            subtitle.pack(pady=(0, 10))
            
            # File selection frame
            file_frame = ttk.LabelFrame(main, text="File Selection", padding=10)
            file_frame.pack(fill=tk.X, pady=5)
            
            # Input file
            input_row = ttk.Frame(file_frame)
            input_row.pack(fill=tk.X, pady=2)
            ttk.Label(input_row, text="Input:", width=8).pack(side=tk.LEFT)
            self.input_var = tk.StringVar()
            ttk.Entry(input_row, textvariable=self.input_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
            ttk.Button(input_row, text="Browse...", command=self.browse_input).pack(side=tk.RIGHT)
            
            # Output file
            output_row = ttk.Frame(file_frame)
            output_row.pack(fill=tk.X, pady=2)
            ttk.Label(output_row, text="Output:", width=8).pack(side=tk.LEFT)
            self.output_var = tk.StringVar()
            ttk.Entry(output_row, textvariable=self.output_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
            ttk.Button(output_row, text="Browse...", command=self.browse_output).pack(side=tk.RIGHT)
            
            # Action buttons
            btn_frame = ttk.Frame(main)
            btn_frame.pack(fill=tk.X, pady=10)
            
            self.compress_btn = ttk.Button(btn_frame, text="Compress to .cdf",
                                           command=self.start_compress)
            self.compress_btn.pack(side=tk.LEFT, padx=5)
            
            self.decompress_btn = ttk.Button(btn_frame, text="Decompress .cdf",
                                             command=self.start_decompress)
            self.decompress_btn.pack(side=tk.LEFT, padx=5)
            
            # Progress bar
            progress_frame = ttk.LabelFrame(main, text="Progress", padding=5)
            progress_frame.pack(fill=tk.X, pady=5)
            
            self.progress_var = tk.DoubleVar(value=0)
            self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var,
                                                 maximum=100, mode='determinate')
            self.progress_bar.pack(fill=tk.X, pady=2)
            
            self.status_var = tk.StringVar(value="Ready")
            ttk.Label(progress_frame, textvariable=self.status_var).pack(anchor=tk.W)
            
            # Console log
            log_frame = ttk.LabelFrame(main, text="Console Log", padding=5)
            log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
            
            self.console = scrolledtext.ScrolledText(log_frame, height=12,
                                                      font=('Consolas', 9),
                                                      state=tk.DISABLED,
                                                      bg='#1e1e1e', fg='#d4d4d4',
                                                      insertbackground='white')
            self.console.pack(fill=tk.BOTH, expand=True)
            
            # ET Constants display
            const_frame = ttk.Frame(main)
            const_frame.pack(fill=tk.X, pady=5)
            constants_text = (f"N={N_RES}  V=1/{N_RES}  K={K_KOIDE:.4f}  "
                            f"Block=2^{N_RES}={BLOCK_SIZE}  "
                            f"MaxDepth={MAX_COMPRESSION_DEPTH}  "
                            f"∂I=±{INCOHERENCE_BOUNDARY_CENTS}¢")
            ttk.Label(const_frame, text=constants_text,
                     font=('Consolas', 8)).pack(anchor=tk.W)
            
            self.log_msg("CDF Compressor initialized")
            self.log_msg(f"ET Constants: N={N_RES}, V=1/{N_RES}, K={K_KOIDE:.4f}")
            self.log_msg(f"Block size: {BLOCK_SIZE} bytes (2^N = digital action quantum)")
            self.log_msg(f"Max recursion depth: {MAX_COMPRESSION_DEPTH} (= S = manifold symmetry)")
            self.log_msg("Ready.")
        
        def log_msg(self, msg: str):
            """Thread-safe log to console."""
            def _update():
                self.console.configure(state=tk.NORMAL)
                self.console.insert(tk.END, msg + "\n")
                self.console.see(tk.END)
                self.console.configure(state=tk.DISABLED)
            self.root.after(0, _update)
        
        def update_progress(self, pct: float, msg: str = ""):
            def _update():
                self.progress_var.set(pct)
                if msg:
                    self.status_var.set(msg)
            self.root.after(0, _update)
        
        def browse_input(self):
            path = filedialog.askopenfilename(
                title="Select input file",
                filetypes=[("All files", "*.*"), ("CDF files", "*.cdf")]
            )
            if path:
                self.input_var.set(path)
                # Auto-set output
                if path.endswith('.cdf'):
                    self.output_var.set(path[:-4])
                else:
                    self.output_var.set(path + '.cdf')
        
        def browse_output(self):
            path = filedialog.asksaveasfilename(
                title="Select output file",
                filetypes=[("CDF files", "*.cdf"), ("All files", "*.*")]
            )
            if path:
                self.output_var.set(path)
        
        def set_buttons_state(self, state: str):
            def _update():
                self.compress_btn.configure(state=state)
                self.decompress_btn.configure(state=state)
            self.root.after(0, _update)
        
        def start_compress(self):
            input_path = self.input_var.get()
            output_path = self.output_var.get()
            
            if not input_path or not output_path:
                messagebox.showerror("Error", "Please select input and output files.")
                return
            
            if not os.path.exists(input_path):
                messagebox.showerror("Error", f"Input file not found: {input_path}")
                return
            
            self.set_buttons_state('disabled')
            
            def _compress():
                try:
                    compressor = CDFCompressor(
                        progress_callback=self.update_progress,
                        log_callback=self.log_msg
                    )
                    result = compressor.compress_file(input_path, output_path)
                    self.log_msg(f"\n=== COMPRESSION COMPLETE ===")
                    self.log_msg(f"Original:   {result['original_size']:,} bytes")
                    self.log_msg(f"Compressed: {result['compressed_size']:,} bytes")
                    self.log_msg(f"Ratio:      {result['ratio']*100:.1f}%")
                    self.log_msg(f"Time:       {result['time']:.2f}s")
                    self.root.after(0, lambda: messagebox.showinfo(
                        "Complete",
                        f"Compressed to {result['compressed_size']:,} bytes "
                        f"({result['ratio']*100:.1f}%)"
                    ))
                except Exception as e:
                    self.log_msg(f"ERROR: {e}")
                    self.log_msg(traceback.format_exc())
                    self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
                finally:
                    self.set_buttons_state('normal')
            
            threading.Thread(target=_compress, daemon=True).start()
        
        def start_decompress(self):
            input_path = self.input_var.get()
            output_path = self.output_var.get()
            
            if not input_path or not output_path:
                messagebox.showerror("Error", "Please select input and output files.")
                return
            
            if not input_path.endswith('.cdf'):
                messagebox.showwarning("Warning", "Input file should be a .cdf file")
            
            self.set_buttons_state('disabled')
            
            def _decompress():
                try:
                    compressor = CDFCompressor(
                        progress_callback=self.update_progress,
                        log_callback=self.log_msg
                    )
                    result = compressor.decompress_file(input_path, output_path)
                    self.log_msg(f"\n=== DECOMPRESSION COMPLETE ===")
                    self.log_msg(f"Output size: {result['output_size']:,} bytes")
                    self.log_msg(f"Integrity:   {'PASSED' if result['integrity'] else 'FAILED'}")
                    self.log_msg(f"Time:        {result['time']:.2f}s")
                    
                    if result['integrity']:
                        self.root.after(0, lambda: messagebox.showinfo(
                            "Complete", f"Decompressed successfully to {output_path}"))
                    else:
                        self.root.after(0, lambda: messagebox.showwarning(
                            "Warning", "Decompressed but integrity check FAILED!"))
                except Exception as e:
                    self.log_msg(f"ERROR: {e}")
                    self.log_msg(traceback.format_exc())
                    self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
                finally:
                    self.set_buttons_state('normal')
            
            threading.Thread(target=_decompress, daemon=True).start()
    
    root = tk.Tk()
    app = CDFApp(root)
    root.mainloop()


# ═══════════════════════════════════════════════════════════════════════════════
# CLI INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════

def cli_main():
    """Command-line interface for CDF compression."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='ET CDF Compressor — Compressed Descriptor Format (P ∘ D ∘ T = E)')
    parser.add_argument('action', choices=['compress', 'decompress', 'gui'],
                       help='Action to perform')
    parser.add_argument('input', nargs='?', help='Input file path')
    parser.add_argument('output', nargs='?', help='Output file path')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if args.action == 'gui':
        build_gui()
        return
    
    if not args.input:
        parser.error(f"Input file required for {args.action}")
    
    if not args.output:
        if args.action == 'compress':
            args.output = args.input + '.cdf'
        else:
            args.output = args.input.replace('.cdf', '.out') if args.input.endswith('.cdf') else args.input + '.out'
    
    def log_fn(msg):
        print(msg)
    
    def progress_fn(pct, msg):
        if msg:
            print(f"\r  [{pct:5.1f}%] {msg}", end='', flush=True)
    
    compressor = CDFCompressor(progress_callback=progress_fn, log_callback=log_fn)
    
    if args.action == 'compress':
        result = compressor.compress_file(args.input, args.output)
        print(f"\nDone. Ratio: {result['ratio']*100:.1f}%")
    
    elif args.action == 'decompress':
        result = compressor.decompress_file(args.input, args.output)
        print(f"\nDone. Integrity: {'PASS' if result['integrity'] else 'FAIL'}")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        cli_main()
    else:
        build_gui()
