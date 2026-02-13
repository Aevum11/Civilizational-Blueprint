"""
Exception Theory Constants Module

All constants derived from Exception Theory primitives: P (Point), D (Descriptor), T (Traverser)

From: "For every exception there is an exception, except the exception."

Author: Derived from M.J.M.'s Exception Theory
"""

import os
import tempfile

# ============================================================================
# CACHE AND ENVIRONMENT CONFIGURATION
# ============================================================================

def _get_cache_file():
    """
    Get cache file path only if writable, else None (memory-only mode).
    
    Returns:
        str or None: Path to cache file if writable, None otherwise
    """
    try:
        tmp_dir = tempfile.gettempdir()
        test_file = os.path.join(tmp_dir, f".et_write_test_{os.getpid()}")
        try:
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            return os.path.join(tmp_dir, "et_compendium_geometry_v3_0.json")
        except (OSError, IOError):
            return None
    except:
        return None

CACHE_FILE = _get_cache_file()
MAX_SCAN_WIDTH = 2048
DEFAULT_TUPLE_DEPTH = 4
ET_CACHE_ENV_VAR = "ET_COMPENDIUM_GEOMETRY_CACHE_V3_0"
ET_SHARED_MEM_NAME = "et_compendium_geometry_shm_v3_0"
ET_SHARED_MEM_SIZE = 8192

# ============================================================================
# PHASE-LOCK DESCRIPTORS (RO Bypass)
# ============================================================================

DEFAULT_NOISE_PATTERN = 0xFF
DEFAULT_INJECTION_COUNT = 1
ALTERNATE_NOISE_PATTERNS = [0xFF, 0xAA, 0x55, 0x00]
PATTERN_NAMES = {
    0xFF: "BIT_INVERT",
    0xAA: "ALT_HIGH",
    0x55: "ALT_LOW",
    0x00: "DISABLED"
}

# ============================================================================
# MEMORY PROTECTION DESCRIPTORS
# ============================================================================

PROT = {
    'NONE': 0x0,
    'READ': 0x1,
    'WRITE': 0x2,
    'EXEC': 0x4
}

PAGE = {
    'NOACCESS': 0x01,
    'READONLY': 0x02,
    'READWRITE': 0x04,
    'EXEC_READ': 0x20,
    'EXEC_READWRITE': 0x40
}

# ============================================================================
# RO BYPASS TIER DESCRIPTORS
# ============================================================================

RO_BYPASS_TIERS = [
    "TUNNEL_PHASE_LOCK",
    "DIRECT_MEMMOVE",
    "MPROTECT_DIRECT",
    "CTYPES_POINTER_CAST",
    "PYOBJECT_STRUCTURE",
    "DISPLACEMENT_HOLOGRAPHIC"
]

# ============================================================================
# ET FUNDAMENTAL CONSTANTS (Derived from Exception Theory)
# ============================================================================

# Core Variance and Symmetry
BASE_VARIANCE = 1.0 / 12.0  # From ET manifold mathematics
MANIFOLD_SYMMETRY = 12      # Fundamental symmetry count
KOIDE_RATIO = 2.0 / 3.0     # Koide formula constant

# Cosmological Ratios (from ET predictions)
DARK_ENERGY_RATIO = 68.3 / 100.0
DARK_MATTER_RATIO = 26.8 / 100.0
ORDINARY_MATTER_RATIO = 4.9 / 100.0

# ============================================================================
# INDETERMINACY CONSTANTS (v2.1+)
# ============================================================================

T_SINGULARITY_THRESHOLD = 1e-9    # Nanosecond precision for T-gap detection
COHERENCE_VARIANCE_FLOOR = 0.0    # Absolute coherence floor

# ============================================================================
# MANIFOLD ARCHITECTURE CONSTANTS (v2.2+)
# ============================================================================

DEFAULT_BLOOM_SIZE = 1024
DEFAULT_BLOOM_HASHES = 3
ZK_DEFAULT_GENERATOR = 5
ZK_DEFAULT_PRIME = 1000000007

# ============================================================================
# DISTRIBUTED CONSCIOUSNESS CONSTANTS (v2.3+)
# ============================================================================

DEFAULT_SWARM_COHERENCE = 1.0
DEFAULT_SWARM_ALIGNMENT_BONUS = 0.1
DEFAULT_SWARM_STABILITY_BONUS = 0.05
PRECOG_HISTORY_SIZE = 5
PRECOG_PROBABILITY_THRESHOLD = 0.5
DEFAULT_VARIANCE_CAPACITY = 100.0
DEFAULT_VARIANCE_REFILL_RATE = 10.0
DEFAULT_POT_DIFFICULTY = 4
DEFAULT_HASH_RING_REPLICAS = 3
FRACTAL_DEFAULT_OCTAVES = 3
FRACTAL_DEFAULT_PERSISTENCE = 0.5

# ============================================================================
# VERSION INFORMATION
# ============================================================================

VERSION = "3.0.0"
VERSION_INFO = (3, 0, 0)
BUILD = "production"

# Version History
VERSION_HISTORY = {
    "2.0": {"lines": 2586, "equations": "1-10", "focus": "Core transmutation"},
    "2.1": {"lines": 3119, "equations": "11-20", "focus": "Batch 1: Computational ET"},
    "2.2": {"lines": 4313, "equations": "21-30", "focus": "Batch 2: Manifold Architectures"},
    "2.3": {"lines": 5799, "equations": "31-40", "focus": "Batch 3: Distributed Consciousness"},
    "3.0": {"lines": "TBD", "equations": "All", "focus": "Library Architecture"}
}

__all__ = [
    # Cache and Environment
    'CACHE_FILE',
    'MAX_SCAN_WIDTH',
    'DEFAULT_TUPLE_DEPTH',
    'ET_CACHE_ENV_VAR',
    'ET_SHARED_MEM_NAME',
    'ET_SHARED_MEM_SIZE',
    
    # Phase-Lock
    'DEFAULT_NOISE_PATTERN',
    'DEFAULT_INJECTION_COUNT',
    'ALTERNATE_NOISE_PATTERNS',
    'PATTERN_NAMES',
    
    # Memory Protection
    'PROT',
    'PAGE',
    'RO_BYPASS_TIERS',
    
    # ET Fundamental Constants
    'BASE_VARIANCE',
    'MANIFOLD_SYMMETRY',
    'KOIDE_RATIO',
    'DARK_ENERGY_RATIO',
    'DARK_MATTER_RATIO',
    'ORDINARY_MATTER_RATIO',
    
    # Indeterminacy
    'T_SINGULARITY_THRESHOLD',
    'COHERENCE_VARIANCE_FLOOR',
    
    # Manifold Architecture
    'DEFAULT_BLOOM_SIZE',
    'DEFAULT_BLOOM_HASHES',
    'ZK_DEFAULT_GENERATOR',
    'ZK_DEFAULT_PRIME',
    
    # Distributed Consciousness
    'DEFAULT_SWARM_COHERENCE',
    'DEFAULT_SWARM_ALIGNMENT_BONUS',
    'DEFAULT_SWARM_STABILITY_BONUS',
    'PRECOG_HISTORY_SIZE',
    'PRECOG_PROBABILITY_THRESHOLD',
    'DEFAULT_VARIANCE_CAPACITY',
    'DEFAULT_VARIANCE_REFILL_RATE',
    'DEFAULT_POT_DIFFICULTY',
    'DEFAULT_HASH_RING_REPLICAS',
    'FRACTAL_DEFAULT_OCTAVES',
    'FRACTAL_DEFAULT_PERSISTENCE',
    
    # Version
    'VERSION',
    'VERSION_INFO',
    'BUILD',
    'VERSION_HISTORY',
]
