"""
ET Compendium Sovereign - Fixed via Exception Theory Mathematics

This implementation applies ET-derived mathematics to unify operations:
- Eq 211: S = D/D² (Structural Density) for geometry detection
- Eq 212: |T|² = |D₁|² + |D₂|² (Traverser Effort/Pythagoras)
- P ∘ D ∘ T = E (Master Equation): All operations as Traverser navigation
- Eq 30: Status_sub = [1 + exp(-G_input)]^(-1) (Phase Transition Sigmoid)
- Eq 77: N_min(C_target) = min(Count(D_set)) (Kolmogorov - Minimal Descriptors)
- Eq 83: D_next = D_current - S_step ∘ Direction(∇V_sys) (Gradient Descent - Variance Minimization)

FIXES APPLIED (from old version regression analysis):
1. Restored reliable RO bypass with multi-tier fallback (sandbox-safe)
2. Fixed UCS2/UCS4 calibration with robust multi-beacon + fallback chains
3. Added comprehensive C-level intern reference displacement
4. Made disk cache optional with graceful memory-only fallback

Original: 3059 lines | Optimized: 1493 lines | Fixed: ~1650 lines
All functions, features, and verbosity preserved.
"""

import ctypes
import sys
import os
import platform
import struct
import gc
import json
import tempfile
import collections.abc
import inspect
import threading
import time
import math
import logging
import mmap
import hashlib
import weakref

try:
    from multiprocessing import shared_memory
    HAS_SHARED_MEMORY = True
except ImportError:
    HAS_SHARED_MEMORY = False

# --- LOGGING SETUP ---
logger = logging.getLogger('ETSovereign')
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setLevel(logging.DEBUG)
    _handler.setFormatter(logging.Formatter('[ET %(levelname)s] %(message)s'))
    logger.addHandler(_handler)

# --- CONFIGURATION (Descriptor Constants) ---
# FIX #4: Make cache file optional - check writability before using
def _get_cache_file():
    """Get cache file path only if writable, else None (memory-only mode)."""
    try:
        tmp_dir = tempfile.gettempdir()
        test_file = os.path.join(tmp_dir, f".et_write_test_{os.getpid()}")
        try:
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            return os.path.join(tmp_dir, "et_compendium_geometry.json")
        except (OSError, IOError):
            return None
    except:
        return None

CACHE_FILE = _get_cache_file()  # May be None in read-only environments
MAX_SCAN_WIDTH = 2048
DEFAULT_TUPLE_DEPTH = 4
ET_CACHE_ENV_VAR = "ET_COMPENDIUM_GEOMETRY_CACHE"
ET_SHARED_MEM_NAME = "et_compendium_geometry_shm"
ET_SHARED_MEM_SIZE = 4096

# Phase-Lock Descriptors
DEFAULT_NOISE_PATTERN = 0xFF
DEFAULT_INJECTION_COUNT = 1
ALTERNATE_NOISE_PATTERNS = [0xFF, 0xAA, 0x55, 0x00]
PATTERN_NAMES = {0xFF: "BIT_INVERT", 0xAA: "ALT_HIGH", 0x55: "ALT_LOW", 0x00: "DISABLED"}

# Memory Protection Descriptors (unified via ET binding)
PROT = {'NONE': 0x0, 'READ': 0x1, 'WRITE': 0x2, 'EXEC': 0x4}
PAGE = {'NOACCESS': 0x01, 'READONLY': 0x02, 'READWRITE': 0x04, 
        'EXEC_READ': 0x20, 'EXEC_READWRITE': 0x40}

# FIX #1: RO Bypass Tier Descriptors (from old version, enhanced)
RO_BYPASS_TIERS = [
    "TUNNEL_PHASE_LOCK",       # Tier 1: Kernel tunnel with noise injection
    "DIRECT_MEMMOVE",          # Tier 2: ctypes.memmove (if pages writable)
    "MPROTECT_DIRECT",         # Tier 2.5: mprotect/VirtualProtect + memmove
    "CTYPES_POINTER_CAST",     # Tier 2.7: Direct pointer manipulation via ctypes
    "PYOBJECT_STRUCTURE",      # Tier 2.8: PyObject structure manipulation
    "DISPLACEMENT_HOLOGRAPHIC" # Tier 3: Reference displacement (always works)
]


class ETMath:
    """
    Operationalized ET Equations - The Core Descriptor Operators.
    These are Traverser functions that navigate the manifold.
    """
    
    @staticmethod
    def density(payload, container):
        """
        Eq 211: S = D/D² (Structural Density) - Payload/Container ratio.
        High density (>0.7) indicates Compact Geometry (inline storage).
        Low density (<0.1) indicates Pointer Geometry (external buffer).
        """
        return float(payload) / float(container) if container else 0.0
    
    @staticmethod
    def effort(observers, byte_delta):
        """
        Eq 212: |T|² = |D₁|² + |D₂|² - Traverser metabolic cost.
        Calculates the "energy" required for transmutation.
        """
        return math.sqrt(observers**2 + byte_delta**2)
    
    @staticmethod
    def bind(p, d, t=None):
        """P ∘ D ∘ T = E - The Master Equation binding operator."""
        return (p, d, t) if t else (p, d)
    
    @staticmethod
    def phase_transition(gradient_input, threshold=0.0):
        """
        Eq 30: Status_sub = [1 + exp(-G_input)]^(-1) (Sigmoid Phase Transition)
        Models the flip from Potential (0) to Real (1) when gradient crosses threshold.
        Used for confidence scoring in calibration.
        """
        try:
            adjusted = gradient_input - threshold
            return 1.0 / (1.0 + math.exp(-adjusted))
        except OverflowError:
            return 1.0 if gradient_input > threshold else 0.0
    
    @staticmethod
    def variance_gradient(current_variance, target_variance, step_size=0.1):
        """
        Eq 83: D_next = D_current - S_step ∘ Direction(∇V_sys)
        Gradient descent toward minimal variance (intelligence minimizes variance).
        """
        delta = target_variance - current_variance
        direction = 1.0 if delta > 0 else -1.0
        magnitude = abs(delta)
        return current_variance + (step_size * direction * magnitude)
    
    @staticmethod
    def kolmogorov_complexity(descriptor_set):
        """
        Eq 77: N_min(C_target) = min(Count(D_set))
        Minimal descriptors needed to substantiate object.
        Used for determining minimal beacon requirements.
        """
        if not descriptor_set:
            return 0
        return len(set(descriptor_set))
    
    @staticmethod
    def encode_width(s, width):
        """
        Encode string to bytes based on descriptor width.
        Returns None if encoding fails (allows graceful fallback).
        """
        if width == 1:
            try:
                return s.encode('latin-1')
            except UnicodeEncodeError:
                return None
        elif width == 2:
            try:
                return b"".join(struct.pack('<H', ord(c)) for c in s)
            except struct.error:
                return None
        elif width == 4:
            try:
                return b"".join(struct.pack('<I', ord(c)) for c in s)
            except struct.error:
                return None
        return None
    
    @staticmethod
    def decode_width(data, width):
        """Decode bytes to string based on descriptor width."""
        if width == 1:
            return data.decode('latin-1')
        elif width == 2:
            chars = [struct.unpack('<H', data[i:i+2])[0] for i in range(0, len(data), 2)]
            return ''.join(chr(c) for c in chars)
        elif width == 4:
            chars = [struct.unpack('<I', data[i:i+4])[0] for i in range(0, len(data), 4)]
            return ''.join(chr(c) for c in chars)
        return None


class ETBeaconField:
    """
    ET Beacon Generator - Unified Descriptor Field for Calibration.
    
    FIX #2: Enhanced beacon generation with multi-tier fallbacks for UCS2/UCS4.
    Applies ET principle: Different beacon types are same operation
    with different descriptor parameters (width determines Unicode range).
    """
    
    # Primary Descriptor pools for each width (D bindings)
    CHARS_PRIMARY = {
        1: "ABCDEFGHIJKLMNOP",  # ASCII
        2: '\u03A9\u0394\u03A3\u03A0\u0416\u042F\u05D0\u4E2D\u65E5\u00C6\u00D8\u0152\u2202\u221E\u2211',  # UCS2
        4: '\U0001F40D\U0001F525\U0001F4A1\U0001F680\U0001F916\U0001F9E0\U0001F4BB\U0001F310\U0001F3AF\U0001F4A0\U0001F52C\U0001F9EC\U0001F300\U0001F31F\U0001F4AB'  # UCS4
    }
    
    # Secondary fallback pools (simpler characters more likely to encode correctly)
    CHARS_SECONDARY = {
        1: "0123456789QRSTUV",
        2: '\u00C0\u00C1\u00C2\u00C3\u00C4\u00C5\u00E0\u00E1\u00E2\u00E3\u00E4\u00E5\u00F0\u00F1\u00F2',  # Latin Extended
        4: '\U00010000\U00010001\U00010002\U00010003\U00010004\U00010005\U00010006\U00010007\U00010008\U00010009\U0001000A\U0001000B\U0001000C\U0001000D\U0001000E'  # Linear B
    }
    
    # Tertiary fallback (guaranteed to exist in all Python builds)
    CHARS_TERTIARY = {
        1: "etbcn0123456789_",
        2: '\u0100\u0101\u0102\u0103\u0104\u0105\u0106\u0107\u0108\u0109\u010A\u010B\u010C\u010D\u010E',  # Latin Extended-A
        4: '\U00020000\U00020001\U00020002\U00020003\U00020004\U00020005\U00020006\U00020007\U00020008\U00020009\U0002000A\U0002000B\U0002000C\U0002000D\U0002000E'  # CJK Extension B
    }
    
    @classmethod
    def generate(cls, width, count=50):
        """Generate beacon field - T navigating through D_char pool with fallbacks."""
        beacons = []
        
        # Try each character pool in order of preference
        for char_pool in [cls.CHARS_PRIMARY, cls.CHARS_SECONDARY, cls.CHARS_TERTIARY]:
            chars = char_pool.get(width, char_pool[1])
            
            for c in chars:
                beacon = f"ET_{c}"
                # Verify beacon can be encoded at this width
                encoded = ETMath.encode_width(beacon, width)
                if encoded is not None:
                    beacons.append(beacon)
            
            # Also generate extended beacons
            for i, c in enumerate(chars * 3):
                beacon = f"ET_W{width}_{c}{i}"
                encoded = ETMath.encode_width(beacon, width)
                if encoded is not None and beacon not in beacons:
                    beacons.append(beacon)
            
            if len(beacons) >= count:
                break
        
        # Pad with simple guaranteed beacons if needed
        while len(beacons) < count:
            pad_beacon = f"ET_PAD_{width}_{len(beacons)}"
            if ETMath.encode_width(pad_beacon, width) is not None:
                beacons.append(pad_beacon)
            else:
                # Final fallback: use width 1 encoding
                beacons.append(f"ET_P{len(beacons)}")
        
        return beacons[:count]
    
    @classmethod
    def generate_simple(cls, prefix, width):
        """
        Generate a single simple beacon - used for fallback calibration.
        Restored from old version for reliability.
        """
        if width == 1:
            return prefix + "A"
        elif width == 2:
            return prefix + "\u03A9"  # Greek Omega
        elif width == 4:
            return prefix + "\U0001F40D"  # Snake emoji
        return prefix + "X"


class ETContainerTraverser:
    """
    Unified Container Reference Displacement via ET Binding.
    
    All container types (dict, list, set, tuple) are Descriptor configurations.
    Displacement is T navigating through D_container to rebind references.
    """
    
    @staticmethod
    def process(ref, target, replacement, dry_run, report, target_hashable, replacement_hashable, 
                patch_tuple_fn, depth_limit, visited, queue):
        """
        Process single container - T ∘ D_container operation.
        Returns number of swaps performed.
        """
        swaps = 0
        ref_type = type(ref).__name__
        
        # D_dict binding
        if isinstance(ref, dict):
            for k, v in list(ref.items()):
                if v is target:
                    if not dry_run:
                        ref[k] = replacement
                    report["locations"]["Dict_Value"] += 1
                    swaps += 1
                elif isinstance(v, (dict, list, set)) and id(v) not in visited:
                    queue.append(v)
            
            if target_hashable:
                try:
                    if target in ref:
                        if replacement_hashable:
                            if not dry_run:
                                val = ref.pop(target)
                                ref[replacement] = val
                            report["locations"]["Dict_Key"] += 1
                            swaps += 1
                        else:
                            report["skipped_unhashable"] += 1
                            report["locations"]["Dict_Key_Skipped"] += 1
                            if not any("Dict key swap skipped" in w for w in report["warnings"]):
                                report["warnings"].append(f"Dict key swap skipped: replacement unhashable ({type(replacement).__name__})")
                except TypeError:
                    pass
        
        # D_list binding
        elif isinstance(ref, list):
            for i, v in enumerate(ref):
                if v is target:
                    if not dry_run:
                        ref[i] = replacement
                    report["locations"]["List_Item"] += 1
                    swaps += 1
                elif isinstance(v, (dict, list, set)) and id(v) not in visited:
                    queue.append(v)
        
        # D_set binding
        elif isinstance(ref, set):
            if target_hashable:
                try:
                    if target in ref:
                        if replacement_hashable:
                            if not dry_run:
                                ref.remove(target)
                                ref.add(replacement)
                            report["locations"]["Set_Element"] += 1
                            swaps += 1
                        else:
                            report["skipped_unhashable"] += 1
                            report["locations"]["Set_Element_Skipped"] += 1
                            if not any("Set element swap skipped" in w for w in report["warnings"]):
                                report["warnings"].append(f"Set element swap skipped: replacement unhashable ({type(replacement).__name__})")
                except TypeError:
                    pass
        
        # D_tuple binding (requires T pointer patching)
        elif isinstance(ref, tuple) and ref is not target:
            s = patch_tuple_fn(ref, target, replacement, depth_limit, dry_run, visited)
            if s > 0:
                report["locations"]["Tuple_Recursive"] += s
                swaps += s
        
        # D_instance binding (__dict__ and __slots__)
        elif hasattr(ref, '__dict__') and not isinstance(ref, type):
            try:
                obj_dict = ref.__dict__
                if isinstance(obj_dict, dict) and id(obj_dict) not in visited:
                    queue.append(obj_dict)
                    report["locations"]["Instance_Dict_Queued"] += 1
            except:
                pass
            
            if hasattr(ref.__class__, '__slots__'):
                try:
                    for slot in ref.__class__.__slots__:
                        if hasattr(ref, slot) and getattr(ref, slot) is target:
                            if not dry_run:
                                setattr(ref, slot, replacement)
                            report["locations"]["Slot_Attr"] += 1
                            swaps += 1
                except Exception as e:
                    logger.debug(f"Slot scan error: {e}")
        
        # D_mappingproxy binding (read-only class __dict__)
        elif ref_type == 'mappingproxy':
            try:
                for k, v in ref.items():
                    if v is target:
                        report["locations"]["MappingProxy_Value"] += 1
                        if not dry_run:
                            report["warnings"].append(f"Cannot modify mappingproxy key '{k}' (class attribute)")
            except:
                pass
        
        return swaps


class ETCompendiumSovereign:
    """
    The Compendium Traverser - Fixed via ET Mathematics.
    
    Applies:
    - Structural Density (Eq 211) for geometry detection
    - Traverser Effort (Eq 212) for risk assessment
    - Phase Transition (Eq 30) for confidence scoring
    - Phase-Locking for COW break
    - Unified Descriptor binding for all operations
    
    FIXES:
    1. Multi-tier RO bypass with sandbox-safe fallbacks
    2. Robust UCS2/UCS4 calibration with fallback chains
    3. Comprehensive C-level intern reference displacement
    4. Memory-only cache fallback for read-only environments
    """
    
    def __init__(self, noise_pattern=None, injection_count=None):
        """Initialize the ET Compendium Sovereign."""
        self.os_type = platform.system()
        self.pid = os.getpid()
        self.is_64bit = sys.maxsize > 2**32
        self.ptr_size = 8 if self.is_64bit else 4
        self.pyapi = ctypes.pythonapi
        self._lock = threading.RLock()
        
        # Phase-Lock Descriptor binding
        self._noise_pattern = self._validate_pattern(noise_pattern if noise_pattern is not None else DEFAULT_NOISE_PATTERN)
        self._injection_count = self._validate_count(injection_count if injection_count is not None else DEFAULT_INJECTION_COUNT)
        
        # FIX #4: Memory-only cache for read-only environments
        self._memory_cache = {}
        
        # Geometry calibration via Density Matrix
        self.offsets = self._load_geometry()
        
        # FIX #3: Cache for intern dict to avoid repeated lookups
        self._intern_dict_cache = None
        self._intern_dict_cache_time = 0
        
        # Tunnel initialization (platform-specific D binding)
        self.wormhole = self.win_handle = self.kernel32 = None
        self._init_tunnel()
        
        # FIX #1: Track which RO bypass methods work
        self._working_bypass_tiers = set()
        
        print(f"[ET] Compendium Sovereign Active. Offsets: {self.offsets}")
    
    def _validate_pattern(self, pattern):
        """Validate noise pattern descriptor."""
        if isinstance(pattern, bytes):
            if len(pattern) != 1:
                raise ValueError("noise_pattern bytes must be length 1")
            return pattern[0]
        if isinstance(pattern, int) and 0 <= pattern <= 255:
            return pattern
        raise ValueError("noise_pattern must be int 0-255 or single byte")
    
    def _validate_count(self, count):
        """Validate injection count descriptor."""
        if isinstance(count, int) and count >= 1:
            return count
        raise ValueError("injection_count must be positive integer")
    
    def configure_phase_lock(self, noise_pattern=None, injection_count=None):
        """Configure phase-locking descriptors at runtime."""
        with self._lock:
            if noise_pattern is not None:
                self._noise_pattern = self._validate_pattern(noise_pattern)
            if injection_count is not None:
                if not (1 <= injection_count <= 10):
                    raise ValueError("injection_count must be 1-10")
                self._injection_count = injection_count
            return self.get_phase_lock_config()
    
    def get_phase_lock_config(self):
        """Get current phase-locking descriptor configuration."""
        return {
            "noise_pattern": self._noise_pattern,
            "noise_pattern_hex": f"0x{self._noise_pattern:02X}",
            "noise_pattern_name": PATTERN_NAMES.get(self._noise_pattern, "CUSTOM"),
            "injection_count": self._injection_count
        }
    
    # =========================================================================
    # GEOMETRY CALIBRATION (FIX #4: Graceful Cache Fallback)
    # =========================================================================
    
    def _load_geometry(self):
        """
        Load calibration - T navigating cache hierarchy (shm → env → file → memory → fresh).
        FIX #4: Added memory-only fallback for read-only environments.
        """
        # Priority 1: Shared memory (fastest D binding)
        if HAS_SHARED_MEMORY:
            try:
                shm = shared_memory.SharedMemory(name=ET_SHARED_MEM_NAME)
                raw = bytes(shm.buf[:]).rstrip(b'\x00')
                if raw:
                    geo = json.loads(raw.decode('utf-8'))
                    logger.debug(f"Loaded geometry from shared memory: {ET_SHARED_MEM_NAME}")
                    shm.close()
                    return geo
                shm.close()
            except FileNotFoundError:
                pass
            except Exception as e:
                logger.debug(f"Shared memory read failed: {e}")
        
        # Priority 2: Environment variable (D_env binding)
        env_cache = os.environ.get(ET_CACHE_ENV_VAR)
        if env_cache:
            try:
                geo = json.loads(env_cache)
                logger.debug(f"Loaded geometry from env var: {ET_CACHE_ENV_VAR}")
                return geo
            except Exception as e:
                logger.debug(f"Env var cache parse failed: {e}")
        
        # Priority 3: File cache (D_file binding) - only if CACHE_FILE is available
        if CACHE_FILE:
            try:
                if os.path.exists(CACHE_FILE):
                    with open(CACHE_FILE, 'r') as f:
                        geo = json.load(f)
                        logger.debug(f"Loaded geometry from file: {CACHE_FILE}")
                        return geo
            except Exception as e:
                logger.debug(f"File cache read failed: {e}")
        
        # Priority 4: Memory cache (for this process instance)
        if self._memory_cache:
            logger.debug("Loaded geometry from memory cache")
            return self._memory_cache.copy()
        
        # Priority 5: Fresh calibration
        geo = self._calibrate_all()
        self._memory_cache = geo.copy()
        self._save_geometry_cross_process(geo)
        return geo
    
    def _save_geometry_cross_process(self, geo):
        """
        Save geometry to all available cache backends - multi-target D binding.
        FIX #4: Gracefully skip unavailable backends.
        """
        json_str = json.dumps(geo)
        json_bytes = json_str.encode('utf-8')
        
        # Always save to memory cache
        self._memory_cache = geo.copy()
        
        if HAS_SHARED_MEMORY:
            try:
                try:
                    shm = shared_memory.SharedMemory(name=ET_SHARED_MEM_NAME, create=True, size=ET_SHARED_MEM_SIZE)
                except FileExistsError:
                    shm = shared_memory.SharedMemory(name=ET_SHARED_MEM_NAME)
                shm.buf[:len(json_bytes)] = json_bytes
                shm.buf[len(json_bytes):] = b'\x00' * (ET_SHARED_MEM_SIZE - len(json_bytes))
                shm.close()
                logger.debug(f"Saved geometry to shared memory: {ET_SHARED_MEM_NAME}")
            except Exception as e:
                logger.debug(f"Shared memory write failed (non-fatal): {e}")
        
        try:
            os.environ[ET_CACHE_ENV_VAR] = json_str
            logger.debug(f"Saved geometry to env var: {ET_CACHE_ENV_VAR}")
        except Exception as e:
            logger.debug(f"Env var write failed (non-fatal): {e}")
        
        # FIX #4: Only attempt file write if CACHE_FILE is available
        if CACHE_FILE:
            try:
                fd, tmp_name = tempfile.mkstemp(dir=os.path.dirname(CACHE_FILE), text=True)
                with os.fdopen(fd, 'w') as f:
                    json.dump(geo, f)
                os.replace(tmp_name, CACHE_FILE)
                logger.debug(f"Saved geometry to file: {CACHE_FILE}")
            except Exception as e:
                logger.debug(f"File cache write failed (non-fatal): {e}")
    
    def get_cache_info(self):
        """Get cache state - D_cache field inspection."""
        info = {"shared_memory_available": HAS_SHARED_MEMORY, "env_var_name": ET_CACHE_ENV_VAR,
                "file_path": CACHE_FILE, "file_path_available": CACHE_FILE is not None,
                "memory_cache_active": bool(self._memory_cache), "backends": {}}
        
        if HAS_SHARED_MEMORY:
            try:
                shm = shared_memory.SharedMemory(name=ET_SHARED_MEM_NAME)
                info["backends"]["shared_memory"] = {"status": "active", "name": ET_SHARED_MEM_NAME, "size": shm.size}
                shm.close()
            except FileNotFoundError:
                info["backends"]["shared_memory"] = {"status": "not_created"}
            except Exception as e:
                info["backends"]["shared_memory"] = {"status": "error", "error": str(e)}
        else:
            info["backends"]["shared_memory"] = {"status": "unavailable", "reason": "Python < 3.8"}
        
        env_val = os.environ.get(ET_CACHE_ENV_VAR)
        info["backends"]["env_var"] = {"status": "set" if env_val else "not_set", "length": len(env_val) if env_val else 0}
        
        if CACHE_FILE:
            if os.path.exists(CACHE_FILE):
                info["backends"]["file"] = {"status": "exists", "size": os.path.getsize(CACHE_FILE), "mtime": os.path.getmtime(CACHE_FILE)}
            else:
                info["backends"]["file"] = {"status": "not_exists"}
        else:
            info["backends"]["file"] = {"status": "unavailable", "reason": "read-only tempdir"}
        
        info["backends"]["memory"] = {"status": "active" if self._memory_cache else "empty"}
        
        return info
    
    def _calibrate_all(self):
        """
        Full calibration via ET Density Matrix - T scanning all D configurations.
        FIX #2: Enhanced with robust fallback chains for UCS2/UCS4.
        """
        fallbacks = {
            '1': 48 if self.is_64bit else 24,
            '2': 48 if self.is_64bit else 24,
            '4': 48 if self.is_64bit else 24
        }
        
        # ET Eq 211: Structural Density for geometry detection
        s_compact = "ET_DENSITY_TEST"
        s_pointer = "X" * 10000
        rho_compact = ETMath.density(len(s_compact), sys.getsizeof(s_compact))
        rho_pointer = ETMath.density(len(s_pointer), sys.getsizeof(s_pointer))
        density_ratio = rho_pointer / rho_compact if rho_compact > 0 else 1.0
        geometry_mode = "pointer" if density_ratio > 3.0 else "compact"
        
        logger.debug(f"Geometry detection: compact_ρ={rho_compact:.3f}, pointer_ρ={rho_pointer:.3f}, "
                    f"ratio={density_ratio:.2f}, mode={geometry_mode}")
        
        # FIX #2: Multi-tier calibration with fallbacks for each width
        offsets = {}
        
        for width in [1, 2, 4]:
            # Tier 1: Unified multi-beacon scan
            offset = self._scan_offset_unified(width)
            
            # Tier 2: Simple single-beacon scan (from old version)
            if offset is None:
                logger.debug(f"Width {width}: Unified scan failed, trying simple scan")
                offset = self._scan_offset_simple_legacy(width)
            
            # Tier 3: Multiple simple beacon attempts
            if offset is None:
                logger.debug(f"Width {width}: Simple scan failed, trying multiple beacons")
                offset = self._scan_offset_multi_simple(width)
            
            # Tier 4: Use fallback
            if offset is None:
                logger.warning(f"Width {width}: All scans failed, using fallback {fallbacks[str(width)]}")
                offset = fallbacks[str(width)]
            
            offsets[str(width)] = offset
        
        # Additional calibrations
        offsets.update({
            'bytes': self._scan_offset_simple(b"ET_BYTES_BEACON", 1, True) or 32,
            'tuple': 24 if self.is_64bit else 12,
            'code': self._calibrate_code_offset(),
            'func': self._calibrate_func_offset(),
            'ob_type': self._calibrate_type_offset(),
            'hash': 24 if self.is_64bit else 12,  # String hash offset
            'geometry': geometry_mode,
            'density_ratio': round(density_ratio, 3)
        })
        
        return offsets
    
    def _scan_offset_unified(self, width):
        """
        Unified offset scanning - T navigating beacon field.
        Uses ETBeaconField for beacon generation (D_beacon pool).
        FIX #2: Lowered confidence threshold and improved error handling.
        """
        beacons = ETBeaconField.generate(width, 50)
        offset_votes = collections.defaultdict(int)
        valid_beacons = 0
        
        for beacon in beacons:
            target = ETMath.encode_width(beacon, width)
            if target is None:
                continue
            
            valid_beacons += 1
            density = ETMath.density(len(beacon), sys.getsizeof(beacon))
            
            p_base = id(beacon)
            c_ptr = ctypes.cast(p_base, ctypes.POINTER(ctypes.c_ubyte))
            
            for offset in range(8, MAX_SCAN_WIDTH):
                try:
                    if all(c_ptr[offset + k] == target[k] for k in range(len(target))):
                        offset_votes[offset] += 1
                        logger.debug(f"Beacon '{beacon[:10]}...' (ρ={density:.3f}) found at offset {offset}")
                        break
                except:
                    break
        
        if offset_votes and valid_beacons > 0:
            best = max(offset_votes, key=offset_votes.get)
            confidence = offset_votes[best] / valid_beacons
            
            # Use phase transition sigmoid for confidence scoring (Eq 30)
            phase_confidence = ETMath.phase_transition(confidence * 10, threshold=1.0)
            
            logger.debug(f"Width {width}: Best offset={best}, votes={offset_votes[best]}/{valid_beacons}, "
                        f"raw_conf={confidence:.1%}, phase_conf={phase_confidence:.3f}")
            
            # FIX #2: Lower threshold (10% instead of 20%) for UCS2/UCS4
            if confidence >= 0.10:
                return best
            logger.warning(f"Width {width}: Low confidence ({confidence:.1%})")
        
        return None
    
    def _scan_offset_simple_legacy(self, width):
        """
        Simple single-beacon offset scan - restored from old version.
        FIX #2: This is the reliable fallback for UCS2/UCS4.
        """
        beacon = ETBeaconField.generate_simple("ET_", width)
        target = ETMath.encode_width(beacon, width)
        
        if target is None:
            return None
        
        p_base = id(beacon)
        c_ptr = ctypes.cast(p_base, ctypes.POINTER(ctypes.c_ubyte))
        
        for i in range(16, MAX_SCAN_WIDTH):
            try:
                match = True
                for k in range(len(target)):
                    if c_ptr[i + k] != target[k]:
                        match = False
                        break
                if match:
                    logger.debug(f"Legacy scan width {width}: Found offset {i}")
                    return i
            except:
                break
        
        return None
    
    def _scan_offset_multi_simple(self, width):
        """
        Try multiple simple beacons - additional fallback tier.
        FIX #2: Tries several different beacon patterns.
        """
        prefixes = ["ET_A", "ET_B", "ET_X", "TEST_", "SCAN_"]
        
        for prefix in prefixes:
            beacon = ETBeaconField.generate_simple(prefix, width)
            target = ETMath.encode_width(beacon, width)
            
            if target is None:
                continue
            
            p_base = id(beacon)
            c_ptr = ctypes.cast(p_base, ctypes.POINTER(ctypes.c_ubyte))
            
            for i in range(16, MAX_SCAN_WIDTH):
                try:
                    if all(c_ptr[i + k] == target[k] for k in range(len(target))):
                        logger.debug(f"Multi-simple scan width {width} prefix '{prefix}': Found offset {i}")
                        return i
                except:
                    break
        
        return None
    
    def _scan_offset_simple(self, beacon, width, is_bytes=False):
        """Simple single-beacon offset scan - legacy T navigation."""
        p_base = id(beacon)
        target = beacon if is_bytes else ETMath.encode_width(beacon, width)
        if target is None:
            return None
        
        c_ptr = ctypes.cast(p_base, ctypes.POINTER(ctypes.c_ubyte))
        for i in range(16, MAX_SCAN_WIDTH):
            try:
                if all(c_ptr[i + k] == target[k] for k in range(len(target))):
                    return i
            except Exception as e:
                logger.debug(f"Scan terminated at offset {i}: {e}")
                break
        return None
    
    def _calibrate_code_offset(self):
        """Calibrate PyCodeObject->co_code - T locating bytecode D_field."""
        def beacon():
            return 0xDEADBEEF
        try:
            code_obj = beacon.__code__
            code_bytes = code_obj.co_code
            if len(code_bytes) < 2:
                logger.debug("Beacon bytecode too short")
                return 96 if self.is_64bit else 48
            
            code_base = id(code_obj)
            c_ptr = ctypes.cast(code_base, ctypes.POINTER(ctypes.c_ubyte))
            target = code_bytes[:min(len(code_bytes), 8)]
            
            for i in range(16, 256):
                try:
                    if all(c_ptr[i + k] == target[k] for k in range(len(target))):
                        logger.debug(f"Code object offset detected: {i}")
                        return i
                except:
                    break
        except Exception as e:
            logger.warning(f"Code offset calibration failed: {e}")
        
        fallback = 96 if self.is_64bit else 48
        logger.debug(f"Using fallback code offset: {fallback}")
        return fallback
    
    def _calibrate_func_offset(self):
        """Calibrate PyFunctionObject->func_code - T locating code pointer."""
        def beacon():
            pass
        try:
            code_id = id(beacon.__code__)
            code_id_bytes = code_id.to_bytes(self.ptr_size, 'little')
            func_base = id(beacon)
            c_ptr = ctypes.cast(func_base, ctypes.POINTER(ctypes.c_ubyte))
            
            for i in range(self.ptr_size, 128, self.ptr_size):
                try:
                    if all(c_ptr[i + k] == code_id_bytes[k] for k in range(self.ptr_size)):
                        logger.debug(f"Function code offset detected: {i}")
                        return i
                except:
                    break
        except Exception as e:
            logger.warning(f"Function offset calibration failed: {e}")
        
        fallback = 16 if self.is_64bit else 12
        logger.debug(f"Using fallback func offset: {fallback}")
        return fallback
    
    def _calibrate_type_offset(self):
        """Calibrate PyObject->ob_type - T locating type pointer."""
        try:
            obj = object()
            type_id_bytes = id(type(obj)).to_bytes(self.ptr_size, 'little')
            obj_base = id(obj)
            c_ptr = ctypes.cast(obj_base, ctypes.POINTER(ctypes.c_ubyte))
            
            # Standard location check first (fast path)
            standard = self.ptr_size
            try:
                if all(c_ptr[standard + k] == type_id_bytes[k] for k in range(self.ptr_size)):
                    logger.debug(f"Type offset verified at standard location: {standard}")
                    return standard
            except:
                pass
            
            # Fallback scan
            for i in range(0, 64, self.ptr_size):
                try:
                    if all(c_ptr[i + k] == type_id_bytes[k] for k in range(self.ptr_size)):
                        logger.debug(f"Type offset detected at non-standard location: {i}")
                        return i
                except:
                    break
        except Exception as e:
            logger.warning(f"Type offset calibration failed: {e}")
        
        return self.ptr_size
    
    # =========================================================================
    # KERNEL TUNNEL & PHASE LOCKING (FIX #1: Multi-tier RO Bypass)
    # =========================================================================
    
    def _init_tunnel(self):
        """Initialize kernel tunnel - platform-specific D binding."""
        try:
            if self.os_type == 'Linux':
                mem_path = f"/proc/{self.pid}/mem"
                if os.access(mem_path, os.W_OK):
                    self.wormhole = open(mem_path, "rb+", buffering=0)
                    logger.debug("Linux wormhole initialized successfully")
                else:
                    logger.debug("Linux mem file not writable (sandbox?)")
            elif self.os_type == 'Windows':
                self.kernel32 = ctypes.windll.kernel32
                # FIX #1: Try multiple access flag combinations
                for access_flags in [0x0038, 0x001F, 0x0010]:
                    try:
                        handle = self.kernel32.OpenProcess(access_flags, False, self.pid)
                        if handle:
                            self.win_handle = handle
                            logger.debug(f"Windows handle opened with flags 0x{access_flags:04X}")
                            break
                    except:
                        continue
        except Exception as e:
            logger.warning(f"Tunnel initialization failed: {e}")
    
    def _tunnel_write(self, address, data):
        """
        Phase-Locking Tunnel Write - T ∘ D_memory binding with noise injection.
        Breaks COW lock via configurable noise pattern before signal write.
        Thread-safe via RLock.
        """
        with self._lock:
            try:
                noise = None if self._noise_pattern == 0x00 else (data[0] ^ self._noise_pattern).to_bytes(1, 'little')
                
                if self.wormhole:
                    try:
                        if noise:
                            for _ in range(self._injection_count):
                                self.wormhole.seek(address)
                                self.wormhole.write(noise)
                        self.wormhole.seek(address)
                        self.wormhole.write(data)
                        self._working_bypass_tiers.add("TUNNEL_PHASE_LOCK")
                        return True
                    except (OSError, IOError) as e:
                        logger.debug(f"Linux tunnel write failed: {e}")
                
                if self.win_handle:
                    try:
                        written = ctypes.c_size_t(0)
                        if noise:
                            for _ in range(self._injection_count):
                                self.kernel32.WriteProcessMemory(
                                    self.win_handle, ctypes.c_void_p(address), 
                                    noise, 1, ctypes.byref(written))
                        result = self.kernel32.WriteProcessMemory(
                            self.win_handle, ctypes.c_void_p(address), data, 
                            ctypes.c_size_t(len(data)), ctypes.byref(written))
                        if result != 0:
                            self._working_bypass_tiers.add("TUNNEL_PHASE_LOCK")
                            return True
                    except Exception as e:
                        logger.debug(f"Windows tunnel write failed: {e}")
            except Exception as e:
                logger.error(f"Tunnel write failed at 0x{address:X}: {e}")
            return False
    
    # =========================================================================
    # FIX #1: MULTI-TIER RO BYPASS (Restored + Enhanced from old version)
    # =========================================================================
    
    def _make_page_writable(self, address, size):
        """Make memory page writable - OS-level D_protection binding."""
        try:
            page_size = mmap.PAGESIZE
            page_start = (address // page_size) * page_size
            page_count = ((address + size + page_size - 1) // page_size * page_size - page_start) // page_size
            
            if self.os_type == 'Windows':
                old_protect = ctypes.c_ulong()
                result = ctypes.windll.kernel32.VirtualProtect(
                    ctypes.c_void_p(page_start), ctypes.c_size_t(page_count * page_size),
                    PAGE['EXEC_READWRITE'], ctypes.byref(old_protect))
                if result:
                    logger.debug(f"VirtualProtect: 0x{page_start:X} -> RWX (was 0x{old_protect.value:X})")
                    return (True, old_protect.value, page_start)
                logger.debug(f"VirtualProtect failed at 0x{page_start:X}")
                return (False, 0, page_start)
            else:
                try:
                    libc = ctypes.CDLL(None)
                    mprotect = libc.mprotect
                    mprotect.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
                    mprotect.restype = ctypes.c_int
                    
                    result = mprotect(ctypes.c_void_p(page_start), page_count * page_size,
                                      PROT['READ'] | PROT['WRITE'] | PROT['EXEC'])
                    if result == 0:
                        logger.debug(f"mprotect: 0x{page_start:X} -> RWX")
                        return (True, PROT['READ'], page_start)
                    logger.debug(f"mprotect failed at 0x{page_start:X}: {result}")
                except Exception as e:
                    logger.debug(f"mprotect call failed: {e}")
                return (False, 0, page_start)
        except Exception as e:
            logger.debug(f"Page protection change failed: {e}")
            return (False, 0, 0)
    
    def _restore_page_protection(self, page_start, old_protection, size):
        """Restore original memory protection."""
        try:
            page_size = mmap.PAGESIZE
            page_count = ((size + page_size - 1) // page_size)
            
            if self.os_type == 'Windows':
                dummy = ctypes.c_ulong()
                ctypes.windll.kernel32.VirtualProtect(
                    ctypes.c_void_p(page_start), ctypes.c_size_t(page_count * page_size),
                    old_protection, ctypes.byref(dummy))
                logger.debug(f"VirtualProtect: restored 0x{page_start:X} to 0x{old_protection:X}")
            else:
                libc = ctypes.CDLL(None)
                mprotect = libc.mprotect
                mprotect.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
                mprotect(ctypes.c_void_p(page_start), page_count * page_size, old_protection)
        except Exception as e:
            logger.debug(f"Page protection restore failed: {e}")
    
    def _ctypes_pointer_write(self, address, data):
        """
        FIX #1: Direct ctypes pointer cast write (Tier 2.7).
        Works when mprotect fails but memory is actually writable.
        """
        try:
            ptr = ctypes.cast(address, ctypes.POINTER(ctypes.c_ubyte * len(data)))
            for i, b in enumerate(data):
                ptr.contents[i] = b
            self._working_bypass_tiers.add("CTYPES_POINTER_CAST")
            return True
        except Exception as e:
            logger.debug(f"ctypes pointer write failed: {e}")
            return False
    
    def _pyobject_structure_write(self, target_obj, offset, data):
        """
        FIX #1: PyObject structure manipulation (Tier 2.8).
        Uses ctypes.py_object to access PyObject fields directly.
        """
        try:
            obj_ptr = ctypes.cast(id(target_obj), ctypes.POINTER(ctypes.c_ubyte))
            for i, b in enumerate(data):
                obj_ptr[offset + i] = b
            self._working_bypass_tiers.add("PYOBJECT_STRUCTURE")
            return True
        except Exception as e:
            logger.debug(f"PyObject structure write failed: {e}")
            return False
    
    def _multi_tier_write(self, target_obj, offset, data):
        """
        FIX #1: Multi-tier write with all available bypass methods.
        Returns (success, tier_used) tuple.
        """
        address = id(target_obj) + offset
        
        # Tier 1: Tunnel (with phase-locking)
        if self._tunnel_write(address, data):
            return (True, "TUNNEL_PHASE_LOCK")
        
        # Tier 2: Direct memmove (if already writable)
        if self._safety_probe(address, len(data)):
            try:
                ctypes.memmove(address, data, len(data))
                self._working_bypass_tiers.add("DIRECT_MEMMOVE")
                return (True, "DIRECT_MEMMOVE")
            except:
                pass
        
        # Tier 2.5: mprotect/VirtualProtect + memmove
        success, old_prot, page_start = self._make_page_writable(address, len(data))
        if success:
            try:
                ctypes.memmove(address, data, len(data))
                self._working_bypass_tiers.add("MPROTECT_DIRECT")
                return (True, "MPROTECT_DIRECT")
            finally:
                self._restore_page_protection(page_start, old_prot, len(data))
        
        # Tier 2.7: ctypes pointer cast
        if self._ctypes_pointer_write(address, data):
            return (True, "CTYPES_POINTER_CAST")
        
        # Tier 2.8: PyObject structure manipulation
        if self._pyobject_structure_write(target_obj, offset, data):
            return (True, "PYOBJECT_STRUCTURE")
        
        # All direct writes failed
        return (False, None)
    
    # =========================================================================
    # FIX #3: C-LEVEL INTERN POOL ACCESS (Enhanced)
    # =========================================================================
    
    def _get_intern_dict(self, force_refresh=False):
        """
        Access CPython's internal string interning dictionary.
        FIX #3: Enhanced with caching and multiple detection strategies.
        """
        # Use cached result if recent (within 1 second)
        if not force_refresh and self._intern_dict_cache is not None:
            if time.time() - self._intern_dict_cache_time < 1.0:
                return self._intern_dict_cache
        
        intern_dict = None
        
        try:
            # Strategy 1: Find via interned string referrers
            test_str = "ET_INTERN_PROBE_" + str(id(self))
            interned = sys.intern(test_str)
            
            for referrer in gc.get_referrers(interned):
                if isinstance(referrer, dict) and len(referrer) > 100:
                    # Check if this looks like the intern dict
                    str_count = sum(1 for k in list(referrer.keys())[:100] if isinstance(k, str))
                    if str_count > 80:
                        intern_dict = referrer
                        logger.debug(f"Found intern dict via referrers (Strategy 1): {len(referrer)} entries")
                        break
            
            # Strategy 2: Check common identifiers
            if intern_dict is None:
                common_interned = ['__name__', '__main__', 'self', 'None', 'True', 'False']
                for common in common_interned:
                    interned = sys.intern(common)
                    for referrer in gc.get_referrers(interned):
                        if isinstance(referrer, dict) and len(referrer) > 100:
                            str_count = sum(1 for k in list(referrer.keys())[:100] if isinstance(k, str))
                            if str_count > 80:
                                intern_dict = referrer
                                logger.debug(f"Found intern dict via '{common}' (Strategy 2): {len(referrer)} entries")
                                break
                    if intern_dict:
                        break
            
            # Strategy 3: Scan all gc objects for large string dicts
            if intern_dict is None:
                for obj in gc.get_objects():
                    if isinstance(obj, dict) and len(obj) > 500:
                        try:
                            sample = list(obj.keys())[:100]
                            str_count = sum(1 for k in sample if isinstance(k, str))
                            if str_count > 90:
                                # Verify it contains known interned strings
                                if any(s in obj for s in common_interned):
                                    intern_dict = obj
                                    logger.debug(f"Found intern dict via gc scan (Strategy 3): {len(obj)} entries")
                                    break
                        except:
                            pass
        
        except Exception as e:
            logger.debug(f"Intern dict detection failed: {e}")
        
        self._intern_dict_cache = intern_dict
        self._intern_dict_cache_time = time.time()
        return intern_dict
    
    def _get_all_interned_refs(self, target):
        """
        Get all references including C-level intern pool.
        FIX #3: More comprehensive reference gathering.
        """
        observers = list(gc.get_referrers(target))
        observer_ids = set(id(o) for o in observers)
        
        # Add intern dict if target is a string
        if isinstance(target, str):
            intern_dict = self._get_intern_dict()
            if intern_dict is not None and id(intern_dict) not in observer_ids:
                observers.append(intern_dict)
                observer_ids.add(id(intern_dict))
                logger.debug("Added intern dict to observer list")
            
            # Check if target is interned
            try:
                if sys.intern(target) is target:
                    logger.debug(f"Target is an interned string: '{target[:30]}...'")
                    
                    # Also scan global identifier dictionaries
                    for mod in list(sys.modules.values()):
                        if mod is None:
                            continue
                        try:
                            if hasattr(mod, '__dict__'):
                                md = mod.__dict__
                                if id(md) not in observer_ids:
                                    # Check if target appears as key or value
                                    if target in md or target in md.values():
                                        observers.append(md)
                                        observer_ids.add(id(md))
                        except:
                            pass
            except:
                pass
        
        return observers
    
    def _check_c_interned(self, target):
        """Check if target is C-level interned - immutability analysis."""
        result = {"is_interned": False, "is_c_interned": False, "intern_type": None, 
                  "warnings": [], "modifiable": True, "refcount": 0}
        
        if not isinstance(target, str):
            return result
        
        try:
            interned_version = sys.intern(target)
            result["is_interned"] = interned_version is target
            result["refcount"] = sys.getrefcount(target)
            
            if result["is_interned"]:
                # C-interned detection hierarchy
                if len(target) == 0:
                    result.update(is_c_interned=True, intern_type="EMPTY_STRING", modifiable=False)
                    result["warnings"].append("Empty string is C-interned (immortal)")
                elif len(target) == 1 and ord(target) < 128:
                    itype = "DIGIT_CHAR" if target.isdigit() else "ASCII_CHAR"
                    result.update(is_c_interned=True, intern_type=itype, modifiable=False)
                    result["warnings"].append(f"{'Digit' if target.isdigit() else 'Single ASCII char'} '{target}' is C-interned")
                elif target in {'None', 'True', 'False', 'self', 'cls', '__name__', '__main__', '__init__', 
                               '__new__', '__del__', '__repr__', '__str__', '__dict__', '__class__',
                               '__doc__', '__module__', '__file__', '__builtins__', '__spec__',
                               '__loader__', '__package__', '__cached__', '__annotations__'}:
                    result.update(is_c_interned=True, intern_type="BUILTIN_IDENTIFIER", modifiable=False)
                    result["warnings"].append(f"'{target}' is a builtin identifier (likely C-interned)")
                else:
                    result["intern_type"] = "USER_INTERNED"
                
                # High refcount check (Python 3.12+ immortal objects)
                refcount = sys.getrefcount(target)
                if refcount > 0xFFFFFFF0 or refcount > 100000:
                    result["is_c_interned"] = True
                    result["modifiable"] = False
                    result["warnings"].append(f"String has {refcount} references (likely immortal)")
                    if result["intern_type"] == "USER_INTERNED":
                        result["intern_type"] = "HIGH_REFCOUNT"
        except Exception as e:
            logger.debug(f"C-interned check failed: {e}")
        
        return result
    
    # =========================================================================
    # HOLOGRAPHIC DISPLACEMENT (FIX #3: Enhanced C-Level Reference Handling)
    # =========================================================================
    
    def _displace_references(self, target, replacement, dry_run=False, depth_limit=3):
        """
        Holographic Displacement - T navigating all D_container configurations.
        Uses ETContainerTraverser for unified container handling.
        FIX #3: Enhanced C-level intern reference displacement.
        Thread-safe via RLock.
        """
        with self._lock:
            report = {
                "status": "SIMULATION" if dry_run else "EXECUTED",
                "swaps": 0, "effort": 0.0,
                "locations": collections.defaultdict(int),
                "warnings": [], "scanned_containers": 0,
                "skipped_unhashable": 0, "intern_info": None,
                "c_interned_detected": False,
                "intern_dict_found": False
            }
            
            # C-interned detection
            intern_check = self._check_c_interned(target)
            report["intern_info"] = intern_check
            
            if intern_check["is_c_interned"]:
                report["c_interned_detected"] = True
                for warning in intern_check["warnings"]:
                    report["warnings"].append(f"C-INTERNED: {warning}")
                if not intern_check["modifiable"]:
                    report["warnings"].append(
                        "Target may be immortal (C-interned). Displacement will update Python-level refs "
                        "but original may persist in C structures.")
            
            # Hashability checks
            target_hashable = replacement_hashable = False
            try:
                hash(target)
                target_hashable = True
            except TypeError:
                pass
            try:
                hash(replacement)
                replacement_hashable = True
            except TypeError:
                pass
            logger.debug(f"Hashability: target={target_hashable}, replacement={replacement_hashable}")
            
            # FIX #3: Comprehensive reference collection with intern dict
            observers = self._get_all_interned_refs(target)
            observer_ids = set(id(o) for o in observers)
            
            # Check if intern dict was found
            intern_dict = self._get_intern_dict()
            if intern_dict is not None and id(intern_dict) in observer_ids:
                report["intern_dict_found"] = True
            
            # Module globals scan
            for mod_name, mod in list(sys.modules.items()):
                if mod is None:
                    continue
                try:
                    if hasattr(mod, "__dict__"):
                        mod_dict = mod.__dict__
                        if id(mod_dict) not in observer_ids:
                            observers.append(mod_dict)
                            observer_ids.add(id(mod_dict))
                    if hasattr(mod, "__all__") and isinstance(mod.__all__, list):
                        if id(mod.__all__) not in observer_ids:
                            observers.append(mod.__all__)
                            observer_ids.add(id(mod.__all__))
                except Exception as e:
                    logger.debug(f"Module scan error for {mod_name}: {e}")
            
            # Globals and stack frames scan
            try:
                g = globals()
                if id(g) not in observer_ids:
                    observers.append(g)
                    observer_ids.add(id(g))
                frame = inspect.currentframe()
                while frame:
                    for d in (frame.f_locals, frame.f_globals):
                        if id(d) not in observer_ids:
                            observers.append(d)
                            observer_ids.add(id(d))
                    frame = frame.f_back
            except Exception as e:
                logger.debug(f"Globals/stack scan error: {e}")
            
            # Class registry scan
            try:
                for obj in gc.get_objects():
                    if isinstance(obj, type):
                        try:
                            class_dict = obj.__dict__
                            if hasattr(class_dict, 'items') and id(class_dict) not in observer_ids:
                                observers.append(class_dict)
                                observer_ids.add(id(class_dict))
                        except:
                            pass
            except Exception as e:
                logger.debug(f"Class registry scan error: {e}")
            
            # FIX #3: Also scan builtin module dicts
            try:
                import builtins
                if id(builtins.__dict__) not in observer_ids:
                    observers.append(builtins.__dict__)
                    observer_ids.add(id(builtins.__dict__))
            except:
                pass
            
            # Cycle detection via visited set
            visited = {id(report), id(target), id(replacement), id(observers)}
            swaps_count = containers_scanned = 0
            queue = list(observers)
            
            while queue:
                ref = queue.pop(0)
                if id(ref) in visited:
                    continue
                visited.add(id(ref))
                containers_scanned += 1
                
                swaps_count += ETContainerTraverser.process(
                    ref, target, replacement, dry_run, report,
                    target_hashable, replacement_hashable,
                    self._patch_tuple_recursive, depth_limit, visited, queue
                )
            
            report["swaps"] = swaps_count
            report["scanned_containers"] = containers_scanned
            report["effort"] = ETMath.effort(len(observers), swaps_count)
            
            # FIX #3: Add intern dict displacement if no swaps found
            if swaps_count == 0 and intern_check["is_interned"] and not dry_run:
                intern_dict = self._get_intern_dict()
                if intern_dict is not None:
                    # Try to replace in intern dict directly
                    try:
                        if target in intern_dict:
                            if replacement_hashable:
                                val = intern_dict.pop(target)
                                intern_dict[replacement] = val
                                report["locations"]["Intern_Dict"] += 1
                                report["swaps"] += 1
                                swaps_count += 1
                                logger.debug("Displaced reference in intern dict")
                    except Exception as e:
                        logger.debug(f"Intern dict displacement failed: {e}")
            
            return report
    
    def _patch_tuple_recursive(self, tpl, target, replacement, depth, dry_run, visited):
        """Patch tuple elements via pointer manipulation - T ∘ D_tuple binding."""
        if depth <= 0 or id(tpl) in visited:
            return 0
        visited.add(id(tpl))
        
        swaps = 0
        offset = self.offsets['tuple']
        
        for i, elem in enumerate(tpl):
            if elem is target:
                if not dry_run:
                    addr = id(tpl) + offset + i * self.ptr_size
                    new_ptr = id(replacement).to_bytes(self.ptr_size, 'little')
                    self.pyapi.Py_IncRef(ctypes.py_object(replacement))
                    
                    # FIX #1: Use multi-tier write
                    success, tier = self._multi_tier_write(tpl, offset + i * self.ptr_size, new_ptr)
                    if success:
                        self.pyapi.Py_DecRef(ctypes.py_object(target))
                        logger.debug(f"Tuple element patched via {tier}")
                    else:
                        # Fallback: try original method
                        if self._safety_probe(addr, self.ptr_size):
                            try:
                                ctypes.memmove(addr, new_ptr, self.ptr_size)
                                self.pyapi.Py_DecRef(ctypes.py_object(target))
                            except:
                                self.pyapi.Py_DecRef(ctypes.py_object(replacement))
                                continue
                        else:
                            self.pyapi.Py_DecRef(ctypes.py_object(replacement))
                            continue
                swaps += 1
            elif isinstance(elem, tuple):
                swaps += self._patch_tuple_recursive(elem, target, replacement, depth - 1, dry_run, visited)
        
        return swaps
    
    # =========================================================================
    # TRANSMUTATION ENGINE (FIX #1: Multi-Tier RO Bypass)
    # =========================================================================
    
    def transmute(self, target, replacement, dry_run=False):
        """
        Core Transmutation - P ∘ D ∘ T = E (Master Equation applied).
        
        Tiered approach (FIX #1: Enhanced with more bypass options):
        - BUFFER: Mutable bytearray direct modification
        - TUNNEL (Tier 1): Phase-locked kernel write
        - DIRECT (Tier 2): ctypes.memmove
        - MPROTECT_DIRECT (Tier 2.5): mprotect/VirtualProtect + memmove
        - CTYPES_POINTER (Tier 2.7): Direct pointer manipulation
        - PYOBJECT_STRUCTURE (Tier 2.8): PyObject structure manipulation
        - DISPLACEMENT (Tier 3): Holographic reference replacement
        """
        with self._lock:
            gc.disable()
            try:
                is_bytes = isinstance(target, (bytes, bytearray))
                
                # BUFFER tier: Mutable bytearray
                if isinstance(target, bytearray):
                    byte_delta = len(replacement)
                    effort = ETMath.effort(1, byte_delta)
                    if dry_run:
                        return {"status": "SIMULATION", "method": "BUFFER", "swaps": 1, 
                                "byte_delta": byte_delta, "effort": effort}
                    target[:] = replacement
                    return {"status": "COMPLETE", "method": "BUFFER", "swaps": 1,
                            "byte_delta": byte_delta, "effort": effort}
                
                # Determine geometry and payload
                width = 1
                if is_bytes:
                    offset = self.offsets['bytes']
                    payload = replacement
                else:
                    max_char = max(ord(c) for c in target) if target else 0
                    width = 4 if max_char > 65535 else 2 if max_char > 255 else 1
                    offset = self.offsets.get(str(width))
                    payload = ETMath.encode_width(replacement, width)
                
                if not offset:
                    return {"status": "ERROR", "method": "NONE", "msg": "Uncalibrated offset", "swaps": 0, "effort": 0.0}
                
                if payload is None:
                    return {"status": "ERROR", "method": "NONE", "msg": "Payload encoding failed", "swaps": 0, "effort": 0.0}
                
                if dry_run:
                    return self._displace_references(target, replacement, dry_run=True)
                
                # TIER 1-2.8: Direct write (if length matches)
                phy_len = len(target) * (1 if is_bytes else width)
                if len(payload) == phy_len:
                    p_base = id(target)
                    p_data = p_base + offset
                    byte_delta = len(payload)
                    
                    # FIX #1: Use multi-tier write
                    success, tier = self._multi_tier_write(target, offset, payload)
                    if success:
                        self._blind_hash_reset(target)
                        if self._verify(p_data, payload):
                            return {"status": "COMPLETE", "method": tier, 
                                    "tier": RO_BYPASS_TIERS.index(tier) + 1 if tier in RO_BYPASS_TIERS else "?",
                                    "swaps": 1, "byte_delta": byte_delta, 
                                    "effort": ETMath.effort(1, byte_delta),
                                    "phase_lock_config": self.get_phase_lock_config() if "TUNNEL" in tier else None,
                                    "address": f"0x{p_data:X}",
                                    "working_bypass_tiers": list(self._working_bypass_tiers)}
                
                # TIER 3: Displacement
                report = self._displace_references(target, replacement)
                report["method"] = "DISPLACEMENT"
                report["tier"] = 3
                report["working_bypass_tiers"] = list(self._working_bypass_tiers)
                return report
            finally:
                gc.enable()
    
    def _verify(self, addr, expected):
        """Verify write - D_memory confirmation."""
        try:
            return ctypes.string_at(addr, len(expected)) == expected
        except Exception as e:
            logger.debug(f"Verify read failed at 0x{addr:X}: {e}")
            return False
    
    def _safety_probe(self, addr, ln):
        """Safety probe - D_memory accessibility check."""
        try:
            ctypes.string_at(addr, 1)
            ctypes.string_at(addr + ln - 1, 1)
            return True
        except Exception as e:
            logger.debug(f"Safety probe failed at 0x{addr:X} (len={ln}): {e}")
            return False
    
    def _blind_hash_reset(self, target):
        """Reset string hash cache - D_hash invalidation."""
        try:
            hash_off = self.offsets.get('hash', 24 if self.is_64bit else 12)
            ctypes.cast(id(target) + hash_off, ctypes.POINTER(ctypes.c_ssize_t)).contents.value = -1
        except Exception as e:
            logger.debug(f"Hash reset failed for object at 0x{id(target):X}: {e}")
    
    # =========================================================================
    # HELPER METHODS: High-Level Manipulation Operations
    # =========================================================================
    
    def replace_bytecode(self, func, new_bytecode):
        """Replace function bytecode - T ∘ D_bytecode binding."""
        with self._lock:
            gc.disable()
            try:
                if not callable(func):
                    return {"status": "ERROR", "msg": "Target must be callable"}
                if not hasattr(func, '__code__'):
                    return {"status": "ERROR", "msg": "Target has no __code__ attribute"}
                if not isinstance(new_bytecode, (bytes, bytearray)):
                    return {"status": "ERROR", "msg": "new_bytecode must be bytes"}
                
                code_obj = func.__code__
                original = code_obj.co_code
                
                if len(new_bytecode) != len(original):
                    return {"status": "ERROR", "msg": f"Length mismatch: original={len(original)}, new={len(new_bytecode)}",
                            "hint": "Bytecode must be exactly the same length"}
                
                code_offset = self.offsets.get('code')
                if not code_offset:
                    return {"status": "ERROR", "msg": "Code offset not calibrated"}
                
                bytecode_addr = id(code_obj) + code_offset
                
                # FIX #1: Use multi-tier write
                success, tier = self._multi_tier_write(code_obj, code_offset, bytes(new_bytecode))
                if success:
                    if self._verify(bytecode_addr, bytes(new_bytecode)):
                        logger.debug(f"Bytecode replaced via {tier} at 0x{bytecode_addr:X}")
                        return {"status": "COMPLETE", "method": tier, "address": f"0x{bytecode_addr:X}"}
                
                return {"status": "ERROR", "msg": "Failed to write bytecode (memory protected)",
                        "hint": "May require elevated privileges or different OS",
                        "tried_tiers": list(self._working_bypass_tiers)}
            finally:
                gc.enable()
    
    def replace_function(self, old_func, new_func):
        """Replace function references - T ∘ D_function holographic displacement."""
        with self._lock:
            gc.disable()
            try:
                if not callable(old_func):
                    return {"status": "ERROR", "msg": "old_func must be callable"}
                if not callable(new_func):
                    return {"status": "ERROR", "msg": "new_func must be callable"}
                
                report = self._displace_references(old_func, new_func, dry_run=False)
                report["method"] = "FUNCTION_REPLACEMENT"
                logger.debug(f"Function replaced: {report['swaps']} references swapped")
                return report
            finally:
                gc.enable()
    
    def change_type(self, obj, new_type):
        """Change object type - T ∘ D_type pointer manipulation."""
        with self._lock:
            gc.disable()
            try:
                if not isinstance(new_type, type):
                    return {"status": "ERROR", "msg": "new_type must be a type object"}
                
                type_offset = self.offsets.get('ob_type')
                if not type_offset:
                    return {"status": "ERROR", "msg": "Type offset not calibrated"}
                
                obj_base = id(obj)
                type_addr = obj_base + type_offset
                old_type_id = id(type(obj))
                type_bytes = id(new_type).to_bytes(self.ptr_size, 'little')
                
                ctypes.pythonapi.Py_IncRef(ctypes.py_object(new_type))
                
                try:
                    # FIX #1: Use multi-tier write
                    success, tier = self._multi_tier_write(obj, type_offset, type_bytes)
                    if success:
                        if self._verify(type_addr, type_bytes):
                            old_type = ctypes.cast(old_type_id, ctypes.py_object).value
                            ctypes.pythonapi.Py_DecRef(ctypes.py_object(old_type))
                            logger.debug(f"Type changed via {tier} at 0x{type_addr:X}")
                            return True
                    
                    ctypes.pythonapi.Py_DecRef(ctypes.py_object(new_type))
                    return {"status": "ERROR", "msg": "Failed to change type (memory protected)",
                            "tried_tiers": list(self._working_bypass_tiers)}
                except Exception as e:
                    ctypes.pythonapi.Py_DecRef(ctypes.py_object(new_type))
                    logger.error(f"Type change error: {e}")
                    return False
            finally:
                gc.enable()
    
    def allocate_executable(self, size):
        """Allocate executable memory region for native code injection."""
        with self._lock:
            try:
                if size <= 0:
                    return {"status": "ERROR", "msg": "Size must be positive"}
                
                page_size = mmap.PAGESIZE
                aligned_size = ((size + page_size - 1) // page_size) * page_size
                
                if self.os_type == 'Windows':
                    try:
                        VirtualAlloc = ctypes.windll.kernel32.VirtualAlloc
                        VirtualAlloc.restype = ctypes.c_void_p
                        VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_ulong, ctypes.c_ulong]
                        
                        addr = VirtualAlloc(None, aligned_size, 0x3000, 0x40)  # MEM_COMMIT|MEM_RESERVE, PAGE_EXECUTE_READWRITE
                        if addr:
                            logger.debug(f"Allocated {aligned_size} bytes executable memory at 0x{addr:X} (Windows)")
                            return (addr, {"type": "windows", "addr": addr, "size": aligned_size})
                        return {"status": "ERROR", "msg": "VirtualAlloc failed"}
                    except Exception as e:
                        return {"status": "ERROR", "msg": f"Windows allocation failed: {e}"}
                else:
                    try:
                        flags = mmap.MAP_PRIVATE
                        if hasattr(mmap, 'MAP_ANONYMOUS'):
                            flags |= mmap.MAP_ANONYMOUS
                        elif hasattr(mmap, 'MAP_ANON'):
                            flags |= mmap.MAP_ANON
                        
                        prot = mmap.PROT_READ | mmap.PROT_WRITE
                        if hasattr(mmap, 'PROT_EXEC'):
                            prot |= mmap.PROT_EXEC
                        
                        buf = mmap.mmap(-1, aligned_size, flags=flags, prot=prot)
                        buf_array = (ctypes.c_char * aligned_size).from_buffer(buf)
                        addr = ctypes.addressof(buf_array)
                        logger.debug(f"Allocated {aligned_size} bytes executable memory at 0x{addr:X} (mmap)")
                        return (addr, buf)
                    except Exception as e:
                        return {"status": "ERROR", "msg": f"mmap allocation failed: {e}"}
            except Exception as e:
                logger.error(f"Executable allocation failed: {e}")
                return {"status": "ERROR", "msg": str(e)}
    
    def free_executable(self, alloc_result):
        """Free previously allocated executable memory."""
        try:
            if isinstance(alloc_result, tuple) and len(alloc_result) == 2:
                addr, buf = alloc_result
                if isinstance(buf, dict) and buf.get("type") == "windows":
                    ctypes.windll.kernel32.VirtualFree(ctypes.c_void_p(buf["addr"]), 0, 0x8000)
                    logger.debug(f"Freed Windows executable memory at 0x{buf['addr']:X}")
                    return True
                elif hasattr(buf, 'close'):
                    buf.close()
                    logger.debug(f"Freed mmap executable memory at 0x{addr:X}")
                    return True
            return False
        except Exception as e:
            logger.warning(f"Failed to free executable memory: {e}")
            return False
    
    def close(self):
        """Release kernel tunnel resources."""
        if self.wormhole:
            try:
                self.wormhole.close()
                logger.debug("Linux wormhole closed successfully")
            except Exception as e:
                logger.warning(f"Failed to close Linux wormhole: {e}")
            self.wormhole = None
        
        if self.win_handle:
            try:
                self.kernel32.CloseHandle(self.win_handle)
                logger.debug("Windows handle closed successfully")
            except Exception as e:
                logger.warning(f"Failed to close Windows handle: {e}")
            self.win_handle = None
    
    @staticmethod
    def cleanup_shared_memory():
        """Clean up shared memory for cross-process calibration cache."""
        if not HAS_SHARED_MEMORY:
            return True
        try:
            shm = shared_memory.SharedMemory(name=ET_SHARED_MEM_NAME)
            shm.close()
            shm.unlink()
            logger.debug(f"Shared memory '{ET_SHARED_MEM_NAME}' unlinked successfully")
            return True
        except FileNotFoundError:
            return True
        except Exception as e:
            logger.warning(f"Failed to cleanup shared memory: {e}")
            return False
    
    @staticmethod
    def clear_all_caches():
        """Clear all calibration caches."""
        results = {}
        
        if HAS_SHARED_MEMORY:
            try:
                shm = shared_memory.SharedMemory(name=ET_SHARED_MEM_NAME)
                shm.close()
                shm.unlink()
                results["shared_memory"] = "cleared"
            except FileNotFoundError:
                results["shared_memory"] = "not_found"
            except Exception as e:
                results["shared_memory"] = f"error: {e}"
        else:
            results["shared_memory"] = "unavailable"
        
        try:
            if ET_CACHE_ENV_VAR in os.environ:
                del os.environ[ET_CACHE_ENV_VAR]
                results["env_var"] = "cleared"
            else:
                results["env_var"] = "not_set"
        except Exception as e:
            results["env_var"] = f"error: {e}"
        
        # FIX #4: Only try to delete file if CACHE_FILE is available
        if CACHE_FILE:
            try:
                if os.path.exists(CACHE_FILE):
                    os.remove(CACHE_FILE)
                    results["file"] = "deleted"
                else:
                    results["file"] = "not_found"
            except Exception as e:
                results["file"] = f"error: {e}"
        else:
            results["file"] = "unavailable (read-only tempdir)"
        
        return results


# =========================================================================
# COMPREHENSIVE TEST SUITE (Enhanced to verify all fixes)
# =========================================================================

if __name__ == "__main__":
    import concurrent.futures
    import io
    
    log_capture = io.StringIO()
    capture_handler = logging.StreamHandler(log_capture)
    capture_handler.setLevel(logging.DEBUG)
    capture_handler.setFormatter(logging.Formatter('[ET %(levelname)s] %(message)s'))
    logger.addHandler(capture_handler)
    
    print("=" * 60)
    print("ET COMPENDIUM SOVEREIGN - FIXED VERSION TEST SUITE")
    print("=" * 60)
    print("\nFixes applied:")
    print("1. Multi-tier RO bypass with sandbox-safe fallbacks")
    print("2. Robust UCS2/UCS4 calibration with fallback chains")
    print("3. Comprehensive C-level intern reference displacement")
    print("4. Memory-only cache fallback for read-only environments")
    print("=" * 60)
    
    sov = ETCompendiumSovereign()
    
    print("\n--- TEST 1: DRY RUN (With Effort Metric) ---")
    s_test = "Effort_Test"
    l = [s_test] * 10
    report = sov.transmute(s_test, "Calculated", dry_run=True)
    print(f"Swaps: {report['swaps']}")
    print(f"Traverser Effort (|T|): {report['effort']:.2f} (Eq 212)")
    
    print("\n--- TEST 2: DENSITY CHECK (Eq 211) ---")
    s_compact = "Compact"
    s_pointer = "X" * 2000
    rho_c = ETMath.density(len(s_compact), sys.getsizeof(s_compact))
    rho_p = ETMath.density(len(s_pointer), sys.getsizeof(s_pointer))
    print(f"Compact Density (S): {rho_c:.3f} (High)")
    print(f"Pointer Density (S): {rho_p:.3f} (Low)")
    
    print("\n--- TEST 3: PHASE TRANSITION CONFIDENCE (Eq 30) ---")
    for conf in [0.05, 0.10, 0.20, 0.50, 0.80]:
        phase = ETMath.phase_transition(conf * 10, threshold=1.0)
        print(f"  Raw confidence {conf:.0%} -> Phase confidence {phase:.3f}")
    
    print("\n--- TEST 4: FIX #4 - CACHE FALLBACK VERIFICATION ---")
    cache_info = sov.get_cache_info()
    print(f"File cache available: {cache_info['file_path_available']}")
    print(f"Memory cache active: {cache_info['memory_cache_active']}")
    print(f"Cache backends: {list(cache_info['backends'].keys())}")
    for backend, info in cache_info['backends'].items():
        print(f"  {backend}: {info['status']}")
    
    print("\n--- TEST 5: FIX #2 - UCS2/UCS4 CALIBRATION ---")
    offset_1 = sov.offsets.get('1', 0)
    offset_2 = sov.offsets.get('2', 0)
    offset_4 = sov.offsets.get('4', 0)
    print(f"ASCII (width 1) offset: {offset_1} {'[OK]' if offset_1 > 16 else '[FAIL]'}")
    print(f"UCS2 (width 2) offset: {offset_2} {'[OK]' if offset_2 > 16 else '[FAIL]'}")
    print(f"UCS4 (width 4) offset: {offset_4} {'[OK]' if offset_4 > 16 else '[FAIL]'}")
    
    # Test UCS2 string
    ucs2_test = "Hello\u03A9World"  # Contains Greek Omega
    ucs2_target = ETMath.encode_width(ucs2_test, 2)
    print(f"UCS2 encoding test: {'[OK]' if ucs2_target is not None else '[FAIL]'}")
    
    # Test UCS4 string
    ucs4_test = "Hello\U0001F40DWorld"  # Contains snake emoji
    ucs4_target = ETMath.encode_width(ucs4_test, 4)
    print(f"UCS4 encoding test: {'[OK]' if ucs4_target is not None else '[FAIL]'}")
    
    print("\n--- TEST 6: FIX #3 - INTERN DICT DETECTION ---")
    intern_dict = sov._get_intern_dict()
    print(f"Intern dict found: {intern_dict is not None}")
    if intern_dict:
        print(f"Intern dict size: {len(intern_dict)} entries")
    
    # Test C-interned detection
    test_cases = [
        ("a", True, "ASCII_CHAR"),
        ("", True, "EMPTY_STRING"),
        ("0", True, "DIGIT_CHAR"),
        ("__name__", True, "BUILTIN_IDENTIFIER"),
        ("random_string_not_interned", False, None)
    ]
    print("\nC-interned detection:")
    for test_str, expected_interned, expected_type in test_cases:
        result = sov._check_c_interned(test_str)
        status = "[OK]" if result['is_c_interned'] == expected_interned else "[FAIL]"
        print(f"  '{test_str}': c_interned={result['is_c_interned']}, type={result['intern_type']} {status}")
    
    print("\n--- TEST 7: FIX #1 - MULTI-TIER RO BYPASS ---")
    print("Available bypass tiers:")
    for i, tier in enumerate(RO_BYPASS_TIERS, 1):
        print(f"  {i}. {tier}")
    
    # Try a transmutation and check which tiers worked
    test_str = "BytpassTest"
    result = sov.transmute(test_str, "BypassDone")
    if isinstance(result, dict):
        print(f"\nTransmutation result: {result.get('status', 'N/A')}")
        print(f"Method used: {result.get('method', 'N/A')}")
        if 'working_bypass_tiers' in result:
            print(f"Working bypass tiers: {result['working_bypass_tiers']}")
    
    print("\n--- TEST 8: THREAD SAFETY VERIFICATION ---")
    print(f"Lock type: {type(sov._lock).__name__}")
    print(f"Lock acquired test: ", end="")
    with sov._lock:
        print("SUCCESS (RLock operational)")
    
    print("\n--- TEST 9: GC SAFETY VERIFICATION ---")
    gc_was_enabled = gc.isenabled()
    print(f"GC enabled before transmute: {gc_was_enabled}")
    test_str = "GC_TEST!"
    sov.transmute(test_str, "GC_PASS!")
    gc_after = gc.isenabled()
    print(f"GC enabled after transmute: {gc_after}")
    print(f"GC properly restored: {'PASS' if gc_after == gc_was_enabled else 'FAIL'}")
    
    print("\n--- TEST 10: CONCURRENT THREAD SAFETY ---")
    shared_results = []
    error_count = [0]
    
    def thread_worker(thread_id, sov_instance):
        try:
            for i in range(10):
                test_val = f"T{thread_id}_V{i}__"
                replacement = f"R{thread_id}_V{i}__"
                result = sov_instance.transmute(test_val, replacement, dry_run=True)
                shared_results.append((thread_id, i, result['swaps']))
            return True
        except Exception as e:
            error_count[0] += 1
            logger.error(f"Thread {thread_id} failed: {e}")
            return False
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(thread_worker, tid, sov) for tid in range(4)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
    
    total_ops = len(shared_results)
    print(f"Concurrent operations completed: {total_ops}")
    print(f"Errors: {error_count[0]}")
    print(f"Thread safety test: {'PASS' if error_count[0] == 0 and total_ops == 40 else 'FAIL'}")
    
    print("\n--- TEST 11: INTERNED STRING DISPLACEMENT (FIX #3) ---")
    # Test with interned string
    s_imm = sys.intern("ET_INTERN_TEST")
    print(f"Original (interned): {s_imm}")
    
    # Create references
    intern_refs = [s_imm] * 5
    intern_dict_test = {"key": s_imm}
    
    result = sov.transmute(s_imm, "ET_INTERN_DONE", dry_run=True)
    print(f"Dry run swaps: {result['swaps']}")
    print(f"Intern dict found: {result.get('intern_dict_found', False)}")
    print(f"C-interned detected: {result.get('c_interned_detected', False)}")
    
    print("\n--- TEST 12: CONFIGURABLE PHASE-LOCKING ---")
    default_config = sov.get_phase_lock_config()
    print(f"Default noise pattern: {default_config['noise_pattern_hex']} ({default_config['noise_pattern_name']})")
    print(f"Default injection count: {default_config['injection_count']}")
    
    new_config = sov.configure_phase_lock(noise_pattern=0xAA, injection_count=2)
    print(f"After configure(0xAA, 2): pattern={new_config['noise_pattern_hex']}, count={new_config['injection_count']}")
    
    sov.configure_phase_lock(noise_pattern=0xFF, injection_count=1)
    restored = sov.get_phase_lock_config()
    print(f"Config restored: {'PASS' if restored['noise_pattern'] == 0xFF else 'FAIL'}")
    
    print("\n--- TEST 13: NEW ET MATH FUNCTIONS ---")
    
    # Phase Transition (Eq 30)
    print("\n13A. Phase Transition Sigmoid (Eq 30):")
    for x in [-2, -1, 0, 1, 2]:
        pt = ETMath.phase_transition(x, threshold=0)
        print(f"  phase_transition({x}) = {pt:.4f}")
    
    # Variance Gradient (Eq 83)
    print("\n13B. Variance Gradient (Eq 83):")
    current = 1.0
    target = 0.0
    for step in range(5):
        next_val = ETMath.variance_gradient(current, target, step_size=0.3)
        print(f"  Step {step}: {current:.3f} -> {next_val:.3f}")
        current = next_val
    
    # Kolmogorov Complexity (Eq 77)
    print("\n13C. Kolmogorov Complexity (Eq 77):")
    test_sets = [
        [1, 2, 3, 4, 5],
        [1, 1, 1, 1, 1],
        [1, 2, 1, 2, 1, 2]
    ]
    for s in test_sets:
        kc = ETMath.kolmogorov_complexity(s)
        print(f"  N_min({s}) = {kc}")
    
    print("\n--- TEST 14: OFFSET VALIDATION ---")
    all_offsets_valid = True
    offset_checks = {
        'code': ('Code object offset', 16, 256),
        'func': ('Function code pointer offset', 8, 128),
        'ob_type': ('Object type pointer offset', 4, 16),
        'hash': ('String hash offset', 8, 32),
        'tuple': ('Tuple items offset', 8, 32),
    }
    
    print("Offset calibrations:")
    for key, (name, min_val, max_val) in offset_checks.items():
        if key in sov.offsets:
            val = sov.offsets[key]
            in_range = min_val <= val <= max_val
            status = 'PASS' if in_range else 'WARN (out of typical range)'
            print(f"  {name}: {val} bytes [{status}]")
            if not in_range:
                all_offsets_valid = False
        else:
            print(f"  {name}: MISSING")
            all_offsets_valid = False
    
    print("\n--- TEST 15: change_type HELPER ---")
    class TypeA:
        value = "A"
    class TypeB:
        value = "B"
    obj_to_change = TypeA()
    print(f"Before: type={type(obj_to_change).__name__}")
    type_change_result = sov.change_type(obj_to_change, TypeB)
    print(f"After: type={type(obj_to_change).__name__}")
    print(f"change_type test: {'PASS' if type(obj_to_change) is TypeB else 'FAIL'}")
    
    sov.close()
    print("\n--- CLEANUP: Resources released ---")
    
    cleanup_result = ETCompendiumSovereign.cleanup_shared_memory()
    print(f"Shared memory cleanup: {'SUCCESS' if cleanup_result else 'FAILED'}")
    
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETE")
    print("=" * 60)
    print(f"\nFix Summary:")
    print(f"  FIX #1 (RO Bypass): Multi-tier fallback chain implemented")
    print(f"  FIX #2 (UCS2/UCS4): Width {2} offset={offset_2}, Width {4} offset={offset_4}")
    print(f"  FIX #3 (C-Intern): Intern dict detection {'active' if intern_dict else 'inactive'}")
    print(f"  FIX #4 (Cache): File cache {'available' if CACHE_FILE else 'unavailable (memory-only mode)'}")
