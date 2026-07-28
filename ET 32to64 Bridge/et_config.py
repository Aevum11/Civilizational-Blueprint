"""
et_bridge/et_config.py
ET32 Bridge — Configuration Management

Derived from P ∘ D ∘ T = E.

Configuration IS the Descriptor set of the bridge's runtime behavior.
The config file is a {P,D} Unsubstantiated configuration waiting for T
(the running bridge process) to traverse it into actuality.

Config structure derived from ET:
  - target_exes    : D-set of 32-bit processes to bridge
  - bridge_options : fine-grained D-constraints per target
  - ipc_config     : Mediation channel descriptors
  - log_config     : Logging Descriptors

The config file is watched for changes using polling (S-second interval =
12 seconds). When changed, the new D-set replaces the old one — the bridge
T-traverses the updated configuration.
"""

import json
import os
import time
import threading
import hashlib
import copy
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path

from et_math import S, K, V_BASE, CONN_TIMEOUT_MS, DIGITAL_ACTION_QUANTUM, IPC_BUFFER_SIZE
from et_errors import (
    ETOperationError, ETWindowsAPIError, ETInjectionError,
    ETIPCError, ETPacketError, ETAWEError, ETHookError,
    ETDispatchError, ETConfigError, ETHandleError,
    ETErrorSeverity, win32_check, win32_check_handle,
    ntstatus_check, et_context, safe_call,
    record_error, record_op, get_registry,
)



# Default config path — sits next to the executable
DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "et32_bridge_config.json"
)


class TargetConfig:
    """
    Configuration for a single 32-bit target process.
    
    Each target is a potential Descriptor: it describes what capabilities to
    extend and under what conditions. The bridge T-traverses this Descriptor
    when the process matching 'exe_name' is detected.
    
    Fields derived from ET:
      exe_name      : D₁ — which process (process identity Descriptor)
      features      : D₂...D₁₂ — which features to enable (lattice positions)
      mem_threshold : D₁ threshold for memory bridging (default K × 4GB)
      dll_paths     : D₄ — extra 64-bit DLL search paths
      gpu_enabled   : D₇ — whether to bridge GPU operations
      python_64     : D₁₁ — whether to bridge 64-bit Python
      inject_mode   : T-mode for injection (debug, iat, shellcode)
    """

    # Available features keyed by ET lattice position
    FEATURE_NAMES = {
        1:  "memory_basic",     # d=1: extended memory allocation
        2:  "memory_map",       # d=2: large memory-mapped files
        3:  "threads",          # d=3: 64-bit thread operations
        4:  "dll_64bit",        # d=4: 64-bit DLL loading
        5:  "process",          # d=5: 64-bit process creation
        6:  "registry_bypass",  # d=6: bypass WOW64 registry redirect
        7:  "gpu_vram",         # d=7: GPU/VRAM extension
        8:  "large_files",      # d=8: files/maps > 4GB
        9:  "sync",             # d=9: 64-bit sync objects
        10: "network",          # d=10: 64-bit network buffers
        11: "python_64",        # d=11: 64-bit Python embedding
        12: "compound",         # d=12: compound/batched operations
    }

    def __init__(self, data: Dict):
        self.exe_name: str = data.get("exe_name", "")
        self.enabled: bool = data.get("enabled", True)
        self.inject_mode: str = data.get("inject_mode", "iat")  # "iat", "debug", "shellcode"

        # Feature flags: which lattice positions to enable
        features = data.get("features", "all")
        if features == "all":
            self.features = set(range(1, S + 1))
        elif features == "none":
            self.features = set()
        else:
            self.features = set(int(f) for f in features if isinstance(f, (int, str)))

        # Memory threshold: size above which allocations are bridged to 64-bit
        # Default = K × 2^31 (Koide threshold of 32-bit user space ceiling)
        default_threshold = int(K * (1 << 31))
        raw_bytes = data.get("mem_threshold_mb", default_threshold >> 20) * (1 << 20)
        # Page-align to ħ_d (DIGITAL_ACTION_QUANTUM) — the ET digital action quantum.
        # Every memory threshold must fall on a ħ_d boundary (2^12 = 4096 bytes).
        self.mem_threshold_bytes: int = (raw_bytes // DIGITAL_ACTION_QUANTUM) * DIGITAL_ACTION_QUANTUM

        # Extra 64-bit DLL search directories
        self.dll_search_paths: List[str] = data.get("dll_64_paths", [])

        # GPU configuration
        self.gpu_enabled: bool = 7 in self.features and data.get("gpu_enabled", True)
        self.gpu_vram_limit_gb: float = data.get("gpu_vram_limit_gb", 0.0)  # 0 = no limit

        # 64-bit Python configuration
        self.python_64: bool = 11 in self.features and data.get("python_64", False)
        self.python_home: str = data.get("python_home", "")  # path to 64-bit Python

        # Registry bypass configuration
        self.reg_bypass: bool = 6 in self.features and data.get("registry_bypass", False)

        # Large file support
        self.large_files: bool = 8 in self.features and data.get("large_files", True)

        # IPC overrides per target
        self.pipe_timeout_ms: int = data.get("pipe_timeout_ms", CONN_TIMEOUT_MS)

        # Process launch settings (if we're the launcher)
        self.launch_before: bool = data.get("launch_before", False)
        self.launch_args: List[str] = data.get("launch_args", [])

    def has_feature(self, lattice_d: int) -> bool:
        """
        Test whether lattice position d is active for this target.

        ET derivation: each feature is a Descriptor at position d (1...S=12)
        in the ET lattice.  has_feature(d) returns True iff the Descriptor
        is present in this target's active D-set, meaning the bridge will
        extend that capability for the process.
        """
        return lattice_d in self.features

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this target's D-set to a JSON-compatible dictionary.

        ET derivation: inverse of __init__ — maps the runtime Descriptor
        set back to the featureless P-substrate (dict) for persistence.
        Every field produced here can round-trip through TargetConfig(data)
        and reproduce the identical D-set (V(round_trip) = 0).
        """
        return {
            "exe_name": self.exe_name,
            "enabled": self.enabled,
            "inject_mode": self.inject_mode,
            "features": sorted(self.features),
            "mem_threshold_mb": self.mem_threshold_bytes >> 20,
            "dll_64_paths": self.dll_search_paths,
            "gpu_enabled": self.gpu_enabled,
            "gpu_vram_limit_gb": self.gpu_vram_limit_gb,
            "python_64": self.python_64,
            "python_home": self.python_home,
            "registry_bypass": self.reg_bypass,
            "large_files": self.large_files,
            "pipe_timeout_ms": self.pipe_timeout_ms,
            "launch_before": self.launch_before,
            "launch_args": self.launch_args,
        }


class ETBridgeConfig:
    """
    Complete bridge configuration. This is the root Descriptor set for the bridge.

    ET-derived structure:
      P = the config file (featureless substrate containing the Descriptor data)
      D = the parsed configuration values (constraints on bridge behavior)
      T = the config watcher (traverses the file and actualizes the Descriptors)

    The config watcher polls every S seconds (12 seconds = MANIFOLD_SYMMETRY).
    On change, the bridge hot-reloads — the new D-set replaces the old.
    """

    # Poll interval = S seconds (manifold symmetry)
    POLL_INTERVAL_S: float = float(S)

    def __init__(self, path: str = None):
        self.path = path or DEFAULT_CONFIG_PATH
        self._data: Dict = {}
        self._hash: str = ""
        self._targets: Dict[str, TargetConfig] = {}
        self._lock = threading.RLock()
        self._watch_thread: Optional[threading.Thread] = None
        self._watching = False
        self._on_change_callbacks: List[Callable] = []

        # Bridge-level IPC config
        self.ipc_buffer_size: int = IPC_BUFFER_SIZE  # 49152 bytes
        self.ipc_queue_depth: int = S * S             # 144

        # Logging config
        self.log_level: str = "INFO"
        self.log_file: Optional[str] = None
        self.log_et_metrics: bool = True  # log ET variance metrics

        # Bridge behavior
        self.auto_inject: bool = True
        self.inject_on_startup: bool = True
        self.helper_path: Optional[str] = None  # path to et_bridge_helper32.exe

        # Bridge mode — ET Identification Principle:
        #   P = all 32-bit processes on the system
        #   whitelist D = {named processes to bridge}  (explicit inclusion)
        #   blacklist D = {all 32-bit} minus {named exclusions} (explicit exclusion)
        # Both modes address the same P from opposite D-directions.
        # mode 'whitelist' (default): only bridge listed exe names.
        # mode 'blacklist': bridge ALL 32-bit processes EXCEPT listed names.
        self.bridge_mode: str = "whitelist"  # "whitelist" | "blacklist"
        self.blacklist: List[str] = []  # lowercase exe names excluded in blacklist mode

        # Default blacklist config used when bridge_mode = "blacklist":
        # populated from config file — NOT hardcoded here.

        # Load initial config
        self._load()

    @classmethod
    def load(cls, path: str) -> Optional["ETBridgeConfig"]:
        """
        Load a config file and return a new ETBridgeConfig instance.

        Returns the loaded config, or None if the file cannot be parsed.

        ET derivation: this is a P→D factory — reads the featureless
        substrate (file on disk) and produces a fully-formed Descriptor set.
        """
        try:
            cfg = cls(path)
            if cfg._data:
                return cfg
            # _load() was called in __init__ — if it produced data, config is valid
            # If file existed but was empty/malformed, _data will be empty
            p = Path(path)
            if p.exists() and p.stat().st_size > 0:
                return cfg
            return None
        except (OSError, json.JSONDecodeError, ValueError):
            return None

    @classmethod
    def generate_default(cls, path: str) -> None:
        """
        Generate a default config file at the given path and return.

        ET derivation: creates a complete D-set template with all lattice
        positions documented. The user then customizes the Descriptor set
        for their specific targets.
        """
        cfg = cls.__new__(cls)
        cfg.path = path
        cfg._data = {}
        cfg._hash = ""
        cfg._targets = {}
        cfg._lock = threading.RLock()
        cfg._watch_thread = None
        cfg._watching = False
        cfg._on_change_callbacks = []
        cfg.ipc_buffer_size = IPC_BUFFER_SIZE
        cfg.ipc_queue_depth = S * S
        cfg.log_level = "INFO"
        cfg.log_file = None
        cfg.log_et_metrics = True
        cfg.auto_inject = True
        cfg.inject_on_startup = True
        cfg.helper_path = None
        cfg.bridge_mode = "whitelist"
        cfg.blacklist = []
        cfg._create_default()

    # ------------------------------------------------------------------
    # Loading and parsing
    # ------------------------------------------------------------------

    def _load(self) -> bool:
        """Load and parse the config file. Returns True if successfully loaded."""
        path = Path(self.path)
        if not path.exists():
            self._create_default()
            return False

        try:
            text = path.read_text(encoding="utf-8")
            data = json.loads(text)
            new_hash = hashlib.sha256(text.encode()).hexdigest()

            if new_hash == self._hash:
                return False  # no change

            with self._lock:
                self._data = data
                self._hash = new_hash
                self._parse(data)

            record_op()
            return True

        except (json.JSONDecodeError, OSError) as e:
            # Descriptor Gap: config file is malformed — record the error and
            # continue with current defaults.  V(config) reflects the gap.
            record_error(ETConfigError(
                f"Config load failed for '{self.path}': {e}",
                severity=ETErrorSeverity.MEDIATION,
                config_path=str(self.path),
            ))
            return False

    def _parse(self, data: Dict):
        """Parse the raw JSON dict into typed configuration objects."""
        # Bridge-level settings
        ipc = data.get("ipc", {})
        self.ipc_buffer_size = ipc.get("buffer_size", IPC_BUFFER_SIZE)
        self.ipc_queue_depth = ipc.get("queue_depth", S * S)

        log = data.get("logging", {})
        self.log_level      = log.get("level", "INFO")
        self.log_et_metrics = log.get("et_metrics", True)
        self.log_variance   = log.get("variance_log", True)

        # Log file path resolution:
        #   "file": absolute path → used as-is
        #   "file": relative path → resolved relative to exe directory
        #   "file": null/absent  → resolved by ETLog.resolve_log_path() at startup
        #   "dir":  directory    → log file placed in this directory as et32_bridge.log
        raw_file = log.get("file", None)
        raw_dir  = log.get("dir",  None)
        if raw_dir and not raw_file:
            # dir specified: log goes into that directory
            import os as _os
            self.log_file = _os.path.join(str(raw_dir), "et32_bridge.log")
        else:
            self.log_file = raw_file  # None or explicit path — resolved at startup

        bridge = data.get("bridge", {})
        self.auto_inject       = bridge.get("auto_inject", True)
        self.inject_on_startup = bridge.get("inject_on_startup", True)
        self.helper_path       = bridge.get("helper_path", None)
        raw_mode = str(bridge.get("mode", "whitelist")).lower()
        self.bridge_mode = raw_mode if raw_mode in ("whitelist", "blacklist") else "whitelist"
        # Blacklist entries: read from 'blacklist' key OR from targets with enabled=False
        # when bridge_mode='blacklist'. User controls the list entirely in the config file.
        raw_bl = data.get("blacklist", [])
        self.blacklist = [str(e).lower() for e in raw_bl if e]

        # Target configurations
        self._targets = {}
        for entry in data.get("targets", []):
            cfg = TargetConfig(entry)
            if cfg.exe_name:
                key = cfg.exe_name.lower()
                self._targets[key] = cfg

    def _create_default(self):
        """Write a default config file if none exists."""
        default = {
            "_comment": "ET32 Bridge Configuration — P∘D∘T = E",
            "_version": "1.0.0",
            "bridge": {
                "auto_inject": True,
                "inject_on_startup": True,
                "helper_path": None
            },
            "ipc": {
                "buffer_size": IPC_BUFFER_SIZE,
                "queue_depth": S * S
            },
            "logging": {
                "_comment":    "Log file is placed in the same folder as the exe by default.",
                "_file_doc":   "Set 'file' to an absolute path, or a relative path (resolved from exe dir), or null for default.",
                "_dir_doc":    "Set 'dir' to a directory path to place et32_bridge.log there instead of beside the exe.",
                "level":       "INFO",
                "file":        None,
                "dir":         None,
                "et_metrics":  True,
                "variance_log": True
            },
            "targets": [
                {
                    "_comment": "Example: bridge a 32-bit game",
                    "exe_name": "example_game.exe",
                    "enabled": False,
                    "inject_mode": "iat",
                    "features": "all",
                    "mem_threshold_mb": int(K * (1 << 31)) >> 20,
                    "dll_64_paths": [],
                    "gpu_enabled": True,
                    "gpu_vram_limit_gb": 0.0,
                    "python_64": False,
                    "python_home": "",
                    "registry_bypass": True,
                    "large_files": True,
                    "pipe_timeout_ms": CONN_TIMEOUT_MS,
                    "launch_before": False,
                    "launch_args": []
                }
            ]
        }

        try:
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            Path(self.path).write_text(
                json.dumps(default, indent=2),
                encoding="utf-8"
            )
            record_op()
        except OSError as exc:
            record_error(ETConfigError(
                f"Failed to write default config to '{self.path}': {exc}",
                severity=ETErrorSeverity.MEDIATION,
                config_path=str(self.path),
            ))

    # ------------------------------------------------------------------
    # Target lookup
    # ------------------------------------------------------------------

    def get_target(self, exe_name: str) -> Optional[TargetConfig]:
        """Return config for a target process (case-insensitive match)."""
        with self._lock:
            return self._targets.get(exe_name.lower())

    def is_blacklisted(self, exe_name: str) -> bool:
        """Return True if exe_name is on the blacklist (case-insensitive)."""
        with self._lock:
            return exe_name.lower() in self.blacklist

    def should_bridge(self, exe_name: str) -> bool:
        """
        Determine whether a 32-bit process should be bridged.

        ET PDT of the decision:
          P = the process (it exists, it is 32-bit)
          D = the bridge_mode + whitelist/blacklist D-set
          T = this function (the decision traverser)
          E = True (bridge) or False (skip)

        whitelist mode:
          Bridge iff exe_name is in _targets AND enabled=True.
          D is the explicit inclusion set.

        blacklist mode:
          Bridge iff exe_name is NOT in blacklist.
          D is the explicit exclusion set; everything else is bridged.
          For unlisted processes in blacklist mode: use a synthetic
          TargetConfig with all features enabled (full S=12 lattice).
        """
        with self._lock:
            key = exe_name.lower()
            if self.bridge_mode == "blacklist":
                return key not in self.blacklist
            else:  # whitelist
                cfg = self._targets.get(key)
                return cfg is not None and cfg.enabled

    def synthetic_config(self, exe_name: str) -> TargetConfig:
        """
        Return a TargetConfig for an unlisted process in blacklist mode.
        Full S=12 lattice (all features), default ET constants.

        ET: if there is no explicit D for this process, use the complete
        D-set (all 12 positions active) — maximum extension, zero V(E).
        """
        with self._lock:
            # Prefer explicit config if it exists
            key = exe_name.lower()
            if key in self._targets:
                return self._targets[key]
        # Synthesize a full-feature config for this process
        return TargetConfig({
            "exe_name":          exe_name,
            "enabled":           True,
            "inject_mode":       "iat",
            "features":          "all",
            "mem_threshold_mb":  int(K * (1 << 31)) >> 20,
            "dll_64_paths":      [],
            "gpu_enabled":       True,
            "gpu_vram_limit_gb": 0.0,
            "python_64":         False,
            "python_home":       "",
            "registry_bypass":   True,
            "large_files":       True,
            "pipe_timeout_ms":   CONN_TIMEOUT_MS,
            "launch_before":     False,
            "launch_args":       [],
        })

    def all_targets(self) -> List[TargetConfig]:
        """Return all enabled target configurations."""
        with self._lock:
            return [t for t in self._targets.values() if t.enabled]

    def target_exe_names(self) -> List[str]:
        """Return lowercase exe names of all enabled targets."""
        with self._lock:
            return [k for k, v in self._targets.items() if v.enabled]

    @property
    def targets(self) -> List[TargetConfig]:
        """
        All enabled target configurations (property alias for all_targets).

        ET derivation: the active D-set — the complete set of Descriptors
        that the bridge Traverser will act upon. Exposed as a property for
        direct attribute access from et32_bridge_main.py.
        """
        return self.all_targets()

    @property
    def config_path(self) -> str:
        """
        Absolute path to the config file on disk.

        ET derivation: the P-substrate address of the config Descriptor.
        Alias for self.path, exposed for clarity in call sites.
        """
        return self.path

    def has_changed(self) -> bool:
        """
        Check whether the config file has changed since last load.

        Reads the file, computes SHA-256, and compares to the stored hash.
        Does NOT reload — just peeks. Use reload() to force a reparse.

        ET derivation: this is a D-comparison — has the Descriptor file
        mutated since we last read it? If the hash differs, the bridge
        should re-traverse the config to maintain V(E) = 0.
        """
        changed = False
        with et_context("config_has_changed", reraise=False,
                        severity=ETErrorSeverity.TRACE,
                        config_path=str(self.path)):
            text = Path(self.path).read_text(encoding="utf-8")
            current_hash = hashlib.sha256(text.encode()).hexdigest()
            changed = current_hash != self._hash
        return changed

    # ------------------------------------------------------------------
    # File watcher
    # ------------------------------------------------------------------

    def start_watching(self, on_change: Callable = None):
        """
        Start the config file watcher thread.
        Polls every S seconds (= 12 seconds = manifold symmetry interval).
        Calls on_change(new_config) when config changes.
        """
        if on_change:
            self._on_change_callbacks.append(on_change)

        if self._watching:
            return

        self._watching = True
        self._watch_thread = threading.Thread(
            target=self._watch_loop,
            name="ET_ConfigWatcher",
            daemon=True
        )
        self._watch_thread.start()

    def stop_watching(self):
        """
        Stop the config file watcher thread.

        ET derivation: terminates the T-traversal loop.  The config
        D-set freezes — no further hot-reloads occur until
        start_watching() is called again.  Joins the watcher thread
        with a timeout of POLL_INTERVAL_S + 1 second to guarantee
        one final poll cycle can complete.
        """
        self._watching = False
        if self._watch_thread:
            self._watch_thread.join(timeout=self.POLL_INTERVAL_S + 1)

    def _watch_loop(self):
        while self._watching:
            # Poll interval = S seconds
            time.sleep(self.POLL_INTERVAL_S)
            if self._load():
                for cb in self._on_change_callbacks:
                    # safe_call: never raises, records any exception in the
                    # global ETErrorRegistry with full context.  Replaces the
                    # bare 'except Exception: pass' with proper ET error tracking.
                    safe_call(
                        cb, self,
                        operation=f"config_change_callback:"
                                  f"{getattr(cb, '__qualname__', str(cb))}",
                        severity=ETErrorSeverity.MEDIATION,
                    )

    # ------------------------------------------------------------------
    # Hot reload
    # ------------------------------------------------------------------

    def reload(self) -> bool:
        """Force immediate reload. Returns True if config changed."""
        # Invalidate hash to force reparse
        old_hash = self._hash
        self._hash = ""
        changed = self._load()
        if not changed:
            self._hash = old_hash
        return changed

    def apply(self, other: "ETBridgeConfig") -> None:
        """
        Apply settings from another ETBridgeConfig instance to this one.

        Used by the main loop for hot-reload: load a new config from disk,
        then apply its D-set to the running config without replacing the
        object (which would break references held by other subsystems).

        ET derivation: D-replacement — the Descriptor set is updated in-place.
        P (the process) and T (the running threads) are unchanged.
        Only D mutates. V(config) is recalculated from the new D-set.
        """
        with self._lock:
            self._data           = copy.deepcopy(other._data) if hasattr(other, '_data') else {}
            self._hash           = getattr(other, '_hash', "")
            self._targets        = copy.deepcopy(getattr(other, '_targets', {}))
            self.ipc_buffer_size = getattr(other, 'ipc_buffer_size', self.ipc_buffer_size)
            self.ipc_queue_depth = getattr(other, 'ipc_queue_depth', self.ipc_queue_depth)
            self.log_level       = getattr(other, 'log_level', self.log_level)
            self.log_file        = getattr(other, 'log_file', self.log_file)
            self.log_et_metrics  = getattr(other, 'log_et_metrics', self.log_et_metrics)
            self.auto_inject     = getattr(other, 'auto_inject', self.auto_inject)
            self.inject_on_startup = getattr(other, 'inject_on_startup', self.inject_on_startup)
            self.helper_path     = getattr(other, 'helper_path', self.helper_path)
            self.bridge_mode     = getattr(other, 'bridge_mode', self.bridge_mode)
            self.blacklist       = list(getattr(other, 'blacklist', self.blacklist))

    # ------------------------------------------------------------------
    # Runtime config modification (for UI/CLI integration)
    # ------------------------------------------------------------------

    def add_target(self, exe_name: str, features: str = "all") -> TargetConfig:
        """Dynamically add a target at runtime. Writes back to config file."""
        entry = {
            "exe_name": exe_name,
            "enabled": True,
            "features": features,
            "inject_mode": "iat",
        }
        cfg = TargetConfig(entry)
        with self._lock:
            self._targets[exe_name.lower()] = cfg

        # Persist to file
        try:
            path = Path(self.path)
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
                data.setdefault("targets", [])
                # Remove existing entry for same exe
                data["targets"] = [
                    t for t in data["targets"]
                    if t.get("exe_name", "").lower() != exe_name.lower()
                ]
                data["targets"].append(cfg.to_dict())
                path.write_text(json.dumps(data, indent=2), encoding="utf-8")
                record_op()
        except (OSError, json.JSONDecodeError) as exc:
            record_error(ETConfigError(
                f"Failed to persist add_target('{exe_name}'): {exc}",
                severity=ETErrorSeverity.MEDIATION,
                config_path=str(self.path),
            ))

        return cfg

    def remove_target(self, exe_name: str) -> bool:
        """Remove a target at runtime."""
        key = exe_name.lower()
        with self._lock:
            if key not in self._targets:
                return False
            del self._targets[key]

        try:
            path = Path(self.path)
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
                data["targets"] = [
                    t for t in data.get("targets", [])
                    if t.get("exe_name", "").lower() != key
                ]
                path.write_text(json.dumps(data, indent=2), encoding="utf-8")
                record_op()
        except (OSError, json.JSONDecodeError) as exc:
            record_error(ETConfigError(
                f"Failed to persist remove_target('{exe_name}'): {exc}",
                severity=ETErrorSeverity.MEDIATION,
                config_path=str(self.path),
            ))

        return True

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def variance(self) -> float:
        """
        V(config) = fraction of targets that have no features enabled.
        V(config) = 0 → all targets fully configured (Exception state).
        V(config) = V_BASE × count_empty → some Descriptor gaps remain.
        """
        with self._lock:
            targets = list(self._targets.values())
        if not targets:
            return V_BASE  # empty config = one V_BASE gap
        empty = sum(1 for t in targets if not t.features)
        return (empty / len(targets)) * V_BASE if empty else 0.0

    def summary(self) -> Dict[str, Any]:
        """
        Return a diagnostic summary of the bridge configuration.

        ET derivation: a complete D-snapshot of the running config.
        V(config) is included to measure Descriptor completeness.
        The global error registry summary is included so that a single
        call gives the operator both config state and system health.
        """
        registry = get_registry()
        with self._lock:
            targets = list(self._targets.values())
        return {
            "config_path": self.path,
            "targets_total": len(targets),
            "targets_enabled": sum(1 for t in targets if t.enabled),
            "targets": [t.exe_name for t in targets if t.enabled],
            "ipc_buffer_size": self.ipc_buffer_size,
            "log_level": self.log_level,
            "variance": self.variance(),
            "bridge_mode": self.bridge_mode,
            "error_registry": registry.summary(),
            "error_domains": self._error_domain_counts(),
        }

    def _error_domain_counts(self) -> Dict[str, int]:
        """
        Count recent errors by domain type for config-level diagnostics.

        ET derivation: each error subclass maps to a domain of the bridge's
        12-position command lattice.  Counting by domain reveals which
        D-families have the most Descriptor Gaps.

        Discovery is DYNAMIC: ETOperationError.__subclasses__() finds all
        registered error types at runtime.  The known imports below serve
        as the seed set (guaranteeing they are loaded), and any new error
        subclasses added to et_errors.py are automatically discovered
        without modifying this method.
        """
        registry = get_registry()
        # self.ipc_queue_depth = S² = 144 — look back one full queue depth
        recent = registry.recent(n=self.ipc_queue_depth)

        # Seed with known imported error types — ensures the imports are
        # exercised and serves as documentation of the bridge's error taxonomy.
        known: Dict[str, type] = {
            "windowsapi":  ETWindowsAPIError,
            "injection":   ETInjectionError,
            "ipc":         ETIPCError,
            "packet":      ETPacketError,
            "awe":         ETAWEError,
            "hook":        ETHookError,
            "dispatch":    ETDispatchError,
            "config":      ETConfigError,
            "handle":      ETHandleError,
        }

        # Dynamic discovery: find any subclasses NOT in the seed set.
        # This ensures newly added error types are tracked without code changes.
        known_classes = set(known.values())
        for subclass in ETOperationError.__subclasses__():
            if subclass not in known_classes:
                name = subclass.__name__
                # Derive domain key: strip "ET" prefix and "Error" suffix
                if name.startswith("ET") and name.endswith("Error"):
                    domain = name[2:-5].lower()
                else:
                    domain = name.lower()
                known[domain] = subclass

        # Count: base (unspecialized) errors separately, then each domain
        counts: Dict[str, int] = {
            "general": sum(1 for e in recent if type(e) is ETOperationError)
        }
        for domain, err_type in known.items():
            counts[domain] = sum(
                1 for e in recent if isinstance(e, err_type)
            )
        return counts

    def validate(self) -> Dict[str, Any]:
        """
        Validate the bridge configuration against the running system.

        ET derivation: V(config_validation) measures how many D-constraints
        are satisfiable on the current P-substrate (the host OS).  A config
        that references non-existent paths or unavailable APIs has Descriptor
        Gaps — this method enumerates them.

        Checks performed:
          1. Config file exists and is valid JSON
          2. Helper exe exists (if configured)
          3. DLL search paths exist (per target)
          4. Python home exists (per target with python_64=True)
          5. Windows API availability (kernel32, ntdll) — Windows only

        Returns a dict with 'valid' (bool) and 'issues' (list of strings).
        """
        results: Dict[str, Any] = {"valid": True, "issues": []}

        # 1. Config file readability
        with et_context("validate_config_file", reraise=False,
                        severity=ETErrorSeverity.TRACE,
                        config_path=str(self.path)):
            p = Path(self.path)
            if not p.exists():
                results["issues"].append(f"Config file not found: {self.path}")
                results["valid"] = False
            elif p.stat().st_size == 0:
                results["issues"].append(f"Config file is empty: {self.path}")
                results["valid"] = False

        # 2. Helper exe
        if self.helper_path:
            if not Path(self.helper_path).exists():
                results["issues"].append(
                    f"helper_path not found: {self.helper_path}"
                )
                results["valid"] = False

        # 3 & 4. Per-target path validation
        with self._lock:
            targets = list(self._targets.values())
        for tgt in targets:
            if not tgt.enabled:
                continue
            for dll_dir in tgt.dll_search_paths:
                if dll_dir and not Path(dll_dir).is_dir():
                    results["issues"].append(
                        f"[{tgt.exe_name}] dll_64_path not found: {dll_dir}"
                    )
            if tgt.python_64 and tgt.python_home:
                if not Path(tgt.python_home).is_dir():
                    results["issues"].append(
                        f"[{tgt.exe_name}] python_home not found: "
                        f"{tgt.python_home}"
                    )

        # 5. Windows API availability — only on Windows
        import sys as _sys
        if _sys.platform == "win32":
            self._validate_windows_apis(results)

        return results

    def _validate_windows_apis(self, results: Dict[str, Any]) -> None:
        """
        Validate that required Windows APIs are accessible.

        Uses win32_check, win32_check_handle, and ntstatus_check from
        et_errors to test each API surface the bridge depends on.

        All windll attribute access uses getattr() for dynamic resolution —
        ctypes.windll returns _DLLT objects whose members cannot be resolved
        statically by type checkers.

        ET derivation: the bridge requires specific P-capabilities from the
        OS.  Each API is a D-constraint on the system substrate.  If an API
        is missing, the corresponding lattice positions cannot function.
        Validation is target-aware: only APIs required by enabled features
        in self._targets are tested.
        """
        import ctypes

        # Collect the union of active features across all enabled targets.
        # In blacklist mode, assume full lattice (any process may be bridged).
        with self._lock:
            if self.bridge_mode == "blacklist":
                active_features = set(range(1, S + 1))
            else:
                active_features = set()
                for tgt in self._targets.values():
                    if tgt.enabled:
                        active_features |= tgt.features

        # Dynamic windll access — getattr resolves _DLLT type-checker limits
        windll = getattr(ctypes, 'windll', None)
        if windll is None:
            results["issues"].append(
                "ctypes.windll not available (non-Windows platform)"
            )
            results["valid"] = False
            return

        # Load kernel32 — required for ALL bridge operations (d=1..12)
        k32 = getattr(windll, 'kernel32', None)
        if k32 is None:
            results["issues"].append("kernel32.dll not loadable")
            results["valid"] = False
            return

        # Resolve API function pointers dynamically via getattr
        get_module_handle = getattr(k32, 'GetModuleHandleW', None)
        get_proc_address = getattr(k32, 'GetProcAddress', None)

        if get_module_handle is None or get_proc_address is None:
            results["issues"].append(
                "kernel32 missing GetModuleHandleW or GetProcAddress"
            )
            results["valid"] = False
            return

        # Test kernel32 API surface
        try:
            k32_handle = win32_check_handle(
                get_module_handle("kernel32.dll"),
                "GetModuleHandleW('kernel32')",
            )
            # Verify VirtualAllocEx is exported (d=1,2 memory operations)
            win32_check(
                get_proc_address(k32_handle, b"VirtualAllocEx"),
                "GetProcAddress(kernel32, VirtualAllocEx)",
            )
        except ETOperationError as exc:
            results["issues"].append(
                f"kernel32 validation failed: {exc.operation}"
            )
            results["valid"] = False

        # Test ntdll — required for WOW64 hooks (d=6), thread ops (d=3),
        # process ops (d=5), and any NT-level syscall routing.
        # Only validated if targets actually use these features.
        ntdll_features = active_features & {3, 5, 6}
        if ntdll_features or self.bridge_mode == "blacklist":
            try:
                ntdll_handle = win32_check_handle(
                    get_module_handle("ntdll.dll"),
                    "GetModuleHandleW('ntdll')",
                )
                # Verify NtQuerySystemInformation is exported
                win32_check(
                    get_proc_address(ntdll_handle,
                                     b"NtQuerySystemInformation"),
                    "GetProcAddress(ntdll, NtQuerySystemInformation)",
                )
                # Functional test: NtQuerySystemInformation with zero buffer.
                # Expected: STATUS_INFO_LENGTH_MISMATCH (0xC0000004).
                ntdll = getattr(windll, 'ntdll', None)
                nt_query = getattr(ntdll, 'NtQuerySystemInformation', None)
                if nt_query is not None:
                    buf_size = ctypes.c_ulong(0)
                    status = nt_query(0, None, 0, ctypes.byref(buf_size))
                    status_unsigned = status & 0xFFFFFFFF
                    # 0xC0000004 is the expected "buffer too small" — not an error.
                    if status_unsigned not in (0, 0xC0000004):
                        ntstatus_check(
                            status,
                            "NtQuerySystemInformation(SystemBasicInformation)",
                        )
            except ETOperationError as exc:
                results["issues"].append(
                    f"ntdll validation failed: {exc.operation}"
                )
                results["valid"] = False