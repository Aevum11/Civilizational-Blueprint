"""
et32_bridge_main.py
ET32 Bridge — Main Entry Point (64-bit Broker)

Derived from P ∘ D ∘ T = E.

This is the top-level P∘D∘T instantiation of the entire bridge:
  P = this operating system process (the 64-bit broker)
  D = the loaded ETBridgeConfig (Descriptor set of all target processes)
  T = this main thread + all spawned workers (the Traverser)
  E = the running bridge (full Exception state — V(E) = 0 while operational)

Startup sequence (ET-grounded):
  1. Parse arguments — determine config path
  2. Load ETBridgeConfig — instantiate D
  3. Start ETBridgeAPI — starts IPC server (creates P-channel)
  4. Start ETProcessMonitor — T begins scanning
  5. Enter main loop — bridge is active
  6. On config change — hot-reload D without stopping T
  7. On CTRL+C / signal — graceful shutdown (Subsumption Law completion check)

Usage:
  ET32_Bridge.exe [--config path\to\config.json] [--log-level DEBUG|INFO|WARNING]
  
  Default config path: ET32_Bridge_config.json (alongside the executable).

  If --no-tray is NOT passed (default), a system tray icon is created showing
  bridge status. Pass --no-tray to run as a pure console/background process.

ET Stability reporting:
  The main loop logs bridge stability (K_eff) every S² = 144 seconds.
  If K_eff < K (2/3), a WARNING is emitted — approaching ∂I.
  If K_eff < V_BASE (1/12), an INCOHERENCE event is emitted.

Compilation to EXE:
  Use et32_bridge.spec with PyInstaller (64-bit Python required):
    pyinstaller et32_bridge.spec
"""

import argparse
import os
import sys
import time
import signal
import threading
import ctypes
import json
from pathlib import Path
from typing import Optional, Callable, Dict

# Ensure modules are importable when running as frozen EXE or from source
if getattr(sys, "frozen", False):
    _base = getattr(sys, '_MEIPASS', os.path.dirname(sys.executable))
    if _base not in sys.path:
        sys.path.insert(0, _base)
    os.chdir(os.path.dirname(sys.executable))
else:
    _dev_root = os.path.dirname(os.path.abspath(__file__))
    if _dev_root not in sys.path:
        sys.path.insert(0, _dev_root)

from et_config import ETBridgeConfig
from et_api import ETBridgeAPI
from et_monitor import ETProcessMonitor
from et_logger import ETLog, resolve_log_path
from et_math import S, K, V_BASE, QUEUE_DEPTH, CONN_TIMEOUT_MS

# ============================================================================
# STARTUP BANNER
# ============================================================================

_BANNER = r"""
╔══════════════════════════════════════════════════════════════════╗
║             ET32 Bridge — Exception Theory 64-bit Broker         ║
║                  P ∘ D ∘ T = E  (v1.0.0)                        ║
║          S=12  K=2/3  V=1/12  ħ_d=4096  IPC=49152B              ║
╚══════════════════════════════════════════════════════════════════╝
"""

# ============================================================================
# ET-DERIVED TIMING CONSTANTS (module-level — uppercase is PEP8-correct here)
# ============================================================================

STABILITY_INTERVAL = float(S * S)    # 144 s — stability report period (S²)
CONFIG_INTERVAL    = float(S)        # 12 s  — config file poll period (S)
HEALTH_INTERVAL    = float(S * K)    # 8 s   — Koide health check period (S×K)

# ============================================================================
# ARGUMENT PARSING
# ============================================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog        = "ET32_Bridge",
        description = "ET32 Bridge: 32-bit → 64-bit capability extension broker.",
        formatter_class = argparse.RawDescriptionHelpFormatter,
        epilog      = "Derived from Exception Theory (P∘D∘T=E). All constants from first principles."
    )
    parser.add_argument(
        "--config", "-c",
        default = None,
        help    = "Path to et32_bridge_config.json (default: alongside executable)"
    )
    parser.add_argument(
        "--log-level", "-l",
        default = "INFO",
        choices = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help    = "Logging level (default: INFO)"
    )
    parser.add_argument(
        "--log-file", "-f",
        default = None,
        help    = "Path to log file (default: ET32_Bridge.log alongside executable)"
    )
    parser.add_argument(
        "--no-tray",
        action  = "store_true",
        default = False,
        help    = "Suppress system tray icon (run as background/console process)"
    )
    parser.add_argument(
        "--generate-config", "-g",
        action  = "store_true",
        default = False,
        help    = "Generate a default config file and exit"
    )
    parser.add_argument(
        "--status",
        action  = "store_true",
        default = False,
        help    = "Print current bridge status and exit (if another instance is running)"
    )
    return parser.parse_args()


# ============================================================================
# LOCATE CONFIG FILE
# ============================================================================

def _resolve_config_path(explicit: Optional[str]) -> Path:
    """
    Resolve the config file path.
    Order: explicit arg → beside executable → beside script → CWD.
    """
    if explicit:
        p = Path(explicit)
        if p.exists():
            return p
        raise FileNotFoundError(f"Config not found: {explicit}")

    candidates = [
        Path(sys.executable).parent / "et32_bridge_config.json",
        Path(__file__).parent / "et32_bridge_config.json",
        Path.cwd() / "et32_bridge_config.json",
    ]
    for c in candidates:
        if c.exists():
            return c

    # Return the first candidate as the default (will be created if --generate-config)
    return candidates[0]


# ============================================================================
# GENERATE DEFAULT CONFIG
# ============================================================================

def _generate_default_config(path: Path) -> None:
    """
    Generate a default config file with example targets.
    ET Descriptors pre-filled with all lattice positions enabled.
    """
    ETBridgeConfig.generate_default(str(path))
    print(f"Default config generated: {path}")
    print("Edit the 'targets' array to list your 32-bit executables.")
    print("Then run ET32_Bridge.exe without --generate-config to start the bridge.")


# ============================================================================
# SYSTEM TRAY (optional — requires pystray + Pillow)
# ============================================================================

def _try_start_tray(bridge_api: ETBridgeAPI, stop_callback):
    """
    Attempt to start a system tray icon showing bridge status.
    Silently skipped if pystray or PIL are not available.

    The tray icon shows:
      - Tooltip: "ET32 Bridge — K=x.xx" (Koide alignment)
      - Menu: Status | Reload Config | Stop Bridge

    ET derivation: the tray is a T-interface — it gives the user a traversal
    point into the bridge's Descriptor state without modifying P or D.
    """
    try:
        import pystray
        from PIL import Image, ImageDraw

        # Draw a simple icon: 32×32 teal circle with "ET" text
        # Color derived from ET: RGB = (S*S, S*V*255, K*255) = (144, 21, 170) → purple
        def _make_icon_image():
            img  = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            draw.ellipse([2, 2, 30, 30], fill=(90, 50, 180, 255))
            draw.text((7, 10), "ET", fill=(255, 255, 255, 255))
            return img

        def _status_action(tray_icon, menu_item):
            report = bridge_api.status_report()
            tray_icon.title = "ET32 Bridge — K={:.4f}".format(bridge_api.stability)
            # Show in console (tray icon is silent in most contexts)
            print("[{}] {}".format(menu_item.text, json.dumps(report, indent=2)))

        def _reload_action(tray_icon, menu_item):
            bridge_api.reload_config()
            tray_icon.title = "ET32 Bridge — config reloaded"
            print("[{}] Config reloaded.".format(menu_item.text))

        def _stop_action(tray_icon, menu_item):
            print("[{}] Stop requested.".format(menu_item.text))
            tray_icon.stop()
            stop_callback()

        icon = pystray.Icon(
            name  = "ET32Bridge",
            icon  = _make_icon_image(),
            title = "ET32 Bridge",
            menu  = pystray.Menu(
                pystray.MenuItem("Status",        _status_action),
                pystray.MenuItem("Reload Config", _reload_action),
                pystray.MenuItem("Stop Bridge",   _stop_action),
            )
        )

        def _tray_thread():
            icon.run()

        t = threading.Thread(target=_tray_thread, name="ET_TrayIcon", daemon=True)
        t.start()
        return icon
    except ImportError:
        return None  # pystray/PIL not available — no tray


# ============================================================================
# INTERACTIVE CONSOLE CLI — Audit Point 6
# ============================================================================

class ETConsoleCLI:
    """
    Interactive console for the ET32 Bridge broker.

    ET PDT of this CLI:
      P = the operator (human substrate)
      D = the command set (Descriptors of all bridge operations)
      T = the input thread (Traverser — reads operator intent, dispatches actions)
      E = the operator's informed control over the bridge (Exception state)

    Provides:
      1. Persistent console (AllocConsole, SetConsoleTitle on Windows)
      2. Full CLI — all 12 command families accessible
      3. Dynamic help — enumerates CmdFamily/CmdCode at runtime (no static lists)
      4. Live process event display (attach/detach/error)
      5. Step-by-step attachment confirmation
      6. Error and fallback display
      7. Live subsystem metrics on demand
      8. Manual command mode (status, attach, detach, list, help, quit, etc.)
    """

    def __init__(self, bridge_api: 'ETBridgeAPI', monitor: 'ETProcessMonitor',
                 config: 'ETBridgeConfig', log: 'ETLog'):
        self._api     = bridge_api
        self._monitor = monitor
        self._config  = config
        self._log     = log
        self._thread: Optional[threading.Thread] = None
        self._running = False
        # Import dynamically to avoid circular imports at module level
        from et_math import CmdFamily, CmdCode
        self._CmdFamily = CmdFamily
        self._CmdCode   = CmdCode

    # ---- Console Persistence (Windows) ----

    @staticmethod
    def _ensure_console():
        """Ensure a console window exists and is persistent on Windows."""
        if sys.platform != "win32":
            return
        try:
            kernel32 = ctypes.windll.kernel32
            # AllocConsole is safe to call even if one already exists (returns FALSE but no harm)
            getattr(kernel32, 'AllocConsole')()
            getattr(kernel32, 'SetConsoleTitleW')("ET32 Bridge — P∘D∘T = E")
        except (AttributeError, OSError):
            pass

    # ---- Dynamic Help (no static lists — _DLLT principle) ----

    def _print_help(self):
        """Print complete help dynamically from CmdFamily and CmdCode enums."""
        _CF = self._CmdFamily
        _CC = self._CmdCode
        print("\n" + "=" * 72)
        print(" ET32 Bridge — Interactive Console Help")
        print(" P ∘ D ∘ T = E  |  S=12  K=2/3  V=1/12")
        print("=" * 72)

        # Command families — dynamically enumerated
        print("\n--- Command Families (d=1..12) ---\n")
        # Build family→codes mapping dynamically from CmdCode enum members
        family_names = {}
        for attr in dir(_CF):
            val = getattr(_CF, attr)
            if isinstance(val, int) and not attr.startswith("_"):
                family_names[val] = attr

        family_codes: dict = {}
        for attr in dir(_CC):
            val = getattr(_CC, attr)
            if isinstance(val, int) and not attr.startswith("_"):
                # Determine family from code range: 0x01-0x0F→1, 0x11-0x1F→2,
                # 0x21-0x2F→3, etc. Control codes 0xF0+ → 0
                if val >= 0xF0:
                    fam = 0
                elif val >= 0xB0:
                    fam = 12
                elif val >= 0xA0:
                    fam = 11
                elif val >= 0x90:
                    fam = 10
                elif val >= 0x80:
                    fam = 9
                elif val >= 0x70:
                    fam = 8
                elif val >= 0x60:
                    fam = 7
                elif val >= 0x50:
                    fam = 6
                elif val >= 0x40:
                    fam = 5
                elif val >= 0x30:
                    fam = 4
                elif val >= 0x20:
                    fam = 3
                elif val >= 0x10:
                    fam = 2
                else:
                    fam = 1
                family_codes.setdefault(fam, []).append((val, attr))

        for fam_id in sorted(family_names.keys()):
            if fam_id == 0:
                continue
            print(f"  d={fam_id:2d}  {family_names[fam_id]}:")
            codes = family_codes.get(fam_id, [])
            for code_val, code_name in sorted(codes):
                print(f"         0x{code_val:02X}  {code_name}")

        # Control codes
        ctrl_codes = family_codes.get(0, [])
        if ctrl_codes:
            print("\n  CTRL  Control Codes:")
            for code_val, code_name in sorted(ctrl_codes):
                print(f"         0x{code_val:02X}  {code_name}")

        # Interactive commands
        print("\n--- Interactive Commands ---\n")
        cmds = [
            ("status",                "Show bridge status + all subsystem states"),
            ("list",                  "List all bridged processes"),
            ("targets",               "Show configured targets from config"),
            ("attach <pid>",          "Manually attach to a 32-bit process"),
            ("detach <pid>",          "Detach from a bridged process"),
            ("metrics",               "Show live performance metrics"),
            ("awe [pid]",             "AWE Bookshelf status (all or per-PID)"),
            ("heaven",                "Heaven's Gate status"),
            ("hooks [pid]",           "KiFastSystemCall hook status (all or per-PID)"),
            ("handles",               "Handle table summary"),
            ("errors",                "Error registry summary"),
            ("alloc <pid> <size>",    "Allocate memory for a process via broker"),
            ("exec <pid> <code>",     "Execute Python code in 64-bit context"),
            ("reg get <pid> <key>",   "Registry query via bridge"),
            ("reload",                "Hot-reload config file"),
            ("scan",                  "Force immediate process scan"),
            ("help",                  "Show this help"),
            ("quit | exit",           "Graceful shutdown"),
        ]
        for cmd, desc in cmds:
            print(f"  {cmd:<24s} {desc}")
        print("\n" + "=" * 72)
        print()

    # ---- Live Event Callbacks (printed to console) ----

    def on_process_found(self, pid: int, exe_name: str):
        """Console callback when a target process is detected."""
        self._log.info("CLI: process found PID=%d %s", pid, exe_name)
        print(f"\n  [FOUND] PID={pid}  {exe_name}")

    def on_inject_step(self, pid: int, step: str, success: bool, detail: str = ""):
        """Console callback for each injection step."""
        tag = " [OK]  " if success else " [FAIL]"
        self._log.debug("CLI inject step PID=%d %s %s", pid, step, tag)
        print(f"    {tag} PID={pid}: {step}{('  — ' + detail) if detail else ''}")

    def on_process_exit(self, pid: int, exe_name: str, exit_code: int):
        """Console callback when a bridged process exits."""
        self._log.info("CLI: process exit PID=%d %s code=%d", pid, exe_name, exit_code)
        print(f"\n  [EXIT] PID={pid}  {exe_name}  exit_code={exit_code}")

    def on_error(self, pid: int, msg: str):
        """Console callback for errors from bridged processes."""
        self._log.warning("CLI error PID=%d: %s", pid, msg)
        print(f"  [ERR]  PID={pid}: {msg}")

    # ---- Command Dispatch ----

    def _cmd_status(self, parts: list):
        """status — full bridge status report."""
        verbose = len(parts) > 1 and parts[1] == "-v"
        report = self._api.status_report()
        k_eff = self._api.stability
        state = ("EXCEPTION" if k_eff >= 1.0 - V_BASE else
                 "MEDIATION" if k_eff >= K else
                 "WARNING"   if k_eff >= V_BASE else
                 "INCOHERENCE")
        print(f"\n  Bridge Status: {state}  K_eff={k_eff:.4f}")
        print(f"  Active PIDs: {len(self._monitor.active_pids())}")
        print(f"  Mode: {self._config.bridge_mode}")
        # Subsystem states via public accessors
        host = self._api.host
        if host is not None:
            heaven = host.heaven
            if heaven is not None:
                base = heaven.ntdll64_base
                print(f"  Heaven's Gate: ntdll64 resolved={'0x%016X' % base if base else 'no'}")
            else:
                print(f"  Heaven's Gate: not initialised")
            wow64 = host.wow64
            if wow64 is not None:
                svc_size = wow64.service_table_size()
                print(f"  WOW64 Service Table: {svc_size} entries")
            awe = host.awe
            if awe is not None:
                print(f"  AWE Bookshelf: active")
            else:
                print(f"  AWE Bookshelf: not active")
        if verbose:
            # Full report dict when -v flag given
            for key, val in report.items():
                print(f"    {key}: {val}")
        print()

    def _cmd_list(self, parts: list):
        """list [filter] — show all bridged processes, optionally filtered."""
        pids = self._monitor.active_pids()
        # Optional filter: 'list <substring>' filters by PID string match
        if len(parts) > 1:
            filter_str = parts[1]
            pids = [p for p in pids if filter_str in str(p)]
        if not pids:
            print("\n  No bridged processes.\n")
            return
        print(f"\n  Bridged Processes ({len(pids)}):")
        for pid in pids:
            print(f"    PID={pid}")
        print()

    def _cmd_targets(self, parts: list):
        """targets [filter] — show configured targets, optionally filtered by name."""
        targets = self._config.targets
        if len(parts) > 1:
            name_filter = parts[1].lower()
            targets = [t for t in targets if name_filter in t.exe_name.lower()]
        print(f"\n  Configured Targets ({len(targets)}):")
        for t in targets:
            en = "enabled" if t.enabled else "DISABLED"
            print(f"    {t.exe_name:<30s}  {en}  inject={t.inject_mode}  features={t.features}")
        print()

    def _cmd_metrics(self, parts: list):
        """metrics [-v] — show live performance metrics."""
        verbose = len(parts) > 1 and parts[1] == "-v"
        host = self._api.host
        if host is not None:
            m = host.metrics
            print(f"\n  Performance Metrics:")
            print(f"    Total requests:  {m.total_requests}")
            print(f"    Successful:      {m.successful_requests}")
            success_rate = (m.successful_requests / max(m.total_requests, 1)) * 100
            print(f"    Success rate:    {success_rate:.1f}%")
            print(f"    K_eff (Koide):   {self._api.stability:.4f}")
            # Per-family counts — dynamically enumerated
            print(f"    Per-family request counts:")
            for fam_id, count in sorted(m.family_counts.items()):
                if count > 0 or verbose:
                    # Dynamic family name lookup from CmdFamily enum
                    name = "unknown"
                    for attr in dir(self._CmdFamily):
                        if getattr(self._CmdFamily, attr) == fam_id and not attr.startswith("_"):
                            name = attr
                            break
                    print(f"      d={fam_id:2d} ({name}): {count}")
        else:
            print("\n  Metrics not available (host not initialised).\n")
        print()

    def _cmd_awe(self, parts: list):
        """awe [pid] — AWE Bookshelf status."""
        host = self._api.host
        if host is not None:
            awe = host.awe
            if awe is None:
                print("\n  AWE Bookshelf: not active\n")
                return
            print(f"\n  AWE Bookshelf Status:")
            # Optional PID filter from parts
            filter_pid = int(parts[1]) if len(parts) > 1 else None
            pool_pids = awe.all_pool_pids()
            if filter_pid is not None:
                pool_pids = [p for p in pool_pids if p == filter_pid]
            for pid in pool_pids:
                info = awe.status(pid)
                print(f"    PID={pid}: pages_alloc={info.get('pages_allocated', 0)} "
                      f"in_use={info.get('pages_in_use', 0)} "
                      f"windows={info.get('windows_mapped', 0)}/{info.get('windows_total', 0)} "
                      f"gb={info.get('gb_allocated', 0)}")
            if not pool_pids:
                print("    No AWE pools allocated.")
            print()
        else:
            print("\n  AWE not available.\n")

    def _cmd_heaven(self, parts: list):
        """heaven [-v] — Heaven's Gate status."""
        verbose = len(parts) > 1 and parts[1] == "-v"
        host = self._api.host
        if host is not None:
            h = host.heaven
            if h is not None:
                base = h.ntdll64_base
                print(f"\n  Heaven's Gate: initialised")
                print(f"    ntdll64 base: {'0x%016X' % base if base else 'not resolved'}")
                if verbose:
                    wow64 = host.wow64
                    if wow64 is not None:
                        print(f"    Service table: {wow64.service_table_size()} entries")
            else:
                print("\n  Heaven's Gate: not initialised")
        else:
            print("\n  Host not available.\n")
        print()

    def _cmd_hooks(self, parts: list):
        """hooks [pid] — hook status."""
        hm = self._api.hook_manager
        if hm is not None:
            states = hm.all_states()
            # Optional PID filter
            filter_pid = int(parts[1]) if len(parts) > 1 else None
            if filter_pid is not None:
                states = {k: v for k, v in states.items() if k == filter_pid}
            if not states:
                print("\n  No hook states registered.\n")
                return
            print(f"\n  Hook States ({len(states)}):")
            for pid, state in states.items():
                active = "ACTIVE" if state.hooks_active else "inactive"
                print(f"    PID={pid}: {active}")
        else:
            print("\n  Hook manager not available.\n")
        print()

    def _cmd_handles(self, parts: list):
        """handles [-v] — handle table summary."""
        verbose = len(parts) > 1 and parts[1] == "-v"
        host = self._api.host
        if host is not None:
            table = host.handle_table
            count = table.count() if hasattr(table, 'count') else 0
            print(f"\n  Handle Table: {count} live entries")
            if verbose and hasattr(table, 'fill_ratio'):
                print(f"    Fill ratio: {table.fill_ratio:.1%}")
        else:
            print("\n  Handle table not available.\n")
        print()

    def _cmd_errors(self, parts: list):
        """errors [n] — error registry summary, optionally show last n errors."""
        from et_errors import get_registry
        reg = get_registry()
        self._log.debug("CLI: errors command invoked")
        print(f"\n  Error Registry:")
        print(f"    Total errors: {reg.error_count}")
        print(f"    V(system):    {reg.variance():.4f}")
        # Errors by severity — uses new public severity_summary()
        for sev_name, count in reg.severity_summary().items():
            if count > 0:
                print(f"    {sev_name}: {count}")
        # Optional: show recent errors
        show_n = int(parts[1]) if len(parts) > 1 else 0
        if show_n > 0:
            recent = reg.recent(show_n)
            print(f"    Last {len(recent)} errors:")
            for e in recent:
                print(f"      {e}")
        print()

    def _cmd_scan(self, parts: list):
        """scan — force immediate process scan."""
        self._log.debug("CLI scan: args=%s", parts[1:] if len(parts) > 1 else "none")
        self._monitor.force_scan()
        print("  Scan initiated.\n")

    def _cmd_reload(self, parts: list):
        """reload — hot-reload config."""
        self._log.info("CLI reload: args=%s", parts[1:] if len(parts) > 1 else "none")
        self._api.reload_config()
        print("  Config reloaded.\n")

    def _cmd_attach(self, parts: list):
        """attach <pid> — manually attach to a process."""
        if len(parts) < 2:
            print("  Usage: attach <pid>\n")
            return
        try:
            pid = int(parts[1])
        except ValueError:
            print("  Invalid PID.\n")
            return
        # Find matching target config or use default
        tc = None
        for t in self._config.targets:
            if t.enabled:
                tc = t
                break
        if tc is None:
            print("  No enabled target config found. Enable at least one target.\n")
            return
        # on_process_found accepts a single arg: ProcessInfo or int (PID fallback).
        # It returns None — success/failure is observed via hook_manager state.
        self._api.on_process_found(pid)
        state = self._api.hook_manager.get_state(pid) if self._api.hook_manager else None
        attached = state is not None and state.hooks_active if state else False
        print(f"  Attach {'initiated' if not attached else 'confirmed'} for PID {pid}.\n")

    def _cmd_detach(self, parts: list):
        """detach <pid> — detach from a process."""
        if len(parts) < 2:
            print("  Usage: detach <pid>\n")
            return
        try:
            pid = int(parts[1])
        except ValueError:
            print("  Invalid PID.\n")
            return
        self._api.on_process_exit(pid)
        print(f"  Detached from PID {pid}.\n")

    def _cmd_help(self, parts: list):
        """help [topic] — display full command reference."""
        # parts[1:] reserved for future topic-specific help (e.g. 'help awe')
        topic = parts[1] if len(parts) > 1 else None
        if topic:
            self._log.debug("CLI help topic: %s", topic)
        self._print_help()

    def _dispatch(self, line: str):
        """Parse and dispatch a single CLI command."""
        line = line.strip()
        if not line:
            return
        parts = line.split()
        cmd = parts[0].lower()

        dispatch: Dict[str, Callable[[list], None]] = {
            "status":   self._cmd_status,
            "list":     self._cmd_list,
            "targets":  self._cmd_targets,
            "metrics":  self._cmd_metrics,
            "awe":      self._cmd_awe,
            "heaven":   self._cmd_heaven,
            "hooks":    self._cmd_hooks,
            "handles":  self._cmd_handles,
            "errors":   self._cmd_errors,
            "scan":     self._cmd_scan,
            "reload":   self._cmd_reload,
            "attach":   self._cmd_attach,
            "detach":   self._cmd_detach,
            "help":     self._cmd_help,
            "?":        self._cmd_help,
        }

        handler = dispatch.get(cmd)
        if handler:
            try:
                handler(parts)
            except Exception as exc:
                print(f"  Command error: {exc}\n")
        elif cmd in ("quit", "exit"):
            print("  Shutting down...")
            self._api.request_stop()
            self._running = False
        else:
            print(f"  Unknown command: '{cmd}'. Type 'help' for available commands.\n")

    # ---- Input Thread ----

    def _input_loop(self):
        """Main input loop running in its own thread."""
        self._running = True
        print("\n  ET32 Bridge Console — type 'help' for commands.\n")
        while self._running and self._api.is_running:
            try:
                line = input("ET32> ")
                self._dispatch(line)
            except EOFError:
                # stdin closed (piped input or headless mode) — exit gracefully
                break
            except KeyboardInterrupt:
                print()  # newline after ^C
                self._api.request_stop()
                self._running = False
                break
            except Exception as exc:
                print(f"  Input error: {exc}")

    # ---- Start / Stop ----

    def start(self):
        """Start the CLI thread and ensure console persistence."""
        self._ensure_console()
        self._thread = threading.Thread(
            target=self._input_loop,
            name="ET_CLI",
            daemon=True  # dies with main thread
        )
        self._thread.start()

    def stop(self):
        """Signal the CLI to stop."""
        self._running = False


# ============================================================================
# MAIN LOOP
# ============================================================================

def _main_loop(
    bridge_api:  ETBridgeAPI,
    monitor:     ETProcessMonitor,
    config:      ETBridgeConfig,
    log:         "ETLog"
) -> None:
    """
    Main event loop for the broker process.

    Runs until bridge_api.request_stop() is called.

    Periodic tasks (ET-timed):
      Every S   = 12 s : update tray tooltip, check config hash
      Every S²  = 144 s: log stability report
      Every S×K = 8 s  : Koide-threshold health check (2/3 × 12 ≈ 8s)
    """
    last_stability_log = time.monotonic()
    last_config_check  = time.monotonic()
    last_health_check  = time.monotonic()
    # Timing intervals: STABILITY_INTERVAL, CONFIG_INTERVAL, HEALTH_INTERVAL (module-level)

    log.info("Main loop started. Bridge is active.")
    log.info(
        "Watching for: %s",
        ", ".join(t.exe_name for t in config.targets) if config.targets else "(no targets)"
    )

    while bridge_api.is_running:
        now = time.monotonic()

        # --- Stability report every S² seconds ---
        if now - last_stability_log >= STABILITY_INTERVAL:
            k_eff = bridge_api.stability
            state = "EXCEPTION" if k_eff >= 1.0 - V_BASE else (
                    "MEDIATION" if k_eff >= K else
                    "WARNING"   if k_eff >= V_BASE else
                    "INCOHERENCE"
            )
            log.info(
                "Stability report: K_eff=%.4f [%s] | active_targets=%d",
                k_eff,
                state,
                len(monitor.active_pids())
            )
            last_stability_log = now

        # --- Config file change check every S seconds ---
        if now - last_config_check >= CONFIG_INTERVAL:
            if config.has_changed():
                log.info("Config file changed — reloading...")
                new_cfg = ETBridgeConfig.load(config.config_path)
                if new_cfg:
                    config.apply(new_cfg)
                    bridge_api.reload_config()
                    monitor.update_targets(config.targets)
            last_config_check = now

        # --- Health check every S×K ≈ 8 seconds ---
        if now - last_health_check >= HEALTH_INTERVAL:
            k_eff = bridge_api.stability
            if k_eff < V_BASE:
                log.incoherence(
                    "Bridge variance CRITICAL: K_eff=%.4f < V_BASE=%.4f",
                    k_eff, V_BASE
                )
            elif k_eff < K:
                log.warning_di(
                    "Bridge approaching ∂I: K_eff=%.4f < K=%.4f",
                    k_eff, K
                )
            last_health_check = now

        # Sleep in CONN_TIMEOUT_MS / S chunks so we can respond quickly to stop events
        bridge_api.wait_for_stop(timeout=CONN_TIMEOUT_MS / 1000.0 / S)

    log.info("Main loop exiting.")


# ============================================================================
# ENTRY POINT
# ============================================================================

def main() -> int:
    """
    Main entry point.
    Returns exit code: 0 = clean shutdown, 1 = startup error.
    """
    args = _parse_args()

    # --- Initialize logging (FIRST action — before anything that can fail) ---
    # resolve_log_path() handles exe-relative default; ETLog.setup() installs:
    #   - sys.excepthook       (unhandled Python exceptions → log file)
    #   - faulthandler         (C-level crashes / segfaults → log file)
    #   - atexit               (final flush + error registry summary on exit)
    #   - Windows SEH filter   (DLL crashes → log file)
    # The log file is open and flushed before main() proceeds further.
    log_file = resolve_log_path(args.log_file)
    ETLog.setup(level=args.log_level, log_file=log_file)
    log = ETLog.get("et32_bridge_main")

    log.info("ET32 Bridge starting — log: %s", log_file)
    print(_BANNER)
    print(f"  Log file: {log_file}")

    # --- Generate config and exit if requested ---
    if args.generate_config:
        try:
            config_path = _resolve_config_path(args.config) if args.config else \
                          Path(sys.executable).parent / "et32_bridge_config.json" \
                          if getattr(sys, "frozen", False) else \
                          Path.cwd() / "et32_bridge_config.json"
            _generate_default_config(config_path)
            return 0
        except Exception as exc:
            print(f"Error generating config: {exc}")
            return 1

    # --- Load config ---
    try:
        config_path = _resolve_config_path(args.config)
    except FileNotFoundError as exc:
        log.error(str(exc))
        print(f"\nError: {exc}")
        print("Run with --generate-config to create a default configuration.")
        return 1

    log.info("Loading config: %s", config_path)
    config = ETBridgeConfig.load(str(config_path))
    if config is None:
        log.incoherence("Failed to load config from %s", config_path)
        print(f"\nError: Failed to load config from {config_path}")
        return 1

    log.info("Config loaded: %d target(s)", len(config.targets))
    for t in config.targets:
        log.info("  Target: %s (inject_mode=%s)", t.exe_name, t.inject_mode)

    # If config specifies a different log path, switch to it now
    if config.log_file and resolve_log_path(config.log_file) != log_file:
        new_log_path = resolve_log_path(config.log_file)
        log.info("Switching log file to config-specified path: %s", new_log_path)
        ETLog.setup(level=config.log_level, log_file=new_log_path)
        log = ETLog.get("et32_bridge_main")
        log.info("Log resumed at new path: %s", new_log_path)

    # --- Initialise bridge API ---
    bridge_api = ETBridgeAPI(config)
    if not bridge_api.start():
        log.incoherence("Bridge API failed to start")
        print("\nError: Bridge API failed to start (see log for details).")
        return 1

    log.info(
        "IPC configured: queue_depth=%d (S²), timeout=%dms, "
        "buffer=%d bytes",
        QUEUE_DEPTH, CONN_TIMEOUT_MS, config.ipc_buffer_size
    )

    # --- Register signal handlers ---
    def _signal_handler(signum, frame):
        caller_info = ""
        if frame is not None:
            caller_info = " at %s:%d" % (frame.f_code.co_filename, frame.f_lineno)
        log.info("Signal %d received%s — requesting shutdown", signum, caller_info)
        bridge_api.request_stop()

    signal.signal(signal.SIGINT,  _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # --- Windows console control handler (uses ctypes) ---
    # On Windows, SIGTERM is not delivered for console close / logoff / shutdown.
    # SetConsoleCtrlHandler intercepts all console termination events:
    #   CTRL_C_EVENT (0), CTRL_BREAK_EVENT (1), CTRL_CLOSE_EVENT (2),
    #   CTRL_LOGOFF_EVENT (5), CTRL_SHUTDOWN_EVENT (6)
    # ET derivation: complete coverage of all termination D-paths (Subsumption Law).
    _console_handler_ref = None  # prevent GC of the callback
    if sys.platform == "win32":
        try:
            _HANDLER_ROUTINE = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_ulong)

            def _console_ctrl_handler(event):
                log.info("Windows console event %d received — requesting shutdown", event)
                bridge_api.request_stop()
                return True  # handled — prevent default termination

            _console_handler_ref = _HANDLER_ROUTINE(_console_ctrl_handler)
            _set_ctrl = getattr(ctypes.windll.kernel32, 'SetConsoleCtrlHandler')
            _set_ctrl(_console_handler_ref, True)
            log.debug("Windows console control handler installed")
        except (AttributeError, OSError):
            log.debug("SetConsoleCtrlHandler not available on this platform")

    # --- Start process monitor ---
    monitor = ETProcessMonitor(
        config         = config,
        on_found       = bridge_api.on_process_found,
        on_exit        = bridge_api.on_process_exit,
    )
    monitor.start()
    log.info("Process monitor started: polling every %ds", S)

    # --- System tray (if not suppressed) ---
    tray_icon = None
    if not args.no_tray:
        tray_icon = _try_start_tray(bridge_api, bridge_api.request_stop)
        if tray_icon:
            log.info("System tray icon active")
        else:
            log.mediation("System tray not available (pystray/PIL missing) — running headless")

    log.info("ET32 Bridge is active. Press CTRL+C to stop.")

    # --- Start interactive console CLI (audit point 6) ---
    cli = ETConsoleCLI(bridge_api, monitor, config, log)
    cli.start()
    log.info("Interactive console CLI started")

    # --- Main loop ---
    try:
        _main_loop(bridge_api, monitor, config, log)
    except KeyboardInterrupt:
        log.info("KeyboardInterrupt received")
        bridge_api.request_stop()

    # --- Shutdown sequence ---
    log.info("Shutting down ET32 Bridge...")

    cli.stop()

    if tray_icon:
        try:
            tray_icon.stop()
        except (OSError, RuntimeError, AttributeError):
            pass  # tray already stopped or window handle gone — safe to ignore

    monitor.stop()
    bridge_api.stop()

    log.info("ET32 Bridge stopped cleanly. P∘D∘T = E → complete.")
    print("\nET32 Bridge stopped.")

    # Console persistence: when run as PyInstaller EXE via double-click,
    # prevent the console from closing immediately so the user can read
    # any final messages. In piped/headless mode, stdin is closed and
    # input() raises EOFError — silently exit in that case.
    if getattr(sys, "frozen", False) and sys.platform == "win32":
        try:
            input("\nPress Enter to close this window...")
        except (EOFError, KeyboardInterrupt):
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())