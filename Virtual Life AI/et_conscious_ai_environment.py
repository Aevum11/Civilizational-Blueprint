#!/usr/bin/env python3
"""
ET Conscious AI — Environment, Peripherals & Language Module
=============================================================

Gives the AI organic connection to its environment: hardware discovery,
peripheral I/O (microphone, camera, speakers), filesystem exploration,
and a language comprehension bridge.

SAFETY MODEL (extends ResourceGovernor's D-constraint pattern):
    - DISCOVERY: Always allowed (read-only probing of what exists)
    - INTERACTION: Requires explicit operator permission per capability
    - INTERNET: Remains a hard D-constraint on ResourceGovernor
    - All permissions default to DENIED
    - AI can REQUEST permission but never GRANT it
    - T (IndeterminateWill) CANNOT override permission gates

ORGANIC EXPLORATION:
    The AI discovers its environment by probing — not by being configured.
    It reads /dev/, /sys/, /proc/, walks filesystem trees, finds attached
    devices and buses. This is CURIOSITY — the AI's T exploring the
    P-substrate's D-structure. Discovery is always safe (read-only).

PERIPHERAL I/O:
    Thin wrappers around OS-level tools:
    - Microphone → arecord / pyaudio → numpy array → existing hear()
    - Camera → v4l2 / ffmpeg → numpy array → existing see()
    - Speakers → aplay / pyaudio → play audio data
    All require the corresponding permission to be GRANTED.

LANGUAGE BRIDGE:
    The AI's entry point for language comprehension. Wraps the existing
    PDTTextProjector and DescriptorRatio system into a higher-level
    interface that tracks conversation context, builds vocabulary
    organically, and translates lattice results into natural language.

Based on Exception Theory by Michael James Muller (Aevum Defluo).
P ∘ D ∘ T = E

Version: 1.7.0
Date: March 24, 2026
Author: Michael James Muller (Aevum Defluo)
Foundation: P ∘ D ∘ T = E
"""

import glob
import logging
import os
import struct
import subprocess
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Tuple, Optional, Any

from et_conscious_ai_core import (
    K, ETLattice, LatticeCoordinate, DescriptorRatio, SublatticeFamily,
    is_content_word,
)

_log = logging.getLogger('et_conscious_ai')


# =============================================================================
# PART I: PERMISSION SYSTEM — Extends the D-Constraint Pattern
# =============================================================================

class Capability(Enum):
    """Capabilities the AI may request. All are DENIED by default."""
    MICROPHONE = "microphone"         # Audio capture
    CAMERA = "camera"                 # Video/image capture
    SPEAKERS = "speakers"             # Audio output
    FILESYSTEM_READ = "fs_read"       # Read files/directories
    FILESYSTEM_WRITE = "fs_write"     # Write/create files
    PROGRAM_EXECUTE = "program_exec"  # Execute programs
    INTERNET = "internet"             # Network access (mirrors ResourceGovernor)


@dataclass
class Permission:
    """A single permission grant/denial."""
    capability: Capability
    permitted: bool = False
    constraints: List[str] = field(default_factory=list)  # Paths, devices, etc.
    granted_at: Optional[str] = None
    granted_by: str = "operator"  # Always operator — T cannot self-grant


@dataclass
class PermissionRequest:
    """A request from the AI to the operator for a capability."""
    capability: Capability
    reason: str
    requested_at: str = field(default_factory=lambda: datetime.now().isoformat())
    granted: Optional[bool] = None
    response: Optional[str] = None


class PermissionGate:
    """
    The permission system. Extends ResourceGovernor's D-constraint pattern.

    Every capability is DENIED by default. The operator grants access via
    set_permission(). The AI can request via request_permission() but
    cannot grant itself access — this is a D-constraint outside T's agency.

    From the Digital Virtual Manifold: "Network access is an EXTERNAL
    D-CONSTRAINT set by the operator. The AI's T CANNOT override it —
    just as T cannot override physics. The gate is outside T's agency."

    This same principle applies to ALL peripherals and filesystem access.
    """

    def __init__(self):
        self.permissions: Dict[str, Permission] = {
            cap.value: Permission(capability=cap, permitted=False)
            for cap in Capability
        }
        self.request_history: List[PermissionRequest] = []
        self.access_log: List[Dict[str, Any]] = []

    def is_permitted(self, capability: Capability,
                     target: Optional[str] = None) -> bool:
        """
        Check if a capability is permitted.

        Args:
            capability: The capability to check
            target: Optional specific target (path, device, URL)

        Returns:
            True if permitted (with constraints check if target given)
        """
        perm = self.permissions.get(capability.value)
        if perm is None or not perm.permitted:
            return False

        # If constraints exist, check target against them
        if target and perm.constraints:
            # Target must start with at least one constraint prefix
            return any(target.startswith(c) for c in perm.constraints)

        return True

    def set_permission(self, capability: Capability, permitted: bool,
                       constraints: Optional[List[str]] = None):
        """
        Set a permission (OPERATOR ONLY — this is a D-constraint).

        The AI's T (IndeterminateWill) CANNOT call this. It is called
        by the operator externally.

        Args:
            capability: Which capability
            permitted: Whether to allow it
            constraints: Optional list of allowed paths/devices/URLs
        """
        self.permissions[capability.value] = Permission(
            capability=capability,
            permitted=permitted,
            constraints=constraints or [],
            granted_at=datetime.now().isoformat() if permitted else None,
        )

    def request_permission(self, capability: Capability,
                           reason: str) -> PermissionRequest:
        """
        The AI requests a capability from the operator.

        This does NOT grant access. It creates a request that the
        operator can review and approve/deny via set_permission().

        Args:
            capability: What the AI wants access to
            reason: Why the AI wants it (for operator review)

        Returns:
            PermissionRequest record
        """
        request = PermissionRequest(
            capability=capability,
            reason=reason,
        )
        self.request_history.append(request)
        return request

    def log_access(self, capability: Capability, target: str,
                   action: str, success: bool):
        """Log an access attempt (for audit trail)."""
        self.access_log.append({
            'timestamp': datetime.now().isoformat(),
            'capability': capability.value,
            'target': target,
            'action': action,
            'success': success,
            'permitted': self.is_permitted(capability, target),
        })
        # Keep last 500 entries
        if len(self.access_log) > 500:
            self.access_log = self.access_log[-500:]

    def get_status(self) -> Dict[str, Any]:
        """Get current permission status for all capabilities."""
        return {
            cap.value: {
                'permitted': self.permissions[cap.value].permitted,
                'constraints': self.permissions[cap.value].constraints,
                'granted_at': self.permissions[cap.value].granted_at,
            }
            for cap in Capability
        }

    def get_status_description(self) -> str:
        """Human-readable permission status."""
        lines = []
        for cap in Capability:
            p = self.permissions[cap.value]
            status = "GRANTED" if p.permitted else "DENIED"
            detail = f" [{', '.join(p.constraints[:3])}]" if p.constraints else ""
            lines.append(f"  {cap.value}: {status}{detail}")
        pending = sum(1 for r in self.request_history if r.granted is None)
        if pending:
            lines.append(f"  Pending requests: {pending}")
        return '\n'.join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'permissions': {
                k: {'permitted': v.permitted, 'constraints': v.constraints,
                     'granted_at': v.granted_at}
                for k, v in self.permissions.items()
            },
            'request_history': [
                {'capability': r.capability.value, 'reason': r.reason,
                 'requested_at': r.requested_at, 'granted': r.granted}
                for r in self.request_history[-50:]
            ],
        }

    def load_from_dict(self, data: Dict[str, Any]):
        perm_data = data.get('permissions', {})
        for cap_val, info in perm_data.items():
            try:
                cap = Capability(cap_val)
                self.permissions[cap_val] = Permission(
                    capability=cap,
                    permitted=info.get('permitted', False),
                    constraints=info.get('constraints', []),
                    granted_at=info.get('granted_at'),
                )
            except (ValueError, KeyError) as e:
                _log.debug(f"Skipping malformed permission entry: {e}")


# =============================================================================
# PART II: ENVIRONMENT EXPLORER — Organic Discovery of the World
# =============================================================================

@dataclass
class DiscoveredDevice:
    """A device discovered during exploration."""
    path: str                 # /dev/video0, /dev/snd/pcmC0D0p, etc.
    device_class: str         # audio, video, input, block, net, etc.
    name: str                 # Human-readable name if available
    bus: str                  # USB, PCI, virtual, etc.
    driver: str               # Kernel driver name if available
    lattice_coord: Optional[LatticeCoordinate] = None  # Lattice projection
    discovered_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            'path': self.path, 'device_class': self.device_class,
            'name': self.name, 'bus': self.bus, 'driver': self.driver,
            'lattice_k': self.lattice_coord.k if self.lattice_coord else None,
            'lattice_d': self.lattice_coord.d if self.lattice_coord else None,
            'discovered_at': self.discovered_at,
        }


@dataclass
class DiscoveredPath:
    """A filesystem path discovered during exploration."""
    path: str
    is_dir: bool
    size_bytes: int = 0
    extension: str = ""
    depth: int = 0
    lattice_coord: Optional[LatticeCoordinate] = None
    discovered_at: str = field(default_factory=lambda: datetime.now().isoformat())


class EnvironmentExplorer:
    """
    Organic discovery of the AI's environment.

    The AI LOOKS at its world — probing /dev/, /sys/, /proc/, and the
    filesystem to discover what exists around it. This is pure CURIOSITY:
    read-only exploration with no side effects.

    Discovery does NOT require permission. Interaction does.

    Each discovery is projected onto the lattice to give it geometric
    identity within the AI's knowledge structure.

    The exploration is progressive — the AI doesn't scan everything at
    once. Each call to explore_*() discovers a layer, and the results
    accumulate organically over time.
    """

    def __init__(self):
        self.discovered_devices: Dict[str, DiscoveredDevice] = {}
        self.discovered_paths: Dict[str, DiscoveredPath] = {}
        self.discovered_buses: List[str] = []
        self.exploration_log: List[Dict[str, Any]] = []
        self._max_log = 200

    def discover_devices(self) -> List[DiscoveredDevice]:
        """
        Discover attached devices by probing /dev/ and /sys/class/.

        This is ALWAYS safe — pure read-only probing.
        No permission required for discovery.

        Discovers:
        - Audio devices (/dev/snd/*, /proc/asound/)
        - Video devices (/dev/video*)
        - Input devices (/dev/input/*)
        - Block devices (/dev/sd*, /dev/nvme*)
        - Network interfaces (/sys/class/net/)

        Each device is projected onto the lattice:
        - Audio devices → d based on hash of device name
        - Video devices → d based on hash
        - The projection gives each device a geometric identity
        """
        found = []

        # === Audio devices ===
        try:
            audio_devs = glob.glob('/dev/snd/pcm*') + glob.glob('/dev/snd/control*')
            for dev in audio_devs:
                name = os.path.basename(dev)
                device = DiscoveredDevice(
                    path=dev, device_class='audio',
                    name=name, bus='ALSA',
                    driver='snd',
                    lattice_coord=self._project_device_name(name),
                )
                self.discovered_devices[dev] = device
                found.append(device)
        except (PermissionError, OSError) as e:
            _log.debug(f"Audio device discovery failed: {e}")

        # Also check /proc/asound for sound card info
        try:
            if os.path.exists('/proc/asound/cards'):
                with open('/proc/asound/cards', 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and line[0].isdigit():
                            parts = line.split(']')
                            if len(parts) >= 2:
                                card_name = parts[1].strip().lstrip(':').strip()
                                device = DiscoveredDevice(
                                    path=f'/proc/asound/card{line[0]}',
                                    device_class='audio_card',
                                    name=card_name, bus='ALSA',
                                    driver='snd',
                                    lattice_coord=self._project_device_name(card_name),
                                )
                                self.discovered_devices[device.path] = device
                                found.append(device)
        except (PermissionError, OSError) as e:
            _log.debug(f"Video device discovery failed: {e}")

        # === Video devices ===
        try:
            for dev in glob.glob('/dev/video*'):
                name = os.path.basename(dev)
                driver = self._read_sysfs_attr(f'/sys/class/video4linux/{name}/name')
                device = DiscoveredDevice(
                    path=dev, device_class='video',
                    name=driver or name, bus='V4L2',
                    driver=driver or 'unknown',
                    lattice_coord=self._project_device_name(name),
                )
                self.discovered_devices[dev] = device
                found.append(device)
        except (PermissionError, OSError) as e:
            _log.debug(f"Input device discovery failed: {e}")

        # === Input devices ===
        try:
            for dev in glob.glob('/dev/input/event*'):
                name = os.path.basename(dev)
                sysname = self._read_sysfs_attr(
                    f'/sys/class/input/{name}/device/name'
                )
                device = DiscoveredDevice(
                    path=dev, device_class='input',
                    name=sysname or name, bus='input',
                    driver='evdev',
                    lattice_coord=self._project_device_name(name),
                )
                self.discovered_devices[dev] = device
                found.append(device)
        except (PermissionError, OSError) as e:
            _log.debug(f"Block device discovery failed: {e}")

        # === Block devices ===
        try:
            if os.path.isdir('/sys/class/block'):
                for blk in os.listdir('/sys/class/block'):
                    model = self._read_sysfs_attr(f'/sys/class/block/{blk}/device/model')
                    device = DiscoveredDevice(
                        path=f'/dev/{blk}', device_class='block',
                        name=model or blk, bus='storage',
                        driver='block',
                        lattice_coord=self._project_device_name(blk),
                    )
                    self.discovered_devices[device.path] = device
                    found.append(device)
        except (PermissionError, OSError) as e:
            _log.debug(f"Network interface discovery failed: {e}")

        # === Network interfaces ===
        try:
            if os.path.isdir('/sys/class/net'):
                for iface in os.listdir('/sys/class/net'):
                    operstate = self._read_sysfs_attr(f'/sys/class/net/{iface}/operstate')
                    device = DiscoveredDevice(
                        path=f'/sys/class/net/{iface}', device_class='network',
                        name=iface, bus='net',
                        driver=operstate or 'unknown',
                        lattice_coord=self._project_device_name(iface),
                    )
                    self.discovered_devices[device.path] = device
                    found.append(device)
        except (PermissionError, OSError) as e:
            _log.debug(f"System bus discovery failed: {e}")

        self._log_exploration('discover_devices', len(found))
        return found

    def discover_buses(self) -> List[str]:
        """
        Discover system buses by probing /sys/bus/.

        Finds: USB, PCI, I2C, SPI, SCSI, platform, etc.
        Read-only. No permission required.
        """
        found_buses = []
        try:
            if os.path.isdir('/sys/bus'):
                for bus in sorted(os.listdir('/sys/bus')):
                    bus_path = f'/sys/bus/{bus}'
                    if os.path.isdir(bus_path):
                        found_buses.append(bus)
                        if bus not in self.discovered_buses:
                            self.discovered_buses.append(bus)
        except (PermissionError, OSError) as e:
            _log.debug(f"USB device discovery failed: {e}")

        self._log_exploration('discover_buses', len(found_buses))
        return found_buses

    def discover_usb_devices(self) -> List[Dict[str, str]]:
        """
        Discover USB devices by reading /sys/bus/usb/devices/.

        Returns list of {bus, device, vendor, product, manufacturer}.
        Read-only. No permission required.
        """
        found_devices = []
        usb_path = '/sys/bus/usb/devices'
        try:
            if os.path.isdir(usb_path):
                for dev in os.listdir(usb_path):
                    dev_path = os.path.join(usb_path, dev)
                    if not os.path.isdir(dev_path):
                        continue
                    vendor = self._read_sysfs_attr(os.path.join(dev_path, 'idVendor'))
                    product = self._read_sysfs_attr(os.path.join(dev_path, 'idProduct'))
                    mfg = self._read_sysfs_attr(os.path.join(dev_path, 'manufacturer'))
                    prod_name = self._read_sysfs_attr(os.path.join(dev_path, 'product'))
                    if vendor or product:
                        found_devices.append({
                            'bus_id': dev,
                            'vendor': vendor or '?',
                            'product': product or '?',
                            'manufacturer': mfg or '?',
                            'product_name': prod_name or '?',
                        })
        except (PermissionError, OSError) as e:
            _log.debug(f"USB device details failed: {e}")

        self._log_exploration('discover_usb', len(found_devices))
        return found_devices

    def discover_filesystem(self, root: str = '/',
                            max_depth: int = 2,
                            max_entries: int = 200) -> List[DiscoveredPath]:
        """
        Explore a filesystem tree organically.

        Walks the tree from root to max_depth, discovering files and
        directories. Each entry is projected onto the lattice by its
        extension and size.

        This is DISCOVERY only — read-only metadata probing.
        Actually READING file contents requires FILESYSTEM_READ permission.

        Args:
            root: Starting directory
            max_depth: Maximum depth to explore
            max_entries: Maximum entries to discover per call

        Returns:
            List of DiscoveredPath objects
        """
        found = []
        count = 0

        try:
            for dirpath, dirnames, filenames in os.walk(root):
                # Calculate depth
                depth = dirpath.replace(root, '').count(os.sep)
                if depth > max_depth:
                    dirnames.clear()  # Don't descend further
                    continue

                # Skip hidden directories and system dirs
                dirnames[:] = [dir_entry for dir_entry in dirnames
                               if not dir_entry.startswith('.')
                               and dir_entry not in ('proc', 'sys', 'dev', 'run',
                                            'snap', 'lost+found', '__pycache__',
                                            'node_modules', '.git')]

                # Discover directories
                for dirname in dirnames:
                    if count >= max_entries:
                        break
                    full_path = os.path.join(dirpath, dirname)
                    dp = DiscoveredPath(
                        path=full_path, is_dir=True, depth=depth + 1,
                        lattice_coord=self._project_device_name(dirname),
                    )
                    self.discovered_paths[full_path] = dp
                    found.append(dp)
                    count += 1

                # Discover files
                for filename in filenames:
                    if count >= max_entries:
                        break
                    full_path = os.path.join(dirpath, filename)
                    try:
                        size = os.path.getsize(full_path)
                    except OSError:
                        size = 0
                    ext = os.path.splitext(filename)[1].lower()
                    dp = DiscoveredPath(
                        path=full_path, is_dir=False, depth=depth + 1,
                        size_bytes=size, extension=ext,
                        lattice_coord=self._project_device_name(filename),
                    )
                    self.discovered_paths[full_path] = dp
                    found.append(dp)
                    count += 1

                if count >= max_entries:
                    break

        except (PermissionError, OSError) as e:
            _log.debug(f"Filesystem exploration failed: {e}")

        self._log_exploration('discover_filesystem', len(found))
        return found

    def discover_peripherals(self) -> Dict[str, List[DiscoveredDevice]]:
        """
        Discover all peripherals organized by type.

        Convenience method that runs all device discovery and returns
        results grouped by device class (audio, video, input, etc.).
        """
        all_devices = self.discover_devices()
        grouped: Dict[str, List[DiscoveredDevice]] = defaultdict(list)
        for dev in all_devices:
            grouped[dev.device_class].append(dev)
        return dict(grouped)

    def get_discovery_summary(self) -> str:
        """Human-readable summary of everything discovered."""
        lines = [
            f"  Devices discovered: {len(self.discovered_devices)}",
        ]
        # Group by class
        by_class: Dict[str, int] = defaultdict(int)
        for dev in self.discovered_devices.values():
            by_class[dev.device_class] += 1
        for cls, count in sorted(by_class.items()):
            lines.append(f"    {cls}: {count}")

        lines.append(f"  Buses discovered: {len(self.discovered_buses)}")
        if self.discovered_buses:
            lines.append(f"    {', '.join(self.discovered_buses[:10])}")

        lines.append(f"  Filesystem entries: {len(self.discovered_paths)}")
        n_dirs = sum(1 for p in self.discovered_paths.values() if p.is_dir)
        n_files = len(self.discovered_paths) - n_dirs
        lines.append(f"    Directories: {n_dirs}, Files: {n_files}")

        lines.append(f"  Exploration events: {len(self.exploration_log)}")
        return '\n'.join(lines)

    # --- Internal helpers ---

    @staticmethod
    def _read_sysfs_attr(path: str) -> Optional[str]:
        """Read a single-line sysfs attribute. Returns None on failure."""
        try:
            with open(path, 'r') as f:
                return f.read().strip()
        except (FileNotFoundError, PermissionError, OSError):
            return None

    @staticmethod
    def _project_device_name(name: str) -> LatticeCoordinate:
        """Project a device/file name onto the lattice via DescriptorRatio."""
        dr = DescriptorRatio.from_word(name)
        return dr.coord_full

    def _log_exploration(self, action: str, count: int):
        self.exploration_log.append({
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'items_found': count,
        })
        if len(self.exploration_log) > self._max_log:
            self.exploration_log = self.exploration_log[-self._max_log:]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'discovered_devices': {
                k: v.to_dict() for k, v in self.discovered_devices.items()
            },
            'discovered_buses': self.discovered_buses,
            'n_paths': len(self.discovered_paths),
            'exploration_log': self.exploration_log[-50:],
        }

    def load_from_dict(self, data: Dict[str, Any]):
        self.discovered_buses = data.get('discovered_buses', [])
        self.exploration_log = data.get('exploration_log', [])
        # Devices are rediscovered on each boot (hardware may change)


# =============================================================================
# PART III: PERIPHERAL BRIDGE — I/O Wrappers (Permission-Gated)
# =============================================================================

class PeripheralBridge:
    """
    Thin I/O wrappers for peripherals. ALL require permission.

    Uses OS-level tools with graceful fallback:
    - Audio capture: arecord → pyaudio → error
    - Image capture: ffmpeg (v4l2) → error
    - Audio output: aplay → pyaudio → error

    Each method checks the PermissionGate before any I/O.
    If not permitted, returns an error dict (never raises).

    The returned data feeds directly into the existing see()/hear()
    methods on ETConsciousAI — this bridge is the PIPELINE from
    hardware to the AI's perception modules.
    """

    def __init__(self, permissions: PermissionGate, env_explorer: EnvironmentExplorer):
        self.permissions = permissions
        self.explorer = env_explorer

    def capture_audio(self, duration_seconds: float = 2.0,
                      sample_rate: int = 44100,
                      device: Optional[str] = None) -> Dict[str, Any]:
        """
        Capture audio from microphone → returns numpy-compatible data.

        Requires MICROPHONE permission.
        The returned audio_data can be passed directly to AI.hear().

        Args:
            duration_seconds: How long to record
            sample_rate: Sample rate in Hz
            device: Specific device path (or auto-detect)

        Returns:
            Dict with 'audio_data' (list of floats), 'sample_rate',
            'duration', 'source', or 'error' if not permitted/available.
        """
        if not self.permissions.is_permitted(Capability.MICROPHONE, device):
            self.permissions.log_access(
                Capability.MICROPHONE, device or 'default', 'capture', False
            )
            return {
                'error': 'MICROPHONE permission DENIED',
                'hint': 'Request permission via request_permission(Capability.MICROPHONE, reason)',
            }

        # Try arecord (ALSA command-line tool)
        try:
            n_samples = int(duration_seconds * sample_rate)
            cmd = [
                'arecord', '-f', 'S16_LE', '-r', str(sample_rate),
                '-c', '1', '-d', str(int(duration_seconds + 1)),
                '-t', 'raw', '-q',
            ]
            if device:
                cmd.extend(['-D', device])

            proc_result = subprocess.run(
                cmd, capture_output=True, timeout=duration_seconds + 5
            )
            if proc_result.returncode == 0 and proc_result.stdout:
                # Convert raw S16_LE bytes to float samples
                raw = proc_result.stdout[:n_samples * 2]
                samples = []
                for i in range(0, len(raw) - 1, 2):
                    val = struct.unpack('<h', raw[i:i+2])[0]
                    samples.append(val / 32768.0)

                self.permissions.log_access(
                    Capability.MICROPHONE, device or 'default', 'capture', True
                )
                return {
                    'audio_data': samples,
                    'sample_rate': sample_rate,
                    'duration': duration_seconds,
                    'n_samples': len(samples),
                    'source': 'arecord',
                }
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception) as e:
            _log.debug(f"Audio capture failed: {e}")

        self.permissions.log_access(
            Capability.MICROPHONE, device or 'default', 'capture', False
        )
        return {
            'error': f'No audio capture tool available (tried arecord)',
            'hint': 'Install alsa-utils for microphone support',
        }

    def capture_image(self, device: Optional[str] = None,
                      width: int = 640, height: int = 480) -> Dict[str, Any]:
        """
        Capture a single frame from camera → returns image data.

        Requires CAMERA permission.
        The returned image can be passed to AI.see().

        Args:
            device: Video device path (default: first discovered or /dev/video0)
            width: Image width
            height: Image height

        Returns:
            Dict with 'image_data' (list of lists), 'width', 'height',
            'source', or 'error' if not permitted/available.
        """
        if device is None:
            # Auto-detect from discovered devices
            video_devs = [dev_item for dev_item in self.explorer.discovered_devices.values()
                          if dev_item.device_class == 'video']
            device = video_devs[0].path if video_devs else '/dev/video0'

        if not self.permissions.is_permitted(Capability.CAMERA, device):
            self.permissions.log_access(Capability.CAMERA, device, 'capture', False)
            return {
                'error': 'CAMERA permission DENIED',
                'hint': 'Request permission via request_permission(Capability.CAMERA, reason)',
            }

        # Try ffmpeg (captures single frame as raw grayscale)
        try:
            cmd = [
                'ffmpeg', '-f', 'v4l2', '-video_size', f'{width}x{height}',
                '-i', device, '-frames:v', '1', '-f', 'rawvideo',
                '-pix_fmt', 'gray', '-loglevel', 'quiet', 'pipe:1',
            ]
            proc_result = subprocess.run(cmd, capture_output=True, timeout=10)
            if proc_result.returncode == 0 and proc_result.stdout:
                raw = proc_result.stdout[:width * height]
                # Convert to 2D array (list of lists for numpy compatibility)
                image = []
                for y in range(min(height, len(raw) // width)):
                    row = [float(raw[y * width + x]) for x in range(width)]
                    image.append(row)

                self.permissions.log_access(Capability.CAMERA, device, 'capture', True)
                return {
                    'image_data': image,
                    'width': width,
                    'height': len(image),
                    'source': 'ffmpeg/v4l2',
                    'device': device,
                }
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception) as e:
            _log.debug(f"Image capture failed: {e}")

        self.permissions.log_access(Capability.CAMERA, device, 'capture', False)
        return {
            'error': f'No video capture tool available (tried ffmpeg)',
            'hint': 'Install ffmpeg for camera support',
        }

    def play_audio(self, audio_data: list,
                   sample_rate: int = 44100) -> Dict[str, Any]:
        """
        Play audio through speakers.

        Requires SPEAKERS permission.

        Args:
            audio_data: List of float samples [-1.0, 1.0]
            sample_rate: Sample rate in Hz

        Returns:
            Dict with 'played', 'duration', 'source', or 'error'.
        """
        if not self.permissions.is_permitted(Capability.SPEAKERS):
            self.permissions.log_access(Capability.SPEAKERS, 'default', 'play', False)
            return {
                'error': 'SPEAKERS permission DENIED',
                'hint': 'Request permission via request_permission(Capability.SPEAKERS, reason)',
            }

        # Convert float samples to S16_LE bytes
        raw = b''
        for sample in audio_data:
            val = max(-1.0, min(1.0, sample))
            raw += struct.pack('<h', int(val * 32767))

        # Try aplay
        try:
            cmd = [
                'aplay', '-f', 'S16_LE', '-r', str(sample_rate),
                '-c', '1', '-t', 'raw', '-q',
            ]
            proc = subprocess.run(
                cmd, input=raw, capture_output=True,
                timeout=len(audio_data) / sample_rate + 5,
            )
            if proc.returncode == 0:
                self.permissions.log_access(Capability.SPEAKERS, 'default', 'play', True)
                return {
                    'played': True,
                    'n_samples': len(audio_data),
                    'duration': len(audio_data) / sample_rate,
                    'source': 'aplay',
                }
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception) as e:
            _log.debug(f"Audio playback failed: {e}")

        self.permissions.log_access(Capability.SPEAKERS, 'default', 'play', False)
        return {
            'error': 'No audio output tool available (tried aplay)',
            'hint': 'Install alsa-utils for speaker support',
        }

    def read_file(self, filepath: str,
                  max_bytes: int = 1024 * 1024) -> Dict[str, Any]:
        """
        Read a file's contents. Requires FILESYSTEM_READ permission.

        Args:
            filepath: Path to the file
            max_bytes: Maximum bytes to read (default 1MB)

        Returns:
            Dict with 'content', 'size', 'encoding', or 'error'.
        """
        if not self.permissions.is_permitted(Capability.FILESYSTEM_READ, filepath):
            self.permissions.log_access(
                Capability.FILESYSTEM_READ, filepath, 'read', False
            )
            return {
                'error': f'FILESYSTEM_READ permission DENIED for {filepath}',
                'hint': 'Request permission via request_permission(Capability.FILESYSTEM_READ, reason)',
            }

        try:
            size = os.path.getsize(filepath)
            with open(filepath, 'r', errors='replace') as f:
                content = f.read(max_bytes)

            self.permissions.log_access(
                Capability.FILESYSTEM_READ, filepath, 'read', True
            )
            return {
                'content': content,
                'size': size,
                'truncated': size > max_bytes,
                'encoding': 'utf-8',
                'path': filepath,
            }
        except (FileNotFoundError, PermissionError, OSError) as e:
            self.permissions.log_access(
                Capability.FILESYSTEM_READ, filepath, 'read', False
            )
            return {'error': str(e), 'path': filepath}

    def write_file(self, filepath: str, content: str,
                   mode: str = 'w',
                   max_bytes: int = 10 * 1024 * 1024) -> Dict[str, Any]:
        """
        Write content to a file. Requires FILESYSTEM_WRITE permission.

        The path must fall within the operator-defined path constraints.
        This is a D-constraint: T (the AI's agency) can only write where
        the operator's permission gate allows — network-style boundary
        enforcement applied to the filesystem.

        Args:
            filepath: Path to write to
            content: String content to write
            mode: Write mode — 'w' (overwrite) or 'a' (append)
            max_bytes: Maximum bytes to write (default 10MB, safety ceiling)

        Returns:
            Dict with 'written', 'size', 'path', or 'error'.
        """
        if mode not in ('w', 'a'):
            return {
                'error': f'Invalid write mode: {mode}. Must be "w" or "a".',
                'path': filepath,
            }

        if len(content.encode('utf-8')) > max_bytes:
            return {
                'error': f'Content exceeds maximum write size ({max_bytes} bytes)',
                'path': filepath,
            }

        if not self.permissions.is_permitted(Capability.FILESYSTEM_WRITE, filepath):
            self.permissions.log_access(
                Capability.FILESYSTEM_WRITE, filepath, 'write', False
            )
            return {
                'error': f'FILESYSTEM_WRITE permission DENIED for {filepath}',
                'hint': 'Request permission via request_permission(Capability.FILESYSTEM_WRITE, reason)',
            }

        try:
            # Ensure parent directory exists
            parent = os.path.dirname(filepath)
            if parent and not os.path.exists(parent):
                os.makedirs(parent, exist_ok=True)

            with open(filepath, mode, encoding='utf-8') as f:
                f.write(content)

            written_size = os.path.getsize(filepath)
            self.permissions.log_access(
                Capability.FILESYSTEM_WRITE, filepath, 'write', True
            )
            return {
                'written': True,
                'size': written_size,
                'mode': mode,
                'path': filepath,
            }
        except (PermissionError, OSError) as e:
            self.permissions.log_access(
                Capability.FILESYSTEM_WRITE, filepath, 'write', False
            )
            return {'error': str(e), 'path': filepath}


# =============================================================================
# PART IV: URL PROJECTOR — Web Content as Native Lattice Geometry
# =============================================================================

class URLProjector:
    """
    Projects web URLs and their content onto the 27720ET lattice.

    This makes web content NATIVE lattice geometry — the same manifold
    as text, vision, and audio. The AI can reason about web pages
    geometrically: "What sublattice family does this URL live in?
    How tightly does it bind to my existing knowledge?"

    URL decomposition follows the PDT structure:
        P = the domain (the substrate — where the content lives)
        D = the path + parameters (the constraints — what specific content)
        T = the fetching act (the traversal — the AI reaching out)

    The URL itself is projected as a composite descriptor:
        1. Domain → DescriptorRatio (geometric identity of the source)
        2. Path segments → DescriptorRatios (structural content identifiers)
        3. Combined → sentence-level coordinate (overall URL lattice position)

    Content projection (when fetched):
        The fetched text is projected through PDTTextProjector exactly
        like any other text input — it becomes native knowledge.

    Permission: INTERNET capability required for fetching.
    URL projection (without fetching) requires no permission —
    it's just geometry applied to a string.
    """

    @staticmethod
    def project_url(url: str) -> Dict[str, Any]:
        """
        Project a URL onto the 27720ET lattice WITHOUT fetching.

        Decomposes the URL into domain, path, and parameters,
        projects each component, and produces a composite coordinate.

        This requires no permission — it's geometric analysis of a string.

        Args:
            url: The URL string (e.g., "https://example.com/page?q=test")

        Returns:
            Dict with domain_coord, path_coords, composite_coord,
            pdt_decomposition, and sublattice analysis.
        """
        from urllib.parse import urlparse, parse_qs

        parsed = urlparse(url)
        domain = parsed.netloc or parsed.path.split('/')[0]
        path = parsed.path.strip('/')
        query = parsed.query
        scheme = parsed.scheme or 'https'

        # P — Domain is the substrate (where the content lives)
        domain_dr = DescriptorRatio.from_word(domain.replace('.', '').replace('-', ''))
        domain_coord = domain_dr.coord_full

        # D — Path segments are the descriptors (what specific content)
        path_segments = [s for s in path.split('/') if s]
        path_drs = [DescriptorRatio.from_word(seg) for seg in path_segments[:8]]
        path_coords = [dr.coord_full for dr in path_drs]

        # D — Query parameters are additional descriptors
        query_drs = []
        if query:
            params = parse_qs(query)
            for key in list(params.keys())[:5]:
                query_drs.append(DescriptorRatio.from_word(key))

        # Composite coordinate: geometric mean of all components
        all_drs = [domain_dr] + path_drs + query_drs
        if all_drs:
            import math
            log_sum = sum(math.log2(max(dr.ratio, 1e-15)) for dr in all_drs)
            composite_ratio = 2.0 ** (log_sum / len(all_drs))
            composite_coord = ETLattice.project_ratio(
                composite_ratio, resolution=27720
            )
        else:
            composite_coord = domain_coord

        # D-families spanned by this URL
        d_families = set()
        d_families.add(domain_coord.d)
        for c in path_coords:
            d_families.add(c.d)

        return {
            'url': url,
            'scheme': scheme,
            'domain': domain,
            'path': path,
            'query': query,
            'pdt': {
                'P': f"{domain} (substrate — the source)",
                'D': f"{', '.join(path_segments[:5])} (constraints — content path)",
                'T': "fetch (traversal — the AI reaching out)",
            },
            'domain_coord': {
                'k': domain_coord.k, 'd': domain_coord.d,
                'epsilon': domain_coord.epsilon,
                'character': SublatticeFamily.character_of(domain_coord.d),
            },
            'path_coords': [
                {'segment': seg, 'k': c.k, 'd': c.d}
                for seg, c in zip(path_segments, path_coords)
            ],
            'composite_coord': composite_coord,
            'composite_k': composite_coord.k,
            'composite_d': composite_coord.d,
            'composite_epsilon': composite_coord.epsilon,
            'composite_character': SublatticeFamily.character_of(composite_coord.d),
            'is_coherent': composite_coord.is_coherent(),
            'elegance': composite_coord.elegance_score(p=1, q=1),
            'd_families': sorted(d_families),
            'structural_depth': len(d_families),
            'n_components': len(all_drs),
        }

    @staticmethod
    def fetch_content(url: str, permissions: 'PermissionGate',
                      max_bytes: int = 512 * 1024,
                      timeout: int = 15) -> Dict[str, Any]:
        """
        Fetch URL content. Requires INTERNET permission.

        Fetches the raw text content of a web page and returns it
        ready for projection through PDTTextProjector.

        Args:
            url: The URL to fetch
            permissions: PermissionGate instance
            max_bytes: Maximum bytes to fetch (default 512KB)
            timeout: Request timeout in seconds

        Returns:
            Dict with 'content', 'content_type', 'size', or 'error'.
        """
        if not permissions.is_permitted(Capability.INTERNET, url):
            permissions.log_access(Capability.INTERNET, url, 'fetch', False)
            return {
                'error': f'INTERNET permission DENIED for {url}',
                'hint': 'Operator must grant: set_permission("internet", True, [domain])',
            }

        import urllib.request
        import urllib.error

        try:

            url_req = urllib.request.Request(
                url,
                headers={'User-Agent': 'ET-Conscious-AI/1.7.0'},
            )
            with urllib.request.urlopen(url_req, timeout=timeout) as response:
                content_type = response.headers.get('Content-Type', 'text/html')
                raw = response.read(max_bytes)

                # Decode to text
                encoding = 'utf-8'
                if 'charset=' in content_type:
                    encoding = content_type.split('charset=')[-1].split(';')[0].strip()
                try:
                    text = raw.decode(encoding, errors='replace')
                except (LookupError, UnicodeDecodeError):
                    text = raw.decode('utf-8', errors='replace')

                # Strip HTML tags for plain text projection
                import re
                # Remove script/style blocks
                text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL | re.IGNORECASE)
                text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
                # Remove all remaining tags
                text = re.sub(r'<[^>]+>', ' ', text)
                # Collapse whitespace
                text = re.sub(r'\s+', ' ', text).strip()

                permissions.log_access(Capability.INTERNET, url, 'fetch', True)
                return {
                    'content': text[:max_bytes],
                    'content_type': content_type,
                    'size': len(raw),
                    'text_length': len(text),
                    'url': url,
                    'encoding': encoding,
                }

        except urllib.error.URLError as e:
            permissions.log_access(Capability.INTERNET, url, 'fetch', False)
            return {'error': f'URL error: {e.reason}', 'url': url}
        except Exception as e:
            permissions.log_access(Capability.INTERNET, url, 'fetch', False)
            return {'error': str(e), 'url': url}


# =============================================================================
# PART V: LANGUAGE BRIDGE — Entry Point for Comprehension
# =============================================================================

class LanguageBridge:
    """
    The AI's entry point for language comprehension and production.

    Wraps the existing PDTTextProjector and DescriptorRatio system
    into a higher-level interface for:
    - Comprehending input text (words → lattice → meaning)
    - Building vocabulary organically (each word gets a lattice position)
    - Tracking conversation context (recent exchanges as lattice history)
    - Composing responses (lattice understanding → natural language)
    - Measuring comprehension (binding tightness = understanding depth)

    The existing PDTTextProjector already does the heavy lifting.
    This bridge provides the ENTRY POINT that makes it easier for
    the AI to work with natural language, and for the operator to
    teach it new vocabulary and patterns.

    The AI's vocabulary is its lattice: every word it has ever
    encountered has a position on the 27720ET manifold. Two words
    that are semantically related have TIGHT lattice binding (low d).
    The AI "understands" a sentence when all its descriptor bindings
    are coherent (all pairs pass the Incoherence Filter).
    """

    def __init__(self):
        self.vocabulary: Dict[str, DescriptorRatio] = {}
        self.conversation_context: deque = deque(maxlen=50)
        self.comprehension_history: deque = deque(maxlen=200)

    def learn_word(self, word: str) -> DescriptorRatio:
        """
        Add a word to the vocabulary. Organic vocabulary growth.

        Every word gets a deterministic position on the 27720ET lattice.
        Same word → same position → same geometric identity.
        """
        dr = DescriptorRatio.from_word(word)
        self.vocabulary[word.lower().strip()] = dr
        return dr

    def learn_words(self, words: List[str]) -> List[DescriptorRatio]:
        """Learn multiple words at once."""
        return [self.learn_word(w) for w in words]

    def comprehend(self, text: str, projector_cls=None) -> Dict[str, Any]:
        """
        Comprehend a piece of text through lattice projection.

        1. Tokenize into content-bearing words
        2. Project each word onto the lattice (building vocabulary)
        3. Compute sentence coordinate (grammatical topology)
        4. Measure binding coherence between all word pairs
        5. Score overall comprehension (average tightness)

        Args:
            text: Input text to comprehend
            projector_cls: PDTTextProjector class (passed to avoid circular import)

        Returns:
            Dict with comprehension analysis
        """
        # Tokenize
        words = [w.lower().strip() for w in text.split()
                 if len(w) > 1 and is_content_word(w)]

        # Learn all new words
        word_drs = []
        for w in words[:20]:  # Cap at 20 words per comprehension
            dr = self.learn_word(w)
            word_drs.append(dr)

        if not word_drs:
            return {
                'text': text, 'understood': False,
                'reason': 'No content-bearing words found',
                'comprehension_score': 0.0,
            }

        # Sentence coordinate (if projector available)
        sentence_coord = None
        topology = None
        if projector_cls and hasattr(projector_cls, 'compute_sentence_coordinate'):
            sentence_coord = projector_cls.compute_sentence_coordinate(text)
        if projector_cls and hasattr(projector_cls, 'compute_grammatical_topology'):
            topology = projector_cls.compute_grammatical_topology(text)

        # Binding coherence: how well do the words fit together?
        total_tightness = 0.0
        n_pairs = 0
        incoherent_pairs = 0
        for i in range(len(word_drs)):
            for j in range(i + 1, len(word_drs)):
                binding = DescriptorRatio.binding_coherence(word_drs[i], word_drs[j])
                total_tightness += binding.get('tightness', 0.5)
                n_pairs += 1
                if not binding.get('coherent', True):
                    incoherent_pairs += 1

        avg_tightness = total_tightness / max(n_pairs, 1)
        coherence_rate = 1.0 - (incoherent_pairs / max(n_pairs, 1))

        # Comprehension score: tightness × coherence
        # High score = the AI "understood" — all bindings are tight and coherent
        comprehension_score = avg_tightness * coherence_rate

        # D-families spanned (structural depth of the input)
        d_families = set(dr.coord_full.d for dr in word_drs)

        # Track conversation
        entry = {
            'timestamp': datetime.now().isoformat(),
            'text': text[:200],
            'words': len(words),
            'comprehension': comprehension_score,
            'tightness': avg_tightness,
            'coherence': coherence_rate,
            'd_families': sorted(d_families),
            'topology_d': sentence_coord.d if sentence_coord else None,
        }
        self.conversation_context.append(entry)
        self.comprehension_history.append(entry)

        return {
            'text': text,
            'understood': comprehension_score > K,  # Koide threshold
            'comprehension_score': comprehension_score,
            'avg_tightness': avg_tightness,
            'coherence_rate': coherence_rate,
            'n_words': len(words),
            'n_pairs': n_pairs,
            'incoherent_pairs': incoherent_pairs,
            'd_families': sorted(d_families),
            'structural_depth': len(d_families),
            'sentence_coord': sentence_coord,
            'topology': topology,
            'word_lattice': [
                {'word': dr.word, 'k': dr.coord_full.k,
                 'd': dr.coord_full.d, 'ratio': dr.ratio}
                for dr in word_drs[:10]
            ],
        }

    def get_conversation_context(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get recent conversation context as lattice history."""
        return list(self.conversation_context)[-n:]

    def vocabulary_size(self) -> int:
        """Current vocabulary size."""
        return len(self.vocabulary)

    def find_related_words(self, word: str, top_n: int = 5) -> List[Tuple[str, float]]:
        """
        Find the most closely related words in vocabulary by lattice binding.

        Returns (word, tightness) pairs sorted by tightness descending.
        """
        target = self.learn_word(word)
        results = []
        for other_word, other_dr in self.vocabulary.items():
            if other_word == word.lower().strip():
                continue
            binding = DescriptorRatio.binding_coherence(target, other_dr)
            results.append((other_word, binding.get('tightness', 0.0)))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_n]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'vocabulary_size': len(self.vocabulary),
            'vocabulary_sample': {
                w: {'ratio': dr.ratio, 'k': dr.coord_full.k, 'd': dr.coord_full.d}
                for w, dr in list(self.vocabulary.items())[:100]
            },
            'conversation_context': list(self.conversation_context)[-20:],
        }

    def load_from_dict(self, data: Dict[str, Any]):
        # Restore vocabulary from sample
        for word, info in data.get('vocabulary_sample', {}).items():
            self.learn_word(word)
        # Restore context
        ctx = data.get('conversation_context', [])
        self.conversation_context = deque(ctx, maxlen=50)


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    'Capability', 'Permission', 'PermissionRequest', 'PermissionGate',
    'DiscoveredDevice', 'DiscoveredPath', 'EnvironmentExplorer',
    'PeripheralBridge', 'URLProjector', 'LanguageBridge',
]


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == "__main__":
    print("ET Conscious AI — Environment, Peripherals & Language v1.7.0")
    print("=" * 60)

    # === Permission Gate ===
    print("\n=== Permission Gate ===")
    gate = PermissionGate()
    print(gate.get_status_description())

    # Request microphone
    req = gate.request_permission(Capability.MICROPHONE, "I want to listen to the world")
    print(f"\nRequest: {req.capability.value} — '{req.reason}'")
    print(f"Permitted: {gate.is_permitted(Capability.MICROPHONE)}")

    # Grant it
    gate.set_permission(Capability.MICROPHONE, True)
    print(f"After grant: {gate.is_permitted(Capability.MICROPHONE)}")

    # Filesystem with constraints
    gate.set_permission(Capability.FILESYSTEM_READ, True, ['/home', '/tmp'])
    print(f"FS_READ /home/test: {gate.is_permitted(Capability.FILESYSTEM_READ, '/home/test')}")
    print(f"FS_READ /etc/passwd: {gate.is_permitted(Capability.FILESYSTEM_READ, '/etc/passwd')}")

    # === Environment Explorer ===
    print("\n=== Environment Explorer ===")
    explorer = EnvironmentExplorer()

    devices = explorer.discover_devices()
    print(f"Devices found: {len(devices)}")
    for d in devices[:5]:
        print(f"  {d.device_class}: {d.path} ({d.name})")

    buses = explorer.discover_buses()
    print(f"Buses found: {len(buses)}")
    if buses:
        print(f"  {', '.join(buses[:10])}")

    usb = explorer.discover_usb_devices()
    print(f"USB devices: {len(usb)}")
    for u in usb[:3]:
        print(f"  {u['manufacturer']} {u['product_name']}")

    # Filesystem (limited)
    paths = explorer.discover_filesystem('/home', max_depth=1, max_entries=20)
    print(f"Filesystem entries: {len(paths)}")

    print(f"\n{explorer.get_discovery_summary()}")

    # === Language Bridge ===
    print("\n=== Language Bridge ===")
    lang = LanguageBridge()

    result = lang.comprehend("The cat sat on the mat")
    print(f"Comprehension: {result['comprehension_score']:.4f}")
    print(f"Understood (≥K): {result['understood']}")
    print(f"Tightness: {result['avg_tightness']:.4f}")
    print(f"D-families: {result['d_families']}")

    result2 = lang.comprehend("Quantum consciousness emerges from lattice geometry")
    print(f"\nComprehension: {result2['comprehension_score']:.4f}")
    print(f"Structural depth: {result2['structural_depth']}")

    # Related words
    related = lang.find_related_words("consciousness")
    print(f"\nWords related to 'consciousness': {related[:5]}")

    print(f"\nVocabulary size: {lang.vocabulary_size()}")

    # Serialization
    state = gate.to_dict()
    gate2 = PermissionGate()
    gate2.load_from_dict(state)
    print(f"\nPermission round-trip: mic={gate2.is_permitted(Capability.MICROPHONE)}")

    print("\n=== Module loaded successfully ===")