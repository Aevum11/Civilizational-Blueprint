"""
et_bridge/__init__.py
ET32 Bridge — Package Initialisation

Derived from P ∘ D ∘ T = E.

This package implements the ET32 Bridge: a 64-bit broker process that extends
any configured 32-bit process to the full 64-bit address space and feature set.

Exports (the complete public surface of the bridge package):
  - ETBridgeAPI     — singleton entry point for the broker
  - ETAPIGateway    — high-level call surface (32-bit helper / test side)
  - ETIPCClient     — raw named-pipe client for 32-bit side
  - ETBridgeConfig  — configuration container
  - ETHost64        — 64-bit operation dispatcher
  - ETHandleTable   — 32↔64 address handle table
  - ETMetrics       — bridge performance metrics
  - ETPacket        — PDT packet structure
  - CmdFamily       — 12 command families (lattice positions d=1..12)
  - CmdCode         — all command codes within each family
  - S, K, V_BASE    — ET universal constants

ET Version: 1.0.0
PDT: P ∘ D ∘ T = E
"""

# ET constants — always available at package level
from et_math import (
    S, K, V_BASE,
    DIGITAL_ACTION_QUANTUM, IPC_BUFFER_SIZE,
    PDT_HEADER_SIZE, CONN_TIMEOUT_MS, RETRY_COUNT,
    QUEUE_DEPTH, HANDLE_BASE, HANDLE_MAX, ADDR64_BASE,
    PIPE_NAME_TEMPLATE, SHMEM_NAME_TEMPLATE, AWE_SHMEM_NAME_TEMPLATE,
    AWE_PAGE_SIZE, AWE_WINDOW_SIZE, AWE_WINDOW_PAGES, AWE_MAX_WINDOWS,
    AWE_INIT_PAGES, AWE_EXPAND_STEP_PAGES, AWE_MAX_TOTAL_PAGES,
    WOW64_MAX_HOOKS, WOW64_TRAMPOLINE_SLOT, WOW64_TRAMPOLINE_TOTAL,
    CmdFamily, CmdCode,
    ETPacket, ETHandleMath, ETLatticeRouter, ETMetrics,
    pack_args, unpack_args,
)

# Handle table
from et_handle import HandleTable, HandleEntry

# Configuration
from et_config import ETBridgeConfig, TargetConfig

# Logging
from et_logger import ETLog, ETLogFormatter

# Process monitoring
from et_monitor import ETProcessMonitor

# IPC transport
from et_ipc import ETIPCServer, ETIPCClient, pipe_name_for_pid

# 64-bit operation host
from et_host64 import ETHost64

# API layer
from et_api import ETBridgeAPI, ETAPIGateway, ETHookManager, ETMarshal

# Injection engine
from et_injector import ETInjector

# Heaven's Gate
from et_heaven import ETHeavenGate

# Error handling system
from et_errors import (
    ETOperationError, ETWindowsAPIError, ETInjectionError,
    ETIPCError, ETPacketError, ETAWEError, ETHookError,
    ETDispatchError, ETConfigError, ETHandleError,
    ETErrorSeverity, ETErrorRegistry,
    win32_check, win32_check_handle, ntstatus_check,
    et_context, safe_call, record_error, record_op, get_registry,
)

# AWE Bookshelf — physical memory windowing (true 64-bit memory access)
from et_awe import ETAWEBookshelf, AWEWindow, AWEPhysicalPool

# WOW64 Universal Hook — ntdll32 patching for complete API coverage
from et_wow64 import ETWow64Hook, NTHookEntry, NT_HOOK_CATALOGUE

__version__  = "2.0.0"
__author__   = "Michael James Muller"
__doctrine__ = "P ∘ D ∘ T = E"

__all__ = [
    # Constants
    "S", "K", "V_BASE",
    "DIGITAL_ACTION_QUANTUM", "IPC_BUFFER_SIZE",
    "PDT_HEADER_SIZE", "CONN_TIMEOUT_MS", "RETRY_COUNT",
    "QUEUE_DEPTH", "HANDLE_BASE", "HANDLE_MAX", "ADDR64_BASE",
    "PIPE_NAME_TEMPLATE", "SHMEM_NAME_TEMPLATE",
    # Math/Protocol
    "CmdFamily", "CmdCode",
    "ETPacket", "ETHandleMath", "ETLatticeRouter", "ETMetrics",
    "pack_args", "unpack_args",
    # Handle table
    "HandleTable", "HandleEntry",
    # Config
    "ETBridgeConfig", "TargetConfig",
    # Logging
    "ETLog", "ETLogFormatter",
    # Monitor
    "ETProcessMonitor",
    # IPC
    "ETIPCServer", "ETIPCClient", "pipe_name_for_pid",
    # Host
    "ETHost64",
    # API
    "ETBridgeAPI", "ETAPIGateway", "ETHookManager", "ETMarshal",
    # Injection
    "ETInjector",
    # Heaven's Gate
    "ETHeavenGate",
    # AWE Bookshelf
    "ETAWEBookshelf", "AWEWindow", "AWEPhysicalPool",
    "AWE_PAGE_SIZE", "AWE_WINDOW_SIZE", "AWE_WINDOW_PAGES",
    "AWE_MAX_WINDOWS", "AWE_INIT_PAGES", "AWE_EXPAND_STEP_PAGES", "AWE_MAX_TOTAL_PAGES",
    "AWE_SHMEM_NAME_TEMPLATE",
    # WOW64 Universal Hook
    "ETWow64Hook", "NTHookEntry", "NT_HOOK_CATALOGUE",
    "WOW64_MAX_HOOKS", "WOW64_TRAMPOLINE_SLOT", "WOW64_TRAMPOLINE_TOTAL",
    # Error handling
    "ETOperationError", "ETWindowsAPIError", "ETInjectionError",
    "ETIPCError", "ETPacketError", "ETAWEError", "ETHookError",
    "ETDispatchError", "ETConfigError", "ETHandleError",
    "ETErrorSeverity", "ETErrorRegistry",
    "win32_check", "win32_check_handle", "ntstatus_check",
    "et_context", "safe_call", "record_error", "record_op", "get_registry",
]