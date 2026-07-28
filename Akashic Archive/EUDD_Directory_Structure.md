# EUDD — Directory Structure

## Project Root: `AkashicArchive/`

```
AkashicArchive/
│
├── CMakeLists.txt                      # Top-level build configuration
│
├── src/                                # All source files (flat — 26 modules, no nesting)
│   │
│   ├── main.cpp                        # Entry point: normal mode or --omniscient mode
│   │
│   │── ── Level 0 ──────────────────────────────────────────────
│   ├── precision_stack.h               # [Module  1] MPFR/GMP/Arb 400-bit wrapper, ET constants,
│   ├── precision_stack.cpp             #             SHA-256, CRC-32, elementary + special functions
│   │
│   │── ── Level 1 ──────────────────────────────────────────────
│   ├── core_lattice.h                  # [Module  2] Projection Π_N(r), bijection, k-arithmetic,
│   ├── core_lattice.cpp                #             derived properties, Gaussian signature, elegance
│   │
│   │── ── Level 2 ──────────────────────────────────────────────
│   ├── akashic_format.h                # [Module  3] Sempaevum.akashic I/O, memoization hash table,
│   ├── akashic_format.cpp              #             page management, section directory, integrity
│   │
│   │── ── Level 3 ──────────────────────────────────────────────
│   ├── wal.h / wal.cpp                 # [Module  4] Write-ahead log, crash recovery
│   ├── home_finding.h / home_finding.cpp # [Module 5] §7.11 core projection, LCM tower, CF method
│   ├── generator_system.h / .cpp       # [Module  6] L₁/L₂/L₃ backbone, interval tree, K-complexity
│   ├── event_system.h / .cpp           # [Module  8] Three-times events, tower context, permanence
│   │
│   │── ── Level 4 ──────────────────────────────────────────────
│   ├── relationship_system.h / .cpp    # [Module  9] Relationship classes, provenance chains
│   ├── pattern_system.h / .cpp         # [Module 10] E_hierarchy, LIFE_THRESHOLD promotion
│   ├── tower_system.h / .cpp           # [Module 11] Multifold towers, birth triads, nesting
│   ├── extension_system.h / .cpp       # [Module 21] JSON extensions, 12 types, validation
│   │
│   │── ── Level 5 ──────────────────────────────────────────────
│   ├── discovery_engine.h / .cpp       # [Module  7] Five discovery modes, generator search
│   ├── query.h / query.cpp             # [Module 15] Lattice-algebraic queries, search, subsumption
│   ├── bootstrap.h / bootstrap.cpp     # [Module 12] Initial Sempaevum.akashic generation (~10⁴ values)
│   ├── self_recording.h / .cpp         # [Module 18] Operational metrics, ≤1% overhead, journals
│   ├── active_probing.h / .cpp         # [Module 22] T-signal probes, response/silence detection
│   ├── gaze_module.h / .cpp            # [Module 23] Complete Gaze Equation, four status levels
│   │
│   │── ── Level 6 ──────────────────────────────────────────────
│   ├── ingest.h / ingest.cpp           # [Module 13] File/stream ingestion, format adapters
│   ├── manual_input.h / .cpp           # [Module 14] Seven input modes, real-time preview
│   ├── backup.h / backup.cpp           # [Module 24] VSS snapshots, CRC-32 verification
│   ├── metabolism.h / metabolism.cpp    # [Module 17] Three-layer resource governance (K,V,α⁻¹)
│   │
│   │── ── Level 7 ──────────────────────────────────────────────
│   ├── api.h / api.cpp                 # [Module 16] Named pipe IPC, 79 operations, JSON protocol
│   │
│   │── ── Level 8 ──────────────────────────────────────────────
│   ├── gpu_rendering.h / .cpp          # [Module 19] OpenGL 4.6, six-level LOD, freecam
│   │
│   │── ── Level 9 ──────────────────────────────────────────────
│   ├── gui_main.h / gui_main.cpp       # [Module 20] ImGui context, GLFW window, panel orchestration
│   ├── gui_dashboard.cpp               #   Dashboard: live metrics, discovery status
│   ├── gui_inspector.cpp               #   Property inspector: 120-digit detail, provenance
│   ├── gui_manual_input.cpp            #   Manual input panel: seven modes
│   ├── gui_ingest.cpp                  #   File/stream ingestion panel
│   ├── gui_search.cpp                  #   Search and retrieval
│   ├── gui_connections.cpp             #   Connection manager
│   ├── gui_query.cpp                   #   Query builder
│   ├── gui_events.cpp                  #   Event log viewer
│   ├── gui_settings.cpp                #   Settings panel
│   ├── gui_shutdown.cpp                #   Shutdown confirmation dialog
│   │
│   │── ── Level 10 ─────────────────────────────────────────────
│   ├── shutdown.h / shutdown.cpp       # [Module 25] 6-phase deterministic shutdown
│   │
│   │── ── Separate Process ─────────────────────────────────────
│   ├── omniscient.h / omniscient.cpp   # [Module 26] Watchdog: tamper, crash, journal
│   └── omniscient_main.cpp             #   Entry point for --omniscient mode
│
├── build/                              # CMake build output (generated, not committed)
│
└── logs/                               # Runtime journals (created by Omniscient on first run)
    ├── Omniscient_001.log              # Error/tamper/crash journal (watchdog perspective)
    └── SelfRecording_001.log           # Operational metrics journal (manager perspective)
```

## Runtime Files (generated, same directory as .exe)

```
<install_dir>/
├── EUDD_Manager.exe                    # The single statically-linked executable
├── Sempaevum.akashic                   # The database file (generated on first run)
└── logs/                               # Journals (created on first run)
    ├── Omniscient_NNN.log              # Rotates at ~10 MB, never deleted
    └── SelfRecording_NNN.log           # Rotates at ~10 MB, never deleted
```

## Build Order (forced by dependency graph)

```
Level 0:  precision_stack
Level 1:  core_lattice
Level 2:  akashic_format
Level 3:  wal, home_finding, generator_system, event_system
Level 4:  relationship_system, pattern_system, tower_system, extension_system
Level 5:  discovery_engine, query, bootstrap, self_recording, active_probing, gaze_module
Level 6:  ingest, manual_input, backup, metabolism
Level 7:  api
Level 8:  gpu_rendering
Level 9:  gui (11 sub-files)
Level 10: shutdown
Separate: omniscient (own main(), same exe with --omniscient flag)
```
