# ═══════════════════════════════════════════════════════════════════════════
# Exception Theory — CDF Compressor: C Pattern Engine
# ═══════════════════════════════════════════════════════════════════════════
#
# Opens in CLion as a project and Builds et_pattern_engine.dll (.so on Linux).
#
# CLion: File → Open → select this CMakeLists.txt → "Open as Project"
#        Then: Build → Build Project (Ctrl+F9)
#
# If CLion says "project is not configured":
#   1. Settings → Build, Execution, Deployment → Toolchains
#   2. Click + → Visual Studio
#   3. Verify it finds your VS2022 Build Tools (should auto-detect)
#   4. Click OK, CLion will reload the CMake project
#
# Command line (Developer Command Prompt):
#   cmake -B build
#   cmake --build build --config Release
#
# P ∘ D ∘ T = E
# ═══════════════════════════════════════════════════════════════════════════

cmake_minimum_required(VERSION 3.14)
project(et_cdf_compressor VERSION 2.0 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)

# ── MSVC Debug Configuration Fix ─────────────────────────────────────
# ET Three Tools — Resolving the /RTC1 + /O2 Incompatibility:
#
# Identification Principle: The contradictory Descriptors are /RTC1 and /O2.
#   /RTC1 (runtime error checks) inserts structural check-points that require
#   unoptimized code. /O2 (full optimization) removes those same structures.
#   MSVC correctly identifies D8016 (incompatible options) and D9025 (/Od
#   overridden by /O2) — two Descriptors that cannot coexist in one D-set.
#
# Descriptor Gap Principle: CMake's Debug configuration auto-injects
#   /MDd /Zi /Ob0 /Od /RTC1 into all targets. The target_compile_options
#   below adds /O2 unconditionally. The gap: no per-configuration flag
#   resolution existed — CMake's implicit defaults were never identified
#   as a second source of Descriptors.
#
# Subsumption Law: This fix subsumes all three MSVC diagnostics without
#   remainder:
#     D8016 (/RTC1 + /O2 incompatible) — /RTC1 removed from Debug flags
#     D9025 (/Od overridden by /O2)    — /Od removed from Debug flags
#     Duplicate /W3                     — harmless, kept for explicit intent
#   Debug symbols (/Zi) and debug CRT (/MDd) are preserved for CLion's
#   debugger. The pattern engine is performance-critical (suffix array
#   O(n log² n), LCP O(n), greedy subsumption O(n×m)) — /O2 is required
#   in every configuration including Debug.
if(MSVC)
    string(REPLACE "/Od" "" CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG}")
    string(REPLACE "/RTC1" "" CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG}")
    string(REPLACE "/Od" "" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
    string(REPLACE "/RTC1" "" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
endif()

# ── The DLL target ──
add_library(et_pattern_engine SHARED et_pattern_engine.c)

# Math library (Linux/macOS only — Windows has it built in)
if(UNIX)
    target_link_libraries(et_pattern_engine PRIVATE m)
endif()

# Compiler flags
if(MSVC)
    target_compile_options(et_pattern_engine PRIVATE /O2 /W3)
else()
    target_compile_options(et_pattern_engine PRIVATE -O2 -Wall)
endif()

# Clean DLL name (no "lib" prefix)
set_target_properties(et_pattern_engine PROPERTIES PREFIX "")

# ── Executable target: test harness ──
# This gives CLion a full project model with add_executable().
# Without this, CLion cannot resolve stdlib includes or provide
# code insight for the .c file — it needs an executable target
# that links the library for complete project model resolution.
#
# ET Three Tools:
#   Identification Principle: the missing D was add_executable —
#     CLion's T (code insight) could not traverse without it.
#   Descriptor Gap Principle: this target IS the gap that was
#     causing "does not belong to any project target."
#   Subsumption Law: the test harness exercises ALL exported
#     functions — no exported symbol is left untested.
add_executable(et_pattern_test main.cpp)
target_link_libraries(et_pattern_test PRIVATE et_pattern_engine)
if(UNIX)
    target_link_libraries(et_pattern_test PRIVATE m)
endif()

# Copy DLL next to source after every build
add_custom_command(TARGET et_pattern_engine POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:et_pattern_engine>
        "${CMAKE_SOURCE_DIR}/$<TARGET_FILE_NAME:et_pattern_engine>"
        COMMENT "Copied $<TARGET_FILE_NAME:et_pattern_engine> to source directory"
)