# ============================================================================
# EUDD Manager — The ET Universal Discovery Database
# CMake Build Configuration
#
# P ∘ D ∘ T = E
#
# Build: cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE=%VCPKG_ROOT%/scripts/buildsystems/vcpkg.cmake
#        cmake --build build --config Release
# ============================================================================

cmake_minimum_required(VERSION 3.24)

project(EUDD_Manager
        VERSION 1.0.0
        DESCRIPTION "ET Universal Discovery Database — The Sempaevum on disk"
        LANGUAGES CXX C
)

# ── C++ Standard ────────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── Static Runtime (single statically-linked .exe per spec §7.10) ───────────
if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    # Disable warnings that fire inside GMP/MPFR/FLINT C headers (not our code)
    add_compile_options(/W4 /WX /wd4244 /wd4267 /wd4100 /wd4459 /wd4146)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS NOMINMAX)
else()
    add_compile_options(-Wall -Wextra -Wpedantic -Werror)
endif()

# ── Find Dependencies ───────────────────────────────────────────────────────
# GMP — Arbitrary-precision integer/rational arithmetic
find_path(GMP_INCLUDE_DIR NAMES gmp.h)
find_library(GMP_LIBRARY NAMES gmp libgmp gmp-10)
if(NOT GMP_INCLUDE_DIR OR NOT GMP_LIBRARY)
    message(FATAL_ERROR
            "GMP not found.\n"
            "Install via vcpkg: vcpkg install gmp:x64-windows-static\n"
            "Or set GMP_INCLUDE_DIR and GMP_LIBRARY manually."
    )
endif()
message(STATUS "GMP include: ${GMP_INCLUDE_DIR}")
message(STATUS "GMP library: ${GMP_LIBRARY}")

# MPFR — 1200-bit (361-dps) floating-point: the ET precision substrate
find_path(MPFR_INCLUDE_DIR NAMES mpfr.h)
find_library(MPFR_LIBRARY NAMES mpfr libmpfr mpfr-6)
if(NOT MPFR_INCLUDE_DIR OR NOT MPFR_LIBRARY)
    message(FATAL_ERROR
            "MPFR not found.\n"
            "Install via vcpkg: vcpkg install mpfr:x64-windows-static\n"
            "Or set MPFR_INCLUDE_DIR and MPFR_LIBRARY manually."
    )
endif()
message(STATUS "MPFR include: ${MPFR_INCLUDE_DIR}")
message(STATUS "MPFR library: ${MPFR_LIBRARY}")

# FLINT (≥3.0, includes Arb) — ζ, Γ, polylog, hypergeometric at 361-dps
find_path(FLINT_INCLUDE_DIR NAMES flint/flint.h)
find_library(FLINT_LIBRARY NAMES flint libflint)
if(NOT FLINT_INCLUDE_DIR OR NOT FLINT_LIBRARY)
    message(FATAL_ERROR
            "FLINT (≥3.0) not found.\n"
            "Install via vcpkg: vcpkg install flint:x64-windows-static\n"
            "Or set FLINT_INCLUDE_DIR and FLINT_LIBRARY manually."
    )
endif()
message(STATUS "FLINT include: ${FLINT_INCLUDE_DIR}")
message(STATUS "FLINT library: ${FLINT_LIBRARY}")

# ── Source Files ────────────────────────────────────────────────────────────
# Stage 1: Precision Stack (Module 1, Level 0)
# Stage 2: Core Lattice Engine (Module 2, Level 1)
# Stage 3: Akashic Format (Module 3, Level 2)
set(STAGE1_SOURCES
        src/precision_stack.h
        src/precision_stack.cpp
        src/core_lattice.h
        src/core_lattice.cpp
        src/akashic_format.h
        src/akashic_format.cpp
        src/main.cpp
)

# ── Executable ──────────────────────────────────────────────────────────────
add_executable(EUDD_Manager ${STAGE1_SOURCES})

target_include_directories(EUDD_Manager PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${GMP_INCLUDE_DIR}
        ${MPFR_INCLUDE_DIR}
        ${FLINT_INCLUDE_DIR}
)

target_link_libraries(EUDD_Manager PRIVATE
        ${FLINT_LIBRARY}
        ${MPFR_LIBRARY}
        ${GMP_LIBRARY}
)

# FLINT uses pthreads for its thread pool
find_library(PTHREAD_LIBRARY NAMES pthreadVC3)
if(PTHREAD_LIBRARY)
    target_link_libraries(EUDD_Manager PRIVATE ${PTHREAD_LIBRARY})
endif()

# On Windows, FLINT may need additional system libraries
if(WIN32)
    target_link_libraries(EUDD_Manager PRIVATE
            bcrypt     # For potential system crypto (not used — we implement SHA-256/CRC-32 natively)
            ws2_32     # For named pipe IPC (future stages)
    )
endif()

# ── Compile Definitions ────────────────────────────────────────────────────
target_compile_definitions(EUDD_Manager PRIVATE
        EUDD_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
        EUDD_VERSION_MINOR=${PROJECT_VERSION_MINOR}
        EUDD_VERSION_PATCH=${PROJECT_VERSION_PATCH}
)

# ── Installation ────────────────────────────────────────────────────────────
install(TARGETS EUDD_Manager RUNTIME DESTINATION .)

message(STATUS "")
message(STATUS "═══════════════════════════════════════════════")
message(STATUS "  EUDD Manager v${PROJECT_VERSION}")
message(STATUS "  P ∘ D ∘ T = E")
message(STATUS "  Stage 1: Precision Stack (Module 1, Level 0)")
message(STATUS "  Stage 2: Core Lattice Engine (Module 2, Level 1)")
message(STATUS "  Stage 3: Akashic Format (Module 3, Level 2)")
message(STATUS "═══════════════════════════════════════════════")
message(STATUS "")