cmake_minimum_required(VERSION 3.20)
project(ET_32to64_Bridge C CXX)

# =============================================================================
# ET32 Bridge — CMake Build Configuration
# Derived from P ∘ D ∘ T = E
# Author: Michael James Muller (Aevum Defluo) — Exception Theory
#
# Primary target: et_bridge32.dll (32-bit C DLL — the injected bridge)
# Secondary target: et32_dll_verify (32-bit C++ tool — verifies DLL exports)
#
# Usage:
#   cmake -B build -G "Visual Studio 17 2022" -A Win32
#   cmake --build build --config Release
#
# Or with MinGW-w64 i686 (32-bit):
#   cmake -B build -G "MinGW Makefiles" \
#       -DCMAKE_C_FLAGS="-m32" -DCMAKE_CXX_FLAGS="-m32"
#   cmake --build build
#
# ET constants governing the build:
#   S  = 12   (lattice cardinality — export families)
#   hd = 4096 (digital action quantum — buffer granularity)
# =============================================================================

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 20)

# ---------------------------------------------------------------------------
# Target 1: et_bridge32.dll — the 32-bit injected bridge DLL
# ---------------------------------------------------------------------------

# Generate the .def file from the canonical export list.
# Every exported function is a D-binding point: a Descriptor the broker
# can call via CreateRemoteThread or the 32-bit client can import.
set(ET_DEF_FILE "${CMAKE_CURRENT_BINARY_DIR}/et_bridge32.def")
file(WRITE "${ET_DEF_FILE}"
"LIBRARY et_bridge32\n"
"EXPORTS\n"
"    ET32_Init\n"
"    ET32_Shutdown\n"
"    ET32_IsConnected\n"
"    ET32_GetVersion\n"
"    ET32_SetKiFastTrampoline\n"
"    ET32_BridgeVirtualAlloc\n"
"    ET32_BridgeVirtualAlloc64\n"
"    ET32_BridgeLoadLibrary64\n"
"    ET32_BridgeGetProcAddress64\n"
"    ET32_BridgeCall64\n"
"    ET32_BridgeRegOpenKey64\n"
"    ET32_PythonExec64\n"
"    ET32_UniversalHook\n"
"    ET32_KiFastHook\n"
"    ET32_DynamicDispatch\n"
"    ET32_AWE_ReserveWindow\n"
"    ET32_AWE_MapPages\n"
"    ET32_AWE_UnmapWindow\n"
"    ET32_AWE_SlideWindow\n"
"    ET32_AWE_ReleaseWindow\n"
"    ET32_AWE_DirectAlloc\n"
)

add_library(et_bridge32 SHARED et_bridge32.c)

target_link_libraries(et_bridge32 PRIVATE kernel32 advapi32)
set_target_properties(et_bridge32 PROPERTIES
    OUTPUT_NAME "et_bridge32"
    PREFIX ""
    SUFFIX ".dll"
    LINK_FLAGS "/DEF:\"${ET_DEF_FILE}\""
)

# MinGW-compatible link flags (selected automatically by generator)
if(CMAKE_C_COMPILER_ID STREQUAL "GNU")
    target_compile_options(et_bridge32 PRIVATE -m32 -O2 -Wall -Wextra -Wno-unused-parameter)
    set_target_properties(et_bridge32 PROPERTIES
        LINK_FLAGS "-m32 -shared -Wl,--subsystem,windows -Wl,--enable-stdcall-fixup ${ET_DEF_FILE}"
    )
endif()

# MSVC-specific flags
if(MSVC)
    target_compile_options(et_bridge32 PRIVATE /Ox /W4)
    # /LD is implicit for SHARED libraries in CMake
endif()

# ---------------------------------------------------------------------------
# Target 2: et32_dll_verify — build verification tool
# Loads et_bridge32.dll and verifies all exports resolve.
# ---------------------------------------------------------------------------

add_executable(et32_dll_verify main.cpp)
target_link_libraries(et32_dll_verify PRIVATE kernel32)
add_dependencies(et32_dll_verify et_bridge32)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(et32_dll_verify PRIVATE -m32)
    target_link_options(et32_dll_verify PRIVATE -m32)
endif()