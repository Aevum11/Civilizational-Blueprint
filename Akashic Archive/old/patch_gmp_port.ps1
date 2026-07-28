# patch_gmp_port.ps1 — Patches vcpkg GMP portfile to fix Windows VPATH issue
# The bug: GMP sets U_FOR_BUILD='_' on Windows, so make looks for gen-fac_.c
# instead of gen-fac.c. Fix: copy source files + create underscore variants.

param(
    [string]$OverlayDir = "C:\vcpkg-overlay\gmp"
)

# Create overlay directory
if (Test-Path $OverlayDir) { Remove-Item -Recurse -Force $OverlayDir }
New-Item -ItemType Directory -Path $OverlayDir -Force | Out-Null

# Copy existing port files
Copy-Item "C:\vcpkg\ports\gmp\*" $OverlayDir -Recurse

# Read the portfile
$portfilePath = Join-Path $OverlayDir "portfile.cmake"
$content = Get-Content $portfilePath -Raw

# The CMake code to insert before vcpkg_install_make()
$patch = @'

# --- EUDD VPATH FIX (gen-fac.exe Windows build failure) ---
# GMP's configure sets U_FOR_BUILD='_' on Windows when MSVC is the target
# but MSYS2 provides the build tools. This causes TWO problems:
#   1. The Makefile looks for gen-fac_.c instead of gen-fac.c
#   2. The cross-build compiler path uses flags MSVC doesn't understand
# Fix: patch the generated Makefile to clear U_FOR_BUILD, and copy source
# files into build dirs for VPATH fallback.
file(GLOB _gmp_build_dirs
  "${CURRENT_BUILDTREES_DIR}/x64-*-dbg"
  "${CURRENT_BUILDTREES_DIR}/x64-*-rel"
  "${CURRENT_BUILDTREES_DIR}/${TARGET_TRIPLET}-dbg"
  "${CURRENT_BUILDTREES_DIR}/${TARGET_TRIPLET}-rel"
)
list(REMOVE_DUPLICATES _gmp_build_dirs)
foreach(_bdir ${_gmp_build_dirs})
  if(IS_DIRECTORY "${_bdir}")
    # Fix 1: Patch ALL generated Makefiles to fix MSVC/libtool conflicts
    file(GLOB _all_makefiles "${_bdir}/Makefile" "${_bdir}/*/Makefile")
    foreach(_makefile ${_all_makefiles})
      file(READ "${_makefile}" _mf_content)
      # Fix 1a: Clear U_FOR_BUILD so gen-*.c files use normal names
      string(REPLACE "U_FOR_BUILD = _" "U_FOR_BUILD = " _mf_content "${_mf_content}")
      # Fix 1b: Strip -Xcompiler prefixes leaked from libtool
      string(REPLACE "-Xcompiler " "" _mf_content "${_mf_content}")
      # Fix 1c: Strip -RTC1 — libtool misparses this as -R (rpath) TC1
      string(REPLACE " -RTC1" "" _mf_content "${_mf_content}")
      # Fix 1d: Strip -Xlinker prefixes — libtool link mode doesn't understand them
      string(REPLACE "-Xlinker " "" _mf_content "${_mf_content}")
      file(WRITE "${_makefile}" "${_mf_content}")
    endforeach()
    list(LENGTH _all_makefiles _mf_count)
    message(STATUS "EUDD FIX: patched ${_mf_count} Makefiles in ${_bdir}")
    # Fix 2: Copy all source .c and .h files into build dir for VPATH safety
    #         Also copy mini-gmp/ subdir (bootstrap.c #includes mini-gmp/mini-gmp.c)
    file(GLOB _gmp_all_src "${SOURCE_PATH}/*.c" "${SOURCE_PATH}/*.h")
    foreach(_f ${_gmp_all_src})
      get_filename_component(_fname "${_f}" NAME)
      if(NOT EXISTS "${_bdir}/${_fname}")
        file(COPY "${_f}" DESTINATION "${_bdir}")
      endif()
    endforeach()
    if(IS_DIRECTORY "${SOURCE_PATH}/mini-gmp" AND NOT IS_DIRECTORY "${_bdir}/mini-gmp")
      file(COPY "${SOURCE_PATH}/mini-gmp" DESTINATION "${_bdir}")
    endif()
    message(STATUS "EUDD FIX: patched ${_bdir}")
  endif()
endforeach()
# --- END EUDD VPATH FIX ---

'@

# Insert the patch right before vcpkg_install_make(
$content = $content -replace 'vcpkg_install_make\(', ($patch + 'vcpkg_install_make(')

# Write the patched portfile
Set-Content -Path $portfilePath -Value $content -NoNewline

Write-Host "  [OK] GMP overlay port created at $OverlayDir"
Write-Host "  [OK] VPATH fix applied to portfile.cmake"
