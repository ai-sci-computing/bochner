cmake_minimum_required(VERSION 3.20)

# Warning flags. NOTE: add_compile_options propagates into FetchContent
# subdirectories too, so the vendored GLFW/ImGui sources are opted back out
# with -w where they are defined (tools/viewer/CMakeLists.txt) rather than
# being silently counted as ours. Enabled after the tree was made clean under
# -Wall -Wextra: without them, "the build has zero warnings" says very little,
# since nothing was asking for any. Deliberately not -Werror: a first Linux/GCC
# build will surface new diagnostics, and failing the build on them would be a
# worse first experience than seeing them.
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wall> $<$<COMPILE_LANGUAGE:CXX>:-Wextra>)

project(bochner
  VERSION 0.0.1
  DESCRIPTION "Connection-Laplacian numerics and vortex-filament fluid visualization"
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Export compile_commands.json for tooling (clangd, etc.)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(BOCHNER_WITH_PETSC "Enable PETSc/SLEPc-backed linear algebra and solvers" OFF)
option(BOCHNER_WITH_VIEWER "Build the interactive OpenGL viewer (fetches GLFW + Dear ImGui)" OFF)
option(BOCHNER_WITH_OPENMP "Parallelise the gauge-multigrid kernels with OpenMP" ON)
option(BOCHNER_WITH_METAL "Build the Metal GPU compute backend (Apple only)" OFF)

# ---------------------------------------------------------------------------
# Metal GPU backend (opt-in, Apple only). Shaders are compiled from source at
# RUNTIME via the Metal framework, so this needs only the Command Line Tools --
# no Xcode `metal`/`metallib` offline compiler. Enables the ObjC++ language for
# the .mm host code and locates the Metal + Foundation frameworks.
# ---------------------------------------------------------------------------
if(BOCHNER_WITH_METAL)
  if(NOT APPLE)
    message(FATAL_ERROR "BOCHNER_WITH_METAL is Apple-only")
  endif()
  enable_language(OBJCXX)
  find_library(METAL_FRAMEWORK Metal REQUIRED)
  find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
  message(STATUS "Metal GPU backend enabled")
endif()

# ---------------------------------------------------------------------------
# OpenMP (gauge-multigrid matvec/smoother parallelism). AppleClang has no
# built-in OpenMP, so fall back to Homebrew libomp when find_package misses.
# ---------------------------------------------------------------------------
if(BOCHNER_WITH_OPENMP)
  find_package(OpenMP QUIET)
  if(NOT OpenMP_CXX_FOUND AND APPLE)
    find_path(LIBOMP_INC omp.h PATHS /opt/homebrew/opt/libomp/include /usr/local/opt/libomp/include)
    find_library(LIBOMP_LIB omp PATHS /opt/homebrew/opt/libomp/lib /usr/local/opt/libomp/lib)
    if(LIBOMP_INC AND LIBOMP_LIB)
      add_library(OpenMP::OpenMP_CXX INTERFACE IMPORTED)
      target_compile_options(OpenMP::OpenMP_CXX INTERFACE -Xpreprocessor -fopenmp)
      target_include_directories(OpenMP::OpenMP_CXX INTERFACE ${LIBOMP_INC})
      target_link_libraries(OpenMP::OpenMP_CXX INTERFACE ${LIBOMP_LIB})
      set(OpenMP_CXX_FOUND TRUE)
    endif()
  endif()
  if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP enabled for gauge-multigrid kernels")
  else()
    message(STATUS "OpenMP not found; gauge-multigrid runs single-threaded")
  endif()
endif()

# ---------------------------------------------------------------------------
# PETSc + SLEPc (built from source; see scripts/build_petsc_slepc.sh).
# Discovered via pkg-config. Override the install prefixes with
#   -DBOCHNER_PETSC_DIR=... -DBOCHNER_SLEPC_DIR=...
# ---------------------------------------------------------------------------
if(BOCHNER_WITH_PETSC)
  set(BOCHNER_PETSC_DIR "$ENV{HOME}/.local/petsc" CACHE PATH "PETSc install prefix")
  set(BOCHNER_SLEPC_DIR "$ENV{HOME}/.local/slepc" CACHE PATH "SLEPc install prefix")
  set(ENV{PKG_CONFIG_PATH}
      "${BOCHNER_PETSC_DIR}/lib/pkgconfig:${BOCHNER_SLEPC_DIR}/lib/pkgconfig:$ENV{PKG_CONFIG_PATH}")
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(PETSC REQUIRED IMPORTED_TARGET PETSc)
  pkg_check_modules(SLEPC REQUIRED IMPORTED_TARGET SLEPc)
  # PETSc/SLEPc headers include <mpi.h>; we compile with clang++ (not mpicxx),
  # so pull in MPI's include/link paths explicitly.
  find_package(MPI REQUIRED COMPONENTS CXX)
  message(STATUS "Found PETSc ${PETSC_VERSION}, SLEPc ${SLEPC_VERSION}, MPI ${MPI_CXX_VERSION}")

  # One interface target bundling the whole backend.
  add_library(bochner_petsc INTERFACE)
  target_link_libraries(bochner_petsc INTERFACE
    PkgConfig::PETSC PkgConfig::SLEPC MPI::MPI_CXX)
else()
  message(WARNING
    "BOCHNER_WITH_PETSC=OFF: building WITHOUT the PETSc/SLEPc backend. The "
    "pressure projection, the CF fluid pipeline, and the SLEPc eigensolver "
    "baselines are EXCLUDED, and 13 of the 31 test suites are skipped (32 "
    "with the opt-in Metal parity suite). The "
    "gauge-MG eigensolver accuracy is still checked PETSc-free "
    "(test_gauge_eigen_accuracy). Configure with -DBOCHNER_WITH_PETSC=ON for "
    "the full build.")
endif()

# ---------------------------------------------------------------------------
# Vendored test framework (doctest, single header)
# ---------------------------------------------------------------------------
add_library(doctest INTERFACE)
target_include_directories(doctest INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/doctest)

# ---------------------------------------------------------------------------
# Core library
# ---------------------------------------------------------------------------
add_subdirectory(src)

# ---------------------------------------------------------------------------
# Tests (ctest + doctest)
# ---------------------------------------------------------------------------
enable_testing()
add_subdirectory(tests)

# ---------------------------------------------------------------------------
# Demo tools + benchmarks. PETSc-free ones (mg_bench) build unconditionally;
# the tools/ CMakeLists gates the SLEPc/CF-pipeline ones on BOCHNER_WITH_PETSC.
# ---------------------------------------------------------------------------
add_subdirectory(tools)

# ---------------------------------------------------------------------------
# Interactive OpenGL viewer (opt-in; vendors GLFW + Dear ImGui via FetchContent)
# ---------------------------------------------------------------------------
if(BOCHNER_WITH_VIEWER)
  # The viewers no longer require PETSc. Previously they did -- not because the
  # live pipeline uses it (it does not), but because ONE solveSpdCG call in the
  # legacy stateless projection gated the whole fluid library, dragging
  # MacProjector out with it. That function now lives in MacProjectionLegacy.cpp
  # and only the SLEPc Lanczos *baseline* inside the viewers is PETSc-gated:
  # without PETSc the extraction-solver combo simply omits that entry.
  if(NOT BOCHNER_WITH_PETSC)
    message(STATUS "Viewer: building without PETSc -- the SLEPc Lanczos extraction "
                   "baseline is unavailable; the gauge-MG paths are unaffected.")
  endif()
  add_subdirectory(tools/viewer)
endif()

# ---------------------------------------------------------------------------
# Doxygen API docs:  cmake --build build --target docs
# ---------------------------------------------------------------------------
find_package(Doxygen)
if(DOXYGEN_FOUND)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in
                 ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
  add_custom_target(docs
    COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Generating API documentation with Doxygen"
    VERBATIM)
else()
  message(STATUS "Doxygen not found; 'docs' target unavailable")
endif()
