#!/usr/bin/env bash
#
# Build PETSc + SLEPc from source for the bochner project.
#
# Rationale: Homebrew's petsc hard-depends on
# hdf5-mpi, which conflicts with the user's existing hdf5. Building from
# source with --with-hdf5=0 sidesteps that, and lets us pull hypre (the AMG
# baseline we benchmark against in Phase 3) and an optimized build.
#
# Installs into $PREFIX/petsc and $PREFIX/slepc (default ~/.local).
# Idempotent-ish: re-running re-fetches/re-builds. Logs to stdout.
#
# VERSIONS ARE PINNED. The published baselines are timed against a specific
# PETSc/SLEPc, so tracking the `release` branch would silently change what the
# reported speedups are speedups *over*. Override for a different pair:
#   PETSC_TAG=v3.26.0 SLEPC_TAG=v3.26.0 ./scripts/build_petsc_slepc.sh
set -euo pipefail

# Pinned to the versions the paper's baseline numbers were measured against.
PETSC_TAG="${PETSC_TAG:-v3.25.2}"
SLEPC_TAG="${SLEPC_TAG:-v3.25.1}"

PREFIX="${PREFIX:-$HOME/.local}"
SRC="${SRC:-$HOME/.local/src}"
PETSC_PREFIX="$PREFIX/petsc"
SLEPC_PREFIX="$PREFIX/slepc"

# --- platform detection ------------------------------------------------------
UNAME="$(uname -s)"
if [ "$UNAME" = "Darwin" ]; then
  JOBS="$(sysctl -n hw.ncpu)"
  # Apple's Accelerate provides BLAS/LAPACK; no download needed.
  BLAS_ARGS=(--with-blaslapack-lib="-framework Accelerate")
  ARCH_GUESS=arch-darwin-c-opt
else
  JOBS="$(nproc 2>/dev/null || echo 4)"
  # Prefer a system BLAS if pkg-config knows one; otherwise let PETSc fetch a
  # reference implementation so the script works on a bare container.
  if pkg-config --exists openblas 2>/dev/null; then
    BLAS_ARGS=(--with-blaslapack-lib="$(pkg-config --libs openblas)")
  else
    BLAS_ARGS=(--download-fblaslapack)
  fi
  ARCH_GUESS=arch-linux-c-opt
fi

# MPI wrappers: take them from PATH, falling back to Homebrew's location on
# macOS (where they are commonly not on PATH).
find_mpi() {  # find_mpi <name> <homebrew-fallback>
  if command -v "$1" >/dev/null 2>&1; then command -v "$1"
  elif [ -x "$2" ]; then echo "$2"
  else echo "ERROR: $1 not found (install MPI, e.g. 'brew install open-mpi' or" \
            "'apt install libopenmpi-dev')" >&2; exit 1
  fi
}
MPICC="$(find_mpi mpicc  /opt/homebrew/bin/mpicc)"
MPICXX="$(find_mpi mpicxx /opt/homebrew/bin/mpicxx)"
MPIFC="$(find_mpi mpif90 /opt/homebrew/bin/mpif90)"

echo "platform=$UNAME jobs=$JOBS"
echo "petsc=$PETSC_TAG slepc=$SLEPC_TAG"
echo "mpicc=$MPICC"

mkdir -p "$SRC"

echo "==================== PETSc ===================="
cd "$SRC"
if [ ! -d petsc ]; then
  git clone --depth 1 -b "$PETSC_TAG" https://gitlab.com/petsc/petsc.git
fi
cd petsc
git fetch --depth 1 origin tag "$PETSC_TAG" >/dev/null 2>&1 || true
git checkout -q "$PETSC_TAG"
git clean -fdx . >/dev/null 2>&1 || true

./configure \
  --prefix="$PETSC_PREFIX" \
  --with-cc="$MPICC" \
  --with-cxx="$MPICXX" \
  --with-fc="$MPIFC" \
  --with-hdf5=0 \
  --with-debugging=0 \
  "${BLAS_ARGS[@]}" \
  --download-hypre \
  COPTFLAGS="-O2" CXXOPTFLAGS="-O2" FOPTFLAGS="-O2"

# configure prints the exact make line; PETSC_ARCH is empty for prefix builds
# Prefix builds put objects under a PETSC_ARCH directory whose name is
# platform-dependent; use whichever configure actually created rather than
# hardcoding the macOS one.
PETSC_ARCH_BUILT="$(ls -d "$SRC"/petsc/arch-*-c-opt 2>/dev/null | head -1)"
PETSC_ARCH_BUILT="$(basename "${PETSC_ARCH_BUILT:-$ARCH_GUESS}")"
echo "PETSC_ARCH=$PETSC_ARCH_BUILT"
make PETSC_DIR="$SRC/petsc" PETSC_ARCH="$PETSC_ARCH_BUILT" all
make PETSC_DIR="$SRC/petsc" PETSC_ARCH="$PETSC_ARCH_BUILT" install
make PETSC_DIR="$PETSC_PREFIX" PETSC_ARCH="" check || true

echo "==================== SLEPc ===================="
export PETSC_DIR="$PETSC_PREFIX"
export PETSC_ARCH=""
cd "$SRC"
if [ ! -d slepc ]; then
  git clone --depth 1 -b "$SLEPC_TAG" https://gitlab.com/slepc/slepc.git
fi
cd slepc
git fetch --depth 1 origin tag "$SLEPC_TAG" >/dev/null 2>&1 || true
git checkout -q "$SLEPC_TAG"
git clean -fdx . >/dev/null 2>&1 || true

./configure --prefix="$SLEPC_PREFIX"
make SLEPC_DIR="$SRC/slepc" PETSC_DIR="$PETSC_PREFIX"
make SLEPC_DIR="$SRC/slepc" PETSC_DIR="$PETSC_PREFIX" install

echo "==================== DONE ===================="
echo "PETSC_DIR=$PETSC_PREFIX"
echo "SLEPC_DIR=$SLEPC_PREFIX"
ls "$PETSC_PREFIX/lib/pkgconfig/" 2>&1
ls "$SLEPC_PREFIX/lib/pkgconfig/" 2>&1
