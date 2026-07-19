/// \file
/// Phase-3 benchmark harness: time the smallest-eigenvector solve of the
/// connection Laplacian for the competing methods, as a function of frustration
/// strength and grid size. Lanczos (SLEPc) is the matvec-only baseline; the
/// inverse-iteration variants (one per inner preconditioner) are what the
/// Phase-3 structured methods (even-odd, geometric MG, Wilson-Dirac aggregation)
/// must beat. Reports wall time, solver iterations, and lambda_min.
///
/// Usage: eig_benchmark [n ...]   (grid sizes; default 16 24 32)
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <petsclog.h>
#include <petscmat.h>
#include <slepc.h>

#include "solvers/EigenSolver.h"
#include "grid/GridOperators.h"
#include "extraction/MacConnectionLaplacian.h"
#include "grid/MacGrid.h"

using namespace bochner;

namespace {

double wobble(int n) { return std::sin(0.8 * n + 0.3) + 0.4 * std::cos(2.3 * n); }

// Connection Laplacian with O(1) random angles scaled by the frustration s.
CooMatrix laplacian(const MacGrid& g, double s) {
  FaceField th = ops::zeroFaceField(g);
  for (size_t f = 0; f < th.x.size(); ++f) th.x[f] = s * wobble((int)f + 2);
  for (size_t f = 0; f < th.y.size(); ++f) th.y[f] = s * wobble((int)f + 5);
  for (size_t f = 0; f < th.z.size(); ++f) th.z[f] = s * wobble((int)f + 11);
  return connectionLaplacian(g, th);
}

template <class F>
double timeMs(F&& f) {
  auto t0 = std::chrono::steady_clock::now();
  f();
  auto t1 = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// Cumulative number of MatMult (matvec) calls so far -- the honest unit of work
// shared by Lanczos, the inverse-iteration inner CG, and LOBPCG. Returns -1 if
// the count is not yet queryable (stage-0 perf array unsized before the first
// event of a run, or grown lazily by GAMG/hypre) -- guarded so the harness never
// reports a misleading 0 nor spews a (fatal) PETSc logging error.
long matvecCount() {
  PetscLogEvent ev = 0;
  if (PetscLogEventGetId("MatMult", &ev) != 0 || ev < 0) return -1;
  PetscEventPerfInfo info;
  PetscPushErrorHandler(PetscReturnErrorHandler, nullptr);
  const PetscErrorCode ierr = PetscLogEventGetPerfInfo(0, ev, &info);  // stage 0 = main
  PetscPopErrorHandler();
  if (ierr != 0) return -1;
  return static_cast<long>(info.count);
}

// Matvec delta between two counts, or -1 ("n/a") if either was unmeasurable.
long mvDelta(long after, long before) {
  return (after < 0 || before < 0) ? -1 : after - before;
}

// Render a matvec count for the table (-1 -> "n/a").
std::string mvStr(long mv) {
  if (mv < 0) return "   n/a";
  char buf[16];
  std::snprintf(buf, sizeof(buf), "%6ld", mv);
  return buf;
}

// Scaling study: re-measure conditioning and the
// *contender* ranking at the interactivity target (100k-500k cells). The
// dominated single-vector inverse-iteration methods are omitted -- they take
// minutes per point at these sizes and are already known to lose. The question
// this answers: does the SPD conditioning grow enough at scale to make the
// linear solve hard (reviving the preconditioner story), or does it stay
// moderate (confirming the reframing)?
void runScaling(const std::vector<int>& sizes) {
  const std::vector<double> frust = {0.1, 0.8};
  std::printf("%5s %8s %5s %8s | %15s | %15s | %15s\n", "n", "cells", "s", "cond",
              "Lanczos(ms/mv)", "sinvert(ms/mv)", "LOBPCG+AMG ms/it");
  std::printf("%s\n", std::string(90, '-').c_str());
  for (int n : sizes) {
    MacGrid g(n, n, n, 1.0 / n);
    for (double s : frust) {
      const CooMatrix E = laplacian(g, s);
      EigenPair lan, lmax, si, lba;
      long m0;
      m0 = matvecCount();
      double tL = timeMs([&] { lan = smallestEigenpairLanczos(E, 1e-7); });
      long mvL = mvDelta(matvecCount(), m0);
      lmax = largestEigenvalue(E);
      m0 = matvecCount();
      double tS = timeMs([&] { si = smallestEigenpairShiftInvert(E, 1e-7, nullptr, InnerPC::AMG); });
      long mvS = mvDelta(matvecCount(), m0);
      double tBA = timeMs([&] { lba = smallestEigenpairLOBPCG(E, 1e-7, nullptr, InnerPC::AMG); });
      std::printf("%5d %8d %5.2f %8.0f | %6.0fms %s | %6.0fms %s | %6.0fms %5d\n", n, g.numCells(),
                  s, lmax.value / lan.value, tL, mvStr(mvL).c_str(), tS, mvStr(mvS).c_str(), tBA,
                  lba.iterations);
      std::fflush(stdout);
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  SlepcInitialize(&argc, &argv, nullptr, nullptr);
  PetscLogDefaultBegin();  // so matvec calls are counted
  {
    // "scale" mode: contender ranking + conditioning at the interactivity target.
    if (argc > 1 && std::string(argv[1]) == "scale") {
      std::vector<int> sizes;
      for (int i = 2; i < argc; ++i) sizes.push_back(std::atoi(argv[i]));
      if (sizes.empty()) sizes = {46, 64, 79};  // ~97k, 262k, 493k cells
      runScaling(sizes);
      SlepcFinalize();
      return 0;
    }
    std::vector<int> sizes;
    for (int i = 1; i < argc; ++i) sizes.push_back(std::atoi(argv[i]));
    if (sizes.empty()) sizes = {16, 24, 32};
    const std::vector<double> frust = {0.1, 0.4, 0.8};

    // Report wall time + a work measure for each method. For the single-vector
    // methods (Lanczos, inverse iteration) the honest shared unit is MATVECS
    // (MatMult count). LOBPCG applies the operator to its whole trial *block*
    // (a block-matvec MatMult cannot see), so its natural, robustly-captured
    // measure is its block-ITERATION count -- hence the "ms/it" header there.
    // Report wall time + a work measure for each eigensolver method. Single-vector
    // methods (Lanczos, shift-invert, backward iteration) report MATVECS; LOBPCG
    // applies the operator to its whole trial *block*, so it reports block ITERATIONS.
    std::printf("%4s %5s %7s | %14s | %14s | %14s | %14s\n", "n", "s", "cond",
                "Lanczos(ms/mv)", "sinvert(ms/mv)", "backward(ms/mv)", "LOBPCG+AMG ms/it");
    std::printf("%s\n", std::string(88, '-').c_str());
    for (int n : sizes) {
      MacGrid g(n, n, n, 1.0 / n);  // unit box; h scales the operator like the sim
      for (double s : frust) {
        const CooMatrix E = laplacian(g, s);
        EigenPair lan, lmax, si, bw, lba;
        long m0;
        m0 = matvecCount();
        double tL = timeMs([&] { lan = smallestEigenpairLanczos(E, 1e-7); });
        long mvL = mvDelta(matvecCount(), m0);
        lmax = largestEigenvalue(E);  // for the true condition number
        // Shift-invert Krylov-Schur: Krylov on A^{-1}, sigma=0; single-vector,
        // work unit = matvecs (incl. the inner CG solves).
        m0 = matvecCount();
        double tS = timeMs([&] { si = smallestEigenpairShiftInvert(E, 1e-7, nullptr, InnerPC::AMG); });
        long mvS = mvDelta(matvecCount(), m0);
        // Backward (inverse) iteration with the chosen backsolve (CG + ICC).
        m0 = matvecCount();
        double tB = timeMs([&] { bw = smallestEigenpair(E, 1e-6, nullptr, InnerPC::ICC); });
        long mvB = mvDelta(matvecCount(), m0);
        // LOBPCG: the block method (the degenerate-pair answer), AMG
        // via STPRECOND. Work = block iterations (block-matvec MatMult can't see).
        double tBA = timeMs([&] { lba = smallestEigenpairLOBPCG(E, 1e-7, nullptr, InnerPC::AMG); });
        double cond = lmax.value / lan.value;
        std::printf("%4d %5.2f %7.0f | %5.0fms %s | %5.0fms %s | %5.0fms %s | %5.0fms %5d\n", n, s,
                    cond, tL, mvStr(mvL).c_str(), tS, mvStr(mvS).c_str(), tB, mvStr(mvB).c_str(), tBA,
                    lba.iterations);
        std::fflush(stdout);
      }
    }
  }
  SlepcFinalize();
  return 0;
}
