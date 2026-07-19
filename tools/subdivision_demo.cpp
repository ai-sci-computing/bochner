/// \file
/// Experiments A & C: can gauge-aware subdivision (no linear solve) feed
/// vortex-filament extraction, and does a little smoothing sharpen it?
///
/// Setup = the seeded-ring ground truth (tests/test_mac_extraction), on a cubic
/// domain so the subdivision hierarchy can coarsen deeply.
///   A. subdivision alone vs the SLEPc smallest eigenpair (the truth).
///   C. subdivision section + K Richardson smoothing sweeps toward the smallest
///      eigenvector (x <- x - alpha*E*x, normalised; alpha = 1/lambda_max damps
///      the high modes -- the multigrid smoother the 2017 post anticipates).
/// We watch the traced loop sharpen (mean radius, planarity, radial deviation)
/// and the Rayleigh quotient fall toward the true lambda_min, all far cheaper
/// than the cold eigensolve (the interactivity bottleneck).
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <slepc.h>

#include "grid/CooMatrix.h"
#include "solvers/EigenSolver.h"
#include "extraction/MacConnectionLaplacian.h"
#include "extraction/MacFilaments.h"
#include "grid/MacGrid.h"
#include "solvers/GaugeEigen.h"
#include "solvers/GaugeMultigrid.h"
#include "solvers/MacSubdivision.h"
#include "fluid/MacVortexRing.h"

using bochner::Filament;
using bochner::MacGrid;
using bochner::Vec3;

namespace {

struct LoopStats {
  int numFilaments = 0, numClosed = 0, points = 0;
  double meanR = 0.0, maxAbsZ = 0.0, maxRadialDev = 0.0, length = 0.0, minMag = 0.0;
};

LoopStats analyze(const MacGrid& g, const std::vector<double>& psi) {
  LoopStats s;
  s.minMag = 1e300;
  for (int c = 0; c < g.numCells(); ++c)
    s.minMag = std::min(s.minMag, std::hypot(psi[2 * c], psi[2 * c + 1]));

  const auto crossings = bochner::traceZeroSet(g, psi);
  const auto filaments = bochner::linkFilaments(g, crossings);
  s.numFilaments = static_cast<int>(filaments.size());

  const Filament* ring = nullptr;
  for (const auto& f : filaments) {
    if (f.closed) ++s.numClosed;
    if (f.closed && (!ring || f.points.size() > ring->points.size())) ring = &f;
  }
  if (!ring) return s;

  const auto& p = ring->points;
  s.points = static_cast<int>(p.size());
  for (std::size_t a = 0; a < p.size(); ++a) {
    s.meanR += std::sqrt(p[a][0] * p[a][0] + p[a][1] * p[a][1]);
    s.maxAbsZ = std::max(s.maxAbsZ, std::abs(p[a][2]));
    const Vec3& b = p[(a + 1) % p.size()];
    s.length += std::sqrt((p[a][0] - b[0]) * (p[a][0] - b[0]) + (p[a][1] - b[1]) * (p[a][1] - b[1]) +
                          (p[a][2] - b[2]) * (p[a][2] - b[2]));
  }
  s.meanR /= p.size();
  for (const auto& q : p)
    s.maxRadialDev =
        std::max(s.maxRadialDev, std::abs(std::sqrt(q[0] * q[0] + q[1] * q[1]) - s.meanR));
  return s;
}

void report(const char* label, double ms, const LoopStats& s, double rq = -1.0) {
  std::printf("%-20s %8.1f ms | filaments=%d closed=%d | loop pts=%d meanR=%.3f max|z|=%.3f "
              "radDev=%.3f",
              label, ms, s.numFilaments, s.numClosed, s.points, s.meanR, s.maxAbsZ, s.maxRadialDev);
  if (rq >= 0.0) std::printf(" | rayleigh=%.5f", rq);
  std::printf(" | min|psi|=%.2e\n", s.minMag);
}

template <class F>
double timeMs(F&& f) {
  const auto t0 = std::chrono::steady_clock::now();
  f();
  const auto t1 = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// Sparse matvec y = E x from compressed COO entries (E symmetric, real embedding).
std::vector<double> matvec(const std::vector<bochner::CooMatrix::Entry>& E, const std::vector<double>& x) {
  std::vector<double> y(x.size(), 0.0);
  for (const auto& e : E) y[e.row] += e.value * x[e.col];
  return y;
}

double dot(const std::vector<double>& a, const std::vector<double>& b) {
  double s = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
  return s;
}

}  // namespace

int main(int argc, char** argv) {
  SlepcInitialize(&argc, &argv, nullptr, nullptr);

  // Cubic domain [-L,L]^3 so the hierarchy can coarsen deeply; ring in z=0.
  // Usage: subdivision_demo [n] [levels] (cube side in cells, default 32,
  // ideally 2^k; levels = subdivision depth, default the deepest the grid
  // allows -- pass a smaller value to probe the depth-dependence of the
  // seed's zero-set topology).
  const int n = (argc > 1) ? std::atoi(argv[1]) : 32;
  const int levelsArg = (argc > 2) ? std::atoi(argv[2]) : -1;
  const double L = 1.4, h = 2.0 * L / n;
  const MacGrid g(n, n, n, h, Vec3{-L, -L, -L});
  const double R = 0.7, Gamma = 1.0, hbar = Gamma / (2.0 * M_PI);
  const bochner::FaceField u =
      bochner::vortexRingFaceField(g, {0, 0, 0}, {0, 0, 1}, R, Gamma, /*coreRadius=*/0.15);
  const bochner::FaceField theta = bochner::connectionAngles(g, u, hbar);
  const auto E = bochner::connectionLaplacian(g, theta);
  const auto Ec = E.compressed();

  std::printf("Ring R=%.2f Gamma=%.1f on %dx%dx%d (h=%.3f). Reference radius R=%.2f.\n\n", R, Gamma,
              g.nx(), g.ny(), g.nz(), g.spacing(), R);

  // (1) Ground truth: smallest eigenvector.
  double lamMin = 0.0;
  {
    std::vector<double> psi;
    const double ms = timeMs([&] {
      const auto e = bochner::smallestEigenpair(E);
      psi = e.vector;
      lamMin = e.value;
    });
    report("eigensolver (truth)", ms, analyze(g, psi), lamMin);
  }

  // (2) Subdivision alone, deepest hierarchy (single-cell base = the post's seed),
  // or the requested depth.
  int maxLevels = 0;
  while ((n % (1 << (maxLevels + 1))) == 0) ++maxLevels;
  const int levels = (levelsArg >= 0) ? std::min(levelsArg, maxLevels) : maxLevels;
  std::vector<double> psi0;
  const double msSub = timeMs([&] { psi0 = bochner::subdivisionSection(g, theta, levels); });
  std::printf("\n");
  char subLabel[32];
  std::snprintf(subLabel, sizeof(subLabel), "subdivision L=%d", levels);
  report(subLabel, msSub, analyze(g, psi0));

  // (2b) Does the seed help covMG-LOBPCG? Outer iterations from a cold (constant)
  // start vs. warm-started with the subdivision seed of this depth.
  {
    const bochner::GaugeLattice lat = bochner::gaugeLatticeFromFaces(g, theta);
    bochner::GaugeEigenResult cold, warm;
    const double msCold = timeMs([&] { cold = bochner::smallestEigenpairGaugeMG(lat); });
    std::vector<std::complex<double>> seed(g.numCells());
    for (int c = 0; c < g.numCells(); ++c) seed[c] = {psi0[2 * c], psi0[2 * c + 1]};
    const double msWarm = timeMs([&] { warm = bochner::smallestEigenpairGaugeMG(lat, &seed); });
    std::printf("covMG-LOBPCG cold start:  %d outer its, %7.1f ms, lambda=%.6f\n", cold.iterations, msCold,
                cold.eigenvalue);
    std::printf("covMG-LOBPCG seed  start: %d outer its, %7.1f ms, lambda=%.6f\n", warm.iterations, msWarm,
                warm.eigenvalue);
  }

  // (3) Experiment C: smooth the subdivision section toward the lowest mode.
  // alpha = 1/lambda_max makes x <- x - alpha*E*x damp the high (rough) modes.
  const double lamMax = bochner::largestEigenvalue(E).value;
  const double alpha = 1.0 / lamMax;
  std::printf("\nExperiment C: Richardson smoothing (alpha=1/lambda_max=%.3g), from subdivision:\n",
              alpha);

  std::vector<double> x = psi0;
  // L2-normalise the start.
  {
    const double nrm = std::sqrt(dot(x, x));
    for (double& v : x) v /= nrm;
  }
  const int checkpoints[] = {0, 1, 2, 5, 10, 20, 50, 100, 200, 400};
  int done = 0;
  double cumMs = msSub;  // smoothing time is on top of the subdivision build
  for (int cp : checkpoints) {
    cumMs += timeMs([&] {
      for (; done < cp; ++done) {
        const std::vector<double> y = matvec(Ec, x);  // E x
        for (std::size_t i = 0; i < x.size(); ++i) x[i] -= alpha * y[i];
        const double nrm = std::sqrt(dot(x, x));
        for (double& v : x) v /= nrm;
      }
    });
    const std::vector<double> Ex = matvec(Ec, x);
    const double rq = dot(x, Ex);  // Rayleigh quotient (x is unit-norm)
    char label[32];
    std::snprintf(label, sizeof(label), "  + %d smooths", cp);
    report(label, cumMs, analyze(g, x), rq);
  }
  std::printf("\n(truth lambda_min=%.5f; quality bar: max|z| < %.3f, radDev < %.3f)\n", lamMin,
              2.0 * g.spacing(), 1.5 * g.spacing());

  SlepcFinalize();
  return 0;
}
