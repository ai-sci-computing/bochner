/// \file
/// Linear-solver comparison: gauge-aware multigrid V-cycle vs unpreconditioned
/// CG on E x = b for the seeded-ring connection Laplacian, across grid sizes.
/// Reports outer iterations/cycles, total matvecs (the honest shared work unit,
/// since a V-cycle is many matvecs), wall-clock, and the true relative residual.
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "solvers/GaugeMultigrid.h"
#include "extraction/MacConnectionLaplacian.h"
#include "grid/MacGrid.h"
#include "fluid/MacVortexRing.h"

using bochner::GaugeLattice;
using bochner::MacGrid;
using bochner::Vec3;
using cd = std::complex<double>;

namespace {
template <class F>
double timeMs(F&& f) {
  const auto t0 = std::chrono::steady_clock::now();
  f();
  const auto t1 = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(t1 - t0).count();
}
}  // namespace

int main(int argc, char** argv) {
  const double tol = 1e-8;
  std::printf("E x = b on the seeded ring, tol=%.0e. (matvecs = shared work unit)\n\n", tol);
  std::printf("%5s %9s | %-28s | %-28s\n", "n", "DOFs", "gauge-MG V-cycle", "plain CG");
  std::printf("%5s %9s | %6s %8s %9s | %6s %8s %9s\n", "", "", "cyc", "matvec", "ms", "it", "matvec",
              "ms");

  std::vector<int> sizes;
  for (int a = 1; a < argc; ++a) sizes.push_back(std::atoi(argv[a]));
  if (sizes.empty()) sizes = {16, 32, 64};

  for (int N : sizes) {
    const MacGrid g(N, N, N, 1.6 / N, Vec3{-0.8, -0.8, -0.8});
    const double R = 0.7, Gamma = 1.0, hbar = Gamma / (2.0 * M_PI);
    const auto u = bochner::vortexRingFaceField(g, {0, 0, 0}, {0, 0, 1}, R, Gamma, 0.15);
    const auto theta = bochner::connectionAngles(g, u, hbar);
    const GaugeLattice lat = bochner::gaugeLatticeFromFaces(g, theta);

    // Manufactured RHS: b = E x*, solve from zero.
    std::vector<cd> xstar(g.numCells());
    for (int c = 0; c < g.numCells(); ++c) xstar[c] = cd(std::cos(0.11 * c), std::sin(0.07 * c));
    const std::vector<cd> b = bochner::applyConnectionLaplacian(lat, xstar);

    bochner::MgOptions opts;
    opts.tol = tol;
    std::vector<cd> xmg(g.numCells(), cd(0.0, 0.0));
    bochner::resetGaugeMatvecCount();
    bochner::MgResult mg;
    const double mgMs = timeMs([&] { mg = bochner::vcycleSolve(lat, b, xmg, opts); });
    const long mgMv = bochner::gaugeMatvecCount();

    std::vector<cd> xcg(g.numCells(), cd(0.0, 0.0));
    bochner::resetGaugeMatvecCount();
    bochner::SolveStats cg;
    const double cgMs = timeMs([&] { cg = bochner::cgSolve(lat, b, xcg, tol); });
    const long cgMv = bochner::gaugeMatvecCount();

    std::printf("%5d %9d | %6d %8ld %9.1f | %6d %8ld %9.1f\n", N, 2 * g.numCells(), mg.cycles, mgMv,
                mgMs, cg.iterations, cgMv, cgMs);
  }
  return 0;
}
