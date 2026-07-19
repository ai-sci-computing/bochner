/// \file
/// Red-green TDD for the real-scalar geometric multigrid pressure solver
/// (PoissonMultigrid). Anchors: (1) the matrix-free Neumann Laplacian
/// annihilates constants and is symmetric; (2) the V-cycle solves L phi = rhs
/// (manufactured, mean-zero) to tolerance, recovering phi up to a constant;
/// (3) the cycle count is ~mesh-independent (the multigrid payoff).
#include <doctest.h>

#include <cmath>
#include <vector>

#include "solvers/PoissonMultigrid.h"

using bochner::applyPoisson;
using bochner::poissonVcycleSolve;

namespace {

// A deterministic mean-zero field on an nx*ny*nz grid.
std::vector<double> wobbleField(int nx, int ny, int nz) {
  std::vector<double> v(static_cast<std::size_t>(nx) * ny * nz);
  double mean = 0.0;
  for (int i = 0; i < nx; ++i)
    for (int j = 0; j < ny; ++j)
      for (int k = 0; k < nz; ++k) {
        const double x = std::sin(0.3 * i + 0.7) * std::cos(0.4 * j) + std::sin(0.2 * k + 1.0);
        v[(i * ny + j) * nz + k] = x;
        mean += x;
      }
  mean /= v.size();
  for (double& x : v) x -= mean;
  return v;
}

double l2(const std::vector<double>& v) {
  double s = 0.0;
  for (double x : v) s += x * x;
  return std::sqrt(s);
}

}  // namespace

TEST_CASE("Neumann Laplacian annihilates constants and is symmetric") {
  const int n = 5;
  std::vector<double> ones(static_cast<std::size_t>(n) * n * n, 1.0);
  const auto y = applyPoisson(n, n, n, 0.2, ones);
  double maxAbs = 0.0;
  for (double v : y) maxAbs = std::max(maxAbs, std::abs(v));
  CHECK(maxAbs < 1e-12);  // L * const = 0

  // Symmetry: <a, L b> == <L a, b> for two arbitrary fields.
  const auto a = wobbleField(n, n, n);
  std::vector<double> b = wobbleField(n, n, n);
  for (double& x : b) x = x * x - 0.5;  // perturb so a != b
  const auto La = applyPoisson(n, n, n, 0.2, a);
  const auto Lb = applyPoisson(n, n, n, 0.2, b);
  double aLb = 0.0, Lab = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    aLb += a[i] * Lb[i];
    Lab += La[i] * b[i];
  }
  CHECK(aLb == doctest::Approx(Lab).epsilon(1e-10));
}

TEST_CASE("V-cycle solves L phi = rhs to tolerance (phi up to a constant)") {
  const int n = 16;
  const double h = 0.1;
  const auto phiStar = wobbleField(n, n, n);
  const auto rhs = applyPoisson(n, n, n, h, phiStar);  // mean-zero by construction

  std::vector<double> phi(static_cast<std::size_t>(n) * n * n, 0.0);
  bochner::PoissonMgOptions opts;
  opts.tol = 1e-8;
  const auto res = poissonVcycleSolve(n, n, n, h, rhs, phi, opts);

  INFO("cycles=" << res.cycles << " relResidual=" << res.relResidual);
  CHECK(res.relResidual < 1e-8);
  CHECK(res.cycles < opts.maxCycles);

  // phi matches phiStar up to an additive constant (the Neumann null space).
  double meanDiff = 0.0;
  for (std::size_t i = 0; i < phi.size(); ++i) meanDiff += phi[i] - phiStar[i];
  meanDiff /= phi.size();
  std::vector<double> d(phi.size());
  for (std::size_t i = 0; i < phi.size(); ++i) d[i] = phi[i] - phiStar[i] - meanDiff;
  CHECK(l2(d) / l2(phiStar) < 1e-5);
}

TEST_CASE("V-cycle count is ~mesh-independent") {
  bochner::PoissonMgOptions opts;
  opts.tol = 1e-8;
  int c16 = 0, c32 = 0;
  for (int n : {16, 32}) {
    const double h = 1.6 / n;
    const auto phiStar = wobbleField(n, n, n);
    const auto rhs = applyPoisson(n, n, n, h, phiStar);
    std::vector<double> phi(static_cast<std::size_t>(n) * n * n, 0.0);
    const int c = poissonVcycleSolve(n, n, n, h, rhs, phi, opts).cycles;
    (n == 16 ? c16 : c32) = c;
  }
  MESSAGE("Poisson V-cycles to 1e-8: n=16 -> " << c16 << ", n=32 -> " << c32);
  // Measured 16 -> 17. The old 2x bound is the non-mesh-independent signature
  // itself; +2 is the honest band around the measured behaviour.
  CHECK(c32 <= c16 + 2);
}

// The hierarchy depth is part of the contract, because coarsening halts at the
// first odd (or <4) extent: a grid the user picks can silently collapse the
// V-cycle to plain smoothing. Callers that expose the grid size (the viewers)
// must be able to see this, so it is reported rather than inferred.
//
// Measured degradation (spike RHS, tol 1e-8, cap 100 cycles):
//   n=32 -> 5 levels, 12 cycles      n=33 -> 1 level,  39 cycles
//   n=48 -> 5 levels, 13 cycles      n=49 -> 1 level,  82 cycles
//   n=64 -> 6 levels, 13 cycles      n=65 -> 1 level,  FAILS at 4.9e-7
//   n=60 -> 3 levels, 16 cycles (even-but-not-2-adic is merely shallow, not fatal)
// So it is not a cliff at every odd n: the cost grows as the single-level
// smoother loses ground, and only becomes an outright failure once the grid is
// large enough. n=65 is the case that actually breaks.
TEST_CASE("poissonVcycleSolve reports hierarchy depth; a large odd extent breaks it") {
  auto solveOn = [](int n) {
    const double h = 1.0 / n;
    std::vector<double> rhs(static_cast<std::size_t>(n) * n * n, 0.0), phi(rhs.size(), 0.0);
    // Single interior spike, mean removed: nonzero and compatible with the
    // homogeneous-Neumann null space, so the solve is well posed.
    rhs[static_cast<std::size_t>((n / 2 * n) + n / 2) * n + n / 2] = 1.0;
    const double mean = 1.0 / static_cast<double>(rhs.size());
    for (double& v : rhs) v -= mean;
    bochner::PoissonMgOptions opts;
    opts.tol = 1e-8;
    opts.maxCycles = 100;
    return bochner::poissonVcycleSolve(n, n, n, h, rhs, phi, opts);
  };

  const auto deep = solveOn(64);
  INFO("n=64 levels=" << deep.levels << " cycles=" << deep.cycles
                      << " relRes=" << deep.relResidual);
  CHECK(deep.levels >= 5);
  CHECK(deep.relResidual < 1e-8);
  CHECK(deep.cycles < 25);

  // n=65 is odd, so the coarsening loop never runs: one level, and each "cycle"
  // is just opts.coarseSweeps Gauss-Seidel sweeps. It exhausts the cycle cap
  // without reaching tolerance. This asserts the known limitation so that a
  // future semi-coarsening fix trips this test rather than passing silently.
  const auto flat = solveOn(65);
  INFO("n=65 levels=" << flat.levels << " cycles=" << flat.cycles
                      << " relRes=" << flat.relResidual);
  CHECK(flat.levels == 1);
  CHECK(flat.relResidual > 1e-8);        // did NOT reach tolerance
  CHECK(flat.cycles >= 100);             // burned the whole cap
  CHECK(flat.cycles > 5 * deep.cycles);  // and the gap is not marginal
}
