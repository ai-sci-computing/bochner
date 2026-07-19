/// \file
/// The multigrid-preconditioned CG (MGPCG) Poisson solver: it must converge to a
/// small residual in FEW iterations, and that count must stay ~flat as the grid
/// refines (mesh independence) -- the whole point over Jacobi-CG, whose count
/// grows with the grid dimension.
#include <doctest.h>

#include <cmath>
#include <random>
#include <vector>

#include "fluid/PoissonMGPCG.h"
#include "grid/CooMatrix.h"

using namespace bochner;

// Cell-centered 7-point Poisson with a Robin (Dirichlet-ghost) term on every
// domain-boundary face: SPD, no null space, all cells active -- the clean model
// of an all-open MAC pressure Poisson.
static SpMat robinPoisson(int nx, int ny, int nz, double h) {
  const double w = 1.0 / (h * h);
  auto idx = [&](int i, int j, int k) { return (i * ny + j) * nz + k; };
  CooMatrix A(nx * ny * nz, nx * ny * nz);
  auto couple = [&](int a, int b) {
    A.add(a, a, w);
    A.add(b, b, w);
    A.add(a, b, -w);
    A.add(b, a, -w);
  };
  for (int i = 1; i < nx; ++i)
    for (int j = 0; j < ny; ++j)
      for (int k = 0; k < nz; ++k) couple(idx(i - 1, j, k), idx(i, j, k));
  for (int i = 0; i < nx; ++i)
    for (int j = 1; j < ny; ++j)
      for (int k = 0; k < nz; ++k) couple(idx(i, j - 1, k), idx(i, j, k));
  for (int i = 0; i < nx; ++i)
    for (int j = 0; j < ny; ++j)
      for (int k = 1; k < nz; ++k) couple(idx(i, j, k - 1), idx(i, j, k));
  // Robin ghost on the six domain-boundary faces.
  for (int j = 0; j < ny; ++j)
    for (int k = 0; k < nz; ++k) { A.add(idx(0, j, k), idx(0, j, k), w); A.add(idx(nx - 1, j, k), idx(nx - 1, j, k), w); }
  for (int i = 0; i < nx; ++i)
    for (int k = 0; k < nz; ++k) { A.add(idx(i, 0, k), idx(i, 0, k), w); A.add(idx(i, ny - 1, k), idx(i, ny - 1, k), w); }
  for (int i = 0; i < nx; ++i)
    for (int j = 0; j < ny; ++j) { A.add(idx(i, j, 0), idx(i, j, 0), w); A.add(idx(i, j, nz - 1), idx(i, j, nz - 1), w); }
  return toSpMat(A);
}

static void apply(const SpMat& A, const std::vector<double>& x, std::vector<double>& y) {
  for (int r = 0; r < A.n(); ++r) {
    double s = 0.0;
    for (int p = A.rowStart[r]; p < A.rowStart[r + 1]; ++p) s += A.val[p] * x[A.col[p]];
    y[r] = s;
  }
}

TEST_CASE("MGPCG converges in few, ~mesh-independent iterations") {
  auto solveN = [](int n) -> int {
    const double h = 1.0 / n;
    const SpMat A = robinPoisson(n, n, n, h);
    const std::vector<char> active(n * n * n, 1);
    const MgHierarchy H = buildPoissonMgHierarchy(n, n, n, A, active);

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> d(-1.0, 1.0);
    std::vector<double> xstar(n * n * n);
    for (double& v : xstar) v = d(rng);
    std::vector<double> b(n * n * n);
    apply(A, xstar, b);  // b = A x*

    std::vector<double> x(n * n * n, 0.0);
    double rel = 1.0;
    const int its = mgpcgSolve(H, b, x, /*tol=*/1e-6, /*maxit=*/300, &rel);

    CHECK(rel < 1e-6);   // reached the requested residual
    CHECK(its < 300);    // did not hit the cap
    // Recovered the true solution (SPD -> unique), to a residual-consistent error.
    double num = 0.0, den = 0.0;
    for (int i = 0; i < A.n(); ++i) { num += (x[i] - xstar[i]) * (x[i] - xstar[i]); den += xstar[i] * xstar[i]; }
    CHECK(std::sqrt(num / den) < 1e-2);
    return its;
  };

  const int it16 = solveN(16), it32 = solveN(32);
  MESSAGE("MGPCG iterations: n=16 -> " << it16 << ", n=32 -> " << it32);
  CHECK(it16 < 40);
  CHECK(it32 < 40);            // stays small as n doubles (Jacobi-CG would ~double)
  CHECK(it32 <= it16 + 12);   // near mesh-independent
}
