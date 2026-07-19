/// \file
/// Red-green TDD: the PETSc backend bridge solves an SPD system.
/// This is the integration smoke test for PETSc:
/// a 1D Laplacian is the discrete scalar Poisson operator, the same shape as
/// the Phase-1 pressure projection.
#include <doctest.h>

#include <cmath>
#include <vector>

#include "grid/CooMatrix.h"
#include "solvers/PetscSolver.h"

using bochner::CooMatrix;

namespace {

/// Dense-ish matvec from compressed triplets, for residual checking in tests.
std::vector<double> matvec(const CooMatrix& A, const std::vector<double>& x) {
  std::vector<double> y(static_cast<std::size_t>(A.rows()), 0.0);
  for (const auto& e : A.compressed()) {
    y[static_cast<std::size_t>(e.row)] +=
        e.value * x[static_cast<std::size_t>(e.col)];
  }
  return y;
}

/// 1D Laplacian with Dirichlet ends: tridiagonal [-1, 2, -1]. SPD.
CooMatrix laplacian1d(int n) {
  CooMatrix A(n, n);
  for (int i = 0; i < n; ++i) {
    A.add(i, i, 2.0);
    if (i > 0) A.add(i, i - 1, -1.0);
    if (i + 1 < n) A.add(i, i + 1, -1.0);
  }
  return A;
}

}  // namespace

TEST_CASE("solveSpdCG solves a 1D Laplacian to the requested tolerance") {
  const int n = 50;
  const CooMatrix A = laplacian1d(n);

  std::vector<double> b(n, 1.0);
  const std::vector<double> x = bochner::solveSpdCG(A, b, 1e-12);

  REQUIRE(x.size() == static_cast<std::size_t>(n));

  // Residual r = b - A x should be tiny.
  const std::vector<double> Ax = matvec(A, x);
  double rmax = 0.0;
  for (int i = 0; i < n; ++i) rmax = std::max(rmax, std::abs(b[i] - Ax[i]));
  CHECK(rmax < 1e-8);
}

TEST_CASE("solveSpdCG recovers a prescribed solution") {
  const int n = 30;
  const CooMatrix A = laplacian1d(n);

  std::vector<double> xexact(n);
  for (int i = 0; i < n; ++i) xexact[i] = std::sin(0.3 * i) + 0.5 * i;
  const std::vector<double> b = matvec(A, xexact);

  const std::vector<double> x = bochner::solveSpdCG(A, b, 1e-12);
  REQUIRE(x.size() == static_cast<std::size_t>(n));

  double emax = 0.0;
  for (int i = 0; i < n; ++i) emax = std::max(emax, std::abs(x[i] - xexact[i]));
  CHECK(emax < 1e-6);
}

TEST_CASE("solveSpdCG rejects non-square or mismatched input") {
  CooMatrix rect(2, 3);
  CHECK_THROWS(bochner::solveSpdCG(rect, std::vector<double>(3, 1.0)));

  const CooMatrix A = laplacian1d(4);
  CHECK_THROWS(bochner::solveSpdCG(A, std::vector<double>(3, 1.0)));
}
