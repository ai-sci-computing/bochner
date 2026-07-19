/// \file
/// Red-green TDD: the SLEPc smallest-eigenpair bridge. Anchors: (1) it recovers
/// the known smallest eigenpair of a diagonal matrix; (2) the returned pair is a
/// genuine eigenvector (small residual ‖A v − λ v‖) of a small non-diagonal SPD
/// matrix. The connection-Laplacian physics (trivial kernel vs frustration mass
/// gap) is covered on the grid in test_mac_connection_laplacian.cpp.
#include <doctest.h>

#include <cmath>
#include <vector>

#include "grid/CooMatrix.h"
#include "solvers/EigenSolver.h"

using bochner::CooMatrix;

namespace {

// ‖A v − λ v‖_inf for a COO matrix.
double eigenResidual(const CooMatrix& A, const bochner::EigenPair& e) {
  std::vector<double> Av(static_cast<std::size_t>(A.rows()), 0.0);
  for (const auto& t : A.compressed())
    Av[static_cast<std::size_t>(t.row)] += t.value * e.vector[static_cast<std::size_t>(t.col)];
  double m = 0.0;
  for (std::size_t i = 0; i < Av.size(); ++i)
    m = std::max(m, std::abs(Av[i] - e.value * e.vector[i]));
  return m;
}

}  // namespace

TEST_CASE("recovers the smallest eigenpair of a diagonal matrix") {
  CooMatrix A(3, 3);
  A.add(0, 0, 3.0);
  A.add(1, 1, 1.0);
  A.add(2, 2, 2.0);

  const bochner::EigenPair e = bochner::smallestEigenpair(A);
  CHECK(e.value == doctest::Approx(1.0).epsilon(1e-8));
  // Eigenvector is ±e_1 (defined up to sign).
  CHECK(std::abs(e.vector[1]) == doctest::Approx(1.0).epsilon(1e-6));
  CHECK(std::abs(e.vector[0]) < 1e-6);
  CHECK(std::abs(e.vector[2]) < 1e-6);
}

TEST_CASE("returns a genuine eigenvector of an SPD Laplacian") {
  // 4x4 1D Dirichlet Laplacian, tridiag(-1, 2, -1). Eigenvalues are
  // lambda_k = 2 - 2 cos(k*pi/5); the smallest is lambda_1 = 2 - 2 cos(36 deg)
  // = 0.381966..., well separated from the rest (a robust, non-degenerate case).
  const int n = 4;
  CooMatrix A(n, n);
  for (int i = 0; i < n; ++i) {
    A.add(i, i, 2.0);
    if (i > 0) A.add(i, i - 1, -1.0);
    if (i + 1 < n) A.add(i, i + 1, -1.0);
  }

  const bochner::EigenPair e = bochner::smallestEigenpair(A);
  CHECK(e.value == doctest::Approx(2.0 - 2.0 * std::cos(M_PI / 5.0)).epsilon(1e-6));
  CHECK(eigenResidual(A, e) < 1e-6);  // genuine eigenvector
}
