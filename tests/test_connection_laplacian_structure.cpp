/// \file
/// Red-green TDD: the *assembled* U(1) connection Laplacian's structural
/// invariants -- the research-core operator (step 4) checked WITHOUT a linear
/// solve, so it runs in the default (PETSc-OFF) build. The eigenvalue physics of
/// frustration (which needs SLEPc) lives in the PETSc-gated
/// test_mac_connection_laplacian.
///
/// Anchors: (1) a trivial connection reduces to two decoupled copies of the
/// step-2 homogeneous-Neumann grid Laplacian (w = 1/h^2 ties the two operators);
/// (2) the real-block matrix is symmetric (the Hermitian embedding); (3)
/// connectionAngles scales velocity by h/hbar.
#include <doctest.h>

#include <cmath>
#include <vector>

#include "grid/GridOperators.h"
#include "extraction/MacConnectionLaplacian.h"
#include "grid/MacGrid.h"
#include "fluid/MacProjection.h"

using bochner::CooMatrix;
using bochner::FaceField;
using bochner::MacGrid;
namespace ops = bochner::ops;

namespace {

using Dense = std::vector<std::vector<double>>;

Dense toDense(const CooMatrix& A) {
  Dense d(A.rows(), std::vector<double>(A.cols(), 0.0));
  for (const auto& e : A.compressed()) d[e.row][e.col] = e.value;
  return d;
}

double wobble(int n) { return std::sin(0.8 * n + 0.3) + 0.4 * std::cos(2.3 * n); }

}  // namespace

TEST_CASE("trivial connection reduces to the Neumann grid Laplacian (x) I2") {
  MacGrid g(4, 3, 4, 0.5);
  const int n = g.numCells();
  Dense E = toDense(bochner::connectionLaplacian(g, ops::zeroFaceField(g)));
  Dense P = toDense(bochner::pressureLaplacian(g, {}));  // unpinned Neumann Laplacian

  double diag = 0.0, cross = 0.0;
  for (int j = 0; j < n; ++j)
    for (int k = 0; k < n; ++k) {
      double l = P[j][k];
      diag = std::max(diag, std::abs(E[2 * j][2 * k] - l));
      diag = std::max(diag, std::abs(E[2 * j + 1][2 * k + 1] - l));
      cross = std::max(cross, std::abs(E[2 * j][2 * k + 1]));
      cross = std::max(cross, std::abs(E[2 * j + 1][2 * k]));
    }
  CHECK(diag < 1e-12);
  CHECK(cross < 1e-12);
}

TEST_CASE("the real-block connection Laplacian is symmetric") {
  MacGrid g(4, 4, 3, 0.4);
  FaceField th = ops::zeroFaceField(g);
  for (size_t f = 0; f < th.x.size(); ++f) th.x[f] = 0.5 * wobble((int)f + 1);
  for (size_t f = 0; f < th.y.size(); ++f) th.y[f] = 0.5 * wobble((int)f + 7);
  for (size_t f = 0; f < th.z.size(); ++f) th.z[f] = 0.5 * wobble((int)f + 13);

  Dense E = toDense(bochner::connectionLaplacian(g, th));
  double asym = 0.0;
  for (size_t i = 0; i < E.size(); ++i)
    for (size_t j = 0; j < E.size(); ++j) asym = std::max(asym, std::abs(E[i][j] - E[j][i]));
  CHECK(asym < 1e-12);
}

TEST_CASE("connectionAngles scales velocity by h/hbar") {
  MacGrid g(3, 3, 3, 0.5);
  FaceField u = ops::zeroFaceField(g);
  u.x[g.faceXIndex(1, 1, 1)] = 2.0;
  const double hbar = 0.25;
  FaceField th = bochner::connectionAngles(g, u, hbar);
  CHECK(th.x[g.faceXIndex(1, 1, 1)] == doctest::Approx(2.0 * 0.5 / 0.25));
}
