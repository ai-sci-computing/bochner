/// \file
/// Red-green TDD: the eigenvalue physics of the U(1) connection Laplacian (step
/// 4) -- the part that needs a SLEPc solve, so this suite is PETSc-gated. The
/// solve-free structural invariants (trivial reduction, symmetry, angle scaling)
/// live in the ungated test_connection_laplacian_structure so the default build
/// covers the assembled operator.
///
/// Anchor: the physics of frustration -- a pure-gauge connection keeps
/// lambda_min at zero (gauge-equivalent to trivial), while a connection with
/// real holonomy lifts lambda_min off zero (the mass gap the QCD/AMG methods
/// target).
#include <doctest.h>

#include <cmath>
#include <vector>

#include "solvers/EigenSolver.h"
#include "grid/GridOperators.h"
#include "extraction/MacConnectionLaplacian.h"
#include "grid/MacGrid.h"

using bochner::FaceField;
using bochner::MacGrid;
namespace ops = bochner::ops;

namespace {

double wobble(int n) { return std::sin(0.8 * n + 0.3) + 0.4 * std::cos(2.3 * n); }

// A pure-gauge connection from a per-cell potential phi: theta_f = phi_b - phi_a.
FaceField pureGaugeAngles(const MacGrid& g, const std::vector<double>& phi) {
  FaceField th = ops::zeroFaceField(g);
  for (int i = 1; i < g.nx(); ++i)
    for (int j = 0; j < g.ny(); ++j)
      for (int k = 0; k < g.nz(); ++k)
        th.x[g.faceXIndex(i, j, k)] = phi[g.cellIndex(i, j, k)] - phi[g.cellIndex(i - 1, j, k)];
  for (int i = 0; i < g.nx(); ++i)
    for (int j = 1; j < g.ny(); ++j)
      for (int k = 0; k < g.nz(); ++k)
        th.y[g.faceYIndex(i, j, k)] = phi[g.cellIndex(i, j, k)] - phi[g.cellIndex(i, j - 1, k)];
  for (int i = 0; i < g.nx(); ++i)
    for (int j = 0; j < g.ny(); ++j)
      for (int k = 1; k < g.nz(); ++k)
        th.z[g.faceZIndex(i, j, k)] = phi[g.cellIndex(i, j, k)] - phi[g.cellIndex(i, j, k - 1)];
  return th;
}

}  // namespace

TEST_CASE("frustration: pure gauge keeps lambda_min at zero, holonomy lifts it") {
  MacGrid g(6, 6, 6, 1.0);

  double lamTrivial = bochner::smallestEigenpair(
                          bochner::connectionLaplacian(g, ops::zeroFaceField(g)))
                          .value;
  CHECK(std::abs(lamTrivial) < 1e-6);  // constant section is a zero mode

  std::vector<double> phi(g.numCells());
  for (int c = 0; c < g.numCells(); ++c) phi[c] = 0.7 * wobble(c);
  double lamGauge =
      bochner::smallestEigenpair(bochner::connectionLaplacian(g, pureGaugeAngles(g, phi))).value;
  CHECK(std::abs(lamGauge) < 1e-6);  // pure gauge is gauge-equivalent to trivial

  // A generic O(1) connection carries real holonomy -> mass gap (lambda_min > 0).
  FaceField th = ops::zeroFaceField(g);
  for (size_t f = 0; f < th.x.size(); ++f) th.x[f] = wobble((int)f + 2);
  for (size_t f = 0; f < th.y.size(); ++f) th.y[f] = wobble((int)f + 5);
  for (size_t f = 0; f < th.z.size(); ++f) th.z[f] = wobble((int)f + 11);
  double lamFrustrated = bochner::smallestEigenpair(bochner::connectionLaplacian(g, th)).value;
  CHECK(lamFrustrated > 1e-2);
  CHECK(lamFrustrated > lamGauge + 1e-3);
}
