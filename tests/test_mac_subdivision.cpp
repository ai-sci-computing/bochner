/// \file
/// Red-green TDD for the gauge-aware subdivision section (MacSubdivision):
/// the covariant-subdivision prolongation that builds a section without
/// solving a linear system. Anchors: (1) a trivial connection prolongs the
/// constant section exactly (no zeros); (2) parallel transport is actually
/// applied (a non-trivial connection rotates the phase); (3) the output has the
/// interleaved layout traceZeroSet expects; (4) input validation.
#include <doctest.h>

#include <cmath>
#include <vector>

#include "grid/GridOperators.h"
#include "extraction/MacFilaments.h"
#include "grid/MacGrid.h"
#include "solvers/MacSubdivision.h"

using bochner::FaceField;
using bochner::MacGrid;

namespace {

// |psi| at cell c from the interleaved [Re,Im,...] section.
double mag(const std::vector<double>& psi, int c) {
  return std::hypot(psi[2 * c], psi[2 * c + 1]);
}

}  // namespace

TEST_CASE("trivial connection: prolongs the constant section exactly") {
  const MacGrid g(8, 8, 4, 1.0);
  FaceField theta = bochner::ops::zeroFaceField(g);  // theta == 0 everywhere

  const std::vector<double> psi = bochner::subdivisionSection(g, theta, 2);

  REQUIRE(psi.size() == static_cast<std::size_t>(2 * g.numCells()));
  for (int c = 0; c < g.numCells(); ++c) {
    CHECK(psi[2 * c] == doctest::Approx(1.0).epsilon(1e-12));      // Re == 1
    CHECK(psi[2 * c + 1] == doctest::Approx(0.0).epsilon(1e-12));  // Im == 0
  }
  // A constant section has no zero set.
  CHECK(bochner::traceZeroSet(g, psi).empty());
}

TEST_CASE("numLevels = 0 returns the constant seed on the full grid") {
  const MacGrid g(5, 3, 2, 1.0);
  FaceField theta = bochner::ops::zeroFaceField(g);

  const std::vector<double> psi = bochner::subdivisionSection(g, theta, 0);
  for (int c = 0; c < g.numCells(); ++c) CHECK(mag(psi, c) == doctest::Approx(1.0));
}

TEST_CASE("parallel transport is applied (non-trivial connection rotates phase)") {
  // A gentle linear gauge potential alpha(i,j,k) = slope * i gives pure-gauge
  // x-links theta = alpha(i+1) - alpha(i) = slope. Curvature is zero, so |psi|
  // stays positive (no spurious zeros), but the phase must wind: starting from
  // psi = 1, transported averaging makes the imaginary part nonzero somewhere.
  const MacGrid g(8, 8, 4, 1.0);
  FaceField theta = bochner::ops::zeroFaceField(g);
  const double slope = 0.3;
  for (double& v : theta.x) v = slope;  // uniform x-links == d(slope*i)

  const std::vector<double> psi = bochner::subdivisionSection(g, theta, 2);

  double maxImag = 0.0, minMag = 1e9;
  for (int c = 0; c < g.numCells(); ++c) {
    maxImag = std::max(maxImag, std::abs(psi[2 * c + 1]));
    minMag = std::min(minMag, mag(psi, c));
  }
  CHECK(maxImag > 1e-3);  // transport rotated the phase
  CHECK(minMag > 0.1);    // zero curvature => no destructive cancellation
}

TEST_CASE("rejects dimensions not divisible by 2^numLevels") {
  const MacGrid g(6, 8, 4, 1.0);  // 6 is not divisible by 4
  FaceField theta = bochner::ops::zeroFaceField(g);
  CHECK_THROWS_AS(bochner::subdivisionSection(g, theta, 2), std::invalid_argument);
  CHECK_THROWS_AS(bochner::subdivisionSection(g, theta, -1), std::invalid_argument);
  CHECK_NOTHROW(bochner::subdivisionSection(g, theta, 1));  // all dims divisible by 2
}
