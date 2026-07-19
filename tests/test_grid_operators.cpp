/// \file
/// Red-green TDD: behaviour spec for the structured MAC div/grad operators.
///
/// These are the discrete `d` (divergence, faces->cells) and its negative
/// adjoint gradient (cells->faces), realised as structured stencils on the
/// uniform grid (structured, matrix-free-friendly, not a
/// general cell complex). The pair must satisfy the MAC projection identities:
///   - constant velocity is divergence-free; a linear ramp has unit divergence;
///   - the gradient of a constant vanishes (and boundary faces carry no
///     gradient -- they are BC-prescribed);
///   - grad = -div^T on the interior (the discrete adjoint);
///   - div(grad(.)) is the homogeneous-Neumann 7-point Laplacian.
#include <doctest.h>

#include <algorithm>
#include <numeric>
#include <vector>

#include "grid/GridOperators.h"
#include "grid/MacGrid.h"

using bochner::FaceField;
using bochner::MacGrid;
namespace ops = bochner::ops;

namespace {

// A deterministic pseudo-random fill so the adjoint test exercises a generic
// field without depending on <random> seeding details.
double wobble(int n) { return std::sin(0.7 * n + 1.0) + 0.3 * std::cos(2.1 * n); }

// Reference: homogeneous-Neumann 7-point Laplacian of a cell field. A missing
// neighbour (domain boundary) contributes nothing -- zero normal gradient.
std::vector<double> neumannLaplacian(const MacGrid& g, const std::vector<double>& p) {
  const double inv_h2 = 1.0 / (g.spacing() * g.spacing());
  std::vector<double> lap(g.numCells(), 0.0);
  auto add = [&](int i, int j, int k, int di, int dj, int dk, double pc) {
    int ni = i + di, nj = j + dj, nk = k + dk;
    if (ni < 0 || ni >= g.nx() || nj < 0 || nj >= g.ny() || nk < 0 || nk >= g.nz()) return;
    lap[g.cellIndex(i, j, k)] += (p[g.cellIndex(ni, nj, nk)] - pc) * inv_h2;
  };
  for (int i = 0; i < g.nx(); ++i)
    for (int j = 0; j < g.ny(); ++j)
      for (int k = 0; k < g.nz(); ++k) {
        double pc = p[g.cellIndex(i, j, k)];
        add(i, j, k, +1, 0, 0, pc);
        add(i, j, k, -1, 0, 0, pc);
        add(i, j, k, 0, +1, 0, pc);
        add(i, j, k, 0, -1, 0, pc);
        add(i, j, k, 0, 0, +1, pc);
        add(i, j, k, 0, 0, -1, pc);
      }
  return lap;
}

}  // namespace

TEST_CASE("divergence: constant velocity is divergence-free") {
  MacGrid g(4, 5, 3, 0.5);
  FaceField u = ops::zeroFaceField(g);
  std::fill(u.x.begin(), u.x.end(), 2.0);  // uniform flow in +x
  auto div = ops::divergence(g, u);
  REQUIRE(div.size() == static_cast<size_t>(g.numCells()));
  for (double d : div) CHECK(d == doctest::Approx(0.0));
}

TEST_CASE("divergence: a unit x-ramp has unit divergence everywhere") {
  MacGrid g(4, 5, 3, 0.5, bochner::Vec3{-1.0, 0.5, 2.0});
  FaceField u = ops::zeroFaceField(g);
  for (int i = 0; i <= g.nx(); ++i)
    for (int j = 0; j < g.ny(); ++j)
      for (int k = 0; k < g.nz(); ++k)
        u.x[g.faceXIndex(i, j, k)] = g.faceXCenter(i, j, k)[0];  // u_x = x
  auto div = ops::divergence(g, u);
  for (double d : div) CHECK(d == doctest::Approx(1.0));
}

TEST_CASE("gradient: a constant has zero gradient, including boundary faces") {
  MacGrid g(3, 4, 5, 0.7);
  std::vector<double> p(g.numCells(), 4.2);
  FaceField gp = ops::gradient(g, p);
  for (double v : gp.x) CHECK(v == doctest::Approx(0.0));
  for (double v : gp.y) CHECK(v == doctest::Approx(0.0));
  for (double v : gp.z) CHECK(v == doctest::Approx(0.0));
}

TEST_CASE("gradient: a unit x-ramp gives unit gradient on interior x-faces only") {
  MacGrid g(4, 3, 3, 0.5);
  std::vector<double> p(g.numCells());
  for (int i = 0; i < g.nx(); ++i)
    for (int j = 0; j < g.ny(); ++j)
      for (int k = 0; k < g.nz(); ++k)
        p[g.cellIndex(i, j, k)] = g.cellCenter(i, j, k)[0];  // p = x
  FaceField gp = ops::gradient(g, p);

  for (int i = 0; i <= g.nx(); ++i)
    for (int j = 0; j < g.ny(); ++j)
      for (int k = 0; k < g.nz(); ++k) {
        double expected = (i == 0 || i == g.nx()) ? 0.0 : 1.0;  // boundary faces: no gradient
        CHECK(gp.x[g.faceXIndex(i, j, k)] == doctest::Approx(expected));
      }
  for (double v : gp.y) CHECK(v == doctest::Approx(0.0));
  for (double v : gp.z) CHECK(v == doctest::Approx(0.0));
}

TEST_CASE("grad = -div^T : the discrete adjoint holds for interior-supported flux") {
  MacGrid g(3, 4, 3, 0.6);

  std::vector<double> p(g.numCells());
  for (int c = 0; c < g.numCells(); ++c) p[c] = wobble(c);

  // A face field supported only on interior faces (boundary fluxes are
  // BC-prescribed, so the adjoint pairing is over interior faces).
  FaceField u = ops::zeroFaceField(g);
  for (int i = 1; i < g.nx(); ++i)
    for (int j = 0; j < g.ny(); ++j)
      for (int k = 0; k < g.nz(); ++k) u.x[g.faceXIndex(i, j, k)] = wobble(i * 31 + j * 7 + k);
  for (int i = 0; i < g.nx(); ++i)
    for (int j = 1; j < g.ny(); ++j)
      for (int k = 0; k < g.nz(); ++k) u.y[g.faceYIndex(i, j, k)] = wobble(i * 5 + j * 29 + k * 3);
  for (int i = 0; i < g.nx(); ++i)
    for (int j = 0; j < g.ny(); ++j)
      for (int k = 1; k < g.nz(); ++k) u.z[g.faceZIndex(i, j, k)] = wobble(i * 11 + j * 2 + k * 23);

  auto div = ops::divergence(g, u);
  FaceField gp = ops::gradient(g, p);

  double lhs = std::inner_product(div.begin(), div.end(), p.begin(), 0.0);
  double rhs = std::inner_product(u.x.begin(), u.x.end(), gp.x.begin(), 0.0) +
               std::inner_product(u.y.begin(), u.y.end(), gp.y.begin(), 0.0) +
               std::inner_product(u.z.begin(), u.z.end(), gp.z.begin(), 0.0);
  CHECK(lhs == doctest::Approx(-rhs));
}

TEST_CASE("div(grad(.)) is the homogeneous-Neumann 7-point Laplacian") {
  MacGrid g(4, 3, 5, 0.5);
  std::vector<double> p(g.numCells());
  for (int c = 0; c < g.numCells(); ++c) p[c] = wobble(c);

  auto lap = ops::divergence(g, ops::gradient(g, p));
  auto ref = neumannLaplacian(g, p);
  REQUIRE(lap.size() == ref.size());
  for (size_t c = 0; c < lap.size(); ++c) CHECK(lap[c] == doctest::Approx(ref[c]));

  // Neumann: the Laplacian annihilates constants.
  std::vector<double> ones(g.numCells(), 1.0);
  auto lap1 = ops::divergence(g, ops::gradient(g, ones));
  for (double v : lap1) CHECK(v == doctest::Approx(0.0));
}
