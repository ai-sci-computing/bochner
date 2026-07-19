/// \file
/// Red-green TDD: behaviour spec for the uniform 3D MAC grid geometry.
///
/// This is the substrate of the grid rebuild: a
/// cubical cell complex with dual/flux placement. Velocity lives as a normal
/// flux on faces; pressure is a scalar at cell centers. Here we pin down only
/// the geometry and indexing; operators (div/grad) are a separate spec.
#include <doctest.h>

#include "grid/MacGrid.h"

using bochner::MacGrid;
using bochner::Vec3;

TEST_CASE("MacGrid reports its dimensions and element counts") {
  // 2 x 3 x 4 cells, spacing 0.5, default origin at the axes.
  MacGrid g(2, 3, 4, 0.5);
  CHECK(g.nx() == 2);
  CHECK(g.ny() == 3);
  CHECK(g.nz() == 4);
  CHECK(g.spacing() == doctest::Approx(0.5));

  CHECK(g.numCells() == 2 * 3 * 4);
  // Faces normal to each axis gain one layer along that axis.
  CHECK(g.numFacesX() == 3 * 3 * 4);
  CHECK(g.numFacesY() == 2 * 4 * 4);
  CHECK(g.numFacesZ() == 2 * 3 * 5);
  CHECK(g.numFaces() == g.numFacesX() + g.numFacesY() + g.numFacesZ());
}

TEST_CASE("cell indexing is a bijection over the lattice (i slowest, k fastest)") {
  MacGrid g(2, 3, 4, 1.0);
  CHECK(g.cellIndex(0, 0, 0) == 0);
  CHECK(g.cellIndex(0, 0, 1) == 1);          // k fastest
  CHECK(g.cellIndex(0, 1, 0) == 4);          // then j (stride nz)
  CHECK(g.cellIndex(1, 0, 0) == 3 * 4);      // i slowest (stride ny*nz)
  CHECK(g.cellIndex(1, 2, 3) == g.numCells() - 1);

  // Every cell index is hit exactly once.
  std::vector<int> seen(g.numCells(), 0);
  for (int i = 0; i < g.nx(); ++i)
    for (int j = 0; j < g.ny(); ++j)
      for (int k = 0; k < g.nz(); ++k) ++seen[g.cellIndex(i, j, k)];
  for (int c : seen) CHECK(c == 1);
}

TEST_CASE("face indexing is a bijection within each axis family") {
  MacGrid g(2, 3, 4, 1.0);

  auto checkFamily = [](int count, auto idx, int ni, int nj, int nk) {
    std::vector<int> seen(count, 0);
    for (int i = 0; i < ni; ++i)
      for (int j = 0; j < nj; ++j)
        for (int k = 0; k < nk; ++k) {
          int f = idx(i, j, k);
          REQUIRE(f >= 0);
          REQUIRE(f < count);
          ++seen[f];
        }
    for (int c : seen) CHECK(c == 1);
  };

  checkFamily(g.numFacesX(), [&](int i, int j, int k) { return g.faceXIndex(i, j, k); }, 3, 3, 4);
  checkFamily(g.numFacesY(), [&](int i, int j, int k) { return g.faceYIndex(i, j, k); }, 2, 4, 4);
  checkFamily(g.numFacesZ(), [&](int i, int j, int k) { return g.faceZIndex(i, j, k); }, 2, 3, 5);
}

TEST_CASE("cell centers sit at the half-integer lattice offset from the origin") {
  MacGrid g(2, 3, 4, 0.5, Vec3{1.0, 2.0, 3.0});
  Vec3 c = g.cellCenter(0, 0, 0);
  CHECK(c[0] == doctest::Approx(1.0 + 0.25));
  CHECK(c[1] == doctest::Approx(2.0 + 0.25));
  CHECK(c[2] == doctest::Approx(3.0 + 0.25));

  Vec3 c2 = g.cellCenter(1, 2, 3);
  CHECK(c2[0] == doctest::Approx(1.0 + 1.5 * 0.5));
  CHECK(c2[1] == doctest::Approx(2.0 + 2.5 * 0.5));
  CHECK(c2[2] == doctest::Approx(3.0 + 3.5 * 0.5));
}

TEST_CASE("face centers lie on the grid planes, offset by half a cell in-plane") {
  MacGrid g(2, 3, 4, 0.5, Vec3{1.0, 2.0, 3.0});

  // An x-face is normal to x: integer x-plane, half offsets in y,z.
  Vec3 fx = g.faceXCenter(1, 0, 0);
  CHECK(fx[0] == doctest::Approx(1.0 + 1 * 0.5));
  CHECK(fx[1] == doctest::Approx(2.0 + 0.25));
  CHECK(fx[2] == doctest::Approx(3.0 + 0.25));

  // A y-face is normal to y.
  Vec3 fy = g.faceYCenter(0, 2, 0);
  CHECK(fy[0] == doctest::Approx(1.0 + 0.25));
  CHECK(fy[1] == doctest::Approx(2.0 + 2 * 0.5));
  CHECK(fy[2] == doctest::Approx(3.0 + 0.25));

  // A z-face is normal to z.
  Vec3 fz = g.faceZCenter(0, 0, 4);
  CHECK(fz[0] == doctest::Approx(1.0 + 0.25));
  CHECK(fz[1] == doctest::Approx(2.0 + 0.25));
  CHECK(fz[2] == doctest::Approx(3.0 + 4 * 0.5));
}

TEST_CASE("MacGrid rejects degenerate dimensions and spacing") {
  CHECK_THROWS(MacGrid(0, 1, 1, 1.0));
  CHECK_THROWS(MacGrid(1, 1, 1, 0.0));
  CHECK_THROWS(MacGrid(1, 1, 1, -1.0));
}
