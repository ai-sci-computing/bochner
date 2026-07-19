/// \file
/// Red-green TDD for the PETSc-free grid-layer physics that the default build
/// must cover (review G1): the obstacle masks (cylinder/sphere/oriented box),
/// no-penetration wall zeroing, and the open-wall (Dirichlet-ghost) gradient
/// overload. None of these needs PETSc, so they live in the non-gated group --
/// unlike the projection tests that also exercise them via MacProjector.
#include <doctest.h>

#include <cmath>

#include "grid/GridOperators.h"
#include "grid/MacGrid.h"
#include "grid/MacObstacle.h"

using namespace bochner;

// --- Obstacle masks -----------------------------------------------------------

TEST_CASE("cylinderMask/sphereMask mark the interior cells solid") {
  MacGrid g(20, 20, 8, 0.1, Vec3{0, 0, 0});  // [0,2]x[0,2]x[0,0.8]
  const Vec3 c{1.0, 1.0, 0.4};
  auto cyl = cylinderMask(g, c, {0, 0, 1}, 0.3);  // axis along z
  CHECK(isSolid(cyl, g.cellIndex(10, 10, 4)));    // near axis -> solid
  CHECK(isSolid(cyl, g.cellIndex(10, 10, 0)));    // spans z (axis-aligned) -> solid at all k
  CHECK_FALSE(isSolid(cyl, g.cellIndex(2, 2, 4)));  // far corner -> fluid

  auto sph = sphereMask(g, c, 0.3);
  CHECK(isSolid(sph, g.cellIndex(10, 10, 4)));      // center -> solid
  CHECK_FALSE(isSolid(sph, g.cellIndex(10, 10, 0)));  // off-center in z -> fluid (unlike cylinder)
}

TEST_CASE("boxMask marks a finite axis-aligned box solid, finite in every direction") {
  MacGrid g(20, 20, 20, 0.1, Vec3{0, 0, 0});  // [0,2]^3
  const Vec3 c{1.0, 1.0, 1.0};
  // Half-extents 0.25 (x), 0.25 (y), 0.2 (z): finite in z (does NOT span the domain).
  auto box = boxMask(g, c, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, Vec3{0.25, 0.25, 0.2});
  CHECK(isSolid(box, g.cellIndex(10, 10, 10)));  // at center -> solid
  CHECK(isSolid(box, g.cellIndex(12, 12, 10)));  // dx=dy=0.2 -> inside the square section
  CHECK_FALSE(isSolid(box, g.cellIndex(14, 10, 10)));  // dx=0.4 > 0.25 -> fluid
  CHECK_FALSE(isSolid(box, g.cellIndex(10, 10, 13)));  // dz=0.3 > 0.2 -> FLUID (finite span!)
  CHECK(isSolid(box, g.cellIndex(10, 10, 11)));        // dz=0.15 <= 0.2 -> solid
}

TEST_CASE("boxMask respects an inclined (yawed) orientation") {
  MacGrid g(40, 40, 6, 0.05, Vec3{0, 0, 0});  // [0,2]x[0,2]x[0,0.3]
  const Vec3 c{1.0, 1.0, 0.15};
  // A thin plate: 45-degree yaw about z, wide along (1,1,0), thin across (1,-1,0).
  const double s = std::sqrt(0.5);
  auto plate = boxMask(g, c, {s, -s, 0}, {s, s, 0}, {0, 0, 1}, Vec3{0.03, 0.4, 0.15});
  CHECK(isSolid(plate, g.cellIndex(20, 20, 3)));  // at center -> solid
  // Wide direction (equal +dx,+dy) stays on the plate; thin direction (+dx,-dy) leaves.
  CHECK(isSolid(plate, g.cellIndex(24, 24, 3)));       // (0.2,0.2) along width -> solid
  CHECK_FALSE(isSolid(plate, g.cellIndex(24, 16, 3)));  // (0.2,-0.2) across thin -> fluid
}

// --- No-penetration -----------------------------------------------------------

TEST_CASE("enforceNoPenetration zeroes all six wall-normal faces, leaves the interior") {
  MacGrid g(6, 5, 4, 0.1, Vec3{0, 0, 0});
  FaceField u = ops::zeroFaceField(g);
  for (double& v : u.x) v = 1.0;
  for (double& v : u.y) v = 1.0;
  for (double& v : u.z) v = 1.0;
  ops::enforceNoPenetration(g, u);

  for (int j = 0; j < g.ny(); ++j)
    for (int k = 0; k < g.nz(); ++k) {
      CHECK(u.x[g.faceXIndex(0, j, k)] == 0.0);
      CHECK(u.x[g.faceXIndex(g.nx(), j, k)] == 0.0);
    }
  for (int i = 0; i < g.nx(); ++i)
    for (int k = 0; k < g.nz(); ++k) {
      CHECK(u.y[g.faceYIndex(i, 0, k)] == 0.0);
      CHECK(u.y[g.faceYIndex(i, g.ny(), k)] == 0.0);
    }
  for (int i = 0; i < g.nx(); ++i)
    for (int j = 0; j < g.ny(); ++j) {
      CHECK(u.z[g.faceZIndex(i, j, 0)] == 0.0);
      CHECK(u.z[g.faceZIndex(i, j, g.nz())] == 0.0);
    }
  CHECK(u.x[g.faceXIndex(3, 2, 2)] == 1.0);  // an interior face is untouched
}

TEST_CASE("channelFlow no-penetration: inlet prescribed, sides closed, open outlet kept") {
  MacGrid g(6, 5, 4, 0.1, Vec3{0, 0, 0});
  const double U = 2.0;
  FaceField u = ops::zeroFaceField(g);
  for (double& v : u.x) v = 1.0;
  for (double& v : u.y) v = 1.0;
  for (double& v : u.z) v = 1.0;
  ops::enforceNoPenetration(g, u, BoundarySpec::channelFlow(U));

  for (int j = 0; j < g.ny(); ++j)
    for (int k = 0; k < g.nz(); ++k) {
      CHECK(u.x[g.faceXIndex(0, j, k)] == U);        // x-lo inlet -> prescribed inflow
      CHECK(u.x[g.faceXIndex(g.nx(), j, k)] == 1.0);  // x-hi open outlet -> untouched
    }
  for (int i = 0; i < g.nx(); ++i)
    for (int k = 0; k < g.nz(); ++k) {
      CHECK(u.y[g.faceYIndex(i, 0, k)] == 0.0);        // closed side walls zeroed
      CHECK(u.y[g.faceYIndex(i, g.ny(), k)] == 0.0);
    }
}

// --- Open-wall gradient overload ----------------------------------------------

TEST_CASE("open-wall gradient adds the Dirichlet (p-0)/h ghost, closed walls stay zero") {
  MacGrid g(6, 5, 4, 0.1, Vec3{0, 0, 0});
  const double h = g.spacing();
  std::vector<double> p(g.numCells());
  for (int i = 0; i < g.nx(); ++i)
    for (int j = 0; j < g.ny(); ++j)
      for (int k = 0; k < g.nz(); ++k) p[g.cellIndex(i, j, k)] = i + 1.0;  // nonzero everywhere

  const FaceField plain = ops::gradient(g, p);           // all boundary-normal faces zero
  const FaceField open = ops::gradient(g, p, BoundarySpec::allOpen());

  for (int j = 0; j < g.ny(); ++j)
    for (int k = 0; k < g.nz(); ++k) {
      CHECK(plain.x[g.faceXIndex(0, j, k)] == 0.0);        // closed convention: zero
      CHECK(plain.x[g.faceXIndex(g.nx(), j, k)] == 0.0);
      // Open ghost: (p_cell - 0)/h at x-lo, (0 - p_cell)/h at x-hi.
      CHECK(open.x[g.faceXIndex(0, j, k)] ==
            doctest::Approx((p[g.cellIndex(0, j, k)] - 0.0) / h));
      CHECK(open.x[g.faceXIndex(g.nx(), j, k)] ==
            doctest::Approx((0.0 - p[g.cellIndex(g.nx() - 1, j, k)]) / h));
    }

  // With only x-hi open (channelFlow), x-lo keeps the closed zero-face convention.
  const FaceField chan = ops::gradient(g, p, BoundarySpec::channelFlow(1.0));
  for (int j = 0; j < g.ny(); ++j)
    for (int k = 0; k < g.nz(); ++k) {
      CHECK(chan.x[g.faceXIndex(0, j, k)] == 0.0);  // x-lo not open -> zero
      CHECK(chan.x[g.faceXIndex(g.nx(), j, k)] ==
            doctest::Approx((0.0 - p[g.cellIndex(g.nx() - 1, j, k)]) / h));
    }

  // Interior faces are identical between the plain and BC-aware gradients.
  for (int i = 1; i < g.nx(); ++i)
    for (int j = 0; j < g.ny(); ++j)
      for (int k = 0; k < g.nz(); ++k)
        CHECK(open.x[g.faceXIndex(i, j, k)] == plain.x[g.faceXIndex(i, j, k)]);
}
