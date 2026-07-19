/// \file
/// Red-green TDD for Phase B: an interior solid obstacle in the pressure
/// projection. The projected flow must not penetrate the solid, must be
/// divergence-free in the fluid, and must accelerate *around* the obstacle
/// (potential-flow signature) -- flow past a cylinder / sphere.
#include <doctest.h>

#include <cmath>

#include "grid/GridOperators.h"
#include "grid/MacGrid.h"
#include "grid/MacObstacle.h"
#include "fluid/MacProjection.h"

using namespace bochner;

// The mask geometry cases (cylinder/sphere/box) moved to the non-gated
// test_grid_physics.cpp -- they are PETSc-free and must run in the default
// build (review G1). The masks are still exercised below as projection setup.

TEST_CASE("flow past a cylinder: no penetration, divergence-free, accelerates around it") {
  MacGrid g(48, 32, 6, 0.1, Vec3{0, 0, 0});  // [0,4.8]x[0,3.2]x[0,0.6]
  const double U = 1.0;
  const Vec3 c{1.5, 1.6, 0.3};                // centered in y, upstream third in x
  const double rad = 0.5;
  SolidMask solid = cylinderMask(g, c, {0, 0, 1}, rad);  // spans z -> quasi-2D
  BoundarySpec bc = BoundarySpec::channelFlow(U);
  MacProjector proj(g, {}, bc, solid);

  // Project a uniform stream: the Hodge projection removes the part that would
  // penetrate the solid, yielding the steady potential flow around the cylinder.
  FaceField u = ops::zeroFaceField(g);
  for (double& v : u.x) v = U;
  const FaceField out = proj.project(u);

  // (1) No penetration: every face touching a solid cell has zero normal velocity.
  int checked = 0;
  for (int i = 1; i < g.nx(); ++i)
    for (int j = 0; j < g.ny(); ++j)
      for (int k = 0; k < g.nz(); ++k)
        if (isSolid(solid, g.cellIndex(i - 1, j, k)) || isSolid(solid, g.cellIndex(i, j, k))) {
          CHECK(out.x[g.faceXIndex(i, j, k)] == doctest::Approx(0.0).epsilon(1e-9));
          ++checked;
        }
  CHECK(checked > 0);

  // (2) Divergence-free in every FLUID cell.
  const auto d = ops::divergence(g, out);
  double maxFluidDiv = 0.0;
  for (int cc = 0; cc < g.numCells(); ++cc)
    if (!isSolid(solid, cc)) maxFluidDiv = std::max(maxFluidDiv, std::abs(d[cc]));
  CHECK(maxFluidDiv < 1e-5);

  // (3) The flow squeezes around the cylinder: at its top edge the streamwise
  // speed exceeds the free stream (potential flow -> up to 2U at the equator).
  const int iTop = static_cast<int>(c[0] / g.spacing());
  const int jTop = static_cast<int>((c[1] + rad + 0.15) / g.spacing());  // a bit above the cylinder
  const int kMid = g.nz() / 2;
  const double uxTop =
      0.5 * (out.x[g.faceXIndex(iTop, jTop, kMid)] + out.x[g.faceXIndex(iTop + 1, jTop, kMid)]);
  CHECK(uxTop > 1.15 * U);  // clearly accelerated (not merely ~U)
}

// --- Projection robustness (guards for the review's O1/O2/R2 findings) --------

TEST_CASE("O1: an enclosed fluid pocket does not NaN-poison the projection") {
  // Every cell solid except one interior cell -> that cell's faces are all solid
  // interfaces, so its Laplacian row has no diagonal. Unpinned, that is a divide-
  // by-zero in the Jacobi preconditioner; the assembler must pin such pockets.
  MacGrid g(5, 5, 5, 0.1, Vec3{0, 0, 0});
  SolidMask solid(g.numCells(), 1);
  const int hole = g.cellIndex(2, 2, 2);
  solid[hole] = 0;
  MacProjector proj(g, {}, BoundarySpec::channelFlow(1.0), solid);
  FaceField u = ops::zeroFaceField(g);
  for (double& v : u.x) v = 1.0;
  const FaceField out = proj.project(u);
  for (double v : out.x) CHECK(std::isfinite(v));
  for (double v : out.y) CHECK(std::isfinite(v));
  for (double v : out.z) CHECK(std::isfinite(v));
  const auto d = ops::divergence(g, out);
  CHECK(std::abs(d[hole]) < 1e-9);  // bounded by solid on all sides -> div 0
}

TEST_CASE("a multi-cell fluid pocket sealed off by solid gets exactly one ground") {
  // A hollow solid shell traps a 2-cell fluid pocket in an OTHERWISE-open domain.
  // The pocket touches no open wall, so its all-Neumann block is singular unless
  // one of its cells is pinned. The single-cell "untouched" pin does not catch it
  // (both pocket cells couple to each other, so both have a diagonal); grounding
  // must be per connected component, not per zero-diagonal cell.
  MacGrid g(8, 8, 8, 0.1, Vec3{0, 0, 0});
  SolidMask solid(g.numCells(), 1);  // all solid...
  const int a = g.cellIndex(3, 3, 3), b = g.cellIndex(3, 3, 4);
  solid[a] = solid[b] = 0;  // ...except a sealed 2-cell fluid pocket
  std::vector<int> pinned;
  bochner::pressureLaplacianObstacle(g, BoundarySpec::channelFlow(1.0), solid, &pinned);
  REQUIRE(pinned.size() == 1);           // exactly one ground for the one component
  CHECK((pinned[0] == a || pinned[0] == b));

  // A convex obstacle in the same open channel seals no pocket: the fluid all
  // connects to the open outlet, so no cell is pinned (the outlet grounds it).
  SolidMask conv = cylinderMask(g, {0.4, 0.4, 0.4}, {0, 0, 1}, 0.1);
  std::vector<int> pinnedConv;
  bochner::pressureLaplacianObstacle(g, BoundarySpec::channelFlow(1.0), conv, &pinnedConv);
  CHECK(pinnedConv.empty());
}

TEST_CASE("O2: fully-closed obstacle domain projects divergence-free (pin RHS zeroed)") {
  MacGrid g(16, 16, 8, 0.1, Vec3{0, 0, 0});
  SolidMask solid = sphereMask(g, {0.8, 0.8, 0.4}, 0.2);
  MacProjector proj(g, {}, BoundarySpec::allClosed(), solid);  // closed -> a fluid cell is pinned
  FaceField u = ops::zeroFaceField(g);
  for (int i = 0; i <= g.nx(); ++i)
    for (int j = 0; j < g.ny(); ++j)
      for (int k = 0; k < g.nz(); ++k) u.x[g.faceXIndex(i, j, k)] = 0.1 * i;  // divergent
  const FaceField out = proj.project(u);
  const auto d = ops::divergence(g, out);
  double maxFluidDiv = 0.0;
  for (int cc = 0; cc < g.numCells(); ++cc)
    if (!isSolid(solid, cc)) maxFluidDiv = std::max(maxFluidDiv, std::abs(d[cc]));
  CHECK(maxFluidDiv < 1e-5);  // if the pin's RHS is not zeroed, phi_pin != 0 spikes div here
  for (double v : out.x) CHECK(std::isfinite(v));
}

TEST_CASE("R2: re-projecting the same field (exact warm start) stays finite") {
  MacGrid g(24, 16, 6, 0.1, Vec3{0, 0, 0});
  SolidMask solid = cylinderMask(g, {0.8, 0.8, 0.3}, {0, 0, 1}, 0.25);
  MacProjector proj(g, {}, BoundarySpec::channelFlow(1.0), solid);
  FaceField u = ops::zeroFaceField(g);
  for (double& v : u.x) v = 1.0;
  proj.project(u);                         // solve once -> phi_ holds the exact solution
  const FaceField out = proj.project(u);   // same input -> initial residual ~0 -> must not 0/0
  for (double v : out.x) CHECK(std::isfinite(v));
  const auto d = ops::divergence(g, out);
  double maxFluidDiv = 0.0;
  for (int cc = 0; cc < g.numCells(); ++cc)
    if (!isSolid(solid, cc)) maxFluidDiv = std::max(maxFluidDiv, std::abs(d[cc]));
  CHECK(maxFluidDiv < 1e-5);
}
