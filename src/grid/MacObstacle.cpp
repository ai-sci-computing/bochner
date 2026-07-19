#include "grid/MacObstacle.h"

#include <algorithm>

#include <cmath>

namespace bochner {

SolidMask cylinderMask(const MacGrid& g, const Vec3& center, const Vec3& axis, double radius) {
  const double an = vnorm(axis);
  const Vec3 ah = an > 1e-12 ? vscale(axis, 1.0 / an) : Vec3{0, 0, 1};
  SolidMask solid(g.numCells(), 0);
  const double r2 = radius * radius;
  for (int i = 0; i < g.nx(); ++i)
    for (int j = 0; j < g.ny(); ++j)
      for (int k = 0; k < g.nz(); ++k) {
        const Vec3 d = vsub(g.cellCenter(i, j, k), center);
        const Vec3 perp = vsub(d, vscale(ah, vdot(d, ah)));  // component perpendicular to the axis
        if (vdot(perp, perp) <= r2) solid[g.cellIndex(i, j, k)] = 1;
      }
  return solid;
}

SolidMask sphereMask(const MacGrid& g, const Vec3& center, double radius) {
  SolidMask solid(g.numCells(), 0);
  const double r2 = radius * radius;
  for (int i = 0; i < g.nx(); ++i)
    for (int j = 0; j < g.ny(); ++j)
      for (int k = 0; k < g.nz(); ++k) {
        const Vec3 d = vsub(g.cellCenter(i, j, k), center);
        if (vdot(d, d) <= r2) solid[g.cellIndex(i, j, k)] = 1;
      }
  return solid;
}

namespace {
/// Normalize \p a, or return \p fallback if it is (near) zero length.
Vec3 unit(const Vec3& a, const Vec3& fallback) {
  const double n = vnorm(a);
  return n > 1e-12 ? vscale(a, 1.0 / n) : fallback;
}
}  // namespace

SolidMask boxMask(const MacGrid& g, const Vec3& center, const Vec3& ex, const Vec3& ey,
                  const Vec3& ez, const Vec3& half) {
  const Vec3 ux = unit(ex, {1, 0, 0}), uy = unit(ey, {0, 1, 0}), uz = unit(ez, {0, 0, 1});
  // A pure cell-center point-in-box test misses a box thinner than the cell
  // spacing ENTIRELY: no center lands inside, so the mask comes back all-zero
  // while the caller -- and the renderer -- still believe there is a body, and
  // fluid passes straight through the drawn obstacle. At an oblique yaw a
  // sub-cell thickness instead gives a gappy staircase that leaks along the
  // diagonal. Both were reachable from the shipped "inclined plate" preset,
  // whose 0.05 half-extent is under one cell at the low end of the resolution
  // slider.
  //
  // Widen each half-extent to at least half a cell so a thin body always
  // resolves to a watertight single layer. This is conservative -- the
  // represented body can be up to one cell thicker than requested along a
  // sub-cell axis -- which is the right direction: a slightly thick plate is a
  // plate, an infinitely thin one is nothing.
  const double hh = 0.5 * g.spacing();
  const Vec3 hw{std::max(half[0], hh), std::max(half[1], hh), std::max(half[2], hh)};
  SolidMask solid(g.numCells(), 0);
  for (int i = 0; i < g.nx(); ++i)
    for (int j = 0; j < g.ny(); ++j)
      for (int k = 0; k < g.nz(); ++k) {
        const Vec3 d = vsub(g.cellCenter(i, j, k), center);
        if (std::abs(vdot(d, ux)) <= hw[0] && std::abs(vdot(d, uy)) <= hw[1] &&
            std::abs(vdot(d, uz)) <= hw[2])
          solid[g.cellIndex(i, j, k)] = 1;
      }
  return solid;
}

}  // namespace bochner
