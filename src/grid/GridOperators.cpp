#include "grid/GridOperators.h"

#include <algorithm>
#include <cmath>

namespace bochner::ops {

FaceField zeroFaceField(const MacGrid& g) {
  return FaceField{std::vector<double>(g.numFacesX(), 0.0),
                   std::vector<double>(g.numFacesY(), 0.0),
                   std::vector<double>(g.numFacesZ(), 0.0)};
}

std::vector<double> divergence(const MacGrid& g, const FaceField& u) {
  const double inv_h = 1.0 / g.spacing();
  std::vector<double> div(g.numCells(), 0.0);
  for (int i = 0; i < g.nx(); ++i)
    for (int j = 0; j < g.ny(); ++j)
      for (int k = 0; k < g.nz(); ++k) {
        // Net outward flux: (high-side face) - (low-side face) along each axis.
        double d = (u.x[g.faceXIndex(i + 1, j, k)] - u.x[g.faceXIndex(i, j, k)]) +
                   (u.y[g.faceYIndex(i, j + 1, k)] - u.y[g.faceYIndex(i, j, k)]) +
                   (u.z[g.faceZIndex(i, j, k + 1)] - u.z[g.faceZIndex(i, j, k)]);
        div[g.cellIndex(i, j, k)] = d * inv_h;
      }
  return div;
}

FaceField gradient(const MacGrid& g, const std::vector<double>& p) {
  const double inv_h = 1.0 / g.spacing();
  FaceField gp = zeroFaceField(g);

  // Interior x-faces (1 <= i <= nx-1) straddle two cells; boundary faces stay 0.
  for (int i = 1; i < g.nx(); ++i)
    for (int j = 0; j < g.ny(); ++j)
      for (int k = 0; k < g.nz(); ++k)
        gp.x[g.faceXIndex(i, j, k)] =
            (p[g.cellIndex(i, j, k)] - p[g.cellIndex(i - 1, j, k)]) * inv_h;

  for (int i = 0; i < g.nx(); ++i)
    for (int j = 1; j < g.ny(); ++j)
      for (int k = 0; k < g.nz(); ++k)
        gp.y[g.faceYIndex(i, j, k)] =
            (p[g.cellIndex(i, j, k)] - p[g.cellIndex(i, j - 1, k)]) * inv_h;

  for (int i = 0; i < g.nx(); ++i)
    for (int j = 0; j < g.ny(); ++j)
      for (int k = 1; k < g.nz(); ++k)
        gp.z[g.faceZIndex(i, j, k)] =
            (p[g.cellIndex(i, j, k)] - p[g.cellIndex(i, j, k - 1)]) * inv_h;

  return gp;
}

FaceField gradient(const MacGrid& g, const std::vector<double>& p, const BoundarySpec& bc) {
  FaceField gp = gradient(g, p);  // interior faces; all boundary-normal faces 0
  const double inv_h = 1.0 / g.spacing();
  // On an OPEN wall the pressure just outside is pinned to 0 (Dirichlet ghost),
  // so the boundary-normal face carries the ghost difference -> outflow.
  if (bc.xlo)
    for (int j = 0; j < g.ny(); ++j)
      for (int k = 0; k < g.nz(); ++k)
        gp.x[g.faceXIndex(0, j, k)] = (p[g.cellIndex(0, j, k)] - 0.0) * inv_h;
  if (bc.xhi)
    for (int j = 0; j < g.ny(); ++j)
      for (int k = 0; k < g.nz(); ++k)
        gp.x[g.faceXIndex(g.nx(), j, k)] = (0.0 - p[g.cellIndex(g.nx() - 1, j, k)]) * inv_h;
  if (bc.ylo)
    for (int i = 0; i < g.nx(); ++i)
      for (int k = 0; k < g.nz(); ++k)
        gp.y[g.faceYIndex(i, 0, k)] = (p[g.cellIndex(i, 0, k)] - 0.0) * inv_h;
  if (bc.yhi)
    for (int i = 0; i < g.nx(); ++i)
      for (int k = 0; k < g.nz(); ++k)
        gp.y[g.faceYIndex(i, g.ny(), k)] = (0.0 - p[g.cellIndex(i, g.ny() - 1, k)]) * inv_h;
  if (bc.zlo)
    for (int i = 0; i < g.nx(); ++i)
      for (int j = 0; j < g.ny(); ++j)
        gp.z[g.faceZIndex(i, j, 0)] = (p[g.cellIndex(i, j, 0)] - 0.0) * inv_h;
  if (bc.zhi)
    for (int i = 0; i < g.nx(); ++i)
      for (int j = 0; j < g.ny(); ++j)
        gp.z[g.faceZIndex(i, j, g.nz())] = (0.0 - p[g.cellIndex(i, j, g.nz() - 1)]) * inv_h;
  return gp;
}

void enforceNoPenetration(const MacGrid& g, FaceField& u, const BoundarySpec& bc) {
  // x-lo: an INLET prescribes u.n = inflowX (inhomogeneous Neumann -- the flux
  // enters div natively and is NOT pressure-corrected); a CLOSED wall zeroes it;
  // an OPEN wall is left for the outflow correction.
  if (bc.inletXlo())
    for (int j = 0; j < g.ny(); ++j)
      for (int k = 0; k < g.nz(); ++k) u.x[g.faceXIndex(0, j, k)] = bc.inflowX;
  else if (!bc.xlo)
    for (int j = 0; j < g.ny(); ++j)
      for (int k = 0; k < g.nz(); ++k) u.x[g.faceXIndex(0, j, k)] = 0.0;
  if (!bc.xhi)
    for (int j = 0; j < g.ny(); ++j)
      for (int k = 0; k < g.nz(); ++k) u.x[g.faceXIndex(g.nx(), j, k)] = 0.0;
  if (!bc.ylo)
    for (int i = 0; i < g.nx(); ++i)
      for (int k = 0; k < g.nz(); ++k) u.y[g.faceYIndex(i, 0, k)] = 0.0;
  if (!bc.yhi)
    for (int i = 0; i < g.nx(); ++i)
      for (int k = 0; k < g.nz(); ++k) u.y[g.faceYIndex(i, g.ny(), k)] = 0.0;
  if (!bc.zlo)
    for (int i = 0; i < g.nx(); ++i)
      for (int j = 0; j < g.ny(); ++j) u.z[g.faceZIndex(i, j, 0)] = 0.0;
  if (!bc.zhi)
    for (int i = 0; i < g.nx(); ++i)
      for (int j = 0; j < g.ny(); ++j) u.z[g.faceZIndex(i, j, g.nz())] = 0.0;
}

void enforceNoPenetration(const MacGrid& g, FaceField& u) {
  // Zero the wall-normal faces on each of the six domain boundaries.
  for (int j = 0; j < g.ny(); ++j)
    for (int k = 0; k < g.nz(); ++k) {
      u.x[g.faceXIndex(0, j, k)] = 0.0;
      u.x[g.faceXIndex(g.nx(), j, k)] = 0.0;
    }
  for (int i = 0; i < g.nx(); ++i)
    for (int k = 0; k < g.nz(); ++k) {
      u.y[g.faceYIndex(i, 0, k)] = 0.0;
      u.y[g.faceYIndex(i, g.ny(), k)] = 0.0;
    }
  for (int i = 0; i < g.nx(); ++i)
    for (int j = 0; j < g.ny(); ++j) {
      u.z[g.faceZIndex(i, j, 0)] = 0.0;
      u.z[g.faceZIndex(i, j, g.nz())] = 0.0;
    }
}

FaceField diffuseVelocity(const MacGrid& g, const FaceField& u, double nu, double dt,
                          const SolidMask& solid) {
  // Explicit forward-Euler diffusion is stable only for c = nu*dt/h^2 <= 1/6 (the
  // 6-neighbour stencil's worst mode amplifies by |1 - 12c|). If the requested
  // step exceeds that, split it into m equal, individually-stable sub-steps, so
  // the call is unconditionally stable with no caller-side sub-stepping guard.
  // For c < 1/6 this is m = 1, byte-identical to a single explicit step.
  const double cTotal = nu * dt / (g.spacing() * g.spacing());
  int m = std::max(1, static_cast<int>(std::ceil(6.0 * cTotal)));
  // Exact multiples land on c = 1/6 per sub-step, where the checkerboard mode's
  // amplification is exactly -1: stable but oscillating forever, never damped.
  // One extra sub-step turns that edge case into genuine decay.
  if (6.0 * cTotal == static_cast<double>(m)) ++m;
  const double c = cTotal / m;

  auto cs = [&](int i, int j, int k) {
    return i >= 0 && i < g.nx() && j >= 0 && j < g.ny() && k >= 0 && k < g.nz() &&
           isSolid(solid, g.cellIndex(i, j, k));
  };
  auto xWall = [&](int i, int j, int k) { return cs(i - 1, j, k) || cs(i, j, k); };
  auto yWall = [&](int i, int j, int k) { return cs(i, j - 1, k) || cs(i, j, k); };
  auto zWall = [&](int i, int j, int k) { return cs(i, j, k - 1) || cs(i, j, k); };

  FaceField out = u;
  for (int step = 0; step < m; ++step) {
    const FaceField in = out;  // each sub-step diffuses the previous one's field
    // Interior x-faces. A same-family neighbour that touches the solid contributes
    // 0 (no-slip); an off-domain neighbour is dropped (Neumann free-slip walls).
    for (int i = 1; i < g.nx(); ++i)
      for (int j = 0; j < g.ny(); ++j)
        for (int k = 0; k < g.nz(); ++k) {
          if (xWall(i, j, k)) continue;
          const double ctr = in.x[g.faceXIndex(i, j, k)];
          double lap = 0.0;
          auto add = [&](int ii, int jj, int kk) {
            lap += (xWall(ii, jj, kk) ? 0.0 : in.x[g.faceXIndex(ii, jj, kk)]) - ctr;
          };
          add(i - 1, j, k);
          add(i + 1, j, k);
          if (j - 1 >= 0) add(i, j - 1, k);
          if (j + 1 < g.ny()) add(i, j + 1, k);
          if (k - 1 >= 0) add(i, j, k - 1);
          if (k + 1 < g.nz()) add(i, j, k + 1);
          out.x[g.faceXIndex(i, j, k)] = ctr + c * lap;
        }
    for (int i = 0; i < g.nx(); ++i)
      for (int j = 1; j < g.ny(); ++j)
        for (int k = 0; k < g.nz(); ++k) {
          if (yWall(i, j, k)) continue;
          const double ctr = in.y[g.faceYIndex(i, j, k)];
          double lap = 0.0;
          auto add = [&](int ii, int jj, int kk) {
            lap += (yWall(ii, jj, kk) ? 0.0 : in.y[g.faceYIndex(ii, jj, kk)]) - ctr;
          };
          add(i, j - 1, k);
          add(i, j + 1, k);
          if (i - 1 >= 0) add(i - 1, j, k);
          if (i + 1 < g.nx()) add(i + 1, j, k);
          if (k - 1 >= 0) add(i, j, k - 1);
          if (k + 1 < g.nz()) add(i, j, k + 1);
          out.y[g.faceYIndex(i, j, k)] = ctr + c * lap;
        }
    for (int i = 0; i < g.nx(); ++i)
      for (int j = 0; j < g.ny(); ++j)
        for (int k = 1; k < g.nz(); ++k) {
          if (zWall(i, j, k)) continue;
          const double ctr = in.z[g.faceZIndex(i, j, k)];
          double lap = 0.0;
          auto add = [&](int ii, int jj, int kk) {
            lap += (zWall(ii, jj, kk) ? 0.0 : in.z[g.faceZIndex(ii, jj, kk)]) - ctr;
          };
          add(i, j, k - 1);
          add(i, j, k + 1);
          if (i - 1 >= 0) add(i - 1, j, k);
          if (i + 1 < g.nx()) add(i + 1, j, k);
          if (j - 1 >= 0) add(i, j - 1, k);
          if (j + 1 < g.ny()) add(i, j + 1, k);
          out.z[g.faceZIndex(i, j, k)] = ctr + c * lap;
        }
  }
  return out;
}

}  // namespace bochner::ops
