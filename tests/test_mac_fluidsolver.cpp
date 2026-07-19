/// \file
/// Red-green TDD: the Covector Fluids time step on the MAC grid (CF step 3d).
///
/// Assembles Alg 1 (freeze -> BFECC covector advection -> projection) and the
/// Alg 2 midpoint variant. The headline property (CF Fig 8): with BFECC the
/// inviscid solver preserves kinetic energy far better than a plain
/// semi-Lagrangian step, which is the cure for the dissipation that collapsed
/// the vortex ring on the tet prototype.
#include <doctest.h>

#include <cmath>

#include "grid/GridOperators.h"
#include "fluid/MacAdvection.h"
#include "fluid/MacFluidSolver.h"
#include "grid/MacGrid.h"
#include "fluid/MacProjection.h"
#include "fluid/MacVortexRing.h"

using bochner::FaceField;
using bochner::MacGrid;
using bochner::Vec3;
namespace ops = bochner::ops;

namespace {

template <class Fx, class Fy, class Fz>
FaceField sampleField(const MacGrid& g, Fx fx, Fy fy, Fz fz) {
  FaceField u = ops::zeroFaceField(g);
  for (int i = 0; i <= g.nx(); ++i)
    for (int j = 0; j < g.ny(); ++j)
      for (int k = 0; k < g.nz(); ++k) u.x[g.faceXIndex(i, j, k)] = fx(g.faceXCenter(i, j, k));
  for (int i = 0; i < g.nx(); ++i)
    for (int j = 0; j <= g.ny(); ++j)
      for (int k = 0; k < g.nz(); ++k) u.y[g.faceYIndex(i, j, k)] = fy(g.faceYCenter(i, j, k));
  for (int i = 0; i < g.nx(); ++i)
    for (int j = 0; j < g.ny(); ++j)
      for (int k = 0; k <= g.nz(); ++k) u.z[g.faceZIndex(i, j, k)] = fz(g.faceZCenter(i, j, k));
  return u;
}

double maxAbsDiv(const MacGrid& g, const FaceField& u) {
  auto d = ops::divergence(g, u);
  double m = 0.0;
  for (double x : d) m = std::max(m, std::abs(x));
  return m;
}

double energy(const FaceField& u) {
  double e = 0.0;
  for (double x : u.x) e += x * x;
  for (double x : u.y) e += x * x;
  for (double x : u.z) e += x * x;
  return e;
}

// A 3D Taylor-Green vortex on [0,pi]^3: analytically divergence-free, and
// no-penetration (normal velocity vanishes on every wall) for this domain.
FaceField taylorGreen(const MacGrid& g) {
  return sampleField(
      g, [](Vec3 p) { return std::sin(2 * p[0]) * std::cos(2 * p[1]); },
      [](Vec3 p) { return -std::cos(2 * p[0]) * std::sin(2 * p[1]); },
      [](Vec3) { return 0.0; });
}

}  // namespace

TEST_CASE("a CF step leaves the velocity discretely divergence-free") {
  MacGrid g(16, 16, 16, M_PI / 16.0);
  FaceField u0 = bochner::projectToDivergenceFree(g, taylorGreen(g));
  CHECK(maxAbsDiv(g, bochner::stepCovectorFluids(g, u0, 0.05)) < 1e-7);
  CHECK(maxAbsDiv(g, bochner::stepCovectorFluidsMidpoint(g, u0, 0.05)) < 1e-7);
}

TEST_CASE("the CF step does not amplify energy (stability)") {
  MacGrid g(16, 16, 16, M_PI / 16.0);
  FaceField u0 = bochner::projectToDivergenceFree(g, taylorGreen(g));
  double e0 = energy(u0);
  FaceField u1 = bochner::stepCovectorFluids(g, u0, 0.05);
  CHECK(energy(u1) <= e0 * 1.001);
}

TEST_CASE("BFECC preserves kinetic energy far better than plain semi-Lagrangian") {
  MacGrid g(24, 24, 24, M_PI / 24.0);
  FaceField u0 = bochner::projectToDivergenceFree(g, taylorGreen(g));
  const double e0 = energy(u0);
  const double dt = 0.04;
  const int steps = 12;

  FaceField uBF = u0, uSL = u0;
  for (int s = 0; s < steps; ++s) {
    uBF = bochner::stepCovectorFluids(g, uBF, dt);  // BFECC advection + projection
    // Reference: identical step but with plain semi-Lagrangian advection.
    uSL = bochner::projectToDivergenceFree(g, bochner::advectCovectorSL(g, uSL, uSL, dt));
  }

  double retBF = energy(uBF) / e0;
  double retSL = energy(uSL) / e0;
  CHECK(retBF > retSL);        // BFECC is the less-dissipative scheme
  CHECK(retBF > 0.85);         // and it conserves most of the energy
  CHECK(retSL < retBF - 0.05); // sL visibly bleeds energy over the same steps
}

TEST_CASE("a self-advecting vortex ring keeps its energy and does not collapse") {
  // The demo-relevant invariant (separate from Taylor-Green): a seeded ring
  // should self-propel without the dissipative ring-collapse that plagued the
  // tet prototype. We check energy retention and that the ring stays a ring of
  // ~constant radius (enstrophy-weighted cylindrical radius about the z-axis).
  const double L = 1.4, h = 2.0 * L / 24;
  MacGrid g(24, 24, 24, h, Vec3{-L, -L, -L});
  const double R = 0.7, Gamma = 1.0, core = 0.15;
  bochner::MacProjector proj(g);
  FaceField u = proj.project(bochner::vortexRingFaceField(g, {0, 0, 0}, {0, 0, 1}, R, Gamma, core));

  // Enstrophy-weighted ring radius from a cell-centered curl (axis = z).
  auto ringRadius = [&](const FaceField& f) {
    std::vector<double> cx(g.numCells()), cy(g.numCells()), cz(g.numCells());
    for (int i = 0; i < g.nx(); ++i)
      for (int j = 0; j < g.ny(); ++j)
        for (int k = 0; k < g.nz(); ++k) {
          const int c = g.cellIndex(i, j, k);
          cx[c] = 0.5 * (f.x[g.faceXIndex(i, j, k)] + f.x[g.faceXIndex(i + 1, j, k)]);
          cy[c] = 0.5 * (f.y[g.faceYIndex(i, j, k)] + f.y[g.faceYIndex(i, j + 1, k)]);
          cz[c] = 0.5 * (f.z[g.faceZIndex(i, j, k)] + f.z[g.faceZIndex(i, j, k + 1)]);
        }
    const double inv2h = 1.0 / (2.0 * h);
    double sw = 0, swr = 0;
    for (int i = 1; i < g.nx() - 1; ++i)
      for (int j = 1; j < g.ny() - 1; ++j)
        for (int k = 1; k < g.nz() - 1; ++k) {
          auto X = [&](int a, int b, int d) { return cx[g.cellIndex(a, b, d)]; };
          auto Y = [&](int a, int b, int d) { return cy[g.cellIndex(a, b, d)]; };
          auto Z = [&](int a, int b, int d) { return cz[g.cellIndex(a, b, d)]; };
          const double wx = (Z(i, j + 1, k) - Z(i, j - 1, k)) * inv2h - (Y(i, j, k + 1) - Y(i, j, k - 1)) * inv2h;
          const double wy = (X(i, j, k + 1) - X(i, j, k - 1)) * inv2h - (Z(i + 1, j, k) - Z(i - 1, j, k)) * inv2h;
          const double wz = (Y(i + 1, j, k) - Y(i - 1, j, k)) * inv2h - (X(i, j + 1, k) - X(i, j - 1, k)) * inv2h;
          const double w2 = wx * wx + wy * wy + wz * wz;
          const Vec3 p = g.cellCenter(i, j, k);
          sw += w2;
          swr += w2 * std::sqrt(p[0] * p[0] + p[1] * p[1]);
        }
    return sw > 0 ? swr / sw : 0.0;
  };

  const double e0 = energy(u), R0 = ringRadius(u);
  for (int s = 0; s < 12; ++s) u = proj.project(bochner::advectCovectorBFECC(g, u, u, 0.04));
  const double ret = energy(u) / e0, Rend = ringRadius(u);

  CHECK(ret > 0.9);                                       // low dissipation (no collapse)
  CHECK(R0 == doctest::Approx(0.7).epsilon(0.1));         // initial radius is the seed radius
  CHECK(Rend == doctest::Approx(R0).epsilon(0.15));       // ring keeps its radius (does not shrink/smear)
}
