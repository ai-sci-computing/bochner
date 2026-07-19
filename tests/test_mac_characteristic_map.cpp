/// \file
/// Red-green TDD: the CF+MCM solver (Covector Fluids Algorithm 5, the long-time
/// characteristic mapping). It must stay divergence-free, preserve a vortex
/// ring's energy at least as well as the base BFECC method (Alg. 1) -- the whole
/// point of advecting the map instead of the velocity -- and reinitialize its
/// maps on schedule.
#include <doctest.h>

#include <cmath>

#include "grid/GridOperators.h"
#include "fluid/MacAdvection.h"
#include "fluid/MacCharacteristicMap.h"
#include "grid/MacGrid.h"
#include "fluid/MacProjection.h"
#include "fluid/MacVortexRing.h"

using bochner::CfMcmSolver;
using bochner::FaceField;
using bochner::MacGrid;
using bochner::Vec3;
namespace ops = bochner::ops;

namespace {

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

FaceField ring(const MacGrid& g) {
  return bochner::vortexRingFaceField(g, {0, 0, 0}, {0, 0, 1}, 0.7, 1.0, 0.15);
}

// Discrete circulation around an axis-aligned rectangular loop in the x-z plane
// (fixed cell row j), built from the STORED MAC face-normal components. Because
// each edge contribution is a stored u.x / u.z face value, a pure-gradient field
// telescopes around the closed loop to exactly zero -- so this isolates the
// projection's circulation-neutrality to round-off (substep audit).
double discreteCirculationXZ(const MacGrid& g, const FaceField& u, int ia, int ib, int ka, int kb,
                             int j) {
  const double h = g.spacing();
  double gamma = 0.0;
  for (int i = ia + 1; i <= ib; ++i) {
    gamma += u.x[g.faceXIndex(i, j, ka)] * h;   // bottom edge, +x
    gamma -= u.x[g.faceXIndex(i, j, kb)] * h;   // top edge,   -x
  }
  for (int k = ka + 1; k <= kb; ++k) {
    gamma += u.z[g.faceZIndex(ib, j, k)] * h;   // right edge, +z
    gamma -= u.z[g.faceZIndex(ia, j, k)] * h;   // left edge,  -z
  }
  return gamma;
}

// L2 difference between two face fields (same grid).
double l2diff(const FaceField& a, const FaceField& b) {
  double s = 0.0;
  for (size_t f = 0; f < a.x.size(); ++f) s += (a.x[f] - b.x[f]) * (a.x[f] - b.x[f]);
  for (size_t f = 0; f < a.y.size(); ++f) s += (a.y[f] - b.y[f]) * (a.y[f] - b.y[f]);
  for (size_t f = 0; f < a.z.size(); ++f) s += (a.z[f] - b.z[f]) * (a.z[f] - b.z[f]);
  return std::sqrt(s);
}

}  // namespace

TEST_CASE("CF+MCM line-3 midpoint is 2nd-order: closer to a time-refined reference than v=u") {
  // Alg 5 line 3 estimates the flow velocity at the midpoint t+dt/2 (2nd order)
  // rather than freezing v=u (1st order). The paper attributes capturing
  // leapfrogging to this 2nd-order scheme (Fig. 7). Manufactured correctness
  // check WITHOUT an analytic solution: a single full step of the 2nd-order
  // method must land closer to a temporally refined reference (two dt/2 steps of
  // the same 2nd-order method) than the 1st-order freeze does -- because the gap
  // to the reference is the leading O(dt^2) error term that the midpoint removes.
  MacGrid g(24, 24, 24, 2.8 / 24, Vec3{-1.4, -1.4, -1.4});
  const double dt = 0.1;  // large enough that the temporal error is resolvable
  const FaceField seed = ring(g);

  // reinitEvery huge so no map reset perturbs this single-step comparison.
  CfMcmSolver ref(g, seed, 100000, bochner::PoissonMgOptions{}, true, 1.0, /*secondOrder=*/true);
  ref.step(0.5 * dt);
  ref.step(0.5 * dt);

  CfMcmSolver s2(g, seed, 100000, bochner::PoissonMgOptions{}, true, 1.0, /*secondOrder=*/true);
  s2.step(dt);
  CfMcmSolver s1(g, seed, 100000, bochner::PoissonMgOptions{}, true, 1.0, /*secondOrder=*/false);
  s1.step(dt);

  const double e2 = l2diff(s2.velocity(), ref.velocity());
  const double e1 = l2diff(s1.velocity(), ref.velocity());
  CHECK(e1 > 1e-7);   // the two orders genuinely differ (the flow is unsteady)
  CHECK(e2 < e1);     // the midpoint estimate is the more accurate one
}

TEST_CASE("CF+MCM keeps a steady solid-body rotation stationary (rotational dynamics)") {
  // Solid-body rotation v = omega*(-y, x, 0) is a STEADY Euler solution (the
  // centripetal acceleration is a pure gradient absorbed by pressure). A faithful
  // self-advecting covector solver must keep it (nearly) stationary -- the
  // rotation-dominated regime that leapfrogging exercises. Drift here is only
  // discretization error; a transport/projection bug would make it blow up.
  const int n = 32;
  MacGrid g(n, n, n, 2.0 / n, Vec3{-1.0, -1.0, -1.0});
  const double omega = 1.0;
  FaceField rot = ops::zeroFaceField(g);
  for (int i = 0; i <= g.nx(); ++i)
    for (int j = 0; j < g.ny(); ++j)
      for (int k = 0; k < g.nz(); ++k)
        rot.x[g.faceXIndex(i, j, k)] = -omega * g.faceXCenter(i, j, k)[1];
  for (int i = 0; i < g.nx(); ++i)
    for (int j = 0; j <= g.ny(); ++j)
      for (int k = 0; k < g.nz(); ++k)
        rot.y[g.faceYIndex(i, j, k)] = omega * g.faceYCenter(i, j, k)[0];

  CfMcmSolver mcm(g, rot, /*reinitEvery=*/1000, bochner::PoissonMgOptions{}, /*useLimiter=*/false,
                  /*remapTol=*/-1.0, /*secondOrder=*/true);
  const FaceField rot0 = mcm.velocity();  // the projected steady state
  for (int s = 0; s < 20; ++s) mcm.step(0.05);

  double num = 0, den = 0;
  const int lo = n / 4, hi = 3 * n / 4;  // central half, away from the walls
  for (int i = lo; i <= hi; ++i)
    for (int j = lo; j <= hi; ++j)
      for (int k = lo; k <= hi; ++k) {
        const int fx = g.faceXIndex(i, j, k), fy = g.faceYIndex(i, j, k);
        const double a = mcm.velocity().x[fx] - rot0.x[fx];
        const double b = mcm.velocity().y[fy] - rot0.y[fy];
        num += a * a + b * b;
        den += rot0.x[fx] * rot0.x[fx] + rot0.y[fy] * rot0.y[fy];
      }
  CHECK(std::sqrt(num / den) < 0.05);  // <5% drift over 20 steps (~57 deg of rotation)
}

TEST_CASE("CF+MCM keeps the velocity discretely divergence-free") {
  MacGrid g(24, 24, 24, 2.8 / 24, Vec3{-1.4, -1.4, -1.4});
  CfMcmSolver solver(g, ring(g), /*reinitEvery=*/8);
  for (int s = 0; s < 6; ++s) solver.step(0.04);
  CHECK(maxAbsDiv(g, solver.velocity()) < 1e-4);  // geometric-MG projector tolerance
}

TEST_CASE("CF+MCM preserves ring energy at least as well as base BFECC (Alg 1)") {
  MacGrid g(24, 24, 24, 2.8 / 24, Vec3{-1.4, -1.4, -1.4});
  const double dt = 0.04;
  const int steps = 16;

  // CF+MCM (Alg 5).
  CfMcmSolver solver(g, ring(g), /*reinitEvery=*/8);
  const double e0 = energy(solver.velocity());
  for (int s = 0; s < steps; ++s) solver.step(dt);
  const double retMcm = energy(solver.velocity()) / e0;

  // Base method (Alg 1): BFECC advection + the same geometric-MG projection.
  bochner::MacProjector proj(g);
  FaceField u = proj.project(ring(g));
  const double e0b = energy(u);
  for (int s = 0; s < steps; ++s) u = proj.project(bochner::advectCovectorBFECC(g, u, u, dt));
  const double retBfecc = energy(u) / e0b;

  CHECK(retMcm > 0.9);                  // low dissipation
  CHECK(retMcm >= retBfecc - 0.02);     // at least as good as re-advecting u
}

TEST_CASE("CF+MCM projection substep is circulation-neutral (pure gradient)") {
  MacGrid g(24, 24, 24, 2.8 / 24, Vec3{-1.4, -1.4, -1.4});
  CfMcmSolver solver(g, ring(g), /*reinitEvery=*/8);
  const int j = 12, ia = 6, ib = 18, ka = 6, kb = 18;
  for (int s = 0; s < 5; ++s) {
    bochner::CfMcmDiagnostics diag;
    solver.step(0.04, &diag);
    // All four intermediate fields must be captured and correctly sized.
    REQUIRE(diag.afterPullback.x.size() == solver.velocity().x.size());
    REQUIRE(diag.afterProjection.z.size() == solver.velocity().z.size());
    // Projection subtracts grad(phi); around an axis-aligned loop built from
    // stored face values the gradient telescopes to zero, so the discrete
    // circulation is unchanged to round-off. NOTE: the telescoping is a
    // structural identity -- it holds for ANY phi, converged or garbage -- so
    // it only pins the correction's pure-gradient FORM. The divergence check
    // below is what pins the solve itself: a broken pressure solve would leave
    // afterProjection visibly non-solenoidal.
    const double before = discreteCirculationXZ(g, diag.afterLimiter, ia, ib, ka, kb, j);
    const double after = discreteCirculationXZ(g, diag.afterProjection, ia, ib, ka, kb, j);
    CHECK(std::abs(after - before) < 1e-9);
    CHECK(maxAbsDiv(g, diag.afterProjection) < 1e-4);  // geometric-MG projector tolerance
  }
}

TEST_CASE("CF+MCM reinitializes its maps on schedule") {
  MacGrid g(16, 16, 16, 2.8 / 16, Vec3{-1.4, -1.4, -1.4});
  // Disable the accuracy criterion so only the frame count governs this test.
  CfMcmSolver solver(g, ring(g), /*reinitEvery=*/3, bochner::PoissonMgOptions{}, /*useLimiter=*/true,
                     /*remapTol=*/0.0);
  CHECK(solver.sinceReinit() == 0);
  solver.step(0.02);
  solver.step(0.02);
  CHECK(solver.sinceReinit() == 2);
  solver.step(0.02);                 // third step hits the reinit -> counter resets
  CHECK(solver.sinceReinit() == 0);
}

TEST_CASE("CF+MCM map error is zero just after a reinit (Phi o Psi == id)") {
  MacGrid g(16, 16, 16, 2.8 / 16, Vec3{-1.4, -1.4, -1.4});
  CfMcmSolver solver(g, ring(g), /*reinitEvery=*/8);
  // Maps start as identity, so Phi(Psi(x)) = x exactly.
  CHECK(solver.mapError() < 1e-9);
}

TEST_CASE("CF+MCM accuracy criterion (CF 5.3) triggers a reinit before the frame budget") {
  MacGrid g(20, 20, 20, 2.8 / 20, Vec3{-1.4, -1.4, -1.4});
  // A huge frame budget so the ONLY way to reinit is the accuracy criterion,
  // plus a tight tolerance so map drift trips it within a few steps.
  CfMcmSolver solver(g, ring(g), /*reinitEvery=*/100000, bochner::PoissonMgOptions{}, /*useLimiter=*/true,
                     /*remapTol=*/0.25);
  bool reset = false;
  for (int s = 0; s < 60; ++s) {
    solver.step(0.04);
    if (solver.accuracyResets() > 0) {
      reset = true;
      CHECK(solver.sinceReinit() == 0);  // an accuracy reset zeroed the counter
      CHECK(solver.mapError() < 1e-9);   // and the maps are identity again
      break;
    }
  }
  CHECK(reset);
}
