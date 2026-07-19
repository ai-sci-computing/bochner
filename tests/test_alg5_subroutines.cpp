/// \file
/// Per-subroutine correctness invariants for the Covector Fluids CF+MCM pipeline
/// (Algorithm 5). Each case pins the DEFINING property of one subroutine on a
/// manufactured/analytic case -- not just that it runs, but that its output is
/// mathematically what the paper requires. Complements the isolated-op tests in
/// test_grid_operators / test_mac_advection / test_mac_projection by checking the
/// guarantees on the ACTUAL Alg-5 building blocks (the geometric-MG MacProjector
/// and the CfMcmSolver substeps).
///
/// The headline invariant the user asked for: after the pressure projection there
/// is NO interior divergence AND NO wall-normal velocity left (closed box).
#include <doctest.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "grid/GridOperators.h"
#include "fluid/MacAdvection.h"
#include "fluid/MacCharacteristicMap.h"
#include "grid/MacGrid.h"
#include "fluid/MacProjection.h"
#include "fluid/MacVortexRing.h"
#include "grid/Vec3.h"

using namespace bochner;
namespace ops = bochner::ops;

namespace {

// Max |div u| over interior cells (the discrete MAC divergence).
double maxAbsDivergence(const MacGrid& g, const FaceField& u) {
  const auto div = ops::divergence(g, u);
  double m = 0.0;
  for (double d : div) m = std::max(m, std::abs(d));
  return m;
}

// Max |u.n| over all six domain walls (wall-normal / penetrating flux).
double maxWallFlux(const MacGrid& g, const FaceField& u) {
  double m = 0.0;
  for (int j = 0; j < g.ny(); ++j)
    for (int k = 0; k < g.nz(); ++k) {
      m = std::max(m, std::abs(u.x[g.faceXIndex(0, j, k)]));
      m = std::max(m, std::abs(u.x[g.faceXIndex(g.nx(), j, k)]));
    }
  for (int i = 0; i < g.nx(); ++i)
    for (int k = 0; k < g.nz(); ++k) {
      m = std::max(m, std::abs(u.y[g.faceYIndex(i, 0, k)]));
      m = std::max(m, std::abs(u.y[g.faceYIndex(i, g.ny(), k)]));
    }
  for (int i = 0; i < g.nx(); ++i)
    for (int j = 0; j < g.ny(); ++j) {
      m = std::max(m, std::abs(u.z[g.faceZIndex(i, j, 0)]));
      m = std::max(m, std::abs(u.z[g.faceZIndex(i, j, g.nz())]));
    }
  return m;
}

double wobble(int n) { return std::sin(0.9 * n + 0.4) + 0.5 * std::cos(1.7 * n); }

FaceField centeredRing(const MacGrid& g) {
  return vortexRingFaceField(g, Vec3{0, 0, 0}, Vec3{0, 0, 1}, /*R=*/0.6, /*Gamma=*/1.0, /*core=*/0.12);
}

// Circulation Gamma = closed-loop integral of u.dl around a polyline loop, via
// midpoint-rule sampling of the reconstructed velocity (Kelvin's quantity).
double circulation(const MacGrid& g, const FaceField& u, const std::vector<Vec3>& loop) {
  double gamma = 0.0;
  const std::size_t N = loop.size();
  for (std::size_t a = 0; a < N; ++a) {
    const Vec3 p0 = loop[a], p1 = loop[(a + 1) % N];
    const Vec3 seg = vsub(p1, p0);
    const Vec3 w = sampleVelocity(g, u, vscale(vadd(p0, p1), 0.5));
    gamma += w[0] * seg[0] + w[1] * seg[1] + w[2] * seg[2];
  }
  return gamma;
}

// A material loop advected one step forward by the frozen flow v (each Lagrangian
// marker moves with the flow; forward advection = a backtrace over -dt).
std::vector<Vec3> advectLoop(const MacGrid& g, const FaceField& v, const std::vector<Vec3>& loop,
                             double dt) {
  std::vector<Vec3> out(loop.size());
  for (std::size_t a = 0; a < loop.size(); ++a) out[a] = backtrace(g, v, loop[a], -dt);
  return out;
}

}  // namespace

// --- Subroutine: pressure projection P(u), the geometric-MG MacProjector that
//     Alg 5 line 10 actually calls (closed no-penetration box). ---

TEST_CASE("MacProjector (closed): projected field has NO divergence AND NO wall-normal velocity") {
  MacGrid g(6, 7, 5, 0.3, Vec3{-0.9, -1.05, -0.75});
  // A shear u_x = sin(y): divergence-free in the interior but penetrating the
  // x-walls. A correct inhomogeneous-Neumann projection (dphi/dn = u.n) must
  // remove BOTH the interior divergence AND the wall-normal flux.
  FaceField u = ops::zeroFaceField(g);
  for (int i = 0; i <= g.nx(); ++i)
    for (int j = 0; j < g.ny(); ++j)
      for (int k = 0; k < g.nz(); ++k)
        u.x[g.faceXIndex(i, j, k)] = std::sin(g.faceXCenter(i, j, k)[1]);
  REQUIRE(maxWallFlux(g, u) > 0.1);  // genuinely penetrating to start

  PoissonMgOptions opts;
  opts.tol = 1e-9;              // drive the MG solve down so divergence -> ~0
  MacProjector projector(g, opts);  // closed box, geometric-MG
  const FaceField p = projector.project(u);
  CHECK(maxAbsDivergence(g, p) < 1e-6);  // no interior divergence
  CHECK(maxWallFlux(g, p) < 1e-6);       // no wall-normal velocity
}

TEST_CASE("MacProjector (closed): a divergence-free no-penetration field is a fixed point") {
  MacGrid g(6, 6, 6, 0.25, Vec3{-0.75, -0.75, -0.75});
  MacProjector projector(g);
  // Project an arbitrary compressible field to reach the constraint manifold,
  // then re-project: the result must not move (idempotence).
  FaceField u = ops::zeroFaceField(g);
  for (int i = 1; i < g.nx(); ++i)
    for (int j = 0; j < g.ny(); ++j)
      for (int k = 0; k < g.nz(); ++k) u.x[g.faceXIndex(i, j, k)] = wobble(i * 13 + j * 5 + k * 2);
  const FaceField p = projector.project(u);
  const FaceField pp = projector.project(p);
  double maxDiff = 0.0;
  for (std::size_t f = 0; f < p.x.size(); ++f) maxDiff = std::max(maxDiff, std::abs(pp.x[f] - p.x[f]));
  for (std::size_t f = 0; f < p.y.size(); ++f) maxDiff = std::max(maxDiff, std::abs(pp.y[f] - p.y[f]));
  for (std::size_t f = 0; f < p.z.size(); ++f) maxDiff = std::max(maxDiff, std::abs(pp.z[f] - p.z[f]));
  CHECK(maxDiff < 1e-5);
}

TEST_CASE("MacProjector (closed): annihilates a pure interior gradient field") {
  MacGrid g(5, 5, 5, 0.25, Vec3{-0.625, -0.625, -0.625});
  std::vector<double> phi(g.numCells());
  for (int c = 0; c < g.numCells(); ++c) phi[c] = wobble(c);
  const FaceField grad = ops::gradient(g, phi);  // curl-free, zero on the walls
  const FaceField p = MacProjector(g).project(grad);
  double m = 0.0;
  for (double v : p.x) m = std::max(m, std::abs(v));
  for (double v : p.y) m = std::max(m, std::abs(v));
  for (double v : p.z) m = std::max(m, std::abs(v));
  CHECK(m < 1e-5);
}

// --- Subroutine: enforceNoPenetration (the wall BC helper). ---

TEST_CASE("enforceNoPenetration zeroes closed wall-normal faces and leaves the interior") {
  MacGrid g(4, 4, 4, 0.25);
  FaceField u = ops::zeroFaceField(g);
  for (double& v : u.x) v = 1.0;  // every x-face (walls + interior) set to 1
  const double interiorBefore = u.x[g.faceXIndex(2, 1, 1)];
  ops::enforceNoPenetration(g, u);
  for (int j = 0; j < g.ny(); ++j)
    for (int k = 0; k < g.nz(); ++k) {
      CHECK(u.x[g.faceXIndex(0, j, k)] == doctest::Approx(0.0));       // xlo wall zeroed
      CHECK(u.x[g.faceXIndex(g.nx(), j, k)] == doctest::Approx(0.0));  // xhi wall zeroed
    }
  CHECK(u.x[g.faceXIndex(2, 1, 1)] == doctest::Approx(interiorBefore));  // interior untouched
}

// --- Subroutine: the covector pullback Jacobian dM^T (Alg 3 / Eq 39-40),
//     exercised through pullbackThroughMap on an affine map. ---

TEST_CASE("pullbackThroughMap applies the transpose-Jacobian of an affine map (Eq 39-40)") {
  MacGrid g(6, 6, 6, 0.2, Vec3{-0.6, -0.6, -0.6});
  // Affine shear map M(x) = (x + s*y, y, z): dM/dy = (s, 1, 0), so pulling back a
  // CONSTANT source c mixes c_x into the y-component: (dM^T c).y = s*c_x + c_y.
  const double s = 0.37;
  const Vec3 c{1.3, -0.7, 0.4};
  std::vector<Vec3> M(g.numCells());
  for (int i = 0; i < g.nx(); ++i)
    for (int j = 0; j < g.ny(); ++j)
      for (int k = 0; k < g.nz(); ++k) {
        const Vec3 x = g.cellCenter(i, j, k);
        M[g.cellIndex(i, j, k)] = Vec3{x[0] + s * x[1], x[1], x[2]};
      }
  FaceField src = ops::zeroFaceField(g);
  for (double& v : src.x) v = c[0];
  for (double& v : src.y) v = c[1];
  for (double& v : src.z) v = c[2];

  const FaceField out = pullbackThroughMap(g, M, src);
  // Interior y-faces: (dM^T c).y = s*c_x + c_y.
  for (int i = 0; i < g.nx(); ++i)
    for (int j = 1; j < g.ny(); ++j)
      for (int k = 0; k < g.nz(); ++k)
        CHECK(out.y[g.faceYIndex(i, j, k)] == doctest::Approx(s * c[0] + c[1]).epsilon(1e-9));
  // Interior x-faces: dM/dx = (1,0,0) -> (dM^T c).x = c_x (unchanged).
  for (int i = 1; i < g.nx(); ++i)
    for (int j = 0; j < g.ny(); ++j)
      for (int k = 0; k < g.nz(); ++k)
        CHECK(out.x[g.faceXIndex(i, j, k)] == doctest::Approx(c[0]).epsilon(1e-9));
}

// --- Subroutine: BFECC covector advection (Alg 4). Its error-correction term
//     must VANISH wherever the semi-Lagrangian step is already exact. ---

TEST_CASE("advectCovectorBFECC == semi-Lagrangian under uniform flow (roundtrip vanishes)") {
  MacGrid g(12, 12, 12, 0.16, Vec3{-0.96, -0.96, -0.96});
  // A LINEAR field u_x = 0.5 + 0.8*x, transported by a uniform +x flow. Trilinear
  // interpolation reproduces a linear field exactly, and the backtrace/Jacobian
  // are exact for uniform flow, so the sL step is EXACT, its roundtrip is exact,
  // and the BFECC error-correction term VANISHES -> BFECC must equal sL. (A curved
  // field would fail this: interpolation error, not the map, is what BFECC fights.)
  // Flow and variation are along x only, so the wall clamp -- which would break
  // linearity -- is never triggered in the deep interior we check.
  FaceField u = ops::zeroFaceField(g);
  for (int i = 0; i <= g.nx(); ++i)
    for (int j = 0; j < g.ny(); ++j)
      for (int k = 0; k < g.nz(); ++k)
        u.x[g.faceXIndex(i, j, k)] = 0.5 + 0.8 * g.faceXCenter(i, j, k)[0];
  FaceField v = ops::zeroFaceField(g);
  for (double& val : v.x) val = 0.3;  // shift 0.3*0.15 = 0.045 < h = 0.16 (< 1 cell)

  const double dt = 0.15;
  const FaceField sl = advectCovectorSL(g, u, v, dt);
  const FaceField bfecc = advectCovectorBFECC(g, u, v, dt);
  double m = 0.0;
  for (int i = 4; i <= g.nx() - 4; ++i)  // deep interior: >3 cells from the x-walls
    for (int j = 0; j < g.ny(); ++j)
      for (int k = 0; k < g.nz(); ++k)
        m = std::max(m, std::abs(bfecc.x[g.faceXIndex(i, j, k)] - sl.x[g.faceXIndex(i, j, k)]));
  CHECK(m < 1e-9);
}

// --- Circulation preservation (Kelvin), the covector scheme's defining
//     property: one covector-advection step preserves the loop integral of u
//     around a MATERIAL loop that moves with the flow. ---

TEST_CASE("covector advection preserves circulation around a material loop (uniform flow)") {
  MacGrid g(24, 24, 24, 0.1, Vec3{-1.2, -1.2, -1.2});
  // Solid-body rotation about z (a LINEAR field: u = (-Om*y, Om*x, 0)), so the
  // reconstruction is exact and the only thing under test is that the pullback +
  // material-loop transport keep Gamma. Under a uniform translation the field and
  // the loop shift together, so Kelvin must hold to ~machine precision.
  const double Om = 0.7;
  FaceField u = ops::zeroFaceField(g);
  for (int i = 0; i <= g.nx(); ++i)
    for (int j = 0; j < g.ny(); ++j)
      for (int k = 0; k < g.nz(); ++k)
        u.x[g.faceXIndex(i, j, k)] = -Om * g.faceXCenter(i, j, k)[1];
  for (int i = 0; i < g.nx(); ++i)
    for (int j = 0; j <= g.ny(); ++j)
      for (int k = 0; k < g.nz(); ++k)
        u.y[g.faceYIndex(i, j, k)] = Om * g.faceYCenter(i, j, k)[0];

  // A circular material loop in the z=0 plane, radius 0.4, centered at the origin.
  std::vector<Vec3> loop;
  const int N = 240;
  for (int a = 0; a < N; ++a) {
    const double th = 2.0 * M_PI * a / N;
    loop.push_back(Vec3{0.4 * std::cos(th), 0.4 * std::sin(th), 0.0});
  }
  const double gammaBefore = circulation(g, u, loop);
  REQUIRE(std::abs(gammaBefore) > 0.1);  // genuinely circulating (~2*Om*pi*0.4^2)

  FaceField vflow = ops::zeroFaceField(g);
  for (double& val : vflow.x) val = 0.25;  // uniform +x flow
  const double dt = 0.2;                    // shift 0.05 < h = 0.1
  const FaceField uAdv = advectCovectorSL(g, u, vflow, dt);
  const std::vector<Vec3> loopAdv = advectLoop(g, vflow, loop, dt);
  const double gammaAfter = circulation(g, uAdv, loopAdv);

  CHECK(gammaAfter == doctest::Approx(gammaBefore).epsilon(1e-6));
}

TEST_CASE("covector advection preserves circulation under a shear (area-preserving deformation)") {
  MacGrid g(24, 24, 24, 0.1, Vec3{-1.2, -1.2, -1.2});
  // Same rotation field, but the flow is now a LINEAR SHEAR v = (S*y, 0, 0). The
  // material loop deforms (circle -> sheared ellipse) yet -- shear being
  // incompressible -- its area, and thus the circulation of the (shear-invariant)
  // rotation field around it, is unchanged. Exercises a non-identity Jacobian dPsi.
  const double Om = 0.7;
  FaceField u = ops::zeroFaceField(g);
  for (int i = 0; i <= g.nx(); ++i)
    for (int j = 0; j < g.ny(); ++j)
      for (int k = 0; k < g.nz(); ++k)
        u.x[g.faceXIndex(i, j, k)] = -Om * g.faceXCenter(i, j, k)[1];
  for (int i = 0; i < g.nx(); ++i)
    for (int j = 0; j <= g.ny(); ++j)
      for (int k = 0; k < g.nz(); ++k)
        u.y[g.faceYIndex(i, j, k)] = Om * g.faceYCenter(i, j, k)[0];

  std::vector<Vec3> loop;
  const int N = 240;
  for (int a = 0; a < N; ++a) {
    const double th = 2.0 * M_PI * a / N;
    loop.push_back(Vec3{0.35 * std::cos(th), 0.35 * std::sin(th), 0.0});
  }
  const double gammaBefore = circulation(g, u, loop);
  REQUIRE(std::abs(gammaBefore) > 0.1);

  const double S = 0.4;
  FaceField vflow = ops::zeroFaceField(g);
  for (int i = 0; i <= g.nx(); ++i)
    for (int j = 0; j < g.ny(); ++j)
      for (int k = 0; k < g.nz(); ++k) vflow.x[g.faceXIndex(i, j, k)] = S * g.faceXCenter(i, j, k)[1];
  const double dt = 0.15;
  const FaceField uAdv = advectCovectorSL(g, u, vflow, dt);
  const std::vector<Vec3> loopAdv = advectLoop(g, vflow, loop, dt);
  const double gammaAfter = circulation(g, uAdv, loopAdv);

  CHECK(gammaAfter == doctest::Approx(gammaBefore).epsilon(1e-4));
}

// --- Full Algorithm-5 step: the committed velocity AND the per-substep
//     projection output both satisfy the projection invariant. ---

TEST_CASE("CF+MCM step output is divergence-free AND no-penetration (closed box)") {
  MacGrid g(16, 16, 16, 1.6 / 16, Vec3{-0.8, -0.8, -0.8});
  CfMcmSolver solver(g, centeredRing(g), /*reinitEvery=*/4);
  for (int s = 0; s < 6; ++s) solver.step(0.02);
  CHECK(maxAbsDivergence(g, solver.velocity()) < 1e-5);
  CHECK(maxWallFlux(g, solver.velocity()) < 1e-6);
}

TEST_CASE("CF+MCM diag.afterProjection satisfies the projection invariant every substep") {
  MacGrid g(16, 16, 16, 1.6 / 16, Vec3{-0.8, -0.8, -0.8});
  CfMcmSolver solver(g, centeredRing(g), /*reinitEvery=*/4);
  for (int s = 0; s < 5; ++s) {
    CfMcmDiagnostics diag;
    solver.step(0.02, &diag);
    // Line 10's output (captured pre-reinit) is the projection's guarantee:
    CHECK(maxAbsDivergence(g, diag.afterProjection) < 1e-5);
    CHECK(maxWallFlux(g, diag.afterProjection) < 1e-6);
    // afterPullback / afterCorrection / afterLimiter are populated and sized.
    CHECK(diag.afterPullback.x.size() == solver.velocity().x.size());
    CHECK(diag.afterCorrection.y.size() == solver.velocity().y.size());
  }
}
