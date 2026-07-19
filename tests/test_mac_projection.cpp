/// \file
/// Red-green TDD: behaviour spec for MAC pressure projection on the grid.
///
/// Step 2 of the grid rebuild. The projection solves the
/// pressure-Poisson system `A phi = -div(u)` (A = -(div.grad), the SPD
/// homogeneous-Neumann Laplacian with one cell pinned to fix the constant null
/// space) and returns `u - grad(phi)`, which is discretely divergence-free.
///
/// The point of dual/flux placement: a prescribed boundary normal velocity is
/// just a boundary-face value of `u`, so it enters `div(u)` natively -- no
/// source-term hack (this is what supersedes the old #19/#22 workaround).
#include <doctest.h>

#include <cmath>
#include <vector>

#include "grid/GridOperators.h"
#include "grid/MacGrid.h"
#include "fluid/MacProjection.h"

using bochner::FaceField;
using bochner::MacGrid;
namespace ops = bochner::ops;

namespace {

double maxAbsDivergence(const MacGrid& g, const FaceField& u) {
  auto div = ops::divergence(g, u);
  double m = 0.0;
  for (double d : div) m = std::max(m, std::abs(d));
  return m;
}

double wobble(int n) { return std::sin(0.9 * n + 0.4) + 0.5 * std::cos(1.7 * n); }

}  // namespace

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

TEST_CASE("projection removes wall-normal flux (no-penetration) and is idempotent") {
  MacGrid g(5, 6, 4, 0.3);
  // Shear u_x = sin(y): divergence-free in the interior but with nonzero normal
  // component sin(y) at the x-walls -- i.e. through-wall flow. A correct
  // no-penetration projection (inhomogeneous Neumann, dphi/dn = u.n) must remove
  // the wall-normal flux as well as any interior divergence.
  FaceField u = ops::zeroFaceField(g);
  for (int i = 0; i <= g.nx(); ++i)
    for (int j = 0; j < g.ny(); ++j)
      for (int k = 0; k < g.nz(); ++k)
        u.x[g.faceXIndex(i, j, k)] = std::sin(g.faceXCenter(i, j, k)[1]);
  REQUIRE(maxWallFlux(g, u) > 0.1);  // genuinely penetrating the walls to start

  FaceField p = bochner::projectToDivergenceFree(g, u);
  CHECK(maxAbsDivergence(g, p) < 1e-6);        // divergence-free interior
  CHECK(maxWallFlux(g, p) == doctest::Approx(0.0));  // AND u.n = 0 on the walls

  // The projection is idempotent: a divergence-free no-penetration field is a
  // fixed point, so re-projecting leaves it unchanged.
  FaceField pp = bochner::projectToDivergenceFree(g, p);
  for (size_t f = 0; f < p.x.size(); ++f) CHECK(pp.x[f] == doctest::Approx(p.x[f]));
  for (size_t f = 0; f < p.y.size(); ++f) CHECK(pp.y[f] == doctest::Approx(p.y[f]));
  for (size_t f = 0; f < p.z.size(); ++f) CHECK(pp.z[f] == doctest::Approx(p.z[f]));
}

TEST_CASE("projection annihilates a pure gradient field") {
  MacGrid g(4, 5, 4, 0.25);
  std::vector<double> phi0(g.numCells());
  for (int c = 0; c < g.numCells(); ++c) phi0[c] = wobble(c);

  FaceField u = ops::gradient(g, phi0);  // a pure gradient (curl-free)
  FaceField p = bochner::projectToDivergenceFree(g, u);

  for (double v : p.x) CHECK(v == doctest::Approx(0.0).epsilon(1e-6));
  for (double v : p.y) CHECK(v == doctest::Approx(0.0).epsilon(1e-6));
  for (double v : p.z) CHECK(v == doctest::Approx(0.0).epsilon(1e-6));
}

TEST_CASE("projection makes an arbitrary no-penetration field divergence-free") {
  MacGrid g(6, 5, 4, 0.2);
  FaceField u = ops::zeroFaceField(g);
  // Random interior fluxes; boundary faces stay 0 (no-penetration walls), which
  // keeps the all-Neumann Poisson system compatible.
  for (int i = 1; i < g.nx(); ++i)
    for (int j = 0; j < g.ny(); ++j)
      for (int k = 0; k < g.nz(); ++k) u.x[g.faceXIndex(i, j, k)] = wobble(i * 13 + j * 5 + k * 2);
  for (int i = 0; i < g.nx(); ++i)
    for (int j = 1; j < g.ny(); ++j)
      for (int k = 0; k < g.nz(); ++k) u.y[g.faceYIndex(i, j, k)] = wobble(i * 3 + j * 17 + k * 7);
  for (int i = 0; i < g.nx(); ++i)
    for (int j = 0; j < g.ny(); ++j)
      for (int k = 1; k < g.nz(); ++k) u.z[g.faceZIndex(i, j, k)] = wobble(i * 11 + j * 4 + k * 19);

  REQUIRE(maxAbsDivergence(g, u) > 0.1);  // genuinely compressible to start
  FaceField p = bochner::projectToDivergenceFree(g, u);
  CHECK(maxAbsDivergence(g, p) < 1e-7);

  // Boundary faces are BC-prescribed: the projection must not touch them.
  for (int j = 0; j < g.ny(); ++j)
    for (int k = 0; k < g.nz(); ++k) {
      CHECK(p.x[g.faceXIndex(0, j, k)] == doctest::Approx(0.0));
      CHECK(p.x[g.faceXIndex(g.nx(), j, k)] == doctest::Approx(0.0));
    }
}

TEST_CASE("open (Dirichlet p=0) MacProjector annihilates a free-surface gradient field") {
  // With every wall open, project(gradBC(phi)) must be ~0: the Dirichlet Poisson
  // recovers phi exactly (SPD, unique), so u - gradBC(phi) cancels. This pins the
  // whole open path -- BC Laplacian assembly, Jacobi-CG solve, and gradientBC.
  MacGrid g(6, 7, 5, 0.2);
  const auto bc = bochner::BoundarySpec::allOpen();
  std::vector<double> phi0(g.numCells());
  for (int c = 0; c < g.numCells(); ++c) phi0[c] = wobble(c);
  FaceField u = ops::gradient(g, phi0, bc);  // a free-surface (Dirichlet) gradient

  bochner::MacProjector projector(g, {}, bc);
  const FaceField p = projector.project(u);
  for (double v : p.x) CHECK(v == doctest::Approx(0.0).epsilon(1e-5));
  for (double v : p.y) CHECK(v == doctest::Approx(0.0).epsilon(1e-5));
  for (double v : p.z) CHECK(v == doctest::Approx(0.0).epsilon(1e-5));
}

TEST_CASE("open MacProjector is divergence-free WITH outflow (not no-penetration)") {
  // A net-divergent field (sources): interior x-flux grows along i, so div > 0
  // everywhere. A closed box could not absorb it, but open walls let it leave:
  // the projected field is divergence-free AND carries nonzero wall-normal flux.
  MacGrid g(8, 8, 8, 0.2);
  FaceField u = ops::zeroFaceField(g);
  for (int i = 1; i < g.nx(); ++i)
    for (int j = 0; j < g.ny(); ++j)
      for (int k = 0; k < g.nz(); ++k) u.x[g.faceXIndex(i, j, k)] = 0.1 * i;
  REQUIRE(maxAbsDivergence(g, u) > 0.1);

  bochner::MacProjector projector(g, {}, bochner::BoundarySpec::allOpen());
  const FaceField p = projector.project(u);
  CHECK(maxAbsDivergence(g, p) < 1e-6);   // divergence-free interior
  CHECK(maxWallFlux(g, p) > 1e-3);        // outflow allowed -- the open BC at work

  // Idempotent: a divergence-free free-surface field is a fixed point.
  const FaceField pp = projector.project(p);
  double maxDiff = 0.0;
  for (std::size_t f = 0; f < p.x.size(); ++f) maxDiff = std::max(maxDiff, std::abs(pp.x[f] - p.x[f]));
  for (std::size_t f = 0; f < p.z.size(); ++f) maxDiff = std::max(maxDiff, std::abs(pp.z[f] - p.z[f]));
  CHECK(maxDiff < 1e-4);
}

TEST_CASE("geometric-MG MacProjector is divergence-free and warm-starts") {
  MacGrid g(8, 8, 8, 0.2);
  FaceField u = ops::zeroFaceField(g);
  for (int i = 1; i < g.nx(); ++i)
    for (int j = 0; j < g.ny(); ++j)
      for (int k = 0; k < g.nz(); ++k) u.x[g.faceXIndex(i, j, k)] = wobble(i * 13 + j * 5 + k * 2);
  for (int i = 0; i < g.nx(); ++i)
    for (int j = 1; j < g.ny(); ++j)
      for (int k = 0; k < g.nz(); ++k) u.y[g.faceYIndex(i, j, k)] = wobble(i * 3 + j * 17 + k * 7);

  bochner::PoissonMgOptions opts;
  opts.tol = 1e-8;
  bochner::MacProjector projector(g, opts);

  REQUIRE(maxAbsDivergence(g, u) > 0.1);  // genuinely compressible to start
  int coldCycles = 0;
  const FaceField p = projector.project(u, &coldCycles);
  CHECK(maxAbsDivergence(g, p) < 1e-6);  // divergence-free

  // The result matches the stateless (pinned PETSc) projection up to the solve
  // tolerance -- grad(phi) is independent of phi's constant offset.
  const FaceField ref = bochner::projectToDivergenceFree(g, u);
  double maxDiff = 0.0;
  for (std::size_t f = 0; f < p.x.size(); ++f) maxDiff = std::max(maxDiff, std::abs(p.x[f] - ref.x[f]));
  for (std::size_t f = 0; f < p.y.size(); ++f) maxDiff = std::max(maxDiff, std::abs(p.y[f] - ref.y[f]));
  CHECK(maxDiff < 1e-4);

  // Re-projecting the same field warm-starts from the now-converged pressure, so
  // it converges in strictly fewer cycles than the cold solve (temporal coherence).
  int warmCycles = 0;
  projector.project(u, &warmCycles);
  CHECK(warmCycles < coldCycles);
}
