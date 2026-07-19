/// \file
/// Red-green TDD: zero-set tracing of the vortex-filament field on the MAC grid
/// (Weissmann-Pinkall extraction, step 5a).
///
/// Ground truth (project memory): the complex field psi = z + i*(x^2+y^2-R^2)
/// has zero set exactly the circle {z=0, r=R}. Sampling it at cell centers and
/// tracing the phase winding around the cell-center lattice plaquettes must
/// recover crossing points lying on that circle.
#include <doctest.h>

#include <cmath>
#include <vector>

#include "extraction/MacConnectionLaplacian.h"
#include "extraction/MacFilaments.h"
#include "fluid/MacVortexRing.h"
#include "grid/GridOperators.h"
#include "grid/MacGrid.h"
#include "solvers/GaugeEigen.h"
#include "solvers/GaugeMultigrid.h"

using bochner::MacGrid;
using bochner::Vec3;

namespace {

// psi = (z - cz) + i*((x-cx)^2 + (y-cy)^2 - R^2) sampled at cell centers,
// interleaved [Re0, Im0, Re1, Im1, ...].
std::vector<double> analyticRingPsi(const MacGrid& g, Vec3 c, double R) {
  std::vector<double> psi(2 * g.numCells());
  for (int i = 0; i < g.nx(); ++i)
    for (int j = 0; j < g.ny(); ++j)
      for (int k = 0; k < g.nz(); ++k) {
        Vec3 p = g.cellCenter(i, j, k);
        double dx = p[0] - c[0], dy = p[1] - c[1], dz = p[2] - c[2];
        int cell = g.cellIndex(i, j, k);
        psi[2 * cell] = dz;                        // Re
        psi[2 * cell + 1] = dx * dx + dy * dy - R * R;  // Im
      }
  return psi;
}

}  // namespace

TEST_CASE("zero-set tracing recovers an analytic vortex ring circle") {
  MacGrid g(28, 28, 16, 0.1, Vec3{-1.4, -1.4, -0.8});  // domain ~[-1.4,1.4]^2 x [-0.8,0.8]
  const Vec3 center{0.0, 0.0, 0.0};
  const double R = 0.8;
  std::vector<double> psi = analyticRingPsi(g, center, R);

  auto crossings = bochner::traceZeroSet(g, psi);

  REQUIRE(crossings.size() >= 20);  // the circle is well resolved
  const double h = g.spacing();
  int oriented = 0;
  for (const auto& x : crossings) {
    double dz = x.point[2] - center[2];
    double r = std::sqrt((x.point[0] - center[0]) * (x.point[0] - center[0]) +
                         (x.point[1] - center[1]) * (x.point[1] - center[1]));
    CHECK(std::abs(dz) < 1.0 * h);       // lies in the z = 0 plane
    CHECK(std::abs(r - R) < 1.0 * h);    // lies on radius R
    CHECK(std::abs(x.orientation) == 1); // a genuine +/-1 winding

    // The sign, not just |1|: linearizing psi at (R,0,0) gives
    // psi ~ dz + i*2R*(x-R), whose phase winds +1 about +y -- i.e. +1 about
    // the local circle tangent (-y, x, 0)/r, and by rotational symmetry
    // everywhere: orientation = sign(tangent . plaquette normal). A planar
    // z=0 circle never winds around a z-normal plaquette (its four corners
    // share Re = dz != 0 on this grid, confining the phase to a half-plane).
    int axis = bochner::plaquetteNormalAxis(g, x.plaquette);
    CHECK(axis != 2);
    double tanDotN = axis == 0 ? -(x.point[1] - center[1]) / r : (x.point[0] - center[0]) / r;
    if (std::abs(tanDotN) > 0.3) {  // transversal piercing: direction unambiguous
      CHECK(x.orientation == (tanDotN > 0.0 ? 1 : -1));
      ++oriented;
    }
  }
  CHECK(oriented >= 10);  // the signed check actually ran
}

TEST_CASE("crossings link into one closed loop tracing the ring circle") {
  MacGrid g(28, 28, 16, 0.1, Vec3{-1.4, -1.4, -0.8});
  const Vec3 center{0.0, 0.0, 0.0};
  const double R = 0.8;
  auto crossings = bochner::traceZeroSet(g, analyticRingPsi(g, center, R));
  auto filaments = bochner::linkFilaments(g, crossings);

  // The dominant component is the ring: a single closed loop.
  REQUIRE(!filaments.empty());
  const bochner::Filament* ring = nullptr;
  for (const auto& f : filaments)
    if (f.closed && (!ring || f.points.size() > ring->points.size())) ring = &f;
  REQUIRE(ring != nullptr);
  CHECK(ring->closed);

  // Every vertex lies on the circle, and the loop's length matches 2*pi*R.
  double length = 0.0;
  const auto& pts = ring->points;
  for (size_t a = 0; a < pts.size(); ++a) {
    double r = std::sqrt(pts[a][0] * pts[a][0] + pts[a][1] * pts[a][1]);
    CHECK(std::abs(pts[a][2]) < 1.0 * g.spacing());
    CHECK(std::abs(r - R) < 1.0 * g.spacing());
    const Vec3& b = pts[(a + 1) % pts.size()];  // closed: wrap last->first
    length += std::sqrt((pts[a][0] - b[0]) * (pts[a][0] - b[0]) +
                        (pts[a][1] - b[1]) * (pts[a][1] - b[1]) +
                        (pts[a][2] - b[2]) * (pts[a][2] - b[2]));
  }
  CHECK(length == doctest::Approx(2.0 * M_PI * R).epsilon(0.15));
}

TEST_CASE("a field with no phase winding yields no crossings") {
  MacGrid g(10, 10, 10, 0.2);
  // psi = 1 + i*0 everywhere: never encircles the origin.
  std::vector<double> psi(2 * g.numCells(), 0.0);
  for (int c = 0; c < g.numCells(); ++c) psi[2 * c] = 1.0;
  CHECK(bochner::traceZeroSet(g, psi).empty());
}

TEST_CASE("traceZeroSet rejects a wrongly sized psi") {
  MacGrid g(3, 3, 3, 1.0);
  CHECK_THROWS(bochner::traceZeroSet(g, std::vector<double>(2 * g.numCells() - 1, 0.0)));
}

// ---------------------------------------------------------------------------
// The GLOBAL SIGN of the phase convention, pinned absolutely.
//
// Conjugating psi (equivalently flipping every connection angle theta -> -theta)
// gives an operator with an identical spectrum, identical residuals, and an
// identical closed-form torus check -- so every existing test passes under a
// global sign flip. It is not harmless: it reverses the extracted filament
// orientation, hence the direction of every traced vortex.
//
// A relative test ("conjugation flips the sign") cannot catch this, because a
// globally inverted convention flips both branches. The pin has to be absolute,
// and derived from the documented convention rather than from a previous run:
//
//   FilamentCrossing::orientation is the winding of psi's phase around the
//   plaquette corners taken CCW about the +axis of plaquetteNormalAxis().
//
// So psi = (x - x0) + i*(y - y0) -- which has winding EXACTLY +1 about
// (x0, y0) in the xy-plane, by inspection -- must produce orientation +1 on the
// z-normal plaquette containing that zero, and -1 for its conjugate.
// ---------------------------------------------------------------------------
TEST_CASE("traceZeroSet's winding sign matches the documented CCW convention") {
  const MacGrid g(8, 8, 4, 0.25, Vec3{-1.0, -1.0, -0.5});
  // Put the zero line at the center of a cell-center plaquette, parallel to z.
  const double x0 = g.cellCenter(3, 3, 0)[0] + 0.5 * g.spacing();
  const double y0 = g.cellCenter(3, 3, 0)[1] + 0.5 * g.spacing();

  auto build = [&](int sign) {
    std::vector<double> psi(2 * g.numCells());
    for (int i = 0; i < g.nx(); ++i)
      for (int j = 0; j < g.ny(); ++j)
        for (int k = 0; k < g.nz(); ++k) {
          const Vec3 p = g.cellCenter(i, j, k);
          const int c = g.cellIndex(i, j, k);
          psi[2 * c] = p[0] - x0;                     // Re
          psi[2 * c + 1] = sign * (p[1] - y0);        // Im (sign = +1 or conjugate)
        }
    return psi;
  };

  for (const int sign : {+1, -1}) {
    const auto crossings = bochner::traceZeroSet(g, build(sign));
    // The zero set is the line x=x0, y=y0 (parallel to z), so it pierces only
    // z-normal plaquettes.
    int zCount = 0;
    for (const auto& c : crossings) {
      if (bochner::plaquetteNormalAxis(g, c.plaquette) != 2) continue;
      ++zCount;
      INFO("sign=" << sign << " orientation=" << c.orientation << " at (" << c.point[0] << ","
                   << c.point[1] << "," << c.point[2] << ")");
      // psi = (x-x0) + i(y-y0) winds +1 CCW about +z; its conjugate winds -1.
      CHECK(c.orientation == sign);
      // and the zero is where it analytically is
      CHECK(std::abs(c.point[0] - x0) < 0.05 * g.spacing());
      CHECK(std::abs(c.point[1] - y0) < 0.05 * g.spacing());
    }
    INFO("sign=" << sign << " z-normal crossings=" << zCount);
    // One z-normal plaquette per cell-center layer, and the zero line runs
    // through all of them.
    CHECK(zCount == g.nz());
  }
}

// ---------------------------------------------------------------------------
// The full chain: seeded CIRCULATION SIGN -> connection -> eigensolver ->
// extracted crossing orientation. PETSc-free (gauge-MG eigensolver).
//
// This is the flip the previous test cannot see. Sending theta -> -theta
// conjugates the operator: the spectrum, every residual, and the closed-form
// torus check are all invariant, so the entire suite stays green -- while every
// extracted filament reverses direction.
//
// The expected sign is derived, not read off a previous run:
//   * circleCurve(center, +z, R) builds e1 = +x, e2 = z x e1 = +y and traverses
//     cos(t)e1 + sin(t)e2, i.e. CCW about +z.
//   * filamentFaceField puts the vorticity along that tangent, scaled by Gamma.
//     So at the ring's +x side the vorticity points along +y for Gamma > 0.
//   * A plaquette whose normal is along omega has holonomy
//     (1/hbar) * closed integral of u.dl = Gamma/hbar = +2pi when hbar=|Gamma|/2pi,
//     i.e. winding +1 (Stokes).
// Hence: Gamma > 0 must give orientation +1 on y-normal plaquettes at the +x
// side of the ring, and Gamma < 0 must give -1.
// ---------------------------------------------------------------------------
TEST_CASE("seeded circulation sign fixes the extracted filament orientation") {
  const int n = 24;
  const double R = 0.55;
  const MacGrid g(n, n, n, 1.6 / n, Vec3{-0.8, -0.8, -0.8});

  auto orientationsAtPlusX = [&](double Gamma) {
    const double hbar = std::abs(Gamma) / (2.0 * M_PI);
    const auto u = bochner::vortexRingFaceField(g, Vec3{0, 0, 0}, Vec3{0, 0, 1}, R, Gamma, 0.15);
    const auto theta = bochner::connectionAngles(g, u, hbar);
    // Guard the premise: past |theta| = pi the holonomy wraps and the sign
    // question is meaningless.
    REQUIRE(bochner::maxConnectionAngle(theta) < M_PI);
    const auto lat = bochner::gaugeLatticeFromFaces(g, theta);
    bochner::GaugeEigenOptions opts;
    opts.tol = 1e-7;
    const auto res = bochner::smallestEigenpairGaugeMG(lat, nullptr, opts);
    REQUIRE(res.converged);
    const auto crossings = bochner::traceZeroSet(g, bochner::toInterleaved(res.vector));

    std::vector<int> os;
    for (const auto& c : crossings) {
      if (bochner::plaquetteNormalAxis(g, c.plaquette) != 1) continue;  // y-normal
      // the +x side of the ring, near the z=0 and y=0 planes
      if (c.point[0] < 0.5 * R) continue;
      if (std::abs(c.point[1]) > 1.5 * g.spacing()) continue;
      if (std::abs(c.point[2]) > 1.5 * g.spacing()) continue;
      os.push_back(c.orientation);
    }
    return os;
  };

  const auto pos = orientationsAtPlusX(+1.0);
  const auto neg = orientationsAtPlusX(-1.0);
  INFO("+Gamma crossings=" << pos.size() << "  -Gamma crossings=" << neg.size());
  REQUIRE(!pos.empty());
  REQUIRE(!neg.empty());
  for (const int o : pos) CHECK(o == +1);
  for (const int o : neg) CHECK(o == -1);
}
