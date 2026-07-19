/// \file
/// Red-green TDD: the end-to-end Weissmann-Pinkall pipeline on the MAC grid.
///
/// Seed a vortex ring's velocity field, turn it into a U(1) connection
/// (theta = u*h/hbar, hbar = Gamma/2*pi so the ring carries one flux quantum),
/// build the connection Laplacian, solve for its smallest-eigenvalue section
/// psi (SLEPc), and trace+link its zero set. The extracted filament must be a
/// closed loop tracing the seeded ring circle -- the closed sim<->extractor
/// loop, now back on the grid.
#include <doctest.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <vector>

#include "solvers/EigenSolver.h"
#include "extraction/MacConnectionLaplacian.h"
#include "extraction/MacFilaments.h"
#include "grid/MacGrid.h"
#include "grid/Vec3.h"
#include "fluid/MacVortexRing.h"

using bochner::Filament;
using bochner::MacGrid;
using bochner::Vec3;

TEST_CASE("seeded vortex ring is extracted as a closed filament loop") {
  // Flat box around z=0; the ring lies in that plane.
  MacGrid g(28, 28, 16, 0.1, Vec3{-1.4, -1.4, -0.8});
  const Vec3 center{0.0, 0.0, 0.0};
  const double R = 0.7;
  const double Gamma = 1.0;
  const double hbar = Gamma / (2.0 * M_PI);  // one flux quantum around the core

  bochner::FaceField u =
      bochner::vortexRingFaceField(g, center, Vec3{0, 0, 1}, R, Gamma, /*coreRadius=*/0.15);

  bochner::FaceField theta = bochner::connectionAngles(g, u, hbar);
  auto E = bochner::connectionLaplacian(g, theta);
  auto psi = bochner::smallestEigenpair(E).vector;

  auto crossings = bochner::traceZeroSet(g, psi);
  auto filaments = bochner::linkFilaments(g, crossings);

  // Pick the longest closed loop -- the extracted ring.
  REQUIRE(!filaments.empty());
  const Filament* ring = nullptr;
  for (const auto& f : filaments)
    if (f.closed && (!ring || f.points.size() > ring->points.size())) ring = &f;
  REQUIRE(ring != nullptr);

  // Its vertices trace a circle in the z=0 plane. The nodal circle of the
  // lowest mode sits at the flux core, slightly inside the nominal ring radius
  // R (finite coreRadius) -- so we check it is a tight planar circle of some
  // radius meanR near R, rather than pinning meanR to R exactly.
  const auto& p = ring->points;
  double meanR = 0.0, maxAbsZ = 0.0, length = 0.0;
  for (size_t a = 0; a < p.size(); ++a) {
    meanR += std::sqrt(p[a][0] * p[a][0] + p[a][1] * p[a][1]);
    maxAbsZ = std::max(maxAbsZ, std::abs(p[a][2]));
    const Vec3& b = p[(a + 1) % p.size()];
    length += std::sqrt((p[a][0] - b[0]) * (p[a][0] - b[0]) + (p[a][1] - b[1]) * (p[a][1] - b[1]) +
                        (p[a][2] - b[2]) * (p[a][2] - b[2]));
  }
  meanR /= p.size();
  double maxRadialDev = 0.0;
  for (const auto& q : p)
    maxRadialDev = std::max(maxRadialDev, std::abs(std::sqrt(q[0] * q[0] + q[1] * q[1]) - meanR));

  INFO("points=" << p.size() << " meanR=" << meanR << " maxAbsZ=" << maxAbsZ
                 << " maxRadialDev=" << maxRadialDev << " length=" << length);
  CHECK(maxAbsZ < 2.0 * g.spacing());                          // planar (z = 0)
  CHECK(maxRadialDev < 1.5 * g.spacing());                     // a tight circle
  CHECK(meanR == doctest::Approx(R).epsilon(0.25));            // near the ring radius
  CHECK(length == doctest::Approx(2.0 * M_PI * meanR).epsilon(0.1));  // circumference of that circle

  // Pin the transport-sign (conjugation) convention, which nothing geometric
  // can see: theta -> -theta leaves the zero set -- and every check above --
  // unchanged and only flips the crossing orientations. The seed circle runs
  // counterclockwise about +z (circleCurve: e1 = x, e2 = axis x e1 = y, angle
  // increasing), and Biot-Savart makes that the vorticity direction, so the
  // filament tangent at (x,y,0) is (-y, x, 0)/r. With theta = u*h/hbar and
  // E_ab = -w e^{-i theta} (transport a->b multiplies by e^{+i theta}), the
  // ground state's phase increases along u; around the core, the loop
  // right-handed about the vorticity carries circulation +Gamma = 2*pi*hbar,
  // so the winding about a plaquette is +1 exactly when the filament pierces
  // it along the plaquette's +axis normal: orientation = sign(tangent . normal).
  int oriented = 0, misoriented = 0;
  for (const auto& c : crossings) {
    double r = std::sqrt(c.point[0] * c.point[0] + c.point[1] * c.point[1]);
    if (std::abs(c.point[2]) > 2.0 * g.spacing() || std::abs(r - R) > 0.25 * R) continue;
    int axis = bochner::plaquetteNormalAxis(g, c.plaquette);
    if (axis == 2) continue;  // planar ring: no z-normal flux
    double tanDotN = axis == 0 ? -c.point[1] / r : c.point[0] / r;
    if (std::abs(tanDotN) < 0.5) continue;  // near-tangential: direction ambiguous
    ++oriented;
    if (c.orientation != (tanDotN > 0.0 ? 1 : -1)) ++misoriented;
  }
  INFO("oriented=" << oriented << " misoriented=" << misoriented);
  REQUIRE(oriented >= 10);  // the convention check actually ran
  CHECK(misoriented == 0);
}

// Run the pipeline on a seeded field and return the extracted closed loops
// (>= 4 points), longest first, so the multi-filament seeds (leapfrog / Hopf
// link / trefoil) can be checked end-to-end -- including their topology.
static std::vector<Filament> extractClosedLoops(const MacGrid& g, const bochner::FaceField& u,
                                                double hbar, const char* what) {
  bochner::FaceField theta = bochner::connectionAngles(g, u, hbar);
  auto E = bochner::connectionLaplacian(g, theta);
  auto psi = bochner::smallestEigenpair(E).vector;
  auto crossings = bochner::traceZeroSet(g, psi);
  auto filaments = bochner::linkFilaments(g, crossings);
  std::vector<Filament> closed;
  for (auto& f : filaments)
    if (f.closed && f.points.size() >= 4) closed.push_back(std::move(f));
  std::sort(closed.begin(), closed.end(),
            [](const Filament& a, const Filament& b) { return a.points.size() > b.points.size(); });
  INFO(what << ": filaments=" << filaments.size() << " closedLoops>=4pts=" << closed.size());
  return closed;
}

static int countClosedLoops(const MacGrid& g, const bochner::FaceField& u, double hbar,
                            const char* what) {
  return static_cast<int>(extractClosedLoops(g, u, hbar, what).size());
}

// Discrete Gauss linking integral of two closed polylines (midpoint rule):
// (1/4pi) sum_ij ((tA_i x tB_j) . r_ij) / |r_ij|^3. Converges to the integer
// linking number for disjoint closed curves; at the resolutions extracted
// here (~50-100 vertices) the quadrature error is well under 0.01.
static double gaussLinking(const std::vector<Vec3>& A, const std::vector<Vec3>& B) {
  double sum = 0.0;
  for (size_t i = 0; i < A.size(); ++i) {
    const Vec3 ta = bochner::vsub(A[(i + 1) % A.size()], A[i]);
    const Vec3 ma = bochner::vadd(A[i], bochner::vscale(ta, 0.5));
    for (size_t j = 0; j < B.size(); ++j) {
      const Vec3 tb = bochner::vsub(B[(j + 1) % B.size()], B[j]);
      const Vec3 mb = bochner::vadd(B[j], bochner::vscale(tb, 0.5));
      const Vec3 r = bochner::vsub(ma, mb);
      const double d = bochner::vnorm(r);
      if (d < 1e-12) continue;  // coincident midpoints: the self-pair diagonal
      sum += bochner::vdot(bochner::vcross(ta, tb), r) / (d * d * d);
    }
  }
  return sum / (4.0 * M_PI);
}

// Writhe = the Gauss self-integral of one closed polyline (the i == j diagonal
// vanishes in the smooth limit and is skipped above). Quadratic in the
// tangent, hence independent of traversal direction -- usable on the
// undirected extracted polylines.
static double writhe(const std::vector<Vec3>& A) { return gaussLinking(A, A); }

TEST_CASE("Hopf-linked rings extract as two interlocked closed loops") {
  MacGrid g(40, 40, 40, 0.08, Vec3{-1.6, -1.6, -1.6});
  const double R = 0.7, Gamma = 1.0, core = 0.15;
  const double hbar = Gamma / (2.0 * M_PI);
  const std::vector<std::vector<Vec3>> rings{
      bochner::circleCurve({-R * 0.5, 0, 0}, {0, 0, 1}, R, 256),
      bochner::circleCurve({R * 0.5, 0, 0}, {0, 1, 0}, R, 256)};
  auto u = bochner::filamentFaceField(g, rings, Gamma, core);
  auto loops = extractClosedLoops(g, u, hbar, "Hopf");
  REQUIRE(loops.size() >= 2);
  // Interlocked, not merely coexisting: two unlinked loops would also pass a
  // count check. The Gauss integral of the two largest loops must give
  // linking number +/-1 (the seed circles link once; the sign is arbitrary
  // because each extracted polyline's traversal direction is).
  const double lk = gaussLinking(loops[0].points, loops[1].points);
  INFO("Hopf: |Lk| = " << std::abs(lk));
  CHECK(std::abs(std::abs(lk) - 1.0) < 0.2);
}

TEST_CASE("head-on colliding rings (opposite axes) extract as two closed loops") {
  MacGrid g(36, 36, 40, 0.08, Vec3{-1.44, -1.44, -1.6});
  const double R = 0.6, Gamma = 1.0, core = 0.15;
  const double hbar = Gamma / (2.0 * M_PI);
  // Opposite axes, same Gamma => the two rings self-propel toward each other.
  const std::vector<std::vector<Vec3>> rings{
      bochner::circleCurve({0, 0, -0.6}, {0, 0, 1}, R, 256),
      bochner::circleCurve({0, 0, 0.6}, {0, 0, -1}, R, 256)};
  auto u = bochner::filamentFaceField(g, rings, Gamma, core);
  CHECK(countClosedLoops(g, u, hbar, "collision") >= 2);
}

TEST_CASE("trefoil knot extracts as a closed filament") {
  MacGrid g(44, 44, 32, 0.08, Vec3{-1.76, -1.76, -1.28});
  const double Gamma = 1.0, core = 0.15;
  const double hbar = Gamma / (2.0 * M_PI);
  auto u = bochner::filamentFaceField(g, bochner::trefoilKnotCurve({0, 0, 0}, 0.3, 256), Gamma, core);
  // The knot is a single closed curve; extraction must find a closed loop.
  // No knottedness assertion here, deliberately: the ground state RELAXES the
  // seeded field, and at this resolution/core the relaxation loses the knot --
  // the extracted zero line is a single closed loop of writhe ~ -0.97 (a
  // reconnected unknot; measured identically under the distance-only and the
  // orientation-constrained pairings). That is field physics, not an
  // extraction defect; the chain's own knot fidelity is pinned by the analytic
  // Milnor-map test below, whose INPUT field's zero set is exactly a trefoil.
  auto loops = extractClosedLoops(g, u, hbar, "trefoil");
  REQUIRE(loops.size() >= 1);
  INFO("trefoil: writhe = " << writhe(loops[0].points));
}

TEST_CASE("analytic Milnor-map trefoil field extracts as a knotted filament") {
  // psi(x) = u^2 - v^3 with (u, v) the inverse stereographic image of x on
  // S^3 in C^2: u = (r^2 - 1 + 2iz)/(r^2 + 1), v = 2(x + iy)/(r^2 + 1)
  // (|u|^2 + |v|^2 = 1). The zero set is exactly a (2,3) torus knot
  // (Brauner/Milnor; the knotted-optical-vortex construction), so this pins
  // traceZeroSet -> linkFilaments's knot fidelity with no eigensolve and no
  // seeded-field relaxation in the way.
  MacGrid g(48, 48, 48, 6.4 / 48, Vec3{-3.2, -3.2, -3.2});
  std::vector<double> psi(2 * static_cast<std::size_t>(g.numCells()));
  for (int i = 0; i < g.nx(); ++i)
    for (int j = 0; j < g.ny(); ++j)
      for (int k = 0; k < g.nz(); ++k) {
        const Vec3 p = g.cellCenter(i, j, k);
        const double r2 = p[0] * p[0] + p[1] * p[1] + p[2] * p[2];
        const std::complex<double> u((r2 - 1.0) / (r2 + 1.0), 2.0 * p[2] / (r2 + 1.0));
        const std::complex<double> v(2.0 * p[0] / (r2 + 1.0), 2.0 * p[1] / (r2 + 1.0));
        const std::complex<double> val = u * u - v * v * v;
        psi[2 * static_cast<std::size_t>(g.cellIndex(i, j, k))] = val.real();
        psi[2 * static_cast<std::size_t>(g.cellIndex(i, j, k)) + 1] = val.imag();
      }
  auto filaments = bochner::linkFilaments(g, bochner::traceZeroSet(g, psi));
  std::vector<Filament> closed;
  for (auto& f : filaments)
    if (f.closed && f.points.size() >= 4) closed.push_back(std::move(f));
  std::sort(closed.begin(), closed.end(),
            [](const Filament& a, const Filament& b) { return a.points.size() > b.points.size(); });
  REQUIRE(!closed.empty());
  // Reference curve: u^2 = v^3 on the Clifford torus |u| = c1, |v| = c2 with
  // c2^3 + c2^2 = 1, c1 = c2^{3/2}, parametrized (c1 e^{3is}, c2 e^{2is}) and
  // mapped back through the same chart, (x, y, z) = (Re v, Im v, Im u)/(1 - Re u).
  // Its writhe comes from the SAME discrete Gauss integral as the extracted
  // loop's, so the two share quadrature; extraction displaces the curve by
  // O(h) on O(1) features, which moves the (geometric) writhe by far less
  // than the ~2-per-crossing a strand swap costs, and an unknot sits near 0.
  const double c2 = 0.7548776662466927;  // real root of c^3 + c^2 = 1
  const double c1 = std::pow(c2, 1.5);
  std::vector<Vec3> ref(256);
  for (int t = 0; t < 256; ++t) {
    const double s = 2.0 * M_PI * t / 256;
    const std::complex<double> u = c1 * std::exp(std::complex<double>(0.0, 3.0 * s));
    const std::complex<double> v = c2 * std::exp(std::complex<double>(0.0, 2.0 * s));
    const double d = 1.0 - u.real();
    ref[t] = Vec3{v.real() / d, v.imag() / d, u.imag() / d};
  }
  const double wref = writhe(ref);
  const double wr = writhe(closed[0].points);
  INFO("milnor trefoil: closedLoops=" << closed.size() << " extracted writhe = " << wr
                                      << " reference = " << wref);
  CHECK(std::abs(wr - wref) < 0.6);
}

TEST_CASE("leapfrogging coaxial rings extract as two closed loops") {
  MacGrid g(36, 36, 40, 0.08, Vec3{-1.44, -1.44, -1.6});
  const double R = 0.6, Gamma = 1.0, core = 0.15;
  const double hbar = Gamma / (2.0 * M_PI);
  const std::vector<std::vector<Vec3>> rings{
      bochner::circleCurve({0, 0, -0.35}, {0, 0, 1}, R, 256),
      bochner::circleCurve({0, 0, 0.35}, {0, 0, 1}, R, 256)};
  auto u = bochner::filamentFaceField(g, rings, Gamma, core);
  CHECK(countClosedLoops(g, u, hbar, "leapfrog") >= 2);
}
