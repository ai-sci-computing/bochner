/// \file
/// Red-green TDD: analytic seeding fields on the MAC grid.
/// Hill's spherical vortex is the distributed-vorticity field used by the
/// multi-hbar decomposition demo; here we anchor its known velocity values.
#include <doctest.h>

#include <cmath>

#include "fluid/MacAdvection.h"
#include "grid/MacGrid.h"
#include "fluid/MacVortexRing.h"

using bochner::MacGrid;
using bochner::Vec3;

TEST_CASE("circleCurve samples a closed planar ring of the right radius") {
  const Vec3 c{0.1, -0.2, 0.3};
  const Vec3 axis{0, 0, 1};
  const double R = 0.7;
  const int segs = 64;
  auto loop = bochner::circleCurve(c, axis, R, segs);
  REQUIRE(static_cast<int>(loop.size()) == segs);
  for (const auto& p : loop) {
    CHECK(std::abs(p[2] - c[2]) < 1e-12);                            // lies in the z=c plane
    const double r = std::hypot(p[0] - c[0], p[1] - c[1]);
    CHECK(r == doctest::Approx(R).epsilon(1e-9));                    // on the radius-R circle
  }
}

TEST_CASE("filamentFaceField on a circle reproduces vortexRingFaceField") {
  MacGrid g(20, 20, 20, 0.2, Vec3{-2.0, -2.0, -2.0});
  const Vec3 c{0, 0, 0}, axis{0, 0, 1};
  const double R = 0.8, Gamma = 1.3, core = 0.1;
  auto ref = bochner::vortexRingFaceField(g, c, axis, R, Gamma, core, 256);
  auto loop = bochner::circleCurve(c, axis, R, 256);
  auto got = bochner::filamentFaceField(g, loop, Gamma, core);
  REQUIRE(got.x.size() == ref.x.size());
  for (size_t i = 0; i < ref.x.size(); ++i) CHECK(got.x[i] == doctest::Approx(ref.x[i]).epsilon(1e-9));
  for (size_t i = 0; i < ref.z.size(); ++i) CHECK(got.z[i] == doctest::Approx(ref.z[i]).epsilon(1e-9));

  // On-axis center velocity of a regularized thin ring: Gamma R^2 / (2 (R^2+core^2)^{3/2}).
  Vec3 vc = bochner::sampleVelocity(g, got, Vec3{0, 0, 0});
  const double expected = Gamma * R * R / (2.0 * std::pow(R * R + core * core, 1.5));
  CHECK(vc[2] == doctest::Approx(expected).epsilon(0.05));
}

TEST_CASE("multi-loop filamentFaceField is the sum of the single-loop fields") {
  MacGrid g(24, 24, 24, 1.0 / 6.0, Vec3{-2.0, -2.0, -2.0});
  const double R = 0.5, Gamma = 1.0, core = 0.12;
  auto l1 = bochner::circleCurve(Vec3{0, 0, -0.6}, Vec3{0, 0, 1}, R, 128);
  auto l2 = bochner::circleCurve(Vec3{0, 0, 0.6}, Vec3{0, 0, 1}, R, 128);
  auto f1 = bochner::filamentFaceField(g, l1, Gamma, core);
  auto f2 = bochner::filamentFaceField(g, l2, Gamma, core);
  auto both = bochner::filamentFaceField(g, std::vector<std::vector<Vec3>>{l1, l2}, Gamma, core);
  REQUIRE(both.x.size() == f1.x.size());
  for (size_t i = 0; i < both.x.size(); ++i)
    CHECK(both.x[i] == doctest::Approx(f1.x[i] + f2.x[i]).epsilon(1e-9));
  for (size_t i = 0; i < both.z.size(); ++i)
    CHECK(both.z[i] == doctest::Approx(f1.z[i] + f2.z[i]).epsilon(1e-9));
}

TEST_CASE("trefoilKnotCurve is closed and genuinely three-dimensional") {
  const int segs = 200;
  auto knot = bochner::trefoilKnotCurve(Vec3{0, 0, 0}, 0.25, segs);
  REQUIRE(static_cast<int>(knot.size()) == segs);
  // Closed: the gap from last to first matches a typical inter-sample spacing.
  const double seg0 = bochner::vnorm(bochner::vsub(knot[1], knot[0]));
  const double wrap = bochner::vnorm(bochner::vsub(knot[0], knot[segs - 1]));
  CHECK(wrap == doctest::Approx(seg0).epsilon(0.5));
  // Spans all three axes (a planar ring would have one near-constant coordinate).
  double zmin = knot[0][2], zmax = knot[0][2];
  for (const auto& p : knot) {
    zmin = std::min(zmin, p[2]);
    zmax = std::max(zmax, p[2]);
  }
  CHECK(zmax - zmin > 0.05);
}

TEST_CASE("leapfrogSlabFaceField imposes the reference two-tier slab profile") {
  // Origin at 0 so x-face x-coords fall exactly on i*h; slab plane at i=20.
  const double h = 0.1;
  MacGrid g(40, 20, 20, h, Vec3{0, 0, 0});
  const Vec3 axisPt{2.0, 1.0, 1.0};       // slab centre; ring axis at (y,z)=(1,1)
  const Vec3 prop{1, 0, 0};               // rings propagate +x
  const double rIn = 0.4, rOut = 0.7, circ = 0.28;
  const double vel = circ / (3.0 * h);    // reference sheet-speed scaling
  auto u = bochner::leapfrogSlabFaceField(g, axisPt, prop, rIn, rOut, circ);

  // Purely axial: no y/z-face velocity anywhere.
  for (double v : u.y) CHECK(v == doctest::Approx(0.0));
  for (double v : u.z) CHECK(v == doctest::Approx(0.0));

  // On the slab plane (i=20 -> x=2.0): inner disk = 2*vel, annulus = vel, outside = 0.
  auto rad = [&](int j, int k) {
    Vec3 p = g.faceXCenter(20, j, k);
    return std::hypot(p[1] - axisPt[1], p[2] - axisPt[2]);
  };
  CHECK(rad(9, 9) < rIn);                                                   // near axis
  CHECK(u.x[g.faceXIndex(20, 9, 9)] == doctest::Approx(2.0 * vel));
  CHECK((rad(14, 9) > rIn && rad(14, 9) < rOut));                          // in the annulus
  CHECK(u.x[g.faceXIndex(20, 14, 9)] == doctest::Approx(vel));
  CHECK(rad(18, 9) > rOut);                                                 // beyond the outer ring
  CHECK(u.x[g.faceXIndex(20, 18, 9)] == doctest::Approx(0.0));

  // Off the slab (i=10 -> x=1.0, well outside the 1.5h-thick slab): zero even on-axis.
  CHECK(u.x[g.faceXIndex(10, 9, 9)] == doctest::Approx(0.0));
}

TEST_CASE("Hill's vortex has the right on-axis interior and far-field speed") {
  MacGrid g(24, 24, 24, 1.0 / 6.0, Vec3{-2.0, -2.0, -2.0});  // [-2,2]^3
  const double a = 0.6, U = 1.0;
  auto u = bochner::hillVortexFaceField(g, {0, 0, 0}, {0, 0, 1}, a, U);

  // Center (interior): on-axis velocity is -3U/2 along the axis.
  Vec3 c = bochner::sampleVelocity(g, u, Vec3{0, 0, 0});
  CHECK(c[2] == doctest::Approx(-1.5 * U).epsilon(0.1));
  CHECK(std::abs(c[0]) < 0.05);
  CHECK(std::abs(c[1]) < 0.05);

  // Far on the axis (exterior): approaches the uniform stream U.
  Vec3 far = bochner::sampleVelocity(g, u, Vec3{0, 0, 1.6});
  double rr = 1.6;
  double expected = 1.0 - (a * a * a) / (rr * rr * rr);  // on-axis potential value
  CHECK(far[2] == doctest::Approx(expected).epsilon(0.1));
  CHECK(far[2] > 0.0);  // far field opposes the recirculating interior
}

// ---------------------------------------------------------------------------
// The on-axis PROFILE, not just the centre point.
//
// Most assertions in this file are tautologies: vortexRingFaceField IS
// filamentFaceField(circleCurve(...)) (MacVortexRing.cpp), and superposition
// holds for any linear kernel, so those comparisons cannot fail on a wrong
// kernel. The one real check above samples the analytic centre velocity -- but
// at a single point, which constrains the scale and the sign and nothing else.
//
// The closed form for a regularized circular filament, derived from
// Biot-Savart independently of the implementation (the transverse components
// integrate to zero around the loop, leaving only the axial one):
//
//   u_z(z) = Gamma R^2 / (2 (R^2 + z^2 + a^2)^{3/2})
//
// Checking it across a range of z pins the FUNCTIONAL FORM: a wrong
// regularization (a^2 added in the wrong place, or not at all), a wrong power
// in the kernel, or a 4*pi slip all change the falloff, not merely the scale,
// and none of them can be absorbed by a single-point tolerance.
// ---------------------------------------------------------------------------
TEST_CASE("regularized ring reproduces the analytic on-axis velocity profile") {
  const MacGrid g(48, 48, 48, 0.1, Vec3{-2.4, -2.4, -2.4});
  const Vec3 c{0, 0, 0}, axis{0, 0, 1};
  const double R = 0.8, Gamma = 1.3, core = 0.12;
  const auto u = bochner::vortexRingFaceField(g, c, axis, R, Gamma, core, 512);

  const double a2 = core * core, R2 = R * R;
  // Sample along the axis, staying well inside the box and away from the ring
  // plane's near field where the polygonal discretization dominates.
  for (const double z : {0.0, 0.2, 0.4, 0.6, 0.8, 1.2, 1.6}) {
    const Vec3 v = bochner::sampleVelocity(g, u, Vec3{0, 0, z});
    const double want = Gamma * R2 / (2.0 * std::pow(R2 + z * z + a2, 1.5));
    INFO("z=" << z << " got=" << v[2] << " want=" << want
              << " rel=" << std::abs(v[2] - want) / want);
    // Sign and magnitude together: a conjugated/negated kernel fails outright.
    CHECK(v[2] > 0.0);
    // Measured agreement is 0.35-0.44%; 1.5% leaves room for discretization
    // and platform variation while still failing on a real kernel error.
    CHECK(v[2] == doctest::Approx(want).epsilon(0.015));
    // Transverse components vanish on the axis by symmetry.
    CHECK(std::abs(v[0]) < 0.02 * want);
    CHECK(std::abs(v[1]) < 0.02 * want);
  }

  // The falloff exponent itself: u_z(z)/u_z(0) must follow the -3/2 power law.
  // A kernel with the wrong power passes a single-point scale check but not
  // this ratio.
  // A FAT core, where the regularization is a large effect rather than a
  // correction. At a=0.12 above, a^2/R^2 = 2%, so dropping the regularization
  // entirely still nearly passes (measured: it trips exactly one assertion).
  // At a=0.45 it is 32%, so the a^2 term is pinned properly.
  {
    const double fat = 0.45, fat2 = fat * fat;
    const auto uf = bochner::vortexRingFaceField(g, c, axis, R, Gamma, fat, 512);
    for (const double z : {0.0, 0.4, 0.8}) {
      const Vec3 v = bochner::sampleVelocity(g, uf, Vec3{0, 0, z});
      const double want = Gamma * R2 / (2.0 * std::pow(R2 + z * z + fat2, 1.5));
      const double wantNoReg = Gamma * R2 / (2.0 * std::pow(R2 + z * z, 1.5));
      INFO("fat core z=" << z << " got=" << v[2] << " want=" << want
                         << " (unregularized would be " << wantNoReg << ")");
      CHECK(v[2] == doctest::Approx(want).epsilon(0.02));
      // and the two predictions are far enough apart that this discriminates
      REQUIRE(std::abs(want - wantNoReg) > 0.1 * want);
    }
  }

  const double u0 = bochner::sampleVelocity(g, u, Vec3{0, 0, 0.0})[2];
  const double u1 = bochner::sampleVelocity(g, u, Vec3{0, 0, 1.2})[2];
  const double wantRatio = std::pow((R2 + a2) / (R2 + 1.44 + a2), 1.5);
  INFO("ratio got=" << u1 / u0 << " want=" << wantRatio);
  CHECK(u1 / u0 == doctest::Approx(wantRatio).epsilon(0.02));  // measured 0.95% off
}
