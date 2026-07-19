/// \file
/// Red-green TDD: MAC velocity reconstruction + RK4 backtrace (CF step 3a).
///
/// Foundation of CF's covector advection: the semi-Lagrangian
/// scheme (Alg 3) needs (a) the full velocity vector reconstructed at arbitrary
/// points -- to integrate the flow and to sample `u . Psi` -- and (b) the RK4
/// backward flow map Psi. Velocity components live staggered on faces; each is
/// trilinearly interpolated on its own lattice and clamped to the domain
/// (the "streak boundary" of §5.4.5).
#include <doctest.h>

#include <cmath>

#include "grid/GridOperators.h"
#include "fluid/MacAdvection.h"
#include "grid/MacGrid.h"

using bochner::FaceField;
using bochner::MacGrid;
using bochner::Vec3;
namespace ops = bochner::ops;

namespace {

// Fill a face field from analytic component functions sampled at face centers.
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

}  // namespace

TEST_CASE("velocity reconstruction reproduces a constant field exactly") {
  MacGrid g(5, 4, 6, 0.3, Vec3{-0.5, 1.0, 2.0});
  FaceField u = sampleField(
      g, [](Vec3) { return 1.5; }, [](Vec3) { return -2.0; }, [](Vec3) { return 0.7; });
  for (Vec3 p : {g.cellCenter(2, 1, 3), g.cellCenter(0, 0, 0), g.cellCenter(4, 3, 5)}) {
    Vec3 w = bochner::sampleVelocity(g, u, p);
    CHECK(w[0] == doctest::Approx(1.5));
    CHECK(w[1] == doctest::Approx(-2.0));
    CHECK(w[2] == doctest::Approx(0.7));
  }
}

TEST_CASE("velocity reconstruction is exact on a linear field (trilinear)") {
  MacGrid g(6, 5, 4, 0.25, Vec3{0.2, -0.3, 0.5});
  // Each component an independent affine function of position.
  auto fx = [](Vec3 p) { return 0.5 + 2.0 * p[0] - 1.0 * p[1] + 0.3 * p[2]; };
  auto fy = [](Vec3 p) { return -1.0 + 0.4 * p[0] + 1.5 * p[1] - 0.2 * p[2]; };
  auto fz = [](Vec3 p) { return 0.1 - 0.6 * p[0] + 0.2 * p[1] + 1.1 * p[2]; };
  FaceField u = sampleField(g, fx, fy, fz);

  for (Vec3 p : {g.cellCenter(3, 2, 1), Vec3{0.7, 0.1, 1.0}, g.faceXCenter(2, 2, 2)}) {
    Vec3 w = bochner::sampleVelocity(g, u, p);
    CHECK(w[0] == doctest::Approx(fx(p)));
    CHECK(w[1] == doctest::Approx(fy(p)));
    CHECK(w[2] == doctest::Approx(fz(p)));
  }
}

TEST_CASE("at a face center, the normal component equals the stored face value") {
  MacGrid g(4, 4, 4, 0.5);
  auto wob = [](Vec3 p) { return std::sin(p[0]) + std::cos(2 * p[1]) - 0.5 * p[2]; };
  FaceField u = sampleField(g, wob, wob, wob);
  // Interior x-face: the x-component samples exactly onto the stored value.
  Vec3 fc = g.faceXCenter(2, 1, 3);
  CHECK(bochner::sampleVelocity(g, u, fc)[0] == doctest::Approx(u.x[g.faceXIndex(2, 1, 3)]));
}

TEST_CASE("sampling clamps to the closest interior point outside the domain") {
  MacGrid g(3, 3, 3, 1.0);  // domain [0,3]^3
  FaceField u =
      sampleField(g, [](Vec3 p) { return p[0]; }, [](Vec3) { return 0.0; },
                  [](Vec3) { return 0.0; });
  // Far outside in +x: x-velocity clamps to its value at the max x-plane (=3).
  Vec3 w = bochner::sampleVelocity(g, u, Vec3{10.0, 1.5, 1.5});
  CHECK(w[0] == doctest::Approx(3.0));
}

TEST_CASE("RK4 backtrace of a uniform flow is exact translation") {
  MacGrid g(8, 8, 8, 0.5, Vec3{-2.0, -2.0, -2.0});  // domain [-2,2]^3
  FaceField v = sampleField(
      g, [](Vec3) { return 1.0; }, [](Vec3) { return -0.5; }, [](Vec3) { return 0.25; });
  Vec3 start{0.0, 0.0, 0.0};
  double dt = 0.4;
  Vec3 back = bochner::backtrace(g, v, start, dt);
  CHECK(back[0] == doctest::Approx(0.0 - 1.0 * dt));
  CHECK(back[1] == doctest::Approx(0.0 + 0.5 * dt));
  CHECK(back[2] == doctest::Approx(0.0 - 0.25 * dt));
}

TEST_CASE("covector advection: zero time step is the identity on interior faces") {
  MacGrid g(5, 5, 5, 0.4);
  auto wob = [](Vec3 p) { return std::sin(p[0]) + 0.5 * p[1] - p[2]; };
  FaceField u = sampleField(g, wob, wob, wob);
  FaceField v = sampleField(g, [](Vec3) { return 0.3; }, [](Vec3) { return -0.2; },
                            [](Vec3) { return 0.1; });
  FaceField out = bochner::advectCovectorSL(g, u, v, 0.0);
  for (int i = 1; i < g.nx(); ++i)
    for (int j = 0; j < g.ny(); ++j)
      for (int k = 0; k < g.nz(); ++k)
        CHECK(out.x[g.faceXIndex(i, j, k)] == doctest::Approx(u.x[g.faceXIndex(i, j, k)]));
}

TEST_CASE("covector advection under uniform flow is exact translation (Jacobian = I)") {
  // Linear field + uniform flow => RK4 backtrace, trilinear sampling, and the
  // FD Jacobian (= identity) are all exact, so the result is the analytic field
  // evaluated at the back-translated face center.
  MacGrid g(10, 10, 10, 0.2, Vec3{0.0, 0.0, 0.0});  // domain [0,2]^3
  auto fx = [](Vec3 p) { return 0.3 + 0.5 * p[0] - 0.2 * p[1] + 0.1 * p[2]; };
  auto fy = [](Vec3 p) { return -0.1 + 0.2 * p[0] + 0.4 * p[1] - 0.3 * p[2]; };
  auto fz = [](Vec3 p) { return 0.2 - 0.1 * p[0] + 0.3 * p[1] + 0.6 * p[2]; };
  FaceField u = sampleField(g, fx, fy, fz);
  const Vec3 vel{0.5, -0.25, 0.4};
  FaceField v = sampleField(g, [&](Vec3) { return vel[0]; }, [&](Vec3) { return vel[1]; },
                            [&](Vec3) { return vel[2]; });
  const double dt = 0.3;
  FaceField out = bochner::advectCovectorSL(g, u, v, dt);

  // Check interior faces whose back-translation stays well inside the domain.
  for (int i = 3; i <= 7; ++i)
    for (int j = 3; j <= 6; ++j)
      for (int k = 3; k <= 6; ++k) {
        Vec3 shifted = bochner::vsub(g.faceXCenter(i, j, k), bochner::vscale(vel, dt));
        CHECK(out.x[g.faceXIndex(i, j, k)] == doctest::Approx(fx(shifted)));
      }
}

TEST_CASE("covector advection reproduces the transpose-Jacobian under shear flow") {
  // Frozen shear v = (c*y, 0, 0). Its backward flow map is affine,
  // Psi(p) = (px - c*py*dt, py, pz), so the exact transpose Jacobian couples
  // the y-face update to u1: u2_new = -c*dt * u1(Psi) + u2(Psi). We reuse the
  // implementation's own backtrace+sample so boundary clamping cancels, and
  // assert only the (dPsi)^T assembly (the -c*dt off-diagonal).
  MacGrid g(12, 12, 12, 0.25, Vec3{0.0, 0.0, 0.0});
  const double c = 0.4;
  FaceField v = sampleField(g, [&](Vec3 p) { return c * p[1]; }, [](Vec3) { return 0.0; },
                            [](Vec3) { return 0.0; });
  auto fx = [](Vec3 p) { return 0.5 + p[0] - 0.3 * p[1] + 0.2 * p[2]; };
  auto fy = [](Vec3 p) { return -0.2 + 0.1 * p[0] + p[1] - 0.4 * p[2]; };
  auto fz = [](Vec3 p) { return 0.3 - 0.2 * p[0] + 0.5 * p[1] + p[2]; };
  FaceField u = sampleField(g, fx, fy, fz);
  const double dt = 0.2;
  FaceField out = bochner::advectCovectorSL(g, u, v, dt);

  for (int i = 3; i <= 8; ++i)
    for (int j = 3; j <= 8; ++j)
      for (int k = 3; k <= 8; ++k) {
        Vec3 psiF = bochner::backtrace(g, v, g.faceYCenter(i, j, k), dt);
        Vec3 w = bochner::sampleVelocity(g, u, psiF);
        double expected = -c * dt * w[0] + w[1];  // (dPsi)^T y-row, analytic Jacobian
        CHECK(out.y[g.faceYIndex(i, j, k)] == doctest::Approx(expected));
      }
}

TEST_CASE("covector advection co-rotates a uniform field under solid-body rotation") {
  // The rotation-dominated correctness check leapfrogging needs (CF Fig. 5):
  // under v = omega*(-y, x, 0) the inverse map is Psi=R(-theta), so a CONSTANT
  // covector u0=(1,0,0) must transport to the UNIFORM rotated field
  // (cos theta, sin theta, 0). A transpose/axis error in the Eq. 39-40 Jacobian
  // (invisible for a translating field) would corrupt this co-rotation.
  MacGrid g(32, 32, 32, 2.0 / 32, Vec3{-1.0, -1.0, -1.0});
  const double omega = 1.0, dt = 0.05, theta = omega * dt;
  FaceField v = sampleField(
      g, [&](Vec3 p) { return -omega * p[1]; }, [&](Vec3 p) { return omega * p[0]; },
      [](Vec3) { return 0.0; });
  FaceField u = sampleField(g, [](Vec3) { return 1.0; }, [](Vec3) { return 0.0; },
                            [](Vec3) { return 0.0; });

  FaceField out = bochner::advectCovectorBFECC(g, u, v, dt);
  // Deep interior (central half), away from the clamped walls.
  for (int i = 12; i <= 20; ++i)
    for (int j = 12; j <= 20; ++j)
      for (int k = 12; k <= 20; ++k) {
        CHECK(out.x[g.faceXIndex(i, j, k)] == doctest::Approx(std::cos(theta)).epsilon(1e-3));
        CHECK(out.y[g.faceYIndex(i, j, k)] == doctest::Approx(std::sin(theta)).epsilon(1e-3));
      }
}

TEST_CASE("accumulated long-time map + pullback reproduces rotation (CF+MCM lines 4-5)") {
  // Replicate the CF+MCM map advection (Alg 5 line 4, psiNew(x)=psi(backtrace(x)))
  // under a fixed solid-body rotation, then pull a constant snapshot through the
  // ACCUMULATED map (line 5). After many steps the result must still match the
  // analytic rotated uniform field -- the long-time mapping must not drift or
  // mis-compose. This is the interpolation-saving heart of CF+MCM.
  MacGrid g(32, 32, 32, 2.0 / 32, Vec3{-1.0, -1.0, -1.0});
  const double omega = 1.0, dt = 0.05;
  FaceField v = sampleField(
      g, [&](Vec3 p) { return -omega * p[1]; }, [&](Vec3 p) { return omega * p[0]; },
      [](Vec3) { return 0.0; });
  FaceField u0 = sampleField(g, [](Vec3) { return 1.0; }, [](Vec3) { return 0.0; },
                             [](Vec3) { return 0.0; });
  std::vector<Vec3> psi(g.numCells());
  for (int i = 0; i < g.nx(); ++i)
    for (int j = 0; j < g.ny(); ++j)
      for (int k = 0; k < g.nz(); ++k) psi[g.cellIndex(i, j, k)] = g.cellCenter(i, j, k);

  const int steps = 16;
  for (int s = 0; s < steps; ++s) {
    std::vector<Vec3> psiNew(g.numCells());
    for (int i = 0; i < g.nx(); ++i)
      for (int j = 0; j < g.ny(); ++j)
        for (int k = 0; k < g.nz(); ++k)
          psiNew[g.cellIndex(i, j, k)] =
              bochner::sampleCellVec3(g, psi, bochner::backtrace(g, v, g.cellCenter(i, j, k), dt));
    psi.swap(psiNew);
  }
  const double theta = omega * dt * steps;
  FaceField u1 = bochner::pullbackThroughMap(g, psi, u0);
  for (int i = 12; i <= 20; ++i)
    for (int j = 12; j <= 20; ++j)
      for (int k = 12; k <= 20; ++k) {
        CHECK(u1.x[g.faceXIndex(i, j, k)] == doctest::Approx(std::cos(theta)).epsilon(2e-3));
        CHECK(u1.y[g.faceYIndex(i, j, k)] == doctest::Approx(std::sin(theta)).epsilon(2e-3));
      }
}

TEST_CASE("BFECC is markedly less dissipative than plain semi-Lagrangian") {
  // A resolved curved field translated by a uniform flow: the exact result is
  // a pure translation. Plain sL smears it (1st order); BFECC's back-and-forth
  // error correction is 2nd order and should track the analytic translate far
  // more closely.
  MacGrid g(24, 24, 24, 0.1, Vec3{0.0, 0.0, 0.0});  // domain [0, 2.4]^3
  const double k = 2.0 * M_PI / 0.6;                 // wavelength 0.6 = 6 cells
  auto fx = [&](Vec3 p) { return std::sin(k * p[0]) * std::cos(k * p[1]); };
  auto fy = [&](Vec3 p) { return std::cos(k * p[0]) * std::sin(k * p[2]); };
  auto fz = [&](Vec3 p) { return std::sin(k * p[1]) * std::cos(k * p[2]); };
  FaceField u = sampleField(g, fx, fy, fz);
  const Vec3 vel{0.7, 0.5, 0.6};
  FaceField v = sampleField(g, [&](Vec3) { return vel[0]; }, [&](Vec3) { return vel[1]; },
                            [&](Vec3) { return vel[2]; });
  const double dt = 0.1;

  FaceField sl = bochner::advectCovectorSL(g, u, v, dt);
  FaceField bf = bochner::advectCovectorBFECC(g, u, v, dt);

  double errSL = 0.0, errBF = 0.0;
  int count = 0;
  for (int i = 6; i <= 18; ++i)
    for (int j = 6; j <= 18; ++j)
      for (int kk = 6; kk <= 18; ++kk) {
        Vec3 shifted = bochner::vsub(g.faceXCenter(i, j, kk), bochner::vscale(vel, dt));
        double exact = fx(shifted);
        errSL += std::abs(sl.x[g.faceXIndex(i, j, kk)] - exact);
        errBF += std::abs(bf.x[g.faceXIndex(i, j, kk)] - exact);
        ++count;
      }
  errSL /= count;
  errBF /= count;
  CHECK(errBF < 0.5 * errSL);  // BFECC cuts the dissipative error substantially
}

TEST_CASE("sampleCellVec3 is exact on an affine cell-centered field") {
  MacGrid g(6, 5, 7, 0.2, Vec3{-0.3, 0.4, -1.0});
  auto F = [](Vec3 p) {
    return Vec3{0.5 + p[0] - 0.3 * p[1], -0.2 + 0.4 * p[1] + p[2], 0.1 - 0.6 * p[0] + 0.2 * p[2]};
  };
  std::vector<Vec3> field(g.numCells());
  for (int i = 0; i < g.nx(); ++i)
    for (int j = 0; j < g.ny(); ++j)
      for (int k = 0; k < g.nz(); ++k) field[g.cellIndex(i, j, k)] = F(g.cellCenter(i, j, k));
  for (Vec3 p : {g.cellCenter(3, 2, 4), Vec3{0.1, 0.9, -0.4}}) {
    Vec3 w = bochner::sampleCellVec3(g, field, p);
    CHECK(w[0] == doctest::Approx(F(p)[0]));
    CHECK(w[1] == doctest::Approx(F(p)[1]));
    CHECK(w[2] == doctest::Approx(F(p)[2]));
  }
}

TEST_CASE("pullbackThroughMap: identity map returns the source on interior faces") {
  MacGrid g(6, 6, 6, 0.3, Vec3{-0.9, -0.9, -0.9});
  auto wob = [](Vec3 p) { return std::sin(p[0]) + 0.5 * p[1] - p[2]; };
  FaceField src = sampleField(g, wob, wob, wob);
  std::vector<Vec3> id(g.numCells());
  for (int i = 0; i < g.nx(); ++i)
    for (int j = 0; j < g.ny(); ++j)
      for (int k = 0; k < g.nz(); ++k) id[g.cellIndex(i, j, k)] = g.cellCenter(i, j, k);
  FaceField out = bochner::pullbackThroughMap(g, id, src);
  for (int i = 1; i < g.nx(); ++i)
    for (int j = 0; j < g.ny(); ++j)
      for (int k = 0; k < g.nz(); ++k)
        CHECK(out.x[g.faceXIndex(i, j, k)] == doctest::Approx(src.x[g.faceXIndex(i, j, k)]));
}

TEST_CASE("pullbackThroughMap: a uniform-shift map translates the field (Jacobian = I)") {
  // Map M(x) = x - c (constant shift, Jacobian = I), so the pullback is the
  // source evaluated at the shifted face center -- a pure translation.
  MacGrid g(12, 12, 12, 0.2, Vec3{0, 0, 0});
  auto fx = [](Vec3 p) { return 0.3 + 0.5 * p[0] - 0.2 * p[1] + 0.1 * p[2]; };
  FaceField src = sampleField(g, fx, fx, fx);
  const Vec3 c{0.3, -0.15, 0.2};
  std::vector<Vec3> shift(g.numCells());
  for (int i = 0; i < g.nx(); ++i)
    for (int j = 0; j < g.ny(); ++j)
      for (int k = 0; k < g.nz(); ++k) shift[g.cellIndex(i, j, k)] = bochner::vsub(g.cellCenter(i, j, k), c);
  FaceField out = bochner::pullbackThroughMap(g, shift, src);
  for (int i = 3; i <= 8; ++i)
    for (int j = 3; j <= 8; ++j)
      for (int k = 3; k <= 8; ++k) {
        Vec3 shifted = bochner::vsub(g.faceXCenter(i, j, k), c);
        CHECK(out.x[g.faceXIndex(i, j, k)] == doctest::Approx(fx(shifted)).epsilon(1e-9));
      }
}

TEST_CASE("BFECC limiter keeps the output within the forward stencil's extrema") {
  // The minmax limiter (CF 5.4.2) clamps each corrected face value to the
  // min/max of the forward-advected field u1 over its immediate same-family
  // neighbour stencil -- so no new extrema are introduced.
  MacGrid g(16, 16, 16, 0.1);
  const double k = 2.0 * M_PI / 0.4;  // short wavelength to stress the limiter
  auto f = [&](Vec3 p) { return std::sin(k * p[0]) + std::cos(k * p[1]) - std::sin(k * p[2]); };
  FaceField u = sampleField(g, f, f, f);
  FaceField v = sampleField(g, [](Vec3) { return 0.9; }, [](Vec3) { return -0.7; },
                            [](Vec3) { return 0.6; });
  const double dt = 0.1;

  FaceField u1 = bochner::advectCovectorSL(g, u, v, dt);
  FaceField bf = bochner::advectCovectorBFECC(g, u, v, dt);

  for (int i = 1; i < g.nx(); ++i)
    for (int j = 0; j < g.ny(); ++j)
      for (int kk = 0; kk < g.nz(); ++kk) {
        double lo = u1.x[g.faceXIndex(i, j, kk)], hi = lo;
        for (int di = -1; di <= 1; ++di)
          for (int dj = -1; dj <= 1; ++dj)
            for (int dk = -1; dk <= 1; ++dk) {
              int ni = std::clamp(i + di, 0, g.nx());
              int nj = std::clamp(j + dj, 0, g.ny() - 1);
              int nk = std::clamp(kk + dk, 0, g.nz() - 1);
              double val = u1.x[g.faceXIndex(ni, nj, nk)];
              lo = std::min(lo, val);
              hi = std::max(hi, val);
            }
        double out = bf.x[g.faceXIndex(i, j, kk)];
        CHECK(out >= doctest::Approx(lo).epsilon(1e-9));
        CHECK(out <= doctest::Approx(hi).epsilon(1e-9));
      }
}
