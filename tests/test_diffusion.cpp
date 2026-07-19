/// \file
/// Red-green TDD for Phase C: the explicit viscous diffusion operator. One step
/// of u += nu*dt*Laplacian(u). Verified against exact discrete Laplacians (a
/// linear field is unchanged, a quadratic gains 2*nu*dt), a decaying sinusoid,
/// and the no-slip drag toward a solid.
#include <doctest.h>

#include <cmath>

#include "grid/GridOperators.h"
#include "grid/MacGrid.h"
#include "grid/MacObstacle.h"

using namespace bochner;

TEST_CASE("diffusion leaves a linear velocity field unchanged (Laplacian = 0)") {
  MacGrid g(20, 16, 12, 0.1, Vec3{0, 0, 0});
  FaceField u = ops::zeroFaceField(g);
  for (int i = 0; i <= g.nx(); ++i)
    for (int j = 0; j < g.ny(); ++j)
      for (int k = 0; k < g.nz(); ++k) u.x[g.faceXIndex(i, j, k)] = 2.0 * g.faceXCenter(i, j, k)[0];

  const FaceField out = ops::diffuseVelocity(g, u, /*nu=*/0.3, /*dt=*/0.01);
  for (int i = 1; i < g.nx(); ++i)
    for (int j = 0; j < g.ny(); ++j)
      for (int k = 0; k < g.nz(); ++k)
        CHECK(out.x[g.faceXIndex(i, j, k)] ==
              doctest::Approx(u.x[g.faceXIndex(i, j, k)]).epsilon(1e-9));
}

TEST_CASE("diffusion of a quadratic field adds exactly 2*nu*dt (Laplacian = 2)") {
  MacGrid g(24, 18, 14, 0.1, Vec3{-1, -1, -1});
  const double nu = 0.5, dt = 0.002;  // c = nu*dt/h^2 = 0.1 <= 1/6: a single stable
                                      // step, so the exact +2*nu*dt identity holds
                                      // (larger c would auto-substep, changing it)
  FaceField u = ops::zeroFaceField(g);
  for (int i = 0; i <= g.nx(); ++i)
    for (int j = 0; j < g.ny(); ++j)
      for (int k = 0; k < g.nz(); ++k) {
        const double x = g.faceXCenter(i, j, k)[0];
        u.x[g.faceXIndex(i, j, k)] = x * x;  // d2/dx2 = 2, others 0
      }
  const FaceField out = ops::diffuseVelocity(g, u, nu, dt);
  for (int i = 1; i < g.nx(); ++i)
    for (int j = 0; j < g.ny(); ++j)
      for (int k = 0; k < g.nz(); ++k)
        CHECK(out.x[g.faceXIndex(i, j, k)] ==
              doctest::Approx(u.x[g.faceXIndex(i, j, k)] + 2.0 * nu * dt).epsilon(1e-9));
}

TEST_CASE("diffusion decays a sinusoidal mode (sign and magnitude)") {
  MacGrid g(40, 8, 8, 2.0 * M_PI / 40, Vec3{0, 0, 0});  // one x-period across the box
  const double nu = 0.4, dt = 0.005, h = g.spacing();
  FaceField u = ops::zeroFaceField(g);
  for (int i = 0; i <= g.nx(); ++i)
    for (int j = 0; j < g.ny(); ++j)
      for (int k = 0; k < g.nz(); ++k) u.x[g.faceXIndex(i, j, k)] = std::sin(g.faceXCenter(i, j, k)[0]);
  const FaceField out = ops::diffuseVelocity(g, u, nu, dt);
  // Discrete Laplacian eigenvalue of sin(x) on spacing h: -2(1-cos h)/h^2.
  const double factor = 1.0 - nu * dt * 2.0 * (1.0 - std::cos(h)) / (h * h);
  CHECK(factor < 1.0);  // decays
  const int i = g.nx() / 4, j = g.ny() / 2, k = g.nz() / 2;  // sin ~ 1 here
  CHECK(out.x[g.faceXIndex(i, j, k)] ==
        doctest::Approx(factor * u.x[g.faceXIndex(i, j, k)]).epsilon(1e-6));
}

TEST_CASE("diffusion auto-substeps: stable and decaying even when nu*dt/h^2 > 1/6") {
  // A single explicit step is unstable for c = nu*dt/h^2 > 1/6: the worst mode
  // (an every-other-face checkerboard) is amplified by |1 - 12c|. Here c = 1.1,
  // so a single step would blow the checkerboard up ~12x. diffuseVelocity must
  // split the step internally so the result stays bounded AND actually decays.
  MacGrid g(8, 8, 8, 1.0, Vec3{0, 0, 0});  // h = 1
  FaceField u = ops::zeroFaceField(g);
  double maxIn = 0.0;
  for (int i = 1; i < g.nx(); ++i)
    for (int j = 0; j < g.ny(); ++j)
      for (int k = 0; k < g.nz(); ++k) {
        const double v = ((i + j + k) & 1) ? -1.0 : 1.0;  // checkerboard on x-faces
        u.x[g.faceXIndex(i, j, k)] = v;
        maxIn = std::max(maxIn, std::abs(v));
      }
  const FaceField out = ops::diffuseVelocity(g, u, /*nu=*/1.0, /*dt=*/1.1, {});  // c = 1.1
  double maxOut = 0.0;
  for (double v : out.x) {
    CHECK(std::isfinite(v));
    maxOut = std::max(maxOut, std::abs(v));
  }
  CHECK(maxOut <= maxIn + 1e-12);  // stable: a single unstable step would give ~12x here
  CHECK(maxOut < 0.99 * maxIn);    // and it genuinely diffuses (the mode decays)
}

TEST_CASE("no-slip: diffusion drags fluid toward zero next to a solid") {
  MacGrid g(20, 20, 8, 0.1, Vec3{-1, -1, -0.4});
  SolidMask solid = sphereMask(g, {0, 0, 0}, 0.35);
  FaceField u = ops::zeroFaceField(g);
  for (double& v : u.x) v = 1.0;  // uniform stream
  const FaceField out = ops::diffuseVelocity(g, u, /*nu=*/0.5, /*dt=*/0.01, solid);

  // A wet x-face with a solid-touching neighbour is dragged below the free stream;
  // a wet face far from the solid is untouched (uniform -> Laplacian 0).
  bool draggedSomewhere = false;
  for (int i = 1; i < g.nx(); ++i)
    for (int j = 0; j < g.ny(); ++j) {
      const int f = g.faceXIndex(i, j, 4);
      if (isSolid(solid, g.cellIndex(i - 1, j, 4)) || isSolid(solid, g.cellIndex(i, j, 4))) continue;
      if (out.x[f] < 0.999) draggedSomewhere = true;             // near the solid: slowed
      const Vec3 p = g.faceXCenter(i, j, 4);
      if (std::hypot(p[0], p[1]) > 0.7) CHECK(out.x[f] == doctest::Approx(1.0).epsilon(1e-9));
    }
  CHECK(draggedSomewhere);
}
