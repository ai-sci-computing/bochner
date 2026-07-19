/// \file
/// Metal GPU backend, first slice: the GPU trilinear velocity sample must agree
/// with the CPU sampleVelocity() to single-precision tolerance (the GPU runs in
/// float; the CPU in double). Gated on BOCHNER_WITH_METAL. If no Metal device is
/// present the test reports and returns (nothing to verify).
#include <doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include <complex>

#include "fluid/MacAdvection.h"
#include "fluid/MacCharacteristicMap.h"
#include "fluid/MacProjection.h"
#include "fluid/MacVortexRing.h"
#include "fluid/PoissonMGPCG.h"
#include "gpu/MetalContext.h"
#include "grid/CooMatrix.h"
#include "grid/GridOperators.h"
#include "grid/MacGrid.h"
#include "grid/MacObstacle.h"
#include "grid/Vec3.h"
#include "solvers/GaugeEigen.h"
#include "solvers/GaugeMultigrid.h"
#include "solvers/PoissonMultigrid.h"

using namespace bochner;

// Cell-centered 7-point Poisson with a Robin ghost on every boundary face (SPD).
static SpMat robinPoissonT(int n, double h) {
  const double w = 1.0 / (h * h);
  auto id = [&](int i, int j, int k) { return (i * n + j) * n + k; };
  CooMatrix A(n * n * n, n * n * n);
  auto couple = [&](int a, int b) {
    A.add(a, a, w); A.add(b, b, w); A.add(a, b, -w); A.add(b, a, -w);
  };
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      for (int k = 0; k < n; ++k) {
        if (i) couple(id(i - 1, j, k), id(i, j, k));
        if (j) couple(id(i, j - 1, k), id(i, j, k));
        if (k) couple(id(i, j, k - 1), id(i, j, k));
        int bnd = (i == 0) + (i == n - 1) + (j == 0) + (j == n - 1) + (k == 0) + (k == n - 1);
        if (bnd) A.add(id(i, j, k), id(i, j, k), bnd * w);
      }
  return toSpMat(A);
}

TEST_CASE("Metal sampleVelocityBatch matches the CPU sampleVelocity") {
  if (!gpu::metalAvailable()) {
    MESSAGE("no Metal device available -- skipping GPU parity check");
    return;
  }

  // A non-trivial staggered field on an offset grid, filled with a smooth but
  // non-affine pattern so trilinear interpolation is genuinely exercised (an
  // affine field would interpolate exactly and hide interpolation error).
  MacGrid g(20, 16, 12, 0.1, Vec3{-1.0, -0.8, -0.6});
  FaceField u = ops::zeroFaceField(g);
  auto fill = [](std::vector<double>& a, int seed) {
    for (size_t i = 0; i < a.size(); ++i) {
      const double t = static_cast<double>(i);
      a[i] = std::sin(0.11 * t + seed) + 0.3 * std::cos(0.023 * t * t);
    }
  };
  fill(u.x, 1);
  fill(u.y, 2);
  fill(u.z, 3);

  // Random query points spanning the interior AND reaching outside the domain
  // (to exercise the clamped streak boundary). Off-grid by construction, so the
  // float/double floor never disagrees at a cell boundary.
  std::mt19937 rng(12345);
  std::uniform_real_distribution<double> dx(-1.6, 1.1), dy(-1.3, 0.9), dz(-1.0, 0.7);
  std::vector<Vec3> pts(4096);
  for (auto& p : pts) p = Vec3{dx(rng), dy(rng), dz(rng)};

  const std::vector<Vec3> gpu = gpu::sampleVelocityBatch(g, u, pts);
  REQUIRE(gpu.size() == pts.size());

  double maxErr = 0.0;
  for (size_t i = 0; i < pts.size(); ++i) {
    const Vec3 cpu = sampleVelocity(g, u, pts[i]);
    for (int c = 0; c < 3; ++c) maxErr = std::max(maxErr, std::abs(cpu[c] - gpu[i][c]));
  }
  MESSAGE("max |CPU(double) - GPU(float)| over 4096 points = " << maxErr);
  CHECK(maxErr < 1e-3);  // single-precision trilerp agreement (values are O(1))
}

TEST_CASE("Metal backtraceBatch matches the CPU RK4 backtrace") {
  if (!gpu::metalAvailable()) {
    MESSAGE("no Metal device available -- skipping GPU parity check");
    return;
  }

  MacGrid g(20, 16, 12, 0.1, Vec3{-1.0, -0.8, -0.6});
  FaceField v = ops::zeroFaceField(g);
  auto fill = [](std::vector<double>& a, int seed) {
    for (size_t i = 0; i < a.size(); ++i) {
      const double t = static_cast<double>(i);
      a[i] = std::sin(0.11 * t + seed) + 0.3 * std::cos(0.023 * t * t);  // O(1) velocity
    }
  };
  fill(v.x, 1);
  fill(v.y, 2);
  fill(v.z, 3);

  std::mt19937 rng(6789);
  std::uniform_real_distribution<double> dx(-0.95, 0.95), dy(-0.75, 0.75), dz(-0.55, 0.55);
  std::vector<Vec3> starts(4096);
  for (auto& p : starts) p = Vec3{dx(rng), dy(rng), dz(rng)};

  const double dt = 0.05;  // ~CFL 0.5 at |v|~1, h=0.1
  const std::vector<Vec3> gpu = gpu::backtraceBatch(g, v, starts, dt);
  REQUIRE(gpu.size() == starts.size());

  double maxErr = 0.0;
  for (size_t i = 0; i < starts.size(); ++i) {
    const Vec3 cpu = backtrace(g, v, starts[i], dt);
    for (int c = 0; c < 3; ++c) maxErr = std::max(maxErr, std::abs(cpu[c] - gpu[i][c]));
  }
  MESSAGE("max |CPU(double) - GPU(float)| backtrace over 4096 points = " << maxErr);
  CHECK(maxErr < 1e-3);  // RK4 with float samples vs double
}

TEST_CASE("Metal advectCovectorSLGPU matches the CPU semi-Lagrangian advection") {
  if (!gpu::metalAvailable()) {
    MESSAGE("no Metal device available -- skipping GPU parity check");
    return;
  }

  MacGrid g(24, 18, 14, 0.1, Vec3{-1.2, -0.9, -0.7});
  FaceField u = ops::zeroFaceField(g), v = ops::zeroFaceField(g);
  auto fill = [](std::vector<double>& a, double s) {
    for (size_t i = 0; i < a.size(); ++i) {
      const double t = static_cast<double>(i);
      a[i] = std::sin(0.09 * t + s) + 0.25 * std::cos(0.017 * t + 2.0 * s);
    }
  };
  fill(v.x, 0.5);
  fill(v.y, 1.5);
  fill(v.z, 2.5);  // the flow field
  fill(u.x, 3.5);
  fill(u.y, 4.5);
  fill(u.z, 5.5);  // the advected covector

  const double dt = 0.05;
  const FaceField cpu = advectCovectorSL(g, u, v, dt);
  const FaceField gpu = gpu::advectCovectorSLGPU(g, u, v, dt);

  REQUIRE(gpu.x.size() == cpu.x.size());
  REQUIRE(gpu.y.size() == cpu.y.size());
  REQUIRE(gpu.z.size() == cpu.z.size());
  double maxErr = 0.0;
  auto cmp = [&](const std::vector<double>& a, const std::vector<double>& b) {
    for (size_t i = 0; i < a.size(); ++i) maxErr = std::max(maxErr, std::abs(a[i] - b[i]));
  };
  cmp(cpu.x, gpu.x);
  cmp(cpu.y, gpu.y);
  cmp(cpu.z, gpu.z);
  MESSAGE("max |CPU(double) - GPU(float)| SL step over all faces = " << maxErr);
  CHECK(maxErr < 2e-3);  // finite-difference dPsi amplifies float error somewhat
}

TEST_CASE("Metal advectCovectorBFECCGpu matches the CPU BFECC advection") {
  if (!gpu::metalAvailable()) {
    MESSAGE("no Metal device available -- skipping GPU parity check");
    return;
  }

  MacGrid g(24, 18, 14, 0.1, Vec3{-1.2, -0.9, -0.7});
  FaceField u = ops::zeroFaceField(g), v = ops::zeroFaceField(g);
  auto fill = [](std::vector<double>& a, double s) {
    for (size_t i = 0; i < a.size(); ++i) {
      const double t = static_cast<double>(i);
      a[i] = std::sin(0.09 * t + s) + 0.25 * std::cos(0.017 * t + 2.0 * s);
    }
  };
  fill(v.x, 0.5);
  fill(v.y, 1.5);
  fill(v.z, 2.5);
  fill(u.x, 3.5);
  fill(u.y, 4.5);
  fill(u.z, 5.5);

  const double dt = 0.05;
  const FaceField cpu = advectCovectorBFECC(g, u, v, dt);
  const FaceField gpu = advectCovectorBFECCGpu(g, u, v, dt);

  REQUIRE(gpu.x.size() == cpu.x.size());
  double maxErr = 0.0;
  auto cmp = [&](const std::vector<double>& a, const std::vector<double>& b) {
    for (size_t i = 0; i < a.size(); ++i) maxErr = std::max(maxErr, std::abs(a[i] - b[i]));
  };
  cmp(cpu.x, gpu.x);
  cmp(cpu.y, gpu.y);
  cmp(cpu.z, gpu.z);
  MESSAGE("max |CPU(double) - GPU(float)| BFECC over all faces = " << maxErr);
  CHECK(maxErr < 3e-3);  // three SL sub-steps + round-trip cancellation, single precision
}

TEST_CASE("Metal advectCovectorBFECCResident (device-resident) matches the CPU BFECC") {
  if (!gpu::metalAvailable()) {
    MESSAGE("no Metal device available -- skipping GPU parity check");
    return;
  }

  MacGrid g(24, 18, 14, 0.1, Vec3{-1.2, -0.9, -0.7});
  FaceField u = ops::zeroFaceField(g), v = ops::zeroFaceField(g);
  auto fill = [](std::vector<double>& a, double s) {
    for (size_t i = 0; i < a.size(); ++i) {
      const double t = static_cast<double>(i);
      a[i] = std::sin(0.09 * t + s) + 0.25 * std::cos(0.017 * t + 2.0 * s);
    }
  };
  fill(v.x, 0.5);
  fill(v.y, 1.5);
  fill(v.z, 2.5);
  fill(u.x, 3.5);
  fill(u.y, 4.5);
  fill(u.z, 5.5);

  const double dt = 0.05;
  const FaceField cpu = advectCovectorBFECC(g, u, v, dt);
  const FaceField res = gpu::advectCovectorBFECCResident(g, u, v, dt);

  REQUIRE(res.x.size() == cpu.x.size());
  double maxErr = 0.0;
  auto cmp = [&](const std::vector<double>& a, const std::vector<double>& b) {
    for (size_t i = 0; i < a.size(); ++i) maxErr = std::max(maxErr, std::abs(a[i] - b[i]));
  };
  cmp(cpu.x, res.x);
  cmp(cpu.y, res.y);
  cmp(cpu.z, res.z);
  MESSAGE("max |CPU(double) - GPU(float)| resident BFECC over all faces = " << maxErr);
  CHECK(maxErr < 3e-3);  // must match the CPU path as closely as the hybrid does
}

// --- CF+MCM (characteristic-map) advection primitives ------------------------

namespace {
// A smooth non-identity map psi(x) = x + small displacement, so finite differences
// (dPsi) and both clamped/extrapolated samples are genuinely exercised.
std::vector<Vec3> smoothMap(const MacGrid& g, double amp, double phase) {
  std::vector<Vec3> m(g.numCells());
  for (int i = 0; i < g.nx(); ++i)
    for (int j = 0; j < g.ny(); ++j)
      for (int k = 0; k < g.nz(); ++k) {
        const Vec3 x = g.cellCenter(i, j, k);
        m[g.cellIndex(i, j, k)] = {x[0] + amp * std::sin(0.7 * j + phase),
                                   x[1] + amp * std::cos(0.5 * k + 0.3 * i + phase),
                                   x[2] + amp * std::sin(0.4 * i + 0.6 * j + phase)};
      }
  return m;
}
}  // namespace

TEST_CASE("Metal mapAdvect matches the CPU sampleCellVec3(backtrace) map advection") {
  if (!gpu::metalAvailable()) {
    MESSAGE("no Metal device available -- skipping GPU parity check");
    return;
  }
  MacGrid g(24, 18, 14, 0.1, Vec3{-1.2, -0.9, -0.7});
  FaceField v = ops::zeroFaceField(g);
  auto fill = [](std::vector<double>& a, double s) {
    for (size_t i = 0; i < a.size(); ++i)
      a[i] = std::sin(0.09 * double(i) + s) + 0.25 * std::cos(0.017 * double(i) + 2.0 * s);
  };
  fill(v.x, 0.5);
  fill(v.y, 1.5);
  fill(v.z, 2.5);
  const std::vector<Vec3> psi = smoothMap(g, 0.15, 0.8);
  const double dt = 0.05;

  std::vector<Vec3> cpu(g.numCells());
  for (int i = 0; i < g.nx(); ++i)
    for (int j = 0; j < g.ny(); ++j)
      for (int k = 0; k < g.nz(); ++k)
        cpu[g.cellIndex(i, j, k)] =
            sampleCellVec3(g, psi, backtrace(g, v, g.cellCenter(i, j, k), dt), /*extrap=*/true);

  const std::vector<Vec3> gpu = gpu::mapAdvect(g, psi, v, dt);
  REQUIRE(gpu.size() == cpu.size());
  double maxErr = 0.0;
  for (size_t c = 0; c < cpu.size(); ++c)
    for (int d = 0; d < 3; ++d) maxErr = std::max(maxErr, std::abs(cpu[c][d] - gpu[c][d]));
  MESSAGE("max |CPU(double) - GPU(float)| map advect = " << maxErr);
  CHECK(maxErr < 2e-5);  // a pure sample+backtrace, no dPsi amplification
}

TEST_CASE("Metal pullbackStored matches the CPU pullbackThroughMap") {
  if (!gpu::metalAvailable()) {
    MESSAGE("no Metal device available -- skipping GPU parity check");
    return;
  }
  MacGrid g(24, 18, 14, 0.1, Vec3{-1.2, -0.9, -0.7});
  FaceField source = ops::zeroFaceField(g);
  auto fill = [](std::vector<double>& a, double s) {
    for (size_t i = 0; i < a.size(); ++i)
      a[i] = std::sin(0.09 * double(i) + s) + 0.25 * std::cos(0.017 * double(i) + 2.0 * s);
  };
  fill(source.x, 3.5);
  fill(source.y, 4.5);
  fill(source.z, 5.5);
  const std::vector<Vec3> map = smoothMap(g, 0.12, 1.7);

  const FaceField cpu = pullbackThroughMap(g, map, source);
  const FaceField gpu = gpu::pullbackStored(g, map, source);
  REQUIRE(gpu.x.size() == cpu.x.size());
  double maxErr = 0.0;
  auto cmp = [&](const std::vector<double>& a, const std::vector<double>& b) {
    for (size_t i = 0; i < a.size(); ++i) maxErr = std::max(maxErr, std::abs(a[i] - b[i]));
  };
  cmp(cpu.x, gpu.x);
  cmp(cpu.y, gpu.y);
  cmp(cpu.z, gpu.z);
  MESSAGE("max |CPU(double) - GPU(float)| stored-map pullback = " << maxErr);
  CHECK(maxErr < 2e-3);  // dM finite-difference amplifies float error (like the SL pullback)
}

TEST_CASE("Metal mapDistortion matches the CPU rawDistortion metric") {
  if (!gpu::metalAvailable()) {
    MESSAGE("no Metal device available -- skipping GPU parity check");
    return;
  }
  MacGrid g(20, 18, 16, 0.1, Vec3{-1.0, -0.9, -0.8});
  // Non-inverse psi/phi so the round trips genuinely depart from the identity.
  const std::vector<Vec3> psi = smoothMap(g, 0.18, 0.4);
  const std::vector<Vec3> phi = smoothMap(g, 0.13, 2.1);

  // CPU reference == CfMcmSolver::rawDistortion (skip the 2-cell border, extrap).
  double maxD2 = 0.0;
  for (int i = 2; i < g.nx() - 2; ++i)
    for (int j = 2; j < g.ny() - 2; ++j)
      for (int k = 2; k < g.nz() - 2; ++k) {
        const int c = g.cellIndex(i, j, k);
        const Vec3 x = g.cellCenter(i, j, k);
        const Vec3 fb = sampleCellVec3(g, phi, psi[c], /*extrap=*/true);
        const Vec3 bf = sampleCellVec3(g, psi, phi[c], /*extrap=*/true);
        const Vec3 dbf = {x[0] - fb[0], x[1] - fb[1], x[2] - fb[2]};
        const Vec3 dfb = {x[0] - bf[0], x[1] - bf[1], x[2] - bf[2]};
        maxD2 = std::max(maxD2, std::max(dbf[0] * dbf[0] + dbf[1] * dbf[1] + dbf[2] * dbf[2],
                                         dfb[0] * dfb[0] + dfb[1] * dfb[1] + dfb[2] * dfb[2]));
      }
  const double cpu = std::sqrt(maxD2);
  const double gpu = gpu::mapDistortion(g, psi, phi);
  MESSAGE("CPU rawDistortion = " << cpu << "  GPU = " << gpu);
  REQUIRE(cpu > 0.01);  // genuinely distorted
  CHECK(std::abs(cpu - gpu) < 1e-4 * cpu + 1e-6);
}

TEST_CASE("Metal cfMcmAdvect matches a CPU CF+MCM advect-and-correct substep") {
  if (!gpu::metalAvailable()) {
    MESSAGE("no Metal device available -- skipping GPU parity check");
    return;
  }
  MacGrid g(20, 18, 16, 0.1, Vec3{-1.0, -0.9, -0.8});
  FaceField v = ops::zeroFaceField(g), u0 = ops::zeroFaceField(g);
  auto fill = [](std::vector<double>& a, double s) {
    for (size_t i = 0; i < a.size(); ++i)
      a[i] = std::sin(0.09 * double(i) + s) + 0.25 * std::cos(0.017 * double(i) + 2.0 * s);
  };
  fill(v.x, 0.5);
  fill(v.y, 1.5);
  fill(v.z, 2.5);
  fill(u0.x, 3.5);
  fill(u0.y, 4.5);
  fill(u0.z, 5.5);
  const double dt = 0.05;

  // CPU reference: one advectAndCorrect (Alg. 5 lines 4-9 + limiter) from identity
  // maps, PRE-projection -- exactly what cfMcmAdvect returns. Built from the public
  // CPU primitives so it does not need CfMcmSolver's private advectAndCorrect.
  std::vector<Vec3> psi(g.numCells()), phi(g.numCells());
  for (int i = 0; i < g.nx(); ++i)
    for (int j = 0; j < g.ny(); ++j)
      for (int k = 0; k < g.nz(); ++k) psi[g.cellIndex(i, j, k)] = phi[g.cellIndex(i, j, k)] =
                                           g.cellCenter(i, j, k);
  std::vector<Vec3> psiN(g.numCells()), phiN(g.numCells());
  for (int i = 0; i < g.nx(); ++i)
    for (int j = 0; j < g.ny(); ++j)
      for (int k = 0; k < g.nz(); ++k) {
        const int c = g.cellIndex(i, j, k);
        psiN[c] = sampleCellVec3(g, psi, backtrace(g, v, g.cellCenter(i, j, k), dt), true);  // line 4
        phiN[c] = backtrace(g, v, phi[c], -dt);                                              // line 6
      }
  const FaceField u1 = pullbackThroughMap(g, psiN, u0);   // line 5
  const FaceField ut0 = pullbackThroughMap(g, phiN, u1);  // line 7
  FaceField he = u0;
  for (size_t f = 0; f < he.x.size(); ++f) he.x[f] = 0.5 * (ut0.x[f] - u0.x[f]);
  for (size_t f = 0; f < he.y.size(); ++f) he.y[f] = 0.5 * (ut0.y[f] - u0.y[f]);
  for (size_t f = 0; f < he.z.size(); ++f) he.z[f] = 0.5 * (ut0.z[f] - u0.z[f]);
  const FaceField corr = pullbackThroughMap(g, psiN, he);  // line 9
  FaceField cpu = u1;
  for (size_t f = 0; f < cpu.x.size(); ++f) cpu.x[f] -= corr.x[f];
  for (size_t f = 0; f < cpu.y.size(); ++f) cpu.y[f] -= corr.y[f];
  for (size_t f = 0; f < cpu.z.size(); ++f) cpu.z[f] -= corr.z[f];
  cpu = bfeccLimit(g, u1, cpu);

  const int h = gpu::cfMcmUpload(g, u0);
  const FaceField gpu = gpu::cfMcmAdvect(h, v, dt, /*scratch=*/false, /*useLimiter=*/true);
  gpu::cfMcmFree(h);

  REQUIRE(gpu.x.size() == cpu.x.size());
  double maxErr = 0.0;
  auto cmp = [&](const std::vector<double>& a, const std::vector<double>& b) {
    for (size_t i = 0; i < a.size(); ++i) maxErr = std::max(maxErr, std::abs(a[i] - b[i]));
  };
  cmp(cpu.x, gpu.x);
  cmp(cpu.y, gpu.y);
  cmp(cpu.z, gpu.z);
  MESSAGE("max |CPU(double) - GPU(float)| CF+MCM substep = " << maxErr);
  CHECK(maxErr < 3e-3);  // three stored-map pullbacks (dM finite-diff) in single precision
}

// CfMcmSolver lives in MacCharacteristicMap.cpp, which bochner_fluid only compiles
// under BOCHNER_WITH_PETSC (MacProjection.cpp pulls in PetscSolver.h). Gate these two
// cases so test_metal_sample still links in the advertised PETSc-free Metal config.
#ifdef BOCHNER_WITH_PETSC
TEST_CASE("GPU CfMcmSolver tracks the CPU CfMcmSolver over a multi-step ring") {
  if (!gpu::metalAvailable()) {
    MESSAGE("no Metal device available -- skipping GPU parity check");
    return;
  }
  // Frame-count-only reinit (remapTol < 0) so the CPU-double and GPU-float solvers
  // reinit on exactly the same steps (a float distortion could otherwise cross the
  // accuracy gate a step apart). A small reinitEvery exercises the GPU reset path.
  MacGrid g(24, 24, 24, 2.0 / 24, Vec3{-1.0, -1.0, -1.0});
  const FaceField u0 = vortexRingFaceField(g, {0, 0, -0.2}, {0, 0, 1}, 0.6, 1.0, 0.15);
  CfMcmSolver cpu(g, u0, /*reinitEvery=*/5, {}, /*useLimiter=*/true, /*remapTol=*/-1.0,
                  /*secondOrder=*/true, BoundarySpec::allClosed(), /*useGpuAdvect=*/false);
  CfMcmSolver gpu(g, u0, /*reinitEvery=*/5, {}, /*useLimiter=*/true, /*remapTol=*/-1.0,
                  /*secondOrder=*/true, BoundarySpec::allClosed(), /*useGpuAdvect=*/true);

  const double dt = 0.02;  // fixed dt for both -> a deterministic step-by-step A/B
  double worstRel = 0.0, worstDiv = 0.0;
  for (int step = 0; step < 12; ++step) {
    cpu.step(dt);
    gpu.step(dt);
    const FaceField& a = cpu.velocity();
    const FaceField& b = gpu.velocity();
    double diff = 0.0, ref = 0.0;
    auto acc = [&](const std::vector<double>& x, const std::vector<double>& y) {
      for (size_t i = 0; i < x.size(); ++i) {
        diff = std::max(diff, std::abs(x[i] - y[i]));
        ref = std::max(ref, std::abs(x[i]));
      }
    };
    acc(a.x, b.x);
    acc(a.y, b.y);
    acc(a.z, b.z);
    worstRel = std::max(worstRel, diff / std::max(1e-9, ref));
    auto div = ops::divergence(g, b);
    for (double d : div) worstDiv = std::max(worstDiv, std::abs(d));
  }
  MESSAGE("12-step CF+MCM GPU vs CPU: worst rel vel diff = " << worstRel
                                                            << ", worst GPU divergence = " << worstDiv);
  CHECK(worstRel < 2e-2);   // float advection drift, bounded and non-growing
  CHECK(worstDiv < 1e-4);   // GPU field stays divergence-free (CPU projection)
}

TEST_CASE("GPU geometric-MG projection keeps CF+MCM stable and tracks the CPU") {
  if (!gpu::metalAvailable()) {
    MESSAGE("no Metal device available -- skipping GPU parity check");
    return;
  }
  // The projection is the whole point: the assembled pin-cell-0 MGPCG at a loose
  // float tol DESTABILIZED CF+MCM (the BiMocq accumulate amplified the residual
  // divergence -> blow-up). The unpinned geometric-MG at 1e-6 must instead stay
  // stable and track the CPU double projection. Isolate the projection: CPU
  // advection on both, GPU vs CPU only for the pressure solve.
  MacGrid g(24, 24, 24, 2.0 / 24, Vec3{-1.0, -1.0, -1.0});
  const FaceField u0 = vortexRingFaceField(g, {0, 0, -0.2}, {0, 0, 1}, 0.6, 1.0, 0.15);
  CfMcmSolver cpu(g, u0, /*reinitEvery=*/20, {}, /*useLimiter=*/true, /*remapTol=*/-1.0,
                  /*secondOrder=*/true, BoundarySpec::allClosed(), /*useGpuAdvect=*/false);
  CfMcmSolver gpu(g, u0, /*reinitEvery=*/20, {}, /*useLimiter=*/true, /*remapTol=*/-1.0,
                  /*secondOrder=*/true, BoundarySpec::allClosed(), /*useGpuAdvect=*/false);
  gpu.setGpuProjection(true);  // only the pressure solve differs

  const double dt = 0.02;
  double worstRel = 0.0, worstDiv = 0.0;
  for (int step = 0; step < 12; ++step) {
    cpu.step(dt);
    gpu.step(dt);
    const FaceField& a = cpu.velocity();
    const FaceField& b = gpu.velocity();
    double diff = 0.0, ref = 0.0;
    auto acc = [&](const std::vector<double>& x, const std::vector<double>& y) {
      for (size_t i = 0; i < x.size(); ++i) {
        diff = std::max(diff, std::abs(x[i] - y[i]));
        ref = std::max(ref, std::abs(x[i]));
      }
    };
    acc(a.x, b.x);
    acc(a.y, b.y);
    acc(a.z, b.z);
    worstRel = std::max(worstRel, diff / std::max(1e-9, ref));
    for (double d : ops::divergence(g, b)) worstDiv = std::max(worstDiv, std::abs(d));
  }
  MESSAGE("12-step CF+MCM GPU-projection vs CPU: worst rel vel diff = "
          << worstRel << ", worst GPU divergence = " << worstDiv);
  CHECK(worstRel < 1e-2);  // stable: bounded float drift, no blow-up (the pinned MGPCG failed here)
  CHECK(worstDiv < 1e-4);  // divergence-free -> the 1e-6 float geometric solve is tight enough
}
#endif  // BOCHNER_WITH_PETSC

TEST_CASE("Metal mgpcgSolvePoisson matches the CPU MGPCG (float precision)") {
  if (!gpu::metalAvailable()) {
    MESSAGE("no Metal device available -- skipping GPU parity check");
    return;
  }

  const int n = 24;
  const int N = n * n * n;
  const double h = 1.0 / n;
  const SpMat A = robinPoissonT(n, h);
  const std::vector<char> active(N, 1);
  const MgHierarchy H = buildPoissonMgHierarchy(n, n, n, A, active);

  std::mt19937 rng(7);
  std::uniform_real_distribution<double> d(-1.0, 1.0);
  std::vector<double> xstar(N);
  for (double& v : xstar) v = d(rng);
  std::vector<double> b(N, 0.0);
  for (int r = 0; r < N; ++r)
    for (int p = A.rowStart[r]; p < A.rowStart[r + 1]; ++p) b[r] += A.val[p] * xstar[A.col[p]];

  std::vector<double> xc(N, 0.0);
  double relc = 1.0;
  const int itc = mgpcgSolve(H, b, xc, 1e-6, 200, &relc);

  const int handle = gpu::uploadPoisson(H);
  REQUIRE(handle >= 0);
  std::vector<double> xg(N, 0.0);
  double relg = 1.0;
  const int itg = gpu::mgpcgSolvePoisson(handle, b, xg, /*tol=*/1e-4, /*maxit=*/200, &relg);
  gpu::freePoisson(handle);

  MESSAGE("MGPCG  CPU: " << itc << " iters, rel " << relc << "  |  GPU(float): " << itg
                         << " iters, rel " << relg);
  CHECK(relg < 1e-3);   // GPU is single precision -> converges to a float residual
  CHECK(itg < 40);      // and does so in few (mesh-independent) iterations
  double num = 0.0, den = 0.0;
  for (int i = 0; i < N; ++i) { num += (xc[i] - xg[i]) * (xc[i] - xg[i]); den += xc[i] * xc[i]; }
  CHECK(std::sqrt(num / den) < 1e-2);  // GPU solution matches the CPU one to float tol
}

#ifdef BOCHNER_WITH_PETSC
TEST_CASE("GPU MGPCG obstacle projection matches the CPU double (solid mask + pin)") {
  if (!gpu::metalAvailable()) {
    MESSAGE("no Metal device available -- skipping GPU parity check");
    return;
  }
  // The production obstacle route (MacProjector::project -> assembled
  // pressureLaplacianObstacle -> gpu::mgpcgSolvePoisson) with a solid mask (inactive
  // cells) and a closed-box single-cell pin -- the path the all-active mgpcg case
  // above never touches. Closed box + interior sphere -> exactly one fluid cell is
  // pinned; the sphere cells are inactive with dropped face couplings.
  MacGrid g(24, 24, 16, 1.0 / 24, Vec3{0, 0, 0});
  SolidMask solid = sphereMask(g, {0.5, 0.5, 0.33}, 0.15);
  REQUIRE(std::count(solid.begin(), solid.end(), std::uint8_t{1}) > 0);  // sphere hit some cells

  FaceField u = ops::zeroFaceField(g);
  auto fill = [](std::vector<double>& a, double s) {
    for (size_t i = 0; i < a.size(); ++i) a[i] = std::sin(0.13 * i + s) + 0.3 * std::cos(0.05 * i);
  };
  fill(u.x, 0.2);
  fill(u.y, 1.0);
  fill(u.z, 2.1);

  // Independent projectors (each owns its warm start) for a clean CPU-vs-GPU A/B.
  MacProjector cpu(g, {}, BoundarySpec::allClosed(), solid);
  MacProjector gpu(g, {}, BoundarySpec::allClosed(), solid);
  gpu.setGpuProjection(true);
  const FaceField uc = cpu.project(u);
  const FaceField ug = gpu.project(u);

  double worst = 0.0, ref = 1e-30;
  auto cmp = [&](const std::vector<double>& a, const std::vector<double>& b) {
    for (size_t i = 0; i < a.size(); ++i) {
      worst = std::max(worst, std::abs(a[i] - b[i]));
      ref = std::max(ref, std::abs(a[i]));
    }
  };
  cmp(uc.x, ug.x);
  cmp(uc.y, ug.y);
  cmp(uc.z, ug.z);

  // Matching the authoritative CPU double projection (itself tested divergence-free
  // in test_obstacle_flow) to single-precision tolerance is the correctness signal
  // for the GPU obstacle path -- an absolute div check would trip on the closed-box
  // pin cell and the uncorrected solid-boundary faces, which retain div by design.
  MESSAGE("obstacle MGPCG: worst |CPU-GPU| = " << worst << " (rel " << worst / ref << ")");
  CHECK(worst / ref < 2e-2);   // same projected velocity to single-precision tolerance
}
#endif  // BOCHNER_WITH_PETSC

TEST_CASE("Metal poissonGeomSolve matches the CPU geometric-MG (unpinned Neumann)") {
  if (!gpu::metalAvailable()) {
    MESSAGE("no Metal device available -- skipping GPU parity check");
    return;
  }
  // The closed-box all-Neumann Poisson, solved UNPINNED by matrix-free geometric
  // V-cycles (the route that beats the CPU geometric MG for the closed box). Both
  // solutions are defined up to a constant (the null space), so compare after
  // removing the mean; the residual is null-space-independent.
  const int nx = 24, ny = 20, nz = 16, N = nx * ny * nz;
  const double h = 2.0 / nx;
  std::vector<double> rhs(N);
  double mean = 0.0;
  for (int i = 0; i < N; ++i) {
    rhs[i] = std::sin(0.11 * i + 0.3) + 0.4 * std::cos(0.027 * i);
    mean += rhs[i];
  }
  mean /= N;
  for (double& r : rhs) r -= mean;  // mean-zero -> compatible all-Neumann RHS

  PoissonMgOptions opts;
  opts.tol = 1e-6;
  std::vector<double> phiC(N, 0.0);
  const PoissonMgResult rc = poissonVcycleSolve(nx, ny, nz, h, rhs, phiC, opts);

  const int handle = gpu::uploadPoissonGeom(nx, ny, nz, h);
  REQUIRE(handle >= 0);
  std::vector<double> phiG(N, 0.0);
  double relg = 1.0;
  const int itg = gpu::poissonGeomSolve(handle, rhs, phiG, /*tol=*/1e-6, /*maxCycles=*/100, opts.nu1,
                                        opts.nu2, opts.coarseSweeps, opts.omega, &relg);
  gpu::freePoissonGeom(handle);

  MESSAGE("geom-MG  CPU: " << rc.cycles << " cyc, rel " << rc.relResidual << "  |  GPU(float): "
                           << itg << " cyc, rel " << relg);
  CHECK(relg < 1e-5);  // float geometric V-cycles reach a tight residual (no pin stall)
  CHECK(itg < 40);     // mesh-independent, comparable to the CPU cycle count
  double mc = 0.0, mgm = 0.0;
  for (int i = 0; i < N; ++i) { mc += phiC[i]; mgm += phiG[i]; }
  mc /= N; mgm /= N;
  double num = 0.0, den = 0.0;
  for (int i = 0; i < N; ++i) {
    const double dc = (phiC[i] - mc) - (phiG[i] - mgm);
    num += dc * dc;
    den += (phiC[i] - mc) * (phiC[i] - mc);
  }
  MESSAGE("geom-MG solution rel-diff (mean-removed) = " << std::sqrt(num / den));
  CHECK(std::sqrt(num / den) < 5e-3);  // same solution as the CPU geometric MG, up to the constant
}

// Faithful reference red-black Gauss-Seidel (SOR) smoother of E x = b, mirroring
// the (file-private) smoother inside GaugeMultigrid's vcycleSolve. The GPU
// smoother must reproduce this sweep for sweep to float tolerance. Stage 2c's
// V-cycle parity test (against the public vcycleSolve) catches any systematic
// mirroring error end-to-end.
static void refSmooth(const GaugeLattice& L, const std::vector<std::complex<double>>& b,
                      std::vector<std::complex<double>>& x, int sweeps, double omega) {
  using cd = std::complex<double>;
  auto xiL = [&](int i, int j, int k) { return (i * L.ly + j) * L.lz + k; };
  auto yiL = [&](int i, int j, int k) { return (i * (L.periodic ? L.ly : L.ly - 1) + j) * L.lz + k; };
  auto ziL = [&](int i, int j, int k) { return (i * L.ly + j) * (L.periodic ? L.lz : L.lz - 1) + k; };
  auto deg = [&](int i, int j, int k) {
    if (L.periodic) return 6;
    return (i > 0) + (i + 1 < L.lx) + (j > 0) + (j + 1 < L.ly) + (k > 0) + (k + 1 < L.lz);
  };
  for (int s = 0; s < sweeps; ++s)
    for (int color = 0; color <= 1; ++color)
      for (int i = 0; i < L.lx; ++i)
        for (int j = 0; j < L.ly; ++j)
          for (int k = 0; k < L.lz; ++k) {
            if (((i + j + k) & 1) != color) continue;
            const int c = L.index(i, j, k);
            const int d = deg(i, j, k);
            if (d == 0) continue;
            cd sum = b[c];
            if (L.periodic) {
              const int im = (i - 1 + L.lx) % L.lx, ip = (i + 1) % L.lx;
              const int jm = (j - 1 + L.ly) % L.ly, jp = (j + 1) % L.ly;
              const int km = (k - 1 + L.lz) % L.lz, kp = (k + 1) % L.lz;
              sum += L.w * L.tx[xiL(im, j, k)] * x[L.index(im, j, k)];
              sum += L.w * std::conj(L.tx[xiL(i, j, k)]) * x[L.index(ip, j, k)];
              sum += L.w * L.ty[yiL(i, jm, k)] * x[L.index(i, jm, k)];
              sum += L.w * std::conj(L.ty[yiL(i, j, k)]) * x[L.index(i, jp, k)];
              sum += L.w * L.tz[ziL(i, j, km)] * x[L.index(i, j, km)];
              sum += L.w * std::conj(L.tz[ziL(i, j, k)]) * x[L.index(i, j, kp)];
            } else {
              if (i > 0) sum += L.w * L.tx[xiL(i - 1, j, k)] * x[L.index(i - 1, j, k)];
              if (i + 1 < L.lx) sum += L.w * std::conj(L.tx[xiL(i, j, k)]) * x[L.index(i + 1, j, k)];
              if (j > 0) sum += L.w * L.ty[yiL(i, j - 1, k)] * x[L.index(i, j - 1, k)];
              if (j + 1 < L.ly) sum += L.w * std::conj(L.ty[yiL(i, j, k)]) * x[L.index(i, j + 1, k)];
              if (k > 0) sum += L.w * L.tz[ziL(i, j, k - 1)] * x[L.index(i, j, k - 1)];
              if (k + 1 < L.lz) sum += L.w * std::conj(L.tz[ziL(i, j, k)]) * x[L.index(i, j, k + 1)];
            }
            x[c] = (1.0 - omega) * x[c] + omega * sum / (L.w * static_cast<double>(d));
          }
}

TEST_CASE("Metal connectionSmooth matches the CPU red-black GS smoother (complex)") {
  if (!gpu::metalAvailable()) {
    MESSAGE("no Metal device available -- skipping GPU parity check");
    return;
  }
  using cd = std::complex<double>;
  std::mt19937 rng(2024);
  std::uniform_real_distribution<double> d(-1.0, 1.0);

  auto runCase = [&](const GaugeLattice& lat, int sweeps, double omega, const char* label) {
    const int n = static_cast<int>(lat.numNodes());
    std::vector<cd> b(n), x0(n);
    for (cd& z : b) z = cd(d(rng), d(rng));
    for (cd& z : x0) z = cd(d(rng), d(rng));

    std::vector<cd> xcpu = x0;
    refSmooth(lat, b, xcpu, sweeps, omega);
    const std::vector<cd> xgpu = gpu::connectionSmooth(lat.lx, lat.ly, lat.lz, lat.periodic, lat.w,
                                                       lat.tx, lat.ty, lat.tz, b, x0, sweeps, omega);
    REQUIRE(xgpu.size() == xcpu.size());
    double maxErr = 0.0, ref = 0.0;
    for (int i = 0; i < n; ++i) {
      maxErr = std::max(maxErr, std::abs(xcpu[i] - xgpu[i]));
      ref = std::max(ref, std::abs(xcpu[i]));
    }
    MESSAGE(label << ": max |CPU - GPU| = " << maxErr << " (ref max |x| = " << ref << ")");
    CHECK(maxErr < 1e-3 * std::max(1.0, ref));
  };

  // Open lattice from MAC faces (exercises the open branch + shrunk yi/zi strides).
  MacGrid g(12, 10, 8, 0.1, Vec3{-0.6, -0.5, -0.4});
  FaceField theta = ops::zeroFaceField(g);
  auto fill = [](std::vector<double>& a, double s) {
    for (size_t i = 0; i < a.size(); ++i) a[i] = 0.7 * std::sin(0.13 * i + s);
  };
  fill(theta.x, 0.3);
  fill(theta.y, 1.1);
  fill(theta.z, 2.0);
  runCase(gaugeLatticeFromFaces(g, theta), /*sweeps=*/5, /*omega=*/1.0, "open GS");

  // Periodic (3-torus) lattice with random link angles (exercises the wrap branch).
  const int p = 8;
  std::vector<double> lkx(p * p * p), lky(p * p * p), lkz(p * p * p);
  for (double& a : lkx) a = d(rng);
  for (double& a : lky) a = d(rng);
  for (double& a : lkz) a = d(rng);
  runCase(gaugeLatticePeriodic(p, p, p, 1.0 / (0.1 * 0.1), lkx, lky, lkz), /*sweeps=*/6,
          /*omega=*/1.1, "periodic SOR");
}

// Faithful reference covariant prolongation, mirroring the (file-private)
// prolong + gatherSources inside the CPU V-cycle. The GPU prolong must match this
// to float tolerance.
static std::vector<std::complex<double>> refProlong(const GaugeLattice& f,
                                                    const std::vector<std::complex<double>>& coarse) {
  using cd = std::complex<double>;
  const int cly = f.ly / 2, clz = f.lz / 2;
  auto cidx = [&](int I, int J, int K) { return (I * cly + J) * clz + K; };
  auto xiL = [&](int i, int j, int k) { return (i * f.ly + j) * f.lz + k; };
  auto yiL = [&](int i, int j, int k) { return (i * (f.periodic ? f.ly : f.ly - 1) + j) * f.lz + k; };
  auto ziL = [&](int i, int j, int k) { return (i * f.ly + j) * (f.periodic ? f.lz : f.lz - 1) + k; };
  auto gather = [&](int i, int j, int k, int* idx, cd* tr) {
    int cnt = 0;
    if (i & 1) {
      const int lo = f.periodic ? (i - 1 + f.lx) % f.lx : i - 1;
      idx[cnt] = f.index(lo, j, k); tr[cnt] = f.tx[xiL(lo, j, k)]; ++cnt;
      if (f.periodic) { idx[cnt] = f.index((i + 1) % f.lx, j, k); tr[cnt] = std::conj(f.tx[xiL(i, j, k)]); ++cnt; }
      else if (i + 1 < f.lx) { idx[cnt] = f.index(i + 1, j, k); tr[cnt] = std::conj(f.tx[xiL(i, j, k)]); ++cnt; }
    }
    if (j & 1) {
      const int lo = f.periodic ? (j - 1 + f.ly) % f.ly : j - 1;
      idx[cnt] = f.index(i, lo, k); tr[cnt] = f.ty[yiL(i, lo, k)]; ++cnt;
      if (f.periodic) { idx[cnt] = f.index(i, (j + 1) % f.ly, k); tr[cnt] = std::conj(f.ty[yiL(i, j, k)]); ++cnt; }
      else if (j + 1 < f.ly) { idx[cnt] = f.index(i, j + 1, k); tr[cnt] = std::conj(f.ty[yiL(i, j, k)]); ++cnt; }
    }
    if (k & 1) {
      const int lo = f.periodic ? (k - 1 + f.lz) % f.lz : k - 1;
      idx[cnt] = f.index(i, j, lo); tr[cnt] = f.tz[ziL(i, j, lo)]; ++cnt;
      if (f.periodic) { idx[cnt] = f.index(i, j, (k + 1) % f.lz); tr[cnt] = std::conj(f.tz[ziL(i, j, k)]); ++cnt; }
      else if (k + 1 < f.lz) { idx[cnt] = f.index(i, j, k + 1); tr[cnt] = std::conj(f.tz[ziL(i, j, k)]); ++cnt; }
    }
    return cnt;
  };
  std::vector<cd> fine(static_cast<size_t>(f.numNodes()), cd(0.0, 0.0));
  for (int i = 0; i < f.lx; i += 2)
    for (int j = 0; j < f.ly; j += 2)
      for (int k = 0; k < f.lz; k += 2) fine[f.index(i, j, k)] = coarse[cidx(i / 2, j / 2, k / 2)];
  for (int pass = 1; pass <= 3; ++pass)
    for (int i = 0; i < f.lx; ++i)
      for (int j = 0; j < f.ly; ++j)
        for (int k = 0; k < f.lz; ++k) {
          if ((i & 1) + (j & 1) + (k & 1) != pass) continue;
          int idx[6]; cd tr[6];
          const int cnt = gather(i, j, k, idx, tr);
          cd sum(0.0, 0.0);
          for (int t = 0; t < cnt; ++t) sum += tr[t] * fine[idx[t]];
          fine[f.index(i, j, k)] = sum / static_cast<double>(cnt);
        }
  return fine;
}

TEST_CASE("Metal connectionProlong/Restrict: prolong parity + restrict = adjoint") {
  if (!gpu::metalAvailable()) {
    MESSAGE("no Metal device available -- skipping GPU parity check");
    return;
  }
  using cd = std::complex<double>;
  std::mt19937 rng(555);
  std::uniform_real_distribution<double> d(-1.0, 1.0);

  auto hdot = [](const std::vector<cd>& a, const std::vector<cd>& b) {
    cd s(0.0, 0.0);
    for (size_t i = 0; i < a.size(); ++i) s += std::conj(a[i]) * b[i];
    return s;
  };

  auto runCase = [&](const GaugeLattice& lat, const char* label) {
    const int nf = static_cast<int>(lat.numNodes());
    const int nc = (lat.lx / 2) * (lat.ly / 2) * (lat.lz / 2);

    // (1) Prolong parity vs the CPU reference.
    std::vector<cd> coarse(nc), fine(nf);
    for (cd& z : coarse) z = cd(d(rng), d(rng));
    for (cd& z : fine) z = cd(d(rng), d(rng));

    const std::vector<cd> pcpu = refProlong(lat, coarse);
    const std::vector<cd> pgpu =
        gpu::connectionProlong(lat.lx, lat.ly, lat.lz, lat.periodic, lat.tx, lat.ty, lat.tz, coarse);
    REQUIRE(pgpu.size() == pcpu.size());
    double maxErr = 0.0, ref = 0.0;
    for (int i = 0; i < nf; ++i) {
      maxErr = std::max(maxErr, std::abs(pcpu[i] - pgpu[i]));
      ref = std::max(ref, std::abs(pcpu[i]));
    }
    MESSAGE(label << " prolong: max |CPU - GPU| = " << maxErr << " (ref max = " << ref << ")");
    CHECK(maxErr < 1e-3 * std::max(1.0, ref));

    // (2) restrict is the EXACT ADJOINT of prolong: <f, P c> = <R f, c>. This
    // pins the GPU restrict against the GPU prolong (verified in (1) to match the
    // CPU), so it needs no separate hand-mirrored restrict reference.
    const std::vector<cd> Rf =
        gpu::connectionRestrict(lat.lx, lat.ly, lat.lz, lat.periodic, lat.tx, lat.ty, lat.tz, fine);
    REQUIRE(static_cast<int>(Rf.size()) == nc);
    const cd lhs = hdot(fine, pgpu);   // <f, P c>_fine
    const cd rhs = hdot(Rf, coarse);   // <R f, c>_coarse
    const double adj = std::abs(lhs - rhs);
    MESSAGE(label << " adjoint: |<f,Pc> - <Rf,c>| = " << adj << " (|<f,Pc>| = " << std::abs(lhs)
                  << ")");
    CHECK(adj < 1e-3 * std::max(1.0, std::abs(lhs)));
  };

  // Open lattice from MAC faces (open branch + shrunk yi/zi strides).
  MacGrid g(12, 10, 8, 0.1, Vec3{-0.6, -0.5, -0.4});
  FaceField theta = ops::zeroFaceField(g);
  auto fillt = [](std::vector<double>& a, double s) {
    for (size_t i = 0; i < a.size(); ++i) a[i] = 0.7 * std::sin(0.13 * i + s);
  };
  fillt(theta.x, 0.3);
  fillt(theta.y, 1.1);
  fillt(theta.z, 2.0);
  runCase(gaugeLatticeFromFaces(g, theta), "open");

  // Periodic (3-torus) lattice with random link angles (wrap branch).
  const int p = 8;
  std::vector<double> lkx(p * p * p), lky(p * p * p), lkz(p * p * p);
  for (double& a : lkx) a = d(rng);
  for (double& a : lky) a = d(rng);
  for (double& a : lkz) a = d(rng);
  runCase(gaugeLatticePeriodic(p, p, p, 1.0 / (0.1 * 0.1), lkx, lky, lkz), "periodic");
}

// Convert a CPU gauge-MG level pyramid into the dependency-free upload form.
static std::vector<gpu::GaugeLevelData> toGpuLevels(const std::vector<GaugeLattice>& levels) {
  std::vector<gpu::GaugeLevelData> out;
  for (const GaugeLattice& L : levels)
    out.push_back({L.lx, L.ly, L.lz, L.periodic, L.w, L.tx, L.ty, L.tz});
  return out;
}

TEST_CASE("Metal vcycleSolveGauge matches the CPU gauge V-cycle (float precision)") {
  if (!gpu::metalAvailable()) {
    MESSAGE("no Metal device available -- skipping GPU parity check");
    return;
  }
  using cd = std::complex<double>;
  std::mt19937 rng(31337);
  std::uniform_real_distribution<double> d(-1.0, 1.0);

  auto runCase = [&](const GaugeLattice& lat, const char* label) {
    const int n = static_cast<int>(lat.numNodes());
    std::vector<cd> b(n);
    for (cd& z : b) z = cd(d(rng), d(rng));

    MgOptions opts;  // nu1=nu2=2, coarseSweeps=30, omega=1.0
    // CPU near-exact reference solution (double, tight tolerance).
    std::vector<cd> xc(n, cd(0.0, 0.0));
    MgOptions optc = opts;
    optc.tol = 1e-11;
    optc.maxCycles = 200;
    const MgResult rc = vcycleSolve(lat, b, xc, optc);

    // GPU single-precision solve on the SAME level pyramid.
    const std::vector<GaugeLattice> levels = buildGaugeLevels(lat);
    const int handle = gpu::uploadGauge(toGpuLevels(levels));
    REQUIRE(handle >= 0);
    std::vector<cd> xg(n, cd(0.0, 0.0));
    double relg = 1.0;
    const int itg = gpu::vcycleSolveGauge(handle, b, xg, opts.nu1, opts.nu2, opts.coarseSweeps,
                                          opts.omega, /*maxCycles=*/30, /*tol=*/1e-5, &relg);
    gpu::freeGauge(handle);

    double num = 0.0, den = 0.0;
    for (int i = 0; i < n; ++i) {
      num += std::norm(xc[i] - xg[i]);
      den += std::norm(xc[i]);
    }
    const double solErr = std::sqrt(num / den);
    MESSAGE(label << ": CPU " << rc.cycles << " cyc (rel " << rc.relResidual << ") | GPU(float) "
                  << itg << " cyc (rel " << relg << "), solution rel-diff " << solErr);
    CHECK(relg < 1e-4);      // single-precision V-cycle reaches a float residual
    CHECK(itg < 20);         // in few, mesh-independent cycles (like the CPU)
    CHECK(solErr < 1e-2);    // and lands on the CPU's solution to float tolerance
  };

  // Open lattice from MAC faces -- the actual fluid extraction path.
  MacGrid g(16, 12, 8, 0.1, Vec3{-0.8, -0.6, -0.4});
  FaceField theta = ops::zeroFaceField(g);
  auto fillt = [](std::vector<double>& a, double s) {
    for (size_t i = 0; i < a.size(); ++i) a[i] = 0.5 * std::sin(0.13 * i + s);
  };
  fillt(theta.x, 0.3);
  fillt(theta.y, 1.1);
  fillt(theta.z, 2.0);
  runCase(gaugeLatticeFromFaces(g, theta), "open");

  // Periodic (3-torus) lattice with random links (wrap branch, /4 coarsening).
  const int p = 8;
  std::vector<double> lkx(p * p * p), lky(p * p * p), lkz(p * p * p);
  for (double& a : lkx) a = 0.6 * d(rng);
  for (double& a : lky) a = 0.6 * d(rng);
  for (double& a : lkz) a = 0.6 * d(rng);
  runCase(gaugeLatticePeriodic(p, p, p, 1.0 / (0.1 * 0.1), lkx, lky, lkz), "periodic");
}

TEST_CASE("Metal lobpcgSolveGauge matches the CPU gauge eigensolver (float precision)") {
  if (!gpu::metalAvailable()) {
    MESSAGE("no Metal device available -- skipping GPU parity check");
    return;
  }
  using cd = std::complex<double>;

  // Open lattice from MAC faces with a generic connection -> a simple (non-
  // degenerate) smallest eigenpair, so CPU and GPU return the same complex line.
  MacGrid g(16, 12, 8, 0.1, Vec3{-0.8, -0.6, -0.4});
  FaceField theta = ops::zeroFaceField(g);
  auto fillt = [](std::vector<double>& a, double s) {
    for (size_t i = 0; i < a.size(); ++i) a[i] = 0.5 * std::sin(0.13 * i + s) + 0.2 * std::cos(0.05 * i);
  };
  fillt(theta.x, 0.3);
  fillt(theta.y, 1.1);
  fillt(theta.z, 2.0);
  const GaugeLattice lat = gaugeLatticeFromFaces(g, theta);
  const int n = static_cast<int>(lat.numNodes());

  // CPU double reference (the authoritative solver).
  GaugeEigenOptions co;
  co.tol = 1e-8;
  co.maxIters = 300;
  const GaugeEigenResult rc = smallestEigenpairGaugeMG(lat, nullptr, co);

  // GPU single-precision covMG-LOBPCG on the same hierarchy.
  const int handle = gpu::uploadGauge(toGpuLevels(buildGaugeLevels(lat)));
  REQUIRE(handle >= 0);
  const gpu::GaugeEigenGpu rg =
      gpu::lobpcgSolveGauge(handle, /*guess=*/{}, /*maxIters=*/100, /*tol=*/1e-4, co.precCycles,
                          co.mg.nu1, co.mg.nu2, co.mg.coarseSweeps, co.mg.omega);
  gpu::freeGauge(handle);

  const double evalErr = std::abs(rg.eigenvalue - rc.eigenvalue) / std::abs(rc.eigenvalue);
  cd ip(0.0, 0.0);  // <vc, vg> (both unit-norm)
  for (int i = 0; i < n; ++i) ip += std::conj(rc.vector[i]) * rg.vector[i];
  const double lineDist = std::sqrt(std::max(0.0, 1.0 - std::norm(ip)));

  MESSAGE("eig CPU " << rc.eigenvalue << " (" << rc.iterations << " it, res " << rc.residual
                     << ") | GPU(float) " << rg.eigenvalue << " (" << rg.iterations << " it, res "
                     << rg.residual << ")");
  MESSAGE("eigenvalue rel-err = " << evalErr << ", eigenvector complex-line distance = "
                                  << lineDist);
  // lobpcgSolveGauge stops on Rayleigh-quotient stagnation, i.e. at the float
  // residual floor (~1e-3 here), so the eigenVALUE and eigenVECTOR are the quality
  // signals, not the raw residual.
  CHECK(rg.residual < 5e-3);        // settles at the float eigen-residual floor
  CHECK(evalErr < 1e-3);            // eigenvalue matches to ~float tolerance
  CHECK(lineDist < 5e-2);           // same eigenvector line (up to global phase)
}

TEST_CASE("Metal lobpcgSolveGauge warm-starts and guards a zero/non-finite guess") {
  if (!gpu::metalAvailable()) {
    MESSAGE("no Metal device available -- skipping GPU parity check");
    return;
  }
  using cd = std::complex<double>;
  // The viewer path: lobpcgSolveGauge is re-run every frame warm-started from the
  // previous filament eigenvector (obstacle_main.cpp / main.cpp pass prevVec). The
  // cold-start case above never exercises that branch or its zero/non-finite guard.
  MacGrid g(16, 12, 8, 0.1, Vec3{-0.8, -0.6, -0.4});
  FaceField theta = ops::zeroFaceField(g);
  auto fillt = [](std::vector<double>& a, double s) {
    for (size_t i = 0; i < a.size(); ++i)
      a[i] = 0.5 * std::sin(0.13 * i + s) + 0.2 * std::cos(0.05 * i);
  };
  fillt(theta.x, 0.3);
  fillt(theta.y, 1.1);
  fillt(theta.z, 2.0);
  const GaugeLattice lat = gaugeLatticeFromFaces(g, theta);
  const int n = static_cast<int>(lat.numNodes());

  GaugeEigenOptions co;
  co.tol = 1e-8;
  co.maxIters = 300;
  const GaugeEigenResult rc = smallestEigenpairGaugeMG(lat, nullptr, co);

  const int handle = gpu::uploadGauge(toGpuLevels(buildGaugeLevels(lat)));
  REQUIRE(handle >= 0);
  auto solve = [&](const std::vector<cd>& guess) {
    return gpu::lobpcgSolveGauge(handle, guess, /*maxIters=*/100, /*tol=*/1e-4, co.precCycles,
                               co.mg.nu1, co.mg.nu2, co.mg.coarseSweeps, co.mg.omega);
  };
  auto lineDistTo = [&](const std::vector<cd>& v) {
    cd ip(0.0, 0.0);
    for (int i = 0; i < n; ++i) ip += std::conj(rc.vector[i]) * v[i];
    return std::sqrt(std::max(0.0, 1.0 - std::norm(ip)));
  };
  auto evalErrOf = [&](double ev) { return std::abs(ev - rc.eigenvalue) / std::abs(rc.eigenvalue); };

  const gpu::GaugeEigenGpu cold = solve({});

  // Warm start = the CPU eigenvector at an arbitrary global phase (the solver
  // returns an arbitrary phase each frame) plus a small perturbation (the field
  // drifts frame-to-frame). It must converge in FEWER iters and to the same line.
  std::mt19937 rng(999);
  std::uniform_real_distribution<double> d(-1.0, 1.0);
  const cd phase = std::polar(1.0, 0.7);
  std::vector<cd> warmGuess(n);
  for (int i = 0; i < n; ++i) warmGuess[i] = phase * rc.vector[i] + cd(0.02 * d(rng), 0.02 * d(rng));
  const gpu::GaugeEigenGpu warm = solve(warmGuess);

  // Zero and non-finite guesses must not yield a false-converged zero eigenpair --
  // the guard resets to all-ones (mirrors GaugeEigen.cpp:91 / EigenSolver.cpp).
  const gpu::GaugeEigenGpu zero = solve(std::vector<cd>(n, cd(0.0, 0.0)));
  std::vector<cd> nanGuess(n, cd(1.0, 0.0));
  nanGuess[0] = cd(std::nan(""), 0.0);
  const gpu::GaugeEigenGpu nan = solve(nanGuess);
  gpu::freeGauge(handle);

  MESSAGE("cold " << cold.iterations << " it -> warm " << warm.iterations << " it; line-dist warm "
                  << lineDistTo(warm.vector) << ", zero " << lineDistTo(zero.vector) << ", nan "
                  << lineDistTo(nan.vector));
  CHECK(warm.iterations < cold.iterations);   // warm start converges in fewer iters
  CHECK(evalErrOf(warm.eigenvalue) < 1e-3);   // to the same eigenvalue
  CHECK(lineDistTo(warm.vector) < 5e-2);      // and the same eigenvector line
  CHECK(std::isfinite(zero.eigenvalue));      // zero guess -> guard -> valid solve
  CHECK(evalErrOf(zero.eigenvalue) < 1e-3);
  CHECK(std::isfinite(nan.eigenvalue));       // non-finite guess -> guard -> valid solve
  CHECK(evalErrOf(nan.eigenvalue) < 1e-3);
}

TEST_CASE("Metal gauge handle registry reuses freed slots (bounded per-frame growth)") {
  if (!gpu::metalAvailable()) {
    MESSAGE("no Metal device available -- skipping GPU parity check");
    return;
  }
  // The viewers upload+free a gauge hierarchy every extraction frame; the registry
  // must reuse freed slots rather than grow one null slot per frame forever.
  MacGrid g(8, 8, 8, 0.1, Vec3{-0.4, -0.4, -0.4});
  FaceField theta = ops::zeroFaceField(g);
  for (size_t i = 0; i < theta.x.size(); ++i) theta.x[i] = 0.1 * std::sin(0.2 * i);
  const auto levels = toGpuLevels(buildGaugeLevels(gaugeLatticeFromFaces(g, theta)));

  const int h0 = gpu::uploadGauge(levels);
  const int h1 = gpu::uploadGauge(levels);
  REQUIRE(h0 >= 0);
  REQUIRE(h1 >= 0);
  CHECK(h1 != h0);
  gpu::freeGauge(h0);
  const int h2 = gpu::uploadGauge(levels);  // must land in the slot h0 vacated
  CHECK(h2 == h0);
  gpu::freeGauge(h1);
  gpu::freeGauge(h2);
}

TEST_CASE("Metal connectionMatvec matches applyConnectionLaplacian (complex)") {
  if (!gpu::metalAvailable()) {
    MESSAGE("no Metal device available -- skipping GPU parity check");
    return;
  }
  using cd = std::complex<double>;

  MacGrid g(12, 10, 8, 0.1, Vec3{-0.6, -0.5, -0.4});
  FaceField theta = ops::zeroFaceField(g);  // non-trivial connection angles
  auto fill = [](std::vector<double>& a, double s) {
    for (size_t i = 0; i < a.size(); ++i) a[i] = 0.7 * std::sin(0.13 * i + s);
  };
  fill(theta.x, 0.3);
  fill(theta.y, 1.1);
  fill(theta.z, 2.0);
  const GaugeLattice lat = gaugeLatticeFromFaces(g, theta);
  const int n = static_cast<int>(lat.numNodes());

  std::mt19937 rng(99);
  std::uniform_real_distribution<double> d(-1.0, 1.0);
  std::vector<cd> x(n);
  for (cd& z : x) z = cd(d(rng), d(rng));

  const std::vector<cd> ycpu = applyConnectionLaplacian(lat, x);
  const std::vector<cd> ygpu =
      gpu::connectionMatvec(lat.lx, lat.ly, lat.lz, lat.periodic, lat.w, lat.tx, lat.ty, lat.tz, x);

  REQUIRE(ygpu.size() == ycpu.size());
  double maxErr = 0.0;
  for (int i = 0; i < n; ++i) maxErr = std::max(maxErr, std::abs(ycpu[i] - ygpu[i]));
  MESSAGE("max |CPU(double) - GPU(float)| connection matvec = " << maxErr);
  CHECK(maxErr < 1e-3);  // single-precision complex stencil (values are O(1), w = 1/h^2 ~ 100)
}
