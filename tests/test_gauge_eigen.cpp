/// \file
/// Red-green TDD for the gauge-MG Rayleigh-quotient eigensolver (GaugeEigen):
/// our standalone smallest-eigenpair method. Anchors (vs the SLEPc ground
/// truth): (1) it converges to the same smallest eigenvalue on a frustrated
/// connection, with a genuine eigen-residual; (2) its eigenvector's zero set
/// traces the seeded vortex ring; (3) a subdivision warm start cuts iterations.
#include <doctest.h>

#include <cmath>
#include <complex>
#include <vector>

#include "solvers/EigenSolver.h"
#include "solvers/GaugeEigen.h"
#include "solvers/GaugeMultigrid.h"
#include "extraction/MacConnectionLaplacian.h"
#include "extraction/MacFilaments.h"
#include "grid/MacGrid.h"
#include "solvers/MacSubdivision.h"
#include "fluid/MacVortexRing.h"

using bochner::Filament;
using bochner::GaugeLattice;
using bochner::MacGrid;
using bochner::Vec3;
using cd = std::complex<double>;

namespace {
bochner::FaceField ringTheta(const MacGrid& g) {
  const double R = 0.7, Gamma = 1.0, hbar = Gamma / (2.0 * M_PI);
  const auto u = bochner::vortexRingFaceField(g, {0, 0, 0}, {0, 0, 1}, R, Gamma, 0.15);
  return bochner::connectionAngles(g, u, hbar);
}
}  // namespace

TEST_CASE("gauge-MG eigensolver matches the SLEPc smallest eigenvalue") {
  const MacGrid g(24, 24, 24, 1.6 / 24, Vec3{-0.8, -0.8, -0.8});
  const auto theta = ringTheta(g);
  const GaugeLattice lat = bochner::gaugeLatticeFromFaces(g, theta);

  // Certified ground truth: Krylov-Schur throws on non-convergence. (The
  // inverse-iteration reference can never satisfy its own stopping rule here
  // -- its inner-CG accuracy floor sits above tol*lambda -- so its converged
  // flag cannot certify; only the quadratic accuracy of its Rayleigh quotient
  // made value comparisons work.)
  const auto slepc = bochner::smallestEigenpairLanczos(bochner::connectionLaplacian(g, theta));
  REQUIRE(slepc.converged);
  const double lamSlepc = slepc.value;

  bochner::GaugeEigenOptions opts;
  opts.tol = 1e-7;
  const auto res = bochner::smallestEigenpairGaugeMG(lat, nullptr, opts);

  INFO("covMG-LOBPCG eig=" << res.eigenvalue << " (SLEPc " << lamSlepc << ") iters=" << res.iterations
                   << " residual=" << res.residual);
  CHECK(res.residual < 1e-6);
  CHECK(res.eigenvalue == doctest::Approx(lamSlepc).epsilon(1e-3));
  CHECK(res.iterations < opts.maxIters);  // converged, not capped
}

TEST_CASE("gauge-MG eigenvector's zero set traces the seeded ring") {
  const MacGrid g(28, 28, 16, 0.1, Vec3{-1.4, -1.4, -0.8});
  const GaugeLattice lat = bochner::gaugeLatticeFromFaces(g, ringTheta(g));

  bochner::GaugeEigenOptions opts;
  opts.tol = 1e-7;
  const auto res = bochner::smallestEigenpairGaugeMG(lat, nullptr, opts);

  const auto psi = bochner::toInterleaved(res.vector);
  const auto filaments = bochner::linkFilaments(g, bochner::traceZeroSet(g, psi));
  const Filament* ring = nullptr;
  for (const auto& f : filaments)
    if (f.closed && (!ring || f.points.size() > ring->points.size())) ring = &f;
  REQUIRE(ring != nullptr);

  double meanR = 0.0, maxAbsZ = 0.0;
  for (const auto& p : ring->points) {
    meanR += std::sqrt(p[0] * p[0] + p[1] * p[1]);
    maxAbsZ = std::max(maxAbsZ, std::abs(p[2]));
  }
  meanR /= ring->points.size();
  INFO("ring meanR=" << meanR << " maxAbsZ=" << maxAbsZ << " pts=" << ring->points.size());
  CHECK(maxAbsZ < 2.0 * g.spacing());                // planar (z = 0)
  CHECK(meanR == doctest::Approx(0.7).epsilon(0.25));  // near the ring radius
}

TEST_CASE("gauge-MG eigensolver works on an elongated (non-cubic) box") {
  // CF runs the trefoil/leapfrog in a 2:1:1 box; the MG must coarsen a long z
  // axis (48 -> 24 -> 12; cross-section limits depth). Ring near the lower end.
  const double h = 0.1;
  const MacGrid g(16, 16, 48, h, Vec3{-0.8, -0.8, -2.4});
  const double R = 0.5, Gamma = 1.0, hbar = Gamma / (2.0 * M_PI);
  const auto u = bochner::vortexRingFaceField(g, {0, 0, -1.6}, {0, 0, 1}, R, Gamma, 0.12);
  const GaugeLattice lat = bochner::gaugeLatticeFromFaces(g, bochner::connectionAngles(g, u, hbar));

  bochner::GaugeEigenOptions opts;
  opts.tol = 1e-7;
  const auto res = bochner::smallestEigenpairGaugeMG(lat, nullptr, opts);

  INFO("elongated eig=" << res.eigenvalue << " iters=" << res.iterations
                        << " residual=" << res.residual);
  CHECK(res.residual < 1e-4);  // small eigen-residual (self-converged)
  // The certified flag, not the iteration count: the m==1 stagnation break also
  // exits below maxIters, so `iterations < maxIters` cannot certify convergence.
  CHECK(res.converged);

  const auto psi = bochner::toInterleaved(res.vector);
  const auto filaments = bochner::linkFilaments(g, bochner::traceZeroSet(g, psi));
  int closed = 0;
  for (const auto& f : filaments)
    if (f.closed && f.points.size() >= 4) ++closed;
  CHECK(closed >= 1);  // the ring extracts on the elongated grid
}

TEST_CASE("subdivision warm start cuts the iteration count under the live policy") {
  const MacGrid g(32, 32, 32, 1.6 / 32, Vec3{-0.8, -0.8, -0.8});
  const auto theta = ringTheta(g);
  const GaugeLattice lat = bochner::gaugeLatticeFromFaces(g, theta);
  const auto seed = bochner::toComplex(bochner::subdivisionSection(g, theta, /*numLevels=*/5));

  // The seed's iteration-cutting claim belongs to the LIVE pipeline policy
  // (absolute-drop early exit, the setting the viewer runs): there the seed
  // must not cost iterations relative to a cold start.
  bochner::GaugeEigenOptions live;
  live.tol = 1e-7;
  live.relativeGsDrop = false;
  const int coldLive = bochner::smallestEigenpairGaugeMG(lat, nullptr, live).iterations;
  const int warmLive = bochner::smallestEigenpairGaugeMG(lat, &seed, live).iterations;
  INFO("live policy: cold=" << coldLive << ", warm=" << warmLive);
  CHECK(warmLive <= coldLive);

  // Under the certifying (relative-drop) policy both starts must certify;
  // the seed is roughly neutral there (measured 28 vs 25 cold), so assert
  // certification and a bounded gap, not superiority.
  bochner::GaugeEigenOptions cert;
  cert.tol = 1e-7;
  const auto coldCert = bochner::smallestEigenpairGaugeMG(lat, nullptr, cert);
  const auto warmCert = bochner::smallestEigenpairGaugeMG(lat, &seed, cert);
  INFO("certifying: cold=" << coldCert.iterations << " (res=" << coldCert.residual
       << "), warm=" << warmCert.iterations << " (res=" << warmCert.residual << ")");
  CHECK(coldCert.converged);
  CHECK(warmCert.converged);
  CHECK(warmCert.iterations <= coldCert.iterations + 5);
}

TEST_CASE("weighted lattice (constant arrays) matches the SLEPc smallest eigenvalue") {
  // End-to-end weighted-path exercise: constant per-edge weights are the same
  // operator as the scalar-w lattice, so covMG-LOBPCG through the weighted code path
  // must reproduce the uniform behaviour (same geometry + criteria as the
  // uniform SLEPc-match test above, including its ~1e-6 residual floor from
  // the 1e-7 Gram-Schmidt drop).
  const MacGrid g(24, 24, 24, 1.6 / 24, Vec3{-0.8, -0.8, -0.8});
  const auto theta = ringTheta(g);
  // Certified ground truth (see the note in the first SLEPc test above).
  const auto slepc = bochner::smallestEigenpairLanczos(bochner::connectionLaplacian(g, theta));
  REQUIRE(slepc.converged);
  const double lamSlepc = slepc.value;

  GaugeLattice lat = bochner::gaugeLatticeFromFaces(g, theta);
  const auto fill = [&](long cnt) { return std::vector<double>(static_cast<std::size_t>(cnt), lat.w); };
  lat.setEdgeWeights(fill(lat.numLinksX()), fill(lat.numLinksY()), fill(lat.numLinksZ()));

  bochner::GaugeEigenOptions opts;
  opts.tol = 1e-7;
  const auto res = bochner::smallestEigenpairGaugeMG(lat, nullptr, opts);
  INFO("weighted covMG-LOBPCG eig=" << res.eigenvalue << " (SLEPc " << lamSlepc << ") iters="
                            << res.iterations << " residual=" << res.residual);
  CHECK(res.residual < 1e-6);
  CHECK(res.eigenvalue == doctest::Approx(lamSlepc).epsilon(1e-3));
  CHECK(res.iterations < opts.maxIters);
}

TEST_CASE("eigensolver converges on a graded-weight lattice with a genuine residual") {
  const MacGrid g(24, 24, 24, 1.6 / 24, Vec3{-0.8, -0.8, -0.8});
  GaugeLattice lat = bochner::gaugeLatticeFromFaces(g, ringTheta(g));
  const auto graded = [&](long cnt, int axis) {
    std::vector<double> w(static_cast<std::size_t>(cnt));
    for (long e = 0; e < cnt; ++e) {
      const double t = std::sin(0.37 * e + 0.7 * axis);
      w[static_cast<std::size_t>(e)] = lat.w * (1.0 + 3.0 * t * t);
    }
    return w;
  };
  lat.setEdgeWeights(graded(lat.numLinksX(), 0), graded(lat.numLinksY(), 1),
                     graded(lat.numLinksZ(), 2));

  bochner::GaugeEigenOptions opts;
  opts.tol = 1e-7;
  const auto res = bochner::smallestEigenpairGaugeMG(lat, nullptr, opts);
  CHECK(res.eigenvalue > 0.0);
  CHECK(res.iterations < opts.maxIters);

  // Recompute the eigen-residual independently of the solver's bookkeeping:
  // it must agree with the reported value and sit at/below the solver's
  // practical floor.
  const auto Ex = bochner::applyConnectionLaplacian(lat, res.vector);
  double num = 0.0, nrm2 = 0.0;
  for (std::size_t i = 0; i < Ex.size(); ++i) {
    num += std::norm(Ex[i] - res.eigenvalue * res.vector[i]);
    nrm2 += std::norm(res.vector[i]);
  }
  const double indep = std::sqrt(num / nrm2) / res.eigenvalue;
  INFO("graded covMG-LOBPCG eig=" << res.eigenvalue << " iters=" << res.iterations
                          << " reported=" << res.residual << " recomputed=" << indep);
  CHECK(indep == doctest::Approx(res.residual).epsilon(1e-6));
  CHECK(indep < 1e-5);
}

// Regression: a block eigensolve that CANNOT build a rank-m block must report
// failure, not a block of zeros. blockLobpcg signals this by returning empty
// eigenvalue/residual/vector arrays with maxResidual = 1e300; the wrapper's
// resize(m) -- correct on the success path, where blockLobpcg returns the
// larger mb -- would GROW those empty arrays with zeros, so the max over m
// zeros came out 0 < tol and the 1e300 sentinel was overwritten. The result was
// converged=true with m zero eigenvalues and m ZERO-LENGTH eigenvectors: any
// caller indexing res.vectors[j][i] reads out of bounds.
TEST_CASE("lowestEigenpairsGaugeMG reports failure, not zeros, when the block cannot be built") {
  // 2^3 open lattice: 8 nodes, but m + blockGuard > 8 trial vectors are wanted,
  // so Gram-Schmidt can never complete a full block.
  const MacGrid g(2, 2, 2, 0.5, Vec3{-0.5, -0.5, -0.5});
  const GaugeLattice lat = bochner::gaugeLatticeFromFaces(g, ringTheta(g));
  REQUIRE(lat.numNodes() == 8);

  bochner::GaugeEigenOptions opts;
  opts.tol = 1e-7;
  const int m = 8;  // + blockGuard pushes the block past the 8 available dimensions
  const auto res = bochner::lowestEigenpairsGaugeMG(lat, m, nullptr, opts);

  INFO("converged=" << res.converged << " maxResidual=" << res.maxResidual
                    << " eigvals=" << res.eigenvalues.size()
                    << " vectors=" << res.vectors.size());
  CHECK_FALSE(res.converged);
  // Pins that this really is the failure path and not a vacuous pass: the
  // 1e300 sentinel surviving proves blockLobpcg could not build the block AND
  // that the wrapper did not overwrite the sentinel with a max over zeros
  // (which is exactly what produced the false converged=true).
  CHECK(res.maxResidual > 1e299);
  // Every returned eigenvector must be usable: either the call reports failure
  // with nothing to index, or it hands back full-length vectors. Never both.
  for (const auto& v : res.vectors) CHECK(v.size() == static_cast<std::size_t>(lat.numNodes()));
}
