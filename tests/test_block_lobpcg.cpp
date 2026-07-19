/// \file
/// Block covMG-LOBPCG (lowest-m eigenpairs) validation. Anchors:
///   (1) EXACT lowest-N reference: the flat-holonomy anisotropic torus has the
///       closed-form spectrum lambda(m_a) = sum_a (2-2cos(2 pi m_a/n_a -
///       theta_a/n_a))/h_a^2 -- the block solver must reproduce the sorted
///       lowest 8 values, weighted path included.
///   (2) DEGENERACY: the uniform-flux torus's lowest Landau level is
///       n_Phi-fold degenerate; single-vector iteration cannot separate it,
///       the block resolves the full multiplet (equal eigenvalues, orthonormal
///       vectors, genuine residual certificates).
///   (3) SU(2): certificates + consistency with the single-vector solver.
#include <doctest.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <vector>

#include "solvers/GaugeEigen.h"
#include "solvers/GaugeMultigrid.h"
#include "solvers/SunGauge.h"

using bochner::GaugeLattice;
using bochner::SunLattice;
using cd = std::complex<double>;

namespace {

GaugeLattice flatHolonomyLattice(int nx, int ny, int nz, double hx, double hy, double hz,
                                 double thx, double thy, double thz) {
  const std::size_t N = static_cast<std::size_t>(nx) * ny * nz;
  const std::vector<double> lkx(N, thx / nx), lky(N, thy / ny), lkz(N, thz / nz);
  GaugeLattice L = bochner::gaugeLatticePeriodic(nx, ny, nz, 1.0, lkx, lky, lkz);
  L.setEdgeWeights(std::vector<double>(N, 1.0 / (hx * hx)), std::vector<double>(N, 1.0 / (hy * hy)),
                   std::vector<double>(N, 1.0 / (hz * hz)));
  return L;
}

// All eigenvalues of the flat-holonomy operator, sorted ascending.
std::vector<double> exactSpectrum(int nx, int ny, int nz, double hx, double hy, double hz,
                                  double thx, double thy, double thz) {
  std::vector<double> ev;
  ev.reserve(static_cast<std::size_t>(nx) * ny * nz);
  const auto axis = [](int m, int n, double h, double th) {
    return (2.0 - 2.0 * std::cos(2.0 * M_PI * m / n - th / n)) / (h * h);
  };
  for (int a = 0; a < nx; ++a)
    for (int b = 0; b < ny; ++b)
      for (int c = 0; c < nz; ++c)
        ev.push_back(axis(a, nx, hx, thx) + axis(b, ny, hy, thy) + axis(c, nz, hz, thz));
  std::sort(ev.begin(), ev.end());
  return ev;
}

GaugeLattice uniformFluxLattice(int n, int nPhi) {
  const double phi_p = 2.0 * M_PI * nPhi / (double(n) * n);
  const std::size_t N = static_cast<std::size_t>(n) * n * n;
  std::vector<double> lkx(N, 0.0), lky(N, 0.0), lkz(N, 0.0);
  const auto idx = [&](int i, int j, int k) { return (i * n + j) * n + k; };
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      for (int k = 0; k < n; ++k) {
        lkx[idx(i, j, k)] = -phi_p * j;
        if (j == n - 1) lky[idx(i, j, k)] = 2.0 * M_PI * nPhi * i / double(n);
      }
  return bochner::gaugeLatticePeriodic(n, n, n, double(n) * n, lkx, lky, lkz);
}

cd cdot(const std::vector<cd>& a, const std::vector<cd>& b) {
  cd s(0, 0);
  for (std::size_t i = 0; i < a.size(); ++i) s += std::conj(a[i]) * b[i];
  return s;
}

}  // namespace

TEST_CASE("block covMG-LOBPCG reproduces the exact lowest-8 spectrum (weighted anisotropic torus)") {
  const int nx = 8, ny = 12, nz = 16;
  const double hx = 0.5, hy = 0.2, hz = 0.1;
  const double thx = 4.0, thy = 2.0, thz = 0.7;
  const GaugeLattice L = flatHolonomyLattice(nx, ny, nz, hx, hy, hz, thx, thy, thz);
  const auto exact = exactSpectrum(nx, ny, nz, hx, hy, hz, thx, thy, thz);

  bochner::GaugeEigenOptions opts;
  opts.tol = 1e-8;
  opts.maxIters = 400;
  const int m = 8;
  const auto res = bochner::lowestEigenpairsGaugeMG(L, m, nullptr, opts);
  REQUIRE(static_cast<int>(res.eigenvalues.size()) == m);
  MESSAGE("block iters=" << res.iterations << " maxRes=" << res.maxResidual);
  for (int j = 0; j < m; ++j) {
    INFO("j=" << j << " got " << res.eigenvalues[j] << " exact " << exact[j]);
    CHECK(res.eigenvalues[j] == doctest::Approx(exact[j]).epsilon(1e-6));
  }
}

TEST_CASE("block covMG-LOBPCG resolves the degenerate lowest Landau level (flux torus)") {
  const int n = 16, nPhi = 4;
  const GaugeLattice L = uniformFluxLattice(n, nPhi);
  bochner::GaugeEigenOptions opts;
  opts.tol = 1e-7;
  opts.maxIters = 400;
  const int m = 6;
  const auto res = bochner::lowestEigenpairsGaugeMG(L, m, nullptr, opts);
  REQUIRE(static_cast<int>(res.eigenvalues.size()) == m);
  MESSAGE("Landau block: iters=" << res.iterations << " evs=" << res.eigenvalues[0] << ", "
                                 << res.eigenvalues[3] << ", " << res.eigenvalues[4]
                                 << " maxRes=" << res.maxResidual);
  // n_Phi = 4 quanta -> 4-fold degenerate lowest level; reference value from
  // the paper's appendix (24.825546 at n=16).
  for (int j = 0; j < 4; ++j)
    CHECK(res.eigenvalues[j] == doctest::Approx(24.825546).epsilon(1e-5));
  CHECK(res.eigenvalues[4] > res.eigenvalues[3] * 1.001);  // genuine gap after the multiplet
  // Certificates: orthonormal block, true eigenpairs.
  for (int i = 0; i < m; ++i) {
    CHECK(res.residuals[i] < 10 * opts.tol);
    for (int j = 0; j < i; ++j) CHECK(std::abs(cdot(res.vectors[i], res.vectors[j])) < 1e-7);
  }
}

TEST_CASE("block covMG-LOBPCG on massless hot SU(2): certificates + single-vector consistency") {
  const SunLattice L = bochner::randomSunLattice(2, 8, 8, 8, /*w=*/1.0, /*mass2=*/0.0, /*seed=*/5);
  bochner::GaugeEigenOptions opts;
  opts.tol = 1e-7;
  opts.maxIters = 400;
  const int m = 4;
  const auto res = bochner::lowestEigenpairsSunMG(L, m, nullptr, opts);
  REQUIRE(static_cast<int>(res.eigenvalues.size()) == m);
  MESSAGE("SU(2) block: iters=" << res.iterations << " evs=" << res.eigenvalues[0] << ".."
                                << res.eigenvalues[3] << " maxRes=" << res.maxResidual);
  for (int i = 0; i < m; ++i) {
    CHECK(res.residuals[i] < 10 * opts.tol);
    CHECK(res.eigenvalues[i] > 0.0);
    if (i) CHECK(res.eigenvalues[i] >= res.eigenvalues[i - 1] * (1.0 - 1e-12));
    for (int j = 0; j < i; ++j) CHECK(std::abs(cdot(res.vectors[i], res.vectors[j])) < 1e-7);
  }
  // The block's smallest agrees with the single-vector solver.
  const auto single = bochner::smallestEigenpairSunMG(L, nullptr, opts);
  CHECK(res.eigenvalues[0] == doctest::Approx(single.eigenvalue).epsilon(1e-6));
}

TEST_CASE("block soft locking: same eigenpairs as the default path, honest certificates") {
  // Soft locking (currently-certified pairs contribute no new search
  // directions but stay in the Rayleigh-Ritz space) must return the same
  // eigenpairs as the published default under the SAME stopping rule -- only
  // the trial basis while a pair holds below tol differs, so the exit
  // certificate must pass just like the default's.
  const SunLattice L = bochner::randomSunLattice(2, 8, 8, 8, /*w=*/1.0, /*mass2=*/0.0, /*seed=*/5);
  bochner::GaugeEigenOptions opts;
  opts.tol = 1e-7;
  opts.maxIters = 400;
  const int m = 4;
  const auto ref = bochner::lowestEigenpairsSunMG(L, m, nullptr, opts);
  opts.blockSoftLockConverged = true;
  const auto soft = bochner::lowestEigenpairsSunMG(L, m, nullptr, opts);
  REQUIRE(static_cast<int>(soft.eigenvalues.size()) == m);
  MESSAGE("soft-lock: iters=" << soft.iterations << " (default " << ref.iterations
                              << ") maxRes=" << soft.maxResidual);
  CHECK(soft.converged);  // identical stopping rule => certified exit
  for (int i = 0; i < m; ++i) {
    CHECK(soft.eigenvalues[i] == doctest::Approx(ref.eigenvalues[i]).epsilon(1e-6));
    CHECK(soft.residuals[i] < 10 * opts.tol);
    for (int j = 0; j < i; ++j) CHECK(std::abs(cdot(soft.vectors[i], soft.vectors[j])) < 1e-7);
  }
}
