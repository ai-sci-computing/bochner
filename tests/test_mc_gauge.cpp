/// \file
/// Validation of the pure-gauge Wilson-action Monte Carlo sampler (WilsonMc) --
/// the source of the "rough but CORRELATED" SU(3) configurations for the
/// mc_gauge_bench referee experiment. The sampler must be trusted before any
/// solver benchmark on its output means anything; these tests pin:
///   (1) links stay in SU(3) across many sweeps (reunitarization works);
///   (2) hot and cold starts converge to the same average plaquette at fixed
///       beta (thermalization to the same equilibrium, no metastability);
///   (3) <P> is monotone increasing in beta and lies in (0,1);
///   (4) the strong-coupling normalization: with S = -(beta/3) sum_p Re tr U_p
///       the leading-order prediction is <P> ~ beta/18 for SU(3) -- the
///       measured slope at beta <= 1 must match (pins the action convention);
///   (5) gauge covariance: a random gauge transformation of a sampled
///       configuration leaves lambda_min of the covariant Laplacian (and the
///       average plaquette) unchanged.
#include <doctest.h>

#include <cmath>
#include <complex>
#include <random>
#include <vector>

#include "solvers/SunGauge.h"
#include "solvers/WilsonMc.h"

using namespace bochner;
using cd = std::complex<double>;

namespace {

// Thermalize, then measure <P> over `meas` further sweeps; returns the mean.
// If sigmaOut is non-null, writes a binned (bin = 10 sweeps) standard error of
// the mean, which absorbs the short heatbath autocorrelation.
double measurePlaquette(int d, int n, double beta, int therm, int meas, std::uint64_t seed,
                        bool hotStart, double* sigmaOut = nullptr) {
  SunLattice L = mcSunLattice(d, n, beta, therm, /*w=*/1.0, /*mass2=*/0.0, seed, hotStart);
  std::mt19937_64 rng(seed * 2654435761u + 17);
  std::vector<double> hist;
  hist.reserve(meas);
  for (int s = 0; s < meas; ++s) {
    wilsonHeatbathSweep(L, beta, rng);
    if ((s + 1) % 50 == 0) reunitarizeSun(L);
    hist.push_back(averagePlaquette(L));
  }
  double mean = 0.0;
  for (double p : hist) mean += p;
  mean /= hist.size();
  if (sigmaOut) {
    const int bin = 10, nb = meas / bin;
    double var = 0.0;
    for (int b = 0; b < nb; ++b) {
      double m = 0.0;
      for (int s = 0; s < bin; ++s) m += hist[b * bin + s];
      m /= bin;
      var += (m - mean) * (m - mean);
    }
    *sigmaOut = std::sqrt(var / (nb * (nb - 1.0)));
  }
  return mean;
}

double maxUnitarityDefect(const SunLattice& L) {
  const int d = L.d, dd = d * d;
  double worst = 0.0;
  for (const auto* u : {&L.ux, &L.uy, &L.uz})
    for (std::size_t e = 0; e < u->size() / dd; ++e) {
      const cd* M = &(*u)[e * dd];
      // ||M^H M - I||_max and |det M - 1|
      for (int i = 0; i < d; ++i)
        for (int j = 0; j < d; ++j) {
          cd s(0.0, 0.0);
          for (int k = 0; k < d; ++k) s += std::conj(M[k * d + i]) * M[k * d + j];
          if (i == j) s -= cd(1.0, 0.0);
          worst = std::max(worst, std::abs(s));
        }
      cd det;
      if (d == 2)
        det = M[0] * M[3] - M[1] * M[2];
      else
        det = M[0] * (M[4] * M[8] - M[5] * M[7]) - M[1] * (M[3] * M[8] - M[5] * M[6]) +
              M[2] * (M[3] * M[7] - M[4] * M[6]);
      worst = std::max(worst, std::abs(det - cd(1.0, 0.0)));
    }
  return worst;
}

}  // namespace

TEST_CASE("WilsonMc links stay in SU(3) and sane limits hold") {
  // 120 sweeps at beta=6 (reunitarized every 50): links must still be SU(3).
  const SunLattice L = mcSunLattice(3, 6, 6.0, 120, 1.0, 0.0, /*seed=*/11);
  CHECK(maxUnitarityDefect(L) < 1e-10);
  // Hot start, zero sweeps = i.i.d. Haar: <P> ~ 0 (sigma_mean ~ 0.006 at n=8).
  const SunLattice hot = mcSunLattice(3, 8, 0.0, 0, 1.0, 0.0, /*seed=*/7);
  CHECK(std::abs(averagePlaquette(hot)) < 0.03);
  // Cold start, zero sweeps = identity links: <P> = 1 exactly.
  const SunLattice cold = mcSunLattice(3, 8, 0.0, 0, 1.0, 0.0, /*seed=*/7, /*hotStart=*/false);
  CHECK(averagePlaquette(cold) == doctest::Approx(1.0).epsilon(1e-12));
}

TEST_CASE("WilsonMc hot and cold starts agree at beta=5 (SU(3), n=8)") {
  double sh = 0.0, sc = 0.0;
  const double ph = measurePlaquette(3, 8, 5.0, /*therm=*/200, /*meas=*/100, /*seed=*/101, true, &sh);
  const double pc = measurePlaquette(3, 8, 5.0, /*therm=*/200, /*meas=*/100, /*seed=*/202, false, &sc);
  MESSAGE("beta=5 n=8: hot <P>=" << ph << " +/- " << sh << ", cold <P>=" << pc << " +/- " << sc);
  CHECK(ph > 0.0);
  CHECK(ph < 1.0);
  CHECK(pc > 0.0);
  CHECK(pc < 1.0);
  const double tol = std::max(5.0 * std::sqrt(sh * sh + sc * sc), 0.003);
  CHECK(std::abs(ph - pc) < tol);
}

TEST_CASE("WilsonMc average plaquette is monotone in beta and in (0,1)") {
  double prev = 0.0;
  for (double beta : {1.0, 3.0, 6.0, 12.0}) {
    const double p = measurePlaquette(3, 8, beta, /*therm=*/150, /*meas=*/50, /*seed=*/40 + int(beta), true);
    MESSAGE("beta=" << beta << ": <P>=" << p);
    CHECK(p > 0.0);
    CHECK(p < 1.0);
    CHECK(p > prev);
    prev = p;
  }
}

TEST_CASE("WilsonMc strong coupling: <P> ~ beta/18 for SU(3) at beta <= 1") {
  // Leading order of S = -(beta/3) sum_p Re tr U_p: <P> = beta/(2*3^2) = beta/18
  // (a wrong action normalization would miss by a factor >= 3/2 -- this pins the
  // implemented convention, it does not tune to it).
  double s05 = 0.0, s10 = 0.0;
  const double p05 = measurePlaquette(3, 8, 0.5, /*therm=*/100, /*meas=*/200, /*seed=*/905, true, &s05);
  const double p10 = measurePlaquette(3, 8, 1.0, /*therm=*/100, /*meas=*/200, /*seed=*/910, true, &s10);
  MESSAGE("beta=0.5: <P>=" << p05 << " +/- " << s05 << " (LO " << 0.5 / 18.0 << ")");
  MESSAGE("beta=1.0: <P>=" << p10 << " +/- " << s10 << " (LO " << 1.0 / 18.0 << ")");
  // Slope check with room only for the O(beta^2) correction + noise, far below
  // any wrong-normalization factor.
  CHECK(std::abs(p05 - 0.5 / 18.0) < 0.10 * (0.5 / 18.0) + 5.0 * s05);
  CHECK(std::abs(p10 - 1.0 / 18.0) < 0.15 * (1.0 / 18.0) + 5.0 * s10);
}

TEST_CASE("WilsonMc gauge covariance: lambda_min and <P> invariant under gauge transforms") {
  SunLattice L = mcSunLattice(3, 8, 6.0, 60, /*w=*/1.0, /*mass2=*/0.0, /*seed=*/321);
  const double plaqBefore = averagePlaquette(L);
  GaugeEigenOptions eo;
  eo.tol = 1e-6;  // covMG-LOBPCG stalls near 1.7e-7 on this MC config (see the bench
                  // verdicts); 1e-6 converges and pins lambda far below the
                  // invariance tolerance checked here.
  eo.maxIters = 400;
  const GaugeEigenResult before = smallestEigenpairSunMG(L, nullptr, eo);
  REQUIRE(before.converged);

  gaugeTransformSun(L, /*seed=*/777);
  CHECK(maxUnitarityDefect(L) < 1e-10);  // transformed links still SU(3)
  const double plaqAfter = averagePlaquette(L);
  // Wilson action is gauge invariant (roundoff only; a broken transform moves <P> by O(1)).
  CHECK(std::abs(plaqAfter - plaqBefore) < 1e-9);

  const GaugeEigenResult after = smallestEigenpairSunMG(L, nullptr, eo);
  REQUIRE(after.converged);
  MESSAGE("lambda_min before=" << before.eigenvalue << " after=" << after.eigenvalue);
  CHECK(std::abs(after.eigenvalue - before.eigenvalue) <
        1e-6 * std::max(1.0, std::abs(before.eigenvalue)));
}

TEST_CASE("relativeGsDrop certifies 1e-7 on an MC config where the absolute drop stalls") {
  // The legacy ABSOLUTE 1e-7 MGS drop threshold floors the certifiable
  // relative eigen-residual at ~3e-7 on MC configs (rho ~ lambda_2, so the
  // preconditioned direction's norm ~ the relative residual itself).
  // The scale-RELATIVE test keeps small
  // but genuinely new directions and must reach the 1e-7 certificate.
  const SunLattice L = mcSunLattice(3, 8, 6.0, 300, /*w=*/64.0, /*mass2=*/0.0,
                                    /*seed=*/2026, /*hotStart=*/false);
  GaugeEigenOptions eo;
  eo.tol = 1e-7;
  eo.maxIters = 300;
  eo.relativeGsDrop = false;  // the legacy absolute test

  const GaugeEigenResult stalled = smallestEigenpairSunMG(L, nullptr, eo);
  CHECK(!stalled.converged);      // floors just above tol...
  CHECK(stalled.residual > 1e-7);
  CHECK(stalled.residual < 1e-5); // ...but does not diverge

  eo.relativeGsDrop = true;
  const GaugeEigenResult certified = smallestEigenpairSunMG(L, nullptr, eo);
  MESSAGE("absolute drop: its=" << stalled.iterations << " res=" << stalled.residual
          << " | relative drop: its=" << certified.iterations
          << " res=" << certified.residual);
  CHECK(certified.converged);
  CHECK(certified.residual < 1e-7);
  CHECK(std::abs(certified.eigenvalue - stalled.eigenvalue) <
        1e-6 * std::abs(stalled.eigenvalue));
}
