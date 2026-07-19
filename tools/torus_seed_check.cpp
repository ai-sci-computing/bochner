/// \file
/// Does the single-cell-base subdivision seed carry the Landau topology on
/// the PERIODIC uniform-flux torus? Companion check to the seeded-ring
/// finding, on the paper's workhorse operator, with no open
/// boundary and no ring geometry.
///
/// Prints, for the deep seed and the converged ground state: min/max |psi|,
/// and the per-xy-plane winding census (sum of raw-phase plaquette windings;
/// the lowest Landau level on the torus carries net winding nPhi per plane).
/// Also checks the trivial connection (nPhi=0): the deep seed must be
/// identically 1 (covariant averages of equal values), so any magnitude
/// structure at nPhi>0 is holonomy interference, not an averaging artifact.
///
/// Usage: torus_seed_check [n] [nPhi]   (dumps a midplane |psi| CSV pair to
/// stdout: deep seed, blank line, ground state; stats on stderr)

#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "solvers/GaugeEigen.h"
#include "solvers/GaugeMultigrid.h"

using namespace bochner;
using cd = std::complex<double>;

namespace {

GaugeLattice uniformFluxLattice(int n, int nPhi, double w) {
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
  return gaugeLatticePeriodic(n, n, n, w, lkx, lky, lkz);
}

double wrap(double d) {
  while (d > M_PI) d -= 2.0 * M_PI;
  while (d <= -M_PI) d += 2.0 * M_PI;
  return d;
}

// Net and gross raw-phase winding over the xy-plaquettes of one z-layer.
void windingCensus(const GaugeLattice& lat, const std::vector<cd>& psi, int n, int k, int& net,
                   int& gross) {
  net = gross = 0;
  const auto ph = [&](int i, int j) { return std::arg(psi[lat.index(i % n, j % n, k)]); };
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) {
      const double s = wrap(ph(i + 1, j) - ph(i, j)) + wrap(ph(i + 1, j + 1) - ph(i + 1, j)) +
                       wrap(ph(i, j + 1) - ph(i + 1, j + 1)) + wrap(ph(i, j) - ph(i, j + 1));
      const int w = static_cast<int>(std::lround(s / (2.0 * M_PI)));
      net += w;
      gross += std::abs(w);
    }
}

void stats(const char* tag, const GaugeLattice& lat, const std::vector<cd>& psi, int n) {
  double lo = 1e300, hi = 0.0;
  for (const cd& z : psi) {
    lo = std::min(lo, std::abs(z));
    hi = std::max(hi, std::abs(z));
  }
  int net0, gross0, netMid, grossMid;
  windingCensus(lat, psi, n, 0, net0, gross0);
  windingCensus(lat, psi, n, n / 2, netMid, grossMid);
  std::fprintf(stderr, "%-14s min|psi|=%.3e max|psi|=%.3e  winding z=0: net %+d gross %d;"
                       " z=n/2: net %+d gross %d\n",
               tag, lo, hi, net0, gross0, netMid, grossMid);
}

void dumpSlice(const GaugeLattice& lat, const std::vector<cd>& psi, int n) {
  const int k = n / 2;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j)
      std::printf("%s%.6e", j ? "," : "", std::abs(psi[lat.index(i, j, k)]));
    std::printf("\n");
  }
}

}  // namespace

int main(int argc, char** argv) {
  const int n = (argc > 1) ? std::atoi(argv[1]) : 32;
  const int nPhi = (argc > 2) ? std::atoi(argv[2]) : 4;
  int maxLevels = 0;
  while ((n % (1 << (maxLevels + 1))) == 0) ++maxLevels;

  // Sanity: trivial connection -> deep seed identically 1.
  {
    const GaugeLattice flat = uniformFluxLattice(n, 0, double(n) * n);
    const std::vector<cd> s = subdivisionSectionFromLattice(flat, maxLevels);
    double lo = 1e300, hi = 0.0;
    for (const cd& z : s) {
      lo = std::min(lo, std::abs(z));
      hi = std::max(hi, std::abs(z));
    }
    std::fprintf(stderr, "trivial conn : min|psi|=%.15f max|psi|=%.15f (expect 1)\n", lo, hi);
  }

  const GaugeLattice lat = uniformFluxLattice(n, nPhi, double(n) * n);
  const std::vector<cd> seed = subdivisionSectionFromLattice(lat, maxLevels);
  stats("deep seed", lat, seed, n);

  GaugeEigenOptions eo;
  eo.tol = 1e-7;
  const GaugeEigenResult gs = smallestEigenpairGaugeMG(lat, &seed, eo);
  std::fprintf(stderr, "ground state : lambda=%.6f iters=%d\n", gs.eigenvalue, gs.iterations);
  stats("ground state", lat, gs.vector, n);

  dumpSlice(lat, seed, n);
  std::printf("\n");
  dumpSlice(lat, gs.vector, n);
  return 0;
}
