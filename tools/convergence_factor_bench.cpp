// Empirical asymptotic V-cycle convergence factors rho(n, flux) for the paper's
// theory section -- the measured substitute for a full magnetic-Bloch LFA, and
// the flux-aliasing stress test.
//
// The coarse plaquette carries 4x the fine plaquette angle, so a fine flux of
// phi_p = pi/2 per plaquette (nPhi = n^2/4 quanta) makes the FIRST coarse level
// exactly flat while the fine level is maximally frustrated -- the point where
// the Galerkin/rediscretized spectral equivalence provably fails (the
// "flux-aliasing counterexample"). The alpha-stabilized cycle must remain
// convergent there (energy descent), just slower: this table quantifies it.
//
// rho = geometric mean of the last few per-cycle residual reduction factors,
// measured by driving vcycleSolve one cycle at a time on a fixed random RHS.
//
// Usage: convergence_factor_bench

#include <cmath>
#include <complex>
#include <cstdio>
#include <random>
#include <vector>

#include "solvers/GaugeMultigrid.h"

using namespace bochner;
using cd = std::complex<double>;

namespace {

GaugeLattice fluxLattice(int n, int nPhi) {
  const double phi_p = 2.0 * M_PI * nPhi / (static_cast<double>(n) * n);
  const std::size_t N = static_cast<std::size_t>(n) * n * n;
  std::vector<double> lkx(N, 0.0), lky(N, 0.0), lkz(N, 0.0);
  const auto idx = [&](int i, int j, int k) { return (i * n + j) * n + k; };
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      for (int k = 0; k < n; ++k) {
        lkx[idx(i, j, k)] = -phi_p * j;
        if (j == n - 1) lky[idx(i, j, k)] = 2.0 * M_PI * nPhi * i / static_cast<double>(n);
      }
  return gaugeLatticePeriodic(n, n, n, static_cast<double>(n) * n, lkx, lky, lkz);
}

// Asymptotic per-cycle residual reduction factor: run V-cycles one at a time,
// geometric-mean the last `tail` ratios before the residual reaches the
// stagnation floor.
double asymptoticRho(const GaugeLattice& lat, int maxCycles = 60, int tail = 5) {
  const auto levels = buildGaugeLevels(lat);
  std::mt19937_64 rng(4321);
  std::normal_distribution<double> g(0.0, 1.0);
  std::vector<cd> b(static_cast<std::size_t>(lat.numNodes()));
  for (auto& z : b) z = cd(g(rng), g(rng));
  std::vector<cd> x(b.size(), cd(0, 0));

  MgOptions one;
  one.maxCycles = 1;
  one.tol = 0.0;  // never stop early inside the single cycle
  std::vector<double> res;
  res.reserve(maxCycles);
  for (int c = 0; c < maxCycles; ++c) {
    const MgResult r = vcycleSolve(levels, b, x, one);
    res.push_back(r.relResidual);
    if (r.relResidual < 1e-13) break;  // machine floor: stop before ratios saturate
  }
  const int m = static_cast<int>(res.size());
  const int use = std::min(tail, m - 1);
  if (use < 1) return 0.0;
  double logsum = 0.0;
  for (int i = m - use; i < m; ++i) logsum += std::log(res[i] / res[i - 1]);
  return std::exp(logsum / use);
}

}  // namespace

int main() {
  std::printf("=== asymptotic V-cycle convergence factor rho(n, flux) ===\n");
  std::printf("fine flux per plaquette phi_p = 2 pi nPhi / n^2; coarse flux = 4 phi_p.\n");
  std::printf("nPhi = n^2/4 is the ALIASING point (phi_p = pi/2: first coarse level flat,\n");
  std::printf("fine level maximally frustrated).\n\n");
  std::printf("  %-4s %-10s %-12s %-8s\n", "n", "nPhi", "phi_p/2pi", "rho");
  for (int n : {8, 16, 32}) {
    const int alias = n * n / 4;
    // weak flux, the paper's standard nPhi=4, half-way to aliasing, aliasing,
    // and just past it.
    const int phis[] = {1, 4, alias / 2, alias, alias + n / 4};
    for (int nPhi : phis) {
      const GaugeLattice lat = fluxLattice(n, nPhi);
      const double rho = asymptoticRho(lat);
      std::printf("  %-4d %-10d %-12.5f %-8.3f%s\n", n, nPhi,
                  nPhi / (static_cast<double>(n) * n), rho,
                  nPhi == alias ? "   <-- aliasing point" : "");
    }
    std::printf("\n");
  }
  return 0;
}
