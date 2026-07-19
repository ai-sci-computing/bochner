// Dump the |psi| midplane slices behind the paper's seed figure:
// the SOLVE-FREE covariant subdivision seed at two hierarchy depths, and the
// converged ground state, on the seeded-ring operator. The ring lies in the
// z = 0 plane; the dumped (x,z) cross-section (y = 0) contains the ring's
// axis, so the core pierces it at x = +-R.
//
// Blocks (blank-line separated, CSV):
//   1. |psi| slice of the seed at [levels] (default 3 -- the shallow,
//      pipeline-like regime: base 8^3 at n=64, zero set empty)
//   2. |psi| slice of the seed at FULL depth (single-cell base -- the regime
//      whose zero set is one closed ring)
//   3. |psi| slice of the converged ground state
//   4. zero-set crossings of the deep seed  (x,y,z,orientation per line)
//   5. zero-set crossings of the ground state
//
// Usage: seed_slice [n] [levels] [boxHalfWidth] > seed_slice.csv
// (stats on stderr). Default box 1.4: the deep seed's zero-set topology
// needs the domain to enclose the ring's flux -- in a tight 0.8 crop the
// single-cell seed's zero set is empty again (measured).

#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "extraction/MacConnectionLaplacian.h"
#include "extraction/MacFilaments.h"
#include "fluid/MacVortexRing.h"
#include "grid/MacGrid.h"
#include "solvers/GaugeEigen.h"
#include "solvers/GaugeMultigrid.h"

using namespace bochner;
using cd = std::complex<double>;

namespace {

void printSlice(const GaugeLattice& lat, int n, const std::vector<cd>& psi) {
  const int j = n / 2;
  for (int i = 0; i < n; ++i) {
    for (int k = 0; k < n; ++k)
      std::printf("%s%.6e", k ? "," : "", std::abs(psi[lat.index(i, j, k)]));
    std::printf("\n");
  }
}

double minAbs(const std::vector<cd>& psi) {
  double m = 1e300;
  for (const cd& z : psi) m = std::min(m, std::abs(z));
  return m;
}

std::vector<FilamentCrossing> crossings(const MacGrid& g, const std::vector<cd>& psi) {
  std::vector<double> ri(2 * psi.size());
  for (std::size_t c = 0; c < psi.size(); ++c) {
    ri[2 * c] = psi[c].real();
    ri[2 * c + 1] = psi[c].imag();
  }
  return traceZeroSet(g, ri);
}

void printCrossings(const std::vector<FilamentCrossing>& xs) {
  for (const auto& c : xs)
    std::printf("%.6e,%.6e,%.6e,%d\n", c.point[0], c.point[1], c.point[2], c.orientation);
}

}  // namespace

int main(int argc, char** argv) {
  const int n = (argc > 1) ? std::atoi(argv[1]) : 64;
  const int shallowLevels = (argc > 2) ? std::atoi(argv[2]) : 3;
  const double L = (argc > 3) ? std::atof(argv[3]) : 1.4;
  const double h = 2.0 * L / n;
  const MacGrid g(n, n, n, h, Vec3{-L, -L, -L});
  const double R = 0.7, Gamma = 1.0, core = 0.15, hbar = Gamma / (2.0 * M_PI);
  const FaceField u = vortexRingFaceField(g, {0, 0, 0}, {0, 0, 1}, R, Gamma, core);
  const FaceField theta = connectionAngles(g, u, hbar);
  const GaugeLattice lat = gaugeLatticeFromFaces(g, theta);

  int maxLevels = 0;
  while ((n % (1 << (maxLevels + 1))) == 0) ++maxLevels;

  // Block 1: shallow seed (pipeline-like depth).
  const std::vector<cd> shallow = subdivisionSectionFromLattice(lat, shallowLevels);
  std::fprintf(stderr, "shallow seed (L=%d): min |psi| = %.6e\n", shallowLevels, minAbs(shallow));
  printSlice(lat, n, shallow);

  // Block 2: full-depth seed (single-cell base).
  const std::vector<cd> deep = subdivisionSectionFromLattice(lat, maxLevels);
  const auto deepX = crossings(g, deep);
  std::fprintf(stderr, "deep seed (L=%d): min |psi| = %.6e, crossings = %zu\n", maxLevels,
               minAbs(deep), deepX.size());
  std::printf("\n");
  printSlice(lat, n, deep);

  // Block 3: converged ground state (warm-started from the shallow seed).
  GaugeEigenOptions eo;
  eo.tol = 1e-7;
  const GaugeEigenResult gs = smallestEigenpairGaugeMG(lat, &shallow, eo);
  const auto gsX = crossings(g, gs.vector);
  std::fprintf(stderr, "ground state: lambda=%.6f iters=%d res=%.2e, crossings = %zu\n",
               gs.eigenvalue, gs.iterations, gs.residual, gsX.size());
  std::printf("\n");
  printSlice(lat, n, gs.vector);

  // Blocks 4+5: zero-set crossings (deep seed, then ground state).
  std::printf("\n");
  printCrossings(deepX);
  std::printf("\n");
  printCrossings(gsX);
  return 0;
}
