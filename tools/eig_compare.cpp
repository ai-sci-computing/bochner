/// \file
/// Eigensolver comparison: our gauge-MG Rayleigh-quotient method vs the SLEPc
/// Lanczos baseline (the Phase-3 wall-clock winner) for the smallest eigenpair
/// of the seeded-ring connection Laplacian, across grid sizes. Both solve to the
/// same relative eigen-residual; we report the eigenvalue (must agree), the
/// outer iterations, and wall-clock.
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <slepc.h>

#include "BenchTiming.h"
#include "solvers/EigenSolver.h"
#include "solvers/GaugeEigen.h"
#include "solvers/GaugeMultigrid.h"
#include "extraction/MacConnectionLaplacian.h"
#include "grid/MacGrid.h"
#include "fluid/MacVortexRing.h"

using bochner::GaugeLattice;
using bochner::MacGrid;
using bochner::Vec3;
using benchstat::medianMs;

bool gRelativeDrop = true;  // "absdrop" argv token selects the legacy absolute test

int main(int argc, char** argv) {
  SlepcInitialize(&argc, &argv, nullptr, nullptr);
  const double tol = 1e-7;

  std::vector<int> sizes;
  for (int a = 1; a < argc; ++a) {
    if (std::string(argv[a]) == "absdrop") { gRelativeDrop = false; continue; }
    // A PETSc/SLEPc option (e.g. -eps_ncv 32) and its value: consumed by the
    // options database at SlepcInitialize, not a lattice size -- atoi would
    // turn it into a 0-sized grid.
    if (argv[a][0] == '-') { ++a; continue; }
    sizes.push_back(std::atoi(argv[a]));
  }
  if (sizes.empty()) sizes = {16, 24, 32, 48};

  std::printf("Smallest eigenpair of the ring connection Laplacian, tol=%.0e.\n\n", tol);
  std::printf("%5s %9s | %-22s | %-30s\n", "n", "DOFs", "SLEPc Lanczos", "covMG-LOBPCG (ours)");
  std::printf("%5s %9s | %10s %9s | %10s %6s %9s\n", "", "", "eig", "ms", "eig", "iter", "ms");

  for (int n : sizes) {
    const MacGrid g(n, n, n, 1.6 / n, Vec3{-0.8, -0.8, -0.8});
    const double R = 0.7, Gamma = 1.0, hbar = Gamma / (2.0 * M_PI);
    const auto u = bochner::vortexRingFaceField(g, {0, 0, 0}, {0, 0, 1}, R, Gamma, 0.15);
    const auto theta = bochner::connectionAngles(g, u, hbar);
    const auto E = bochner::connectionLaplacian(g, theta);
    const GaugeLattice lat = bochner::gaugeLatticeFromFaces(g, theta);

    double lamL = 0.0, lamR = 0.0, resR = 0.0;
    int itR = 0;
    bool convR = false;
    const double msL = medianMs([&] { lamL = bochner::smallestEigenpairLanczos(E, tol).value; });

    bochner::GaugeEigenOptions opts;
    opts.tol = tol;
    opts.relativeGsDrop = gRelativeDrop;
    const double msR = medianMs([&] {
      const auto r = bochner::smallestEigenpairGaugeMG(lat, nullptr, opts);
      lamR = r.eigenvalue;
      itR = r.iterations;
      resR = r.residual;
      convR = r.converged;
    });

    std::printf("%5d %9d | %10.5f %9.1f | %10.5f %6d %9.1f  res=%.1e%s\n", n, 2 * g.numCells(),
                lamL, msL, lamR, itR, msR, resR, convR ? "" : " NOT-CERTIFIED");
  }
  benchstat::printTimingSummary();
  SlepcFinalize();
  return 0;
}
