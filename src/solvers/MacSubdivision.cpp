#include "solvers/MacSubdivision.h"

#include <complex>
#include <cstddef>
#include <stdexcept>

#include "solvers/GaugeMultigrid.h"

namespace bochner {

std::vector<double> subdivisionSection(const MacGrid& g, const FaceField& theta, int numLevels) {
  if (numLevels < 0) throw std::invalid_argument("subdivisionSection: numLevels must be >= 0");
  const int factor = 1 << numLevels;
  if (g.nx() % factor || g.ny() % factor || g.nz() % factor)
    throw std::invalid_argument(
        "subdivisionSection: every grid dimension must be divisible by 2^numLevels");

  // Build the finest connection lattice from theta, then let the gauge-multigrid
  // transfer do the covariant subdivision (seed psi = 1 on the coarsest, prolong
  // up). The lattice + coarsen/prolong live in one place (GaugeMultigrid); this
  // is the grid-level wrapper that reads the connection and emits the section.
  const GaugeLattice finest = gaugeLatticeFromFaces(g, theta);
  const std::vector<std::complex<double>> sec = subdivisionSectionFromLattice(finest, numLevels);

  // Emit interleaved [Re, Im, ...] for traceZeroSet / smallestEigenpair.
  std::vector<double> psi(2 * static_cast<std::size_t>(g.numCells()));
  for (std::size_t c = 0; c < sec.size(); ++c) {
    psi[2 * c] = sec[c].real();
    psi[2 * c + 1] = sec[c].imag();
  }
  return psi;
}

}  // namespace bochner
