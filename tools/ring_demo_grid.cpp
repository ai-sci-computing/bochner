/// \file
/// Demo: the closed simulator <-> extractor loop on the MAC grid. Seed a vortex
/// ring, self-advect it with the Covector Fluids stepper (BFECC covector
/// advection -> pressure projection) in a closed box, and extract its vortex
/// filament every frame via the Weissmann-Pinkall pipeline (connection
/// Laplacian -> smallest eigenvector -> zero-set trace -> link). Writes per-frame
/// filament polylines for visualization (see tools/animate_filaments.py).
///
/// Usage: ring_demo_grid [out.txt] [frames] [n] [startZ]
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <slepc.h>

#include "solvers/EigenSolver.h"
#include "grid/GridOperators.h"
#include "extraction/MacConnectionLaplacian.h"
#include "extraction/MacFilaments.h"
#include "fluid/MacFluidSolver.h"
#include "grid/MacGrid.h"
#include "fluid/MacProjection.h"
#include "fluid/MacVortexRing.h"

using namespace bochner;

namespace {

// Largest filament extracted from the current velocity field (or empty).
// `psi` is warm-started in and updated out for the next frame.
Filament extractRing(const MacGrid& g, const FaceField& u, double hbar, std::vector<double>& psi) {
  const FaceField theta = connectionAngles(g, u, hbar);
  const std::vector<double>* guess =
      psi.size() == static_cast<std::size_t>(2 * g.numCells()) ? &psi : nullptr;
  psi = smallestEigenpair(connectionLaplacian(g, theta), 1e-5, guess).vector;
  const std::vector<Filament> fils = linkFilaments(g, traceZeroSet(g, psi));
  if (fils.empty()) return Filament{{}, false};
  std::size_t big = 0;
  for (std::size_t i = 1; i < fils.size(); ++i)
    if (fils[i].points.size() > fils[big].points.size()) big = i;
  return fils[big];
}

}  // namespace

int main(int argc, char** argv) {
  SlepcInitialize(&argc, &argv, nullptr, nullptr);
  {
    const std::string out = (argc > 1) ? argv[1] : "frames_grid.txt";
    const int frames = (argc > 2) ? std::atoi(argv[2]) : 30;
    const int n = (argc > 3) ? std::atoi(argv[3]) : 28;
    // Start the ring low so it has the whole box to self-propel through (+z).
    const double startZ = (argc > 4) ? std::atof(argv[4]) : -1.0;

    const double L = 1.4, h = 2.0 * L / n;
    const MacGrid g(n, n, n, h, Vec3{-L, -L, -L});  // cubic box [-L, L]^3
    std::printf("grid: %d^3 = %d cells, h=%.3f\n", n, g.numCells(), h);

    const double R = 0.7, Gamma = 1.0, core = 0.15, dt = 0.04;
    const double hbar = Gamma / (2.0 * M_PI);

    // Seed the ring (axis +z) and make it divergence-free in the closed box.
    FaceField u = vortexRingFaceField(g, {0, 0, startZ}, {0, 0, 1}, R, Gamma, core);
    u = projectToDivergenceFree(g, u);

    std::vector<double> psi;  // warm start carried across frames
    std::FILE* f = std::fopen(out.c_str(), "w");
    std::fprintf(f, "# box %.3f %.3f  nframes %d\n", -L, L, frames);
    for (int frame = 0; frame < frames; ++frame) {
      const Filament fil = extractRing(g, u, hbar, psi);
      std::fprintf(f, "FRAME %d %d %zu\n", frame, fil.closed ? 1 : 0, fil.points.size());
      for (const auto& q : fil.points) std::fprintf(f, "%.6f %.6f %.6f\n", q[0], q[1], q[2]);
      std::printf("frame %2d: %3zu filament points%s\n", frame, fil.points.size(),
                  fil.closed ? " (closed)" : "");
      std::fflush(stdout);

      u = stepCovectorFluids(g, u, dt);  // BFECC covector advection + projection
    }
    std::fclose(f);
    std::printf("wrote %s\n", out.c_str());
  }
  SlepcFinalize();
  return 0;
}
