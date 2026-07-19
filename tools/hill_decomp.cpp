/// \file
/// Demo: Weissmann-Pinkall re-quantization. Seed Hill's spherical vortex (a
/// sphere of distributed vorticity) and extract its vortex filaments at several
/// flux quanta hbar = Gamma/(2*pi*N). As hbar shrinks the smooth vorticity
/// decomposes into an increasing nest of coaxial filament rings -- the "level
/// of detail" knob of W-P (their Fig 11). Writes all filament loops per hbar.
///
/// Usage: hill_decomp [out.txt] [n] [Nmax]
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <slepc.h>

#include "solvers/EigenSolver.h"
#include "fluid/MacAdvection.h"
#include "extraction/MacConnectionLaplacian.h"
#include "extraction/MacFilaments.h"
#include "grid/MacGrid.h"
#include "fluid/MacProjection.h"
#include "fluid/MacVortexRing.h"

using namespace bochner;

namespace {

// Circulation around a meridional rectangle (x-z plane, +x side) enclosing the
// vortical sphere: Gamma = closed-loop integral of u . dl. The number of
// filament quanta threading this loop at quantum hbar is Gamma / (2*pi*hbar).
double meridionalCirculation(const MacGrid& g, const FaceField& u, double W, double H) {
  const int M = 400;
  double gamma = 0.0;
  auto edge = [&](Vec3 from, Vec3 to) {
    Vec3 d = vsub(to, from);
    for (int s = 0; s < M; ++s) {
      Vec3 p = vadd(from, vscale(d, (s + 0.5) / M));
      gamma += vdot(sampleVelocity(g, u, p), vscale(d, 1.0 / M));
    }
  };
  edge({0, 0, -H}, {W, 0, -H});
  edge({W, 0, -H}, {W, 0, H});
  edge({W, 0, H}, {0, 0, H});
  edge({0, 0, H}, {0, 0, -H});
  return gamma;
}

}  // namespace

int main(int argc, char** argv) {
  SlepcInitialize(&argc, &argv, nullptr, nullptr);
  {
    const std::string out = (argc > 1) ? argv[1] : "hill_decomp.txt";
    const int n = (argc > 2) ? std::atoi(argv[2]) : 40;
    const int Nmax = (argc > 3) ? std::atoi(argv[3]) : 4;

    const double L = 1.6, h = 2.0 * L / n;
    const MacGrid g(n, n, n, h, Vec3{-L, -L, -L});
    const double a = 0.95, U = 1.0;
    std::printf("grid: %d^3 = %d cells, h=%.3f; Hill radius a=%.2f\n", n, g.numCells(), h, a);

    FaceField u = hillVortexFaceField(g, {0, 0, 0}, {0, 0, 1}, a, U);
    u = projectToDivergenceFree(g, u);

    const double Gamma = std::abs(meridionalCirculation(g, u, 1.3, 1.3));
    std::printf("meridional circulation Gamma = %.4f\n", Gamma);

    std::FILE* f = std::fopen(out.c_str(), "w");
    std::fprintf(f, "# box %.3f %.3f\n", -L, L);
    for (int N = 1; N <= Nmax; ++N) {
      const double hbar = Gamma / (2.0 * M_PI * N);  // aim for N flux quanta
      const FaceField theta = connectionAngles(g, u, hbar);
      const std::vector<double> psi = smallestEigenpair(connectionLaplacian(g, theta)).vector;
      std::vector<Filament> fils = linkFilaments(g, traceZeroSet(g, psi));
      // Keep only the closed loops (the ring nest) with a few points.
      std::vector<Filament> loops;
      for (auto& fl : fils)
        if (fl.closed && fl.points.size() >= 4) loops.push_back(std::move(fl));

      std::fprintf(f, "PANEL %d %.5f %zu\n", N, hbar, loops.size());
      for (const auto& fl : loops) {
        std::fprintf(f, "CURVE %d %zu\n", fl.closed ? 1 : 0, fl.points.size());
        for (const auto& q : fl.points) std::fprintf(f, "%.6f %.6f %.6f\n", q[0], q[1], q[2]);
      }
      std::printf("N=%d  hbar=%.4f  ->  %zu closed loops\n", N, hbar, loops.size());
      std::fflush(stdout);
    }
    std::fclose(f);
    std::printf("wrote %s\n", out.c_str());
  }
  SlepcFinalize();
  return 0;
}
