/// \file
/// Full per-frame pipeline profiler. Now that the eigensolve is no longer the
/// bottleneck (covMG-LOBPCG, ~100 ms at 110k cells), this measures where the
/// remaining time goes so we know the new bottleneck before optimising further.
///
/// Per frame on the live self-advecting ring it times the four stages:
///   1. sim step      -- stepCovectorFluids (BFECC advection + PETSc projection)
///   2. connection    -- connectionAngles + gaugeLatticeFromFaces
///   3. eigensolve    -- smallestEigenpairGaugeMG (warm-started from prev frame)
///   4. extraction    -- traceZeroSet + linkFilaments
/// and reports the breakdown, total, and implied fps.
///
/// Usage: pipeline_profile [n] [frames] [dt]   (default 48 10 0.02)
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <slepc.h>

#include "solvers/GaugeEigen.h"
#include "solvers/GaugeMultigrid.h"
#include "fluid/MacAdvection.h"
#include "extraction/MacConnectionLaplacian.h"
#include "extraction/MacFilaments.h"
#include "fluid/MacFluidSolver.h"
#include "grid/MacGrid.h"
#include "fluid/MacProjection.h"
#include "fluid/MacVortexRing.h"

using namespace bochner;

namespace {
template <class F>
double timeMs(F&& f) {
  const auto t0 = std::chrono::steady_clock::now();
  f();
  const auto t1 = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(t1 - t0).count();
}
}  // namespace

int main(int argc, char** argv) {
  SlepcInitialize(&argc, &argv, nullptr, nullptr);
  {
    const int n = (argc > 1) ? std::atoi(argv[1]) : 48;
    const int frames = (argc > 2) ? std::atoi(argv[2]) : 10;
    const double dt = (argc > 3) ? std::atof(argv[3]) : 0.02;
    const double L = 1.4, h = 2.0 * L / n;
    const MacGrid g(n, n, n, h, Vec3{-L, -L, -L});
    const double R = 0.7, Gamma = 1.0, core = 0.15, hbar = Gamma / (2.0 * M_PI);

    std::printf("pipeline profile: %d^3 = %d cells (%d DOFs), %d frames, dt=%.3f\n\n", n,
                g.numCells(), 2 * g.numCells(), frames, dt);
    std::printf("%5s | %8s %8s %8s %8s %8s | %9s %6s\n", "frame", "advect", "project", "conn",
                "eigen", "extract", "total", "fps");

    // Geometric-MG pressure projector (matrix-free, warm-started across frames).
    MacProjector projector(g);
    FaceField u = projector.project(vortexRingFaceField(g, {0, 0, -1.0}, {0, 0, 1}, R, Gamma, core));

    // Decoupled extraction cadence: advect+project every frame; extract the
    // filament (connection + eigensolve + trace/link) only every k-th frame.
    const int extractEvery = (argc > 4) ? std::atoi(argv[4]) : 1;

    GaugeEigenOptions opts;
      opts.relativeGsDrop = false;  // live path: keep the absolute-drop warm-start early-exit
    opts.tol = 1e-7;
    std::vector<std::complex<double>> prev;
    double sAdv = 0, sProj = 0, sConn = 0, sEig = 0, sExt = 0;
    int counted = 0;

    for (int f = 0; f < frames; ++f) {
      FaceField advected;
      const double tAdv = timeMs([&] { advected = advectCovectorBFECC(g, u, u, dt); });
      const double tProj = timeMs([&] { u = projector.project(advected); });

      double tConn = 0, tEig = 0, tExt = 0;
      const bool extract = (f % extractEvery == 0);
      if (extract) {
        GaugeLattice lat;
        tConn = timeMs([&] {
          const auto theta = connectionAngles(g, u, hbar);
          lat = gaugeLatticeFromFaces(g, theta);
        });
        GaugeEigenResult e;
        tEig = timeMs([&] {
          e = smallestEigenpairGaugeMG(lat, prev.empty() ? nullptr : &prev, opts);
        });
        prev = e.vector;  // warm start carries from one extraction to the next
        tExt = timeMs([&] {
          const auto psi = toInterleaved(e.vector);
          (void)linkFilaments(g, traceZeroSet(g, psi));
        });
      }

      const double total = tAdv + tProj + tConn + tEig + tExt;
      std::printf("%5d%s | %8.1f %8.1f %8.1f %8.1f %8.1f | %9.1f %6.1f\n", f, extract ? "*" : " ",
                  tAdv, tProj, tConn, tEig, tExt, total, 1000.0 / total);
      if (f > 0) {  // skip frame 0 (cold eigensolve, no warm start)
        sAdv += tAdv;
        sProj += tProj;
        sConn += tConn;
        sEig += tEig;
        sExt += tExt;
        ++counted;
      }
    }

    if (counted > 0) {
      const double sum = sAdv + sProj + sConn + sEig + sExt;
      const double amortized = sum / counted;          // average over all frames
      const double simOnly = (sAdv + sProj) / counted;  // the per-frame sim floor
      std::printf("\nextract every %d frame(s).\n", extractEvery);
      std::printf("per-extraction stages: conn %.1f  eigen %.1f  extract %.1f ms\n",
                  sConn / counted * extractEvery, sEig / counted * extractEvery,
                  sExt / counted * extractEvery);
      std::printf("sim floor (advect+project): %.1f ms = %.1f fps\n", simOnly, 1000.0 / simOnly);
      std::printf("amortized full pipeline: %.1f ms = %.1f fps\n", amortized, 1000.0 / amortized);
    }
  }
  SlepcFinalize();
  return 0;
}
