/// \file
/// Phase-3 warm-start study: the make-or-break
/// interactivity experiment. At the target scale no method reaches >=10 fps from
/// a *cold* start, so the only remaining lever is temporal coherence -- across
/// sim frames the connection changes slowly, so the previous frame's eigenvector
/// is a near-converged start (W-P sec 4.5).
///
/// This drives the real self-advecting vortex-ring sim (the same loop as
/// ring_demo_grid: BFECC covector advection -> pressure projection), and at every
/// frame extracts the smallest eigenpair of the connection Laplacian both COLD
/// (no guess) and WARM (each method warm-started from its own previous frame's
/// eigenvector), reporting the per-frame speedup. The question it answers: does
/// warm-starting cut the per-frame eigensolve enough to reach interactivity?
///
/// Usage: warmstart_bench [n] [frames]   (default 46 8 -> ~97k cells)
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>

#include <slepc.h>

#include "BenchTiming.h"
#include "solvers/EigenSolver.h"
#include "solvers/GaugeEigen.h"
#include "solvers/GaugeMultigrid.h"
#include "grid/GridOperators.h"
#include "extraction/MacConnectionLaplacian.h"
#include "fluid/MacFluidSolver.h"
#include "grid/MacGrid.h"
#include "fluid/MacProjection.h"
#include "fluid/MacVortexRing.h"

using namespace bochner;

namespace {

using benchstat::medianMs;

// One competing method: name + a solve taking (operator, optional warm guess).
struct Method {
  std::string name;
  std::function<EigenPair(const CooMatrix&, const std::vector<double>*)> solve;
  std::vector<double> warm;  // its own previous-frame eigenvector
  double coldT = 0, warmT = 0;
  long coldIt = 0, warmIt = 0;
  int cnt = 0;  // frames counted (frame 0 excluded -- no prior to warm from)
};

}  // namespace

int gEigUncertified = 0;  // covMG-LOBPCG solves that missed their residual certificate

int main(int argc, char** argv) {
  SlepcInitialize(&argc, &argv, nullptr, nullptr);
  {
    const int n = (argc > 1) ? std::atoi(argv[1]) : 46;
    const int frames = (argc > 2) ? std::atoi(argv[2]) : 8;
    // dt sets the per-frame field change = the temporal-coherence knob: a smaller
    // dt (finer extraction cadence) should make the warm start closer/cheaper.
    const double dt = (argc > 3) ? std::atof(argv[3]) : 0.04;
    const double tolK = 1e-7, tolI = 1e-6;  // Krylov/Davidson vs inverse-iter tol

    const double L = 1.4, h = 2.0 * L / n;
    const MacGrid g(n, n, n, h, Vec3{-L, -L, -L});
    const int dofs = 2 * g.numCells();
    std::printf("warm-start study: %d^3 = %d cells (%d real DOFs), %d frames, dt=%.3f\n", n,
                g.numCells(), dofs, frames, dt);

    const double R = 0.7, Gamma = 1.0, core = 0.15;
    const double hbar = Gamma / (2.0 * M_PI);
    FaceField u = vortexRingFaceField(g, {0, 0, -1.0}, {0, 0, 1}, R, Gamma, core);
    u = projectToDivergenceFree(g, u);

    // The connection changes every frame, so the gauge lattice is rebuilt per
    // frame right next to the CooMatrix; the harness lambda captures it.
    GaugeLattice eigLat;
    std::vector<Method> methods;
    methods.push_back({"covMG-LOBPCG (ours)", [&](const CooMatrix&, const std::vector<double>* gptr) {
                         GaugeEigenOptions eo;
                         eo.tol = tolK;
                         std::vector<std::complex<double>> guess;
                         const std::vector<std::complex<double>>* gc = nullptr;
                         if (gptr) {
                           guess = toComplex(*gptr);
                           gc = &guess;
                         }
                         const GaugeEigenResult r = smallestEigenpairGaugeMG(eigLat, gc, eo);
                         if (!r.converged) ++gEigUncertified;
                         EigenPair p;
                         p.value = r.eigenvalue;
                         p.iterations = r.iterations;
                         p.vector = toInterleaved(r.vector);
                         return p;
                       }, {}, 0, 0, 0, 0, 0});
    methods.push_back({"Lanczos", [&](const CooMatrix& E, const std::vector<double>* gptr) {
                         return smallestEigenpairLanczos(E, tolK, gptr);
                       }, {}, 0, 0, 0, 0, 0});
    methods.push_back({"LOBPCG+AMG", [&](const CooMatrix& E, const std::vector<double>* gptr) {
                         return smallestEigenpairLOBPCG(E, tolK, gptr, InnerPC::AMG);
                       }, {}, 0, 0, 0, 0, 0});
    methods.push_back({"GD+AMG", [&](const CooMatrix& E, const std::vector<double>* gptr) {
                         return smallestEigenpairDavidson(E, tolK, gptr, InnerPC::AMG);
                       }, {}, 0, 0, 0, 0, 0});
    // Backward (inverse) iteration is the classic warm-start method, but also the
    // dominated, slow one -- include it only at tractable sizes.
    if (n <= 32)
      methods.push_back({"backward", [&](const CooMatrix& E, const std::vector<double>* gptr) {
                           return smallestEigenpair(E, tolI, gptr, InnerPC::ICC);
                         }, {}, 0, 0, 0, 0, 0});

    for (int frame = 0; frame < frames; ++frame) {
      const FaceField theta = connectionAngles(g, u, hbar);
      const CooMatrix E = connectionLaplacian(g, theta);
      eigLat = gaugeLatticeFromFaces(g, theta);
      std::printf("frame %2d:", frame);
      for (auto& m : methods) {
        EigenPair cold, warm;
        const double tc = medianMs([&] { cold = m.solve(E, nullptr); });
        const std::vector<double>* guess =
            (frame > 0 && m.warm.size() == static_cast<std::size_t>(dofs)) ? &m.warm : nullptr;
        const double tw = medianMs([&] { warm = m.solve(E, guess); });
        m.warm = warm.vector;  // carry to next frame
        if (frame > 0) {       // frame 0 warm == cold (no prior); exclude
          m.coldT += tc;
          m.warmT += tw;
          m.coldIt += cold.iterations;
          m.warmIt += warm.iterations;
          ++m.cnt;
        }
        std::printf("  %s %.0f/%.0fms", m.name.c_str(), tc, tw);
      }
      std::printf("\n");
      std::fflush(stdout);
      u = stepCovectorFluids(g, u, dt);  // advance the field
    }

    std::printf("\n%-12s | %10s | %10s | %7s | %9s | %9s\n", "method", "cold ms", "warm ms",
                "speedup", "cold its", "warm its");
    std::printf("%s\n", std::string(72, '-').c_str());
    for (const auto& m : methods) {
      if (m.cnt == 0) continue;
      const double c = m.coldT / m.cnt, w = m.warmT / m.cnt;
      std::printf("%-12s | %8.0fms | %8.0fms | %6.1fx | %9.1f | %9.1f\n", m.name.c_str(), c, w,
                  c / w, static_cast<double>(m.coldIt) / m.cnt,
                  static_cast<double>(m.warmIt) / m.cnt);
    }
  }
  if (gEigUncertified > 0)
    std::printf("\nWARNING: %d covMG-LOBPCG solve(s) did NOT meet the residual certificate\n",
                gEigUncertified);
  benchstat::printTimingSummary();
  SlepcFinalize();
  return 0;
}
