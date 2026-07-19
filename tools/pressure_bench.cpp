/// \file
/// Pressure-projection solver study. The projection is a standard *real scalar*
/// Poisson solve (CF defers to Bridson 2015 / the Qu et al. 2019 codebase) -- so
/// the right tools are the classical ones (ICC, MGPCG/AMG), not our C-valued
/// gauge multigrid. Here we sweep preconditioner x tolerance on a realistic
/// divergent field (one BFECC-advected ring step), reporting CG iterations and
/// wall-clock, to pick the projection solver. The current default (CG+Jacobi,
/// rtol 1e-10) is the baseline to beat.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <slepc.h>

#include "grid/GridOperators.h"
#include "fluid/MacAdvection.h"
#include "grid/MacGrid.h"
#include "fluid/MacProjection.h"
#include "fluid/MacVortexRing.h"
#include "solvers/PetscSolver.h"
#include "solvers/PoissonMultigrid.h"

using namespace bochner;

namespace {
template <class F>
double timeMs(F&& f) {
  const auto t0 = std::chrono::steady_clock::now();
  f();
  const auto t1 = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(t1 - t0).count();
}
const char* pcName(SpdPC p) {
  return p == SpdPC::Jacobi ? "Jacobi" : p == SpdPC::ICC ? "ICC" : "AMG(hypre)";
}
}  // namespace

int main(int argc, char** argv) {
  SlepcInitialize(&argc, &argv, nullptr, nullptr);
  {
    const int n = (argc > 1) ? std::atoi(argv[1]) : 48;
    const double L = 1.4, h = 2.0 * L / n;
    const MacGrid g(n, n, n, h, Vec3{-L, -L, -L});
    const double R = 0.7, Gamma = 1.0, core = 0.15;

    // Realistic divergent RHS: seed + project, then one BFECC advection step.
    FaceField u = projectToDivergenceFree(g, vortexRingFaceField(g, {0, 0, -1.0}, {0, 0, 1}, R,
                                                                 Gamma, core));
    u = advectCovectorBFECC(g, u, u, 0.02);

    const CooMatrix A = pressureLaplacian(g, {0});
    std::vector<double> rhs = ops::divergence(g, u);
    for (double& r : rhs) r = -r;
    rhs[0] = 0.0;

    std::printf("Pressure Poisson on %d^3 = %d cells. Operator is CONSTANT across frames;\n", n,
                g.numCells());
    std::printf("only the RHS changes -- so any setup (ICC/AMG hierarchy) is amortisable.\n\n");
    std::printf("%-12s %8s | %5s %9s | %5s %9s\n", "PC", "", "it", "ms@1e-6", "it", "ms@1e-10");

    for (SpdPC pc : {SpdPC::Jacobi, SpdPC::ICC, SpdPC::AMG}) {
      int it6 = 0, it10 = 0;
      double ms6 = 0, ms10 = 0;
      ms6 = timeMs([&] { (void)solveSpdCG(A, rhs, 1e-6, pc, &it6); });
      ms10 = timeMs([&] { (void)solveSpdCG(A, rhs, 1e-10, pc, &it10); });
      std::printf("%-12s %8s | %5d %9.1f | %5d %9.1f\n", pcName(pc), "", it6, ms6, it10, ms10);
    }
    std::printf("\n(baseline today: CG+Jacobi @ 1e-10)\n");

    // The operator is constant -> cache it. Build the solver once (paying the AMG
    // setup), then time the per-frame solve (what actually runs each frame).
    std::printf("\nCached (setup once, then per-frame solve), AMG @ 1e-6:\n");
    double setupMs = 0;
    CachedSpdSolver* solver = nullptr;
    setupMs = timeMs([&] { solver = new CachedSpdSolver(A, SpdPC::AMG, 1e-6); });
    int it = 0;
    double solveMs = 0;
    for (int rep = 0; rep < 5; ++rep) solveMs = timeMs([&] { (void)solver->solve(rhs, &it); });
    std::printf("  setup %.1f ms (once), per-frame solve %.1f ms (%d its)\n", setupMs, solveMs, it);
    delete solver;

    // Our real-scalar geometric multigrid (matrix-free -> NO setup, parallel).
    // Solve the unpinned Neumann system (mean-zero rhs; the projection's grad
    // removes the constant) -- the standard MGPCG-style pressure solver.
    std::vector<double> rhs0 = ops::divergence(g, u);
    double mean = 0.0;
    for (double r : rhs0) mean += r;
    mean /= rhs0.size();
    for (double& r : rhs0) r = -r + mean;  // -div(u), projected to mean-zero
    bochner::PoissonMgOptions mopts;
    mopts.tol = 1e-6;
    std::vector<double> phi(g.numCells(), 0.0);
    bochner::PoissonMgResult mr;
    const double mgMs = timeMs([&] { mr = bochner::poissonVcycleSolve(g.nx(), g.ny(), g.nz(),
                                                                      g.spacing(), rhs0, phi, mopts); });
    std::printf("\nGeometric MG (ours, real scalar, matrix-free, no setup) @ 1e-6:\n");
    std::printf("  per-frame solve %.1f ms (%d V-cycles), no setup\n", mgMs, mr.cycles);
  }
  SlepcFinalize();
  return 0;
}
