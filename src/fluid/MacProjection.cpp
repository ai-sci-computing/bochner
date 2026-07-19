#include "fluid/MacProjection.h"

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <unordered_set>
#include <utility>

#ifdef BOCHNER_WITH_METAL
#include "gpu/MetalContext.h"
#endif

namespace bochner {

namespace {

// Zero every face that touches a solid cell (no-penetration around the
// obstacle). Faces internal to the solid are zeroed too (no flow inside), and
// so are domain-wall faces of solid boundary cells: a mask reaching a wall
// would otherwise keep the inlet/open flux enforceNoPenetration wrote there,
// pumping mass into a solid cell whose RHS row is zeroed -- silently discarded.
// (Shipped demos keep obstacles interior, so this only bites wall-touching masks.)
void zeroSolidFaces(const MacGrid& g, FaceField& u, const SolidMask& solid) {
  if (solid.empty()) return;
  for (int i = 0; i <= g.nx(); ++i)
    for (int j = 0; j < g.ny(); ++j)
      for (int k = 0; k < g.nz(); ++k)
        if ((i > 0 && isSolid(solid, g.cellIndex(i - 1, j, k))) ||
            (i < g.nx() && isSolid(solid, g.cellIndex(i, j, k))))
          u.x[g.faceXIndex(i, j, k)] = 0.0;
  for (int i = 0; i < g.nx(); ++i)
    for (int j = 0; j <= g.ny(); ++j)
      for (int k = 0; k < g.nz(); ++k)
        if ((j > 0 && isSolid(solid, g.cellIndex(i, j - 1, k))) ||
            (j < g.ny() && isSolid(solid, g.cellIndex(i, j, k))))
          u.y[g.faceYIndex(i, j, k)] = 0.0;
  for (int i = 0; i < g.nx(); ++i)
    for (int j = 0; j < g.ny(); ++j)
      for (int k = 0; k <= g.nz(); ++k)
        if ((k > 0 && isSolid(solid, g.cellIndex(i, j, k - 1))) ||
            (k < g.nz() && isSolid(solid, g.cellIndex(i, j, k))))
          u.z[g.faceZIndex(i, j, k)] = 0.0;
}

}  // namespace

MacProjector::MacProjector(const MacGrid& g, PoissonMgOptions opts, BoundarySpec bc, SolidMask solid)
    : g_(g), opts_(opts), bc_(bc), solid_(std::move(solid)), phi_(g.numCells(), 0.0) {}

MacProjector::~MacProjector() {
#ifdef BOCHNER_WITH_METAL
  if (gpuHandle_ >= 0) gpu::freePoisson(gpuHandle_);
  if (geomHandle_ >= 0) gpu::freePoissonGeom(geomHandle_);
#endif
}

void MacProjector::buildSystem(const CooMatrix& A) {
  SpMat m = toSpMat(A);
  std::vector<char> active(g_.numCells(), 1);
  for (int c = 0; c < g_.numCells(); ++c)
    if (isSolid(solid_, c)) active[c] = 0;  // solid cells carry no pressure DOF
  for (int c : pinned_) active[c] = 0;       // pinned Dirichlet cells
  mg_ = buildPoissonMgHierarchy(g_.nx(), g_.ny(), g_.nz(), m, active);
  sysReady_ = true;
}

int MacProjector::solvePoisson(const std::vector<double>& rhs, double* relRes) {
  double rr = 0.0;
  int its = 0;
#ifdef BOCHNER_WITH_METAL
  if (useGpu_ && gpu::metalAvailable()) {
    if (gpuHandle_ < 0) gpuHandle_ = gpu::uploadPoisson(mg_);
    try {
      its = gpu::mgpcgSolvePoisson(gpuHandle_, rhs, phi_, /*tol=*/1e-4, /*maxit=*/200, &rr);
      if (rr > 1e-3)
        std::fprintf(stderr, "MacProjector(GPU): Poisson did not converge (relRes=%.2e)\n", rr);
      if (relRes) *relRes = rr;
      return its;
    } catch (const std::exception& e) {
      // GPU fault (command-buffer error): phi_ is untouched by a thrown GPU solve,
      // so fall through to the authoritative CPU MGPCG from the same warm start.
      std::fprintf(stderr, "MacProjector(GPU): %s -- falling back to CPU MGPCG\n", e.what());
    }
  }
#endif
  its = mgpcgSolve(mg_, rhs, phi_, /*tol=*/1e-8, /*maxit=*/200, &rr);
  if (rr > 1e-6)
    std::fprintf(stderr, "MacProjector: Poisson did not converge (relRes=%.2e)\n", rr);
  if (relRes) *relRes = rr;
  return its;
}

FaceField MacProjector::project(const FaceField& u, int* cycles) {
  // Obstacle present: solid-aware Poisson (interior no-flux around the solid),
  // solved by Jacobi-CG. Faces touching the solid are zeroed (no penetration) and
  // left uncorrected; the open outlet carries the through-flow.
  if (!solid_.empty()) {
    FaceField out = u;
    ops::enforceNoPenetration(g_, out, bc_);
    zeroSolidFaces(g_, out, solid_);
    std::vector<double> rhs = ops::divergence(g_, out);
    for (double& r : rhs) r = -r;
    if (!sysReady_) buildSystem(pressureLaplacianObstacle(g_, bc_, solid_, &pinned_));
    for (int c = 0; c < g_.numCells(); ++c)
      if (isSolid(solid_, c)) rhs[c] = 0.0;  // solid cells pinned (phi = 0)
    for (int c : pinned_) rhs[c] = 0.0;       // pinned fluid cell: phi = 0
    const int its = solvePoisson(rhs, nullptr);
    if (cycles) *cycles = its;
    FaceField grad = ops::gradient(g_, phi_, bc_);
    zeroSolidFaces(g_, grad, solid_);  // no gradient correction across the solid surface
    for (size_t f = 0; f < out.x.size(); ++f) out.x[f] -= grad.x[f];
    for (size_t f = 0; f < out.y.size(); ++f) out.y[f] -= grad.y[f];
    for (size_t f = 0; f < out.z.size(); ++f) out.z[f] -= grad.z[f];
    return out;
  }

  // Open (free-surface) walls: Dirichlet p=0 Poisson. Only the CLOSED walls get
  // no-penetration; the open walls carry outflow, which enters the divergence
  // natively and gets an outflow gradient correction.
  if (bc_.anyOpen()) {
    FaceField out = u;
    ops::enforceNoPenetration(g_, out, bc_);
    std::vector<double> rhs = ops::divergence(g_, out);
    for (double& r : rhs) r = -r;
    if (!sysReady_) buildSystem(pressureLaplacianBC(g_, bc_));
    const int its = solvePoisson(rhs, nullptr);
    if (cycles) *cycles = its;
    const FaceField grad = ops::gradient(g_, phi_, bc_);
    for (size_t f = 0; f < out.x.size(); ++f) out.x[f] -= grad.x[f];
    for (size_t f = 0; f < out.y.size(); ++f) out.y[f] -= grad.y[f];
    for (size_t f = 0; f < out.z.size(); ++f) out.z[f] -= grad.z[f];
    return out;
  }

  // Closed no-penetration domain: enforce u.n = 0 on the walls (the inhomogeneous
  // Neumann BC -- the projection removes the wall-normal flux, not just the
  // interior divergence). With the walls zeroed, div(u) sums to exactly zero, so
  // the all-Neumann Poisson is compatible; the mean-removal below then only
  // cleans up round-off before the unpinned MG solve.
  FaceField out = u;
  ops::enforceNoPenetration(g_, out);

  std::vector<double> rhs = ops::divergence(g_, out);
  double mean = 0.0;
  for (double r : rhs) mean += r;
  mean /= rhs.size();
  for (double& r : rhs) r = -r + mean;

  double relRes = 0.0;
  bool solved = false;
#ifdef BOCHNER_WITH_METAL
  // Opt-in GPU: the SAME unpinned matrix-free geometric multigrid on Metal (float,
  // resident hierarchy, warm-started phi_). The pinned assembled MGPCG stalls on
  // this all-Neumann box; the geometric MG keeps its fast convergence in float, so
  // this is the closed-box projection GPU route that actually wins.
  if (useGpu_ && gpu::metalAvailable()) {
    if (geomHandle_ < 0)
      geomHandle_ = gpu::uploadPoissonGeom(g_.nx(), g_.ny(), g_.nz(), g_.spacing());
    const int cyc = gpu::poissonGeomSolve(geomHandle_, rhs, phi_, opts_.tol, opts_.maxCycles,
                                          opts_.nu1, opts_.nu2, opts_.coarseSweeps, opts_.omega,
                                          &relRes);
    if (cycles) *cycles = cyc;
    solved = true;
  }
#endif
  if (!solved) {
    const PoissonMgResult res = poissonVcycleSolve(g_.nx(), g_.ny(), g_.nz(), g_.spacing(), rhs, phi_,
                                                   opts_);  // phi_ warm-started in/out
    if (cycles) *cycles = res.cycles;
    relRes = res.relResidual;
    if (relRes > opts_.tol)
      std::fprintf(stderr,
                   "MacProjector: closed Poisson did not converge (relRes=%.2e, tol=%.1e)\n", relRes,
                   opts_.tol);
  } else if (relRes > 1e-3) {
    // The GPU float geometric MG floors a little above 1e-6 (expected); only a
    // gross residual signals genuine non-convergence.
    std::fprintf(stderr, "MacProjector(GPU): closed Poisson did not converge (relRes=%.2e)\n",
                 relRes);
  }

  const FaceField grad = ops::gradient(g_, phi_);  // zero on the walls -> out.n stays 0
  for (size_t f = 0; f < out.x.size(); ++f) out.x[f] -= grad.x[f];
  for (size_t f = 0; f < out.y.size(); ++f) out.y[f] -= grad.y[f];
  for (size_t f = 0; f < out.z.size(); ++f) out.z[f] -= grad.z[f];
  return out;
}

}  // namespace bochner
