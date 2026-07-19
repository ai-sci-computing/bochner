#pragma once

#include <vector>

namespace bochner {

struct PoissonMgOptions {
  int nu1 = 2;          ///< pre-smoothing sweeps per level
  int nu2 = 2;          ///< post-smoothing sweeps per level
  int coarseSweeps = 30;  ///< smoothing sweeps at the coarsest level
  int maxCycles = 100;  ///< V-cycle iteration cap
  double omega = 1.0;   ///< red-black GS relaxation (1 = GS)
  double tol = 1e-6;    ///< target relative residual ||rhs - L phi|| / ||rhs||
};

struct PoissonMgResult {
  int cycles = 0;
  double relResidual = 0.0;
  /// Number of levels in the hierarchy actually built. Coarsening halts as soon
  /// as ANY dimension is odd or below 4, so a grid with an odd (or barely
  /// 2-divisible) extent yields a shallow hierarchy: at `levels == 1` the
  /// V-cycle degenerates to plain Gauss-Seidel smoothing and will not converge
  /// at any useful resolution. Callers that let a user pick the grid size
  /// should check this.
  int levels = 0;
};

/// \brief Real-scalar geometric multigrid for the MAC homogeneous-Neumann
/// pressure Poisson -- the standard, cheap pressure solver (CF defers to this;
/// Bridson 2015 / McAdams et al. 2010 MGPCG), and the right tool instead of the
/// C-valued gauge multigrid (which would double the dimension for a real field).
///
/// Solves \f$L\,\phi = \mathrm{rhs}\f$ on an `nx*ny*nz` cell-centred grid, where
/// \f$L = -\nabla\!\cdot\nabla\f$ is the 7-point homogeneous-Neumann Laplacian
/// (weight \f$1/h^2\f$). It is the trivial-connection case of \ref vcycleSolve,
/// specialised to **real scalars**: decimation hierarchy, red-black Gauss-Seidel
/// smoother, transport-free averaging transfers, rediscretised coarse operators,
/// optimal A-energy step on the coarse correction. Matrix-free, so there is no
/// per-frame setup cost (unlike a cached AMG hierarchy), and parallel (OpenMP).
///
/// The operator is singular (constant null space): the pressure projection only
/// uses \f$\nabla\phi\f$, which annihilates the constant, so no pin is needed --
/// `rhs` must be consistent (mean-zero), which the closed-domain divergence is.
///
/// \param nx,ny,nz Cell counts of the grid.
/// \param h        Cell spacing.
/// \param rhs      Right-hand side (must be mean-zero; length `nx*ny*nz`).
/// \param phi in/out: initial guess in (zero or a warm start), solution out.
/// \param opts     V-cycle controls (max cycles, tolerance, smoothing).
PoissonMgResult poissonVcycleSolve(int nx, int ny, int nz, double h,
                                   const std::vector<double>& rhs, std::vector<double>& phi,
                                   const PoissonMgOptions& opts = {});

/// Matrix-free Neumann-Laplacian matvec `y = L x` (exposed for tests).
std::vector<double> applyPoisson(int nx, int ny, int nz, double h, const std::vector<double>& x);

}  // namespace bochner
