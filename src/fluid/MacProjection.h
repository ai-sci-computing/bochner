#pragma once

#include <vector>

#include "grid/CooMatrix.h"
#include "grid/GridOperators.h"
#include "grid/MacGrid.h"
#include "grid/MacObstacle.h"
#include "fluid/PoissonMGPCG.h"
#include "solvers/PoissonMultigrid.h"

namespace bochner {

/// \brief The SPD pressure-Poisson matrix on cell centers, \f$A = -\nabla\!\cdot\nabla\f$.
///
/// This is the homogeneous-Neumann 7-point Laplacian of the MAC grid, negated
/// to be symmetric positive (semi-)definite, assembled as a graph Laplacian
/// over interior faces (boundary faces contribute no coupling -- the natural
/// no-penetration Neumann condition). Cells listed in
/// \p dirichletCells are pinned to zero: their rows/columns are removed and a
/// unit diagonal is set, which makes the matrix strictly positive definite.
///
/// With \p dirichletCells empty the matrix is the pure-Neumann Laplacian and is
/// singular (constant null space); callers must pin at least one cell.
CooMatrix pressureLaplacian(const MacGrid& g, const std::vector<int>& dirichletCells);

/// \brief BC-aware pressure Laplacian \f$A = -\nabla\!\cdot\nabla\f$ for mixed
/// no-penetration (Neumann) / free-surface (Dirichlet p=0) walls.
///
/// Interior faces couple their two cells as in \ref pressureLaplacian. A CLOSED
/// wall contributes nothing (natural Neumann). Each OPEN wall face adds `+w` to
/// its adjacent cell's diagonal -- the coupling to a ghost cell pinned to `p=0`
/// -- which makes the matrix strictly positive definite (no null space, no pin).
/// If no wall is open the system is the singular pure-Neumann Laplacian, so
/// cell 0 is pinned (as in \ref pressureLaplacian) to fix the constant.
CooMatrix pressureLaplacianBC(const MacGrid& g, const BoundarySpec& bc);

/// \brief BC-aware pressure Laplacian with an interior solid obstacle.
///
/// A face is dropped (no coupling, no diagonal) when either of its cells is
/// solid -- the interior no-flux (Neumann) obstacle surface. Solid cells get a
/// unit diagonal (pinned, disconnected). Open walls add a Robin ghost diagonal
/// on their fluid boundary cells (SPD). To keep the matrix strictly positive
/// definite, every fluid connected component (through fluid-fluid faces) needs
/// one pressure ground: components that touch an open wall are grounded by that
/// Robin diagonal; any component that touches no open wall -- the whole domain
/// when no wall is open, or a fluid pocket sealed off by solid (a concave/shell
/// obstacle) -- gets one of its cells pinned to `p=0`. \p pinnedOut, if given,
/// receives the pinned fluid cells (so the caller can zero their RHS).
CooMatrix pressureLaplacianObstacle(const MacGrid& g, const BoundarySpec& bc, const SolidMask& solid,
                                    std::vector<int>* pinnedOut = nullptr);

/// \brief Project a face velocity field onto its discretely divergence-free
/// part (the MAC pressure projection / discrete Helmholtz-Hodge decomposition).
///
/// Solves `A phi = -div(u)` with one cell pinned to fix the constant null space
/// (closed domain: all walls no-penetration), then returns `u - grad(phi)`,
/// which satisfies `div(.) = 0` in every cell. Boundary faces are left
/// untouched (their values are BC-prescribed); prescribed boundary normal
/// velocities enter `div(u)` natively, so no source-term hack is needed
///.
FaceField projectToDivergenceFree(const MacGrid& g, const FaceField& u);

/// \brief Reusable closed-domain pressure projector for a live simulation --
/// the standard real-scalar Poisson solve (CF / Bridson 2015), solved by our
/// matrix-free geometric multigrid (\ref poissonVcycleSolve).
///
/// It solves the **unpinned** homogeneous-Neumann system (the projection's
/// gradient annihilates the constant null space, so no pin is needed), and
/// **warm-starts** the pressure field from the previous frame -- the pressure
/// changes slowly between frames, so the prior solution is a near-converged
/// start. Matrix-free, so there is no per-frame setup (unlike a cached AMG
/// hierarchy), and parallel. Being stateful (the warm start), \ref project is
/// non-const.
/// The pressure boundary condition is selectable: the default is
/// the closed no-penetration box (solved by the fast geometric multigrid); if
/// any wall of \p bc is open, the free-surface Dirichlet Poisson is assembled
/// and solved by a self-contained Jacobi-CG (no PETSc), so open runs stay as
/// portable as closed ones -- correctness over speed for that path.
class MacProjector {
public:
  /// \p solid marks obstacle cells (empty = no obstacle). A non-empty mask forces
  /// the solid-aware path: faces touching a solid cell are no-penetration
  /// (Neumann) walls in the interior, so the flow is projected to go *around* the
  /// obstacle (the solid cells carry no pressure DOF).
  explicit MacProjector(const MacGrid& g, PoissonMgOptions opts = {},
                        BoundarySpec bc = BoundarySpec::allClosed(), SolidMask solid = {});
  ~MacProjector();

  /// Divergence-free part of \p u; \p cycles out: V-cycles (closed) or MGPCG
  /// iterations (open/obstacle) taken this frame.
  FaceField project(const FaceField& u, int* cycles = nullptr);

  /// Run the pressure solve on the Metal GPU (single precision) when available.
  /// Obstacle/open walls use the assembled MGPCG; the CLOSED box uses the same
  /// unpinned matrix-free geometric multigrid as the CPU default, ported to Metal
  /// (poissonGeomSolve) -- the closed all-Neumann box needs the unpinned solver to
  /// converge fast, so the assembled MGPCG is not used for it. Off by default (the
  /// authoritative CPU double solvers run). No effect unless Metal is built.
  void setGpuProjection(bool on) { useGpu_ = on; }

  const BoundarySpec& boundary() const { return bc_; }
  const SolidMask& solid() const { return solid_; }

private:
  /// Assemble the SpMat + MGPCG hierarchy from \p A once (the operator is fixed);
  /// \ref pinned_ must already hold the Dirichlet cells, and \ref solid_ the mask.
  void buildSystem(const CooMatrix& A);
  /// Solve `A phi_ = rhs` with the cached hierarchy (GPU float if enabled+built,
  /// else CPU double), warm-started; returns iterations, reports rel residual.
  int solvePoisson(const std::vector<double>& rhs, double* relRes);

  MacGrid g_;
  PoissonMgOptions opts_;
  BoundarySpec bc_;
  SolidMask solid_;          ///< obstacle cells (empty = none)
  std::vector<double> phi_;  ///< previous-frame pressure (warm start)

  // Assembled matrix + MGPCG hierarchy for the obstacle / open paths, built once
  // (the operator is fixed for the projector's life) and reused every frame.
  bool sysReady_ = false;
  std::vector<int> pinned_;  ///< pinned (Dirichlet phi=0) fluid cells
  MgHierarchy mg_;
  bool useGpu_ = false;   ///< use the Metal GPU solve (single precision) when built + available
  // Read only under BOCHNER_WITH_METAL; kept unconditionally so the class
  // layout does not depend on the backend. [[maybe_unused]] because a
  // non-Metal build never touches them.
  [[maybe_unused]] int gpuHandle_ = -1;   ///< assembled MGPCG hierarchy handle (obstacle/open); -1 = not uploaded
  [[maybe_unused]] int geomHandle_ = -1;  ///< geometric-MG hierarchy handle (closed box); -1 = not uploaded
};

}  // namespace bochner
