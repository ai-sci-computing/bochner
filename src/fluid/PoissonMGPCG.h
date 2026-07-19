#pragma once

#include <vector>

#include "grid/CooMatrix.h"
#include "grid/SparseMg.h"  // SpMat, MgLevel, MgHierarchy

namespace bochner {

/// \file
/// A multigrid-preconditioned CG (MGPCG) solver for the cell-centered pressure
/// Poisson, including the obstacle case. Jacobi-preconditioned CG on a 3-D
/// Poisson needs O(grid dimension) iterations (hundreds at interactive
/// resolution); one V-cycle of a geometric-aggregation multigrid as the CG
/// preconditioner cuts that to ~10 -- mesh-independently -- which is the standard
/// fast fluid Poisson solve (McAdams et al. 2010, Bridson 2015).
///
/// The multigrid is **Galerkin on the assembled matrix**: the coarse operator is
/// \f$P^\top A P\f$ for a piecewise-constant 2x2x2 aggregation \f$P\f$, so the
/// solid mask, the open-outlet Robin diagonal, and the pinned cells that \ref
/// pressureLaplacianObstacle bakes into \f$A\f$ are all inherited automatically.
/// The V-cycle uses a symmetric (weighted-Jacobi) smoother and \f$R=P^\top\f$, so
/// it is an SPD preconditioner -- valid for CG.

/// Compress a COO matrix to \ref SpMat (rows sorted, duplicates summed).
SpMat toSpMat(const CooMatrix& A);

/// Build the Galerkin-aggregation hierarchy for a cell-indexed Poisson operator
/// \p A on an \p nx x \p ny x \p nz grid. \p active marks the free cells (length
/// `nx*ny*nz`); solid/pinned cells are excluded from the coarse problem.
MgHierarchy buildPoissonMgHierarchy(int nx, int ny, int nz, const SpMat& A,
                                    const std::vector<char>& active);

/// Solve `A x = b` (A = the finest level of \p H) by CG preconditioned with one
/// V-cycle of \p H. Warm-starts from \p x. Returns the iteration count; if
/// \p relResidualOut is non-null, reports the final `||b - A x|| / ||b||`.
int mgpcgSolve(const MgHierarchy& H, const std::vector<double>& b, std::vector<double>& x,
               double tol, int maxit, double* relResidualOut = nullptr);

}  // namespace bochner
