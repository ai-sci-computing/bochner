#pragma once

#include <memory>
#include <vector>

#include "grid/CooMatrix.h"

namespace bochner {

/// Preconditioner for \ref solveSpdCG. `Jacobi` is the cheap baseline; `ICC`
/// (incomplete Cholesky) is the standard moderate-size choice; `AMG` (hypre
/// BoomerAMG) is mesh-independent and ideal for the real scalar Poisson of the
/// pressure projection (where there is no connection to break it).
enum class SpdPC { Jacobi, ICC, AMG };

/// \brief Solve a symmetric positive-definite system `A x = b` with PETSc.
///
/// Bridges the backend-neutral ::bochner::CooMatrix to PETSc: assembles a
/// sequential PETSc `Mat`, wraps \p b in a `Vec`, and solves with a Krylov
/// method (conjugate gradients). This is the first concrete use of the PETSc
/// backend and the kernel that the Phase-3 preconditioner study will extend
/// (the connection-Laplacian solves inside inverse iteration are SPD solves of
/// exactly this shape).
///
/// \param A     Square SPD matrix in COO form.
/// \param b     Right-hand side; size must equal `A.rows()`.
/// \param rtol  Relative residual tolerance for the Krylov solve.
/// \param pc    Preconditioner (\ref SpdPC).
/// \param iters Optional out: number of CG iterations taken.
/// \returns     The solution vector `x` of length `A.rows()`.
/// \throws std::invalid_argument if \p A is not square or sizes mismatch.
/// \throws std::runtime_error    if the Krylov solver fails to converge.
std::vector<double> solveSpdCG(const CooMatrix& A, const std::vector<double>& b,
                               double rtol = 1e-10, SpdPC pc = SpdPC::Jacobi,
                               int* iters = nullptr);

/// \brief A reusable SPD solver for a **fixed** matrix and many right-hand sides.
///
/// Assembles the PETSc `Mat`, builds the KSP/PC, and performs the preconditioner
/// setup (e.g. the AMG hierarchy, or an incomplete-Cholesky factorisation) once
/// in the constructor; each \ref solve then reuses it for a new RHS. This is the
/// right shape for the pressure projection, whose Laplacian is **constant across
/// frames** -- only the divergence RHS changes -- so the (otherwise dominant)
/// AMG setup is paid once instead of every frame.
class CachedSpdSolver {
public:
  CachedSpdSolver(const CooMatrix& A, SpdPC pc = SpdPC::AMG, double rtol = 1e-6);
  ~CachedSpdSolver();
  CachedSpdSolver(CachedSpdSolver&&) noexcept;
  CachedSpdSolver& operator=(CachedSpdSolver&&) noexcept;
  CachedSpdSolver(const CachedSpdSolver&) = delete;
  CachedSpdSolver& operator=(const CachedSpdSolver&) = delete;

  /// Solve `A x = b` (b sized to the matrix). \p iters out: CG iterations taken.
  std::vector<double> solve(const std::vector<double>& b, int* iters = nullptr) const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace bochner
