#pragma once

#include <vector>

#include "grid/CooMatrix.h"

namespace bochner {

/// An eigenvalue and its (unit-norm) eigenvector.
struct EigenPair {
  double value;
  std::vector<double> vector;  ///< length = matrix dimension
  int iterations = 0;          ///< total inner work (CG its) / Lanczos steps
  int outerIterations = 0;     ///< inverse-iteration outer steps (0 for Lanczos)
  /// Stopping criterion met. False = the iteration cap was hit: the Rayleigh
  /// quotient is then still quadratically accurate, but the *vector* may be an
  /// unconverged mix (e.g. of a degenerate pair) -- callers using the vector as
  /// ground truth must check this, not just the eigenvalue.
  bool converged = false;
};

/// Preconditioner for the eigensolver inner work. The broad linear-solver
/// comparison having been settled (the connection Laplacian is a well-conditioned
/// SPD system; incomplete Cholesky is the fastest backsolve), only two survive:
/// `ICC` — incomplete Cholesky, the inner solve for the CG-based methods
/// (backward iteration, shift-invert); and `AMG` — the preconditioner for the
/// block methods (LOBPCG / Generalized Davidson), where it maps to PETSc GAMG.
enum class InnerPC { ICC, AMG };

/// \brief Smallest-eigenvalue eigenpair of a real symmetric PSD matrix, via
/// **inverse (backward) iteration** — no general eigensolver needed.
///
/// Weissmann–Pinkall extraction needs only the *lowest* mode of the connection
/// Laplacian, so we iterate \f$x \leftarrow (A+\varepsilon I)^{-1}x\f$ (normalising
/// each step): the smallest eigenpair is the dominant one of the inverse, so a
/// few solves suffice. The tiny shift \f$\varepsilon\f$ keeps the
/// system SPD/factorisable when the smallest eigenvalue is ~0 (trivial /
/// pure-gauge connection); the returned eigenvalue is the Rayleigh quotient of
/// the *un*shifted \p A. The inner SPD solve is CG + incomplete Cholesky.
///
/// \p A must be square and symmetric. The eigenvector is defined up to sign —
/// and, for the real embedding of a Hermitian operator, up to a global phase —
/// which does not affect the zero set used for filament extraction.
///
/// \param A            Square symmetric PSD matrix in COO form.
/// \param tol          Relative convergence tolerance on the Rayleigh quotient.
/// \param initialGuess Optional warm start (e.g. the previous frame's
///                     eigenvector); length must equal `A.rows()` to be used.
///                     The field changes slowly between frames, so a warm start
///                     cuts the iteration count substantially (W-P §4.5).
/// \param pc           Inner preconditioner for the SPD backsolve (\ref InnerPC).
/// \returns   The smallest eigenvalue and its eigenvector (length `A.rows()`).
/// \throws std::invalid_argument if \p A is not square.
/// \throws std::runtime_error    if the solve fails.
EigenPair smallestEigenpair(const CooMatrix& A, double tol = 1e-8,
                            const std::vector<double>* initialGuess = nullptr,
                            InnerPC pc = InnerPC::ICC);

/// \brief The same smallest eigenpair via a SLEPc **Lanczos / Krylov-Schur**
/// solve — the matvec-only baseline the Phase-3 preconditioned inverse iteration
/// is benchmarked against. No factorization or inner solve, so it sidesteps the
/// near-singular system that makes inverse iteration hard; competitive at modest
/// sizes. Optional warm start via the initial subspace.
/// \throws std::runtime_error if no eigenpair converges.
EigenPair smallestEigenpairLanczos(const CooMatrix& A, double tol = 1e-8,
                                   const std::vector<double>* initialGuess = nullptr);

/// \brief The \p nev smallest eigenpairs (SLEPc Krylov-Schur), ascending — for
/// diagnosing the low spectrum (e.g. whether two seeded rings produce a
/// near-degenerate cluster of low modes that a single-vector solve would return
/// as an arbitrary mix). The real 2x2 embedding doubles every
/// eigenvalue, so expect pairs.
/// \throws std::runtime_error if fewer than \p nev converge.
std::vector<EigenPair> smallestEigenpairsLanczos(const CooMatrix& A, int nev, double tol = 1e-8);

/// \brief The smallest eigenpair via **shift-invert Krylov-Schur** (SLEPc) —
/// Krylov on \f$(A-\sigma I)^{-1}\f$ with \f$\sigma=0\f$, so the smallest
/// eigenvalue maps to the *dominant*, well-separated mode \f$1/\lambda_{\min}\f$.
///
/// This is the natural method once the method survey showed the SPD solve is only
/// moderately conditioned (the *linear* solve is easy; the eigen-*gap* is the
/// hardness). Each Arnoldi step is one preconditioned SPD solve (CG + \p pc),
/// which shift-invert amplifies into very fast convergence on the lowest mode —
/// the standard "smallest few eigenvalues" approach, and the missing point in
/// the comparison (the \ref smallestEigenpairLanczos baseline is the σ=0
/// *matvec-only* variant). Requires \p A nonsingular at σ=0 (i.e. a genuinely
/// frustrated, SPD connection Laplacian; a trivial/pure-gauge λ_min≈0 would
/// need a small shift).
/// \throws std::runtime_error if no eigenpair converges.
EigenPair smallestEigenpairShiftInvert(const CooMatrix& A, double tol = 1e-8,
                                       const std::vector<double>* initialGuess = nullptr,
                                       InnerPC pc = InnerPC::AMG);

/// \brief The smallest eigenpair via **preconditioned LOBPCG** (SLEPc) — a
/// *block* method, the Phase-3 answer to the finding that the limiter
/// is the **clustered/degenerate low spectrum**, not the inner solve.
///
/// The real 2×2 embedding makes every eigenvalue of the connection
/// Laplacian a degenerate pair, so the gap *within* the lowest pair is exactly
/// zero — single-vector inverse iteration converges to an arbitrary mix and is
/// governed by the gap to the *next* pair. LOBPCG carries a block of \p blockSize
/// trial vectors, so it captures the whole degenerate pair at once and converges
/// on the gap to the next *distinct* eigenvalue. Unlike Lanczos it accepts a
/// **preconditioner** (\p pc via SLEPc `STPRECOND`), so it can ride the same
/// structured preconditioners the inverse-iteration path uses (even-odd,
/// geometric MG, Wilson–Dirac aggregation) while sidestepping the inner KSP.
///
/// \param A          Square symmetric PSD matrix in COO form.
/// \param tol        Relative convergence tolerance on the eigenpair.
/// \param initialGuess Optional warm start (previous frame's eigenvector).
/// \param pc         Preconditioner applied through `STPRECOND` (None = plain).
/// \param blockSize  LOBPCG block size; ≥2 to span the degenerate lowest pair.
/// \returns   The smallest eigenvalue and its eigenvector (length `A.rows()`).
/// \throws std::invalid_argument if \p A is not square or \p blockSize < 1.
/// \throws std::runtime_error    if no eigenpair converges.
EigenPair smallestEigenpairLOBPCG(const CooMatrix& A, double tol = 1e-7,
                                  const std::vector<double>* initialGuess = nullptr,
                                  InnerPC pc = InnerPC::AMG, int blockSize = 4);

/// \brief The smallest eigenpair via **Generalized Davidson** (SLEPc `EPSGD`) —
/// a preconditioned subspace eigensolver, the family built to exploit a good
/// initial guess + preconditioner (so a strong candidate in the warm-start
/// regime identified as the interactivity lever).
///
/// Unlike Krylov-Schur it grows the search space by the *preconditioned*
/// residual (no Krylov structure, no inner solve-to-tolerance), so it can warm
/// up from the previous frame's eigenvector and converge in a handful of
/// expansions. Preconditioner via `STPRECOND` (\p pc); `AMG` maps to PETSc GAMG
/// for the same block-apply reason as \ref smallestEigenpairLOBPCG.
/// \throws std::runtime_error if no eigenpair converges.
EigenPair smallestEigenpairDavidson(const CooMatrix& A, double tol = 1e-7,
                                    const std::vector<double>* initialGuess = nullptr,
                                    InnerPC pc = InnerPC::AMG);

/// \brief Largest eigenvalue (SLEPc Lanczos) — for reporting the true condition
/// number `lambda_max / lambda_min` of the SPD connection Laplacian.
EigenPair largestEigenvalue(const CooMatrix& A, double tol = 1e-6);

}  // namespace bochner
