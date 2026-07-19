#pragma once

#include <complex>
#include <functional>
#include <vector>

namespace bochner {

/// \brief Fiber-agnostic block Rayleigh-quotient iteration (block LOBPCG):
/// the lowest \p m eigenpairs of a Hermitian positive-(semi)definite operator,
/// preconditioned by a caller-supplied approximate inverse.
///
/// This is the block generalization of the 3-term single-vector covMG-LOBPCG in
/// GaugeEigen/SunGauge: the trial space per step is
/// [X, M(residuals), X_prev] (up to 3m vectors after modified Gram--Schmidt
/// with a drop tolerance), Rayleigh--Ritz picks the lowest m Ritz pairs, and
/// the previous block supplies the conjugate directions. Blocks resolve
/// degenerate and clustered levels (e.g.\ Landau levels) that single-vector
/// iteration cannot separate, and amortize the operator hierarchy across all
/// m vectors -- the "distillation" use case (lowest-N eigenspace of the
/// SU(3) covariant Laplacian).
///
/// The operator and preconditioner enter as callables so the same engine
/// serves the U(1) lattice (GaugeEigen) and the SU(d) lattice (SunGauge);
/// both wrappers pass one gauge-covariant V-cycle as \p prec.
struct BlockEigOptions {
  int maxIters = 500;     ///< outer block iterations
  double tol = 1e-7;      ///< per-vector relative eigen-residual target
  double dropTol = 1e-7;  ///< modified-Gram--Schmidt drop tolerance
  /// Interpret \ref dropTol RELATIVE to each direction's
  /// pre-orthogonalization norm (the correct linear-dependence test; matches
  /// GaugeEigenOptions::relativeGsDrop). False = legacy absolute threshold.
  bool relativeDrop = true;
  /// Converge when the first `wanted` (lowest) pairs meet \ref tol; 0 = all m.
  /// A block larger than `wanted` supplies GUARD vectors: a wanted pair's
  /// convergence rate is governed by its gap to the first Ritz value BEYOND
  /// the block, so guards keep the rate nonzero even when the block boundary
  /// would otherwise cut through a degenerate cluster (e.g.\ a Landau level).
  int wanted = 0;
  /// Absolute certificate floor: residuals divide by max(|rho|, certFloor).
  /// 0 = purely relative (structurally unattainable as lambda -> 0; see
  /// GaugeEigenOptions::certFloorRel, from which the lattice wrappers set
  /// this as certFloorRel * the operator's diagonal scale).
  double certFloor = 0.0;
  /// Hard-lock certified pairs: a wanted pair whose residual certifies is
  /// FROZEN (removed from the active block; later iterates stay orthogonal
  /// to it), so it can never regress and the active block shrinks as pairs
  /// land. Cures the certification-flicker tail measured at large n x wide
  /// bands (nearly-converged pairs bouncing across tol under Rayleigh-Ritz
  /// redistribution while termination demanded ALL pairs below tol
  /// simultaneously). Default OFF: measured on the smooth SU(3) band, hard
  /// locking cuts wall time ~30% at wide bands (the active block shrinks)
  /// but can COST iterations at healthy sizes (n=32/m=20: 28 -> 34; the
  /// frozen pairs' directions leave the shared Rayleigh-Ritz space, and the
  /// remaining pairs converge in a strictly poorer subspace -- the classical
  /// hard-vs-soft-locking trade-off). \ref softLockConverged keeps the
  /// subspace instead and is the better default choice of the two.
  bool lockConverged = false;
  /// Soft-lock certified pairs (Knyazev / scipy-LOBPCG active mask): a wanted
  /// pair whose residual is CURRENTLY below tol contributes no
  /// preconditioned-residual or conjugate-direction column to the trial basis
  /// (saving one preconditioner apply and two operator applies per certified
  /// pair per iteration), while its X column STAYS in the Rayleigh-Ritz
  /// space -- the remaining pairs keep the full trial subspace, avoiding hard
  /// locking's iteration cost at healthy sizes. The mask is re-evaluated
  /// every iteration: a pair that drifts back above tol regains its search
  /// directions and gates termination again, so the stopping rule -- ALL
  /// wanted pairs below tol simultaneously -- and the exit certificate are
  /// IDENTICAL to the default path (a sticky once-certified-always-certified
  /// variant was tried first and rejected: it exits ~2 iterations earlier but
  /// can return a pair drifted marginally above tol, i.e. uncertified).
  /// Ignored when \ref lockConverged is set (hard locking wins). Default OFF
  /// (published behavior).
  bool softLockConverged = false;
};

struct BlockEigResult {
  std::vector<double> eigenvalues;                        ///< ascending, size m
  std::vector<std::vector<std::complex<double>>> vectors; ///< m orthonormal columns
  std::vector<double> residuals;  ///< relative eigen-residual per returned pair (guards included)
  int iterations = 0;
  /// Max over the `wanted` lowest pairs (recomputed at exit) -- the certified
  /// set, matching the stopping rule; guard residuals are excluded but remain
  /// visible in \ref residuals.
  double maxResidual = 0.0;
  bool converged = false;    ///< maxResidual < tol
};

using BlockApplyFn =
    std::function<std::vector<std::complex<double>>(const std::vector<std::complex<double>>&)>;

/// Lowest \p m eigenpairs of the Hermitian operator \p apply (dimension \p n)
/// preconditioned by \p prec. \p guess optionally seeds up to m columns
/// (e.g. a previous solve for warm starts); missing columns are filled with a
/// deterministic pseudo-random basis.
BlockEigResult blockLobpcg(std::size_t n, int m, const BlockApplyFn& apply,
                         const BlockApplyFn& prec,
                         const std::vector<std::vector<std::complex<double>>>* guess = nullptr,
                         const BlockEigOptions& opts = {});

}  // namespace bochner
