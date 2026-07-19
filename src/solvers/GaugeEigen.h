#pragma once

#include <complex>
#include <vector>

#include "solvers/BlockLobpcg.h"
#include "solvers/GaugeMultigrid.h"

namespace bochner {

struct GaugeEigenOptions {
  int maxIters = 300;   ///< outer Rayleigh-quotient minimization steps
  double tol = 1e-6;    ///< relative eigen-residual ||Ex - rho x|| / max(|rho|, floor)
  /// Certificate floor, relative to the operator's diagonal scale: the
  /// certified residual divides by max(|rho|, certFloorRel * diagScale). A
  /// purely relative certificate (certFloorRel = 0, the pre-2026-07-17
  /// behavior) is structurally unattainable as lambda_min -> 0: for
  /// x = v1 + eps v2, ||r|| ~ eps lambda2 while rho -> lambda_min, so the
  /// ratio DIVERGES as the iterate improves. The default floor sits orders of
  /// magnitude below every gapped operator's lambda_min (inactive there) and
  /// merely caps that divergence in the near-pure-gauge limit.
  double certFloorRel = 1e-6;
  int precCycles = 1;   ///< gauge-MG V-cycles per preconditioner apply
  int blockGuard = 2;   ///< extra guard vectors in the BLOCK solver (see BlockLobpcg.h)
  /// Hard-lock certified pairs in the BLOCK solver (BlockEigOptions::
  /// lockConverged; experimental, default off -- see the trade-off note
  /// there). The bench toggle is block_eig_bench's `lock=hard`.
  bool blockLockConverged = false;
  /// Soft-lock certified pairs in the BLOCK solver (BlockEigOptions::
  /// softLockConverged; experimental, default off -- currently-certified
  /// pairs contribute no new search directions but stay in the Rayleigh-Ritz
  /// space; stopping rule and certificate identical to the default). The
  /// bench toggle is block_eig_bench's `lock=soft`. Hard locking wins if
  /// both are set.
  bool blockSoftLockConverged = false;
  /// Modified-Gram-Schmidt drop test: RELATIVE (default, the correct
  /// linear-dependence test -- keep a direction if its orthogonal remainder
  /// exceeds 1e-7 of its own norm; the descent/Knyazev-Neymeyr statements
  /// apply to this variant at every scale) vs the legacy ABSOLUTE 1e-7
  /// (set false), which doubles as a cheap warm-start early-exit for the
  /// live pipeline (tol 1e-6) but floors the certifiable relative
  /// eigen-residual at ~3e-7 on dense-edge operators (lambda_min/w ~ 1,
  /// e.g. Monte-Carlo gauge ensembles, where rho ~ lambda_2 so the
  /// preconditioned direction's norm is itself of the order of the relative
  /// residual, and an ABSOLUTE drop test discards genuinely new directions).
  bool relativeGsDrop = true;
  MgOptions mg;         ///< inner V-cycle configuration
};

struct GaugeEigenResult {
  double eigenvalue = 0.0;
  std::vector<std::complex<double>> vector;  ///< unit-norm smallest eigenvector
  int iterations = 0;
  double residual = 0.0;   ///< relative eigen-residual of \ref vector at return
  bool converged = false;  ///< true iff \ref residual < the requested tol
};

/// \brief Smallest eigenpair of the connection Laplacian by **Rayleigh-quotient
/// minimization preconditioned by the gauge-aware multigrid V-cycle** -- the
/// project's "own eigensolver", standalone (no SLEPc) and warm-startable.
///
/// This is the eigen-analogue of the linear V-cycle (decision: route (b),
/// "subdivision + smoothing"): each step takes the eigen-residual
/// \f$r = Ex - \rho x\f$ (\f$\rho\f$ the Rayleigh quotient), preconditions it
/// with one gauge-MG V-cycle \f$w \approx E^{-1}r\f$, and replaces \f$x\f$ by the
/// **Rayleigh-quotient-optimal** combination of \f$x\f$ and \f$w\f$ (a 2x2
/// Hermitian eigenproblem -- the locally optimal step that cannot increase
/// \f$\rho\f$). The gauge-MG preconditioner gives the multigrid acceleration the
/// degenerate/clustered low spectrum otherwise denies single-
/// vector iteration; the mass gap keeps \f$E\f$ SPD so the preconditioner solve
/// is well-posed.
///
/// \param lat          The connection lattice (e.g. from \ref gaugeLatticeFromFaces).
/// \param initialGuess Optional warm start -- the \ref subdivisionSection (a
///                     solve-free, connection-adapted seed; with a single-site
///                     coarsest level it carries the ground state's zero-set
///                     topology, but its warm-start effect is measured neutral
///                     either way) or the previous frame's
///                     eigenvector (temporal coherence). Length must be the node
///                     count; otherwise a constant start is used.
/// \param opts         covMG-LOBPCG controls (tolerance, iteration caps, V-cycle count).
GaugeEigenResult smallestEigenpairGaugeMG(
    const GaugeLattice& lat, const std::vector<std::complex<double>>* initialGuess = nullptr,
    const GaugeEigenOptions& opts = {});

/// \brief Lowest \p m eigenpairs by BLOCK Rayleigh-quotient iteration
/// preconditioned by the gauge-covariant V-cycle (one cycle per residual
/// vector per step) -- the block generalization of
/// \ref smallestEigenpairGaugeMG. Blocks resolve degenerate/clustered levels
/// (Landau levels) that single-vector iteration cannot separate. \p guess
/// optionally seeds up to m columns (warm start).
BlockEigResult lowestEigenpairsGaugeMG(
    const GaugeLattice& lat, int m,
    const std::vector<std::vector<std::complex<double>>>* guess = nullptr,
    const GaugeEigenOptions& opts = {});

}  // namespace bochner
