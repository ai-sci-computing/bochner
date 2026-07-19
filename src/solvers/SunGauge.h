#pragma once

#include <complex>
#include <cstdint>
#include <vector>

#include "solvers/BlockLobpcg.h"
#include "solvers/GaugeEigen.h"     // GaugeEigenOptions/Result (fiber-agnostic)
#include "solvers/GaugeMultigrid.h"  // MgOptions/MgResult/SolveStats

namespace bochner {

/// \brief A **non-abelian** U(d) / SU(d) connection on a regular 3D lattice --
/// the direct generalization of \ref GaugeLattice from scalar U(1) phases
/// \f$e^{i\theta}\f$ to \f$d\times d\f$ unitary link matrices.
///
/// This is the paper's transferability demonstration: the closed-form
/// gauge-aware geometric multigrid needs nothing but a *transport per edge*, so
/// replacing the scalar phase with a matrix \f$U\in\mathrm U(d)\f$ carries the
/// whole method across with two substitutions --- scalar `transport * value`
/// becomes a matrix-vector product, and `conj(transport)` becomes the Hermitian
/// adjoint \f$U^\dagger\f$. The diagonal stays \f$(w\,\deg + m^2)\,I_d\f$, so
/// even the red-black Gauss-Seidel smoother is structurally identical. The
/// covariant subdivision, adjoint restriction, and optimal-A-energy correction
/// are unchanged; only the fiber algebra (\f$\mathbb C \to \mathbb C^d\f$) grows.
///
/// A section is a `d`-vector per node, stored flat: node `c` occupies
/// `[c*d, c*d + d)`. Link matrices are row-major `d*d` complex, one forward link
/// (low->high) per node per axis, indexed exactly as \ref GaugeLattice's link
/// arrays. Setting `d == 1` recovers the scalar U(1) operator (used as a
/// cross-check against the trusted \ref applyConnectionLaplacian).
struct SunLattice {
  int d = 2;                          ///< fiber dimension (U(1)->1, SU(2)->2, SU(3)->3)
  int lx = 0, ly = 0, lz = 0;
  bool periodic = false;              ///< true = 3-torus (links wrap); false = open (Neumann)
  double w = 1.0;                     ///< connection-Laplacian edge weight (1/h^2)
  double mass2 = 0.0;                 ///< diagonal mass m^2 (SPD shift); 0 = pure covariant Laplacian
  std::vector<std::complex<double>> ux, uy, uz;  ///< forward link matrices (d*d row-major), low->high

  int index(int i, int j, int k) const { return (i * ly + j) * lz + k; }
  long numNodes() const { return static_cast<long>(lx) * ly * lz; }
  long dof() const { return numNodes() * d; }

  long numLinksX() const { return static_cast<long>(periodic ? lx : (lx > 0 ? lx - 1 : 0)) * ly * lz; }
  long numLinksY() const { return static_cast<long>(lx) * (periodic ? ly : (ly > 0 ? ly - 1 : 0)) * lz; }
  long numLinksZ() const { return static_cast<long>(lx) * ly * (periodic ? lz : (lz > 0 ? lz - 1 : 0)); }

  /// \name Optional per-edge weights (variable edge lengths / graded resolution)
  /// Scalar weight per edge (the fiber stays d-dimensional; the weight does
  /// not). Same indexing/length as one d*d block per entry of ux/uy/uz, i.e.
  /// numLinksX/Y/Z entries. Empty = uniform \ref w. Semantics identical to
  /// GaugeLattice: weighted covariant averaging in the transfers, diagonal =
  /// (sum of incident weights + m^2) I, series-conductance coarsening
  /// W = wa*wb/(2(wa+wb)) (uniform -> w/4).
  /// \{
  std::vector<double> wx, wy, wz;
  bool weighted() const { return !wx.empty(); }
  /// \throws std::invalid_argument on size mismatch or non-positive weight.
  void setEdgeWeights(std::vector<double> wx_, std::vector<double> wy_, std::vector<double> wz_);
  /// \}
};

/// \name Config generators
/// \{
/// Random SU(d) (for d>=2; Haar-random U(1) phases for d==1) link field on a
/// periodic torus, deterministic in \p seed. `mass2` shifts the diagonal so the
/// operator is SPD with a spectral gap even for a maximally-disordered ("hot")
/// gauge field -- the hardest case for constant-near-kernel methods. Every
/// dimension must be even. \throws std::invalid_argument otherwise.
SunLattice randomSunLattice(int d, int lx, int ly, int lz, double w, double mass2, std::uint64_t seed,
                            bool periodic = true);

/// Trivial connection: all link matrices are the identity. The operator then
/// decouples into `d` independent copies of the ordinary scalar Laplacian --
/// a correctness anchor.
SunLattice identitySunLattice(int d, int lx, int ly, int lz, double w, double mass2,
                              bool periodic = true);
/// \}

/// \brief Matrix-free non-abelian connection Laplacian \f$y = (E + m^2)x\f$,
/// \f$(Ex)_c = w\sum_n\big(x_c - U_{n\to c}\,x_n\big) + m^2 x_c\f$ over the
/// existing axis neighbours (homogeneous-Neumann boundary = missing neighbours
/// absent). Hermitian positive (semi-)definite; \p x has length \ref dof.
std::vector<std::complex<double>> applySunLaplacian(const SunLattice& lat,
                                                    const std::vector<std::complex<double>>& x);

/// Build the V-cycle level pyramid (coarsen by decimation while every dimension
/// stays coarsenable), reusable across many solves. A coarse link is the
/// **ordered product** of the two fine links it spans.
std::vector<SunLattice> buildSunLevels(const SunLattice& lat);

/// Gauge-aware multigrid V-cycle solve of \f$(E+m^2)x = b\f$ (prebuilt levels).
MgResult vcycleSolveSun(const std::vector<SunLattice>& levels,
                        const std::vector<std::complex<double>>& b,
                        std::vector<std::complex<double>>& x, const MgOptions& opts = {});
/// Single-solve convenience (builds the pyramid, then solves).
MgResult vcycleSolveSun(const SunLattice& lat, const std::vector<std::complex<double>>& b,
                        std::vector<std::complex<double>>& x, const MgOptions& opts = {});

/// Baseline: unpreconditioned conjugate gradients on the same Hermitian-SPD
/// operator -- the reference the gauge multigrid's mesh-independence is measured
/// against.
SolveStats cgSolveSun(const SunLattice& lat, const std::vector<std::complex<double>>& b,
                      std::vector<std::complex<double>>& x, double tol = 1e-8, int maxIters = 20000);

/// The gauge-aware subdivision seed: constant frame on the coarsest of
/// \p numLevels decimations, prolonged up by covariant transport averaging (the
/// non-abelian covariant-subdivision warm start).
std::vector<std::complex<double>> subdivisionSectionSun(const SunLattice& finest, int numLevels);

/// \name Intergrid transfers (prolongation P and its exact adjoint R = P^H)
/// The SU(d) analogue of \ref prolongGauge / \ref restrictGauge, exposed for
/// the same reason: the adjoint identity <f, P c> = <R f, c> is the only direct
/// test of the transfer pair (the alpha step makes the V-cycle converge for any
/// transfer). Note R is the Hermitian ADJOINT, not the transpose -- the
/// restriction transports with U where the prolongation used U^H and vice
/// versa. Coarse sections have (lx/2)*(ly/2)*(lz/2)*d entries; every dimension
/// of \p fine must be even.
/// \{
std::vector<std::complex<double>> prolongSun(const SunLattice& fine,
                                             const std::vector<std::complex<double>>& coarse);
std::vector<std::complex<double>> restrictSun(const SunLattice& fine,
                                              const std::vector<std::complex<double>>& fineVec);
/// \}

/// \brief Smallest eigenpair of the non-abelian connection Laplacian by
/// Rayleigh-quotient minimization preconditioned by the gauge-MG V-cycle -- the
/// exact SU(d) analogue of \ref smallestEigenpairGaugeMG (the covMG-LOBPCG outer loop is
/// fiber-agnostic; only the operator and preconditioner change).
GaugeEigenResult smallestEigenpairSunMG(const SunLattice& lat,
                                        const std::vector<std::complex<double>>* initialGuess = nullptr,
                                        const GaugeEigenOptions& opts = {});

/// \brief Lowest \p m eigenpairs of the non-abelian covariant Laplacian by
/// block Rayleigh-quotient iteration preconditioned by the gauge-covariant
/// V-cycle -- the SU(d) block solver. With d = 3 this is the distillation /
/// Laplacian-Heaviside lowest-band task.
BlockEigResult lowestEigenpairsSunMG(
    const SunLattice& lat, int m,
    const std::vector<std::vector<std::complex<double>>>* guess = nullptr,
    const GaugeEigenOptions& opts = {});

}  // namespace bochner
