#pragma once

#include <complex>
#include <vector>

#include "grid/GridOperators.h"
#include "grid/MacGrid.h"

namespace bochner {

/// \brief A U(1) connection on a regular 3D lattice -- the substrate-independent
/// core of the gauge-aware multigrid.
///
/// This deliberately knows nothing about the MAC grid, PETSc, or the fluid
/// solver: it is just a lattice of complex nodes with a forward parallel
/// transport \f$e^{i\theta}\f$ on each axis edge and a connection-Laplacian edge
/// weight \f$w\f$ (= \f$1/h^2\f$ at this level). Everything the multigrid needs
/// -- the operator, smoother, and intergrid transfers -- is expressed from these
/// fields alone, so the solver can be lifted into a stand-alone library.
///
/// `lkx[i,j,k]` is the angle transporting node `(i,j,k)` to `(i+1,j,k)` (low to
/// high), with `i` in `0..lx-2`; `lky`/`lkz` analogously. Node linear index is
/// `(i*ly+j)*lz+k` (\ref index), matching MacGrid::cellIndex.
struct GaugeLattice {
  int lx = 0, ly = 0, lz = 0;
  bool periodic = false;              ///< true = 3-torus (links wrap); false = open (Neumann)
  double w = 1.0;                     ///< uniform edge weight (1/h^2) -- used iff wx/wy/wz are empty
  std::vector<double> lkx, lky, lkz;  ///< forward link angle (low->high) per axis
  std::vector<std::complex<double>> tx, ty, tz;  ///< precomputed e^{i*lk} (built by buildTransports)

  /// \name Optional per-edge weights (variable edge lengths / graded resolution)
  /// Same indexing and length as lkx/lky/lkz. Empty (the default) means the
  /// uniform weight \ref w on every edge -- the original solver, bit-for-bit.
  /// Non-empty replaces \f$w\f$ per edge: the operator diagonal becomes the sum
  /// of incident edge weights, the prolongation averages with the connecting
  /// edges' weights (the A-harmonic fill), and coarsening combines the two fine
  /// weights of a coarse edge by series conductance,
  /// \f$W = w_a w_b / (2(w_a + w_b))\f$ (uniform \f$w \to w/4\f$, so the
  /// weighted rules contain the uniform ones). All three arrays must be set
  /// together (\ref setEdgeWeights validates this).
  /// \{
  std::vector<double> wx, wy, wz;
  bool weighted() const { return !wx.empty(); }
  /// Install per-edge weights (sizes must match numLinksX/Y/Z; all entries > 0).
  /// \throws std::invalid_argument on a size mismatch or a non-positive weight.
  void setEdgeWeights(std::vector<double> wx_, std::vector<double> wy_, std::vector<double> wz_);
  /// \}

  int index(int i, int j, int k) const { return (i * ly + j) * lz + k; }
  long numNodes() const { return static_cast<long>(lx) * ly * lz; }

  /// Number of forward links along each axis: `lx` per (j,k) row when periodic
  /// (the i=lx-1 link wraps lx-1 -> 0), else `lx-1` (open, no wrap). The link
  /// index formula `(i*ly+j)*lz+k` is shared by both -- only the range of `i`
  /// (and hence the array length) differs.
  long numLinksX() const { return static_cast<long>(periodic ? lx : (lx > 0 ? lx - 1 : 0)) * ly * lz; }
  long numLinksY() const { return static_cast<long>(lx) * (periodic ? ly : (ly > 0 ? ly - 1 : 0)) * lz; }
  long numLinksZ() const { return static_cast<long>(lx) * ly * (periodic ? lz : (lz > 0 ? lz - 1 : 0)); }

  /// Populate tx/ty/tz = e^{i*lk} from the link angles -- call after the lk
  /// arrays are set, so the matvec/smoother multiply by a stored transport
  /// instead of evaluating a transcendental per edge per sweep.
  void buildTransports();
};

/// Build the finest lattice from a MAC face connection \p theta (interior faces)
/// with weight \f$w = 1/h^2\f$ -- the bridge from the fluid side to the solver.
GaugeLattice gaugeLatticeFromFaces(const MacGrid& g, const FaceField& theta);

/// \brief Build a **periodic** (3-torus) lattice directly from raw forward-link
/// angle arrays -- the bridge for the lattice-gauge benchmark, whose operator is
/// periodic and whose connection is a bare gauge field (not a MAC velocity).
///
/// Each array is `lx*ly*lz` long (one forward link per node per axis; the
/// `i=lx-1` x-link wraps `lx-1 -> 0`, etc.), indexed by the low node
/// `(i*ly+j)*lz+k`. `lkx[i,j,k]` is the angle transporting node `(i,j,k)` to
/// `((i+1) mod lx, j, k)`. Every dimension must be even (the wrap link would put
/// two same-parity nodes on one red-black color otherwise). \throws
/// std::invalid_argument on an odd dimension or a link-array size mismatch.
GaugeLattice gaugeLatticePeriodic(int lx, int ly, int lz, double w,
                                  const std::vector<double>& lkx, const std::vector<double>& lky,
                                  const std::vector<double>& lkz);

/// \brief The gauge-aware subdivision section: seed \f$\psi = 1\f$ on the
/// coarsest of \p numLevels decimations of \p finest and prolong up by covariant
/// transport averaging. This is the SAME prolongation the linear V-cycle uses,
/// exposed so \ref subdivisionSection (the covariant-subdivision warm
/// start) shares one lattice + transfer implementation with the multigrid.
std::vector<std::complex<double>> subdivisionSectionFromLattice(const GaugeLattice& finest,
                                                                int numLevels);

/// \name Intergrid transfers (the V-cycle's prolongation P and its exact adjoint R)
/// Exposed so the adjoint identity \f$\langle f, Pc\rangle = \langle Rf, c\rangle\f$
/// is directly testable in double precision: the alpha-scaled coarse correction
/// guarantees descent for ANY transfer, so a scaling or conjugation error
/// confined to the transfers would pass every solve-to-tolerance test. The
/// coarse vector lives on the decimated lattice ((lx/2)*(ly/2)*(lz/2) nodes);
/// every dimension of \p fine must be even.
/// \{
std::vector<std::complex<double>> prolongGauge(const GaugeLattice& fine,
                                               const std::vector<std::complex<double>>& coarse);
std::vector<std::complex<double>> restrictGauge(const GaugeLattice& fine,
                                                const std::vector<std::complex<double>>& fineVec);
/// \}

/// \brief Matrix-free connection ("magnetic") Laplacian: \f$y = E x\f$.
///
/// \f$(Ex)_c = w\sum_{n}\big(x_c - P^\nabla_{n\to c}\,x_n\big)\f$ over the
/// existing axis neighbours \f$n\f$ of cell \f$c\f$ (homogeneous-Neumann
/// boundary = missing neighbours simply absent). This is exactly the operator
/// \ref connectionLaplacian assembles, evaluated without forming the matrix.
std::vector<std::complex<double>> applyConnectionLaplacian(
    const GaugeLattice& lat, const std::vector<std::complex<double>>& x);

/// Tuning + reporting for the multigrid linear solve.
struct MgOptions {
  int nu1 = 2;           ///< pre-smoothing sweeps per level
  int nu2 = 2;           ///< post-smoothing sweeps per level
  int coarseSweeps = 30;  ///< smoothing sweeps at the coarsest level
  int maxCycles = 100;   ///< V-cycle iteration cap
  double omega = 1.0;    ///< red-black GS relaxation (1 = Gauss-Seidel; <2 = SOR)
  double tol = 1e-8;     ///< target relative residual ||b - Ex|| / ||b||

  /// \name Ablation switches (benchmarking only -- leave true in production)
  /// \{
  /// Apply the A-energy-optimal step alpha = <p,r>/<p,Ap> to each coarse
  /// correction. false = raw injection (alpha = 1), the historical PTMG
  /// behaviour -- the non-Galerkin cycle then has no descent guarantee and can
  /// diverge on frustrated configurations.
  bool alphaStep = true;
  /// Parallel-transport the transferred values through the links. false =
  /// plain (non-covariant) averaging with the same stencil -- isolates the
  /// covariant transfer as the load-bearing ingredient.
  bool covariantTransfer = true;
  /// \}
};

struct MgResult {
  int cycles = 0;
  /// True relative residual after the last cycle; -1 when not computed
  /// (tol <= 0 = fixed-cycle preconditioner mode skips the residual matvec).
  double relResidual = 0.0;
};

/// Result of an iterative linear solve (for the CG baseline / comparisons).
struct SolveStats {
  int iterations = 0;
  double relResidual = 0.0;
};

/// \brief Baseline: solve \f$E x = b\f$ by **unpreconditioned conjugate
/// gradients** on the same matrix-free Hermitian-SPD operator -- the reference
/// the gauge multigrid is measured against.
/// \param lat      The connection lattice (the matrix-free Hermitian-SPD operator).
/// \param b        Right-hand side.
/// \param x in/out: initial guess in, solution out.
/// \param tol      Relative-residual stopping tolerance.
/// \param maxIters Iteration cap.
SolveStats cgSolve(const GaugeLattice& lat, const std::vector<std::complex<double>>& b,
                   std::vector<std::complex<double>>& x, double tol = 1e-8, int maxIters = 5000);

/// \name Matvec instrumentation (benchmarking) -- counts applyConnectionLaplacian calls.
/// \{
long gaugeMatvecCount();
/// Node-touches accumulated over all applies + smoother sweeps; divide by the
/// finest level's node count for the FINE-GRID-EQUIVALENT apply count (the fair
/// multilevel work metric -- a raw sweep count over-weights cheap coarse levels).
long long gaugeMatvecNodeWork();
void resetGaugeMatvecCount();
/// \}

/// \brief Solve \f$E x = b\f$ for the connection Laplacian by gauge-aware
/// multigrid V-cycles (matrix-free).
///
/// The hierarchy coarsens by decimation (a coarse node is every other fine node;
/// a coarse link is the sum of the two fine links it spans -- the restricted
/// U(1) connection); the prolongation is the parallel-transport averaging of the
/// covariant subdivision, the restriction is its exact
/// adjoint, and each coarse operator is the connection Laplacian re-evaluated on
/// that level's links. Smoother is red-black Gauss-Seidel (parallel within a
/// colour); each coarse correction is scaled by the A-energy-optimal step.
///
/// \warning **The V-cycle is neither symmetric nor linear -- by design.** The
/// *operator* \f$E\f$ is Hermitian-SPD (as the surrounding docs say), but do not
/// transfer that to the preconditioner this cycle defines. Two independent
/// reasons: (1) the smoother always sweeps red-then-black, whose A-adjoint is
/// black-then-red, giving a measured relative asymmetry
/// \f$|\langle u,Mv\rangle - \langle Mu,v\rangle| / |\langle u,Mv\rangle|
/// \approx 2\cdot 10^{-1}\f$ at production settings; and (2), the binding one,
/// the A-energy-optimal step \ref MgOptions::alphaStep makes \f$M\f$ *nonlinear*
/// (homogeneous of degree 1 but not additive; measured additivity defect
/// \f$2.3\cdot 10^{-2}\f$ at scale 3.2, versus \f$4\cdot 10^{-15}\f$ with the
/// step off). So this must **not** be used as a CG preconditioner. It is used as
/// a LOBPCG preconditioner (GaugeEigen, SunGauge), which only needs search
/// directions, not an SPD linear \f$T\f$ -- note that this is weaker than the
/// hypothesis of Knyazev's convergence theory.
///
/// Reversing the colour order would fix (1) (measured: asymmetry to
/// \f$10^{-16}\f$) but buys nothing -- V-cycle counts are unchanged to slightly
/// worse and LOBPCG iteration counts are bit-identical, because the dual cell
/// graph is bipartite, so red-then-black and black-then-red are the same
/// two-block Gauss-Seidel and share a smoothing factor. Fixing (2) requires
/// `alphaStep = false`, which `tools/ablation_bench` shows **diverges** on the
/// flux operator. The non-symmetry is intrinsic to why the method converges.
///
/// \param lat  The connection lattice (the finest level).
/// \param b    Right-hand side.
/// \param x in/out: initial guess on entry (zero or a warm start), solution out.
/// \param opts V-cycle controls (max cycles, tolerance, smoothing).
/// \returns the V-cycle count and final relative residual.
MgResult vcycleSolve(const GaugeLattice& lat, const std::vector<std::complex<double>>& b,
                     std::vector<std::complex<double>>& x, const MgOptions& opts = {});

/// \brief Build the V-cycle level pyramid for \p lat once (coarsen while every
/// dimension stays coarsenable: open needs each dim even and >= 4; periodic needs
/// each divisible by 4 so the coarsened dim stays even). `levels.front()` is the
/// finest (== \p lat). Reuse the result across many \ref vcycleSolve calls -- the
/// eigensolver preconditions with one V-cycle per covMG-LOBPCG step -- to avoid
/// rebuilding the hierarchy (and recomputing every coarse level's transports) on
/// each solve.
std::vector<GaugeLattice> buildGaugeLevels(const GaugeLattice& lat);

/// \brief V-cycle solve reusing a prebuilt \p levels pyramid (see \ref
/// buildGaugeLevels) -- identical to the single-lattice \ref vcycleSolve but
/// skips reconstructing the hierarchy per call.
MgResult vcycleSolve(const std::vector<GaugeLattice>& levels,
                     const std::vector<std::complex<double>>& b,
                     std::vector<std::complex<double>>& x, const MgOptions& opts = {});

/// \name Interleaved <-> complex conversion (bridge to traceZeroSet / eigensolver)
/// \{
std::vector<std::complex<double>> toComplex(const std::vector<double>& interleaved);
std::vector<double> toInterleaved(const std::vector<std::complex<double>>& z);
/// \}

}  // namespace bochner
