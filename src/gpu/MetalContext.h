#pragma once

#include <complex>
#include <vector>

#include "grid/GridOperators.h"  // FaceField
#include "grid/MacGrid.h"
#include "grid/SparseMg.h"  // MgHierarchy
#include "grid/Vec3.h"

/// \file
/// Metal GPU compute backend (opt-in, `BOCHNER_WITH_METAL`). This header is pure
/// C++: all Objective-C / Metal interop lives in MetalContext.mm behind it. The
/// GPU works in **single precision** (Metal has no `double`), so results match
/// the double-precision CPU path only to float tolerance -- which is the
/// accuracy the reference Covector Fluids GPU code runs at.
namespace bochner::gpu {

/// \brief True if a Metal device is present and the compute pipeline compiled.
/// When false, the GPU entry points below throw; callers fall back to the CPU.
bool metalAvailable();

/// \brief GPU trilinear sample of the staggered MAC velocity at each point --
/// the batched, single-precision equivalent of \ref bochner::sampleVelocity for
/// a whole array of query points (the hot inner op of semi-Lagrangian advection).
///
/// Uses the same clamped ("streak") boundary as the CPU path. Returns one
/// velocity per input point, in the same order.
/// \throws std::runtime_error if \ref metalAvailable is false.
std::vector<Vec3> sampleVelocityBatch(const MacGrid& g, const FaceField& u,
                                      const std::vector<Vec3>& pts);

/// \brief GPU RK4 backward flow-map -- the batched, single-precision equivalent
/// of \ref bochner::backtrace: for each start point, integrate `dx/dt = v`
/// backward over \p dt with classic RK4 (four velocity samples of \p v) and
/// return the traced-back position. Same clamped boundary as the CPU path.
/// \throws std::runtime_error if \ref metalAvailable is false.
std::vector<Vec3> backtraceBatch(const MacGrid& g, const FaceField& v,
                                 const std::vector<Vec3>& starts, double dt);

/// \brief GPU semi-Lagrangian covector advection -- the single-precision
/// equivalent of \ref bochner::advectCovectorSL (Covector Fluids Alg. 3). Runs
/// the whole step on the GPU: backtrace the cell-center flow map through \p v,
/// then set each interior face to the transpose-Jacobian column (finite
/// difference of the map) dotted with \p u sampled at the face's backtrace.
/// Boundary faces keep their prescribed \p u value.
/// \throws std::runtime_error if \ref metalAvailable is false.
FaceField advectCovectorSLGPU(const MacGrid& g, const FaceField& u, const FaceField& v, double dt);

/// \brief GPU BFECC covector advection, fully device-resident: the single-
/// precision equivalent of \ref bochner::advectCovectorBFECC with the three
/// semi-Lagrangian sub-steps, the error blend, and the CF 5.4.2 extrema limiter
/// all run as kernels on buffers that stay on the device -- \p u and \p v are
/// uploaded once and only the result is read back, so there are no per-sub-step
/// host<->device transfers. Matches \ref bochner::advectCovectorBFECC to float
/// tolerance. \throws std::runtime_error if \ref metalAvailable is false.
FaceField advectCovectorBFECCResident(const MacGrid& g, const FaceField& u, const FaceField& v,
                                      double dt);

// --- CF+MCM (characteristic-map, Alg. 5) advection primitives -----------------
// The single-precision GPU twins of the CF+MCM map operators. Raw entry points
// (upload/compute/download) used for parity testing; the resident per-step
// orchestration reuses the same kernels without the transfers.

/// GPU map advection (Alg. 5 line 4): \p psiNew(x) = \p psiOld(backtrace(x; v, dt)),
/// the stored map sampled with EXTRAPOLATION. Matches the CPU
/// sampleCellVec3(psiOld, backtrace(.), extrap=true) loop to float tolerance.
std::vector<Vec3> mapAdvect(const MacGrid& g, const std::vector<Vec3>& psiOld, const FaceField& v,
                            double dt);

/// GPU pullback through a stored map (== \ref bochner::pullbackThroughMap): each
/// interior face = dM . source(M(faceCenter)), dM = (M[hi]-M[lo])/h the map's
/// transpose-Jacobian column; the map sample at the face is CLAMPED (extrap
/// false, as the CPU does). Boundary faces keep their \p source value.
FaceField pullbackStored(const MacGrid& g, const std::vector<Vec3>& map, const FaceField& source);

/// GPU reference distortion (== \ref bochner CfMcmSolver::rawDistortion): the
/// worst map round-trip departure from the identity over the deep interior (skip
/// a 2-cell border), max(|x-Phi(Psi(x))|, |x-Psi(Phi(x))|), in physical length.
double mapDistortion(const MacGrid& g, const std::vector<Vec3>& psi, const std::vector<Vec3>& phi);

// --- Resident CF+MCM state (the per-step orchestration, maps stay on device) --
// An opaque handle (like the Poisson/gauge handles) owning the inverse/forward
// flow maps Psi/Phi (+ predictor scratch copies) and the velocity snapshot u0,
// all resident across steps. Only the velocity crosses to the host for the CPU
// pressure projection; the O(cells) maps never move.

/// Allocate resident CF+MCM state: Psi, Phi <- identity, snapshot u0 <- \p u0.
/// Returns a handle for the calls below; free with \ref cfMcmFree.
int cfMcmUpload(const MacGrid& g, const FaceField& u0);
void cfMcmFree(int handle);

/// One CF+MCM advect-and-correct (Alg. 5 lines 4-9 + the CF 5.4.2 limiter) with a
/// frozen flow \p v over \p dt, advancing the map set in place and reconstructing
/// the velocity from the snapshot. \p scratch=true runs it on a fresh copy of the
/// real maps (the midpoint PREDICTOR -- the real Psi/Phi are untouched); false on
/// the real maps (the committed CORRECTOR). Returns the PRE-projection velocity
/// (the caller runs the pressure projection on the host) and keeps it resident
/// for \ref cfMcmAccumulate.
FaceField cfMcmAdvect(int handle, const FaceField& v, double dt, bool scratch, bool useLimiter);

/// BiMocq accumulate (corrector/commit only): fold the post-projection change
/// back into the snapshot through the real forward map,
/// u0 += dPhi^T((\p uPost - uPre) o Phi), where uPre is the field \ref cfMcmAdvect
/// last returned. Keeps u0 consistent with the divergence-free constraint.
void cfMcmAccumulate(int handle, const FaceField& uPost);

/// Reinit (Alg. 5 lines 11-13): real maps <- identity, snapshot u0 <- \p uPost.
void cfMcmReset(int handle, const FaceField& uPost);

/// Reference distortion on the real maps (== \ref mapDistortion / rawDistortion).
double cfMcmDistortion(int handle);

// --- GPU multigrid-preconditioned CG for the pressure Poisson ----------------
// The V-cycle (SpMV + Jacobi smoother + restrict/prolong) is bandwidth-bound, so
// it runs far faster on the GPU. The hierarchy is built on the CPU
// (buildPoissonMgHierarchy) and uploaded once; each frame solves in single
// precision (Metal has no double), reaching a float-appropriate residual.

/// Upload a multigrid hierarchy to the GPU and return a handle for repeated
/// solves. Returns -1 if Metal is unavailable.
int uploadPoisson(const MgHierarchy& H);

/// Release an uploaded hierarchy (safe on -1 / already-freed handles).
void freePoisson(int handle);

/// Solve `A x = b` (A = the finest level of the uploaded hierarchy \p handle) by
/// GPU MGPCG, warm-started from \p x. Single precision. Returns the iteration
/// count; if \p relResidualOut is non-null, reports `||b - A x|| / ||b||`.
/// \throws std::runtime_error if \p handle is invalid or Metal is unavailable.
int mgpcgSolvePoisson(int handle, const std::vector<double>& b, std::vector<double>& x, double tol,
                      int maxit, double* relResidualOut = nullptr);

// --- GPU matrix-free geometric multigrid for the closed-box Neumann Poisson ---
// The single-precision device twin of \ref bochner::poissonVcycleSolve: the
// UNPINNED all-Neumann Laplacian solved by optimal-alpha V-cycles, resident on
// the GPU. Unlike the assembled MGPCG above (which pins one cell and so converges
// slowly on the all-Neumann box), this keeps the fast matrix-free structure -- the
// route that actually beats the CPU geometric MG for the closed box.

/// Allocate the resident geometric-MG hierarchy for an \p nx x \p ny x \p nz grid
/// of spacing \p h (levels coarsened by 2 while all dims are even and >= 4).
/// Returns a handle; free with \ref freePoissonGeom. -1 if Metal is unavailable.
int uploadPoissonGeom(int nx, int ny, int nz, double h);
void freePoissonGeom(int handle);

/// Solve the all-Neumann Poisson `L phi = rhs` (rhs must be mean-zero) by
/// optimal-alpha geometric V-cycles on the uploaded hierarchy \p handle, warm-
/// started from \p phi (in/out). Single precision. Returns the cycle count; if
/// \p relResidualOut is non-null, reports `||rhs - L phi|| / ||rhs||`.
/// \throws std::runtime_error if \p handle is invalid or Metal is unavailable.
int poissonGeomSolve(int handle, const std::vector<double>& rhs, std::vector<double>& phi,
                     double tol, int maxCycles, int nu1, int nu2, int coarseSweeps, double omega,
                     double* relResidualOut = nullptr);

// --- GPU gauge (connection-Laplacian) eigensolver primitives -----------------
// Foundation for a single-precision GPU covMG-LOBPCG eigensolver (the "smoke rings"
// extraction) -- a viewer-speed path; the double CPU/Lanczos solvers stay
// authoritative. Complex vectors are std::complex<double> on the C++ side,
// float2 on the GPU. Lattice data is passed as raw arrays to avoid a gpu->solvers
// dependency (the GaugeLattice type lives in solvers).

/// GPU apply of the U(1) connection ("Bochner") Laplacian, `y = E x` (single
/// precision): the matrix-free 7-point complex stencil with per-edge link
/// transports \p tx/\p ty/\p tz (= e^{i theta}, the GaugeLattice::t* arrays).
/// \p periodic selects the 3-torus wrap vs open (Neumann) boundary. Mirrors
/// \ref bochner::applyConnectionLaplacian.
/// \throws std::runtime_error if \ref metalAvailable is false.
std::vector<std::complex<double>> connectionMatvec(
    int lx, int ly, int lz, bool periodic, double w, const std::vector<std::complex<double>>& tx,
    const std::vector<std::complex<double>>& ty, const std::vector<std::complex<double>>& tz,
    const std::vector<std::complex<double>>& x);

/// GPU red-black Gauss-Seidel (SOR) smoother of the connection Laplacian: run
/// \p sweeps full red-black sweeps of `E x = b` from the initial guess \p x0,
/// relaxed by \p omega (single precision). Mirrors the CPU smoother inside
/// \ref bochner::vcycleSolve; the building block of the GPU gauge V-cycle.
/// Returns the smoothed vector. \throws std::runtime_error if \ref metalAvailable
/// is false.
std::vector<std::complex<double>> connectionSmooth(
    int lx, int ly, int lz, bool periodic, double w, const std::vector<std::complex<double>>& tx,
    const std::vector<std::complex<double>>& ty, const std::vector<std::complex<double>>& tz,
    const std::vector<std::complex<double>>& b, const std::vector<std::complex<double>>& x0,
    int sweeps, double omega);

/// GPU covariant prolongation ("smoke rings" subdivision): lift a coarse vector
/// (length `(lx/2)(ly/2)(lz/2)`) to the fine lattice by transport averaging --
/// even fine nodes copy the coarse value, then three parity-sum passes fill the
/// rest from their already-filled even neighbours. \p tx/\p ty/\p tz are the FINE
/// lattice's link transports. Mirrors the (file-private) prolong inside the CPU
/// V-cycle. \throws std::runtime_error if \ref metalAvailable is false.
std::vector<std::complex<double>> connectionProlong(
    int lx, int ly, int lz, bool periodic, const std::vector<std::complex<double>>& tx,
    const std::vector<std::complex<double>>& ty, const std::vector<std::complex<double>>& tz,
    const std::vector<std::complex<double>>& coarse);

/// GPU restriction = the EXACT ADJOINT of \ref connectionProlong: map a fine
/// residual (length `lx*ly*lz`) down to the coarse lattice by running the three
/// prolong passes in reverse, each scattering a node's value to its sources with
/// conjugate transport (atomic float2 add), then picking the even nodes. \p tx/
/// \p ty/\p tz are the FINE lattice's link transports.
/// \throws std::runtime_error if \ref metalAvailable is false.
std::vector<std::complex<double>> connectionRestrict(
    int lx, int ly, int lz, bool periodic, const std::vector<std::complex<double>>& tx,
    const std::vector<std::complex<double>>& ty, const std::vector<std::complex<double>>& tz,
    const std::vector<std::complex<double>>& fineR);

/// One level of a gauge multigrid hierarchy, as raw arrays (no dependency on the
/// GaugeLattice type, which lives in solvers). Built on the CPU by
/// \ref bochner::buildGaugeLevels and copied into this dependency-free form.
struct GaugeLevelData {
  int lx = 0, ly = 0, lz = 0;
  bool periodic = false;
  double w = 1.0;
  std::vector<std::complex<double>> tx, ty, tz;  ///< this level's link transports (e^{i theta})
};

/// Upload a gauge multigrid hierarchy (finest first, coarsened by decimation) to
/// the GPU and return a handle for repeated V-cycle solves. Returns -1 if Metal
/// is unavailable. Release with \ref freeGauge.
int uploadGauge(const std::vector<GaugeLevelData>& levels);

/// Release an uploaded gauge hierarchy (safe on -1 / already-freed handles).
void freeGauge(int handle);

/// Solve `E x = b` (E = the finest level of hierarchy \p handle) by GPU gauge
/// multigrid V-cycles, warm-started from \p x, in single precision. Each V-cycle
/// mirrors the CPU \ref bochner::vcycleSolve: red-black GS pre/post smoothing,
/// covariant restrict/prolong, and the energy-optimal coarse-correction step
/// `alpha = Re<pe,r> / Re<pe,E pe>`. Runs until the relative residual falls below
/// \p tol or \p maxCycles is reached; returns the cycle count and (if
/// \p relResidualOut is non-null) `||b - E x|| / ||b||`.
/// \throws std::runtime_error if \p handle is invalid or Metal is unavailable.
int vcycleSolveGauge(int handle, const std::vector<std::complex<double>>& b,
                     std::vector<std::complex<double>>& x, int nu1, int nu2, int coarseSweeps,
                     double omega, int maxCycles, double tol, double* relResidualOut = nullptr);

/// Smallest eigenpair returned by \ref lobpcgSolveGauge (the GPU analogue of
/// \ref bochner::GaugeEigenResult).
struct GaugeEigenGpu {
  double eigenvalue = 0.0;
  double residual = 0.0;   ///< relative eigen-residual of \ref vector at return
  int iterations = 0;
  bool converged = false;
  std::vector<std::complex<double>> vector;  ///< unit-norm smallest eigenvector
};

/// Smallest eigenpair of the connection Laplacian on hierarchy \p handle by GPU
/// gauge-covariant MG-preconditioned eigensolver (covMG-LOBPCG) -- the "smoke
/// rings" extraction at viewer
/// speed, fully DEVICE-RESIDENT. The guess is normalized in double and uploaded
/// once; each iteration (matvec, Rayleigh quotient, residual, \p precCycles
/// V-cycles as the preconditioner, Gram-Schmidt of {x, w, x_prev} and the <=3x3
/// Ritz matrix) is encoded into one command buffer, and only the per-iteration
/// scalars (rho, two Gram-Schmidt norms, the Ritz matrix) are read back -- one
/// host sync per iteration. Iteration stops on rho STAGNATION (relative change
/// < 1e-5 twice in a row: the float residual floors above any tight tol; rho is
/// the monotone signal). The final eigenvalue/residual are recomputed in DOUBLE
/// from the downloaded vector, so the Rayleigh quotient's quadratic accuracy
/// gives ~1e-6 eigenvalues from float vectors (see docs/gauge-multigrid.md
/// section 5a). The double CPU / SLEPc solvers stay authoritative. \p guess is
/// an optional warm start (the previous frame's eigenvector); pass empty for a
/// constant start.
/// \throws std::runtime_error if \p handle is invalid or Metal is unavailable.
GaugeEigenGpu lobpcgSolveGauge(int handle, const std::vector<std::complex<double>>& guess,
                             int maxIters, double tol, int precCycles, int nu1, int nu2,
                             int coarseSweeps, double omega);

}  // namespace bochner::gpu
