#include "fluid/MacCharacteristicMap.h"

#include <algorithm>

#include "fluid/MacAdvection.h"
#ifdef BOCHNER_WITH_METAL
#include "gpu/MetalContext.h"
#endif

namespace bochner {

namespace {

std::vector<Vec3> identityMap(const MacGrid& g) {
  std::vector<Vec3> m(g.numCells());
  for (int i = 0; i < g.nx(); ++i)
    for (int j = 0; j < g.ny(); ++j)
      for (int k = 0; k < g.nz(); ++k) m[g.cellIndex(i, j, k)] = g.cellCenter(i, j, k);
  return m;
}

}  // namespace

CfMcmSolver::CfMcmSolver(const MacGrid& g, const FaceField& u0, int reinitEvery,
                         PoissonMgOptions opts, bool useLimiter, double remapTol, bool secondOrder,
                         BoundarySpec bc, bool useGpuAdvect)
    : g_(g),
      proj_(g, opts, bc),
      u_(u0),
      u0_(u0),
      psi_(identityMap(g)),
      phi_(identityMap(g)),
      reinitEvery_(reinitEvery < 1 ? 1 : reinitEvery),
      useLimiter_(useLimiter),
      secondOrder_(secondOrder),
      remapTol_(remapTol) {
  u_ = proj_.project(u0);  // start from a divergence-free state
  u0_ = u_;
#ifdef BOCHNER_WITH_METAL
  // Opt-in GPU advection: upload the resident CF+MCM state (maps <- identity to
  // match psi_/phi_, snapshot u0 <- the projected initial velocity). The GPU path
  // reads the snapshot reference (roundtripRefCurrent_ off, the default).
  if (useGpuAdvect && gpu::metalAvailable()) gpuHandle_ = gpu::cfMcmUpload(g_, u_);
#else
  (void)useGpuAdvect;
#endif
}

CfMcmSolver::~CfMcmSolver() {
#ifdef BOCHNER_WITH_METAL
  if (gpuHandle_ >= 0) gpu::cfMcmFree(gpuHandle_);
#endif
}

double CfMcmSolver::rawDistortion() const {
  // The reference distortion metric, reproduced from MapperBase::estimateDistortion
  // + estimate_kernel (Mapping.cpp / CPU_kernel.cpp): for every interior point x,
  // the WORSE of the two round-trip departures from the identity,
  //   max( ||x - Phi(Psi(x))||, ||x - Psi(Phi(x))|| ),
  // maximized over x, returned in PHYSICAL length. Both round trips equal x for a
  // perfectly invertible map, so this is zero just after a reinit. The reference
  // skips a 2-cell border (i>1 && i<ni-2); we skip the same, which also keeps the
  // composed samples in the deep interior where its clamp and our extrapolation
  // agree. On a grid too small for the full border (dim < 5) the
  // border shrinks so at least one cell is always measured -- otherwise the
  // remapTol gate would silently never fire and reinit would fall back to the
  // frame cap alone.
  const int bx = std::min(2, (g_.nx() - 1) / 2);
  const int by = std::min(2, (g_.ny() - 1) / 2);
  const int bz = std::min(2, (g_.nz() - 1) / 2);
  double maxD2 = 0.0;
#pragma omp parallel for schedule(static) reduction(max : maxD2)
  for (int i = bx; i < g_.nx() - bx; ++i)
    for (int j = by; j < g_.ny() - by; ++j)
      for (int k = bz; k < g_.nz() - bz; ++k) {
        const int c = g_.cellIndex(i, j, k);
        const Vec3 x = g_.cellCenter(i, j, k);
        const Vec3 fwdBack = sampleCellVec3(g_, phi_, psi_[c], /*extrap=*/true);  // Phi(Psi(x))
        const Vec3 backFwd = sampleCellVec3(g_, psi_, phi_[c], /*extrap=*/true);  // Psi(Phi(x))
        const Vec3 dbf = vsub(x, fwdBack);
        const Vec3 dfb = vsub(x, backFwd);
        maxD2 = std::max(maxD2, std::max(vdot(dbf, dbf), vdot(dfb, dfb)));
      }
  return std::sqrt(maxD2);
}

double CfMcmSolver::maxVelComponent() const {
  // Reference CovectorSolver::getCFL: max over all faces of the absolute velocity
  // COMPONENT (an L-infinity of the staggered components, not the vector norm),
  // floored at 1e-4 so a near-still field can't divide by zero.
  double m = 1e-4;
  for (double v : u_.x) m = std::max(m, std::abs(v));
  for (double v : u_.y) m = std::max(m, std::abs(v));
  for (double v : u_.z) m = std::max(m, std::abs(v));
  return m;
}

double CfMcmSolver::mapError() const { return rawDistortion() / g_.spacing(); }

void CfMcmSolver::resetMaps() {
#ifdef BOCHNER_WITH_METAL
  if (gpuHandle_ >= 0) {
    gpu::cfMcmReset(gpuHandle_, u_);  // resident maps <- identity, snapshot u0 <- u_
    u0_ = u_;
    sinceReinit_ = 0;
    return;
  }
#endif
  psi_ = identityMap(g_);
  phi_ = identityMap(g_);
  u0_ = u_;
  sinceReinit_ = 0;
}

FaceField CfMcmSolver::advectAndCorrect(const FaceField& v, double dt, std::vector<Vec3>& psi,
                                        std::vector<Vec3>& phi, CfMcmDiagnostics* diag,
                                        bool commit) {
  const int nc = g_.numCells();

  // Line 4: advect the inverse flow map Psi backward (semi-Lagrangian map
  // advection -- Psi_new(x) = Psi_old(backtrace(x))).
  std::vector<Vec3> psiNew(nc);
#pragma omp parallel for schedule(static)
  for (int i = 0; i < g_.nx(); ++i)
    for (int j = 0; j < g_.ny(); ++j)
      for (int k = 0; k < g_.nz(); ++k)
        psiNew[g_.cellIndex(i, j, k)] =
            sampleCellVec3(g_, psi, backtrace(g_, v, g_.cellCenter(i, j, k), dt),
                           /*extrap=*/true);  // a MAP must be extrapolated, not clamped
  psi.swap(psiNew);

  // Line 5: u1 = dPsi^T (u0 o Psi) -- a single pullback of the snapshot through
  // the accumulated long-time map (the interpolation-saving heart of CF+MCM).
  FaceField u1 = pullbackThroughMap(g_, psi, u0_);

  // Line 6: march the forward flow map Phi (advance each stored position forward;
  // a backtrace over -dt is a forward RK4 integration).
  std::vector<Vec3> phiNew(nc);
#pragma omp parallel for schedule(static)
  for (int i = 0; i < g_.nx(); ++i)
    for (int j = 0; j < g_.ny(); ++j)
      for (int k = 0; k < g_.nz(); ++k)
        phiNew[g_.cellIndex(i, j, k)] = backtrace(g_, v, phi[g_.cellIndex(i, j, k)], -dt);
  phi.swap(phiNew);

  // Line 7: ut0 <- dPhi^T (u1 o Phi) -- transport u1 back through the forward map.
  FaceField ut0 = pullbackThroughMap(g_, phi, u1);

  // Line 8: e <- ut0 - u  (roundtrip error). The paper's "u" here is the SNAPSHOT
  // u0 (Sec 5.3: "u is a snapshot of the velocity from the last reinitialization";
  // Eq 35 transports u0). ut0 is a Lagrangian/reinit-time field, so ut0 - u0 is
  // the BFECC-consistent error that VANISHES for a perfect map -- and it is the
  // only reading that reproduces CF's non-dissipative result (measured: single
  // ring holds peak vorticity ~0.97 vs ~0.53 with the running velocity, whose
  // spurious correction grows with map age). roundtripRefCurrent_=true uses the
  // literally-typeset running velocity u_ -- a diagnostic, not the default.
  // Line 9: u <- u1 - dPsi^T(1/2 e o Psi).
  const FaceField& ref = roundtripRefCurrent_ ? u_ : u0_;
  FaceField he = u0_;  // half roundtrip error (boundary faces -> 0)
  for (size_t f = 0; f < he.x.size(); ++f) he.x[f] = 0.5 * (ut0.x[f] - ref.x[f]);
  for (size_t f = 0; f < he.y.size(); ++f) he.y[f] = 0.5 * (ut0.y[f] - ref.y[f]);
  for (size_t f = 0; f < he.z.size(); ++f) he.z[f] = 0.5 * (ut0.z[f] - ref.z[f]);
  const FaceField corr = pullbackThroughMap(g_, psi, he);  // dPsi^T(1/2 e o Psi)
  FaceField u = u1;
  for (size_t f = 0; f < u.x.size(); ++f) u.x[f] -= corr.x[f];
  for (size_t f = 0; f < u.y.size(); ++f) u.y[f] -= corr.y[f];
  for (size_t f = 0; f < u.z.size(); ++f) u.z[f] -= corr.z[f];
  if (diag) {
    diag->afterPullback = u1;
    diag->afterCorrection = u;
  }
  if (useLimiter_) u = bfeccLimit(g_, u1, u);
  if (diag) diag->afterLimiter = u;

  // Line 10: pressure projection.
  const FaceField uPreProj = u;  // reference point for the accumulated change
  u = proj_.project(u);
  if (diag) diag->afterProjection = u;

  // Reference BiMocq accumulation (CovectorSolver::accumulateVelocityCovector):
  // fold the non-advective velocity change Du = (post-projection) - (post-advect)
  // back into the snapshot u0 through the FORWARD map as a covector, so u0 stays
  // consistent with the divergence-free constraint over the whole map lifetime.
  // Only on the committed corrector step (the predictor must not mutate state).
  if (commit && accumulate_) {
    FaceField du = u;
    for (size_t f = 0; f < du.x.size(); ++f) du.x[f] -= uPreProj.x[f];
    for (size_t f = 0; f < du.y.size(); ++f) du.y[f] -= uPreProj.y[f];
    for (size_t f = 0; f < du.z.size(); ++f) du.z[f] -= uPreProj.z[f];
    const FaceField duBack = pullbackThroughMap(g_, phi, du);  // dPhi^T(Du o Phi)
    for (size_t f = 0; f < u0_.x.size(); ++f) u0_.x[f] += duBack.x[f];
    for (size_t f = 0; f < u0_.y.size(); ++f) u0_.y[f] += duBack.y[f];
    for (size_t f = 0; f < u0_.z.size(); ++f) u0_.z[f] += duBack.z[f];
  }
  return u;
}

void CfMcmSolver::step(double dt, CfMcmDiagnostics* diag) {
  // Line 3: v <- EstimateVelocity_{t+dt/2} for the midpoint method. The paper:
  // "we estimate the flow velocity at line 3 by the result of line 4-10 using
  // flow v = u". So the PREDICTOR runs lines 4-10 with the current velocity as the
  // frozen flow, over dt/2, on SCRATCH copies of the maps (it must not commit the
  // map advance). The resulting velocity is the midpoint estimate that then
  // freezes the flow for the real (committed) step. secondOrder_=false falls back
  // to the 1st-order freeze v=u (Alg. 1 style) for A/B diagnostics.
  // The reference normalizes the distortion by the per-step advection distance
  // max_v * dt, where max_v (getCFL) is measured on the velocity at the STEP
  // START; capture it before advection mutates u_.
  const double maxV = maxVelComponent();

#ifdef BOCHNER_WITH_METAL
  // The GPU kernels hard-wire the BFECC reference to the resident snapshot u0 (the
  // paper-faithful choice). The roundtripRefCurrent_ diagnostic asks for e = ut0 - u_
  // (running velocity) instead, which the device path cannot honor -- so fall through
  // to the CPU path in that mode, as MacCharacteristicMap.h documents.
  if (gpuHandle_ >= 0 && !roundtripRefCurrent_) {
    // GPU advection path: the map advection + pullbacks + BFECC correction run on
    // the device (resident maps); the pressure projection stays on the CPU (the
    // authoritative geometric MG) between the predictor and corrector, mirroring
    // the CPU step exactly. Both cfMcmAdvect calls return the PRE-projection
    // velocity. diag is not filled here (GPU mode is the viewer's, which never
    // requests it). The predictor runs on scratch map copies (no commit).
    FaceField v;
    if (secondOrder_) {
      FaceField vPre =
          gpu::cfMcmAdvect(gpuHandle_, u_, midpointFrac_ * dt, /*scratch=*/true, useLimiter_);
      v = proj_.project(vPre);
    } else {
      v = u_;
    }
    FaceField uPre = gpu::cfMcmAdvect(gpuHandle_, v, dt, /*scratch=*/false, useLimiter_);
    u_ = proj_.project(uPre);
    if (accumulate_) gpu::cfMcmAccumulate(gpuHandle_, u_);  // BiMocq fold-back into u0

    ++sinceReinit_;
    const bool distortionGated = remapTol_ > 0.0;
    lastDistortion_ = distortionGated ? gpu::cfMcmDistortion(gpuHandle_) / (maxV * dt) : 0.0;
    const bool byFrames = sinceReinit_ >= reinitEvery_;
    const bool byAccuracy = distortionGated && lastDistortion_ > remapTol_;
    if (byAccuracy && !byFrames) ++accuracyResets_;
    if (byFrames || byAccuracy) resetMaps();
    return;
  }
#endif

  FaceField v;
  if (secondOrder_) {
    std::vector<Vec3> psiScratch = psi_;
    std::vector<Vec3> phiScratch = phi_;
    v = advectAndCorrect(u_, midpointFrac_ * dt, psiScratch, phiScratch, nullptr, /*commit=*/false);
  } else {
    v = u_;
  }

  // Lines 4-10 (committed): the real step, using the estimated flow velocity v to
  // advect the accumulated maps and reconstruct/correct/project the velocity.
  u_ = advectAndCorrect(v, dt, psi_, phi_, diag, /*commit=*/true);

  // Lines 11-13: reinitialize the maps before they accumulate too much error --
  // EXACTLY the reference gate (CovectorSolver::advanceCovector):
  //   VelocityDistortion = estimateDistortion / (max_v * dt);
  //   if (VelocityDistortion > threshold  ||  framenum - lastReinit >= cap) reinit;
  // i.e. distortion-gated OR the frame budget, whichever trips first.
  ++sinceReinit_;
  // rawDistortion() is O(cells) with two trilinear samples per cell; only compute
  // it when reinit is actually distortion-gated (remapTol_ > 0). Otherwise the
  // gate is frame-count only and the distortion is unused, so skip the work.
  const bool distortionGated = remapTol_ > 0.0;
  lastDistortion_ = distortionGated ? rawDistortion() / (maxV * dt) : 0.0;
  const bool byFrames = sinceReinit_ >= reinitEvery_;
  const bool byAccuracy = distortionGated && lastDistortion_ > remapTol_;
  if (byAccuracy && !byFrames) ++accuracyResets_;
  if (byFrames || byAccuracy) resetMaps();
}

}  // namespace bochner
