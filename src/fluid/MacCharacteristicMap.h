#pragma once

#include <vector>

#include "grid/GridOperators.h"
#include "grid/MacGrid.h"
#include "fluid/MacProjection.h"
#include "solvers/PoissonMultigrid.h"
#include "grid/Vec3.h"

namespace bochner {

/// \brief Intermediate velocity fields captured during one \ref CfMcmSolver
/// step, for the per-substep circulation audit.
///
/// All four fields live at the SAME (post-advection) time level, so evaluating
/// the circulation of each around one fixed loop isolates how much each substep
/// changes Gamma: the pullback sets the advected baseline, the BFECC correction
/// and minmax limiter may add/remove circulation, and the projection (a pure
/// gradient) must leave every closed-loop integral UNCHANGED.
struct CfMcmDiagnostics {
  FaceField afterPullback;    ///< u1 = dPsi^T(u0 o Psi) (line 5)
  FaceField afterCorrection;  ///< u after the BFECC error correction (line 9, pre-limiter)
  FaceField afterLimiter;     ///< u after the minmax limiter (== afterCorrection if limiter off)
  FaceField afterProjection;  ///< u after pressure projection (line 10, the final field)
};

/// \brief Covector Fluids with a long-time characteristic mapping
/// (**CF+MCM**, Covector Fluids Algorithm 5) -- the paper's least-dissipative
/// variant, used for its hero trefoil (Fig. 9) and leapfrogging (Fig. 10) results.
///
/// Where the base method (\ref stepCovectorFluids, Alg. 1) re-advects the
/// velocity field every step -- accumulating one interpolation per step -- CF+MCM
/// maintains long-time flow maps and advects only **those**, reconstructing the
/// velocity from a snapshot \f$u_0\f$ via a *single* pullback. The paper: "this
/// largely reduces the amount of interpolation during subsequent advection
/// steps, by only needing to advect the map \f$\Psi\f$ and not the velocity
/// components." That is exactly the interpolation diffusion that otherwise smears
/// close vortex cores until they merge.
///
/// State carried between steps: the inverse (backward) flow map \f$\Psi:
/// M_t\to M_0\f$ and the forward map \f$\Phi: M_0\to M_t\f$ (both cell-centered
/// position fields, identity at the last reinitialization), plus the velocity
/// snapshot \f$u_0\f$. Each \ref step (Alg. 5):
///   0. estimate the frozen flow velocity at the midpoint
///      \f$v\leftarrow\mathcal P(\mathcal A^{covec}(u; u, \tfrac{\Delta t}2))\f$
///      (Alg. 5 line 3 -- the 2nd-order midpoint method; the paper ties this to
///      capturing leapfrogging, Fig. 7). The maps (lines 4, 6) are then advected
///      with this \f$v\f$. Disable via \p secondOrder=false to freeze \f$v=u\f$
///      (1st order, Alg. 1 style) -- kept only for A/B diagnostics.
///   1. advect \f$\Psi\f$ backward by the frozen flow (line 4);
///   2. \f$u_1 = d\Psi^\top(u_0\circ\Psi)\f$ -- one pullback of the snapshot (line 5);
///   3. march \f$\Phi\f$ forward (line 6);
///   4. \f$\tilde u_0 = d\Phi^\top(u_1\circ\Phi)\f$, error \f$e=\tilde u_0-u_0\f$,
///      correction \f$u = u_1 - d\Psi^\top(\tfrac12 e\circ\Psi)\f$ + minmax limiter
///      (lines 7-9, the BFECC treatment);
///   5. pressure projection (line 10);
///   6. reset \f$\Psi,\Phi\to\mathrm{id}\f$ and \f$u_0\to u\f$ (lines 11-13)
///      before the map accumulates too much error -- CF Sec 5.3 resets "either
///      after a certain number of frames OR when the map is no longer accurate
///      (i.e. \f$\Phi\circ\Psi\neq\Psi\circ\Phi\f$)". Both criteria are applied:
///      the frame count \p reinitEvery and the accuracy tolerance \p remapTol.
class CfMcmSolver {
public:
  /// Build the solver around an initial velocity \p u0 (projected to
  /// divergence-free). \p reinitEvery sets the max steps the maps accumulate;
  /// \p remapTol is the CF 5.3 accuracy threshold, built as **the reference's
  /// exact distortion gate** (`CovectorSolver::advanceCovector`): the map is
  /// reset once the normalized distortion \f$\hat D=D/(v_\max\,\Delta t)\f$
  /// exceeds it, where
  /// \f$D=\max_x\max(\lVert x-\Phi(\Psi(x))\rVert,\lVert x-\Psi(\Phi(x))\rVert)\f$
  /// (the reference `estimateDistortion`) and \f$v_\max\f$ is the max absolute
  /// velocity **component** at the step start (the reference `getCFL` max_v).
  /// \f$\hat D\f$ is dimensionless -- a fraction of one step's advection
  /// distance -- so the default **0.5 matches the reference's velocity threshold**
  /// for the covector (CF+MCM) scheme. \p remapTol \f$\le 0\f$ disables the
  /// accuracy criterion, leaving pure frame-count reinit.
  /// \p secondOrder selects the Alg. 5 line-3 midpoint flow-velocity estimate
  /// (true, faithful) versus the 1st-order freeze \f$v=u\f$ (false, diagnostic).
  /// \p bc selects the pressure boundary condition: the default
  /// closed no-penetration box, or a free-surface Dirichlet-p=0 box (any open
  /// wall) that removes the wall-image confinement stalling the leapfrog rings.
  /// \p useGpuAdvect runs the map advection + pullbacks + BFECC correction on the
  /// Metal GPU (single precision, device-resident maps) -- the opt-in viewer-speed
  /// overlay; the pressure projection still runs on the CPU (the authoritative
  /// geometric MG) between the predictor and corrector. Ignored if Metal is not
  /// built/available or the diagnostic \ref setRoundtripRefCurrent is on (the GPU
  /// path assumes the paper-faithful snapshot reference).
  CfMcmSolver(const MacGrid& g, const FaceField& u0, int reinitEvery = 8,
              PoissonMgOptions opts = {}, bool useLimiter = true, double remapTol = 0.5,
              bool secondOrder = true, BoundarySpec bc = BoundarySpec::allClosed(),
              bool useGpuAdvect = false);
  ~CfMcmSolver();

  /// Advance one CF+MCM step of size \p dt; updates the internal velocity.
  /// If \p diag is non-null it is filled with the per-substep intermediate
  /// fields (for the circulation audit; otherwise no extra work / copies).
  void step(double dt, CfMcmDiagnostics* diag = nullptr);

  /// The current divergence-free velocity field.
  const FaceField& velocity() const { return u_; }

  /// Route the internal pressure projection through the Metal GPU (the closed-box
  /// geometric multigrid, \ref MacProjector::setGpuProjection) -- the opt-in
  /// viewer-speed overlay for the projection, independent of \ref useGpuAdvect.
  /// Both closed-box projections per step (predictor + corrector) use it.
  void setGpuProjection(bool on) { proj_.setGpuProjection(on); }

  int reinitEvery() const { return reinitEvery_; }
  int sinceReinit() const { return sinceReinit_; }

  /// The reference distortion \f$D=\max_x\max(\lVert x-\Phi(\Psi(x))\rVert,
  /// \lVert x-\Psi(\Phi(x))\rVert)\f$ (the `estimateDistortion` metric) reported
  /// in **cell units** (\f$D/h\f$): how far the two map round trips have drifted
  /// from the identity. Zero just after a reinit (the maps are identity). This is
  /// the raw, un-normalized map property; the reinit GATE uses the velocity-
  /// normalized \ref lastDistortion.
  double mapError() const;

  /// The normalized distortion \f$\hat D=D/(v_\max\,\Delta t)\f$ evaluated at the
  /// most recent \ref step -- the exact quantity the reference compares to its
  /// threshold (0 before any step). Dimensionless; \ref remapTol is its gate.
  double lastDistortion() const { return lastDistortion_; }

  /// The accuracy threshold on the normalized distortion \ref lastDistortion
  /// (dimensionless; reference velocity value 0.5); \f$\le 0\f$ means disabled.
  double remapTol() const { return remapTol_; }

  /// How many reinitializations so far were triggered by the accuracy criterion
  /// (rather than the frame count) -- for diagnostics/auditing.
  int accuracyResets() const { return accuracyResets_; }

  /// Select the reference field for the Alg. 5 line-8 roundtrip error
  /// \f$e\leftarrow\tilde u_0 - u\f$. Default (\p current=false) = the snapshot
  /// \f$u_0\f$: \f$\tilde u_0\f$ is a Lagrangian (reinit-time) field, so
  /// \f$\tilde u_0-u_0\f$ is the BFECC-consistent error, it VANISHES for a perfect
  /// map, and it matches Eq. 35, the Sec 5.3 prose ("u is a snapshot of the
  /// velocity from the last reinitialization"), AND the paper's non-dissipative
  /// result. MEASURED: a single ring holds peak vorticity ~0.97 with the snapshot
  /// vs ~0.53 with the current velocity (the latter's spurious correction grows
  /// with map age and shreds the vorticity). \p current=true uses the
  /// literally-typeset running velocity \f$u\f$ -- a DIAGNOSTIC ONLY; it does not
  /// reproduce the paper's non-dissipation, so it is not the faithful reading.
  void setRoundtripRefCurrent(bool current) { roundtripRefCurrent_ = current; }
  bool roundtripRefCurrent() const { return roundtripRefCurrent_; }

  /// Fraction of dt the Alg. 5 line-3 midpoint PREDICTOR advances (only used when
  /// secondOrder). 0.5 (default) = the literal midpoint method, estimating the
  /// flow velocity at t+dt/2; 1.0 = a full-step (endpoint) predictor. Diagnostic
  /// knob for the "what stepsize" A/B -- 0.5 is the paper-faithful value.
  void setMidpointFraction(double frac) { midpointFrac_ = frac; }
  double midpointFraction() const { return midpointFrac_; }

  /// Maintain the snapshot as a PERSISTENT accumulator (the reference's BiMocq
  /// long-term mapping, `CovectorSolver::accumulateVelocityCovector`): after each
  /// committed step, the non-advective velocity change \f$\Delta u\f$ from the
  /// pressure projection (and any external forces) is folded back into the
  /// snapshot \f$u_0\f$ through the FORWARD map as a covector,
  /// \f$u_0 \mathrel{+}= \mathrm{d}\Phi^{\!\top}(\Delta u\circ\Phi)\f$, so the
  /// snapshot stays consistent with the divergence-free constraint over the
  /// whole map lifetime instead of being frozen at the last reinit. Default true
  /// (faithful). \p false = the frozen-snapshot behaviour (diagnostic A/B): the
  /// projection is re-derived from scratch every step, which lets the cumulative
  /// near-wall projection error compound and diffuse the cores.
  void setAccumulate(bool on) { accumulate_ = on; }
  bool accumulate() const { return accumulate_; }

private:
  void resetMaps();  ///< Psi, Phi <- id; u0 <- current velocity.

  /// Reference `estimateDistortion`: the worst round-trip departure from the
  /// identity over the interior, \f$\max_x\max(\lVert x-\Phi(\Psi(x))\rVert,
  /// \lVert x-\Psi(\Phi(x))\rVert)\f$, in PHYSICAL length (both directions, like
  /// `estimate_kernel`; skips the same 2-cell border).
  double rawDistortion() const;

  /// Max absolute velocity COMPONENT of the current field, floored at 1e-4 --
  /// the reference `getCFL` max_v used to normalize the distortion.
  double maxVelComponent() const;

  /// Alg. 5 lines 4-10 with a frozen flow velocity \p v and step \p dt, mutating
  /// the passed flow maps \p psi, \p phi in place and returning the projected
  /// velocity. Called twice per step: once as the midpoint PREDICTOR (line 3
  /// "result of lines 4-10 using flow v=u", on scratch map copies over dt/2) and
  /// once as the committed CORRECTOR (over dt, on the real maps). \p diag, if
  /// non-null, captures the per-substep fields of that call.
  FaceField advectAndCorrect(const FaceField& v, double dt, std::vector<Vec3>& psi,
                             std::vector<Vec3>& phi, CfMcmDiagnostics* diag, bool commit);

  MacGrid g_;
  MacProjector proj_;
  FaceField u_;             ///< current velocity (divergence-free)
  FaceField u0_;            ///< velocity snapshot at the last reinitialization
  std::vector<Vec3> psi_;   ///< inverse (backward) flow map, cell-centered
  std::vector<Vec3> phi_;   ///< forward flow map, cell-centered
  int reinitEvery_;
  int sinceReinit_ = 0;
  bool useLimiter_ = true;  ///< apply the CF Sec 5.4.2 BFECC minmax limiter
  bool secondOrder_ = true; ///< Alg. 5 line 3 midpoint flow estimate (vs 1st-order v=u)
  double remapTol_ = 0.5;   ///< reference distortion gate: normalized-D threshold (<=0 disables)
  double lastDistortion_ = 0.0;  ///< normalized distortion at the last step (reference VelocityDistortion)
  int accuracyResets_ = 0;  ///< count of accuracy-triggered reinits
  bool roundtripRefCurrent_ = false;  ///< line-8 e = ut0 - u0 (snapshot, curvature-preserving); true = current u (diagnostic, diffusive)
  double midpointFrac_ = 0.5;        ///< line-3 predictor step as a fraction of dt (0.5 = midpoint, faithful)
  bool accumulate_ = true;           ///< persistent-snapshot BiMocq accumulation (reference-faithful) vs frozen snapshot
  // Read only under BOCHNER_WITH_METAL (see MacProjection.h for the same
  // pattern); unused, not absent, in a non-Metal build.
  [[maybe_unused]] int gpuHandle_ = -1;  ///< resident GPU CF+MCM state handle (>=0 = GPU advection active); -1 = CPU
};

}  // namespace bochner
