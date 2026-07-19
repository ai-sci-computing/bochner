#pragma once

#include <vector>

#include "grid/GridOperators.h"
#include "grid/MacGrid.h"
#include "grid/Vec3.h"

namespace bochner {

/// \brief Reconstruct the full velocity vector at an arbitrary world point.
///
/// Each MAC component lives staggered on its own face lattice; this trilinearly
/// interpolates each on its lattice and returns the assembled 3-vector. Points
/// outside the domain are clamped to the closest interior sample (CF §5.4.5
/// "streak boundary"), so backtraces that leave the box stay well-defined.
/// Trilinear interpolation is exact for fields that are affine per component.
Vec3 sampleVelocity(const MacGrid& g, const FaceField& u, const Vec3& p);

/// \brief One RK4 step of the backward flow map: integrate `dx/dt = v` from
/// \p start over a span \p dt *backwards*, returning \f$\Psi(\texttt{start})\f$.
///
/// This is the inverse-flow-map evaluation used by the semi-Lagrangian covector
/// advection (CF Alg 3): from a grid location we trace back along the frozen
/// flow velocity \p v to find where the advected quantity came from.
Vec3 backtrace(const MacGrid& g, const FaceField& v, const Vec3& start, double dt);

/// \brief Semi-Lagrangian covector advection \f$\mathcal{A}^{sL}(u; v, \Delta t)\f$
/// (Covector Fluids Alg. 3) -- the staggered pullback \f$u \leftarrow (d\Psi)^\top
/// (u\circ\Psi)\f$.
///
/// For each *interior* face with normal along axis \f$n\f$, the new normal
/// component is the transpose-Jacobian column for \f$n\f$ dotted with the full
/// velocity sampled at the backtraced face center (CF Eq. 39):
/// \f[ u_n|_F \leftarrow \big[\partial\Psi_x/\partial n,\ \partial\Psi_y/\partial n,\
///     \partial\Psi_z/\partial n\big]_F \cdot \big[u_1,u_2,u_3\big]^\top_{\Psi(F)}, \f]
/// where each \f$\partial\Psi_\alpha/\partial n\f$ is the finite difference of the
/// backtraced cell-center map across the face (CF Eq. 40). Boundary faces are
/// left untouched (BC-prescribed). \p u is the field advected; \p v is the
/// frozen flow velocity.
FaceField advectCovectorSL(const MacGrid& g, const FaceField& u, const FaceField& v, double dt);

/// \brief BFECC covector advection \f$\mathcal{A}^{BFECC}(u; v, \Delta t)\f$
/// (Covector Fluids Alg. 4) -- back-and-forth error compensation around
/// \ref advectCovectorSL to remove its first-order numerical dissipation.
///
/// Computes the forward advection `u1`, advects it back to estimate the
/// round-trip error `e`, and corrects: `out = u1 - A^sL(e/2; v, dt)`. The result
/// is then run through the §5.4.2 extrema (minmax) limiter: each interior face
/// value is clamped to the min/max of `u1` over its immediate same-family
/// neighbour box, so no new extrema (oscillations) are introduced. This is the
/// non-dissipative advection that preserves vortex structure (CF Fig. 5d).
FaceField advectCovectorBFECC(const MacGrid& g, const FaceField& u, const FaceField& v, double dt);

/// \brief Trilinearly sample a **cell-centered** vector field at a world point.
///
/// \p field holds one \ref Vec3 per cell (length `g.numCells()`, indexed by
/// \ref MacGrid::cellIndex). Used to evaluate an accumulated flow map (stored at
/// cell centers) at arbitrary points. With \p extrap `false` (default),
/// out-of-domain queries clamp to the closest interior sample (streak boundary,
/// correct for a velocity). With \p extrap `true`, out-of-domain queries are
/// linearly EXTRAPOLATED -- the correct treatment for a flow MAP, whose values
/// are Lagrangian coordinates that legitimately lie outside the domain (clamping
/// a map injects a wall error the flow advects inward).
Vec3 sampleCellVec3(const MacGrid& g, const std::vector<Vec3>& field, const Vec3& p,
                    bool extrap = false);

/// \brief Covector pullback through an **arbitrary cell-centered map**:
/// \f$u_1 \leftarrow (dM)^\top (u_0 \circ M)\f$ (CF Eq. 39/40), where \p map is a
/// field of source-space positions (e.g. the accumulated inverse flow map
/// \f$\Psi\f$ or the forward map \f$\Phi\f$ of CF+MCM, Alg. 5).
///
/// This generalizes \ref advectCovectorSL (which builds a *single-step* map by
/// backtracing): the per-face transpose-Jacobian column \f$\partial M/\partial n\f$
/// is the finite difference of \p map across the face, and the source velocity
/// \p source is sampled at the face's mapped position \f$M(F)\f$. Interior faces
/// only; boundary faces keep their \p source value.
FaceField pullbackThroughMap(const MacGrid& g, const std::vector<Vec3>& map,
                             const FaceField& source);

/// \brief The CF §5.4.2 extrema (minmax) limiter: clamp each interior face of
/// \p candidate to the min/max of \p reference over its immediate same-family
/// neighbour box, so a BFECC/CF+MCM correction introduces no new extrema.
FaceField bfeccLimit(const MacGrid& g, const FaceField& reference, const FaceField& candidate);

#ifdef BOCHNER_WITH_METAL
/// \brief GPU BFECC covector advection -- the three semi-Lagrangian sub-steps of
/// \ref advectCovectorBFECC run on the Metal backend (\ref gpu::advectCovectorSLGPU),
/// with the error blend and the CF 5.4.2 extrema limiter on the host. Single
/// precision; matches \ref advectCovectorBFECC to float tolerance. Declared only
/// when the Metal backend is built.
FaceField advectCovectorBFECCGpu(const MacGrid& g, const FaceField& u, const FaceField& v, double dt);
#endif

}  // namespace bochner
