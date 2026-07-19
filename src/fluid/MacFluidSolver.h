#pragma once

#include "grid/GridOperators.h"
#include "grid/MacGrid.h"

namespace bochner {

/// \brief One Covector Fluids time step, 1st order (CF Alg. 1).
///
/// Freeze the flow velocity `v <- u`, advect the velocity covector with BFECC
/// (\ref advectCovectorBFECC), then pressure-project to enforce
/// incompressibility (\ref projectToDivergenceFree, closed domain). The
/// returned field is discretely divergence-free.
FaceField stepCovectorFluids(const MacGrid& g, const FaceField& u, double dt);

/// \brief One Covector Fluids time step, 2nd-order midpoint (CF Alg. 2).
///
/// Estimate the flow velocity at the midpoint `v <- P(A_covec(u; u, dt/2))`,
/// then take the full step `u <- P(A_covec(u; v, dt))`. The midpoint flow
/// estimate reduces the truncation error from freezing the velocity (CF Fig. 7).
FaceField stepCovectorFluidsMidpoint(const MacGrid& g, const FaceField& u, double dt);

}  // namespace bochner
