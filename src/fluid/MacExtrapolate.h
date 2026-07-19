#pragma once

#include "grid/GridOperators.h"
#include "grid/MacGrid.h"
#include "grid/MacObstacle.h"

namespace bochner {

/// \brief Extrapolate fluid velocity into a band of solid cells so that
/// semi-Lagrangian backtraces near an obstacle sample a plausible field
/// instead of the identically-zero interior (Bridson sec. 6).
///
/// Faces with at least one adjacent fluid cell keep their values; the rest are
/// filled by repeated averaging from already-filled neighbours, one cell-layer
/// per sweep, for \p band sweeps. Call this on the velocity field *before*
/// advection. The projection re-imposes `u.n = 0` on solid faces afterwards, so
/// the extrapolated values never leak into the physical solution -- they exist
/// only to be sampled by backtraces.
///
/// \p band should cover the furthest backtrace, i.e. `ceil(CFL) + 2` cells.
/// A no-op when \p solid is empty or \p band <= 0.
void extrapolateIntoSolid(const MacGrid& g, FaceField& u, const SolidMask& solid, int band);

}  // namespace bochner
