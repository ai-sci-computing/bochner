#pragma once

#include <vector>

#include "grid/MacGrid.h"
#include "grid/Vec3.h"

namespace bochner {

/// A point where a vortex filament (the zero set of the complex field psi)
/// pierces an elementary plaquette of the cell-center lattice.
struct FilamentCrossing {
  Vec3 point;       ///< 3D location of the zero (bilinear interpolation)
  int orientation;  ///< +1/-1: sign of psi's phase winding around the plaquette
                    ///< (corners CCW about the +axis of ::plaquetteNormalAxis);
                    ///< +1 = the filament pierces the plaquette along +axis
  int plaquette;    ///< global plaquette id this crossing lies on (for linking)
};

/// \brief Crossing points of the vortex-filament set (zero set of psi) with the
/// elementary plaquettes of the cell-center lattice -- stage 1 of
/// Weissmann-Pinkall extraction on the MAC grid.
///
/// psi is a complex 0-form at cell centers, supplied interleaved as
/// `[Re psi_0, Im psi_0, Re psi_1, Im psi_1, ...]` of length `2*numCells`
/// (the layout from ::connectionLaplacian / ::smallestEigenpair, cell index =
/// MacGrid::cellIndex). Around each unit square spanned by four neighbouring
/// cell centers, the phase of psi either winds by +/-2*pi (a filament pierces
/// the square, curvature/vorticity flux = +/-1) or not. When it winds, the
/// enclosed zero is located by a few Newton steps on the bilinear field, and
/// the winding sign records which way the filament pierces the plaquette
/// (see FilamentCrossing::orientation). Linking the crossings into curves is
/// the next stage.
///
/// \param g   The MAC grid (cell/face indexing and spacing).
/// \param psi Interleaved complex cell field, length `2*g.numCells()`.
/// \throws std::invalid_argument if \p psi has the wrong length.
std::vector<FilamentCrossing> traceZeroSet(const MacGrid& g, const std::vector<double>& psi);

/// Normal axis (0=x, 1=y, 2=z) of a global plaquette id as assigned by
/// ::traceZeroSet -- FilamentCrossing::orientation is the phase-winding sign
/// about this +axis.
/// \throws std::invalid_argument if \p plaquette is not a valid id for \p g.
int plaquetteNormalAxis(const MacGrid& g, int plaquette);

/// A single extracted vortex filament: an ordered polyline of points. The
/// polyline is undirected -- per-crossing piercing directions live in
/// FilamentCrossing::orientation and are consumed by the pairing inside
/// ::linkFilaments, but no global direction is propagated onto the curve.
struct Filament {
  std::vector<Vec3> points;
  bool closed;  ///< true = closed loop (connect last->first); false = open path
};

/// \brief Link ::FilamentCrossing points into vortex-filament curves -- stage 2
/// of Weissmann-Pinkall extraction on the MAC grid.
///
/// Each elementary cube of the cell-center lattice (8 corner cells) is bounded
/// by 6 plaquettes; a filament passing through it enters and exits via two of
/// them, so the cube joins the crossings on those plaquettes -- pairing an
/// entering with a leaving crossing (orientation against the face's outward
/// normal), closest first, which disambiguates cubes shared by two strands.
/// A plaquette is shared by two cubes (interior) -- giving its crossing degree
/// 2, a curve passing through -- or one (lattice boundary) -- degree 1, an
/// open endpoint. Chaining the per-cube segments yields the filaments:
/// components with no boundary endpoint come out as **closed loops** (a vortex
/// ring), components reaching the lattice boundary come out **open**.
///
/// \returns One ::Filament per connected component of the zero set.
std::vector<Filament> linkFilaments(const MacGrid& g, const std::vector<FilamentCrossing>& crossings);

/// Diagnostic: cubes seen by ::linkFilaments whose crossings could not be
/// fully paired entering-with-leaving (odd or same-sense leftovers). Only
/// exact phase degeneracies -- an edge phase step of exactly pi, or a
/// |winding|=2 plaquette collapsed to one crossing -- produce them, so a
/// nonzero count flags input in need of scrutiny. Monotone across calls.
long filamentDegenerateCubeCount();
/// Reset ::filamentDegenerateCubeCount to zero.
void resetFilamentDegenerateCubeCount();

}  // namespace bochner
