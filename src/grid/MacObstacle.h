#pragma once

#include <cstdint>
#include <vector>

#include "grid/MacGrid.h"
#include "grid/Vec3.h"

namespace bochner {

/// \brief A per-cell solid mask: `1` where the cell center lies inside a solid
/// obstacle, `0` in the fluid. Sized to \ref MacGrid::numCells.
///
/// The pressure projection treats a face between a fluid and a solid cell as a
/// no-penetration (Neumann) wall in the interior: the normal velocity is forced
/// to zero and the two cells are not coupled -- exactly the closed-wall
/// treatment, applied around the obstacle. Solid cells carry
/// no pressure degree of freedom.
using SolidMask = std::vector<std::uint8_t>;

/// True if cell \p c is solid in \p solid (an empty mask means "all fluid").
inline bool isSolid(const SolidMask& solid, int c) {
  return !solid.empty() && solid[c] != 0;
}

/// \brief Solid mask of an infinite circular cylinder of \p radius whose axis
/// passes through \p center along \p axis (need not be unit).
///
/// A cell is solid when its center's perpendicular distance to the axis line is
/// \f$\le\f$ \p radius. For a cylinder spanning the domain (axis along a grid
/// direction) the flow is quasi-2D in the plane perpendicular to the axis -- the
/// von Karman / flow-past-a-cylinder setup.
SolidMask cylinderMask(const MacGrid& g, const Vec3& center, const Vec3& axis, double radius);

/// \brief Solid mask of a sphere of \p radius centered at \p center: cells whose
/// center is within \p radius are solid.
SolidMask sphereMask(const MacGrid& g, const Vec3& center, double radius);

/// \brief Solid mask of a finite oriented rectangular box centered at \p center,
/// with half-extents \p half along the box's local axes \p ex, \p ey, \p ez
/// (each need not be unit; assumed mutually orthogonal).
///
/// A cell is solid when its offset from \p center projects to within the
/// respective half-extent along each local axis. Because the box is FINITE in
/// every direction it need not touch the domain walls (flow can pass around all
/// sides -- a genuinely 3D wake), and any orientation is possible. It subsumes
/// the square cylinder (square cross-section, long span) and the inclined flat
/// plate (one thin extent, yawed); an inclined/asymmetric box sheds NATURALLY
/// with no symmetry-breaking offset needed.
SolidMask boxMask(const MacGrid& g, const Vec3& center, const Vec3& ex, const Vec3& ey,
                  const Vec3& ez, const Vec3& half);

}  // namespace bochner
