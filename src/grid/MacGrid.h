#pragma once

#include "grid/Vec3.h"

namespace bochner {

/// \brief A uniform 3D MAC (marker-and-cell) grid as a cubical cell complex.
///
/// This is the substrate of the grid rebuild,
/// replacing the retired tetrahedral mesh. Placement is **dual/flux**:
///   - **cells** (centers) carry scalars — pressure, and the connection
///     Laplacian's section \f$\psi\f$ (dual 0-forms);
///   - **faces** carry the velocity as a *normal flux* (primal 2-form / dual
///     1-form). The component normal to a face is the MAC face velocity.
///
/// The grid has `nx * ny * nz` cubic cells of side \ref spacing(). Faces split
/// into three axis-aligned families; the family normal to an axis has one extra
/// layer along that axis (the two domain boundaries). All operators (`d`, the
/// diagonal Hodge stars, divergence/gradient) are realised as **structured
/// stencils** over this indexing rather than as a general cell complex.
///
/// Linear indices run with the first axis slowest and the last fastest
/// (C/row-major over `(i,j,k)`).
class MacGrid {
public:
  /// Construct a `nx * ny * nz` grid of cubic cells with side \p h, whose
  /// minimum corner is at \p origin.
  /// \throws std::invalid_argument if any dimension is < 1 or \p h is not > 0.
  MacGrid(int nx, int ny, int nz, double h, Vec3 origin = {0.0, 0.0, 0.0});

  /// \name Dimensions
  /// \{
  int nx() const noexcept { return nx_; }
  int ny() const noexcept { return ny_; }
  int nz() const noexcept { return nz_; }
  double spacing() const noexcept { return h_; }
  Vec3 origin() const noexcept { return origin_; }
  /// \}

  /// \name Element counts
  /// \{
  int numCells() const noexcept { return nx_ * ny_ * nz_; }
  int numFacesX() const noexcept { return (nx_ + 1) * ny_ * nz_; }
  int numFacesY() const noexcept { return nx_ * (ny_ + 1) * nz_; }
  int numFacesZ() const noexcept { return nx_ * ny_ * (nz_ + 1); }
  int numFaces() const noexcept { return numFacesX() + numFacesY() + numFacesZ(); }
  /// \}

  /// \name Linear indexing (no bounds checking; callers stay within range)
  /// \{
  int cellIndex(int i, int j, int k) const noexcept { return (i * ny_ + j) * nz_ + k; }
  int faceXIndex(int i, int j, int k) const noexcept { return (i * ny_ + j) * nz_ + k; }
  int faceYIndex(int i, int j, int k) const noexcept { return (i * (ny_ + 1) + j) * nz_ + k; }
  int faceZIndex(int i, int j, int k) const noexcept { return (i * ny_ + j) * (nz_ + 1) + k; }
  /// \}

  /// \name Geometry (positions in world space)
  /// \{
  /// Center of cell `(i,j,k)`, at the half-integer lattice offset.
  Vec3 cellCenter(int i, int j, int k) const noexcept;
  /// Center of the x-normal face at the integer x-plane `i`, in-plane half offsets.
  Vec3 faceXCenter(int i, int j, int k) const noexcept;
  /// Center of the y-normal face at the integer y-plane `j`.
  Vec3 faceYCenter(int i, int j, int k) const noexcept;
  /// Center of the z-normal face at the integer z-plane `k`.
  Vec3 faceZCenter(int i, int j, int k) const noexcept;
  /// \}

private:
  int nx_, ny_, nz_;
  double h_;
  Vec3 origin_;
};

}  // namespace bochner
