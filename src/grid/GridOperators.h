#pragma once

#include <vector>

#include "grid/MacGrid.h"
#include "grid/MacObstacle.h"

namespace bochner {

/// \brief Velocity stored as the normal component on each MAC face.
///
/// One coefficient per face, split into the three axis-aligned families
/// (sizes \ref MacGrid::numFacesX / Y / Z). `x[faceXIndex(i,j,k)]` is the
/// x-component of velocity on the x-normal face at integer x-plane `i`, and so
/// on. This is the dual 1-form / primal 2-form flux density placement.
struct FaceField {
  std::vector<double> x, y, z;
};

/// \brief Which of the six domain walls are OPEN (free-surface, Dirichlet
/// pressure \f$p=0\f$) versus CLOSED (solid, no-penetration \f$u\cdot n=0\f$).
///
/// Covector Fluids / Bridson 2015 (CF 5.4.5) support both: a solid wall is a
/// no-stick Neumann condition (\f$u\cdot n=0\f$), while a free surface pins the
/// pressure to zero on the interface, letting flow leave the domain. The paper's
/// leapfrogging experiment (Fig. 10) puts the rings within 0.5 m of the y,z
/// walls, so only open boundaries avoid the wall-image confinement that stalls
/// the outer ring. A wall flagged `true` here is open. All false
/// (the default) reproduces the closed no-penetration box.
struct BoundarySpec {
  bool xlo = false, xhi = false, ylo = false, yhi = false, zlo = false, zhi = false;

  /// Prescribed uniform \f$+x\f$ inlet speed on the x-lo wall (0 = no inlet).
  /// An INLET is a third wall type: the boundary normal velocity
  /// is *prescribed* (\f$u\cdot n = \text{inflowX}\f$), so -- unlike a CLOSED
  /// wall -- it is not zeroed, and -- unlike an OPEN wall -- it is not
  /// pressure-corrected. In the pressure Poisson it behaves as a closed (Neumann)
  /// wall; the prescribed flux enters the divergence natively and is balanced by
  /// an OPEN outlet (the flow-past-obstacle setup needs both). x-lo only for now
  /// (the single inflow direction of a flow tank); generalize when a scenario
  /// needs it.
  double inflowX = 0.0;

  static BoundarySpec allClosed() { return {}; }
  static BoundarySpec allOpen() { return {true, true, true, true, true, true}; }
  /// Flow tank: a uniform \f$+x\f$ stream \p speed enters the x-lo inlet and
  /// leaves the OPEN x-hi outlet; the other four walls are closed (free-slip
  /// channel). This is the base setup for flow past an obstacle.
  static BoundarySpec channelFlow(double speed) {
    BoundarySpec b;
    b.xhi = true;       // open outlet balances the inflow
    b.inflowX = speed;  // x-lo inlet
    return b;
  }
  bool inletXlo() const { return inflowX != 0.0; }
  bool anyOpen() const { return xlo || xhi || ylo || yhi || zlo || zhi; }
  bool allOpenWalls() const { return xlo && xhi && ylo && yhi && zlo && zhi; }
};

namespace ops {

/// A zero-initialised face field sized to grid \p g.
FaceField zeroFaceField(const MacGrid& g);

/// \brief Discrete divergence (the MAC `d`): faces -> cells.
///
/// `div[c] = sum over the cell's 6 faces of (outward normal velocity)/h`, i.e.
/// the finite-volume net flux per unit volume. Returns one value per cell.
std::vector<double> divergence(const MacGrid& g, const FaceField& u);

/// \brief Discrete gradient (cells -> faces), the negative adjoint of \ref
/// divergence on the interior.
///
/// `grad[f] = (p[c+] - p[c-]) / h` across each *interior* face. Boundary faces
/// carry **no** gradient (set to zero): they are prescribed by the boundary
/// conditions, so the pressure projection corrects only interior faces. With
/// this convention `grad = -div^T` on interior faces and
/// `divergence(gradient(p))` is the homogeneous-Neumann 7-point Laplacian.
FaceField gradient(const MacGrid& g, const std::vector<double>& p);

/// \brief BC-aware gradient: like \ref gradient, but on each OPEN wall the
/// boundary-normal face gets the Dirichlet ghost difference \f$(p_{cell}-0)/h\f$
/// (the outflow correction), while CLOSED walls keep the zero-face convention.
FaceField gradient(const MacGrid& g, const std::vector<double>& p, const BoundarySpec& bc);

/// \brief Enforce no-penetration `u.n = 0` on the six domain walls by zeroing
/// the wall-normal boundary faces of \p u in place.
///
/// This is the boundary condition of the pressure projection: with the walls
/// zeroed, `divergence(u)` sums to exactly zero (the discrete divergence theorem
/// with zero boundary flux), so the all-Neumann pressure Poisson is compatible
/// with no mean-subtraction, and the projected field has `u.n = 0` at the walls.
/// Equivalently, this makes the projection solve the *inhomogeneous* Neumann
/// problem `dphi/dn = u.n` (which removes the wall-normal flux) rather than the
/// homogeneous one (which would leave it).
void enforceNoPenetration(const MacGrid& g, FaceField& u);

/// \brief Enforce no-penetration only on the CLOSED walls of \p bc, leaving the
/// OPEN walls' normal faces untouched (they carry legitimate outflow).
void enforceNoPenetration(const MacGrid& g, FaceField& u, const BoundarySpec& bc);

/// \brief Explicit viscous diffusion of a MAC velocity field: one forward-Euler
/// step of \f$\partial_t u = \nu\,\nabla^2 u\f$, i.e.
/// \f$u \mathrel{+}= \nu\,\Delta t\,\nabla^2 u\f$ applied component-wise (the
/// vector Laplacian = per-family scalar Laplacian on the MAC grid).
///
/// This is the vorticity SINK/generator of the flow-past-obstacle demo: it
/// dissipates the wake (bounding it and setting the Reynolds number) and, via a
/// **no-slip** obstacle, generates the boundary-layer vorticity that sheds. Each
/// interior face is diffused against its six same-family neighbours; a neighbour
/// that touches a \p solid cell contributes value 0 (no-slip Dirichlet at the
/// obstacle), a neighbour off the domain edge is dropped (Neumann = free-slip
/// walls). Faces on/inside the obstacle are left untouched. Explicit forward
/// Euler, stable only for \f$\nu\,\Delta t/h^2 \le 1/6\f$; a larger step is split
/// internally into equal, individually-stable sub-steps, so the call is
/// unconditionally stable (and is a single step, byte-identical to before, for
/// \f$\nu\,\Delta t/h^2 < 1/6\f$ strictly -- at an exact multiple of the bound
/// one extra sub-step avoids the undamped checkerboard edge case).
FaceField diffuseVelocity(const MacGrid& g, const FaceField& u, double nu, double dt,
                          const SolidMask& solid = {});

}  // namespace ops
}  // namespace bochner
