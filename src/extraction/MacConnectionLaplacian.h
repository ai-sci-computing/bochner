#pragma once

#include "grid/CooMatrix.h"
#include "grid/GridOperators.h"
#include "grid/MacGrid.h"

namespace bochner {

/// \brief The U(1) connection ("magnetic Schrodinger") Laplacian on the dual
/// cell graph -- the operator at the heart of Weissmann-Pinkall vortex-filament
/// extraction and the project's research target, now on the regular MAC grid.
///
/// The complex section \f$\psi\f$ lives at **cell centers** (dual 0-form). Each
/// interior face is a graph edge linking the two cells it separates and carries
/// a holonomy phase \f$e^{i\theta_f}\f$, where \f$\theta_f\f$ is the connection
/// 1-form integrated along the dual edge -- i.e. the stored MAC face velocity
/// scaled by \f$1/\hbar\f$ (see \ref connectionAngles). This makes \f$E\f$
/// literally a U(1) lattice gauge theory on a regular lattice:
/// the curvature (holonomy around a cell loop) is the vorticity flux, and the
/// smallest-eigenvalue eigenvector's zero set is the vortex-filament set.
///
/// With \f$(d^\nabla\psi)_f = \psi_b - e^{i\theta_f}\psi_a\f$ (a,b the low/high
/// cells across the face) and edge weight \f$w = 1/h^2\f$, the Hermitian
/// operator \f$E = (d^\nabla)^\dagger d^\nabla\f$ is returned as a **real
/// symmetric** \f$2N\times2N\f$ matrix embedding each complex entry \f$a+ib\f$
/// as \f$\begin{psmallmatrix} a & -b \\ b & a \end{psmallmatrix}\f$ (decision
/// #7); cell \f$j\f$ occupies real rows/cols \f$\{2j, 2j+1\}\f$. A trivial
/// connection (\f$\theta\equiv0\f$) reduces to two decoupled copies of the
/// homogeneous-Neumann grid Laplacian (\ref pressureLaplacian).
///
/// \param g     The MAC grid (cell/face indexing and spacing \f$h\f$).
/// \param theta Connection angle per face (only interior faces are used).
CooMatrix connectionLaplacian(const MacGrid& g, const FaceField& theta);

/// \brief Connection angles from a velocity field: \f$\theta_f = u_f\,h/\hbar\f$.
///
/// The line integral of the velocity 1-form along the dual edge crossing face
/// \f$f\f$ is the stored normal component \f$u_f\f$ times the dual-edge length
/// \f$h\f$; dividing by \f$\hbar = h_{\text{strength}}/2\pi\f$ gives the U(1)
/// connection angle (Weissmann-Pinkall). The filament strength sets the flux
/// quantum: one filament per vortex of circulation \f$h_{\text{strength}}\f$.
/// \throws std::invalid_argument if \p hbar is not positive.
///
/// \warning The Weissmann-Pinkall construction requires \f$|\theta| < \pi\f$ on
/// every face. Past that the link phase wraps and the plaquette holonomy no
/// longer equals the enclosed vorticity flux, so the extracted windings
/// correspond to no physical filament. The failure is *not* obvious from the
/// output: aliasing increases the filament count, so it superficially
/// resembles the intended effect of lowering \f$\hbar\f$. Use
/// \ref maxConnectionAngle or \ref hbarAliasingFloor to check before trusting
/// an extraction.
FaceField connectionAngles(const MacGrid& g, const FaceField& u, double hbar);

/// Largest \f$|\theta|\f$ over all faces of \p theta -- the aliasing
/// diagnostic. Values \f$\ge \pi\f$ mean the holonomy has wrapped somewhere.
double maxConnectionAngle(const FaceField& theta);

/// Smallest \f$\hbar\f$ that keeps \f$|\theta| < \pi\f$ everywhere for the
/// velocity field \p u on grid \p g, i.e. \f$h\,\max|u| / \pi\f$. Below this
/// the extraction aliases. Returns 0 for a zero field.
double hbarAliasingFloor(const MacGrid& g, const FaceField& u);

}  // namespace bochner
