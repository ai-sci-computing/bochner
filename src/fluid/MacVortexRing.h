#pragma once

#include <vector>

#include "grid/GridOperators.h"
#include "grid/MacGrid.h"
#include "grid/Vec3.h"

namespace bochner {

/// \brief Sample a circle as a closed polyline of \p segments points.
///
/// Points lie on the radius-\p radius circle in the plane through \p center
/// perpendicular to \p axis (which need not be unit). Suitable as a \ref
/// filamentFaceField loop; \ref vortexRingFaceField is exactly this fed to it.
std::vector<Vec3> circleCurve(const Vec3& center, const Vec3& axis, double radius,
                              int segments = 256);

/// \brief Sample the standard trefoil knot as a closed polyline.
///
/// Uses the classic parametrization scaled by \p scale about \p center:
/// `(sin t + 2 sin 2t, cos t - 2 cos 2t, -sin 3t)`. The simplest *knotted*
/// vortex filament -- seeded with \ref filamentFaceField it reproduces the
/// Covector Fluids / Kleckner-Irvine trefoil that unties itself via reconnection
/// (CF Fig. 9).
std::vector<Vec3> trefoilKnotCurve(const Vec3& center, double scale, int segments = 256);

/// \brief Seed the regularized Biot-Savart velocity of one closed vortex
/// filament (any shape) as MAC face-normal components.
///
/// \p loop is a closed polyline (treated cyclically: the segment from the last
/// point back to the first is included) of total \p circulation Gamma; the
/// Biot-Savart kernel is regularized with finite \p coreRadius (Rosenhead-Moore,
/// `|r|^2 -> |r|^2 + coreRadius^2`). This is the general seeder behind
/// \ref vortexRingFaceField (a circle) and the knot/leapfrog/link demos.
FaceField filamentFaceField(const MacGrid& g, const std::vector<Vec3>& loop, double circulation,
                            double coreRadius);

/// \brief Seed several closed filaments at once (their Biot-Savart velocities
/// superpose), each with the same \p circulation and \p coreRadius.
///
/// Use for leapfrogging rings (two coaxial circles) or a Hopf link (two
/// interlocked circles).
FaceField filamentFaceField(const MacGrid& g, const std::vector<std::vector<Vec3>>& loops,
                            double circulation, double coreRadius);

/// \brief Seed a vortex ring's velocity field as MAC face-normal components.
///
/// Samples the regularized Biot-Savart velocity of a circular vortex filament
/// at each face center and stores its normal component (seeded
/// vortex rings are the project's vorticity source). The ring has the given
/// \p center, \p axis (need not be unit), \p radius, and total \p circulation
/// Gamma; the Biot-Savart kernel is regularized with a finite \p coreRadius
/// (Rosenhead-Moore, `|r|^2 -> |r|^2 + coreRadius^2`).
///
/// The vorticity is concentrated near the ring circle and the field is
/// irrotational elsewhere, so the U(1) connection (\ref connectionAngles) has
/// holonomy ~2*pi around the core (one flux quantum when hbar = Gamma/2*pi) and
/// ~0 elsewhere -- the lowest connection-Laplacian mode then has its zero set
/// on the ring (Weissmann-Pinkall).
FaceField vortexRingFaceField(const MacGrid& g, const Vec3& center, const Vec3& axis,
                              double radius, double circulation, double coreRadius,
                              int segments = 256);

/// \brief Seed Hill's spherical vortex as MAC face-normal velocity components.
///
/// Hill's vortex is a sphere of radius \p a (about \p center, symmetry \p axis)
/// of *distributed* vorticity (azimuthal, \f$\omega_\phi \propto \rho\f$), with
/// potential flow-past-a-sphere outside; the field is solenoidal and velocity
/// is continuous across \f$r=a\f$ (no vortex sheet). Unlike a thin ring, its
/// spread-out vorticity decomposes (Weissmann-Pinkall) into a *nest* of coaxial
/// filament rings -- more of them as the flux quantum hbar shrinks.
///
/// With \p labFrame `false` the field is given in the frame moving with the
/// vortex (far field = uniform stream \p speed along the axis), so the vortex is
/// stationary -- and in a closed box the projection removes that uniform stream,
/// leaving it motionless. With \p labFrame `true` the uniform stream is
/// subtracted (fluid at rest at infinity), so the field decays away from the
/// sphere and the vortex **self-propels** along its axis, like the ring.
FaceField hillVortexFaceField(const MacGrid& g, const Vec3& center, const Vec3& axis, double a,
                              double speed, bool labFrame = false);

/// \brief Seed the Covector Fluids leapfrog initial condition as a thin
/// *vortex-sheet slab*, exactly as the reference code (Nabizadeh et al. 2022,
/// supplementary `CovectorFluids_code`, experiment 1) constructs it.
///
/// This is the **faithful** leapfrog seed, and the choice of IC is decisive: the
/// reference does NOT seed two Biot-Savart filament rings
/// (\ref filamentFaceField) -- it imposes a slug of axial velocity on a thin slab
/// perpendicular to \p propAxis and lets the pressure projection turn the slab's
/// two radial velocity jumps into a pair of nested, coaxial vortex sheets. With
/// this IC the rings leapfrog (the inner sheet expands, the outer contracts,
/// their radii swap); with the filament seed they merge/bounce and never thread.
///
/// Inside a slab of thickness \f$1.5\,h\f$ about the plane through \p slabCenter
/// normal to \p propAxis, the velocity along \p propAxis is
/// \f$2\,\mathrm{vel}\f$ for cylindrical radius \f$r\le r_\text{inner}\f$,
/// \f$\mathrm{vel}\f$ for \f$r_\text{inner}<r\le r_\text{outer}\f$, and 0 outside,
/// where \f$\mathrm{vel}=\Gamma/(3h)\f$ (reference scaling; \p circulation is
/// \f$\Gamma\f$). All other components are 0. The **caller must project** the
/// result to divergence-free (this is what materializes the sheets), matching the
/// other seeders' contract.
FaceField leapfrogSlabFaceField(const MacGrid& g, const Vec3& slabCenter, const Vec3& propAxis,
                                double rInner, double rOuter, double circulation);

}  // namespace bochner
