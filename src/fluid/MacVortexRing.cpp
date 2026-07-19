#include "fluid/MacVortexRing.h"

#include <cmath>
#include <vector>

namespace bochner {

namespace {

// Fill a MAC face field with the normal components of a velocity function.
template <class Vel>
FaceField fillFaces(const MacGrid& g, Vel&& velocity) {
  FaceField u = ops::zeroFaceField(g);
  for (int i = 0; i <= g.nx(); ++i)
    for (int j = 0; j < g.ny(); ++j)
      for (int k = 0; k < g.nz(); ++k) u.x[g.faceXIndex(i, j, k)] = velocity(g.faceXCenter(i, j, k))[0];
  for (int i = 0; i < g.nx(); ++i)
    for (int j = 0; j <= g.ny(); ++j)
      for (int k = 0; k < g.nz(); ++k) u.y[g.faceYIndex(i, j, k)] = velocity(g.faceYCenter(i, j, k))[1];
  for (int i = 0; i < g.nx(); ++i)
    for (int j = 0; j < g.ny(); ++j)
      for (int k = 0; k <= g.nz(); ++k) u.z[g.faceZIndex(i, j, k)] = velocity(g.faceZCenter(i, j, k))[2];
  return u;
}

}  // namespace

std::vector<Vec3> circleCurve(const Vec3& center, const Vec3& axis, double radius, int segments) {
  // Orthonormal frame (e1, e2) spanning the ring plane (perpendicular to axis).
  Vec3 n = axis;
  const double nlen = vnorm(n);
  if (nlen > 0.0) n = vscale(n, 1.0 / nlen);
  Vec3 t = (std::abs(n[0]) < 0.9) ? Vec3{1, 0, 0} : Vec3{0, 1, 0};
  Vec3 e1 = vsub(t, vscale(n, vdot(t, n)));
  e1 = vscale(e1, 1.0 / vnorm(e1));
  const Vec3 e2 = vcross(n, e1);

  std::vector<Vec3> loop(segments);
  for (int i = 0; i < segments; ++i) {
    const double a = 2.0 * M_PI * i / segments;
    loop[i] = vadd(center, vadd(vscale(e1, radius * std::cos(a)), vscale(e2, radius * std::sin(a))));
  }
  return loop;
}

std::vector<Vec3> trefoilKnotCurve(const Vec3& center, double scale, int segments) {
  std::vector<Vec3> loop(segments);
  for (int i = 0; i < segments; ++i) {
    const double t = 2.0 * M_PI * i / segments;
    const Vec3 p{std::sin(t) + 2.0 * std::sin(2.0 * t),  //
                 std::cos(t) - 2.0 * std::cos(2.0 * t),  //
                 -std::sin(3.0 * t)};
    loop[i] = vadd(center, vscale(p, scale));
  }
  return loop;
}

FaceField filamentFaceField(const MacGrid& g, const std::vector<std::vector<Vec3>>& loops,
                            double circulation, double coreRadius) {
  // Discretize every loop into segment midpoints and tangents (cyclic: the last
  // point connects back to the first).
  std::vector<Vec3> mid, dl;
  for (const auto& loop : loops) {
    const int n = static_cast<int>(loop.size());
    for (int i = 0; i < n; ++i) {
      const Vec3& p0 = loop[i];
      const Vec3& p1 = loop[(i + 1) % n];
      mid.push_back(vscale(vadd(p0, p1), 0.5));
      dl.push_back(vsub(p1, p0));
    }
  }

  const double c2 = coreRadius * coreRadius;
  const double kc = circulation / (4.0 * M_PI);
  const int segs = static_cast<int>(mid.size());
  auto velocity = [&](const Vec3& x) {
    Vec3 v{0, 0, 0};
    for (int i = 0; i < segs; ++i) {
      const Vec3 r = vsub(x, mid[i]);
      const double d2 = vdot(r, r) + c2;
      const double inv = 1.0 / (d2 * std::sqrt(d2));  // (|r|^2 + core^2)^{-3/2}
      v = vadd(v, vscale(vcross(dl[i], r), inv));
    }
    return vscale(v, kc);
  };
  return fillFaces(g, velocity);
}

FaceField filamentFaceField(const MacGrid& g, const std::vector<Vec3>& loop, double circulation,
                            double coreRadius) {
  return filamentFaceField(g, std::vector<std::vector<Vec3>>{loop}, circulation, coreRadius);
}

FaceField vortexRingFaceField(const MacGrid& g, const Vec3& center, const Vec3& axis,
                              double radius, double circulation, double coreRadius, int segments) {
  return filamentFaceField(g, circleCurve(center, axis, radius, segments), circulation, coreRadius);
}

FaceField hillVortexFaceField(const MacGrid& g, const Vec3& center, const Vec3& axis, double a,
                              double U, bool labFrame) {
  Vec3 n = axis;
  const double nlen = vnorm(n);
  if (nlen > 0.0) n = vscale(n, 1.0 / nlen);
  const double a2 = a * a, a3 = a2 * a;
  // Lab frame = co-moving field minus the uniform stream U along the axis
  // (so the fluid is at rest at infinity and the vortex translates).
  const double bg = labFrame ? U : 0.0;

  // Hill's spherical vortex velocity at world point x, in the frame moving with
  // the vortex (far field = uniform U along the axis). Axisymmetric, swirl-free:
  // build axial (zc) and radial (rho) coordinates about the axis through center.
  auto velocity = [&](const Vec3& x) {
    const Vec3 d = vsub(x, center);
    const double zc = vdot(d, n);
    const Vec3 radial = vsub(d, vscale(n, zc));
    const double rho = vnorm(radial);
    const double r2 = rho * rho + zc * zc;
    const double r = std::sqrt(r2);

    double uRho, uZ;
    if (r <= a) {  // interior: C = -3U/(4a^2), omega_phi proportional to rho
      const double k = -1.5 * U / a2;
      uRho = k * rho * zc;
      uZ = k * (a2 - zc * zc - 2.0 * rho * rho);
    } else {  // exterior: potential flow past a sphere of radius a
      const double r5 = r2 * r2 * r;
      uRho = -1.5 * U * a3 * rho * zc / r5;
      uZ = 0.5 * U * (2.0 * (1.0 - a3 / (r2 * r)) + 3.0 * a3 * rho * rho / r5);
    }
    Vec3 v = vscale(n, uZ - bg);  // subtract the uniform stream in the lab frame
    if (rho > 1e-12) v = vadd(v, vscale(radial, uRho / rho));
    return v;
  };

  FaceField u = ops::zeroFaceField(g);
  for (int i = 0; i <= g.nx(); ++i)
    for (int j = 0; j < g.ny(); ++j)
      for (int k = 0; k < g.nz(); ++k) u.x[g.faceXIndex(i, j, k)] = velocity(g.faceXCenter(i, j, k))[0];
  for (int i = 0; i < g.nx(); ++i)
    for (int j = 0; j <= g.ny(); ++j)
      for (int k = 0; k < g.nz(); ++k) u.y[g.faceYIndex(i, j, k)] = velocity(g.faceYCenter(i, j, k))[1];
  for (int i = 0; i < g.nx(); ++i)
    for (int j = 0; j < g.ny(); ++j)
      for (int k = 0; k <= g.nz(); ++k) u.z[g.faceZIndex(i, j, k)] = velocity(g.faceZCenter(i, j, k))[2];
  return u;
}

FaceField leapfrogSlabFaceField(const MacGrid& g, const Vec3& slabCenter, const Vec3& propAxis,
                                double rInner, double rOuter, double circulation) {
  const double h = g.spacing();
  const double halfThick = 1.5 * h;              // reference makeSlab thickness parameter
  const double vel = circulation / (3.0 * h);    // reference sheet-speed scaling
  const double an = vnorm(propAxis);
  const Vec3 nhat = an > 1e-12 ? vscale(propAxis, 1.0 / an) : Vec3{0, 0, 1};
  // Impose vmag * nhat inside the slab; fillFaces keeps each face's normal
  // component, so an axis-aligned nhat lands the slug on exactly one face set
  // (matching the reference's u_x-only construction), and the caller's pressure
  // projection turns the two radial jumps into the nested vortex-sheet pair.
  return fillFaces(g, [&](const Vec3& p) -> Vec3 {
    const Vec3 d = vsub(p, slabCenter);
    const double along = vdot(d, nhat);
    if (std::abs(along) > halfThick) return Vec3{0, 0, 0};
    const double r = vnorm(vsub(d, vscale(nhat, along)));  // radial distance in the slab plane
    double vmag = 0.0;
    if (r <= rInner)
      vmag = 2.0 * vel;
    else if (r <= rOuter)
      vmag = vel;
    return vscale(nhat, vmag);
  });
}

}  // namespace bochner
