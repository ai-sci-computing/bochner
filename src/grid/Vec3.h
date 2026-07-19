#pragma once

#include <array>
#include <cmath>

namespace bochner {

/// A 3D point/vector. Free functions below avoid operator-ADL pitfalls with
/// std::array.
using Vec3 = std::array<double, 3>;

inline Vec3 vadd(const Vec3& a, const Vec3& b) { return {a[0] + b[0], a[1] + b[1], a[2] + b[2]}; }
inline Vec3 vsub(const Vec3& a, const Vec3& b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
inline Vec3 vscale(const Vec3& a, double s) { return {a[0] * s, a[1] * s, a[2] * s}; }
inline double vdot(const Vec3& a, const Vec3& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
inline Vec3 vcross(const Vec3& a, const Vec3& b) {
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
inline double vnorm(const Vec3& a) { return std::sqrt(vdot(a, a)); }

}  // namespace bochner
