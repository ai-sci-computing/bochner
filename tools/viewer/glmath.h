#pragma once
// Minimal column-major 4x4 matrix helpers for the viewer (avoids a glm dep).
#include <array>
#include <cmath>

#include "grid/Vec3.h"

namespace viewer {

using Mat4 = std::array<float, 16>;  // column-major, OpenGL-ready

inline Mat4 identity() {
  Mat4 m{};
  m[0] = m[5] = m[10] = m[15] = 1.0f;
  return m;
}

// Column-major multiply: result = a * b.
inline Mat4 mul(const Mat4& a, const Mat4& b) {
  Mat4 r{};
  for (int c = 0; c < 4; ++c)
    for (int row = 0; row < 4; ++row) {
      float s = 0.0f;
      for (int k = 0; k < 4; ++k) s += a[k * 4 + row] * b[c * 4 + k];
      r[c * 4 + row] = s;
    }
  return r;
}

inline Mat4 perspective(float fovyRad, float aspect, float zn, float zf) {
  const float f = 1.0f / std::tan(fovyRad * 0.5f);
  Mat4 m{};
  m[0] = f / aspect;
  m[5] = f;
  m[10] = (zf + zn) / (zn - zf);
  m[11] = -1.0f;
  m[14] = (2.0f * zf * zn) / (zn - zf);
  return m;
}

inline std::array<float, 3> sub(const std::array<float, 3>& a, const std::array<float, 3>& b) {
  return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}
inline std::array<float, 3> cross(const std::array<float, 3>& a, const std::array<float, 3>& b) {
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
inline std::array<float, 3> normalize(std::array<float, 3> v) {
  float n = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
  if (n > 0) {
    v[0] /= n;
    v[1] /= n;
    v[2] /= n;
  }
  return v;
}

inline Mat4 lookAt(const std::array<float, 3>& eye, const std::array<float, 3>& center,
                   const std::array<float, 3>& up) {
  auto f = normalize(sub(center, eye));
  auto s = normalize(cross(f, up));
  auto u = cross(s, f);
  auto dot = [](const std::array<float, 3>& a, const std::array<float, 3>& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  };
  Mat4 m = identity();
  m[0] = s[0];  m[4] = s[1];  m[8] = s[2];
  m[1] = u[0];  m[5] = u[1];  m[9] = u[2];
  m[2] = -f[0]; m[6] = -f[1]; m[10] = -f[2];
  m[12] = -dot(s, eye);
  m[13] = -dot(u, eye);
  m[14] = dot(f, eye);
  return m;
}

}  // namespace viewer
