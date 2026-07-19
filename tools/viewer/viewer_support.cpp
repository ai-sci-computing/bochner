#include "viewer_support.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

#include <sys/resource.h>  // getrusage: peak RSS (high-water mark)
#ifdef __APPLE__
#include <mach/mach.h>     // task_info: current resident set size
#endif

using bochner::Vec3;
using bochner::vadd;
using bochner::vcross;
using bochner::vdot;
using bochner::vnorm;
using bochner::vscale;
using bochner::vsub;

namespace viewer {

void memoryMB(double& current, double& peak) {
  rusage ru{};
  getrusage(RUSAGE_SELF, &ru);
#ifdef __APPLE__
  peak = ru.ru_maxrss / (1024.0 * 1024.0);  // Darwin: bytes
  mach_task_basic_info info{};
  mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
  current = (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                       reinterpret_cast<task_info_t>(&info), &count) == KERN_SUCCESS)
                ? info.resident_size / (1024.0 * 1024.0)
                : 0.0;
#else
  peak = ru.ru_maxrss / 1024.0;  // Linux: kilobytes
  current = peak;                // no cheap current-RSS here; report peak
#endif
}

const Vec3 kLight = {0.4, 0.55, 0.85};

Vec3 nrm(const Vec3& a) {
  double n = vnorm(a);
  return n > 1e-12 ? vscale(a, 1.0 / n) : a;
}

void addTube(Mesh& m, const std::vector<Vec3>& loop, double radius, bool closed, float br, float bg,
             float bb) {
  const int K = 10;  // sides
  const Vec3 light = nrm({0.4, 0.5, 0.85});
  const size_t N = loop.size();
  if (N < 3) return;
  std::vector<std::array<Vec3, K>> ring(N), nn(N);
  for (size_t i = 0; i < N; ++i) {
    const Vec3 tangent = nrm(vsub(loop[(i + 1) % N], loop[(i + N - 1) % N]));
    const Vec3 ref = (std::abs(tangent[2]) < 0.9) ? Vec3{0, 0, 1} : Vec3{1, 0, 0};
    const Vec3 n1 = nrm(vcross(tangent, ref));
    const Vec3 n2 = vcross(tangent, n1);
    for (int k = 0; k < K; ++k) {
      const double ang = 2.0 * M_PI * k / K;
      Vec3 rn = vadd(vscale(n1, std::cos(ang)), vscale(n2, std::sin(ang)));
      nn[i][k] = rn;
      ring[i][k] = vadd(loop[i], vscale(rn, radius));
    }
  }
  auto emit = [&](const Vec3& p, const Vec3& n) {
    float s = 0.35f + 0.65f * float(std::max(0.0, vdot(n, light)));
    m.vert(p, br * s, bg * s, bb * s);
  };
  const size_t segEnd = closed ? N : N - 1;  // open filaments skip the wrap segment
  for (size_t i = 0; i < segEnd; ++i) {
    const size_t j = (i + 1) % N;
    for (int k = 0; k < K; ++k) {
      const int k2 = (k + 1) % K;
      emit(ring[i][k], nn[i][k]); emit(ring[j][k], nn[j][k]); emit(ring[j][k2], nn[j][k2]);
      emit(ring[i][k], nn[i][k]); emit(ring[j][k2], nn[j][k2]); emit(ring[i][k2], nn[i][k2]);
    }
  }
}

void addCylinder(Mesh& m, const Vec3& c, double radius, double zlo, double zhi, float cr, float cg,
                 float cb) {
  const int K = 28;
  auto shade = [&](const Vec3& p, const Vec3& n) {
    float s = 0.4f + 0.6f * float(std::max(0.0, vdot(n, nrm(kLight))));
    m.vert(p, cr * s, cg * s, cb * s);
  };
  for (int k = 0; k < K; ++k) {
    const double a0 = 2 * M_PI * k / K, a1 = 2 * M_PI * (k + 1) / K;
    const Vec3 n0{std::cos(a0), std::sin(a0), 0}, n1{std::cos(a1), std::sin(a1), 0};
    const Vec3 blo0{c[0] + radius * n0[0], c[1] + radius * n0[1], zlo};
    const Vec3 bhi0{c[0] + radius * n0[0], c[1] + radius * n0[1], zhi};
    const Vec3 blo1{c[0] + radius * n1[0], c[1] + radius * n1[1], zlo};
    const Vec3 bhi1{c[0] + radius * n1[0], c[1] + radius * n1[1], zhi};
    shade(blo0, n0); shade(blo1, n1); shade(bhi1, n1);
    shade(blo0, n0); shade(bhi1, n1); shade(bhi0, n0);
  }
}

void addSphere(Mesh& m, const Vec3& c, double radius, float cr, float cg, float cb) {
  const int NLat = 16, NLon = 24;
  auto pt = [&](int a, int b) {
    double th = M_PI * a / NLat, ph = 2 * M_PI * b / NLon;
    return Vec3{std::sin(th) * std::cos(ph), std::sin(th) * std::sin(ph), std::cos(th)};
  };
  auto shade = [&](const Vec3& n) {
    float s = 0.4f + 0.6f * float(std::max(0.0, vdot(n, nrm(kLight))));
    m.vert(vadd(c, vscale(n, radius)), cr * s, cg * s, cb * s);
  };
  for (int a = 0; a < NLat; ++a)
    for (int b = 0; b < NLon; ++b) {
      Vec3 n00 = pt(a, b), n01 = pt(a, b + 1), n10 = pt(a + 1, b), n11 = pt(a + 1, b + 1);
      shade(n00); shade(n10); shade(n11);
      shade(n00); shade(n11); shade(n01);
    }
}

void addOrientedBox(Mesh& m, const Vec3& c, const Vec3& ex, const Vec3& ey, const Vec3& ez,
                    double hx, double hy, double hz, float cr, float cg, float cb) {
  auto shade = [&](const Vec3& p, const Vec3& n) {
    float s = 0.4f + 0.6f * float(std::max(0.0, vdot(nrm(n), nrm(kLight))));
    m.vert(p, cr * s, cg * s, cb * s);
  };
  auto corner = [&](double sx, double sy, double sz) {
    return Vec3{c[0] + sx * hx * ex[0] + sy * hy * ey[0] + sz * hz * ez[0],
                c[1] + sx * hx * ex[1] + sy * hy * ey[1] + sz * hz * ez[1],
                c[2] + sx * hx * ex[2] + sy * hy * ey[2] + sz * hz * ez[2]};
  };
  auto quad = [&](const Vec3& a, const Vec3& b, const Vec3& d, const Vec3& e, const Vec3& n) {
    shade(a, n); shade(b, n); shade(d, n);
    shade(a, n); shade(d, n); shade(e, n);
  };
  // Six faces, outward normals = +/- each local axis.
  quad(corner(+1, -1, -1), corner(+1, +1, -1), corner(+1, +1, +1), corner(+1, -1, +1), ex);
  quad(corner(-1, +1, -1), corner(-1, -1, -1), corner(-1, -1, +1), corner(-1, +1, +1), vscale(ex, -1));
  quad(corner(-1, +1, -1), corner(+1, +1, -1), corner(+1, +1, +1), corner(-1, +1, +1), ey);
  quad(corner(+1, -1, -1), corner(-1, -1, -1), corner(-1, -1, +1), corner(+1, -1, +1), vscale(ey, -1));
  quad(corner(-1, -1, +1), corner(+1, -1, +1), corner(+1, +1, +1), corner(-1, +1, +1), ez);
  quad(corner(+1, -1, -1), corner(-1, -1, -1), corner(-1, +1, -1), corner(+1, +1, -1), vscale(ez, -1));
}

GLuint compileShader(GLenum type, const char* src) {
  GLuint s = glCreateShader(type);
  glShaderSource(s, 1, &src, nullptr);
  glCompileShader(s);
  GLint ok = 0;
  glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
  if (!ok) {
    char log[1024];
    glGetShaderInfoLog(s, 1024, nullptr, log);
    std::fprintf(stderr, "shader compile error: %s\n", log);
  }
  return s;
}

GLuint makeFlatProgram(GLint& uMVP, GLint& uPointSize) {
  const char* vs = R"(#version 330 core
    layout(location=0) in vec3 aPos;
    layout(location=1) in vec3 aColor;
    uniform mat4 uMVP;
    uniform float uPointSize;
    out vec3 vColor;
    void main(){ gl_Position = uMVP * vec4(aPos,1.0); gl_PointSize = uPointSize; vColor = aColor; })";
  const char* fs = R"(#version 330 core
    in vec3 vColor; out vec4 FragColor;
    void main(){ FragColor = vec4(vColor,1.0); })";
  GLuint prog = glCreateProgram();
  GLuint v = compileShader(GL_VERTEX_SHADER, vs), f = compileShader(GL_FRAGMENT_SHADER, fs);
  glAttachShader(prog, v);
  glAttachShader(prog, f);
  glBindAttribLocation(prog, 0, "aPos");
  glBindAttribLocation(prog, 1, "aColor");
  glLinkProgram(prog);
  glDeleteShader(v);
  glDeleteShader(f);
  uMVP = glGetUniformLocation(prog, "uMVP");
  uPointSize = glGetUniformLocation(prog, "uPointSize");
  return prog;
}

}  // namespace viewer
