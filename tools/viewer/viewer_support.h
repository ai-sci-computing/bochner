#pragma once
/// \file
/// Shared support for the interactive viewers (bochner_viewer and
/// bochner_obstacle_viewer): the interleaved vertex batch, the orbit camera, the
/// flat vertex-color shader program, CPU-shaded mesh primitives, and the RSS
/// readout. Both apps use these so nothing is duplicated between the two mains.
#include <array>
#include <vector>

#ifdef __APPLE__
#include <OpenGL/gl3.h>
#endif

#include "glmath.h"
#include "grid/Vec3.h"

namespace viewer {

/// Process memory in MB, for the HUD. \p current = resident set size right now;
/// \p peak = the high-water mark that never comes back down (getrusage ru_maxrss).
void memoryMB(double& current, double& peak);

/// Interleaved pos(3)+color(3) vertex batch (lines or triangles).
struct Mesh {
  std::vector<float> v;
  void vert(const bochner::Vec3& p, float r, float g, float b) {
    v.insert(v.end(), {(float)p[0], (float)p[1], (float)p[2], r, g, b});
  }
  void seg(const bochner::Vec3& a, const bochner::Vec3& b, float r, float g, float bl) {
    vert(a, r, g, bl);
    vert(b, r, g, bl);
  }
  GLsizei count() const { return GLsizei(v.size() / 6); }
};

/// Orbit camera: az orbits the vertical axis, el tilts above the floor, dist is
/// the eye distance. Defaults differ per app, so construct with brace-init.
struct Camera {
  float az, el, dist;
};

/// Fixed directional light used by the CPU Lambert shading below.
extern const bochner::Vec3 kLight;
/// Normalize \p a (returns it unchanged if ~zero length).
bochner::Vec3 nrm(const bochner::Vec3& a);

/// Append a shaded tube of \p radius around polyline \p loop. \p closed wraps the
/// tube end-to-end (a ring); an open filament skips the wrap segment.
void addTube(Mesh& m, const std::vector<bochner::Vec3>& loop, double radius, bool closed, float br,
             float bg, float bb);
/// Shaded solid cylinder along +z from \p zlo to \p zhi.
void addCylinder(Mesh& m, const bochner::Vec3& c, double radius, double zlo, double zhi, float cr,
                 float cg, float cb);
/// Shaded solid sphere.
void addSphere(Mesh& m, const bochner::Vec3& c, double radius, float cr, float cg, float cb);
/// Shaded finite oriented box: half-extents (hx,hy,hz) along local axes (ex,ey,ez).
void addOrientedBox(Mesh& m, const bochner::Vec3& c, const bochner::Vec3& ex, const bochner::Vec3& ey,
                    const bochner::Vec3& ez, double hx, double hy, double hz, float cr, float cg,
                    float cb);

/// Compile a shader stage, logging any error to stderr.
GLuint compileShader(GLenum type, const char* src);
/// Build the viewer's flat vertex-color program (MVP transform, per-vertex color,
/// gl_PointSize for GL_POINTS). Returns the linked program and outputs the uMVP
/// and uPointSize uniform locations.
GLuint makeFlatProgram(GLint& uMVP, GLint& uPointSize);

}  // namespace viewer
