/// \file
/// Interactive viewer for FLOW PAST AN OBSTACLE (the vortex-shedding demo, in
/// progress). A constant +x inflow (inhomogeneous-Neumann inlet) enters a
/// horizontal tank, an obstacle -- cylinder or sphere -- sits in the stream, and
/// the pressure projection routes the flow around it. This is the sibling of the
/// covector-fluids viewer, stripped of the example vortex ICs and given an
/// obstacle to choose instead. Vorticity generation (viscosity) and the vortex
/// street come in the next phase; for now it shows the steady flow-around.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <vector>

#include <complex>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#ifdef __APPLE__
#include <OpenGL/gl3.h>
#endif

#ifdef BOCHNER_WITH_PETSC
#include <slepc.h>
#endif

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include "solvers/EigenSolver.h"
#include "solvers/GaugeEigen.h"
#include "solvers/GaugeMultigrid.h"
#include "fluid/MacAdvection.h"
#include "fluid/MacExtrapolate.h"
#ifdef BOCHNER_WITH_METAL
#include "gpu/MetalContext.h"
#endif
#include "extraction/MacConnectionLaplacian.h"
#include "extraction/MacFilaments.h"
#include "grid/MacGrid.h"
#include "grid/MacObstacle.h"
#include "fluid/MacProjection.h"
#include "glmath.h"
#include "viewer_support.h"

using namespace bochner;
using namespace viewer;

// y-up orbit camera (long x-axis of the tank shows HORIZONTAL, flow left->right).
Camera g_cam{0.35f, 0.32f, 9.0f};
double g_lastX = 0, g_lastY = 0;
bool g_dragging = false;

void scrollCallback(GLFWwindow*, double, double dy) {
  if (ImGui::GetCurrentContext() && ImGui::GetIO().WantCaptureMouse) return;
  g_cam.dist *= std::pow(0.9f, static_cast<float>(dy));
  g_cam.dist = std::clamp(g_cam.dist, 2.0f, 40.0f);
}

// ---------------------------------------------------------------------------
// Flow-past-obstacle simulation state.
// ---------------------------------------------------------------------------
struct Sim {
  enum Obstacle { CYLINDER, SPHERE, SQUARE, PLATE };
  int obstacle = CYLINDER;
  int res = 32;          // cells across the tank height Ly (master resolution)
  float U = 1.0f;        // constant inflow speed (+x)
  float radius = 0.35f;  // obstacle radius (cylinder / sphere)
  // Oriented finite box (drives both SQUARE and PLATE). Half-extents along the
  // box's local axes: thickness (streamwise), width (cross-stream), span. The
  // span is finite -> the body need not touch the z-walls (a 3D wake). yaw/tilt
  // orient it; an inclined box sheds naturally with no symmetry trigger.
  float boxThick = 0.30f;  // local-x half-extent (streamwise thickness/depth)
  float boxWidth = 0.30f;  // local-y half-extent (cross-stream width)
  float boxSpan = 0.35f;   // local-z half-extent (finite span; < Lz/2 leaves wall gaps)
  float yawDeg = 20.0f;    // rotation about +z (in-plane angle of attack)
  float tiltDeg = 0.0f;    // rotation of the span axis off +z (out-of-plane tilt)

  // Local orthonormal frame (ex,ey,ez) of the oriented box: world axes yawed
  // about +z, then tilted about the yawed +y. Shared by the mask and the render.
  void boxFrame(Vec3& ex, Vec3& ey, Vec3& ez) const {
    const double a = yawDeg * M_PI / 180.0, b = tiltDeg * M_PI / 180.0;
    const Vec3 x0{std::cos(a), std::sin(a), 0}, y0{-std::sin(a), std::cos(a), 0}, z0{0, 0, 1};
    ex = vadd(vscale(x0, std::cos(b)), vscale(z0, std::sin(b)));
    ez = vadd(vscale(x0, -std::sin(b)), vscale(z0, std::cos(b)));
    ey = y0;
  }

  // Characteristic transverse length D (for the Reynolds number and the HUD):
  // the diameter for cylinder/sphere, the cross-stream width for the box/plate.
  double charDim() const {
    return (obstacle == SQUARE || obstacle == PLATE) ? 2.0 * boxWidth : 2.0 * radius;
  }
  float nu = 0.008f;     // kinematic viscosity: the wake sink + no-slip source (0 = inviscid)
  bool cflAdaptive = true;
  float cflTarget = 0.6f;
  float dt = 0.02f;

  // Vortex-filament extraction (the "smoke rings"): the connection-Laplacian
  // eigensolver traces the shed vortices to filament loops. hbar = flux quantum
  // per traced tube; smaller hbar -> more/thinner filaments.
  bool extractOn = true;
  // The GPU covMG-LOBPCG is fast enough to extract EVERY frame (measured); the CPU/Lanczos
  // solvers want a larger cadence (raise this after switching to them).
  int extractEvery = 1;
  // Which eigensolver traces the filaments. All three solve the SAME connection
  // Laplacian (a clean A/B): gauge-MG on the CPU (double, authoritative), the same
  // covMG-LOBPCG on the GPU (Metal, float -- viewer-speed), or SLEPc Lanczos.
  enum ExtractSolver { EXTRACT_GAUGE_CPU = 0, EXTRACT_GAUGE_GPU = 1, EXTRACT_LANCZOS = 2 };
#ifdef BOCHNER_WITH_METAL
  int extractSolver = EXTRACT_GAUGE_GPU;  // default: the fast GPU path when Metal is built
#else
  int extractSolver = EXTRACT_GAUGE_CPU;
#endif
#ifdef BOCHNER_WITH_METAL
  bool useGpuAdvect = true;   // Metal GPU BFECC advection (vs the OpenMP CPU path)
  bool useGpuProject = true;  // Metal GPU MGPCG projection (float) vs the CPU double solve
#endif
  float hbar = 0.1f;        // flux quantum; ~0.1 gives a clean set of shed tubes (measured)
  float filamentSmooth = 0.5f;  // temporal EMA on the eigenvector (0 = off) -- steadies flicker
  double extractMs = 0.0;

  // Fixed physical tank, centered at origin, long in x (the flow direction).
  static constexpr float Lx = 6.0f, Ly = 2.0f, Lz = 1.2f;

  int frame = 0, nx = 0, ny = 0, nz = 0, solidCells = 0, lastCycles = 0;
  double h = 0, dtEff = 0.02, curCfl = 0.0, simMs = 0.0;
  bool diverged = false;  // set when the field goes non-finite; halts stepping/extraction

  std::unique_ptr<MacGrid> grid;
  std::unique_ptr<MacProjector> proj;
  FaceField u;
  SolidMask solid;
  Vec3 center{-1.5, 0, 0};
  std::vector<std::vector<Vec3>> loops;         // extracted filament polylines
  std::vector<char> loopClosed;                 // per filament: closed ring vs open spanwise tube
  std::vector<std::complex<double>> prevVec;    // gauge-MG eigensolver warm start
  std::vector<std::complex<double>> smoothVec;  // temporally-smoothed eigenvector (EMA state)

  // Passive white "smoke": each frame we release emitPerFrame tracer points,
  // randomly scattered across the inlet; they drift with the flow (RK2) and are
  // culled when they leave the domain, enter the solid, or age out. Cheap -- just
  // velocity samples, no solve. The steady count ~ emitPerFrame * crossing-time.
  int emitPerFrame = 120;
  std::vector<Vec3> pPos;
  std::vector<int> pAge;
  std::uint32_t rng = 2463534242u;
  float randf() {  // xorshift PRNG in [0,1); deterministic (no <random> / Date)
    rng ^= rng << 13;
    rng ^= rng >> 17;
    rng ^= rng << 5;
    return (rng & 0xffffffu) / float(0x1000000);
  }
  Vec3 inletPoint() {  // a random point on the x-lo inlet face
    const float hy = Ly / 2, hz = Lz / 2, hx = Lx / 2;
    return Vec3{-hx + 0.02, (2 * randf() - 1) * 0.98 * hy, (2 * randf() - 1) * 0.94 * hz};
  }
  void clearParticles() {
    pPos.clear();
    pAge.clear();
  }
  bool inSolidOrOut(const Vec3& p) const {
    const float hx = Lx / 2, hy = Ly / 2, hz = Lz / 2;
    if (p[0] > hx || p[0] < -hx || std::abs(p[1]) > hy || std::abs(p[2]) > hz) return true;
    const int i = std::clamp(int((p[0] - grid->origin()[0]) / h), 0, grid->nx() - 1);
    const int j = std::clamp(int((p[1] - grid->origin()[1]) / h), 0, grid->ny() - 1);
    const int k = std::clamp(int((p[2] - grid->origin()[2]) / h), 0, grid->nz() - 1);
    return isSolid(solid, grid->cellIndex(i, j, k));
  }
  void advectParticles(double dt) {
    // Advect (RK2) + cull dead in one compaction pass, then emit at the inlet.
    size_t w = 0;
    for (size_t r = 0; r < pPos.size(); ++r) {
      const Vec3 k1 = sampleVelocity(*grid, u, pPos[r]);
      const Vec3 k2 = sampleVelocity(*grid, u, vadd(pPos[r], vscale(k1, 0.5 * dt)));
      const Vec3 np = vadd(pPos[r], vscale(k2, dt));
      if (pAge[r] < 3000 && !inSolidOrOut(np)) {  // survives
        pPos[w] = np;
        pAge[w] = pAge[r] + 1;
        ++w;
      }
    }
    pPos.resize(w);
    pAge.resize(w);
    for (int e = 0; e < emitPerFrame && pPos.size() < 400000; ++e) {  // release new smoke
      pPos.push_back(inletPoint());
      pAge.push_back(0);
    }
  }

  static double nowMs() {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
  }

  void reseed() {
    h = Ly / res;
    nx = std::max(12, int(std::lround(Lx / h)));
    ny = res;
    nz = std::max(4, int(std::lround(Lz / h)));
    grid = std::make_unique<MacGrid>(nx, ny, nz, h, Vec3{-Lx / 2, -Ly / 2, -Lz / 2});
    // Upstream third. The sub-cell y-offset that used to be applied here broke
    // the exact top/bottom symmetry to seed the von Karman instability. It is
    // no longer needed -- the cylinder sheds 20-24 filaments centred exactly,
    // measured over 500 frames at res=32 -- and it was actively wrong for the
    // SPHERE, whose wake at this Reynolds number (~87, onset ~270) is steady
    // and axisymmetric: the offset manufactured a top/bottom asymmetry the
    // physics does not have.
    center = Vec3{-Lx * 0.25, 0.0, 0.0};
    switch (obstacle) {
      case CYLINDER:
        solid = cylinderMask(*grid, center, {0, 0, 1}, radius);
        break;
      case SPHERE:
        solid = sphereMask(*grid, center, radius);
        break;
      case SQUARE:
      case PLATE: {
        // Finite oriented box: square = square section + long span; plate = one
        // thin extent. Both use the same frame + half-extents (differ only in the
        // slider presets set when the shape is chosen).
        Vec3 ex, ey, ez;
        boxFrame(ex, ey, ez);
        solid = boxMask(*grid, center, ex, ey, ez, Vec3{boxThick, boxWidth, boxSpan});
        break;
      }
    }
    solidCells = 0;
    for (auto s : solid) solidCells += s;
    proj = std::make_unique<MacProjector>(*grid, PoissonMgOptions{}, BoundarySpec::channelFlow(U),
                                          solid);
    u = ops::zeroFaceField(*grid);
    for (double& v : u.x) v = U;
    u = proj->project(u, &lastCycles);
    frame = 0;
    diverged = false;
    loops.clear();
    loopClosed.clear();
    prevVec.clear();
    smoothVec.clear();
    clearParticles();
    g_cam.dist = std::clamp(1.3f * Lx, 4.0f, 40.0f);
  }

  double maxSpeed() const {
    const MacGrid& g = *grid;
    double m = 0.0;
    for (int i = 0; i < g.nx(); ++i)
      for (int j = 0; j < g.ny(); ++j)
        for (int k = 0; k < g.nz(); ++k) {
          const double ux = 0.5 * (u.x[g.faceXIndex(i, j, k)] + u.x[g.faceXIndex(i + 1, j, k)]);
          const double uy = 0.5 * (u.y[g.faceYIndex(i, j, k)] + u.y[g.faceYIndex(i, j + 1, k)]);
          const double uz = 0.5 * (u.z[g.faceZIndex(i, j, k)] + u.z[g.faceZIndex(i, j, k + 1)]);
          const double s = std::sqrt(ux * ux + uy * uy + uz * uz);
          if (!std::isfinite(s)) return s;  // propagate NaN/Inf so step() can halt on blow-up
          m = std::max(m, s);
        }
    return m;
  }

  void step() {
    const double t0 = nowMs();
    const double rawMax = maxSpeed();
    if (!std::isfinite(rawMax)) {  // field blew up: halt rather than step/extract NaN silently
      diverged = true;
      simMs = nowMs() - t0;
      return;
    }
    const double Umax = std::max(1e-6, rawMax);
    dtEff = cflAdaptive ? std::min(0.1, double(cflTarget) * h / Umax) : double(dt);
    curCfl = Umax * dtEff / h;
    // Give the solid band a plausible velocity before backtracing into it
    // (Bridson sec. 6). Without it, faces one layer outside the body sample the
    // identically-zero interior every step. Measured cost is nil and, on this
    // demo, so is the measured effect -- but the projection's zeroing is a
    // pressure-solve boundary condition, not a statement about where the fluid
    // was, so advecting against it is wrong regardless.
    if (solid.size() == static_cast<std::size_t>(grid->numCells()))
      extrapolateIntoSolid(*grid, u, solid, static_cast<int>(std::ceil(cflTarget)) + 2);

    // Covector advection -> viscous diffusion (no-slip vorticity source + wake
    // sink) -> project around the obstacle. The projector re-imposes the inlet
    // and the no-penetration solid each call.
    FaceField w;
#ifdef BOCHNER_WITH_METAL
    if (useGpuAdvect && gpu::metalAvailable())
      w = advectCovectorBFECCGpu(*grid, u, u, dtEff);
    else
#endif
      w = advectCovectorBFECC(*grid, u, u, dtEff);
    if (nu > 0.0f) {
      // Sub-step the explicit diffusion so nu*dt/h^2 stays under ~1/6 (stable).
      const double lim = 0.16 * h * h;
      const int nsub = std::max(1, int(std::ceil(nu * dtEff / lim)));
      for (int s = 0; s < nsub; ++s) w = ops::diffuseVelocity(*grid, w, nu, dtEff / nsub, solid);
    }
#ifdef BOCHNER_WITH_METAL
    proj->setGpuProjection(useGpuProject);
#endif
    u = proj->project(w, &lastCycles);
    advectParticles(dtEff);
    simMs = nowMs() - t0;
    ++frame;
    if (extractOn && frame % extractEvery == 0) extract();
  }

  // Weissmann-Pinkall extraction: the smallest-eigenvalue section of the
  // connection Laplacian, traced to its zero set = the vortex filaments ("smoke
  // rings"). theta = u.h/hbar is the U(1) connection; the two eigensolvers share
  // it, so the traced filament is solver-independent (a clean A/B).
  // HUD label for the effective extraction solver (reflects GPU fallback).
  const char* extractLabel() const {
    if (extractSolver == EXTRACT_LANCZOS) return "SLEPc Lanczos";
#ifdef BOCHNER_WITH_METAL
    if (extractSolver == EXTRACT_GAUGE_GPU)
      return gpu::metalAvailable() ? "gauge-MG (GPU Metal, float)" : "gauge-MG (CPU; no GPU)";
#endif
    return "gauge-MG (CPU double)";
  }

  // Phase-aligned EMA of the eigenvector for temporal coherence; returns the
  // interleaved [Re,Im] section to trace. Only the DISPLAYED vector is smoothed --
  // the eigensolver is warm-started from the raw vector (feeding the lagged smooth
  // vector back adds jitter; measured). The blend must phase-align first: the
  // eigenvector is defined only up to a global phase (the solver returns an
  // arbitrary one each frame), so misaligned representatives would cancel -- but
  // the alignment changes no line (the zero set / filaments are phase-invariant).
  std::vector<double> smoothEigenvector(const std::vector<std::complex<double>>& vec) {
    const double s = std::clamp(filamentSmooth, 0.0f, 0.95f);
    if (s > 0.0 && smoothVec.size() == vec.size() && !vec.empty()) {
      std::complex<double> ip(0.0, 0.0);  // <smoothVec, vec>, whose phase aligns vec to smoothVec
      for (size_t i = 0; i < vec.size(); ++i) ip += std::conj(smoothVec[i]) * vec[i];
      const double mag = std::abs(ip);
      const std::complex<double> cph = mag > 1e-30 ? std::conj(ip / mag) : std::complex<double>(1, 0);
      const double a = 1.0 - s;  // weight of the (phase-aligned) new vector
      double nrm = 0.0;
      for (size_t i = 0; i < vec.size(); ++i) {
        smoothVec[i] = (1.0 - a) * smoothVec[i] + a * (cph * vec[i]);
        nrm += std::norm(smoothVec[i]);
      }
      nrm = std::sqrt(nrm);
      if (nrm > 0.0)
        for (auto& z : smoothVec) z /= nrm;
    } else {
      smoothVec = vec;
    }
    return toInterleaved(smoothVec);
  }

  void extract() {
    const double t0 = nowMs();
    const FaceField theta = connectionAngles(*grid, u, hbar);
    std::vector<double> psi;
    bool gpuExtract = false;
#ifdef BOCHNER_WITH_METAL
    gpuExtract = (extractSolver == EXTRACT_GAUGE_GPU) && gpu::metalAvailable();
#endif
    std::vector<std::complex<double>> vec;  // this frame's raw eigenvector (complex)
#ifdef BOCHNER_WITH_PETSC
    const bool useLanczos = (extractSolver == EXTRACT_LANCZOS);
#else
    // No SLEPc in this build: the Lanczos baseline is unavailable and the
    // combo entry is hidden below, so this branch is never taken.
    const bool useLanczos = false;
#endif
    if (useLanczos) {
#ifdef BOCHNER_WITH_PETSC
      const CooMatrix E = connectionLaplacian(*grid, theta);
      vec = toComplex(smallestEigenpairLanczos(E, 1e-6).vector);
#endif
    } else if (gpuExtract) {
#ifdef BOCHNER_WITH_METAL
      // Same covMG-LOBPCG as the CPU path, on the GPU in float (viewer speed). Rebuild +
      // upload the hierarchy each extract -- the connection (theta = u.h/hbar)
      // changes every frame, so its transports do too. Warm-started from the
      // previous (smoothed) filament eigenvector, like the CPU path.
      const GaugeLattice lat = gaugeLatticeFromFaces(*grid, theta);
      std::vector<gpu::GaugeLevelData> levels;
      for (const GaugeLattice& L : buildGaugeLevels(lat))
        levels.push_back({L.lx, L.ly, L.lz, L.periodic, L.w, L.tx, L.ty, L.tz});
      const int handle = gpu::uploadGauge(levels);
      GaugeEigenOptions opts;
      opts.relativeGsDrop = false;  // live path: keep the absolute-drop warm-start early-exit
      // lobpcgSolveGauge stops on Rayleigh-quotient stagnation (the float residual
      // floors out above tol at large sizes), so a warm start converges in far
      // fewer iterations than a cold one; maxIters is just a safety cap.
      vec = gpu::lobpcgSolveGauge(handle, prevVec, /*maxIters=*/100, /*tol=*/1e-4, opts.precCycles,
                                opts.mg.nu1, opts.mg.nu2, opts.mg.coarseSweeps, opts.mg.omega)
                .vector;
      gpu::freeGauge(handle);
#endif
    } else {  // EXTRACT_GAUGE_CPU (also the fallback when the GPU is unavailable)
      const GaugeLattice lat = gaugeLatticeFromFaces(*grid, theta);
      GaugeEigenOptions opts;
      opts.relativeGsDrop = false;  // live path: keep the absolute-drop warm-start early-exit
      opts.tol = 1e-6;
      vec = smallestEigenpairGaugeMG(lat, prevVec.empty() ? nullptr : &prevVec, opts).vector;
    }
    // Warm-start the next solve from the RAW vector; smooth only what is drawn.
    if (!vec.empty()) prevVec = vec;
    // Temporal smoothing: the shed vortices are stable but the weakest near-
    // threshold filaments flicker frame to frame; a phase-aligned EMA on the
    // eigenvector low-passes that. filamentSmooth in [0,0.95]: 0 = off, higher =
    // steadier (fewer filament pops) but laggier.
    psi = smoothEigenvector(vec);
    loops.clear();
    loopClosed.clear();
    // Keep OPEN filaments too: the von Karman shed vortices are spanwise tubes
    // (open, ending at the z-walls), not closed rings.
    for (auto& fl : linkFilaments(*grid, traceZeroSet(*grid, psi)))
      if (fl.points.size() >= 4) {
        const bool c = fl.closed;
        loops.push_back(std::move(fl.points));
        loopClosed.push_back(c);
      }
    extractMs = nowMs() - t0;
  }
};

// omega_z on the mid-z x-y plane (the cylinder's shedding plane): each shed
// vortex is a +/- blob. Flat quads, red +, blue -, auto-normalized * gain.
static void addVortSliceXY(Mesh& m, const MacGrid& g, const FaceField& u, float gain) {
  const int k = g.nz() / 2, nx = g.nx(), ny = g.ny();
  const double inv2h = 1.0 / (2.0 * g.spacing());
  auto cx = [&](int i, int j) {
    return 0.5 * (u.x[g.faceXIndex(i, j, k)] + u.x[g.faceXIndex(i + 1, j, k)]);
  };
  auto cy = [&](int i, int j) {
    return 0.5 * (u.y[g.faceYIndex(i, j, k)] + u.y[g.faceYIndex(i, j + 1, k)]);
  };
  std::vector<double> w(nx * ny, 0.0);
  double peak = 1e-12;
  for (int i = 1; i < nx - 1; ++i)
    for (int j = 1; j < ny - 1; ++j) {
      const double wz = (cy(i + 1, j) - cy(i - 1, j)) * inv2h - (cx(i, j + 1) - cx(i, j - 1)) * inv2h;
      w[i * ny + j] = wz;
      peak = std::max(peak, std::abs(wz));
    }
  const double z = g.cellCenter(0, 0, k)[2];
  const double hh = 0.5 * g.spacing();
  for (int i = 1; i < nx - 1; ++i)
    for (int j = 1; j < ny - 1; ++j) {
      const float t = std::clamp(float(gain * w[i * ny + j] / peak), -1.0f, 1.0f);
      const float r = 0.08f + 0.92f * std::max(0.0f, t);
      const float b = 0.08f + 0.92f * std::max(0.0f, -t);
      const Vec3 c = g.cellCenter(i, j, k);
      const Vec3 p00{c[0] - hh, c[1] - hh, z}, p10{c[0] + hh, c[1] - hh, z};
      const Vec3 p11{c[0] + hh, c[1] + hh, z}, p01{c[0] - hh, c[1] + hh, z};
      m.vert(p00, r, 0.08f, b); m.vert(p10, r, 0.08f, b); m.vert(p11, r, 0.08f, b);
      m.vert(p00, r, 0.08f, b); m.vert(p11, r, 0.08f, b); m.vert(p01, r, 0.08f, b);
    }
}

// Cell-centered vorticity magnitude |curl u| over the whole grid (0 on the one-cell
// boundary shell), for coloring the smoke tracers by local swirl strength. Two cheap
// O(cells) passes: average the MAC faces to cell centers, then central-difference.
static std::vector<float> vorticityMagField(const MacGrid& g, const FaceField& u) {
  const int nx = g.nx(), ny = g.ny(), nz = g.nz();
  const int N = nx * ny * nz;
  std::vector<float> ux(N, 0.0f), uy(N, 0.0f), uz(N, 0.0f);
  for (int i = 0; i < nx; ++i)
    for (int j = 0; j < ny; ++j)
      for (int k = 0; k < nz; ++k) {
        const int c = g.cellIndex(i, j, k);
        ux[c] = 0.5f * float(u.x[g.faceXIndex(i, j, k)] + u.x[g.faceXIndex(i + 1, j, k)]);
        uy[c] = 0.5f * float(u.y[g.faceYIndex(i, j, k)] + u.y[g.faceYIndex(i, j + 1, k)]);
        uz[c] = 0.5f * float(u.z[g.faceZIndex(i, j, k)] + u.z[g.faceZIndex(i, j, k + 1)]);
      }
  std::vector<float> mag(N, 0.0f);
  const float inv2h = float(1.0 / (2.0 * g.spacing()));
  auto C = [&](int i, int j, int k) { return g.cellIndex(i, j, k); };
  for (int i = 1; i < nx - 1; ++i)
    for (int j = 1; j < ny - 1; ++j)
      for (int k = 1; k < nz - 1; ++k) {
        const float wx = (uz[C(i, j + 1, k)] - uz[C(i, j - 1, k)]) * inv2h -
                         (uy[C(i, j, k + 1)] - uy[C(i, j, k - 1)]) * inv2h;
        const float wy = (ux[C(i, j, k + 1)] - ux[C(i, j, k - 1)]) * inv2h -
                         (uz[C(i + 1, j, k)] - uz[C(i - 1, j, k)]) * inv2h;
        const float wz = (uy[C(i + 1, j, k)] - uy[C(i - 1, j, k)]) * inv2h -
                         (ux[C(i, j + 1, k)] - ux[C(i, j - 1, k)]) * inv2h;
        mag[C(i, j, k)] = std::sqrt(wx * wx + wy * wy + wz * wz);
      }
  return mag;
}

int main(int argc, char** argv) {
#ifdef BOCHNER_WITH_PETSC
  SlepcInitialize(&argc, &argv, nullptr, nullptr);  // for the Lanczos extraction path
#endif
  if (!glfwInit()) {
    std::fprintf(stderr, "glfwInit failed\n");
    return 1;
  }
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
  GLFWwindow* win = glfwCreateWindow(1280, 800, "Bochner - Flow past an obstacle", nullptr, nullptr);
  if (!win) {
    std::fprintf(stderr, "window creation failed\n");
    glfwTerminate();
    return 1;
  }
  glfwMakeContextCurrent(win);
  glfwSwapInterval(1);
  glfwSetScrollCallback(win, scrollCallback);

  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGui::StyleColorsDark();
  ImGui_ImplGlfw_InitForOpenGL(win, true);
  ImGui_ImplOpenGL3_Init("#version 330");

  // Flat vertex-color program (shared with the covector-fluids viewer); the
  // point-size uniform sizes the GL_POINTS smoke tracers.
  GLint uMVP = -1, uPointSize = -1;
  GLuint prog = makeFlatProgram(uMVP, uPointSize);

  GLuint vao, vbo;
  glGenVertexArrays(1, &vao);
  glGenBuffers(1, &vbo);
  glBindVertexArray(vao);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
  glEnableVertexAttribArray(1);
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_PROGRAM_POINT_SIZE);  // let the vertex shader size the smoke points

  Sim sim;
  sim.reseed();

  bool playing = false, showObstacle = true, showGlyphs = false, showBox = true, showSlice = false,
       showFilament = true, showParticles = true, colorByVorticity = true;
  int glyphRes = 10;
  float glyphScale = 0.12f, sliceGain = 1.0f, tubeRadius = 0.02f, pointSize = 1.5f;
  float vortColorGain = 1.0f;   // scales the vorticity-to-color mapping
  float vortPeakEma = 0.0f;     // smoothed frame peak |omega| so the coloring doesn't flicker

  while (!glfwWindowShouldClose(win)) {
    glfwPollEvents();
    ImGuiIO& io = ImGui::GetIO();
    double mx, my;
    glfwGetCursorPos(win, &mx, &my);
    if (glfwGetMouseButton(win, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS && !io.WantCaptureMouse) {
      if (g_dragging) {
        g_cam.az -= 0.01f * float(mx - g_lastX);
        g_cam.el = std::clamp(g_cam.el + 0.01f * float(my - g_lastY), -1.5f, 1.5f);
      }
      g_dragging = true;
    } else {
      g_dragging = false;
    }
    g_lastX = mx;
    g_lastY = my;

    if (playing) {
      sim.step();
      if (sim.diverged) playing = false;  // stop auto-stepping a blown-up field
    }

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    ImGui::Begin("Flow past an obstacle");
    ImGui::Text("frame %d   grid %dx%dx%d = %d cells   filaments %zu", sim.frame, sim.nx, sim.ny,
                sim.nz, sim.nx * sim.ny * sim.nz, sim.loops.size());
    if (sim.diverged)
      ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f),
                         "DIVERGED -- sim halted. Lower dt / CFL target or raise viscosity, then "
                         "Reseed.");
    ImGui::Text("h=%.4f | size %.1f cells | %d solid cells | CG its %d", sim.h,
                sim.charDim() / sim.h, sim.solidCells, sim.lastCycles);
    ImGui::Text("sim %.1f ms (%.1f fps)  |  dt %.4f  CFL %.2f", sim.simMs,
                sim.simMs > 0 ? 1000.0 / sim.simMs : 0.0, sim.dtEff, sim.curCfl);
#ifdef BOCHNER_WITH_METAL
    ImGui::Text("advection: %s   |   projection: %s",
                (sim.useGpuAdvect && gpu::metalAvailable()) ? "GPU" : "CPU",
                (sim.useGpuProject && gpu::metalAvailable()) ? "GPU (MGPCG)" : "CPU");
#endif
    ImGui::Text("extract %.1f ms (every %d frames, %s)", sim.extractMs, sim.extractEvery,
                sim.extractLabel());
    {
      double cur = 0, pk = 0;
      memoryMB(cur, pk);
      ImGui::Text("mem %.0f MB (peak %.0f MB)", cur, pk);
    }
    if (ImGui::Button(playing ? "Pause" : "Play")) playing = !playing;
    ImGui::SameLine();
    if (ImGui::Button("Step")) sim.step();
    ImGui::SameLine();
    if (ImGui::Button("Reset")) sim.reseed();

    ImGui::SeparatorText("Obstacle");
    static const char* kNames[] = {"Cylinder (spanwise)", "Sphere", "Box / square cylinder",
                                   "Inclined plate"};
    if (ImGui::Combo("shape", &sim.obstacle, kNames, IM_ARRAYSIZE(kNames))) {
      // Preset the box dimensions per shape (a square section vs a thin plate).
      if (sim.obstacle == Sim::SQUARE) {
        sim.boxThick = 0.30f;
        sim.boxWidth = 0.30f;
      } else if (sim.obstacle == Sim::PLATE) {
        sim.boxThick = 0.05f;
        sim.boxWidth = 0.40f;
      }
      sim.reseed();
    }
    if (sim.obstacle == Sim::SQUARE || sim.obstacle == Sim::PLATE) {
      bool ch = false;
      ch |= ImGui::SliderFloat("thickness (streamwise)", &sim.boxThick, 0.03f, 0.7f, "%.3f");
      ch |= ImGui::SliderFloat("width (cross-stream)", &sim.boxWidth, 0.05f, 0.9f, "%.3f");
      ch |= ImGui::SliderFloat("span (half-length)", &sim.boxSpan, 0.05f, 0.6f, "%.3f");
      ch |= ImGui::SliderFloat("yaw (deg)", &sim.yawDeg, -80.0f, 80.0f, "%.0f");
      ch |= ImGui::SliderFloat("tilt (deg)", &sim.tiltDeg, -60.0f, 60.0f, "%.0f");
      if (ch) sim.reseed();
      ImGui::TextDisabled("(finite: shrink span below %.2f to clear the z-walls)", Sim::Lz / 2);
    } else if (ImGui::SliderFloat("radius", &sim.radius, 0.15f, 0.7f)) {
      sim.reseed();
    }
    if (ImGui::SliderFloat("inflow U", &sim.U, 0.2f, 3.0f)) sim.reseed();
    // Snapped to multiples of 8 -- see the grid-n note in main.cpp: the
    // pressure hierarchy collapses on odd (or barely even) extents.
    if (ImGui::SliderInt("resolution (cells across height)", &sim.res, 16, 80, "%d",
                         ImGuiSliderFlags_AlwaysClamp)) {
      sim.res = ((sim.res + 4) / 8) * 8;
      sim.reseed();
    }
    ImGui::TextDisabled("(constant +x inflow, open outlet, free-slip walls; changing rebuilds)");
    ImGui::SliderFloat("viscosity nu", &sim.nu, 0.0f, 0.04f, "%.4f");
    if (sim.nu > 0.0f)
      ImGui::Text("Reynolds  U*D/nu = %.0f   (von Karman ~50-300)", sim.U * sim.charDim() / sim.nu);
    else
      ImGui::TextDisabled("inviscid (nu=0): no sink -> wake vorticity grows unbounded");

    ImGui::SeparatorText("Step");
#ifdef BOCHNER_WITH_METAL
    if (gpu::metalAvailable()) {
      ImGui::Checkbox("GPU advection (Metal)", &sim.useGpuAdvect);
      ImGui::Checkbox("GPU projection (Metal MGPCG)", &sim.useGpuProject);
    } else {
      ImGui::TextDisabled("GPU compute: no Metal device");
    }
#endif
    ImGui::Checkbox("CFL-adaptive dt", &sim.cflAdaptive);
    ImGui::BeginDisabled(!sim.cflAdaptive);
    ImGui::SliderFloat("target CFL", &sim.cflTarget, 0.1f, 1.5f, "%.2f");
    ImGui::EndDisabled();
    ImGui::BeginDisabled(sim.cflAdaptive);
    ImGui::SliderFloat("dt (manual)", &sim.dt, 0.005f, 0.08f);
    ImGui::EndDisabled();

    ImGui::SeparatorText("Smoke (tracers)");
    // Release emitPerFrame white points across the inlet each frame; they drift
    // with the flow and are culled on exit. The live count ~ emit * crossing-time.
    ImGui::Checkbox("smoke particles", &showParticles);
    ImGui::SliderInt("emit / frame", &sim.emitPerFrame, 0, 600);
    ImGui::SliderFloat("point size", &pointSize, 1.0f, 6.0f, "%.1f");
    ImGui::Checkbox("color by vorticity", &colorByVorticity);
    if (colorByVorticity)
      ImGui::SliderFloat("vorticity color gain", &vortColorGain, 0.2f, 5.0f, "%.1f");
    ImGui::Text("live particles: %zu", sim.pPos.size());

    ImGui::SeparatorText("Extraction (smoke rings)");
    // The eigensolver payoff: trace the shed vortices to filament loops via the
    // connection Laplacian. hbar sets the flux quantum per tube. The three solvers
    // are a clean A/B -- all solve the SAME connection Laplacian; the GPU one is
    // the same covMG-LOBPCG in float (viewer speed), the CPU/Lanczos ones stay the
    // double-precision ground truth.
    ImGui::Checkbox("extract filaments", &sim.extractOn);
    static const char* kSolvers[] = {"gauge-MG (CPU double)", "gauge-MG (GPU Metal, float)",
                                     "SLEPc Lanczos"};
    const int prevSolver = sim.extractSolver;
#ifdef BOCHNER_WITH_PETSC
    ImGui::Combo("extraction solver", &sim.extractSolver, kSolvers, 3);
#else
    // Built without SLEPc: drop the Lanczos entry rather than offering a
    // control that cannot run.
    ImGui::Combo("extraction solver", &sim.extractSolver, kSolvers, 2);
#endif
    if (sim.extractSolver != prevSolver) {
      sim.prevVec.clear();  // drop the stale warm start
      sim.smoothVec.clear();
    }
#ifndef BOCHNER_WITH_METAL
    if (sim.extractSolver == Sim::EXTRACT_GAUGE_GPU)
      ImGui::TextDisabled("(built without Metal -> using the CPU gauge-MG)");
#endif
    // Weissmann-Pinkall needs |theta| = |u|h/hbar < pi on every face. Below the
    // floor the holonomy wraps and the extracted windings are meaningless --
    // and because aliasing INCREASES the filament count, the failure looks
    // exactly like the intended effect of lowering hbar. Show the floor and
    // flag the crossing rather than letting the slider quietly lie.
    {
      const double floorH = sim.grid ? bochner::hbarAliasingFloor(*sim.grid, sim.u) : 0.0;
      ImGui::SliderFloat("hbar (flux quantum)", &sim.hbar, 0.01f, 0.3f, "%.3f");
      if (floorH > 0.0) {
        if (sim.hbar < floorH) {
          // |theta|max = h*umax/hbar = pi * (floor/hbar), so floor/hbar is the
          // overshoot in units of pi.
          const double overshoot = floorH / std::max(1e-6, (double)sim.hbar);
          ImGui::TextColored(ImVec4(1.0f, 0.35f, 0.25f, 1.0f),
                             "ALIASED: hbar %.3f < floor %.3f (|theta|max ~ %.1f pi) -- the "
                             "extra filaments are wrap artifacts, not structure",
                             sim.hbar, floorH, overshoot);
        }
        else
          ImGui::TextDisabled("(aliasing floor %.3f; below it the holonomy wraps)", floorH);
      }
    }
    ImGui::SliderFloat("filament smoothing", &sim.filamentSmooth, 0.0f, 0.95f, "%.2f");
    ImGui::SliderInt("extract every N frames", &sim.extractEvery, 1, 12);
    ImGui::TextDisabled("(smaller hbar -> more/thinner filaments; eigensolve is the cost)");

    ImGui::SeparatorText("Render");
    // Two fixed columns so every toggle stays on-panel (SameLine with no offset
    // wrapped past the panel width and hid the last boxes).
    const float col2 = 150.0f;
    ImGui::Checkbox("obstacle", &showObstacle);
    ImGui::SameLine(col2);
    ImGui::Checkbox("domain box", &showBox);
    ImGui::Checkbox("filaments", &showFilament);
    ImGui::SameLine(col2);
    ImGui::Checkbox("vorticity", &showSlice);
    ImGui::Checkbox("glyphs", &showGlyphs);
    ImGui::TextDisabled("(omega_z slice: red +, blue -; filaments = red tubes)");
    ImGui::SliderFloat("slice gain", &sliceGain, 0.2f, 5.0f, "%.1f");
    ImGui::SliderFloat("tube radius", &tubeRadius, 0.005f, 0.06f, "%.3f");
    ImGui::SliderInt("glyph density", &glyphRes, 4, 20);
    ImGui::SliderFloat("glyph scale", &glyphScale, 0.02f, 0.4f);
    ImGui::End();

    Mesh lines, tris;
    const float hx = Sim::Lx / 2, hy = Sim::Ly / 2, hz = Sim::Lz / 2;
    if (showBox) {
      const float c[8][3] = {{-hx, -hy, -hz}, {hx, -hy, -hz}, {hx, hy, -hz}, {-hx, hy, -hz},
                             {-hx, -hy, hz},  {hx, -hy, hz},  {hx, hy, hz},  {-hx, hy, hz}};
      const int e[12][2] = {{0, 1}, {1, 2}, {2, 3}, {3, 0}, {4, 5}, {5, 6},
                            {6, 7}, {7, 4}, {0, 4}, {1, 5}, {2, 6}, {3, 7}};
      for (auto& ed : e)
        lines.seg({c[ed[0]][0], c[ed[0]][1], c[ed[0]][2]},
                  {c[ed[1]][0], c[ed[1]][1], c[ed[1]][2]}, 0.4f, 0.4f, 0.45f);
    }
    if (showGlyphs && sim.grid) {
      const int gy = glyphRes, gx = glyphRes * int(Sim::Lx / Sim::Ly), gz = std::max(2, glyphRes / 2);
      for (int i = 0; i < gx; ++i)
        for (int j = 0; j < gy; ++j)
          for (int k = 0; k < gz; ++k) {
            Vec3 p{-hx + Sim::Lx * (i + 0.5) / gx, -hy + Sim::Ly * (j + 0.5) / gy,
                   -hz + Sim::Lz * (k + 0.5) / gz};
            Vec3 vel = sampleVelocity(*sim.grid, sim.u, p);
            double sp = vnorm(vel);
            Vec3 q = vadd(p, vscale(vel, glyphScale));
            float t = std::min(1.0f, float(sp / std::max(0.1f, sim.U)));
            lines.seg(p, q, 0.35f + 0.35f * t, 0.6f + 0.25f * t, 1.0f);
          }
    }
    if (showSlice && sim.grid) addVortSliceXY(tris, *sim.grid, sim.u, sliceGain);
    if (showFilament)
      for (size_t i = 0; i < sim.loops.size(); ++i)
        addTube(tris, sim.loops[i], tubeRadius, sim.loopClosed[i] != 0, 0.95f, 0.13f, 0.11f);
    if (showObstacle) {
      const float cr = 0.75f, cg = 0.78f, cb = 0.82f;
      if (sim.obstacle == Sim::CYLINDER)
        addCylinder(tris, sim.center, sim.radius, -hz, hz, cr, cg, cb);
      else if (sim.obstacle == Sim::SPHERE)
        addSphere(tris, sim.center, sim.radius, cr, cg, cb);
      else {  // SQUARE or PLATE: the finite oriented box
        Vec3 ex, ey, ez;
        sim.boxFrame(ex, ey, ez);
        addOrientedBox(tris, sim.center, ex, ey, ez, sim.boxThick, sim.boxWidth, sim.boxSpan, cr, cg,
                       cb);
      }
    }
    Mesh points;  // smoke tracers: white, or colored by local vorticity magnitude
    if (showParticles && colorByVorticity && sim.grid && !sim.diverged) {
      const MacGrid& g = *sim.grid;
      const std::vector<float> vmag = vorticityMagField(g, sim.u);
      float peak = 1e-6f;
      for (float v : vmag) peak = std::max(peak, v);
      // Smooth the normalizing peak across frames so the color scale is steady.
      vortPeakEma = vortPeakEma > 0.0f ? 0.85f * vortPeakEma + 0.15f * peak : peak;
      const double invh = 1.0 / g.spacing();
      const Vec3 org = g.origin();
      for (const auto& p : sim.pPos) {
        const int i = std::clamp(int((p[0] - org[0]) * invh), 0, g.nx() - 1);
        const int j = std::clamp(int((p[1] - org[1]) * invh), 0, g.ny() - 1);
        const int k = std::clamp(int((p[2] - org[2]) * invh), 0, g.nz() - 1);
        const float t =
            std::clamp(vortColorGain * vmag[g.cellIndex(i, j, k)] / vortPeakEma, 0.0f, 1.0f);
        // Calm smoke stays white; higher swirl ramps to red, so the shed vortices
        // read directly off the tracer field.
        points.vert(p, 0.92f + t * (0.95f - 0.92f), 0.92f + t * (0.12f - 0.92f),
                    0.96f + t * (0.10f - 0.96f));
      }
    } else if (showParticles) {
      for (const auto& p : sim.pPos) points.vert(p, 0.92f, 0.92f, 0.96f);
    }

    int fbw, fbh;
    glfwGetFramebufferSize(win, &fbw, &fbh);
    glViewport(0, 0, fbw, fbh);
    glClearColor(0.08f, 0.09f, 0.11f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    std::array<float, 3> eye{g_cam.dist * std::cos(g_cam.el) * std::sin(g_cam.az),
                             g_cam.dist * std::sin(g_cam.el),
                             g_cam.dist * std::cos(g_cam.el) * std::cos(g_cam.az)};
    Mat4 view = viewer::lookAt(eye, {0, 0, 0}, {0, 1, 0});
    Mat4 projm = viewer::perspective(0.9f, float(fbw) / std::max(1, fbh), 0.05f, 100.0f);
    Mat4 mvp = viewer::mul(projm, view);

    glUseProgram(prog);
    glUniformMatrix4fv(uMVP, 1, GL_FALSE, mvp.data());
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    if (!lines.v.empty()) {
      glBufferData(GL_ARRAY_BUFFER, lines.v.size() * sizeof(float), lines.v.data(), GL_DYNAMIC_DRAW);
      glDrawArrays(GL_LINES, 0, lines.count());
    }
    if (!tris.v.empty()) {
      glBufferData(GL_ARRAY_BUFFER, tris.v.size() * sizeof(float), tris.v.data(), GL_DYNAMIC_DRAW);
      glDrawArrays(GL_TRIANGLES, 0, tris.count());
    }
    if (!points.v.empty()) {
      glUniform1f(uPointSize, pointSize);
      glBufferData(GL_ARRAY_BUFFER, points.v.size() * sizeof(float), points.v.data(),
                   GL_DYNAMIC_DRAW);
      glDrawArrays(GL_POINTS, 0, points.count());
    }

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    glfwSwapBuffers(win);
  }

  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();
  glfwDestroyWindow(win);
  glfwTerminate();
#ifdef BOCHNER_WITH_PETSC
  SlepcFinalize();
#endif
  return 0;
}
