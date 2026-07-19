/// \file
/// Interactive OpenGL viewer for the grid Covector-Fluids <-> Weissmann-Pinkall
/// pipeline. Runs the simulation live (BFECC covector advection + projection)
/// and re-extracts the vortex filament every frame, rendering the filament,
/// the domain box, and velocity glyphs, with ImGui controls for the parameters.
///
/// Build: cmake -DBOCHNER_WITH_PETSC=ON -DBOCHNER_WITH_VIEWER=ON ...
/// Note: the per-frame eigensolve is sub-second but not yet real-time (Phase 3),
/// so playback runs at a few fps -- exactly the cost the preconditioner study
/// aims to remove.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <memory>
#include <utility>
#include <vector>

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
#ifdef BOCHNER_WITH_METAL
#include "gpu/MetalContext.h"
#endif
#include "extraction/MacConnectionLaplacian.h"
#include "extraction/MacFilaments.h"
#include "grid/MacGrid.h"
#include "fluid/MacProjection.h"
#include "fluid/MacCharacteristicMap.h"
#include "fluid/MacVortexRing.h"
#include "glmath.h"
#include "viewer_support.h"

using namespace bochner;

using namespace viewer;

// Orbit camera state (y-up: the elongated tank's long z-axis lies HORIZONTAL on
// screen, flow left->right, for the flow-past-obstacle setup). az orbits around
// the vertical y-axis, el tilts above the floor; declared early so Sim::reseed
// can re-frame it. Default az~1.15 gives a 3/4 side view (az=pi/2 = pure side).
Camera g_cam{1.15f, 0.32f, 6.0f};

// ---------------------------------------------------------------------------
// Simulation state: parameters + grid + velocity field + extracted filaments.
// ---------------------------------------------------------------------------
struct Sim {
  enum Seed { RING, HILL, LEAPFROG, TREFOIL, HOPF, COLLIDE };
  int seed = TREFOIL;  // default to the self-untying trefoil knot (CF Fig. 9): at the
                       // interactive n=32 it extracts as a genuinely KNOTTED closed
                       // filament (verified by the extracted loop's Alexander
                       // determinant = 3, not just "one closed loop"), a striking
                       // opening view where the leapfrog needs a resolution the
                       // interactive grid cannot afford to actually thread. The core
                       // default matters here -- see `core` below.
  int n = 32;          // cross-section cells per axis
  int zMult = 1;       // cube box: the knot is compact (and a cube is half the cells of
                       // the old 2:1:1 leapfrog box, so the default is cheaper too)
  float L = 1.4f;      // half box size (cross-section)
  float R = 0.7f;      // ring radius
  float Gamma = 0.3f;  // circulation. Lowered from 1.0: MEASURED that a strong
                       // circulation makes the leapfrog cores merge (the solver's
                       // numerical dissipation grows with velocity magnitude even
                       // at fixed CFL), while ~0.3 (the reference's 0.28) lets them
                       // thread. hbar auto-follows (=Gamma/2pi) so the seed's flux
                       // quantum -- and thus the extraction -- stays consistent.
  float core = 0.10f;  // Biot-Savart core regularization a. Default set by the
                       // TREFOIL seed's TOPOLOGY: at n=32 (h~0.088) the knot's
                       // strands are ~4.4 cells apart, so a core wide enough to
                       // bridge them merges the three lobes into one blob and the
                       // extracted vorticity comes out UNKNOTTED. MEASURED (Alexander
                       // determinant of the extracted loop): a>=0.14 -> det 1 (unknot),
                       // a<=0.13 -> det 3 (trefoil); a=0.10 (~1.1-cell core, still
                       // resolved) stays a knot through >=30 steps of evolution, with
                       // margin. Larger a (e.g. old 0.15) reads smoother but unknots;
                       // the LEAPFROG seed instead wants a bigger core so its two rings
                       // hold together (raise the slider for that seed).
  float a = 0.9f;      // Hill sphere radius
  float U = 1.0f;      // Hill stream speed
  float startZ = -1.0f;
  float dt = 0.04f;        // manual timestep (used when cflAdaptive is off)
  bool cflAdaptive = true;  // pick dt from a target CFL so stability is RESOLUTION-INDEPENDENT
  float cflTarget = 0.5f;   // CF runs ~0.5 (paper 5.4.3; their critical CFL ~6.18)
  double dtEff = 0.04;      // the dt actually used this frame (HUD)
  double curCfl = 0.0;      // measured CFL U*dt/h this frame (HUD)
  float hbar = 1.0f;
  bool hbarAuto = true;     // hbar follows Gamma/2pi (one flux quantum); off = manual
  int frame = 0;
  // The GPU covMG-LOBPCG is fast enough to extract every frame; CPU/Lanczos want a larger
  // cadence (raise this after switching to them).
  int extractEvery = 1;
  // Extraction eigensolver -- all three solve the SAME connection Laplacian (a
  // clean A/B): gauge-MG on the CPU (double, authoritative), the same covMG-LOBPCG on the
  // GPU (Metal, float -- viewer speed), or SLEPc Lanczos.
  enum ExtractSolver { EXTRACT_GAUGE_CPU = 0, EXTRACT_GAUGE_GPU = 1, EXTRACT_LANCZOS = 2 };
#ifdef BOCHNER_WITH_METAL
  int extractSolver = EXTRACT_GAUGE_GPU;  // default: the fast GPU path when Metal is built
#else
  int extractSolver = EXTRACT_GAUGE_CPU;
#endif
  bool useMcm = true;    // CF+MCM (Alg 5, long-time flow map) vs base BFECC (Alg 1); the
                         // validated port -- default ON (matches refmatch_leapfrog)
#ifdef BOCHNER_WITH_METAL
  bool useGpuAdvect = true;   // CF+MCM map advection + pullbacks on the Metal GPU (resident maps)
  bool useGpuProject = true;  // closed-box projection on the Metal GPU (geometric MG, float)
#else
  bool useGpuAdvect = false;
  bool useGpuProject = false;
#endif
  bool useLimiter = true;  // CF 5.4.2 BFECC minmax limiter (CF+MCM only; energy sink)
  int reinitEvery = 20;    // CF+MCM map lifetime: longer map = fewer interpolations = less
                           // dissipation (the BiMocq mechanism). The paper's 5 is for ~51
                           // cells/radius; coarse grids need a longer map (measured).
  double framedLz = -1.0;  // last box half-length the camera was framed for (avoid re-zoom on reseed)

  // Live timing (ms) for the HUD.
  double simMs = 0.0;      // advect + project (every frame)
  double extractMs = 0.0;  // connection + eigensolve + trace/link (every k frames)
  bool diverged = false;   // set when the field goes non-finite; halts stepping/extraction

  std::unique_ptr<MacGrid> grid;
  std::unique_ptr<MacProjector> projector;        // geometric-MG pressure solve (warm-started)
  std::unique_ptr<CfMcmSolver> mcm;               // CF+MCM state (lazy; null when using BFECC)
  FaceField u;
  std::vector<std::vector<Vec3>> loops;           // closed filament loops
  std::vector<std::complex<double>> prevVec;      // gauge-MG eigensolver warm start

  static double nowMs() {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
  }

  void reseed() {
    const double h = 2.0 * L / n;       // cubic cells; cross-section sets resolution
    const int nz = n * zMult;           // elongate along z (CF's 2:1:1 trefoil/leapfrog box)
    const double Lz = L * zMult;
    grid = std::make_unique<MacGrid>(n, n, nz, h, Vec3{-L, -L, -Lz});
    projector = std::make_unique<MacProjector>(*grid, PoissonMgOptions{}, BoundarySpec::allClosed());
    // Travelling structures start near the lower wall of a long box so they have
    // room to evolve; in a cube the user-set startZ is used instead.
    const double seedZ = (zMult > 1) ? (-Lz + 0.8) : startZ;
    if (seed == RING) {
      u = vortexRingFaceField(*grid, {0, 0, seedZ}, {0, 0, 1}, R, Gamma, core);
    } else if (seed == LEAPFROG) {
      // CF Fig.10 leapfrog, seeded EXACTLY as the reference code does it: a thin
      // vortex-sheet SLAB, not two Biot-Savart filament rings. The IC is the
      // decisive fidelity point -- the reference imposes
      // a two-tier slug of axial velocity (2*vel for r<=0.6R, vel for 0.6R<r<=R)
      // on a thin slab and lets the projection below turn the two radial jumps
      // into a pair of nested coaxial sheets. With THIS IC the rings leapfrog
      // (inner expands, outer contracts, radii swap); the old filament seed made
      // them merge/bounce and never thread. Radius
      // ratio 0.6 (=1.2/2.0) matches the reference's inner/outer rings.
      // Reference PROPORTIONS: at the default R=0.7 the rings fill ~80% of the
      // cross-section (rOuter = 0.8*L, like the reference's 2.0 in a 2.5 half-width;
      // ratio 0.6). MEASURED: only near these proportions do the two vortex sheets
      // sit ~25 cells/radius apart at n=64 and thread instead of merging -- so the R
      // slider now SCALES this tuned default (R/0.7) rather than being ignored, but
      // shrinking R much below 0.7 makes the rings merge (too few cells/radius).
      // Clamp the outer sheet inside the box.
      const double rOut = std::min(0.9 * L, 0.8 * L * (R / 0.7)), rIn = 0.6 * rOut;
      u = leapfrogSlabFaceField(*grid, {0, 0, seedZ}, {0, 0, 1}, rIn, rOut, Gamma);
    } else if (seed == COLLIDE) {
      // Two coaxial rings with opposite axes (same Gamma) move toward each other
      // and reconnect in place at z = 0 (a head-on collision).
      const double d = (zMult > 1) ? Lz * 0.5 : 0.6;
      const std::vector<std::vector<Vec3>> rings{
          circleCurve({0, 0, -d}, {0, 0, 1}, R, 256),    // lower ring moves +z
          circleCurve({0, 0, d}, {0, 0, -1}, R, 256)};   // upper ring moves -z
      u = filamentFaceField(*grid, rings, Gamma, core);
    } else if (seed == TREFOIL) {
      // The self-untying knotted filament (CF Fig. 9); R scales the knot. It
      // self-propels, so start it low in a long box, centered in a cube. The
      // 0.45 factor fills the cube well while staying a single resolved loop at
      // n=32 (headless-checked: 1 closed filament for scale 0.24-0.34 at n=32).
      const double knotZ = (zMult > 1) ? seedZ : 0.0;
      u = filamentFaceField(*grid, trefoilKnotCurve({0, 0, knotZ}, R * 0.45, 256), Gamma, core);
    } else if (seed == HOPF) {
      // Two interlocked rings (Hopf link): ring 2 pierces ring 1's disk once.
      const std::vector<std::vector<Vec3>> rings{
          circleCurve({-R * 0.5, 0, 0}, {0, 0, 1}, R, 256),
          circleCurve({R * 0.5, 0, 0}, {0, 1, 0}, R, 256)};
      u = filamentFaceField(*grid, rings, Gamma, core);
    } else {  // lab frame so Hill's vortex self-propels (axis -z so it moves +z, up)
      u = hillVortexFaceField(*grid, {0, 0, 0}, {0, 0, -1}, a, U, /*labFrame=*/true);
    }
    u = projector->project(u);
    mcm.reset();  // rebuilt lazily from the fresh field on the first CF+MCM step
    if (hbarAuto) hbar = Gamma / (2.0 * M_PI);  // keep a user-set hbar across reseeds
    frame = 0;
    diverged = false;
    prevVec.clear();
    // Re-frame the camera ONLY when the box size actually changed, so re-seeding
    // (e.g. dragging the core slider) does not jolt the view the user has set.
    if (std::abs(framedLz - Lz) > 1e-6) {
      g_cam.dist = std::clamp(2.4f * static_cast<float>(Lz), 4.0f, 30.0f);
      framedLz = Lz;
    }
    extract();
  }

  // Max fluid speed |u| at cell centers (for the CFL timestep / HUD).
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
    // Sim: covector advection + warm-started geometric-MG projection. CF+MCM
    // (Alg 5, long-time flow map) or the base BFECC (Alg 1). CF+MCM carries map
    // state, rebuilt from the current field whenever it is (re)enabled.
    const double t0 = nowMs();
    // Pick the timestep from a target CFL (dt = C*h/U_max) so stability is
    // RESOLUTION-INDEPENDENT: a fixed dt at finer h raises the CFL and the BFECC/
    // covector-stretching energy growth blows up (measured); scaling dt with h
    // holds it flat. The limiter does not prevent this (it clamps dispersion, not
    // the CFL-driven growth). U is capped away from 0 so a near-still field can't
    // produce a huge dt.
    const double h = grid->spacing();
    const double raw = maxSpeed();
    if (!std::isfinite(raw)) {  // field blew up: halt rather than step/extract NaN silently
      diverged = true;
      simMs = nowMs() - t0;
      return;
    }
    const double U = std::max(1e-6, raw);
    dtEff = cflAdaptive ? std::min(0.2, double(cflTarget) * h / U) : double(dt);
    curCfl = U * dtEff / h;
    bool gpuAdvect = false, gpuProject = false;
#ifdef BOCHNER_WITH_METAL
    gpuAdvect = useGpuAdvect && gpu::metalAvailable();
    gpuProject = useGpuProject && gpu::metalAvailable();
#endif
    if (useMcm) {
      if (!mcm)
        mcm = std::make_unique<CfMcmSolver>(*grid, u, reinitEvery, PoissonMgOptions{}, useLimiter,
                                            /*remapTol=*/0.5, /*secondOrder=*/true,
                                            BoundarySpec::allClosed(), gpuAdvect);
      mcm->setGpuProjection(gpuProject);
      mcm->step(dtEff);
      u = mcm->velocity();
    } else {
      mcm.reset();  // BFECC owns u; drop any stale map state
      projector->setGpuProjection(gpuProject);
      u = projector->project(advectCovectorBFECC(*grid, u, u, dtEff));
    }
    simMs = nowMs() - t0;
    ++frame;
    if (frame % extractEvery == 0) extract();  // decoupled extraction cadence
  }

  // HUD label for the effective extraction solver (reflects GPU fallback).
  const char* extractLabel() const {
    if (extractSolver == EXTRACT_LANCZOS) return "SLEPc Lanczos";
#ifdef BOCHNER_WITH_METAL
    if (extractSolver == EXTRACT_GAUGE_GPU)
      return gpu::metalAvailable() ? "gauge-MG (GPU Metal, float)" : "gauge-MG (CPU; no GPU)";
#endif
    return "gauge-MG (CPU double)";
  }

  void extract() {
    // W-P extraction: smallest-eigenvalue section of the connection Laplacian,
    // traced to its zero set. The link phases theta are shared by all solvers;
    // only the eigensolve differs, so the toggle is a clean A/B (the filament
    // must be solver-INDEPENDENT). psi is the interleaved [Re,Im] section.
    const double t0 = nowMs();
    const FaceField theta = connectionAngles(*grid, u, hbar);
    std::vector<double> psi;
    bool gpuExtract = false;
#ifdef BOCHNER_WITH_METAL
    gpuExtract = (extractSolver == EXTRACT_GAUGE_GPU) && gpu::metalAvailable();
#endif
#ifdef BOCHNER_WITH_PETSC
    const bool useLanczos = (extractSolver == EXTRACT_LANCZOS);
#else
    // No SLEPc in this build: the Lanczos baseline is unavailable and the
    // combo entry is hidden below, so this branch is never taken.
    const bool useLanczos = false;
#endif
    if (useLanczos) {
#ifdef BOCHNER_WITH_PETSC
      // SLEPc Lanczos baseline: assemble the connection Laplacian and solve.
      const CooMatrix E = connectionLaplacian(*grid, theta);
      psi = smallestEigenpairLanczos(E, 1e-6).vector;  // already interleaved [Re,Im]
#endif
    } else if (gpuExtract) {
#ifdef BOCHNER_WITH_METAL
      // Same covMG-LOBPCG as the CPU path, on the GPU in float (viewer speed). Rebuild +
      // upload the hierarchy each extract (the connection changes every frame);
      // warm-started from the previous filament eigenvector.
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
      const gpu::GaugeEigenGpu res =
          gpu::lobpcgSolveGauge(handle, prevVec, /*maxIters=*/100, /*tol=*/1e-4, opts.precCycles,
                              opts.mg.nu1, opts.mg.nu2, opts.mg.coarseSweeps, opts.mg.omega);
      gpu::freeGauge(handle);
      prevVec = res.vector;
      psi = toInterleaved(res.vector);
#endif
    } else {  // EXTRACT_GAUGE_CPU (also the fallback when the GPU is unavailable)
      // Our gauge-MG Rayleigh-quotient eigensolver (no SLEPc), warm-started from
      // the previous filament eigenvector.
      const GaugeLattice lat = gaugeLatticeFromFaces(*grid, theta);
      GaugeEigenOptions opts;
      opts.relativeGsDrop = false;  // live path: keep the absolute-drop warm-start early-exit
      opts.tol = 1e-6;
      const auto res = smallestEigenpairGaugeMG(lat, prevVec.empty() ? nullptr : &prevVec, opts);
      prevVec = res.vector;
      psi = toInterleaved(res.vector);
    }
    loops.clear();
    for (auto& f : linkFilaments(*grid, traceZeroSet(*grid, psi)))
      if (f.closed && f.points.size() >= 4) loops.push_back(std::move(f.points));
    extractMs = nowMs() - t0;
  }
};

// ---------------------------------------------------------------------------
// Orbit camera input (z-up, orbiting the origin); Camera/g_cam declared above.
// ---------------------------------------------------------------------------
double g_lastX = 0, g_lastY = 0;
bool g_dragging = false;

void scrollCallback(GLFWwindow*, double, double dy) {
  // Don't zoom the camera when the wheel is scrolling the ImGui panel (ImGui's
  // own chained scroll callback handles the panel); only orbit-zoom over the 3D view.
  if (ImGui::GetCurrentContext() && ImGui::GetIO().WantCaptureMouse) return;
  g_cam.dist *= std::pow(0.9f, static_cast<float>(dy));
  g_cam.dist = std::clamp(g_cam.dist, 1.5f, 30.0f);
}

// Colored vorticity slice on the central x-z plane (j = ny/2) -- the standard
// vortex-ring cross-section: each ring shows as a +/- omega_y blob pair at
// x = +-R, so the user can watch BOTH rings and the moment the sim breaks. We
// emit one flat quad per interior cell, colored by the azimuthal vorticity
// omega_y = d u_x/dz - d u_z/dx (diverging map: red +, blue -, dark ~0),
// auto-normalized by the per-frame peak and scaled by `gain`.
static void addVorticitySlice(Mesh& m, const MacGrid& g, const FaceField& u, float gain) {
  const int j = g.ny() / 2;
  const int nx = g.nx(), nz = g.nz();
  const double inv2h = 1.0 / (2.0 * g.spacing());
  auto cx = [&](int i, int k) {
    return 0.5 * (u.x[g.faceXIndex(i, j, k)] + u.x[g.faceXIndex(i + 1, j, k)]);
  };
  auto cz = [&](int i, int k) {
    return 0.5 * (u.z[g.faceZIndex(i, j, k)] + u.z[g.faceZIndex(i, j, k + 1)]);
  };
  std::vector<double> w(nx * nz, 0.0);
  double peak = 1e-12;
  for (int i = 1; i < nx - 1; ++i)
    for (int k = 1; k < nz - 1; ++k) {
      const double wy = (cx(i, k + 1) - cx(i, k - 1)) * inv2h - (cz(i + 1, k) - cz(i - 1, k)) * inv2h;
      w[i * nz + k] = wy;
      peak = std::max(peak, std::abs(wy));
    }
  const double y = g.cellCenter(0, j, 0)[1];  // plane height (~0 for a symmetric box)
  const double hh = 0.5 * g.spacing();
  for (int i = 1; i < nx - 1; ++i)
    for (int k = 1; k < nz - 1; ++k) {
      const float t = std::clamp(float(gain * w[i * nz + k] / peak), -1.0f, 1.0f);
      const float r = 0.08f + 0.92f * std::max(0.0f, t);
      const float b = 0.08f + 0.92f * std::max(0.0f, -t);
      const Vec3 c = g.cellCenter(i, j, k);
      const Vec3 p00{c[0] - hh, y, c[2] - hh}, p10{c[0] + hh, y, c[2] - hh};
      const Vec3 p11{c[0] + hh, y, c[2] + hh}, p01{c[0] - hh, y, c[2] + hh};
      m.vert(p00, r, 0.08f, b); m.vert(p10, r, 0.08f, b); m.vert(p11, r, 0.08f, b);
      m.vert(p00, r, 0.08f, b); m.vert(p11, r, 0.08f, b); m.vert(p01, r, 0.08f, b);
    }
}

int main(int argc, char** argv) {
#ifdef BOCHNER_WITH_PETSC
  SlepcInitialize(&argc, &argv, nullptr, nullptr);
#endif

  if (!glfwInit()) {
    std::fprintf(stderr, "glfwInit failed\n");
    return 1;
  }
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
  GLFWwindow* win = glfwCreateWindow(1280, 800, "Bochner - Covector Fluids / W-P viewer", nullptr,
                                     nullptr);
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

  // Flat vertex-color program (shared with the obstacle viewer). This viewer
  // draws no GL_POINTS, so the point-size uniform is unused here.
  GLint uMVP = -1, uPointSize = -1;
  GLuint prog = makeFlatProgram(uMVP, uPointSize);
  (void)uPointSize;

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

  Sim sim;
  sim.reseed();

  bool playing = false;
  bool showFilament = true, showGlyphs = true, showBox = true, showSlice = false;
  int glyphRes = 7;
  float glyphScale = 0.15f;
  float tubeRadius = 0.03f;
  float sliceGain = 1.0f;  // saturation gain for the vorticity slice colormap

  while (!glfwWindowShouldClose(win)) {
    glfwPollEvents();

    // Orbit drag (when not over the UI).
    ImGuiIO& io = ImGui::GetIO();
    double mx, my;
    glfwGetCursorPos(win, &mx, &my);
    bool leftDown = glfwGetMouseButton(win, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
    if (leftDown && !io.WantCaptureMouse) {
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

    // --- UI ---
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    ImGui::Begin("Controls");
    ImGui::Text("frame %d   loops: %zu", sim.frame, sim.loops.size());
    if (sim.diverged)
      ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f),
                         "DIVERGED -- sim halted. Lower dt / CFL target, then Reseed.");
    ImGui::Text("grid %dx%dx%d = %d cells", sim.n, sim.n, sim.n * sim.zMult,
                sim.n * sim.n * sim.n * sim.zMult);
    // Resolution per structure: CF runs ~51 cells/ring-radius; we are far coarser,
    // which is why thickness/resolution coupling and blow-ups appear. core in cells
    // < ~1.5 is under-resolved.
    {
      const double h = 2.0 * sim.L / sim.n;
      const double rad = (sim.seed == Sim::HILL) ? sim.a : sim.R;
      ImGui::Text("h=%.4f | %.0f cells/radius | core %.1f cells (CF ~51 cells/radius)", h, rad / h,
                  sim.core / h);
    }
    // Live cost: sim every frame, extraction every k frames -> amortized rate.
    {
      const double amortized = sim.simMs + sim.extractMs / std::max(1, sim.extractEvery);
      const char* advectLabel = "CPU";
      const char* projLabel = "CPU";
#ifdef BOCHNER_WITH_METAL
      advectLabel = (sim.useGpuAdvect && gpu::metalAvailable()) ? "GPU" : "CPU";
      projLabel = (sim.useGpuProject && gpu::metalAvailable()) ? "GPU" : "CPU";
#endif
      ImGui::Text("sim %.1f ms (advect %s, project %s) | extract %.1f ms (every %d, %s)", sim.simMs,
                  advectLabel, projLabel, sim.extractMs, sim.extractEvery, sim.extractLabel());
      ImGui::Text("amortized %.1f ms = %.1f fps   sim-only %.1f fps", amortized,
                  amortized > 0 ? 1000.0 / amortized : 0.0, sim.simMs > 0 ? 1000.0 / sim.simMs : 0.0);
    }
    // Memory: current resident vs the peak high-water mark. If you raise the grid
    // then lower it, `peak` stays at the high value while `current` should drop
    // back near the low-res baseline -- if current does NOT drop, that's a leak.
    {
      double curMB = 0.0, peakMB = 0.0;
      memoryMB(curMB, peakMB);
      ImGui::Text("mem %.0f MB (peak %.0f MB)", curMB, peakMB);
    }
    if (ImGui::Button(playing ? "Pause" : "Play")) playing = !playing;
    ImGui::SameLine();
    if (ImGui::Button("Step")) sim.step();
    ImGui::SameLine();
    if (ImGui::Button("Reseed")) sim.reseed();

    ImGui::SeparatorText("Seed");
    // Order must match the Sim::Seed enum (the combo index is the enum value).
    static const char* kSeedNames[] = {"Vortex ring",  "Hill's vortex", "Leapfrog rings",
                                       "Trefoil knot", "Hopf link",     "Head-on collision"};
    ImGui::Combo("example", &sim.seed, kSeedNames, IM_ARRAYSIZE(kSeedNames));
    if (sim.seed == Sim::HILL) {
      ImGui::SliderFloat("sphere a", &sim.a, 0.4f, 1.2f);
      ImGui::SliderFloat("speed U", &sim.U, 0.2f, 3.0f);
    } else {  // all filament seeds share radius/scale, circulation, core
      // Re-seed live on release (like core below) so dragging R/circulation shows
      // the new seed immediately instead of only after a manual Reseed.
      ImGui::SliderFloat(sim.seed == Sim::TREFOIL ? "knot scale R" : "radius R", &sim.R, 0.2f, 1.2f);
      if (ImGui::IsItemDeactivatedAfterEdit() && sim.grid) sim.reseed();
      ImGui::SliderFloat("circulation", &sim.Gamma, 0.2f, 4.0f);
      if (ImGui::IsItemDeactivatedAfterEdit() && sim.grid) sim.reseed();
      // Core thickness sets where the extracted filament sits relative to the
      // vorticity core (thin -> on it at hbar=Gamma/2pi). Resets to frame 0.
      ImGui::SliderFloat("core (thickness)", &sim.core, 0.03f, 0.30f, "%.3f");
      if (ImGui::IsItemDeactivatedAfterEdit() && sim.grid) sim.reseed();
      if ((sim.seed == Sim::RING || sim.seed == Sim::LEAPFROG) && sim.zMult == 1)
        ImGui::SliderFloat("start z", &sim.startZ, -1.3f, 0.0f);
    }
    // Cross-section cells. Default 32 stays interactive; the cap reaches 64 so a
    // leapfrog at n=64, zMult=2 is 64x64x128 -- the SAME cell count/shape as the
    // 128x64x64 reference tests (25.6 cells/ring-radius), where the cores stay
    // sharp and the swap is crisp. High n is seconds/frame (watch the fps HUD);
    // drop 'extract every N' or pause extraction to keep the sim responsive.
    // Snapped to multiples of 8: the pressure multigrid coarsens only while
    // every extent is even and >= 4, so an odd n (or one barely divisible by 2)
    // collapses the hierarchy to one or two levels and the projection silently
    // under-converges. Stepping by 8 keeps at least four levels across the
    // whole range.
    if (ImGui::SliderInt("grid n", &sim.n, 16, 64, "%d", ImGuiSliderFlags_AlwaysClamp))
      sim.n = ((sim.n + 4) / 8) * 8;
    ImGui::SliderInt("box length (xN z)", &sim.zMult, 1, 3);
    ImGui::TextDisabled(sim.zMult > 1 ? "(long box: seed near the lower end; n=64,x2 ~= the ref tests)"
                                      : "(Reseed applies seed/grid changes)");

    ImGui::SeparatorText("Solve / step");
    ImGui::Checkbox("CF+MCM (Alg 5)", &sim.useMcm);
    ImGui::SameLine();
    // The BFECC minmax limiter is a CF+MCM option (base BFECC always limits);
    // toggling it rebuilds the solver so the change takes effect immediately.
    ImGui::BeginDisabled(!sim.useMcm);
    if (ImGui::Checkbox("limiter (5.4.2)", &sim.useLimiter)) sim.mcm.reset();
    // Map lifetime: a longer map re-interpolates the field less often -> less
    // dissipation (BiMocq long-term mapping). Coarse grids need this longer than
    // the paper's 5. Changing it rebuilds the solver.
    if (ImGui::SliderInt("reinit every N steps", &sim.reinitEvery, 1, 60)) sim.mcm.reset();
    ImGui::EndDisabled();
    ImGui::TextDisabled("(CF+MCM long-time map; longer reinit = less dissipation; limiter = energy sink)");
#ifdef BOCHNER_WITH_METAL
    // The CF+MCM map advection + pullbacks are the CPU bottleneck; run them on the
    // GPU (resident maps, float). Rebuild the solver so the change takes effect.
    if (ImGui::Checkbox("GPU advection (Metal, CF+MCM maps)", &sim.useGpuAdvect)) sim.mcm.reset();
    ImGui::SameLine();
    ImGui::TextDisabled(gpu::metalAvailable() ? "(resident flow maps)" : "(no Metal device -> CPU)");
    // Closed-box projection on the GPU: the unpinned matrix-free geometric MG
    // (float) -- the route that beats the CPU for the all-Neumann box.
    ImGui::Checkbox("GPU projection (Metal geometric MG)", &sim.useGpuProject);
    ImGui::SameLine();
    ImGui::TextDisabled(gpu::metalAvailable() ? "(the step's biggest cost)" : "(no Metal device -> CPU)");
#endif
    // The reference leapfrog runs in a closed no-penetration box; open (Dirichlet
    // p=0) boundaries were dropped -- the projection there was unreliable and the
    // reference does not use them.
    // CFL-adaptive timestep: dt = CFL*h/U_max makes stability resolution-
    // INDEPENDENT (a fixed dt blows up at finer h; the limiter cannot stop it).
    ImGui::Checkbox("CFL-adaptive dt", &sim.cflAdaptive);
    ImGui::BeginDisabled(!sim.cflAdaptive);
    ImGui::SliderFloat("target CFL", &sim.cflTarget, 0.1f, 2.0f, "%.2f");
    ImGui::EndDisabled();
    ImGui::BeginDisabled(sim.cflAdaptive);
    ImGui::SliderFloat("dt (manual)", &sim.dt, 0.005f, 0.1f);
    ImGui::EndDisabled();
    ImGui::TextDisabled("dt used %.4f  |  CFL %.2f  (CF runs ~0.5, critical ~6.2)", sim.dtEff,
                        sim.curCfl);
    ImGui::SliderInt("extract every N frames", &sim.extractEvery, 1, 5);
    ImGui::TextDisabled("(sim runs every frame; filament refreshes every N)");
    // Eigensolver for the extraction: gauge-MG on the CPU (double), the same covMG-LOBPCG
    // on the GPU (Metal, float), or the SLEPc Lanczos baseline. Same connection
    // Laplacian, so the filament must match (an in-app A/B). Re-extract immediately
    // when paused so the switch is visible at once.
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
      sim.prevVec.clear();  // drop the stale warm start when the solver changes
      if (!playing && sim.grid && !sim.diverged) sim.extract();
    }
#ifndef BOCHNER_WITH_METAL
    if (sim.extractSolver == Sim::EXTRACT_GAUGE_GPU)
      ImGui::TextDisabled("(built without Metal -> using the CPU gauge-MG)");
#endif
    ImGui::TextDisabled("(extraction eigensolver A/B; filament should be identical)");
    // hbar = Gamma/2pi is one flux quantum (filament on the core for a thin ring).
    // Auto keeps it linked; uncheck to fit it by hand without reseed clobbering it.
    ImGui::Checkbox("auto hbar (=Gamma/2pi)", &sim.hbarAuto);
    if (sim.hbarAuto) sim.hbar = sim.Gamma / (2.0 * M_PI);
    ImGui::BeginDisabled(sim.hbarAuto);
    bool hbarChanged = ImGui::SliderFloat("hbar (flux quantum)", &sim.hbar, 0.02f, 1.0f, "%.4f");
    ImGui::EndDisabled();
    ImGui::TextDisabled("smaller hbar -> more nested filaments");
    if (hbarChanged && !playing && sim.grid && !sim.diverged) sim.extract();

    ImGui::SeparatorText("Render");
    ImGui::Checkbox("filament", &showFilament);
    ImGui::SameLine();
    ImGui::Checkbox("glyphs", &showGlyphs);
    ImGui::SameLine();
    ImGui::Checkbox("box", &showBox);
    ImGui::SameLine();
    ImGui::Checkbox("vorticity slice", &showSlice);
    ImGui::TextDisabled("(omega_y on the central x-z plane: red +, blue -)");
    ImGui::SliderFloat("slice gain", &sliceGain, 0.2f, 5.0f, "%.1f");
    ImGui::SliderFloat("tube radius", &tubeRadius, 0.005f, 0.08f, "%.3f");
    ImGui::SliderInt("glyph density", &glyphRes, 3, 12);
    ImGui::SliderFloat("glyph scale", &glyphScale, 0.02f, 0.5f);
    ImGui::End();

    // --- Build geometry (lines: box + glyphs; tris: filament tubes) ---
    Mesh lines, tris;
    const float L = sim.L;
    const float Lz = sim.L * sim.zMult;  // box may be elongated along z (CF 2:1:1)
    if (showBox) {
      const float c[8][3] = {{-L, -L, -Lz}, {L, -L, -Lz}, {L, L, -Lz}, {-L, L, -Lz},
                             {-L, -L, Lz},  {L, -L, Lz},  {L, L, Lz},  {-L, L, Lz}};
      const int e[12][2] = {{0, 1}, {1, 2}, {2, 3}, {3, 0}, {4, 5}, {5, 6},
                            {6, 7}, {7, 4}, {0, 4}, {1, 5}, {2, 6}, {3, 7}};
      for (auto& ed : e)
        lines.seg({c[ed[0]][0], c[ed[0]][1], c[ed[0]][2]},
                  {c[ed[1]][0], c[ed[1]][1], c[ed[1]][2]}, 0.4f, 0.4f, 0.45f);
    }
    if (showGlyphs && sim.grid) {
      const int glyphResZ = glyphRes * sim.zMult;  // keep glyph spacing uniform
      for (int i = 0; i < glyphRes; ++i)
        for (int j = 0; j < glyphRes; ++j)
          for (int k = 0; k < glyphResZ; ++k) {
            auto coord = [&](int q) { return -L + 2 * L * (q + 0.5) / glyphRes; };
            Vec3 p{coord(i), coord(j), -Lz + 2 * L * (k + 0.5) / glyphRes};
            Vec3 vel = sampleVelocity(*sim.grid, sim.u, p);
            double sp = vnorm(vel);
            Vec3 q = vadd(p, vscale(vel, glyphScale));
            float t = std::min(1.0f, float(sp));  // light blue, brighter = faster
            lines.seg(p, q, 0.35f + 0.25f * t, 0.65f + 0.25f * t, 1.0f);
          }
    }
    if (showSlice && sim.grid) addVorticitySlice(tris, *sim.grid, sim.u, sliceGain);
    if (showFilament)
      for (const auto& loop : sim.loops) addTube(tris, loop, tubeRadius, true, 0.95f, 0.13f, 0.11f);

    // --- Render ---
    int fbw, fbh;
    glfwGetFramebufferSize(win, &fbw, &fbh);
    glViewport(0, 0, fbw, fbh);
    glClearColor(0.08f, 0.09f, 0.11f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // y-up orbit: y is vertical, so the long z-axis (and x) lie in the horizontal
    // plane -> the elongated tank shows HORIZONTAL. el tilts above the floor.
    std::array<float, 3> eye{g_cam.dist * std::cos(g_cam.el) * std::sin(g_cam.az),
                             g_cam.dist * std::sin(g_cam.el),
                             g_cam.dist * std::cos(g_cam.el) * std::cos(g_cam.az)};
    Mat4 view = viewer::lookAt(eye, {0, 0, 0}, {0, 1, 0});
    Mat4 proj = viewer::perspective(0.9f, float(fbw) / std::max(1, fbh), 0.05f, 100.0f);
    Mat4 mvp = viewer::mul(proj, view);

    glUseProgram(prog);
    glUniformMatrix4fv(uMVP, 1, GL_FALSE, mvp.data());
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    if (!lines.v.empty()) {
      glBufferData(GL_ARRAY_BUFFER, lines.v.size() * sizeof(float), lines.v.data(),
                   GL_DYNAMIC_DRAW);
      glDrawArrays(GL_LINES, 0, lines.count());
    }
    if (!tris.v.empty()) {
      glBufferData(GL_ARRAY_BUFFER, tris.v.size() * sizeof(float), tris.v.data(), GL_DYNAMIC_DRAW);
      glDrawArrays(GL_TRIANGLES, 0, tris.count());
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
