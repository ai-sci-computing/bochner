/// \file
/// Headless per-frame profiler for the FLOW-PAST-OBSTACLE demo (the paper's
/// Section "demonstration" figure): the same numerical loop as the obstacle
/// viewer (tools/viewer/obstacle_main.cpp), stripped of GL/imgui/smoke, so the
/// demo numbers are reproducible without a display.
///
/// Setup mirrors the viewer defaults: fixed tank 6 x 2 x 1.2 (long x = flow),
/// res cells across the height (res=32 -> 96 x 32 x 19 cells), SPHERE obstacle
/// (radius 0.35) at the upstream third with a half-cell y-offset (symmetry
/// breaker), inflow U = 1, explicit viscous substeps nu = 0.008 (the no-slip
/// vorticity source), CFL-adaptive dt (target 0.6, cap 0.1). Extraction every
/// frame: theta = u h / hbar (hbar = 0.1), warm-started covMG-LOBPCG (absolute-drop
/// early exit -- the live policy), zero-set trace + filament linking.
///
/// Per frame it times the stages
///   sim:      cfl scan | BFECC covector advect | viscous diffuse | MGPCG project
///   extract:  connection build | eigensolve | trace+link
/// and reports the median [min..max] over the last `window` frames (the flow
/// evolves, so this is a sequence statistic, not repeated identical solves).
/// At the last frame it also runs the OTHER eigensolvers on the same operator:
/// CPU double covMG-LOBPCG and SLEPc Lanczos (both warm-started where supported), and
/// reports wall time, eigenvalue agreement, and the chordal projective distance
/// between the eigenvector lines (the A/B check that the float GPU path traces
/// the same filaments).
///
/// Usage: obstacle_profile [res] [frames] [mode]
///   res    = cells across the tank height  (default 32)
///   frames = frames to run                 (default 200; stats over last 50)
///   mode   = gpu | cpu                     (default gpu when Metal is built)
/// CPU thread count comes from OMP_NUM_THREADS.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <slepc.h>

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
#include "grid/GridOperators.h"
#include "grid/MacGrid.h"
#include "grid/MacObstacle.h"
#include "fluid/MacProjection.h"

using namespace bochner;
using cd = std::complex<double>;

namespace {

double nowMs() {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

struct StageStat {
  std::vector<double> v;
  void add(double ms) { v.push_back(ms); }
  double median() const {
    std::vector<double> t = v;
    std::sort(t.begin(), t.end());
    return t.empty() ? 0.0 : (t.size() % 2 ? t[t.size() / 2]
                                           : 0.5 * (t[t.size() / 2 - 1] + t[t.size() / 2]));
  }
  double mn() const { return v.empty() ? 0.0 : *std::min_element(v.begin(), v.end()); }
  double mx() const { return v.empty() ? 0.0 : *std::max_element(v.begin(), v.end()); }
};

void printStat(const char* name, const StageStat& s) {
  std::printf("  %-10s %8.2f ms   [%7.2f .. %7.2f]\n", name, s.median(), s.mn(), s.mx());
}

double maxSpeed(const MacGrid& g, const FaceField& u) {
  double m = 0.0;
  for (int i = 0; i < g.nx(); ++i)
    for (int j = 0; j < g.ny(); ++j)
      for (int k = 0; k < g.nz(); ++k) {
        const double ux = 0.5 * (u.x[g.faceXIndex(i, j, k)] + u.x[g.faceXIndex(i + 1, j, k)]);
        const double uy = 0.5 * (u.y[g.faceYIndex(i, j, k)] + u.y[g.faceYIndex(i, j + 1, k)]);
        const double uz = 0.5 * (u.z[g.faceZIndex(i, j, k)] + u.z[g.faceZIndex(i, j, k + 1)]);
        m = std::max(m, std::sqrt(ux * ux + uy * uy + uz * uz));
      }
  return m;
}

// Chordal projective distance between the complex lines spanned by two unit
// eigenvectors: sqrt(1 - |<a,b>|^2) (phase-invariant; 0 = same filaments).
double lineDistance(const std::vector<cd>& a, const std::vector<cd>& b) {
  cd ip(0, 0);
  double na = 0, nb = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    ip += std::conj(a[i]) * b[i];
    na += std::norm(a[i]);
    nb += std::norm(b[i]);
  }
  const double ov = std::min(1.0, std::abs(ip) / std::sqrt(std::max(1e-300, na * nb)));
  return std::sqrt(std::max(0.0, 1.0 - ov * ov));
}

}  // namespace

int main(int argc, char** argv) {
  SlepcInitialize(&argc, &argv, nullptr, nullptr);
  {
    const int res = (argc > 1) ? std::atoi(argv[1]) : 32;
    const int frames = (argc > 2) ? std::atoi(argv[2]) : 200;
    std::string mode = (argc > 3) ? argv[3] : "";
#ifdef BOCHNER_WITH_METAL
    const bool haveGpu = gpu::metalAvailable();
#else
    const bool haveGpu = false;
#endif
    if (mode.empty()) mode = haveGpu ? "gpu" : "cpu";
    const bool useGpu = (mode == "gpu");
    if (useGpu && !haveGpu) {
      std::fprintf(stderr, "error: mode=gpu but Metal is unavailable\n");
      return 2;
    }
    const int window = std::min(50, std::max(1, frames / 4));

    // --- viewer-default setup (sphere; see file header) ---
    const float Lx = 6.0f, Ly = 2.0f, Lz = 1.2f, radius = 0.35f;
    // BOCHNER_U / BOCHNER_NU sweep the Reynolds number Re = U*(2*radius)/nu,
    // for checking the sphere-wake regime transitions (steady axisymmetric ->
    // steady planar-symmetric ~210 -> unsteady loop shedding ~270).
    const char* uEnv = std::getenv("BOCHNER_U");
    const char* nuEnv = std::getenv("BOCHNER_NU");
    const float U = uEnv ? std::atof(uEnv) : 1.0f;
    const float nu = nuEnv ? std::atof(nuEnv) : 0.008f;
    const double hbar = 0.1, cflTarget = 0.6;
    const double h = double(Ly) / res;
    const int nx = std::max(12, int(std::lround(Lx / h)));
    const int ny = res;
    const int nz = std::max(4, int(std::lround(Lz / h)));
    const MacGrid grid(nx, ny, nz, h, Vec3{-Lx / 2, -Ly / 2, -Lz / 2});
    // Centred exactly. The half-cell y-offset this used to carry is no longer
    // needed to seed shedding (the cylinder sheds 20-24 filaments centred) and
    // was wrong for the sphere, whose sub-critical wake is axisymmetric.
    // BOCHNER_YOFFSET=1 restores it for comparison.
    const bool yoff = std::getenv("BOCHNER_YOFFSET") != nullptr;
    const Vec3 center{-Lx * 0.25, yoff ? 0.5 * h : 0.0, 0.0};
    // BOCHNER_OBSTACLE=cylinder selects the shedding geometry. The sphere's
    // shedding onset is Re ~ 270, so at this demo's Re ~ 87 a steady
    // axisymmetric wake is correct physics and the wake treatment cannot be
    // assessed on it; a cylinder sheds from Re ~ 47.
    const char* obsEnv = std::getenv("BOCHNER_OBSTACLE");
    const bool useCylinder = obsEnv && std::string(obsEnv) == "cylinder";
    const SolidMask solid = useCylinder
        ? cylinderMask(grid, center, Vec3{0, 0, 1}, radius)
        : sphereMask(grid, center, radius);
    int solidCells = 0;
    for (auto s : solid) solidCells += s;

    MacProjector proj(grid, PoissonMgOptions{}, BoundarySpec::channelFlow(U), solid);
    proj.setGpuProjection(useGpu);
    FaceField u = ops::zeroFaceField(grid);
    for (double& v : u.x) v = U;
    int cycles = 0;
    u = proj.project(u, &cycles);

    std::printf("obstacle profile: %dx%dx%d = %d cells (%d solid), %s r=%.2f, U=%g,"
                " nu=%g, hbar=%g\n",
                nx, ny, nz, grid.numCells(), solidCells, useCylinder ? "cylinder" : "sphere",
                radius, U, nu, hbar);
    std::printf("Re = U*D/nu = %.0f\n", U * 2.0 * radius / nu);
    std::printf("mode=%s  OMP_NUM_THREADS=%s  frames=%d (stats over last %d)\n\n", mode.c_str(),
                std::getenv("OMP_NUM_THREADS") ? std::getenv("OMP_NUM_THREADS") : "(unset)",
                frames, window);

    StageStat sCfl, sAdv, sDif, sPrj, sSim, sConn, sEig, sTrc, sExt;
    std::vector<cd> prevVec;
    std::size_t nFilaments = 0;
    GaugeLattice lastLat;  // last frame's operator, for the A/B block below
    std::vector<cd> lastVec;
    int lastIts = 0;

    for (int f = 0; f < frames; ++f) {
      const bool inWindow = f >= frames - window;
      // -- sim --
      double t0 = nowMs();
      const double Umax = std::max(1e-6, maxSpeed(grid, u));
      const double dtEff = std::min(0.1, cflTarget * h / Umax);
      // Give the solid band a plausible velocity before backtracing into it.
      // Without this, faces one layer outside the body sample the identically
      // zero interior and a sheet of dead fluid is injected every step.
      // BOCHNER_NO_EXTRAP=1 restores the old behaviour for A/B measurement.
      static const bool kNoExtrap = std::getenv("BOCHNER_NO_EXTRAP") != nullptr;
      if (!kNoExtrap) {
        const int band = static_cast<int>(std::ceil(cflTarget)) + 2;
        extrapolateIntoSolid(grid, u, solid, band);
      }
      double t1 = nowMs();
      FaceField w;
#ifdef BOCHNER_WITH_METAL
      if (useGpu)
        w = advectCovectorBFECCGpu(grid, u, u, dtEff);
      else
#endif
        w = advectCovectorBFECC(grid, u, u, dtEff);
      double t2 = nowMs();
      const double lim = 0.16 * h * h;
      const int nsub = std::max(1, int(std::ceil(nu * dtEff / lim)));
      for (int s = 0; s < nsub; ++s) w = ops::diffuseVelocity(grid, w, nu, dtEff / nsub, solid);
      double t3 = nowMs();
      u = proj.project(w, &cycles);
      double t4 = nowMs();

      // -- extraction (every frame, as in the GPU viewer) --
      const FaceField theta = connectionAngles(grid, u, hbar);
      const GaugeLattice lat = gaugeLatticeFromFaces(grid, theta);
      double t5 = nowMs();
      std::vector<cd> vec;
      int its = 0;
      if (useGpu) {
#ifdef BOCHNER_WITH_METAL
        std::vector<gpu::GaugeLevelData> levels;
        for (const GaugeLattice& L : buildGaugeLevels(lat))
          levels.push_back({L.lx, L.ly, L.lz, L.periodic, L.w, L.tx, L.ty, L.tz});
        const int handle = gpu::uploadGauge(levels);
        GaugeEigenOptions opts;
        opts.relativeGsDrop = false;  // live path: absolute-drop warm-start early exit
        const gpu::GaugeEigenGpu r =
            gpu::lobpcgSolveGauge(handle, prevVec, /*maxIters=*/100, /*tol=*/1e-4, opts.precCycles,
                                opts.mg.nu1, opts.mg.nu2, opts.mg.coarseSweeps, opts.mg.omega);
        gpu::freeGauge(handle);
        vec = r.vector;
        its = r.iterations;
#endif
      } else {
        GaugeEigenOptions opts;
        opts.relativeGsDrop = false;  // live path (same policy as the viewer's CPU solver)
        opts.tol = 1e-6;
        const GaugeEigenResult r =
            smallestEigenpairGaugeMG(lat, prevVec.empty() ? nullptr : &prevVec, opts);
        vec = r.vector;
        its = r.iterations;
      }
      if (!vec.empty()) prevVec = vec;
      double t6 = nowMs();
      const std::vector<double> psi = toInterleaved(vec);
      nFilaments = 0;
      for (auto& fl : linkFilaments(grid, traceZeroSet(grid, psi)))
        if (fl.points.size() >= 4) ++nFilaments;
      double t7 = nowMs();

      if (inWindow) {
        sCfl.add(t1 - t0);
        sAdv.add(t2 - t1);
        sDif.add(t3 - t2);
        sPrj.add(t4 - t3);
        sSim.add(t4 - t0);
        sConn.add(t5 - t4);
        sEig.add(t6 - t5);
        sTrc.add(t7 - t6);
        sExt.add(t7 - t4);
      }
      if (f == frames - 1) {
        lastLat = lat;
        lastVec = vec;
        lastIts = its;
      }
      if ((f + 1) % 50 == 0) {
        std::printf("  frame %4d: sim %.1f ms, extract %.1f ms, %zu filaments, eig its %d\n",
                    f + 1, t4 - t0, t7 - t4, nFilaments, its);
        std::fflush(stdout);
      }
    }

    std::printf("\nper-frame stages, median [min..max] over the last %d frames:\n", window);
    printStat("cfl", sCfl);
    printStat("advect", sAdv);
    printStat("diffuse", sDif);
    printStat("project", sPrj);
    printStat("SIM", sSim);
    printStat("connection", sConn);
    printStat("eigensolve", sEig);
    printStat("trace+link", sTrc);
    printStat("EXTRACT", sExt);
    const double frameMs = sSim.median() + sExt.median();
    std::printf("  full loop  %8.2f ms  = %.1f fps (extraction every frame)\n", frameMs,
                1000.0 / frameMs);

    // --- A/B on the final operator: CPU double covMG-LOBPCG and SLEPc Lanczos ---
    std::printf("\nA/B on the final frame's operator (%zu filaments traced):\n", nFilaments);
    {
      GaugeEigenOptions opts;
      opts.relativeGsDrop = false;
      opts.tol = 1e-6;
      const double a0 = nowMs();
      const GaugeEigenResult r =
          smallestEigenpairGaugeMG(lastLat, prevVec.empty() ? nullptr : &prevVec, opts);
      const double a1 = nowMs();
      std::printf("  CPU double covMG-LOBPCG (warm): %7.1f ms, %d its, lambda %.6f, line dist to %s"
                  " vec %.2e\n",
                  a1 - a0, r.iterations, r.eigenvalue, mode.c_str(), lineDistance(r.vector, lastVec));
    }
    {
      const FaceField thetaL = connectionAngles(grid, u, hbar);
      const CooMatrix E = connectionLaplacian(grid, thetaL);
      const std::vector<double> guess = toInterleaved(prevVec);
      const double b0 = nowMs();
      const EigenPair lp = smallestEigenpairLanczos(E, 1e-6, prevVec.empty() ? nullptr : &guess);
      const double b1 = nowMs();
      std::printf("  SLEPc Lanczos   (warm): %7.1f ms, %d its, lambda %.6f\n", b1 - b0,
                  lp.iterations, lp.value);
    }
    std::printf("  (%s eigensolve at the last frame: %d iterations)\n", mode.c_str(), lastIts);
  }
  SlepcFinalize();
  return 0;
}
