/// \file
/// Red-green TDD for the gauge-aware multigrid linear solver (GaugeMultigrid).
/// Anchors: (1) the matrix-free magnetic Laplacian agrees with the assembled
/// CooMatrix connectionLaplacian; (2) restriction is the exact adjoint of
/// prolongation (tested directly: <f, P c> = <R f, c> in double precision, for
/// the U(1) and SU(d) transfer pairs); (3) the V-cycle solves E x = b to
/// tolerance on a frustrated (SPD) connection; (4) the V-cycle iteration count
/// is ~mesh-independent (the multigrid payoff).
#include <doctest.h>

#include <cmath>
#include <complex>
#include <stdexcept>
#include <vector>

#include "grid/CooMatrix.h"
#include "solvers/GaugeMultigrid.h"
#include "solvers/SunGauge.h"
#include "extraction/MacConnectionLaplacian.h"
#include "grid/MacGrid.h"
#include "fluid/MacVortexRing.h"

using bochner::FaceField;
using bochner::GaugeLattice;
using bochner::MacGrid;
using bochner::Vec3;
using cd = std::complex<double>;

namespace {

// A frustrated connection: the seeded vortex ring (SPD connection Laplacian).
FaceField ringTheta(const MacGrid& g) {
  const double R = 0.7, Gamma = 1.0, hbar = Gamma / (2.0 * M_PI);
  const FaceField u = bochner::vortexRingFaceField(g, {0, 0, 0}, {0, 0, 1}, R, Gamma, 0.15);
  return bochner::connectionAngles(g, u, hbar);
}

// y = E x via the assembled CooMatrix (real 2x2 embedding), returned complex.
std::vector<cd> assembledApply(const bochner::CooMatrix& E, const std::vector<cd>& x) {
  std::vector<double> xr = bochner::toInterleaved(x), yr(xr.size(), 0.0);
  for (const auto& t : E.compressed()) yr[t.row] += t.value * xr[t.col];
  return bochner::toComplex(yr);
}

double l2(const std::vector<cd>& v) {
  double s = 0.0;
  for (const cd& z : v) s += std::norm(z);
  return std::sqrt(s);
}

// Hermitian inner product <a,b> = sum conj(a) b.
cd cdot(const std::vector<cd>& a, const std::vector<cd>& b) {
  cd s(0, 0);
  for (std::size_t i = 0; i < a.size(); ++i) s += std::conj(a[i]) * b[i];
  return s;
}

// A periodic (3-torus) uniform magnetic field of `nPhi` flux quanta through the
// x-y plane, in the lattice Landau/seam gauge -- the exact gauge of the sibling
// lattice-gauge-solvers' Examples::uniformField, expressed as bochner forward
// links: lkx[i,j,k] = -phi_p*j (the wrap link included), and the seam row
// lky[i, n-1, k] = 2*pi*nPhi*i/n; all other links zero.
bochner::GaugeLattice uniformFluxLattice(int n, int nPhi, double h) {
  const double phi_p = 2.0 * M_PI * nPhi / (double(n) * n);
  const std::size_t N = static_cast<std::size_t>(n) * n * n;
  std::vector<double> lkx(N, 0.0), lky(N, 0.0), lkz(N, 0.0);
  const auto idx = [&](int i, int j, int k) { return (i * n + j) * n + k; };
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      for (int k = 0; k < n; ++k) {
        lkx[idx(i, j, k)] = -phi_p * j;
        if (j == n - 1) lky[idx(i, j, k)] = 2.0 * M_PI * nPhi * i / double(n);
      }
  return bochner::gaugeLatticePeriodic(n, n, n, 1.0 / (h * h), lkx, lky, lkz);
}

}  // namespace

TEST_CASE("matrix-free magnetic Laplacian matches the assembled CooMatrix") {
  const MacGrid g(6, 5, 4, 0.3, Vec3{-0.9, -0.75, -0.6});
  const FaceField theta = ringTheta(g);
  const GaugeLattice lat = bochner::gaugeLatticeFromFaces(g, theta);
  const auto E = bochner::connectionLaplacian(g, theta);

  // A deterministic non-trivial test vector.
  std::vector<cd> x(g.numCells());
  for (int c = 0; c < g.numCells(); ++c) x[c] = cd(std::cos(0.7 * c), std::sin(0.3 * c + 1.0));

  const std::vector<cd> yFree = bochner::applyConnectionLaplacian(lat, x);
  const std::vector<cd> yMat = assembledApply(E, x);

  REQUIRE(yFree.size() == yMat.size());
  double maxDiff = 0.0;
  for (std::size_t i = 0; i < yFree.size(); ++i)
    maxDiff = std::max(maxDiff, std::abs(yFree[i] - yMat[i]));
  CHECK(maxDiff < 1e-12);
}

TEST_CASE("V-cycle solves E x = b to tolerance on a frustrated connection") {
  const MacGrid g(16, 16, 16, 0.1, Vec3{-0.8, -0.8, -0.8});
  const GaugeLattice lat = bochner::gaugeLatticeFromFaces(g, ringTheta(g));

  // Manufacture a solution: pick x*, set b = E x*, solve from zero, recover x*.
  std::vector<cd> xstar(g.numCells());
  for (int c = 0; c < g.numCells(); ++c) xstar[c] = cd(std::cos(0.11 * c), std::sin(0.07 * c));
  const std::vector<cd> b = bochner::applyConnectionLaplacian(lat, xstar);

  std::vector<cd> x(g.numCells(), cd(0.0, 0.0));
  bochner::MgOptions opts;
  opts.tol = 1e-8;
  const bochner::MgResult res = bochner::vcycleSolve(lat, b, x, opts);

  INFO("cycles=" << res.cycles << " relResidual=" << res.relResidual);
  CHECK(res.relResidual < 1e-8);
  CHECK(res.cycles < opts.maxCycles);  // converged, not capped

  // x matches x* up to the operator's nullspace-free SPD solution (unique here).
  std::vector<cd> diff(x.size());
  for (std::size_t i = 0; i < x.size(); ++i) diff[i] = x[i] - xstar[i];
  CHECK(l2(diff) / l2(xstar) < 1e-6);
}

TEST_CASE("CG baseline solves E x = b to tolerance") {
  const MacGrid g(16, 16, 16, 0.1, Vec3{-0.8, -0.8, -0.8});
  const GaugeLattice lat = bochner::gaugeLatticeFromFaces(g, ringTheta(g));
  std::vector<cd> xstar(g.numCells());
  for (int c = 0; c < g.numCells(); ++c) xstar[c] = cd(std::cos(0.11 * c), std::sin(0.07 * c));
  const std::vector<cd> b = bochner::applyConnectionLaplacian(lat, xstar);

  std::vector<cd> x(g.numCells(), cd(0.0, 0.0));
  const bochner::SolveStats st = bochner::cgSolve(lat, b, x, 1e-8);
  INFO("CG iterations=" << st.iterations << " relResidual=" << st.relResidual);
  CHECK(st.relResidual < 1e-8);
  std::vector<cd> diff(x.size());
  for (std::size_t i = 0; i < x.size(); ++i) diff[i] = x[i] - xstar[i];
  CHECK(l2(diff) / l2(xstar) < 1e-6);
}

TEST_CASE("CG with an exact warm start returns cleanly (no 0/0 NaN)") {
  const MacGrid g(8, 8, 8, 0.2, Vec3{-0.8, -0.8, -0.8});
  const GaugeLattice lat = bochner::gaugeLatticeFromFaces(g, ringTheta(g));
  std::vector<cd> xstar(g.numCells());
  for (int c = 0; c < g.numCells(); ++c) xstar[c] = cd(std::cos(0.11 * c), std::sin(0.07 * c));
  const std::vector<cd> b = bochner::applyConnectionLaplacian(lat, xstar);

  // Warm-start x AT the solution: the initial residual is ~0, so the first search
  // direction p = r is null (pAp ~ 0). Without the pAp>0 guard, alpha = rs/pAp is
  // 0/0 = NaN and poisons x. The guard must short-circuit and return cleanly.
  std::vector<cd> x = xstar;
  const bochner::SolveStats st = bochner::cgSolve(lat, b, x, 1e-8);
  CHECK(std::isfinite(st.relResidual));
  CHECK(st.relResidual < 1e-8);
  for (const cd& z : x) {
    CHECK(std::isfinite(z.real()));
    CHECK(std::isfinite(z.imag()));
  }
}

TEST_CASE("periodic trivial connection Laplacian annihilates the constant") {
  // No flux (nPhi=0): every link phase 0, so E is the periodic graph Laplacian,
  // whose only null vector is the constant. E*const = 0; E*(non-constant) != 0.
  const bochner::GaugeLattice lat = uniformFluxLattice(8, /*nPhi=*/0, 0.1);
  std::vector<cd> ones(lat.numNodes(), cd(1.0, 0.0));
  const std::vector<cd> Eones = bochner::applyConnectionLaplacian(lat, ones);
  CHECK(l2(Eones) < 1e-12);

  std::vector<cd> v(lat.numNodes());
  for (std::size_t c = 0; c < v.size(); ++c) v[c] = cd(std::cos(0.9 * c), 0.0);
  CHECK(l2(bochner::applyConnectionLaplacian(lat, v)) > 1e-6);  // not in the null space
}

TEST_CASE("periodic uniform-flux connection Laplacian is Hermitian and SPD") {
  const bochner::GaugeLattice lat = uniformFluxLattice(8, /*nPhi=*/2, 0.1);
  std::vector<cd> x(lat.numNodes()), y(lat.numNodes());
  for (std::size_t c = 0; c < x.size(); ++c) {
    x[c] = cd(std::cos(0.7 * c), std::sin(0.3 * c + 1.0));
    y[c] = cd(std::sin(0.5 * c), std::cos(0.2 * c));
  }
  const std::vector<cd> Ex = bochner::applyConnectionLaplacian(lat, x);
  const std::vector<cd> Ey = bochner::applyConnectionLaplacian(lat, y);
  // Hermitian: <Ex,y> == <x,Ey>.
  CHECK(std::abs(cdot(Ex, y) - cdot(x, Ey)) < 1e-10);
  // SPD with flux (a real mass gap, no null space): <x,Ex> > 0 and real.
  const cd q = cdot(x, Ex);
  CHECK(q.real() > 1e-6);
  CHECK(std::abs(q.imag()) < 1e-10);
}

TEST_CASE("periodic uniform-flux V-cycle solves E x = b, ~mesh-independent") {
  bochner::MgOptions opts;
  opts.tol = 1e-8;
  int cyc8 = 0, cyc16 = 0;
  for (int n : {8, 16}) {
    const bochner::GaugeLattice lat = uniformFluxLattice(n, /*nPhi=*/2, 1.0 / n);
    std::vector<cd> xstar(lat.numNodes());
    for (std::size_t c = 0; c < xstar.size(); ++c)
      xstar[c] = cd(std::cos(0.11 * c), std::sin(0.07 * c));
    const std::vector<cd> b = bochner::applyConnectionLaplacian(lat, xstar);
    std::vector<cd> x(lat.numNodes(), cd(0.0, 0.0));
    const bochner::MgResult res = bochner::vcycleSolve(lat, b, x, opts);
    INFO("n=" << n << " cycles=" << res.cycles << " relResidual=" << res.relResidual);
    CHECK(res.relResidual < 1e-8);
    CHECK(res.cycles < opts.maxCycles);  // converged, not capped
    std::vector<cd> diff(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) diff[i] = x[i] - xstar[i];
    CHECK(l2(diff) / l2(xstar) < 1e-6);  // recovered the manufactured solution
    (n == 8 ? cyc8 : cyc16) = res.cycles;
  }
  MESSAGE("periodic uniform-flux V-cycles: n=8 -> " << cyc8 << ", n=16 -> " << cyc16);
  // Mesh independence, asserted at a strength that can actually fail. The old
  // bound (2x per doubling) IS the non-mesh-independent signature, so it could
  // not distinguish the claim from its negation. Measured: 6 -> 5 (the count
  // DECREASES); +1 leaves room for platform FP variation, nothing more.
  CHECK(cyc16 <= cyc8 + 1);
}

TEST_CASE("V-cycle iteration count is ~mesh-independent") {
  bochner::MgOptions opts;
  opts.tol = 1e-8;
  int cyc16 = 0, cyc32 = 0;
  for (int n : {16, 32}) {
    const MacGrid g(n, n, n, 1.6 / n, Vec3{-0.8, -0.8, -0.8});
    const GaugeLattice lat = bochner::gaugeLatticeFromFaces(g, ringTheta(g));
    std::vector<cd> xstar(g.numCells());
    for (int c = 0; c < g.numCells(); ++c) xstar[c] = cd(std::cos(0.11 * c), std::sin(0.07 * c));
    const std::vector<cd> b = bochner::applyConnectionLaplacian(lat, xstar);
    std::vector<cd> x(g.numCells(), cd(0.0, 0.0));
    const int cyc = bochner::vcycleSolve(lat, b, x, opts).cycles;
    (n == 16 ? cyc16 : cyc32) = cyc;
  }
  MESSAGE("V-cycles to 1e-8: n=16 -> " << cyc16 << ", n=32 -> " << cyc32);
  // Measured 14 -> 12: the count decreases under refinement. Assert that, not
  // the far weaker "does not double" (which a non-mesh-independent method
  // would also satisfy).
  CHECK(cyc32 <= cyc16 + 1);
}

TEST_CASE("subdivisionSectionFromLattice rejects an indivisible dimension") {
  // 6 is not divisible by 2^2 = 4 -> coarsen would truncate and prolong read OOB.
  const MacGrid g(6, 8, 8, 0.25, Vec3{-0.75, -0.75, -0.75});
  const GaugeLattice lat = bochner::gaugeLatticeFromFaces(g, ringTheta(g));
  CHECK_THROWS_AS(bochner::subdivisionSectionFromLattice(lat, 2), std::invalid_argument);
  CHECK_THROWS_AS(bochner::subdivisionSectionFromLattice(lat, -1), std::invalid_argument);
  // 8 is divisible by 4 along every used axis at 2 levels (lx=6 unused here): a
  // fully-divisible lattice still succeeds.
  const MacGrid g2(8, 8, 8, 0.25, Vec3{-1, -1, -1});
  const GaugeLattice lat2 = bochner::gaugeLatticeFromFaces(g2, ringTheta(g2));
  CHECK_NOTHROW(bochner::subdivisionSectionFromLattice(lat2, 2));
}

TEST_CASE("gaugeLatticePeriodic rejects an odd dimension") {
  // A periodic dim must be even: an odd dim wraps node (n-1) to node 0, whose
  // (i+j+k) parities are equal, so the red-black smoother would couple two
  // same-color nodes -> an OpenMP read/write race and an invalid Gauss-Seidel
  // bipartition. Reject it at construction.
  const std::vector<double> ok(4 * 4 * 4, 0.0);
  CHECK_NOTHROW(bochner::gaugeLatticePeriodic(4, 4, 4, 1.0, ok, ok, ok));
  const std::vector<double> odd(5 * 4 * 4, 0.0);
  CHECK_THROWS_AS(bochner::gaugeLatticePeriodic(5, 4, 4, 1.0, odd, odd, odd),
                  std::invalid_argument);
  const std::vector<double> oddz(4 * 4 * 3, 0.0);
  CHECK_THROWS_AS(bochner::gaugeLatticePeriodic(4, 4, 3, 1.0, oddz, oddz, oddz),
                  std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Per-edge weights (variable edge lengths / graded resolution)
// ---------------------------------------------------------------------------

namespace {

// Deterministic smoothly graded positive weights (~4x contrast), one entry per
// forward link, indexed like lkx/lky/lkz by the low node of the edge.
std::vector<double> gradedWeights(const GaugeLattice& L, int axis) {
  const long n = axis == 0 ? L.numLinksX() : axis == 1 ? L.numLinksY() : L.numLinksZ();
  std::vector<double> w(static_cast<std::size_t>(n));
  for (long e = 0; e < n; ++e) {
    const double t = std::sin(0.37 * e + 0.7 * axis);
    w[static_cast<std::size_t>(e)] = 1.0 + 3.0 * t * t;  // in [1, 4]
  }
  return w;
}

GaugeLattice withGradedWeights(GaugeLattice L) {
  L.setEdgeWeights(gradedWeights(L, 0), gradedWeights(L, 1), gradedWeights(L, 2));
  return L;
}

}  // namespace

TEST_CASE("setEdgeWeights validates sizes and positivity") {
  GaugeLattice lat = uniformFluxLattice(4, 1, 0.25);
  std::vector<double> ok(4 * 4 * 4, 1.0);
  CHECK_NOTHROW(lat.setEdgeWeights(ok, ok, ok));
  CHECK(lat.weighted());
  std::vector<double> tooShort(4 * 4 * 3, 1.0);
  CHECK_THROWS_AS(lat.setEdgeWeights(tooShort, ok, ok), std::invalid_argument);
  std::vector<double> zero(4 * 4 * 4, 1.0);
  zero[7] = 0.0;
  CHECK_THROWS_AS(lat.setEdgeWeights(ok, zero, ok), std::invalid_argument);
}

TEST_CASE("constant per-edge weights reproduce the uniform solver") {
  // Filling the weight arrays with the uniform w must give the same operator,
  // smoother, and V-cycle as the scalar-w path (up to roundoff reassociation).
  const int n = 8;
  const GaugeLattice uni = uniformFluxLattice(n, 2, 1.0 / n);
  GaugeLattice wtd = uni;
  const auto fill = [&](long cnt) { return std::vector<double>(static_cast<std::size_t>(cnt), uni.w); };
  wtd.setEdgeWeights(fill(uni.numLinksX()), fill(uni.numLinksY()), fill(uni.numLinksZ()));

  std::vector<cd> x(static_cast<std::size_t>(uni.numNodes()));
  for (std::size_t c = 0; c < x.size(); ++c) x[c] = cd(std::cos(0.7 * c), std::sin(0.3 * c + 1.0));
  const auto yu = bochner::applyConnectionLaplacian(uni, x);
  const auto yw = bochner::applyConnectionLaplacian(wtd, x);
  double scale = 0.0, diff = 0.0;
  for (std::size_t i = 0; i < yu.size(); ++i) {
    scale = std::max(scale, std::abs(yu[i]));
    diff = std::max(diff, std::abs(yu[i] - yw[i]));
  }
  CHECK(diff < 1e-12 * scale);

  std::vector<cd> b(x.size(), cd(1.0, 0.5)), xu(x.size(), cd(0, 0)), xw(x.size(), cd(0, 0));
  bochner::MgOptions opts;
  opts.tol = 1e-8;
  const auto ru = bochner::vcycleSolve(uni, b, xu, opts);
  const auto rw = bochner::vcycleSolve(wtd, b, xw, opts);
  CHECK(ru.cycles == rw.cycles);
  CHECK(rw.relResidual < opts.tol);
}

TEST_CASE("weighted connection Laplacian is Hermitian and PSD") {
  GaugeLattice lat = withGradedWeights(uniformFluxLattice(8, 3, 0.125));
  std::vector<cd> x(static_cast<std::size_t>(lat.numNodes())), y(x.size());
  for (std::size_t c = 0; c < x.size(); ++c) {
    x[c] = cd(std::cos(1.3 * c + 0.2), std::sin(0.9 * c));
    y[c] = cd(std::sin(0.5 * c + 1.1), std::cos(2.1 * c));
  }
  const auto Ex = bochner::applyConnectionLaplacian(lat, x);
  const auto Ey = bochner::applyConnectionLaplacian(lat, y);
  const cd lhs = cdot(x, Ey), rhs = cdot(Ex, y);
  CHECK(std::abs(lhs - rhs) < 1e-10 * std::abs(lhs));
  CHECK(cdot(x, Ex).real() > 0.0);  // frustrated (flux) connection: PD
  CHECK(std::abs(cdot(x, Ex).imag()) < 1e-10 * cdot(x, Ex).real());
}

TEST_CASE("weighted V-cycle solves E x = b; count stays ~mesh-independent") {
  // Graded weights (~4x contrast) on the uniform-flux torus: the weighted
  // covariant transfer + series-conductance coarsening must keep the flat
  // V-cycle behaviour of the uniform solver (spectral-equivalence constants
  // verified nearly unchanged in the scratchpad study).
  int cycles[2] = {0, 0};
  int idx = 0;
  for (int n : {8, 16}) {
    GaugeLattice lat = withGradedWeights(uniformFluxLattice(n, 2, 1.0 / n));
    std::vector<cd> b(static_cast<std::size_t>(lat.numNodes()), cd(1.0, -0.25));
    std::vector<cd> x(b.size(), cd(0, 0));
    bochner::MgOptions opts;
    opts.tol = 1e-8;
    const auto res = bochner::vcycleSolve(lat, b, x, opts);
    CHECK(res.relResidual < opts.tol);
    cycles[idx++] = res.cycles;
  }
  MESSAGE("weighted graded V-cycles: n=8 -> " << cycles[0] << ", n=16 -> " << cycles[1]);
  // Measured 8 -> 8 (flat). +3 on a base of 8 admitted ~40% growth.
  CHECK(cycles[1] <= cycles[0] + 1);

  // Open (Neumann) topology with graded weights converges too.
  const MacGrid g(16, 16, 16, 0.1, Vec3{-0.8, -0.8, -0.8});
  GaugeLattice lat = withGradedWeights(bochner::gaugeLatticeFromFaces(g, ringTheta(g)));
  std::vector<cd> b(static_cast<std::size_t>(lat.numNodes()), cd(0.5, 1.0));
  std::vector<cd> x(b.size(), cd(0, 0));
  bochner::MgOptions opts;
  opts.tol = 1e-8;
  const auto res = bochner::vcycleSolve(lat, b, x, opts);
  CHECK(res.relResidual < opts.tol);
}

// ---------------------------------------------------------------------------
// Restriction adjointness: <f, P c> = <R f, c> in double precision. This pins
// the transfer pair DIRECTLY: the alpha-scaled coarse correction guarantees
// descent for any transfer, so a scaling or conjugation error confined to
// restrict/prolong would still pass every solve-to-tolerance test above (the
// mesh-independence checks carry 2x slack). Until now the only explicit
// adjointness test was the float Metal one, gated behind BOCHNER_WITH_METAL.
// ---------------------------------------------------------------------------

namespace {

// Deterministic dense complex test vector (generic phases: nothing affine, so
// no transfer reproduces it exactly).
std::vector<cd> denseVec(std::size_t n, double a, double b) {
  std::vector<cd> v(n);
  for (std::size_t c = 0; c < n; ++c)
    v[c] = cd(std::cos(a * c + 0.3), std::sin(b * c + 1.1));
  return v;
}

// Graded positive weights by link count (the SunLattice sibling of gradedWeights).
std::vector<double> gradedByCount(long cnt, int axis) {
  std::vector<double> w(static_cast<std::size_t>(cnt));
  for (long e = 0; e < cnt; ++e) {
    const double t = std::sin(0.37 * e + 0.7 * axis);
    w[static_cast<std::size_t>(e)] = 1.0 + 3.0 * t * t;  // in [1, 4]
  }
  return w;
}

// Link-array index helpers, mirroring GaugeMultigrid.cpp's xi/yi/zi (the y and z
// strides differ from x because the open lattice drops the wrap link on that
// axis only).
std::size_t lxi(const GaugeLattice& L, int i, int j, int k) {
  return static_cast<std::size_t>((i * L.ly + j) * L.lz + k);
}
std::size_t lyi(const GaugeLattice& L, int i, int j, int k) {
  return static_cast<std::size_t>((i * (L.periodic ? L.ly : L.ly - 1) + j) * L.lz + k);
}
std::size_t lzi(const GaugeLattice& L, int i, int j, int k) {
  return static_cast<std::size_t>((i * L.ly + j) * (L.periodic ? L.lz : L.lz - 1) + k);
}

/// Apply a U(1) gauge transform \f$g(v) = e^{i\varphi(v)}\f$: the forward link
/// angle (low->high) picks up \f$\varphi(\text{high}) - \varphi(\text{low})\f$.
/// The connection Laplacian becomes \f$G E G^H\f$ -- unitarily equivalent -- and
/// gauge *covariance* of the whole method (transfers included) is the identity
/// the contribution rests on.
GaugeLattice gaugeTransform(const GaugeLattice& f, const std::vector<double>& phi) {
  GaugeLattice t = f;
  const int lastI = f.periodic ? f.lx : f.lx - 1;
  const int lastJ = f.periodic ? f.ly : f.ly - 1;
  const int lastK = f.periodic ? f.lz : f.lz - 1;
  for (int i = 0; i < lastI; ++i)
    for (int j = 0; j < f.ly; ++j)
      for (int k = 0; k < f.lz; ++k)
        t.lkx[lxi(f, i, j, k)] +=
            phi[f.index((i + 1) % f.lx, j, k)] - phi[f.index(i, j, k)];
  for (int i = 0; i < f.lx; ++i)
    for (int j = 0; j < lastJ; ++j)
      for (int k = 0; k < f.lz; ++k)
        t.lky[lyi(f, i, j, k)] +=
            phi[f.index(i, (j + 1) % f.ly, k)] - phi[f.index(i, j, k)];
  for (int i = 0; i < f.lx; ++i)
    for (int j = 0; j < f.ly; ++j)
      for (int k = 0; k < lastK; ++k)
        t.lkz[lzi(f, i, j, k)] +=
            phi[f.index(i, j, (k + 1) % f.lz)] - phi[f.index(i, j, k)];
  t.buildTransports();
  return t;
}

// A deterministic pseudo-random gauge -- an arbitrary phase per node, so nothing
// about the transform is aligned with the lattice or the connection.
std::vector<double> randomGauge(const GaugeLattice& L, double seed) {
  std::vector<double> phi(static_cast<std::size_t>(L.numNodes()));
  for (std::size_t v = 0; v < phi.size(); ++v)
    phi[v] = 2.3 * std::sin(0.731 * static_cast<double>(v) + seed) +
             1.7 * std::cos(0.194 * static_cast<double>(v) - seed);
  return phi;
}

// Multiply componentwise by g = e^{i phi} sampled at the given node stride: for
// the fine vector stride 1, for the coarse vector the coarse node (I,J,K) is the
// fine node (2I,2J,2K) -- the same identification `coarsen` uses when it sums
// the two fine link angles into one coarse link.
std::vector<cd> applyGaugeFine(const GaugeLattice&, const std::vector<double>& phi,
                               const std::vector<cd>& v) {
  std::vector<cd> out(v.size());
  for (std::size_t i = 0; i < v.size(); ++i) out[i] = std::polar(1.0, phi[i]) * v[i];
  return out;
}

std::vector<cd> applyGaugeCoarse(const GaugeLattice& L, const std::vector<double>& phi,
                                 const std::vector<cd>& v) {
  const int cx = L.lx / 2, cy = L.ly / 2, cz = L.lz / 2;
  std::vector<cd> out(v.size());
  for (int I = 0; I < cx; ++I)
    for (int J = 0; J < cy; ++J)
      for (int K = 0; K < cz; ++K) {
        const std::size_t ci = static_cast<std::size_t>((I * cy + J) * cz + K);
        out[ci] = std::polar(1.0, phi[L.index(2 * I, 2 * J, 2 * K)]) * v[ci];
      }
  return out;
}

}  // namespace

// The central identity of the method: the covariant-subdivision prolongation
// commutes with a gauge transform, so the whole hierarchy is gauge covariant and
// the solver's behaviour cannot depend on the gauge. Adjointness (the test
// below) is a different property and does not imply this one: plain scalar
// averaging is exactly self-adjoint too, and it is precisely the transfer that
// is *not* covariant -- which is why scalar AMG fails on this operator.
TEST_CASE("U(1) prolongation is gauge covariant: P' (g_c c) = g_f (P c)") {
  const MacGrid g(8, 8, 8, 0.2, Vec3{-0.8, -0.8, -0.8});
  const GaugeLattice open = bochner::gaugeLatticeFromFaces(g, ringTheta(g));
  const GaugeLattice torus = uniformFluxLattice(8, /*nPhi=*/2, 0.125);
  for (const GaugeLattice* base : {&open, &torus})
    for (const bool weighted : {false, true}) {
      GaugeLattice lat = *base;
      if (weighted)
        lat.setEdgeWeights(gradedWeights(lat, 0), gradedWeights(lat, 1), gradedWeights(lat, 2));
      const std::vector<double> phi = randomGauge(lat, 0.41);
      const GaugeLattice lat2 = gaugeTransform(lat, phi);
      const std::size_t nc = static_cast<std::size_t>(lat.lx / 2) * (lat.ly / 2) * (lat.lz / 2);
      const auto c = denseVec(nc, 1.3, 0.9);

      const std::vector<cd> lhs = bochner::prolongGauge(lat2, applyGaugeCoarse(lat, phi, c));
      const std::vector<cd> rhs = applyGaugeFine(lat, phi, bochner::prolongGauge(lat, c));
      REQUIRE(lhs.size() == rhs.size());
      double err = 0.0;
      for (std::size_t i = 0; i < lhs.size(); ++i) err = std::max(err, std::abs(lhs[i] - rhs[i]));
      INFO("periodic=" << lat.periodic << " weighted=" << weighted << " maxerr=" << err
                       << " scale=" << l2(rhs));
      CHECK(err < 1e-12 * std::max(1.0, l2(rhs)));

      // Restriction inherits covariance from P via R = P^H, but pin it directly
      // so a future change to restrict() cannot break it silently.
      const auto f = denseVec(static_cast<std::size_t>(lat.numNodes()), 0.7, 0.4);
      const std::vector<cd> rl = bochner::restrictGauge(lat2, applyGaugeFine(lat, phi, f));
      const std::vector<cd> rr = applyGaugeCoarse(lat, phi, bochner::restrictGauge(lat, f));
      double rerr = 0.0;
      for (std::size_t i = 0; i < rl.size(); ++i) rerr = std::max(rerr, std::abs(rl[i] - rr[i]));
      INFO("restrict maxerr=" << rerr);
      CHECK(rerr < 1e-12 * std::max(1.0, l2(rr)));
    }
}

// The invariance with teeth. lambda_min alone is a weak check -- it is invariant
// for *any* correct eigensolver by unitary equivalence, gauge-covariant transfers
// or not. What actually distinguishes a covariant hierarchy is that the solver
// takes the SAME PATH: identical V-cycle counts on the gauge-transformed
// operator. A non-covariant transfer would still land on the right eigenvalue,
// just more slowly (or not at all).
TEST_CASE("U(1) V-cycle iteration count is gauge invariant, not just the spectrum") {
  const MacGrid g(16, 16, 16, 0.1, Vec3{-0.8, -0.8, -0.8});
  GaugeLattice lat = bochner::gaugeLatticeFromFaces(g, ringTheta(g));
  const std::vector<double> phi = randomGauge(lat, 1.27);
  const GaugeLattice lat2 = gaugeTransform(lat, phi);

  const std::size_t n = static_cast<std::size_t>(lat.numNodes());
  const auto b = denseVec(n, 0.55, 0.31);
  bochner::MgOptions opts;
  opts.tol = 1e-10;
  opts.maxCycles = 100;

  std::vector<cd> x1(n, cd(0, 0));
  const bochner::MgResult r1 = bochner::vcycleSolve(lat, b, x1, opts);
  // Same RHS in the transformed gauge, so the two solves are the *same* problem.
  std::vector<cd> x2(n, cd(0, 0));
  const bochner::MgResult r2 = bochner::vcycleSolve(lat2, applyGaugeFine(lat, phi, b), x2, opts);

  INFO("cycles " << r1.cycles << " vs " << r2.cycles << ", relRes " << r1.relResidual << " vs "
                 << r2.relResidual);
  CHECK(r1.cycles == r2.cycles);
  CHECK(r1.relResidual == doctest::Approx(r2.relResidual).epsilon(1e-9));

  // ...and the solution itself is the gauge-rotated one, to solver tolerance.
  const std::vector<cd> want = applyGaugeFine(lat, phi, x1);
  double err = 0.0;
  for (std::size_t i = 0; i < n; ++i) err = std::max(err, std::abs(x2[i] - want[i]));
  INFO("solution maxerr=" << err << " scale=" << l2(want));
  CHECK(err < 1e-8 * std::max(1.0, l2(want)));
}

TEST_CASE("U(1) restriction is the exact adjoint of prolongation") {
  const MacGrid g(8, 8, 8, 0.2, Vec3{-0.8, -0.8, -0.8});
  const GaugeLattice open = bochner::gaugeLatticeFromFaces(g, ringTheta(g));
  const GaugeLattice torus = uniformFluxLattice(8, /*nPhi=*/2, 0.125);
  for (const GaugeLattice* base : {&open, &torus})
    for (const bool weighted : {false, true}) {
      GaugeLattice lat = *base;
      if (weighted)
        lat.setEdgeWeights(gradedWeights(lat, 0), gradedWeights(lat, 1), gradedWeights(lat, 2));
      const std::size_t nf = static_cast<std::size_t>(lat.numNodes());
      const std::size_t nc = static_cast<std::size_t>(lat.lx / 2) * (lat.ly / 2) * (lat.lz / 2);
      const auto f = denseVec(nf, 0.7, 0.4);
      const auto c = denseVec(nc, 1.3, 0.9);
      const cd lhs = cdot(f, bochner::prolongGauge(lat, c));
      const cd rhs = cdot(bochner::restrictGauge(lat, f), c);
      INFO("periodic=" << lat.periodic << " weighted=" << weighted << " lhs=" << lhs
                       << " rhs=" << rhs);
      CHECK(std::abs(lhs - rhs) < 1e-12 * std::max(1.0, std::abs(lhs)));
    }
}

TEST_CASE("SU(d) restriction is the exact adjoint (P^H, not P^T) of prolongation") {
  // Random SU(d) links: any transpose-vs-adjoint slip in the matrix transports
  // breaks the identity at O(1), unlike the U(1) case where conj == adjoint on
  // scalars makes some slips invisible.
  for (const int d : {2, 3})
    for (const bool periodic : {false, true})
      for (const bool weighted : {false, true}) {
        bochner::SunLattice lat =
            bochner::randomSunLattice(d, 8, 8, 8, 1.0, 0.5, /*seed=*/17u + d, periodic);
        if (weighted)
          lat.setEdgeWeights(gradedByCount(lat.numLinksX(), 0), gradedByCount(lat.numLinksY(), 1),
                             gradedByCount(lat.numLinksZ(), 2));
        const std::size_t nf = static_cast<std::size_t>(lat.dof());
        const std::size_t nc =
            static_cast<std::size_t>(lat.lx / 2) * (lat.ly / 2) * (lat.lz / 2) * d;
        const auto f = denseVec(nf, 0.7, 0.4);
        const auto c = denseVec(nc, 1.3, 0.9);
        const cd lhs = cdot(f, bochner::prolongSun(lat, c));
        const cd rhs = cdot(bochner::restrictSun(lat, f), c);
        INFO("d=" << d << " periodic=" << periodic << " weighted=" << weighted << " lhs=" << lhs
                  << " rhs=" << rhs);
        CHECK(std::abs(lhs - rhs) < 1e-12 * std::max(1.0, std::abs(lhs)));
      }
}

TEST_CASE("weighted V-cycle beats CG on the graded-weight torus") {
  // The point of the weighted extension: the gauge-aware hierarchy keeps its
  // advantage when the lattice is no longer combinatorial.
  GaugeLattice lat = withGradedWeights(uniformFluxLattice(16, 2, 1.0 / 16));
  std::vector<cd> b(static_cast<std::size_t>(lat.numNodes()), cd(1.0, 0.0));
  std::vector<cd> xmg(b.size(), cd(0, 0)), xcg(b.size(), cd(0, 0));
  bochner::MgOptions opts;
  opts.tol = 1e-8;
  const auto mg = bochner::vcycleSolve(lat, b, xmg, opts);
  const auto cg = bochner::cgSolve(lat, b, xcg, 1e-8);
  MESSAGE("graded torus n=16: MG " << mg.cycles << " cycles vs CG " << cg.iterations << " iters");
  CHECK(mg.relResidual < 1e-8);
  CHECK(cg.relResidual < 1e-8);
  CHECK(mg.cycles < cg.iterations);
}
