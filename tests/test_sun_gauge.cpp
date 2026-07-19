/// \file
/// Correctness anchors for the non-abelian SU(d) gauge multigrid (SunGauge),
/// the paper's transferability demonstration. The claim is that the closed-form
/// gauge-aware multigrid carries from U(1) to SU(d) by replacing scalar
/// transport with matrix-vector and conjugate with Hermitian adjoint; these
/// tests pin that:
///   (1) d=1 reproduces the trusted scalar applyConnectionLaplacian exactly;
///   (2) an identity (trivial) connection decouples into d scalar Laplacians;
///   (3) the operator is Hermitian and positive (semi-)definite;
///   (4) the V-cycle solves E x = b to tolerance on a random SU(d) field;
///   (5) covMG-LOBPCG returns a certified smallest eigenpair (and m^2 on a trivial one).
#include <doctest.h>

#include <complex>
#include <random>
#include <vector>

#include "solvers/GaugeMultigrid.h"
#include "solvers/SunGauge.h"

using namespace bochner;
using cd = std::complex<double>;

namespace {

std::vector<cd> randomVec(long n, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::normal_distribution<double> g(0.0, 1.0);
  std::vector<cd> v(n);
  for (auto& z : v) z = cd(g(rng), g(rng));
  return v;
}

cd cdot(const std::vector<cd>& a, const std::vector<cd>& b) {
  cd s(0, 0);
  for (size_t i = 0; i < a.size(); ++i) s += std::conj(a[i]) * b[i];
  return s;
}
double maxAbsDiff(const std::vector<cd>& a, const std::vector<cd>& b) {
  double m = 0.0;
  for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::abs(a[i] - b[i]));
  return m;
}

// Assemble the operator as a dense m x m matrix (column i = E e_i) -- exact,
// used only at tiny n as an independent reference.
std::vector<cd> assembleDense(const SunLattice& L) {
  const int m = static_cast<int>(L.dof());
  std::vector<cd> E(static_cast<size_t>(m) * m);
  std::vector<cd> e(m, cd(0, 0));
  for (int i = 0; i < m; ++i) {
    e[i] = cd(1, 0);
    const std::vector<cd> col = applySunLaplacian(L, e);
    for (int r = 0; r < m; ++r) E[static_cast<size_t>(r) * m + i] = col[r];
    e[i] = cd(0, 0);
  }
  return E;
}

// Smallest eigenvalue of a dense m x m complex-Hermitian matrix by cyclic
// Hermitian Jacobi (computes the whole spectrum; returns the minimum). An
// independent oracle for the covMG-LOBPCG smallest eigenvalue -- stronger than Lanczos
// here (exact full spectrum, no iterative approximation, no dependency).
double smallestEigenvalueDense(std::vector<cd> D, int m) {
  for (int sweep = 0; sweep < 100; ++sweep) {
    double off = 0.0;
    for (int p = 0; p < m; ++p)
      for (int q = p + 1; q < m; ++q) off += std::norm(D[static_cast<size_t>(p) * m + q]);
    if (off < 1e-26) break;
    for (int p = 0; p < m; ++p)
      for (int q = p + 1; q < m; ++q) {
        const cd dpq = D[static_cast<size_t>(p) * m + q];
        if (std::abs(dpq) < 1e-18) continue;
        const double a = D[static_cast<size_t>(p) * m + p].real();
        const double b = D[static_cast<size_t>(q) * m + q].real();
        const double mag = std::abs(dpq);
        const cd g = std::exp(cd(0.0, -std::arg(dpq)));  // makes the off-diagonal real
        const double theta = 0.5 * std::atan2(2.0 * mag, a - b);
        const double cc = std::cos(theta), ss = std::sin(theta);
        const cd U00(cc, 0.0), U01(-ss, 0.0), U10 = g * ss, U11 = g * cc;
        const cd c00 = std::conj(U00), c01 = std::conj(U01), c10 = std::conj(U10), c11 = std::conj(U11);
        for (int i = 0; i < m; ++i) {  // D <- D U (columns p,q)
          const cd dip = D[static_cast<size_t>(i) * m + p], diq = D[static_cast<size_t>(i) * m + q];
          D[static_cast<size_t>(i) * m + p] = dip * U00 + diq * U10;
          D[static_cast<size_t>(i) * m + q] = dip * U01 + diq * U11;
        }
        for (int j = 0; j < m; ++j) {  // D <- U^H D (rows p,q)
          const cd dpj = D[static_cast<size_t>(p) * m + j], dqj = D[static_cast<size_t>(q) * m + j];
          D[static_cast<size_t>(p) * m + j] = c00 * dpj + c10 * dqj;
          D[static_cast<size_t>(q) * m + j] = c01 * dpj + c11 * dqj;
        }
      }
  }
  double lam = D[0].real();
  for (int i = 1; i < m; ++i) lam = std::min(lam, D[static_cast<size_t>(i) * m + i].real());
  return lam;
}

}  // namespace

TEST_CASE("SunGauge d=1 reproduces the scalar U(1) connection Laplacian") {
  const int n = 8;
  const double w = 1.3;
  std::mt19937_64 rng(2024);
  std::uniform_real_distribution<double> u(-M_PI, M_PI);
  const long links = static_cast<long>(n) * n * n;  // periodic: one forward link per node per axis
  std::vector<double> lkx(links), lky(links), lkz(links);
  for (long e = 0; e < links; ++e) {
    lkx[e] = u(rng);
    lky[e] = u(rng);
    lkz[e] = u(rng);
  }
  const GaugeLattice scalar = gaugeLatticePeriodic(n, n, n, w, lkx, lky, lkz);

  SunLattice sun;
  sun.d = 1;
  sun.lx = sun.ly = sun.lz = n;
  sun.periodic = true;
  sun.w = w;
  sun.ux.resize(links);
  sun.uy.resize(links);
  sun.uz.resize(links);
  for (long e = 0; e < links; ++e) {
    sun.ux[e] = cd(std::cos(lkx[e]), std::sin(lkx[e]));
    sun.uy[e] = cd(std::cos(lky[e]), std::sin(lky[e]));
    sun.uz[e] = cd(std::cos(lkz[e]), std::sin(lkz[e]));
  }

  const std::vector<cd> x = randomVec(links, 7);
  const std::vector<cd> ys = applyConnectionLaplacian(scalar, x);
  const std::vector<cd> yn = applySunLaplacian(sun, x);
  CHECK(maxAbsDiff(ys, yn) < 1e-12);
}

TEST_CASE("SunGauge identity connection decouples into d scalar Laplacians") {
  const int n = 8, d = 3;
  const double w = 1.0;
  const SunLattice L = identitySunLattice(d, n, n, n, w, /*mass2=*/0.0);
  // Trivial scalar lattice (all link angles zero).
  const long links = static_cast<long>(n) * n * n;
  const GaugeLattice trivial =
      gaugeLatticePeriodic(n, n, n, w, std::vector<double>(links, 0.0), std::vector<double>(links, 0.0),
                           std::vector<double>(links, 0.0));
  const long N = L.numNodes();
  const std::vector<cd> x = randomVec(L.dof(), 11);
  const std::vector<cd> y = applySunLaplacian(L, x);
  for (int a = 0; a < d; ++a) {
    std::vector<cd> fa(N), expect;
    for (long c = 0; c < N; ++c) fa[c] = x[c * d + a];
    expect = applyConnectionLaplacian(trivial, fa);
    for (long c = 0; c < N; ++c) CHECK(std::abs(y[c * d + a] - expect[c]) < 1e-12);
  }
}

TEST_CASE("SunGauge SU(3) operator is Hermitian and positive semidefinite") {
  const SunLattice L = randomSunLattice(3, 8, 8, 8, /*w=*/1.0, /*mass2=*/0.0, /*seed=*/5);
  const std::vector<cd> x = randomVec(L.dof(), 1), yv = randomVec(L.dof(), 2);
  const std::vector<cd> Ex = applySunLaplacian(L, x), Ey = applySunLaplacian(L, yv);
  // <x, E y> == conj(<y, E x>)  (Hermitian).
  const cd a = cdot(x, Ey), b = cdot(yv, Ex);
  CHECK(std::abs(a - std::conj(b)) < 1e-10);
  // <x, E x> real and >= 0  (PSD).
  const cd q = cdot(x, Ex);
  CHECK(std::abs(q.imag()) < 1e-10);
  CHECK(q.real() >= -1e-10);
}

TEST_CASE("SunGauge V-cycle solves E x = b on a random SU(2) field") {
  const SunLattice L = randomSunLattice(2, 16, 16, 16, /*w=*/1.0, /*mass2=*/0.2, /*seed=*/42);
  const std::vector<cd> b = randomVec(L.dof(), 314);
  std::vector<cd> x(L.dof(), cd(0, 0));
  MgOptions mg;
  mg.tol = 1e-8;
  mg.maxCycles = 100;
  const MgResult r = vcycleSolveSun(L, b, x, mg);
  CHECK(r.relResidual < 1e-8);
  CHECK(r.cycles < 40);  // gauge-aware => few cycles (mesh-independent)
}

TEST_CASE("SunGauge covMG-LOBPCG returns a certified smallest eigenpair (SU(2))") {
  const SunLattice L = randomSunLattice(2, 16, 16, 16, /*w=*/1.0, /*mass2=*/0.2, /*seed=*/9);
  // A maximally-disordered ("hot") random field is a deliberately hard,
  // clustered eigenproblem; 1e-6 relative eigen-residual is a solid certificate.
  GaugeEigenOptions eo;
  eo.tol = 1e-6;
  eo.maxIters = 200;
  const GaugeEigenResult res = smallestEigenpairSunMG(L, nullptr, eo);
  CHECK(res.converged);
  CHECK(res.residual < 1e-6);
  CHECK(res.eigenvalue > 0.0);
  // It is the SMALLEST: no random unit vector has a lower Rayleigh quotient.
  for (std::uint64_t s = 100; s < 106; ++s) {
    std::vector<cd> v = randomVec(L.dof(), s);
    const double nv = std::sqrt(cdot(v, v).real());
    for (auto& z : v) z /= nv;
    const double rq = cdot(v, applySunLaplacian(L, v)).real();
    CHECK(res.eigenvalue <= rq + 1e-6);
  }
}

TEST_CASE("SunGauge covMG-LOBPCG finds lambda = m^2 on a trivial connection") {
  const double m2 = 0.5;
  const SunLattice L = identitySunLattice(2, 8, 8, 8, /*w=*/1.0, m2);
  GaugeEigenOptions eo;
  eo.tol = 1e-8;
  eo.maxIters = 200;
  const GaugeEigenResult res = smallestEigenpairSunMG(L, nullptr, eo);
  // Trivial connection => d copies of (-Delta + m^2) periodic; the constant mode
  // has zero Dirichlet energy, so lambda_min = m^2 exactly.
  CHECK(std::abs(res.eigenvalue - m2) < 1e-5);
}

TEST_CASE("SunGauge V-cycle and CG converge to the same solution vector (SU(3))") {
  const SunLattice L = randomSunLattice(3, 16, 16, 16, /*w=*/1.0, /*mass2=*/0.3, /*seed=*/71);
  const std::vector<cd> b = randomVec(L.dof(), 2718);
  MgOptions mg;
  mg.tol = 1e-10;
  mg.maxCycles = 200;
  std::vector<cd> xmg(L.dof(), cd(0, 0));
  const MgResult mr = vcycleSolveSun(L, b, xmg, mg);
  std::vector<cd> xcg(L.dof(), cd(0, 0));
  const SolveStats cg = cgSolveSun(L, b, xcg, 1e-10, 20000);
  REQUIRE(mr.relResidual < 1e-10);
  REQUIRE(cg.relResidual < 1e-10);
  // Two independent solvers of the same SPD system must land on the same vector.
  double num = 0.0, den = 0.0;
  for (size_t i = 0; i < xmg.size(); ++i) {
    num += std::norm(xmg[i] - xcg[i]);
    den += std::norm(xcg[i]);
  }
  CHECK(std::sqrt(num / den) < 1e-6);
}

TEST_CASE("SunGauge covMG-LOBPCG matches a dense reference eigenvalue at tiny n") {
  for (int d : {2, 3}) {
    const SunLattice L = randomSunLattice(d, 4, 4, 4, /*w=*/1.0, /*mass2=*/0.3, /*seed=*/13);
    const double denseMin = smallestEigenvalueDense(assembleDense(L), static_cast<int>(L.dof()));
    GaugeEigenOptions eo;
    eo.tol = 1e-9;
    eo.maxIters = 400;
    const GaugeEigenResult res = smallestEigenpairSunMG(L, nullptr, eo);
    // covMG-LOBPCG must find the TRUE smallest eigenvalue (whole spectrum from the dense
    // reference), not merely a certified eigenpair.
    CHECK(std::abs(res.eigenvalue - denseMin) < 1e-6);
  }
}

// ---------------------------------------------------------------------------
// Per-edge weights (variable edge lengths), SU(d)
// ---------------------------------------------------------------------------

namespace {
std::vector<double> gradedW(long cnt, int axis, double base) {
  std::vector<double> w(static_cast<std::size_t>(cnt));
  for (long e = 0; e < cnt; ++e) {
    const double t = std::sin(0.37 * e + 0.7 * axis);
    w[static_cast<std::size_t>(e)] = base * (1.0 + 3.0 * t * t);
  }
  return w;
}
}  // namespace

TEST_CASE("weighted SunGauge d=1 reproduces the weighted scalar U(1) operator") {
  // The same graded weights installed on both fibers must give the same
  // operator -- pins the SU(d) weighted path to the independently tested U(1)
  // weighted path.
  const int n = 8;
  const double w = 1.3;
  std::mt19937_64 rng(2024);
  std::uniform_real_distribution<double> u(-M_PI, M_PI);
  const long links = static_cast<long>(n) * n * n;
  std::vector<double> lkx(links), lky(links), lkz(links);
  for (long e = 0; e < links; ++e) {
    lkx[e] = u(rng);
    lky[e] = u(rng);
    lkz[e] = u(rng);
  }
  GaugeLattice scalar = gaugeLatticePeriodic(n, n, n, w, lkx, lky, lkz);
  scalar.setEdgeWeights(gradedW(links, 0, w), gradedW(links, 1, w), gradedW(links, 2, w));

  SunLattice sun;
  sun.d = 1;
  sun.lx = sun.ly = sun.lz = n;
  sun.periodic = true;
  sun.w = w;
  sun.ux.resize(links);
  sun.uy.resize(links);
  sun.uz.resize(links);
  for (long e = 0; e < links; ++e) {
    sun.ux[e] = cd(std::cos(lkx[e]), std::sin(lkx[e]));
    sun.uy[e] = cd(std::cos(lky[e]), std::sin(lky[e]));
    sun.uz[e] = cd(std::cos(lkz[e]), std::sin(lkz[e]));
  }
  sun.setEdgeWeights(gradedW(links, 0, w), gradedW(links, 1, w), gradedW(links, 2, w));

  const std::vector<cd> x = randomVec(links, 7);
  const std::vector<cd> ys = applyConnectionLaplacian(scalar, x);
  const std::vector<cd> yn = applySunLaplacian(sun, x);
  CHECK(maxAbsDiff(ys, yn) < 1e-12);
}

TEST_CASE("weighted SunGauge SU(2) V-cycle solves the massless graded operator") {
  // m^2 = 0 (pure covariant Laplacian -- the paper's operator) with graded
  // per-edge weights: the weighted covariant hierarchy must still converge and
  // beat CG. A hot SU(2) field keeps lambda_min > 0 (not a pure gauge).
  SunLattice L = randomSunLattice(2, 16, 16, 16, /*w=*/1.0, /*mass2=*/0.0, /*seed=*/42);
  L.setEdgeWeights(gradedW(L.numLinksX(), 0, 1.0), gradedW(L.numLinksY(), 1, 1.0),
                   gradedW(L.numLinksZ(), 2, 1.0));
  const std::vector<cd> b = randomVec(L.dof(), 3);
  std::vector<cd> x(b.size(), cd(0, 0)), xcg(b.size(), cd(0, 0));
  bochner::MgOptions opts;
  opts.tol = 1e-8;
  opts.maxCycles = 60;
  const auto mg = vcycleSolveSun(L, b, x, opts);
  const auto cg = cgSolveSun(L, b, xcg, 1e-8);
  MESSAGE("weighted massless SU(2) n=16: MG " << mg.cycles << " cycles vs CG " << cg.iterations
                                              << " iters");
  CHECK(mg.relResidual < opts.tol);
  CHECK(cg.relResidual < 1e-8);
  CHECK(mg.cycles < cg.iterations);
}

TEST_CASE("weighted SunGauge covMG-LOBPCG returns a certified eigenpair (SU(3), massless)") {
  SunLattice L = randomSunLattice(3, 12, 12, 12, /*w=*/1.0, /*mass2=*/0.0, /*seed=*/17);
  L.setEdgeWeights(gradedW(L.numLinksX(), 0, 1.0), gradedW(L.numLinksY(), 1, 1.0),
                   gradedW(L.numLinksZ(), 2, 1.0));
  bochner::GaugeEigenOptions opts;
  opts.tol = 1e-6;
  const auto res = smallestEigenpairSunMG(L, nullptr, opts);
  CHECK(res.eigenvalue > 0.0);
  CHECK(res.iterations < opts.maxIters);
  // Independent residual recompute (certificate).
  const auto Ex = applySunLaplacian(L, res.vector);
  double num = 0.0, nrm = 0.0;
  for (std::size_t i = 0; i < Ex.size(); ++i) {
    num += std::norm(Ex[i] - res.eigenvalue * res.vector[i]);
    nrm += std::norm(res.vector[i]);
  }
  const double indep = std::sqrt(num / nrm) / res.eigenvalue;
  MESSAGE("weighted massless SU(3) covMG-LOBPCG: eig=" << res.eigenvalue << " iters=" << res.iterations
                                               << " residual=" << indep);
  CHECK(indep < 1e-5);
}
