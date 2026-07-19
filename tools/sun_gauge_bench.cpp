// Non-abelian transferability demonstration for the paper.
//
// The closed-form gauge-aware geometric multigrid needs only a transport per
// edge, so it carries from U(1) phases to SU(d) matrices unchanged (scalar
// transport -> matrix-vector, conjugate -> Hermitian adjoint). This tool shows
// the payoff on the SU(2)/SU(3) covariant Laplacian -Delta^nabla + m^2 on a
// periodic torus:
//
//   (1) mesh-independence: a SMOOTH background field refined in the continuum
//       (h = 1/n, w = n^2, fixed physical mass) -- the V-cycle count stays flat
//       while unpreconditioned CG grows ~n, exactly as in the U(1) tables;
//   (2) robustness: a maximally-disordered ("hot") random-link field, where the
//       V-cycle still converges;
//   plus the covMG-LOBPCG ground-state solve on the same operator.
//
// Usage:  sun_gauge_bench [d] [m^2]   (d = 2 for SU(2) [default], 3 for SU(3);
//                                      m^2 defaults to 0 -- the pure covariant
//                                      Laplacian, the paper's operator)

#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <random>
#include <vector>

#include "BenchTiming.h"
#include "solvers/SunGauge.h"
#ifdef BOCHNER_WITH_PETSC
#include <slepc.h>

#include "grid/CooMatrix.h"
#include "solvers/EigenSolver.h"
#endif

using namespace bochner;
using cd = std::complex<double>;
using Clock = std::chrono::steady_clock;

namespace {

[[maybe_unused]] double ms(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

// Median-of-k wall time (ms) of a solve (BenchTiming.h), the paper-wide
// timing statistic.
using benchstat::medianMs;

// SU(2) exponential exp(i v.sigma) = cos|v| I + i (sin|v|/|v|) v.sigma, written
// into the (p,q) 2x2 block of a d x d identity (row-major).
void embedSu2(cd* M, int d, int p, int q, double v0, double v1, double v2) {
  const double a = std::sqrt(v0 * v0 + v1 * v1 + v2 * v2);
  const double c = std::cos(a), s = (a > 1e-12) ? std::sin(a) / a : 1.0;
  const cd m00(c, s * v2), m01(s * v1, s * v0), m10(-s * v1, s * v0), m11(c, -s * v2);
  M[p * d + p] = m00;
  M[p * d + q] = m01;
  M[q * d + p] = m10;
  M[q * d + q] = m11;
}

void setIdentity(cd* M, int d) {
  for (int i = 0; i < d * d; ++i) M[i] = cd(0, 0);
  for (int i = 0; i < d; ++i) M[i * d + i] = cd(1, 0);
}
void matmulInto(const cd* A, const cd* B, int d, cd* out) {
  for (int i = 0; i < d; ++i)
    for (int j = 0; j < d; ++j) {
      cd sum(0, 0);
      for (int k = 0; k < d; ++k) sum += A[i * d + k] * B[k * d + j];
      out[i * d + j] = sum;
    }
}

// A smooth SU(d) link = exp of a smooth su(d) field along the edge, amplitude
// ~ amp*h so links -> I under continuum refinement (flux per plaquette ~ h^2 F).
// d==2: one SU(2) rotation; d==3: product of three embedded SU(2) rotations in
// the (0,1),(0,2),(1,2) blocks with distinct position dependence -> genuine
// non-abelian curvature.
void smoothLink(int d, int axis, int i, int j, int k, int n, double amp, cd* M) {
  const double h = 1.0 / n;
  auto S = [n](int idx) { return std::sin(2.0 * M_PI * idx / n); };
  // per-axis su(2) coefficient vectors (kept non-collinear across axes)
  double v[3][3];
  v[0][0] = S(j); v[0][1] = S(k); v[0][2] = 0.0;   // block (0,1)
  v[1][0] = S(k); v[1][1] = 0.0;  v[1][2] = S(i);   // block (0,2)
  v[2][0] = 0.0;  v[2][1] = S(i); v[2][2] = S(j);   // block (1,2)
  const double rot = (axis == 0) ? 0.0 : (axis == 1) ? 2.09439510239 : 4.18879020479;  // +/-120deg phase
  for (int b = 0; b < 3; ++b)
    for (int c = 0; c < 3; ++c) v[b][c] = amp * h * (v[b][c] + 0.5 * std::sin(2.0 * M_PI * (i + j + k) / n + rot));
  if (d == 2) {
    setIdentity(M, 2);
    embedSu2(M, 2, 0, 1, v[0][0], v[0][1], v[0][2]);
    return;
  }
  cd R01[9], R02[9], R12[9], T[9];
  setIdentity(R01, 3); embedSu2(R01, 3, 0, 1, v[0][0], v[0][1], v[0][2]);
  setIdentity(R02, 3); embedSu2(R02, 3, 0, 2, v[1][0], v[1][1], v[1][2]);
  setIdentity(R12, 3); embedSu2(R12, 3, 1, 2, v[2][0], v[2][1], v[2][2]);
  matmulInto(R01, R02, 3, T);
  matmulInto(T, R12, 3, M);
}

SunLattice smoothLattice(int d, int n, double mass2, double amp) {
  SunLattice L;
  L.d = d;
  L.lx = L.ly = L.lz = n;
  L.periodic = true;
  L.w = static_cast<double>(n) * n;  // 1/h^2 with h = 1/n (continuum refinement)
  L.mass2 = mass2;
  const int dd = d * d;
  L.ux.resize(static_cast<size_t>(L.numLinksX()) * dd);
  L.uy.resize(static_cast<size_t>(L.numLinksY()) * dd);
  L.uz.resize(static_cast<size_t>(L.numLinksZ()) * dd);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      for (int k = 0; k < n; ++k) {
        const size_t e = static_cast<size_t>((i * n + j) * n + k) * dd;
        smoothLink(d, 0, i, j, k, n, amp, &L.ux[e]);
        smoothLink(d, 1, i, j, k, n, amp, &L.uy[e]);
        smoothLink(d, 2, i, j, k, n, amp, &L.uz[e]);
      }
  return L;
}

std::vector<cd> randomRhs(long dof, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::normal_distribution<double> g(0.0, 1.0);
  std::vector<cd> b(dof);
  for (auto& z : b) z = cd(g(rng), g(rng));
  return b;
}

#ifdef BOCHNER_WITH_PETSC
// Assemble the (periodic) SU(d) covariant Laplacian as a real-symmetric CooMatrix
// of dimension 2*d*N, via the complex->real embedding a+ib -> [[a,-b],[b,a]] --
// the same embedding the U(1) connectionLaplacian uses -- so it can be handed to
// the production SLEPc Lanczos baseline (EigenSolver.h), exactly as in the U(1)
// eigen comparison.
CooMatrix assembleReal(const SunLattice& L) {
  const int d = L.d, dd = d * d;
  const long N = L.numNodes();
  const int M = static_cast<int>(2 * d * N);
  CooMatrix A(M, M);
  auto cplx = [&](long p, long q, cd v) {  // complex dof (p,q) -> real 2x2 block
    A.add(2 * p, 2 * q, v.real());
    A.add(2 * p, 2 * q + 1, -v.imag());
    A.add(2 * p + 1, 2 * q, v.imag());
    A.add(2 * p + 1, 2 * q + 1, v.real());
  };
  auto block = [&](long c, long n, const cd* U, bool adj) {  // -w * U_{n->c}[a,b]
    for (int a = 0; a < d; ++a)
      for (int b = 0; b < d; ++b) {
        const cd t = adj ? std::conj(U[b * d + a]) : U[a * d + b];
        if (t != cd(0, 0)) cplx(c * d + a, n * d + b, -L.w * t);
      }
  };
  const int lx = L.lx, ly = L.ly, lz = L.lz;
  const double diag = L.w * 6.0 + L.mass2;  // periodic degree = 6
  for (int i = 0; i < lx; ++i)
    for (int j = 0; j < ly; ++j)
      for (int k = 0; k < lz; ++k) {
        const long c = L.index(i, j, k);
        for (int a = 0; a < d; ++a) cplx(c * d + a, c * d + a, cd(diag, 0));
        const int im = (i - 1 + lx) % lx, ip = (i + 1) % lx;
        const int jm = (j - 1 + ly) % ly, jp = (j + 1) % ly;
        const int km = (k - 1 + lz) % lz, kp = (k + 1) % lz;
        block(c, L.index(im, j, k), &L.ux[static_cast<size_t>(L.index(im, j, k)) * dd], false);
        block(c, L.index(ip, j, k), &L.ux[static_cast<size_t>(c) * dd], true);
        block(c, L.index(i, jm, k), &L.uy[static_cast<size_t>(L.index(i, jm, k)) * dd], false);
        block(c, L.index(i, jp, k), &L.uy[static_cast<size_t>(c) * dd], true);
        block(c, L.index(i, j, km), &L.uz[static_cast<size_t>(L.index(i, j, km)) * dd], false);
        block(c, L.index(i, j, kp), &L.uz[static_cast<size_t>(c) * dd], true);
      }
  return A;
}
#endif

#ifndef BOCHNER_WITH_PETSC
double cdotRe(const std::vector<cd>& a, const std::vector<cd>& b) {
  double s = 0.0;
  for (size_t i = 0; i < a.size(); ++i) s += (std::conj(a[i]) * b[i]).real();
  return s;
}

// Smallest eigenvalue of a symmetric tridiagonal (diag alpha[0..m-1], off-diag
// beta[1..m-1]) by Sturm-sequence bisection -- the reference for Lanczos's Ritz
// values.
int sturmCount(const std::vector<double>& al, const std::vector<double>& be, int m, double x) {
  int c = 0;
  double d = al[0] - x;
  if (d < 0) ++c;
  for (int i = 1; i < m; ++i) {
    if (d == 0.0) d = 1e-300;
    d = (al[i] - x) - be[i] * be[i] / d;
    if (d < 0) ++c;
  }
  return c;
}
double smallestTridiag(const std::vector<double>& al, const std::vector<double>& be, int m) {
  double lo = 1e300, hi = -1e300;
  for (int i = 0; i < m; ++i) {
    const double r = (i > 0 ? std::abs(be[i]) : 0.0) + (i + 1 < m ? std::abs(be[i + 1]) : 0.0);
    lo = std::min(lo, al[i] - r);
    hi = std::max(hi, al[i] + r);
  }
  for (int it = 0; it < 200 && hi - lo > 1e-11 * (std::abs(lo) + std::abs(hi) + 1e-30); ++it) {
    const double mid = 0.5 * (lo + hi);
    if (sturmCount(al, be, m, mid) >= 1)
      hi = mid;
    else
      lo = mid;
  }
  return 0.5 * (lo + hi);
}

// Baseline eigensolver: Lanczos with full reorthogonalization on the matrix-free
// SU(d) operator (the natural competitor to covMG-LOBPCG at scale -- the U(1) paper's
// baseline is SLEPc Lanczos; this is the dependency-free equivalent). Stops when
// the smallest Ritz value stops changing relatively by `tol`.
double lanczosSmallest(const SunLattice& L, double tol, int maxIters, int& itersOut) {
  const long n = L.dof();
  std::mt19937_64 rng(1);
  std::normal_distribution<double> g(0.0, 1.0);
  std::vector<cd> v(n);
  for (auto& z : v) z = cd(g(rng), g(rng));
  double nv = std::sqrt(cdotRe(v, v));
  for (auto& z : v) z /= nv;
  std::vector<std::vector<cd>> basis{v};
  std::vector<double> al, be{0.0};  // be[0] unused; be[i] couples rows i-1,i
  double prev = 0.0;
  for (int j = 0; j < maxIters; ++j) {
    std::vector<cd> w = applySunLaplacian(L, basis[j]);
    if (j > 0)
      for (long i = 0; i < n; ++i) w[i] -= be[j] * basis[j - 1][i];
    const double aj = cdotRe(basis[j], w);
    al.push_back(aj);
    for (long i = 0; i < n; ++i) w[i] -= aj * basis[j][i];
    for (const auto& q : basis) {  // full reorthogonalization
      const cd p = [&] { cd s(0, 0); for (long i = 0; i < n; ++i) s += std::conj(q[i]) * w[i]; return s; }();
      for (long i = 0; i < n; ++i) w[i] -= p * q[i];
    }
    const double lam = smallestTridiag(al, be, j + 1);
    if (j > 0 && std::abs(lam - prev) < tol * std::abs(lam)) {
      itersOut = j + 1;
      return lam;
    }
    prev = lam;
    const double bj = std::sqrt(cdotRe(w, w));
    if (bj < 1e-13) {
      itersOut = j + 1;
      return lam;
    }
    be.push_back(bj);
    for (auto& z : w) z /= bj;
    basis.push_back(std::move(w));
  }
  itersOut = maxIters;
  return prev;
}
#endif

}  // namespace

int main(int argc, char** argv) {
#ifdef BOCHNER_WITH_PETSC
  SlepcInitialize(&argc, &argv, nullptr, nullptr);
#endif
  const int d = (argc > 1) ? std::atoi(argv[1]) : 2;
  const double mass2 = (argc > 2) ? std::atof(argv[2]) : 0.0;  // 0 = pure covariant Laplacian
  const double amp = 4.0;

  std::printf("=== SU(%d) covariant Laplacian  -Delta^nabla%s ===\n", d,
              mass2 != 0.0 ? "  + m^2" : "  (massless)");
  if (mass2 != 0.0) std::printf("    m^2 = %g\n", mass2);

  // (2) helper timings share this MG config.
  MgOptions mg;
  mg.tol = 1e-8;
  mg.maxCycles = 200;

  // (1) Mesh-independence: smooth field, continuum refinement (w = n^2).
  std::printf("\n[1] SMOOTH field, continuum refinement (h = 1/n, w = n^2): mesh-independence\n");
  std::printf("  %-4s %-9s  %-19s  %-20s  %-8s %-9s\n", "n", "DOF", "gauge-MG (cyc, ms)",
              "CG(none) (its, ms)", "CG/MG", "lambda_min");
  for (int n : {8, 16, 24, 32, 48}) {
    const SunLattice L = smoothLattice(d, n, mass2, amp);
    const auto levels = buildSunLevels(L);
    const std::vector<cd> b = randomRhs(L.dof(), 12345);

    MgResult mr;
    const double mgMs = medianMs([&] {
      std::vector<cd> x(L.dof(), cd(0, 0));
      mr = vcycleSolveSun(levels, b, x, mg);
    });
    SolveStats cg;
    const double cgMs = medianMs([&] {
      std::vector<cd> x(L.dof(), cd(0, 0));
      cg = cgSolveSun(L, b, x, 1e-8, 20000);
    });

    GaugeEigenOptions eo;
    eo.tol = 1e-7;
    eo.maxIters = 200;
    const GaugeEigenResult er = smallestEigenpairSunMG(L, nullptr, eo);

    std::printf("  %-4d %-9ld  %2d (%.0e, %6.1f)  %5d (%.0e, %7.1f)  %5.1fx  %.5f\n", n, L.dof(),
                mr.cycles, mr.relResidual, mgMs, cg.iterations, cg.relResidual, cgMs, cgMs / mgMs,
                er.eigenvalue);
  }

  // (2) Robustness: maximally-disordered "hot" random links, fixed lattice.
  std::printf("\n[2] RANDOM hot links, fixed lattice (w = 1): robustness of the V-cycle\n");
  std::printf("  %-4s %-9s  %-19s  %-20s  %-8s\n", "n", "DOF", "gauge-MG (cyc, ms)",
              "CG(none) (its, ms)", "CG/MG");
  for (int n : {8, 16, 24, 32, 48}) {
    const SunLattice L = randomSunLattice(d, n, n, n, /*w=*/1.0, mass2, /*seed=*/777);
    const auto levels = buildSunLevels(L);
    const std::vector<cd> b = randomRhs(L.dof(), 999);
    MgResult mr;
    const double mgMs = medianMs([&] {
      std::vector<cd> x(L.dof(), cd(0, 0));
      mr = vcycleSolveSun(levels, b, x, mg);
    });
    SolveStats cg;
    const double cgMs = medianMs([&] {
      std::vector<cd> x(L.dof(), cd(0, 0));
      cg = cgSolveSun(L, b, x, 1e-8, 20000);
    });
    std::printf("  %-4d %-9ld  %2d (%.0e, %6.1f)  %5d (%.0e, %7.1f)  %5.1fx\n", n, L.dof(), mr.cycles,
                mr.relResidual, mgMs, cg.iterations, cg.relResidual, cgMs, cgMs / mgMs);
  }

  // (2b) Variable edge weights: a fixed smooth conductance metric refined in
  // the continuum -- w_e = n^2 * c(edge midpoint), c(x) = 1 + 3 sin^2(pi x_1)
  // sin^2(pi x_2) (contrast up to 4x). Exercises the weighted covariant
  // averaging + series-conductance coarsening; the V-cycle count should stay
  // as flat as the uniform series in [1].
  std::printf("\n[2b] SMOOTH field + GRADED edge weights (metric contrast 4x): weighted transfer\n");
  std::printf("  %-4s %-9s  %-19s  %-20s  %-8s\n", "n", "DOF", "gauge-MG (cyc, ms)",
              "CG(none) (its, ms)", "CG/MG");
  for (int n : {8, 16, 24, 32, 48}) {
    SunLattice L = smoothLattice(d, n, mass2, amp);
    const double h = 1.0 / n;
    const auto cfun = [&](double x, double y) {
      const double sx = std::sin(M_PI * x), sy = std::sin(M_PI * y);
      return 1.0 + 3.0 * sx * sx * sy * sy;
    };
    const auto graded = [&](int axis) {
      std::vector<double> wv(static_cast<std::size_t>(axis == 0   ? L.numLinksX()
                                                      : axis == 1 ? L.numLinksY()
                                                                  : L.numLinksZ()));
      for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
          for (int k = 0; k < n; ++k) {
            // edge midpoint (periodic: every node has a forward link per axis)
            const double x = (i + (axis == 0 ? 0.5 : 0.0)) * h;
            const double y = (j + (axis == 1 ? 0.5 : 0.0)) * h;
            const double z = (k + (axis == 2 ? 0.5 : 0.0)) * h;
            wv[static_cast<std::size_t>((i * n + j) * n + k)] =
                static_cast<double>(n) * n * cfun(x + z, y + z);
          }
      return wv;
    };
    L.setEdgeWeights(graded(0), graded(1), graded(2));
    const auto levels = buildSunLevels(L);
    const std::vector<cd> b = randomRhs(L.dof(), 4242);
    MgResult mr;
    const double mgMs = medianMs([&] {
      std::vector<cd> x(L.dof(), cd(0, 0));
      mr = vcycleSolveSun(levels, b, x, mg);
    });
    SolveStats cg;
    const double cgMs = medianMs([&] {
      std::vector<cd> x(L.dof(), cd(0, 0));
      cg = cgSolveSun(L, b, x, 1e-8, 20000);
    });
    std::printf("  %-4d %-9ld  %2d (%.0e, %6.1f)  %5d (%.0e, %7.1f)  %5.1fx\n", n, L.dof(),
                mr.cycles, mr.relResidual, mgMs, cg.iterations, cg.relResidual, cgMs, cgMs / mgMs);
  }

  // (3) Ground-state wall time: covMG-LOBPCG vs the production SLEPc Lanczos
  // baseline (the SAME solver the U(1) tables use), for a fair, consistent
  // comparison. Run single-threaded (SLEPc has no OpenMP), matching the U(1)
  // methodology. Without PETSc, fall back to a matrix-free full-reorth Lanczos
  // (an upper-bound baseline only -- not used for the paper numbers).
#ifdef BOCHNER_WITH_PETSC
  std::printf("\n[3] GROUND STATE, smooth field: covMG-LOBPCG vs SLEPc Lanczos (run single-threaded!)\n");
  std::printf("  %-4s %-9s  %-18s  %-20s  %-9s %-9s\n", "n", "DOF", "covMG-LOBPCG (outer, ms)",
              "SLEPc-Lanc (its, ms)", "Lanc/ours", "|d lambda|");
  for (int n : {8, 16, 24, 32, 48}) {
    const SunLattice L = smoothLattice(d, n, mass2, amp);
    GaugeEigenOptions eo;
    eo.tol = 1e-7;
    eo.maxIters = 300;
    GaugeEigenResult er;
    const double rqMs = medianMs([&] { er = smallestEigenpairSunMG(L, nullptr, eo); });
    const CooMatrix A = assembleReal(L);
    EigenPair lp;
    const double lanMs = medianMs([&] { lp = smallestEigenpairLanczos(A, 1e-7); });
    std::printf("  %-4d %-9ld  %3d%s (%7.1f)     %5d (%8.1f)     %5.1fx  %.1e\n", n, L.dof(),
                er.iterations, er.converged ? "" : "!", rqMs, lp.iterations, lanMs, lanMs / rqMs,
                std::abs(er.eigenvalue - lp.value));
  }
#else
  std::printf("\n[3] GROUND STATE: (build with -DBOCHNER_WITH_PETSC=ON for the SLEPc Lanczos comparison)\n");
  std::printf("    fallback matrix-free full-reorth Lanczos (upper-bound baseline only):\n");
  for (int n : {8, 16, 24, 32}) {
    const SunLattice L = smoothLattice(d, n, mass2, amp);
    GaugeEigenOptions eo;
    eo.tol = 1e-7;
    eo.maxIters = 300;
    GaugeEigenResult er;
    const double rqMs = medianMs([&] { er = smallestEigenpairSunMG(L, nullptr, eo); });
    int lit = 0;
    double llam = 0.0;
    const double lanMs = medianMs([&] { llam = lanczosSmallest(L, 1e-8, 2000, lit); });
    std::printf("  n=%-3d DOF=%-8ld covMG-LOBPCG %3d%s it %7.1f ms   Lanczos %5d it %8.1f ms   |dl|=%.1e\n", n,
                L.dof(), er.iterations, er.converged ? "" : "!", rqMs, lit, lanMs,
                std::abs(er.eigenvalue - llam));
  }
#endif
  benchstat::printTimingSummary();
#ifdef BOCHNER_WITH_PETSC
  SlepcFinalize();
#endif
  return 0;
}
