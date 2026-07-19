// Lowest-band (block) eigensolver benchmark for the paper: block covMG-LOBPCG vs the
// production SLEPc Krylov-Schur baseline computing the same m lowest COMPLEX
// eigenpairs of the massless SU(3) covariant Laplacian on the smooth field --
// the distillation-shaped task (Laplacian-Heaviside needs the lowest O(10^2)
// modes of exactly this operator).
//
// SLEPc receives the real 2N x 2N embedding, in which every complex eigenvalue
// is a degenerate real pair, so it is asked for nev = 2m real eigenpairs to
// recover the same m complex ones; this doubling is charged to the baseline
// because it is intrinsic to giving a real solver the complex operator (the
// same convention as the paper's single-pair tables).
//
// Usage: block_eig_bench [d] [field] [n...] [m=LIST]
//   d      = 1 (U(1)), 2, 3            (default 3)
//   field  = smooth | hot | u1flux     (default smooth; u1flux forces d=1:
//            the Landau-gauge uniform-flux torus, nPhi = 4)
//   n...   = lattice sizes             (default 16 24 32)
//   m=LIST = comma-separated band sizes (default 1,2,4,8,16,32)
//   noks   = skip the Krylov-Schur baseline (sizes whose assembled matrix
//            + basis exceed RAM, e.g. n=128: block-solver-only measurement;
//            prints the full computed band so band-edge gaps are measurable)
//   g=N    = guard vectors (GaugeEigenOptions::blockGuard; default 2)
//   pc=N   = V-cycles per preconditioner apply (GaugeEigenOptions::precCycles,
//            default 1; diagnosis knob -- tests whether a rate is limited by
//            preconditioner quality)
//   lock=M = certified-pair locking mode: off (default; the published
//            iteration), hard (freeze + remove from the Rayleigh-Ritz space;
//            BlockEigOptions::lockConverged), soft (sticky certification,
//            the column stays in the Rayleigh-Ritz space;
//            BlockEigOptions::softLockConverged). `nolock` = legacy alias
//            for off (the flag predates the OFF default).
// At m = 1 the dedicated single-vector solver is timed alongside (the block
// pays ~3x for its guard vectors there).

#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
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

// --- smooth SU(d) field (identical construction to sun_gauge_bench) ---
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
void smoothLink(int d, int axis, int i, int j, int k, int n, double amp, cd* M) {
  const double h = 1.0 / n;
  auto S = [n](int idx) { return std::sin(2.0 * M_PI * idx / n); };
  double v[3][3];
  v[0][0] = S(j); v[0][1] = S(k); v[0][2] = 0.0;
  v[1][0] = S(k); v[1][1] = 0.0;  v[1][2] = S(i);
  v[2][0] = 0.0;  v[2][1] = S(i); v[2][2] = S(j);
  const double rot = (axis == 0) ? 0.0 : (axis == 1) ? 2.09439510239 : 4.18879020479;
  for (int b = 0; b < 3; ++b)
    for (int c = 0; c < 3; ++c)
      v[b][c] = amp * h * (v[b][c] + 0.5 * std::sin(2.0 * M_PI * (i + j + k) / n + rot));
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
// U(1) uniform-flux torus (Landau/seam gauge, nPhi quanta) as a d = 1
// SunLattice, so the same block machinery runs the abelian case.
SunLattice u1FluxLattice(int n, int nPhi) {
  SunLattice L;
  L.d = 1;
  L.lx = L.ly = L.lz = n;
  L.periodic = true;
  L.w = static_cast<double>(n) * n;
  L.mass2 = 0.0;
  const double phi_p = 2.0 * M_PI * nPhi / (static_cast<double>(n) * n);
  L.ux.resize(static_cast<size_t>(L.numLinksX()));
  L.uy.resize(static_cast<size_t>(L.numLinksY()));
  L.uz.assign(static_cast<size_t>(L.numLinksZ()), cd(1, 0));
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      for (int k = 0; k < n; ++k) {
        const size_t e = static_cast<size_t>((i * n + j) * n + k);
        L.ux[e] = std::exp(cd(0.0, -phi_p * j));
        L.uy[e] = (j == n - 1) ? std::exp(cd(0.0, 2.0 * M_PI * nPhi * i / n)) : cd(1, 0);
      }
  return L;
}

SunLattice smoothLattice(int d, int n, double amp) {
  SunLattice L;
  L.d = d;
  L.lx = L.ly = L.lz = n;
  L.periodic = true;
  L.w = static_cast<double>(n) * n;
  L.mass2 = 0.0;  // massless: the paper's operator
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

#ifdef BOCHNER_WITH_PETSC
CooMatrix assembleReal(const SunLattice& L) {
  const int d = L.d, dd = d * d;
  const long N = L.numNodes();
  const int M = static_cast<int>(2 * d * N);
  CooMatrix A(M, M);
  auto cplx = [&](long p, long q, cd v) {
    A.add(2 * p, 2 * q, v.real());
    A.add(2 * p, 2 * q + 1, -v.imag());
    A.add(2 * p + 1, 2 * q, v.imag());
    A.add(2 * p + 1, 2 * q + 1, v.real());
  };
  auto block = [&](long c, long nn, const cd* U, bool adj) {
    for (int a = 0; a < d; ++a)
      for (int b = 0; b < d; ++b) {
        const cd t = adj ? std::conj(U[b * d + a]) : U[a * d + b];
        if (t != cd(0, 0)) cplx(c * d + a, nn * d + b, -L.w * t);
      }
  };
  const int lx = L.lx, ly = L.ly, lz = L.lz;
  const double diag = L.w * 6.0 + L.mass2;
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

}  // namespace

int main(int argc, char** argv) {
#ifdef BOCHNER_WITH_PETSC
  SlepcInitialize(&argc, &argv, nullptr, nullptr);
#endif
  const int d = (argc > 1) ? std::atoi(argv[1]) : 3;
  const std::string field = (argc > 2) ? argv[2] : "smooth";
  std::vector<int> sizes;
  std::vector<int> ms;
  // Read only under BOCHNER_WITH_PETSC (there is no baseline to skip without
  // it), so it is genuinely unused in the PETSc-free build.
  [[maybe_unused]] bool noks = false;
  std::string lock = "off";
  int guard = -1;
  int precCycles = -1;
  for (int a = 3; a < argc; ++a) {
    if (std::strcmp(argv[a], "noks") == 0) {
      noks = true;
      continue;
    }
    if (std::strncmp(argv[a], "g=", 2) == 0) {
      guard = std::atoi(argv[a] + 2);
      continue;
    }
    if (std::strncmp(argv[a], "pc=", 3) == 0) {
      precCycles = std::atoi(argv[a] + 3);
      continue;
    }
    if (std::strncmp(argv[a], "lock=", 5) == 0) {
      lock = argv[a] + 5;
      continue;
    }
    if (std::strcmp(argv[a], "nolock") == 0) {  // legacy alias
      lock = "off";
      continue;
    }
    if (std::strncmp(argv[a], "m=", 2) == 0) {
      for (const char* p = argv[a] + 2; *p;) {
        ms.push_back(std::atoi(p));
        while (*p && *p != ',') ++p;
        if (*p == ',') ++p;
      }
    } else {
      sizes.push_back(std::atoi(argv[a]));
    }
  }
  if (sizes.empty()) sizes = {16, 24, 32};
  if (ms.empty()) ms = {1, 2, 4, 8, 16, 32};
  const double amp = 4.0;

  std::printf("=== lowest-band: block covMG-LOBPCG vs SLEPc Krylov-Schur (nev = 2m real),"
              " d=%d field=%s, massless, tol 1e-7, lock=%s ===\n", d, field.c_str(), lock.c_str());
  std::printf("  (at m=1 the third column is the dedicated single-vector covMG-LOBPCG)\n");
  std::printf("  %-4s %-9s %-4s  %-22s  %-22s  %-22s  %-8s %-10s\n", "n", "cDOF", "m",
              "block covMG-LOBPCG (it, ms)", "single covMG-LOBPCG (it, ms)", "SLEPc 2m (its, ms)", "KS/blk",
              "|d lambda|");
  for (int n : sizes) {
    const SunLattice L = field == "u1flux" ? u1FluxLattice(n, 4)
                         : field == "hot"
                             ? randomSunLattice(d, n, n, n, /*w=*/1.0, /*mass2=*/0.0, 777)
                             : smoothLattice(d, n, amp);
    for (int m : ms) {
      GaugeEigenOptions eo;
      eo.tol = 1e-7;
      eo.maxIters = 500;
      if (guard >= 0) eo.blockGuard = guard;
      if (precCycles > 0) eo.precCycles = precCycles;
      if (lock == "hard") eo.blockLockConverged = true;
      if (lock == "soft") eo.blockSoftLockConverged = true;
      BlockEigResult blk;
      const double blkMs = benchstat::medianMs([&] { blk = lowestEigenpairsSunMG(L, m, nullptr, eo); });
      char single[40] = "";
      if (m == 1) {
        GaugeEigenResult sv;
        const double svMs = benchstat::medianMs([&] { sv = smallestEigenpairSunMG(L, nullptr, eo); });
        std::snprintf(single, sizeof single, "%4d it %9.1f ms", sv.iterations, svMs);
      }

#ifdef BOCHNER_WITH_PETSC
      if (noks) {
        char cert[48] = "";
        if (!blk.converged)
          std::snprintf(cert, sizeof cert, "  [blk maxRes %.2e > tol]", blk.maxResidual);
        std::printf("  %-4d %-9ld %-4d  %4d it %9.1f ms    %-22s  (baseline skipped)"
                    "  lambda[0]=%.5f%s\n",
                    n, L.dof(), m, blk.iterations, blkMs, single, blk.eigenvalues[0], cert);
        std::printf("      band:");
        for (int j = 0; j < m; ++j) std::printf(" %.6f", blk.eigenvalues[j]);
        std::printf("\n");
        std::fflush(stdout);
        continue;
      }
      const CooMatrix A = assembleReal(L);
      std::vector<EigenPair> ks;
      const double ksMs = benchstat::medianMs([&] { ks = smallestEigenpairsLanczos(A, 2 * m, 1e-7); });
      // Compare the m distinct complex eigenvalues: SLEPc's 2m real values come
      // in degenerate pairs; take every second one.
      double dl = 0.0;
      long ksIts = 0;
      for (int j = 0; j < m; ++j)
        dl = std::max(dl, std::abs(blk.eigenvalues[j] - ks[2 * j].value));
      for (const auto& p : ks) ksIts = std::max<long>(ksIts, p.iterations);
      std::printf("  %-4d %-9ld %-4d  %4d it %9.1f ms    %-22s  %5ld it %9.1f ms    %5.1fx  %.1e%s\n",
                  n, L.dof(), m, blk.iterations, blkMs, single, ksIts, ksMs, ksMs / blkMs, dl,
                  blk.converged ? "" : "  [blk maxRes>tol]");
#else
      std::printf("  %-4d %-9ld %-4d  %4d it %9.1f ms  %s  (build with PETSc for the baseline)%s\n",
                  n, L.dof(), m, blk.iterations, blkMs, single,
                  blk.converged ? "" : "  [blk maxRes>tol]");
#endif
      std::fflush(stdout);
    }
  }
  benchstat::printTimingSummary();
#ifdef BOCHNER_WITH_PETSC
  SlepcFinalize();
#endif
  return 0;
}
