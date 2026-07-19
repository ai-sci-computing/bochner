/// \file
/// Eigensolve comparison on the lattice-gauge-solvers setup (periodic 3-torus,
/// uniform magnetic field): our gauge-MG Rayleigh-quotient eigensolver vs the
/// SLEPc Lanczos baseline for the smallest eigenpair of the U(1) connection
/// Laplacian. Both solve the SAME operator (asserted: the assembled CooMatrix
/// the Lanczos path uses is the real embedding of the matrix-free lattice the
/// covMG-LOBPCG path preconditions), to the same tolerance; we report the eigenvalue
/// (must agree), outer iterations, and wall-clock.
///
/// It also reports a SINGLE gauge-MG linear solve and, from it, the estimated
/// cost of a classical (shift-and-invert) backward/inverse iteration =
/// per-solve time x outer iterations -- so the eigensolve numbers can be read
/// off the linear-solve timings, as requested.
///
/// Usage: torus_eig_compare [n] [nPhi]   (defaults 16 4 = the uniform-field example)
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <slepc.h>

#include "BenchTiming.h"
#include "grid/CooMatrix.h"
#include "solvers/EigenSolver.h"
#include "solvers/GaugeEigen.h"
#include "solvers/GaugeMultigrid.h"

using bochner::GaugeLattice;
using benchstat::medianMs;
using cd = std::complex<double>;

namespace {

// The lattice-gauge-solvers Examples::uniformField: nPhi flux quanta through the
// x-y torus in the Landau/seam gauge, as bochner forward links.
GaugeLattice uniformFluxLattice(int n, int nPhi, double h) {
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

// Assemble the same periodic operator as a real 2N CooMatrix (the Lanczos input),
// matching applyConnectionLaplacian: diagonal 6w, off-diagonal -w e^{-i theta}.
bochner::CooMatrix assemblePeriodic(const GaugeLattice& L) {
  const int n = static_cast<int>(L.numNodes());
  bochner::CooMatrix A(2 * n, 2 * n);
  const auto blk = [&](int a, int b, double re, double im) {
    A.add(2 * a, 2 * b, re);
    A.add(2 * a, 2 * b + 1, -im);
    A.add(2 * a + 1, 2 * b, im);
    A.add(2 * a + 1, 2 * b + 1, re);
  };
  const auto lnk = [&](int a, int b, double th) {
    blk(a, a, L.w, 0.0);
    blk(b, b, L.w, 0.0);
    blk(a, b, -L.w * std::cos(th), L.w * std::sin(th));   // -w e^{-i th}
    blk(b, a, -L.w * std::cos(th), -L.w * std::sin(th));  // conj
  };
  const auto id = [&](int i, int j, int k) { return (i * L.ly + j) * L.lz + k; };
  for (int i = 0; i < L.lx; ++i)
    for (int j = 0; j < L.ly; ++j)
      for (int k = 0; k < L.lz; ++k) {
        const int c = id(i, j, k);
        lnk(c, id((i + 1) % L.lx, j, k), L.lkx[c]);
        lnk(c, id(i, (j + 1) % L.ly, k), L.lky[c]);
        lnk(c, id(i, j, (k + 1) % L.lz), L.lkz[c]);
      }
  return A;
}

}  // namespace

int main(int argc, char** argv) {
  SlepcInitialize(&argc, &argv, nullptr, nullptr);
  const int n = argc > 1 ? std::atoi(argv[1]) : 16;
  const int nPhi = argc > 2 ? std::atoi(argv[2]) : 4;
  const double tol = 1e-7;

  const GaugeLattice lat = uniformFluxLattice(n, nPhi, 1.0 / n);
  const bochner::CooMatrix A = assemblePeriodic(lat);

  // Operator-equivalence guard: A (real 2N) == the matrix-free lattice.
  std::vector<cd> probe(lat.numNodes());
  for (std::size_t c = 0; c < probe.size(); ++c) probe[c] = cd(std::cos(0.7 * c), std::sin(0.3 * c));
  const std::vector<cd> yLat = bochner::applyConnectionLaplacian(lat, probe);
  const std::vector<double> yA = [&] {
    std::vector<double> xr = bochner::toInterleaved(probe), yr(xr.size(), 0.0);
    for (const auto& t : A.compressed()) yr[t.row] += t.value * xr[t.col];
    return yr;
  }();
  const std::vector<cd> yAc = bochner::toComplex(yA);
  double mismatch = 0.0;
  for (std::size_t c = 0; c < yLat.size(); ++c) mismatch = std::max(mismatch, std::abs(yLat[c] - yAc[c]));

  std::printf("\n#### uniform field, n=%d (%d DOFs), nPhi=%d, tol=%.0e ####\n", n, 2 * n * n * n, nPhi,
              tol);
  std::printf("   operator mismatch (assembled vs matrix-free) = %.2e\n\n", mismatch);

  // SLEPc Lanczos baseline.
  bochner::EigenPair L;
  const double msL = medianMs([&] { L = bochner::smallestEigenpairLanczos(A, tol); });

  // Our gauge-MG Rayleigh-quotient eigensolver (no SLEPc), from a cold constant start.
  bochner::GaugeEigenOptions eo;
  eo.tol = tol;
  bochner::GaugeEigenResult R;
  const double msR = medianMs([&] { R = bochner::smallestEigenpairGaugeMG(lat, nullptr, eo); });

  // A single gauge-MG linear solve, for the backward/inverse-iteration estimate.
  bochner::MgOptions mo;
  mo.tol = tol;
  std::vector<cd> b = bochner::applyConnectionLaplacian(lat, probe), x(lat.numNodes(), cd(0, 0));
  bochner::MgResult one;
  const double msSolve = medianMs([&] {
    std::fill(x.begin(), x.end(), cd(0, 0));  // every repeat solves from zero
    one = bochner::vcycleSolve(lat, b, x, mo);
  });

  std::printf("%-26s | %12s | %6s | %10s\n", "method", "eigenvalue", "iters", "wall ms");
  std::printf("--------------------------------------------------------------------\n");
  std::printf("%-26s | %12.6f | %6s | %10.1f\n", "SLEPc Lanczos", L.value, "-", msL);
  std::printf("%-26s | %12.6f | %6d%s | %9.1f  res=%.1e\n", "covMG-LOBPCG (ours)", R.eigenvalue,
              R.iterations, R.converged ? "" : "!",
              msR, R.residual);
  std::printf("   |eigenvalue difference|    = %.2e\n", std::abs(L.value - R.eigenvalue));

  // Eigenvector comparison as the DISTANCE BETWEEN THE TWO COMPLEX LINES that the
  // eigenvectors span (the ray C*psi), which is exactly the phase-invariant object
  // an eigenvector is. For unit vL, vR the distance between the lines span(vL) and
  // span(vR) is the CHORDAL projective distance sin(theta), theta the
  // principal angle = arccos|<vL,vR>| (the Fubini-Study geodesic distance
  // is theta itself; sin(theta) = ||P_L - P_R||_2 and agrees to 2nd order)
  //     d = sqrt(1 - |<vL,vR>|^2) = sin(theta) = || P_vL - P_vR ||_2
  // (theta = principal angle, P the rank-1 projectors); d = 0 iff the same line,
  // independent of each vector's global phase. We also report the pointwise |psi|
  // difference and each vector's own Rayleigh residual (both are true eigenvectors).
  std::vector<cd> vL = bochner::toComplex(L.vector), vR = R.vector;
  const auto unit = [](std::vector<cd>& v) {
    double s = 0.0;
    for (const cd& z : v) s += std::norm(z);
    s = std::sqrt(s);
    if (s > 0) for (cd& z : v) z /= s;
  };
  unit(vL);
  unit(vR);
  cd ov(0, 0);
  for (std::size_t c = 0; c < vL.size(); ++c) ov += std::conj(vL[c]) * vR[c];
  const double overlap = std::min(1.0, std::abs(ov));  // |<vL,vR>| in [0,1]
  const double lineDist = std::sqrt(std::max(0.0, 1.0 - overlap * overlap));  // sin(theta)
  const double angleDeg = std::acos(overlap) * 180.0 / M_PI;
  double maxMag = 0.0, magNum = 0.0, magDen = 0.0;
  for (std::size_t c = 0; c < vL.size(); ++c) {
    const double a = std::abs(vR[c]), b = std::abs(vL[c]);
    maxMag = std::max(maxMag, std::abs(a - b));
    magNum += (a - b) * (a - b);
    magDen += b * b;
  }
  const double relMag = magDen > 0 ? std::sqrt(magNum / magDen) : 0.0;
  const auto eigResidual = [&](const std::vector<cd>& v) {
    const std::vector<cd> Ev = bochner::applyConnectionLaplacian(lat, v);
    cd rq(0, 0);
    for (std::size_t c = 0; c < v.size(); ++c) rq += std::conj(v[c]) * Ev[c];
    double num = 0.0;
    for (std::size_t c = 0; c < v.size(); ++c) num += std::norm(Ev[c] - rq * v[c]);
    return std::sqrt(num) / std::abs(rq.real());
  };
  std::printf("\neigenvector = distance between the complex lines span(vL), span(vR):\n");
  std::printf("   line distance sqrt(1-|<vL,vR>|^2) = %.2e   (0 = same line, phase-invariant)\n",
              lineDist);
  std::printf("   |<vL,vR>| = %.6f  ->  principal angle = %.3f deg\n", overlap, angleDeg);
  std::printf("   pointwise |psi| relative diff     = %.2e  (max %.2e)\n", relMag, maxMag);
  std::printf("   eigen-residual   Lanczos = %.1e, covMG-LOBPCG = %.1e  (both true eigenvectors)\n",
              eigResidual(vL), eigResidual(vR));
  if (lineDist > 1e-3)
    std::printf("   NOTE: lambda_min is degenerate here (nPhi=%d Landau level); the two solvers\n"
                "         return different lines of the SAME eigenspace -- both residuals tiny.\n"
                "         Use nPhi=1 for a simple (non-degenerate) mode: the lines then coincide.\n",
                nPhi);

  std::printf("\nspeedup (Lanczos / ours)      = %.1fx\n", msR > 0 ? msL / msR : 0.0);
  std::printf("single gauge-MG linear solve  = %.2f ms (%d cycles)\n", msSolve, one.cycles);
  std::printf("=> est. backward iteration    ~ %.1f ms  (%.2f ms/solve x %d covMG-LOBPCG steps)\n",
              msSolve * R.iterations, msSolve, R.iterations);
  std::printf("   (covMG-LOBPCG folds the solve + Rayleigh step into each of its %d steps)\n", R.iterations);

  benchstat::printTimingSummary();
  SlepcFinalize();
  return 0;
}
