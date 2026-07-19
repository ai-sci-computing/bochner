/// \file
/// Weight-PLACEMENT validation for the per-edge-weight extension -- tests that
/// the weights land on the right edges with the right magnitudes, beyond the
/// uniform-reproduction check:
///
///   (1) EXACT at finite n: a flat connection with holonomy (theta_x, theta_y,
///       theta_z) on an ANISOTROPIC periodic torus has the closed-form discrete
///       spectrum  lambda(m) = sum_a (2 - 2 cos(2 pi m_a/n_a - theta_a/n_a)) / h_a^2.
///       The eigensolver through the weighted path must hit the m = 0 value to
///       solver accuracy at every resolution -- a swapped weight axis or an
///       off-by-one edge index breaks the equality at finite n, no refinement
///       needed. Axis-weight contrast here: 25x.
///
///   (2) Two discretizations of ONE continuum problem (a uniform-flux magnetic
///       Laplacian on the stretched torus (2,1,1)): (a) cubic cells on a
///       2n x n x n grid through the trusted uniform-scalar-w path; (b) an
///       n x n x n grid stretched to the same box through the weighted path
///       (w_x = n^2/4, w_y = w_z = n^2). Their smallest eigenvalues must
///       converge to each other under refinement (each is O(h^2) accurate to
///       the same continuum limit, so the gap shrinks ~4x per doubling).
#include <doctest.h>

#include <cmath>
#include <complex>
#include <vector>

#include "solvers/GaugeEigen.h"
#include "solvers/GaugeMultigrid.h"

using bochner::GaugeLattice;
using cd = std::complex<double>;

namespace {

// Periodic lattice with CONSTANT link angle alpha_a on every a-link (a flat
// connection; holonomy around the a-cycle = n_a * alpha_a) and per-axis
// spacings h = (hx, hy, hz) installed as per-edge weights 1/h_a^2.
GaugeLattice flatHolonomyLattice(int nx, int ny, int nz, double hx, double hy, double hz,
                                 double thx, double thy, double thz) {
  const std::size_t N = static_cast<std::size_t>(nx) * ny * nz;
  const std::vector<double> lkx(N, thx / nx), lky(N, thy / ny), lkz(N, thz / nz);
  GaugeLattice L = bochner::gaugeLatticePeriodic(nx, ny, nz, 1.0, lkx, lky, lkz);
  L.setEdgeWeights(std::vector<double>(N, 1.0 / (hx * hx)), std::vector<double>(N, 1.0 / (hy * hy)),
                   std::vector<double>(N, 1.0 / (hz * hz)));
  return L;
}

// Exact smallest eigenvalue of the flat-holonomy operator (plane waves; the
// spectrum is separable per axis, and for |theta_a/n_a| < pi/n_a the minimum
// over the momentum integer m_a is at m_a = 0).
double exactLambdaMin(int nx, int ny, int nz, double hx, double hy, double hz, double thx,
                      double thy, double thz) {
  const auto axis = [](int n, double h, double th) {
    double best = 1e300;
    for (int m = 0; m < n; ++m)
      best = std::min(best, (2.0 - 2.0 * std::cos(2.0 * M_PI * m / n - th / n)) / (h * h));
    return best;
  };
  return axis(nx, hx, thx) + axis(ny, hy, thy) + axis(nz, hz, thz);
}

// Uniform-flux (nPhi quanta through the x-y cross-section) periodic lattice in
// the Landau/seam gauge, for a general Nx x Ny x Nz grid: lkx = -phi_p * j with
// phi_p = 2 pi nPhi / (Nx Ny), seam row lky[i, Ny-1, k] = 2 pi nPhi i / Nx.
// The flux (plaquette angle) is metric-free; the spacings enter only through
// the weights the caller installs afterwards.
GaugeLattice fluxLattice(int Nx, int Ny, int Nz, int nPhi) {
  const double phi_p = 2.0 * M_PI * nPhi / (static_cast<double>(Nx) * Ny);
  const std::size_t N = static_cast<std::size_t>(Nx) * Ny * Nz;
  std::vector<double> lkx(N, 0.0), lky(N, 0.0), lkz(N, 0.0);
  const auto idx = [&](int i, int j, int k) { return (i * Ny + j) * Nz + k; };
  for (int i = 0; i < Nx; ++i)
    for (int j = 0; j < Ny; ++j)
      for (int k = 0; k < Nz; ++k) {
        lkx[idx(i, j, k)] = -phi_p * j;
        if (j == Ny - 1) lky[idx(i, j, k)] = 2.0 * M_PI * nPhi * i / static_cast<double>(Nx);
      }
  return bochner::gaugeLatticePeriodic(Nx, Ny, Nz, 1.0, lkx, lky, lkz);
}

double smallestEig(const GaugeLattice& L, int& iters) {
  bochner::GaugeEigenOptions opts;
  opts.tol = 1e-8;
  opts.maxIters = 400;
  // Generic (deterministic pseudo-random) start. The default all-ones start is
  // an EXACT excited eigenvector of the constant-angle flat connection (the
  // m = 0 plane wave), and single-vector Rayleigh-quotient iteration certifies
  // stationarity only -- from that measure-zero start it would stop at the
  // excited level with a zero residual. A generic start has ground-state
  // overlap, and the monotone Rayleigh-quotient descent does the rest.
  std::vector<cd> guess(static_cast<std::size_t>(L.numNodes()));
  for (std::size_t i = 0; i < guess.size(); ++i)
    guess[i] = cd(std::cos(0.7 * i + 0.3), std::sin(1.3 * i));
  const auto res = bochner::smallestEigenpairGaugeMG(L, &guess, opts);
  iters = res.iterations;
  return res.eigenvalue;
}

}  // namespace

TEST_CASE("flat holonomy on an anisotropic torus: eigenvalue EXACT at finite n (25x contrast)") {
  // theta distinct per axis (no accidental degeneracy); spacings 0.5/0.2/0.1
  // -> weights 4/25/100. The closed form is exact at THIS resolution: any
  // misplacement of a weight (wrong axis, wrong edge) shifts the value.
  const int nx = 8, ny = 12, nz = 16;
  const double hx = 0.5, hy = 0.2, hz = 0.1;
  // thx > pi puts the minimizing momentum at m_x = 1: the ground state is then
  // a genuine plane wave, NOT the constant section the solver starts from --
  // so the eigensolver must actually iterate through the weighted V-cycle.
  const double thx = 4.0, thy = 2.0, thz = 0.7;

  const GaugeLattice L = flatHolonomyLattice(nx, ny, nz, hx, hy, hz, thx, thy, thz);
  const double exact = exactLambdaMin(nx, ny, nz, hx, hy, hz, thx, thy, thz);
  int iters = 0;
  const double lam = smallestEig(L, iters);
  MESSAGE("flat holonomy: exact=" << exact << " covMG-LOBPCG=" << lam << " iters=" << iters);
  CHECK(lam == doctest::Approx(exact).epsilon(1e-7));

  // Second geometry: permuted dims/spacings/holonomies -- catches axis swaps
  // that the first combination might alias.
  const GaugeLattice L2 = flatHolonomyLattice(16, 8, 12, 0.1, 0.5, 0.2, 0.7, 4.0, 2.0);
  const double exact2 = exactLambdaMin(16, 8, 12, 0.1, 0.5, 0.2, 0.7, 4.0, 2.0);
  int iters2 = 0;
  const double lam2 = smallestEig(L2, iters2);
  CHECK(lam2 == doctest::Approx(exact2).epsilon(1e-7));

  // The weighted V-cycle must also SOLVE this operator to tolerance.
  std::vector<cd> b(static_cast<std::size_t>(L.numNodes()), cd(1.0, -0.5));
  std::vector<cd> x(b.size(), cd(0, 0));
  bochner::MgOptions mg;
  mg.tol = 1e-8;
  mg.maxCycles = 100;
  const auto r = bochner::vcycleSolve(L, b, x, mg);
  MESSAGE("flat holonomy V-cycle: " << r.cycles << " cycles");
  CHECK(r.relResidual < mg.tol);
}

TEST_CASE("stretched box two ways: cubic grid (uniform path) vs stretched grid (weighted path)") {
  // One continuum problem -- uniform B_z with nPhi = 2 flux quanta on the
  // periodic box (2, 1, 1) -- discretized (a) with cubic cells 2n x n x n via
  // the trusted scalar-w path and (b) with an n^3 grid stretched 2x along x via
  // the weighted path. Both are O(h^2) discretizations of the same operator,
  // so their smallest eigenvalues must approach each other under refinement.
  const int nPhi = 2;
  double gap[3] = {0, 0, 0};
  double lamB_last = 0.0;
  int idx = 0;
  for (int n : {8, 16, 24}) {
    // (a) cubic cells, uniform scalar weight w = n^2 (the pre-weights solver).
    GaugeLattice A = fluxLattice(2 * n, n, n, nPhi);
    A.w = static_cast<double>(n) * n;
    A.buildTransports();
    // (b) stretched cells hx = 2/n, hy = hz = 1/n via per-edge weights.
    GaugeLattice B = fluxLattice(n, n, n, nPhi);
    const std::size_t NB = static_cast<std::size_t>(n) * n * n;
    const double wx = 0.25 * n * n, wyz = static_cast<double>(n) * n;
    B.setEdgeWeights(std::vector<double>(NB, wx), std::vector<double>(NB, wyz),
                     std::vector<double>(NB, wyz));

    int itA = 0, itB = 0;
    const double lamA = smallestEig(A, itA);
    const double lamB = smallestEig(B, itB);
    gap[idx++] = std::abs(lamA - lamB);
    lamB_last = lamB;
    MESSAGE("n=" << n << ": cubic-2n lambda=" << lamA << " (" << itA
                 << " it), stretched-n lambda=" << lamB << " (" << itB
                 << " it), |gap|=" << std::abs(lamA - lamB));
  }
  // O(h^2) mutual convergence: the gap must shrink under refinement (~4x per
  // doubling; allow slack for the n=24 step being 1.5x).
  CHECK(gap[1] < 0.5 * gap[0]);
  CHECK(gap[2] < 0.7 * gap[1]);
  // Ballpark sanity: the lowest Landau level of -D^2 at B = 2 pi nPhi/(Lx Ly)
  // = 2 pi is lambda ~ B; catch gross (factor-2/axis-swap) errors.
  CHECK(lamB_last > 4.0);
  CHECK(lamB_last < 9.0);
}
