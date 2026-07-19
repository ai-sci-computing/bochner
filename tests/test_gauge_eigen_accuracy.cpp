/// \file
/// PETSc-free accuracy check for the research kernel. The default build has
/// BOCHNER_WITH_PETSC OFF, which gates the SLEPc-comparison eigensolver tests --
/// so without this the gauge-MG eigensolver's *accuracy* goes unverified in the
/// default `ctest` (only the linear V-cycle runs). Here the smallest eigenvalue
/// found by `smallestEigenpairGaugeMG` (Rayleigh-quotient min preconditioned by
/// the gauge-MG V-cycle) is checked against an independent reference: inverse
/// power iteration driven by the matrix-free operator + CG (both PETSc-free, and
/// the operator is separately verified against the assembled matrix). Two very
/// different eigen-methods agreeing on the smallest eigenvalue is the check.
#include <doctest.h>

#include <cmath>
#include <complex>
#include <vector>

#include "extraction/MacConnectionLaplacian.h"
#include "fluid/MacVortexRing.h"
#include "grid/MacGrid.h"
#include "solvers/GaugeEigen.h"
#include "solvers/GaugeMultigrid.h"

using bochner::FaceField;
using bochner::GaugeLattice;
using bochner::MacGrid;
using bochner::Vec3;
using cd = std::complex<double>;

namespace {

// A frustrated connection (SPD connection Laplacian with a positive mass gap):
// the seeded vortex ring, as used by the V-cycle tests.
FaceField ringTheta(const MacGrid& g) {
  const double R = 0.7, Gamma = 1.0, hbar = Gamma / (2.0 * M_PI);
  const FaceField u = bochner::vortexRingFaceField(g, {0, 0, 0}, {0, 0, 1}, R, Gamma, 0.15);
  return bochner::connectionAngles(g, u, hbar);
}

double reDot(const std::vector<cd>& a, const std::vector<cd>& b) {
  double s = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) s += std::real(std::conj(a[i]) * b[i]);
  return s;
}
void normalize(std::vector<cd>& v) {
  const double n = std::sqrt(reDot(v, v));
  for (cd& z : v) z /= n;
}

// Smallest eigenvalue via inverse power iteration: repeatedly apply E^{-1} (by
// CG), normalize; the iterate converges into the smallest eigenspace, and its
// Rayleigh quotient converges down to the smallest eigenvalue.
double smallestByInversePower(const GaugeLattice& lat, int N) {
  std::vector<cd> x(N);
  for (int i = 0; i < N; ++i) x[i] = cd(std::cos(0.3 * i), std::sin(0.17 * i));
  normalize(x);
  for (int it = 0; it < 120; ++it) {
    std::vector<cd> y(N, cd(0, 0));
    bochner::cgSolve(lat, x, y, 1e-10);  // y ~= E^{-1} x
    normalize(y);
    x.swap(y);
  }
  const std::vector<cd> Ex = bochner::applyConnectionLaplacian(lat, x);
  return reDot(x, Ex) / reDot(x, x);
}

}  // namespace

TEST_CASE("gauge-MG eigensolver finds the true smallest eigenvalue (PETSc-free)") {
  const MacGrid g(6, 6, 6, 0.25, Vec3{-0.75, -0.75, -0.75});
  const GaugeLattice lat = bochner::gaugeLatticeFromFaces(g, ringTheta(g));
  const int N = g.numCells();

  const double lamRef = smallestByInversePower(lat, N);

  bochner::GaugeEigenOptions opts;
  opts.tol = 1e-8;
  const bochner::GaugeEigenResult r = bochner::smallestEigenpairGaugeMG(lat, nullptr, opts);

  INFO("gauge-MG lambda=" << r.eigenvalue << "  ref(inverse-power)=" << lamRef
                          << "  residual=" << r.residual);
  CHECK(r.eigenvalue > 0.0);                                     // frustrated -> mass gap
  CHECK(r.residual < 1e-6);                                      // a genuine eigenpair of E
  CHECK(r.eigenvalue == doctest::Approx(lamRef).epsilon(0.01));  // = the smallest (independent ref)
}

TEST_CASE("a zero/non-finite initial guess falls back to a valid start (A3)") {
  const MacGrid g(6, 6, 6, 0.25, Vec3{-0.75, -0.75, -0.75});
  const GaugeLattice lat = bochner::gaugeLatticeFromFaces(g, ringTheta(g));
  const int N = g.numCells();

  bochner::GaugeEigenOptions opts;
  opts.tol = 1e-8;

  // An all-zero guess must NOT short-circuit into a false-converged zero
  // "eigenpair" (norm2(x)==0 skips the normalize, Ex=rho=res=0 < tol). The
  // solver should reset to a valid start and return the true smallest eigenpair.
  SUBCASE("all-zero guess") {
    const std::vector<cd> zero(N, cd(0.0, 0.0));
    const bochner::GaugeEigenResult r = bochner::smallestEigenpairGaugeMG(lat, &zero, opts);
    double vnorm = 0.0;
    for (const cd& z : r.vector) vnorm += std::norm(z);
    CHECK(vnorm > 0.5);              // a real, unit-norm eigenvector -- not the zero vector
    CHECK(r.eigenvalue > 1e-6);      // the frustrated mass gap, not a spurious 0
    CHECK(r.residual < 1e-6);        // and it is a genuine eigenpair
  }

  // A denormal/near-zero guess would make 1/norm2 overflow -> NaN iterate; the
  // same fallback (non-finite norm -> reset) must catch it.
  SUBCASE("near-zero denormal guess") {
    std::vector<cd> tiny(N, cd(5e-309, 0.0));  // subnormal; sum of norms underflows to 0
    const bochner::GaugeEigenResult r = bochner::smallestEigenpairGaugeMG(lat, &tiny, opts);
    CHECK(std::isfinite(r.eigenvalue));
    CHECK(r.eigenvalue > 1e-6);
    CHECK(r.residual < 1e-6);
  }
}
