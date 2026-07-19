/// \file
/// Red-green TDD: the smallest-eigenvector methods agree. Backward (inverse)
/// iteration, shift-invert Krylov-Schur, LOBPCG, and Generalized Davidson must
/// all return the same smallest eigenvalue of the connection Laplacian as the
/// SLEPc Lanczos baseline, with a genuine eigenvector. This anchors the
/// eig_benchmark comparison and is the baseline for the planned MG eigensolver.
#include <doctest.h>

#include <cmath>
#include <vector>

#include "solvers/EigenSolver.h"
#include "grid/GridOperators.h"
#include "extraction/MacConnectionLaplacian.h"
#include "grid/MacGrid.h"

using bochner::CooMatrix;
using bochner::FaceField;
using bochner::InnerPC;
using bochner::MacGrid;
namespace ops = bochner::ops;

namespace {

double wobble(int n) { return std::sin(0.8 * n + 0.3) + 0.4 * std::cos(2.3 * n); }

// A frustrated connection Laplacian (generic O(1) angles -> real holonomy).
CooMatrix frustrated(const MacGrid& g) {
  FaceField th = ops::zeroFaceField(g);
  for (size_t f = 0; f < th.x.size(); ++f) th.x[f] = 0.6 * wobble((int)f + 2);
  for (size_t f = 0; f < th.y.size(); ++f) th.y[f] = 0.6 * wobble((int)f + 5);
  for (size_t f = 0; f < th.z.size(); ++f) th.z[f] = 0.6 * wobble((int)f + 11);
  return bochner::connectionLaplacian(g, th);
}

double residual(const CooMatrix& A, const bochner::EigenPair& e) {
  std::vector<double> Av(A.rows(), 0.0);
  for (const auto& t : A.compressed()) Av[t.row] += t.value * e.vector[t.col];
  double m = 0.0;
  for (size_t i = 0; i < Av.size(); ++i) m = std::max(m, std::abs(Av[i] - e.value * e.vector[i]));
  return m;
}

}  // namespace

TEST_CASE("backward iteration and Lanczos agree on the smallest eigenvalue") {
  MacGrid g(8, 8, 8, 1.0);
  const CooMatrix E = frustrated(g);

  const auto lan = bochner::smallestEigenpairLanczos(E, 1e-9);
  const auto inv = bochner::smallestEigenpair(E, 1e-7);  // CG + ICC backsolve

  CHECK(lan.value > 1e-3);  // genuinely frustrated (mass gap)
  CHECK(inv.value == doctest::Approx(lan.value).epsilon(2e-3));
  CHECK(inv.iterations > 0);
  CHECK(lan.iterations > 0);
  CHECK(residual(E, inv) < 1e-3);
  CHECK(residual(E, lan) < 1e-5);
}

TEST_CASE("backward iteration reaches the same eigenvalue with either inner PC") {
  MacGrid g(6, 6, 6, 1.0);
  const CooMatrix E = frustrated(g);
  const double ref = bochner::smallestEigenpairLanczos(E, 1e-9).value;

  for (InnerPC pc : {InnerPC::ICC, InnerPC::AMG}) {
    const auto e = bochner::smallestEigenpair(E, 1e-7, nullptr, pc);
    CHECK(e.value == doctest::Approx(ref).epsilon(3e-3));
  }
}

TEST_CASE("preconditioned LOBPCG agrees on the smallest eigenvalue") {
  MacGrid g(8, 8, 8, 1.0);
  const CooMatrix E = frustrated(g);
  const auto lan = bochner::smallestEigenpairLanczos(E, 1e-9);

  // The block method must clear the degenerate lowest pair (the #29 limiter).
  const auto e = bochner::smallestEigenpairLOBPCG(E, 1e-8, nullptr, InnerPC::AMG);
  CHECK(e.value == doctest::Approx(lan.value).epsilon(2e-3));
  CHECK(e.iterations > 0);
  CHECK(residual(E, e) < 1e-3);
}

TEST_CASE("shift-invert Krylov-Schur reaches the same eigenvalue") {
  MacGrid g(8, 8, 8, 1.0);
  const CooMatrix E = frustrated(g);
  const auto lan = bochner::smallestEigenpairLanczos(E, 1e-9);

  const auto si = bochner::smallestEigenpairShiftInvert(E, 1e-8, nullptr, InnerPC::AMG);
  CHECK(si.value == doctest::Approx(lan.value).epsilon(2e-3));
  CHECK(si.iterations > 0);
  CHECK(residual(E, si) < 1e-3);
}

TEST_CASE("Generalized Davidson reaches the same eigenvalue") {
  MacGrid g(8, 8, 8, 1.0);
  const CooMatrix E = frustrated(g);
  const auto lan = bochner::smallestEigenpairLanczos(E, 1e-9);

  const auto gd = bochner::smallestEigenpairDavidson(E, 1e-8, nullptr, InnerPC::AMG);
  CHECK(gd.value == doctest::Approx(lan.value).epsilon(2e-3));
  CHECK(gd.iterations > 0);
  CHECK(residual(E, gd) < 1e-3);
}

TEST_CASE("smallestEigenpairsLanczos returns ascending eigenvalues, lowest matching") {
  MacGrid g(8, 8, 8, 1.0);
  const CooMatrix E = frustrated(g);
  const double ref = bochner::smallestEigenpairLanczos(E, 1e-9).value;

  const auto pairs = bochner::smallestEigenpairsLanczos(E, 4, 1e-9);
  REQUIRE(pairs.size() == 4);
  CHECK(pairs[0].value == doctest::Approx(ref).epsilon(1e-6));  // lowest matches the single solve
  for (size_t i = 1; i < pairs.size(); ++i) CHECK(pairs[i].value >= pairs[i - 1].value - 1e-9);
  CHECK(residual(E, pairs[0]) < 1e-5);  // genuine eigenvector
}

TEST_CASE("warm start does not change the converged eigenvalue") {
  MacGrid g(8, 8, 8, 1.0);
  const CooMatrix E = frustrated(g);
  const auto cold = bochner::smallestEigenpair(E, 1e-7);
  // Warm-starting from the already-converged vector should reproduce it.
  const auto warm = bochner::smallestEigenpair(E, 1e-7, &cold.vector);
  CHECK(warm.value == doctest::Approx(cold.value).epsilon(2e-3));
  CHECK(warm.iterations > 0);
}
