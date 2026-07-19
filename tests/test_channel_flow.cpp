/// \file
/// Red-green TDD for Phase A of the flow-past-obstacle work: a constant inflow
/// (inhomogeneous-Neumann inlet) balanced by an open (Dirichlet p=0) outlet.
/// The pressure projection must (1) leave a uniform through-flow untouched,
/// (2) return a divergence-free field while preserving the prescribed inlet, and
/// (3) conserve mass (inlet flux == outlet flux).
#include <doctest.h>

#include <cmath>

#include "grid/GridOperators.h"
#include "grid/MacGrid.h"
#include "fluid/MacProjection.h"

using bochner::BoundarySpec;
using bochner::FaceField;
using bochner::MacGrid;
using bochner::MacProjector;
using bochner::Vec3;

namespace {

// Uniform +x stream of speed U on every face.
FaceField uniformStream(const MacGrid& g, double U) {
  FaceField u = bochner::ops::zeroFaceField(g);
  for (double& v : u.x) v = U;  // x-faces carry U; y/z faces stay 0
  return u;
}

double maxAbsDiv(const MacGrid& g, const FaceField& u) {
  const auto d = bochner::ops::divergence(g, u);
  double m = 0.0;
  for (double v : d) m = std::max(m, std::abs(v));
  return m;
}

}  // namespace

TEST_CASE("uniform through-flow is a fixed point of the inflow/outflow projection") {
  MacGrid g(16, 12, 8, 0.1, Vec3{0, 0, 0});
  const double U = 1.3;
  BoundarySpec bc = BoundarySpec::channelFlow(U);  // x-lo inlet, x-hi open, sides closed
  MacProjector proj(g, {}, bc);

  const FaceField in = uniformStream(g, U);
  const FaceField out = proj.project(in);

  // A uniform stream is already divergence-free with this BC, so the projection
  // must not change it.
  for (size_t f = 0; f < in.x.size(); ++f) CHECK(out.x[f] == doctest::Approx(U).epsilon(1e-9));
  for (size_t f = 0; f < in.y.size(); ++f) CHECK(out.y[f] == doctest::Approx(0.0).epsilon(1e-9));
  CHECK(maxAbsDiv(g, out) < 1e-9);
}

TEST_CASE("projection makes a perturbed inflow divergence-free and preserves the inlet") {
  MacGrid g(20, 16, 6, 0.1, Vec3{0, 0, 0});
  const double U = 1.0;
  BoundarySpec bc = BoundarySpec::channelFlow(U);
  MacProjector proj(g, {}, bc);

  // Uniform stream plus a divergent interior blob (breaks div-freeness).
  FaceField u = uniformStream(g, U);
  for (int i = 1; i < g.nx(); ++i)
    for (int j = 0; j < g.ny(); ++j)
      for (int k = 0; k < g.nz(); ++k)
        u.x[g.faceXIndex(i, j, k)] += 0.4 * std::sin(0.7 * i) * std::cos(0.5 * j);

  const FaceField out = proj.project(u);

  CHECK(maxAbsDiv(g, out) < 1e-6);  // divergence-free everywhere
  // The inlet normal velocity stays exactly the prescribed U (not corrected).
  for (int j = 0; j < g.ny(); ++j)
    for (int k = 0; k < g.nz(); ++k)
      CHECK(out.x[g.faceXIndex(0, j, k)] == doctest::Approx(U).epsilon(1e-9));
}

TEST_CASE("inflow/outflow projection conserves mass (inlet flux == outlet flux)") {
  MacGrid g(18, 14, 10, 0.1, Vec3{0, 0, 0});
  const double U = 0.8, h = g.spacing();
  BoundarySpec bc = BoundarySpec::channelFlow(U);
  MacProjector proj(g, {}, bc);

  // Start from zero interior + prescribed inlet: strongly out of balance, so the
  // outlet must open up to carry exactly the inflow.
  FaceField u = bochner::ops::zeroFaceField(g);
  const FaceField out = proj.project(u);

  double inflow = 0.0, outflow = 0.0;
  const double faceArea = h * h;
  for (int j = 0; j < g.ny(); ++j)
    for (int k = 0; k < g.nz(); ++k) {
      inflow += out.x[g.faceXIndex(0, j, k)] * faceArea;         // +x into the domain
      outflow += out.x[g.faceXIndex(g.nx(), j, k)] * faceArea;   // +x out of the domain
    }
  CHECK(inflow == doctest::Approx(U * g.ny() * g.nz() * faceArea).epsilon(1e-9));  // inlet = U*area
  CHECK(outflow == doctest::Approx(inflow).epsilon(1e-4));                          // mass balance
}
