#include "fluid/MacFluidSolver.h"

#include "fluid/MacAdvection.h"
#include "fluid/MacProjection.h"

namespace bochner {

FaceField stepCovectorFluids(const MacGrid& g, const FaceField& u, double dt) {
  // Alg 1: freeze v <- u; covector advection (BFECC); pressure projection.
  FaceField advected = advectCovectorBFECC(g, u, u, dt);
  return projectToDivergenceFree(g, advected);
}

FaceField stepCovectorFluidsMidpoint(const MacGrid& g, const FaceField& u, double dt) {
  // Alg 2: estimate the flow velocity at dt/2, then take the full step. Each
  // advection is followed by a projection (the flow estimate is divergence-free).
  FaceField vHalf = projectToDivergenceFree(g, advectCovectorBFECC(g, u, u, 0.5 * dt));
  return projectToDivergenceFree(g, advectCovectorBFECC(g, u, vHalf, dt));
}

}  // namespace bochner
