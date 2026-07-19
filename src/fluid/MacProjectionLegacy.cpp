/// \file
/// The legacy stateless pressure projection -- the ONLY PETSc dependency in
/// src/fluid/.
///
/// Split into its own translation unit so that MacProjection.cpp (MacProjector,
/// the geometric-multigrid projector both viewers use), MacFluidSolver.cpp and
/// MacCharacteristicMap.cpp no longer have to be gated behind
/// BOCHNER_WITH_PETSC. Before the split, this single solveSpdCG call forced the
/// flagship interactive demo to require a from-source PETSc+SLEPc install even
/// though the algorithm it runs is PETSc-free. Same pattern already applied to
/// MacPressureAssembly in fcc1dbe.
#include "fluid/MacProjection.h"

#include "grid/GridOperators.h"
#include "solvers/PetscSolver.h"

namespace bochner {

FaceField projectToDivergenceFree(const MacGrid& g, const FaceField& u) {
  // Closed no-penetration domain: enforce u.n = 0 on the walls (the inhomogeneous
  // Neumann BC -- the projection removes the wall-normal flux too, not just the
  // interior divergence). With the walls zeroed the RHS -div(u) sums to zero. We
  // still pin cell 0 to fix the remaining constant null space.
  FaceField out = u;
  ops::enforceNoPenetration(g, out);

  const std::vector<int> pinned = {0};
  CooMatrix A = pressureLaplacian(g, pinned);

  std::vector<double> rhs = ops::divergence(g, out);
  for (double& r : rhs) r = -r;
  rhs[0] = 0.0;  // pinned cell

  std::vector<double> phi = solveSpdCG(A, rhs);

  FaceField grad = ops::gradient(g, phi);  // zero on the walls -> out.n stays 0
  for (size_t f = 0; f < out.x.size(); ++f) out.x[f] -= grad.x[f];
  for (size_t f = 0; f < out.y.size(); ++f) out.y[f] -= grad.y[f];
  for (size_t f = 0; f < out.z.size(); ++f) out.z[f] -= grad.z[f];
  return out;
}

}  // namespace bochner
