# Covector Fluids simulation: covector advection, pressure projection, the
# CF+MCM long-time flow map, and the vortex-ring seeders.
add_library(bochner_fluid
  MacAdvection.cpp
  MacExtrapolate.cpp
  MacPressureAssembly.cpp
  MacVortexRing.cpp
  PoissonMGPCG.cpp
  # PETSc-free: MacProjector (the geometric-MG projector both viewers use), the
  # CF+MCM flow map, and the seeders. The only PETSc dependency in this library
  # is the legacy stateless projection, isolated in MacProjectionLegacy.cpp
  # below -- without that split, one solveSpdCG call gated this whole set and
  # forced the interactive demo to require a from-source PETSc install.
  MacProjection.cpp
  MacCharacteristicMap.cpp
)
target_link_libraries(bochner_fluid PUBLIC bochner_grid bochner_solvers)

if(BOCHNER_WITH_METAL)
  target_link_libraries(bochner_fluid PUBLIC bochner_gpu)  # GPU advection sub-steps
endif()

if(BOCHNER_WITH_PETSC)
  # projectToDivergenceFree (MacProjectionLegacy.cpp) is the sole PETSc user;
  # MacFluidSolver.cpp is gated only because it calls it.
  target_sources(bochner_fluid PRIVATE
    MacProjectionLegacy.cpp MacFluidSolver.cpp)
  target_link_libraries(bochner_fluid PUBLIC bochner_petsc)
endif()
