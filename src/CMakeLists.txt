# The core is split into layered modules, each its own library so the dependency
# direction is enforced by the linker (a cross-layer include becomes a link error):
#   grid        -- foundation: MAC grid, matrix-free stencils, solid masks, sparse matrix
#   solvers     -- linear systems + eigensolvers (incl. the gauge-multigrid research core)
#   fluid       -- Covector Fluids simulation
#   extraction  -- Weissmann-Pinkall vortex-filament extraction
# fluid and extraction are independent siblings layered on grid + solvers.
add_subdirectory(grid)
add_subdirectory(solvers)
add_subdirectory(fluid)
add_subdirectory(extraction)
if(BOCHNER_WITH_METAL)
  add_subdirectory(gpu)  # Metal compute backend (bochner_gpu), on grid
endif()

# Convenience aggregate: link this to pull in the whole core (used by tests/tools/viewer).
add_library(bochner_core INTERFACE)
target_link_libraries(bochner_core INTERFACE
  bochner_grid bochner_solvers bochner_fluid bochner_extraction)
if(BOCHNER_WITH_METAL)
  target_link_libraries(bochner_core INTERFACE bochner_gpu)
endif()
