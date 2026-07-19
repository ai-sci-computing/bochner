# Weissmann-Pinkall vortex-filament extraction: the connection Laplacian on the
# dual cell graph and the zero-set trace/link into filament curves. The eigensolve
# itself is composed in from solvers by the pipeline (tests/tools/viewer).
add_library(bochner_extraction
  MacConnectionLaplacian.cpp
  MacFilaments.cpp
)
target_link_libraries(bochner_extraction PUBLIC bochner_grid)
