# Foundation layer: geometry + stencils everything else builds on. Carries the
# shared PUBLIC build settings (include root, C++20, OpenMP, the PETSc define),
# which propagate to every module that links grid and to all consumers.
add_library(bochner_grid
  CooMatrix.cpp
  MacGrid.cpp
  MacObstacle.cpp
  GridOperators.cpp
)
target_include_directories(bochner_grid PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_compile_features(bochner_grid PUBLIC cxx_std_20)

if(BOCHNER_WITH_OPENMP AND OpenMP_CXX_FOUND)
  target_link_libraries(bochner_grid PUBLIC OpenMP::OpenMP_CXX)
endif()
if(BOCHNER_WITH_PETSC)
  target_compile_definitions(bochner_grid PUBLIC BOCHNER_WITH_PETSC)
endif()
