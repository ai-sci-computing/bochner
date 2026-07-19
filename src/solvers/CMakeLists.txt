# Linear systems + eigensolvers: the scalar/gauge multigrids, the gauge-aware
# prolongation, and the SLEPc-backed baselines. The research core lives here.
add_library(bochner_solvers
  PoissonMultigrid.cpp
  BlockLobpcg.cpp
  GaugeMultigrid.cpp
  GaugeEigen.cpp
  SunGauge.cpp
  WilsonMc.cpp
  MacSubdivision.cpp
)
target_link_libraries(bochner_solvers PUBLIC bochner_grid)

if(BOCHNER_WITH_PETSC)
  target_sources(bochner_solvers PRIVATE PetscSolver.cpp EigenSolver.cpp)
  target_link_libraries(bochner_solvers PUBLIC bochner_petsc)
endif()
