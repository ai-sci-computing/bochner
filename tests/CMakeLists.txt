# Compile doctest's main() once and share it across test executables.
add_library(doctest_main STATIC doctest_main.cpp)
target_link_libraries(doctest_main PUBLIC doctest)
if(BOCHNER_WITH_PETSC)
  target_compile_definitions(doctest_main PUBLIC BOCHNER_WITH_PETSC)
  target_link_libraries(doctest_main PUBLIC bochner_petsc)
endif()

# Helper: register a doctest-based ctest target.
function(bochner_add_test name)
  add_executable(${name} ${ARGN})
  target_link_libraries(${name} PRIVATE bochner_core doctest_main doctest)
  add_test(NAME ${name} COMMAND ${name})
endfunction()

# Helper: register a suite that is NOT built in this configuration as an
# explicit ctest "skipped" entry.
#
# Without this, a gated suite simply does not exist, so `ctest` reports
# "100% tests passed" for both the PETSc-off (18 suites) and PETSc-on (31)
# configurations with nothing to distinguish them -- a first-time reader has no
# way to see that 13 suites were never compiled, including the only
# conjugation-direction and Milnor-trefoil coverage. Registering them keeps the
# green bar honest: ctest prints them as Skipped with the reason.
function(bochner_skip_test name reason)
  add_test(NAME ${name} COMMAND ${CMAKE_COMMAND} -E echo
           "SKIPPED: ${name} is not built in this configuration (${reason})")
  set_tests_properties(${name} PROPERTIES SKIP_REGULAR_EXPRESSION "^SKIPPED:")
endfunction()

bochner_add_test(test_coo_matrix test_coo_matrix.cpp)
bochner_add_test(test_macgrid test_macgrid.cpp)
bochner_add_test(test_grid_operators test_grid_operators.cpp)
bochner_add_test(test_mac_advection test_mac_advection.cpp)
bochner_add_test(test_mac_filaments test_mac_filaments.cpp)
bochner_add_test(test_mac_vortexring test_mac_vortexring.cpp)
bochner_add_test(test_mac_subdivision test_mac_subdivision.cpp)
bochner_add_test(test_gauge_multigrid test_gauge_multigrid.cpp)
bochner_add_test(test_sun_gauge test_sun_gauge.cpp)
bochner_add_test(test_mc_gauge test_mc_gauge.cpp)
bochner_add_test(test_weighted_anisotropic test_weighted_anisotropic.cpp)
bochner_add_test(test_block_lobpcg test_block_lobpcg.cpp)
bochner_add_test(test_gauge_eigen_accuracy test_gauge_eigen_accuracy.cpp)
bochner_add_test(test_poisson_multigrid test_poisson_multigrid.cpp)
# PETSc-free grid-layer physics (masks, no-penetration, open-wall gradient) and
# the explicit viscous diffusion operator: no linear solve, so run by default.
bochner_add_test(test_grid_physics test_grid_physics.cpp)
bochner_add_test(test_diffusion test_diffusion.cpp)
# The assembled connection Laplacian's structural invariants (trivial reduction,
# symmetry, angle scaling) -- no linear solve, so run by default. The frustration
# eigenvalue physics needs SLEPc and stays gated (test_mac_connection_laplacian).
bochner_add_test(test_connection_laplacian_structure test_connection_laplacian_structure.cpp)
bochner_add_test(test_poisson_mgpcg test_poisson_mgpcg.cpp)

# Metal GPU backend parity (GPU float vs CPU double). Opt-in with the backend.
if(BOCHNER_WITH_METAL)
  bochner_add_test(test_metal_sample test_metal_sample.cpp)
endif()

if(BOCHNER_WITH_PETSC)
  bochner_add_test(test_petsc_solver test_petsc_solver.cpp)
  bochner_add_test(test_eigensolver test_eigensolver.cpp)
  bochner_add_test(test_mac_projection test_mac_projection.cpp)
  bochner_add_test(test_channel_flow test_channel_flow.cpp)
  bochner_add_test(test_obstacle_flow test_obstacle_flow.cpp)
  bochner_add_test(test_mac_fluidsolver test_mac_fluidsolver.cpp)
  bochner_add_test(test_mac_characteristic_map test_mac_characteristic_map.cpp)
  bochner_add_test(test_alg5_subroutines test_alg5_subroutines.cpp)
  bochner_add_test(test_mac_connection_laplacian test_mac_connection_laplacian.cpp)
  bochner_add_test(test_mac_extraction test_mac_extraction.cpp)
  bochner_add_test(test_eig_methods test_eig_methods.cpp)
  bochner_add_test(test_gauge_eigen test_gauge_eigen.cpp)
  bochner_add_test(test_convergence_guarantees test_convergence_guarantees.cpp)
else()
  # Same list as above, registered as skipped so the absence is visible.
  foreach(t
      test_petsc_solver test_eigensolver test_mac_projection test_channel_flow
      test_obstacle_flow test_mac_fluidsolver test_mac_characteristic_map
      test_alg5_subroutines test_mac_connection_laplacian test_mac_extraction
      test_eig_methods test_gauge_eigen test_convergence_guarantees)
    bochner_skip_test(${t} "requires BOCHNER_WITH_PETSC=ON")
  endforeach()
endif()

if(NOT BOCHNER_WITH_METAL)
  bochner_skip_test(test_metal_sample "requires BOCHNER_WITH_METAL=ON, Apple only")
endif()
