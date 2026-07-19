# PETSc-free tools build unconditionally.

# Linear-solver comparison: gauge-MG V-cycle vs plain CG (no PETSc/SLEPc needed).
add_executable(torus_seed_check torus_seed_check.cpp)
target_link_libraries(torus_seed_check PRIVATE bochner_core)

add_executable(aniso_bench aniso_bench.cpp)
target_link_libraries(aniso_bench PRIVATE bochner_core)

add_executable(mg_bench mg_bench.cpp)
target_link_libraries(mg_bench PRIVATE bochner_core)

# Non-abelian transferability demo: SU(2)/SU(3) covariant Laplacian, gauge-MG
# mesh-independence vs CG + covMG-LOBPCG ground state (no PETSc/SLEPc needed).
add_executable(sun_gauge_bench sun_gauge_bench.cpp)
target_link_libraries(sun_gauge_bench PRIVATE bochner_core)

# Referee experiment: Wilson-action Monte-Carlo SU(3) configurations (rough but
# CORRELATED fields) -- beta sweep + refinement series for the paper. Links the
# SLEPc Lanczos baseline when PETSc is on (same pattern as sun_gauge_bench).
add_executable(mc_gauge_bench mc_gauge_bench.cpp)
target_link_libraries(mc_gauge_bench PRIVATE bochner_core)

add_executable(ablation_bench ablation_bench.cpp)
target_link_libraries(ablation_bench PRIVATE bochner_core)

add_executable(convergence_factor_bench convergence_factor_bench.cpp)
target_link_libraries(convergence_factor_bench PRIVATE bochner_core)

add_executable(block_eig_bench block_eig_bench.cpp)
target_link_libraries(block_eig_bench PRIVATE bochner_core)

add_executable(seed_slice seed_slice.cpp)
target_link_libraries(seed_slice PRIVATE bochner_core)

# Matrix export for external eigensolver baselines (PyAMG LOBPCG): the U(1)
# uniform-flux torus / smooth SU(d) operator as a complex MatrixMarket file.
add_executable(export_operator export_operator.cpp)
target_link_libraries(export_operator PRIVATE bochner_core)

# The remaining tools use the SLEPc-backed EigenSolver / CF pipeline (PETSc only).
if(BOCHNER_WITH_PETSC)
  # Grid demo: closed CF sim <-> W-P extractor loop on the MAC grid.
  add_executable(ring_demo_grid ring_demo_grid.cpp)
  target_link_libraries(ring_demo_grid PRIVATE bochner_core)

  # Re-quantization demo: Hill's vortex decomposed into nested rings vs hbar.
  add_executable(hill_decomp hill_decomp.cpp)
  target_link_libraries(hill_decomp PRIVATE bochner_core)

  # Phase-3 benchmark: smallest-eigenvector solve methods vs frustration & size.
  add_executable(eig_benchmark eig_benchmark.cpp)
  target_link_libraries(eig_benchmark PRIVATE bochner_core)

  # Phase-3 warm-start study: cold vs warm per-frame eigensolve over a real ring
  # sim sequence (the interactivity lever).
  add_executable(warmstart_bench warmstart_bench.cpp)
  target_link_libraries(warmstart_bench PRIVATE bochner_core)

  # Experiment A: gauge-aware subdivision vs the eigensolver for extraction.
  add_executable(subdivision_demo subdivision_demo.cpp)
  target_link_libraries(subdivision_demo PRIVATE bochner_core)

  # Eigensolver comparison: gauge-MG covMG-LOBPCG vs SLEPc Lanczos.
  add_executable(eig_compare eig_compare.cpp)
  target_link_libraries(eig_compare PRIVATE bochner_core)

  # Eigensolve on the lattice-gauge-solvers setup (periodic torus, uniform flux):
  # gauge-MG covMG-LOBPCG vs SLEPc Lanczos + the backward-iteration estimate.
  add_executable(torus_eig_compare torus_eig_compare.cpp)
  target_link_libraries(torus_eig_compare PRIVATE bochner_core)

  # Chebyshev-filtered Lanczos baseline (SLEPc STFILTER) vs plain Lanczos /
  # Krylov-Schur, on the same U(1)/SU(d) operators as export_operator.
  add_executable(chebyshev_lanczos_bench chebyshev_lanczos_bench.cpp)
  target_link_libraries(chebyshev_lanczos_bench PRIVATE bochner_core)

  # Pressure-projection solver study (PC x tolerance on a real scalar Poisson).
  add_executable(pressure_bench pressure_bench.cpp)
  target_link_libraries(pressure_bench PRIVATE bochner_core)

  # Full per-frame pipeline profiler (sim / connection / eigensolve / extraction).
  add_executable(pipeline_profile pipeline_profile.cpp)
  target_link_libraries(pipeline_profile PRIVATE bochner_core)

  # Headless flow-past-obstacle demo profiler (the paper's demo numbers:
  # CPU 1T / CPU nT / GPU per-stage timings + eigensolver A/B).
  add_executable(obstacle_profile obstacle_profile.cpp)
  target_link_libraries(obstacle_profile PRIVATE bochner_core)
endif()
