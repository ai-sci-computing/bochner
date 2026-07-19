#!/bin/zsh
# Regenerate every benchmark that involves the covMG-LOBPCG eigensolvers, plus
# the demo profiles -- the tables invalidated by the 2026-07-17 C7 (certificate
# floor) + C14 (fixed-cycle preconditioner skips the dead residual matvec)
# changes. Linear-solve tables are NOT regenerated: the linear V-cycle path
# (tol > 0) is untouched by both changes.
#
# Protocol: single-threaded (OMP_NUM_THREADS=1), median of BENCH_REPS=5
# identical runs per solve (tools/BenchTiming.h), matching the certified
# 2026-07-11 record. Demo profiles run at their documented thread counts.
# Tuned-baseline points rerun with -eps_ncv.
#
# Usage: scripts/regen_eigen_tables.sh <output-dir>   (from the build dir)
set -e
out=${1:?output dir required}
mkdir -p "$out"
export BENCH_REPS=5
stage() { echo "=== STAGE $1 start $(date +%H:%M:%S) ==="; }

stage su3            ; OMP_NUM_THREADS=1 ./tools/sun_gauge_bench 3            > "$out/su3.log" 2>&1
stage su3-ncv32      ; OMP_NUM_THREADS=1 ./tools/sun_gauge_bench 3 -eps_ncv 32 > "$out/su3_ncv32.log" 2>&1
stage su2            ; OMP_NUM_THREADS=1 ./tools/sun_gauge_bench 2            > "$out/su2.log" 2>&1
stage su2-ncv32      ; OMP_NUM_THREADS=1 ./tools/sun_gauge_bench 2 -eps_ncv 32 > "$out/su2_ncv32.log" 2>&1
stage ring           ; OMP_NUM_THREADS=1 ./tools/eig_compare                  > "$out/ring.log" 2>&1
stage ring-ncv32     ; OMP_NUM_THREADS=1 ./tools/eig_compare 48 64 -eps_ncv 32 > "$out/ring_ncv32.log" 2>&1
stage ring-ncv48     ; OMP_NUM_THREADS=1 ./tools/eig_compare 64 -eps_ncv 48   > "$out/ring_ncv48.log" 2>&1
stage torus          ; for n in 16 24 32 48 64 96; do OMP_NUM_THREADS=1 ./tools/torus_eig_compare $n 4 >> "$out/torus.log" 2>&1; done
stage torus96-ncv32  ; OMP_NUM_THREADS=1 ./tools/torus_eig_compare 96 4 -eps_ncv 32 > "$out/torus96_ncv32.log" 2>&1
stage warmstart      ; OMP_NUM_THREADS=1 ./tools/warmstart_bench 46 8         > "$out/warmstart.log" 2>&1
stage mc             ; OMP_NUM_THREADS=1 ./tools/mc_gauge_bench all 3         > "$out/mc.log" 2>&1
stage mc48           ; OMP_NUM_THREADS=1 ./tools/mc_gauge_bench refine48 3    > "$out/mc48.log" 2>&1
stage block          ; OMP_NUM_THREADS=1 ./tools/block_eig_bench 3 smooth 16 24 32 > "$out/block.log" 2>&1
stage obstacle-gpu   ; ./tools/obstacle_profile 32 200 gpu                    > "$out/obstacle_gpu.log" 2>&1
stage obstacle-cpu4  ; OMP_NUM_THREADS=4 ./tools/obstacle_profile 32 200 cpu  > "$out/obstacle_cpu4.log" 2>&1
stage obstacle-cpu1  ; OMP_NUM_THREADS=1 ./tools/obstacle_profile 32 200 cpu  > "$out/obstacle_cpu1.log" 2>&1
stage pipeline-4t    ; OMP_NUM_THREADS=4 ./tools/pipeline_profile 46 10       > "$out/pipeline_4t.log" 2>&1
stage pipeline-1t    ; OMP_NUM_THREADS=1 ./tools/pipeline_profile 46 10       > "$out/pipeline_1t.log" 2>&1
stage block64        ; OMP_NUM_THREADS=1 ./tools/block_eig_bench 3 smooth 64  > "$out/block64.log" 2>&1
echo "=== REGEN COMPLETE $(date +%H:%M:%S) ==="
