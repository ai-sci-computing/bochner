#!/bin/zsh
# Continuation of regen_eigen_tables.sh after the eig_compare argv fix (the
# original run died at ring-ncv48: option tokens were parsed as lattice sizes).
# Completed and valid from the first run: su3, su3-ncv32, su2, su2-ncv32, ring
# (default sizes; superseded by ring-full below for the n=64 row).
set -e
out=${1:?output dir required}
mkdir -p "$out"
export BENCH_REPS=5
stage() { echo "=== STAGE $1 start $(date +%H:%M:%S) ==="; }

stage ring-full      ; OMP_NUM_THREADS=1 ./tools/eig_compare 16 24 32 48 64   > "$out/ring_full.log" 2>&1
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
# n=64 wide-band cut (user decision 2026-07-17): m=32 omitted -- the wide-band
# verdict (baseline leads) is settled at n=16..32 and the cell alone costs ~2 h
# of single-threaded protocol time; the measured cells bracket the crossover.
stage block64        ; OMP_NUM_THREADS=1 ./tools/block_eig_bench 3 smooth 64 m=1,2,4,8,16 > "$out/block64.log" 2>&1
echo "=== REGEN CONTINUATION COMPLETE $(date +%H:%M:%S) ==="
