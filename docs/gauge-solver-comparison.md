# Gauge-solver comparison: bochner's multigrid vs the lattice-gauge-solvers suite

The connection Laplacian at the heart of the vortex-filament extraction is a **U(1) lattice gauge theory on a regular lattice**. That exact operator is the subject of a [companion study](https://coding.wirbellinie.de/when-preconditioners-make-things-worse-solving-the-u1-connection-laplacian/) by the same author — a controlled comparison of 18 linear solvers / preconditioners on the periodic 3-torus connection Laplacian. Its headline finding: the solver cost is a **near-null-space** phenomenon, not a conditioning one, so classical hypre AMG pays a **3× gauge penalty** (its constant near-null interpolation misses the oscillatory `e^{iφ}` low modes), while *gauge-aware* methods recover the gauge-blind cost.

> **Reproducibility note.** That companion harness is published as a write-up, **not as code**. Everything in §1 below (the linear-solve table, the flat-cycle count, the GAMG comparison) was produced inside it and therefore **cannot be re-run from this repository**. The *eigensolver* results — which are this project's headline claim — use tools committed here (`tools/torus_eig_compare`, `tools/eig_compare`) and are fully reproducible.

bochner's own solver is a **geometric** gauge-aware multigrid: a parallel-transport prolongation (covariant subdivision) makes multigrid work on the connection Laplacian where scalar AMG fails. That is a *different* gauge-aware mechanism from the sibling's algebraic methods (GAMG-adaptive computed-near-null, deflation, SAP), so running it on the same operator is a genuinely new data point.

This document records that head-to-head. Numbers are the **certified median-of-5 regeneration** whose runs the companion article's tables publish, with the certified relative-drop stopping rule (`relativeGsDrop=true`), which is the tools' default. Two tools produced them:

- `lattice-gauge-solvers/tools/bochner_compare` — **not public** (see the note above): runs bochner's geometric gauge-MG V-cycle *inside the sibling harness*, next to all 18 methods, on the same operator + RHS + binary (so wall time is honestly comparable).
- `bochner/tools/torus_eig_compare` — the smallest-eigenpair solve on the same uniform-flux operator: our gauge-MG Rayleigh-quotient eigensolver vs SLEPc Lanczos, with the eigenvector compared as the distance between complex lines.

Both tools first **assert operator equivalence**: bochner's periodic matrix-free apply equals the sibling's assembled `CooMatrix` to ~1e-16, so the comparison is on one and the same system.

## 1. Linear solve — bochner's gauge-MG in the sibling harness

Uniform magnetic field (Landau / Hofstadter example), `nPhi = 4` flux quanta, `rtol = 1e-8`, one RHS. Same binary, same machine. `solve ms` is wall time; the `matvecs` column is discussed in its own subsection below. **All methods run single-threaded — a fair wall-time comparison:** the sibling harness builds bochner *without* OpenMP and PETSc uses one MPI rank, so no method (ours included) gets a threading advantage here (verified: every solve time, bochner's included, is flat from 1 → 11 threads). `bochner_compare 16 4`:

| method | taxonomy | its | matvecs | setup ms | solve ms |
|---|---|---:|---:|---:|---:|
| CG(none) | baseline | 56 | 56 | 0.0 | 2.4 |
| SSOR | gauge-blind | 29 | 29 | 0.0 | 2.9 |
| ICC | gauge-naive | 27 | 27 | 0.4 | 2.4 |
| AMG-hypre | gauge-naive | 14 | 14* | 19.2 | 18.5 |
| GAMG | gauge-naive | 24 | 244 | 12.4 | 7.2 |
| Schwarz-add(DD) | gauge-aware | 32 | 32 | 2.8 | 5.6 |
| GAMG-adaptive | gauge-aware | 16 | 290 | 41.4 | 7.3 |
| deflated-CG | gauge-aware | 76 | 164 | 6.6 | 8.4 |
| **bochner-gauge-MG** | **gauge-aware** | **6** | **294** | **0.0** | **2.1** |

At n=16 bochner's V-cycle has the **fastest solve wall time of every method** and **zero setup** (no hierarchy / factor / near-null build).

### h-optimality under refinement (fixed total flux, `κ ~ n²`)

The multigrid payoff is that the V-cycle count stays flat while gauge-blind CG grows `~n`. Refining n = 8 → 64 at fixed `nPhi = 4` (524 288 DOFs at n=64):

| method | n=8 | n=16 | n=24 | n=32 | n=48 | n=64 |
|---|---:|---:|---:|---:|---:|---:|
| CG(none), its | 28 | 56 | 77 | 108 | 136 | 159 |
| deflated-CG, matvecs | 129 | 164 | 185 | 217 | 245 | 246 |
| AMG-hypre, its | 10 | 14 | 15 | 16 | 17 | 17 |
| GAMG-adaptive, matvecs | 158 | 290 | 342 | 381 | 606 | 691 |
| **bochner-gauge-MG, V-cycles** | **7** | **6** | **5** | **6** | **5** | **5** |

The competitors reproduce the sibling project's *own published* growth factors (CG ~2.7×, deflated-CG 1.43×, AMG-hypre 1.50× over n=8→24) — confirming the integration is faithful — while bochner's geometric gauge-MG holds a **flat 5–7 cycle count across n=8→64**, the best h-optimality on the table. (The cycle-count row above stops at n=64; the n=128 / 4.2M-DOF result in the wall-time table below is a timing, not a logged cycle count. Corrected 2026-07-19 — the earlier wording claimed flat cycles "all the way to n=128", which this table does not show.)

### Wall time — where the geometric multigrid dominates

By wall time the gap widens with resolution — exactly the regime where a geometric multigrid should win. `solve ms` (+ `setup ms`), fastest competitor solve (always ICC) and the best gauge-aware algebraic method (GAMG-adaptive) vs bochner:

| n | DOFs | ICC solve | GAMG-adaptive (solve + setup) | **bochner (solve + 0 setup)** | bochner vs ICC | vs GAMG-adaptive total |
|---|---:|---:|---:|---:|---:|---:|
| 32 | 65 536 | 25.9 | 87.7 + 532 | **17.3** | 1.5× | 36× |
| 48 | 221 184 | 103.1 | 427.8 + 1 986 | **45.1** | 2.3× | 54× |
| 64 | 524 288 | 256.3 | 1 257.8 + 4 819 | **115.2** | 2.2× | 53× |
| 96 | 1 769 472 | 1 254.9 | 5 608.6 + 17 029 | **389.0** | 3.2× | 58× |
| 128 | 4 194 304 | 3 354.3 | 16 532.3 + 42 182 | **966.3** | 3.5× | 61× |

bochner's solve scales ~linearly in DOFs (O(N), flat cycles) and carries **zero setup**, while the algebraic gauge-aware methods drown in a near-null / hierarchy build whose cost explodes: GAMG-adaptive's setup alone grows **0.5 s → 42.2 s** over this range. So the advantage over the best competitor *solve* widens with resolution (1.5× → 3.5×), and against GAMG-adaptive's *total* it holds ~53–61× across n = 48–128 — **966 ms vs ≈58.7 s at 4.2M DOFs.** The direct methods are off the chart entirely: Cholesky's fill makes its setup ~88 s already at n=32.

### Reading the matvecs column (it is not what it looks like)

By raw matvec count bochner (~300) looks worse than CG (56) or AMG-hypre (14). That is a **metric artifact, not a work deficit**, for two reasons:

- **bochner counts every smoother sweep at every level as one matvec** — including the 30 sweeps on the tiny coarsest grid (8–216 nodes), each ticking the counter like a full 524k-node apply. AMG-hypre reads a misleadingly *low* 14 for the opposite reason: its V-cycle sweeps happen inside hypre and PETSc's `MatMult` log never sees them.
- The right peer comparison is the **other true multigrid**: GAMG (244–822) and GAMG-adaptive (290–691), whose PETSc `MatMult` counts likewise include every coarse-level apply. Against those, bochner (~300, flat) is squarely **in the same regime — and it wins wall time by 5–50×.**

The FLOP-honest metric is **fine-grid-equivalent applies** (node-touches ÷ finest node count), which `bochner_compare` now prints. Measured on this operator: **47 applies at n=16, 39 at n=64 — flat, and *below* unpreconditioned CG (56 → 159).** So bochner does *less* actual work than CG, mesh-independently; the high raw count is nearly-free coarse sweeps.

## 2. Eigensolve — covMG-LOBPCG vs SLEPc Lanczos

Smallest eigenpair of the same uniform-flux operator, `tol = 1e-7`, **single-thread — the fair, apples-to-apples setup**: the SLEPc Lanczos baseline runs one MPI rank with no OpenMP, so its wall time is flat across `OMP_NUM_THREADS` (measured: ~640 ms at n=48 whether 1, 4, or 8 threads). Our covMG-LOBPCG *can* additionally use OpenMP (the interactive pipeline does, ~1.5–1.8×), but this table runs it serial too so the speed-up reflects the algorithm, not our parallelism. **And single-thread is not merely fair but *conservative* for us: SLEPc's native parallelism is MPI, and giving Lanczos more cores via `mpirun` *regresses* at this scale — measured 640 ms (1 rank) → 832 (2) → 1399 (4) at n=48, because the sparse eigensolve is communication-bound (a global reduction per orthogonalization step) below ~10⁶ DOFs per rank.** So the serial Lanczos time is close to its best case here; a "both get 4 cores via their native model" comparison (MPI-Lanczos vs our 4-thread covMG-LOBPCG) would be *wider* than the single-thread figures below, not narrower. `torus_eig_compare n 4`:

| n   |      DOFs | Lanczos ms | covMG-LOBPCG ms (iters) |   speedup | \|Δλ\| |
| --- | --------: | ---------: | ----------------------: | --------: | -----: |
| 16  |     8 192 |       10.8 |               4.3 (10)  |      2.5× |  1e-12 |
| 24  |    27 648 |       39.7 |              17.6 (12)  |      2.3× |  1e-12 |
| 32  |    65 536 |      117.2 |              39.6 (11)  |      3.0× |  1e-12 |
| 48  |   221 184 |      658.4 |             112.5 (10)  |  **5.9×** |  1e-12 |
| 64  |   524 288 |    2 413.5 |             288.3 (10)  |  **8.4×** |  1e-12 |
| 96  | 1 769 472 |   13 295.3 |           1 072.2 (11)  | **12.4×** |  1e-12 |

Our gauge-MG Rayleigh-quotient eigensolver (covMG-LOBPCG) is mesh-independent (10–12 certified outer steps at every n) and its **advantage over Lanczos grows with resolution — ~2.3–3× at the small sizes to 12.4× at 1.77M** — because the flat-cycle V-cycle preconditioner keeps the per-step cost O(N) while Lanczos's Krylov depth grows. Eigenvalues agree to ~1e-12 throughout. (The default `ncv` is the swept Lanczos optimum through n=64; tuning `ncv=32` improves the n=96 baseline ~21%, giving 10.3× there. On the near-critical seeded-ring spectrum the margin is larger still even serial — 12.8× at 524k single-threaded vs SLEPc defaults, 8.9× vs the tuned baseline, see `eig_compare` and the companion article's ring table; the pipeline additionally threads our solver where the serial SLEPc baseline cannot, and warm starts in the interactive regime widen it further.) The **backward / inverse-iteration** cost reads off the linear solve: ≈ (single V-cycle solve) × (outer steps) ≈ 114 ms × 10 ≈ 1.1 s at n=64 — covMG-LOBPCG's 288 ms beats that because it folds a *single* V-cycle, not a full solve, into each step.

### Eigenvector — distance between the two complex lines

An eigenvector is defined only up to a global phase, so the right object to compare is the **complex line** `ℂ·ψ` it spans. The distance between the two lines is the **chordal** projective distance `d = √(1 − |⟨v_L, v_R⟩|²) = sin θ` (θ the principal angle; equal to the operator-norm distance between the two line projectors). Note this is not the Fubini–Study *geodesic* distance, which is θ = arccos|⟨v_L,v_R⟩| itself; the two agree to second order at small angles. `d = 0` iff the same line, independent of each vector's phase.

- **Simple mode** (`nPhi = 1`, non-degenerate lowest state): line distance **7e-8**, principal angle **0.000°**, pointwise `|ψ|` relative difference 4e-8. The two solvers land on the **same complex line** to solver tolerance.
- **Degenerate mode** (`nPhi ≥ 2`: the lowest state is an `nPhi`-fold Landau level): the two solvers return *different* lines of the same eigenspace (at `nPhi = 4`, line distance ~0.07), each a genuine eigenvector (residuals ~1e-8). The tool detects the degeneracy and reports it.

## Reproduce

```sh
# Linear solve -- requires the companion harness, which is NOT public. These
# commands are recorded for provenance, not as something a reader can run.
cd ../lattice-gauge-solvers && cmake --build build --target bochner_compare
for n in 8 16 24 32; do ./build/tools/bochner_compare $n 4; done
for n in 48 64; do ./build/tools/bochner_compare $n 4 fast; done  # skip O(n^6) direct methods

# Eigensolve + eigenvector (in the bochner build, PETSc/SLEPc on):
cmake --build build --target torus_eig_compare
for n in 16 24 32 48 64 96; do OMP_NUM_THREADS=1 ./build/tools/torus_eig_compare $n 4; done
OMP_NUM_THREADS=1 ./build/tools/torus_eig_compare 16 1   # non-degenerate eigenvector match
```

Wall times are machine-dependent; the fine-grid-equivalent work and the qualitative ranking (flat V-cycle count, fastest solve, zero setup) are the portable results.
