# The gauge-aware multigrid method

This document describes the project's research contribution: a **gauge-aware multigrid** for the U(1) connection Laplacian, used to find the smallest eigenvector — the vortex-filament field — fast enough for interactive extraction. It covers the operator and why it is hard, the gauge-aware prolongation (the covariant subdivision), the linear V-cycle, the Rayleigh-quotient eigensolver built on it, and the measured results. The companion article is the authoritative record of the method, its theory, and every benchmark table; this is the code-side tour. (The article generalizes everything here to per-edge weights with series-conductance coarsening and to SU(d) matrix links — `src/solvers/SunGauge.*` — while this document stays with the uniform-weight U(1) case the extraction pipeline uses.)

Context for the surrounding pipeline is in [`simulation-algorithm.md`](simulation-algorithm.md). Code: `src/solvers/GaugeMultigrid.*`, `src/solvers/GaugeEigen.*`, `src/solvers/MacSubdivision.*`.

---

## 1. The operator

On the dual lattice (cell centers of the MAC grid, linked through faces) the **connection Laplacian** acts on a complex section `ψ` (one complex value per node). Each axis edge carries a parallel transport `e^{iθ}` (the link variable), where `θ` is the velocity-derived connection angle (`θ_f = u_f·h/ħ`; see the simulation doc). With edge weight `w = 1/h²`:

```
(E ψ)_c = w · Σ_n ( ψ_c − P^∇_{n→c} ψ_n )
```

summed over the existing axis neighbours `n` of cell `c`. The transport from the **low** neighbour `(i−1)` into `c` is `e^{+iθ}`, and from the **high** neighbour `(i+1)` it is `e^{−iθ}` (its conjugate). Missing neighbours at the boundary are simply absent — the homogeneous-Neumann condition. This is `E = (d^∇)† d^∇` with `(d^∇ψ)_f = ψ_b − e^{iθ_f}ψ_a`; it is Hermitian and positive (semi-)definite. (The article writes this operator `L`.)

`E` is matrix-free (`applyConnectionLaplacian`), and verified identical to the assembled real `2N×2N` matrix (`connectionLaplacian`) to 1e-12.

**Why the smallest eigenvector is hard.** The eigenvector encodes a localized phase singularity (the filament). Three obstacles, all measured:

- **Frustration / holonomy.** A non-pure-gauge connection lifts `λ_min` off zero (a gap). That holonomy-driven near-kernel is exactly what *standard* smoothers and constant-near-kernel AMG miss: the slow modes are constant only **up to parallel transport**, so gauge-blind coarse spaces misrepresent them (classical AMG pays a measured ~3× gauge penalty on this operator; see [`gauge-solver-comparison.md`](gauge-solver-comparison.md)).
- **Degenerate pairs.** The real 2×2 embedding makes every eigenvalue a degenerate pair, so single-vector inverse iteration converges on the gap to the next *distinct* pair — which shrinks with frustration (an ~80× blow-up in the early method survey).
- **It is not a conditioning problem.** The operator is SPD and its conditioning is unremarkable — under continuum refinement at fixed physical flux, κ grows ~n² just like the scalar Laplacian's (the matched-conditioning control in the article and the sibling comparison is built on this). The hardness is the *gauge structure* of the near-kernel, not ill-conditioning, which is why better generic linear-solve preconditioners did not help and the matvec-only **Lanczos** baseline was the method to beat. Lanczos, however, grows super-linearly in wall-clock with size and never reached interactivity (~2 s/frame at 97k cells).

The fix is not a better linear solve — it is a **gauge-aware multigrid** that captures the holonomy in its intergrid transfer.

---

## 2. The gauge-aware prolongation (covariant subdivision)

The hardest ingredient of any multigrid for a connection Laplacian is the prolongation. Standard interpolation averages neighbouring coarse values with plain weights, which implicitly assumes the near-kernel is **locally constant** (smooth). For a connection that is wrong: the slow modes are constant only **up to parallel transport** (covariantly constant). Averaging across a phase jump injects frustration energy — the cause of the gauge-blind slowdown.

The cure is to insert the transport into the average — a **covariant subdivision** that carries each coarse value along the link phase before combining it, so a covariantly-constant mode prolongs to itself. Implemented for the cell-centered placement in `subdivisionSection` (`src/solvers/MacSubdivision.*`).

**Hierarchy by decimation.** Coarsening keeps every other node along each axis, so a coarse node *coincides* with a fine node (the levels nest with no geometric interpolation error). A coarse edge spans two fine edges, so its transport is the composition of the two — i.e. the **coarse link angle is the sum of the two fine link angles** (a U(1) connection restricts by summing links). Every dimension must be divisible by `2^levels`.

**One covariant-averaging rule.** Seed the coarsest lattice with `ψ ≡ 1`, then prolong up one level at a time. Classify each fine node by how many of its coordinates are odd (how many axes it was *not* inherited along):

- **0 odd** — a retained coarse node: copy its value.
- **k odd** (k = 1,2,3) — average its immediate neighbours along each odd axis (2k of them, fewer at the boundary), each **parallel-transported to this node**:

  ```
  ψ_m = (1/#N) Σ_{n ∈ N} P^∇_{n→m} ψ_n
  ```

Processing passes in order k = 1,2,3 guarantees every neighbour used is already filled. This single rule reproduces the edge-/face-/cube-midpoint averaging of the original cube formulation (1-odd = edge midpoint, 2-odd = face, 3-odd = cube center). For a trivial connection it is ordinary multilinear interpolation; for a pure gauge it transports exactly.

**What the bare subdivision produces — measured precisely.** Prolonging `ψ ≡ 1` with no solve yields a smooth connection-adapted section, and its zero-set topology depends on the coarsest base. Coarsened all the way to a **single site**, the zero set of the section on a seeded-ring operator is exactly one closed ring at every tested size (topology correct; geometry rough — the loop is displaced and tilted, and its winding is accumulated holonomy from the one seed value). With any coarsest base of `2³` or larger — including every practical hierarchy — the transported averages interfere less coherently and the zero set is **empty** (`min |ψ|` ≈ 0.7–0.98). As an eigensolver warm start the seed is **iteration-neutral** in both regimes: the strong V-cycle preconditioner erases the head start within an iteration or two. So the subdivision is the *transfer operator* of the method and an instructive illustration of holonomy interference — not a shortcut extractor; the eigensolve below is what traces filaments.

On a structured grid with a *known* connection, this writes down in closed form the gauge-aware interpolation that lattice-QCD adaptive aggregation (Babich et al. 2010; Frommer et al. DD-αAMG) and Maxwell auxiliary-space AMG (Hiptmair–Xu) have to *learn*. That is the payoff of the regular lattice.

---

## 3. The linear V-cycle

`vcycleSolve` solves `E x = b` by gauge-aware multigrid V-cycles (`src/solvers/GaugeMultigrid.*`). Components:

- **Operator (matvec).** Matrix-free complex `E·x` as above. The link transports `e^{iθ}` are **precomputed once** per level (`buildTransports`) so the hot loop multiplies a stored complex instead of evaluating a transcendental per edge per sweep (~3× single-thread).
- **Smoother.** Red-black **Gauss–Seidel**. The dual lattice is bipartite under parity `(i+j+k) mod 2`, so one colour then the other is a Gauss–Seidel sweep in which every update within a colour is independent — a strong smoother that is also parallel within each colour.
- **Restriction.** The **exact adjoint** of the prolongation (`R = P^H`), implemented by running the prolongation cascade in reverse with conjugate transports.
- **Coarse operator.** The connection Laplacian **rediscretized** on each level's summed links (`w → w/4` per level).
- **Coarse correction with an optimal step.** The transport-cascade prolongation is not standard trilinear, so a naive rediscretized + adjoint-restriction V-cycle is scale-inconsistent and *diverges*. The fix is an **optimal A-energy step** on the coarse correction `p = P e_c`:

  ```
  α = Re⟨p, r⟩ / ⟨p, E p⟩ ,     x ← x + α p
  ```

For SPD `E` this is a guaranteed descent step in the energy norm, so it cannot diverge and it self-scales the correction — decoupling convergence from any transfer/operator scale mismatch (and matching the Rayleigh-quotient spirit of the eigensolver in §4).

  **What the α step costs: linearity.** Because α depends on the residual it is applied to, the V-cycle is a *nonlinear* operator — homogeneous of degree 1 but not additive (measured additivity defect `|M(u+v) − Mu − Mv|` ≈ 2.3e-2 at scale 3.2, versus ~4e-15 with the step off). Together with the smoother's fixed red-then-black colour order (relative asymmetry ≈ 2e-1 at production settings), this means **the gauge V-cycle is not an SPD linear preconditioner and must not be used inside CG.** It is used only as a *stationary* solver and as a **LOBPCG** preconditioner, which needs search directions rather than an SPD linear `T`. This is exactly the asymmetry between the two multigrids in this project: the pressure MGPCG V-cycle *is* genuinely SPD (`R = Pᵀ`, true Galerkin `PᵀAP`, symmetric sweeps) and therefore *is* wrapped in CG; this one is not, and is not. The trade is deliberate and the ablation table below is the evidence for it — dropping α diverges on the flux operator.

  The article proves the spectral-equivalence facts behind this (`P^H E P ⪰ 2Ẽ` for every connection, exact two-sided bounds in the flat case) and shows by ablation that removing either ingredient — the α-step or the covariant transfer — destroys convergence or mesh-independence.

**Result.** Convergence is **mesh-independent**: on the uniform-flux torus (fixed flux quanta) the V-cycle count is flat at 5–7 cycles to 1e-8 across n = 8→64, while unpreconditioned CG grows ~n (28 → 159 iterations over the same range). The wall-time series extends to n = 128 (4.2M real DOFs) at the same O(N) per-cycle cost; the *cycle-count* table stops at n = 64, so the flat-count claim is stated only over the range it covers. Head-to-head against 18 assembled solvers/preconditioners on the same operator, the V-cycle has the fastest solve wall time with zero setup at every measured size — see [`gauge-solver-comparison.md`](gauge-solver-comparison.md) for the tables.

All kernels (matvec, GS sweeps) are OpenMP-parallel; being stencil kernels they are memory-bandwidth-bound, so the practical sweet spot is ~4 threads.

---

## 4. The eigensolver (covMG-LOBPCG)

The smallest eigenpair is found by **LOBPCG preconditioned by the gauge-MG V-cycle** — a standalone method (no SLEPc), warm-startable (`smallestEigenpairGaugeMG`, `src/solvers/GaugeEigen.*`). It is the eigen-analogue of the linear V-cycle: the covariant transfer supplies the coarse space; the V-cycle does the smoothing.

Each outer step, with the current unit-norm `x` and `ρ = ⟨x, Ex⟩`:

1. **eigen-residual** `r = Ex − ρ x`; stop if `‖r‖/max(|ρ|, floor) < tol` (the floor, `certFloorRel`, only guards the near-pure-gauge limit `λ→0`; it is inactive on every gapped operator).
2. **precondition** `w ≈ E⁻¹ r` with exactly **one** gauge-MG V-cycle (`precCycles`, default 1 — measured as the sweet spot; the option exists as a knob).
3. **locally-optimal step (3-term LOBPCG).** Form the trial subspace `{x, w, x_prev}` (the previous iterate is the conjugate direction), orthonormalize by modified Gram–Schmidt (dropping near-dependent directions with a scale-relative test), and take the **Rayleigh-quotient-optimal** combination via a small complex-Hermitian eigenproblem (`H = Q^H E Q`, solved by a cyclic Hermitian Jacobi routine for `m ≤ 3`). Replace `x` by that combination; remember the old `x` for next step.

The V-cycle preconditioner supplies the multigrid acceleration the degenerate/clustered low spectrum otherwise denies single-vector iteration; the 3-term LOBPCG roughly halves the outer iterations versus plain preconditioned steepest descent. The gap keeps `E` SPD so the preconditioner apply is well-posed.

A **block variant** (`lowestEigenpairsGaugeMG` / `BlockLobpcg.*`) computes the lowest-m band with per-pair certificates — it resolves exact multiplets (Landau levels, Kramers-doubled SU(2) spectra) that single-vector Krylov methods silently truncate. Its options include guard vectors and optional soft/hard locking of converged pairs (defaults off; see the header docs).

**Warm starts.** `x` enters as the initial guess — in the pipeline, the previous frame's eigenvector (temporal coherence). On the evolving ring field at 46³ this cuts the certified solve from ~22 outer iterations (cold) to ~15 (warm), about 1.5× in wall time. (The covariant-subdivision seed is iteration-neutral as a start; see §2.)

---

## 5. Results

Smallest eigenpair of the seeded-ring connection Laplacian to a certified 1e-7 relative eigen-residual, our covMG-LOBPCG vs the SLEPc Lanczos baseline; eigenvalues agree to ~1e-12. **Single-threaded — the fair comparison:** the SLEPc baseline runs one MPI rank with no OpenMP, so its wall time is flat across `OMP_NUM_THREADS`, and only a serial-vs-serial table isolates the *algorithm* rather than our implementation's parallelism. Certified median-of-5 protocol (the article's ring table; `tools/eig_compare`):

| n | DOFs | ours: outer its | speed-up vs SLEPc defaults | vs `ncv`-tuned baseline |
|---|---|---|---|---|
| 24 | 27k | 22 | 3.1× | — |
| 32 | 66k | 25 | 4.0× | — |
| 48 | 221k | 28 | 8.0× | 5.5× |
| 64 | 524k | 30 | 12.8× | 8.9× |

The outer count grows only mildly (22 → 30) on this deliberately **near-critical** operator, whose relative gap narrows under refinement; on gapped operators the counts are genuinely flat (uniform-flux torus: 10–12 its to n = 96, speedups 2.3–12.4×; smooth SU(3): 14–16 its flat, 8.0×, reproducible with `tools/sun_gauge_bench 3`; see [`gauge-solver-comparison.md`](gauge-solver-comparison.md) for the torus tables and the article for the SU(3) ones). The advantage over Lanczos widens with size in every family because the flat-cycle V-cycle keeps the per-step cost O(N) while Lanczos's Krylov depth grows.

**Threading, stated honestly.** In the 4-threaded interactive pipeline the wall-clock gap is wider still, because our solver additionally threads (~1.5–1.8× end-to-end; the serial Gram–Schmidt/Rayleigh–Ritz limits it below the kernels' raw scaling) while the one-MPI-rank SLEPc baseline structurally cannot. The single-threaded table above is the honest *algorithmic* advantage.

The win comes from stacking: the gauge-aware transfer (mesh-independence), the red-black GS smoother, precomputed transports (~3× single-thread), 3-term LOBPCG (~½ the outer iterations), and OpenMP. In the live pipeline this turned the per-frame eigensolve from the ~2 s/frame blocker (Lanczos) into an ~88 ms component at 97k cells (4 threads, warm), bringing sim + extraction to ≥10 fps at an every-3rd-frame extraction cadence — and the GPU path (§5a) runs the full obstacle-demo loop at ~47 fps with extraction every frame.

## 5a. The GPU (Metal) eigensolver

The whole eigensolver runs on the GPU as an opt-in single-precision path (`-DBOCHNER_WITH_METAL=ON`, Apple only): the operator apply, the red-black smoother, the covariant restrict/prolong, the V-cycle, and the outer LOBPCG loop are all Metal kernels (`src/gpu/MetalContext.mm`), mirroring the CPU stages one-for-one and each parity-tested against them. It is a **viewer-speed path** — Metal has no `double`, so it converges to a float residual floor (~1e-4); the double CPU / SLEPc solvers above stay authoritative for the accuracy claims. The eigen*vector* (hence the traced filament topology) converges well before that floor, so the extraction is visually identical.

The eigensolver is **fully device-resident**: the iterate, its operator image, the residual, the preconditioned residual, and the ≤3 Gram–Schmidt trial vectors and their operator images all stay in GPU buffers, and every O(n) operation — `E x`, the V-cycle preconditioner, the complex dots, the Gram–Schmidt axpys, the normalizations — is a Metal kernel. The V-cycle's optimal α is reduced on the device too, so the whole cycle is one command buffer. Per outer iteration only a handful of *scalars* cross the bus: the Rayleigh quotient, the two Gram–Schmidt norms, and the 3×3 Ritz matrix are read back in a single sync, the tiny 3×3 Hermitian eig is solved on the CPU, and its three coefficients are uploaded — the O(n) vectors never move. The reported eigenvalue is recomputed once at the end in double from the (single-precision) vectors, so it stays ~1e-6 accurate (the Rayleigh quotient is quadratically accurate) even though the vectors are float.

**Stopping.** In float the eigen-*residual* floors out above a tight `tol` (and is noisy), so keying on it alone would run every solve to `maxIters` — cold *or* warm — hiding the warm-start speed-up. Instead the loop stops when the **Rayleigh quotient ρ stagnates** (relative change < 1e-5 for two consecutive iterations — a tighter 1e-6 gate never triggers on near-degenerate spectra and runs every solve to `maxIters`; see the comment at the gate in `MetalContext.mm`): ρ is monotone and quadratically accurate, so it plateaus exactly when the eigenpair has converged as far as single precision allows. A warm start plateaus in far fewer iterations than a cold one (≈5–9 vs ≈15–49 here), so this restores the warm-start advantage with no hand-tuned iteration cap.

Per-extract wall time on the real flow-past-cylinder connection Laplacian (M3 Pro, best of 5, includes building + uploading the hierarchy each extract; outer iterations in parentheses), **cold** (constant start) and **warm** (previous frame's eigenvector, the interactive per-frame case):

| grid | cells | CPU cold | CPU warm | GPU cold | GPU warm |
|---|---|---|---|---|---|
| 72×24×14 | 24k | 79 (21) | 40 (11) | 14 (15) | **5** (5) |
| 96×32×19 | 58k | 347 (51) | 140 (21) | 33 (30) | **9** (8) |
| 120×40×24 | 115k | 278 (27) | 143 (14) | 29 (16) | **12** (6) |
| 144×48×29 | 200k | 1458 (79) | 532 (29) | 131 (41) | **38** (11) |

The GPU and CPU return the same eigenpair (eigenvalue to ~1e-6, eigenvector complex-line distance ≤ a few ×1e-3). In the **interactive per-frame (warm) case the GPU is ~8–15× faster** than the double CPU — so the obstacle and covector-fluids viewers re-extract the filaments **every frame** on the GPU by default; cold starts (once per reseed) are ~6–11× faster. (The headless profiler `tools/obstacle_profile` reproduces this margin on the current build: ~5.6 ms warm GPU eigensolve at 58k cells inside a ~47 fps full loop.) The strong shed vortices trace identically; the count of the *weakest* near-threshold filaments can differ by ±1–2 (the float eigenvector is line-distance ~1e-3 from the double one, and this borderline count already varies with grid size even for the CPU) — a threshold effect on marginal tubes, not a topology change in the wake.

An earlier hybrid version (CPU-double orchestration, only `E x` and the preconditioner on the GPU) was ~2× slower per frame here: it round-tripped the ≤3 trial vectors host↔device every iteration and did the Gram–Schmidt in CPU double, so the per-frame time was dominated by transfer + orchestration, not compute. Moving to full residency removed both.

---

## 6. Code map

| piece | file | key entry points |
|---|---|---|
| operator + linear V-cycle | `src/solvers/GaugeMultigrid.{h,cpp}` | `GaugeLattice`, `applyConnectionLaplacian`, `vcycleSolve`, `cgSolve` (baseline) |
| eigensolver (single vector) | `src/solvers/GaugeEigen.{h,cpp}` | `smallestEigenpairGaugeMG`, `lowestEigenpairsGaugeMG` (block wrapper) |
| block engine | `src/solvers/BlockLobpcg.{h,cpp}` | `blockLobpcg` |
| SU(d) generalization | `src/solvers/SunGauge.{h,cpp}` | `smallestEigenpairSunMG`, `lowestEigenpairsSunMG` |
| subdivision / transfer | `src/solvers/MacSubdivision.{h,cpp}` | `subdivisionSection` |
| GPU (Metal, float) | `src/gpu/MetalContext.{h,mm}` | `connectionMatvec`/`Smooth`/`Prolong`/`Restrict` (primitives), `uploadGauge` + `vcycleSolveGauge` + `lobpcgSolveGauge` (handle API) |
| benchmarks | `tools/` | `mg_bench` (vs CG), `eig_compare` / `torus_eig_compare` (vs Lanczos), `sun_gauge_bench`, `block_eig_bench`, `warmstart_bench`, `subdivision_demo` |
| tests | `tests/` | `test_gauge_multigrid`, `test_gauge_eigen`, `test_block_lobpcg`, `test_sun_gauge`, `test_mac_subdivision`, `test_metal_sample` (GPU parity) |

The core (`GaugeLattice` + the V-cycle + the eigensolver) is substrate-independent — it depends only on lattice dimensions, link angles, and an edge weight, not on the MAC grid, PETSc, or the fluid solver — so it can be lifted into a stand-alone solver library.
