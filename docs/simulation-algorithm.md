# The simulation + extraction algorithm

This document describes the full per-frame pipeline: an incompressible 3D fluid simulation (Covector Fluids) whose vortex structure is extracted, every frame, as discrete vortex filaments (Weissmann–Pinkall). The research kernel inside the extraction step — the smallest-eigenvector solve of the connection Laplacian — is documented separately in [`gauge-multigrid.md`](gauge-multigrid.md).

Sources: Nabizadeh et al., *Covector Fluids* (SIGGRAPH 2022) for the fluid scheme; Weissmann & Pinkall, *Smoke Rings from Smoke* (2014) for the extraction.

---

## 1. The grid and where quantities live

The substrate is a uniform **MAC (marker-and-cell) grid** of `nx · ny · nz` cubic cells of side `h`, treated as a cubical cell complex with **dual/flux placement**:

| quantity | lives on | k-form |
|---|---|---|
| velocity `u` | **face normals** (the component normal to each face) | dual 1-form / primal 2-form (flux) |
| pressure `p`, section `ψ` | **cell centers** | dual 0-form |

Faces split into three axis families (`x`, `y`, `z`); the family normal to an axis has one extra layer along that axis (the two domain boundaries). Storing the velocity *component* (not the area-integrated flux) keeps the `h` factors explicit and makes a face value directly equal to the Weissmann–Pinkall gauge link later. Code: `src/grid/MacGrid.*`, `FaceField` in `src/grid/GridOperators.h`.

The two structured operators, realized as matrix-free stencils:

- **divergence** (`d`, faces → cells): `div[c] = Σ (outward normal velocity)/h` over the cell's six faces — the finite-volume net flux per unit volume.
- **gradient** (cells → faces): `grad[f] = (p[c⁺] − p[c⁻])/h` on **interior** faces only; boundary faces carry **no** gradient (they are BC-prescribed). With this convention `grad = −divᵀ` on the interior, and `div(grad ·)` is the homogeneous-Neumann 7-point Laplacian.

The domain is **closed** (no-penetration walls): boundary face velocities stay zero, so a prescribed normal velocity would enter `div` natively — no source-term hack is needed.

---

## 2. One time step (Covector Fluids)

Velocity is a **covector** (1-form), not a vector field — this is what distinguishes Covector Fluids and what preserves vortex dynamics. A first-order step (CF Algorithm 1; `stepCovectorFluids`) is:

1. freeze the flow velocity `v ← u`;
2. **advect** the covector: `u ← A_covec(u; v, Δt)` (BFECC, §2.1);
3. **project** to divergence-free: `u ← P(u)` (§2.2).

A second-order midpoint variant (Alg 2) estimates `v` at `Δt/2` first. The obstacle viewer uses this first-order step with BFECC advection; the covector- fluids (leapfrog) viewer defaults to the **CF+MCM long-time flow map** (Alg 5, `CfMcmSolver` in `src/fluid/MacCharacteristicMap.*`) — a persistent bidirectional characteristic map reinitialized only when it distorts (the BiMocq mechanism), which advects through the accumulated map instead of re-interpolating the grid every step.

### 2.1 Covector advection — the `(dΨ)ᵀ` pullback with BFECC

Advecting a 1-form is the **pullback by the inverse flow map** `Ψ = Φ⁻¹`: `u(x) ← (dΨ(x))ᵀ u(Ψ(x))` (CF Eq. 3). On the staggered grid this is realized exactly per CF §5; code: `src/fluid/MacAdvection.*`.

**Semi-Lagrangian core (`advectCovectorSL`, CF Eq. 39/40):**

- Backtrace every **cell center** once: integrate `dx/dt = v` *backward* over `Δt` by **RK4**, giving `Ψ(cellCenter)`. (`backtrace`, `sampleVelocity` with trilinear interpolation of the staggered components; queries are clamped to the domain — CF §5.4.5's "closest interior point".)
- For each interior face, the transpose-Jacobian column `∂Ψ/∂n` is the **finite difference of the backtraced cell-center map across that face** (Eq. 40), divided by `h`.
- The new face value is that column **dotted with the velocity sampled at the backtraced face center** `Ψ(faceCenter)` (Eq. 39).

**BFECC (`advectCovectorBFECC`, CF Alg. 4):** plain semi-Lagrangian advection is diffusive and collapses the vortex ring over ~10–14 frames (a key lesson from the tet prototype: exact circulation preservation did *not* fix it — the vorticity *distribution* smears). BFECC removes the first-order error:

```
u1 = SL(u, v, +Δt)            # forward
u0b = SL(u1, v, −Δt)            # back
eh = (u0b − u) / 2              # half the round-trip error
u* = u1 − SL(eh, v, +Δt)        # corrected
u   = clampToStencil(u*, u1)     # §5.4.2 extrema limiter
```

The minmax limiter clamps each corrected face value to the min/max of `u1` over the 27-cell neighbourhood, which suppresses the overshoots BFECC can introduce. BFECC keeps >85% of a Taylor–Green vortex's kinetic energy where plain SL visibly bleeds it (CF Fig. 8).

All advection loops are independent per cell/face and are **OpenMP-parallel**.

### 2.2 Pressure projection — divergence-free part

Projection extracts the divergence-free representative (discrete Helmholtz–Hodge): solve a scalar Poisson for the pressure, then subtract its gradient. Code: `MacProjector` in `src/fluid/MacProjection.*`.

```
rhs = −div(u), projected to mean-zero       # consistent Neumann data
L φ = rhs                                     # homogeneous-Neumann Poisson
u ← u − grad(φ)
```

`L = −div·grad` is the real-scalar 7-point Neumann Laplacian. Two points make this fast (and are *not* a job for the gauge multigrid — that would double the dimension for a real field):

- It is solved by a **real-scalar geometric multigrid** (`poissonVcycleSolve`, `src/solvers/PoissonMultigrid.*`): matrix-free, red-black Gauss–Seidel smoother, OpenMP, mesh-independent (~16 V-cycles), and with **no per-frame setup** (unlike an algebraic-AMG hierarchy).
- The operator is constant across frames and the system is left **unpinned** (the projection uses only `grad(φ)`, which annihilates the constant null space), so `MacProjector` **warm-starts** `φ` from the previous frame — the pressure field changes slowly, so this converges in a couple of cycles.

The result is divergence-free in every interior cell; boundary faces are untouched.

### 2.3 Flow past an obstacle (inflow/outflow + solid + viscosity)

The obstacle viewer runs the same step in an open channel with an interior solid, adding three pieces (code: `src/grid/MacObstacle.*`, `src/fluid/*`):

- **Inflow/outflow BCs.** A `BoundarySpec` marks each wall closed, an open outlet (Dirichlet `p = 0` ghost), or a prescribed-velocity inlet. Open runs assemble a BC-aware Poisson solved by an assembled Galerkin-aggregation **MGPCG** (`PoissonMGPCG`, driven from `MacProjection.cpp`), with an opt-in Metal GPU path that falls back to the CPU solver on a GPU fault.
- **Interior solid.** A per-cell `SolidMask` (`cylinderMask`/`sphereMask`/ `boxMask`) marks obstacle cells. A face touching a solid cell is a no-penetration interior wall (dropped from the Laplacian, zeroed in the velocity); solid cells carry no pressure DOF.
- **Viscosity.** An explicit forward-Euler vector-Laplacian diffusion substep (`ops::diffuseVelocity`, no-slip at the solid, free-slip at walls) is the vorticity source/sink that turns the free-slip potential flow into a shedding von Kármán wake. The per-frame step is advect → diffuse → project.

**Physical validation: the sphere-wake regime sequence.** The demo reproduces the
textbook transitions of a sphere wake, which is a stronger check than any single
timing. Sweeping `nu` at fixed `U = 1`, `D = 0.7` (`BOCHNER_NU=... obstacle_profile`),
using the warm-started eigensolver's iteration count as an unsteadiness probe — a
*steady* wake re-converges in ~1 iteration because the previous eigenvector still
fits, an unsteady one keeps needing 10+:

| Re | eig its at late time | regime |
|---|---|---|
| 87  | 6 → 1 (settles)        | steady, axisymmetric recirculating torus |
| 250 | 4 → 3 (still settling) | steady |
| 318 | 13 → 13 → 13 (flat)    | **sustained unsteadiness** |
| 438 | 15–18, filament count varies | unsteady loop shedding |

So the onset of unsteadiness is bracketed at **250 < Re < 318**, against a
literature value of **Re ≈ 270** for sphere vortex shedding — on a body only ~11
cells across. Visually the progression is the correct one: a stationary
recirculation ring, then a wobble as axisymmetry is lost, then loop vortices
transported downstream — less regular than a cylinder's von Kármán street, which
is how spheres actually shed (hairpin/loop vortices, not a 2D alternating street).

Caveats, so this is not over-read: the probe is an eigensolver iteration count
rather than a direct wake measurement; the transition is bracketed by two points,
not resolved; and the channel is confined (D/Ly = 35%, D/Lz = 58%), which shifts
transitions relative to an unbounded sphere. The agreement is "right regime, right
ballpark", not a quantitative benchmark.

The shed wake is a live, physically-generated vorticity field for the same extraction pipeline (§3).

---

## 3. Vortex-filament extraction (Weissmann–Pinkall)

The filaments are recovered as the **zero set of the smallest-eigenvalue eigenvector** of a U(1) connection Laplacian built from the velocity. Code: `src/extraction/MacConnectionLaplacian.*`, `src/extraction/MacFilaments.*`.

### 3.1 The connection

The velocity 1-form becomes a U(1) connection on the **dual graph** (cell centers linked through faces). The connection angle on face `f`:

```
θ_f = u_f · h / ħ ,     ħ = h_strength / 2π
```

`u_f` is the stored face-normal velocity; `h` is the dual-edge length; `ħ` sets the **flux quantum**. With `ħ = Γ/2π`, a vortex of circulation `Γ` carries exactly **one flux quantum**, hence **one filament**. (`connectionAngles`.)

The complex section `ψ` lives at cell centers; each interior face carries the parallel transport `e^{iθ_f}`. The covariant difference across a face (low cell `a`, high cell `b`) is `(d^∇ψ)_f = ψ_b − e^{iθ_f} ψ_a`, and the **connection Laplacian** is the Hermitian operator `E = (d^∇)† d^∇` with edge weight `w = 1/h²`. Curvature (holonomy around a cell loop) is the vorticity flux; where that holonomy is ~2π the lowest mode develops a phase singularity — a filament.

`E` is stored as a **real symmetric `2N × 2N`** matrix using the embedding `a + ib → [[a, −b], [b, a]]`; cell `j` occupies real rows/cols `{2j, 2j+1}`. A trivial connection (`θ ≡ 0`) reduces to two decoupled copies of the Neumann Laplacian. (`connectionLaplacian`.)

### 3.2 The smallest eigenvector

`ψ = argmin Rayleigh quotient of E` is the vortex field. This is the per-frame bottleneck and the project's research contribution; it is solved by the **gauge-aware multigrid Rayleigh-quotient eigensolver** — **covMG-LOBPCG** (code: `smallestEigenpairGaugeMG`) — warm-started from the previous frame's eigenvector. See [`gauge-multigrid.md`](gauge-multigrid.md) for the full method.

### 3.3 Trace and link

`ψ` is supplied interleaved as `[Re ψ₀, Im ψ₀, …]`. The zero set is extracted in two stages:

- **`traceZeroSet`** — around each elementary plaquette (unit square of four neighbouring cell centers) the phase of `ψ` either winds by `±2π` (a filament pierces the square, vorticity flux `±1`) or not. When it winds, the enclosed zero is located by a few bilinear-Newton steps, and the winding sign orients the crossing.
- **`linkFilaments`** — each elementary cube joins the crossings on its two pierced plaquettes; chaining the per-cube segments yields the filaments. Components with no boundary endpoint come out as **closed loops** (a vortex ring); components reaching the lattice boundary come out **open**.

A seeded ring round-trips: ring velocity → connection → eigenvector → trace/link recovers a single closed planar loop coaxial with the ring (verified in `tests/test_mac_extraction.cpp` to geometric-fidelity tolerances — planar to within 2 cells, radial deviation < 1.5 cells, mean radius within 25% and circumference within 10% of the seeded ring — not machine precision; the loop is a discrete trace through a smooth eigenvector's zero set, not an exact-arithmetic object).

---

## 4. Putting it together: the live frame

Per frame the viewer (`tools/viewer`) and profiler (`tools/pipeline_profile`) do:

```
advect (BFECC)                          # every frame
project (geometric MG, warm-started)    # every frame   -> "sim"
─────────────────────────────────────────────────────────────────
connection angles θ = u·h/ħ            # every k frames
smallest eigenvector ψ (covMG-LOBPCG)  # every k frames -> "extract"
trace + link → filament loops           # every k frames
```

The sim (advect + project) advances the fluid every frame; the **extraction is decoupled** to run every `k` frames, since the filament refreshes can lag the fluid without being noticed. The amortized per-frame cost is therefore `sim + extract/k`.

Measured on the seeded ring at 46³ ≈ 97k cells (4 threads, `tools/pipeline_profile`): sim floor ≈ 59 ms (≈17 fps); warm eigensolve ≈ 88 ms/frame; combined sim + extraction reaches ≥ 10 fps at an every-3rd-frame extraction cadence (every 2nd: ≈ 9 fps) — meeting the project's ≥10 fps interactivity criterion at the low end of the 100k–500k-cell target range at the coarser cadence. (The GPU obstacle demo runs its full loop, extraction every frame, at ~47 fps — see `gauge-multigrid.md` §5a and the companion article's demo section.)

**Thread-count sensitivity is grid-size-dependent, and it matters at the top of the range.** At n = 48 (110k) the working set is small enough to be *memory-bandwidth-bound*, so going past 4 threads barely helps (sim floor 52.2 ms at 4 threads, 50.9 at 8) — hence "4 threads" as the reported config. At n = 64 (262k) the larger grid stays *compute-bound* and keeps scaling with cores: the sim floor is **8.4 fps at 4 threads, 9.6 at 8, and ~10.1 fps only using all 11 cores.** So the ≥10 fps sim at 262k is reachable but needs the whole machine, and the full sim + extraction there is still ~5 fps — closing the upper half of the target range is the remaining work. (By contrast the eigensolver-vs-Lanczos comparisons in `docs/gauge-multigrid.md` are reported single-threaded, because the SLEPc baseline has no OpenMP and only a serial-vs-serial table is a fair one.)

### Pipeline cost breakdown (seeded ring, 4 threads, per extracting frame)

| stage | ms | notes |
|---|---|---|
| advect (BFECC) | ~38 | compute-bound, scales to 8 threads |
| project (geom-MG, warm) | ~15 | matrix-free, no setup |
| connection build | ~1 | matrix-free |
| eigensolve (covMG-LOBPCG, warm) | ~88 | the research kernel |
| trace + link | ~12 | |

(The advect/project/trace rows are the earlier profile's stage split; the 2026-07-17 re-measurement recorded only the totals — sim floor 59 ms, warm eigensolve ~88 ms — so the split rows are indicative, the totals authoritative.)

The eigensolve — which was ~2 s/frame with Lanczos at this scale before the gauge multigrid — is now the largest single stage but no longer the blocker.

---

## Code map

| step | files |
|---|---|
| grid, operators | `MacGrid.*`, `GridOperators.*` |
| advection (BFECC) | `MacAdvection.*` |
| pressure projection | `MacProjection.*`, `PoissonMultigrid.*` |
| connection Laplacian | `MacConnectionLaplacian.*` |
| eigensolver (research kernel) | `GaugeMultigrid.*`, `GaugeEigen.*`, `MacSubdivision.*` |
| trace / link | `MacFilaments.*` |
| seeding | `MacVortexRing.*` |
| live viewer | `tools/viewer/` |
