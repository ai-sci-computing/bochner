#!/usr/bin/env python3
"""PyAMG-preconditioned LOBPCG baseline for the connection Laplacian.

Loads a complex Hermitian operator exported by tools/export_operator (.mtx,
MatrixMarket coordinate complex general) and solves for the smallest eigenpair
with scipy.sparse.linalg.lobpcg (one vector, largest=False), preconditioned by

  1. pyamg.aggregation.adaptive_sa_solver  (adaptive smoothed aggregation),
  2. pyamg.smoothed_aggregation_solver     (classical smoothed aggregation),
  3. nothing                               (unpreconditioned reference).

If a pyamg solver cannot be built on the complex matrix, that method falls
back to the real 2N x 2N embedding a+ib -> [[a,-b],[b,a]] (built here from the
complex matrix; the spectrum is unchanged, every eigenvalue doubles its
multiplicity) and the report says so.

Tolerance: "1e-7 equivalent" means the SLEPc-style RELATIVE criterion
||A x - lambda x|| <= 1e-7 * lambda, implemented by handing lobpcg the
absolute tolerance 1e-7 * lambda_est with lambda_est from a short
SA-preconditioned pre-solve (reported). Iteration counts come from
retResidualNormsHistory. Per method: setup wall s, solve wall s, iterations,
eigenvalue.

Usage: pyamg_baseline.py <matrix.mtx>
"""

import sys
import time
import warnings

import numpy as np
import scipy.io
import scipy.sparse as sp
from scipy.sparse.linalg import lobpcg

MAXITER = 500
RELTOL = 1e-7


def real_embedding(a):
    """The real 2N x 2N embedding [[Re, -Im], [Im, Re]] of a complex matrix."""
    re, im = a.real.tocsr(), a.imag.tocsr()
    return sp.bmat([[re, -im], [im, re]], format="csr")


def start_vector(n, complex_valued, seed=7):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, 1))
    if complex_valued:
        x = x + 1j * rng.standard_normal((n, 1))
    return x


def run_lobpcg(a, m, tol, seed=7, maxiter=MAXITER):
    """One smallest-eigenpair LOBPCG solve; returns (lambda, iters, seconds, resid)."""
    x = start_vector(a.shape[0], np.iscomplexobj(a), seed)
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # lobpcg warns about single-vector blocks
        vals, vecs, hist = lobpcg(a, x, M=m, largest=False, tol=tol,
                                  maxiter=maxiter, retResidualNormsHistory=True)
    dt = time.perf_counter() - t0
    lam = float(vals[0])
    v = vecs[:, 0]
    r = a @ v - lam * v
    resid = float(np.linalg.norm(r) / np.linalg.norm(v))
    return lam, len(hist), dt, resid


def build_preconditioner(name, a):
    """Build the requested pyamg hierarchy on `a` (complex first, real embedding
    fallback). Returns (M, operator, setup_s, used_embedding, error_note)."""
    import pyamg

    def attempt(mat):
        t0 = time.perf_counter()
        if name == "adaptive-SA":
            ml, _work = pyamg.aggregation.adaptive_sa_solver(mat)
        elif name == "SA":
            ml = pyamg.smoothed_aggregation_solver(mat)
        else:
            raise ValueError(name)
        m = ml.aspreconditioner()
        # One trial apply: pyamg can build a hierarchy whose *cycle* then fails.
        m @ np.ones(mat.shape[0], dtype=mat.dtype)
        return m, time.perf_counter() - t0

    try:
        m, dt = attempt(a)
        return m, a, dt, False, None
    except Exception as exc:  # noqa: BLE001 -- report and fall back, per task
        note = f"{type(exc).__name__}: {exc}"
        m, dt = attempt(real_embedding(a))
        return m, real_embedding(a), dt, True, note


def main(path):
    import pyamg

    a = scipy.io.mmread(path).tocsr()
    n = a.shape[0]
    herm = abs(a - a.conjugate().T).max()
    print(f"matrix: {path}")
    print(f"  dim {n}, nnz {a.nnz}, dtype {a.dtype}, hermitian defect {herm:.2e}")
    print(f"  pyamg {pyamg.__version__}, scipy {scipy.__version__}, numpy {np.__version__}")

    # lambda estimate for the relative tolerance (short SA-preconditioned solve).
    t0 = time.perf_counter()
    ml_est = pyamg.smoothed_aggregation_solver(a)
    lam_est, its_est, dt_est, _ = run_lobpcg(a, ml_est.aspreconditioner(),
                                             tol=1e-3, maxiter=100)
    print(f"  lambda_est = {lam_est:.6f}  (SA-LOBPCG tol 1e-3: {its_est} its, "
          f"{time.perf_counter() - t0:.2f} s incl. setup)")
    tol = RELTOL * abs(lam_est)
    print(f"  lobpcg tol = {tol:.3e}  (= {RELTOL:.0e} * lambda_est), maxiter {MAXITER}\n")

    rows = []
    for name in ("adaptive-SA", "SA"):
        try:
            m, op, setup_s, embedded, note = build_preconditioner(name, a)
        except Exception as exc:  # noqa: BLE001 -- even the embedding failed
            rows.append((name, None, f"setup failed on complex AND embedding: {exc}"))
            continue
        lam, its, solve_s, resid = run_lobpcg(op, m, tol)
        label = name + (" [real 2Nx2N embedding]" if embedded else " [complex]")
        rows.append((label, (setup_s, solve_s, its, lam, resid), note))
    lam, its, solve_s, resid = run_lobpcg(a, None, tol)
    rows.append(("none (unpreconditioned) [complex]", (0.0, solve_s, its, lam, resid), None))

    hdr = f"{'preconditioner':<36} | {'setup s':>8} | {'solve s':>8} | {'iters':>5} | {'eigenvalue':>12} | {'rel resid':>9}"
    print(hdr)
    print("-" * len(hdr))
    for label, data, note in rows:
        if data is None:
            print(f"{label:<36} | FAILED: {note}")
            continue
        setup_s, solve_s, its, lam, resid = data
        print(f"{label:<36} | {setup_s:8.3f} | {solve_s:8.3f} | {its:5d} | {lam:12.6f} | {resid:9.2e}")
        if note:
            print(f"{'':<36} | complex build failed -> embedding: {note}")
    conv = [d[2] for _, d, _ in rows if d is not None]
    if any(c >= MAXITER for c in conv):
        print(f"\nnote: iters == {MAXITER} means the maxiter cap, not convergence "
              "(see rel resid).")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(f"usage: {sys.argv[0]} <matrix.mtx>")
    main(sys.argv[1])
