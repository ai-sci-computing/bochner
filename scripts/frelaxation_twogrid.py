#!/usr/bin/env python3
"""F-relaxation polish experiment (paper, harmonic-extension remark).

Two-grid asymptotic factors for the alpha-stabilized cycle when the
covariant fill P is polished toward the exact harmonic extension by k
F-Jacobi sweeps (odd nodes only, even values fixed; k = infinity = ideal
interpolation). Tested verdict (2D, n=16): unchanged at aliasing, marginal
at worst flux / hot, WORSE in the physical regime -- with a rediscretized
coarse operator the matched pair (fill, L_c) beats the ideal extension.
Builders are shared with verify_lower_bound.py.
"""
import numpy as np
import os, sys

here = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(here, "verify_lower_bound.py")).read().split("def check(")[0])


def fjacobi_polish(P, L, even, odd, k):
    LOO = L[np.ix_(odd, odd)]; LOE = L[np.ix_(odd, even)]
    D = np.diag(LOO).real
    Pk = P.copy()
    for _ in range(k):
        R = LOO @ Pk[odd, :] + LOE @ Pk[even, :]
        Pk[odd, :] -= (R.T / D).T
    return Pk


def ideal_P(P, L, even, odd):
    Pi = P.copy()
    Pi[odd, :] = -np.linalg.solve(L[np.ix_(odd, odd)], L[np.ix_(odd, even)] @ P[even, :])
    return Pi


def twogrid_rho(L, P, Lc, sweeps=2, ncyc=60, seed=0):
    n2 = L.shape[0]
    r = np.random.default_rng(seed)
    b = r.normal(size=n2) + 1j * r.normal(size=n2)
    x = np.zeros(n2, complex)
    n = int(np.sqrt(n2))
    red = [i * n + j for i in range(n) for j in range(n) if (i + j) % 2 == 0]
    blk = [i * n + j for i in range(n) for j in range(n) if (i + j) % 2 == 1]
    D = np.diag(L).real

    def gs(x):
        for grp in (red, blk):
            x[grp] += (b[grp] - (L[grp, :] @ x)) / D[grp]
        return x

    res = []
    for c in range(ncyc):
        for _ in range(sweeps):
            x = gs(x)
        rr = b - L @ x
        p = P @ np.linalg.solve(Lc, P.conj().T @ rr)
        den = np.real(np.vdot(p, L @ p))
        if den > 0:
            x += (np.real(np.vdot(p, rr)) / den) * p
        for _ in range(sweeps):
            x = gs(x)
        res.append(np.linalg.norm(b - L @ x))
        if res[-1] < 1e-13 * res[0]:
            break
    res = np.array(res)
    return float(np.exp(np.mean(np.log(res[-5:] / res[-6:-1]))))


if __name__ == "__main__":
    n = 16
    one = np.ones((n, n))
    cases = []
    for nphi, tag in ((4, "physical flux nphi=4"), (32, "WORST flux phi=2pi/8"), (60, "near aliasing")):
        phi = 2 * np.pi * nphi / n**2
        thx = np.zeros((n, n)); thy = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                thx[i, j] = -phi * j
        for i in range(n):
            thy[i, n - 1] = phi * n * i
        cases.append((tag, thx, thy, one, one))
    r = np.random.default_rng(5)
    cases.append(("hot + random weights", r.uniform(0, 2 * np.pi, (n, n)),
                  r.uniform(0, 2 * np.pi, (n, n)), r.uniform(0.2, 3, (n, n)),
                  r.uniform(0.2, 3, (n, n))))

    print(f"{'case':24s} {'k=0 (ours)':>11s} {'k=1':>7s} {'k=2':>7s} {'ideal':>7s}")
    for tag, thx, thy, wx, wy in cases:
        N = n * n
        even = [(i % n) * n + (j % n) for i in range(0, n, 2) for j in range(0, n, 2)]
        odd = [k for k in range(N) if k not in set(even)]
        L = build(n, thx, thy, wx, wy)
        P0 = build_P(n, thx, thy, wx, wy)
        Lc = coarse(n, thx, thy, wx, wy)
        out = [twogrid_rho(L, Pv, Lc) for Pv in
               (P0, fjacobi_polish(P0, L, even, odd, 1),
                fjacobi_polish(P0, L, even, odd, 2), ideal_P(P0, L, even, odd))]
        print(f"{tag:24s} {out[0]:11.3f} {out[1]:7.3f} {out[2]:7.3f} {out[3]:7.3f}")
