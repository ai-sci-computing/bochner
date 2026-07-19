"""Numerical verification of Theorem [universal lower bound] (paper sec. 5)
and its harmonic-extension proof chain:
    P^H L P  >=  S (Schur/ideal-interpolation energy)  >=  2 L_c
on 2D U(1) lattices with RANDOM angles (hot) and RANDOM per-edge weights,
plus the uniform-flux case. P = the paper's weighted multi-pass covariant
fill; L_c = composed links + series-conductance weights W = wawb/(2(wa+wb)).
Also checks the defect identity P^H L P = S + D^H L_OO D.
"""
import numpy as np

rng = np.random.default_rng(11)


def build(n, thx, thy, wx, wy):
    N = n * n
    idx = lambda i, j: (i % n) * n + (j % n)
    L = np.zeros((N, N), complex)
    for i in range(n):
        for j in range(n):
            a = idx(i, j)
            for (b, th, w) in ((idx(i + 1, j), thx[i, j], wx[i, j]),
                               (idx(i, j + 1), thy[i, j], wy[i, j])):
                t = np.exp(1j * th)
                L[a, a] += w; L[b, b] += w
                L[b, a] += -w * t
                L[a, b] += -w * np.conj(t)
    return L


def build_P(n, thx, thy, wx, wy):
    nc = n // 2
    N, Nc = n * n, nc * nc
    idx = lambda i, j: (i % n) * n + (j % n)
    cidx = lambda I, J: (I % nc) * nc + (J % nc)
    P = np.zeros((N, Nc), complex)
    for I in range(nc):
        for J in range(nc):
            P[idx(2 * I, 2 * J), cidx(I, J)] = 1.0
    # pass 1: one odd coordinate -- weighted covariant average of the 2 axis nbrs
    for i in range(n):
        for j in range(n):
            if (i % 2) + (j % 2) != 1:
                continue
            a = idx(i, j)
            if i % 2 == 1:
                lo, hi = idx(i - 1, j), idx(i + 1, j)
                wlo, whi = wx[(i - 1) % n, j], wx[i, j]
                tlo, thi = np.exp(1j * thx[(i - 1) % n, j]), np.exp(-1j * thx[i, j])
            else:
                lo, hi = idx(i, j - 1), idx(i, j + 1)
                wlo, whi = wy[i, (j - 1) % n], wy[i, j]
                tlo, thi = np.exp(1j * thy[i, (j - 1) % n]), np.exp(-1j * thy[i, j])
            P[a, :] = (wlo * tlo * P[lo, :] + whi * thi * P[hi, :]) / (wlo + whi)
    # pass 2: both odd -- weighted covariant average of all 4 nbrs
    for i in range(n):
        for j in range(n):
            if (i % 2) == 1 and (j % 2) == 1:
                a = idx(i, j)
                nbrs = [(idx(i - 1, j), wx[(i - 1) % n, j], np.exp(1j * thx[(i - 1) % n, j])),
                        (idx(i + 1, j), wx[i, j], np.exp(-1j * thx[i, j])),
                        (idx(i, j - 1), wy[i, (j - 1) % n], np.exp(1j * thy[i, (j - 1) % n])),
                        (idx(i, j + 1), wy[i, j], np.exp(-1j * thy[i, j]))]
                sw = sum(w for _, w, _ in nbrs)
                P[a, :] = sum(w * t * P[b, :] for b, w, t in nbrs) / sw
    return P


def coarse(n, thx, thy, wx, wy):
    nc = n // 2
    cthx = np.zeros((nc, nc)); cthy = np.zeros((nc, nc))
    cwx = np.zeros((nc, nc)); cwy = np.zeros((nc, nc))
    for I in range(nc):
        for J in range(nc):
            a, b = wx[2 * I, 2 * J], wx[(2 * I + 1) % n, 2 * J]
            cthx[I, J] = thx[2 * I, 2 * J] + thx[(2 * I + 1) % n, 2 * J]
            cwx[I, J] = a * b / (2 * (a + b))
            a, b = wy[2 * I, 2 * J], wy[2 * I, (2 * J + 1) % n]
            cthy[I, J] = thy[2 * I, 2 * J] + thy[2 * I, (2 * J + 1) % n]
            cwy[I, J] = a * b / (2 * (a + b))
    return build(nc, cthx, cthy, cwx, cwy)


def check(label, n, thx, thy, wx, wy):
    N = n * n
    even = [(i % n) * n + (j % n) for i in range(0, n, 2) for j in range(0, n, 2)]
    odd = [k for k in range(N) if k not in set(even)]
    L = build(n, thx, thy, wx, wy)
    P = build_P(n, thx, thy, wx, wy)
    Lc = coarse(n, thx, thy, wx, wy)
    A = P.conj().T @ L @ P

    LOO = L[np.ix_(odd, odd)]
    LEO = L[np.ix_(even, odd)]
    LEE = L[np.ix_(even, even)]
    S = LEE - LEO @ np.linalg.solve(LOO, LEO.conj().T)

    # reorder coarse dofs to match `even` ordering (they already do: row-major)
    e1 = np.linalg.eigvalsh((A - S + (A - S).conj().T) / 2).min()
    e2 = np.linalg.eigvalsh((S - 2 * Lc + (S - 2 * Lc).conj().T) / 2).min()
    # defect identity
    Pideal = np.zeros_like(P)
    Pideal[even, :] = np.eye(len(even))
    Pideal[odd, :] = -np.linalg.solve(LOO, LEO.conj().T @ np.eye(len(even)))
    D = (P - Pideal)[odd, :]
    ident = np.abs(A - (S + D.conj().T @ LOO @ D)).max()
    # pencil lower bound
    lam, V = np.linalg.eigh(Lc)
    keep = lam > 1e-10 * lam.max()
    Wb = V[:, keep] / np.sqrt(lam[keep])
    pmin = np.linalg.eigvalsh(Wb.conj().T @ A @ Wb).min()
    print(f"{label:34s} min eig(PHLP-S)={e1:9.2e}  min eig(S-2Lc)={e2:9.2e}  "
          f"|defect identity|={ident:.1e}  min pencil={pmin:.4f}")


n = 8
for seed in (1, 2, 3):
    r = np.random.default_rng(seed)
    thx = r.uniform(0, 2 * np.pi, (n, n)); thy = r.uniform(0, 2 * np.pi, (n, n))
    wx = r.uniform(0.2, 3.0, (n, n)); wy = r.uniform(0.2, 3.0, (n, n))
    check(f"hot links + random weights s={seed}", n, thx, thy, wx, wy)

# uniform flux, uniform weights (Landau gauge), incl. near-aliasing
for nphi in (1, 8, 15):
    phi = 2 * np.pi * nphi / n**2
    thx = np.zeros((n, n)); thy = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            thx[i, j] = -phi * j
    for i in range(n):
        thy[i, n - 1] = phi * n * i
    w1 = np.ones((n, n))
    check(f"uniform flux nphi={nphi}", n, thx, thy, w1, w1)

# arithmetic-mean coarse weight must VIOLATE S >= 2 Lc' (sanity of sharpness)
r = np.random.default_rng(7)
thx = r.uniform(0, 2 * np.pi, (n, n)); thy = r.uniform(0, 2 * np.pi, (n, n))
wx = r.uniform(0.2, 3.0, (n, n)); wy = r.uniform(0.2, 3.0, (n, n))
nc = n // 2
cthx = np.zeros((nc, nc)); cthy = np.zeros((nc, nc))
cwx = np.zeros((nc, nc)); cwy = np.zeros((nc, nc))
for I in range(nc):
    for J in range(nc):
        a, b = wx[2 * I, 2 * J], wx[(2 * I + 1) % n, 2 * J]
        cthx[I, J] = thx[2 * I, 2 * J] + thx[(2 * I + 1) % n, 2 * J]
        cwx[I, J] = (a + b) / 4.0
        a, b = wy[2 * I, 2 * J], wy[2 * I, (2 * J + 1) % n]
        cthy[I, J] = thy[2 * I, 2 * J] + thy[2 * I, (2 * J + 1) % n]
        cwy[I, J] = (a + b) / 4.0
LcAM = build(nc, cthx, cthy, cwx, cwy)
L = build(n, thx, thy, wx, wy)
P = build_P(n, thx, thy, wx, wy)
A = P.conj().T @ L @ P
lam, V = np.linalg.eigh(LcAM)
keep = lam > 1e-10 * lam.max()
Wb = V[:, keep] / np.sqrt(lam[keep])
pmin = np.linalg.eigvalsh(Wb.conj().T @ A @ Wb).min()
print(f"{'AM rule (wa+wb)/4 (doubled)':34s} min pencil={pmin:.4f}  (expected < 2)")
# normalized AM rule (wa+wb)/8 (same uniform limit w/4 as the series rule):
# the coarse operator is linear in its weights, so the pencil doubles exactly.
LcAM8 = build(nc, cthx, cthy, cwx / 2.0, cwy / 2.0)
lam, V = np.linalg.eigh(LcAM8)
keep = lam > 1e-10 * lam.max()
Wb = V[:, keep] / np.sqrt(lam[keep])
pmin8 = np.linalg.eigvalsh(Wb.conj().T @ A @ Wb).min()
print(f"{'AM rule (wa+wb)/8 (normalized)':34s} min pencil={pmin8:.4f}  (expected < 2; = 2x the /4 value)")

# approach to aliasing (paper Example ex:alias): on the n=16 uniform-flux
# torus (uniform weights), the LARGEST pencil eigenvalue of (P^H L P, Lc)
# grows from 4.0 = 2^d at zero flux through ~7.8 at phi = 2pi/8 and blows
# up as 4*phi -> 2pi (Lc singular at the aliasing point nphi = n^2/4).
n2 = 16
print()
for nphi in (0, 32, 56, 62, 63):
    phi = 2 * np.pi * nphi / n2**2
    thx = np.zeros((n2, n2)); thy = np.zeros((n2, n2))
    for i in range(n2):
        for j in range(n2):
            thx[i, j] = -phi * j
    for i in range(n2):
        thy[i, n2 - 1] = phi * n2 * i
    w1 = np.ones((n2, n2))
    L = build(n2, thx, thy, w1, w1)
    P = build_P(n2, thx, thy, w1, w1)
    Lc = coarse(n2, thx, thy, w1, w1)
    A = P.conj().T @ L @ P
    lam, V = np.linalg.eigh(Lc)
    keep = lam > 1e-10 * lam.max()
    Wb = V[:, keep] / np.sqrt(lam[keep])
    ev = np.linalg.eigvalsh(Wb.conj().T @ A @ Wb)
    print(f"aliasing approach n=16 nphi={nphi:3d} (phi={phi/np.pi:.3f}pi)  "
          f"max pencil={ev.max():9.3f}  min pencil={ev.min():.4f}")

# harmonic-extension headroom (see the companion article's section on the
# harmonic extension): pencil [min,max] of
# (P^H L P, Lc) at n=16, our one-sweep fill vs the EXACT harmonic (ideal)
# extension [I; -Loo^-1 Loe]. Headroom is negligible where the connection
# is coherent and real only in smoother-owned regimes.
def pencil_range(L, P, Lc):
    A = P.conj().T @ L @ P
    lam, V = np.linalg.eigh(Lc)
    keep = lam > 1e-10 * lam.max()
    Wb = V[:, keep] / np.sqrt(lam[keep])
    ev = np.linalg.eigvalsh(Wb.conj().T @ A @ Wb)
    return ev.min(), ev.max()

def ideal_of(n, L):
    N = n * n
    even = [(i % n) * n + (j % n) for i in range(0, n, 2) for j in range(0, n, 2)]
    odd = [k for k in range(N) if k not in set(even)]
    Pid = np.zeros((N, len(even)), complex)
    Pid[even, :] = np.eye(len(even))
    LOO = L[np.ix_(odd, odd)]; LEO = L[np.ix_(even, odd)]
    Pid[odd, :] = -np.linalg.solve(LOO, LEO.conj().T)
    return Pid

print()
def headroom(label, thx, thy, wx, wy, n):
    L = build(n, thx, thy, wx, wy)
    P0 = build_P(n, thx, thy, wx, wy)
    Lc = coarse(n, thx, thy, wx, wy)
    lo0, hi0 = pencil_range(L, P0, Lc)
    lo1, hi1 = pencil_range(L, ideal_of(n, L), Lc)
    print(f"headroom {label:26s} ours=[{lo0:6.3f},{hi0:7.3f}]  ideal=[{lo1:6.3f},{hi1:7.3f}]")

n3 = 16
w1 = np.ones((n3, n3))
headroom("flat theta=(0.3,0.7)", 0.3 * np.ones((n3, n3)), 0.7 * np.ones((n3, n3)), w1, w1, n3)
for nphi, tag in ((4, "physical nphi=4"), (32, "worst phi=2pi/8")):
    phi = 2 * np.pi * nphi / n3**2
    thx = np.zeros((n3, n3)); thy = np.zeros((n3, n3))
    for i in range(n3):
        for j in range(n3):
            thx[i, j] = -phi * j
    for i in range(n3):
        thy[i, n3 - 1] = phi * n3 * i
    headroom(tag, thx, thy, w1, w1, n3)
r = np.random.default_rng(1)
headroom("hot links seed=1", r.uniform(0, 2 * np.pi, (n3, n3)),
         r.uniform(0, 2 * np.pi, (n3, n3)), w1, w1, n3)
