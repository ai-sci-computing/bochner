#include "fluid/PoissonMGPCG.h"

#include <algorithm>
#include <cmath>
#include <unordered_map>

namespace bochner {

SpMat toSpMat(const CooMatrix& A) {
  const auto entries = A.compressed();  // sorted by (row, col), duplicates summed
  SpMat m;
  m.rowStart.assign(A.rows() + 1, 0);
  m.col.reserve(entries.size());
  m.val.reserve(entries.size());
  m.diag.assign(A.rows(), 0.0);
  int r = 0;
  for (const auto& e : entries) {
    while (r < e.row) m.rowStart[++r] = static_cast<int>(m.col.size());
    m.col.push_back(e.col);
    m.val.push_back(e.value);
    if (e.row == e.col) m.diag[e.row] = e.value;
  }
  while (r < A.rows()) m.rowStart[++r] = static_cast<int>(m.col.size());
  return m;
}

namespace {

void spmv(const SpMat& m, const std::vector<double>& x, std::vector<double>& y) {
  const int n = m.n();
#pragma omp parallel for schedule(static)
  for (int i = 0; i < n; ++i) {
    double s = 0.0;
    for (int p = m.rowStart[i]; p < m.rowStart[i + 1]; ++p) s += m.val[p] * x[m.col[p]];
    y[i] = s;
  }
}

double dot(const std::vector<double>& a, const std::vector<double>& b) {
  double s = 0.0;
  const int n = static_cast<int>(a.size());
#pragma omp parallel for reduction(+ : s) schedule(static)
  for (int i = 0; i < n; ++i) s += a[i] * b[i];
  return s;
}

// Per-level scratch, allocated once per solve (not per V-cycle) to avoid malloc
// churn: `Ax`, the residual `r`, and the coarse correction `e` at each level.
struct Workspace {
  std::vector<std::vector<double>> Ax, r, e;
  explicit Workspace(const MgHierarchy& H) {
    for (const MgLevel& L : H.levels) {
      const int n = L.A.n();
      Ax.emplace_back(n);
      r.emplace_back(n);
      e.emplace_back(n);
    }
  }
};

// Weighted-Jacobi smoother (symmetric): x <- x + omega * D^{-1} (b - A x), applied
// only on active cells. omega = 0.8 damps the oscillatory error the coarse grid
// cannot see. `Ax` is caller-provided scratch.
void smooth(const MgLevel& L, std::vector<double>& x, const std::vector<double>& b, int sweeps,
            std::vector<double>& Ax) {
  constexpr double omega = 0.8;
  const int n = L.A.n();
  for (int s = 0; s < sweeps; ++s) {
    spmv(L.A, x, Ax);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i)
      if (L.active[i] && L.A.diag[i] != 0.0) x[i] += omega * (b[i] - Ax[i]) / L.A.diag[i];
  }
}

// One symmetric V-cycle: pre-smooth, restrict the residual (R = P^T = sum over the
// aggregate), recurse, prolong the correction (P = piecewise constant), post-smooth.
void vcycle(const MgHierarchy& H, Workspace& ws, int l, std::vector<double>& x,
            const std::vector<double>& b) {
  const MgLevel& L = H.levels[l];
  if (l + 1 == static_cast<int>(H.levels.size())) {
    smooth(L, x, b, 40, ws.Ax[l]);  // coarsest: a few dozen Jacobi sweeps solve it approximately
    return;
  }
  smooth(L, x, b, 2, ws.Ax[l]);

  const int n = L.A.n();
  spmv(L.A, x, ws.Ax[l]);
  for (int i = 0; i < n; ++i) ws.r[l][i] = b[i] - ws.Ax[l][i];

  std::vector<double>& rc = ws.r[l + 1];
  std::vector<double>& ec = ws.e[l + 1];
  std::fill(rc.begin(), rc.end(), 0.0);
  std::fill(ec.begin(), ec.end(), 0.0);
  for (int i = 0; i < n; ++i)
    if (L.active[i]) rc[L.aggUp[i]] += ws.r[l][i];  // restrict = P^T (sum)

  vcycle(H, ws, l + 1, ec, rc);

  for (int i = 0; i < n; ++i)
    if (L.active[i]) x[i] += ec[L.aggUp[i]];  // prolong = P (piecewise constant)

  smooth(L, x, b, 2, ws.Ax[l]);
}

}  // namespace

MgHierarchy buildPoissonMgHierarchy(int nx, int ny, int nz, const SpMat& A,
                                    const std::vector<char>& active) {
  MgHierarchy H;
  H.levels.push_back(MgLevel{nx, ny, nz, A, active, {}});

  while (true) {
    MgLevel& fine = H.levels.back();
    int activeCount = 0;
    for (char a : fine.active) activeCount += a;
    if (activeCount <= 64 || (fine.nx <= 2 && fine.ny <= 2 && fine.nz <= 2)) break;

    const int cnx = (fine.nx + 1) / 2, cny = (fine.ny + 1) / 2, cnz = (fine.nz + 1) / 2;
    const int cn = cnx * cny * cnz;
    std::vector<char> cactive(cn, 0);
    std::vector<int> agg(fine.A.n(), -1);
    auto cidx = [&](int I, int J, int K) { return (I * cny + J) * cnz + K; };
    for (int i = 0; i < fine.nx; ++i)
      for (int j = 0; j < fine.ny; ++j)
        for (int k = 0; k < fine.nz; ++k) {
          const int c = (i * fine.ny + j) * fine.nz + k;
          if (!fine.active[c]) continue;
          const int C = cidx(i / 2, j / 2, k / 2);
          agg[c] = C;
          cactive[C] = 1;
        }

    // Coarse operator A_c = P^T A P: sum each fine coupling into its aggregate pair.
    std::vector<std::unordered_map<int, double>> crows(cn);
    for (int r = 0; r < fine.A.n(); ++r) {
      if (!fine.active[r]) continue;
      const int R = agg[r];
      for (int p = fine.A.rowStart[r]; p < fine.A.rowStart[r + 1]; ++p) {
        const int c = fine.A.col[p];
        if (!fine.active[c]) continue;  // active rows never couple to inactive cells
        crows[R][agg[c]] += fine.A.val[p];
      }
    }
    SpMat Ac;
    Ac.rowStart.assign(cn + 1, 0);
    Ac.diag.assign(cn, 0.0);
    for (int R = 0; R < cn; ++R) {
      Ac.rowStart[R + 1] = Ac.rowStart[R] + static_cast<int>(crows[R].size());
    }
    Ac.col.resize(Ac.rowStart[cn]);
    Ac.val.resize(Ac.rowStart[cn]);
    for (int R = 0; R < cn; ++R) {
      // Sort columns for a canonical, cache-friendly row.
      std::vector<std::pair<int, double>> row(crows[R].begin(), crows[R].end());
      std::sort(row.begin(), row.end());
      int p = Ac.rowStart[R];
      for (auto& [C, v] : row) {
        Ac.col[p] = C;
        Ac.val[p] = v;
        if (C == R) Ac.diag[R] = v;
        ++p;
      }
    }

    fine.aggUp = std::move(agg);
    H.levels.push_back(MgLevel{cnx, cny, cnz, std::move(Ac), std::move(cactive), {}});
  }
  return H;
}

int mgpcgSolve(const MgHierarchy& H, const std::vector<double>& b, std::vector<double>& x,
               double tol, int maxit, double* relResidualOut) {
  const SpMat& A = H.levels[0].A;
  const std::vector<char>& active = H.levels[0].active;
  const int n = A.n();

  Workspace ws(H);
  // One V-cycle from a zero guess as the preconditioner z ~ M^{-1} r. Inactive
  // (solid/pinned) cells keep the identity action z = r so M stays SPD there.
  auto precond = [&](const std::vector<double>& r, std::vector<double>& z) {
    std::fill(z.begin(), z.end(), 0.0);
    vcycle(H, ws, 0, z, r);
    for (int i = 0; i < n; ++i)
      if (!active[i]) z[i] = r[i];
  };

  std::vector<double> r(n), z(n), p(n), Ap(n);
  spmv(A, x, Ap);
  double bnorm = 0.0, rnorm = 0.0;
  for (int i = 0; i < n; ++i) {
    r[i] = b[i] - Ap[i];
    bnorm += b[i] * b[i];
    rnorm += r[i] * r[i];
  }
  auto report = [&](double rn) {
    if (relResidualOut) *relResidualOut = bnorm > 0.0 ? std::sqrt(rn / bnorm) : 0.0;
  };
  if (bnorm == 0.0) {
    // b = 0: the SPD solution is x = 0, so return it explicitly. Leaving the
    // warm start in x would make the projection subtract grad(phi_prev) from a
    // zero field (phantom pressure on a viewer-style reset/re-seed).
    std::fill(x.begin(), x.end(), 0.0);
    return report(0.0), 0;
  }
  const double thresh = tol * tol * bnorm;
  if (rnorm <= thresh) return report(rnorm), 0;  // exact warm start: p would be 0

  precond(r, z);
  p = z;
  double rz = dot(r, z);
  int it = 0;
  for (; it < maxit; ++it) {
    spmv(A, p, Ap);
    const double pAp = dot(p, Ap);
    if (!(pAp > 0.0)) break;  // breakdown / NaN guard
    const double alpha = rz / pAp;
    rnorm = 0.0;
    for (int i = 0; i < n; ++i) {
      x[i] += alpha * p[i];
      r[i] -= alpha * Ap[i];
      rnorm += r[i] * r[i];
    }
    if (rnorm < thresh) return report(rnorm), it + 1;
    precond(r, z);
    const double rzNew = dot(r, z);
    const double beta = rzNew / rz;
    for (int i = 0; i < n; ++i) p[i] = z[i] + beta * p[i];
    rz = rzNew;
  }
  return report(rnorm), it;
}

}  // namespace bochner
