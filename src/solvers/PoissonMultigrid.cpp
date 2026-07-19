#include "solvers/PoissonMultigrid.h"

#include <algorithm>
#include <cstdio>
#include <cmath>
#include <cstddef>

namespace bochner {
namespace {

// One grid level: the homogeneous-Neumann Laplacian needs only dims + weight
// (no link variables -- this is the trivial-connection, real-scalar case).
struct Level {
  int nx = 0, ny = 0, nz = 0;
  double w = 1.0;
  int index(int i, int j, int k) const { return (i * ny + j) * nz + k; }
  long numNodes() const { return static_cast<long>(nx) * ny * nz; }
};

int degree(const Level& L, int i, int j, int k) {
  return (i > 0) + (i + 1 < L.nx) + (j > 0) + (j + 1 < L.ny) + (k > 0) + (k + 1 < L.nz);
}

// Matvec y = L x: (Lx)_c = w * (deg_c x_c - sum of existing neighbours).
std::vector<double> matvec(const Level& L, const std::vector<double>& x) {
  std::vector<double> y(x.size(), 0.0);
#pragma omp parallel for schedule(static)
  for (int i = 0; i < L.nx; ++i)
    for (int j = 0; j < L.ny; ++j)
      for (int k = 0; k < L.nz; ++k) {
        const int c = L.index(i, j, k);
        double acc = 0.0;
        if (i > 0) acc += x[c] - x[L.index(i - 1, j, k)];
        if (i + 1 < L.nx) acc += x[c] - x[L.index(i + 1, j, k)];
        if (j > 0) acc += x[c] - x[L.index(i, j - 1, k)];
        if (j + 1 < L.ny) acc += x[c] - x[L.index(i, j + 1, k)];
        if (k > 0) acc += x[c] - x[L.index(i, j, k - 1)];
        if (k + 1 < L.nz) acc += x[c] - x[L.index(i, j, k + 1)];
        y[c] = L.w * acc;
      }
  return y;
}

// Red-black Gauss-Seidel (SOR) smoothing of L x = b: parallel within each colour.
void smooth(const Level& L, const std::vector<double>& b, std::vector<double>& x, int sweeps,
            double omega) {
  for (int s = 0; s < sweeps; ++s)
    for (int color = 0; color <= 1; ++color)
#pragma omp parallel for schedule(static)
      for (int i = 0; i < L.nx; ++i)
        for (int j = 0; j < L.ny; ++j)
          for (int k = 0; k < L.nz; ++k) {
            if (((i + j + k) & 1) != color) continue;
            const int deg = degree(L, i, j, k);
            if (deg == 0) continue;
            const int c = L.index(i, j, k);
            double sum = b[c] / L.w;
            if (i > 0) sum += x[L.index(i - 1, j, k)];
            if (i + 1 < L.nx) sum += x[L.index(i + 1, j, k)];
            if (j > 0) sum += x[L.index(i, j - 1, k)];
            if (j + 1 < L.ny) sum += x[L.index(i, j + 1, k)];
            if (k > 0) sum += x[L.index(i, j, k - 1)];
            if (k + 1 < L.nz) sum += x[L.index(i, j, k + 1)];
            x[c] = (1.0 - omega) * x[c] + omega * sum / deg;
          }
}

Level coarsen(const Level& f) {
  Level c;
  c.nx = f.nx / 2;
  c.ny = f.ny / 2;
  c.nz = f.nz / 2;
  c.w = f.w / 4.0;
  return c;
}

// Neighbour sources of a new (odd-coordinate) fine node, weight 1/cnt each.
void gatherSources(const Level& f, int i, int j, int k, int out[6], int& cnt) {
  cnt = 0;
  if (i & 1) {
    out[cnt++] = f.index(i - 1, j, k);
    if (i + 1 < f.nx) out[cnt++] = f.index(i + 1, j, k);
  }
  if (j & 1) {
    out[cnt++] = f.index(i, j - 1, k);
    if (j + 1 < f.ny) out[cnt++] = f.index(i, j + 1, k);
  }
  if (k & 1) {
    out[cnt++] = f.index(i, j, k - 1);
    if (k + 1 < f.nz) out[cnt++] = f.index(i, j, k + 1);
  }
}

std::vector<double> prolong(const Level& f, const std::vector<double>& coarse) {
  const int cny = f.ny / 2, cnz = f.nz / 2;  // cnx unused: cidx strides on cny/cnz only
  const auto cidx = [&](int I, int J, int K) { return (I * cny + J) * cnz + K; };
  std::vector<double> fine(static_cast<std::size_t>(f.numNodes()), 0.0);
  for (int i = 0; i < f.nx; i += 2)
    for (int j = 0; j < f.ny; j += 2)
      for (int k = 0; k < f.nz; k += 2) fine[f.index(i, j, k)] = coarse[cidx(i / 2, j / 2, k / 2)];
  for (int pass = 1; pass <= 3; ++pass)
    for (int i = 0; i < f.nx; ++i)
      for (int j = 0; j < f.ny; ++j)
        for (int k = 0; k < f.nz; ++k) {
          if ((i & 1) + (j & 1) + (k & 1) != pass) continue;
          int s[6], cnt = 0;
          gatherSources(f, i, j, k, s, cnt);
          double sum = 0.0;
          for (int t = 0; t < cnt; ++t) sum += fine[s[t]];
          fine[f.index(i, j, k)] = sum / cnt;
        }
  return fine;
}

std::vector<double> restrict(const Level& f, std::vector<double> r) {
  for (int pass = 3; pass >= 1; --pass)
    for (int i = 0; i < f.nx; ++i)
      for (int j = 0; j < f.ny; ++j)
        for (int k = 0; k < f.nz; ++k) {
          if ((i & 1) + (j & 1) + (k & 1) != pass) continue;
          int s[6], cnt = 0;
          gatherSources(f, i, j, k, s, cnt);
          const double val = r[f.index(i, j, k)];
          r[f.index(i, j, k)] = 0.0;
          for (int t = 0; t < cnt; ++t) r[s[t]] += val / cnt;
        }
  const int cnx = f.nx / 2, cny = f.ny / 2, cnz = f.nz / 2;
  std::vector<double> coarse(static_cast<std::size_t>(cnx) * cny * cnz, 0.0);
  for (int I = 0; I < cnx; ++I)
    for (int J = 0; J < cny; ++J)
      for (int K = 0; K < cnz; ++K) coarse[(I * cny + J) * cnz + K] = r[f.index(2 * I, 2 * J, 2 * K)];
  return coarse;
}

double dot(const std::vector<double>& a, const std::vector<double>& b) {
  double s = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
  return s;
}

void vcycle(const std::vector<Level>& levels, int l, std::vector<double>& x,
            const std::vector<double>& b, const PoissonMgOptions& opts) {
  const Level& A = levels[l];
  if (l + 1 == static_cast<int>(levels.size())) {
    smooth(A, b, x, opts.coarseSweeps, opts.omega);
    return;
  }
  smooth(A, b, x, opts.nu1, opts.omega);
  const std::vector<double> Ax = matvec(A, x);
  std::vector<double> r(b.size());
  for (std::size_t i = 0; i < r.size(); ++i) r[i] = b[i] - Ax[i];

  const std::vector<double> rc = restrict(A, r);
  std::vector<double> ec(static_cast<std::size_t>(levels[l + 1].numNodes()), 0.0);
  vcycle(levels, l + 1, ec, rc, opts);

  // Optimal A-energy step on the coarse correction (robust to transfer scaling).
  const std::vector<double> pe = prolong(A, ec);
  const std::vector<double> Ape = matvec(A, pe);
  const double den = dot(pe, Ape);
  const double alpha = den > 0.0 ? dot(pe, r) / den : 0.0;
  for (std::size_t i = 0; i < x.size(); ++i) x[i] += alpha * pe[i];
  smooth(A, b, x, opts.nu2, opts.omega);
}

}  // namespace

std::vector<double> applyPoisson(int nx, int ny, int nz, double h, const std::vector<double>& x) {
  Level L{nx, ny, nz, 1.0 / (h * h)};
  return matvec(L, x);
}

PoissonMgResult poissonVcycleSolve(int nx, int ny, int nz, double h, const std::vector<double>& rhs,
                                   std::vector<double>& phi, const PoissonMgOptions& opts) {
  std::vector<Level> levels{Level{nx, ny, nz, 1.0 / (h * h)}};
  while (levels.back().nx % 2 == 0 && levels.back().ny % 2 == 0 && levels.back().nz % 2 == 0 &&
         levels.back().nx >= 4 && levels.back().ny >= 4 && levels.back().nz >= 4)
    levels.push_back(coarsen(levels.back()));

  const double bnorm = std::sqrt(dot(rhs, rhs));
  PoissonMgResult res;
  res.levels = static_cast<int>(levels.size());
  // Coarsening stops at the first odd (or <4) extent, so e.g. n=65 gives ONE
  // level and the "V-cycle" below is just opts.coarseSweeps Gauss-Seidel sweeps
  // per cycle -- which will not converge at that resolution. Warn once rather
  // than per solve: this is called every frame from the projection.
  {
    const Level& c = levels.back();
    const int coarsest = std::max({c.nx, c.ny, c.nz});
    if (coarsest > 8) {
      static bool warned = false;
      if (!warned) {
        warned = true;
        std::fprintf(stderr,
                     "poissonVcycleSolve: hierarchy stops at %dx%dx%d after %zu level(s) -- "
                     "coarsening halts at the first odd or <4 extent, so convergence will be "
                     "poor. Use grid extents divisible by a high power of 2 (e.g. 32, 48, 64).\n",
                     c.nx, c.ny, c.nz, levels.size());
      }
    }
  }
  if (bnorm == 0.0) {
    // rhs = 0: the (minimum-norm) solution is phi = 0, so return it explicitly.
    // Leaving a warm start in phi would make the projection subtract
    // grad(phi_prev) from a zero field (phantom pressure on a re-seed).
    std::fill(phi.begin(), phi.end(), 0.0);
    return res;
  }
  for (int cyc = 1; cyc <= opts.maxCycles; ++cyc) {
    vcycle(levels, 0, phi, rhs, opts);
    const std::vector<double> Ax = matvec(levels[0], phi);
    std::vector<double> r(rhs.size());
    for (std::size_t i = 0; i < r.size(); ++i) r[i] = rhs[i] - Ax[i];
    res.cycles = cyc;
    res.relResidual = std::sqrt(dot(r, r)) / bnorm;
    if (res.relResidual < opts.tol) break;
  }
  return res;
}

}  // namespace bochner
