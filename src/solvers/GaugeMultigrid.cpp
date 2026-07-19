#include "solvers/GaugeMultigrid.h"

#include <atomic>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace bochner {
namespace {

using cd = std::complex<double>;

// Matvec instrumentation. g_matvecs counts every operator apply AND every
// smoother sweep as one, regardless of level size -- honest as a call count but
// it over-weights a geometric-MG's cheap coarse sweeps. g_matNodeWork instead
// accumulates the node count touched, so dividing by the finest level's node
// count gives the FINE-GRID-EQUIVALENT apply count -- the fair work metric
// across a multilevel hierarchy. Relaxed atomics: the counters are mutated on
// every production apply/sweep, so a concurrent solve + benchmark read must not
// tear them; no ordering is implied between the two counters.
std::atomic<long> g_matvecs{0};
std::atomic<long long> g_matNodeWork{0};

// Hermitian inner product <a,b> = sum conj(a_i) b_i.
cd cdot(const std::vector<cd>& a, const std::vector<cd>& b) {
  cd s(0.0, 0.0);
  for (std::size_t i = 0; i < a.size(); ++i) s += std::conj(a[i]) * b[i];
  return s;
}

// Axis link linear indices (forward link arrays). i/j/k is the *low* node of
// the edge. Open: the linked axis is one shorter (no wrap), so its stride is
// n-1; periodic: a full link per node (n), stride n. The x helper already uses
// the full ly/lz strides (correct for both); y/z pick the stride by topology.
int xi(const GaugeLattice& L, int i, int j, int k) { return (i * L.ly + j) * L.lz + k; }
int yi(const GaugeLattice& L, int i, int j, int k) {
  return (i * (L.periodic ? L.ly : L.ly - 1) + j) * L.lz + k;
}
int zi(const GaugeLattice& L, int i, int j, int k) {
  return (i * L.ly + j) * (L.periodic ? L.lz : L.lz - 1) + k;
}

cd expi(double a) { return cd(std::cos(a), std::sin(a)); }

// Number of existing axis neighbours of node (i,j,k) -- the connection
// Laplacian's diagonal degree (its diagonal entry is w * deg). A periodic node
// always has all six (each axis wraps); an open node drops its boundary edges.
int degree(const GaugeLattice& L, int i, int j, int k) {
  if (L.periodic) return 6;
  return (i > 0) + (i + 1 < L.lx) + (j > 0) + (j + 1 < L.ly) + (k > 0) + (k + 1 < L.lz);
}

// Per-edge weight lookup for the TRANSFERS. The weighted average is invariant
// under a global weight scale, so the uniform case uses 1.0 -- keeping the
// prolongation arithmetic bit-identical to the historical sum/count mean --
// while the weighted case uses the true per-edge weights (A-harmonic fill).
double pwx(const GaugeLattice& L, int i, int j, int k) {
  return L.weighted() ? L.wx[xi(L, i, j, k)] : 1.0;
}
double pwy(const GaugeLattice& L, int i, int j, int k) {
  return L.weighted() ? L.wy[yi(L, i, j, k)] : 1.0;
}
double pwz(const GaugeLattice& L, int i, int j, int k) {
  return L.weighted() ? L.wz[zi(L, i, j, k)] : 1.0;
}

// One source neighbour of a fine node during prolongation: its linear index,
// the parallel transport bringing that neighbour's value *to* this node, and
// the weight of the connecting edge (drives the A-harmonic weighted average;
// uniform lattices get equal weights, reducing to the plain mean).
struct Src {
  int idx;
  cd transport;
  double w;
};

// The immediate neighbours a new (odd-coordinate) fine node averages from: along
// each odd axis, the +/-1 neighbours, transported in. The forward link l takes
// low->high, so the low neighbour (i-1) comes in as e^{+i l[i-1]} and the high
// neighbour (i+1) as e^{-i l[i]}.
void gatherSources(const GaugeLattice& f, int i, int j, int k, Src out[6], int& cnt,
                   bool covariant = true) {
  cnt = 0;
  // A new (odd-coordinate) fine node averages from its two even neighbours along
  // each odd axis. Open: a boundary node has only one. Periodic (lx even, so i
  // odd => i+-1 even): both always exist, the high neighbour of i=lx-1 wrapping
  // to 0 via the wrap link f.tx[xi(f, lx-1, ...)].
  if (i & 1) {
    const int lo = f.periodic ? (i - 1 + f.lx) % f.lx : i - 1;
    out[cnt++] = {f.index(lo, j, k), f.tx[xi(f, lo, j, k)], pwx(f, lo, j, k)};
    if (f.periodic)
      out[cnt++] = {f.index((i + 1) % f.lx, j, k), std::conj(f.tx[xi(f, i, j, k)]), pwx(f, i, j, k)};
    else if (i + 1 < f.lx)
      out[cnt++] = {f.index(i + 1, j, k), std::conj(f.tx[xi(f, i, j, k)]), pwx(f, i, j, k)};
  }
  if (j & 1) {
    const int lo = f.periodic ? (j - 1 + f.ly) % f.ly : j - 1;
    out[cnt++] = {f.index(i, lo, k), f.ty[yi(f, i, lo, k)], pwy(f, i, lo, k)};
    if (f.periodic)
      out[cnt++] = {f.index(i, (j + 1) % f.ly, k), std::conj(f.ty[yi(f, i, j, k)]), pwy(f, i, j, k)};
    else if (j + 1 < f.ly)
      out[cnt++] = {f.index(i, j + 1, k), std::conj(f.ty[yi(f, i, j, k)]), pwy(f, i, j, k)};
  }
  if (k & 1) {
    const int lo = f.periodic ? (k - 1 + f.lz) % f.lz : k - 1;
    out[cnt++] = {f.index(i, j, lo), f.tz[zi(f, i, j, lo)], pwz(f, i, j, lo)};
    if (f.periodic)
      out[cnt++] = {f.index(i, j, (k + 1) % f.lz), std::conj(f.tz[zi(f, i, j, k)]), pwz(f, i, j, k)};
    else if (k + 1 < f.lz)
      out[cnt++] = {f.index(i, j, k + 1), std::conj(f.tz[zi(f, i, j, k)]), pwz(f, i, j, k)};
  }
  if (!covariant)
    for (int t = 0; t < cnt; ++t) out[t].transport = cd(1.0, 0.0);
}

// Coarsen by decimation: dims/2, w/4, and a coarse link is the sum of the two
// fine links it spans (the restricted U(1) connection). Weighted lattices also
// combine the two fine edge weights by series conductance,
//   W = wa*wb / (2*(wa+wb)),
// the Kron-reduction rule whose uniform case is exactly w/4 -- so the weighted
// hierarchy contains the uniform one.
double seriesW(double wa, double wb) { return wa * wb / (2.0 * (wa + wb)); }

GaugeLattice coarsen(const GaugeLattice& f) {
  GaugeLattice c;
  c.lx = f.lx / 2;
  c.ly = f.ly / 2;
  c.lz = f.lz / 2;
  c.periodic = f.periodic;
  c.w = f.w / 4.0;
  c.lkx.resize(static_cast<std::size_t>(c.numLinksX()));
  c.lky.resize(static_cast<std::size_t>(c.numLinksY()));
  c.lkz.resize(static_cast<std::size_t>(c.numLinksZ()));
  const bool wt = f.weighted();
  if (wt) {
    c.wx.resize(c.lkx.size());
    c.wy.resize(c.lky.size());
    c.wz.resize(c.lkz.size());
  }
  // A coarse forward link is the sum of the two fine links it spans (the
  // restricted U(1) connection). Periodic keeps the wrap link (I = clx-1, whose
  // second fine link 2I+1 = lx-1 is the fine wrap link); open stops one short.
  const int lastI = c.periodic ? c.lx : c.lx - 1;
  const int lastJ = c.periodic ? c.ly : c.ly - 1;
  const int lastK = c.periodic ? c.lz : c.lz - 1;
  for (int I = 0; I < lastI; ++I)
    for (int J = 0; J < c.ly; ++J)
      for (int K = 0; K < c.lz; ++K) {
        c.lkx[xi(c, I, J, K)] =
            f.lkx[xi(f, 2 * I, 2 * J, 2 * K)] + f.lkx[xi(f, 2 * I + 1, 2 * J, 2 * K)];
        if (wt)
          c.wx[xi(c, I, J, K)] =
              seriesW(f.wx[xi(f, 2 * I, 2 * J, 2 * K)], f.wx[xi(f, 2 * I + 1, 2 * J, 2 * K)]);
      }
  for (int I = 0; I < c.lx; ++I)
    for (int J = 0; J < lastJ; ++J)
      for (int K = 0; K < c.lz; ++K) {
        c.lky[yi(c, I, J, K)] =
            f.lky[yi(f, 2 * I, 2 * J, 2 * K)] + f.lky[yi(f, 2 * I, 2 * J + 1, 2 * K)];
        if (wt)
          c.wy[yi(c, I, J, K)] =
              seriesW(f.wy[yi(f, 2 * I, 2 * J, 2 * K)], f.wy[yi(f, 2 * I, 2 * J + 1, 2 * K)]);
      }
  for (int I = 0; I < c.lx; ++I)
    for (int J = 0; J < c.ly; ++J)
      for (int K = 0; K < lastK; ++K) {
        c.lkz[zi(c, I, J, K)] =
            f.lkz[zi(f, 2 * I, 2 * J, 2 * K)] + f.lkz[zi(f, 2 * I, 2 * J, 2 * K + 1)];
        if (wt)
          c.wz[zi(c, I, J, K)] =
              seriesW(f.wz[zi(f, 2 * I, 2 * J, 2 * K)], f.wz[zi(f, 2 * I, 2 * J, 2 * K + 1)]);
      }
  c.buildTransports();
  return c;
}

// Prolong a coarse correction to fine level \p f (transport averaging). Even
// nodes are copied; k-odd nodes are averaged from their sources, in passes
// k = 1,2,3 so every source is already filled.
std::vector<cd> prolong(const GaugeLattice& f, const std::vector<cd>& coarse,
                        bool covariant = true) {
  const int cly = f.ly / 2, clz = f.lz / 2;  // clx unused: cidx strides on cly/clz only
  const auto cidx = [&](int I, int J, int K) { return (I * cly + J) * clz + K; };
  std::vector<cd> fine(static_cast<std::size_t>(f.numNodes()), cd(0.0, 0.0));
  for (int i = 0; i < f.lx; i += 2)
    for (int j = 0; j < f.ly; j += 2)
      for (int k = 0; k < f.lz; k += 2)
        fine[f.index(i, j, k)] = coarse[cidx(i / 2, j / 2, k / 2)];
  for (int pass = 1; pass <= 3; ++pass)
    for (int i = 0; i < f.lx; ++i)
      for (int j = 0; j < f.ly; ++j)
        for (int k = 0; k < f.lz; ++k) {
          if ((i & 1) + (j & 1) + (k & 1) != pass) continue;
          Src s[6];
          int cnt = 0;
          gatherSources(f, i, j, k, s, cnt, covariant);
          cd sum(0.0, 0.0);
          double wsum = 0.0;
          for (int t = 0; t < cnt; ++t) {
            sum += s[t].w * s[t].transport * fine[s[t].idx];
            wsum += s[t].w;
          }
          fine[f.index(i, j, k)] = sum / wsum;
        }
  return fine;
}

// Restriction = exact adjoint of \ref prolong. prolong = M3 . M2 . M1 . Copy, so
// the adjoint applies M3^H, M2^H, M1^H (each scatters a node's value to its
// sources with conjugate transport, then clears it) and finally Copy^H (picks
// the even nodes).
std::vector<cd> restrict(const GaugeLattice& f, std::vector<cd> r, bool covariant = true) {
  for (int pass = 3; pass >= 1; --pass)
    for (int i = 0; i < f.lx; ++i)
      for (int j = 0; j < f.ly; ++j)
        for (int k = 0; k < f.lz; ++k) {
          if ((i & 1) + (j & 1) + (k & 1) != pass) continue;
          Src s[6];
          int cnt = 0;
          gatherSources(f, i, j, k, s, cnt, covariant);
          double wsum = 0.0;
          for (int t = 0; t < cnt; ++t) wsum += s[t].w;
          const cd val = r[f.index(i, j, k)];
          r[f.index(i, j, k)] = cd(0.0, 0.0);
          // Coefficient w_t/wsum applied as a division by (wsum/w_t): the
          // uniform case divides by exactly cnt, bit-identical to the
          // historical conj(t)/cnt.
          for (int t = 0; t < cnt; ++t)
            r[s[t].idx] += std::conj(s[t].transport) / (wsum / s[t].w) * val;
        }
  const int clx = f.lx / 2, cly = f.ly / 2, clz = f.lz / 2;
  std::vector<cd> coarse(static_cast<std::size_t>(clx) * cly * clz, cd(0.0, 0.0));
  for (int I = 0; I < clx; ++I)
    for (int J = 0; J < cly; ++J)
      for (int K = 0; K < clz; ++K) coarse[(I * cly + J) * clz + K] = r[f.index(2 * I, 2 * J, 2 * K)];
  return coarse;
}

// Red-black Gauss-Seidel (SOR) smoothing of E x = b. The cell graph is bipartite
// under parity (i+j+k)&1, so sweeping one colour then the other
// is a Gauss-Seidel sweep in which every update within a colour is independent
// -> a much stronger smoother than Jacobi and parallel within each colour (the
// natural OpenMP target). For row c, E x = b gives
//   x_c = (b_c + w * sum_n transport_{n->c} x_n) / (w * deg_c),
// relaxed by omega (omega = 1 is plain GS; 1 < omega < 2 over-relaxes).
// Weighted red-black GS sweep body: per-edge weights on every neighbour term
// and the diagonal = sum of incident edge weights (uniform: w * deg).
void smoothWeighted(const GaugeLattice& L, const std::vector<cd>& b, std::vector<cd>& x, int sweeps,
                    double omega) {
  for (int s = 0; s < sweeps; ++s) {
    g_matvecs.fetch_add(1, std::memory_order_relaxed);
    g_matNodeWork.fetch_add(L.numNodes(), std::memory_order_relaxed);
    for (int color = 0; color <= 1; ++color)
#pragma omp parallel for schedule(static)
      for (int i = 0; i < L.lx; ++i)
        for (int j = 0; j < L.ly; ++j)
          for (int k = 0; k < L.lz; ++k) {
            if (((i + j + k) & 1) != color) continue;
            const int c = L.index(i, j, k);
            cd sum = b[c];
            double diag = 0.0;
            if (L.periodic) {
              const int im = (i - 1 + L.lx) % L.lx, ip = (i + 1) % L.lx;
              const int jm = (j - 1 + L.ly) % L.ly, jp = (j + 1) % L.ly;
              const int km = (k - 1 + L.lz) % L.lz, kp = (k + 1) % L.lz;
              double we;
              we = L.wx[xi(L, im, j, k)]; diag += we;
              sum += we * L.tx[xi(L, im, j, k)] * x[L.index(im, j, k)];
              we = L.wx[xi(L, i, j, k)]; diag += we;
              sum += we * std::conj(L.tx[xi(L, i, j, k)]) * x[L.index(ip, j, k)];
              we = L.wy[yi(L, i, jm, k)]; diag += we;
              sum += we * L.ty[yi(L, i, jm, k)] * x[L.index(i, jm, k)];
              we = L.wy[yi(L, i, j, k)]; diag += we;
              sum += we * std::conj(L.ty[yi(L, i, j, k)]) * x[L.index(i, jp, k)];
              we = L.wz[zi(L, i, j, km)]; diag += we;
              sum += we * L.tz[zi(L, i, j, km)] * x[L.index(i, j, km)];
              we = L.wz[zi(L, i, j, k)]; diag += we;
              sum += we * std::conj(L.tz[zi(L, i, j, k)]) * x[L.index(i, j, kp)];
            } else {
              double we;
              if (i > 0) {
                we = L.wx[xi(L, i - 1, j, k)]; diag += we;
                sum += we * L.tx[xi(L, i - 1, j, k)] * x[L.index(i - 1, j, k)];
              }
              if (i + 1 < L.lx) {
                we = L.wx[xi(L, i, j, k)]; diag += we;
                sum += we * std::conj(L.tx[xi(L, i, j, k)]) * x[L.index(i + 1, j, k)];
              }
              if (j > 0) {
                we = L.wy[yi(L, i, j - 1, k)]; diag += we;
                sum += we * L.ty[yi(L, i, j - 1, k)] * x[L.index(i, j - 1, k)];
              }
              if (j + 1 < L.ly) {
                we = L.wy[yi(L, i, j, k)]; diag += we;
                sum += we * std::conj(L.ty[yi(L, i, j, k)]) * x[L.index(i, j + 1, k)];
              }
              if (k > 0) {
                we = L.wz[zi(L, i, j, k - 1)]; diag += we;
                sum += we * L.tz[zi(L, i, j, k - 1)] * x[L.index(i, j, k - 1)];
              }
              if (k + 1 < L.lz) {
                we = L.wz[zi(L, i, j, k)]; diag += we;
                sum += we * std::conj(L.tz[zi(L, i, j, k)]) * x[L.index(i, j, k + 1)];
              }
            }
            if (diag == 0.0) continue;
            x[c] = (1.0 - omega) * x[c] + omega * sum / diag;
          }
  }
}

void smooth(const GaugeLattice& L, const std::vector<cd>& b, std::vector<cd>& x, int sweeps,
            double omega) {
  if (L.weighted()) {
    smoothWeighted(L, b, x, sweeps, omega);
    return;
  }
  for (int s = 0; s < sweeps; ++s) {
    g_matvecs.fetch_add(1, std::memory_order_relaxed);  // a full sweep ~ one matvec
    g_matNodeWork.fetch_add(L.numNodes(), std::memory_order_relaxed);  // ...weighted by level size
    for (int color = 0; color <= 1; ++color)
#pragma omp parallel for schedule(static)
      for (int i = 0; i < L.lx; ++i)
        for (int j = 0; j < L.ly; ++j)
          for (int k = 0; k < L.lz; ++k) {
            if (((i + j + k) & 1) != color) continue;
            const int c = L.index(i, j, k);
            const int deg = degree(L, i, j, k);
            if (deg == 0) continue;
            cd sum = b[c];
            if (L.periodic) {
              const int im = (i - 1 + L.lx) % L.lx, ip = (i + 1) % L.lx;
              const int jm = (j - 1 + L.ly) % L.ly, jp = (j + 1) % L.ly;
              const int km = (k - 1 + L.lz) % L.lz, kp = (k + 1) % L.lz;
              sum += L.w * L.tx[xi(L, im, j, k)] * x[L.index(im, j, k)];
              sum += L.w * std::conj(L.tx[xi(L, i, j, k)]) * x[L.index(ip, j, k)];
              sum += L.w * L.ty[yi(L, i, jm, k)] * x[L.index(i, jm, k)];
              sum += L.w * std::conj(L.ty[yi(L, i, j, k)]) * x[L.index(i, jp, k)];
              sum += L.w * L.tz[zi(L, i, j, km)] * x[L.index(i, j, km)];
              sum += L.w * std::conj(L.tz[zi(L, i, j, k)]) * x[L.index(i, j, kp)];
            } else {
              if (i > 0) sum += L.w * L.tx[xi(L, i - 1, j, k)] * x[L.index(i - 1, j, k)];
              if (i + 1 < L.lx) sum += L.w * std::conj(L.tx[xi(L, i, j, k)]) * x[L.index(i + 1, j, k)];
              if (j > 0) sum += L.w * L.ty[yi(L, i, j - 1, k)] * x[L.index(i, j - 1, k)];
              if (j + 1 < L.ly) sum += L.w * std::conj(L.ty[yi(L, i, j, k)]) * x[L.index(i, j + 1, k)];
              if (k > 0) sum += L.w * L.tz[zi(L, i, j, k - 1)] * x[L.index(i, j, k - 1)];
              if (k + 1 < L.lz) sum += L.w * std::conj(L.tz[zi(L, i, j, k)]) * x[L.index(i, j, k + 1)];
            }
            x[c] = (1.0 - omega) * x[c] + omega * sum / (L.w * deg);
          }
  }
}

void vcycle(const std::vector<GaugeLattice>& levels, int l, std::vector<cd>& x,
            const std::vector<cd>& b, const MgOptions& opts) {
  const GaugeLattice& A = levels[l];
  if (l + 1 == static_cast<int>(levels.size())) {  // coarsest: smooth hard
    smooth(A, b, x, opts.coarseSweeps, opts.omega);
    return;
  }
  smooth(A, b, x, opts.nu1, opts.omega);
  const std::vector<cd> Ax = applyConnectionLaplacian(A, x);
  std::vector<cd> r(b.size());
  for (std::size_t i = 0; i < r.size(); ++i) r[i] = b[i] - Ax[i];

  const std::vector<cd> rc = restrict(A, r, opts.covariantTransfer);
  std::vector<cd> ec(static_cast<std::size_t>(levels[l + 1].numNodes()), cd(0.0, 0.0));
  vcycle(levels, l + 1, ec, rc, opts);

  // Coarse correction with an optimal A-energy step alpha = Re<p,r>/<p,A p>.
  // For SPD A this is a guaranteed descent direction, so it cannot diverge and
  // self-scales the correction -- decoupling convergence from any prolongation/
  // coarse-operator scale mismatch (and matching the Rayleigh-quotient spirit of
  // the eigensolver to come).
  const std::vector<cd> pe = prolong(A, ec, opts.covariantTransfer);
  double alpha = 1.0;  // raw injection (the ablation / historical-PTMG mode)
  if (opts.alphaStep) {
    const std::vector<cd> Ape = applyConnectionLaplacian(A, pe);
    cd num(0.0, 0.0);
    double den = 0.0;
    for (std::size_t i = 0; i < pe.size(); ++i) {
      num += std::conj(pe[i]) * r[i];
      den += std::real(std::conj(pe[i]) * Ape[i]);
    }
    alpha = den > 0.0 ? num.real() / den : 0.0;
  }
  for (std::size_t i = 0; i < x.size(); ++i) x[i] += alpha * pe[i];
  smooth(A, b, x, opts.nu2, opts.omega);
}

double l2(const std::vector<cd>& v) {
  double s = 0.0;
  for (const cd& z : v) s += std::norm(z);
  return std::sqrt(s);
}

}  // namespace

void GaugeLattice::setEdgeWeights(std::vector<double> wx_, std::vector<double> wy_,
                                  std::vector<double> wz_) {
  if (wx_.size() != static_cast<std::size_t>(numLinksX()) ||
      wy_.size() != static_cast<std::size_t>(numLinksY()) ||
      wz_.size() != static_cast<std::size_t>(numLinksZ()))
    throw std::invalid_argument("setEdgeWeights: weight array sizes must match the link counts");
  for (const auto* v : {&wx_, &wy_, &wz_})
    for (double e : *v)
      if (!(e > 0.0)) throw std::invalid_argument("setEdgeWeights: every weight must be > 0");
  wx = std::move(wx_);
  wy = std::move(wy_);
  wz = std::move(wz_);
}

void GaugeLattice::buildTransports() {
  tx.resize(lkx.size());
  ty.resize(lky.size());
  tz.resize(lkz.size());
  for (std::size_t i = 0; i < lkx.size(); ++i) tx[i] = expi(lkx[i]);
  for (std::size_t i = 0; i < lky.size(); ++i) ty[i] = expi(lky[i]);
  for (std::size_t i = 0; i < lkz.size(); ++i) tz[i] = expi(lkz[i]);
}

namespace {
// Weighted matvec: y_c = sum_e w_e (x_c - t_e x_n) over incident edges.
std::vector<cd> applyWeighted(const GaugeLattice& L, const std::vector<cd>& x) {
  std::vector<cd> y(x.size(), cd(0.0, 0.0));
#pragma omp parallel for schedule(static)
  for (int i = 0; i < L.lx; ++i)
    for (int j = 0; j < L.ly; ++j)
      for (int k = 0; k < L.lz; ++k) {
        const int c = L.index(i, j, k);
        const cd xc = x[c];
        cd acc(0.0, 0.0);
        if (L.periodic) {
          const int im = (i - 1 + L.lx) % L.lx, ip = (i + 1) % L.lx;
          const int jm = (j - 1 + L.ly) % L.ly, jp = (j + 1) % L.ly;
          const int km = (k - 1 + L.lz) % L.lz, kp = (k + 1) % L.lz;
          acc += L.wx[xi(L, im, j, k)] * (xc - L.tx[xi(L, im, j, k)] * x[L.index(im, j, k)]);
          acc += L.wx[xi(L, i, j, k)] * (xc - std::conj(L.tx[xi(L, i, j, k)]) * x[L.index(ip, j, k)]);
          acc += L.wy[yi(L, i, jm, k)] * (xc - L.ty[yi(L, i, jm, k)] * x[L.index(i, jm, k)]);
          acc += L.wy[yi(L, i, j, k)] * (xc - std::conj(L.ty[yi(L, i, j, k)]) * x[L.index(i, jp, k)]);
          acc += L.wz[zi(L, i, j, km)] * (xc - L.tz[zi(L, i, j, km)] * x[L.index(i, j, km)]);
          acc += L.wz[zi(L, i, j, k)] * (xc - std::conj(L.tz[zi(L, i, j, k)]) * x[L.index(i, j, kp)]);
        } else {
          if (i > 0)
            acc += L.wx[xi(L, i - 1, j, k)] * (xc - L.tx[xi(L, i - 1, j, k)] * x[L.index(i - 1, j, k)]);
          if (i + 1 < L.lx)
            acc += L.wx[xi(L, i, j, k)] * (xc - std::conj(L.tx[xi(L, i, j, k)]) * x[L.index(i + 1, j, k)]);
          if (j > 0)
            acc += L.wy[yi(L, i, j - 1, k)] * (xc - L.ty[yi(L, i, j - 1, k)] * x[L.index(i, j - 1, k)]);
          if (j + 1 < L.ly)
            acc += L.wy[yi(L, i, j, k)] * (xc - std::conj(L.ty[yi(L, i, j, k)]) * x[L.index(i, j + 1, k)]);
          if (k > 0)
            acc += L.wz[zi(L, i, j, k - 1)] * (xc - L.tz[zi(L, i, j, k - 1)] * x[L.index(i, j, k - 1)]);
          if (k + 1 < L.lz)
            acc += L.wz[zi(L, i, j, k)] * (xc - std::conj(L.tz[zi(L, i, j, k)]) * x[L.index(i, j, k + 1)]);
        }
        y[c] = acc;
      }
  return y;
}

// Mirror of SunGauge.cpp's validateSunLattice/requireVecSize. GaugeLattice is a
// public aggregate with no invariant, and hand-built lattices are the normal
// pattern across tools/ and tests/, so the constraints the factory functions
// enforce have to be re-checked where the operator is entered.
void validateGaugeLattice(const GaugeLattice& L, const char* who) {
  if (L.lx <= 0 || L.ly <= 0 || L.lz <= 0)
    throw std::invalid_argument(std::string(who) + ": lattice dimensions must be positive");
  // gaugeLatticePeriodic enforces this at construction, but a hand-built
  // periodic lattice with an odd dimension bypasses it: the wrap link joins
  // node n-1 to node 0, whose (i+j+k) parities are EQUAL for odd n, so the
  // red-black smoother would update two coupled nodes in the same colour --
  // an OpenMP read/write race and an invalid Gauss-Seidel bipartition.
  if (L.periodic && (L.lx % 2 || L.ly % 2 || L.lz % 2))
    throw std::invalid_argument(std::string(who) + ": every periodic dimension must be even");
}

void requireNodeVec(const GaugeLattice& L, const std::vector<cd>& v, const char* who,
                    const char* what) {
  if (v.size() != static_cast<std::size_t>(L.numNodes()))
    throw std::invalid_argument(std::string(who) + ": " + what + " must have numNodes() entries");
}

}  // namespace

std::vector<cd> applyConnectionLaplacian(const GaugeLattice& L, const std::vector<cd>& x) {
  // y is sized from x, but the loops below write y[c] for every c < numNodes
  // inside an omp parallel for -- a short x is a parallel heap overflow plus
  // out-of-bounds reads, with no diagnostic.
  validateGaugeLattice(L, "applyConnectionLaplacian");
  requireNodeVec(L, x, "applyConnectionLaplacian", "x");
  g_matvecs.fetch_add(1, std::memory_order_relaxed);
  g_matNodeWork.fetch_add(L.numNodes(), std::memory_order_relaxed);
  if (L.weighted()) return applyWeighted(L, x);
  std::vector<cd> y(x.size(), cd(0.0, 0.0));
#pragma omp parallel for schedule(static)
  for (int i = 0; i < L.lx; ++i)
    for (int j = 0; j < L.ly; ++j)
      for (int k = 0; k < L.lz; ++k) {
        const int c = L.index(i, j, k);
        const cd xc = x[c];
        cd acc(0.0, 0.0);
        if (L.periodic) {
          const int im = (i - 1 + L.lx) % L.lx, ip = (i + 1) % L.lx;
          const int jm = (j - 1 + L.ly) % L.ly, jp = (j + 1) % L.ly;
          const int km = (k - 1 + L.lz) % L.lz, kp = (k + 1) % L.lz;
          acc += xc - L.tx[xi(L, im, j, k)] * x[L.index(im, j, k)];
          acc += xc - std::conj(L.tx[xi(L, i, j, k)]) * x[L.index(ip, j, k)];
          acc += xc - L.ty[yi(L, i, jm, k)] * x[L.index(i, jm, k)];
          acc += xc - std::conj(L.ty[yi(L, i, j, k)]) * x[L.index(i, jp, k)];
          acc += xc - L.tz[zi(L, i, j, km)] * x[L.index(i, j, km)];
          acc += xc - std::conj(L.tz[zi(L, i, j, k)]) * x[L.index(i, j, kp)];
        } else {
          if (i > 0) acc += xc - L.tx[xi(L, i - 1, j, k)] * x[L.index(i - 1, j, k)];
          if (i + 1 < L.lx) acc += xc - std::conj(L.tx[xi(L, i, j, k)]) * x[L.index(i + 1, j, k)];
          if (j > 0) acc += xc - L.ty[yi(L, i, j - 1, k)] * x[L.index(i, j - 1, k)];
          if (j + 1 < L.ly) acc += xc - std::conj(L.ty[yi(L, i, j, k)]) * x[L.index(i, j + 1, k)];
          if (k > 0) acc += xc - L.tz[zi(L, i, j, k - 1)] * x[L.index(i, j, k - 1)];
          if (k + 1 < L.lz) acc += xc - std::conj(L.tz[zi(L, i, j, k)]) * x[L.index(i, j, k + 1)];
        }
        y[c] = L.w * acc;
      }
  return y;
}

GaugeLattice gaugeLatticeFromFaces(const MacGrid& g, const FaceField& theta) {
  GaugeLattice f;
  f.lx = g.nx();
  f.ly = g.ny();
  f.lz = g.nz();
  f.w = 1.0 / (g.spacing() * g.spacing());
  f.lkx.resize(static_cast<std::size_t>(f.lx > 0 ? f.lx - 1 : 0) * f.ly * f.lz);
  f.lky.resize(static_cast<std::size_t>(f.lx) * (f.ly > 0 ? f.ly - 1 : 0) * f.lz);
  f.lkz.resize(static_cast<std::size_t>(f.lx) * f.ly * (f.lz > 0 ? f.lz - 1 : 0));
  for (int i = 0; i + 1 < f.lx; ++i)
    for (int j = 0; j < f.ly; ++j)
      for (int k = 0; k < f.lz; ++k) f.lkx[xi(f, i, j, k)] = theta.x[g.faceXIndex(i + 1, j, k)];
  for (int i = 0; i < f.lx; ++i)
    for (int j = 0; j + 1 < f.ly; ++j)
      for (int k = 0; k < f.lz; ++k) f.lky[yi(f, i, j, k)] = theta.y[g.faceYIndex(i, j + 1, k)];
  for (int i = 0; i < f.lx; ++i)
    for (int j = 0; j < f.ly; ++j)
      for (int k = 0; k + 1 < f.lz; ++k) f.lkz[zi(f, i, j, k)] = theta.z[g.faceZIndex(i, j, k + 1)];
  f.buildTransports();
  return f;
}

GaugeLattice gaugeLatticePeriodic(int lx, int ly, int lz, double w, const std::vector<double>& lkx,
                                  const std::vector<double>& lky, const std::vector<double>& lkz) {
  GaugeLattice f;
  f.lx = lx;
  f.ly = ly;
  f.lz = lz;
  f.periodic = true;
  f.w = w;
  // Every periodic dimension must be even. The wrap link joins node n-1 to node 0,
  // whose (i+j+k) parities are equal for odd n, so the red-black smoother would
  // couple two same-color nodes -> an OpenMP read/write race and an invalid
  // Gauss-Seidel bipartition. (buildGaugeLevels needs /4 to keep coarse levels
  // even too, but the finest level must at least be even to smooth on.)
  if (lx % 2 || ly % 2 || lz % 2)
    throw std::invalid_argument("gaugeLatticePeriodic: every dimension must be even");
  const std::size_t n = static_cast<std::size_t>(lx) * ly * lz;
  if (lkx.size() != n || lky.size() != n || lkz.size() != n)
    throw std::invalid_argument("gaugeLatticePeriodic: each link array must be lx*ly*lz long");
  f.lkx = lkx;
  f.lky = lky;
  f.lkz = lkz;
  f.buildTransports();
  return f;
}

std::vector<cd> subdivisionSectionFromLattice(const GaugeLattice& finest, int numLevels) {
  // Precondition: every dimension must be divisible by 2^numLevels. Otherwise
  // `coarsen` truncates (odd/2) and the subsequent `prolong` reads past the
  // coarse extent -- out of bounds. The production caller (subdivisionSection)
  // validates this, but the function is public, so guard it here too.
  if (numLevels < 0)
    throw std::invalid_argument("subdivisionSectionFromLattice: numLevels must be >= 0");
  const int factor = 1 << numLevels;
  if (finest.lx % factor || finest.ly % factor || finest.lz % factor)
    throw std::invalid_argument(
        "subdivisionSectionFromLattice: every lattice dimension must be divisible by 2^numLevels");

  // Coarsen to `numLevels` levels, seed psi = 1 on the coarsest, prolong up. This
  // reuses the V-cycle's own coarsen/prolong (the covariant transport averaging),
  // so the covariant subdivision and the multigrid share one
  // implementation of the lattice + transfer.
  std::vector<GaugeLattice> levels(numLevels + 1);
  levels[numLevels] = finest;
  for (int l = numLevels - 1; l >= 0; --l) levels[l] = coarsen(levels[l + 1]);
  std::vector<cd> sec(static_cast<std::size_t>(levels[0].numNodes()), cd(1.0, 0.0));
  for (int l = 1; l <= numLevels; ++l) sec = prolong(levels[l], sec);
  return sec;
}

std::vector<cd> prolongGauge(const GaugeLattice& fine, const std::vector<cd>& coarse) {
  validateGaugeLattice(fine, "prolongGauge");
  return prolong(fine, coarse);
}

std::vector<cd> restrictGauge(const GaugeLattice& fine, const std::vector<cd>& fineVec) {
  validateGaugeLattice(fine, "restrictGauge");
  requireNodeVec(fine, fineVec, "restrictGauge", "fineVec");
  return restrict(fine, fineVec);
}

std::vector<GaugeLattice> buildGaugeLevels(const GaugeLattice& lat) {
  validateGaugeLattice(lat, "buildGaugeLevels");
  // Build the level pyramid: coarsen while every dimension stays coarsenable. Open
  // needs each dim even (>= 4 so the coarse level is >= 2). Periodic needs each
  // dim divisible by 4, so the *coarsened* dim is still even -- an odd periodic
  // level would put a wrap link between two same-parity nodes and break the
  // red-black smoother's bipartition.
  const int div = lat.periodic ? 4 : 2;
  std::vector<GaugeLattice> levels{lat};
  while (levels.back().lx % div == 0 && levels.back().ly % div == 0 && levels.back().lz % div == 0 &&
         levels.back().lx >= 4 && levels.back().ly >= 4 && levels.back().lz >= 4)
    levels.push_back(coarsen(levels.back()));
  return levels;
}

MgResult vcycleSolve(const std::vector<GaugeLattice>& levels, const std::vector<cd>& b,
                     std::vector<cd>& x, const MgOptions& opts) {
  if (levels.empty()) throw std::invalid_argument("vcycleSolve: empty level hierarchy");
  const GaugeLattice& lat = levels.front();  // finest
  // x is never resized on this path, so an empty or short x sends smooth() into
  // out-of-bounds writes.
  validateGaugeLattice(lat, "vcycleSolve");
  requireNodeVec(lat, b, "vcycleSolve", "b");
  requireNodeVec(lat, x, "vcycleSolve", "x");
  const double bnorm = l2(b);
  MgResult res;
  if (bnorm == 0.0) {
    // b = 0: the SPD solution is x = 0. A warm start must be overwritten here,
    // or the report (cycles=0, relResidual=0) would certify an x with an O(1)
    // true residual.
    x.assign(b.size(), cd(0.0, 0.0));
    res.relResidual = 0.0;
    return res;
  }
  // tol <= 0 means "run exactly maxCycles" (the eigensolver's preconditioner
  // mode): the residual test can never pass, so evaluating it would burn one
  // fine-level matvec + norm per cycle for a report the caller discards.
  // relResidual is -1 (not computed) -- a 0 would certify an unmeasured x.
  if (opts.tol <= 0.0) {
    for (int cyc = 1; cyc <= opts.maxCycles; ++cyc) {
      vcycle(levels, 0, x, b, opts);
      res.cycles = cyc;
    }
    res.relResidual = -1.0;
    return res;
  }
  for (int cyc = 1; cyc <= opts.maxCycles; ++cyc) {
    vcycle(levels, 0, x, b, opts);
    const std::vector<cd> Ax = applyConnectionLaplacian(lat, x);
    std::vector<cd> r(b.size());
    for (std::size_t i = 0; i < r.size(); ++i) r[i] = b[i] - Ax[i];
    res.cycles = cyc;
    res.relResidual = l2(r) / bnorm;
    if (res.relResidual < opts.tol) break;
  }
  return res;
}

MgResult vcycleSolve(const GaugeLattice& lat, const std::vector<cd>& b, std::vector<cd>& x,
                     const MgOptions& opts) {
  // Single-solve convenience: build the pyramid, then solve. Callers that solve
  // repeatedly on the same lattice (the eigensolver preconditions with one
  // V-cycle per covMG-LOBPCG step) should buildGaugeLevels once and reuse the overload
  // above, so the coarse hierarchy and its transports are not rebuilt each solve.
  return vcycleSolve(buildGaugeLevels(lat), b, x, opts);
}

SolveStats cgSolve(const GaugeLattice& lat, const std::vector<cd>& b, std::vector<cd>& x, double tol,
                   int maxIters) {
  validateGaugeLattice(lat, "cgSolve");
  requireNodeVec(lat, b, "cgSolve", "b");
  requireNodeVec(lat, x, "cgSolve", "x");
  SolveStats st;
  const double bnorm = std::sqrt(cdot(b, b).real());
  if (bnorm == 0.0) return st;

  std::vector<cd> r = applyConnectionLaplacian(lat, x);
  for (std::size_t i = 0; i < r.size(); ++i) r[i] = b[i] - r[i];
  double rs = cdot(r, r).real();
  st.relResidual = std::sqrt(rs) / bnorm;
  // Exact/near-exact warm start: x already solves E x = b. Return before forming
  // the search direction, which would otherwise be p = 0 -> pAp = 0 -> alpha NaN.
  if (st.relResidual < tol) return st;
  std::vector<cd> p = r;
  for (int it = 1; it <= maxIters; ++it) {
    const std::vector<cd> Ap = applyConnectionLaplacian(lat, p);
    const double pAp = cdot(p, Ap).real();
    if (!(pAp > 0.0)) break;  // breakdown (or NaN): stop rather than divide by ~0
    const double alpha = rs / pAp;
    for (std::size_t i = 0; i < x.size(); ++i) {
      x[i] += alpha * p[i];
      r[i] -= alpha * Ap[i];
    }
    const double rsNew = cdot(r, r).real();
    st.iterations = it;
    st.relResidual = std::sqrt(rsNew) / bnorm;
    if (st.relResidual < tol) break;
    const double beta = rsNew / rs;
    for (std::size_t i = 0; i < p.size(); ++i) p[i] = r[i] + beta * p[i];
    rs = rsNew;
  }
  return st;
}

long gaugeMatvecCount() { return g_matvecs.load(std::memory_order_relaxed); }
long long gaugeMatvecNodeWork() { return g_matNodeWork.load(std::memory_order_relaxed); }
void resetGaugeMatvecCount() {
  g_matvecs.store(0, std::memory_order_relaxed);
  g_matNodeWork.store(0, std::memory_order_relaxed);
}

std::vector<cd> toComplex(const std::vector<double>& interleaved) {
  std::vector<cd> z(interleaved.size() / 2);
  for (std::size_t c = 0; c < z.size(); ++c) z[c] = cd(interleaved[2 * c], interleaved[2 * c + 1]);
  return z;
}

std::vector<double> toInterleaved(const std::vector<cd>& z) {
  std::vector<double> v(2 * z.size());
  for (std::size_t c = 0; c < z.size(); ++c) {
    v[2 * c] = z[c].real();
    v[2 * c + 1] = z[c].imag();
  }
  return v;
}

}  // namespace bochner
