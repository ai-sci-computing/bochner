#include "solvers/SunGauge.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <random>
#include <stdexcept>
#include <string>

namespace bochner {
namespace {

using cd = std::complex<double>;

// ---- small dense d x d complex linear algebra (row-major, d <= 3) ----

// out = M v.
void matvec(const cd* M, const cd* v, int d, cd* out) {
  for (int i = 0; i < d; ++i) {
    cd s(0.0, 0.0);
    for (int j = 0; j < d; ++j) s += M[i * d + j] * v[j];
    out[i] = s;
  }
}

// out = M^H v  (Hermitian adjoint: the inverse transport for unitary M).
void adjMatvec(const cd* M, const cd* v, int d, cd* out) {
  for (int i = 0; i < d; ++i) {
    cd s(0.0, 0.0);
    for (int j = 0; j < d; ++j) s += std::conj(M[j * d + i]) * v[j];
    out[i] = s;
  }
}

// out = A B  (composition of transports along two collinear fine edges).
void matmul(const cd* A, const cd* B, int d, cd* out) {
  for (int i = 0; i < d; ++i)
    for (int j = 0; j < d; ++j) {
      cd s(0.0, 0.0);
      for (int k = 0; k < d; ++k) s += A[i * d + k] * B[k * d + j];
      out[i * d + j] = s;
    }
}

cd detSmall(const cd* M, int d) {
  if (d == 1) return M[0];
  if (d == 2) return M[0] * M[3] - M[1] * M[2];
  // d == 3, rule of Sarrus on row-major M[r*3+c].
  return M[0] * (M[4] * M[8] - M[5] * M[7]) - M[1] * (M[3] * M[8] - M[5] * M[6]) +
         M[2] * (M[3] * M[7] - M[4] * M[6]);
}

cd expi(double a) { return cd(std::cos(a), std::sin(a)); }

// A random SU(d) matrix (Haar U(1) phase for d==1): Gram-Schmidt of Gaussian
// columns -> Haar U(d), then scale column 0 by 1/det to force det = 1 -> SU(d).
void randomSU(int d, std::mt19937_64& rng, std::normal_distribution<double>& g,
              std::uniform_real_distribution<double>& u, cd* M) {
  if (d == 1) {
    M[0] = expi(u(rng));
    return;
  }
  for (int r = 0; r < d; ++r)
    for (int c = 0; c < d; ++c) M[r * d + c] = cd(g(rng), g(rng));
  // Modified Gram-Schmidt on the columns M[:,c].
  for (int c = 0; c < d; ++c) {
    for (int p = 0; p < c; ++p) {
      cd proj(0.0, 0.0);
      for (int r = 0; r < d; ++r) proj += std::conj(M[r * d + p]) * M[r * d + c];
      for (int r = 0; r < d; ++r) M[r * d + c] -= proj * M[r * d + p];
    }
    double nrm = 0.0;
    for (int r = 0; r < d; ++r) nrm += std::norm(M[r * d + c]);
    nrm = std::sqrt(nrm);
    for (int r = 0; r < d; ++r) M[r * d + c] /= nrm;
  }
  const cd det = detSmall(M, d);
  const cd phi = cd(1.0, 0.0) / det;  // |det| ~ 1, so this is ~ conj(det)
  for (int r = 0; r < d; ++r) M[r * d + d - 1] *= phi;  // scaling one column: det *= phi -> 1
}

// Axis forward-link linear indices (the low node of the edge), identical to the
// U(1) GaugeLattice layout.
int xi(const SunLattice& L, int i, int j, int k) { return (i * L.ly + j) * L.lz + k; }
int yi(const SunLattice& L, int i, int j, int k) {
  return (i * (L.periodic ? L.ly : L.ly - 1) + j) * L.lz + k;
}
int zi(const SunLattice& L, int i, int j, int k) {
  return (i * L.ly + j) * (L.periodic ? L.lz : L.lz - 1) + k;
}

// One axis neighbour of a node: its linear index, the forward link matrix on the
// connecting edge, and whether the transport n->c is that matrix (adj=false, low
// neighbour) or its Hermitian adjoint (adj=true, high neighbour).
struct Nbr {
  int nidx;
  const cd* U;
  bool adj;
  double w;  ///< connecting-edge weight (operator semantics; transfers use 1.0 when uniform)
};

// All existing axis neighbours of node (i,j,k) -- the operator/smoother
// neighbourhood (degree = cnt). Mirrors the U(1) six-neighbour stencil.
void fillOpNeighbours(const SunLattice& L, int i, int j, int k, Nbr out[6], int& cnt) {
  cnt = 0;
  const int dd = L.d * L.d;
  const bool wt = L.weighted();
  if (L.periodic || i > 0) {
    const int im = L.periodic ? (i - 1 + L.lx) % L.lx : i - 1;
    const int e = xi(L, im, j, k);
    out[cnt++] = {L.index(im, j, k), &L.ux[static_cast<std::size_t>(e) * dd], false, wt ? L.wx[e] : L.w};
  }
  if (L.periodic || i + 1 < L.lx) {
    const int ip = L.periodic ? (i + 1) % L.lx : i + 1;
    const int e = xi(L, i, j, k);
    out[cnt++] = {L.index(ip, j, k), &L.ux[static_cast<std::size_t>(e) * dd], true, wt ? L.wx[e] : L.w};
  }
  if (L.periodic || j > 0) {
    const int jm = L.periodic ? (j - 1 + L.ly) % L.ly : j - 1;
    const int e = yi(L, i, jm, k);
    out[cnt++] = {L.index(i, jm, k), &L.uy[static_cast<std::size_t>(e) * dd], false, wt ? L.wy[e] : L.w};
  }
  if (L.periodic || j + 1 < L.ly) {
    const int jp = L.periodic ? (j + 1) % L.ly : j + 1;
    const int e = yi(L, i, j, k);
    out[cnt++] = {L.index(i, jp, k), &L.uy[static_cast<std::size_t>(e) * dd], true, wt ? L.wy[e] : L.w};
  }
  if (L.periodic || k > 0) {
    const int km = L.periodic ? (k - 1 + L.lz) % L.lz : k - 1;
    const int e = zi(L, i, j, km);
    out[cnt++] = {L.index(i, j, km), &L.uz[static_cast<std::size_t>(e) * dd], false, wt ? L.wz[e] : L.w};
  }
  if (L.periodic || k + 1 < L.lz) {
    const int kp = L.periodic ? (k + 1) % L.lz : k + 1;
    const int e = zi(L, i, j, k);
    out[cnt++] = {L.index(i, j, kp), &L.uz[static_cast<std::size_t>(e) * dd], true, wt ? L.wz[e] : L.w};
  }
}

// Apply a neighbour's transport, or the identity when the covariant transfer is
// ablated off. The U(1) twin does this by overwriting the scalar transport with
// 1 in gatherSources; for matrix transports the equivalent is to skip the
// matvec, which is what MgOptions::covariantTransfer=false must mean here.
void applyTransport(const Nbr& nb, const cd* v, int d, cd* tmp, bool covariant, bool adjoint) {
  if (!covariant) {
    for (int a = 0; a < d; ++a) tmp[a] = v[a];
    return;
  }
  if (adjoint)
    adjMatvec(nb.U, v, d, tmp);
  else
    matvec(nb.U, v, d, tmp);
}

// The immediate neighbours a new (odd-coordinate) fine node averages from during
// prolongation: along each odd axis, the +/-1 neighbours transported in. Direct
// SU(d) analogue of the U(1) gatherSources.
void gatherSources(const SunLattice& f, int i, int j, int k, Nbr out[6], int& cnt) {
  cnt = 0;
  const int dd = f.d * f.d;
  const bool wt = f.weighted();
  if (i & 1) {
    const int lo = f.periodic ? (i - 1 + f.lx) % f.lx : i - 1;
    const int el = xi(f, lo, j, k), eh = xi(f, i, j, k);
    out[cnt++] = {f.index(lo, j, k), &f.ux[static_cast<std::size_t>(el) * dd], false, wt ? f.wx[el] : 1.0};
    if (f.periodic)
      out[cnt++] = {f.index((i + 1) % f.lx, j, k), &f.ux[static_cast<std::size_t>(eh) * dd], true, wt ? f.wx[eh] : 1.0};
    else if (i + 1 < f.lx)
      out[cnt++] = {f.index(i + 1, j, k), &f.ux[static_cast<std::size_t>(eh) * dd], true, wt ? f.wx[eh] : 1.0};
  }
  if (j & 1) {
    const int lo = f.periodic ? (j - 1 + f.ly) % f.ly : j - 1;
    const int el = yi(f, i, lo, k), eh = yi(f, i, j, k);
    out[cnt++] = {f.index(i, lo, k), &f.uy[static_cast<std::size_t>(el) * dd], false, wt ? f.wy[el] : 1.0};
    if (f.periodic)
      out[cnt++] = {f.index(i, (j + 1) % f.ly, k), &f.uy[static_cast<std::size_t>(eh) * dd], true, wt ? f.wy[eh] : 1.0};
    else if (j + 1 < f.ly)
      out[cnt++] = {f.index(i, j + 1, k), &f.uy[static_cast<std::size_t>(eh) * dd], true, wt ? f.wy[eh] : 1.0};
  }
  if (k & 1) {
    const int lo = f.periodic ? (k - 1 + f.lz) % f.lz : k - 1;
    const int el = zi(f, i, j, lo), eh = zi(f, i, j, k);
    out[cnt++] = {f.index(i, j, lo), &f.uz[static_cast<std::size_t>(el) * dd], false, wt ? f.wz[el] : 1.0};
    if (f.periodic)
      out[cnt++] = {f.index(i, j, (k + 1) % f.lz), &f.uz[static_cast<std::size_t>(eh) * dd], true, wt ? f.wz[eh] : 1.0};
    else if (k + 1 < f.lz)
      out[cnt++] = {f.index(i, j, k + 1), &f.uz[static_cast<std::size_t>(eh) * dd], true, wt ? f.wz[eh] : 1.0};
  }
}

cd cdot(const std::vector<cd>& a, const std::vector<cd>& b) {
  cd s(0.0, 0.0);
  for (std::size_t i = 0; i < a.size(); ++i) s += std::conj(a[i]) * b[i];
  return s;
}
double l2(const std::vector<cd>& v) {
  double s = 0.0;
  for (const cd& z : v) s += std::norm(z);
  return std::sqrt(s);
}
double norm2(const std::vector<cd>& v) { return l2(v); }

// Operator diagonal scale for the certificate floor: 6w + m^2 uniform, or an
// upper bound from the per-axis weight maxima when weighted. Tightness is
// irrelevant -- the floor sits orders below any gapped operator's lambda_min.
double diagScale(const SunLattice& L) {
  if (!L.weighted()) return 6.0 * L.w + L.mass2;
  auto mx = [](const std::vector<double>& v) {
    double m = 0.0;
    for (double x : v) m = std::max(m, x);
    return m;
  };
  return 2.0 * (mx(L.wx) + mx(L.wy) + mx(L.wz)) + L.mass2;
}
void scale(std::vector<cd>& v, cd s) {
  for (cd& z : v) z *= s;
}

// Coarsen by decimation: dims/2, w/4, mass2 unchanged (m^2 is h-independent),
// and a coarse forward link is the ORDERED PRODUCT U_{2I+1} U_{2I} of the two
// fine links it spans (the restricted non-abelian connection).
SunLattice coarsen(const SunLattice& f) {
  SunLattice c;
  c.d = f.d;
  c.lx = f.lx / 2;
  c.ly = f.ly / 2;
  c.lz = f.lz / 2;
  c.periodic = f.periodic;
  c.w = f.w / 4.0;
  c.mass2 = f.mass2;
  const int dd = f.d * f.d;
  c.ux.resize(static_cast<std::size_t>(c.numLinksX()) * dd);
  c.uy.resize(static_cast<std::size_t>(c.numLinksY()) * dd);
  c.uz.resize(static_cast<std::size_t>(c.numLinksZ()) * dd);
  const int lastI = c.periodic ? c.lx : c.lx - 1;
  const int lastJ = c.periodic ? c.ly : c.ly - 1;
  const int lastK = c.periodic ? c.lz : c.lz - 1;
  const bool wt = f.weighted();
  // Series-conductance combination of the two fine weights of a coarse edge
  // (Kron-reduction rule; uniform case = w/4).
  const auto seriesW = [](double wa, double wb) { return wa * wb / (2.0 * (wa + wb)); };
  if (wt) {
    c.wx.resize(static_cast<std::size_t>(c.numLinksX()));
    c.wy.resize(static_cast<std::size_t>(c.numLinksY()));
    c.wz.resize(static_cast<std::size_t>(c.numLinksZ()));
  }
  for (int I = 0; I < lastI; ++I)
    for (int J = 0; J < c.ly; ++J)
      for (int K = 0; K < c.lz; ++K) {
        matmul(&f.ux[static_cast<std::size_t>(xi(f, 2 * I + 1, 2 * J, 2 * K)) * dd],
               &f.ux[static_cast<std::size_t>(xi(f, 2 * I, 2 * J, 2 * K)) * dd], f.d,
               &c.ux[static_cast<std::size_t>(xi(c, I, J, K)) * dd]);
        if (wt)
          c.wx[xi(c, I, J, K)] =
              seriesW(f.wx[xi(f, 2 * I, 2 * J, 2 * K)], f.wx[xi(f, 2 * I + 1, 2 * J, 2 * K)]);
      }
  for (int I = 0; I < c.lx; ++I)
    for (int J = 0; J < lastJ; ++J)
      for (int K = 0; K < c.lz; ++K) {
        matmul(&f.uy[static_cast<std::size_t>(yi(f, 2 * I, 2 * J + 1, 2 * K)) * dd],
               &f.uy[static_cast<std::size_t>(yi(f, 2 * I, 2 * J, 2 * K)) * dd], f.d,
               &c.uy[static_cast<std::size_t>(yi(c, I, J, K)) * dd]);
        if (wt)
          c.wy[yi(c, I, J, K)] =
              seriesW(f.wy[yi(f, 2 * I, 2 * J, 2 * K)], f.wy[yi(f, 2 * I, 2 * J + 1, 2 * K)]);
      }
  for (int I = 0; I < c.lx; ++I)
    for (int J = 0; J < c.ly; ++J)
      for (int K = 0; K < lastK; ++K) {
        matmul(&f.uz[static_cast<std::size_t>(zi(f, 2 * I, 2 * J, 2 * K + 1)) * dd],
               &f.uz[static_cast<std::size_t>(zi(f, 2 * I, 2 * J, 2 * K)) * dd], f.d,
               &c.uz[static_cast<std::size_t>(zi(c, I, J, K)) * dd]);
        if (wt)
          c.wz[zi(c, I, J, K)] =
              seriesW(f.wz[zi(f, 2 * I, 2 * J, 2 * K)], f.wz[zi(f, 2 * I, 2 * J, 2 * K + 1)]);
      }
  return c;
}

// Prolong a coarse correction to fine level f by covariant transport averaging.
std::vector<cd> prolong(const SunLattice& f, const std::vector<cd>& coarse,
                        bool covariant = true) {
  const int d = f.d;
  const int cly = f.ly / 2, clz = f.lz / 2;
  const auto cidx = [&](int I, int J, int K) { return (I * cly + J) * clz + K; };
  std::vector<cd> fine(static_cast<std::size_t>(f.dof()), cd(0.0, 0.0));
  for (int i = 0; i < f.lx; i += 2)
    for (int j = 0; j < f.ly; j += 2)
      for (int k = 0; k < f.lz; k += 2)
        for (int a = 0; a < d; ++a)
          fine[static_cast<std::size_t>(f.index(i, j, k)) * d + a] =
              coarse[static_cast<std::size_t>(cidx(i / 2, j / 2, k / 2)) * d + a];
  for (int pass = 1; pass <= 3; ++pass)
    for (int i = 0; i < f.lx; ++i)
      for (int j = 0; j < f.ly; ++j)
        for (int k = 0; k < f.lz; ++k) {
          if ((i & 1) + (j & 1) + (k & 1) != pass) continue;
          Nbr s[6];
          int cnt = 0;
          gatherSources(f, i, j, k, s, cnt);
          cd sum[3] = {cd(0, 0), cd(0, 0), cd(0, 0)}, tmp[3];
          double wsum = 0.0;
          for (int t = 0; t < cnt; ++t) {
            const cd* v = &fine[static_cast<std::size_t>(s[t].nidx) * d];
            applyTransport(s[t], v, d, tmp, covariant, s[t].adj);
            for (int a = 0; a < d; ++a) sum[a] += s[t].w * tmp[a];
            wsum += s[t].w;
          }
          for (int a = 0; a < d; ++a)
            fine[static_cast<std::size_t>(f.index(i, j, k)) * d + a] = sum[a] / wsum;
        }
  return fine;
}

// Restriction = exact adjoint of prolong (transpose the transport cascade with
// Hermitian-adjoint transports, then pick the even nodes).
std::vector<cd> restrict(const SunLattice& f, std::vector<cd> r, bool covariant = true) {
  const int d = f.d;
  for (int pass = 3; pass >= 1; --pass)
    for (int i = 0; i < f.lx; ++i)
      for (int j = 0; j < f.ly; ++j)
        for (int k = 0; k < f.lz; ++k) {
          if ((i & 1) + (j & 1) + (k & 1) != pass) continue;
          Nbr s[6];
          int cnt = 0;
          gatherSources(f, i, j, k, s, cnt);
          cd val[3], tmp[3];
          double wsum = 0.0;
          for (int t = 0; t < cnt; ++t) wsum += s[t].w;
          for (int a = 0; a < d; ++a) {
            val[a] = r[static_cast<std::size_t>(f.index(i, j, k)) * d + a];
            r[static_cast<std::size_t>(f.index(i, j, k)) * d + a] = cd(0.0, 0.0);
          }
          for (int t = 0; t < cnt; ++t) {
            // adjoint of the transport prolong applied: adj ? U : U^H, with the
            // same real coefficient w_t / wsum the prolongation used
            applyTransport(s[t], val, d, tmp, covariant, !s[t].adj);
            cd* dst = &r[static_cast<std::size_t>(s[t].nidx) * d];
            for (int a = 0; a < d; ++a) dst[a] += (s[t].w / wsum) * tmp[a];
          }
        }
  const int clx = f.lx / 2, cly = f.ly / 2, clz = f.lz / 2;
  std::vector<cd> coarse(static_cast<std::size_t>(clx) * cly * clz * d, cd(0.0, 0.0));
  for (int I = 0; I < clx; ++I)
    for (int J = 0; J < cly; ++J)
      for (int K = 0; K < clz; ++K)
        for (int a = 0; a < d; ++a)
          coarse[(static_cast<std::size_t>((I * cly + J) * clz + K)) * d + a] =
              r[static_cast<std::size_t>(f.index(2 * I, 2 * J, 2 * K)) * d + a];
  return coarse;
}

// Red-black Gauss-Seidel: x_c = (b_c + w sum_n U_{n->c} x_n) / (w*deg + m^2),
// relaxed by omega. Diagonal is a scalar multiple of the identity, so the block
// update is a plain divide -- as cheap as the U(1) case.
void smooth(const SunLattice& L, const std::vector<cd>& b, std::vector<cd>& x, int sweeps,
            double omega) {
  const int d = L.d;
  for (int s = 0; s < sweeps; ++s)
    for (int color = 0; color <= 1; ++color)
#pragma omp parallel for schedule(static)
      for (int i = 0; i < L.lx; ++i)
        for (int j = 0; j < L.ly; ++j)
          for (int k = 0; k < L.lz; ++k) {
            if (((i + j + k) & 1) != color) continue;
            Nbr nb[6];
            int cnt = 0;
            fillOpNeighbours(L, i, j, k, nb, cnt);
            if (cnt == 0 && L.mass2 == 0.0) continue;
            const std::size_t c = static_cast<std::size_t>(L.index(i, j, k)) * d;
            cd sum[3], tmp[3];
            for (int a = 0; a < d; ++a) sum[a] = b[c + a];
            double wdeg = 0.0;
            for (int t = 0; t < cnt; ++t) {
              const cd* v = &x[static_cast<std::size_t>(nb[t].nidx) * d];
              if (nb[t].adj)
                adjMatvec(nb[t].U, v, d, tmp);
              else
                matvec(nb[t].U, v, d, tmp);
              for (int a = 0; a < d; ++a) sum[a] += nb[t].w * tmp[a];
              wdeg += nb[t].w;
            }
            const double diag = wdeg + L.mass2;
            for (int a = 0; a < d; ++a) x[c + a] = (1.0 - omega) * x[c + a] + omega * sum[a] / diag;
          }
}

void vcycle(const std::vector<SunLattice>& levels, int l, std::vector<cd>& x, const std::vector<cd>& b,
            const MgOptions& opts) {
  const SunLattice& A = levels[l];
  if (l + 1 == static_cast<int>(levels.size())) {
    smooth(A, b, x, opts.coarseSweeps, opts.omega);
    return;
  }
  smooth(A, b, x, opts.nu1, opts.omega);
  const std::vector<cd> Ax = applySunLaplacian(A, x);
  std::vector<cd> r(b.size());
  for (std::size_t i = 0; i < r.size(); ++i) r[i] = b[i] - Ax[i];

  const std::vector<cd> rc = restrict(A, r, opts.covariantTransfer);
  std::vector<cd> ec(static_cast<std::size_t>(levels[l + 1].dof()), cd(0.0, 0.0));
  vcycle(levels, l + 1, ec, rc, opts);

  // Optimal A-energy coarse correction alpha = Re<p,r>/<p,A p> -- unchanged from
  // the U(1) cycle (needs only SPD-ness), and what keeps the non-Galerkin
  // covariant cycle from diverging.
  const std::vector<cd> pe = prolong(A, ec, opts.covariantTransfer);
  if (!opts.alphaStep) {
    // Ablation: raw injection of the coarse correction (the historical PTMG
    // mode). Kept switchable so a referee can reproduce the ablation table on
    // the SU(d) operator, not only on U(1).
    for (std::size_t i = 0; i < x.size(); ++i) x[i] += pe[i];
    smooth(A, b, x, opts.nu2, opts.omega);
    return;
  }
  const std::vector<cd> Ape = applySunLaplacian(A, pe);
  cd num(0.0, 0.0);
  double den = 0.0;
  for (std::size_t i = 0; i < pe.size(); ++i) {
    num += std::conj(pe[i]) * r[i];
    den += std::real(std::conj(pe[i]) * Ape[i]);
  }
  const double alpha = den > 0.0 ? num.real() / den : 0.0;
  for (std::size_t i = 0; i < x.size(); ++i) x[i] += alpha * pe[i];
  smooth(A, b, x, opts.nu2, opts.omega);
}

// Smallest eigenpair of a small (m <= 3) dense complex-Hermitian matrix by
// cyclic Hermitian Jacobi -- identical to the U(1) GaugeEigen helper.
void smallestHermitianEig(int m, cd H[3][3], double& lam, cd c[3]) {
  cd D[3][3], V[3][3];
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < m; ++j) {
      D[i][j] = H[i][j];
      V[i][j] = (i == j) ? cd(1.0, 0.0) : cd(0.0, 0.0);
    }
  for (int sweep = 0; sweep < 40; ++sweep) {
    double off = 0.0;
    for (int p = 0; p < m; ++p)
      for (int q = p + 1; q < m; ++q) off += std::norm(D[p][q]);
    if (off < 1e-30) break;
    for (int p = 0; p < m; ++p)
      for (int q = p + 1; q < m; ++q) {
        const cd dpq = D[p][q];
        if (std::abs(dpq) < 1e-18) continue;
        const double a = D[p][p].real(), b = D[q][q].real(), mag = std::abs(dpq);
        const cd g = std::exp(cd(0.0, -std::arg(dpq)));
        const double theta = 0.5 * std::atan2(2.0 * mag, a - b);
        const double cc = std::cos(theta), ss = std::sin(theta);
        const cd U00(cc, 0.0), U01(-ss, 0.0), U10 = g * ss, U11 = g * cc;
        const cd c00 = std::conj(U00), c01 = std::conj(U01), c10 = std::conj(U10), c11 = std::conj(U11);
        for (int i = 0; i < m; ++i) {
          const cd dip = D[i][p], diq = D[i][q];
          D[i][p] = dip * U00 + diq * U10;
          D[i][q] = dip * U01 + diq * U11;
        }
        for (int j = 0; j < m; ++j) {
          const cd dpj = D[p][j], dqj = D[q][j];
          D[p][j] = c00 * dpj + c10 * dqj;
          D[q][j] = c01 * dpj + c11 * dqj;
        }
        for (int i = 0; i < m; ++i) {
          const cd vip = V[i][p], viq = V[i][q];
          V[i][p] = vip * U00 + viq * U10;
          V[i][q] = vip * U01 + viq * U11;
        }
      }
  }
  int jmin = 0;
  for (int j = 1; j < m; ++j)
    if (D[j][j].real() < D[jmin][jmin].real()) jmin = j;
  lam = D[jmin][jmin].real();
  for (int i = 0; i < m; ++i) c[i] = V[i][jmin];
}

// SunLattice::d is a public field with no invariant, and hand-built lattices are
// the normal pattern in tools/ and tests/ -- so the constraints the factory
// functions enforce have to be re-checked at the operator entry points, not
// just at construction. Every fixed-size buffer on the operator paths (acc/tmp
// here, T1..S in WilsonMc, the prolong/restrict/smooth scratch) is sized 3, so
// d >= 4 smashes the stack silently.
void validateSunLattice(const SunLattice& L, const char* who) {
  if (L.d < 1 || L.d > 3)
    throw std::invalid_argument(std::string(who) + ": SunLattice::d must be 1..3");
  // Same invariant as the factory functions: an odd periodic dim wraps two
  // same-parity nodes onto one red-black color -> OpenMP data race in smooth
  // plus an invalid Gauss-Seidel bipartition.
  if (L.periodic && (L.lx % 2 || L.ly % 2 || L.lz % 2))
    throw std::invalid_argument(std::string(who) + ": every periodic dimension must be even");
}

void requireVecSize(const SunLattice& L, const std::vector<cd>& v, const char* who,
                    const char* what) {
  const std::size_t want = static_cast<std::size_t>(L.numNodes()) * static_cast<std::size_t>(L.d);
  if (v.size() != want)
    throw std::invalid_argument(std::string(who) + ": " + what + " has wrong length");
}

}  // namespace

std::vector<cd> applySunLaplacian(const SunLattice& L, const std::vector<cd>& x) {
  validateSunLattice(L, "applySunLaplacian");
  requireVecSize(L, x, "applySunLaplacian", "x");
  const int d = L.d;
  std::vector<cd> y(x.size(), cd(0.0, 0.0));
#pragma omp parallel for schedule(static)
  for (int i = 0; i < L.lx; ++i)
    for (int j = 0; j < L.ly; ++j)
      for (int k = 0; k < L.lz; ++k) {
        Nbr nb[6];
        int cnt = 0;
        fillOpNeighbours(L, i, j, k, nb, cnt);
        const std::size_t c = static_cast<std::size_t>(L.index(i, j, k)) * d;
        cd acc[3] = {cd(0, 0), cd(0, 0), cd(0, 0)}, tmp[3];
        for (int t = 0; t < cnt; ++t) {
          const cd* v = &x[static_cast<std::size_t>(nb[t].nidx) * d];
          if (nb[t].adj)
            adjMatvec(nb[t].U, v, d, tmp);
          else
            matvec(nb[t].U, v, d, tmp);
          for (int a = 0; a < d; ++a) acc[a] += nb[t].w * (x[c + a] - tmp[a]);
        }
        for (int a = 0; a < d; ++a) y[c + a] = acc[a] + L.mass2 * x[c + a];
      }
  return y;
}

std::vector<cd> prolongSun(const SunLattice& fine, const std::vector<cd>& coarse) {
  validateSunLattice(fine, "prolongSun");
  return prolong(fine, coarse);
}

std::vector<cd> restrictSun(const SunLattice& fine, const std::vector<cd>& fineVec) {
  validateSunLattice(fine, "restrictSun");
  requireVecSize(fine, fineVec, "restrictSun", "fineVec");
  return restrict(fine, fineVec);
}

std::vector<SunLattice> buildSunLevels(const SunLattice& lat) {
  validateSunLattice(lat, "buildSunLevels");
  const int div = lat.periodic ? 4 : 2;
  std::vector<SunLattice> levels{lat};
  while (levels.back().lx % div == 0 && levels.back().ly % div == 0 && levels.back().lz % div == 0 &&
         levels.back().lx >= 4 && levels.back().ly >= 4 && levels.back().lz >= 4)
    levels.push_back(coarsen(levels.back()));
  return levels;
}

MgResult vcycleSolveSun(const std::vector<SunLattice>& levels, const std::vector<cd>& b,
                        std::vector<cd>& x, const MgOptions& opts) {
  const SunLattice& lat = levels.front();
  // x is never resized on this path, so an empty or short x would send smooth()
  // into out-of-bounds writes with no diagnostic.
  requireVecSize(lat, b, "vcycleSolveSun", "b");
  requireVecSize(lat, x, "vcycleSolveSun", "x");
  const double bnorm = l2(b);
  MgResult res;
  if (bnorm == 0.0) {
    // b = 0: the SPD solution is x = 0. A warm start must be overwritten here,
    // or the report (cycles=0, relResidual=0) would certify an x with an O(1)
    // true residual.
    x.assign(b.size(), cd(0.0, 0.0));
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
    const std::vector<cd> Ax = applySunLaplacian(lat, x);
    std::vector<cd> r(b.size());
    for (std::size_t i = 0; i < r.size(); ++i) r[i] = b[i] - Ax[i];
    res.cycles = cyc;
    res.relResidual = l2(r) / bnorm;
    if (res.relResidual < opts.tol) break;
  }
  return res;
}

MgResult vcycleSolveSun(const SunLattice& lat, const std::vector<cd>& b, std::vector<cd>& x,
                        const MgOptions& opts) {
  return vcycleSolveSun(buildSunLevels(lat), b, x, opts);
}

SolveStats cgSolveSun(const SunLattice& lat, const std::vector<cd>& b, std::vector<cd>& x, double tol,
                      int maxIters) {
  validateSunLattice(lat, "cgSolveSun");
  requireVecSize(lat, b, "cgSolveSun", "b");
  requireVecSize(lat, x, "cgSolveSun", "x");
  SolveStats st;
  const double bnorm = std::sqrt(cdot(b, b).real());
  if (bnorm == 0.0) return st;
  std::vector<cd> r = applySunLaplacian(lat, x);
  for (std::size_t i = 0; i < r.size(); ++i) r[i] = b[i] - r[i];
  double rs = cdot(r, r).real();
  st.relResidual = std::sqrt(rs) / bnorm;
  if (st.relResidual < tol) return st;
  std::vector<cd> p = r;
  for (int it = 1; it <= maxIters; ++it) {
    const std::vector<cd> Ap = applySunLaplacian(lat, p);
    const double pAp = cdot(p, Ap).real();
    if (!(pAp > 0.0)) break;
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

std::vector<cd> subdivisionSectionSun(const SunLattice& finest, int numLevels) {
  validateSunLattice(finest, "subdivisionSectionSun");
  if (numLevels < 0) throw std::invalid_argument("subdivisionSectionSun: numLevels must be >= 0");
  const int factor = 1 << numLevels;
  if (finest.lx % factor || finest.ly % factor || finest.lz % factor)
    throw std::invalid_argument("subdivisionSectionSun: every dimension must be divisible by 2^numLevels");
  std::vector<SunLattice> levels(numLevels + 1);
  levels[numLevels] = finest;
  for (int l = numLevels - 1; l >= 0; --l) levels[l] = coarsen(levels[l + 1]);
  std::vector<cd> sec(static_cast<std::size_t>(levels[0].dof()), cd(1.0, 0.0));  // constant frame
  for (int l = 1; l <= numLevels; ++l) sec = prolong(levels[l], sec);
  return sec;
}

GaugeEigenResult smallestEigenpairSunMG(const SunLattice& lat, const std::vector<cd>* initialGuess,
                                        const GaugeEigenOptions& opts) {
  const std::size_t n = static_cast<std::size_t>(lat.dof());
  std::vector<cd> x(n, cd(1.0, 0.0));
  if (initialGuess && initialGuess->size() == n) x = *initialGuess;
  {
    double nx = norm2(x);
    if (!std::isfinite(nx) || !(nx > 0.0) || !std::isfinite(1.0 / nx)) {
      x.assign(n, cd(1.0, 0.0));
      nx = norm2(x);
    }
    scale(x, cd(1.0 / nx, 0.0));
  }

  MgOptions pmg = opts.mg;
  pmg.tol = 0.0;
  pmg.maxCycles = opts.precCycles;
  const std::vector<SunLattice> pmgLevels = buildSunLevels(lat);
  const double certFloor = opts.certFloorRel * diagScale(lat);

  GaugeEigenResult res;
  res.vector = x;
  std::vector<cd> xprev;

  for (int it = 1; it <= opts.maxIters; ++it) {
    const std::vector<cd> Ex = applySunLaplacian(lat, x);
    const double rho = cdot(x, Ex).real();
    res.eigenvalue = rho;
    res.iterations = it;

    std::vector<cd> r(n);
    for (std::size_t i = 0; i < n; ++i) r[i] = Ex[i] - rho * x[i];
    res.residual = norm2(r) / std::max({std::abs(rho), certFloor, 1e-300});
    if (res.residual < opts.tol) break;

    std::vector<cd> w(n, cd(0.0, 0.0));
    vcycleSolveSun(pmgLevels, r, w, pmg);

    std::vector<std::vector<cd>> q;
    q.push_back(x);
    auto orthonormalAdd = [&](std::vector<cd> v) {
      const double nv0 = norm2(v);
      for (const auto& qi : q) {
        const cd proj = cdot(qi, v);
        for (std::size_t t = 0; t < n; ++t) v[t] -= proj * qi[t];
      }
      const double nv = norm2(v);
      // Drop near-dependent directions: relative to the direction's own norm
      // (certifies tight tolerances on dense-edge spectra) or the legacy
      // absolute 1e-7 (doubles as the warm-start early-exit; see the option).
      if (nv > (opts.relativeGsDrop ? 1e-7 * nv0 : 1e-7)) {
        scale(v, cd(1.0 / nv, 0.0));
        q.push_back(std::move(v));
      }
    };
    orthonormalAdd(w);
    if (!xprev.empty()) orthonormalAdd(xprev);
    const int m = static_cast<int>(q.size());
    if (m == 1) break;

    std::vector<std::vector<cd>> Aq(m);
    Aq[0] = Ex;
    for (int i = 1; i < m; ++i) Aq[i] = applySunLaplacian(lat, q[i]);
    cd H[3][3];
    for (int i = 0; i < m; ++i)
      for (int j = 0; j < m; ++j) H[i][j] = cdot(q[i], Aq[j]);

    double lam = 0.0;
    cd cc[3];
    smallestHermitianEig(m, H, lam, cc);

    std::vector<cd> xn(n, cd(0.0, 0.0));
    for (int i = 0; i < m; ++i)
      for (std::size_t t = 0; t < n; ++t) xn[t] += cc[i] * q[i][t];
    const double nn = norm2(xn);
    if (nn < 1e-300) break;
    scale(xn, cd(1.0 / nn, 0.0));
    xprev = std::move(x);
    x = std::move(xn);
    res.vector = x;
  }

  const std::vector<cd> Ex = applySunLaplacian(lat, x);
  const double rho = cdot(x, Ex).real();
  std::vector<cd> r(n);
  for (std::size_t i = 0; i < n; ++i) r[i] = Ex[i] - rho * x[i];
  res.eigenvalue = rho;
  // Same certificate denominator as the in-loop break test above: without the
  // certFloor the loop can break believing it converged while this final
  // assignment inflates the residual by certFloor/|rho| and reports
  // converged=false -- exactly the mass2=0 near-pure-gauge case (lambda_min
  // below the floor) that this port advertises as its correctness anchor.
  res.residual = norm2(r) / std::max({std::abs(rho), certFloor, 1e-300});
  res.vector = x;
  res.converged = res.residual < opts.tol;
  return res;
}

void SunLattice::setEdgeWeights(std::vector<double> wx_, std::vector<double> wy_,
                                std::vector<double> wz_) {
  if (wx_.size() != static_cast<std::size_t>(numLinksX()) ||
      wy_.size() != static_cast<std::size_t>(numLinksY()) ||
      wz_.size() != static_cast<std::size_t>(numLinksZ()))
    throw std::invalid_argument("SunLattice::setEdgeWeights: weight sizes must match link counts");
  for (const auto* v : {&wx_, &wy_, &wz_})
    for (double e : *v)
      if (!(e > 0.0)) throw std::invalid_argument("SunLattice::setEdgeWeights: weights must be > 0");
  wx = std::move(wx_);
  wy = std::move(wy_);
  wz = std::move(wz_);
}

SunLattice randomSunLattice(int d, int lx, int ly, int lz, double w, double mass2, std::uint64_t seed,
                            bool periodic) {
  if (d < 1 || d > 3) throw std::invalid_argument("randomSunLattice: d must be 1..3");
  if (periodic && (lx % 2 || ly % 2 || lz % 2))
    throw std::invalid_argument("randomSunLattice: every periodic dimension must be even");
  SunLattice L;
  L.d = d;
  L.lx = lx;
  L.ly = ly;
  L.lz = lz;
  L.periodic = periodic;
  L.w = w;
  L.mass2 = mass2;
  const int dd = d * d;
  L.ux.resize(static_cast<std::size_t>(L.numLinksX()) * dd);
  L.uy.resize(static_cast<std::size_t>(L.numLinksY()) * dd);
  L.uz.resize(static_cast<std::size_t>(L.numLinksZ()) * dd);
  std::mt19937_64 rng(seed);
  std::normal_distribution<double> g(0.0, 1.0);
  std::uniform_real_distribution<double> u(-M_PI, M_PI);
  for (std::size_t e = 0; e < L.ux.size() / dd; ++e) randomSU(d, rng, g, u, &L.ux[e * dd]);
  for (std::size_t e = 0; e < L.uy.size() / dd; ++e) randomSU(d, rng, g, u, &L.uy[e * dd]);
  for (std::size_t e = 0; e < L.uz.size() / dd; ++e) randomSU(d, rng, g, u, &L.uz[e * dd]);
  return L;
}

SunLattice identitySunLattice(int d, int lx, int ly, int lz, double w, double mass2, bool periodic) {
  if (d < 1 || d > 3) throw std::invalid_argument("identitySunLattice: d must be 1..3");
  // Same invariant as randomSunLattice/gaugeLatticePeriodic: an odd periodic
  // dim wraps two same-parity nodes onto one red-black color -> an OpenMP data
  // race in smooth and an invalid Gauss-Seidel bipartition.
  if (periodic && (lx % 2 || ly % 2 || lz % 2))
    throw std::invalid_argument("identitySunLattice: every periodic dimension must be even");
  SunLattice L;
  L.d = d;
  L.lx = lx;
  L.ly = ly;
  L.lz = lz;
  L.periodic = periodic;
  L.w = w;
  L.mass2 = mass2;
  const int dd = d * d;
  auto fillI = [d, dd](std::vector<cd>& u, long links) {
    u.assign(static_cast<std::size_t>(links) * dd, cd(0.0, 0.0));
    for (long e = 0; e < links; ++e)
      for (int a = 0; a < d; ++a) u[static_cast<std::size_t>(e) * dd + a * d + a] = cd(1.0, 0.0);
  };
  fillI(L.ux, L.numLinksX());
  fillI(L.uy, L.numLinksY());
  fillI(L.uz, L.numLinksZ());
  return L;
}

BlockEigResult lowestEigenpairsSunMG(const SunLattice& lat, int m,
                                     const std::vector<std::vector<cd>>* guess,
                                     const GaugeEigenOptions& opts) {
  MgOptions pmg = opts.mg;
  pmg.tol = 0.0;
  pmg.maxCycles = opts.precCycles;
  const std::vector<SunLattice> levels = buildSunLevels(lat);
  BlockEigOptions bo;
  bo.maxIters = opts.maxIters;
  bo.tol = opts.tol;
  bo.relativeDrop = opts.relativeGsDrop;
  bo.wanted = m;
  bo.certFloor = opts.certFloorRel * diagScale(lat);
  bo.lockConverged = opts.blockLockConverged;
  bo.softLockConverged = opts.blockSoftLockConverged;
  const int mb = m + std::max(0, opts.blockGuard);
  const auto apply = [&](const std::vector<cd>& x) { return applySunLaplacian(lat, x); };
  const auto prec = [&](const std::vector<cd>& r) {
    std::vector<cd> w(r.size(), cd(0.0, 0.0));
    vcycleSolveSun(levels, r, w, pmg);
    return w;
  };
  BlockEigResult r = blockLobpcg(static_cast<std::size_t>(lat.dof()), mb, apply, prec, guess, bo);
  // Mirror of the U(1) guard in lowestEigenpairsGaugeMG. blockLobpcg signals
  // "could not build a rank-mb block" (dof < mb, or guesses that collapse the
  // block) with EMPTY arrays and maxResidual = 1e300; the resize below only
  // shrinks on the success path, but would GROW those empty arrays with zeros,
  // making the max over m zeros 0 < tol and destroying the sentinel -- a hard
  // failure returned as converged=true with m zero-length eigenvectors, which
  // any caller indexing r.vectors[j][i] reads out of bounds. Reachable e.g. via
  // a 4^3 d=3 lattice (dof=192) with m+blockGuard > 192.
  if (static_cast<int>(r.vectors.size()) < m || static_cast<int>(r.eigenvalues.size()) < m ||
      static_cast<int>(r.residuals.size()) < m) {
    r.converged = false;
    return r;
  }
  r.eigenvalues.resize(m);
  r.residuals.resize(m);
  r.vectors.resize(m);
  r.maxResidual = 0.0;
  for (double d : r.residuals) r.maxResidual = std::max(r.maxResidual, d);
  r.converged = r.maxResidual < opts.tol;
  return r;
}

}  // namespace bochner
