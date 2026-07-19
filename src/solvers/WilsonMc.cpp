#include "solvers/WilsonMc.h"

#include <cmath>
#include <stdexcept>

namespace bochner {
namespace {

using cd = std::complex<double>;

// ---- small dense d x d complex helpers (row-major, d <= 3) ----

// out = A B
void mm(const cd* A, const cd* B, int d, cd* out) {
  for (int i = 0; i < d; ++i)
    for (int j = 0; j < d; ++j) {
      cd s(0.0, 0.0);
      for (int k = 0; k < d; ++k) s += A[i * d + k] * B[k * d + j];
      out[i * d + j] = s;
    }
}
// out = A B^H
void mmAdjB(const cd* A, const cd* B, int d, cd* out) {
  for (int i = 0; i < d; ++i)
    for (int j = 0; j < d; ++j) {
      cd s(0.0, 0.0);
      for (int k = 0; k < d; ++k) s += A[i * d + k] * std::conj(B[j * d + k]);
      out[i * d + j] = s;
    }
}
// out = A^H B
void mmAdjA(const cd* A, const cd* B, int d, cd* out) {
  for (int i = 0; i < d; ++i)
    for (int j = 0; j < d; ++j) {
      cd s(0.0, 0.0);
      for (int k = 0; k < d; ++k) s += std::conj(A[k * d + i]) * B[k * d + j];
      out[i * d + j] = s;
    }
}
// Re tr(A B^H) = sum_ij A_ij conj(B_ij)
double reTrAdjB(const cd* A, const cd* B, int d) {
  double s = 0.0;
  for (int i = 0; i < d * d; ++i) s += (A[i] * std::conj(B[i])).real();
  return s;
}

cd detSmall(const cd* M, int d) {
  if (d == 1) return M[0];
  if (d == 2) return M[0] * M[3] - M[1] * M[2];
  return M[0] * (M[4] * M[8] - M[5] * M[7]) - M[1] * (M[3] * M[8] - M[5] * M[6]) +
         M[2] * (M[3] * M[7] - M[4] * M[6]);
}

// Modified Gram-Schmidt on the columns, then rotate the last column by
// conj(det) (unit modulus after MGS) so det = 1: the SU(d) projection used both
// for reunitarization and to finish Haar sampling.
void projectSU(cd* M, int d) {
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
  const cd phi = std::conj(det) / std::abs(det);
  for (int r = 0; r < d; ++r) M[r * d + d - 1] *= phi;
}

// ---- SU(2) heatbath: sample a0 with density ~ sqrt(1-a0^2) exp(y a0) ----
//
// y > ~1: Kennedy-Pendleton.  Substituting a0 = 1 - delta, the target factors
// as [sqrt(delta) e^{-y delta}] * sqrt(2 - delta): sample delta ~ Gamma(3/2, y)
// exactly (Exp(1)/y + cos^2(2 pi r) Exp(1)/y), accept with prob
// sqrt(1 - delta/2)  (i.e. r4^2 <= 1 - delta/2).
// y <= ~1: plain rejection from the semicircle, accept prob
// sqrt(1-x^2) e^{y(x-1)} <= 1  (KP's acceptance collapses as y -> 0; this
// branch stays O(1) there, and y = 0 gives exact Haar).
double sampleA0(double y, std::mt19937_64& rng) {
  std::uniform_real_distribution<double> u01(0.0, 1.0);
  if (y > 1.0) {
    for (int t = 0; t < 100000; ++t) {
      const double r1 = 1.0 - u01(rng);  // (0,1]
      const double r2 = u01(rng);
      const double r3 = 1.0 - u01(rng);
      const double c = std::cos(2.0 * M_PI * r2);
      const double delta = -(std::log(r1) + c * c * std::log(r3)) / y;
      const double r4 = u01(rng);
      if (r4 * r4 <= 1.0 - 0.5 * delta) return 1.0 - delta;
    }
    return 1.0;  // unreachable in practice (acceptance is O(1) for y > 1)
  }
  for (int t = 0; t < 100000; ++t) {
    const double x = 2.0 * u01(rng) - 1.0;
    const double acc = std::sqrt(std::max(0.0, 1.0 - x * x)) * std::exp(y * (x - 1.0));
    if (u01(rng) <= acc) return x;
  }
  return 0.0;  // unreachable in practice (acceptance >= ~0.17 for y <= 1)
}

// Sample u in SU(2) (as (u00,u01); u10 = -conj(u01), u11 = conj(u00)) with
// density ~ exp((beta/d) k Re tr(u Vhat)), Vhat in SU(2), i.e. g = u Vhat with
// density ~ exp(y g0), y = 2 beta k / d, then u = g Vhat^H.
void su2Heatbath(double y, const cd& vhat00, const cd& vhat01, std::mt19937_64& rng, cd& u00,
                 cd& u01) {
  std::uniform_real_distribution<double> u01d(0.0, 1.0);
  const double g0 = sampleA0(y, rng);
  const double gn = std::sqrt(std::max(0.0, 1.0 - g0 * g0));
  const double ct = 2.0 * u01d(rng) - 1.0;  // uniform direction on S^2
  const double st = std::sqrt(std::max(0.0, 1.0 - ct * ct));
  const double ph = 2.0 * M_PI * u01d(rng);
  const double g1 = gn * st * std::cos(ph), g2 = gn * st * std::sin(ph), g3 = gn * ct;
  // g = g0 I + i (g1 sx + g2 sy + g3 sz)  ->  2x2:
  const cd G00(g0, g3), G01(g2, g1);  // G10 = -conj(G01), G11 = conj(G00)
  // u = g Vhat^H with Vhat = [[a,b],[-conj(b),conj(a)]] (unit "quaternion"):
  // Vhat^H = [[conj(a), -b],[conj(b), a]].
  u00 = G00 * std::conj(vhat00) + G01 * std::conj(vhat01);
  u01 = -G00 * vhat01 + G01 * vhat00;
}

// Wrapped +/-1 neighbour index.
inline int wrapIdx(int i, int n) { return (i + n) % n; }

}  // namespace

void haarSuMatrix(int d, std::mt19937_64& rng, cd* M) {
  if (d < 1 || d > 3) throw std::invalid_argument("haarSuMatrix: d must be 1..3");
  std::normal_distribution<double> g(0.0, 1.0);
  if (d == 1) {
    std::uniform_real_distribution<double> u(-M_PI, M_PI);
    const double a = u(rng);
    M[0] = cd(std::cos(a), std::sin(a));
    return;
  }
  for (int i = 0; i < d * d; ++i) M[i] = cd(g(rng), g(rng));
  projectSU(M, d);
}

void reunitarizeSun(SunLattice& lat) {
  const int d = lat.d, dd = d * d;
  for (auto* u : {&lat.ux, &lat.uy, &lat.uz})
    for (std::size_t e = 0; e < u->size() / dd; ++e) projectSU(&(*u)[e * dd], d);
}

double averagePlaquette(const SunLattice& lat) {
  if (!lat.periodic) throw std::invalid_argument("averagePlaquette: periodic lattice required");
  const int d = lat.d, dd = d * d;
  const int nx = lat.lx, ny = lat.ly, nz = lat.lz;
  const std::vector<cd>* U[3] = {&lat.ux, &lat.uy, &lat.uz};
  auto link = [&](int axis, int i, int j, int k) {
    return &(*U[axis])[static_cast<std::size_t>(lat.index(i, j, k)) * dd];
  };
  double sum = 0.0;
  long np = 0;
  cd T1[9], T2[9];
  for (int i = 0; i < nx; ++i)
    for (int j = 0; j < ny; ++j)
      for (int k = 0; k < nz; ++k)
        for (int mu = 0; mu < 3; ++mu)
          for (int nu = mu + 1; nu < 3; ++nu) {
            const int ip = wrapIdx(i + (mu == 0), nx), jp = wrapIdx(j + (mu == 1), ny),
                      kp = wrapIdx(k + (mu == 2), nz);
            const int iq = wrapIdx(i + (nu == 0), nx), jq = wrapIdx(j + (nu == 1), ny),
                      kq = wrapIdx(k + (nu == 2), nz);
            // Two transport routes x -> x+mu+nu (links here compose by LEFT
            // multiplication: U stored at c transports c -> c+mu and gauge-
            // transforms as U -> g_head U g_tail^H, matching SunGauge):
            //   T1 = U_nu(x+mu) U_mu(x),  T2 = U_mu(x+nu) U_nu(x);
            // Re tr(T2^H T1) is the gauge-invariant plaquette trace.
            mm(link(nu, ip, jp, kp), link(mu, i, j, k), d, T1);
            mm(link(mu, iq, jq, kq), link(nu, i, j, k), d, T2);
            sum += reTrAdjB(T1, T2, d);
            ++np;
          }
  return sum / (static_cast<double>(np) * d);
}

void wilsonHeatbathSweep(SunLattice& lat, double beta, std::mt19937_64& rng) {
  if (!lat.periodic) throw std::invalid_argument("wilsonHeatbathSweep: periodic lattice required");
  if (lat.d < 2 || lat.d > 3)
    throw std::invalid_argument("wilsonHeatbathSweep: d must be 2 or 3");
  const int d = lat.d, dd = d * d;
  const int nx = lat.lx, ny = lat.ly, nz = lat.lz;
  std::vector<cd>* U[3] = {&lat.ux, &lat.uy, &lat.uz};
  auto link = [&](int axis, int i, int j, int k) {
    return &(*U[axis])[static_cast<std::size_t>(lat.index(i, j, k)) * dd];
  };
  const int npairs = (d == 2) ? 1 : 3;
  const int pairs[3][2] = {{0, 1}, {0, 2}, {1, 2}};
  cd A[9], T[9], W[9];
  for (int mu = 0; mu < 3; ++mu)
    for (int i = 0; i < nx; ++i)
      for (int j = 0; j < ny; ++j)
        for (int k = 0; k < nz; ++k) {
          // Sum of the 4 staples of U_mu(x) in 3D (codebase convention: links
          // transport tail -> head and compose by LEFT multiplication, so the
          // gauge-invariant plaquette at base x is Re tr[(U_mu(x+nu)U_nu(x))^H
          // U_nu(x+mu) U_mu(x)]): for each transverse nu,
          //   forward  (U_mu(x+nu) U_nu(x))^H U_nu(x+mu)
          //   backward U_nu(x-nu) U_mu(x-nu)^H U_nu(x+mu-nu)^H
          // so that sum_{p ni U} Re tr U_p = Re tr(U_mu(x) A).
          for (int t = 0; t < dd; ++t) A[t] = cd(0.0, 0.0);
          const int ip = wrapIdx(i + (mu == 0), nx), jp = wrapIdx(j + (mu == 1), ny),
                    kp = wrapIdx(k + (mu == 2), nz);
          for (int nu = 0; nu < 3; ++nu) {
            if (nu == mu) continue;
            const int di = (nu == 0), dj = (nu == 1), dk = (nu == 2);
            const int iq = wrapIdx(i + di, nx), jq = wrapIdx(j + dj, ny), kq = wrapIdx(k + dk, nz);
            const int im = wrapIdx(i - di, nx), jm = wrapIdx(j - dj, ny), km = wrapIdx(k - dk, nz);
            const int ipm = wrapIdx(ip - di, nx), jpm = wrapIdx(jp - dj, ny),
                      kpm = wrapIdx(kp - dk, nz);
            cd S[9];
            // forward: T = U_mu(x+nu) U_nu(x) ; S = T^H U_nu(x+mu)
            mm(link(mu, iq, jq, kq), link(nu, i, j, k), d, T);
            mmAdjA(T, link(nu, ip, jp, kp), d, S);
            for (int t = 0; t < dd; ++t) A[t] += S[t];
            // backward: T = U_nu(x-nu) U_mu(x-nu)^H ; S = T U_nu(x+mu-nu)^H
            mmAdjB(link(nu, im, jm, km), link(mu, im, jm, km), d, T);
            mmAdjB(T, link(nu, ipm, jpm, kpm), d, S);
            for (int t = 0; t < dd; ++t) A[t] += S[t];
          }
          cd* Ul = link(mu, i, j, k);
          for (int s = 0; s < npairs; ++s) {
            const int p = pairs[s][0], q = pairs[s][1];
            // Local weight exp((beta/d) Re tr(U' A)), U' = u_emb U: only the
            // quaternionic projection of the (p,q) block of W = U A couples.
            mm(Ul, A, d, W);
            const cd wa = 0.5 * (W[p * d + p] + std::conj(W[q * d + q]));
            const cd wb = 0.5 * (W[p * d + q] - std::conj(W[q * d + p]));
            const double kn = std::sqrt(std::norm(wa) + std::norm(wb));
            cd u00, u01;
            if (kn < 1e-14) {
              su2Heatbath(0.0, cd(1.0, 0.0), cd(0.0, 0.0), rng, u00, u01);
            } else {
              su2Heatbath(2.0 * beta * kn / d, wa / kn, wb / kn, rng, u00, u01);
            }
            // U <- u_emb U (left-multiply rows p and q).
            for (int c = 0; c < d; ++c) {
              const cd rp = Ul[p * d + c], rq = Ul[q * d + c];
              Ul[p * d + c] = u00 * rp + u01 * rq;
              Ul[q * d + c] = -std::conj(u01) * rp + std::conj(u00) * rq;
            }
          }
        }
}

void gaugeTransformSun(SunLattice& lat, std::uint64_t seed) {
  if (!lat.periodic) throw std::invalid_argument("gaugeTransformSun: periodic lattice required");
  const int d = lat.d, dd = d * d;
  const long N = lat.numNodes();
  std::mt19937_64 rng(seed);
  std::vector<cd> g(static_cast<std::size_t>(N) * dd);
  for (long c = 0; c < N; ++c) haarSuMatrix(d, rng, &g[static_cast<std::size_t>(c) * dd]);
  const int nx = lat.lx, ny = lat.ly, nz = lat.lz;
  std::vector<cd>* U[3] = {&lat.ux, &lat.uy, &lat.uz};
  cd T[9], V[9];
  for (int i = 0; i < nx; ++i)
    for (int j = 0; j < ny; ++j)
      for (int k = 0; k < nz; ++k) {
        const long c = lat.index(i, j, k);
        for (int mu = 0; mu < 3; ++mu) {
          const int ip = wrapIdx(i + (mu == 0), nx), jp = wrapIdx(j + (mu == 1), ny),
                    kp = wrapIdx(k + (mu == 2), nz);
          const long h = lat.index(ip, jp, kp);
          cd* Ul = &(*U[mu])[static_cast<std::size_t>(c) * dd];
          // U' = g_head U g_tail^H
          mm(&g[static_cast<std::size_t>(h) * dd], Ul, d, T);
          mmAdjB(T, &g[static_cast<std::size_t>(c) * dd], d, V);
          for (int t = 0; t < dd; ++t) Ul[t] = V[t];
        }
      }
}

SunLattice mcSunLattice(int d, int n, double beta, int sweeps, double w, double mass2,
                        std::uint64_t seed, bool hotStart, std::vector<double>* plaqHistory,
                        int reunitEvery) {
  SunLattice L = hotStart ? randomSunLattice(d, n, n, n, w, mass2, seed)
                          : identitySunLattice(d, n, n, n, w, mass2);
  // Decouple the sweep stream from the hot-start stream so beta=0/sweeps=0
  // reproduces randomSunLattice(seed) exactly.
  std::mt19937_64 rng(seed ^ 0x9E3779B97F4A7C15ull);
  for (int s = 1; s <= sweeps; ++s) {
    wilsonHeatbathSweep(L, beta, rng);
    if (reunitEvery > 0 && s % reunitEvery == 0) reunitarizeSun(L);
    if (plaqHistory) plaqHistory->push_back(averagePlaquette(L));
  }
  if (sweeps > 0) reunitarizeSun(L);
  return L;
}

}  // namespace bochner
