#include "solvers/BlockLobpcg.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <random>
#include <utility>
#include <cstdio>
#include <cstdlib>

namespace bochner {
namespace {

using cd = std::complex<double>;

cd cdot(const std::vector<cd>& a, const std::vector<cd>& b) {
  cd s(0.0, 0.0);
  for (std::size_t i = 0; i < a.size(); ++i) s += std::conj(a[i]) * b[i];
  return s;
}

double norm2(const std::vector<cd>& v) { return std::sqrt(cdot(v, v).real()); }

// Full spectrum of a dense q x q complex-Hermitian matrix H (row-major) by
// cyclic Hermitian Jacobi rotations -- the general-size version of the fixed
// 3x3 routine in GaugeEigen. Returns eigenvalues ascending with matching
// eigenvector columns in V (row-major, V[i*q+j] = component i of vector j).
// q stays small (<= 3m, m tens), so O(q^3) sweeps are negligible next to the
// n-sized vector work.
void hermitianEig(int q, std::vector<cd>& H, std::vector<double>& evals, std::vector<cd>& V) {
  V.assign(static_cast<std::size_t>(q) * q, cd(0.0, 0.0));
  for (int i = 0; i < q; ++i) V[static_cast<std::size_t>(i) * q + i] = cd(1.0, 0.0);
  const auto at = [q](std::vector<cd>& M, int i, int j) -> cd& {
    return M[static_cast<std::size_t>(i) * q + j];
  };
  for (int sweep = 0; sweep < 80; ++sweep) {
    double off = 0.0, diag = 0.0;
    for (int p = 0; p < q; ++p) {
      diag += std::norm(at(H, p, p));
      for (int r = p + 1; r < q; ++r) off += std::norm(at(H, p, r));
    }
    if (off <= 1e-30 * std::max(diag, 1e-300)) break;
    for (int p = 0; p < q; ++p)
      for (int r = p + 1; r < q; ++r) {
        const cd hpr = at(H, p, r);
        if (std::abs(hpr) < 1e-18) continue;
        const double a = at(H, p, p).real(), b = at(H, r, r).real(), mag = std::abs(hpr);
        const cd g = std::exp(cd(0.0, -std::arg(hpr)));  // makes the off-diag real
        const double theta = 0.5 * std::atan2(2.0 * mag, a - b);
        const double c = std::cos(theta), s = std::sin(theta);
        const cd U00(c, 0.0), U01(-s, 0.0), U10 = g * s, U11 = g * c;
        const cd c00 = std::conj(U00), c01 = std::conj(U01), c10 = std::conj(U10),
                 c11 = std::conj(U11);
        for (int i = 0; i < q; ++i) {  // H <- H U (columns p,r)
          const cd hip = at(H, i, p), hir = at(H, i, r);
          at(H, i, p) = hip * U00 + hir * U10;
          at(H, i, r) = hip * U01 + hir * U11;
        }
        for (int j = 0; j < q; ++j) {  // H <- U^H H (rows p,r)
          const cd hpj = at(H, p, j), hrj = at(H, r, j);
          at(H, p, j) = c00 * hpj + c10 * hrj;
          at(H, r, j) = c01 * hpj + c11 * hrj;
        }
        for (int i = 0; i < q; ++i) {  // V <- V U
          const cd vip = at(V, i, p), vir = at(V, i, r);
          at(V, i, p) = vip * U00 + vir * U10;
          at(V, i, r) = vip * U01 + vir * U11;
        }
      }
  }
  // Sort ascending by diagonal, permuting V's columns along.
  std::vector<int> perm(q);
  std::iota(perm.begin(), perm.end(), 0);
  std::sort(perm.begin(), perm.end(), [&](int i, int j) {
    return H[static_cast<std::size_t>(i) * q + i].real() <
           H[static_cast<std::size_t>(j) * q + j].real();
  });
  evals.resize(q);
  std::vector<cd> Vs(V.size());
  for (int j = 0; j < q; ++j) {
    evals[j] = H[static_cast<std::size_t>(perm[j]) * q + perm[j]].real();
    for (int i = 0; i < q; ++i)
      Vs[static_cast<std::size_t>(i) * q + j] = V[static_cast<std::size_t>(i) * q + perm[j]];
  }
  V = std::move(Vs);
}

}  // namespace

BlockEigResult blockLobpcg(std::size_t n, int m, const BlockApplyFn& apply, const BlockApplyFn& prec,
                         const std::vector<std::vector<cd>>* guess, const BlockEigOptions& opts) {
  BlockEigResult res;
  if (m < 1 || n == 0) return res;
  const int wanted = (opts.wanted > 0 && opts.wanted < m) ? opts.wanted : m;

  // --- initial block: caller columns first, deterministic pseudo-random fill,
  // then modified Gram-Schmidt (a degenerate guess column is simply dropped
  // and replaced by the fill).
  std::vector<std::vector<cd>> X;
  const auto mgsAdd = [&](std::vector<cd> v, std::vector<std::vector<cd>>& basis,
                          double drop) -> bool {
    const double nv0 = norm2(v);
    for (const auto& b : basis) {
      const cd proj = cdot(b, v);
      for (std::size_t t = 0; t < n; ++t) v[t] -= proj * b[t];
    }
    const double nv = norm2(v);
    if (!(nv > (opts.relativeDrop ? drop * nv0 : drop)) || !std::isfinite(nv)) return false;
    for (auto& z : v) z *= cd(1.0 / nv, 0.0);
    basis.push_back(std::move(v));
    return true;
  };
  if (guess)
    for (const auto& g : *guess) {
      if (static_cast<int>(X.size()) == m) break;
      if (g.size() == n) mgsAdd(g, X, opts.dropTol);
    }
  {
    // Deterministic PRNG fill (a fixed seed, so runs are reproducible). Trig
    // formulas are NOT usable here: cos(a i + phi_j) spans a fixed 2-dim space
    // over j, so a whole block of them is rank-deficient and Gram-Schmidt
    // could never complete. Attempts are capped defensively.
    std::mt19937_64 rng(0x9e3779b97f4a7c15ULL);
    std::normal_distribution<double> g(0.0, 1.0);
    for (int attempt = 0; static_cast<int>(X.size()) < m && attempt < 8 * m + 64; ++attempt) {
      std::vector<cd> v(n);
      for (std::size_t i = 0; i < n; ++i) v[i] = cd(g(rng), g(rng));
      mgsAdd(std::move(v), X, opts.dropTol);
    }
    if (static_cast<int>(X.size()) < m) {
      // Could not build a rank-m block (n < m, or adversarial guesses): return
      // what exists, honestly unconverged.
      res.maxResidual = 1e300;
      return res;
    }
  }

  std::vector<std::vector<cd>> Xprev;  // conjugate directions (empty on step 1)
  std::vector<double> rho(m, 0.0);
  // BLOCK_DIAG=1: per-iteration convergence structure on stderr (which pair
  // holds the max residual, how many are uncertified, how many trial
  // directions the Gram-Schmidt drop test removed) -- the discriminating
  // observables for slow-pair vs basis-thinning diagnoses. No effect on the
  // iteration itself.
  const bool diag = std::getenv("BLOCK_DIAG") != nullptr;

  // Hard-locked (frozen) certified pairs. Later iterates are kept orthogonal
  // to them (they head the Gram-Schmidt basis) but they are excluded from the
  // Rayleigh-Ritz step, so their vectors -- and hence their certified
  // residuals -- can never change again.
  std::vector<std::vector<cd>> locked;
  std::vector<double> lockedRho;

  // Soft locking (see BlockEigOptions): pairs currently below tol contribute
  // no new search directions; the mask (`certify`) is re-evaluated every
  // iteration, so the stopping rule and exit certificate match the default.
  const bool softLock = opts.softLockConverged && !opts.lockConverged;

  for (int it = 1; it <= opts.maxIters; ++it) {
    res.iterations = it;
    const int mA = static_cast<int>(X.size());                 // active block
    const int wantedA = wanted - static_cast<int>(locked.size());  // still open

    // Rayleigh quotients + residuals of the active block.
    std::vector<std::vector<cd>> AX(mA);
    double maxRes = 0.0;
    int worstPair = -1;
    int uncertified = 0;
    std::vector<std::vector<cd>> R(mA);
    std::vector<char> certify(mA, 0);
    rho.assign(mA, 0.0);
    for (int j = 0; j < mA; ++j) {
      AX[j] = apply(X[j]);
      rho[j] = cdot(X[j], AX[j]).real();  // X orthonormal
      R[j].resize(n);
      for (std::size_t t = 0; t < n; ++t) R[j][t] = AX[j][t] - rho[j] * X[j][t];
      // Convergence is judged on the still-open wanted pairs only; the rest
      // of the block are guards (X is Ritz-sorted ascending after step 1).
      if (j < wantedA) {
        const double rj =
            norm2(R[j]) / std::max({std::abs(rho[j]), opts.certFloor, 1e-300});
        if (it > 1 && rj < opts.tol) certify[j] = 1;
        if (rj >= opts.tol) ++uncertified;
        if (rj > maxRes) {
          maxRes = rj;
          worstPair = j;
        }
      }
    }
    res.maxResidual = maxRes;
    if (diag)
      std::fprintf(stderr,
                   "[block-diag] it %3d  maxRes %.3e @pair %d  uncertified %d/%d  locked %d"
                   "  soft %d\n",
                   it, maxRes, worstPair, uncertified, wantedA, static_cast<int>(locked.size()),
                   softLock ? static_cast<int>(std::count(certify.begin(), certify.end(), 1)) : 0);
    if (diag && mA > wantedA) {
      // Guard telemetry (diagnosis only): the top wanted pair's effective
      // shield is the guards' CURRENT Ritz accuracy, not the exact spectrum
      // above the band -- print their Rayleigh quotients and residuals.
      std::fprintf(stderr, "[block-diag]         guards:");
      for (int j = wantedA; j < mA; ++j) {
        const double rj =
            norm2(R[j]) / std::max({std::abs(rho[j]), opts.certFloor, 1e-300});
        std::fprintf(stderr, "  rho %.6f res %.2e", rho[j], rj);
      }
      std::fprintf(stderr, "\n");
    }
    if (it > 1 && maxRes < opts.tol) {
      // Every still-open wanted pair certified this iteration: done. (In
      // locking mode the certified pairs join `locked` below only if the
      // block continues; here the active copies are already final.)
      break;
    }
    if (opts.lockConverged) {
      // Freeze certified pairs: move them out of the active block. Guards and
      // uncertified pairs stay; the active block shrinks.
      std::vector<std::vector<cd>> Xa;
      std::vector<std::vector<cd>> Ra;
      std::vector<double> rhoA;
      for (int j = 0; j < mA; ++j) {
        if (j < wantedA && certify[j]) {
          locked.push_back(std::move(X[j]));
          lockedRho.push_back(rho[j]);
        } else {
          Xa.push_back(std::move(X[j]));
          Ra.push_back(std::move(R[j]));
          rhoA.push_back(rho[j]);
        }
      }
      X = std::move(Xa);
      R = std::move(Ra);
      rho = std::move(rhoA);
      if (static_cast<int>(locked.size()) == wanted) break;  // all wanted frozen
    }
    const int mAct = static_cast<int>(X.size());

    // Trial basis Z = MGS[locked, X, M R, X_prev]; the locked prefix is an
    // orthogonality constraint only and takes no further part. EVERY column
    // goes through modified Gram-Schmidt, including X: treating X as exactly
    // orthonormal lets its roundoff-level orthogonality error feed back
    // through the Rayleigh-Ritz step (whose small eigenproblem assumes
    // Z^H Z = I) and grow geometrically -- the classic LOBPCG basis-collapse
    // instability, observed here as a blow-up right after the residuals get
    // small. The full MGS costs the m reusable A-applies (AX cannot be
    // recycled) and removes the instability entirely.
    std::vector<std::vector<cd>> Z;
    Z.reserve(static_cast<std::size_t>(locked.size()) + 3 * static_cast<std::size_t>(mAct));
    for (const auto& lv : locked) mgsAdd(lv, Z, opts.dropTol);
    const int L0 = static_cast<int>(Z.size());
    for (int j = 0; j < mAct; ++j) mgsAdd(X[j], Z, opts.dropTol);
    int softSkipped = 0;  // currently-certified pairs contribute no new search directions
    for (int j = 0; j < mAct; ++j) {
      if (softLock && certify[j]) {
        ++softSkipped;
        continue;
      }
      mgsAdd(prec(R[j]), Z, opts.dropTol);
    }
    for (std::size_t j = 0; j < Xprev.size(); ++j) {
      if (softLock && j < certify.size() && certify[j]) {
        ++softSkipped;
        continue;
      }
      mgsAdd(Xprev[j], Z, opts.dropTol);
    }
    const int q = static_cast<int>(Z.size());
    if (diag) {
      const int attempted = L0 + 2 * mAct + static_cast<int>(Xprev.size()) - softSkipped;
      std::fprintf(stderr, "[block-diag]         basis %d of %d (%d dropped)\n", q,
                   attempted, attempted - q);
    }
    const int qA = q - L0;  // active-subspace dimension for the Rayleigh-Ritz
    if (qA <= mAct) break;  // no new directions: stagnation (reported honestly)

    // Rayleigh-Ritz on the active columns Z[L0..q) (orthogonal to `locked`).
    std::vector<std::vector<cd>> AZ(qA);
    for (int j = 0; j < qA; ++j) AZ[j] = apply(Z[L0 + j]);
    std::vector<cd> H(static_cast<std::size_t>(qA) * qA);
    for (int i = 0; i < qA; ++i)
      for (int j = 0; j < qA; ++j)
        H[static_cast<std::size_t>(i) * qA + j] = cdot(Z[L0 + i], AZ[j]);
    std::vector<double> evals;
    std::vector<cd> C;
    hermitianEig(qA, H, evals, C);

    // New active block = lowest mAct Ritz vectors; the previous active block
    // becomes the conjugate directions.
    std::vector<std::vector<cd>> Xnew(mAct);
    for (int j = 0; j < mAct; ++j) {
      Xnew[j].assign(n, cd(0.0, 0.0));
      for (int k = 0; k < qA; ++k) {
        const cd c = C[static_cast<std::size_t>(k) * qA + j];
        if (c == cd(0.0, 0.0)) continue;
        const auto& zk = Z[L0 + k];
        for (std::size_t t = 0; t < n; ++t) Xnew[j][t] += c * zk[t];
      }
      const double nx = norm2(Xnew[j]);
      if (nx > 0.0)
        for (auto& z : Xnew[j]) z *= cd(1.0 / nx, 0.0);
    }
    rho.assign(evals.begin(), evals.begin() + mAct);  // keys stay current for
                                                      // the exit reassembly
    Xprev = std::move(X);
    X = std::move(Xnew);
  }

  // Reassemble the returned block: certified frozen pairs plus the remaining
  // active ones. Wanted pairs are merged in ascending Rayleigh-quotient order
  // among THEMSELVES; guards keep their Ritz order strictly BEHIND the wanted
  // set (a raw global sort could let an unconverged guard tie into the wanted
  // prefix inside a degenerate cluster -- the failure mode the exit comment
  // below records).
  if (!locked.empty()) {
    std::vector<std::pair<double, std::vector<cd>>> wantedPairs;
    wantedPairs.reserve(static_cast<std::size_t>(wanted));
    for (std::size_t j = 0; j < locked.size(); ++j)
      wantedPairs.emplace_back(lockedRho[j], std::move(locked[j]));
    const int wantedA = wanted - static_cast<int>(lockedRho.size());
    for (int j = 0; j < wantedA && j < static_cast<int>(X.size()); ++j)
      wantedPairs.emplace_back(rho[j], std::move(X[j]));
    std::stable_sort(wantedPairs.begin(), wantedPairs.end(),
                     [](const auto& a, const auto& b) { return a.first < b.first; });
    std::vector<std::vector<cd>> Xall;
    Xall.reserve(static_cast<std::size_t>(m));
    for (auto& p : wantedPairs) Xall.push_back(std::move(p.second));
    for (int j = wantedA; j < static_cast<int>(X.size()); ++j)
      Xall.push_back(std::move(X[j]));  // guards, in Ritz order, behind wanted
    X = std::move(Xall);
  }

  // Report the eigenpairs OF THE RETURNED BLOCK, recomputed independently of
  // the loop bookkeeping (mirrors the single-vector solvers). The certificate
  // (`maxResidual`/`converged`) covers the `wanted` lowest pairs only -- the
  // same rule the loop stops on; guard vectors are deliberately unconverged, so
  // folding them in would report failure for a successful solve. Per-pair
  // residuals stay visible for the whole block, guards included.
  res.eigenvalues.resize(m);
  res.residuals.resize(m);
  res.vectors = X;
  res.maxResidual = 0.0;
  for (int j = 0; j < m; ++j) {
    const std::vector<cd> Ax = apply(X[j]);
    const double r = cdot(X[j], Ax).real();
    double num = 0.0;
    for (std::size_t t = 0; t < n; ++t) num += std::norm(Ax[t] - r * X[j][t]);
    res.eigenvalues[j] = r;
    res.residuals[j] = std::sqrt(num) / std::max({std::abs(r), opts.certFloor, 1e-300});
    if (j < wanted) res.maxResidual = std::max(res.maxResidual, res.residuals[j]);
  }
  // Keep RITZ order (ascending by construction): a final re-sort on the
  // recomputed Rayleigh quotients is unstable inside degenerate clusters,
  // where an UNCONVERGED guard vector ties with a converged wanted vector at
  // roundoff level and can leak ahead of it -- observed as spurious
  // unconverged flags and inflated residuals on Landau/Kramers multiplets.
  // The Ritz step already ordered the block; the recompute above is only the
  // honest certificate of each returned pair.
  res.converged = res.maxResidual < opts.tol;
  return res;
}

}  // namespace bochner
