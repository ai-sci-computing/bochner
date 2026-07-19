/// \file
/// Chebyshev-filtered Lanczos baseline (the lattice-QCD "distillation" practice
/// solver) for the smallest eigenpair of the connection Laplacian, as the paper
/// reviews requested. Solves the SAME operators as export_operator /
/// torus_eig_compare / sun_gauge_bench, in the real 2N embedding SLEPc gets
/// everywhere else in this repo, with four SLEPc configurations:
///
///   1. EPSKRYLOVSCHUR, defaults              (the repo's production baseline)
///   2. EPSLANCZOS, defaults                  (plain Lanczos)
///   3. EPSKRYLOVSCHUR + STFILTER (Chebyshev) (polynomial-filtered: EPS_ALL on
///   4. EPSKRYLOVSCHUR + STFILTER (FILTLAN)    the interval [0, 1.05*lambda_est])
///
/// The filter needs (a) a rough bound on the wanted eigenvalue -- taken from a
/// loose (tol 1e-2) Krylov-Schur pre-solve, itself reported -- and (b) the
/// operator's numerical range, taken as [0, 1.1*lambda_max] with lambda_max
/// from ~20 power iterations (the operator is PSD). All final solves use tol
/// 1e-7, smallest real. Per method we report outer iterations, MatMult count
/// (PETSc log event -- the filtered solve hides degree-many matvecs per apply),
/// wall ms, and the eigenvalue. Run single-threaded (OMP_NUM_THREADS=1).
///
/// Usage: chebyshev_lanczos_bench u1 <n> <nPhi> | sun <d> <n>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include <slepc.h>

#include "grid/CooMatrix.h"
#include "solvers/GaugeMultigrid.h"
#include "solvers/SunGauge.h"

using namespace bochner;
using cd = std::complex<double>;

namespace {

void check(PetscErrorCode ierr, const char* what) {
  if (ierr != 0)
    throw std::runtime_error(std::string("PETSc/SLEPc error (") + std::to_string(ierr) + ") in " + what);
}

template <class T, PetscErrorCode (*Destroy)(T*)>
struct Owned {
  T obj = nullptr;
  ~Owned() {
    if (obj) Destroy(&obj);
  }
};

template <class F>
double timeMs(F&& f) {
  const auto t0 = std::chrono::steady_clock::now();
  f();
  const auto t1 = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// ---------------------------------------------------------------------------
// Operators (identical constructions to torus_eig_compare / sun_gauge_bench).
// ---------------------------------------------------------------------------

// The lattice-gauge-solvers Examples::uniformField: nPhi flux quanta through the
// x-y torus in the Landau/seam gauge, as bochner forward links.
GaugeLattice uniformFluxLattice(int n, int nPhi, double h) {
  const double phi_p = 2.0 * M_PI * nPhi / (double(n) * n);
  const std::size_t N = static_cast<std::size_t>(n) * n * n;
  std::vector<double> lkx(N, 0.0), lky(N, 0.0), lkz(N, 0.0);
  const auto idx = [&](int i, int j, int k) { return (i * n + j) * n + k; };
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      for (int k = 0; k < n; ++k) {
        lkx[idx(i, j, k)] = -phi_p * j;
        if (j == n - 1) lky[idx(i, j, k)] = 2.0 * M_PI * nPhi * i / double(n);
      }
  return gaugeLatticePeriodic(n, n, n, 1.0 / (h * h), lkx, lky, lkz);
}

// The periodic U(1) operator as a real 2N CooMatrix (same as torus_eig_compare).
CooMatrix assemblePeriodicU1(const GaugeLattice& L) {
  const int n = static_cast<int>(L.numNodes());
  CooMatrix A(2 * n, 2 * n);
  const auto blk = [&](int a, int b, double re, double im) {
    A.add(2 * a, 2 * b, re);
    A.add(2 * a, 2 * b + 1, -im);
    A.add(2 * a + 1, 2 * b, im);
    A.add(2 * a + 1, 2 * b + 1, re);
  };
  const auto lnk = [&](int a, int b, double th) {
    blk(a, a, L.w, 0.0);
    blk(b, b, L.w, 0.0);
    blk(a, b, -L.w * std::cos(th), L.w * std::sin(th));   // -w e^{-i th}
    blk(b, a, -L.w * std::cos(th), -L.w * std::sin(th));  // conj
  };
  const auto id = [&](int i, int j, int k) { return (i * L.ly + j) * L.lz + k; };
  for (int i = 0; i < L.lx; ++i)
    for (int j = 0; j < L.ly; ++j)
      for (int k = 0; k < L.lz; ++k) {
        const int c = id(i, j, k);
        lnk(c, id((i + 1) % L.lx, j, k), L.lkx[c]);
        lnk(c, id(i, (j + 1) % L.ly, k), L.lky[c]);
        lnk(c, id(i, j, (k + 1) % L.lz), L.lkz[c]);
      }
  return A;
}

// Smooth SU(d) field helpers (copied from sun_gauge_bench).
void embedSu2(cd* M, int d, int p, int q, double v0, double v1, double v2) {
  const double a = std::sqrt(v0 * v0 + v1 * v1 + v2 * v2);
  const double c = std::cos(a), s = (a > 1e-12) ? std::sin(a) / a : 1.0;
  const cd m00(c, s * v2), m01(s * v1, s * v0), m10(-s * v1, s * v0), m11(c, -s * v2);
  M[p * d + p] = m00;
  M[p * d + q] = m01;
  M[q * d + p] = m10;
  M[q * d + q] = m11;
}

void setIdentity(cd* M, int d) {
  for (int i = 0; i < d * d; ++i) M[i] = cd(0, 0);
  for (int i = 0; i < d; ++i) M[i * d + i] = cd(1, 0);
}
void matmulInto(const cd* A, const cd* B, int d, cd* out) {
  for (int i = 0; i < d; ++i)
    for (int j = 0; j < d; ++j) {
      cd sum(0, 0);
      for (int k = 0; k < d; ++k) sum += A[i * d + k] * B[k * d + j];
      out[i * d + j] = sum;
    }
}

void smoothLink(int d, int axis, int i, int j, int k, int n, double amp, cd* M) {
  const double h = 1.0 / n;
  auto S = [n](int idx) { return std::sin(2.0 * M_PI * idx / n); };
  double v[3][3];
  v[0][0] = S(j); v[0][1] = S(k); v[0][2] = 0.0;   // block (0,1)
  v[1][0] = S(k); v[1][1] = 0.0;  v[1][2] = S(i);   // block (0,2)
  v[2][0] = 0.0;  v[2][1] = S(i); v[2][2] = S(j);   // block (1,2)
  const double rot = (axis == 0) ? 0.0 : (axis == 1) ? 2.09439510239 : 4.18879020479;  // +/-120deg phase
  for (int b = 0; b < 3; ++b)
    for (int c = 0; c < 3; ++c) v[b][c] = amp * h * (v[b][c] + 0.5 * std::sin(2.0 * M_PI * (i + j + k) / n + rot));
  if (d == 2) {
    setIdentity(M, 2);
    embedSu2(M, 2, 0, 1, v[0][0], v[0][1], v[0][2]);
    return;
  }
  cd R01[9], R02[9], R12[9], T[9];
  setIdentity(R01, 3); embedSu2(R01, 3, 0, 1, v[0][0], v[0][1], v[0][2]);
  setIdentity(R02, 3); embedSu2(R02, 3, 0, 2, v[1][0], v[1][1], v[1][2]);
  setIdentity(R12, 3); embedSu2(R12, 3, 1, 2, v[2][0], v[2][1], v[2][2]);
  matmulInto(R01, R02, 3, T);
  matmulInto(T, R12, 3, M);
}

SunLattice smoothLattice(int d, int n, double mass2, double amp) {
  SunLattice L;
  L.d = d;
  L.lx = L.ly = L.lz = n;
  L.periodic = true;
  L.w = static_cast<double>(n) * n;  // 1/h^2 with h = 1/n (continuum refinement)
  L.mass2 = mass2;
  const int dd = d * d;
  L.ux.resize(static_cast<size_t>(L.numLinksX()) * dd);
  L.uy.resize(static_cast<size_t>(L.numLinksY()) * dd);
  L.uz.resize(static_cast<size_t>(L.numLinksZ()) * dd);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      for (int k = 0; k < n; ++k) {
        const size_t e = static_cast<size_t>((i * n + j) * n + k) * dd;
        smoothLink(d, 0, i, j, k, n, amp, &L.ux[e]);
        smoothLink(d, 1, i, j, k, n, amp, &L.uy[e]);
        smoothLink(d, 2, i, j, k, n, amp, &L.uz[e]);
      }
  return L;
}

// The periodic SU(d) operator as a real 2dN CooMatrix (same as sun_gauge_bench).
CooMatrix assembleRealSun(const SunLattice& L) {
  const int d = L.d, dd = d * d;
  const long N = L.numNodes();
  const int M = static_cast<int>(2 * d * N);
  CooMatrix A(M, M);
  auto cplx = [&](long p, long q, cd v) {  // complex dof (p,q) -> real 2x2 block
    A.add(2 * p, 2 * q, v.real());
    A.add(2 * p, 2 * q + 1, -v.imag());
    A.add(2 * p + 1, 2 * q, v.imag());
    A.add(2 * p + 1, 2 * q + 1, v.real());
  };
  auto block = [&](long c, long n, const cd* U, bool adj) {  // -w * U_{n->c}[a,b]
    for (int a = 0; a < d; ++a)
      for (int b = 0; b < d; ++b) {
        const cd t = adj ? std::conj(U[b * d + a]) : U[a * d + b];
        if (t != cd(0, 0)) cplx(c * d + a, n * d + b, -L.w * t);
      }
  };
  const int lx = L.lx, ly = L.ly, lz = L.lz;
  const double diag = L.w * 6.0 + L.mass2;  // periodic degree = 6
  for (int i = 0; i < lx; ++i)
    for (int j = 0; j < ly; ++j)
      for (int k = 0; k < lz; ++k) {
        const long c = L.index(i, j, k);
        for (int a = 0; a < d; ++a) cplx(c * d + a, c * d + a, cd(diag, 0));
        const int im = (i - 1 + lx) % lx, ip = (i + 1) % lx;
        const int jm = (j - 1 + ly) % ly, jp = (j + 1) % ly;
        const int km = (k - 1 + lz) % lz, kp = (k + 1) % lz;
        block(c, L.index(im, j, k), &L.ux[static_cast<size_t>(L.index(im, j, k)) * dd], false);
        block(c, L.index(ip, j, k), &L.ux[static_cast<size_t>(c) * dd], true);
        block(c, L.index(i, jm, k), &L.uy[static_cast<size_t>(L.index(i, jm, k)) * dd], false);
        block(c, L.index(i, jp, k), &L.uy[static_cast<size_t>(c) * dd], true);
        block(c, L.index(i, j, km), &L.uz[static_cast<size_t>(L.index(i, j, km)) * dd], false);
        block(c, L.index(i, j, kp), &L.uz[static_cast<size_t>(c) * dd], true);
      }
  return A;
}

// ---------------------------------------------------------------------------
// SLEPc plumbing.
// ---------------------------------------------------------------------------

// Sequential AIJ Mat from the COO operator (the EigenSolver.cpp pattern).
void buildMat(const CooMatrix& A, Mat* out) {
  const PetscInt n = A.rows();
  const auto entries = A.compressed();
  std::vector<PetscInt> nnzPerRow(static_cast<std::size_t>(n), 0);
  for (const auto& e : entries) ++nnzPerRow[static_cast<std::size_t>(e.row)];
  check(MatCreateSeqAIJ(PETSC_COMM_SELF, n, n, 0, nnzPerRow.data(), out), "MatCreateSeqAIJ");
  check(MatSetOption(*out, MAT_SYMMETRIC, PETSC_TRUE), "MatSetOption");
  for (const auto& e : entries)
    check(MatSetValue(*out, e.row, e.col, e.value, INSERT_VALUES), "MatSetValue");
  check(MatAssemblyBegin(*out, MAT_FINAL_ASSEMBLY), "MatAssemblyBegin");
  check(MatAssemblyEnd(*out, MAT_FINAL_ASSEMBLY), "MatAssemblyEnd");
}

// Matvec work is reported as FLOP-based matvec EQUIVALENTS, delta(PetscGetFlops)
// / (2 nnz): the polynomial filters apply the operator through *blocked*
// MatMatMult (one call = ncv matvecs), so counting the MatMult event alone
// under-reports them; flops capture both forms (vector ops add a few percent).
double flopsNow() {
  PetscLogDouble f = 0.0;
  PetscGetFlops(&f);
  return static_cast<double>(f);
}

// lambda_max upper estimate: ~20 power iterations (the operator is PSD).
double powerLambdaMax(Mat M, int iters = 20) {
  Owned<Vec, VecDestroy> x, y;
  check(MatCreateVecs(M, &x.obj, &y.obj), "MatCreateVecs power");
  PetscScalar* xa = nullptr;
  PetscInt n = 0;
  check(VecGetLocalSize(x.obj, &n), "VecGetLocalSize");
  check(VecGetArray(x.obj, &xa), "VecGetArray power");
  for (PetscInt i = 0; i < n; ++i) xa[i] = std::cos(0.37 * i) + 0.5;  // deterministic, non-degenerate
  check(VecRestoreArray(x.obj, &xa), "VecRestoreArray power");
  check(VecNormalize(x.obj, nullptr), "VecNormalize power");
  PetscReal lam = 0.0;
  for (int it = 0; it < iters; ++it) {
    check(MatMult(M, x.obj, y.obj), "MatMult power");
    check(VecNorm(y.obj, NORM_2, &lam), "VecNorm power");
    check(VecCopy(y.obj, x.obj), "VecCopy power");
    check(VecNormalize(x.obj, nullptr), "VecNormalize power it");
  }
  return static_cast<double>(lam);
}

struct MethodResult {
  std::string name;
  bool ok = false;
  std::string note;      // failure reason / remark
  int iterations = 0;
  long matvecEq = 0;     // flop-based matvec equivalents (see flopsNow)
  double wallMs = 0.0;
  double eigenvalue = 0.0;
  int nconv = 0;
};

enum class Filter { None, Chebyshev, Filtlan };

// One SLEPc solve for the smallest eigenvalue: EPS `type` at tolerance `tol`;
// with a Filter, Krylov-Schur computes ALL eigenvalues in [0, intervalTop] via
// the STFILTER polynomial filter over the numerical range [0, rangeTop].
MethodResult runEps(Mat M, const char* name, EPSType type, double tol, Filter filter,
                    double intervalTop = 0.0, double rangeTop = 0.0) {
  MethodResult r;
  r.name = name;
  double nnz = 1.0;
  {
    MatInfo mi;
    if (MatGetInfo(M, MAT_LOCAL, &mi) == 0) nnz = std::max(1.0, mi.nz_used);
  }
  const double fl0 = flopsNow();
  try {
    Owned<EPS, EPSDestroy> eps;
    check(EPSCreate(PETSC_COMM_SELF, &eps.obj), "EPSCreate");
    check(EPSSetOperators(eps.obj, M, nullptr), "EPSSetOperators");
    check(EPSSetProblemType(eps.obj, EPS_HEP), "EPSSetProblemType");
    check(EPSSetType(eps.obj, type), "EPSSetType");
    if (std::string(type) == EPSLANCZOS)  // required explicitly by SLEPc >= 3.x
      check(EPSLanczosSetReorthog(eps.obj, EPS_LANCZOS_REORTHOG_FULL), "EPSLanczosSetReorthog");
    check(EPSSetTolerances(eps.obj, tol, PETSC_DEFAULT), "EPSSetTolerances");
    if (filter == Filter::None) {
      check(EPSSetWhichEigenpairs(eps.obj, EPS_SMALLEST_REAL), "EPSSetWhichEigenpairs");
      check(EPSSetDimensions(eps.obj, 1, PETSC_DEFAULT, PETSC_DEFAULT), "EPSSetDimensions");
    } else {
      check(EPSSetWhichEigenpairs(eps.obj, EPS_ALL), "EPSSetWhichEigenpairs all");
      check(EPSSetInterval(eps.obj, 0.0, intervalTop), "EPSSetInterval");
      ST st = nullptr;
      check(EPSGetST(eps.obj, &st), "EPSGetST");
      check(STSetType(st, STFILTER), "STSetType filter");
      check(STFilterSetType(st, filter == Filter::Chebyshev ? ST_FILTER_CHEBYSHEV
                                                            : ST_FILTER_FILTLAN),
            "STFilterSetType");
      check(STFilterSetRange(st, 0.0, rangeTop), "STFilterSetRange");
    }
    check(EPSSetFromOptions(eps.obj), "EPSSetFromOptions");
    r.wallMs = timeMs([&] { check(EPSSolve(eps.obj), "EPSSolve"); });
    PetscInt nconv = 0, its = 0;
    check(EPSGetConverged(eps.obj, &nconv), "EPSGetConverged");
    check(EPSGetIterationNumber(eps.obj, &its), "EPSGetIterationNumber");
    r.iterations = static_cast<int>(its);
    r.nconv = static_cast<int>(nconv);
    if (nconv < 1) {
      r.note = "no eigenpair converged";
      return r;
    }
    double best = 0.0;
    for (PetscInt e = 0; e < nconv; ++e) {  // smallest converged (EPS_ALL is unsorted-ish)
      PetscScalar kr = 0.0, ki = 0.0;
      check(EPSGetEigenpair(eps.obj, e, &kr, &ki, nullptr, nullptr), "EPSGetEigenpair");
      const double lam = static_cast<double>(PetscRealPart(kr));
      if (e == 0 || lam < best) best = lam;
    }
    r.eigenvalue = best;
    r.ok = true;
  } catch (const std::exception& ex) {
    r.note = ex.what();
  }
  r.matvecEq = static_cast<long>((flopsNow() - fl0) / (2.0 * nnz));
  return r;
}

int usage() {
  std::fprintf(stderr,
               "usage: chebyshev_lanczos_bench u1 <n> <nPhi>\n"
               "       chebyshev_lanczos_bench sun <d> <n>\n");
  return 1;
}

}  // namespace

int main(int argc, char** argv) {
  SlepcInitialize(&argc, &argv, nullptr, nullptr);
  PetscLogDefaultBegin();  // enables the flop counter behind the matvec metric

  if (argc != 4) {
    const int rc = usage();
    SlepcFinalize();
    return rc;
  }

  {  // scope: every PETSc object must be destroyed before SlepcFinalize
    const std::string kind = argv[1];
    CooMatrix A(1, 1);
    char label[128];
    if (kind == "u1") {
      const int n = std::atoi(argv[2]), nPhi = std::atoi(argv[3]);
      const GaugeLattice lat = uniformFluxLattice(n, nPhi, 1.0 / n);
      A = assemblePeriodicU1(lat);
      std::snprintf(label, sizeof label, "U(1) uniform flux, n=%d, nPhi=%d", n, nPhi);
    } else if (kind == "sun") {
      const int d = std::atoi(argv[2]), n = std::atoi(argv[3]);
      if (d < 2 || d > 3) {
        const int rc = usage();
        SlepcFinalize();
        return rc;
      }
      const SunLattice lat = smoothLattice(d, n, /*mass2=*/0.0, /*amp=*/4.0);
      A = assembleRealSun(lat);
      std::snprintf(label, sizeof label, "SU(%d) smooth field (amp=4), n=%d", d, n);
    } else {
      const int rc = usage();
      SlepcFinalize();
      return rc;
    }

    const double tol = 1e-7;
    Owned<Mat, MatDestroy> M;
    buildMat(A, &M.obj);

    std::printf("\n#### %s -- real embedding dim %d, tol %.0e ####\n", label, A.rows(), tol);

    // Filter prerequisites: lambda_max (power iteration) and a rough lambda_min
    // bound (loose Krylov-Schur), both reported so their cost is on the table.
    double lamMax = 0.0;
    const double msPow = timeMs([&] { lamMax = powerLambdaMax(M.obj); });
    const MethodResult est =
        runEps(M.obj, "KS estimate (tol 1e-2)", EPSKRYLOVSCHUR, 1e-2, Filter::None);
    std::printf("lambda_max (20 power its)  = %.4f   (%.1f ms)\n", lamMax, msPow);
    std::printf("lambda_est (loose KS)      = %.6f   (%.1f ms, %d its)\n", est.eigenvalue,
                est.wallMs, est.iterations);
    const double intervalTop = 1.05 * est.eigenvalue;
    const double rangeTop = 1.1 * lamMax;
    std::printf("filter: interval [0, %.6f], numerical range [0, %.4f]\n\n", intervalTop, rangeTop);

    const MethodResult results[] = {
        runEps(M.obj, "Krylov-Schur (default)", EPSKRYLOVSCHUR, tol, Filter::None),
        runEps(M.obj, "Lanczos (full reorthog)", EPSLANCZOS, tol, Filter::None),
        runEps(M.obj, "KS + STFILTER Chebyshev", EPSKRYLOVSCHUR, tol, Filter::Chebyshev,
               intervalTop, rangeTop),
        runEps(M.obj, "KS + STFILTER FILTLAN", EPSKRYLOVSCHUR, tol, Filter::Filtlan, intervalTop,
               rangeTop),
    };

    std::printf("%-24s | %6s | %9s | %10s | %12s | %5s\n", "method", "iters", "matvec-eq",
                "wall ms", "eigenvalue", "nconv");
    std::printf("---------------------------------------------------------------------------------\n");
    for (const MethodResult& r : results) {
      if (r.ok)
        std::printf("%-24s | %6d | %9ld | %10.1f | %12.6f | %5d\n", r.name.c_str(), r.iterations,
                    r.matvecEq, r.wallMs, r.eigenvalue, r.nconv);
      else
        std::printf("%-24s | FAILED: %s (its=%d, %.1f ms)\n", r.name.c_str(), r.note.c_str(),
                    r.iterations, r.wallMs);
    }
  }

  SlepcFinalize();
  return 0;
}
