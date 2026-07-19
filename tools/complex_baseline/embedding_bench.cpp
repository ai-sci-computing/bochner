/// \file
/// Quantify the real-embedding penalty of the paper's SLEPc baseline
/// (review 2026-07-10, major point 1).
///
/// One source, two binaries: compiled against a REAL-scalar PETSc/SLEPc it
/// loads a complex MatrixMarket operator (written by tools/export_operator)
/// and assembles the real 2Nx2N embedding a+ib -> [a -b; b a] -- exactly what
/// the paper's production baseline receives.  Compiled against a
/// COMPLEX-scalar PETSc/SLEPc it assembles the NxN complex operator natively.
/// Both then run the identical solver configuration as
/// src/solvers/EigenSolver.cpp::smallestEigenpairLanczos: assembled AIJ,
/// default EPS type (Krylov-Schur), EPS_HEP, EPS_SMALLEST_REAL, nev=1,
/// tol from argv (paper: 1e-7), everything else default.
///
/// Usage: embedding_bench <matrix.mtx> [tol=1e-7] [reps=3] [nev=1]
/// With nev>1 the nev smallest eigenvalues are printed (spectrum diagnostics).
/// Prints: scalar type, N, nnz, eigenvalue, EPS iterations, best-of-reps ms.
#include <petscmat.h>
#include <slepceps.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Coo {
  PetscInt n = 0;
  std::vector<PetscInt> row, col;
  std::vector<double> re, im;
};

Coo readMtx(const char* path) {
  std::FILE* f = std::fopen(path, "r");
  if (!f) throw std::runtime_error(std::string("cannot open ") + path);
  char line[512];
  if (!std::fgets(line, sizeof line, f)) throw std::runtime_error("empty file");
  if (!std::strstr(line, "coordinate") || !std::strstr(line, "complex"))
    throw std::runtime_error("expected MatrixMarket coordinate complex header");
  do {
    if (!std::fgets(line, sizeof line, f)) throw std::runtime_error("truncated header");
  } while (line[0] == '%');
  long rows = 0, cols = 0, nnz = 0;
  if (std::sscanf(line, "%ld %ld %ld", &rows, &cols, &nnz) != 3 || rows != cols)
    throw std::runtime_error("bad size line");
  Coo c;
  c.n = static_cast<PetscInt>(rows);
  c.row.reserve(nnz); c.col.reserve(nnz); c.re.reserve(nnz); c.im.reserve(nnz);
  for (long k = 0; k < nnz; ++k) {
    long i = 0, j = 0; double a = 0.0, b = 0.0;
    if (std::fscanf(f, "%ld %ld %lf %lf", &i, &j, &a, &b) != 4)
      throw std::runtime_error("truncated entries");
    c.row.push_back(static_cast<PetscInt>(i - 1));
    c.col.push_back(static_cast<PetscInt>(j - 1));
    c.re.push_back(a);
    c.im.push_back(b);
  }
  std::fclose(f);
  return c;
}

void check(PetscErrorCode e, const char* what) {
  if (e) { std::fprintf(stderr, "PETSc error in %s\n", what); std::exit(2); }
}

Mat assemble(const Coo& c) {
  Mat M = nullptr;
#if defined(PETSC_USE_COMPLEX)
  const PetscInt N = c.n;
#else
  const PetscInt N = 2 * c.n;
#endif
  std::vector<PetscInt> nnzRow(static_cast<std::size_t>(N), 0);
  for (std::size_t k = 0; k < c.row.size(); ++k) {
#if defined(PETSC_USE_COMPLEX)
    ++nnzRow[static_cast<std::size_t>(c.row[k])];
#else
    nnzRow[static_cast<std::size_t>(2 * c.row[k])] += 2;
    nnzRow[static_cast<std::size_t>(2 * c.row[k] + 1)] += 2;
#endif
  }
  check(MatCreateSeqAIJ(PETSC_COMM_SELF, N, N, 0, nnzRow.data(), &M), "MatCreateSeqAIJ");
  for (std::size_t k = 0; k < c.row.size(); ++k) {
    const PetscInt i = c.row[k], j = c.col[k];
    const double a = c.re[k], b = c.im[k];
#if defined(PETSC_USE_COMPLEX)
    const PetscScalar v = a + b * PETSC_i;
    check(MatSetValue(M, i, j, v, INSERT_VALUES), "MatSetValue");
#else
    // a+ib -> [a -b; b a]
    check(MatSetValue(M, 2 * i,     2 * j,     a, INSERT_VALUES), "MatSetValue");
    check(MatSetValue(M, 2 * i,     2 * j + 1, -b, INSERT_VALUES), "MatSetValue");
    check(MatSetValue(M, 2 * i + 1, 2 * j,     b, INSERT_VALUES), "MatSetValue");
    check(MatSetValue(M, 2 * i + 1, 2 * j + 1, a, INSERT_VALUES), "MatSetValue");
#endif
  }
  check(MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY), "MatAssemblyBegin");
  check(MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY), "MatAssemblyEnd");
  return M;
}

}  // namespace

int main(int argc, char** argv) {
  check(SlepcInitialize(&argc, &argv, nullptr, nullptr), "SlepcInitialize");
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <matrix.mtx> [tol] [reps]\n", argv[0]);
    return 1;
  }
  const double tol = argc > 2 ? std::atof(argv[2]) : 1e-7;
  const int reps = argc > 3 ? std::atoi(argv[3]) : 3;
  const int nev = argc > 4 ? std::atoi(argv[4]) : 1;

  const Coo coo = readMtx(argv[1]);
  Mat M = assemble(coo);
  PetscInt N = 0, dummy = 0;
  check(MatGetSize(M, &N, &dummy), "MatGetSize");
  MatInfo info;
  check(MatGetInfo(M, MAT_LOCAL, &info), "MatGetInfo");

  double bestMs = 1e300, lambda = 0.0;
  PetscInt its = 0;
  for (int r = 0; r < reps; ++r) {
    EPS eps = nullptr;
    check(EPSCreate(PETSC_COMM_SELF, &eps), "EPSCreate");
    check(EPSSetOperators(eps, M, nullptr), "EPSSetOperators");
    check(EPSSetProblemType(eps, EPS_HEP), "EPSSetProblemType");
    check(EPSSetWhichEigenpairs(eps, EPS_SMALLEST_REAL), "EPSSetWhich");
    check(EPSSetDimensions(eps, nev, PETSC_DEFAULT, PETSC_DEFAULT), "EPSSetDimensions");
    check(EPSSetTolerances(eps, tol, PETSC_DEFAULT), "EPSSetTolerances");
    check(EPSSetFromOptions(eps), "EPSSetFromOptions");
    const auto t0 = std::chrono::steady_clock::now();
    check(EPSSolve(eps), "EPSSolve");
    const auto t1 = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    PetscInt nconv = 0;
    check(EPSGetConverged(eps, &nconv), "EPSGetConverged");
    if (nconv < nev) { std::fprintf(stderr, "fewer than nev eigenpairs converged\n"); return 3; }
    PetscScalar kr = 0.0, ki = 0.0;
    check(EPSGetEigenpair(eps, 0, &kr, &ki, nullptr, nullptr), "EPSGetEigenpair");
    check(EPSGetIterationNumber(eps, &its), "EPSGetIterationNumber");
    lambda = PetscRealPart(kr);
    if (nev > 1 && r == 0) {
      std::printf("  lowest %d eigenvalues:\n", nev);
      double prev = 0.0;
      for (PetscInt e = 0; e < nev; ++e) {
        PetscScalar er = 0.0, ei = 0.0;
        check(EPSGetEigenpair(eps, e, &er, &ei, nullptr, nullptr), "EPSGetEigenpair");
        const double le = PetscRealPart(er);
        if (e == 0)
          std::printf("    lam[%2d] = %.12g\n", int(e + 1), le);
        else
          std::printf("    lam[%2d] = %.12g   (lam[k]-lam[k-1])/lam[1] = %.3e\n",
                      int(e + 1), le, (le - prev) / lambda);
        prev = le;
      }
    }
    if (ms < bestMs) bestMs = ms;
    check(EPSDestroy(&eps), "EPSDestroy");
  }
#if defined(PETSC_USE_COMPLEX)
  const char* kind = "complex-native";
#else
  const char* kind = "real-embedded ";
#endif
  std::printf("%s  file=%s  N=%d  nnz=%.0f  eig=%.10g  its=%d  best_ms=%.1f\n",
              kind, argv[1], static_cast<int>(N), info.nz_used, lambda,
              static_cast<int>(its), bestMs);
  check(MatDestroy(&M), "MatDestroy");
  check(SlepcFinalize(), "SlepcFinalize");
  return 0;
}
