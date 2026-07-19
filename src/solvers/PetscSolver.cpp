#include "solvers/PetscSolver.h"

#include <stdexcept>
#include <string>

#include <petscksp.h>

namespace bochner {

namespace {

/// Translate a nonzero PetscErrorCode into a C++ exception.
void check(PetscErrorCode ierr, const char* what) {
  if (ierr != 0) {
    throw std::runtime_error(std::string("PETSc error (") + std::to_string(ierr) +
                             ") in " + what);
  }
}

/// RAII guard that destroys a PETSc object on scope exit.
template <class T, PetscErrorCode (*Destroy)(T*)>
struct Owned {
  T obj = nullptr;
  ~Owned() {
    if (obj) Destroy(&obj);
  }
};

}  // namespace

std::vector<double> solveSpdCG(const CooMatrix& A, const std::vector<double>& b,
                               double rtol, SpdPC pcType, int* iters) {
  if (A.rows() != A.cols()) {
    throw std::invalid_argument("solveSpdCG: matrix must be square");
  }
  if (b.size() != static_cast<std::size_t>(A.rows())) {
    throw std::invalid_argument("solveSpdCG: rhs size does not match matrix");
  }

  const PetscInt n = A.rows();
  const auto entries = A.compressed();

  // Per-row nonzero counts for exact preallocation.
  std::vector<PetscInt> nnzPerRow(static_cast<std::size_t>(n), 0);
  for (const auto& e : entries) {
    ++nnzPerRow[static_cast<std::size_t>(e.row)];
  }

  Owned<Mat, MatDestroy> M;
  Owned<Vec, VecDestroy> rhs, x;
  Owned<KSP, KSPDestroy> ksp;

  check(MatCreateSeqAIJ(PETSC_COMM_SELF, n, n, 0, nnzPerRow.data(), &M.obj),
        "MatCreateSeqAIJ");
  check(MatSetOption(M.obj, MAT_SYMMETRIC, PETSC_TRUE), "MatSetOption");
  for (const auto& e : entries) {
    check(MatSetValue(M.obj, e.row, e.col, e.value, INSERT_VALUES),
          "MatSetValue");
  }
  check(MatAssemblyBegin(M.obj, MAT_FINAL_ASSEMBLY), "MatAssemblyBegin");
  check(MatAssemblyEnd(M.obj, MAT_FINAL_ASSEMBLY), "MatAssemblyEnd");

  check(VecCreateSeq(PETSC_COMM_SELF, n, &rhs.obj), "VecCreateSeq");
  for (PetscInt i = 0; i < n; ++i) {
    check(VecSetValue(rhs.obj, i, b[static_cast<std::size_t>(i)], INSERT_VALUES),
          "VecSetValue");
  }
  check(VecAssemblyBegin(rhs.obj), "VecAssemblyBegin");
  check(VecAssemblyEnd(rhs.obj), "VecAssemblyEnd");
  check(VecDuplicate(rhs.obj, &x.obj), "VecDuplicate");

  check(KSPCreate(PETSC_COMM_SELF, &ksp.obj), "KSPCreate");
  check(KSPSetOperators(ksp.obj, M.obj, M.obj), "KSPSetOperators");
  check(KSPSetType(ksp.obj, KSPCG), "KSPSetType");
  // NOTE: KSP judges rtol in PETSc's default norm -- the PRECONDITIONED
  // residual -- not the true residual norm.
  check(KSPSetTolerances(ksp.obj, rtol, PETSC_DEFAULT, PETSC_DEFAULT,
                         PETSC_DEFAULT),
        "KSPSetTolerances");
  {
    PC pc = nullptr;
    check(KSPGetPC(ksp.obj, &pc), "KSPGetPC");
    switch (pcType) {
      case SpdPC::Jacobi:
        check(PCSetType(pc, PCJACOBI), "PCSetType");
        break;
      case SpdPC::ICC:
        check(PCSetType(pc, PCICC), "PCSetType");
        break;
      case SpdPC::AMG:
        check(PCSetType(pc, PCHYPRE), "PCSetType");
        check(PCHYPRESetType(pc, "boomeramg"), "PCHYPRESetType");
        break;
    }
  }
  check(KSPSetFromOptions(ksp.obj), "KSPSetFromOptions");  // runtime override
  check(KSPSolve(ksp.obj, rhs.obj, x.obj), "KSPSolve");

  KSPConvergedReason reason{};
  check(KSPGetConvergedReason(ksp.obj, &reason), "KSPGetConvergedReason");
  if (reason < 0) {
    throw std::runtime_error("solveSpdCG: KSP did not converge (reason " +
                             std::to_string(static_cast<int>(reason)) + ")");
  }
  if (iters) {
    PetscInt it = 0;
    check(KSPGetIterationNumber(ksp.obj, &it), "KSPGetIterationNumber");
    *iters = static_cast<int>(it);
  }

  std::vector<double> result(static_cast<std::size_t>(n));
  const PetscScalar* xa = nullptr;
  check(VecGetArrayRead(x.obj, &xa), "VecGetArrayRead");
  for (PetscInt i = 0; i < n; ++i) {
    result[static_cast<std::size_t>(i)] = PetscRealPart(xa[i]);
  }
  check(VecRestoreArrayRead(x.obj, &xa), "VecRestoreArrayRead");

  return result;
}

// ---------------------------------------------------------------------------
// CachedSpdSolver: fixed matrix, reused KSP/PC, many right-hand sides.
// ---------------------------------------------------------------------------
struct CachedSpdSolver::Impl {
  PetscInt n = 0;
  Mat M = nullptr;
  KSP ksp = nullptr;
  Vec rhs = nullptr, x = nullptr;
  ~Impl() {
    if (x) VecDestroy(&x);
    if (rhs) VecDestroy(&rhs);
    if (ksp) KSPDestroy(&ksp);
    if (M) MatDestroy(&M);
  }
};

CachedSpdSolver::CachedSpdSolver(const CooMatrix& A, SpdPC pcType, double rtol)
    : impl_(std::make_unique<Impl>()) {
  if (A.rows() != A.cols()) throw std::invalid_argument("CachedSpdSolver: matrix must be square");
  Impl& s = *impl_;
  s.n = A.rows();
  const auto entries = A.compressed();
  std::vector<PetscInt> nnzPerRow(static_cast<std::size_t>(s.n), 0);
  for (const auto& e : entries) ++nnzPerRow[static_cast<std::size_t>(e.row)];

  check(MatCreateSeqAIJ(PETSC_COMM_SELF, s.n, s.n, 0, nnzPerRow.data(), &s.M), "MatCreateSeqAIJ");
  check(MatSetOption(s.M, MAT_SYMMETRIC, PETSC_TRUE), "MatSetOption");
  for (const auto& e : entries) check(MatSetValue(s.M, e.row, e.col, e.value, INSERT_VALUES), "MatSetValue");
  check(MatAssemblyBegin(s.M, MAT_FINAL_ASSEMBLY), "MatAssemblyBegin");
  check(MatAssemblyEnd(s.M, MAT_FINAL_ASSEMBLY), "MatAssemblyEnd");

  check(VecCreateSeq(PETSC_COMM_SELF, s.n, &s.rhs), "VecCreateSeq");
  check(VecDuplicate(s.rhs, &s.x), "VecDuplicate");

  check(KSPCreate(PETSC_COMM_SELF, &s.ksp), "KSPCreate");
  check(KSPSetOperators(s.ksp, s.M, s.M), "KSPSetOperators");
  check(KSPSetType(s.ksp, KSPCG), "KSPSetType");
  check(KSPSetTolerances(s.ksp, rtol, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT), "KSPSetTolerances");
  PC pc = nullptr;
  check(KSPGetPC(s.ksp, &pc), "KSPGetPC");
  switch (pcType) {
    case SpdPC::Jacobi: check(PCSetType(pc, PCJACOBI), "PCSetType"); break;
    case SpdPC::ICC: check(PCSetType(pc, PCICC), "PCSetType"); break;
    case SpdPC::AMG:
      check(PCSetType(pc, PCHYPRE), "PCSetType");
      check(PCHYPRESetType(pc, "boomeramg"), "PCHYPRESetType");
      break;
  }
  check(KSPSetFromOptions(s.ksp), "KSPSetFromOptions");
  check(KSPSetUp(s.ksp), "KSPSetUp");  // build the PC (AMG hierarchy / factorisation) now
}

CachedSpdSolver::~CachedSpdSolver() = default;
CachedSpdSolver::CachedSpdSolver(CachedSpdSolver&&) noexcept = default;
CachedSpdSolver& CachedSpdSolver::operator=(CachedSpdSolver&&) noexcept = default;

std::vector<double> CachedSpdSolver::solve(const std::vector<double>& b, int* iters) const {
  const Impl& s = *impl_;
  if (b.size() != static_cast<std::size_t>(s.n)) throw std::invalid_argument("CachedSpdSolver::solve: rhs size mismatch");
  for (PetscInt i = 0; i < s.n; ++i) check(VecSetValue(s.rhs, i, b[static_cast<std::size_t>(i)], INSERT_VALUES), "VecSetValue");
  check(VecAssemblyBegin(s.rhs), "VecAssemblyBegin");
  check(VecAssemblyEnd(s.rhs), "VecAssemblyEnd");
  check(KSPSolve(s.ksp, s.rhs, s.x), "KSPSolve");
  KSPConvergedReason reason{};
  check(KSPGetConvergedReason(s.ksp, &reason), "KSPGetConvergedReason");
  if (reason < 0) throw std::runtime_error("CachedSpdSolver::solve: KSP did not converge (" + std::to_string(static_cast<int>(reason)) + ")");
  if (iters) {
    PetscInt it = 0;
    check(KSPGetIterationNumber(s.ksp, &it), "KSPGetIterationNumber");
    *iters = static_cast<int>(it);
  }
  std::vector<double> result(static_cast<std::size_t>(s.n));
  const PetscScalar* xa = nullptr;
  check(VecGetArrayRead(s.x, &xa), "VecGetArrayRead");
  for (PetscInt i = 0; i < s.n; ++i) result[static_cast<std::size_t>(i)] = PetscRealPart(xa[i]);
  check(VecRestoreArrayRead(s.x, &xa), "VecRestoreArrayRead");
  return result;
}

}  // namespace bochner
