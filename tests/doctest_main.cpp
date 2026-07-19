/// \file
/// Provides doctest's main() for all test executables. When the PETSc backend
/// is enabled, wraps the test run in Slepc/PetscInitialize..Finalize so PETSc
/// objects can be created inside test cases.
#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest.h>

#ifdef BOCHNER_WITH_PETSC
#include <slepc.h>  // SlepcInitialize also initializes PETSc
#endif

int main(int argc, char** argv) {
#ifdef BOCHNER_WITH_PETSC
  SlepcInitialize(&argc, &argv, nullptr, nullptr);
#endif
  const int res = doctest::Context(argc, argv).run();
#ifdef BOCHNER_WITH_PETSC
  SlepcFinalize();
#endif
  return res;
}
