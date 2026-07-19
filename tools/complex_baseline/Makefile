# Builds embedding_bench twice: against the real-scalar PETSc/SLEPc the
# project uses, and against the complex-scalar builds in
# ~/.local/{petsc,slepc}-complex. Isolates how much of the SLEPc baseline's
# cost is the real 2x2 embedding of the complex operator rather than the
# algorithm: a complex-scalar PETSc solves the same operator natively.
#
#   make            # both binaries
#   make run        # export the paper's operators and run the comparison

REAL_PETSC    ?= $(HOME)/.local/petsc
REAL_SLEPC    ?= $(HOME)/.local/slepc
COMPLEX_PETSC ?= $(HOME)/.local/petsc-complex
COMPLEX_SLEPC ?= $(HOME)/.local/slepc-complex

CXX      = /opt/homebrew/bin/mpicxx
CXXFLAGS = -O2 -std=c++17

all: bench_real bench_complex

bench_real: embedding_bench.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ \
	  -I$(REAL_PETSC)/include -I$(REAL_SLEPC)/include \
	  -L$(REAL_PETSC)/lib -L$(REAL_SLEPC)/lib -lslepc -lpetsc \
	  -Wl,-rpath,$(REAL_PETSC)/lib -Wl,-rpath,$(REAL_SLEPC)/lib

bench_complex: embedding_bench.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ \
	  -I$(COMPLEX_PETSC)/include -I$(COMPLEX_SLEPC)/include \
	  -L$(COMPLEX_PETSC)/lib -L$(COMPLEX_SLEPC)/lib -lslepc -lpetsc \
	  -Wl,-rpath,$(COMPLEX_PETSC)/lib -Wl,-rpath,$(COMPLEX_SLEPC)/lib

clean:
	rm -f bench_real bench_complex
