#pragma once

#include <vector>

namespace bochner {

/// \file
/// Plain data structures for a cell-indexed sparse matrix and an aggregation
/// multigrid hierarchy, in the grid layer so both the CPU solver (fluid) and the
/// GPU solver (gpu) can share them. Behaviour (assembly, build, solve) lives with
/// the consumers; this header is types only.

/// Cell-indexed sparse matrix (CSR + explicit diagonal); row index is the
/// MacGrid cell index.
struct SpMat {
  std::vector<int> rowStart;
  std::vector<int> col;
  std::vector<double> val;
  std::vector<double> diag;
  int n() const { return static_cast<int>(diag.size()); }
};

/// One level of the aggregation multigrid hierarchy.
struct MgLevel {
  int nx = 0, ny = 0, nz = 0;
  SpMat A;
  std::vector<char> active;  ///< cells that participate (free = non-solid, non-pinned)
  std::vector<int> aggUp;    ///< cell -> next-coarser cell index (empty on the coarsest level)
};

struct MgHierarchy {
  std::vector<MgLevel> levels;  ///< levels[0] is the finest
};

}  // namespace bochner
