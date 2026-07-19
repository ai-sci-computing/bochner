#pragma once

#include <cstddef>
#include <vector>

namespace bochner {

/// \brief Backend-neutral sparse matrix in coordinate (triplet) format.
///
/// Discrete operators (incidence matrices, Hodge stars, the connection
/// Laplacian) are assembled here as `(row, col, value)` triplets, independent
/// of any linear-algebra backend, then handed to a concrete backend (PETSc,
/// later possibly Trilinos) for solving. Keeping assembly backend-neutral is a
/// deliberate project decision so that alternative
/// solver libraries can be benchmarked in Phase 3 without re-assembling.
///
/// Duplicate `(row, col)` contributions accumulate, matching standard
/// finite-element / DEC assembly semantics.
class CooMatrix {
public:
  /// One compressed (duplicates summed) nonzero entry.
  struct Entry {
    int row;
    int col;
    double value;
  };

  /// Construct an all-zero \p rows by \p cols matrix.
  CooMatrix(int rows, int cols);

  /// Number of matrix rows.
  int rows() const noexcept { return rows_; }
  /// Number of matrix columns.
  int cols() const noexcept { return cols_; }

  /// Number of raw triplets stored (one per add() call), before merging.
  std::size_t numTriplets() const noexcept { return triplets_.size(); }

  /// Accumulate \p value into entry (\p i, \p j).
  /// \throws std::out_of_range if (\p i, \p j) lies outside the declared shape.
  void add(int i, int j, double value);

  /// Triplets with duplicate `(row, col)` summed and exact zeros dropped,
  /// sorted ascending by `(row, col)`.
  std::vector<Entry> compressed() const;

private:
  int rows_;
  int cols_;
  std::vector<Entry> triplets_;
};

}  // namespace bochner
