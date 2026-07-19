#include "grid/CooMatrix.h"

#include <algorithm>
#include <stdexcept>

namespace bochner {

CooMatrix::CooMatrix(int rows, int cols) : rows_(rows), cols_(cols) {
  if (rows < 0 || cols < 0) {
    throw std::out_of_range("CooMatrix: negative dimension");
  }
}

void CooMatrix::add(int i, int j, double value) {
  if (i < 0 || i >= rows_ || j < 0 || j >= cols_) {
    throw std::out_of_range("CooMatrix::add: index out of range");
  }
  triplets_.push_back({i, j, value});
}

std::vector<CooMatrix::Entry> CooMatrix::compressed() const {
  std::vector<Entry> sorted = triplets_;
  // stable_sort, not sort: duplicate (row,col) triplets are summed below, and
  // floating-point addition is not associative, so an unstable sort would make
  // the assembled matrix depend on the standard library's sort implementation.
  // This project's headline results are reproducible iteration-count tables fed
  // by this assembly, so bit-reproducibility here is worth the (negligible)
  // cost -- the comparator already orders every distinct key, so stability only
  // fixes the order *within* a duplicate group.
  std::stable_sort(sorted.begin(), sorted.end(), [](const Entry& a, const Entry& b) {
    return a.row != b.row ? a.row < b.row : a.col < b.col;
  });

  std::vector<Entry> merged;
  merged.reserve(sorted.size());
  for (const Entry& e : sorted) {
    if (!merged.empty() && merged.back().row == e.row &&
        merged.back().col == e.col) {
      merged.back().value += e.value;
    } else {
      merged.push_back(e);
    }
  }

  // Drop entries that summed to exactly zero.
  merged.erase(std::remove_if(merged.begin(), merged.end(),
                              [](const Entry& e) { return e.value == 0.0; }),
               merged.end());
  return merged;
}

}  // namespace bochner
