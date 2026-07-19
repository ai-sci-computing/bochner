/// \file
/// Red-green TDD: behaviour spec for the backend-neutral COO sparse matrix.
#include <doctest.h>

#include "grid/CooMatrix.h"

using bochner::CooMatrix;

TEST_CASE("CooMatrix reports its declared shape") {
  CooMatrix A(3, 4);
  CHECK(A.rows() == 3);
  CHECK(A.cols() == 4);
  CHECK(A.numTriplets() == 0);
  CHECK(A.compressed().empty());
}

TEST_CASE("add() stores triplets and accumulates duplicates") {
  CooMatrix A(2, 2);
  A.add(0, 0, 1.0);
  A.add(0, 0, 2.0);  // same (i,j) -> should sum to 3.0
  A.add(1, 1, -4.0);

  // Raw triplet count is the number of add() calls (no eager merge).
  CHECK(A.numTriplets() == 3);

  const auto entries = A.compressed();
  REQUIRE(entries.size() == 2);

  // compressed() is sorted by (row, col).
  CHECK(entries[0].row == 0);
  CHECK(entries[0].col == 0);
  CHECK(entries[0].value == doctest::Approx(3.0));
  CHECK(entries[1].row == 1);
  CHECK(entries[1].col == 1);
  CHECK(entries[1].value == doctest::Approx(-4.0));
}

TEST_CASE("compressed() drops entries that sum to zero") {
  CooMatrix A(2, 2);
  A.add(0, 1, 5.0);
  A.add(0, 1, -5.0);  // cancels exactly
  CHECK(A.compressed().empty());
}

TEST_CASE("add() rejects out-of-range indices") {
  CooMatrix A(2, 2);
  CHECK_THROWS_AS(A.add(2, 0, 1.0), std::out_of_range);
  CHECK_THROWS_AS(A.add(0, 2, 1.0), std::out_of_range);
  CHECK_THROWS_AS(A.add(-1, 0, 1.0), std::out_of_range);
}
