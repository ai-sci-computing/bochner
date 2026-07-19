#include "grid/MacGrid.h"

#include <stdexcept>

namespace bochner {

MacGrid::MacGrid(int nx, int ny, int nz, double h, Vec3 origin)
    : nx_(nx), ny_(ny), nz_(nz), h_(h), origin_(origin) {
  if (nx < 1 || ny < 1 || nz < 1)
    throw std::invalid_argument("MacGrid: each dimension must be at least 1");
  if (!(h > 0.0)) throw std::invalid_argument("MacGrid: spacing must be positive");
}

Vec3 MacGrid::cellCenter(int i, int j, int k) const noexcept {
  return {origin_[0] + (i + 0.5) * h_, origin_[1] + (j + 0.5) * h_, origin_[2] + (k + 0.5) * h_};
}

Vec3 MacGrid::faceXCenter(int i, int j, int k) const noexcept {
  return {origin_[0] + i * h_, origin_[1] + (j + 0.5) * h_, origin_[2] + (k + 0.5) * h_};
}

Vec3 MacGrid::faceYCenter(int i, int j, int k) const noexcept {
  return {origin_[0] + (i + 0.5) * h_, origin_[1] + j * h_, origin_[2] + (k + 0.5) * h_};
}

Vec3 MacGrid::faceZCenter(int i, int j, int k) const noexcept {
  return {origin_[0] + (i + 0.5) * h_, origin_[1] + (j + 0.5) * h_, origin_[2] + k * h_};
}

}  // namespace bochner
