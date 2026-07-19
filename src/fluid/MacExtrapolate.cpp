/// \file
/// Velocity extrapolation into solid cells -- the missing half of the
/// obstacle boundary treatment (Bridson, *Fluid Simulation for Computer
/// Graphics*, sec. 6).
///
/// `zeroSolidFaces` holds in-solid faces at exactly 0 forever, which is right
/// for the projection but wrong for advection: a semi-Lagrangian backtrace from
/// a face one layer outside the body lands inside it and samples ~0, injecting
/// a sheet of dead fluid off the body every step. That is a numerical wall
/// viscosity, and on the shipped obstacle demo it dwarfs the physical `nu`.
///
/// The cure is to give the solid band a plausible velocity *before* advecting,
/// then re-zero afterwards (the projection re-imposes u.n = 0 regardless). This
/// is a preprocessing pass on the FaceField, so the CPU and GPU advection paths
/// both benefit with no change to either kernel.
#include "fluid/MacExtrapolate.h"

#include <vector>

namespace bochner {

namespace {

// One component of the staggered field. `valid` marks faces whose value is
// physically meaningful; unknown faces are filled from valid neighbours by
// repeated averaging, one cell-layer per sweep.
void extrapolateComponent(std::vector<double>& f, std::vector<std::uint8_t>& valid, int ex, int ey,
                          int ez, int band) {
  const auto idx = [&](int i, int j, int k) { return (i * ey + j) * ez + k; };
  std::vector<double> next;
  std::vector<int> filled;
  for (int sweep = 0; sweep < band; ++sweep) {
    next.assign(f.begin(), f.end());
    filled.clear();
    for (int i = 0; i < ex; ++i)
      for (int j = 0; j < ey; ++j)
        for (int k = 0; k < ez; ++k) {
          const int c = idx(i, j, k);
          if (valid[c]) continue;
          double sum = 0.0;
          int cnt = 0;
          if (i > 0 && valid[idx(i - 1, j, k)]) { sum += f[idx(i - 1, j, k)]; ++cnt; }
          if (i + 1 < ex && valid[idx(i + 1, j, k)]) { sum += f[idx(i + 1, j, k)]; ++cnt; }
          if (j > 0 && valid[idx(i, j - 1, k)]) { sum += f[idx(i, j - 1, k)]; ++cnt; }
          if (j + 1 < ey && valid[idx(i, j + 1, k)]) { sum += f[idx(i, j + 1, k)]; ++cnt; }
          if (k > 0 && valid[idx(i, j, k - 1)]) { sum += f[idx(i, j, k - 1)]; ++cnt; }
          if (k + 1 < ez && valid[idx(i, j, k + 1)]) { sum += f[idx(i, j, k + 1)]; ++cnt; }
          if (cnt == 0) continue;
          next[c] = sum / cnt;
          filled.push_back(c);
        }
    if (filled.empty()) break;  // nothing further to reach
    f.swap(next);
    for (const int c : filled) valid[c] = 1;
  }
}

}  // namespace

void extrapolateIntoSolid(const MacGrid& g, FaceField& u, const SolidMask& solid, int band) {
  if (solid.empty() || band <= 0) return;

  // A face carries real velocity iff EVERY existing adjacent cell is fluid --
  // the exact complement of zeroSolidFaces, which zeroes a face when EITHER
  // neighbour is solid. Marking "at least one fluid neighbour" as valid would
  // leave the solid-boundary faces pinned at zero and make this whole pass a
  // no-op (measured: it was).
  std::vector<std::uint8_t> vx(u.x.size(), 0), vy(u.y.size(), 0), vz(u.z.size(), 0);
  for (int i = 0; i <= g.nx(); ++i)
    for (int j = 0; j < g.ny(); ++j)
      for (int k = 0; k < g.nz(); ++k) {
        const bool loSolid = i > 0 && isSolid(solid, g.cellIndex(i - 1, j, k));
        const bool hiSolid = i < g.nx() && isSolid(solid, g.cellIndex(i, j, k));
        vx[g.faceXIndex(i, j, k)] = (loSolid || hiSolid) ? 0 : 1;
      }
  for (int i = 0; i < g.nx(); ++i)
    for (int j = 0; j <= g.ny(); ++j)
      for (int k = 0; k < g.nz(); ++k) {
        const bool loSolid = j > 0 && isSolid(solid, g.cellIndex(i, j - 1, k));
        const bool hiSolid = j < g.ny() && isSolid(solid, g.cellIndex(i, j, k));
        vy[g.faceYIndex(i, j, k)] = (loSolid || hiSolid) ? 0 : 1;
      }
  for (int i = 0; i < g.nx(); ++i)
    for (int j = 0; j < g.ny(); ++j)
      for (int k = 0; k <= g.nz(); ++k) {
        const bool loSolid = k > 0 && isSolid(solid, g.cellIndex(i, j, k - 1));
        const bool hiSolid = k < g.nz() && isSolid(solid, g.cellIndex(i, j, k));
        vz[g.faceZIndex(i, j, k)] = (loSolid || hiSolid) ? 0 : 1;
      }

  extrapolateComponent(u.x, vx, g.nx() + 1, g.ny(), g.nz(), band);
  extrapolateComponent(u.y, vy, g.nx(), g.ny() + 1, g.nz(), band);
  extrapolateComponent(u.z, vz, g.nx(), g.ny(), g.nz() + 1, band);
}

}  // namespace bochner
