#include "fluid/MacAdvection.h"

#include <algorithm>
#include <cmath>

#ifdef BOCHNER_WITH_METAL
#include "gpu/MetalContext.h"
#endif

namespace bochner {

namespace {

// Trilinear interpolation of one staggered component. The component is sampled
// on an (ni x nj x nk) lattice whose node (a,b,c) sits at
//   origin + (a+ox, b+oy, c+oz) * h,
// and `at(a,b,c)` returns the stored value there. Query coordinates are clamped
// to the lattice extent (closest-interior-point boundary, CF §5.4.5).
template <class At>
double trilerpComponent(const MacGrid& g, const Vec3& p, int ni, int nj, int nk, double ox,
                        double oy, double oz, At at, bool extrap = false) {
  const double h = g.spacing();
  const Vec3 o = g.origin();

  // For one axis: convert a world coordinate to lattice-node units and split into
  // a base index + fractional weight. Default: clamp the query into [0, n-1] (the
  // streak boundary, CF §5.4.5) so out-of-domain values use the closest interior
  // sample -- correct for a VELOCITY. With \p extrap the query is NOT clamped:
  // only the base index is bounded to [0, n-2] and the weight t is left free, so
  // an out-of-domain query is linearly EXTRAPOLATED. This is the correct
  // treatment for a MAP (a Lagrangian coordinate field legitimately points
  // outside the domain); clamping a map injects a wall error that the flow then
  // advects inward (verified: an affine shear map stays exact only with extrap).
  auto setup = [&](double world, double orig, double off, int n, int& i0, double& t) {
    double c = (world - orig) / h - off;  // coordinate in lattice-node units
    if (!extrap) {
      if (c < 0.0) c = 0.0;
      double maxc = static_cast<double>(n - 1);
      if (c > maxc) c = maxc;
    }
    i0 = static_cast<int>(std::floor(c));
    if (i0 < 0) i0 = 0;
    if (i0 > n - 2) i0 = std::max(0, n - 2);
    t = c - i0;  // with extrap, t may fall outside [0,1] -> linear extrapolation
    if (n == 1) {
      i0 = 0;
      t = 0.0;
    }
  };

  int i0, j0, k0;
  double tx, ty, tz;
  setup(p[0], o[0], ox, ni, i0, tx);
  setup(p[1], o[1], oy, nj, j0, ty);
  setup(p[2], o[2], oz, nk, k0, tz);
  int i1 = std::min(i0 + 1, ni - 1);
  int j1 = std::min(j0 + 1, nj - 1);
  int k1 = std::min(k0 + 1, nk - 1);

  double c000 = at(i0, j0, k0), c100 = at(i1, j0, k0);
  double c010 = at(i0, j1, k0), c110 = at(i1, j1, k0);
  double c001 = at(i0, j0, k1), c101 = at(i1, j0, k1);
  double c011 = at(i0, j1, k1), c111 = at(i1, j1, k1);

  double c00 = c000 * (1 - tx) + c100 * tx;
  double c10 = c010 * (1 - tx) + c110 * tx;
  double c01 = c001 * (1 - tx) + c101 * tx;
  double c11 = c011 * (1 - tx) + c111 * tx;
  double c0 = c00 * (1 - ty) + c10 * ty;
  double c1 = c01 * (1 - ty) + c11 * ty;
  return c0 * (1 - tz) + c1 * tz;
}

}  // namespace

Vec3 sampleVelocity(const MacGrid& g, const FaceField& u, const Vec3& p) {
  // x-faces: lattice (nx+1, ny, nz), node offset (0, 0.5, 0.5).
  double ux = trilerpComponent(g, p, g.nx() + 1, g.ny(), g.nz(), 0.0, 0.5, 0.5,
                               [&](int i, int j, int k) { return u.x[g.faceXIndex(i, j, k)]; });
  // y-faces: lattice (nx, ny+1, nz), node offset (0.5, 0, 0.5).
  double uy = trilerpComponent(g, p, g.nx(), g.ny() + 1, g.nz(), 0.5, 0.0, 0.5,
                               [&](int i, int j, int k) { return u.y[g.faceYIndex(i, j, k)]; });
  // z-faces: lattice (nx, ny, nz+1), node offset (0.5, 0.5, 0).
  double uz = trilerpComponent(g, p, g.nx(), g.ny(), g.nz() + 1, 0.5, 0.5, 0.0,
                               [&](int i, int j, int k) { return u.z[g.faceZIndex(i, j, k)]; });
  return {ux, uy, uz};
}

Vec3 backtrace(const MacGrid& g, const FaceField& v, const Vec3& start, double dt) {
  // Classic RK4 for dx/dt = v, integrated backward over dt.
  Vec3 k1 = sampleVelocity(g, v, start);
  Vec3 x2 = vsub(start, vscale(k1, 0.5 * dt));
  Vec3 k2 = sampleVelocity(g, v, x2);
  Vec3 x3 = vsub(start, vscale(k2, 0.5 * dt));
  Vec3 k3 = sampleVelocity(g, v, x3);
  Vec3 x4 = vsub(start, vscale(k3, dt));
  Vec3 k4 = sampleVelocity(g, v, x4);
  Vec3 sum = vadd(vadd(k1, vscale(k2, 2.0)), vadd(vscale(k3, 2.0), k4));
  return vsub(start, vscale(sum, dt / 6.0));
}

FaceField advectCovectorSL(const MacGrid& g, const FaceField& u, const FaceField& v, double dt) {
  const double inv_h = 1.0 / g.spacing();

  // Backtrace every cell center once; the per-face transpose-Jacobian column is
  // a finite difference of this map across the face (CF Eq. 40). Each cell is
  // independent (writes its own psiCell, reads only v) -> parallel.
  std::vector<Vec3> psiCell(g.numCells());
#pragma omp parallel for schedule(static)
  for (int i = 0; i < g.nx(); ++i)
    for (int j = 0; j < g.ny(); ++j)
      for (int k = 0; k < g.nz(); ++k)
        psiCell[g.cellIndex(i, j, k)] = backtrace(g, v, g.cellCenter(i, j, k), dt);

  // Dot a transpose-Jacobian column (dPsi/dn) with the sampled velocity (Eq 39).
  auto pullback = [&](const Vec3& dPsi_dn, const Vec3& faceCenter) {
    Vec3 w = sampleVelocity(g, u, backtrace(g, v, faceCenter, dt));
    return dPsi_dn[0] * w[0] + dPsi_dn[1] * w[1] + dPsi_dn[2] * w[2];
  };

  FaceField out = u;  // boundary faces stay at their prescribed values

  // Each interior face is independent (writes its own out entry) -> parallel.
#pragma omp parallel for schedule(static)
  for (int i = 1; i < g.nx(); ++i)  // interior x-faces (between cells i-1 and i)
    for (int j = 0; j < g.ny(); ++j)
      for (int k = 0; k < g.nz(); ++k) {
        Vec3 dPsi = vscale(vsub(psiCell[g.cellIndex(i, j, k)], psiCell[g.cellIndex(i - 1, j, k)]),
                           inv_h);
        out.x[g.faceXIndex(i, j, k)] = pullback(dPsi, g.faceXCenter(i, j, k));
      }

#pragma omp parallel for schedule(static)
  for (int i = 0; i < g.nx(); ++i)  // interior y-faces
    for (int j = 1; j < g.ny(); ++j)
      for (int k = 0; k < g.nz(); ++k) {
        Vec3 dPsi = vscale(vsub(psiCell[g.cellIndex(i, j, k)], psiCell[g.cellIndex(i, j - 1, k)]),
                           inv_h);
        out.y[g.faceYIndex(i, j, k)] = pullback(dPsi, g.faceYCenter(i, j, k));
      }

#pragma omp parallel for schedule(static)
  for (int i = 0; i < g.nx(); ++i)  // interior z-faces
    for (int j = 0; j < g.ny(); ++j)
      for (int k = 1; k < g.nz(); ++k) {
        Vec3 dPsi = vscale(vsub(psiCell[g.cellIndex(i, j, k)], psiCell[g.cellIndex(i, j, k - 1)]),
                           inv_h);
        out.z[g.faceZIndex(i, j, k)] = pullback(dPsi, g.faceZCenter(i, j, k));
      }

  return out;
}

namespace {

// Clamp `value` at same-family face (i,j,k) to the min/max of `u1fam` over the
// immediate neighbour box (one cell each direction), with neighbour indices
// clamped to the family extent (ni x nj x nk). CF §5.4.2 extrema limiter.
template <class Idx>
double limitToStencil(const std::vector<double>& u1fam, double value, int i, int j, int k, int ni,
                      int nj, int nk, Idx idx) {
  // Bounds come from u1 over the neighbour box only -- never from `value`
  // itself, or an overshooting value would become its own (useless) bound.
  double lo = u1fam[idx(i, j, k)], hi = lo;
  for (int di = -1; di <= 1; ++di)
    for (int dj = -1; dj <= 1; ++dj)
      for (int dk = -1; dk <= 1; ++dk) {
        int a = std::clamp(i + di, 0, ni - 1);
        int b = std::clamp(j + dj, 0, nj - 1);
        int c = std::clamp(k + dk, 0, nk - 1);
        double val = u1fam[idx(a, b, c)];
        lo = std::min(lo, val);
        hi = std::max(hi, val);
      }
  return std::clamp(value, lo, hi);
}

}  // namespace

Vec3 sampleCellVec3(const MacGrid& g, const std::vector<Vec3>& field, const Vec3& p, bool extrap) {
  auto comp = [&](int c) {
    return trilerpComponent(
        g, p, g.nx(), g.ny(), g.nz(), 0.5, 0.5, 0.5,
        [&](int i, int j, int k) { return field[g.cellIndex(i, j, k)][c]; }, extrap);
  };
  return {comp(0), comp(1), comp(2)};
}

FaceField pullbackThroughMap(const MacGrid& g, const std::vector<Vec3>& map,
                             const FaceField& source) {
  const double inv_h = 1.0 / g.spacing();
  // (dM/dn) . source(M(F)): the map's transpose-Jacobian column along the face
  // normal, dotted with the source velocity sampled at the face's mapped point.
  auto pull = [&](const Vec3& dM, const Vec3& faceCenter) {
    const Vec3 w = sampleVelocity(g, source, sampleCellVec3(g, map, faceCenter));
    return dM[0] * w[0] + dM[1] * w[1] + dM[2] * w[2];
  };
  FaceField out = source;  // boundary faces keep their (BC-prescribed) source value

#pragma omp parallel for schedule(static)
  for (int i = 1; i < g.nx(); ++i)
    for (int j = 0; j < g.ny(); ++j)
      for (int k = 0; k < g.nz(); ++k) {
        const Vec3 dM =
            vscale(vsub(map[g.cellIndex(i, j, k)], map[g.cellIndex(i - 1, j, k)]), inv_h);
        out.x[g.faceXIndex(i, j, k)] = pull(dM, g.faceXCenter(i, j, k));
      }
#pragma omp parallel for schedule(static)
  for (int i = 0; i < g.nx(); ++i)
    for (int j = 1; j < g.ny(); ++j)
      for (int k = 0; k < g.nz(); ++k) {
        const Vec3 dM =
            vscale(vsub(map[g.cellIndex(i, j, k)], map[g.cellIndex(i, j - 1, k)]), inv_h);
        out.y[g.faceYIndex(i, j, k)] = pull(dM, g.faceYCenter(i, j, k));
      }
#pragma omp parallel for schedule(static)
  for (int i = 0; i < g.nx(); ++i)
    for (int j = 0; j < g.ny(); ++j)
      for (int k = 1; k < g.nz(); ++k) {
        const Vec3 dM =
            vscale(vsub(map[g.cellIndex(i, j, k)], map[g.cellIndex(i, j, k - 1)]), inv_h);
        out.z[g.faceZIndex(i, j, k)] = pull(dM, g.faceZCenter(i, j, k));
      }
  return out;
}

FaceField bfeccLimit(const MacGrid& g, const FaceField& reference, const FaceField& candidate) {
  FaceField out = candidate;  // boundary faces stay as given
#pragma omp parallel for schedule(static)
  for (int i = 1; i < g.nx(); ++i)
    for (int j = 0; j < g.ny(); ++j)
      for (int k = 0; k < g.nz(); ++k) {
        const int f = g.faceXIndex(i, j, k);
        out.x[f] = limitToStencil(reference.x, candidate.x[f], i, j, k, g.nx() + 1, g.ny(), g.nz(),
                                  [&](int a, int b, int c) { return g.faceXIndex(a, b, c); });
      }
#pragma omp parallel for schedule(static)
  for (int i = 0; i < g.nx(); ++i)
    for (int j = 1; j < g.ny(); ++j)
      for (int k = 0; k < g.nz(); ++k) {
        const int f = g.faceYIndex(i, j, k);
        out.y[f] = limitToStencil(reference.y, candidate.y[f], i, j, k, g.nx(), g.ny() + 1, g.nz(),
                                  [&](int a, int b, int c) { return g.faceYIndex(a, b, c); });
      }
#pragma omp parallel for schedule(static)
  for (int i = 0; i < g.nx(); ++i)
    for (int j = 0; j < g.ny(); ++j)
      for (int k = 1; k < g.nz(); ++k) {
        const int f = g.faceZIndex(i, j, k);
        out.z[f] = limitToStencil(reference.z, candidate.z[f], i, j, k, g.nx(), g.ny(), g.nz() + 1,
                                  [&](int a, int b, int c) { return g.faceZIndex(a, b, c); });
      }
  return out;
}

FaceField advectCovectorBFECC(const MacGrid& g, const FaceField& u, const FaceField& v, double dt) {
  FaceField u1 = advectCovectorSL(g, u, v, dt);        // forward
  FaceField u0b = advectCovectorSL(g, u1, v, -dt);     // back-and-forth

  // Half the round-trip error, e/2 = (u0b - u)/2 (boundary faces cancel to 0).
  FaceField eh = u;
  for (size_t f = 0; f < eh.x.size(); ++f) eh.x[f] = 0.5 * (u0b.x[f] - u.x[f]);
  for (size_t f = 0; f < eh.y.size(); ++f) eh.y[f] = 0.5 * (u0b.y[f] - u.y[f]);
  for (size_t f = 0; f < eh.z.size(); ++f) eh.z[f] = 0.5 * (u0b.z[f] - u.z[f]);
  FaceField corr = advectCovectorSL(g, eh, v, dt);

  // corrected = u1 - corr on interior faces (boundary faces keep u1's BC value),
  // then clamp to u1's neighbourhood extrema (CF 5.4.2) -> no new oscillations.
  FaceField corrected = u1;
#pragma omp parallel for schedule(static)
  for (int i = 1; i < g.nx(); ++i)
    for (int j = 0; j < g.ny(); ++j)
      for (int k = 0; k < g.nz(); ++k) corrected.x[g.faceXIndex(i, j, k)] -= corr.x[g.faceXIndex(i, j, k)];
#pragma omp parallel for schedule(static)
  for (int i = 0; i < g.nx(); ++i)
    for (int j = 1; j < g.ny(); ++j)
      for (int k = 0; k < g.nz(); ++k) corrected.y[g.faceYIndex(i, j, k)] -= corr.y[g.faceYIndex(i, j, k)];
#pragma omp parallel for schedule(static)
  for (int i = 0; i < g.nx(); ++i)
    for (int j = 0; j < g.ny(); ++j)
      for (int k = 1; k < g.nz(); ++k) corrected.z[g.faceZIndex(i, j, k)] -= corr.z[g.faceZIndex(i, j, k)];
  return bfeccLimit(g, u1, corrected);
}

#ifdef BOCHNER_WITH_METAL
FaceField advectCovectorBFECCGpu(const MacGrid& g, const FaceField& u, const FaceField& v, double dt) {
  // The fully device-resident GPU BFECC: u and v are uploaded once and the whole
  // step (three SL sub-steps + error blend + CF 5.4.2 limiter) runs as kernels on
  // resident buffers, with only the result read back. This is the fluid-layer
  // entry point the pipeline calls; the Metal implementation lives in the gpu
  // module. Single precision; matches advectCovectorBFECC to float tolerance.
  return gpu::advectCovectorBFECCResident(g, u, v, dt);
}
#endif

}  // namespace bochner
