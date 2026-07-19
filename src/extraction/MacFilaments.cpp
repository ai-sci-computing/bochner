#include "extraction/MacFilaments.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_map>

namespace bochner {

namespace {

constexpr double kPi = 3.14159265358979323846;

// Wrap an angle difference into [-pi, pi]. Exactly-antipodal differences keep
// their sign (+pi stays +pi, -pi stays -pi): opposite traversals of a shared
// edge must contribute antisymmetrically, or the per-cube even-crossing parity
// that linkFilaments relies on breaks at exact-pi degeneracies.
double wrap(double d) {
  while (d > kPi) d -= 2.0 * kPi;
  while (d < -kPi) d += 2.0 * kPi;
  return d;
}

// Phase winding number around four corners given in cyclic order (their phases
// are phi[0..3]); returns the net number of 2*pi turns, normally -1, 0 or +1.
int winding(const std::array<double, 4>& phi) {
  double sum = 0.0;
  for (int e = 0; e < 4; ++e) sum += wrap(phi[(e + 1) % 4] - phi[e]);
  return static_cast<int>(std::lround(sum / (2.0 * kPi)));
}

// Bilinear value of corner data f00,f10,f11,f01 at (s,t) in [0,1]^2, plus its
// (s,t) partial derivatives. Corner order matches the cyclic plaquette order:
// index 0=(0,0), 1=(1,0), 2=(1,1), 3=(0,1).
struct Bilinear {
  double v, ds, dt;
};
Bilinear bilinear(const std::array<double, 4>& f, double s, double t) {
  double v = f[0] * (1 - s) * (1 - t) + f[1] * s * (1 - t) + f[2] * s * t + f[3] * (1 - s) * t;
  double ds = (1 - t) * (f[1] - f[0]) + t * (f[2] - f[3]);
  double dt = (1 - s) * (f[3] - f[0]) + s * (f[2] - f[1]);
  return {v, ds, dt};
}

// Locate the (s,t) where Re=Im=0 by Newton on the bilinear field, then evaluate
// the world position. `re`,`im` are the four corner values; `px,py,pz` the four
// corner positions, all in the same cyclic order. The clamped Newton need not
// converge (near-singular Jacobians, degenerate corner data), so the iterate
// with the smallest |psi|^2 residual is the one returned -- never worse than
// the plaquette-center start, and identical to the final iterate whenever
// Newton converges.
Vec3 locateZero(const std::array<double, 4>& re, const std::array<double, 4>& im,
                const std::array<double, 4>& px, const std::array<double, 4>& py,
                const std::array<double, 4>& pz) {
  double s = 0.5, t = 0.5;
  double bs = s, bt = t;
  double bres = std::numeric_limits<double>::infinity();
  for (int it = 0; it <= 8; ++it) {
    Bilinear R = bilinear(re, s, t), I = bilinear(im, s, t);
    const double res = R.v * R.v + I.v * I.v;
    if (res < bres) {
      bres = res;
      bs = s;
      bt = t;
    }
    if (it == 8) break;
    double det = R.ds * I.dt - R.dt * I.ds;
    if (std::abs(det) < 1e-14) break;
    double ds = (-R.v * I.dt + I.v * R.dt) / det;
    double dt = (-R.ds * I.v + I.ds * R.v) / det;
    s += ds;
    t += dt;
    s = std::clamp(s, 0.0, 1.0);
    t = std::clamp(t, 0.0, 1.0);
  }
  return {bilinear(px, bs, bt).v, bilinear(py, bs, bt).v, bilinear(pz, bs, bt).v};
}

// Global plaquette ids over the cell-center lattice. Three families, laid out
// z-normal | x-normal | y-normal, so each pierced plaquette has a unique id for
// linking. (nx,ny,nz are cell counts = lattice point counts per axis.)
struct PlaquetteIds {
  int nx, ny, nz, zc, xc;
  explicit PlaquetteIds(const MacGrid& g)
      : nx(g.nx()), ny(g.ny()), nz(g.nz()),
        zc((g.nx() - 1) * (g.ny() - 1) * g.nz()),
        xc(g.nx() * (g.ny() - 1) * (g.nz() - 1)) {}
  int z(int i, int j, int k) const { return (i * (ny - 1) + j) * nz + k; }
  int x(int i, int j, int k) const { return zc + (i * (ny - 1) + j) * (nz - 1) + k; }
  int y(int i, int j, int k) const { return zc + xc + (i * ny + j) * (nz - 1) + k; }
};

// Cubes whose crossings could not be fully paired entering-with-leaving --
// only exact phase degeneracies (pi-stubs, collapsed |winding|=2 plaquettes)
// produce them, so a nonzero count flags input in need of scrutiny.
std::atomic<long> g_degenerateCubes{0};

}  // namespace

long filamentDegenerateCubeCount() { return g_degenerateCubes.load(std::memory_order_relaxed); }
void resetFilamentDegenerateCubeCount() { g_degenerateCubes.store(0, std::memory_order_relaxed); }

int plaquetteNormalAxis(const MacGrid& g, int plaquette) {
  const PlaquetteIds ids(g);
  const int yc = g.nx() * g.ny() * (g.nz() - 1);
  if (plaquette < 0 || plaquette >= ids.zc + ids.xc + yc)
    throw std::invalid_argument("plaquetteNormalAxis: id out of range");
  if (plaquette < ids.zc) return 2;
  return plaquette < ids.zc + ids.xc ? 0 : 1;
}

std::vector<FilamentCrossing> traceZeroSet(const MacGrid& g, const std::vector<double>& psi) {
  if (psi.size() != static_cast<std::size_t>(2 * g.numCells()))
    throw std::invalid_argument("traceZeroSet: psi must have length 2*numCells");

  std::vector<FilamentCrossing> out;
  const PlaquetteIds ids(g);
  auto re = [&](int c) { return psi[2 * c]; };
  auto im = [&](int c) { return psi[2 * c + 1]; };

  // For four corner cells in cyclic order with their positions and the
  // plaquette id, test the plaquette and emit a crossing if psi winds around it.
  auto emit = [&](int plaq, std::array<int, 4> cells, std::array<Vec3, 4> pos) {
    std::array<double, 4> phi, R, I, px, py, pz;
    for (int n = 0; n < 4; ++n) {
      R[n] = re(cells[n]);
      I[n] = im(cells[n]);
      // An exactly-zero cell has no phase (atan2(0,0) fabricates 0). Pin it to
      // +0 real explicitly -- phase 0, same value seen by every plaquette
      // sharing the cell, so winding conservation still telescopes over cube
      // faces -- and keep the bilinear Newton field nonzero at the corner.
      if (R[n] == 0.0 && I[n] == 0.0) R[n] = std::numeric_limits<double>::min();
      phi[n] = std::atan2(I[n], R[n]);
      px[n] = pos[n][0];
      py[n] = pos[n][1];
      pz[n] = pos[n][2];
    }
    int w = winding(phi);
    if (w == 0) return;
    out.push_back({locateZero(R, I, px, py, pz), w > 0 ? 1 : -1, plaq});
  };

  auto C = [&](int i, int j, int k) { return g.cellIndex(i, j, k); };
  auto P = [&](int i, int j, int k) { return g.cellCenter(i, j, k); };

  // z-normal plaquettes (xy square), corners CCW about +z.
  for (int i = 0; i + 1 < g.nx(); ++i)
    for (int j = 0; j + 1 < g.ny(); ++j)
      for (int k = 0; k < g.nz(); ++k)
        emit(ids.z(i, j, k), {C(i, j, k), C(i + 1, j, k), C(i + 1, j + 1, k), C(i, j + 1, k)},
             {P(i, j, k), P(i + 1, j, k), P(i + 1, j + 1, k), P(i, j + 1, k)});

  // x-normal plaquettes (yz square), corners CCW about +x.
  for (int i = 0; i < g.nx(); ++i)
    for (int j = 0; j + 1 < g.ny(); ++j)
      for (int k = 0; k + 1 < g.nz(); ++k)
        emit(ids.x(i, j, k), {C(i, j, k), C(i, j + 1, k), C(i, j + 1, k + 1), C(i, j, k + 1)},
             {P(i, j, k), P(i, j + 1, k), P(i, j + 1, k + 1), P(i, j, k + 1)});

  // y-normal plaquettes (zx square), corners CCW about +y.
  for (int i = 0; i + 1 < g.nx(); ++i)
    for (int j = 0; j < g.ny(); ++j)
      for (int k = 0; k + 1 < g.nz(); ++k)
        emit(ids.y(i, j, k), {C(i, j, k), C(i, j, k + 1), C(i + 1, j, k + 1), C(i + 1, j, k)},
             {P(i, j, k), P(i, j, k + 1), P(i + 1, j, k + 1), P(i + 1, j, k)});

  return out;
}

std::vector<Filament> linkFilaments(const MacGrid& g,
                                    const std::vector<FilamentCrossing>& crossings) {
  const int n = static_cast<int>(crossings.size());
  const PlaquetteIds ids(g);

  // plaquette id -> crossing index.
  std::unordered_map<int, int> byPlaq;
  byPlaq.reserve(crossings.size() * 2);
  for (int c = 0; c < n; ++c) byPlaq.emplace(crossings[c].plaquette, c);

  std::vector<std::vector<int>> adj(n);
  auto distance2 = [&](int a, int b) {
    Vec3 d = vsub(crossings[a].point, crossings[b].point);
    return vdot(d, d);
  };

  // For each cube, gather the crossings on its 6 plaquettes and pair them.
  // Winding conservation over the closed cube surface splits the crossings
  // into equal entering/leaving counts, and a physical strand joins one of
  // each -- so only in<->out pairs are admissible, closest first (a distance
  // tie-break alone can swap strands when two share a cube). With one strand
  // -- one in, one out -- this reduces to the unique pair as before.
  for (int i = 0; i + 1 < g.nx(); ++i)
    for (int j = 0; j + 1 < g.ny(); ++j)
      for (int k = 0; k + 1 < g.nz(); ++k) {
        // Faces come in low/high pairs per axis: on a high face (odd index)
        // the plaquette's +axis winding normal points out of the cube, on a
        // low face into it -- so orientation * side gives +1 = strand leaves
        // the cube through this face, -1 = enters.
        const int plaqs[6] = {ids.z(i, j, k),     ids.z(i, j, k + 1), ids.x(i, j, k),
                              ids.x(i + 1, j, k), ids.y(i, j, k),     ids.y(i, j + 1, k)};
        std::vector<int> present;
        std::vector<int> flow;
        for (int f = 0; f < 6; ++f) {
          auto it = byPlaq.find(plaqs[f]);
          if (it == byPlaq.end()) continue;
          present.push_back(it->second);
          flow.push_back(crossings[it->second].orientation * ((f & 1) ? 1 : -1));
        }
        // Greedily connect the two closest admissible crossings until <2
        // remain. If in<->out pairing cannot complete (odd or same-sense
        // leftovers -- only exact phase degeneracies produce them), fall back
        // to distance-only pairing so the walk still sees degree <= 2.
        std::vector<char> used(present.size(), 0);
        int remaining = static_cast<int>(present.size());
        bool inOutOnly = true;
        bool degenerate = false;
        while (remaining >= 2) {
          double best = -1.0;
          int ba = -1, bb = -1;
          for (size_t a = 0; a < present.size(); ++a) {
            if (used[a]) continue;
            for (size_t b = a + 1; b < present.size(); ++b) {
              if (used[b]) continue;
              if (inOutOnly && flow[a] + flow[b] != 0) continue;
              double d2 = distance2(present[a], present[b]);
              if (best < 0.0 || d2 < best) {
                best = d2;
                ba = static_cast<int>(a);
                bb = static_cast<int>(b);
              }
            }
          }
          if (ba < 0) {
            if (!inOutOnly) break;
            inOutOnly = false;
            degenerate = true;
            continue;
          }
          adj[present[ba]].push_back(present[bb]);
          adj[present[bb]].push_back(present[ba]);
          used[ba] = used[bb] = 1;
          remaining -= 2;
        }
        if (degenerate || remaining != 0)
          g_degenerateCubes.fetch_add(1, std::memory_order_relaxed);
      }

  // Walk the degree-<=2 graph into polylines. Endpoints (degree 1) start open
  // paths; whatever remains forms closed loops.
  std::vector<char> visited(n, 0);
  std::vector<Filament> result;
  auto walk = [&](int start, bool closed) {
    Filament f;
    f.closed = closed;
    int prev = -1, cur = start;
    while (cur != -1 && !visited[cur]) {
      visited[cur] = 1;
      f.points.push_back(crossings[cur].point);
      int next = -1;
      for (int nb : adj[cur])
        if (nb != prev && !visited[nb]) {
          next = nb;
          break;
        }
      prev = cur;
      cur = next;
    }
    result.push_back(std::move(f));
  };

  for (int c = 0; c < n; ++c)
    if (!visited[c] && adj[c].size() == 1) walk(c, /*closed=*/false);
  for (int c = 0; c < n; ++c)
    if (!visited[c]) walk(c, /*closed=*/adj[c].size() == 2);

  return result;
}

}  // namespace bochner
