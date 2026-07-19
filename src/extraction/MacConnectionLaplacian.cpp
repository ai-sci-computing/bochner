#include "extraction/MacConnectionLaplacian.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace bochner {

CooMatrix connectionLaplacian(const MacGrid& g, const FaceField& theta) {
  const int n = g.numCells();
  const double w = 1.0 / (g.spacing() * g.spacing());
  CooMatrix L(2 * n, 2 * n);

  // Place a complex value (re + i*im) at logical (j,k) as the 2x2 real block
  // [[re,-im],[im,re]] over real rows/cols {2j,2j+1} x {2k,2k+1}.
  auto addBlock = [&](int j, int k, double re, double im) {
    L.add(2 * j, 2 * k, re);
    L.add(2 * j, 2 * k + 1, -im);
    L.add(2 * j + 1, 2 * k, im);
    L.add(2 * j + 1, 2 * k + 1, re);
  };

  // One graph edge a--b with connection angle th: diagonal weight w on both
  // endpoints; off-diagonals E_ab = -w e^{-i th}, E_ba = conj(E_ab).
  auto link = [&](int a, int b, double th) {
    addBlock(a, a, w, 0.0);
    addBlock(b, b, w, 0.0);
    const double c = std::cos(th), s = std::sin(th);
    addBlock(a, b, -w * c, w * s);
    addBlock(b, a, -w * c, -w * s);
  };

  for (int i = 1; i < g.nx(); ++i)  // interior x-faces link cells (i-1) -- (i)
    for (int j = 0; j < g.ny(); ++j)
      for (int k = 0; k < g.nz(); ++k)
        link(g.cellIndex(i - 1, j, k), g.cellIndex(i, j, k), theta.x[g.faceXIndex(i, j, k)]);
  for (int i = 0; i < g.nx(); ++i)  // interior y-faces
    for (int j = 1; j < g.ny(); ++j)
      for (int k = 0; k < g.nz(); ++k)
        link(g.cellIndex(i, j - 1, k), g.cellIndex(i, j, k), theta.y[g.faceYIndex(i, j, k)]);
  for (int i = 0; i < g.nx(); ++i)  // interior z-faces
    for (int j = 0; j < g.ny(); ++j)
      for (int k = 1; k < g.nz(); ++k)
        link(g.cellIndex(i, j, k - 1), g.cellIndex(i, j, k), theta.z[g.faceZIndex(i, j, k)]);

  return L;
}

double maxConnectionAngle(const FaceField& theta) {
  double m = 0.0;
  for (const double v : theta.x) m = std::max(m, std::abs(v));
  for (const double v : theta.y) m = std::max(m, std::abs(v));
  for (const double v : theta.z) m = std::max(m, std::abs(v));
  return m;
}

double hbarAliasingFloor(const MacGrid& g, const FaceField& u) {
  double umax = 0.0;
  for (const double v : u.x) umax = std::max(umax, std::abs(v));
  for (const double v : u.y) umax = std::max(umax, std::abs(v));
  for (const double v : u.z) umax = std::max(umax, std::abs(v));
  return g.spacing() * umax / M_PI;
}

FaceField connectionAngles(const MacGrid& g, const FaceField& u, double hbar) {
  if (!(hbar > 0.0)) throw std::invalid_argument("connectionAngles: hbar must be positive");
  const double scale = g.spacing() / hbar;
  FaceField theta = u;
  for (double& v : theta.x) v *= scale;
  for (double& v : theta.y) v *= scale;
  for (double& v : theta.z) v *= scale;
  return theta;
}

}  // namespace bochner
