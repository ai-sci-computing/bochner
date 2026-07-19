#include "fluid/MacProjection.h"

#include <cstddef>
#include <unordered_set>
#include <vector>

// Pure assembly of the pressure Poisson operators (CooMatrix from grid data).
// Deliberately PETSc-free: tests and tools inspect these operators in builds
// without BOCHNER_WITH_PETSC; the projection SOLVERS live in MacProjection.cpp.

namespace bochner {

namespace {

// Accumulate the graph-Laplacian contribution of one interior face that couples
// cells `a` and `b` with weight `w`. Rows/columns of pinned (Dirichlet) cells
// are dropped (their phi is known = 0), so the coupling only touches free cells.
void addFaceCoupling(CooMatrix& A, int a, int b, double w,
                     const std::unordered_set<int>& pinned) {
  const bool pa = pinned.count(a) != 0;
  const bool pb = pinned.count(b) != 0;
  if (!pa) A.add(a, a, w);
  if (!pb) A.add(b, b, w);
  if (!pa && !pb) {
    A.add(a, b, -w);
    A.add(b, a, -w);
  }
}

}  // namespace

CooMatrix pressureLaplacianObstacle(const MacGrid& g, const BoundarySpec& bc, const SolidMask& solid,
                                    std::vector<int>* pinnedOut) {
  const int n = g.numCells();
  const double w = 1.0 / (g.spacing() * g.spacing());
  const int nx = g.nx(), ny = g.ny(), nz = g.nz();

  // A fluid cell on an OPEN wall is grounded by the Robin ghost diagonal added
  // below, so its connected component needs no pin. Mark those up front.
  std::vector<char> grounded(n, 0);
  auto markOpen = [&](int c) {
    if (!isSolid(solid, c)) grounded[c] = 1;
  };
  if (bc.xlo)
    for (int j = 0; j < ny; ++j)
      for (int k = 0; k < nz; ++k) markOpen(g.cellIndex(0, j, k));
  if (bc.xhi)
    for (int j = 0; j < ny; ++j)
      for (int k = 0; k < nz; ++k) markOpen(g.cellIndex(nx - 1, j, k));
  if (bc.ylo)
    for (int i = 0; i < nx; ++i)
      for (int k = 0; k < nz; ++k) markOpen(g.cellIndex(i, 0, k));
  if (bc.yhi)
    for (int i = 0; i < nx; ++i)
      for (int k = 0; k < nz; ++k) markOpen(g.cellIndex(i, ny - 1, k));
  if (bc.zlo)
    for (int i = 0; i < nx; ++i)
      for (int j = 0; j < ny; ++j) markOpen(g.cellIndex(i, j, 0));
  if (bc.zhi)
    for (int i = 0; i < nx; ++i)
      for (int j = 0; j < ny; ++j) markOpen(g.cellIndex(i, j, nz - 1));

  // Every fluid connected component (through fluid-fluid faces) needs exactly one
  // pressure ground or its all-Neumann block is singular. Flood-fill the fluid;
  // pin one cell of any component that touches no open wall -- the whole domain
  // when no wall is open, or a fluid pocket sealed off by solid (a concave/shell
  // obstacle; the convex shipped masks never seal one). This subsumes both the
  // fully-closed single-cell pin and the isolated-cell pin.
  std::unordered_set<int> pinned;
  std::vector<char> seen(n, 0);
  std::vector<int> stack;
  for (int s = 0; s < n; ++s) {
    if (isSolid(solid, s) || seen[s]) continue;
    stack.clear();
    stack.push_back(s);
    seen[s] = 1;
    bool anyGround = false;
    for (std::size_t head = 0; head < stack.size(); ++head) {
      const int c = stack[head];
      anyGround = anyGround || grounded[c] != 0;
      const int i = c / (ny * nz), j = (c / nz) % ny, k = c % nz;
      auto visit = [&](int nc) {
        if (!isSolid(solid, nc) && !seen[nc]) {
          seen[nc] = 1;
          stack.push_back(nc);
        }
      };
      if (i > 0) visit(g.cellIndex(i - 1, j, k));
      if (i + 1 < nx) visit(g.cellIndex(i + 1, j, k));
      if (j > 0) visit(g.cellIndex(i, j - 1, k));
      if (j + 1 < ny) visit(g.cellIndex(i, j + 1, k));
      if (k > 0) visit(g.cellIndex(i, j, k - 1));
      if (k + 1 < nz) visit(g.cellIndex(i, j, k + 1));
    }
    if (!anyGround) pinned.insert(s);  // s is the min-index cell of this component
  }

  CooMatrix A(n, n);
  auto couple = [&](int a, int b) {
    if (isSolid(solid, a) || isSolid(solid, b)) return;  // solid interface: Neumann, no coupling
    addFaceCoupling(A, a, b, w, pinned);
  };
  for (int i = 1; i < nx; ++i)
    for (int j = 0; j < ny; ++j)
      for (int k = 0; k < nz; ++k) couple(g.cellIndex(i - 1, j, k), g.cellIndex(i, j, k));
  for (int i = 0; i < nx; ++i)
    for (int j = 1; j < ny; ++j)
      for (int k = 0; k < nz; ++k) couple(g.cellIndex(i, j - 1, k), g.cellIndex(i, j, k));
  for (int i = 0; i < nx; ++i)
    for (int j = 0; j < ny; ++j)
      for (int k = 1; k < nz; ++k) couple(g.cellIndex(i, j, k - 1), g.cellIndex(i, j, k));

  auto openDiag = [&](int c) {
    if (!isSolid(solid, c) && pinned.count(c) == 0) A.add(c, c, w);
  };
  if (bc.xlo)
    for (int j = 0; j < ny; ++j)
      for (int k = 0; k < nz; ++k) openDiag(g.cellIndex(0, j, k));
  if (bc.xhi)
    for (int j = 0; j < ny; ++j)
      for (int k = 0; k < nz; ++k) openDiag(g.cellIndex(nx - 1, j, k));
  if (bc.ylo)
    for (int i = 0; i < nx; ++i)
      for (int k = 0; k < nz; ++k) openDiag(g.cellIndex(i, 0, k));
  if (bc.yhi)
    for (int i = 0; i < nx; ++i)
      for (int k = 0; k < nz; ++k) openDiag(g.cellIndex(i, ny - 1, k));
  if (bc.zlo)
    for (int i = 0; i < nx; ++i)
      for (int j = 0; j < ny; ++j) openDiag(g.cellIndex(i, j, 0));
  if (bc.zhi)
    for (int i = 0; i < nx; ++i)
      for (int j = 0; j < ny; ++j) openDiag(g.cellIndex(i, j, nz - 1));

  for (int c = 0; c < n; ++c)
    if (isSolid(solid, c)) A.add(c, c, 1.0);  // solid cell: unit diagonal, phi=0
  for (int c : pinned) A.add(c, c, 1.0);
  if (pinnedOut) pinnedOut->assign(pinned.begin(), pinned.end());
  return A;
}

CooMatrix pressureLaplacian(const MacGrid& g, const std::vector<int>& dirichletCells) {
  const int n = g.numCells();
  const double w = 1.0 / (g.spacing() * g.spacing());
  const std::unordered_set<int> pinned(dirichletCells.begin(), dirichletCells.end());

  CooMatrix A(n, n);
  // Interior x-faces couple cell (i-1,j,k) to (i,j,k); likewise y and z.
  for (int i = 1; i < g.nx(); ++i)
    for (int j = 0; j < g.ny(); ++j)
      for (int k = 0; k < g.nz(); ++k)
        addFaceCoupling(A, g.cellIndex(i - 1, j, k), g.cellIndex(i, j, k), w, pinned);
  for (int i = 0; i < g.nx(); ++i)
    for (int j = 1; j < g.ny(); ++j)
      for (int k = 0; k < g.nz(); ++k)
        addFaceCoupling(A, g.cellIndex(i, j - 1, k), g.cellIndex(i, j, k), w, pinned);
  for (int i = 0; i < g.nx(); ++i)
    for (int j = 0; j < g.ny(); ++j)
      for (int k = 1; k < g.nz(); ++k)
        addFaceCoupling(A, g.cellIndex(i, j, k - 1), g.cellIndex(i, j, k), w, pinned);

  // Unit diagonal for each pinned cell (its row/column is otherwise empty).
  for (int c : pinned) A.add(c, c, 1.0);
  return A;
}

CooMatrix pressureLaplacianBC(const MacGrid& g, const BoundarySpec& bc) {
  const int n = g.numCells();
  const double w = 1.0 / (g.spacing() * g.spacing());
  // Fully closed -> singular pure-Neumann: pin cell 0. Any open wall -> Dirichlet
  // ghost makes it SPD, so no pin.
  const std::unordered_set<int> pinned =
      bc.anyOpen() ? std::unordered_set<int>{} : std::unordered_set<int>{0};

  CooMatrix A(n, n);
  for (int i = 1; i < g.nx(); ++i)
    for (int j = 0; j < g.ny(); ++j)
      for (int k = 0; k < g.nz(); ++k)
        addFaceCoupling(A, g.cellIndex(i - 1, j, k), g.cellIndex(i, j, k), w, pinned);
  for (int i = 0; i < g.nx(); ++i)
    for (int j = 1; j < g.ny(); ++j)
      for (int k = 0; k < g.nz(); ++k)
        addFaceCoupling(A, g.cellIndex(i, j - 1, k), g.cellIndex(i, j, k), w, pinned);
  for (int i = 0; i < g.nx(); ++i)
    for (int j = 0; j < g.ny(); ++j)
      for (int k = 1; k < g.nz(); ++k)
        addFaceCoupling(A, g.cellIndex(i, j, k - 1), g.cellIndex(i, j, k), w, pinned);

  // Open wall: each boundary cell couples to a p=0 ghost -> +w on its diagonal.
  auto openDiag = [&](int c) {
    if (pinned.count(c) == 0) A.add(c, c, w);
  };
  if (bc.xlo)
    for (int j = 0; j < g.ny(); ++j)
      for (int k = 0; k < g.nz(); ++k) openDiag(g.cellIndex(0, j, k));
  if (bc.xhi)
    for (int j = 0; j < g.ny(); ++j)
      for (int k = 0; k < g.nz(); ++k) openDiag(g.cellIndex(g.nx() - 1, j, k));
  if (bc.ylo)
    for (int i = 0; i < g.nx(); ++i)
      for (int k = 0; k < g.nz(); ++k) openDiag(g.cellIndex(i, 0, k));
  if (bc.yhi)
    for (int i = 0; i < g.nx(); ++i)
      for (int k = 0; k < g.nz(); ++k) openDiag(g.cellIndex(i, g.ny() - 1, k));
  if (bc.zlo)
    for (int i = 0; i < g.nx(); ++i)
      for (int j = 0; j < g.ny(); ++j) openDiag(g.cellIndex(i, j, 0));
  if (bc.zhi)
    for (int i = 0; i < g.nx(); ++i)
      for (int j = 0; j < g.ny(); ++j) openDiag(g.cellIndex(i, j, g.nz() - 1));

  for (int c : pinned) A.add(c, c, 1.0);
  return A;
}

}  // namespace bochner
