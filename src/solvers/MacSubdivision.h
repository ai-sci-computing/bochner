#pragma once

#include <vector>

#include "grid/GridOperators.h"
#include "grid/MacGrid.h"

namespace bochner {

/// \brief Build a smooth section \f$\psi\f$ of the U(1) connection by
/// **gauge-aware subdivision** -- no linear system is solved.
///
/// The section is built by parallel transport alone (covariant subdivision,
/// here in our dual-0-form placement), and is the candidate fast
/// path for vortex-filament extraction: instead of solving for the smallest
/// eigenvector of the connection Laplacian (the per-frame bottleneck), we
/// *prolong* a constant section from a coarse lattice up to the full grid using
/// parallel transport. The result is not the ground state, but where the
/// connection has holonomy the transported averages interfere destructively and
/// \f$|\psi|\f$ pinches to zero -- so the **zero set still decomposes into
/// vortex rings**, which is all the extractor needs.
///
/// ### The lattice and its connection
/// The section lives at cell centers (dual 0-form); cell centers
/// form a regular 3D lattice whose nearest-neighbour edges are the interior MAC
/// faces, each carrying a parallel transport \f$e^{i\theta_f}\f$ (the link
/// variable). Transport from the low cell to the high cell across a face is
/// \f$e^{+i\theta_f}\f$, matching \ref connectionLaplacian (where
/// \f$(d^\nabla\psi)_f = \psi_b - e^{i\theta_f}\psi_a\f$). \p theta is the
/// per-face angle (see \ref connectionAngles); only interior faces are used.
///
/// ### The multigrid hierarchy (decimation, so levels nest exactly)
/// Coarsening halves the lattice along each axis by keeping every other node
/// (the even-indexed ones); a coarse node coincides with a fine node, so the
/// levels nest with no geometric interpolation error. A coarse edge spans two
/// fine edges through the dropped node, so its transport is the composition of
/// the two -- i.e. the **coarse link angle is the sum of the two fine link
/// angles** along it (a U(1) connection restricts by summing links). The finest
/// level is the full grid \p g; there are \p numLevels coarsenings, so every
/// grid dimension must be divisible by \f$2^{\text{numLevels}}\f$.
///
/// ### Prolongation = one covariant-averaging rule
/// Seed the coarsest lattice with \f$\psi\equiv 1\f$, then prolong up one level
/// at a time. Going to a finer level, classify each fine node by how many of its
/// coordinates are odd (i.e. how many axes it was *not* inherited along):
///   - **0 odd** -- it is a retained coarse node: copy its value.
///   - **k odd** (k = 1,2,3) -- it is new along those k axes; set it to the
///     average of its immediate neighbours along each odd axis (2k of them, or
///     fewer at the boundary), **each parallel-transported to this node**:
///     \f[ \psi_m = \frac{1}{\#N}\sum_{n\in N} P^\nabla_{n\to m}\,\psi_n. \f]
/// Processing the passes in order k = 1, 2, 3 guarantees every neighbour used is
/// already filled (a k-odd node's neighbours along an odd axis are (k-1)-odd).
/// This single rule reproduces the edge-/face-/cube-midpoint averaging of the
/// original cube formulation: 1-odd = edge midpoint (2 neighbours), 2-odd = face
/// midpoint (4), 3-odd = cube centre (6).
///
/// \param g         The finest grid (cell-centre lattice).
/// \param theta     Per-face connection angles on \p g (interior faces used).
/// \param numLevels Number of coarsenings; each grid dim must be divisible by
///                  \f$2^{\text{numLevels}}\f$. `numLevels = 0` returns the
///                  constant section.
/// \returns The section interleaved as `[Re psi_0, Im psi_0, Re psi_1, ...]` of
///          length `2 * g.numCells()` -- the exact layout \ref traceZeroSet and
///          \ref smallestEigenpair use, so it can be traced or used as a warm
///          start directly.
/// \throws std::invalid_argument if \p numLevels < 0 or a grid dimension is not
///         divisible by \f$2^{\text{numLevels}}\f$.
std::vector<double> subdivisionSection(const MacGrid& g, const FaceField& theta, int numLevels);

}  // namespace bochner
