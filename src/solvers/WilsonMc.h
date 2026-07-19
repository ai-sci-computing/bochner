#pragma once

#include <complex>
#include <cstdint>
#include <random>
#include <vector>

#include "solvers/SunGauge.h"

namespace bochner {

/// \file
/// \brief Pure-gauge Wilson-action Monte Carlo sampler on the periodic 3D
/// \ref SunLattice -- the "hard middle" between the paper's two easy extremes
/// (smooth fields with flux -> 0, and i.i.d. hot links): Monte-Carlo-sampled
/// configurations at finite coupling are ROUGH but CORRELATED, which is the
/// regime a referee would probe.
///
/// Action (SU(d), d in {2,3}):  S = -(beta/d) * sum_p Re tr U_p,  where the sum
/// runs over all unoriented plaquettes of the periodic n^3 lattice (3 per site).
/// With this normalization the leading strong-coupling prediction for the
/// average plaquette  P = <(1/d) Re tr U_p>  is  P ~ beta/(2 d^2)  (= beta/18
/// for SU(3), beta/4 for SU(2)) -- pinned by tests/test_mc_gauge.cpp.
///
/// Sampling: Cabibbo-Marinari sweep over the SU(2) subgroups (all (p,q) pairs),
/// each subgroup updated by the exact Kennedy-Pendleton SU(2) heatbath (with a
/// semicircle-rejection fallback at small effective coupling, where KP's
/// acceptance degrades). Each link sees the sum of its 4 staples in 3D (2
/// orthogonal planes x 2 orientations). Everything is deterministic in the
/// caller-provided std::mt19937_64.

/// One full heatbath sweep (all links, all SU(2) subgroups once) of the Wilson
/// action at inverse coupling \p beta. Requires a periodic lattice with
/// d in {2,3}. \throws std::invalid_argument otherwise.
void wilsonHeatbathSweep(SunLattice& lat, double beta, std::mt19937_64& rng);

/// Average plaquette  <(1/d) Re tr U_p>  over all 3*N plaquettes of the
/// periodic lattice. 1 on a trivial (identity/pure-gauge) configuration; 0 in
/// expectation on i.i.d. Haar links.
double averagePlaquette(const SunLattice& lat);

/// Project every link back onto SU(d) (modified Gram-Schmidt on the columns +
/// det-phase fix on the last column) -- call periodically against roundoff
/// drift accumulated by repeated left-multiplications in the heatbath.
void reunitarizeSun(SunLattice& lat);

/// Haar-random SU(d) matrix (d in {1,2,3}) written to \p M (row-major d*d).
void haarSuMatrix(int d, std::mt19937_64& rng, std::complex<double>* M);

/// Random gauge transformation, deterministic in \p seed: a Haar-random
/// g_c in SU(d) per node, links  U_mu(c) -> g_{c+mu} U_mu(c) g_c^H  (the
/// transformation under which the covariant Laplacian is unitarily equivalent,
/// x_c -> g_c x_c). Spectrum-preserving by construction -- the validation
/// tests use it to pin gauge covariance of the solvers.
void gaugeTransformSun(SunLattice& lat, std::uint64_t seed);

/// \brief Thermalized Monte-Carlo gauge configuration: hot (i.i.d. Haar) or
/// cold (identity) start, then \p sweeps heatbath sweeps, reunitarizing every
/// \p reunitEvery sweeps (and once at the end). Deterministic in \p seed.
///
/// \p w / \p mass2 are the connection-Laplacian weight and mass installed on
/// the returned lattice (the MC dynamics itself does not use them).
/// If \p plaqHistory is non-null, the average plaquette after every sweep is
/// appended to it (the thermalization diagnostic: flat over the last half).
/// beta = 0 with sweeps = 0 reproduces exactly the existing hot-link path
/// (randomSunLattice at the same seed).
SunLattice mcSunLattice(int d, int n, double beta, int sweeps, double w, double mass2,
                        std::uint64_t seed, bool hotStart = true,
                        std::vector<double>* plaqHistory = nullptr, int reunitEvery = 50);

}  // namespace bochner
