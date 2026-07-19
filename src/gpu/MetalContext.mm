// Metal GPU compute backend implementation (Objective-C++). All ObjC/Metal
// interop is isolated here behind the pure-C++ MetalContext.h. Shaders are
// compiled from the embedded MSL source at runtime (newLibraryWithSource:), so
// only the Metal framework is needed -- no offline `metal` compiler.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "gpu/MetalContext.h"

namespace bochner::gpu {
namespace {

// Grid parameters passed to the kernels. All fields are 4 bytes and 4-aligned,
// so this struct has the same layout as its MSL twin below (no internal pad).
struct GridParams {
  int nx, ny, nz;
  float h, ox, oy, oz;
};

// Embedded Metal Shading Language. `trilerp` mirrors trilerpComponent() in
// MacAdvection.cpp exactly, on the clamped ("streak") path used for velocity;
// `sampleVel` assembles the three staggered components (== sampleVelocity), and
// `backtrace` is classic RK4 over v (== bochner::backtrace). The flat index
// (i*nj+j)*nk+k matches MacGrid::faceX/Y/ZIndex for the lattice (ni,nj,nk).
const char* kSource = R"METAL(
#include <metal_stdlib>
using namespace metal;

struct GridParams { int nx, ny, nz; float h, ox, oy, oz; };

static float trilerp(device const float* d, int ni, int nj, int nk,
                     float offx, float offy, float offz,
                     float3 p, constant GridParams& g) {
  float cx = (p.x - g.ox) / g.h - offx;
  float cy = (p.y - g.oy) / g.h - offy;
  float cz = (p.z - g.oz) / g.h - offz;
  cx = clamp(cx, 0.0f, float(ni - 1));
  cy = clamp(cy, 0.0f, float(nj - 1));
  cz = clamp(cz, 0.0f, float(nk - 1));
  int i0 = clamp(int(floor(cx)), 0, max(0, ni - 2));
  int j0 = clamp(int(floor(cy)), 0, max(0, nj - 2));
  int k0 = clamp(int(floor(cz)), 0, max(0, nk - 2));
  float tx = cx - float(i0);
  float ty = cy - float(j0);
  float tz = cz - float(k0);
  if (ni == 1) { i0 = 0; tx = 0.0f; }
  if (nj == 1) { j0 = 0; ty = 0.0f; }
  if (nk == 1) { k0 = 0; tz = 0.0f; }
  int i1 = min(i0 + 1, ni - 1);
  int j1 = min(j0 + 1, nj - 1);
  int k1 = min(k0 + 1, nk - 1);
  int b00 = (i0 * nj + j0) * nk, b10 = (i1 * nj + j0) * nk;
  int b01 = (i0 * nj + j1) * nk, b11 = (i1 * nj + j1) * nk;
  float c000 = d[b00 + k0], c100 = d[b10 + k0];
  float c010 = d[b01 + k0], c110 = d[b11 + k0];
  float c001 = d[b00 + k1], c101 = d[b10 + k1];
  float c011 = d[b01 + k1], c111 = d[b11 + k1];
  float c00 = mix(c000, c100, tx), c10 = mix(c010, c110, tx);
  float c01 = mix(c001, c101, tx), c11 = mix(c011, c111, tx);
  float c0 = mix(c00, c10, ty), c1 = mix(c01, c11, ty);
  return mix(c0, c1, tz);
}

static float3 sampleVel(device const float* ux, device const float* uy, device const float* uz,
                        float3 p, constant GridParams& g) {
  return float3(trilerp(ux, g.nx + 1, g.ny, g.nz, 0.0f, 0.5f, 0.5f, p, g),
                trilerp(uy, g.nx, g.ny + 1, g.nz, 0.5f, 0.0f, 0.5f, p, g),
                trilerp(uz, g.nx, g.ny, g.nz + 1, 0.5f, 0.5f, 0.0f, p, g));
}

// Classic RK4 for dx/dt = v integrated backward over dt (== bochner::backtrace).
static float3 rk4Backtrace(device const float* vx, device const float* vy, device const float* vz,
                           float3 s, float dt, constant GridParams& g) {
  float3 k1 = sampleVel(vx, vy, vz, s, g);
  float3 k2 = sampleVel(vx, vy, vz, s - 0.5f * dt * k1, g);
  float3 k3 = sampleVel(vx, vy, vz, s - 0.5f * dt * k2, g);
  float3 k4 = sampleVel(vx, vy, vz, s - dt * k3, g);
  return s - (dt / 6.0f) * (k1 + 2.0f * k2 + 2.0f * k3 + k4);
}

kernel void sample_velocity(device float* out        [[buffer(0)]],  // 3*n
                            device const float* pts   [[buffer(1)]],  // 3*n
                            device const float* ux    [[buffer(2)]],
                            device const float* uy    [[buffer(3)]],
                            device const float* uz    [[buffer(4)]],
                            constant GridParams& g    [[buffer(5)]],
                            constant uint& n          [[buffer(6)]],
                            uint tid [[thread_position_in_grid]]) {
  if (tid >= n) return;
  float3 v = sampleVel(ux, uy, uz, float3(pts[3 * tid], pts[3 * tid + 1], pts[3 * tid + 2]), g);
  out[3 * tid] = v.x; out[3 * tid + 1] = v.y; out[3 * tid + 2] = v.z;
}

kernel void backtrace(device float* out          [[buffer(0)]],  // 3*n
                      device const float* starts  [[buffer(1)]],  // 3*n
                      device const float* vx       [[buffer(2)]],
                      device const float* vy       [[buffer(3)]],
                      device const float* vz       [[buffer(4)]],
                      constant GridParams& g       [[buffer(5)]],
                      constant uint& n             [[buffer(6)]],
                      constant float& dt           [[buffer(7)]],
                      uint tid [[thread_position_in_grid]]) {
  if (tid >= n) return;
  float3 s = float3(starts[3 * tid], starts[3 * tid + 1], starts[3 * tid + 2]);
  float3 r = rk4Backtrace(vx, vy, vz, s, dt, g);
  out[3 * tid] = r.x; out[3 * tid + 1] = r.y; out[3 * tid + 2] = r.z;
}

// Semi-Lagrangian covector advection (== advectCovectorSL), two kernels.
// (1) sl_cells: backtrace every cell center through v -> the flow map psiCell.
kernel void sl_cells(device float* psi            [[buffer(0)]],  // 3*numCells
                     device const float* vx        [[buffer(1)]],
                     device const float* vy        [[buffer(2)]],
                     device const float* vz        [[buffer(3)]],
                     constant GridParams& g        [[buffer(4)]],
                     constant float& dt            [[buffer(5)]],
                     uint tid [[thread_position_in_grid]]) {
  int total = g.nx * g.ny * g.nz;
  if (int(tid) >= total) return;
  int i = int(tid) / (g.ny * g.nz);
  int rem = int(tid) % (g.ny * g.nz);
  int j = rem / g.nz, k = rem % g.nz;
  float3 cc = float3(g.ox + (i + 0.5f) * g.h, g.oy + (j + 0.5f) * g.h, g.oz + (k + 0.5f) * g.h);
  float3 r = rk4Backtrace(vx, vy, vz, cc, dt, g);
  psi[3 * tid] = r.x; psi[3 * tid + 1] = r.y; psi[3 * tid + 2] = r.z;
}

// Face-family (0=x,1=y,2=z) lattice dimensions for the staggered arrays.
static void familyDims(int axis, constant GridParams& g, thread int& ni, thread int& nj,
                       thread int& nk) {
  if (axis == 0) { ni = g.nx + 1; nj = g.ny; nk = g.nz; }
  else if (axis == 1) { ni = g.nx; nj = g.ny + 1; nk = g.nz; }
  else { ni = g.nx; nj = g.ny; nk = g.nz + 1; }
}

// True if face (i,j,k) of family `axis` is a domain-boundary face (skipped by the
// interior operators, so it keeps whatever the out buffer was prefilled with).
static bool boundaryFace(int axis, int i, int j, int k, constant GridParams& g) {
  return (axis == 0 && (i == 0 || i == g.nx)) || (axis == 1 && (j == 0 || j == g.ny)) ||
         (axis == 2 && (k == 0 || k == g.nz));
}

// (2) sl_pullback: for each interior face of family `axis` (0=x,1=y,2=z), the
// transpose-Jacobian column dPsi/dn (finite difference of psiCell across the
// face) dotted with u sampled at the face center's own v-backtrace. Boundary
// faces are skipped (the out buffer is prefilled with u).
kernel void sl_pullback(device float* out           [[buffer(0)]],   // this family's face array
                        device const float* psi       [[buffer(1)]],  // 3*numCells
                        device const float* ux        [[buffer(2)]],
                        device const float* uy        [[buffer(3)]],
                        device const float* uz        [[buffer(4)]],
                        device const float* vx        [[buffer(5)]],
                        device const float* vy        [[buffer(6)]],
                        device const float* vz        [[buffer(7)]],
                        constant GridParams& g        [[buffer(8)]],
                        constant int& axis            [[buffer(9)]],
                        constant float& dt            [[buffer(10)]],
                        uint tid [[thread_position_in_grid]]) {
  int ni, nj, nk;
  familyDims(axis, g, ni, nj, nk);
  if (int(tid) >= ni * nj * nk) return;
  int i = int(tid) / (nj * nk);
  int rem = int(tid) % (nj * nk);
  int j = rem / nk, k = rem % nk;
  if (boundaryFace(axis, i, j, k, g)) return;  // boundary face keeps prefilled u

  int hi, lo;
  float3 fc;
  if (axis == 0) {
    hi = (i * g.ny + j) * g.nz + k; lo = ((i - 1) * g.ny + j) * g.nz + k;
    fc = float3(g.ox + i * g.h, g.oy + (j + 0.5f) * g.h, g.oz + (k + 0.5f) * g.h);
  } else if (axis == 1) {
    hi = (i * g.ny + j) * g.nz + k; lo = (i * g.ny + (j - 1)) * g.nz + k;
    fc = float3(g.ox + (i + 0.5f) * g.h, g.oy + j * g.h, g.oz + (k + 0.5f) * g.h);
  } else {
    hi = (i * g.ny + j) * g.nz + k; lo = (i * g.ny + j) * g.nz + (k - 1);
    fc = float3(g.ox + (i + 0.5f) * g.h, g.oy + (j + 0.5f) * g.h, g.oz + k * g.h);
  }
  float3 psiHi = float3(psi[3 * hi], psi[3 * hi + 1], psi[3 * hi + 2]);
  float3 psiLo = float3(psi[3 * lo], psi[3 * lo + 1], psi[3 * lo + 2]);
  float3 dpsi = (psiHi - psiLo) / g.h;
  float3 w = sampleVel(ux, uy, uz, rk4Backtrace(vx, vy, vz, fc, dt, g), g);
  out[(i * nj + j) * nk + k] = dot(dpsi, w);
}

// Elementwise half round-trip error: o = 0.5*(a - b) over one face array.
kernel void half_diff(device float* o        [[buffer(0)]],
                      device const float* a   [[buffer(1)]],
                      device const float* b   [[buffer(2)]],
                      constant uint& n        [[buffer(3)]],
                      uint tid [[thread_position_in_grid]]) {
  if (tid >= n) return;
  o[tid] = 0.5f * (a[tid] - b[tid]);
}

// corrected -= corr on the interior faces of family `axis` (the face index is the
// linear thread id, since faceX/Y/ZIndex enumerates (i*nj+j)*nk+k). Boundary
// faces keep their prefilled value.
kernel void sub_interior(device float* corrected  [[buffer(0)]],
                         device const float* corr   [[buffer(1)]],
                         constant GridParams& g     [[buffer(2)]],
                         constant int& axis         [[buffer(3)]],
                         uint tid [[thread_position_in_grid]]) {
  int ni, nj, nk;
  familyDims(axis, g, ni, nj, nk);
  if (int(tid) >= ni * nj * nk) return;
  int i = int(tid) / (nj * nk), rem = int(tid) % (nj * nk), j = rem / nk, k = rem % nk;
  if (boundaryFace(axis, i, j, k, g)) return;
  corrected[tid] -= corr[tid];
}

// CF 5.4.2 extrema limiter (== limitToStencil): clamp each interior candidate
// face to the min/max of `ref` over the 3x3x3 same-family neighbour box (indices
// clamped to the family extent). In place on the candidate buffer -- each thread
// reads only its own candidate entry plus the const reference, so no hazard.
kernel void limiter(device float* cand        [[buffer(0)]],  // in-place candidate
                    device const float* ref     [[buffer(1)]],  // reference (u1)
                    constant GridParams& g       [[buffer(2)]],
                    constant int& axis           [[buffer(3)]],
                    uint tid [[thread_position_in_grid]]) {
  int ni, nj, nk;
  familyDims(axis, g, ni, nj, nk);
  if (int(tid) >= ni * nj * nk) return;
  int i = int(tid) / (nj * nk), rem = int(tid) % (nj * nk), j = rem / nk, k = rem % nk;
  if (boundaryFace(axis, i, j, k, g)) return;
  float lo = ref[tid], hi = lo;
  for (int di = -1; di <= 1; ++di)
    for (int dj = -1; dj <= 1; ++dj)
      for (int dk = -1; dk <= 1; ++dk) {
        int a = clamp(i + di, 0, ni - 1);
        int b = clamp(j + dj, 0, nj - 1);
        int c = clamp(k + dk, 0, nk - 1);
        float val = ref[(a * nj + b) * nk + c];
        lo = min(lo, val);
        hi = max(hi, val);
      }
  cand[tid] = clamp(cand[tid], lo, hi);
}

// --- Poisson MGPCG kernels ---------------------------------------------------

// CSR sparse matvec: y = A x.
kernel void csr_spmv(device float* y             [[buffer(0)]],
                     device const int* rowStart   [[buffer(1)]],
                     device const int* col        [[buffer(2)]],
                     device const float* val      [[buffer(3)]],
                     device const float* x        [[buffer(4)]],
                     constant uint& n             [[buffer(5)]],
                     uint i [[thread_position_in_grid]]) {
  if (i >= n) return;
  float s = 0.0f;
  for (int p = rowStart[i]; p < rowStart[i + 1]; ++p) s += val[p] * x[col[p]];
  y[i] = s;
}

// One weighted-Jacobi sweep, out of place: xout = xin + omega D^{-1}(b - A xin)
// on active cells (identity on inactive). Out-of-place so the reads all use xin.
kernel void jacobi_update(device float* xout       [[buffer(0)]],
                          device const float* xin    [[buffer(1)]],
                          device const int* rowStart [[buffer(2)]],
                          device const int* col      [[buffer(3)]],
                          device const float* val    [[buffer(4)]],
                          device const float* diag   [[buffer(5)]],
                          device const uchar* active [[buffer(6)]],
                          device const float* b      [[buffer(7)]],
                          constant uint& n           [[buffer(8)]],
                          uint i [[thread_position_in_grid]]) {
  if (i >= n) return;
  if (!active[i] || diag[i] == 0.0f) { xout[i] = xin[i]; return; }
  float s = 0.0f;
  for (int p = rowStart[i]; p < rowStart[i + 1]; ++p) s += val[p] * xin[col[p]];
  xout[i] = xin[i] + 0.8f * (b[i] - s) / diag[i];
}

// Restrict r to the coarse level: rc[agg] += r (over active fine cells). rc must
// be pre-zeroed; the scatter is an atomic add (several fine cells share one
// coarse cell).
kernel void restrict_scatter(device atomic_float* rc [[buffer(0)]],
                             device const float* r     [[buffer(1)]],
                             device const int* aggUp    [[buffer(2)]],
                             device const uchar* active [[buffer(3)]],
                             constant uint& n           [[buffer(4)]],
                             uint i [[thread_position_in_grid]]) {
  if (i >= n) return;
  if (!active[i]) return;
  atomic_fetch_add_explicit(&rc[aggUp[i]], r[i], memory_order_relaxed);
}

// Prolong the coarse correction to the fine level: xfine[i] += ecoarse[agg[i]].
kernel void prolong_add(device float* xfine        [[buffer(0)]],
                        device const float* ecoarse [[buffer(1)]],
                        device const int* aggUp      [[buffer(2)]],
                        device const uchar* active   [[buffer(3)]],
                        constant uint& n             [[buffer(4)]],
                        uint i [[thread_position_in_grid]]) {
  if (i >= n) return;
  if (active[i]) xfine[i] += ecoarse[aggUp[i]];
}

kernel void vec_sub(device float* r [[buffer(0)]], device const float* b [[buffer(1)]],
                    device const float* ax [[buffer(2)]], constant uint& n [[buffer(3)]],
                    uint i [[thread_position_in_grid]]) {
  if (i < n) r[i] = b[i] - ax[i];
}

// y += a * x
kernel void vec_axpy(device float* y [[buffer(0)]], device const float* x [[buffer(1)]],
                     constant float& a [[buffer(2)]], constant uint& n [[buffer(3)]],
                     uint i [[thread_position_in_grid]]) {
  if (i < n) y[i] += a * x[i];
}

// p = z + beta * p
kernel void vec_xpay(device float* p [[buffer(0)]], device const float* z [[buffer(1)]],
                     constant float& beta [[buffer(2)]], constant uint& n [[buffer(3)]],
                     uint i [[thread_position_in_grid]]) {
  if (i < n) p[i] = z[i] + beta * p[i];
}

// Partial dot products: one float per threadgroup, summed on the host.
kernel void dot_partial(device float* partials [[buffer(0)]], device const float* a [[buffer(1)]],
                        device const float* b [[buffer(2)]], constant uint& n [[buffer(3)]],
                        uint gid [[threadgroup_position_in_grid]],
                        uint lid [[thread_position_in_threadgroup]],
                        uint tpg [[threads_per_threadgroup]]) {
  threadgroup float scratch[256];
  const uint i = gid * tpg + lid;
  scratch[lid] = (i < n) ? a[i] * b[i] : 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint s = tpg / 2; s > 0; s >>= 1) {
    if (lid < s) scratch[lid] += scratch[lid + s];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lid == 0) partials[gid] = scratch[0];
}

// Finish the optimal-alpha reduction on the device: sum the num/den partial
// arrays (each `P` long) in one threadgroup and write alpha = num/den (0 if den
// <= 0) to alpha[0]. Keeps the V-cycle's coarse-correction scale on the GPU, so
// the whole cycle is one command buffer with no mid-cycle host sync.
kernel void alpha_reduce(device float* alpha       [[buffer(0)]],
                         device const float* pnum    [[buffer(1)]],
                         device const float* pden    [[buffer(2)]],
                         constant uint& P            [[buffer(3)]],
                         uint lid [[thread_position_in_threadgroup]],
                         uint tpg [[threads_per_threadgroup]]) {
  threadgroup float sn[256], sd[256];
  float an = 0.0f, ad = 0.0f;
  for (uint i = lid; i < P; i += tpg) { an += pnum[i]; ad += pden[i]; }
  sn[lid] = an; sd[lid] = ad;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint s = tpg / 2; s > 0; s >>= 1) {
    if (lid < s) { sn[lid] += sn[lid + s]; sd[lid] += sd[lid + s]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lid == 0) alpha[0] = (sd[0] > 0.0f) ? (sn[0] / sd[0]) : 0.0f;
}

// y += a[0] * x  (scalar a read from a device buffer, not a setBytes constant).
kernel void axpy_buf(device float* y [[buffer(0)]], device const float* x [[buffer(1)]],
                     device const float* a [[buffer(2)]], constant uint& n [[buffer(3)]],
                     uint i [[thread_position_in_grid]]) {
  if (i < n) y[i] += a[0] * x[i];
}

// --- device-resident covMG-LOBPCG primitives (complex, scalars kept on-device) --------
// Complex Hermitian dot partials: float2 per threadgroup = (Re, Im) of
// sum conj(a)*b over this threadgroup's complex nodes (n = node count).
kernel void cdot_partial(device float2* partials [[buffer(0)]],
                         device const float2* a [[buffer(1)]],
                         device const float2* b [[buffer(2)]], constant uint& n [[buffer(3)]],
                         uint gid [[threadgroup_position_in_grid]],
                         uint lid [[thread_position_in_threadgroup]],
                         uint tpg [[threads_per_threadgroup]]) {
  threadgroup float sr[256], si[256];
  const uint i = gid * tpg + lid;
  const float2 va = (i < n) ? a[i] : float2(0.0f, 0.0f);
  const float2 vb = (i < n) ? b[i] : float2(0.0f, 0.0f);
  sr[lid] = va.x * vb.x + va.y * vb.y;  // Re conj(a) b
  si[lid] = va.x * vb.y - va.y * vb.x;  // Im conj(a) b
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint s = tpg / 2; s > 0; s >>= 1) {
    if (lid < s) { sr[lid] += sr[lid + s]; si[lid] += si[lid + s]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lid == 0) partials[gid] = float2(sr[0], si[0]);
}

// Sum the P float2 partials into out[0] (one threadgroup).
kernel void cdot_finish(device float2* out [[buffer(0)]], device const float2* partials [[buffer(1)]],
                        constant uint& P [[buffer(2)]], uint lid [[thread_position_in_threadgroup]],
                        uint tpg [[threads_per_threadgroup]]) {
  threadgroup float sr[256], si[256];
  float2 acc = float2(0.0f, 0.0f);
  for (uint i = lid; i < P; i += tpg) acc += partials[i];
  sr[lid] = acc.x; si[lid] = acc.y;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint s = tpg / 2; s > 0; s >>= 1) {
    if (lid < s) { sr[lid] += sr[lid + s]; si[lid] += si[lid + s]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lid == 0) out[0] = float2(sr[0], si[0]);
}

// y[i] += sign * c * x[i], complex c = scal[0], over n complex nodes (sign = +1
// combine, -1 Gram-Schmidt subtract).
kernel void caxpy_c(device float2* y [[buffer(0)]], device const float2* x [[buffer(1)]],
                    device const float2* scal [[buffer(2)]], constant float& sign [[buffer(3)]],
                    constant uint& n [[buffer(4)]], uint i [[thread_position_in_grid]]) {
  if (i >= n) return;
  const float2 c = scal[0], xi = x[i];
  y[i] += sign * float2(c.x * xi.x - c.y * xi.y, c.x * xi.y + c.y * xi.x);
}

// r[i] = ex[i] - rho * x[i], real rho = scal[0].x.
kernel void resid_c(device float2* r [[buffer(0)]], device const float2* ex [[buffer(1)]],
                    device const float2* x [[buffer(2)]], device const float2* scal [[buffer(3)]],
                    constant uint& n [[buffer(4)]], uint i [[thread_position_in_grid]]) {
  if (i < n) r[i] = ex[i] - scal[0].x * x[i];
}

// x[i] *= 1/sqrt(max(scal[0].x, tiny))  -- normalize by the norm whose square is
// scal[0].x (a near-null vector stays bounded and unit-ish; it is excluded on the
// host by patching the Ritz matrix, so its exact direction does not matter).
kernel void cnormalize(device float2* x [[buffer(0)]], device const float2* scal [[buffer(1)]],
                       constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i < n) x[i] *= rsqrt(max(scal[0].x, 1e-30f));
}

// --- Gauge (connection-Laplacian) matvec -------------------------------------
struct LatParams { int lx, ly, lz, periodic; float w; };

static float2 cmul(float2 a, float2 b) {  // a * b
  return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}
static float2 cconjmul(float2 t, float2 x) {  // conj(t) * x
  return float2(t.x * x.x + t.y * x.y, t.x * x.y - t.y * x.x);
}

// y = E x for the U(1) connection Laplacian (== applyConnectionLaplacian). Node
// index (i*ly+j)*lz+k; link indices match xi/yi/zi (yi/zi strides shrink by one
// on an open boundary).
kernel void connection_matvec(device float2* y            [[buffer(0)]],
                              device const float2* x        [[buffer(1)]],
                              device const float2* tx       [[buffer(2)]],
                              device const float2* ty       [[buffer(3)]],
                              device const float2* tz       [[buffer(4)]],
                              constant LatParams& L         [[buffer(5)]],
                              uint tid [[thread_position_in_grid]]) {
  const int total = L.lx * L.ly * L.lz;
  if (int(tid) >= total) return;
  const int i = int(tid) / (L.ly * L.lz), rem = int(tid) % (L.ly * L.lz), j = rem / L.lz,
            k = rem % L.lz;
  const int myl = L.periodic ? L.ly : L.ly - 1;  // y-link middle stride
  const int mzl = L.periodic ? L.lz : L.lz - 1;  // z-link last stride
  auto nidx = [&](int a, int b, int c) { return (a * L.ly + b) * L.lz + c; };
  auto xiL = [&](int a, int b, int c) { return (a * L.ly + b) * L.lz + c; };
  auto yiL = [&](int a, int b, int c) { return (a * myl + b) * L.lz + c; };
  auto ziL = [&](int a, int b, int c) { return (a * L.ly + b) * mzl + c; };

  const float2 xc = x[tid];
  float2 acc = float2(0.0f, 0.0f);
  if (L.periodic) {
    const int im = (i - 1 + L.lx) % L.lx, ip = (i + 1) % L.lx;
    const int jm = (j - 1 + L.ly) % L.ly, jp = (j + 1) % L.ly;
    const int km = (k - 1 + L.lz) % L.lz, kp = (k + 1) % L.lz;
    acc += xc - cmul(tx[xiL(im, j, k)], x[nidx(im, j, k)]);
    acc += xc - cconjmul(tx[xiL(i, j, k)], x[nidx(ip, j, k)]);
    acc += xc - cmul(ty[yiL(i, jm, k)], x[nidx(i, jm, k)]);
    acc += xc - cconjmul(ty[yiL(i, j, k)], x[nidx(i, jp, k)]);
    acc += xc - cmul(tz[ziL(i, j, km)], x[nidx(i, j, km)]);
    acc += xc - cconjmul(tz[ziL(i, j, k)], x[nidx(i, j, kp)]);
  } else {
    if (i > 0) acc += xc - cmul(tx[xiL(i - 1, j, k)], x[nidx(i - 1, j, k)]);
    if (i + 1 < L.lx) acc += xc - cconjmul(tx[xiL(i, j, k)], x[nidx(i + 1, j, k)]);
    if (j > 0) acc += xc - cmul(ty[yiL(i, j - 1, k)], x[nidx(i, j - 1, k)]);
    if (j + 1 < L.ly) acc += xc - cconjmul(ty[yiL(i, j, k)], x[nidx(i, j + 1, k)]);
    if (k > 0) acc += xc - cmul(tz[ziL(i, j, k - 1)], x[nidx(i, j, k - 1)]);
    if (k + 1 < L.lz) acc += xc - cconjmul(tz[ziL(i, j, k)], x[nidx(i, j, k + 1)]);
  }
  y[tid] = L.w * acc;
}

// Existing-neighbour count of node (i,j,k) = the connection Laplacian's diagonal
// degree (== bochner::degree). Periodic nodes always have six.
static int nodeDegree(constant LatParams& L, int i, int j, int k) {
  if (L.periodic) return 6;
  return (i > 0) + (i + 1 < L.lx) + (j > 0) + (j + 1 < L.ly) + (k > 0) + (k + 1 < L.lz);
}

// One red-black Gauss-Seidel (SOR) COLOR pass of E x = b (== GaugeMultigrid
// smooth). A single dispatch over all nodes updates only the `color` parity
// (i+j+k)&1; those nodes' neighbours are all the OTHER color, untouched this
// pass, so the in-place writes are race-free. Run color 0 then color 1 (separate
// encoders) for a full Gauss-Seidel sweep.
//   x_c = (1-omega) x_c + omega (b_c + w sum_n transport_{n->c} x_n) / (w deg_c).
kernel void connection_smooth(device float2* x            [[buffer(0)]],  // in/out
                              device const float2* b        [[buffer(1)]],
                              device const float2* tx       [[buffer(2)]],
                              device const float2* ty       [[buffer(3)]],
                              device const float2* tz       [[buffer(4)]],
                              constant LatParams& L         [[buffer(5)]],
                              constant int& color           [[buffer(6)]],
                              constant float& omega         [[buffer(7)]],
                              uint tid [[thread_position_in_grid]]) {
  const int total = L.lx * L.ly * L.lz;
  if (int(tid) >= total) return;
  const int i = int(tid) / (L.ly * L.lz), rem = int(tid) % (L.ly * L.lz), j = rem / L.lz,
            k = rem % L.lz;
  if (((i + j + k) & 1) != color) return;
  const int deg = nodeDegree(L, i, j, k);
  if (deg == 0) return;
  const int myl = L.periodic ? L.ly : L.ly - 1;
  const int mzl = L.periodic ? L.lz : L.lz - 1;
  auto nidx = [&](int a, int b2, int c) { return (a * L.ly + b2) * L.lz + c; };
  auto xiL = [&](int a, int b2, int c) { return (a * L.ly + b2) * L.lz + c; };
  auto yiL = [&](int a, int b2, int c) { return (a * myl + b2) * L.lz + c; };
  auto ziL = [&](int a, int b2, int c) { return (a * L.ly + b2) * mzl + c; };

  float2 sum = b[tid];
  if (L.periodic) {
    const int im = (i - 1 + L.lx) % L.lx, ip = (i + 1) % L.lx;
    const int jm = (j - 1 + L.ly) % L.ly, jp = (j + 1) % L.ly;
    const int km = (k - 1 + L.lz) % L.lz, kp = (k + 1) % L.lz;
    sum += L.w * cmul(tx[xiL(im, j, k)], x[nidx(im, j, k)]);
    sum += L.w * cconjmul(tx[xiL(i, j, k)], x[nidx(ip, j, k)]);
    sum += L.w * cmul(ty[yiL(i, jm, k)], x[nidx(i, jm, k)]);
    sum += L.w * cconjmul(ty[yiL(i, j, k)], x[nidx(i, jp, k)]);
    sum += L.w * cmul(tz[ziL(i, j, km)], x[nidx(i, j, km)]);
    sum += L.w * cconjmul(tz[ziL(i, j, k)], x[nidx(i, j, kp)]);
  } else {
    if (i > 0) sum += L.w * cmul(tx[xiL(i - 1, j, k)], x[nidx(i - 1, j, k)]);
    if (i + 1 < L.lx) sum += L.w * cconjmul(tx[xiL(i, j, k)], x[nidx(i + 1, j, k)]);
    if (j > 0) sum += L.w * cmul(ty[yiL(i, j - 1, k)], x[nidx(i, j - 1, k)]);
    if (j + 1 < L.ly) sum += L.w * cconjmul(ty[yiL(i, j, k)], x[nidx(i, j + 1, k)]);
    if (k > 0) sum += L.w * cmul(tz[ziL(i, j, k - 1)], x[nidx(i, j, k - 1)]);
    if (k + 1 < L.lz) sum += L.w * cconjmul(tz[ziL(i, j, k)], x[nidx(i, j, k + 1)]);
  }
  x[tid] = (1.0f - omega) * x[tid] + (omega / (L.w * float(deg))) * sum;
}

// Sources a fine node (i,j,k) averages from during prolongation (== bochner::
// gatherSources): along each ODD axis its two even neighbours (open drops the
// missing side). Returns the count; fills the node index, the RAW forward link
// transport, and a high flag (the high neighbour's transport is used conjugated).
static int gatherSources(constant LatParams& L, device const float2* tx,
                         device const float2* ty, device const float2* tz, int i, int j, int k,
                         thread int* sidx, thread float2* st, thread int* shigh) {
  const int myl = L.periodic ? L.ly : L.ly - 1;
  const int mzl = L.periodic ? L.lz : L.lz - 1;
  auto nidx = [&](int a, int b, int c) { return (a * L.ly + b) * L.lz + c; };
  auto xiL = [&](int a, int b, int c) { return (a * L.ly + b) * L.lz + c; };
  auto yiL = [&](int a, int b, int c) { return (a * myl + b) * L.lz + c; };
  auto ziL = [&](int a, int b, int c) { return (a * L.ly + b) * mzl + c; };
  int cnt = 0;
  if (i & 1) {
    const int lo = L.periodic ? (i - 1 + L.lx) % L.lx : i - 1;
    sidx[cnt] = nidx(lo, j, k); st[cnt] = tx[xiL(lo, j, k)]; shigh[cnt] = 0; ++cnt;
    if (L.periodic) { sidx[cnt] = nidx((i + 1) % L.lx, j, k); st[cnt] = tx[xiL(i, j, k)]; shigh[cnt] = 1; ++cnt; }
    else if (i + 1 < L.lx) { sidx[cnt] = nidx(i + 1, j, k); st[cnt] = tx[xiL(i, j, k)]; shigh[cnt] = 1; ++cnt; }
  }
  if (j & 1) {
    const int lo = L.periodic ? (j - 1 + L.ly) % L.ly : j - 1;
    sidx[cnt] = nidx(i, lo, k); st[cnt] = ty[yiL(i, lo, k)]; shigh[cnt] = 0; ++cnt;
    if (L.periodic) { sidx[cnt] = nidx(i, (j + 1) % L.ly, k); st[cnt] = ty[yiL(i, j, k)]; shigh[cnt] = 1; ++cnt; }
    else if (j + 1 < L.ly) { sidx[cnt] = nidx(i, j + 1, k); st[cnt] = ty[yiL(i, j, k)]; shigh[cnt] = 1; ++cnt; }
  }
  if (k & 1) {
    const int lo = L.periodic ? (k - 1 + L.lz) % L.lz : k - 1;
    sidx[cnt] = nidx(i, j, lo); st[cnt] = tz[ziL(i, j, lo)]; shigh[cnt] = 0; ++cnt;
    if (L.periodic) { sidx[cnt] = nidx(i, j, (k + 1) % L.lz); st[cnt] = tz[ziL(i, j, k)]; shigh[cnt] = 1; ++cnt; }
    else if (k + 1 < L.lz) { sidx[cnt] = nidx(i, j, k + 1); st[cnt] = tz[ziL(i, j, k)]; shigh[cnt] = 1; ++cnt; }
  }
  return cnt;
}

// Copy the coarse vector onto the even (all-even-coordinate) fine nodes.
kernel void connection_prolong_copy(device float2* fine        [[buffer(0)]],
                                    device const float2* coarse [[buffer(1)]],
                                    constant LatParams& L        [[buffer(2)]],
                                    uint tid [[thread_position_in_grid]]) {
  const int total = L.lx * L.ly * L.lz;
  if (int(tid) >= total) return;
  const int i = int(tid) / (L.ly * L.lz), rem = int(tid) % (L.ly * L.lz), j = rem / L.lz,
            k = rem % L.lz;
  if ((i & 1) || (j & 1) || (k & 1)) return;
  const int cly = L.ly / 2, clz = L.lz / 2;
  fine[tid] = coarse[((i / 2) * cly + j / 2) * clz + k / 2];
}

// Prolong pass p (p=1,2,3): fill fine nodes with parity-sum p from their sources
// (all parity-sum p-1, filled by the even-copy or a prior pass -> nodes within a
// pass are independent). node = (1/cnt) sum transport * fine[src].
kernel void connection_prolong(device float2* fine        [[buffer(0)]],
                               device const float2* tx      [[buffer(1)]],
                               device const float2* ty      [[buffer(2)]],
                               device const float2* tz      [[buffer(3)]],
                               constant LatParams& L        [[buffer(4)]],
                               constant int& pass           [[buffer(5)]],
                               uint tid [[thread_position_in_grid]]) {
  const int total = L.lx * L.ly * L.lz;
  if (int(tid) >= total) return;
  const int i = int(tid) / (L.ly * L.lz), rem = int(tid) % (L.ly * L.lz), j = rem / L.lz,
            k = rem % L.lz;
  if ((i & 1) + (j & 1) + (k & 1) != pass) return;
  int sidx[6]; float2 st[6]; int shigh[6];
  const int cnt = gatherSources(L, tx, ty, tz, i, j, k, sidx, st, shigh);
  float2 sum = float2(0.0f, 0.0f);
  for (int t = 0; t < cnt; ++t)
    sum += shigh[t] ? cconjmul(st[t], fine[sidx[t]]) : cmul(st[t], fine[sidx[t]]);
  fine[tid] = sum / float(cnt);
}

// Restrict pass p (p=3,2,1): the exact adjoint of prolong pass p. Read the node's
// (accumulated) value, clear it, and atomic-scatter conj(transport)/cnt * val to
// each source (several parity-sum-p nodes share a parity-sum p-1 source -> atomic
// float2 add, per component). r is 2*n interleaved (re,im).
kernel void connection_restrict(device atomic_float* r       [[buffer(0)]],
                                device const float2* tx       [[buffer(1)]],
                                device const float2* ty       [[buffer(2)]],
                                device const float2* tz       [[buffer(3)]],
                                constant LatParams& L         [[buffer(4)]],
                                constant int& pass            [[buffer(5)]],
                                uint tid [[thread_position_in_grid]]) {
  const int total = L.lx * L.ly * L.lz;
  if (int(tid) >= total) return;
  const int i = int(tid) / (L.ly * L.lz), rem = int(tid) % (L.ly * L.lz), j = rem / L.lz,
            k = rem % L.lz;
  if ((i & 1) + (j & 1) + (k & 1) != pass) return;
  const float2 val = float2(atomic_load_explicit(&r[2 * tid], memory_order_relaxed),
                            atomic_load_explicit(&r[2 * tid + 1], memory_order_relaxed));
  atomic_store_explicit(&r[2 * tid], 0.0f, memory_order_relaxed);
  atomic_store_explicit(&r[2 * tid + 1], 0.0f, memory_order_relaxed);
  int sidx[6]; float2 st[6]; int shigh[6];
  const int cnt = gatherSources(L, tx, ty, tz, i, j, k, sidx, st, shigh);
  for (int t = 0; t < cnt; ++t) {
    const float2 c = (shigh[t] ? cmul(st[t], val) : cconjmul(st[t], val)) / float(cnt);
    atomic_fetch_add_explicit(&r[2 * sidx[t]], c.x, memory_order_relaxed);
    atomic_fetch_add_explicit(&r[2 * sidx[t] + 1], c.y, memory_order_relaxed);
  }
}

// Pick the even fine nodes into the coarse vector (Copy^H) after the restrict
// passes. r is now read as plain float2 (no atomics remain).
kernel void connection_restrict_pick(device float2* coarse      [[buffer(0)]],
                                     device const float2* r       [[buffer(1)]],
                                     constant LatParams& L        [[buffer(2)]],
                                     uint tid [[thread_position_in_grid]]) {
  const int clx = L.lx / 2, cly = L.ly / 2, clz = L.lz / 2;
  const int cn = clx * cly * clz;
  if (int(tid) >= cn) return;
  const int I = int(tid) / (cly * clz), rem = int(tid) % (cly * clz), J = rem / clz, K = rem % clz;
  coarse[tid] = r[((2 * I) * L.ly + 2 * J) * L.lz + 2 * K];
}

// --- CF+MCM (characteristic-map) advection kernels ---------------------------

// Trilinear sample of a cell-centered Vec3 field (3 floats/cell, node offset
// 0.5,0.5,0.5) at world point p. Matches bochner::sampleCellVec3 / the CPU
// trilerpComponent: extrap!=0 leaves the fractional weight free (linear
// EXTRAPOLATION -- correct for a map coordinate field); extrap==0 clamps the
// query into the lattice (the streak boundary, correct for a velocity).
static float3 sampleCellVec3(device const float* f, float3 p, constant GridParams& g, int extrap) {
  int ni = g.nx, nj = g.ny, nk = g.nz;
  float cx = (p.x - g.ox) / g.h - 0.5f;
  float cy = (p.y - g.oy) / g.h - 0.5f;
  float cz = (p.z - g.oz) / g.h - 0.5f;
  if (extrap == 0) {
    cx = clamp(cx, 0.0f, float(ni - 1));
    cy = clamp(cy, 0.0f, float(nj - 1));
    cz = clamp(cz, 0.0f, float(nk - 1));
  }
  int i0 = clamp(int(floor(cx)), 0, max(0, ni - 2));
  int j0 = clamp(int(floor(cy)), 0, max(0, nj - 2));
  int k0 = clamp(int(floor(cz)), 0, max(0, nk - 2));
  float tx = cx - float(i0), ty = cy - float(j0), tz = cz - float(k0);
  if (ni == 1) { i0 = 0; tx = 0.0f; }
  if (nj == 1) { j0 = 0; ty = 0.0f; }
  if (nk == 1) { k0 = 0; tz = 0.0f; }
  int i1 = min(i0 + 1, ni - 1), j1 = min(j0 + 1, nj - 1), k1 = min(k0 + 1, nk - 1);
  int b00 = (i0 * nj + j0) * nk, b10 = (i1 * nj + j0) * nk;
  int b01 = (i0 * nj + j1) * nk, b11 = (i1 * nj + j1) * nk;
  float3 r;
  for (int c = 0; c < 3; ++c) {
    float c000 = f[3 * (b00 + k0) + c], c100 = f[3 * (b10 + k0) + c];
    float c010 = f[3 * (b01 + k0) + c], c110 = f[3 * (b11 + k0) + c];
    float c001 = f[3 * (b00 + k1) + c], c101 = f[3 * (b10 + k1) + c];
    float c011 = f[3 * (b01 + k1) + c], c111 = f[3 * (b11 + k1) + c];
    float c00 = mix(c000, c100, tx), c10 = mix(c010, c110, tx);
    float c01 = mix(c001, c101, tx), c11 = mix(c011, c111, tx);
    float c0 = mix(c00, c10, ty), c1 = mix(c01, c11, ty);
    r[c] = mix(c0, c1, tz);
  }
  return r;
}

static float3 cellCenter(int tid, constant GridParams& g) {
  int i = tid / (g.ny * g.nz), rem = tid % (g.ny * g.nz), j = rem / g.nz, k = rem % g.nz;
  return float3(g.ox + (i + 0.5f) * g.h, g.oy + (j + 0.5f) * g.h, g.oz + (k + 0.5f) * g.h);
}

// Alg. 5 line 4: advect the inverse map backward -- Psi_new(x) = Psi_old(backtrace(x)).
// A map is sampled with EXTRAPOLATION. Out of place (psiNew != psiOld).
kernel void map_advect(device float* psiNew        [[buffer(0)]],  // 3*cells
                       device const float* psiOld   [[buffer(1)]],  // 3*cells
                       device const float* vx        [[buffer(2)]],
                       device const float* vy        [[buffer(3)]],
                       device const float* vz        [[buffer(4)]],
                       constant GridParams& g        [[buffer(5)]],
                       constant float& dt            [[buffer(6)]],
                       uint tid [[thread_position_in_grid]]) {
  int total = g.nx * g.ny * g.nz;
  if (int(tid) >= total) return;
  float3 bt = rk4Backtrace(vx, vy, vz, cellCenter(int(tid), g), dt, g);
  float3 m = sampleCellVec3(psiOld, bt, g, 1);
  psiNew[3 * tid] = m.x; psiNew[3 * tid + 1] = m.y; psiNew[3 * tid + 2] = m.z;
}

// Reset a map to the identity (cell centers) -- the reinit (Alg. 5 line 12).
kernel void map_identity(device float* map          [[buffer(0)]],  // 3*cells
                         constant GridParams& g       [[buffer(1)]],
                         uint tid [[thread_position_in_grid]]) {
  if (int(tid) >= g.nx * g.ny * g.nz) return;
  float3 cc = cellCenter(int(tid), g);
  map[3 * tid] = cc.x; map[3 * tid + 1] = cc.y; map[3 * tid + 2] = cc.z;
}

// Alg. 5 lines 5/7/9: u1 = dM^T (source o M) through a STORED map M (== the CPU
// pullbackThroughMap). Per interior face of family `axis`: dM = (M[hi]-M[lo])/h;
// sample `source` at M(faceCenter) (a map point, but pullbackThroughMap CLAMPS
// this sample -- extrap=0); out = dM . source(M(fc)). Boundary faces keep the
// prefilled source value.
kernel void pullback_stored(device float* out         [[buffer(0)]],  // this family's face array
                            device const float* map     [[buffer(1)]],  // 3*cells
                            device const float* sx       [[buffer(2)]],
                            device const float* sy       [[buffer(3)]],
                            device const float* sz       [[buffer(4)]],
                            constant GridParams& g       [[buffer(5)]],
                            constant int& axis           [[buffer(6)]],
                            uint tid [[thread_position_in_grid]]) {
  int ni, nj, nk;
  familyDims(axis, g, ni, nj, nk);
  if (int(tid) >= ni * nj * nk) return;
  int i = int(tid) / (nj * nk), rem = int(tid) % (nj * nk), j = rem / nk, k = rem % nk;
  if (boundaryFace(axis, i, j, k, g)) return;
  int hi, lo;
  float3 fc;
  if (axis == 0) {
    hi = (i * g.ny + j) * g.nz + k; lo = ((i - 1) * g.ny + j) * g.nz + k;
    fc = float3(g.ox + i * g.h, g.oy + (j + 0.5f) * g.h, g.oz + (k + 0.5f) * g.h);
  } else if (axis == 1) {
    hi = (i * g.ny + j) * g.nz + k; lo = (i * g.ny + (j - 1)) * g.nz + k;
    fc = float3(g.ox + (i + 0.5f) * g.h, g.oy + j * g.h, g.oz + (k + 0.5f) * g.h);
  } else {
    hi = (i * g.ny + j) * g.nz + k; lo = (i * g.ny + j) * g.nz + (k - 1);
    fc = float3(g.ox + (i + 0.5f) * g.h, g.oy + (j + 0.5f) * g.h, g.oz + k * g.h);
  }
  float3 mHi = float3(map[3 * hi], map[3 * hi + 1], map[3 * hi + 2]);
  float3 mLo = float3(map[3 * lo], map[3 * lo + 1], map[3 * lo + 2]);
  float3 dM = (mHi - mLo) / g.h;
  float3 w = sampleVel(sx, sy, sz, sampleCellVec3(map, fc, g, 0), g);
  out[(i * nj + j) * nk + k] = dot(dM, w);
}

// Per-cell squared distortion d2(x) = max(|x-Phi(Psi(x))|^2, |x-Psi(Phi(x))|^2)
// on the deep interior (skip a 2-cell border, == rawDistortion); 0 on the border.
// The border shrinks per axis on grids too small for it (min(2, (dim-1)/2), as
// the CPU does) so at least one cell is always measured -- a full 2-cell border
// would zero every cell of any dim < 5 grid and the distortion gate would
// silently never fire. Both composed samples use EXTRAPOLATION (extrap=1),
// matching the CPU.
kernel void distortion_cell(device float* d2          [[buffer(0)]],
                            device const float* psi     [[buffer(1)]],  // 3*cells
                            device const float* phi     [[buffer(2)]],  // 3*cells
                            constant GridParams& g       [[buffer(3)]],
                            uint tid [[thread_position_in_grid]]) {
  int total = g.nx * g.ny * g.nz;
  if (int(tid) >= total) return;
  int i = int(tid) / (g.ny * g.nz), rem = int(tid) % (g.ny * g.nz), j = rem / g.nz, k = rem % g.nz;
  int bi = min(2, (g.nx - 1) / 2), bj = min(2, (g.ny - 1) / 2), bk = min(2, (g.nz - 1) / 2);
  if (i < bi || i >= g.nx - bi || j < bj || j >= g.ny - bj || k < bk || k >= g.nz - bk) {
    d2[tid] = 0.0f;
    return;
  }
  float3 x = cellCenter(int(tid), g);
  float3 psic = float3(psi[3 * tid], psi[3 * tid + 1], psi[3 * tid + 2]);
  float3 phic = float3(phi[3 * tid], phi[3 * tid + 1], phi[3 * tid + 2]);
  float3 fwdBack = sampleCellVec3(phi, psic, g, 1);  // Phi(Psi(x))
  float3 backFwd = sampleCellVec3(psi, phic, g, 1);  // Psi(Phi(x))
  float3 dbf = x - fwdBack, dfb = x - backFwd;
  d2[tid] = max(dot(dbf, dbf), dot(dfb, dfb));
}

// Threadgroup max-reduction: one partial max per threadgroup into `partials`.
kernel void max_partial(device float* partials      [[buffer(0)]],
                        device const float* a         [[buffer(1)]],
                        constant uint& n             [[buffer(2)]],
                        uint tid [[thread_position_in_grid]],
                        uint lid [[thread_position_in_threadgroup]],
                        uint tgid [[threadgroup_position_in_grid]],
                        uint tgsize [[threads_per_threadgroup]]) {
  threadgroup float scratch[256];
  scratch[lid] = (tid < n) ? a[tid] : 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint s = tgsize / 2; s > 0; s >>= 1) {
    if (lid < s) scratch[lid] = max(scratch[lid], scratch[lid + s]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lid == 0) partials[tgid] = scratch[0];
}

// --- real geometric multigrid for the closed-box Neumann Poisson -------------
// The trivial-connection (transport = 1), real, non-periodic special case of the
// gauge multigrid above (== solvers/PoissonMultigrid.cpp poissonVcycleSolve). It
// solves the UNPINNED all-Neumann Laplacian (the projection's gradient kills the
// constant null space, so no pin -- which is what lets it converge fast, unlike
// the assembled single-cell-pin MGPCG). LatParams.w = 1/h^2; periodic ignored.

static int rDegree(constant LatParams& L, int i, int j, int k) {
  return (i > 0) + (i + 1 < L.lx) + (j > 0) + (j + 1 < L.ly) + (k > 0) + (k + 1 < L.lz);
}

// y = L x: (Lx)_c = w (deg_c x_c - sum existing neighbours).
kernel void poisson_matvec(device float* y            [[buffer(0)]],
                           device const float* x        [[buffer(1)]],
                           constant LatParams& L        [[buffer(2)]],
                           uint tid [[thread_position_in_grid]]) {
  const int total = L.lx * L.ly * L.lz;
  if (int(tid) >= total) return;
  const int i = int(tid) / (L.ly * L.lz), rem = int(tid) % (L.ly * L.lz), j = rem / L.lz,
            k = rem % L.lz;
  auto nidx = [&](int a, int b, int c) { return (a * L.ly + b) * L.lz + c; };
  const float xc = x[tid];
  float acc = 0.0f;
  if (i > 0) acc += xc - x[nidx(i - 1, j, k)];
  if (i + 1 < L.lx) acc += xc - x[nidx(i + 1, j, k)];
  if (j > 0) acc += xc - x[nidx(i, j - 1, k)];
  if (j + 1 < L.ly) acc += xc - x[nidx(i, j + 1, k)];
  if (k > 0) acc += xc - x[nidx(i, j, k - 1)];
  if (k + 1 < L.lz) acc += xc - x[nidx(i, j, k + 1)];
  y[tid] = L.w * acc;
}

// One red-black GS (SOR) COLOR pass of L x = b (== PoissonMultigrid smooth):
//   x_c = (1-omega) x_c + omega (b_c + w sum_n x_n) / (w deg_c).
kernel void poisson_smooth(device float* x            [[buffer(0)]],  // in/out
                           device const float* b        [[buffer(1)]],
                           constant LatParams& L        [[buffer(2)]],
                           constant int& color          [[buffer(3)]],
                           constant float& omega        [[buffer(4)]],
                           uint tid [[thread_position_in_grid]]) {
  const int total = L.lx * L.ly * L.lz;
  if (int(tid) >= total) return;
  const int i = int(tid) / (L.ly * L.lz), rem = int(tid) % (L.ly * L.lz), j = rem / L.lz,
            k = rem % L.lz;
  if (((i + j + k) & 1) != color) return;
  const int deg = rDegree(L, i, j, k);
  if (deg == 0) return;
  auto nidx = [&](int a, int b, int c) { return (a * L.ly + b) * L.lz + c; };
  float sum = b[tid];
  if (i > 0) sum += L.w * x[nidx(i - 1, j, k)];
  if (i + 1 < L.lx) sum += L.w * x[nidx(i + 1, j, k)];
  if (j > 0) sum += L.w * x[nidx(i, j - 1, k)];
  if (j + 1 < L.ly) sum += L.w * x[nidx(i, j + 1, k)];
  if (k > 0) sum += L.w * x[nidx(i, j, k - 1)];
  if (k + 1 < L.lz) sum += L.w * x[nidx(i, j, k + 1)];
  x[tid] = (1.0f - omega) * x[tid] + (omega / (L.w * float(deg))) * sum;
}

// Even-neighbour sources of a new (odd-coordinate) fine node (== gatherSources),
// indices only (transport = 1), non-periodic.
static int rGather(constant LatParams& L, int i, int j, int k, thread int* s) {
  auto nidx = [&](int a, int b, int c) { return (a * L.ly + b) * L.lz + c; };
  int cnt = 0;
  if (i & 1) {
    s[cnt++] = nidx(i - 1, j, k);
    if (i + 1 < L.lx) s[cnt++] = nidx(i + 1, j, k);
  }
  if (j & 1) {
    s[cnt++] = nidx(i, j - 1, k);
    if (j + 1 < L.ly) s[cnt++] = nidx(i, j + 1, k);
  }
  if (k & 1) {
    s[cnt++] = nidx(i, j, k - 1);
    if (k + 1 < L.lz) s[cnt++] = nidx(i, j, k + 1);
  }
  return cnt;
}

kernel void poisson_prolong_copy(device float* fine        [[buffer(0)]],
                                 device const float* coarse [[buffer(1)]],
                                 constant LatParams& L        [[buffer(2)]],
                                 uint tid [[thread_position_in_grid]]) {
  const int total = L.lx * L.ly * L.lz;
  if (int(tid) >= total) return;
  const int i = int(tid) / (L.ly * L.lz), rem = int(tid) % (L.ly * L.lz), j = rem / L.lz,
            k = rem % L.lz;
  if ((i & 1) || (j & 1) || (k & 1)) return;
  const int cly = L.ly / 2, clz = L.lz / 2;
  fine[tid] = coarse[((i / 2) * cly + j / 2) * clz + k / 2];
}

kernel void poisson_prolong(device float* fine        [[buffer(0)]],
                            constant LatParams& L        [[buffer(1)]],
                            constant int& pass           [[buffer(2)]],
                            uint tid [[thread_position_in_grid]]) {
  const int total = L.lx * L.ly * L.lz;
  if (int(tid) >= total) return;
  const int i = int(tid) / (L.ly * L.lz), rem = int(tid) % (L.ly * L.lz), j = rem / L.lz,
            k = rem % L.lz;
  if ((i & 1) + (j & 1) + (k & 1) != pass) return;
  int s[6];
  const int cnt = rGather(L, i, j, k, s);
  float sum = 0.0f;
  for (int t = 0; t < cnt; ++t) sum += fine[s[t]];
  fine[tid] = sum / float(cnt);
}

// Exact adjoint of prolong pass p: read the node's accumulated value, clear it,
// and atomic-scatter val/cnt to each source (several pass-p nodes share a source).
kernel void poisson_restrict(device atomic_float* r       [[buffer(0)]],
                             constant LatParams& L         [[buffer(1)]],
                             constant int& pass            [[buffer(2)]],
                             uint tid [[thread_position_in_grid]]) {
  const int total = L.lx * L.ly * L.lz;
  if (int(tid) >= total) return;
  const int i = int(tid) / (L.ly * L.lz), rem = int(tid) % (L.ly * L.lz), j = rem / L.lz,
            k = rem % L.lz;
  if ((i & 1) + (j & 1) + (k & 1) != pass) return;
  const float val = atomic_load_explicit(&r[tid], memory_order_relaxed);
  atomic_store_explicit(&r[tid], 0.0f, memory_order_relaxed);
  int s[6];
  const int cnt = rGather(L, i, j, k, s);
  for (int t = 0; t < cnt; ++t)
    atomic_fetch_add_explicit(&r[s[t]], val / float(cnt), memory_order_relaxed);
}

kernel void poisson_restrict_pick(device float* coarse      [[buffer(0)]],
                                  device const float* r       [[buffer(1)]],
                                  constant LatParams& L        [[buffer(2)]],
                                  uint tid [[thread_position_in_grid]]) {
  const int clx = L.lx / 2, cly = L.ly / 2, clz = L.lz / 2;
  const int cn = clx * cly * clz;
  if (int(tid) >= cn) return;
  const int I = int(tid) / (cly * clz), rem = int(tid) % (cly * clz), J = rem / clz, K = rem % clz;
  coarse[tid] = r[((2 * I) * L.ly + 2 * J) * L.lz + 2 * K];
}
)METAL";

// Lazily-initialized device/queue/pipelines, built once on first use.
struct Context {
  id<MTLDevice> device = nil;
  id<MTLCommandQueue> queue = nil;
  id<MTLComputePipelineState> samplePso = nil;
  id<MTLComputePipelineState> backtracePso = nil;
  id<MTLComputePipelineState> slCellsPso = nil;
  id<MTLComputePipelineState> slPullbackPso = nil;
  id<MTLComputePipelineState> halfDiffPso = nil;
  id<MTLComputePipelineState> subInteriorPso = nil;
  id<MTLComputePipelineState> limiterPso = nil;
  id<MTLComputePipelineState> csrSpmvPso = nil;
  id<MTLComputePipelineState> jacobiPso = nil;
  id<MTLComputePipelineState> restrictPso = nil;
  id<MTLComputePipelineState> prolongPso = nil;
  id<MTLComputePipelineState> subPso = nil;
  id<MTLComputePipelineState> axpyPso = nil;
  id<MTLComputePipelineState> xpayPso = nil;
  id<MTLComputePipelineState> dotPso = nil;
  id<MTLComputePipelineState> connMatvecPso = nil;
  id<MTLComputePipelineState> connSmoothPso = nil;
  id<MTLComputePipelineState> connProlongCopyPso = nil;
  id<MTLComputePipelineState> connProlongPso = nil;
  id<MTLComputePipelineState> connRestrictPso = nil;
  id<MTLComputePipelineState> connRestrictPickPso = nil;
  id<MTLComputePipelineState> alphaReducePso = nil;
  id<MTLComputePipelineState> axpyBufPso = nil;
  id<MTLComputePipelineState> cdotPartialPso = nil;
  id<MTLComputePipelineState> cdotFinishPso = nil;
  id<MTLComputePipelineState> caxpyCPso = nil;
  id<MTLComputePipelineState> residCPso = nil;
  id<MTLComputePipelineState> cnormalizePso = nil;
  id<MTLComputePipelineState> mapAdvectPso = nil;
  id<MTLComputePipelineState> mapIdentityPso = nil;
  id<MTLComputePipelineState> pullbackStoredPso = nil;
  id<MTLComputePipelineState> distortionCellPso = nil;
  id<MTLComputePipelineState> maxPartialPso = nil;
  id<MTLComputePipelineState> poissonMatvecPso = nil;
  id<MTLComputePipelineState> poissonSmoothPso = nil;
  id<MTLComputePipelineState> poissonProlongCopyPso = nil;
  id<MTLComputePipelineState> poissonProlongPso = nil;
  id<MTLComputePipelineState> poissonRestrictPso = nil;
  id<MTLComputePipelineState> poissonRestrictPickPso = nil;
  bool ok = false;

  Context() {
    @autoreleasepool {
      device = MTLCreateSystemDefaultDevice();
      if (!device) return;
      queue = [device newCommandQueue];
      NSError* err = nil;
      id<MTLLibrary> lib =
          [device newLibraryWithSource:[NSString stringWithUTF8String:kSource]
                               options:nil
                                 error:&err];
      if (!lib) {
        // The whole backend is disabled (metalAvailable()==false) if the MSL library
        // fails to compile; log why instead of silently vanishing.
        std::fprintf(stderr, "bochner::gpu: Metal library compile failed: %s\n",
                     err ? err.localizedDescription.UTF8String : "(no NSError)");
        return;
      }
      samplePso = pipeline(lib, @"sample_velocity");
      backtracePso = pipeline(lib, @"backtrace");
      slCellsPso = pipeline(lib, @"sl_cells");
      slPullbackPso = pipeline(lib, @"sl_pullback");
      halfDiffPso = pipeline(lib, @"half_diff");
      subInteriorPso = pipeline(lib, @"sub_interior");
      limiterPso = pipeline(lib, @"limiter");
      csrSpmvPso = pipeline(lib, @"csr_spmv");
      jacobiPso = pipeline(lib, @"jacobi_update");
      restrictPso = pipeline(lib, @"restrict_scatter");
      prolongPso = pipeline(lib, @"prolong_add");
      subPso = pipeline(lib, @"vec_sub");
      axpyPso = pipeline(lib, @"vec_axpy");
      xpayPso = pipeline(lib, @"vec_xpay");
      dotPso = pipeline(lib, @"dot_partial");
      connMatvecPso = pipeline(lib, @"connection_matvec");
      connSmoothPso = pipeline(lib, @"connection_smooth");
      connProlongCopyPso = pipeline(lib, @"connection_prolong_copy");
      connProlongPso = pipeline(lib, @"connection_prolong");
      connRestrictPso = pipeline(lib, @"connection_restrict");
      connRestrictPickPso = pipeline(lib, @"connection_restrict_pick");
      alphaReducePso = pipeline(lib, @"alpha_reduce");
      axpyBufPso = pipeline(lib, @"axpy_buf");
      cdotPartialPso = pipeline(lib, @"cdot_partial");
      cdotFinishPso = pipeline(lib, @"cdot_finish");
      caxpyCPso = pipeline(lib, @"caxpy_c");
      residCPso = pipeline(lib, @"resid_c");
      cnormalizePso = pipeline(lib, @"cnormalize");
      mapAdvectPso = pipeline(lib, @"map_advect");
      mapIdentityPso = pipeline(lib, @"map_identity");
      pullbackStoredPso = pipeline(lib, @"pullback_stored");
      distortionCellPso = pipeline(lib, @"distortion_cell");
      maxPartialPso = pipeline(lib, @"max_partial");
      poissonMatvecPso = pipeline(lib, @"poisson_matvec");
      poissonSmoothPso = pipeline(lib, @"poisson_smooth");
      poissonProlongCopyPso = pipeline(lib, @"poisson_prolong_copy");
      poissonProlongPso = pipeline(lib, @"poisson_prolong");
      poissonRestrictPso = pipeline(lib, @"poisson_restrict");
      poissonRestrictPickPso = pipeline(lib, @"poisson_restrict_pick");
      ok = (samplePso && backtracePso && slCellsPso && slPullbackPso && halfDiffPso &&
            subInteriorPso && limiterPso && csrSpmvPso && jacobiPso && restrictPso && prolongPso &&
            subPso && axpyPso && xpayPso && dotPso && connMatvecPso && connSmoothPso &&
            connProlongCopyPso && connProlongPso && connRestrictPso && connRestrictPickPso &&
            alphaReducePso && axpyBufPso && cdotPartialPso && cdotFinishPso && caxpyCPso &&
            residCPso && cnormalizePso && mapAdvectPso && mapIdentityPso && pullbackStoredPso &&
            distortionCellPso && maxPartialPso && poissonMatvecPso && poissonSmoothPso &&
            poissonProlongCopyPso && poissonProlongPso && poissonRestrictPso &&
            poissonRestrictPickPso);
    }
  }

  id<MTLComputePipelineState> pipeline(id<MTLLibrary> lib, NSString* name) {
    id<MTLFunction> fn = [lib newFunctionWithName:name];
    if (!fn) {
      std::fprintf(stderr, "bochner::gpu: Metal function not found: %s\n", name.UTF8String);
      return nil;
    }
    NSError* err = nil;
    id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:fn error:&err];
    if (!pso)
      std::fprintf(stderr, "bochner::gpu: pipeline '%s' failed: %s\n", name.UTF8String,
                   err ? err.localizedDescription.UTF8String : "(no NSError)");
    return pso;
  }
};

Context& context() {
  static Context ctx;
  return ctx;
}

std::vector<float> toF32(const std::vector<double>& a) { return {a.begin(), a.end()}; }

std::vector<float> packPoints(const std::vector<Vec3>& pts) {
  std::vector<float> fp(3 * pts.size());
  for (size_t i = 0; i < pts.size(); ++i) {
    fp[3 * i] = static_cast<float>(pts[i][0]);
    fp[3 * i + 1] = static_cast<float>(pts[i][1]);
    fp[3 * i + 2] = static_cast<float>(pts[i][2]);
  }
  return fp;
}

GridParams makeGridParams(const MacGrid& g) {
  return {g.nx(),
          g.ny(),
          g.nz(),
          static_cast<float>(g.spacing()),
          static_cast<float>(g.origin()[0]),
          static_cast<float>(g.origin()[1]),
          static_cast<float>(g.origin()[2])};
}

id<MTLBuffer> sharedBuffer(id<MTLDevice> dev, const void* bytes, size_t len) {
  return [dev newBufferWithBytes:bytes length:len options:MTLResourceStorageModeShared];
}

// Host twin of the MSL LatParams (all 4-byte, no pad) for the gauge kernels.
struct LatParamsHost {
  int lx, ly, lz, periodic;
  float w;
};

// complex<double> vector -> interleaved float2 (re,im,re,im,...).
std::vector<float> packC(const std::vector<std::complex<double>>& v) {
  std::vector<float> f(2 * v.size());
  for (size_t i = 0; i < v.size(); ++i) {
    f[2 * i] = static_cast<float>(v[i].real());
    f[2 * i + 1] = static_cast<float>(v[i].imag());
  }
  return f;
}

// Upload a complex vector as a float2 buffer (a dummy 8-byte buffer if empty, so
// an unused link array like tz on a degenerate lattice still binds).
id<MTLBuffer> complexBuffer(id<MTLDevice> d, const std::vector<std::complex<double>>& v) {
  const std::vector<float> f = packC(v);
  return f.empty()
             ? [d newBufferWithLength:8 options:MTLResourceStorageModeShared]
             : [d newBufferWithBytes:f.data() length:f.size() * sizeof(float)
                             options:MTLResourceStorageModeShared];
}

// float2 buffer contents -> complex<double> vector (widen back to the CPU type).
std::vector<std::complex<double>> downloadC(id<MTLBuffer> buf, uint32_t n) {
  const float* r = static_cast<const float*>(buf.contents);
  std::vector<std::complex<double>> out(n);
  for (uint32_t i = 0; i < n; ++i) out[i] = std::complex<double>(r[2 * i], r[2 * i + 1]);
  return out;
}

std::vector<Vec3> unpack(id<MTLBuffer> buf, uint32_t n) {
  const float* r = static_cast<const float*>(buf.contents);
  std::vector<Vec3> out(n);
  for (uint32_t i = 0; i < n; ++i)
    out[i] = Vec3{static_cast<double>(r[3 * i]), static_cast<double>(r[3 * i + 1]),
                  static_cast<double>(r[3 * i + 2])};
  return out;
}

// float buffer -> vector<double> (widen back to the CPU field type).
std::vector<double> downloadF64(id<MTLBuffer> buf, size_t n) {
  const float* r = static_cast<const float*>(buf.contents);
  return std::vector<double>(r, r + n);
}

void dispatch1D(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pso, uint32_t n) {
  NSUInteger tg = pso.maxTotalThreadsPerThreadgroup;
  if (tg > 256) tg = 256;
  [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
}

void blitCopy(id<MTLCommandBuffer> cb, id<MTLBuffer> src, id<MTLBuffer> dst, size_t len) {
  id<MTLBlitCommandEncoder> b = [cb blitCommandEncoder];
  [b copyFromBuffer:src sourceOffset:0 toBuffer:dst destinationOffset:0 size:len];
  [b endEncoding];
}

// A faulted/timed-out command buffer would otherwise be ignored and the
// undefined buffer contents copied out as the "result". Throw instead so the
// caller can fall back to the CPU path. Call after waitUntilCompleted (the
// error is only settled once the buffer has completed).
void checkCb(id<MTLCommandBuffer> cb, const char* where) {
  if (cb.error)
    throw std::runtime_error(std::string("bochner::gpu command-buffer error @") + where + ": " +
                             cb.error.localizedDescription.UTF8String);
}

// Encode one semi-Lagrangian step SL(src; v, dt) into `cb`, operating entirely on
// device buffers: out[3] must already hold a copy of `src` (so boundary faces
// keep it), `psi` is 3*cells scratch. Two compute encoders -- the cell backtrace
// then the per-family pullback -- so hazard tracking orders the psi write before
// the read. `counts` = {numFacesX, numFacesY, numFacesZ}.
void encodeSLStep(id<MTLCommandBuffer> cb, Context& ctx, id<MTLBuffer> outX, id<MTLBuffer> outY,
                  id<MTLBuffer> outZ, id<MTLBuffer> srcX, id<MTLBuffer> srcY, id<MTLBuffer> srcZ,
                  id<MTLBuffer> vX, id<MTLBuffer> vY, id<MTLBuffer> vZ, id<MTLBuffer> psi,
                  const GridParams& gp, float dtf, uint32_t cells, const uint32_t counts[3]) {
  id<MTLBuffer> out[3] = {outX, outY, outZ};
  id<MTLBuffer> src[3] = {srcX, srcY, srcZ};
  id<MTLBuffer> vel[3] = {vX, vY, vZ};
  {
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:ctx.slCellsPso];
    [enc setBuffer:psi offset:0 atIndex:0];
    [enc setBuffer:vel[0] offset:0 atIndex:1];
    [enc setBuffer:vel[1] offset:0 atIndex:2];
    [enc setBuffer:vel[2] offset:0 atIndex:3];
    [enc setBytes:&gp length:sizeof(gp) atIndex:4];
    [enc setBytes:&dtf length:sizeof(dtf) atIndex:5];
    dispatch1D(enc, ctx.slCellsPso, cells);
    [enc endEncoding];
  }
  {
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:ctx.slPullbackPso];
    [enc setBuffer:psi offset:0 atIndex:1];
    [enc setBuffer:src[0] offset:0 atIndex:2];
    [enc setBuffer:src[1] offset:0 atIndex:3];
    [enc setBuffer:src[2] offset:0 atIndex:4];
    [enc setBuffer:vel[0] offset:0 atIndex:5];
    [enc setBuffer:vel[1] offset:0 atIndex:6];
    [enc setBuffer:vel[2] offset:0 atIndex:7];
    [enc setBytes:&gp length:sizeof(gp) atIndex:8];
    [enc setBytes:&dtf length:sizeof(dtf) atIndex:10];
    for (int a = 0; a < 3; ++a) {
      [enc setBuffer:out[a] offset:0 atIndex:0];
      [enc setBytes:&a length:sizeof(int) atIndex:9];
      dispatch1D(enc, ctx.slPullbackPso, counts[a]);
    }
    [enc endEncoding];
  }
}

}  // namespace

bool metalAvailable() { return context().ok; }

std::vector<Vec3> sampleVelocityBatch(const MacGrid& g, const FaceField& u,
                                      const std::vector<Vec3>& pts) {
  Context& ctx = context();
  if (!ctx.ok) throw std::runtime_error("bochner::gpu: Metal device/pipeline unavailable");
  const uint32_t n = static_cast<uint32_t>(pts.size());
  if (n == 0) return {};

  std::vector<Vec3> out;
  @autoreleasepool {
    const std::vector<float> fp = packPoints(pts), fx = toF32(u.x), fy = toF32(u.y), fz = toF32(u.z);
    const GridParams gp = makeGridParams(g);

    id<MTLBuffer> bOut = [ctx.device newBufferWithLength:3 * n * sizeof(float)
                                                 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bPts = sharedBuffer(ctx.device, fp.data(), fp.size() * sizeof(float));
    id<MTLBuffer> bUx = sharedBuffer(ctx.device, fx.data(), fx.size() * sizeof(float));
    id<MTLBuffer> bUy = sharedBuffer(ctx.device, fy.data(), fy.size() * sizeof(float));
    id<MTLBuffer> bUz = sharedBuffer(ctx.device, fz.data(), fz.size() * sizeof(float));

    id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:ctx.samplePso];
    [enc setBuffer:bOut offset:0 atIndex:0];
    [enc setBuffer:bPts offset:0 atIndex:1];
    [enc setBuffer:bUx offset:0 atIndex:2];
    [enc setBuffer:bUy offset:0 atIndex:3];
    [enc setBuffer:bUz offset:0 atIndex:4];
    [enc setBytes:&gp length:sizeof(gp) atIndex:5];
    [enc setBytes:&n length:sizeof(n) atIndex:6];
    NSUInteger tg = ctx.samplePso.maxTotalThreadsPerThreadgroup;
    if (tg > 256) tg = 256;
    [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    checkCb(cb, "sampleVelocityBatch");
    out = unpack(bOut, n);
  }
  return out;
}

std::vector<Vec3> backtraceBatch(const MacGrid& g, const FaceField& v,
                                 const std::vector<Vec3>& starts, double dt) {
  Context& ctx = context();
  if (!ctx.ok) throw std::runtime_error("bochner::gpu: Metal device/pipeline unavailable");
  const uint32_t n = static_cast<uint32_t>(starts.size());
  if (n == 0) return {};

  std::vector<Vec3> out;
  @autoreleasepool {
    const std::vector<float> fs = packPoints(starts), fx = toF32(v.x), fy = toF32(v.y),
                             fz = toF32(v.z);
    const GridParams gp = makeGridParams(g);
    const float dtf = static_cast<float>(dt);

    id<MTLBuffer> bOut = [ctx.device newBufferWithLength:3 * n * sizeof(float)
                                                 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bStarts = sharedBuffer(ctx.device, fs.data(), fs.size() * sizeof(float));
    id<MTLBuffer> bVx = sharedBuffer(ctx.device, fx.data(), fx.size() * sizeof(float));
    id<MTLBuffer> bVy = sharedBuffer(ctx.device, fy.data(), fy.size() * sizeof(float));
    id<MTLBuffer> bVz = sharedBuffer(ctx.device, fz.data(), fz.size() * sizeof(float));

    id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:ctx.backtracePso];
    [enc setBuffer:bOut offset:0 atIndex:0];
    [enc setBuffer:bStarts offset:0 atIndex:1];
    [enc setBuffer:bVx offset:0 atIndex:2];
    [enc setBuffer:bVy offset:0 atIndex:3];
    [enc setBuffer:bVz offset:0 atIndex:4];
    [enc setBytes:&gp length:sizeof(gp) atIndex:5];
    [enc setBytes:&n length:sizeof(n) atIndex:6];
    [enc setBytes:&dtf length:sizeof(dtf) atIndex:7];
    NSUInteger tg = ctx.backtracePso.maxTotalThreadsPerThreadgroup;
    if (tg > 256) tg = 256;
    [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    checkCb(cb, "backtraceBatch");
    out = unpack(bOut, n);
  }
  return out;
}

FaceField advectCovectorSLGPU(const MacGrid& g, const FaceField& u, const FaceField& v, double dt) {
  Context& ctx = context();
  if (!ctx.ok) throw std::runtime_error("bochner::gpu: Metal device/pipeline unavailable");

  const uint32_t cells = static_cast<uint32_t>(g.numCells());
  const uint32_t nfx = static_cast<uint32_t>(g.numFacesX());
  const uint32_t nfy = static_cast<uint32_t>(g.numFacesY());
  const uint32_t nfz = static_cast<uint32_t>(g.numFacesZ());

  FaceField out;
  @autoreleasepool {
    const std::vector<float> vx = toF32(v.x), vy = toF32(v.y), vz = toF32(v.z);
    const std::vector<float> ux = toF32(u.x), uy = toF32(u.y), uz = toF32(u.z);
    const GridParams gp = makeGridParams(g);
    const float dtf = static_cast<float>(dt);

    id<MTLDevice> d = ctx.device;
    id<MTLBuffer> bPsi = [d newBufferWithLength:3 * cells * sizeof(float)
                                        options:MTLResourceStorageModeShared];
    id<MTLBuffer> bVx = sharedBuffer(d, vx.data(), vx.size() * sizeof(float));
    id<MTLBuffer> bVy = sharedBuffer(d, vy.data(), vy.size() * sizeof(float));
    id<MTLBuffer> bVz = sharedBuffer(d, vz.data(), vz.size() * sizeof(float));
    id<MTLBuffer> bUx = sharedBuffer(d, ux.data(), ux.size() * sizeof(float));
    id<MTLBuffer> bUy = sharedBuffer(d, uy.data(), uy.size() * sizeof(float));
    id<MTLBuffer> bUz = sharedBuffer(d, uz.data(), uz.size() * sizeof(float));
    // Out buffers start as a copy of u, so skipped boundary faces keep u's value.
    id<MTLBuffer> bOutX = sharedBuffer(d, ux.data(), ux.size() * sizeof(float));
    id<MTLBuffer> bOutY = sharedBuffer(d, uy.data(), uy.size() * sizeof(float));
    id<MTLBuffer> bOutZ = sharedBuffer(d, uz.data(), uz.size() * sizeof(float));

    id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];

    // Pass 1: the cell-center flow map. Separate encoder so its writes to bPsi are
    // ordered before the pullback reads them (automatic hazard tracking).
    {
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      [enc setComputePipelineState:ctx.slCellsPso];
      [enc setBuffer:bPsi offset:0 atIndex:0];
      [enc setBuffer:bVx offset:0 atIndex:1];
      [enc setBuffer:bVy offset:0 atIndex:2];
      [enc setBuffer:bVz offset:0 atIndex:3];
      [enc setBytes:&gp length:sizeof(gp) atIndex:4];
      [enc setBytes:&dtf length:sizeof(dtf) atIndex:5];
      dispatch1D(enc, ctx.slCellsPso, cells);
      [enc endEncoding];
    }
    // Pass 2: per-face pullback for the three families (different out buffer +
    // axis each; the shared inputs stay bound).
    {
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      [enc setComputePipelineState:ctx.slPullbackPso];
      [enc setBuffer:bPsi offset:0 atIndex:1];
      [enc setBuffer:bUx offset:0 atIndex:2];
      [enc setBuffer:bUy offset:0 atIndex:3];
      [enc setBuffer:bUz offset:0 atIndex:4];
      [enc setBuffer:bVx offset:0 atIndex:5];
      [enc setBuffer:bVy offset:0 atIndex:6];
      [enc setBuffer:bVz offset:0 atIndex:7];
      [enc setBytes:&gp length:sizeof(gp) atIndex:8];
      [enc setBytes:&dtf length:sizeof(dtf) atIndex:10];
      const int axes[3] = {0, 1, 2};
      id<MTLBuffer> outs[3] = {bOutX, bOutY, bOutZ};
      const uint32_t counts[3] = {nfx, nfy, nfz};
      for (int a = 0; a < 3; ++a) {
        [enc setBuffer:outs[a] offset:0 atIndex:0];
        [enc setBytes:&axes[a] length:sizeof(int) atIndex:9];
        dispatch1D(enc, ctx.slPullbackPso, counts[a]);
      }
      [enc endEncoding];
    }

    [cb commit];
    [cb waitUntilCompleted];
    checkCb(cb, "advectCovectorSL");

    out.x = downloadF64(bOutX, nfx);
    out.y = downloadF64(bOutY, nfy);
    out.z = downloadF64(bOutZ, nfz);
  }
  return out;
}

FaceField advectCovectorBFECCResident(const MacGrid& g, const FaceField& u, const FaceField& v,
                                      double dt) {
  Context& ctx = context();
  if (!ctx.ok) throw std::runtime_error("bochner::gpu: Metal device/pipeline unavailable");

  const uint32_t cells = static_cast<uint32_t>(g.numCells());
  const uint32_t counts[3] = {static_cast<uint32_t>(g.numFacesX()),
                              static_cast<uint32_t>(g.numFacesY()),
                              static_cast<uint32_t>(g.numFacesZ())};

  FaceField out;
  @autoreleasepool {
    id<MTLDevice> d = ctx.device;
    const GridParams gp = makeGridParams(g);
    const float dtf = static_cast<float>(dt);

    // Everything stays on the device: u and v are uploaded once; u1/u0b/eh/corr/
    // corrected and the psi map are scratch buffers never read back until the end.
    const std::vector<float> uf[3] = {toF32(u.x), toF32(u.y), toF32(u.z)};
    const std::vector<float> vf[3] = {toF32(v.x), toF32(v.y), toF32(v.z)};
    id<MTLBuffer> U[3] = {}, V[3] = {}, u1[3] = {}, u0b[3] = {}, eh[3] = {}, corr[3] = {},
                  corrected[3] = {};
    for (int a = 0; a < 3; ++a) {
      const size_t bytes = counts[a] * sizeof(float);
      U[a] = sharedBuffer(d, uf[a].data(), bytes);
      V[a] = sharedBuffer(d, vf[a].data(), bytes);
      u1[a] = [d newBufferWithLength:bytes options:MTLResourceStorageModeShared];
      u0b[a] = [d newBufferWithLength:bytes options:MTLResourceStorageModeShared];
      eh[a] = [d newBufferWithLength:bytes options:MTLResourceStorageModeShared];
      corr[a] = [d newBufferWithLength:bytes options:MTLResourceStorageModeShared];
      corrected[a] = [d newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    }
    id<MTLBuffer> psi = [d newBufferWithLength:3 * cells * sizeof(float)
                                      options:MTLResourceStorageModeShared];

    id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];

    // 1. u1 = SL(u; v, dt)   (prefill u1 <- u so boundary faces keep u)
    for (int a = 0; a < 3; ++a) blitCopy(cb, U[a], u1[a], counts[a] * sizeof(float));
    encodeSLStep(cb, ctx, u1[0], u1[1], u1[2], U[0], U[1], U[2], V[0], V[1], V[2], psi, gp,
                 dtf, cells, counts);

    // 2. u0b = SL(u1; v, -dt)
    for (int a = 0; a < 3; ++a) blitCopy(cb, u1[a], u0b[a], counts[a] * sizeof(float));
    encodeSLStep(cb, ctx, u0b[0], u0b[1], u0b[2], u1[0], u1[1], u1[2], V[0], V[1], V[2], psi,
                 gp, -dtf, cells, counts);

    // 3. eh = 0.5*(u0b - u)
    {
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      [enc setComputePipelineState:ctx.halfDiffPso];
      for (int a = 0; a < 3; ++a) {
        [enc setBuffer:eh[a] offset:0 atIndex:0];
        [enc setBuffer:u0b[a] offset:0 atIndex:1];
        [enc setBuffer:U[a] offset:0 atIndex:2];
        [enc setBytes:&counts[a] length:sizeof(uint32_t) atIndex:3];
        dispatch1D(enc, ctx.halfDiffPso, counts[a]);
      }
      [enc endEncoding];
    }

    // 4. corr = SL(eh; v, dt)
    for (int a = 0; a < 3; ++a) blitCopy(cb, eh[a], corr[a], counts[a] * sizeof(float));
    encodeSLStep(cb, ctx, corr[0], corr[1], corr[2], eh[0], eh[1], eh[2], V[0], V[1], V[2],
                 psi, gp, dtf, cells, counts);

    // 5. corrected = u1 - corr on interior faces (prefill corrected <- u1)
    for (int a = 0; a < 3; ++a) blitCopy(cb, u1[a], corrected[a], counts[a] * sizeof(float));
    {
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      [enc setComputePipelineState:ctx.subInteriorPso];
      [enc setBytes:&gp length:sizeof(gp) atIndex:2];
      for (int a = 0; a < 3; ++a) {
        [enc setBuffer:corrected[a] offset:0 atIndex:0];
        [enc setBuffer:corr[a] offset:0 atIndex:1];
        [enc setBytes:&a length:sizeof(int) atIndex:3];
        dispatch1D(enc, ctx.subInteriorPso, counts[a]);
      }
      [enc endEncoding];
    }

    // 6. limiter: clamp corrected in place to u1's neighbour extrema (CF 5.4.2)
    {
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      [enc setComputePipelineState:ctx.limiterPso];
      [enc setBytes:&gp length:sizeof(gp) atIndex:2];
      for (int a = 0; a < 3; ++a) {
        [enc setBuffer:corrected[a] offset:0 atIndex:0];
        [enc setBuffer:u1[a] offset:0 atIndex:1];
        [enc setBytes:&a length:sizeof(int) atIndex:3];
        dispatch1D(enc, ctx.limiterPso, counts[a]);
      }
      [enc endEncoding];
    }

    [cb commit];
    [cb waitUntilCompleted];
    checkCb(cb, "advectCovectorBFECC");

    out.x = downloadF64(corrected[0], counts[0]);
    out.y = downloadF64(corrected[1], counts[1]);
    out.z = downloadF64(corrected[2], counts[2]);
  }
  return out;
}

// --- CF+MCM (characteristic-map) advection primitives ------------------------
namespace {
// Pack a Vec3 map to interleaved float (3/cell); unpack the reverse.
std::vector<float> packMap(const std::vector<Vec3>& m) { return packPoints(m); }
std::vector<Vec3> unpackMap(id<MTLBuffer> buf, size_t cells) {
  const float* r = static_cast<const float*>(buf.contents);
  std::vector<Vec3> m(cells);
  for (size_t c = 0; c < cells; ++c) m[c] = {r[3 * c], r[3 * c + 1], r[3 * c + 2]};
  return m;
}
}  // namespace

std::vector<Vec3> mapAdvect(const MacGrid& g, const std::vector<Vec3>& psiOld, const FaceField& v,
                            double dt) {
  Context& ctx = context();
  if (!ctx.ok) throw std::runtime_error("bochner::gpu: Metal device/pipeline unavailable");
  const uint32_t cells = static_cast<uint32_t>(g.numCells());
  std::vector<Vec3> out;
  @autoreleasepool {
    id<MTLDevice> d = ctx.device;
    const GridParams gp = makeGridParams(g);
    const float dtf = static_cast<float>(dt);
    const std::vector<float> pf = packMap(psiOld);
    const std::vector<float> vf[3] = {toF32(v.x), toF32(v.y), toF32(v.z)};
    id<MTLBuffer> psiOldB = sharedBuffer(d, pf.data(), pf.size() * sizeof(float));
    id<MTLBuffer> psiNewB = [d newBufferWithLength:3 * cells * sizeof(float)
                                           options:MTLResourceStorageModeShared];
    id<MTLBuffer> V[3];
    for (int a = 0; a < 3; ++a) V[a] = sharedBuffer(d, vf[a].data(), vf[a].size() * sizeof(float));
    id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:ctx.mapAdvectPso];
    [enc setBuffer:psiNewB offset:0 atIndex:0];
    [enc setBuffer:psiOldB offset:0 atIndex:1];
    [enc setBuffer:V[0] offset:0 atIndex:2];
    [enc setBuffer:V[1] offset:0 atIndex:3];
    [enc setBuffer:V[2] offset:0 atIndex:4];
    [enc setBytes:&gp length:sizeof(gp) atIndex:5];
    [enc setBytes:&dtf length:sizeof(dtf) atIndex:6];
    dispatch1D(enc, ctx.mapAdvectPso, cells);
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    checkCb(cb, "mapAdvect");
    out = unpackMap(psiNewB, cells);
  }
  return out;
}

FaceField pullbackStored(const MacGrid& g, const std::vector<Vec3>& map, const FaceField& source) {
  Context& ctx = context();
  if (!ctx.ok) throw std::runtime_error("bochner::gpu: Metal device/pipeline unavailable");
  const uint32_t cells = static_cast<uint32_t>(g.numCells());
  const uint32_t counts[3] = {static_cast<uint32_t>(g.numFacesX()),
                              static_cast<uint32_t>(g.numFacesY()),
                              static_cast<uint32_t>(g.numFacesZ())};
  FaceField out;
  @autoreleasepool {
    id<MTLDevice> d = ctx.device;
    const GridParams gp = makeGridParams(g);
    const std::vector<float> mf = packMap(map);
    const std::vector<float> sf[3] = {toF32(source.x), toF32(source.y), toF32(source.z)};
    id<MTLBuffer> mapB = sharedBuffer(d, mf.data(), mf.size() * sizeof(float));
    id<MTLBuffer> S[3], O[3];
    for (int a = 0; a < 3; ++a) {
      S[a] = sharedBuffer(d, sf[a].data(), counts[a] * sizeof(float));
      O[a] = sharedBuffer(d, sf[a].data(), counts[a] * sizeof(float));  // prefill = source
    }
    id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:ctx.pullbackStoredPso];
    [enc setBuffer:mapB offset:0 atIndex:1];
    [enc setBuffer:S[0] offset:0 atIndex:2];
    [enc setBuffer:S[1] offset:0 atIndex:3];
    [enc setBuffer:S[2] offset:0 atIndex:4];
    [enc setBytes:&gp length:sizeof(gp) atIndex:5];
    for (int a = 0; a < 3; ++a) {
      [enc setBuffer:O[a] offset:0 atIndex:0];
      [enc setBytes:&a length:sizeof(int) atIndex:6];
      dispatch1D(enc, ctx.pullbackStoredPso, counts[a]);
    }
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    checkCb(cb, "pullbackStored");
    out.x = downloadF64(O[0], counts[0]);
    out.y = downloadF64(O[1], counts[1]);
    out.z = downloadF64(O[2], counts[2]);
  }
  return out;
}

double mapDistortion(const MacGrid& g, const std::vector<Vec3>& psi, const std::vector<Vec3>& phi) {
  Context& ctx = context();
  if (!ctx.ok) throw std::runtime_error("bochner::gpu: Metal device/pipeline unavailable");
  const uint32_t cells = static_cast<uint32_t>(g.numCells());
  double result = 0.0;
  @autoreleasepool {
    id<MTLDevice> d = ctx.device;
    const GridParams gp = makeGridParams(g);
    const std::vector<float> pf = packMap(psi), qf = packMap(phi);
    id<MTLBuffer> psiB = sharedBuffer(d, pf.data(), pf.size() * sizeof(float));
    id<MTLBuffer> phiB = sharedBuffer(d, qf.data(), qf.size() * sizeof(float));
    id<MTLBuffer> d2 = [d newBufferWithLength:cells * sizeof(float)
                                      options:MTLResourceStorageModeShared];
    const uint32_t nTG = (cells + 255) / 256;
    id<MTLBuffer> partials = [d newBufferWithLength:nTG * sizeof(float)
                                            options:MTLResourceStorageModeShared];
    id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
    {
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      [enc setComputePipelineState:ctx.distortionCellPso];
      [enc setBuffer:d2 offset:0 atIndex:0];
      [enc setBuffer:psiB offset:0 atIndex:1];
      [enc setBuffer:phiB offset:0 atIndex:2];
      [enc setBytes:&gp length:sizeof(gp) atIndex:3];
      dispatch1D(enc, ctx.distortionCellPso, cells);
      [enc endEncoding];
    }
    {
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      [enc setComputePipelineState:ctx.maxPartialPso];
      [enc setBuffer:partials offset:0 atIndex:0];
      [enc setBuffer:d2 offset:0 atIndex:1];
      [enc setBytes:&cells length:sizeof(uint32_t) atIndex:2];
      [enc dispatchThreads:MTLSizeMake(nTG * 256, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [enc endEncoding];
    }
    [cb commit];
    [cb waitUntilCompleted];
    checkCb(cb, "mapDistortion");
    const float* pr = static_cast<const float*>(partials.contents);
    float m = 0.0f;
    for (uint32_t t = 0; t < nTG; ++t) m = std::max(m, pr[t]);
    result = std::sqrt(static_cast<double>(m));
  }
  return result;
}

// --- Resident CF+MCM state ---------------------------------------------------
namespace {

// Copy a double face/vector into an existing shared float buffer.
void writeF32(id<MTLBuffer> buf, const std::vector<double>& a) {
  float* p = static_cast<float*>(buf.contents);
  for (size_t i = 0; i < a.size(); ++i) p[i] = static_cast<float>(a[i]);
}

struct GpuCfMcm {
  bool valid = false;
  int nx = 0, ny = 0, nz = 0;
  uint32_t cells = 0;
  uint32_t counts[3] = {0, 0, 0};
  GridParams gp{};
  // Flow maps: real (advanced by the corrector) + predictor scratch, each with an
  // alt buffer for the out-of-place advect (swapped after each map step).
  id<MTLBuffer> psi = nil, phi = nil, psiAlt = nil, phiAlt = nil;
  id<MTLBuffer> psiS = nil, phiS = nil, psiSAlt = nil, phiSAlt = nil;
  // Persistent snapshot u0; per-substep flow V and the face scratch triples.
  id<MTLBuffer> u0[3] = {}, V[3] = {}, u1[3] = {}, ut0[3] = {}, he[3] = {}, corr[3] = {},
                uPre[3] = {}, uPost[3] = {};
  // Distortion reduction scratch.
  id<MTLBuffer> d2 = nil, partials = nil;
  int nTG = 0;
};

// Register a resident handle, reusing the first slot a free*() left empty. The
// viewers upload+free a gauge hierarchy every extraction frame, so a plain
// push_back would grow the registry (and realloc) unbounded for the process
// lifetime; the free-list keeps it bounded by the live-handle high-water mark.
template <typename T>
int registerHandle(std::vector<std::unique_ptr<T>>& reg, std::unique_ptr<T> h) {
  for (std::size_t i = 0; i < reg.size(); ++i)
    if (!reg[i]) {
      reg[i] = std::move(h);
      return static_cast<int>(i);
    }
  reg.push_back(std::move(h));
  return static_cast<int>(reg.size()) - 1;
}

std::vector<std::unique_ptr<GpuCfMcm>>& cfMcmRegistry() {
  static std::vector<std::unique_ptr<GpuCfMcm>> reg;
  return reg;
}

// Encode one map advect (out-of-place) into `cb`: dst(x) = src(backtrace(x; v, dt)).
void encodeMapAdvect(id<MTLCommandBuffer> cb, Context& ctx, id<MTLBuffer> dst, id<MTLBuffer> src,
                     id<MTLBuffer> vX, id<MTLBuffer> vY, id<MTLBuffer> vZ, const GridParams& gp,
                     float dtf, uint32_t cells) {
  id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
  [enc setComputePipelineState:ctx.mapAdvectPso];
  [enc setBuffer:dst offset:0 atIndex:0];
  [enc setBuffer:src offset:0 atIndex:1];
  [enc setBuffer:vX offset:0 atIndex:2];
  [enc setBuffer:vY offset:0 atIndex:3];
  [enc setBuffer:vZ offset:0 atIndex:4];
  [enc setBytes:&gp length:sizeof(gp) atIndex:5];
  [enc setBytes:&dtf length:sizeof(dtf) atIndex:6];
  dispatch1D(enc, ctx.mapAdvectPso, cells);
  [enc endEncoding];
}

// Encode the forward-map march (Alg. 5 line 6): dst = backtrace(src positions; v, -dt).
void encodeMapMarch(id<MTLCommandBuffer> cb, Context& ctx, id<MTLBuffer> dst, id<MTLBuffer> src,
                    id<MTLBuffer> vX, id<MTLBuffer> vY, id<MTLBuffer> vZ, const GridParams& gp,
                    float negdt, uint32_t cells) {
  id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
  [enc setComputePipelineState:ctx.backtracePso];
  [enc setBuffer:dst offset:0 atIndex:0];
  [enc setBuffer:src offset:0 atIndex:1];
  [enc setBuffer:vX offset:0 atIndex:2];
  [enc setBuffer:vY offset:0 atIndex:3];
  [enc setBuffer:vZ offset:0 atIndex:4];
  [enc setBytes:&gp length:sizeof(gp) atIndex:5];
  [enc setBytes:&cells length:sizeof(uint32_t) atIndex:6];
  [enc setBytes:&negdt length:sizeof(float) atIndex:7];
  dispatch1D(enc, ctx.backtracePso, cells);
  [enc endEncoding];
}

// Encode a stored-map pullback (== pullbackThroughMap): out[a] = dMap . src(Map(fc)),
// interior faces only (out must be prefilled with src for the boundary faces).
void encodePullbackStored(id<MTLCommandBuffer> cb, Context& ctx, __strong id<MTLBuffer> out[3],
                          id<MTLBuffer> map, __strong id<MTLBuffer> src[3], const GridParams& gp,
                          const uint32_t counts[3]) {
  id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
  [enc setComputePipelineState:ctx.pullbackStoredPso];
  [enc setBuffer:map offset:0 atIndex:1];
  [enc setBuffer:src[0] offset:0 atIndex:2];
  [enc setBuffer:src[1] offset:0 atIndex:3];
  [enc setBuffer:src[2] offset:0 atIndex:4];
  [enc setBytes:&gp length:sizeof(gp) atIndex:5];
  for (int a = 0; a < 3; ++a) {
    [enc setBuffer:out[a] offset:0 atIndex:0];
    [enc setBytes:&a length:sizeof(int) atIndex:6];
    dispatch1D(enc, ctx.pullbackStoredPso, counts[a]);
  }
  [enc endEncoding];
}

}  // namespace

int cfMcmUpload(const MacGrid& g, const FaceField& u0) {
  Context& ctx = context();
  if (!ctx.ok) throw std::runtime_error("bochner::gpu: Metal device/pipeline unavailable");
  if (u0.x.size() != static_cast<size_t>(g.numFacesX()) ||
      u0.y.size() != static_cast<size_t>(g.numFacesY()) ||
      u0.z.size() != static_cast<size_t>(g.numFacesZ()))
    throw std::runtime_error("bochner::gpu: cfMcmUpload u0 size mismatch with grid");
  auto h = std::make_unique<GpuCfMcm>();
  h->nx = g.nx(); h->ny = g.ny(); h->nz = g.nz();
  h->cells = static_cast<uint32_t>(g.numCells());
  h->counts[0] = static_cast<uint32_t>(g.numFacesX());
  h->counts[1] = static_cast<uint32_t>(g.numFacesY());
  h->counts[2] = static_cast<uint32_t>(g.numFacesZ());
  h->gp = makeGridParams(g);
  @autoreleasepool {
    id<MTLDevice> d = ctx.device;
    const size_t mapBytes = 3 * h->cells * sizeof(float);
    auto mk = [&](size_t bytes) {
      return [d newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    };
    h->psi = mk(mapBytes); h->phi = mk(mapBytes);
    h->psiAlt = mk(mapBytes); h->phiAlt = mk(mapBytes);
    h->psiS = mk(mapBytes); h->phiS = mk(mapBytes);
    h->psiSAlt = mk(mapBytes); h->phiSAlt = mk(mapBytes);
    const std::vector<double>* u0c[3] = {&u0.x, &u0.y, &u0.z};
    for (int a = 0; a < 3; ++a) {
      const size_t b = h->counts[a] * sizeof(float);
      h->u0[a] = mk(b); writeF32(h->u0[a], *u0c[a]);
      h->V[a] = mk(b); h->u1[a] = mk(b); h->ut0[a] = mk(b);
      h->he[a] = mk(b); h->corr[a] = mk(b); h->uPre[a] = mk(b); h->uPost[a] = mk(b);
    }
    h->d2 = mk(h->cells * sizeof(float));
    h->nTG = (h->cells + 255) / 256;
    h->partials = mk(h->nTG * sizeof(float));

    // Psi, Phi <- identity.
    id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:ctx.mapIdentityPso];
    [enc setBytes:&h->gp length:sizeof(h->gp) atIndex:1];
    for (id<MTLBuffer> m : {h->psi, h->phi}) {
      [enc setBuffer:m offset:0 atIndex:0];
      dispatch1D(enc, ctx.mapIdentityPso, h->cells);
    }
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    checkCb(cb, "cfMcmUpload");
  }
  h->valid = true;
  return registerHandle(cfMcmRegistry(), std::move(h));
}

void cfMcmFree(int handle) {
  auto& reg = cfMcmRegistry();
  if (handle >= 0 && handle < static_cast<int>(reg.size())) reg[handle].reset();
}

FaceField cfMcmAdvect(int handle, const FaceField& v, double dt, bool scratch, bool useLimiter) {
  Context& ctx = context();
  auto& reg = cfMcmRegistry();
  if (!ctx.ok || handle < 0 || handle >= static_cast<int>(reg.size()) || !reg[handle] ||
      !reg[handle]->valid)
    throw std::runtime_error("bochner::gpu: invalid CF+MCM handle / Metal unavailable");
  GpuCfMcm& H = *reg[handle];
  const uint32_t* counts = H.counts;
  if (v.x.size() != counts[0] || v.y.size() != counts[1] || v.z.size() != counts[2])
    throw std::runtime_error("bochner::gpu: cfMcmAdvect velocity size mismatch with handle grid");
  FaceField out;
  @autoreleasepool {
    const float dtf = static_cast<float>(dt);
    const std::vector<double>* vc[3] = {&v.x, &v.y, &v.z};
    for (int a = 0; a < 3; ++a) writeF32(H.V[a], *vc[a]);

    // Select the map set: scratch (predictor, on a fresh copy of the real maps)
    // or real (corrector). Track the "current" map buffer as a local; the advect
    // is out-of-place into the alt, then we point current at the alt.
    id<MTLBuffer> psiCur = scratch ? H.psiS : H.psi;
    id<MTLBuffer> psiAlt = scratch ? H.psiSAlt : H.psiAlt;
    id<MTLBuffer> phiCur = scratch ? H.phiS : H.phi;
    id<MTLBuffer> phiAlt = scratch ? H.phiSAlt : H.phiAlt;

    id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
    if (scratch) {
      const size_t mb = 3 * H.cells * sizeof(float);
      blitCopy(cb, H.psi, psiCur, mb);
      blitCopy(cb, H.phi, phiCur, mb);
    }
    // Line 4: advect Psi backward -> psiAlt; psiCur becomes psiAlt.
    encodeMapAdvect(cb, ctx, psiAlt, psiCur, H.V[0], H.V[1], H.V[2], H.gp, dtf, H.cells);
    std::swap(psiCur, psiAlt);
    // Line 5: u1 = dPsi^T(u0 o Psi). Prefill u1 <- u0 (boundary faces keep u0).
    for (int a = 0; a < 3; ++a) blitCopy(cb, H.u0[a], H.u1[a], counts[a] * sizeof(float));
    encodePullbackStored(cb, ctx, H.u1, psiCur, H.u0, H.gp, counts);
    // Line 6: march Phi forward -> phiAlt; phiCur becomes phiAlt.
    encodeMapMarch(cb, ctx, phiAlt, phiCur, H.V[0], H.V[1], H.V[2], H.gp, -dtf, H.cells);
    std::swap(phiCur, phiAlt);
    // Line 7: ut0 = dPhi^T(u1 o Phi). Prefill ut0 <- u1.
    for (int a = 0; a < 3; ++a) blitCopy(cb, H.u1[a], H.ut0[a], counts[a] * sizeof(float));
    encodePullbackStored(cb, ctx, H.ut0, phiCur, H.u1, H.gp, counts);
    // Line 8: he = 0.5*(ut0 - u0) over all faces (boundary faces -> 0).
    {
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      [enc setComputePipelineState:ctx.halfDiffPso];
      for (int a = 0; a < 3; ++a) {
        [enc setBuffer:H.he[a] offset:0 atIndex:0];
        [enc setBuffer:H.ut0[a] offset:0 atIndex:1];
        [enc setBuffer:H.u0[a] offset:0 atIndex:2];
        [enc setBytes:&counts[a] length:sizeof(uint32_t) atIndex:3];
        dispatch1D(enc, ctx.halfDiffPso, counts[a]);
      }
      [enc endEncoding];
    }
    // Line 9: corr = dPsi^T(he o Psi) (prefill corr <- he); uPre = u1 - corr (all faces).
    for (int a = 0; a < 3; ++a) blitCopy(cb, H.he[a], H.corr[a], counts[a] * sizeof(float));
    encodePullbackStored(cb, ctx, H.corr, psiCur, H.he, H.gp, counts);
    {
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      [enc setComputePipelineState:ctx.subPso];
      for (int a = 0; a < 3; ++a) {
        [enc setBuffer:H.uPre[a] offset:0 atIndex:0];
        [enc setBuffer:H.u1[a] offset:0 atIndex:1];
        [enc setBuffer:H.corr[a] offset:0 atIndex:2];
        [enc setBytes:&counts[a] length:sizeof(uint32_t) atIndex:3];
        dispatch1D(enc, ctx.subPso, counts[a]);
      }
      [enc endEncoding];
    }
    // Limiter: clamp uPre (interior) to u1's neighbour extrema (CF 5.4.2).
    if (useLimiter) {
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      [enc setComputePipelineState:ctx.limiterPso];
      [enc setBytes:&H.gp length:sizeof(H.gp) atIndex:2];
      for (int a = 0; a < 3; ++a) {
        [enc setBuffer:H.uPre[a] offset:0 atIndex:0];
        [enc setBuffer:H.u1[a] offset:0 atIndex:1];
        [enc setBytes:&a length:sizeof(int) atIndex:3];
        dispatch1D(enc, ctx.limiterPso, counts[a]);
      }
      [enc endEncoding];
    }
    [cb commit];
    [cb waitUntilCompleted];
    // Throw BEFORE persisting the swap: a fault leaves the handle's real maps
    // naming the pre-step buffers, which this command buffer only read.
    checkCb(cb, "cfMcmAdvect");

    // Persist the advanced map buffers back into the handle (swap so `psi`/`phi`
    // name the just-written buffer for the next step / the accumulate).
    if (scratch) { H.psiS = psiCur; H.psiSAlt = psiAlt; H.phiS = phiCur; H.phiSAlt = phiAlt; }
    else { H.psi = psiCur; H.psiAlt = psiAlt; H.phi = phiCur; H.phiAlt = phiAlt; }

    out.x = downloadF64(H.uPre[0], counts[0]);
    out.y = downloadF64(H.uPre[1], counts[1]);
    out.z = downloadF64(H.uPre[2], counts[2]);
  }
  return out;
}

void cfMcmAccumulate(int handle, const FaceField& uPost) {
  Context& ctx = context();
  auto& reg = cfMcmRegistry();
  if (!ctx.ok || handle < 0 || handle >= static_cast<int>(reg.size()) || !reg[handle] ||
      !reg[handle]->valid)
    throw std::runtime_error("bochner::gpu: invalid CF+MCM handle / Metal unavailable");
  GpuCfMcm& H = *reg[handle];
  const uint32_t* counts = H.counts;
  if (uPost.x.size() != counts[0] || uPost.y.size() != counts[1] || uPost.z.size() != counts[2])
    throw std::runtime_error("bochner::gpu: cfMcmAccumulate uPost size mismatch with handle grid");
  @autoreleasepool {
    const std::vector<double>* uc[3] = {&uPost.x, &uPost.y, &uPost.z};
    for (int a = 0; a < 3; ++a) writeF32(H.uPost[a], *uc[a]);
    id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
    // du = uPost - uPre (reuse he as du); duBack = dPhi^T(du o Phi) (reuse corr).
    {
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      [enc setComputePipelineState:ctx.subPso];
      for (int a = 0; a < 3; ++a) {
        [enc setBuffer:H.he[a] offset:0 atIndex:0];
        [enc setBuffer:H.uPost[a] offset:0 atIndex:1];
        [enc setBuffer:H.uPre[a] offset:0 atIndex:2];
        [enc setBytes:&counts[a] length:sizeof(uint32_t) atIndex:3];
        dispatch1D(enc, ctx.subPso, counts[a]);
      }
      [enc endEncoding];
    }
    for (int a = 0; a < 3; ++a) blitCopy(cb, H.he[a], H.corr[a], counts[a] * sizeof(float));
    encodePullbackStored(cb, ctx, H.corr, H.phi, H.he, H.gp, counts);
    // u0 += duBack.
    {
      const float one = 1.0f;
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      [enc setComputePipelineState:ctx.axpyPso];
      for (int a = 0; a < 3; ++a) {
        [enc setBuffer:H.u0[a] offset:0 atIndex:0];
        [enc setBuffer:H.corr[a] offset:0 atIndex:1];
        [enc setBytes:&one length:sizeof(float) atIndex:2];
        [enc setBytes:&counts[a] length:sizeof(uint32_t) atIndex:3];
        dispatch1D(enc, ctx.axpyPso, counts[a]);
      }
      [enc endEncoding];
    }
    [cb commit];
    [cb waitUntilCompleted];
    checkCb(cb, "cfMcmAccumulate");
  }
}

void cfMcmReset(int handle, const FaceField& uPost) {
  Context& ctx = context();
  auto& reg = cfMcmRegistry();
  if (!ctx.ok || handle < 0 || handle >= static_cast<int>(reg.size()) || !reg[handle] ||
      !reg[handle]->valid)
    throw std::runtime_error("bochner::gpu: invalid CF+MCM handle / Metal unavailable");
  GpuCfMcm& H = *reg[handle];
  if (uPost.x.size() != H.counts[0] || uPost.y.size() != H.counts[1] ||
      uPost.z.size() != H.counts[2])
    throw std::runtime_error("bochner::gpu: cfMcmReset uPost size mismatch with handle grid");
  @autoreleasepool {
    const std::vector<double>* uc[3] = {&uPost.x, &uPost.y, &uPost.z};
    for (int a = 0; a < 3; ++a) writeF32(H.u0[a], *uc[a]);
    id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:ctx.mapIdentityPso];
    [enc setBytes:&H.gp length:sizeof(H.gp) atIndex:1];
    for (id<MTLBuffer> m : {H.psi, H.phi}) {
      [enc setBuffer:m offset:0 atIndex:0];
      dispatch1D(enc, ctx.mapIdentityPso, H.cells);
    }
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    checkCb(cb, "cfMcmReset");
  }
}

double cfMcmDistortion(int handle) {
  Context& ctx = context();
  auto& reg = cfMcmRegistry();
  if (!ctx.ok || handle < 0 || handle >= static_cast<int>(reg.size()) || !reg[handle] ||
      !reg[handle]->valid)
    throw std::runtime_error("bochner::gpu: invalid CF+MCM handle / Metal unavailable");
  GpuCfMcm& H = *reg[handle];
  double result = 0.0;
  @autoreleasepool {
    id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
    {
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      [enc setComputePipelineState:ctx.distortionCellPso];
      [enc setBuffer:H.d2 offset:0 atIndex:0];
      [enc setBuffer:H.psi offset:0 atIndex:1];
      [enc setBuffer:H.phi offset:0 atIndex:2];
      [enc setBytes:&H.gp length:sizeof(H.gp) atIndex:3];
      dispatch1D(enc, ctx.distortionCellPso, H.cells);
      [enc endEncoding];
    }
    {
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      [enc setComputePipelineState:ctx.maxPartialPso];
      [enc setBuffer:H.partials offset:0 atIndex:0];
      [enc setBuffer:H.d2 offset:0 atIndex:1];
      [enc setBytes:&H.cells length:sizeof(uint32_t) atIndex:2];
      [enc dispatchThreads:MTLSizeMake(H.nTG * 256, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [enc endEncoding];
    }
    [cb commit];
    [cb waitUntilCompleted];
    checkCb(cb, "cfMcmDistortion");
    const float* pr = static_cast<const float*>(H.partials.contents);
    float m = 0.0f;
    for (int t = 0; t < H.nTG; ++t) m = std::max(m, pr[t]);
    result = std::sqrt(static_cast<double>(m));
  }
  return result;
}

// --- GPU Poisson MGPCG -------------------------------------------------------
namespace {

struct GpuLevel {
  int n = 0;
  id<MTLBuffer> rowStart = nil, col = nil, val = nil, diag = nil, active = nil, aggUp = nil;
  id<MTLBuffer> xLev = nil, xTmp = nil, bLev = nil, Ax = nil;  // V-cycle scratch
};

struct GpuHierarchy {
  bool valid = false;
  std::vector<GpuLevel> levels;
  // CG vectors on the fine level (separate from the V-cycle's level-0 scratch).
  id<MTLBuffer> bCG = nil, xCG = nil, rBuf = nil, zBuf = nil, pBuf = nil, ApBuf = nil, partials = nil;
  int nPartials = 0;
};

std::vector<std::unique_ptr<GpuHierarchy>>& poissonRegistry() {
  static std::vector<std::unique_ptr<GpuHierarchy>> reg;
  return reg;
}

constexpr int kTG = 256;  // threadgroup size for the dot reductions

// Encode one kernel that touches only the (up to 5) given buffers plus a trailing
// constant uint n, over `n` threads, in its own compute encoder (auto hazard
// ordering across encoders).
void encode(id<MTLCommandBuffer> cb, id<MTLComputePipelineState> pso,
            NSArray<id<MTLBuffer>>* bufs, uint32_t n) {
  id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
  [enc setComputePipelineState:pso];
  for (NSUInteger i = 0; i < bufs.count; ++i) [enc setBuffer:bufs[i] offset:0 atIndex:i];
  [enc setBytes:&n length:sizeof(uint32_t) atIndex:bufs.count];
  NSUInteger tg = pso.maxTotalThreadsPerThreadgroup;
  if (tg > 256) tg = 256;
  [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
  [enc endEncoding];
}

void fill0(id<MTLCommandBuffer> cb, id<MTLBuffer> b, size_t len) {
  id<MTLBlitCommandEncoder> e = [cb blitCommandEncoder];
  [e fillBuffer:b range:NSMakeRange(0, len) value:0];
  [e endEncoding];
}

void encodeSmooth(id<MTLCommandBuffer> cb, Context& ctx, GpuLevel& L, int sweeps) {
  for (int s = 0; s < sweeps; ++s) {
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:ctx.jacobiPso];
    id<MTLBuffer> ins[] = {L.xTmp, L.xLev, L.rowStart, L.col, L.val, L.diag, L.active, L.bLev};
    for (int i = 0; i < 8; ++i) [enc setBuffer:ins[i] offset:0 atIndex:i];
    const uint32_t n = static_cast<uint32_t>(L.n);
    [enc setBytes:&n length:sizeof(n) atIndex:8];
    NSUInteger tg = ctx.jacobiPso.maxTotalThreadsPerThreadgroup;
    if (tg > 256) tg = 256;
    [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [enc endEncoding];
    std::swap(L.xLev, L.xTmp);  // result is now in xLev
  }
}

// Encode a full symmetric V-cycle: precondition rIn -> zOut (both fine-level).
void encodeVcycle(id<MTLCommandBuffer> cb, Context& ctx, GpuHierarchy& H, id<MTLBuffer> rIn,
                  id<MTLBuffer> zOut) {
  const int L = static_cast<int>(H.levels.size());
  auto NB = [](int n) { return static_cast<size_t>(n) * sizeof(float); };

  // Level 0: bLev = rIn, xLev = 0.
  {
    id<MTLBlitCommandEncoder> e = [cb blitCommandEncoder];
    [e copyFromBuffer:rIn sourceOffset:0 toBuffer:H.levels[0].bLev destinationOffset:0
                 size:NB(H.levels[0].n)];
    [e endEncoding];
  }
  fill0(cb, H.levels[0].xLev, NB(H.levels[0].n));

  for (int l = 0; l < L - 1; ++l) {
    GpuLevel& cur = H.levels[l];
    GpuLevel& nxt = H.levels[l + 1];
    encodeSmooth(cb, ctx, cur, 2);
    encode(cb, ctx.csrSpmvPso, @[ cur.Ax, cur.rowStart, cur.col, cur.val, cur.xLev ], cur.n);
    encode(cb, ctx.subPso, @[ cur.xTmp, cur.bLev, cur.Ax ], cur.n);  // xTmp = residual
    fill0(cb, nxt.bLev, NB(nxt.n));
    encode(cb, ctx.restrictPso, @[ nxt.bLev, cur.xTmp, cur.aggUp, cur.active ], cur.n);
    fill0(cb, nxt.xLev, NB(nxt.n));
  }
  encodeSmooth(cb, ctx, H.levels[L - 1], 40);
  for (int l = L - 2; l >= 0; --l) {
    GpuLevel& cur = H.levels[l];
    encode(cb, ctx.prolongPso, @[ cur.xLev, H.levels[l + 1].xLev, cur.aggUp, cur.active ], cur.n);
    encodeSmooth(cb, ctx, cur, 2);
  }
  {
    id<MTLBlitCommandEncoder> e = [cb blitCommandEncoder];
    [e copyFromBuffer:H.levels[0].xLev sourceOffset:0 toBuffer:zOut destinationOffset:0
                 size:NB(H.levels[0].n)];
    [e endEncoding];
  }
}

// dot(a,b): encode the partial reduction, then (after the buffer completes) sum
// the per-threadgroup partials on the host in double.
void encodeDot(id<MTLCommandBuffer> cb, Context& ctx, GpuHierarchy& H, id<MTLBuffer> a,
               id<MTLBuffer> b, int n) {
  const uint32_t groups = (n + kTG - 1) / kTG;
  id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
  [enc setComputePipelineState:ctx.dotPso];
  [enc setBuffer:H.partials offset:0 atIndex:0];
  [enc setBuffer:a offset:0 atIndex:1];
  [enc setBuffer:b offset:0 atIndex:2];
  const uint32_t un = static_cast<uint32_t>(n);
  [enc setBytes:&un length:sizeof(un) atIndex:3];
  [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(kTG, 1, 1)];
  [enc endEncoding];
}

double readDot(GpuHierarchy& H, int n) {
  const uint32_t groups = (n + kTG - 1) / kTG;
  const float* p = static_cast<const float*>(H.partials.contents);
  double s = 0.0;
  for (uint32_t i = 0; i < groups; ++i) s += p[i];
  return s;
}

}  // namespace

int uploadPoisson(const MgHierarchy& H) {
  Context& ctx = context();
  if (!ctx.ok) return -1;
  auto gh = std::make_unique<GpuHierarchy>();
  @autoreleasepool {
    id<MTLDevice> d = ctx.device;
    auto lbuf = [&](size_t bytes) {
      return [d newBufferWithLength:(bytes ? bytes : 4) options:MTLResourceStorageModeShared];
    };
    auto sbuf = [&](const void* p, size_t bytes) {
      // Empty arrays (e.g. the coarsest level's aggUp) have a null data pointer;
      // never memcpy from it -- hand back a dummy buffer instead.
      if (bytes == 0) return lbuf(0);
      return [d newBufferWithBytes:p length:bytes options:MTLResourceStorageModeShared];
    };
    for (const MgLevel& lv : H.levels) {
      GpuLevel gl;
      gl.n = lv.A.n();
      const std::vector<float> valf(lv.A.val.begin(), lv.A.val.end());
      const std::vector<float> diagf(lv.A.diag.begin(), lv.A.diag.end());
      const std::vector<uint8_t> act(lv.active.begin(), lv.active.end());
      gl.rowStart = sbuf(lv.A.rowStart.data(), lv.A.rowStart.size() * sizeof(int));
      gl.col = sbuf(lv.A.col.data(), lv.A.col.size() * sizeof(int));
      gl.val = sbuf(valf.data(), valf.size() * sizeof(float));
      gl.diag = sbuf(diagf.data(), diagf.size() * sizeof(float));
      gl.active = sbuf(act.data(), act.size());
      gl.aggUp = sbuf(lv.aggUp.data(), lv.aggUp.size() * sizeof(int));
      const size_t nb = static_cast<size_t>(gl.n) * sizeof(float);
      gl.xLev = lbuf(nb);
      gl.xTmp = lbuf(nb);
      gl.bLev = lbuf(nb);
      gl.Ax = lbuf(nb);
      gh->levels.push_back(gl);
    }
    const int n0 = H.levels[0].A.n();
    const size_t nb0 = static_cast<size_t>(n0) * sizeof(float);
    gh->bCG = lbuf(nb0);
    gh->xCG = lbuf(nb0);
    gh->rBuf = lbuf(nb0);
    gh->zBuf = lbuf(nb0);
    gh->pBuf = lbuf(nb0);
    gh->ApBuf = lbuf(nb0);
    gh->nPartials = (n0 + kTG - 1) / kTG;
    gh->partials = lbuf(static_cast<size_t>(gh->nPartials) * sizeof(float));
    gh->valid = true;
  }
  return registerHandle(poissonRegistry(), std::move(gh));
}

void freePoisson(int handle) {
  auto& reg = poissonRegistry();
  if (handle >= 0 && handle < static_cast<int>(reg.size())) reg[handle].reset();
}

int mgpcgSolvePoisson(int handle, const std::vector<double>& b, std::vector<double>& x, double tol,
                      int maxit, double* relResidualOut) {
  Context& ctx = context();
  auto& reg = poissonRegistry();
  if (!ctx.ok || handle < 0 || handle >= static_cast<int>(reg.size()) || !reg[handle] ||
      !reg[handle]->valid)
    throw std::runtime_error("bochner::gpu: invalid Poisson handle / Metal unavailable");
  GpuHierarchy& H = *reg[handle];
  const int n = H.levels[0].n;
  if (static_cast<int>(b.size()) != n || static_cast<int>(x.size()) != n)
    throw std::runtime_error("bochner::gpu: mgpcgSolvePoisson b/x size mismatch with hierarchy");

  int iters = 0;
  double relRes = 0.0;
  @autoreleasepool {
    // Upload b and the warm start x (double -> float) into the CG buffers.
    GpuLevel& fine = H.levels[0];
    id<MTLBuffer> bBuf = H.bCG, xBuf = H.xCG;
    {
      float* bp = static_cast<float*>(bBuf.contents);
      float* xp = static_cast<float*>(xBuf.contents);
      for (int i = 0; i < n; ++i) {
        bp[i] = static_cast<float>(b[i]);
        xp[i] = static_cast<float>(x[i]);
      }
    }
    id<MTLBuffer> rBuf = H.rBuf, zBuf = H.zBuf, pBuf = H.pBuf, ApBuf = H.ApBuf;

    // checkCb after every wait: x (the caller's phi_) is only written at the
    // final copy-out below, so a throw leaves the warm start intact.
    auto runDot = [&](id<MTLBuffer> a, id<MTLBuffer> c) {
      id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
      encodeDot(cb, ctx, H, a, c, n);
      [cb commit];
      [cb waitUntilCompleted];
      checkCb(cb, "mgpcg/dot");
      return readDot(H, n);
    };

    // Setup: r = b - A x; z = M r; p = z; rz = <r,z>; and the initial ||r||.
    {
      id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
      encode(cb, ctx.csrSpmvPso, @[ ApBuf, fine.rowStart, fine.col, fine.val, xBuf ], n);
      encode(cb, ctx.subPso, @[ rBuf, bBuf, ApBuf ], n);
      encodeVcycle(cb, ctx, H, rBuf, zBuf);
      {
        id<MTLBlitCommandEncoder> e = [cb blitCommandEncoder];
        [e copyFromBuffer:zBuf sourceOffset:0 toBuffer:pBuf destinationOffset:0
                     size:static_cast<size_t>(n) * sizeof(float)];
        [e endEncoding];
      }
      [cb commit];
      [cb waitUntilCompleted];
      checkCb(cb, "mgpcg/setup");
    }
    const double bnorm = runDot(bBuf, bBuf);
    double rr = runDot(rBuf, rBuf);
    const double thresh = tol * tol * bnorm;
    auto finish = [&](double rn) {
      relRes = bnorm > 0.0 ? std::sqrt(rn / bnorm) : 0.0;
    };
    if (bnorm == 0.0) {
      // b = 0: the exact SPD solution is x = 0, NOT the untouched warm start
      // (whose gradient the projector subtracts unconditionally) -- the same
      // contract as the CPU solvers' zero-RHS branch.
      std::fill(x.begin(), x.end(), 0.0);
      if (relResidualOut) *relResidualOut = 0.0;
      return 0;
    }
    if (rr <= thresh) {
      finish(rr);
    } else {
      double rz = runDot(rBuf, zBuf);
      for (iters = 0; iters < maxit; ++iters) {
        // Ap = A p ; pAp = <p, Ap>
        double pAp;
        {
          id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
          encode(cb, ctx.csrSpmvPso, @[ ApBuf, fine.rowStart, fine.col, fine.val, pBuf ], n);
          encodeDot(cb, ctx, H, pBuf, ApBuf, n);
          [cb commit];
          [cb waitUntilCompleted];
          checkCb(cb, "mgpcg/spmv");
          pAp = readDot(H, n);
        }
        if (!(pAp > 0.0)) break;
        const float alpha = static_cast<float>(rz / pAp);
        const float nalpha = -alpha;
        // x += alpha p ; r -= alpha Ap ; z = M r ; then read ||r|| and <r,z>.
        double rzNew;
        {
          id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
          encode(cb, ctx.axpyPso, @[ xBuf, pBuf, [ctx.device newBufferWithBytes:&alpha length:4 options:MTLResourceStorageModeShared] ], n);
          encode(cb, ctx.axpyPso, @[ rBuf, ApBuf, [ctx.device newBufferWithBytes:&nalpha length:4 options:MTLResourceStorageModeShared] ], n);
          encodeVcycle(cb, ctx, H, rBuf, zBuf);
          encodeDot(cb, ctx, H, rBuf, rBuf, n);  // partials overwritten; read rr first
          [cb commit];
          [cb waitUntilCompleted];
          checkCb(cb, "mgpcg/axpy-vcycle");
          rr = readDot(H, n);
        }
        if (rr < thresh) {
          ++iters;
          break;
        }
        rzNew = runDot(rBuf, zBuf);
        const float beta = static_cast<float>(rzNew / rz);
        {
          id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
          encode(cb, ctx.xpayPso, @[ pBuf, zBuf, [ctx.device newBufferWithBytes:&beta length:4 options:MTLResourceStorageModeShared] ], n);
          [cb commit];
          [cb waitUntilCompleted];
          checkCb(cb, "mgpcg/xpay");
        }
        rz = rzNew;
      }
      finish(rr);
    }

    // Read the solution back (float -> double).
    const float* xp = static_cast<const float*>(xBuf.contents);
    for (int i = 0; i < n; ++i) x[i] = static_cast<double>(xp[i]);
  }
  if (relResidualOut) *relResidualOut = relRes;
  return iters;
}

std::vector<std::complex<double>> connectionMatvec(
    int lx, int ly, int lz, bool periodic, double w, const std::vector<std::complex<double>>& tx,
    const std::vector<std::complex<double>>& ty, const std::vector<std::complex<double>>& tz,
    const std::vector<std::complex<double>>& x) {
  Context& ctx = context();
  if (!ctx.ok) throw std::runtime_error("bochner::gpu: Metal device/pipeline unavailable");
  const uint32_t n = static_cast<uint32_t>(x.size());
  std::vector<std::complex<double>> out(n);
  if (n == 0) return out;

  @autoreleasepool {
    id<MTLDevice> d = ctx.device;
    id<MTLBuffer> bx = complexBuffer(d, x), btx = complexBuffer(d, tx), bty = complexBuffer(d, ty),
                  btz = complexBuffer(d, tz);
    id<MTLBuffer> by = [d newBufferWithLength:2 * n * sizeof(float)
                                     options:MTLResourceStorageModeShared];
    const LatParamsHost lp{lx, ly, lz, periodic ? 1 : 0, static_cast<float>(w)};

    id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:ctx.connMatvecPso];
    [enc setBuffer:by offset:0 atIndex:0];
    [enc setBuffer:bx offset:0 atIndex:1];
    [enc setBuffer:btx offset:0 atIndex:2];
    [enc setBuffer:bty offset:0 atIndex:3];
    [enc setBuffer:btz offset:0 atIndex:4];
    [enc setBytes:&lp length:sizeof(lp) atIndex:5];
    dispatch1D(enc, ctx.connMatvecPso, n);
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    checkCb(cb, "connectionMatvec");
    out = downloadC(by, n);
  }
  return out;
}

std::vector<std::complex<double>> connectionSmooth(
    int lx, int ly, int lz, bool periodic, double w, const std::vector<std::complex<double>>& tx,
    const std::vector<std::complex<double>>& ty, const std::vector<std::complex<double>>& tz,
    const std::vector<std::complex<double>>& b, const std::vector<std::complex<double>>& x0,
    int sweeps, double omega) {
  Context& ctx = context();
  if (!ctx.ok) throw std::runtime_error("bochner::gpu: Metal device/pipeline unavailable");
  const uint32_t n = static_cast<uint32_t>(x0.size());
  std::vector<std::complex<double>> out(x0);
  if (n == 0 || sweeps <= 0) return out;

  @autoreleasepool {
    id<MTLDevice> d = ctx.device;
    id<MTLBuffer> bx = complexBuffer(d, x0), bb = complexBuffer(d, b), btx = complexBuffer(d, tx),
                  bty = complexBuffer(d, ty), btz = complexBuffer(d, tz);
    const LatParamsHost lp{lx, ly, lz, periodic ? 1 : 0, static_cast<float>(w)};
    const float omf = static_cast<float>(omega);

    // One command buffer; two color passes per sweep, each its own encoder so
    // automatic hazard tracking orders color 0's in-place writes before color 1
    // reads them (a full red-black Gauss-Seidel sweep).
    id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
    for (int s = 0; s < sweeps; ++s)
      for (int color = 0; color <= 1; ++color) {
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:ctx.connSmoothPso];
        [enc setBuffer:bx offset:0 atIndex:0];
        [enc setBuffer:bb offset:0 atIndex:1];
        [enc setBuffer:btx offset:0 atIndex:2];
        [enc setBuffer:bty offset:0 atIndex:3];
        [enc setBuffer:btz offset:0 atIndex:4];
        [enc setBytes:&lp length:sizeof(lp) atIndex:5];
        [enc setBytes:&color length:sizeof(int) atIndex:6];
        [enc setBytes:&omf length:sizeof(float) atIndex:7];
        dispatch1D(enc, ctx.connSmoothPso, n);
        [enc endEncoding];
      }
    [cb commit];
    [cb waitUntilCompleted];
    checkCb(cb, "connectionSmooth");
    out = downloadC(bx, n);
  }
  return out;
}

std::vector<std::complex<double>> connectionProlong(
    int lx, int ly, int lz, bool periodic, const std::vector<std::complex<double>>& tx,
    const std::vector<std::complex<double>>& ty, const std::vector<std::complex<double>>& tz,
    const std::vector<std::complex<double>>& coarse) {
  Context& ctx = context();
  if (!ctx.ok) throw std::runtime_error("bochner::gpu: Metal device/pipeline unavailable");
  const uint32_t nf = static_cast<uint32_t>(lx) * ly * lz;
  std::vector<std::complex<double>> out(nf);
  if (nf == 0) return out;

  @autoreleasepool {
    id<MTLDevice> d = ctx.device;
    // fine is fully written (even nodes by the copy, odd-parity by the passes), so
    // no pre-zero is needed.
    id<MTLBuffer> bfine = [d newBufferWithLength:2 * nf * sizeof(float)
                                        options:MTLResourceStorageModeShared];
    id<MTLBuffer> bcoarse = complexBuffer(d, coarse), btx = complexBuffer(d, tx),
                  bty = complexBuffer(d, ty), btz = complexBuffer(d, tz);
    const LatParamsHost lp{lx, ly, lz, periodic ? 1 : 0, 0.0f};  // w unused by the transfer

    id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
    {  // even nodes <- coarse
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      [enc setComputePipelineState:ctx.connProlongCopyPso];
      [enc setBuffer:bfine offset:0 atIndex:0];
      [enc setBuffer:bcoarse offset:0 atIndex:1];
      [enc setBytes:&lp length:sizeof(lp) atIndex:2];
      dispatch1D(enc, ctx.connProlongCopyPso, nf);
      [enc endEncoding];
    }
    for (int pass = 1; pass <= 3; ++pass) {  // separate encoders: pass p reads pass p-1
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      [enc setComputePipelineState:ctx.connProlongPso];
      [enc setBuffer:bfine offset:0 atIndex:0];
      [enc setBuffer:btx offset:0 atIndex:1];
      [enc setBuffer:bty offset:0 atIndex:2];
      [enc setBuffer:btz offset:0 atIndex:3];
      [enc setBytes:&lp length:sizeof(lp) atIndex:4];
      [enc setBytes:&pass length:sizeof(int) atIndex:5];
      dispatch1D(enc, ctx.connProlongPso, nf);
      [enc endEncoding];
    }
    [cb commit];
    [cb waitUntilCompleted];
    checkCb(cb, "connectionProlong");
    out = downloadC(bfine, nf);
  }
  return out;
}

std::vector<std::complex<double>> connectionRestrict(
    int lx, int ly, int lz, bool periodic, const std::vector<std::complex<double>>& tx,
    const std::vector<std::complex<double>>& ty, const std::vector<std::complex<double>>& tz,
    const std::vector<std::complex<double>>& fineR) {
  Context& ctx = context();
  if (!ctx.ok) throw std::runtime_error("bochner::gpu: Metal device/pipeline unavailable");
  const uint32_t nc = static_cast<uint32_t>(lx / 2) * (ly / 2) * (lz / 2);
  std::vector<std::complex<double>> out(nc);
  if (nc == 0) return out;

  @autoreleasepool {
    id<MTLDevice> d = ctx.device;
    // r is the fine residual, scattered into IN PLACE (atomic float2), then the
    // even nodes are picked off into the coarse vector.
    id<MTLBuffer> br = complexBuffer(d, fineR), btx = complexBuffer(d, tx),
                  bty = complexBuffer(d, ty), btz = complexBuffer(d, tz);
    id<MTLBuffer> bcoarse = [d newBufferWithLength:2 * nc * sizeof(float)
                                           options:MTLResourceStorageModeShared];
    const LatParamsHost lp{lx, ly, lz, periodic ? 1 : 0, 0.0f};
    const uint32_t nf = static_cast<uint32_t>(lx) * ly * lz;

    id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
    for (int pass = 3; pass >= 1; --pass) {  // reverse of prolong: adjoint order
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      [enc setComputePipelineState:ctx.connRestrictPso];
      [enc setBuffer:br offset:0 atIndex:0];
      [enc setBuffer:btx offset:0 atIndex:1];
      [enc setBuffer:bty offset:0 atIndex:2];
      [enc setBuffer:btz offset:0 atIndex:3];
      [enc setBytes:&lp length:sizeof(lp) atIndex:4];
      [enc setBytes:&pass length:sizeof(int) atIndex:5];
      dispatch1D(enc, ctx.connRestrictPso, nf);
      [enc endEncoding];
    }
    {  // coarse <- even fine nodes
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      [enc setComputePipelineState:ctx.connRestrictPickPso];
      [enc setBuffer:bcoarse offset:0 atIndex:0];
      [enc setBuffer:br offset:0 atIndex:1];
      [enc setBytes:&lp length:sizeof(lp) atIndex:2];
      dispatch1D(enc, ctx.connRestrictPickPso, nc);
      [enc endEncoding];
    }
    [cb commit];
    [cb waitUntilCompleted];
    checkCb(cb, "connectionRestrict");
    out = downloadC(bcoarse, nc);
  }
  return out;
}

// --- GPU gauge multigrid V-cycle (handle-based, resident) --------------------
namespace {

struct GpuGaugeLevel {
  LatParamsHost lp{0, 0, 0, 0, 0.0f};
  int n = 0;  // node count lx*ly*lz
  id<MTLBuffer> tx = nil, ty = nil, tz = nil;
  // Complex scratch (2n interleaved floats): the iterate/correction x, rhs b,
  // saved residual r (needed for the alpha numerator), prolonged correction pe,
  // and a general scratch tmp (E x, restrict-scatter buffer, then E pe).
  id<MTLBuffer> x = nil, b = nil, r = nil, pe = nil, tmp = nil;
};

struct GpuGaugeHierarchy {
  bool valid = false;
  std::vector<GpuGaugeLevel> levels;
  id<MTLBuffer> partialsNum = nil, partialsDen = nil;  // alpha (and residual) reductions
  id<MTLBuffer> alphaBuf = nil;                        // on-device V-cycle coarse-correction scale
  // Device-resident covMG-LOBPCG state (fine-level complex vectors + on-device scalars),
  // so the eigensolver's per-iteration vector work never leaves the GPU.
  id<MTLBuffer> rx = nil, rex = nil, rr = nil, rw = nil, rxp = nil, rq2 = nil, raq1 = nil,
                raq2 = nil, rxn = nil;
  id<MTLBuffer> cpart = nil;                            // float2 complex-dot partials
  id<MTLBuffer> scRho = nil, scProj = nil, scNvw = nil, scNvxp = nil, scNn = nil;  // scalars (float2)
  id<MTLBuffer> scH = nil, scCC = nil;                 // 3x3 Ritz matrix + coefficients (float2 x 9 / 3)
};

std::vector<std::unique_ptr<GpuGaugeHierarchy>>& gaugeRegistry() {
  static std::vector<std::unique_ptr<GpuGaugeHierarchy>> reg;
  return reg;
}

// --- small double-precision linear algebra for the covMG-LOBPCG orchestration --------
// (mirrors the file-private helpers in solvers/GaugeEigen.cpp; duplicated here so
// the gpu module stays standalone -- it must not depend on solvers.)
using cd = std::complex<double>;

cd cdot(const std::vector<cd>& a, const std::vector<cd>& b) {
  cd s(0.0, 0.0);
  for (std::size_t i = 0; i < a.size(); ++i) s += std::conj(a[i]) * b[i];
  return s;
}
double norm2(const std::vector<cd>& v) { return std::sqrt(cdot(v, v).real()); }
void cscale(std::vector<cd>& v, cd s) {
  for (cd& z : v) z *= s;
}

// Smallest eigenpair of a small (m <= 3) dense complex-Hermitian matrix by cyclic
// Hermitian Jacobi rotations (== GaugeEigen.cpp::smallestHermitianEig).
void smallestHermitianEig(int m, cd H[3][3], double& lam, cd c[3]) {
  cd D[3][3], V[3][3];
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < m; ++j) {
      D[i][j] = H[i][j];
      V[i][j] = (i == j) ? cd(1.0, 0.0) : cd(0.0, 0.0);
    }
  for (int sweep = 0; sweep < 40; ++sweep) {
    double off = 0.0;
    for (int p = 0; p < m; ++p)
      for (int q = p + 1; q < m; ++q) off += std::norm(D[p][q]);
    if (off < 1e-30) break;
    for (int p = 0; p < m; ++p)
      for (int q = p + 1; q < m; ++q) {
        const cd dpq = D[p][q];
        if (std::abs(dpq) < 1e-18) continue;
        const double a = D[p][p].real(), b = D[q][q].real(), mag = std::abs(dpq);
        const cd g = std::exp(cd(0.0, -std::arg(dpq)));
        const double theta = 0.5 * std::atan2(2.0 * mag, a - b);
        const double cc = std::cos(theta), ss = std::sin(theta);
        const cd U00(cc, 0.0), U01(-ss, 0.0), U10 = g * ss, U11 = g * cc;
        const cd c00 = std::conj(U00), c01 = std::conj(U01), c10 = std::conj(U10),
                 c11 = std::conj(U11);
        for (int i = 0; i < m; ++i) {
          const cd dip = D[i][p], diq = D[i][q];
          D[i][p] = dip * U00 + diq * U10;
          D[i][q] = dip * U01 + diq * U11;
        }
        for (int j = 0; j < m; ++j) {
          const cd dpj = D[p][j], dqj = D[q][j];
          D[p][j] = c00 * dpj + c10 * dqj;
          D[q][j] = c01 * dpj + c11 * dqj;
        }
        for (int i = 0; i < m; ++i) {
          const cd vip = V[i][p], viq = V[i][q];
          V[i][p] = vip * U00 + viq * U10;
          V[i][q] = vip * U01 + viq * U11;
        }
      }
  }
  int jmin = 0;
  for (int j = 1; j < m; ++j)
    if (D[j][j].real() < D[jmin][jmin].real()) jmin = j;
  lam = D[jmin][jmin].real();
  for (int i = 0; i < m; ++i) c[i] = V[i][jmin];
}

void gBindLat(id<MTLComputeCommandEncoder> enc, const GpuGaugeLevel& L, int atTx) {
  [enc setBuffer:L.tx offset:0 atIndex:atTx];
  [enc setBuffer:L.ty offset:0 atIndex:atTx + 1];
  [enc setBuffer:L.tz offset:0 atIndex:atTx + 2];
}

// y = E x on level L (connection_matvec).
void gMatvec(id<MTLCommandBuffer> cb, Context& ctx, const GpuGaugeLevel& L, id<MTLBuffer> x,
            id<MTLBuffer> y) {
  id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
  [enc setComputePipelineState:ctx.connMatvecPso];
  [enc setBuffer:y offset:0 atIndex:0];
  [enc setBuffer:x offset:0 atIndex:1];
  gBindLat(enc, L, 2);
  [enc setBytes:&L.lp length:sizeof(L.lp) atIndex:5];
  dispatch1D(enc, ctx.connMatvecPso, L.n);
  [enc endEncoding];
}

// `sweeps` red-black GS sweeps of E x = b on level L (color 0 then color 1 per
// sweep, separate encoders for hazard ordering).
void gSmooth(id<MTLCommandBuffer> cb, Context& ctx, const GpuGaugeLevel& L, id<MTLBuffer> x,
            id<MTLBuffer> b, int sweeps, float omega) {
  for (int s = 0; s < sweeps; ++s)
    for (int color = 0; color <= 1; ++color) {
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      [enc setComputePipelineState:ctx.connSmoothPso];
      [enc setBuffer:x offset:0 atIndex:0];
      [enc setBuffer:b offset:0 atIndex:1];
      gBindLat(enc, L, 2);
      [enc setBytes:&L.lp length:sizeof(L.lp) atIndex:5];
      [enc setBytes:&color length:sizeof(int) atIndex:6];
      [enc setBytes:&omega length:sizeof(float) atIndex:7];
      dispatch1D(enc, ctx.connSmoothPso, L.n);
      [enc endEncoding];
    }
}

// Restrict fine.r (preserved) to coarse.b, using fine.tmp as the scatter scratch.
void gRestrict(id<MTLCommandBuffer> cb, Context& ctx, const GpuGaugeLevel& fine,
               const GpuGaugeLevel& coarse) {
  blitCopy(cb, fine.r, fine.tmp, static_cast<size_t>(2 * fine.n) * sizeof(float));
  for (int pass = 3; pass >= 1; --pass) {
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:ctx.connRestrictPso];
    [enc setBuffer:fine.tmp offset:0 atIndex:0];
    gBindLat(enc, fine, 1);
    [enc setBytes:&fine.lp length:sizeof(fine.lp) atIndex:4];
    [enc setBytes:&pass length:sizeof(int) atIndex:5];
    dispatch1D(enc, ctx.connRestrictPso, fine.n);
    [enc endEncoding];
  }
  id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
  [enc setComputePipelineState:ctx.connRestrictPickPso];
  [enc setBuffer:coarse.b offset:0 atIndex:0];
  [enc setBuffer:fine.tmp offset:0 atIndex:1];
  [enc setBytes:&fine.lp length:sizeof(fine.lp) atIndex:2];
  dispatch1D(enc, ctx.connRestrictPickPso, coarse.n);
  [enc endEncoding];
}

// Prolong coarseX (levels[l+1].x) to fine.pe.
void gProlong(id<MTLCommandBuffer> cb, Context& ctx, const GpuGaugeLevel& fine,
              id<MTLBuffer> coarseX) {
  {
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:ctx.connProlongCopyPso];
    [enc setBuffer:fine.pe offset:0 atIndex:0];
    [enc setBuffer:coarseX offset:0 atIndex:1];
    [enc setBytes:&fine.lp length:sizeof(fine.lp) atIndex:2];
    dispatch1D(enc, ctx.connProlongCopyPso, fine.n);
    [enc endEncoding];
  }
  for (int pass = 1; pass <= 3; ++pass) {
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:ctx.connProlongPso];
    [enc setBuffer:fine.pe offset:0 atIndex:0];
    gBindLat(enc, fine, 1);
    [enc setBytes:&fine.lp length:sizeof(fine.lp) atIndex:4];
    [enc setBytes:&pass length:sizeof(int) atIndex:5];
    dispatch1D(enc, ctx.connProlongPso, fine.n);
    [enc endEncoding];
  }
}

// r = a - b over `n` floats (vec_sub: complex subtraction is componentwise).
void gVecSub(id<MTLCommandBuffer> cb, Context& ctx, id<MTLBuffer> r, id<MTLBuffer> a,
             id<MTLBuffer> b, uint32_t n) {
  encode(cb, ctx.subPso, @[ r, a, b ], n);
}

// y += a * x over `n` floats.
void gAxpy(id<MTLCommandBuffer> cb, Context& ctx, id<MTLBuffer> y, id<MTLBuffer> x, float a,
           uint32_t n) {
  id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
  [enc setComputePipelineState:ctx.axpyPso];
  [enc setBuffer:y offset:0 atIndex:0];
  [enc setBuffer:x offset:0 atIndex:1];
  [enc setBytes:&a length:sizeof(float) atIndex:2];
  [enc setBytes:&n length:sizeof(uint32_t) atIndex:3];
  dispatch1D(enc, ctx.axpyPso, n);
  [enc endEncoding];
}

// Encode the partial reduction sum_i a[i]*b[i] over `n` floats into `partials`;
// treating a complex buffer's 2n floats as reals gives Re<a,b> = a.re b.re +
// a.im b.im summed over the nodes.
void gDot(id<MTLCommandBuffer> cb, Context& ctx, id<MTLBuffer> partials, id<MTLBuffer> a,
          id<MTLBuffer> b, uint32_t n) {
  const uint32_t groups = (n + kTG - 1) / kTG;
  id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
  [enc setComputePipelineState:ctx.dotPso];
  [enc setBuffer:partials offset:0 atIndex:0];
  [enc setBuffer:a offset:0 atIndex:1];
  [enc setBuffer:b offset:0 atIndex:2];
  [enc setBytes:&n length:sizeof(uint32_t) atIndex:3];
  [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(kTG, 1, 1)];
  [enc endEncoding];
}

double gReadDot(id<MTLBuffer> partials, uint32_t n) {
  const uint32_t groups = (n + kTG - 1) / kTG;
  const float* p = static_cast<const float*>(partials.contents);
  double s = 0.0;
  for (uint32_t i = 0; i < groups; ++i) s += p[i];
  return s;
}

// Sum the num/den partials (each `nFloats` -> `groups` partials) and write
// alpha = num/den into `alphaOut[0]` on the device (one threadgroup).
void gAlphaReduce(id<MTLCommandBuffer> cb, Context& ctx, id<MTLBuffer> alphaOut,
                  id<MTLBuffer> pnum, id<MTLBuffer> pden, uint32_t nFloats) {
  const uint32_t P = (nFloats + kTG - 1) / kTG;
  id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
  [enc setComputePipelineState:ctx.alphaReducePso];
  [enc setBuffer:alphaOut offset:0 atIndex:0];
  [enc setBuffer:pnum offset:0 atIndex:1];
  [enc setBuffer:pden offset:0 atIndex:2];
  [enc setBytes:&P length:sizeof(uint32_t) atIndex:3];
  [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(kTG, 1, 1)];
  [enc endEncoding];
}

// y += a[0] * x over `n` floats (scalar a from a device buffer).
void gAxpyBuf(id<MTLCommandBuffer> cb, Context& ctx, id<MTLBuffer> y, id<MTLBuffer> x,
              id<MTLBuffer> a, uint32_t n) {
  encode(cb, ctx.axpyBufPso, @[ y, x, a ], n);
}

// --- device-resident covMG-LOBPCG encode helpers (n = complex node count) ------------
// <a,b> (Hermitian) -> out[slot] (a float2 = Re,Im), fully reduced on the device.
void gCdot(id<MTLCommandBuffer> cb, Context& ctx, GpuGaugeHierarchy& H, id<MTLBuffer> out, int slot,
           id<MTLBuffer> a, id<MTLBuffer> b, int n) {
  const uint32_t un = static_cast<uint32_t>(n);
  const uint32_t groups = (un + kTG - 1) / kTG;
  {
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:ctx.cdotPartialPso];
    [enc setBuffer:H.cpart offset:0 atIndex:0];
    [enc setBuffer:a offset:0 atIndex:1];
    [enc setBuffer:b offset:0 atIndex:2];
    [enc setBytes:&un length:sizeof(un) atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(kTG, 1, 1)];
    [enc endEncoding];
  }
  id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
  [enc setComputePipelineState:ctx.cdotFinishPso];
  [enc setBuffer:out offset:static_cast<NSUInteger>(slot) * 2 * sizeof(float) atIndex:0];
  [enc setBuffer:H.cpart offset:0 atIndex:1];
  [enc setBytes:&groups length:sizeof(groups) atIndex:2];
  [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(kTG, 1, 1)];
  [enc endEncoding];
}

// y += sign * scal[slot] * x  (complex scalar from a device buffer).
void gCaxpy(id<MTLCommandBuffer> cb, Context& ctx, id<MTLBuffer> y, id<MTLBuffer> x,
            id<MTLBuffer> scal, int slot, float sign, int n) {
  const uint32_t un = static_cast<uint32_t>(n);
  id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
  [enc setComputePipelineState:ctx.caxpyCPso];
  [enc setBuffer:y offset:0 atIndex:0];
  [enc setBuffer:x offset:0 atIndex:1];
  [enc setBuffer:scal offset:static_cast<NSUInteger>(slot) * 2 * sizeof(float) atIndex:2];
  [enc setBytes:&sign length:sizeof(float) atIndex:3];
  [enc setBytes:&un length:sizeof(un) atIndex:4];
  dispatch1D(enc, ctx.caxpyCPso, un);
  [enc endEncoding];
}

// r = ex - scal[0].x * x  (real Rayleigh quotient in scal).
void gResid(id<MTLCommandBuffer> cb, Context& ctx, id<MTLBuffer> r, id<MTLBuffer> ex,
            id<MTLBuffer> x, id<MTLBuffer> scal, int n) {
  const uint32_t un = static_cast<uint32_t>(n);
  id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
  [enc setComputePipelineState:ctx.residCPso];
  [enc setBuffer:r offset:0 atIndex:0];
  [enc setBuffer:ex offset:0 atIndex:1];
  [enc setBuffer:x offset:0 atIndex:2];
  [enc setBuffer:scal offset:0 atIndex:3];
  [enc setBytes:&un length:sizeof(un) atIndex:4];
  dispatch1D(enc, ctx.residCPso, un);
  [enc endEncoding];
}

// x *= 1/sqrt(scal[0].x)  (normalize by the norm whose square is scal[0].x).
void gCnormalize(id<MTLCommandBuffer> cb, Context& ctx, id<MTLBuffer> x, id<MTLBuffer> scal, int n) {
  const uint32_t un = static_cast<uint32_t>(n);
  id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
  [enc setComputePipelineState:ctx.cnormalizePso];
  [enc setBuffer:x offset:0 atIndex:0];
  [enc setBuffer:scal offset:0 atIndex:1];
  [enc setBytes:&un length:sizeof(un) atIndex:2];
  dispatch1D(enc, ctx.cnormalizePso, un);
  [enc endEncoding];
}
// Encode a full V-cycle into `cb` (no commit/wait) -- used by runGaugeVcycle and
// inlined into the resident covMG-LOBPCG's per-iteration command buffer. Solves with
// levels[0].b as the rhs and levels[0].x as the (in-place) solution. The optimal
// coarse-correction alpha = Re<pe,r>/Re<pe,E pe> is reduced on the DEVICE
// (alpha_reduce -> alphaBuf) and consumed by axpy_buf, so no scalar is read back
// mid-cycle -- the whole descent + coarsest + ascent needs no host sync.
void encodeGaugeVcycle(id<MTLCommandBuffer> cb, Context& ctx, GpuGaugeHierarchy& H, int nu1, int nu2,
                       int coarseSweeps, float omega) {
  const int L = static_cast<int>(H.levels.size());
  auto NB = [](int n) { return static_cast<size_t>(2 * n) * sizeof(float); };
  for (int l = 0; l < L - 1; ++l) {
    GpuGaugeLevel& cur = H.levels[l];
    GpuGaugeLevel& nxt = H.levels[l + 1];
    gSmooth(cb, ctx, cur, cur.x, cur.b, nu1, omega);
    gMatvec(cb, ctx, cur, cur.x, cur.tmp);              // tmp = E x
    gVecSub(cb, ctx, cur.r, cur.b, cur.tmp, 2 * cur.n);  // r = b - E x
    gRestrict(cb, ctx, cur, nxt);                        // nxt.b = R r (uses cur.tmp)
    fill0(cb, nxt.x, NB(nxt.n));                          // x_{l+1} = 0
  }
  gSmooth(cb, ctx, H.levels[L - 1], H.levels[L - 1].x, H.levels[L - 1].b, coarseSweeps, omega);
  for (int l = L - 2; l >= 0; --l) {
    GpuGaugeLevel& cur = H.levels[l];
    GpuGaugeLevel& nxt = H.levels[l + 1];
    gProlong(cb, ctx, cur, nxt.x);                              // pe = P e_{l+1}
    gMatvec(cb, ctx, cur, cur.pe, cur.tmp);                     // tmp = E pe
    gDot(cb, ctx, H.partialsNum, cur.pe, cur.r, 2 * cur.n);     // num = Re<pe,r>
    gDot(cb, ctx, H.partialsDen, cur.pe, cur.tmp, 2 * cur.n);   // den = Re<pe,E pe>
    gAlphaReduce(cb, ctx, H.alphaBuf, H.partialsNum, H.partialsDen, 2 * cur.n);  // alpha on device
    gAxpyBuf(cb, ctx, cur.x, cur.pe, H.alphaBuf, 2 * cur.n);    // x += alpha pe
    gSmooth(cb, ctx, cur, cur.x, cur.b, nu2, omega);
  }
}

void runGaugeVcycle(Context& ctx, GpuGaugeHierarchy& H, int nu1, int nu2, int coarseSweeps,
                    float omega) {
  id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
  encodeGaugeVcycle(cb, ctx, H, nu1, nu2, coarseSweeps, omega);
  [cb commit];
  [cb waitUntilCompleted];
  checkCb(cb, "gaugeVcycle");
}

}  // namespace

int uploadGauge(const std::vector<GaugeLevelData>& levels) {
  Context& ctx = context();
  if (!ctx.ok || levels.empty()) return -1;
  auto gh = std::make_unique<GpuGaugeHierarchy>();
  @autoreleasepool {
    id<MTLDevice> d = ctx.device;
    for (const GaugeLevelData& lv : levels) {
      GpuGaugeLevel gl;
      gl.lp = LatParamsHost{lv.lx, lv.ly, lv.lz, lv.periodic ? 1 : 0, static_cast<float>(lv.w)};
      gl.n = lv.lx * lv.ly * lv.lz;
      gl.tx = complexBuffer(d, lv.tx);
      gl.ty = complexBuffer(d, lv.ty);
      gl.tz = complexBuffer(d, lv.tz);
      const size_t nb = static_cast<size_t>(2 * gl.n) * sizeof(float);
      gl.x = [d newBufferWithLength:nb options:MTLResourceStorageModeShared];
      gl.b = [d newBufferWithLength:nb options:MTLResourceStorageModeShared];
      gl.r = [d newBufferWithLength:nb options:MTLResourceStorageModeShared];
      gl.pe = [d newBufferWithLength:nb options:MTLResourceStorageModeShared];
      gl.tmp = [d newBufferWithLength:nb options:MTLResourceStorageModeShared];
      gh->levels.push_back(gl);
    }
    const int nFine = gh->levels.front().n;
    const size_t pb = static_cast<size_t>((2 * nFine + kTG - 1) / kTG) * sizeof(float);
    gh->partialsNum = [d newBufferWithLength:pb options:MTLResourceStorageModeShared];
    gh->partialsDen = [d newBufferWithLength:pb options:MTLResourceStorageModeShared];
    gh->alphaBuf = [d newBufferWithLength:sizeof(float) options:MTLResourceStorageModeShared];
    // Device-resident covMG-LOBPCG buffers.
    const size_t vbf = static_cast<size_t>(2 * nFine) * sizeof(float);
    auto vecBuf = [&] { return [d newBufferWithLength:vbf options:MTLResourceStorageModeShared]; };
    gh->rx = vecBuf();  gh->rex = vecBuf();  gh->rr = vecBuf();  gh->rw = vecBuf();
    gh->rxp = vecBuf(); gh->rq2 = vecBuf();  gh->raq1 = vecBuf(); gh->raq2 = vecBuf();
    gh->rxn = vecBuf();
    const size_t cpb = static_cast<size_t>((nFine + kTG - 1) / kTG) * 2 * sizeof(float);
    gh->cpart = [d newBufferWithLength:cpb options:MTLResourceStorageModeShared];
    auto sBuf = [&] { return [d newBufferWithLength:2 * sizeof(float) options:MTLResourceStorageModeShared]; };
    gh->scRho = sBuf(); gh->scProj = sBuf(); gh->scNvw = sBuf(); gh->scNvxp = sBuf(); gh->scNn = sBuf();
    gh->scH = [d newBufferWithLength:9 * 2 * sizeof(float) options:MTLResourceStorageModeShared];
    gh->scCC = [d newBufferWithLength:3 * 2 * sizeof(float) options:MTLResourceStorageModeShared];
    gh->valid = true;
  }
  return registerHandle(gaugeRegistry(), std::move(gh));
}

void freeGauge(int handle) {
  auto& reg = gaugeRegistry();
  if (handle >= 0 && handle < static_cast<int>(reg.size())) reg[handle].reset();
}

int vcycleSolveGauge(int handle, const std::vector<std::complex<double>>& b,
                     std::vector<std::complex<double>>& x, int nu1, int nu2, int coarseSweeps,
                     double omega, int maxCycles, double tol, double* relResidualOut) {
  Context& ctx = context();
  auto& reg = gaugeRegistry();
  if (!ctx.ok || handle < 0 || handle >= static_cast<int>(reg.size()) || !reg[handle] ||
      !reg[handle]->valid)
    throw std::runtime_error("bochner::gpu: invalid gauge handle / Metal unavailable");
  GpuGaugeHierarchy& H = *reg[handle];
  GpuGaugeLevel& fine = H.levels[0];
  const int n = fine.n;
  const uint32_t n2 = static_cast<uint32_t>(2 * n);
  if (static_cast<int>(b.size()) != n || static_cast<int>(x.size()) != n)
    throw std::runtime_error("bochner::gpu: vcycleSolveGauge b/x size mismatch with hierarchy");

  int cycles = 0;
  double rel = 0.0;
  @autoreleasepool {
    // Upload b and the warm start x (double -> float2).
    {
      const std::vector<float> bf = packC(b), xf = packC(x);
      std::memcpy(fine.b.contents, bf.data(), bf.size() * sizeof(float));
      std::memcpy(fine.x.contents, xf.data(), xf.size() * sizeof(float));
    }
    // ||b|| for the relative residual.
    double bnorm;
    {
      id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
      gDot(cb, ctx, H.partialsNum, fine.b, fine.b, n2);
      [cb commit];
      [cb waitUntilCompleted];
      checkCb(cb, "vcycleSolveGauge/bnorm");
      bnorm = std::sqrt(gReadDot(H.partialsNum, n2));
    }
    const float omf = static_cast<float>(omega);
    // ||b - E x||^2 of the current iterate: tmp = E x; r = b - tmp; rr = <r,r>.
    auto residual2 = [&]() {
      id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
      gMatvec(cb, ctx, fine, fine.x, fine.tmp);
      gVecSub(cb, ctx, fine.r, fine.b, fine.tmp, n2);
      gDot(cb, ctx, H.partialsNum, fine.r, fine.r, n2);
      [cb commit];
      [cb waitUntilCompleted];
      checkCb(cb, "vcycleSolveGauge/residual");
      return gReadDot(H.partialsNum, n2);
    };
    if (bnorm == 0.0) {
      // b = 0: the SPD solution is x = 0 (the CPU vcycleSolve contract);
      // return it rather than the warm start with a claimed 0 residual.
      std::fill(x.begin(), x.end(), std::complex<double>(0.0, 0.0));
      if (relResidualOut) *relResidualOut = 0.0;
      return 0;
    } else if (maxCycles <= 0) {
      // No cycles requested: still report the warm start's TRUE relative
      // residual (one matvec on this cold path) rather than claiming 0.
      rel = std::sqrt(residual2()) / bnorm;
    } else {
      for (cycles = 1; cycles <= maxCycles; ++cycles) {
        runGaugeVcycle(ctx, H, nu1, nu2, coarseSweeps, omf);
        rel = std::sqrt(residual2()) / bnorm;
        if (rel < tol) break;
      }
      if (cycles > maxCycles) cycles = maxCycles;
    }
    x = downloadC(fine.x, static_cast<uint32_t>(n));
  }
  if (relResidualOut) *relResidualOut = rel;
  return cycles;
}

GaugeEigenGpu lobpcgSolveGauge(int handle, const std::vector<std::complex<double>>& guess,
                             int maxIters, double tol, int precCycles, int nu1, int nu2,
                             int coarseSweeps, double omega) {
  Context& ctx = context();
  auto& reg = gaugeRegistry();
  if (!ctx.ok || handle < 0 || handle >= static_cast<int>(reg.size()) || !reg[handle] ||
      !reg[handle]->valid)
    throw std::runtime_error("bochner::gpu: invalid gauge handle / Metal unavailable");
  GpuGaugeHierarchy& H = *reg[handle];
  GpuGaugeLevel& fine = H.levels[0];
  const int n = fine.n;
  const float omf = static_cast<float>(omega);
  const size_t vb = static_cast<size_t>(2 * n) * sizeof(float);

  GaugeEigenGpu res;
  @autoreleasepool {
    // rx = normalized guess (double-normalized on the CPU, then uploaded once);
    // rxp = 0 so the first iteration drops the (absent) x_prev direction naturally.
    // Everything after this stays on the device -- only per-iteration scalars (the
    // Rayleigh quotient, the two Gram-Schmidt norms, and the 3x3 Ritz matrix) are
    // read back, one sync per iteration.
    {
      std::vector<cd> x0(n, cd(1.0, 0.0));
      if (static_cast<int>(guess.size()) == n) x0 = guess;
      double nx = norm2(x0);
      if (!std::isfinite(nx) || !(nx > 0.0) || !std::isfinite(1.0 / nx)) {
        x0.assign(n, cd(1.0, 0.0));
        nx = norm2(x0);
      }
      cscale(x0, cd(1.0 / nx, 0.0));
      const std::vector<float> xf = packC(x0);
      std::memcpy(H.rx.contents, xf.data(), xf.size() * sizeof(float));
      std::memset(H.rxp.contents, 0, vb);
    }
    auto readReal = [](id<MTLBuffer> b) { return static_cast<double>(static_cast<const float*>(b.contents)[0]); };
    auto readC = [](id<MTLBuffer> b, int slot) {
      const float* p = static_cast<const float*>(b.contents);
      return cd(p[2 * slot], p[2 * slot + 1]);
    };

    double prevRho = 1e300;
    int stall = 0, iters = 0;
    // cbB is committed without its own wait (see below); the serial queue means
    // it has completed by the time the NEXT wait returns, so its error is
    // checked there via cbPrev.
    id<MTLCommandBuffer> cbPrev = nil;
    for (int it = 1; it <= maxIters; ++it) {
      iters = it;
      // Command buffer A: Ex, rho, r, the V-cycle preconditioner w, the
      // Gram-Schmidt of {x, w, x_prev} into orthonormal q0=rx, q1=rw, q2=rq2 with
      // their operator images, and the 3x3 Ritz matrix H[i][j] = <q_i, E q_j>.
      id<MTLCommandBuffer> cbA = [ctx.queue commandBuffer];
      gMatvec(cbA, ctx, fine, H.rx, H.rex);                 // Ex
      gCdot(cbA, ctx, H, H.scRho, 0, H.rx, H.rex, n);        // rho = Re<x,Ex>
      gResid(cbA, ctx, H.rr, H.rex, H.rx, H.scRho, n);       // r = Ex - rho x
      blitCopy(cbA, H.rr, fine.b, vb);
      fill0(cbA, fine.x, vb);
      for (int c = 0; c < std::max(1, precCycles); ++c)
        encodeGaugeVcycle(cbA, ctx, H, nu1, nu2, coarseSweeps, omf);  // fine.x = w
      blitCopy(cbA, fine.x, H.rw, vb);
      // q1 = normalize(w - <x,w> x)
      gCdot(cbA, ctx, H, H.scProj, 0, H.rx, H.rw, n);
      gCaxpy(cbA, ctx, H.rw, H.rx, H.scProj, 0, -1.0f, n);
      gCdot(cbA, ctx, H, H.scNvw, 0, H.rw, H.rw, n);
      gCnormalize(cbA, ctx, H.rw, H.scNvw, n);
      gMatvec(cbA, ctx, fine, H.rw, H.raq1);
      // q2 = normalize(x_prev - <x,x_prev> x - <q1,x_prev> q1)
      blitCopy(cbA, H.rxp, H.rq2, vb);
      gCdot(cbA, ctx, H, H.scProj, 0, H.rx, H.rq2, n);
      gCaxpy(cbA, ctx, H.rq2, H.rx, H.scProj, 0, -1.0f, n);
      gCdot(cbA, ctx, H, H.scProj, 0, H.rw, H.rq2, n);
      gCaxpy(cbA, ctx, H.rq2, H.rw, H.scProj, 0, -1.0f, n);
      gCdot(cbA, ctx, H, H.scNvxp, 0, H.rq2, H.rq2, n);
      gCnormalize(cbA, ctx, H.rq2, H.scNvxp, n);
      gMatvec(cbA, ctx, fine, H.rq2, H.raq2);
      id<MTLBuffer> qs[3] = {H.rx, H.rw, H.rq2};
      id<MTLBuffer> aqs[3] = {H.rex, H.raq1, H.raq2};
      for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) gCdot(cbA, ctx, H, H.scH, i * 3 + j, qs[i], aqs[j], n);
      [cbA commit];
      [cbA waitUntilCompleted];
      checkCb(cbA, "lobpcg/iter");
      if (cbPrev) checkCb(cbPrev, "lobpcg/update");

      const double rho = readReal(H.scRho);
      res.eigenvalue = rho;
      // rho-stagnation stop (the float residual floors out above tol; rho is the
      // monotone, quadratically-accurate signal -- see the CPU-orchestrated notes).
      // Threshold 1e-5 (relative): the eigenVALUE, hence the filament zero-set
      // TOPOLOGY, is converged to the viewer's float accuracy well before the
      // residual floor; a near-degenerate spectrum (e.g. the leapfrog's nested
      // sheets at 64^2x128) makes rho descend slowly-but-steadily, so a tighter
      // 1e-6 threshold never triggers and the solve runs to maxIters -- 4-5x the
      // work for an identical eigenvalue and filament count.
      if (std::abs(prevRho - rho) < 1e-5 * std::max(std::abs(rho), 1e-12)) {
        if (++stall >= 2) break;
      } else {
        stall = 0;
      }
      prevRho = rho;

      // Keep x (always), q1 if ||w'|| > 1e-7, q2 if ||x_prev'|| > 1e-7 (the first
      // iteration's zero x_prev is dropped here). Solve the m x m Ritz problem.
      int kept[3], m = 0;
      kept[m++] = 0;
      if (readReal(H.scNvw) > 1e-14) kept[m++] = 1;
      if (readReal(H.scNvxp) > 1e-14) kept[m++] = 2;
      if (m == 1) break;  // no usable search direction => converged
      cd Hs[3][3];
      for (int i = 0; i < m; ++i)
        for (int j = 0; j < m; ++j) Hs[i][j] = readC(H.scH, kept[i] * 3 + kept[j]);
      double lam = 0.0;
      cd ccs[3];
      smallestHermitianEig(m, Hs, lam, ccs);
      cd cc3[3] = {cd(0, 0), cd(0, 0), cd(0, 0)};
      for (int i = 0; i < m; ++i) cc3[kept[i]] = ccs[i];
      float* ccp = static_cast<float*>(H.scCC.contents);
      for (int k = 0; k < 3; ++k) {
        ccp[2 * k] = static_cast<float>(cc3[k].real());
        ccp[2 * k + 1] = static_cast<float>(cc3[k].imag());
      }

      // Command buffer B: xn = sum cc_k q_k ; normalize ; x_prev = x ; x = xn.
      id<MTLCommandBuffer> cbB = [ctx.queue commandBuffer];
      fill0(cbB, H.rxn, vb);
      gCaxpy(cbB, ctx, H.rxn, H.rx, H.scCC, 0, 1.0f, n);
      gCaxpy(cbB, ctx, H.rxn, H.rw, H.scCC, 1, 1.0f, n);
      gCaxpy(cbB, ctx, H.rxn, H.rq2, H.scCC, 2, 1.0f, n);
      gCdot(cbB, ctx, H, H.scNn, 0, H.rxn, H.rxn, n);
      gCnormalize(cbB, ctx, H.rxn, H.scNn, n);
      blitCopy(cbB, H.rx, H.rxp, vb);
      blitCopy(cbB, H.rxn, H.rx, vb);
      [cbB commit];  // next iteration's cbA (or the final report) waits on the serial queue
      cbPrev = cbB;
    }
    res.iterations = iters;

    // Final report: recompute Ex, then the eigenvalue and residual in DOUBLE from
    // the (single-precision) vectors -- the Rayleigh quotient's quadratic accuracy
    // keeps rho ~1e-6 even though x is stored in float.
    {
      id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
      gMatvec(cb, ctx, fine, H.rx, H.rex);
      [cb commit];
      [cb waitUntilCompleted];
      checkCb(cb, "lobpcg/report");
      if (cbPrev) checkCb(cbPrev, "lobpcg/update");
    }
    std::vector<cd> x = downloadC(H.rx, n), Ex = downloadC(H.rex, n);
    const double nx2 = cdot(x, x).real();
    const double inv = 1.0 / std::sqrt(std::max(nx2, 1e-300));
    cscale(x, cd(inv, 0.0));
    cscale(Ex, cd(inv, 0.0));
    const double rho = cdot(x, Ex).real();
    std::vector<cd> r(n);
    for (int i = 0; i < n; ++i) r[i] = Ex[i] - rho * x[i];
    res.eigenvalue = rho;
    res.residual = norm2(r) / std::max(std::abs(rho), 1e-300);
    res.vector = std::move(x);
    res.converged = res.residual < tol;
  }
  return res;
}

// --- GPU matrix-free geometric multigrid for the closed-box Neumann Poisson ---
namespace {

struct GpuPoissonGeomLevel {
  LatParamsHost lp{0, 0, 0, 0, 0.0f};
  int n = 0;
  id<MTLBuffer> x = nil, b = nil, r = nil, pe = nil, tmp = nil;  // real, n floats each
};

struct GpuPoissonGeom {
  bool valid = false;
  std::vector<GpuPoissonGeomLevel> levels;
  id<MTLBuffer> partialsNum = nil, partialsDen = nil, alphaBuf = nil;
};

std::vector<std::unique_ptr<GpuPoissonGeom>>& poissonGeomRegistry() {
  static std::vector<std::unique_ptr<GpuPoissonGeom>> reg;
  return reg;
}

void pMatvec(id<MTLCommandBuffer> cb, Context& ctx, const GpuPoissonGeomLevel& L, id<MTLBuffer> x,
             id<MTLBuffer> y) {
  id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
  [enc setComputePipelineState:ctx.poissonMatvecPso];
  [enc setBuffer:y offset:0 atIndex:0];
  [enc setBuffer:x offset:0 atIndex:1];
  [enc setBytes:&L.lp length:sizeof(L.lp) atIndex:2];
  dispatch1D(enc, ctx.poissonMatvecPso, L.n);
  [enc endEncoding];
}

void pSmooth(id<MTLCommandBuffer> cb, Context& ctx, const GpuPoissonGeomLevel& L, id<MTLBuffer> x,
             id<MTLBuffer> b, int sweeps, float omega) {
  for (int s = 0; s < sweeps; ++s)
    for (int color = 0; color <= 1; ++color) {
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      [enc setComputePipelineState:ctx.poissonSmoothPso];
      [enc setBuffer:x offset:0 atIndex:0];
      [enc setBuffer:b offset:0 atIndex:1];
      [enc setBytes:&L.lp length:sizeof(L.lp) atIndex:2];
      [enc setBytes:&color length:sizeof(int) atIndex:3];
      [enc setBytes:&omega length:sizeof(float) atIndex:4];
      dispatch1D(enc, ctx.poissonSmoothPso, L.n);
      [enc endEncoding];
    }
}

// Restrict fine.r (preserved) into coarse.b, using fine.tmp as the scatter scratch.
void pRestrict(id<MTLCommandBuffer> cb, Context& ctx, const GpuPoissonGeomLevel& fine,
               const GpuPoissonGeomLevel& coarse) {
  blitCopy(cb, fine.r, fine.tmp, static_cast<size_t>(fine.n) * sizeof(float));
  for (int pass = 3; pass >= 1; --pass) {
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:ctx.poissonRestrictPso];
    [enc setBuffer:fine.tmp offset:0 atIndex:0];
    [enc setBytes:&fine.lp length:sizeof(fine.lp) atIndex:1];
    [enc setBytes:&pass length:sizeof(int) atIndex:2];
    dispatch1D(enc, ctx.poissonRestrictPso, fine.n);
    [enc endEncoding];
  }
  id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
  [enc setComputePipelineState:ctx.poissonRestrictPickPso];
  [enc setBuffer:coarse.b offset:0 atIndex:0];
  [enc setBuffer:fine.tmp offset:0 atIndex:1];
  [enc setBytes:&fine.lp length:sizeof(fine.lp) atIndex:2];
  dispatch1D(enc, ctx.poissonRestrictPickPso, coarse.n);
  [enc endEncoding];
}

// Prolong coarseX (levels[l+1].x) into fine.pe.
void pProlong(id<MTLCommandBuffer> cb, Context& ctx, const GpuPoissonGeomLevel& fine,
              id<MTLBuffer> coarseX) {
  {
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:ctx.poissonProlongCopyPso];
    [enc setBuffer:fine.pe offset:0 atIndex:0];
    [enc setBuffer:coarseX offset:0 atIndex:1];
    [enc setBytes:&fine.lp length:sizeof(fine.lp) atIndex:2];
    dispatch1D(enc, ctx.poissonProlongCopyPso, fine.n);
    [enc endEncoding];
  }
  for (int pass = 1; pass <= 3; ++pass) {
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:ctx.poissonProlongPso];
    [enc setBuffer:fine.pe offset:0 atIndex:0];
    [enc setBytes:&fine.lp length:sizeof(fine.lp) atIndex:1];
    [enc setBytes:&pass length:sizeof(int) atIndex:2];
    dispatch1D(enc, ctx.poissonProlongPso, fine.n);
    [enc endEncoding];
  }
}

// One V-cycle (== PoissonMultigrid vcycle): smooth, restrict the residual, recurse,
// prolong with the optimal A-energy alpha = <pe,r>/<pe,L pe> (reduced on device),
// post-smooth. levels[0].b = rhs, levels[0].x = the in-place iterate.
void encodePoissonGeomVcycle(id<MTLCommandBuffer> cb, Context& ctx, GpuPoissonGeom& H, int nu1,
                             int nu2, int coarseSweeps, float omega) {
  const int L = static_cast<int>(H.levels.size());
  for (int l = 0; l < L - 1; ++l) {
    GpuPoissonGeomLevel& cur = H.levels[l];
    GpuPoissonGeomLevel& nxt = H.levels[l + 1];
    pSmooth(cb, ctx, cur, cur.x, cur.b, nu1, omega);
    pMatvec(cb, ctx, cur, cur.x, cur.tmp);
    gVecSub(cb, ctx, cur.r, cur.b, cur.tmp, cur.n);  // r = b - L x
    pRestrict(cb, ctx, cur, nxt);                    // nxt.b = R r
    fill0(cb, nxt.x, static_cast<size_t>(nxt.n) * sizeof(float));
  }
  pSmooth(cb, ctx, H.levels[L - 1], H.levels[L - 1].x, H.levels[L - 1].b, coarseSweeps, omega);
  for (int l = L - 2; l >= 0; --l) {
    GpuPoissonGeomLevel& cur = H.levels[l];
    GpuPoissonGeomLevel& nxt = H.levels[l + 1];
    pProlong(cb, ctx, cur, nxt.x);                                 // pe = P e_{l+1}
    pMatvec(cb, ctx, cur, cur.pe, cur.tmp);                        // tmp = L pe
    gDot(cb, ctx, H.partialsNum, cur.pe, cur.r, cur.n);            // num = <pe,r>
    gDot(cb, ctx, H.partialsDen, cur.pe, cur.tmp, cur.n);          // den = <pe,L pe>
    gAlphaReduce(cb, ctx, H.alphaBuf, H.partialsNum, H.partialsDen, cur.n);
    gAxpyBuf(cb, ctx, cur.x, cur.pe, H.alphaBuf, cur.n);           // x += alpha pe
    pSmooth(cb, ctx, cur, cur.x, cur.b, nu2, omega);
  }
}

}  // namespace

int uploadPoissonGeom(int nx, int ny, int nz, double h) {
  Context& ctx = context();
  if (!ctx.ok) return -1;
  auto gh = std::make_unique<GpuPoissonGeom>();
  @autoreleasepool {
    id<MTLDevice> d = ctx.device;
    int lx = nx, ly = ny, lz = nz;
    double w = 1.0 / (h * h);
    for (;;) {
      GpuPoissonGeomLevel gl;
      gl.lp = LatParamsHost{lx, ly, lz, 0, static_cast<float>(w)};
      gl.n = lx * ly * lz;
      const size_t nb = static_cast<size_t>(gl.n) * sizeof(float);
      gl.x = [d newBufferWithLength:nb options:MTLResourceStorageModeShared];
      gl.b = [d newBufferWithLength:nb options:MTLResourceStorageModeShared];
      gl.r = [d newBufferWithLength:nb options:MTLResourceStorageModeShared];
      gl.pe = [d newBufferWithLength:nb options:MTLResourceStorageModeShared];
      gl.tmp = [d newBufferWithLength:nb options:MTLResourceStorageModeShared];
      gh->levels.push_back(gl);
      if (!(lx % 2 == 0 && ly % 2 == 0 && lz % 2 == 0 && lx >= 4 && ly >= 4 && lz >= 4)) break;
      lx /= 2; ly /= 2; lz /= 2; w /= 4.0;
    }
    const int nFine = gh->levels.front().n;
    const size_t pb = static_cast<size_t>((nFine + kTG - 1) / kTG) * sizeof(float);
    gh->partialsNum = [d newBufferWithLength:pb options:MTLResourceStorageModeShared];
    gh->partialsDen = [d newBufferWithLength:pb options:MTLResourceStorageModeShared];
    gh->alphaBuf = [d newBufferWithLength:sizeof(float) options:MTLResourceStorageModeShared];
    gh->valid = true;
  }
  return registerHandle(poissonGeomRegistry(), std::move(gh));
}

void freePoissonGeom(int handle) {
  auto& reg = poissonGeomRegistry();
  if (handle >= 0 && handle < static_cast<int>(reg.size())) reg[handle].reset();
}

int poissonGeomSolve(int handle, const std::vector<double>& rhs, std::vector<double>& phi,
                     double tol, int maxCycles, int nu1, int nu2, int coarseSweeps, double omega,
                     double* relResidualOut) {
  Context& ctx = context();
  auto& reg = poissonGeomRegistry();
  if (!ctx.ok || handle < 0 || handle >= static_cast<int>(reg.size()) || !reg[handle] ||
      !reg[handle]->valid)
    throw std::runtime_error("bochner::gpu: invalid poisson-geom handle / Metal unavailable");
  GpuPoissonGeom& H = *reg[handle];
  GpuPoissonGeomLevel& fine = H.levels[0];
  const uint32_t n = static_cast<uint32_t>(fine.n);
  if (rhs.size() != n || phi.size() != n)
    throw std::runtime_error("bochner::gpu: poissonGeomSolve rhs/phi size mismatch with hierarchy");
  const float omf = static_cast<float>(omega);
  int cycles = 0;
  double rel = 0.0;
  @autoreleasepool {
    writeF32(fine.b, rhs);
    writeF32(fine.x, phi);
    double bnorm = 0.0;
    {
      id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
      gDot(cb, ctx, H.partialsNum, fine.b, fine.b, n);
      [cb commit];
      [cb waitUntilCompleted];
      checkCb(cb, "poissonGeom/bnorm");
      bnorm = std::sqrt(gReadDot(H.partialsNum, n));
    }
    if (bnorm == 0.0) {
      // rhs = 0: the (mean-zero) solution is phi = 0, NOT the untouched warm
      // start -- same zero-RHS contract as the CPU solvers.
      std::fill(phi.begin(), phi.end(), 0.0);
      if (relResidualOut) *relResidualOut = 0.0;
      return 0;
    }
    // In single precision the geometric-MG residual FLOORS a little above 1e-6
    // (and the floor grows with ||rhs||), so a fixed tol below it would run every
    // solve to maxCycles. Stop on STAGNATION instead (the eigensolver's lesson):
    // once a cycle no longer cuts the residual by ~2x we have hit the float floor,
    // which the well-conditioned unpinned MG reaches in ~15 cycles. tol still wins
    // if genuinely met (the double CPU path is unaffected; it calls poissonVcycleSolve).
    double prevRel = 1e30;
    for (int cyc = 1; cyc <= maxCycles; ++cyc) {
      id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
      encodePoissonGeomVcycle(cb, ctx, H, nu1, nu2, coarseSweeps, omf);
      pMatvec(cb, ctx, fine, fine.x, fine.tmp);
      gVecSub(cb, ctx, fine.r, fine.b, fine.tmp, n);
      gDot(cb, ctx, H.partialsDen, fine.r, fine.r, n);
      [cb commit];
      [cb waitUntilCompleted];
      checkCb(cb, "poissonGeom/cycle");
      cycles = cyc;
      rel = std::sqrt(gReadDot(H.partialsDen, n)) / bnorm;
      if (rel < tol) break;
      // Only trust the "<2x reduction => at the float floor" heuristic once the
      // residual is actually near that floor (rel < 1e-3). Otherwise a weak-smoothing
      // / high-aspect solve whose true MG factor exceeds 0.5 would trip this at cycle 3
      // with rel ~ 0.3 and return a badly under-solved phi (non-divergence-free field).
      if (cyc >= 3 && rel < 1e-3 && rel > 0.5 * prevRel) break;
      prevRel = rel;
    }
    const float* xr = static_cast<const float*>(fine.x.contents);
    phi.assign(xr, xr + n);
  }
  if (relResidualOut) *relResidualOut = rel;
  return cycles;
}

}  // namespace bochner::gpu
