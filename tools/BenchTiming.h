/// \file
/// Median-of-k wall-time measurement shared by the paper's benchmark tools.
///
/// Every reported wall time is the median of k identical repeats of the same
/// solve (k from the BENCH_REPS environment variable, default 5) -- the
/// paper-wide timing statistic. The helper also tracks the largest relative
/// spread (max-min)/median across all timed solves in the process, so a tool
/// can end with one summary line and the paper can quote a measured
/// "run-to-run spread below X%".
///
/// Repeats must be side-effect-free: reset solution vectors inside the
/// timed lambda, and never carry state from one repeat into the next.
#pragma once

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace benchstat {

inline int reps() {
  static const int k = [] {
    const char* s = std::getenv("BENCH_REPS");
    const int v = s ? std::atoi(s) : 0;
    return v >= 1 ? v : 5;
  }();
  return k;
}

inline double& maxRelSpreadRef() {
  static double s = 0.0;
  return s;
}

/// Median-of-k wall time (ms) of solve().
template <class F>
double medianMs(F&& solve, int k = reps()) {
  std::vector<double> t(static_cast<std::size_t>(k));
  for (int r = 0; r < k; ++r) {
    const auto t0 = std::chrono::steady_clock::now();
    solve();
    const auto t1 = std::chrono::steady_clock::now();
    t[static_cast<std::size_t>(r)] =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
  }
  std::sort(t.begin(), t.end());
  const double med = (k % 2) ? t[static_cast<std::size_t>(k / 2)]
                             : 0.5 * (t[static_cast<std::size_t>(k / 2 - 1)] +
                                      t[static_cast<std::size_t>(k / 2)]);
  if (med > 0.0) {
    maxRelSpreadRef() = std::max(maxRelSpreadRef(), (t.back() - t.front()) / med);
    if (std::getenv("BENCH_VERBOSE"))
      std::fprintf(stderr, "[timing] solve %.2f ms, spread %.1f%%\n", med,
                   100.0 * (t.back() - t.front()) / med);
  }
  return med;
}

/// One-line spread summary for the tail of a tool's output.
inline void printTimingSummary() {
  std::printf(
      "\n[timing] median of %d runs per solve; max rel spread (max-min)/median = %.1f%%\n",
      reps(), 100.0 * maxRelSpreadRef());
}

}  // namespace benchstat
