// Copyright (c) 2016-2020 Andreas Goulas
// Licensed under the MIT license.

#ifndef METASINF_INCLUDE_METASINF_PBIL_H_
#define METASINF_INCLUDE_METASINF_PBIL_H_

#include <array>
#include <random>

#include "metasinf/population.h"

namespace snf {

/// Encapsulates a distribution.
template <typename ProbT, size_t Size>
struct PbilDist {
  PbilDist() {
    prob.fill(0.5);
  }

  /// Probability vector.
  std::array<ProbT, Size> prob;

  template <typename T, typename Rng>
  void operator()(T& value, Rng& rng) {
    for (size_t i = 0; i < Size; ++i) {
      std::bernoulli_distribution dist(prob[i]);
      value[i] = dist(rng);
    }
  }
};

/// Population-based incremental learning algorithm implementation.
template <typename ProbT, size_t Size>
struct PbilUpdate {
  PbilUpdate(ProbT rate, size_t best_count,
             double mutation_prob, ProbT mutation_shift,
             ProbT lower_bound, ProbT upper_bound)
      : rate(rate),
        best_count(best_count),
        mutation_prob(mutation_prob),
        mutation_shift(mutation_shift),
        lower_bound(lower_bound),
        upper_bound(upper_bound) {}

  /// Learning rate.
  ProbT rate;

  /// Number of individuals to sample.
  size_t best_count;

  /// Mutation probability.
  double mutation_prob;

  /// Mutation shift.
  ProbT mutation_shift;

  /// Lower bound.
  ProbT lower_bound;

  /// Upper bound.
  ProbT upper_bound;

  template <typename T, typename F, typename Rng>
  void operator()(PbilDist<ProbT, Size>& dist, Population<T, F>& pop,
                  Rng& rng) {
    assert(best_count > 0 && best_count <= pop.size());

    std::bernoulli_distribution flip_dist, mutation_dist(mutation_prob);
    std::sort(pop.begin(), pop.end(), std::greater<Individual<T, F>>());
    for (size_t i = 0; i < Size; ++i) {
      ProbT p = dist.prob[i] * (1.0 - rate);
      ProbT lr = rate / best_count;
      for (size_t j = 0; j < best_count; ++j) {
        if (pop[j].data[i]) {
          p += lr;
        }
      }

      if (mutation_dist(rng)) {
        p *= 1.0 - mutation_shift;
        if (flip_dist(rng)) {
          p += mutation_shift;
        }
      }

      dist.prob[i] = std::min(std::max(p, lower_bound), upper_bound);
    }
  }
};

}  // namespace snf

#endif  // METASINF_INCLUDE_METASINF_PBIL_H_
