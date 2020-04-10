// Copyright (c) 2016-2020 Andreas Goulas
// Licensed under the MIT license.

#ifndef METASINF_INCLUDE_METASINF_MUTATION_H_
#define METASINF_INCLUDE_METASINF_MUTATION_H_

#include <algorithm>
#include <climits>
#include <random>
#include <utility>

namespace snf {

/// Binary mutation.
///
/// The bits of the individual are flipped uniformly at random.
struct MutationFlip {
  explicit MutationFlip(double prob) : prob(prob) {}

  /// Mutation probability.
  double prob;

  template <typename T, typename Rng>
  void operator()(T& value, Rng& rng) {
    std::bernoulli_distribution dist(prob);
    for (size_t i = 0; i < value.size(); ++i) {
      if (dist(rng)) {
        value[i] = !value[i];
      }
    }
  }
};

/// Swap mutation.
///
/// The elements of the individual are interchanged uniformly at random.
struct MutationSwap {
  explicit MutationSwap(int count) : count(count) {}

  /// Number of swaps.
  int count;

  template <typename T, typename Rng>
  void operator()(T& value, Rng& rng) {
    std::uniform_int_distribution<size_t> dist(0, value.size() - 1);
    for (int i = 0; i < count; ++i) {
      size_t index0 = dist(rng);
      size_t index1;
      do {
        index1 = dist(rng);
      } while (index0 == index1);
      std::swap(value[index0], value[index1]);
    }
  }
};

/// Inversion mutation.
///
/// A segment of the individual is reversed uniformly at random.
struct MutationInvert {
  template <typename T, typename Rng>
  void operator()(T& value, Rng& rng) {
    std::uniform_int_distribution<size_t> dist(0, value.size() - 1);
    size_t index0 = dist(rng);
    size_t index1;
    do {
      index1 = dist(rng);
    } while (index0 == index1);
    std::reverse(value.begin() + std::min(index0, index1),
                 value.begin() + std::max(index0, index1));
  }
};

/// Move mutation.
///
/// An element of the individual is moved to a random position.
struct MutationMove {
  template <typename T, typename Rng>
  void operator()(T& value, Rng& rng) {
    std::uniform_int_distribution<size_t> dist(0, value.size() - 1);
    size_t index0 = dist(rng);
    size_t index1;
    do {
      index1 = dist(rng);
    } while (index0 == index1);

    size_t min_index = std::min(index0, index1);
    size_t max_index = std::max(index0, index1);
    typename T::value_type tmp = value[max_index];
    for (size_t i = max_index; i > min_index; --i) {
      value[i] = value[i - 1];
    }
    value[min_index] = tmp;
  }
};

/// Boundary mutation.
///
/// The value of the individual is randomly replaced by its lower or upper
/// bound.
template <typename T>
struct MutationBoundary {
  MutationBoundary(T lower_bound, T upper_bound)
      : lower_bound(lower_bound), upper_bound(upper_bound) {}

  /// Lower bound.
  T lower_bound;

  /// Upper bound.
  T upper_bound;

  template <typename Rng>
  void operator()(T& value, Rng& rng) {
    std::bernoulli_distribution dist;
    if (dist(rng)) {
      value = lower_bound;
    } else {
      value = upper_bound;
    }
  }
};

/// Mutate the specified real according to a normal distribution.
template <typename T>
struct MutationNormal {
  MutationNormal(T std_dev, T lower_bound, T upper_bound)
      : std_dev(std_dev), lower_bound(lower_bound), upper_bound(upper_bound) {}

  /// Standard deviation.
  T std_dev;

  /// Lower bound.
  T lower_bound;

  /// Upper bound.
  T upper_bound;

  template <typename Rng>
  void operator()(T& value, Rng& rng) {
    std::normal_distribution<T> dist(0.0, std_dev);
    value += dist(rng);
    value = std::min(std::max(value, lower_bound), upper_bound);
  }
};

/// Mutate the specified real according to a uniform distribution.
template <typename T>
struct MutationUniform {
  MutationUniform(T range, T lower_bound, T upper_bound)
      : range(range),
        lower_bound(lower_bound),
        upper_bound(upper_bound) {}

  /// Mutation range.
  T range;

  /// Lower bound.
  T lower_bound;

  /// Upper bound.
  T upper_bound;

  template <typename Rng>
  void operator()(T& value, Rng& rng) {
    std::uniform_real_distribution<T> dist(-range, range);
    value += dist(rng);
    value = std::min(std::max(value, lower_bound), upper_bound);
  }
};

/// Apply a mutation algorithm to each element of a vector.
template <typename MutationFunc>
struct MutationVector {
  MutationVector(double prob, const MutationFunc& func = MutationFunc())
      : prob(prob), func(func) {}

  /// Mutation probability.
  double prob;

  /// Wrapped mutation algorithm.
  MutationFunc func;

  template <typename T, typename Rng>
  void operator()(T& value, Rng& rng) {
    std::bernoulli_distribution dist(prob);
    for (auto& it : value) {
      if (dist(rng)) {
        func(it, rng);
      }
    }
  }
};

}  // namespace snf

#endif  // METASINF_INCLUDE_METASINF_MUTATION_H_
