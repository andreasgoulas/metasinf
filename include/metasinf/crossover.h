// Copyright (c) 2016-2020 Andreas Goulas
// Licensed under the MIT license.

#ifndef METASINF_INCLUDE_METASINF_CROSSOVER_H_
#define METASINF_INCLUDE_METASINF_CROSSOVER_H_

#include <cassert>
#include <climits>
#include <random>

namespace snf {

/// N-point crossover.
///
/// A number of crossover points are chosen at random. The elements between
/// successive points are exchanged between the two parents.
struct CrossoverPoint {
  explicit CrossoverPoint(int point_count = 1)
      : point_count(point_count) {}

  /// Number of crossover points.
  int point_count;

  template <typename T, typename Rng>
  void operator()(T& value0, T& value1, Rng& rng) {
    size_t size = std::max(value0.size(), value1.size());
    std::uniform_int_distribution<size_t> dist(0, size - 1);
    for (int i = 0; i < point_count; ++i) {
      size_t index = dist(rng);
      std::swap_ranges(value0.begin(), value0.begin() + index, value1.begin());
    }
  }
};

/// Uniform crossover.
///
/// The uniform crossover evaluates each element in the parents for exchange
/// with a probability of 0.5.
struct CrossoverUniform {
  template <typename T, typename Rng>
  void operator()(T& value0, T& value1, Rng& rng) {
    size_t size = std::min(value0.size(), value1.size());
    std::bernoulli_distribution dist;
    for (size_t i = 0; i < size; ++i) {
      if (dist(rng)) {
        std::swap(value0[i], value1[i]);
      }
    }
  }
};

/// Partially-matched crossover.
///
/// Two crossover points are selected at random and PMX proceeds by
/// position-wise exchanges.
struct CrossoverPmx {
  template <typename T, typename Rng>
  void operator()(T& value0, T& value1, Rng& rng) {
    thread_local std::vector<size_t> p0, p1;

    size_t size = std::min(value0.size(), value1.size());
    std::uniform_int_distribution<size_t> dist(0, size);
    size_t index0 = dist(rng);
    size_t index1 = dist(rng);
    if (index0 > index1) {
      std::swap(index0, index1);
    }

    if (index0 == index1) {
      return;
    }

    p0.assign(size, 0);
    p1.assign(size, 0);
    for (size_t i = 0; i < size; ++i) {
      assert(value0[i] < size);
      assert(value1[i] < size);
      p0[value0[i]] = i;
      p1[value1[i]] = i;
    }

    for (size_t i = index0; i < index1; ++i) {
      auto tmp0 = value0[i];
      auto tmp1 = value1[i];

      value0[i] = tmp1;
      value1[i] = tmp0;
      value0[p0[tmp1]] = tmp0;
      value1[p1[tmp0]] = tmp1;

      std::swap(p0[tmp0], p0[tmp1]);
      std::swap(p1[tmp0], p1[tmp1]);
    }
  }
};

/// Intermediate recombination.
///
/// The values of the offspring are selected around and between the values of
/// the parents.
template <typename T>
struct CrossoverReal {
  explicit CrossoverReal(T delta = 0.0) : delta(delta) {}

  /// Value range.
  T delta;

  template <typename Rng>
  void operator()(T& value0, T& value1, Rng& rng) {
    std::uniform_real_distribution<T> dist(-delta, delta + 1.0);
    T alpha0 = dist(rng);
    T alpha1 = dist(rng);
    T tmp0 = value0 * alpha0 + value1 * (1.0 - alpha0);
    T tmp1 = value1 * alpha1 + value0 * (1.0 - alpha1);
    value0 = tmp0;
    value1 = tmp1;
  }
};

/// Simulated binary crossover.
///
/// A large value of eta gives a higher probability for creating near parent
/// solutions and a small value of eta allows distant solutions to be
/// selected as children solutions.
template <typename T>
struct CrossoverSbx {
  explicit CrossoverSbx(T eta) : eta(eta) {}

  /// Distribution index.
  T eta;

  template <typename Rng>
  void operator()(T& value0, T& value1, Rng& rng) {
    assert(eta >= 0.0);
    std::uniform_real_distribution<T> dist;
    T u = dist(rng);
    T beta = 1.0;
    if (u < 0.5) {
      beta = std::pow(2 * u, 1.0 / (eta + 1.0));
    } else if (u > 0.5) {
      beta = std::pow(0.5 / (1.0 - u), 1.0 / (eta + 1.0));
    }

    T average = (value0 + value1) / 2.0;
    T diff = std::abs(value0 - value1) / 2.0;
    value0 = average - beta * diff;
    value1 = average + beta * diff;
  }
};

}  // namespace snf

#endif  // METASINF_INCLUDE_METASINF_CROSSOVER_H_
