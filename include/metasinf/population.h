// Copyright (c) 2016-2020 Andreas Goulas
// Licensed under the MIT license.

#ifndef METASINF_INCLUDE_METASINF_POPULATION_H_
#define METASINF_INCLUDE_METASINF_POPULATION_H_

#include <cassert>
#include <cmath>
#include <vector>
#include <functional>
#include <algorithm>

namespace snf {

/// Helper class used to specify a selection size.
struct SelectionSize {
  SelectionSize() : count(0), percentage(0.0) {}
  explicit SelectionSize(size_t count) : count(count), percentage(0.0) {}
  explicit SelectionSize(double percentage)
      : count(0), percentage(percentage) {}

  /// Number of samples.
  size_t count;

  /// Percentage.
  double percentage;

  size_t operator()(size_t size) const {
    if (count > 0) {
      return std::min(count, size);
    } else {
      assert(percentage >= 0.0 && percentage <= 1.0);
      return static_cast<size_t>(std::ceil(percentage * size));
    }
  }
};

/// Encapsulates an individual and his fitness score.
template <typename T, typename F>
struct Individual {
  Individual() : fitness(-1.0) {}
  explicit Individual(const T& data) : data(data), fitness(-1.0) {}
  Individual(const T& data, F fitness) : data(data), fitness(fitness) {}

  /// Data value.
  T data;

  /// Fitness value.
  F fitness;

  /// Return whether the individual is dirty, i.e. whether his fitness needs to
  /// be recomputed.
  bool is_dirty() const { return fitness < 0.0; }

  /// Mark the individual as dirty.
  void mark_dirty() { fitness = -1.0; }

  bool operator<(const Individual& rhs) const { return fitness < rhs.fitness; }
  bool operator>(const Individual& rhs) const { return fitness > rhs.fitness; }
};

template <typename T, typename F>
using Population = std::vector<Individual<T, F>>;

/// Compute the fitness of the individuals.
template <typename T, typename F, typename EvaluationFunc, typename Rng>
void Evaluate(Population<T, F>& pop, EvaluationFunc& func, Rng& rng) {
  for (auto& it : pop) {
    if (it.is_dirty()) {
      it.fitness = func(it.data, rng);
      assert(it.fitness >= 0.0);
    }
  }
}

}  // namespace snf

#endif  // METASINF_INCLUDE_METASINF_POPULATION_H_
