// Copyright (c) 2016-2020 Andreas Goulas
// Licensed under the MIT license.

#ifndef METASINF_INCLUDE_METASINF_SELECTION_H_
#define METASINF_INCLUDE_METASINF_SELECTION_H_

#include <cassert>
#include <cmath>
#include <vector>

#include "metasinf/population.h"

namespace snf {

/// Random selection.
///
/// The individuals are selected uniformly at random.
struct SelectionRandom {
  explicit SelectionRandom(SelectionSize size) : size(size) {}

  /// Number of individuals to select.
  SelectionSize size;

  template <typename T, typename F, typename Rng>
  void operator()(Population<T, F>& src, Population<T, F>& dst, Rng& rng) {
    size_t samples = size(src.size());
    dst.reserve(samples);

    std::uniform_int_distribution<size_t> dist(0, src.size() - 1);
    for (size_t i = 0; i < samples; ++i) {
      size_t index = dist(rng);
      dst.push_back(src[index]);
    }
  }
};

/// Truncation selection.
///
/// The individuals are sorted according to their fitness and the best are
/// selected.
struct SelectionTruncate {
  explicit SelectionTruncate(SelectionSize size) : size(size) {}

  /// Number of individuals to select.
  SelectionSize size;

  template <typename T, typename F, typename Rng>
  void operator()(Population<T, F>& src, Population<T, F>& dst, Rng& rng) {
    std::sort(src.begin(), src.end(), std::greater<Individual<T, F>>());

    size_t samples = size(src.size());
    dst.reserve(dst.size() + samples);
    dst.insert(dst.end(), src.begin(), src.begin() + samples);
  }
};

/// Roulette-wheel selection (also called stochastic sampling with replacement).
///
/// The individuals are mapped to contiguous segments of a line, such that
/// each individual's segment is equal in size to its fitness. A random
/// number is generated and the individual whose segment spans the random
/// number is selected. The process is repeated until the desired number of
/// individuals is obtained.
///
/// The roulette-wheel selection algorithm provides a zero bias but does not
/// guarantee minimum spread.
struct SelectionRouletteWheel {
  explicit SelectionRouletteWheel(SelectionSize size) : size(size) {}

  /// Number of individuals to select.
  SelectionSize size;

  template <typename T, typename F, typename Rng>
  void operator()(Population<T, F>& src, Population<T, F>& dst, Rng& rng) {
    thread_local std::vector<F> cum_fitness;

    if (src.empty()) {
      return;
    }

    cum_fitness.resize(src.size());
    cum_fitness[0] = src[0].fitness;
    for (size_t i = 1; i < src.size(); ++i) {
      cum_fitness[i] = src[i].fitness + cum_fitness[i - 1];
    }

    F total_fitness = 0.0;
    for (const auto& it : src) {
      total_fitness += it.fitness;
    }

    std::uniform_real_distribution<F> dist(0.0, total_fitness);
    size_t samples = size(src.size());
    dst.reserve(samples);
    for (size_t i = 0; i < samples; ++i) {
      F selection = dist(rng);
      size_t index = std::distance(
          cum_fitness.begin(),
          std::lower_bound(cum_fitness.begin(), cum_fitness.end(), selection));
      dst.push_back(src[index]);
    }
  }
};

/// Stochastic universal sampling.
///
/// The individuals are mapped to contiguous segments of a line, such that
/// each individual's segment is equal in size to its fitness exactly as in
/// roulette-wheel selection. Here equally spaced pointers are placed over
/// the line as many as there are individuals to be selected.
///
/// Stochastic universal sampling provides zero bias and minimum spread.
struct SelectionSus {
  explicit SelectionSus(SelectionSize size) : size(size) {}

  /// Number of individuals to select.
  SelectionSize size;

  template <typename T, typename F, typename Rng>
  void operator()(Population<T, F>& src, Population<T, F>& dst, Rng& rng) {
    size_t samples = size(src.size());
    dst.reserve(samples);

    F total_fitness = 0.0;
    for (const auto& it : src) {
      total_fitness += it.fitness;
    }

    std::uniform_real_distribution<F> dist;
    F offset = dist(rng);

    F cum_exp = 0.0;
    size_t index = 0;
    for (size_t i = 0; i < src.size(); ++i) {
      cum_exp += samples * src[i].fitness / total_fitness;
      while (cum_exp > offset + index) {
        dst.push_back(src[i]);
        ++index;
      }
    }
  }
};

/// Tournament selection.
///
/// In tournament selection a number of individuals are chosen randomly from
/// the population and the best individual from this group is selected as
/// parent. This process is repeated as often as individuals must be chosen.
struct SelectionTournament {
  SelectionTournament(SelectionSize size, int tournament_size)
      : size(size), tournament_size(tournament_size) {}

  /// Number of individuals to select.
  SelectionSize size;

  /// Size of each tournament.
  int tournament_size;

  template <typename T, typename F, typename Rng>
  void operator()(Population<T, F>& src, Population<T, F>& dst, Rng& rng) {
    assert(tournament_size > 0);

    std::uniform_int_distribution<size_t> dist(0, src.size() - 1);
    size_t samples = size(src.size());
    dst.reserve(samples);
    for (size_t i = 0; i < samples; ++i) {
      size_t best = dist(rng);
      for (int j = 0; j < tournament_size - 1; ++j) {
        size_t index = dist(rng);
        if (src[index].fitness > src[best].fitness) {
          best = index;
        }
      }

      dst.push_back(src[best]);
    }
  }
};

/// Linear rank-based fitness assignment.
struct FitnessRankLinear {
  template <typename F>
  F operator()(size_t rank, size_t size) {
    return size - rank;
  }
};

/// Rank-based selection.
///
/// The individuals are sorted according their fitness. The selection
/// probability of the individuals is adjusted according to their rank.
///
/// Rank-based fitness assignment overcomes the scaling problems of the
/// proportional fitness assignment.
template <typename SelectionFunc, typename FitnessFunc = FitnessRankLinear>
struct SelectionRank {
  SelectionRank(const SelectionFunc& selection = SelectionFunc(),
                const FitnessFunc& fitness = FitnessFunc())
      : selection(selection), fitness(fitness) {}

  /// Wrapped selection algorithm.
  SelectionFunc selection;

  /// Fitness assignment function.
  FitnessFunc fitness;

  template <typename T, typename F, typename Rng>
  void operator()(Population<T, F>& src, Population<T, F>& dst, Rng& rng) {
    thread_local Population<T, F> tmp;

    tmp = src;
    std::sort(tmp.begin(), tmp.end(), std::greater<Individual<T, F>>());
    for (size_t i = 0; i < tmp.size(); ++i) {
      tmp[i].fitness = fitness(i, tmp.size());
    }

    selection(tmp, dst, rng);
  }
};

/// Default sigma scaling.
struct FitnessSigmaDefault {
  template <typename F>
  F operator()(F fitness, F mean, F std_dev) {
    if (std_dev == 0.0) {
      return 1.0;
    }

    fitness = 1.0 + (fitness - mean) / (2.0 * std_dev);
    return fitness > 0.0 ? fitness : 0.1;
  }
};

/// Sigma-scaling selection.
///
/// The selection probablity of the individuals is adjusted according to the
/// mean population fitness and the fitness standard deviation.
///
/// Sigma-scaling helps avoid premature convergence and amplifies minor
/// fitness differences.
template <typename SelectionFunc, typename FitnessFunc = FitnessSigmaDefault>
struct SelectionSigma {
  SelectionSigma(const SelectionFunc& selection = SelectionFunc(),
                 const FitnessFunc& fitness = FitnessFunc())
      : selection(selection), fitness(fitness) {}

  /// Wrapped selection algorithm.
  SelectionFunc selection;

  /// Fitness assignment function.
  FitnessFunc fitness;

  template <typename T, typename F, typename Rng>
  void operator()(Population<T, F>& src, Population<T, F>& dst, Rng& rng) {
    thread_local Population<T, F> tmp;

    tmp = src;
    if (tmp.empty()) {
      return;
    }

    F mean_fitness = 0.0;
    for (const auto& it : src) {
      mean_fitness += it.fitness;
    }
    mean_fitness /= src.size();

    F std_dev = 0.0;
    for (const auto& it : src) {
      F diff = it.fitness - mean_fitness;
      std_dev += diff * diff;
    }
    std_dev = std::sqrt(std_dev / src.size());

    for (auto& it : tmp) {
      it.fitness = fitness(it.fitness, mean_fitness, std_dev);
    }

    selection(tmp, dst, rng);
  }
};

}  // namespace snf

#endif  // METASINF_INCLUDE_METASINF_SELECTION_H_
