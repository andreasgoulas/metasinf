// Copyright (c) 2016-2020 Andreas Goulas
// Licensed under the MIT license.

#ifndef METASINF_INCLUDE_METASINF_GA_H_
#define METASINF_INCLUDE_METASINF_GA_H_

#include <random>

#include "metasinf/population.h"

namespace snf {

/// Implementation of a genetic algorithm.
template <
    typename EvaluationFunc,
    typename SelectionFunc,
    typename CrossoverFunc,
    typename MutationFunc,
    typename ReplacementFunc,
    typename TerminationFunc>
struct Ga {
  /// Construct a new simulation.
  Ga(double mutation_rate, double crossover_rate,
     const EvaluationFunc& evaluation = EvaluationFunc(),
     const SelectionFunc& selection = SelectionFunc(),
     const CrossoverFunc& crossover = CrossoverFunc(),
     const MutationFunc& mutation = MutationFunc(),
     const ReplacementFunc& replacement = ReplacementFunc(),
     const TerminationFunc& termination = TerminationFunc())
      : mutation_rate(mutation_rate),
        crossover_rate(crossover_rate),
        evaluation(evaluation),
        selection(selection),
        crossover(crossover),
        mutation(mutation),
        replacement(replacement),
        termination(termination) {}

  /// Mutation rate.
  double mutation_rate;

  /// Crossover rate.
  double crossover_rate;

  /// Evaluation functor.
  EvaluationFunc evaluation;

  /// Selection functor.
  SelectionFunc selection;

  /// Crossover functor.
  CrossoverFunc crossover;

  /// Mutation functor.
  MutationFunc mutation;

  /// Replacement functor.
  ReplacementFunc replacement;

  /// Termination functor.
  TerminationFunc termination;

  /// Perform the next evolution step.
  template <typename T, typename F, typename Rng>
  bool operator()(Population<T, F>& pop, Rng& rng) {
    thread_local Population<T, F> tmp;

    assert(mutation_rate >= 0.0 && mutation_rate <= 1.0);
    assert(crossover_rate >= 0.0 && crossover_rate <= 1.0);
    if (pop.empty()) {
      return true;
    }

    Evaluate(pop, evaluation, rng);

    tmp.clear();
    selection(pop, tmp, rng);
    if (tmp.empty()) {
      return false;
    }

    std::shuffle(tmp.begin(), tmp.end(), rng);
    std::bernoulli_distribution mutation_dist(mutation_rate);
    std::bernoulli_distribution crossover_dist(crossover_rate);
    for (size_t i = 0; i < tmp.size() / 2; ++i) {
      Individual<T, F>& child0 = tmp[2 * i + 0];
      Individual<T, F>& child1 = tmp[2 * i + 1];

      if (crossover_dist(rng)) {
        crossover(child0.data, child1.data, rng);
        child0.mark_dirty();
        child1.mark_dirty();
      }

      if (mutation_dist(rng)) {
        mutation(child0.data, rng);
        child0.mark_dirty();
      }

      if (mutation_dist(rng)) {
        mutation(child1.data, rng);
        child1.mark_dirty();
      }
    }

    replacement(tmp, pop, rng);
    return termination(pop, rng);
  }

  /// Run the algorithm until the termination conditions have been met.
  template <typename T, typename F, typename Rng>
  void Run(Population<T, F>& pop, Rng& rng) {
    while (!operator()(pop, rng)) {}
  }
};

template <typename... Args>
Ga<Args...> make_ga(double mutation_rate, double crossover_rate, Args... args) {
  return {mutation_rate, crossover_rate, args...};
}

}  // namespace snf

#endif  // METASINF_INCLUDE_METASINF_H_
