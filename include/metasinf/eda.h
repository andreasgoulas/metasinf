// Copyright (c) 2016-2020 Andreas Goulas
// Licensed under the MIT license.

#ifndef METASINF_INCLUDE_METASINF_EDA_H_
#define METASINF_INCLUDE_METASINF_EDA_H_

#include "metasinf/population.h"
#include "metasinf/termination.h"

namespace snf {

/// Estimation of distribution algorithm (also called probabilistic
/// model-building genetic algorithm).
template <
    typename EvaluationFunc,
    typename UpdateFunc,
    typename TerminationFunc>
struct Eda {
  /// Construct a new simulation.
  Eda(size_t pop_size,
      const EvaluationFunc& evaluation = EvaluationFunc(),
      const UpdateFunc& update = UpdateFunc(),
      const TerminationFunc& termination = TerminationFunc())
      : pop_size(pop_size),
        evaluation(evaluation),
        update(update),
        termination(termination) {}

  /// Population size.
  size_t pop_size;

  /// Evaluation functor.
  EvaluationFunc evaluation;

  /// Distribution update functor.
  UpdateFunc update;

  /// Termination functor.
  TerminationFunc termination;

  /// Perform the next evolution step.
  template <typename T, typename F, typename DistFunc, typename Rng>
  bool operator()(DistFunc& dist, Rng& rng) {
    thread_local Population<T, F> pop;

    pop.clear();
    pop.resize(pop_size);
    for (auto& it : pop) {
      dist(it.data, rng);
    }

    Evaluate(pop, evaluation, rng);
    update(dist, pop, rng);
    return termination(pop, rng);
  }

  /// Run the algorithm until the termination conditions have been met.
  template <typename T, typename F, typename DistFunc, typename Rng>
  void Run(DistFunc& dist, Rng& rng) {
    while (!operator()<T, F>(dist, rng)) {}
  }
};

template <typename... Args>
Eda<Args...> make_eda(size_t pop_size, Args... args) {
  return {pop_size, args...};
}

}  // namespace snf

#endif  // METASINF_INCLUDE_METASINF_EDA_H_
