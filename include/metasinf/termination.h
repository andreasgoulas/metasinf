// Copyright (c) 2016-2020 Andreas Goulas
// Licensed under the MIT license.

#ifndef METASINF_INCLUDE_METASINF_TERMINATION_H_
#define METASINF_INCLUDE_METASINF_TERMINATION_H_

#include <chrono>
#include <tuple>

#include "metasinf/population.h"

namespace snf {

/// Terminate the simulation after the specified amount of generations.
struct TerminationGeneration {
  explicit TerminationGeneration(int max_generations)
      : max_generations(max_generations), curr_generation_(0) {}

  /// Maximum number of generations.
  int max_generations;

  template <typename T, typename F, typename Rng>
  bool operator()(Population<T, F>& pop, Rng& rng) {
    ++curr_generation_;
    return curr_generation_ >= max_generations;
  }

 private:
  int curr_generation_;
};

/// Terminate the simulation when the specified fitness has been achieved.
template <typename F>
struct TerminationFitness {
  explicit TerminationFitness(F target_fitness)
      : target_fitness(target_fitness) {}

  /// Target fitness.
  F target_fitness;

  template <typename T, typename Rng>
  bool operator()(Population<T, F>& pop, Rng& rng) {
    auto best = std::max_element(pop.begin(), pop.end());
    if (best == pop.end()) {
      return true;
    }

    return best->fitness >= target_fitness;
  }
};

/// Terminate the simulation when the specified amount of time has elapsed.
struct TerminationTime {
  explicit TerminationTime(std::chrono::seconds max_time) : max_time(max_time) {
    start_time_ = Clock::now();
  }

  /// Maximum number of seconds.
  std::chrono::seconds max_time;

  template <typename T, typename F, typename Rng>
  bool operator()(Population<T, F>& pop, Rng& rng) {
    Clock::time_point now = Clock::now();
    return (now - start_time_) >= max_time;
  }

 private:
  using Clock = std::chrono::high_resolution_clock;
  Clock::time_point start_time_;
};

/// Terminate the simulation when no fitness improvement has been observed.
template <typename F>
struct TerminationStagnation {
  explicit TerminationStagnation(int max_generations)
      : max_generations(max_generations),
        curr_generation_(0),
        best_fitness_(0.0) {}

  /// Maximum number of generations without improvement.
  int max_generations;

  template <typename T, typename Rng>
  bool operator()(Population<T, F>& pop, Rng& rng) {
    auto best = std::max_element(pop.begin(), pop.end());
    if (best == pop.end()) {
      return true;
    }

    if (best->fitness > best_fitness_) {
      best_fitness_ = best->fitness;
      curr_generation_ = 0;
      return false;
    }

    ++curr_generation_;
    return curr_generation_ >= max_generations;
  }

 private:
  int curr_generation_;
  F best_fitness_;
};

/// Terminate the simulation based on a flag.
struct TerminationFlag {
  TerminationFlag() : flag(false) {}

  /// Flag indicating whether to terminate the simulation.
  bool flag;

  template <typename T, typename F, typename Rng>
  bool operator()(Population<T, F>& pop, Rng& rng) {
    return flag;
  }
};

/// Terminate the simulation when at least one of the specified termination
/// conditions has been met.
template <typename... Tp>
struct TerminationOr {
  TerminationOr(Tp... funcs) : funcs(funcs...) {}

  /// Termination conditions;
  std::tuple<Tp...> funcs;

  template <int I = 0, typename... Args>
  typename std::enable_if<I == sizeof...(Tp), bool>::type Check(Args... args) {
    return false;
  }

  template <int I = 0, typename... Args>
  typename std::enable_if<I < sizeof...(Tp), bool>::type Check(Args... args) {
    return std::get<I>(funcs)(args...) || Check<I + 1>(args...);
  }

  template <typename T, typename F, typename Rng>
  bool operator()(Population<T, F>& pop, Rng& rng) {
    return Check(pop, rng);
  }
};

/// Terminate the simulation when all of the specified termination conditions
/// have been met.
template <typename... Tp>
struct TerminationAnd {
  TerminationAnd(Tp... funcs) : funcs(funcs...) {}

  /// Termination conditions;
  std::tuple<Tp...> funcs;

  template <int I = 0, typename... Args>
  typename std::enable_if<I == sizeof...(Tp), bool>::type Check(Args... args) {
    return true;
  }

  template <int I = 0, typename... Args>
  typename std::enable_if<I < sizeof...(Tp), bool>::type Check(Args... args) {
    return std::get<I>(funcs)(args...) && Check<I + 1>(args...);
  }

  template <typename T, typename F, typename Rng>
  bool operator()(Population<T, F>& pop, Rng& rng) {
    return Check(pop, rng);
  }
};

}  // namespace snf

#endif  // METASINF_INCLUDE_METASINF_TERMINATION_H_
