// Copyright (c) 2016-2020 Andreas Goulas
// Licensed under the MIT license.

#ifndef METASINF_INCLUDE_METASINF_ISLAND_MODEL_H_
#define METASINF_INCLUDE_METASINF_ISLAND_MODEL_H_

#include "metasinf/population.h"

namespace snf {

/// Encapsulates a subpopulation.
template <typename T, typename F, typename Ga>
struct Island {
  explicit Island(const Ga& ga = Ga()) : ga(ga) {}

  /// Island population.
  Population<T, F> pop;

  /// Genetic algorithm used to step the island.
  Ga ga;

  /// Perform the next evolution step.
  template <typename Rng>
  bool operator()(Rng& rng) {
    return ga(pop, rng);
  }
};

/// Island model implementation.
///
/// The population is divided into multiple subpopulations. These
/// subpopulations evolve independently for a certain number of generations.
/// A number of individuals are then distributed between the subpopulations.
template <typename MigrationFunc>
struct IslandModel {
  /// Construct a new simulation.
  IslandModel(int migration_rate,
              const MigrationFunc& migration = MigrationFunc())
      : migration_rate(migration_rate), migration(migration) {}

  /// Migration rate.
  int migration_rate;

  /// Migration functor.
  MigrationFunc migration;

  /// Perform the next evolution step.
  template <typename T, typename F, typename Ga, typename Rng>
  bool operator()(std::vector<Island<T, F, Ga>>& islands, Rng& rng) {
    assert(migration_rate > 0);
    for (int i = 0; i < migration_rate; ++i) {
      bool result = false;
      for (auto& it : islands) {
        if (it(rng)) {
          result = true;
        }
      }

      if (result) {
        return true;
      }
    }

    migration(islands, rng);
    return false;
  }

  /// Run the algorithm until the termination conditions have been met.
  template <typename T, typename F, typename Ga, typename Rng>
  void Run(std::vector<Island<T, F, Ga>>& islands, Rng& rng) {
    while (!operator()(islands, rng)) {}
  }
};

}  // namespace snf

#endif  // METASINF_INCLUDE_METASINF_ISLAND_MODEL_H_
