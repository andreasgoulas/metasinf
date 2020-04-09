// Copyright (c) 2016-2020 Andreas Goulas
// Licensed under the MIT license.

#include <iostream>

#include "metasinf/crossover.h"
#include "metasinf/ga.h"
#include "metasinf/island_model.h"
#include "metasinf/migration.h"
#include "metasinf/mutation.h"
#include "metasinf/replacement.h"
#include "metasinf/selection.h"
#include "metasinf/termination.h"

using Rng = std::mt19937;

// Maximize y = sin^6(8x) 0<x<1
double f(double& value, Rng& rng) {
  return std::pow(std::sin(8.0 * value), 6);
}

int main() {
  Rng rng;
  rng.seed(static_cast<unsigned int>(time(nullptr)));

  snf::MigrationRing migration(snf::SelectionSize(0.1));
  snf::IslandModel<snf::MigrationRing> island_model(50, migration);

  auto ga = snf::make_ga(
      0.2, 0.8, f,
      snf::SelectionSus(snf::SelectionSize(0.4)),
      snf::CrossoverSbx<double>(3.0),
      snf::MutationNormal<double>(0.5, 0.0, 1.0),
      snf::ReplacementElitist(snf::SelectionSize(0.6)),
      snf::TerminationGeneration(1000));

  using Island = snf::Island<double, double, decltype(ga)>;

  std::vector<Island> islands;
  for (int i = 0; i < 6; ++i) {
    Island island(ga);
    island.pop.resize(20);
    for (auto& it : island.pop) {
      std::uniform_real_distribution<double> dist;
      it.data = dist(rng);
    }

    islands.push_back(island);
  }

  island_model.Run(islands, rng);

  int i = 0;
  for (auto& island : islands) {
    if (!island.pop.empty()) {
      snf::Evaluate(island.pop, f, rng);
      std::sort(island.pop.begin(), island.pop.end());

      auto best = island.pop.back();
      std::cout << "Island " << ++i << ": " << best.data
                << " (Fitness: " << best.fitness << ")" << std::endl;
    }
  }

  return 0;
}
