// Copyright (c) 2016-2020 Andreas Goulas
// Licensed under the MIT license.

#include <iostream>

#include "metasinf/crossover.h"
#include "metasinf/ga.h"
#include "metasinf/mutation.h"
#include "metasinf/replacement.h"
#include "metasinf/selection.h"
#include "metasinf/termination.h"

using Rng = std::mt19937;

// Maximize y = sin^6(4x) 0<x<1
double f(double& value, Rng rng) {
  return std::pow(std::sin(4.0 * value), 6);
}

int main() {
  Rng rng;
  rng.seed(static_cast<unsigned int>(time(nullptr)));

  auto ga = snf::make_ga(
      0.2, 0.8, f,
      snf::SelectionSus(snf::SelectionSize(0.4)),
      snf::CrossoverSbx<double>(3.0),
      snf::MutationNormal<double>(0.5, 0.0, 1.0),
      snf::ReplacementElitist(snf::SelectionSize(0.6)),
      snf::TerminationStagnation<double>(10));

  snf::Population<double, double> pop(20);
  for (auto& it : pop) {
    std::uniform_real_distribution<double> dist;
    it.data = dist(rng);
  }

  ga.Run(pop, rng);

  if (!pop.empty()) {
    snf::Evaluate(pop, f, rng);
    std::sort(pop.begin(), pop.end());

    auto best = pop.back();
    std::cout << best.data << " (Fitness: " << best.fitness << ")" << std::endl;
  }

  return 0;
}
