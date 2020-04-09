// Copyright (c) 2016-2020 Andreas Goulas
// Licensed under the MIT license.

#include <array>
#include <iostream>

#include "metasinf/crossover.h"
#include "metasinf/ga.h"
#include "metasinf/mutation.h"
#include "metasinf/replacement.h"
#include "metasinf/selection.h"
#include "metasinf/termination.h"

static constexpr int kSize = 64;
using State = std::array<uint8_t, kSize>;
using Rng = std::mt19937;

void PrintState(const State& state) {
  for (int y = 0; y < kSize; ++y) {
    for (int x = 0; x < kSize; ++x) {
      if (y == state[x]) {
        std::cout << "Q";
      } else {
        std::cout << " ";
      }

      if (x != kSize - 1) {
        std::cout << "|";
      }
    }

    std::cout << std::endl;
  }
}

bool CheckQueen(State& value, int i) {
  for (int j = 0; j < kSize; ++j) {
    if (i == j) {
      continue;
    }

    int dx = i - j;
    int dy = value[i] - value[j];
    if (std::abs(dx) == std::abs(dy)) {
      return false;
    }
  }

  return true;
}

double f(State& value, Rng& rng) {
  double fitness = 0.0;
  for (int i = 0; i < kSize; ++i) {
    if (CheckQueen(value, i)) {
      fitness += 1.0;
    }
  }

  return fitness;
}

int main() {
  Rng rng;
  rng.seed(static_cast<unsigned int>(time(nullptr)));

  auto ga = snf::make_ga(
    0.2, 0.8, f,
    snf::SelectionSus(snf::SelectionSize(0.4)),
    snf::CrossoverPmx(),
    snf::MutationSwap(1),
    snf::ReplacementElitist(snf::SelectionSize(0.6)),
    snf::TerminationFitness<double>(kSize));

  snf::Population<State, double> pop(20);
  for (auto& it : pop) {
    for (int i = 0; i < kSize; ++i) {
      it.data[i] = i;
    }

    std::shuffle(it.data.begin(), it.data.end(), rng);
  }

  ga.Run(pop, rng);

  if (!pop.empty()) {
    snf::Evaluate(pop, f, rng);
    std::sort(pop.begin(), pop.end());

    const auto& best = pop.back();
    std::cout << "Fitness: " << best.fitness << std::endl;
    PrintState(best.data);
  }

  return 0;
}
