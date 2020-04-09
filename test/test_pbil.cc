// Copyright (c) 2016-2020 Andreas Goulas
// Licensed under the MIT license.

#include <array>
#include <bitset>
#include <iomanip>
#include <iostream>

#include "metasinf/eda.h"
#include "metasinf/pbil.h"

static constexpr int kSize = 80;
using State = std::bitset<kSize>;
using Rng = std::mt19937;

// Four peaks
double f(State& value, Rng& rng) {
  static constexpr int kThreshold = 10;
  static constexpr int kReward = 100;

  int start;
  for (start = 0; start < kSize; ++start) {
    if (value[start]) {
      break;
    }
  }

  int end;
  for (end = kSize - 1; end >= 0; --end) {
    if (!value[end]) {
      break;
    }
  }

  int head = start;
  int tail = kSize - end - 1;
  int fitness = std::max(head, tail);
  if (head > kThreshold && tail > kThreshold) {
    fitness += kReward;
  }

  return fitness;
}

int main() {
  Rng rng;
  rng.seed(static_cast<unsigned int>(time(nullptr)));

  auto eda = snf::make_eda(
      100, f,
      snf::PbilUpdate<double, kSize>(0.1, 1, 0.02, 0.05, 0.0, 1.0),
      snf::TerminationGeneration(10000));

  snf::PbilDist<double, kSize> dist;
  eda.Run<State, double>(dist, rng);

  std::cout << std::setprecision(2);
  for (int i = 0; i < kSize; ++i) {
    std::cout << dist.prob[i];
    if (i == kSize - 1) {
      std::cout << std::endl;
    } else {
      std::cout << ", ";
    }
  }

  return 0;
}
