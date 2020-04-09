// Copyright (c) 2016-2020 Andreas Goulas
// Licensed under the MIT license.

#ifndef METASINF_INCLUDE_METASINF_MIGRATION_H_
#define METASINF_INCLUDE_METASINF_MIGRATION_H_

#include <random>
#include <utility>
#include <vector>

#include "metasinf/island_model.h"

namespace snf {

/// Migrate individuals uniformly at random.
struct MigrationRandom {
  explicit MigrationRandom(const SelectionSize& size) : size(size) {}

  /// Number of individuals to migrate per island.
  SelectionSize size;

  template <typename T, typename F, typename Ga, typename Rng>
  void operator()(std::vector<Island<T, F, Ga>>& islands, Rng& rng) {
    assert(islands.size() > 1);
    std::uniform_int_distribution<size_t> dist(1, islands.size() - 1);
    for (size_t i = 0; i < islands.size(); ++i) {
      Population<T, F>& src = islands[i].pop;
      std::shuffle(src.begin(), src.end(), rng);

      size_t count = size(src.size());
      for (auto it = src.end() - count; it != src.end(); ++it) {
        size_t index = dist(rng);
        if (index == i) {
          index = 0;
        }

        islands[index].pop.push_back(std::move(*it));
      }

      src.erase(src.end() - count, src.end());
    }
  }
};

/// Migrate individuals between adjacent islands arranged in a ring topology.
struct MigrationRing {
  explicit MigrationRing(const SelectionSize& size) : size(size) {}

  /// Number of individuals to migrate per island.
  SelectionSize size;

  template <typename T, typename F, typename Ga, typename Rng>
  void operator()(std::vector<Island<T, F, Ga>>& islands, Rng& rng) {
    assert(islands.size() > 1);
    for (size_t i = 0; i < islands.size(); ++i) {
      Population<T, F>& src = islands[i].pop;
      std::shuffle(src.begin(), src.end(), rng);

      size_t index = i + 1;
      if (index >= islands.size()) {
        index = 0;
      }

      size_t count = size(src.size());
      Population<T, F>& dst = islands[index].pop;
      dst.reserve(dst.size() + count);
      std::move(src.end() - count, src.end(), std::back_inserter(dst));
      src.erase(src.end() - count, src.end());
    }
  }
};

}  // namespace snf

#endif  // METASINF_INCLUDE_METASINF_MIGRATION_H_
