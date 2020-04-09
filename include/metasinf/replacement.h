// Copyright (c) 2016-2020 Andreas Goulas
// Licensed under the MIT license.

#ifndef METASINF_INCLUDE_METASINF_REPLACEMENT_H_
#define METASINF_INCLUDE_METASINF_REPLACEMENT_H_

#include <algorithm>
#include <functional>
#include <random>
#include <utility>

#include "metasinf/population.h"

namespace snf {

/// Replace the entire population.
struct ReplacementAll {
  template <typename T, typename F, typename Rng>
  void operator()(Population<T, F>& src, Population<T, F>& dst, Rng& rng) {
    dst = std::move(src);
  }
};

/// Replace the worst individuals.
struct ReplacementElitist {
  explicit ReplacementElitist(SelectionSize size) : size(size) {}

  /// Elitism size.
  SelectionSize size;

  template <typename T, typename F, typename Rng>
  void operator()(Population<T, F>& src, Population<T, F>& dst, Rng& rng) {
    size_t count = size(dst.size());
    std::sort(dst.begin(), dst.end(), std::greater<Individual<T, F>>());
    dst.resize(count);

    dst.reserve(dst.size() + src.size());
    std::move(src.begin(), src.end(), std::back_inserter(dst));
    src.clear();
  }
};

}  // namespace snf

#endif  // METASINF_INCLUDE_METASINF_REPLACEMENT_H_
