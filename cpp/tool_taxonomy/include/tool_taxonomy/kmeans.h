// Lightweight k-means clustering.

#pragma once

#include <cstddef>
#include <vector>

namespace tool_taxonomy {

struct ClusterResult {
    std::vector<int> assignments;
    std::vector<std::vector<float>> centroids;
};

ClusterResult RunKMeans(const std::vector<std::vector<float>>& embeddings,
                        std::size_t k,
                        std::size_t max_iters,
                        double tol,
                        unsigned int seed);

}  // namespace tool_taxonomy
