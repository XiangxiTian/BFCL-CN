// Simple k-means implementation.

#include "tool_taxonomy/kmeans.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>

namespace tool_taxonomy {
namespace {

double DistanceSquared(const std::vector<float>& a, const std::vector<float>& b) {
    double sum = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double diff = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        sum += diff * diff;
    }
    return sum;
}

std::vector<std::vector<float>> InitializeCentroids(const std::vector<std::vector<float>>& data,
                                                    std::size_t k,
                                                    unsigned int seed) {
    std::vector<std::vector<float>> centroids;
    centroids.reserve(k);
    std::mt19937 gen(seed);
    std::uniform_int_distribution<std::size_t> dist(0, data.size() - 1);
    centroids.push_back(data[dist(gen)]);

    while (centroids.size() < k) {
        std::vector<double> distances(data.size(), 0.0);
        double sum = 0.0;
        for (std::size_t i = 0; i < data.size(); ++i) {
            double min_dist = std::numeric_limits<double>::max();
            for (const auto& centroid : centroids) {
                min_dist = std::min(min_dist, DistanceSquared(data[i], centroid));
            }
            distances[i] = min_dist;
            sum += min_dist;
        }

        if (sum == 0.0) {
            centroids.push_back(data[dist(gen)]);
            continue;
        }

        std::uniform_real_distribution<double> pick(0.0, sum);
        double target = pick(gen);
        double cumulative = 0.0;
        for (std::size_t i = 0; i < data.size(); ++i) {
            cumulative += distances[i];
            if (cumulative >= target) {
                centroids.push_back(data[i]);
                break;
            }
        }
    }
    return centroids;
}

}  // namespace

ClusterResult RunKMeans(const std::vector<std::vector<float>>& embeddings,
                        std::size_t k,
                        std::size_t max_iters,
                        double tol,
                        unsigned int seed) {
    ClusterResult result;
    if (embeddings.empty() || k == 0) {
        return result;
    }

    const std::size_t dim = embeddings.front().size();
    if (dim == 0) {
        throw std::runtime_error("Embeddings must have non-zero dimensionality.");
    }

    k = std::min<std::size_t>(k, embeddings.size());
    auto centroids = InitializeCentroids(embeddings, k, seed);
    result.assignments.assign(embeddings.size(), 0);

    for (std::size_t iter = 0; iter < max_iters; ++iter) {
        bool changed = false;
        for (std::size_t i = 0; i < embeddings.size(); ++i) {
            double best_dist = std::numeric_limits<double>::max();
            int best_cluster = 0;
            for (std::size_t cluster = 0; cluster < centroids.size(); ++cluster) {
                const double dist = DistanceSquared(embeddings[i], centroids[cluster]);
                if (dist < best_dist) {
                    best_dist = dist;
                    best_cluster = static_cast<int>(cluster);
                }
            }
            if (result.assignments[i] != best_cluster) {
                result.assignments[i] = best_cluster;
                changed = true;
            }
        }

        std::vector<std::vector<float>> new_centroids(k, std::vector<float>(dim, 0.0f));
        std::vector<std::size_t> counts(k, 0);
        for (std::size_t i = 0; i < embeddings.size(); ++i) {
            const int cluster = result.assignments[i];
            ++counts[cluster];
            for (std::size_t d = 0; d < dim; ++d) {
                new_centroids[cluster][d] += embeddings[i][d];
            }
        }

        std::mt19937 gen(seed + static_cast<unsigned int>(iter));
        std::uniform_int_distribution<std::size_t> dist(0, embeddings.size() - 1);
        double max_shift = 0.0;
        for (std::size_t cluster = 0; cluster < k; ++cluster) {
            if (counts[cluster] == 0) {
                new_centroids[cluster] = embeddings[dist(gen)];
                max_shift = std::numeric_limits<double>::infinity();
                continue;
            }
            for (std::size_t d = 0; d < dim; ++d) {
                new_centroids[cluster][d] /= static_cast<float>(counts[cluster]);
            }
            max_shift =
                std::max(max_shift, DistanceSquared(new_centroids[cluster], centroids[cluster]));
        }

        centroids = new_centroids;
        if (!changed || max_shift < tol * tol) {
            break;
        }
    }

    result.centroids = std::move(centroids);
    return result;
}

}  // namespace tool_taxonomy
