// Utility implementations.

#include "tool_taxonomy/utils.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>

#include <boost/property_tree/json_parser.hpp>

namespace tool_taxonomy {
namespace {

void FlattenPropertyMap(const boost::property_tree::ptree& props,
                        int depth,
                        std::ostringstream& oss) {
    const std::string indent(depth * 2, ' ');
    for (const auto& entry : props) {
        const auto& key = entry.first;
        const auto& node = entry.second;
        const auto type = node.get<std::string>("type", "any");
        const auto description = node.get<std::string>("description", "");
        oss << indent << "- " << key << " (" << type << ")";
        if (!description.empty()) {
            oss << ": " << description;
        }
        oss << '\n';

        if (auto nested = node.get_child_optional("properties")) {
            FlattenPropertyMap(*nested, depth + 1, oss);
        } else if (auto items = node.get_child_optional("items")) {
            const auto item_type = items->get<std::string>("type", "any");
            oss << indent << "  [items: " << item_type << "]\n";
        }
    }
}

}  // namespace

std::string CurrentTimestamp() {
    const auto now = std::chrono::system_clock::now();
    const auto time = std::chrono::system_clock::to_time_t(now);
    std::ostringstream oss;
    oss << std::put_time(std::gmtime(&time), "%Y-%m-%dT%H:%M:%SZ");
    return oss.str();
}

std::string FlattenParameters(const boost::property_tree::ptree& parameters) {
    std::ostringstream oss;
    if (auto props = parameters.get_child_optional("properties")) {
        FlattenPropertyMap(*props, 0, oss);
    }
    return oss.str();
}

std::string SerializeTree(const boost::property_tree::ptree& tree) {
    std::ostringstream oss;
    boost::property_tree::write_json(oss, tree, false);
    return oss.str();
}

double CosineSimilarity(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.empty() || b.empty() || a.size() != b.size()) {
        return -1.0;
    }
    double dot = 0.0;
    double norm_a = 0.0;
    double norm_b = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        dot += static_cast<double>(a[i]) * static_cast<double>(b[i]);
        norm_a += static_cast<double>(a[i]) * static_cast<double>(a[i]);
        norm_b += static_cast<double>(b[i]) * static_cast<double>(b[i]);
    }
    if (norm_a == 0.0 || norm_b == 0.0) {
        return -1.0;
    }
    return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
}

std::size_t DetermineClusterCount(std::size_t item_count,
                                  std::size_t target_cluster_size,
                                  std::size_t max_clusters) {
    if (item_count == 0) {
        return 0;
    }
    const std::size_t safe_target = std::max<std::size_t>(1, target_cluster_size);
    const std::size_t raw = static_cast<std::size_t>(
        std::ceil(static_cast<double>(item_count) / static_cast<double>(safe_target)));
    const std::size_t bounded = std::max<std::size_t>(
        1, std::min<std::size_t>(max_clusters == 0 ? raw : max_clusters, std::max<std::size_t>(1, raw)));
    return std::min<std::size_t>(item_count, bounded);
}

}  // namespace tool_taxonomy
