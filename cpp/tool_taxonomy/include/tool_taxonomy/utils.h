// Helper utilities.

#pragma once

#include <string>
#include <vector>

#include <boost/property_tree/ptree.hpp>

namespace tool_taxonomy {

std::string CurrentTimestamp();
std::string FlattenParameters(const boost::property_tree::ptree& parameters);
std::string SerializeTree(const boost::property_tree::ptree& tree);

double CosineSimilarity(const std::vector<float>& a, const std::vector<float>& b);
std::size_t DetermineClusterCount(std::size_t item_count,
                                  std::size_t target_cluster_size,
                                  std::size_t max_clusters);

}  // namespace tool_taxonomy
