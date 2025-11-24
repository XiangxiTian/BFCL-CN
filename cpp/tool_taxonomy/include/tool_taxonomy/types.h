// Copyright 2025
// Tool taxonomy data structures.

#pragma once

#include <string>
#include <vector>

#include <boost/property_tree/ptree.hpp>

namespace tool_taxonomy {

struct Tool {
    std::string name;
    std::string description;
    boost::property_tree::ptree parameters;
    std::string source_file;
    std::string entry_id;
    std::string summary;
    std::vector<float> embedding;
    std::string primary_class;
    double primary_similarity = 0.0;
    int cluster_id = -1;
};

struct PrimaryClass {
    std::string label;
    std::string description;
    std::vector<float> embedding;
};

}  // namespace tool_taxonomy
