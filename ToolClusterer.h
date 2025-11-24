#ifndef TOOL_CLUSTERER_H
#define TOOL_CLUSTERER_H

#include <string>
#include <vector>
#include <map>
#include "json.hpp"
#include "GemmaEncoder.h"

using json = nlohmann::json;

// --- Data Structures ---

struct Tool {
    std::string id;
    std::string name;
    std::string description;
    json parameters;
    std::vector<float> embedding;
    std::string primary_class;
    int cluster_id = -1;
};

struct PrimaryClass {
    std::string name;
    std::vector<float> embedding;
};

// --- Core Functions ---

// Main pipeline function to execute the whole process
void run_tool_clustering_pipeline(const std::string& data_path);

// Helper functions exposed for modularity if needed
std::vector<Tool> get_tool_list(const std::string& filepath);
float cosine_similarity(const std::vector<float>& a, const std::vector<float>& b);
void cluster_tools_kmeans(std::vector<Tool*>& tools, int k);

#endif // TOOL_CLUSTERER_H
