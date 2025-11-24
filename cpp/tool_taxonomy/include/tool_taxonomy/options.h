// Command-line options parsing.

#pragma once

#include <cstddef>
#include <filesystem>
#include <string>

namespace tool_taxonomy {

enum class ApiAuthMode {
    kQueryParam = 0,
    kBearerHeader = 1,
};

struct ProgramOptions {
    std::filesystem::path data_root;
    std::filesystem::path primary_classes_path;
    std::filesystem::path output_path;
    std::string gemma_endpoint;
    std::string gemma_model;
    std::string api_key;
    ApiAuthMode auth_mode = ApiAuthMode::kQueryParam;
    std::string api_key_header = "Authorization";
    std::string api_key_prefix = "Bearer ";
    double min_class_similarity = 0.2;
    std::size_t target_cluster_size = 12;
    std::size_t max_clusters = 6;
    std::size_t max_tools = 0;
    bool deduplicate = true;
    unsigned int random_seed = 42;
};

ProgramOptions ParseOptions(int argc, char** argv);
void PrintUsage(const char* program_name);

}  // namespace tool_taxonomy
