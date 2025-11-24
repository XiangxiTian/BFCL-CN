// Command-line parsing.

#include "tool_taxonomy/options.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace tool_taxonomy {
namespace {

std::string RequireValue(int argc, char** argv, int& index, const std::string& flag) {
    if (index + 1 >= argc) {
        throw std::runtime_error("Flag requires a value: " + flag);
    }
    ++index;
    return argv[index];
}

}  // namespace

ProgramOptions ParseOptions(int argc, char** argv) {
    ProgramOptions options;
    options.data_root = std::filesystem::current_path() / "data";
    options.output_path = "tool_clusters.json";
    options.gemma_endpoint =
        "https://generativelanguage.googleapis.com/v1beta/models/gemma-0.3b-embedder:embedContent";
    options.gemma_model = "models/gemma-0.3b-embedder";

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            PrintUsage(argv[0]);
            std::exit(EXIT_SUCCESS);
        } else if (arg == "--data-root") {
            options.data_root = RequireValue(argc, argv, i, arg);
        } else if (arg == "--primary-classes") {
            options.primary_classes_path = RequireValue(argc, argv, i, arg);
        } else if (arg == "--output") {
            options.output_path = RequireValue(argc, argv, i, arg);
        } else if (arg == "--gemma-endpoint") {
            options.gemma_endpoint = RequireValue(argc, argv, i, arg);
        } else if (arg == "--gemma-model") {
            options.gemma_model = RequireValue(argc, argv, i, arg);
        } else if (arg == "--api-key") {
            options.api_key = RequireValue(argc, argv, i, arg);
        } else if (arg == "--min-class-sim") {
            options.min_class_similarity = std::stod(RequireValue(argc, argv, i, arg));
        } else if (arg == "--target-cluster-size") {
            options.target_cluster_size = std::stoul(RequireValue(argc, argv, i, arg));
        } else if (arg == "--max-clusters") {
            options.max_clusters = std::stoul(RequireValue(argc, argv, i, arg));
        } else if (arg == "--max-tools") {
            options.max_tools = std::stoul(RequireValue(argc, argv, i, arg));
        } else if (arg == "--seed") {
            options.random_seed = static_cast<unsigned int>(std::stoul(RequireValue(argc, argv, i, arg)));
        } else if (arg == "--no-dedupe") {
            options.deduplicate = false;
        } else if (arg == "--auth-mode") {
            const auto value = RequireValue(argc, argv, i, arg);
            if (value == "query") {
                options.auth_mode = ApiAuthMode::kQueryParam;
            } else if (value == "header") {
                options.auth_mode = ApiAuthMode::kBearerHeader;
            } else {
                throw std::runtime_error("Unknown auth mode: " + value);
            }
        } else if (arg == "--api-key-header") {
            options.api_key_header = RequireValue(argc, argv, i, arg);
        } else if (arg == "--api-key-prefix") {
            options.api_key_prefix = RequireValue(argc, argv, i, arg);
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    if (options.primary_classes_path.empty()) {
        throw std::runtime_error("--primary-classes is required.");
    }
    if (options.min_class_similarity < -1.0 || options.min_class_similarity > 1.0) {
        throw std::runtime_error("--min-class-sim must be between -1 and 1.");
    }
    return options;
}

void PrintUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " --primary-classes <file> [options]\n\n"
              << "Required arguments:\n"
              << "  --primary-classes    Path to JSON file describing primary tool classes.\n\n"
              << "Common options:\n"
              << "  --data-root PATH             Root directory containing BFCL data (default: ./data)\n"
              << "  --output PATH                Output JSON file (default: tool_clusters.json)\n"
              << "  --gemma-endpoint URL         Gemma embedding endpoint URL.\n"
              << "  --gemma-model ID             Gemma model identifier to use.\n"
              << "  --api-key KEY                API key for the embedding endpoint.\n"
              << "  --auth-mode [query|header]   Authentication strategy (default: query).\n"
              << "  --api-key-header NAME        Header used when auth-mode=header (default: Authorization).\n"
              << "  --api-key-prefix VALUE       Prefix applied to API key for header auth (default: \"Bearer \").\n"
              << "  --min-class-sim VALUE        Minimum cosine similarity for class assignment (default: 0.2).\n"
              << "  --target-cluster-size N      Target tools per cluster (default: 12).\n"
              << "  --max-clusters N             Maximum clusters per class (default: 6).\n"
              << "  --max-tools N                Limit the number of tools processed (default: all).\n"
              << "  --no-dedupe                  Disable deduplication of identical tools.\n"
              << "  --seed VALUE                 Random seed for clustering (default: 42).\n"
              << "  --help                       Show this message.\n";
}

}  // namespace tool_taxonomy
