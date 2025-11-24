// Tool clustering entry point.

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <boost/property_tree/json_parser.hpp>

#include "tool_taxonomy/embedder.h"
#include "tool_taxonomy/kmeans.h"
#include "tool_taxonomy/options.h"
#include "tool_taxonomy/tool_loader.h"
#include "tool_taxonomy/types.h"
#include "tool_taxonomy/utils.h"

namespace tool_taxonomy {
namespace {

using boost::property_tree::ptree;

constexpr char kUnassignedLabel[] = "unassigned";

std::vector<PrimaryClass> LoadPrimaryClasses(const std::filesystem::path& path) {
    if (!std::filesystem::exists(path)) {
        throw std::runtime_error("Primary classes file not found: " + path.string());
    }
    ptree tree;
    boost::property_tree::read_json(path.string(), tree);
    std::vector<PrimaryClass> classes;
    for (const auto& entry : tree) {
        PrimaryClass cls;
        if (entry.second.count("label") != 0U) {
            cls.label = entry.second.get<std::string>("label");
            cls.description = entry.second.get<std::string>("description", cls.label);
        } else {
            cls.label = entry.second.get_value<std::string>();
            cls.description = cls.label;
        }
        if (cls.label.empty()) {
            continue;
        }
        classes.emplace_back(std::move(cls));
    }
    if (classes.empty()) {
        throw std::runtime_error("Primary classes list is empty.");
    }
    return classes;
}

ptree BuildReport(const ProgramOptions& options,
                  const std::vector<Tool>& tools,
                  const std::vector<PrimaryClass>& classes,
                  const std::vector<std::string>& class_order,
                  const std::unordered_map<std::string, std::vector<const Tool*>>& class_lookup,
                  const std::unordered_map<std::string, double>& avg_similarity,
                  const std::unordered_map<std::string, std::size_t>& cluster_sizes,
                  const std::unordered_map<std::string, std::string>& descriptions,
                  std::size_t embedding_dim) {
    ptree root;
    root.put("generated_at", CurrentTimestamp());
    root.put("data_root", options.data_root.string());
    root.put("tool_count", tools.size());
    root.put("class_count", classes.size());
    root.put("embedding_model", options.gemma_model);
    root.put("embedding_endpoint", options.gemma_endpoint);
    root.put("embedding_dimensionality", embedding_dim);
    root.put("min_class_similarity", options.min_class_similarity);
    root.put("deduplicated", options.deduplicate);
    root.put("auth_mode", options.auth_mode == ApiAuthMode::kQueryParam ? "query" : "header");
    root.put("unassigned_label", kUnassignedLabel);

    ptree classes_array;
    for (const auto& label : class_order) {
        ptree class_node;
        class_node.put("label", label);
        auto desc_it = descriptions.find(label);
        class_node.put("description", desc_it == descriptions.end() ? "" : desc_it->second);

        const auto tools_it = class_lookup.find(label);
        const auto& tool_refs =
            tools_it == class_lookup.end() ? std::vector<const Tool*>() : tools_it->second;
        class_node.put("tool_count", tool_refs.size());
        const auto avg_it = avg_similarity.find(label);
        class_node.put("avg_similarity", avg_it == avg_similarity.end() ? 0.0 : avg_it->second);
        const auto cluster_it = cluster_sizes.find(label);
        class_node.put("cluster_count", cluster_it == cluster_sizes.end() ? 0 : cluster_it->second);

        ptree clusters_array;
        std::map<int, std::vector<const Tool*>> cluster_map;
        for (const auto* tool : tool_refs) {
            cluster_map[tool->cluster_id].push_back(tool);
        }
        for (const auto& [cluster_id, members] : cluster_map) {
            ptree cluster_node;
            cluster_node.put("cluster_id", cluster_id);
            cluster_node.put("size", members.size());
            ptree members_array;
            for (const auto* tool : members) {
                ptree tool_node;
                tool_node.put("name", tool->name);
                tool_node.put("description", tool->description);
                tool_node.put("source_file", tool->source_file);
                if (!tool->entry_id.empty()) {
                    tool_node.put("entry_id", tool->entry_id);
                }
                tool_node.put("primary_similarity", tool->primary_similarity);
                tool_node.put("primary_class", tool->primary_class);
                tool_node.put("cluster_id", tool->cluster_id);
                tool_node.put("summary", tool->summary);
                tool_node.add_child("parameters", tool->parameters);
                members_array.push_back({"", tool_node});
            }
            cluster_node.add_child("members", members_array);
            clusters_array.push_back({"", cluster_node});
        }

        class_node.add_child("clusters", clusters_array);
        classes_array.push_back({"", class_node});
    }

    root.add_child("classes", classes_array);
    return root;
}

}  // namespace
}  // namespace tool_taxonomy

int main(int argc, char** argv) {
    using namespace tool_taxonomy;
    try {
        const auto options = ParseOptions(argc, argv);
        auto primary_classes = LoadPrimaryClasses(options.primary_classes_path);
        GemmaEmbeddingClient embedder(options);

        std::cout << "[INFO] Encoding " << primary_classes.size() << " primary classes..." << std::endl;
        std::size_t embedding_dim = 0;
        for (auto& cls : primary_classes) {
            cls.embedding = embedder.Embed(cls.description);
            if (cls.embedding.empty()) {
                throw std::runtime_error("Empty embedding for class: " + cls.label);
            }
            if (embedding_dim == 0) {
                embedding_dim = cls.embedding.size();
            } else if (cls.embedding.size() != embedding_dim) {
                throw std::runtime_error("Inconsistent embedding dimension for class: " + cls.label);
            }
        }

        std::cout << "[INFO] Loading tools from " << options.data_root << "..." << std::endl;
        auto tools = LoadTools(options.data_root, options.max_tools, options.deduplicate);
        std::cout << "[INFO] Loaded " << tools.size() << " tools." << std::endl;

        std::unordered_map<std::string, std::string> class_descriptions;
        for (const auto& cls : primary_classes) {
            class_descriptions[cls.label] = cls.description;
        }
        class_descriptions[kUnassignedLabel] =
            "Tools that fell below the similarity threshold for any primary class.";

        std::cout << "[INFO] Encoding tools with Gemma..." << std::endl;
        for (auto& tool : tools) {
            tool.embedding = embedder.Embed(tool.summary);
            if (!tool.embedding.empty() && tool.embedding.size() != embedding_dim) {
                throw std::runtime_error("Embedding dimension mismatch for tool: " + tool.name);
            }
        }

        std::unordered_map<std::string, std::vector<std::size_t>> class_index_map;
        class_index_map[kUnassignedLabel] = {};

        for (std::size_t idx = 0; idx < tools.size(); ++idx) {
            auto& tool = tools[idx];
            double best_sim = -2.0;
            std::string best_label = kUnassignedLabel;
            for (const auto& cls : primary_classes) {
                const double sim = CosineSimilarity(tool.embedding, cls.embedding);
                if (sim > best_sim) {
                    best_sim = sim;
                    best_label = cls.label;
                }
            }
            if (best_sim < options.min_class_similarity) {
                best_label = kUnassignedLabel;
            }
            tool.primary_class = best_label;
            tool.primary_similarity = best_sim;
            class_index_map[best_label].push_back(idx);
        }

        std::vector<std::string> class_order;
        class_order.reserve(primary_classes.size() + 1);
        for (const auto& cls : primary_classes) {
            class_order.push_back(cls.label);
            if (!class_index_map.count(cls.label)) {
                class_index_map[cls.label] = {};
            }
        }
        class_order.push_back(kUnassignedLabel);

        std::unordered_map<std::string, std::vector<const Tool*>> class_lookup;
        std::unordered_map<std::string, double> avg_similarity;
        std::unordered_map<std::string, std::size_t> cluster_sizes;

        for (const auto& label : class_order) {
            const auto& indexes = class_index_map[label];
            std::vector<const Tool*> references;
            references.reserve(indexes.size());
            for (const auto idx : indexes) {
                references.push_back(&tools[idx]);
            }
            class_lookup[label] = std::move(references);

            if (indexes.empty()) {
                avg_similarity[label] = 0.0;
                cluster_sizes[label] = 0;
                continue;
            }

            double sum_similarity = 0.0;
            std::vector<std::vector<float>> subset_embeddings;
            subset_embeddings.reserve(indexes.size());
            for (const auto idx : indexes) {
                subset_embeddings.push_back(tools[idx].embedding);
                sum_similarity += tools[idx].primary_similarity;
            }
            avg_similarity[label] = sum_similarity / static_cast<double>(indexes.size());

            const std::size_t cluster_count =
                DetermineClusterCount(indexes.size(), options.target_cluster_size, options.max_clusters);
            const std::size_t resolved_cluster_count = std::max<std::size_t>(1, cluster_count);
            ClusterResult clusters =
                RunKMeans(subset_embeddings, resolved_cluster_count, 100, 1e-4, options.random_seed);
            const std::size_t realized_clusters =
                clusters.centroids.empty() ? resolved_cluster_count : clusters.centroids.size();
            cluster_sizes[label] = realized_clusters;
            if (clusters.assignments.empty()) {
                for (const auto idx : indexes) {
                    tools[idx].cluster_id = 0;
                }
            } else {
                for (std::size_t local = 0; local < indexes.size(); ++local) {
                    tools[indexes[local]].cluster_id = clusters.assignments[local];
                }
            }
        }

        const auto report = BuildReport(options,
                                        tools,
                                        primary_classes,
                                        class_order,
                                        class_lookup,
                                        avg_similarity,
                                        cluster_sizes,
                                        class_descriptions,
                                        embedding_dim);
        boost::property_tree::write_json(options.output_path.string(), report);
        std::cout << "[INFO] Clustering summary written to " << options.output_path << std::endl;
        return EXIT_SUCCESS;
    } catch (const std::exception& err) {
        std::cerr << "[ERROR] " << err.what() << std::endl;
        return EXIT_FAILURE;
    }
}
