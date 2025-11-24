// Load tools from BFCL datasets.

#include "tool_taxonomy/tool_loader.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <unordered_set>

#include <boost/property_tree/json_parser.hpp>

#include "tool_taxonomy/utils.h"

namespace tool_taxonomy {
namespace {

using boost::property_tree::ptree;

bool ShouldSkipDir(const std::filesystem::path& path) {
    static const std::unordered_set<std::string> ignored = {
        "possible_answer", "sanity_check", "result", "score", ".git", "__pycache__", ".idea"};
    const auto name = path.filename().string();
    if (name.starts_with(".")) {
        return true;
    }
    return ignored.find(name) != ignored.end();
}

bool ShouldIncludeFile(const std::filesystem::path& path) {
    return path.has_extension() && path.extension() == ".json";
}

void AppendToolFromNode(const ptree& node,
                        const std::filesystem::path& source_file,
                        const std::string& entry_id,
                        bool deduplicate,
                        std::unordered_set<std::string>& seen,
                        std::vector<Tool>& tools) {
    Tool tool;
    tool.name = node.get<std::string>("name", "");
    if (tool.name.empty()) {
        return;
    }
    tool.description = node.get<std::string>("description", "");
    if (auto params = node.get_child_optional("parameters")) {
        tool.parameters = *params;
    }
    tool.source_file = source_file.string();
    tool.entry_id = entry_id;
    tool.summary = tool.name + "\n" + tool.description;
    if (!tool.parameters.empty()) {
        tool.summary.append("\nParameters:\n");
        tool.summary.append(FlattenParameters(tool.parameters));
    }

    if (deduplicate) {
        std::string key = tool.name + "::" + tool.description + "::" + SerializeTree(tool.parameters);
        if (!seen.insert(std::move(key)).second) {
            return;
        }
    }

    tools.emplace_back(std::move(tool));
}

void ProcessJsonLine(const std::string& line,
                     const std::filesystem::path& source_file,
                     bool deduplicate,
                     std::unordered_set<std::string>& seen,
                     std::vector<Tool>& tools,
                     std::size_t max_tools) {
    if (line.empty()) {
        return;
    }
    ptree tree;
    std::istringstream iss(line);
    try {
        boost::property_tree::read_json(iss, tree);
    } catch (const boost::property_tree::json_parser::json_parser_error& err) {
        std::cerr << "[WARN] Failed to parse JSON line in " << source_file << ": " << err.what()
                  << '\n';
        return;
    }

    const auto entry_id = tree.get<std::string>("id", "");
    if (auto functions = tree.get_child_optional("function")) {
        for (const auto& fn : *functions) {
            AppendToolFromNode(fn.second, source_file, entry_id, deduplicate, seen, tools);
            if (max_tools > 0 && tools.size() >= max_tools) {
                return;
            }
        }
        return;
    }

    if (tree.get<std::string>("name", "").empty()) {
        return;
    }
    AppendToolFromNode(tree, source_file, entry_id, deduplicate, seen, tools);
}

void LoadFromFile(const std::filesystem::path& file_path,
                  bool deduplicate,
                  std::unordered_set<std::string>& seen,
                  std::vector<Tool>& tools,
                  std::size_t max_tools) {
    std::ifstream input(file_path);
    if (!input.is_open()) {
        std::cerr << "[WARN] Unable to open " << file_path << '\n';
        return;
    }
    std::string line;
    while (std::getline(input, line)) {
        ProcessJsonLine(line, file_path, deduplicate, seen, tools, max_tools);
        if (max_tools > 0 && tools.size() >= max_tools) {
            break;
        }
    }
}

}  // namespace

std::vector<Tool> LoadTools(const std::filesystem::path& data_root,
                            std::size_t max_tools,
                            bool deduplicate) {
    std::vector<Tool> tools;
    std::unordered_set<std::string> seen;
    if (!std::filesystem::exists(data_root)) {
        throw std::runtime_error("Data root does not exist: " + data_root.string());
    }

    for (std::filesystem::recursive_directory_iterator it(data_root), end; it != end; ++it) {
        const auto& path = it->path();
        if (it->is_directory()) {
            if (ShouldSkipDir(path)) {
                it.disable_recursion_pending();
            }
            continue;
        }
        if (!it->is_regular_file() || !ShouldIncludeFile(path)) {
            continue;
        }
        LoadFromFile(path, deduplicate, seen, tools, max_tools);
        if (max_tools > 0 && tools.size() >= max_tools) {
            break;
        }
    }

    return tools;
}

}  // namespace tool_taxonomy
