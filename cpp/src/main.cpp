#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <system_error>

#include "simple_json.h"

namespace fs = std::filesystem;
using simplejson::JsonArray;
using simplejson::JsonObject;
using simplejson::JsonValue;

struct ToolParameter {
    std::string name;
    std::string type;
    std::string description;
    bool required = false;
};

struct ToolDoc {
    std::string name;
    std::string description;
    std::vector<ToolParameter> parameters;
    std::string source;
    std::string raw_json;

    std::string SurfaceText() const {
        std::ostringstream oss;
        oss << name << ". " << description;
        if (!parameters.empty()) {
            oss << " Parameters: ";
            for (size_t i = 0; i < parameters.size(); ++i) {
                const auto& p = parameters[i];
                oss << p.name << " (" << (p.type.empty() ? "unknown" : p.type) << ", "
                    << (p.required ? "required" : "optional") << "): " << p.description;
                if (i + 1 != parameters.size()) {
                    oss << " | ";
                }
            }
        }
        return oss.str();
    }
};

struct PrimaryClass {
    std::string id;
    std::string name;
    std::string description;
    std::vector<float> embedding;
    std::vector<float> normalized;
};

struct Options {
    fs::path tool_root = "data";
    fs::path primary_class_file = "config/primary_classes.json";
    fs::path output_file = "tool_clusters.json";
    std::string api_key;
    std::string gemma_model = "text-embedding-004";
    size_t max_tools = 0;
    size_t batch_size = 8;
    double similarity_threshold = 0.35;
    bool offline = false;
    size_t offline_dim = 128;
    uint64_t seed = 42;
    size_t min_cluster_size = 6;
    size_t max_clusters = 6;
};

void PrintUsage() {
    std::cout << "Tool classifier using Gemma embeddings\n"
              << "Options:\n"
              << "  --tool-root <path>           Root directory or JSON file with tool docs (default: data)\n"
              << "  --primary-classes <path>     JSON file describing primary classes (default: config/primary_classes.json)\n"
              << "  --output <path>              Output JSON file (default: tool_clusters.json)\n"
              << "  --api-key <key>              Google Generative AI API key (fallback to GOOGLE_API_KEY env)\n"
              << "  --gemma-model <name>         Gemma embedding model id (default: text-embedding-004)\n"
              << "  --max-tools <n>              Optional limit on number of tools to process\n"
              << "  --batch-size <n>             Batch size per embedding request (default: 8)\n"
              << "  --similarity-threshold <v>   Minimum cosine similarity to assign to a primary class (default: 0.35)\n"
              << "  --min-cluster-size <n>       Minimum cluster size per primary class (default: 6)\n"
              << "  --max-clusters <n>           Maximum k-means clusters per primary class (default: 6)\n"
              << "  --offline                    Use deterministic synthetic embeddings (no API calls)\n"
              << "  --offline-dim <n>            Embedding dimension in offline mode (default: 128)\n"
              << "  --seed <n>                   Random seed (default: 42)\n"
              << "  --help                       Show this message\n";
}

Options ParseArgs(int argc, char** argv) {
    Options opts;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto require_value = [&](const std::string& flag) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for " + flag);
            }
            return argv[++i];
        };
        if (arg == "--tool-root") {
            opts.tool_root = require_value(arg);
        } else if (arg == "--primary-classes") {
            opts.primary_class_file = require_value(arg);
        } else if (arg == "--output") {
            opts.output_file = require_value(arg);
        } else if (arg == "--api-key") {
            opts.api_key = require_value(arg);
        } else if (arg == "--gemma-model") {
            opts.gemma_model = require_value(arg);
        } else if (arg == "--max-tools") {
            opts.max_tools = static_cast<size_t>(std::stoull(require_value(arg)));
        } else if (arg == "--batch-size") {
            opts.batch_size = static_cast<size_t>(std::stoull(require_value(arg)));
        } else if (arg == "--similarity-threshold") {
            opts.similarity_threshold = std::stod(require_value(arg));
        } else if (arg == "--min-cluster-size") {
            opts.min_cluster_size = static_cast<size_t>(std::stoull(require_value(arg)));
        } else if (arg == "--max-clusters") {
            opts.max_clusters = static_cast<size_t>(std::stoull(require_value(arg)));
        } else if (arg == "--offline") {
            opts.offline = true;
        } else if (arg == "--offline-dim") {
            opts.offline_dim = static_cast<size_t>(std::stoull(require_value(arg)));
        } else if (arg == "--seed") {
            opts.seed = std::stoull(require_value(arg));
        } else if (arg == "--help") {
            PrintUsage();
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }
    if (opts.batch_size == 0) {
        opts.batch_size = 1;
    }
    if (opts.max_clusters == 0) {
        opts.max_clusters = 1;
    }
    if (opts.min_cluster_size == 0) {
        opts.min_cluster_size = 1;
    }
    return opts;
}

std::string Trim(const std::string& input) {
    size_t start = 0;
    while (start < input.size() && std::isspace(static_cast<unsigned char>(input[start]))) {
        ++start;
    }
    size_t end = input.size();
    while (end > start && std::isspace(static_cast<unsigned char>(input[end - 1]))) {
        --end;
    }
    return input.substr(start, end - start);
}

std::string ReadFileToString(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        return {};
    }
    std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

std::string ShellQuote(const std::string& value) {
    std::string quoted = "'";
    for (char ch : value) {
        if (ch == '\'') {
            quoted += "'\"'\"'";
        } else {
            quoted.push_back(ch);
        }
    }
    quoted.push_back('\'');
    return quoted;
}

std::string Slugify(std::string_view text) {
    std::string slug;
    slug.reserve(text.size());
    for (char ch : text) {
        if (std::isalnum(static_cast<unsigned char>(ch))) {
            slug.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
        } else if (std::isspace(static_cast<unsigned char>(ch)) || ch == '-' || ch == '_') {
            if (!slug.empty() && slug.back() != '_') {
                slug.push_back('_');
            }
        }
    }
    if (slug.empty()) {
        slug = "class";
    }
    return slug;
}

std::string JsonValueToString(const JsonValue& value) {
    if (value.IsString()) {
        return value.AsString();
    }
    return value.Dump();
}

std::vector<ToolParameter> ExtractParameters(const JsonValue& parameters_value) {
    std::vector<ToolParameter> result;
    if (!parameters_value.IsObject()) {
        return result;
    }
    const auto& param_obj = parameters_value.AsObject();
    std::unordered_set<std::string> required;
    auto req_it = param_obj.find("required");
    if (req_it != param_obj.end() && req_it->second.IsArray()) {
        for (const auto& entry : req_it->second.AsArray()) {
            if (entry.IsString()) {
                required.insert(entry.AsString());
            }
        }
    }
    auto prop_it = param_obj.find("properties");
    if (prop_it == param_obj.end() || !prop_it->second.IsObject()) {
        return result;
    }
    const auto& properties = prop_it->second.AsObject();
    for (const auto& [name, spec] : properties) {
        ToolParameter param;
        param.name = name;
        if (spec.IsObject()) {
            const auto& spec_obj = spec.AsObject();
            auto type_it = spec_obj.find("type");
            if (type_it != spec_obj.end() && type_it->second.IsString()) {
                param.type = type_it->second.AsString();
            }
            auto desc_it = spec_obj.find("description");
            if (desc_it != spec_obj.end()) {
                param.description = JsonValueToString(desc_it->second);
            }
            auto enum_it = spec_obj.find("enum");
            if (enum_it != spec_obj.end() && enum_it->second.IsArray()) {
                param.description += " Enum options: " + enum_it->second.Dump();
            }
            auto default_it = spec_obj.find("default");
            if (default_it != spec_obj.end()) {
                param.description += " Default: " + JsonValueToString(default_it->second);
            }
        } else {
            param.description = JsonValueToString(spec);
        }
        param.required = required.count(name) > 0;
        result.push_back(std::move(param));
    }
    return result;
}

std::optional<ToolDoc> ToolFromJson(const JsonValue& value, const std::string& source_path) {
    if (!value.IsObject()) {
        return std::nullopt;
    }
    const auto& obj = value.AsObject();
    auto name_it = obj.find("name");
    auto desc_it = obj.find("description");
    auto param_it = obj.find("parameters");
    if (name_it == obj.end() || !name_it->second.IsString()) {
        return std::nullopt;
    }
    ToolDoc doc;
    doc.name = name_it->second.AsString();
    if (desc_it != obj.end() && desc_it->second.IsString()) {
        doc.description = desc_it->second.AsString();
    }
    if (param_it != obj.end()) {
        doc.parameters = ExtractParameters(param_it->second);
    }
    doc.source = source_path;
    doc.raw_json = value.Dump();
    return doc;
}

std::vector<fs::path> CollectJsonFiles(const fs::path& root) {
    std::vector<fs::path> files;
    if (fs::is_regular_file(root)) {
        files.push_back(root);
        return files;
    }
    if (!fs::exists(root) || !fs::is_directory(root)) {
        throw std::runtime_error("Invalid tool root: " + root.string());
    }
    for (const auto& entry : fs::recursive_directory_iterator(root)) {
        if (entry.is_regular_file() && entry.path().extension() == ".json") {
            files.push_back(entry.path());
        }
    }
    return files;
}

std::vector<ToolDoc> GetToolList(const fs::path& root, size_t max_tools) {
    std::vector<ToolDoc> tools;
    std::unordered_set<std::string> seen;
    for (const auto& file : CollectJsonFiles(root)) {
        std::ifstream input(file);
        if (!input) {
            std::cerr << "Warning: Unable to open " << file << '\n';
            continue;
        }
        std::string line;
        size_t line_no = 0;
        while (std::getline(input, line)) {
            ++line_no;
            std::string trimmed = Trim(line);
            if (trimmed.empty()) {
                continue;
            }
            try {
                JsonValue json = simplejson::Parse(trimmed);
                if (json.IsObject() && json.Contains("function")) {
                    const auto& funcs = json.At("function").AsArray();
                    for (const auto& func : funcs) {
                        auto maybe_doc = ToolFromJson(func, file.string());
                        if (maybe_doc) {
                            std::string key = maybe_doc->name + "|" + maybe_doc->description;
                            if (seen.insert(key).second) {
                                tools.push_back(std::move(*maybe_doc));
                            }
                        }
                    }
                } else {
                    auto maybe_doc = ToolFromJson(json, file.string());
                    if (maybe_doc) {
                        std::string key = maybe_doc->name + "|" + maybe_doc->description;
                        if (seen.insert(key).second) {
                            tools.push_back(std::move(*maybe_doc));
                        }
                    }
                }
            } catch (const std::exception& ex) {
                std::cerr << "Warning: Failed to parse " << file << ":" << line_no << " (" << ex.what() << ")\n";
            }
            if (max_tools != 0 && tools.size() >= max_tools) {
                return tools;
            }
        }
    }
    return tools;
}

std::vector<PrimaryClass> LoadPrimaryClasses(const fs::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Unable to open primary class file: " + path.string());
    }
    std::stringstream buffer;
    buffer << input.rdbuf();
    JsonValue root = simplejson::Parse(buffer.str());
    JsonArray entries;
    if (root.IsArray()) {
        entries = root.AsArray();
    } else if (root.IsObject() && root.Contains("classes")) {
        entries = root.At("classes").AsArray();
    } else {
        throw std::runtime_error("Primary class file must be an array or object with 'classes'");
    }
    std::vector<PrimaryClass> classes;
    for (const auto& entry : entries) {
        if (!entry.IsObject()) {
            continue;
        }
        const auto& obj = entry.AsObject();
        auto name_it = obj.find("name");
        if (name_it == obj.end() || !name_it->second.IsString()) {
            continue;
        }
        PrimaryClass cls;
        cls.name = name_it->second.AsString();
        auto id_it = obj.find("id");
        cls.id = (id_it != obj.end() && id_it->second.IsString()) ? id_it->second.AsString() : Slugify(cls.name);
        auto desc_it = obj.find("description");
        if (desc_it != obj.end() && desc_it->second.IsString()) {
            cls.description = desc_it->second.AsString();
        }
        classes.push_back(std::move(cls));
    }
    if (classes.empty()) {
        throw std::runtime_error("No primary classes found in " + path.string());
    }
    return classes;
}

class EmbeddingProvider {
  public:
    virtual ~EmbeddingProvider() = default;
    virtual std::vector<std::vector<float>> Embed(const std::vector<std::string>& texts) = 0;
    virtual size_t Dim() const = 0;
};

class SimulatedEncoder : public EmbeddingProvider {
  public:
    SimulatedEncoder(size_t dim, uint64_t seed) : dim_(dim), seed_(seed) {}

    std::vector<std::vector<float>> Embed(const std::vector<std::string>& texts) override {
        std::vector<std::vector<float>> embeddings;
        embeddings.reserve(texts.size());
        for (const auto& text : texts) {
            std::vector<float> vec(dim_);
            std::mt19937 gen(static_cast<uint64_t>(std::hash<std::string>{}(text)) ^ seed_);
            std::normal_distribution<float> dist(0.0f, 1.0f);
            float norm = 0.0f;
            for (size_t i = 0; i < dim_; ++i) {
                vec[i] = dist(gen);
                norm += vec[i] * vec[i];
            }
            norm = std::sqrt(std::max(norm, std::numeric_limits<float>::epsilon()));
            for (auto& v : vec) {
                v /= norm;
            }
            embeddings.push_back(std::move(vec));
        }
        return embeddings;
    }

    size_t Dim() const override { return dim_; }

  private:
    size_t dim_;
    uint64_t seed_;
};

class GemmaEncoder : public EmbeddingProvider {
  public:
    GemmaEncoder(std::string api_key, std::string model_id, size_t batch_size, uint64_t seed)
        : api_key_(std::move(api_key)),
          model_id_(std::move(model_id)),
          batch_size_(batch_size),
          rng_seed_(seed) {}

    std::vector<std::vector<float>> Embed(const std::vector<std::string>& texts) override {
        std::vector<std::vector<float>> embeddings;
        embeddings.reserve(texts.size());
        for (size_t i = 0; i < texts.size(); i += batch_size_) {
            size_t end = std::min(texts.size(), i + batch_size_);
            std::vector<std::string> batch(texts.begin() + static_cast<long>(i),
                                           texts.begin() + static_cast<long>(end));
            auto batch_embeddings = EmbedBatch(batch);
            embeddings.insert(embeddings.end(), batch_embeddings.begin(), batch_embeddings.end());
        }
        return embeddings;
    }

    size_t Dim() const override { return dim_; }

  private:
    std::string api_key_;
    std::string model_id_;
    size_t batch_size_;
    size_t dim_ = 0;
    uint64_t rng_seed_;
    mutable uint64_t temp_counter_ = 0;
    const std::string base_url_ = "https://generativelanguage.googleapis.com/v1beta/models/";

    static std::string Escape(const std::string& input) {
        std::string out;
        out.reserve(input.size() + 2);
        out.push_back('"');
        for (char ch : input) {
            switch (ch) {
                case '"':
                case '\\':
                    out.push_back('\\');
                    out.push_back(ch);
                    break;
                case '\n':
                    out += "\\n";
                    break;
                case '\r':
                    out += "\\r";
                    break;
                case '\t':
                    out += "\\t";
                    break;
                default:
                    out.push_back(ch);
            }
        }
        out.push_back('"');
        return out;
    }

    std::string BuildBatchPayload(const std::vector<std::string>& texts) const {
        std::ostringstream oss;
        oss << "{";
        oss << "\"model\":\"models/" << model_id_ << "\",";
        oss << "\"requests\":[";
        for (size_t i = 0; i < texts.size(); ++i) {
            if (i != 0) {
                oss << ",";
            }
            oss << "{\"content\":{\"parts\":[{\"text\":" << Escape(texts[i]) << "}]}}";
        }
        oss << "]}";
        return oss.str();
    }

    fs::path MakeTempFile(const std::string& suffix) const {
        auto timestamp = std::chrono::steady_clock::now().time_since_epoch().count();
        std::ostringstream name;
        name << "gemma_tmp_" << timestamp << "_" << (temp_counter_++) << suffix;
        return fs::temp_directory_path() / name.str();
    }

    std::string ExecuteCurl(const std::string& endpoint_suffix, const std::string& payload) const {
        fs::path payload_path = MakeTempFile(".json");
        fs::path response_path = MakeTempFile(".out");
        fs::path status_path = MakeTempFile(".status");
        fs::path error_path = MakeTempFile(".err");
        {
            std::ofstream payload_file(payload_path);
            if (!payload_file) {
                throw std::runtime_error("Failed to create temporary payload file");
            }
            payload_file << payload;
        }
        std::string url = base_url_ + model_id_ + endpoint_suffix;
        std::ostringstream cmd;
        cmd << "curl -sS -o " << ShellQuote(response_path.string()) << " -w \"%{http_code}\" "
            << "-H " << ShellQuote("Content-Type: application/json") << " "
            << "-H " << ShellQuote("x-goog-api-key: " + api_key_) << " "
            << "--data-binary @" << ShellQuote(payload_path.string()) << " "
            << ShellQuote(url) << " > " << ShellQuote(status_path.string()) << " 2> " << ShellQuote(error_path.string());
        int rc = std::system(cmd.str().c_str());
        std::string status_text = Trim(ReadFileToString(status_path));
        std::string error_text = ReadFileToString(error_path);
        auto cleanup = [&](const fs::path& p) {
            std::error_code ec;
            fs::remove(p, ec);
        };
        cleanup(payload_path);
        cleanup(status_path);
        cleanup(error_path);
        std::string response_body = ReadFileToString(response_path);
        cleanup(response_path);
        if (rc != 0) {
            throw std::runtime_error("curl failed (code " + std::to_string(rc) + "): " + error_text);
        }
        if (status_text.size() < 3) {
            throw std::runtime_error("Invalid HTTP status from Gemma API: " + status_text + " " + error_text);
        }
        int status_code = 0;
        try {
            status_code = std::stoi(status_text.substr(0, 3));
        } catch (...) {
            throw std::runtime_error("Unable to parse HTTP status: " + status_text);
        }
        if (status_code < 200 || status_code >= 300) {
            throw std::runtime_error("Gemma API returned HTTP " + std::to_string(status_code) + ": " + response_body);
        }
        return response_body;
    }

    std::vector<std::vector<float>> ExtractEmbeddings(const JsonValue& json) {
        std::vector<std::vector<float>> embeddings;
        auto parse_values = [&](const JsonValue& node) -> std::vector<float> {
            if (node.IsObject()) {
                const auto& obj = node.AsObject();
                auto values_it = obj.find("values");
                auto value_it = obj.find("value");
                const JsonArray* arr = nullptr;
                if (values_it != obj.end() && values_it->second.IsArray()) {
                    arr = &values_it->second.AsArray();
                } else if (value_it != obj.end() && value_it->second.IsArray()) {
                    arr = &value_it->second.AsArray();
                }
                if (arr) {
                    std::vector<float> vec;
                    vec.reserve(arr->size());
                    for (const auto& val : *arr) {
                        vec.push_back(static_cast<float>(val.AsNumber()));
                    }
                    return vec;
                }
            }
            if (json.IsArray()) {
                std::vector<float> vec;
                for (const auto& val : json.AsArray()) {
                    vec.push_back(static_cast<float>(val.AsNumber()));
                }
                return vec;
            }
            throw std::runtime_error("Unable to parse embedding values");
        };

        if (json.IsObject()) {
            const auto& obj = json.AsObject();
            auto responses_it = obj.find("responses");
            auto embeddings_it = obj.find("embeddings");
            if (responses_it != obj.end() && responses_it->second.IsArray()) {
                for (const auto& resp : responses_it->second.AsArray()) {
                    embeddings.push_back(parse_values(resp));
                }
            } else if (embeddings_it != obj.end() && embeddings_it->second.IsArray()) {
                for (const auto& resp : embeddings_it->second.AsArray()) {
                    embeddings.push_back(parse_values(resp));
                }
            } else if (obj.find("embedding") != obj.end()) {
                embeddings.push_back(parse_values(obj.at("embedding")));
            } else {
                throw std::runtime_error("Unexpected embedding response payload");
            }
        } else {
            throw std::runtime_error("Unexpected embedding response type");
        }
        if (dim_ == 0 && !embeddings.empty()) {
            dim_ = embeddings.front().size();
        }
        return embeddings;
    }

    std::vector<std::vector<float>> EmbedBatch(const std::vector<std::string>& texts) {
        if (texts.empty()) {
            return {};
        }
        const std::string payload = BuildBatchPayload(texts);
        const std::string response = ExecuteCurl(":batchEmbedContents", payload);
        JsonValue json = simplejson::Parse(response);
        return ExtractEmbeddings(json);
    }
};

std::vector<float> Normalize(const std::vector<float>& vec) {
    double norm = 0.0;
    for (float v : vec) {
        norm += static_cast<double>(v) * static_cast<double>(v);
    }
    norm = std::sqrt(std::max(norm, std::numeric_limits<double>::epsilon()));
    std::vector<float> normalized(vec.size());
    for (size_t i = 0; i < vec.size(); ++i) {
        normalized[i] = static_cast<float>(vec[i] / norm);
    }
    return normalized;
}

float CosineSim(const std::vector<float>& a, const std::vector<float>& b) {
    double dot = 0.0;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
        dot += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    }
    return static_cast<float>(dot);
}

struct ClusterResult {
    std::vector<int> assignments;
    std::vector<std::vector<float>> centroids;
};

double SquaredDistance(const std::vector<float>& a, const std::vector<float>& b) {
    double sum = 0.0;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
        double diff = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        sum += diff * diff;
    }
    return sum;
}

ClusterResult RunKMeans(const std::vector<std::vector<float>>& data, int k, uint64_t seed, int max_iters = 100) {
    ClusterResult result;
    if (data.empty() || k <= 0) {
        return result;
    }
    const int n = static_cast<int>(data.size());
    k = std::min(k, n);
    std::mt19937 gen(seed);
    std::uniform_int_distribution<int> dist(0, n - 1);
    std::vector<int> centers;
    centers.reserve(k);
    centers.push_back(dist(gen));
    while (static_cast<int>(centers.size()) < k) {
        std::vector<double> dists(n, std::numeric_limits<double>::max());
        for (int i = 0; i < n; ++i) {
            for (int center : centers) {
                dists[i] = std::min(dists[i], SquaredDistance(data[i], data[center]));
            }
        }
        double sum = std::accumulate(dists.begin(), dists.end(), 0.0);
        if (sum == 0.0) {
            centers.push_back(dist(gen));
            continue;
        }
        std::uniform_real_distribution<double> prob_dist(0.0, sum);
        double target = prob_dist(gen);
        double cumulative = 0.0;
        for (int i = 0; i < n; ++i) {
            cumulative += dists[i];
            if (cumulative >= target) {
                centers.push_back(i);
                break;
            }
        }
    }
    result.centroids.resize(k, std::vector<float>(data.front().size(), 0.0f));
    for (int idx = 0; idx < k; ++idx) {
        result.centroids[idx] = data[centers[idx]];
    }
    result.assignments.assign(n, 0);

    for (int iter = 0; iter < max_iters; ++iter) {
        bool changed = false;
        for (int i = 0; i < n; ++i) {
            double best_dist = std::numeric_limits<double>::max();
            int best_cluster = 0;
            for (int c = 0; c < k; ++c) {
                double d = SquaredDistance(data[i], result.centroids[c]);
                if (d < best_dist) {
                    best_dist = d;
                    best_cluster = c;
                }
            }
            if (result.assignments[i] != best_cluster) {
                changed = true;
                result.assignments[i] = best_cluster;
            }
        }
        if (!changed && iter > 0) {
            break;
        }
        std::vector<std::vector<double>> sums(k, std::vector<double>(data.front().size(), 0.0));
        std::vector<int> counts(k, 0);
        for (int i = 0; i < n; ++i) {
            int cluster = result.assignments[i];
            ++counts[cluster];
            for (size_t dim = 0; dim < data[i].size(); ++dim) {
                sums[cluster][dim] += data[i][dim];
            }
        }
        for (int c = 0; c < k; ++c) {
            if (counts[c] == 0) {
                result.centroids[c] = data[dist(gen)];
            } else {
                for (size_t dim = 0; dim < result.centroids[c].size(); ++dim) {
                    result.centroids[c][dim] = static_cast<float>(sums[c][dim] / counts[c]);
                }
            }
        }
    }
    return result;
}

int DetermineClusterCount(size_t tool_count, size_t min_cluster, size_t max_clusters) {
    if (tool_count == 0) {
        return 0;
    }
    if (tool_count <= min_cluster) {
        return 1;
    }
    size_t tentative = static_cast<size_t>(std::ceil(static_cast<double>(tool_count) / min_cluster));
    tentative = std::min(tentative, max_clusters);
    tentative = std::max<size_t>(1, tentative);
    tentative = std::min(tentative, tool_count);
    return static_cast<int>(tentative);
}

std::unique_ptr<EmbeddingProvider> BuildEncoder(const Options& opts) {
    if (opts.offline) {
        return std::make_unique<SimulatedEncoder>(opts.offline_dim, opts.seed);
    }
    std::string api_key = opts.api_key;
    if (api_key.empty()) {
        const char* env = std::getenv("GOOGLE_API_KEY");
        if (env) {
            api_key = env;
        }
    }
    if (api_key.empty()) {
        throw std::runtime_error("Missing API key. Provide --api-key or set GOOGLE_API_KEY, or use --offline.");
    }
    return std::make_unique<GemmaEncoder>(api_key, opts.gemma_model, opts.batch_size, opts.seed);
}

int main(int argc, char** argv) {
    try {
        Options opts = ParseArgs(argc, argv);
        std::cout << "Loading tool docs from " << opts.tool_root << "...\n";
        auto tools = GetToolList(opts.tool_root, opts.max_tools);
        if (tools.empty()) {
            throw std::runtime_error("No tools discovered. Check --tool-root path.");
        }
        std::cout << "Collected " << tools.size() << " unique tools\n";
        std::cout << "Loading primary classes from " << opts.primary_class_file << "...\n";
        auto primary_classes = LoadPrimaryClasses(opts.primary_class_file);
        std::cout << "Loaded " << primary_classes.size() << " primary classes\n";

        std::vector<std::string> tool_texts;
        tool_texts.reserve(tools.size());
        for (const auto& tool : tools) {
            tool_texts.push_back(tool.SurfaceText());
        }
        std::vector<std::string> class_texts;
        class_texts.reserve(primary_classes.size());
        for (const auto& cls : primary_classes) {
            class_texts.push_back(cls.name + ". " + cls.description);
        }

        auto encoder = BuildEncoder(opts);
        std::cout << (opts.offline ? "Using offline synthetic embeddings\n" : "Requesting Gemma embeddings...\n");
        auto class_embeddings = encoder->Embed(class_texts);
        auto tool_embeddings = encoder->Embed(tool_texts);
        if (class_embeddings.size() != primary_classes.size()) {
            throw std::runtime_error("Failed to embed all primary classes");
        }
        if (tool_embeddings.size() != tools.size()) {
            throw std::runtime_error("Failed to embed all tools");
        }
        for (size_t i = 0; i < primary_classes.size(); ++i) {
            primary_classes[i].embedding = class_embeddings[i];
            primary_classes[i].normalized = Normalize(class_embeddings[i]);
        }
        std::vector<std::vector<float>> normalized_tools;
        normalized_tools.reserve(tool_embeddings.size());
        for (const auto& emb : tool_embeddings) {
            normalized_tools.push_back(Normalize(emb));
        }

        std::vector<int> tool_assignments(tools.size(), -1);
        std::vector<float> tool_similarities(tools.size(), -1.0f);
        std::vector<std::vector<size_t>> class_members(primary_classes.size());
        std::vector<size_t> unassigned;
        for (size_t i = 0; i < tools.size(); ++i) {
            float best_sim = -2.0f;
            int best_class = -1;
            for (size_t c = 0; c < primary_classes.size(); ++c) {
                float sim = CosineSim(normalized_tools[i], primary_classes[c].normalized);
                if (sim > best_sim) {
                    best_sim = sim;
                    best_class = static_cast<int>(c);
                }
            }
            if (best_class >= 0 && best_sim >= static_cast<float>(opts.similarity_threshold)) {
                tool_assignments[i] = best_class;
                tool_similarities[i] = best_sim;
                class_members[best_class].push_back(i);
            } else {
                unassigned.push_back(i);
            }
        }

        JsonObject root;
        root["tool_count"] = JsonValue(static_cast<double>(tools.size()));
        root["primary_class_count"] = JsonValue(static_cast<double>(primary_classes.size()));
        root["similarity_threshold"] = JsonValue(opts.similarity_threshold);
        root["offline_mode"] = JsonValue(opts.offline);
        root["embedding_dimension"] = JsonValue(static_cast<double>(encoder->Dim()));

        JsonArray class_entries;
        for (size_t c = 0; c < primary_classes.size(); ++c) {
            JsonObject cls_obj;
            cls_obj["id"] = primary_classes[c].id;
            cls_obj["name"] = primary_classes[c].name;
            cls_obj["description"] = primary_classes[c].description;
            const auto& members = class_members[c];
            cls_obj["tool_count"] = JsonValue(static_cast<double>(members.size()));
            if (!members.empty()) {
                double sim_sum = 0.0;
                double sim_max = -2.0;
                int k = DetermineClusterCount(members.size(), opts.min_cluster_size, opts.max_clusters);
                std::vector<std::vector<float>> class_vectors;
                class_vectors.reserve(members.size());
                for (size_t idx : members) {
                    class_vectors.push_back(normalized_tools[idx]);
                }
                ClusterResult cluster = RunKMeans(class_vectors, k, opts.seed);
                JsonArray cluster_entries;
                std::unordered_map<int, std::vector<size_t>> cluster_to_members;
                for (size_t idx = 0; idx < members.size(); ++idx) {
                    int assignment = cluster.assignments.empty() ? 0 : cluster.assignments[idx];
                    cluster_to_members[assignment].push_back(members[idx]);
                }
                for (const auto& [cluster_id, mlist] : cluster_to_members) {
                    JsonObject cluster_obj;
                    cluster_obj["cluster_id"] = JsonValue(static_cast<double>(cluster_id));
                    cluster_obj["size"] = JsonValue(static_cast<double>(mlist.size()));
                    JsonArray member_arr;
                    double cluster_sim_sum = 0.0;
                    std::vector<std::pair<float, std::string>> ranked;
                    for (size_t tool_idx : mlist) {
                        float sim = tool_similarities[tool_idx];
                        cluster_sim_sum += sim;
                        sim_sum += sim;
                        sim_max = std::max(sim_max, static_cast<double>(sim));
                        JsonObject entry;
                        entry["name"] = tools[tool_idx].name;
                        entry["source"] = tools[tool_idx].source;
                        entry["similarity"] = JsonValue(sim);
                        entry["description"] = tools[tool_idx].description;
                        member_arr.emplace_back(entry);
                        ranked.push_back({sim, tools[tool_idx].name});
                    }
                    cluster_obj["avg_similarity"] =
                        JsonValue(cluster_sim_sum / std::max<double>(1.0, static_cast<double>(mlist.size())));
                    JsonArray centroid_preview;
                    if (cluster_id >= 0 && cluster_id < static_cast<int>(cluster.centroids.size())) {
                        const auto& centroid = cluster.centroids[cluster_id];
                        size_t preview_len = std::min<size_t>(8, centroid.size());
                        for (size_t i = 0; i < preview_len; ++i) {
                            centroid_preview.emplace_back(JsonValue(centroid[i]));
                        }
                    }
                    cluster_obj["centroid_preview"] = centroid_preview;
                    std::sort(ranked.begin(), ranked.end(),
                              [](const auto& lhs, const auto& rhs) { return lhs.first > rhs.first; });
                    JsonArray representatives;
                    for (size_t i = 0; i < std::min<size_t>(3, ranked.size()); ++i) {
                        JsonObject rep;
                        rep["name"] = ranked[i].second;
                        rep["similarity"] = JsonValue(ranked[i].first);
                        representatives.emplace_back(rep);
                    }
                    cluster_obj["representative_tools"] = representatives;
                    cluster_obj["members"] = member_arr;
                    cluster_entries.emplace_back(cluster_obj);
                }
                cls_obj["clusters"] = cluster_entries;
                cls_obj["avg_similarity"] = JsonValue(sim_sum / std::max<double>(1.0, static_cast<double>(members.size())));
                cls_obj["max_similarity"] = JsonValue(sim_max);
            } else {
                cls_obj["clusters"] = JsonArray{};
                cls_obj["avg_similarity"] = JsonValue(0.0);
                cls_obj["max_similarity"] = JsonValue(-1.0);
            }
            class_entries.emplace_back(cls_obj);
        }
        root["classes"] = class_entries;

        if (!unassigned.empty()) {
            JsonArray unassigned_entries;
            for (size_t idx : unassigned) {
                JsonObject entry;
                entry["name"] = tools[idx].name;
                entry["source"] = tools[idx].source;
                entry["description"] = tools[idx].description;
                entry["best_similarity"] = JsonValue(tool_similarities[idx]);
                unassigned_entries.emplace_back(entry);
            }
            root["unassigned"] = unassigned_entries;
        }

        JsonValue output(root);
        if (!opts.output_file.parent_path().empty()) {
            fs::create_directories(opts.output_file.parent_path());
        }
        std::ofstream out(opts.output_file);
        if (!out) {
            throw std::runtime_error("Failed to open output file: " + opts.output_file.string());
        }
        out << output.Dump(2) << std::endl;
        std::cout << "Wrote clustering report to " << opts.output_file << '\n';
        if (!unassigned.empty()) {
            std::cout << "Unassigned tools due to low similarity: " << unassigned.size() << '\n';
        }
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << '\n';
        return 1;
    }
    return 0;
}

