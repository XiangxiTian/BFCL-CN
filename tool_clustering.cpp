#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <memory>
#include <random>
#include <limits>

// JSON parsing (using simple approach - in production, use nlohmann/json)
#include <regex>

// For ONNX Runtime (Google Gemma model)
#ifdef USE_ONNX
#include <onnxruntime_cxx_api.h>
#endif

// Tool structure
struct Tool {
    std::string name;
    std::string description;
    std::map<std::string, std::string> parameters;
    
    // Embedding vector (from Gemma model)
    std::vector<float> embedding;
    
    // Classification info
    std::string primary_class;
    int cluster_id;
    
    Tool(const std::string& n, const std::string& d) 
        : name(n), description(d), cluster_id(-1) {}
};

// Primary class structure
struct PrimaryClass {
    std::string name;
    std::string description;
    std::vector<float> embedding;
    
    PrimaryClass(const std::string& n, const std::string& d)
        : name(n), description(d) {}
};

// Simple JSON parser (handles both single objects and arrays)
class SimpleJSONParser {
public:
    static std::vector<Tool> parseToolsFromJSON(const std::string& jsonContent) {
        std::vector<Tool> tools;
        
        // Handle JSONL format (one JSON object per line) or single JSON array
        std::istringstream stream(jsonContent);
        std::string line;
        
        while (std::getline(stream, line)) {
            if (line.empty() || line[0] != '{') continue;
            
            // Extract function array from JSON object
            std::vector<Tool> tools_from_line = parseJSONObject(line);
            tools.insert(tools.end(), tools_from_line.begin(), tools_from_line.end());
        }
        
        return tools;
    }
    
private:
    static std::vector<Tool> parseJSONObject(const std::string& json) {
        std::vector<Tool> tools;
        
        // Find "function" array
        size_t func_pos = json.find("\"function\"");
        if (func_pos == std::string::npos) {
            return tools;
        }
        
        // Find the array start
        size_t array_start = json.find('[', func_pos);
        if (array_start == std::string::npos) {
            return tools;
        }
        
        // Extract function objects from array
        int depth = 0;
        size_t obj_start = array_start;
        bool in_string = false;
        bool escape_next = false;
        
        for (size_t i = array_start + 1; i < json.length(); ++i) {
            if (escape_next) {
                escape_next = false;
                continue;
            }
            
            char c = json[i];
            
            if (c == '\\') {
                escape_next = true;
                continue;
            }
            
            if (c == '"') {
                in_string = !in_string;
                continue;
            }
            
            if (in_string) continue;
            
            if (c == '{') {
                if (depth == 0) {
                    obj_start = i;
                }
                depth++;
            } else if (c == '}') {
                depth--;
                if (depth == 0) {
                    // Extract function object
                    std::string func_obj = json.substr(obj_start, i - obj_start + 1);
                    Tool tool = parseFunctionObject(func_obj);
                    if (!tool.name.empty()) {
                        tools.push_back(tool);
                    }
                }
            } else if (c == ']' && depth == 0) {
                break;
            }
        }
        
        return tools;
    }
    
    static Tool parseFunctionObject(const std::string& obj_json) {
        Tool tool("", "");
        
        // Extract name - use escaped quotes in raw string
        std::regex name_regex(R"(\"name\"\s*:\s*\"([^\"]+)\")");
        std::smatch name_match;
        if (std::regex_search(obj_json, name_match, name_regex)) {
            tool.name = name_match[1].str();
        }
        
        // Extract description - handle escaped characters
        std::regex desc_regex(R"(\"description\"\s*:\s*\"((?:[^\"\\]|\\.)*)\")");
        std::smatch desc_match;
        if (std::regex_search(obj_json, desc_match, desc_regex)) {
            tool.description = unescapeString(desc_match[1].str());
        }
        
        return tool;
    }
    
    static std::string unescapeString(const std::string& str) {
        std::string result;
        for (size_t i = 0; i < str.length(); ++i) {
            if (str[i] == '\\' && i + 1 < str.length()) {
                switch (str[i + 1]) {
                    case 'n': result += '\n'; ++i; break;
                    case 't': result += '\t'; ++i; break;
                    case 'r': result += '\r'; ++i; break;
                    case '\\': result += '\\'; ++i; break;
                    case '"': result += '"'; ++i; break;
                    case '/': result += '/'; ++i; break;
                    case 'u': 
                        // Handle unicode escape (simplified)
                        if (i + 5 < str.length()) {
                            result += '?'; // Placeholder for unicode
                            i += 5;
                        }
                        break;
                    default: result += str[i]; break;
                }
            } else {
                result += str[i];
            }
        }
        return result;
    }
};

// Embedding model interface (for Google Gemma 300M)
class EmbeddingModel {
public:
    virtual ~EmbeddingModel() = default;
    virtual std::vector<float> encode(const std::string& text) = 0;
    virtual bool initialize() { return true; }
};

// Mock implementation (replace with actual ONNX Runtime integration)
class GemmaEmbeddingModel : public EmbeddingModel {
private:
    size_t embedding_dim_;
    bool initialized_;
    
public:
    GemmaEmbeddingModel(size_t dim = 768) : embedding_dim_(dim), initialized_(false) {}
    
    bool initialize() override {
        // TODO: Initialize ONNX Runtime session with Gemma 300M model
        // For now, we'll use a mock implementation
        initialized_ = true;
        std::cout << "[INFO] Initializing Gemma 300M embedding model..." << std::endl;
        return initialized_;
    }
    
    std::vector<float> encode(const std::string& text) override {
        if (!initialized_) {
            initialize();
        }
        
        // Mock embedding generation (replace with actual model inference)
        // In production, this would call ONNX Runtime to run Gemma model
        std::vector<float> embedding(embedding_dim_);
        
        // Simple hash-based mock embedding (replace with actual model)
        std::hash<std::string> hasher;
        size_t hash = hasher(text);
        std::mt19937 gen(hash);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        
        for (size_t i = 0; i < embedding_dim_; ++i) {
            embedding[i] = dist(gen);
        }
        
        // Normalize to unit vector
        float norm = 0.0f;
        for (float val : embedding) {
            norm += val * val;
        }
        norm = std::sqrt(norm);
        if (norm > 0.0f) {
            for (float& val : embedding) {
                val /= norm;
            }
        }
        
        return embedding;
    }
};

// Similarity calculator
class SimilarityCalculator {
public:
    static float cosineSimilarity(const std::vector<float>& vec1, 
                                  const std::vector<float>& vec2) {
        if (vec1.size() != vec2.size()) {
            return 0.0f;
        }
        
        float dot_product = 0.0f;
        float norm1 = 0.0f;
        float norm2 = 0.0f;
        
        for (size_t i = 0; i < vec1.size(); ++i) {
            dot_product += vec1[i] * vec2[i];
            norm1 += vec1[i] * vec1[i];
            norm2 += vec2[i] * vec2[i];
        }
        
        norm1 = std::sqrt(norm1);
        norm2 = std::sqrt(norm2);
        
        if (norm1 == 0.0f || norm2 == 0.0f) {
            return 0.0f;
        }
        
        return dot_product / (norm1 * norm2);
    }
};

// K-means clustering
class KMeansClusterer {
private:
    int k_;
    int max_iterations_;
    float tolerance_;
    
public:
    KMeansClusterer(int k, int max_iter = 100, float tol = 1e-4)
        : k_(k), max_iterations_(max_iter), tolerance_(tol) {}
    
    std::vector<int> cluster(const std::vector<std::vector<float>>& embeddings) {
        if (embeddings.empty()) {
            return {};
        }
        
        size_t n = embeddings.size();
        size_t dim = embeddings[0].size();
        
        // Initialize centroids randomly
        std::vector<std::vector<float>> centroids(k_);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<size_t> dist(0, n - 1);
        
        for (int i = 0; i < k_; ++i) {
            centroids[i] = embeddings[dist(gen)];
        }
        
        std::vector<int> assignments(n, -1);
        std::vector<int> prev_assignments(n, -1);
        
        for (int iter = 0; iter < max_iterations_; ++iter) {
            // Assign points to nearest centroid
            for (size_t i = 0; i < n; ++i) {
                float max_sim = -1.0f;
                int best_cluster = 0;
                
                for (int j = 0; j < k_; ++j) {
                    float sim = SimilarityCalculator::cosineSimilarity(
                        embeddings[i], centroids[j]);
                    if (sim > max_sim) {
                        max_sim = sim;
                        best_cluster = j;
                    }
                }
                
                assignments[i] = best_cluster;
            }
            
            // Check convergence
            bool converged = true;
            for (size_t i = 0; i < n; ++i) {
                if (assignments[i] != prev_assignments[i]) {
                    converged = false;
                    break;
                }
            }
            
            if (converged) {
                break;
            }
            
            prev_assignments = assignments;
            
            // Update centroids
            std::vector<std::vector<float>> new_centroids(k_, 
                std::vector<float>(dim, 0.0f));
            std::vector<int> counts(k_, 0);
            
            for (size_t i = 0; i < n; ++i) {
                int cluster = assignments[i];
                for (size_t j = 0; j < dim; ++j) {
                    new_centroids[cluster][j] += embeddings[i][j];
                }
                counts[cluster]++;
            }
            
            for (int i = 0; i < k_; ++i) {
                if (counts[i] > 0) {
                    for (size_t j = 0; j < dim; ++j) {
                        new_centroids[i][j] /= counts[i];
                    }
                    // Normalize
                    float norm = 0.0f;
                    for (float val : new_centroids[i]) {
                        norm += val * val;
                    }
                    norm = std::sqrt(norm);
                    if (norm > 0.0f) {
                        for (float& val : new_centroids[i]) {
                            val /= norm;
                        }
                    }
                } else {
                    new_centroids[i] = centroids[i];
                }
            }
            
            centroids = new_centroids;
        }
        
        return assignments;
    }
};

// Tool classifier and clusterer
class ToolClassifier {
private:
    std::shared_ptr<EmbeddingModel> model_;
    std::vector<PrimaryClass> primary_classes_;
    
public:
    ToolClassifier(std::shared_ptr<EmbeddingModel> model)
        : model_(model) {}
    
    void addPrimaryClass(const std::string& name, const std::string& description) {
        PrimaryClass pc(name, description);
        // Encode primary class description
        pc.embedding = model_->encode(description);
        primary_classes_.push_back(pc);
    }
    
    void classifyTools(std::vector<Tool>& tools) {
        std::cout << "[INFO] Encoding " << tools.size() << " tools..." << std::endl;
        
        // Encode all tools
        for (auto& tool : tools) {
            std::string text = tool.name + " " + tool.description;
            tool.embedding = model_->encode(text);
        }
        
        std::cout << "[INFO] Classifying tools into primary classes..." << std::endl;
        
        // Classify each tool to primary class
        for (auto& tool : tools) {
            float max_similarity = -1.0f;
            int best_class_idx = 0;
            
            for (size_t i = 0; i < primary_classes_.size(); ++i) {
                float similarity = SimilarityCalculator::cosineSimilarity(
                    tool.embedding, primary_classes_[i].embedding);
                
                if (similarity > max_similarity) {
                    max_similarity = similarity;
                    best_class_idx = i;
                }
            }
            
            tool.primary_class = primary_classes_[best_class_idx].name;
        }
    }
    
    void clusterToolsInClasses(std::vector<Tool>& tools, int clusters_per_class = 3) {
        std::cout << "[INFO] Clustering tools within each primary class..." << std::endl;
        
        // Group tools by primary class
        std::map<std::string, std::vector<size_t>> class_groups;
        for (size_t i = 0; i < tools.size(); ++i) {
            class_groups[tools[i].primary_class].push_back(i);
        }
        
        // Cluster within each class
        for (const auto& [class_name, indices] : class_groups) {
            if (static_cast<int>(indices.size()) <= clusters_per_class) {
                // Too few tools, assign each to its own cluster
                for (size_t idx : indices) {
                    tools[idx].cluster_id = idx;
                }
                continue;
            }
            
            // Extract embeddings for this class
            std::vector<std::vector<float>> class_embeddings;
            for (size_t idx : indices) {
                class_embeddings.push_back(tools[idx].embedding);
            }
            
            // Determine number of clusters (min of clusters_per_class and size)
            int k = std::min(clusters_per_class, static_cast<int>(indices.size()));
            
            // Perform clustering
            KMeansClusterer clusterer(k);
            std::vector<int> cluster_assignments = clusterer.cluster(class_embeddings);
            
            // Assign cluster IDs
            for (size_t i = 0; i < indices.size(); ++i) {
                tools[indices[i]].cluster_id = cluster_assignments[i];
            }
            
            std::cout << "[INFO] Class '" << class_name << "': " 
                      << indices.size() << " tools clustered into " 
                      << k << " clusters" << std::endl;
        }
    }
    
    void printResults(const std::vector<Tool>& tools) {
        std::cout << "\n=== Classification and Clustering Results ===\n" << std::endl;
        
        // Group by class and cluster
        std::map<std::string, std::map<int, std::vector<const Tool*>>> results;
        
        for (const auto& tool : tools) {
            results[tool.primary_class][tool.cluster_id].push_back(&tool);
        }
        
        for (const auto& [class_name, clusters] : results) {
            std::cout << "\nPrimary Class: " << class_name << std::endl;
            std::cout << "----------------------------------------" << std::endl;
            
            for (const auto& [cluster_id, cluster_tools] : clusters) {
                std::cout << "\n  Cluster " << cluster_id << " (" 
                          << cluster_tools.size() << " tools):" << std::endl;
                
                for (const auto& tool : cluster_tools) {
                    std::cout << "    - " << tool->name << std::endl;
                    std::cout << "      " << tool->description.substr(0, 80) 
                              << (tool->description.length() > 80 ? "..." : "") 
                              << std::endl;
                }
            }
        }
    }
};

// Function to get tools (placeholder - replace with actual implementation)
std::vector<Tool> getToolList(const std::string& json_file_path) {
    std::ifstream file(json_file_path);
    if (!file.is_open()) {
        std::cerr << "[ERROR] Cannot open file: " << json_file_path << std::endl;
        return {};
    }
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string json_content = buffer.str();
    
    return SimpleJSONParser::parseToolsFromJSON(json_content);
}

int main(int argc, char* argv[]) {
    std::cout << "=== Tool Classification and Clustering System ===" << std::endl;
    std::cout << "Using Google Gemma 300M for embeddings\n" << std::endl;
    
    // Initialize embedding model
    auto model = std::make_shared<GemmaEmbeddingModel>(768);
    if (!model->initialize()) {
        std::cerr << "[ERROR] Failed to initialize embedding model" << std::endl;
        return 1;
    }
    
    // Get tools (from JSON file or API)
    std::string json_file = (argc > 1) ? argv[1] : "data/BFCL_v3_simple.json";
    std::vector<Tool> tools = getToolList(json_file);
    
    if (tools.empty()) {
        std::cerr << "[ERROR] No tools found. Please provide a valid JSON file." << std::endl;
        return 1;
    }
    
    std::cout << "[INFO] Loaded " << tools.size() << " tools" << std::endl;
    
    // Define primary classes
    ToolClassifier classifier(model);
    
    // Example primary classes (customize as needed)
    classifier.addPrimaryClass("Mathematics", 
        "Mathematical operations, calculations, algebra, geometry, calculus");
    classifier.addPrimaryClass("Physics", 
        "Physical calculations, mechanics, electromagnetism, thermodynamics");
    classifier.addPrimaryClass("Travel", 
        "Travel planning, directions, routes, itineraries, locations");
    classifier.addPrimaryClass("Data", 
        "Data manipulation, analysis, storage, retrieval");
    classifier.addPrimaryClass("Communication", 
        "Messaging, posting, social media, user interactions");
    
    // Classify tools into primary classes
    classifier.classifyTools(tools);
    
    // Cluster tools within each primary class
    int clusters_per_class = (argc > 2) ? std::stoi(argv[2]) : 3;
    classifier.clusterToolsInClasses(tools, clusters_per_class);
    
    // Print results
    classifier.printResults(tools);
    
    // Optional: Save results to file
    if (argc > 3) {
        std::ofstream out(argv[3]);
        out << "Tool Name,Primary Class,Cluster ID,Description\n";
        for (const auto& tool : tools) {
            out << "\"" << tool.name << "\","
                << "\"" << tool.primary_class << "\","
                << tool.cluster_id << ","
                << "\"" << tool.description << "\"\n";
        }
        std::cout << "\n[INFO] Results saved to: " << argv[3] << std::endl;
    }
    
    return 0;
}
