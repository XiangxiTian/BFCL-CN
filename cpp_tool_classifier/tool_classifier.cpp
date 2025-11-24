#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <map>
#include <memory>
#include <sstream>
#include <random>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

// Structure to represent a tool
struct Tool {
    std::string id;
    std::string name;
    std::string description;
    json parameters;
    std::vector<float> embedding;
};

// Structure to represent a primary class
struct PrimaryClass {
    std::string name;
    std::string description;
    std::vector<float> embedding;
    std::vector<Tool> tools;
};

// Structure for cluster
struct Cluster {
    int id;
    std::vector<float> centroid;
    std::vector<Tool> tools;
};

class ToolClassifier {
private:
    std::vector<Tool> tools;
    std::vector<PrimaryClass> primaryClasses;
    int embeddingDim = 768; // Default embedding dimension

    // Calculate cosine similarity between two vectors
    float cosineSimilarity(const std::vector<float>& a, const std::vector<float>& b) {
        if (a.size() != b.size()) {
            std::cerr << "Error: Vector dimensions don't match!" << std::endl;
            return 0.0f;
        }
        
        float dotProduct = 0.0f;
        float normA = 0.0f;
        float normB = 0.0f;
        
        for (size_t i = 0; i < a.size(); ++i) {
            dotProduct += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }
        
        normA = std::sqrt(normA);
        normB = std::sqrt(normB);
        
        if (normA == 0.0f || normB == 0.0f) {
            return 0.0f;
        }
        
        return dotProduct / (normA * normB);
    }

    // Generate mock embedding (replace with actual Gemma 300m encoding)
    std::vector<float> generateEmbedding(const std::string& text) {
        // This is a placeholder. In production, you would call Gemma 300m model
        // For now, we use a simple hash-based approach to generate deterministic embeddings
        std::vector<float> embedding(embeddingDim, 0.0f);
        
        std::hash<std::string> hasher;
        size_t hash = hasher(text);
        
        std::mt19937 gen(hash);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        
        for (int i = 0; i < embeddingDim; ++i) {
            embedding[i] = dist(gen);
        }
        
        // Normalize the embedding
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

    // K-means clustering implementation
    std::vector<Cluster> kMeansClustering(const std::vector<Tool>& tools, int k, int maxIterations = 100) {
        if (tools.empty() || k <= 0 || k > tools.size()) {
            std::cerr << "Invalid clustering parameters!" << std::endl;
            return {};
        }

        std::vector<Cluster> clusters(k);
        std::random_device rd;
        std::mt19937 gen(rd());
        
        // Initialize centroids randomly from tools
        std::vector<int> indices(tools.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), gen);
        
        for (int i = 0; i < k; ++i) {
            clusters[i].id = i;
            clusters[i].centroid = tools[indices[i]].embedding;
        }

        // K-means iterations
        for (int iter = 0; iter < maxIterations; ++iter) {
            // Clear cluster assignments
            for (auto& cluster : clusters) {
                cluster.tools.clear();
            }

            // Assign each tool to nearest cluster
            for (const auto& tool : tools) {
                float maxSim = -1.0f;
                int bestCluster = 0;
                
                for (int i = 0; i < k; ++i) {
                    float sim = cosineSimilarity(tool.embedding, clusters[i].centroid);
                    if (sim > maxSim) {
                        maxSim = sim;
                        bestCluster = i;
                    }
                }
                
                clusters[bestCluster].tools.push_back(tool);
            }

            // Update centroids
            bool changed = false;
            for (auto& cluster : clusters) {
                if (cluster.tools.empty()) continue;
                
                std::vector<float> newCentroid(embeddingDim, 0.0f);
                
                for (const auto& tool : cluster.tools) {
                    for (size_t i = 0; i < embeddingDim; ++i) {
                        newCentroid[i] += tool.embedding[i];
                    }
                }
                
                for (float& val : newCentroid) {
                    val /= cluster.tools.size();
                }
                
                // Normalize centroid
                float norm = 0.0f;
                for (float val : newCentroid) {
                    norm += val * val;
                }
                norm = std::sqrt(norm);
                
                if (norm > 0.0f) {
                    for (float& val : newCentroid) {
                        val /= norm;
                    }
                }
                
                // Check if centroid changed significantly
                float centroidChange = 0.0f;
                for (size_t i = 0; i < embeddingDim; ++i) {
                    centroidChange += std::abs(newCentroid[i] - cluster.centroid[i]);
                }
                
                if (centroidChange > 0.001f) {
                    changed = true;
                }
                
                cluster.centroid = newCentroid;
            }

            if (!changed) {
                std::cout << "Converged after " << iter + 1 << " iterations." << std::endl;
                break;
            }
        }

        return clusters;
    }

public:
    // Load tools from JSON file
    void loadToolsFromJson(const std::string& filepath) {
        std::ifstream file(filepath);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file " << filepath << std::endl;
            return;
        }

        std::string line;
        while (std::getline(file, line)) {
            try {
                json j = json::parse(line);
                
                if (j.contains("function") && j["function"].is_array()) {
                    for (const auto& func : j["function"]) {
                        Tool tool;
                        tool.id = j.value("id", "unknown");
                        tool.name = func.value("name", "");
                        tool.description = func.value("description", "");
                        tool.parameters = func.value("parameters", json::object());
                        
                        // Generate embedding from name and description
                        std::string text = tool.name + " " + tool.description;
                        tool.embedding = generateEmbedding(text);
                        
                        tools.push_back(tool);
                    }
                }
            } catch (const json::exception& e) {
                std::cerr << "JSON parse error: " << e.what() << std::endl;
            }
        }

        file.close();
        std::cout << "Loaded " << tools.size() << " tools from " << filepath << std::endl;
    }

    // Initialize primary classes with predefined categories
    void initializePrimaryClasses() {
        std::vector<std::pair<std::string, std::string>> classDefinitions = {
            {"Mathematics", "Mathematical operations including geometry, algebra, calculus, and numerical computations"},
            {"Physics", "Physics-related calculations including mechanics, electromagnetism, thermodynamics"},
            {"Travel & Navigation", "Travel planning, route finding, location services, and navigation"},
            {"Data & Numbers", "Number theory, factorization, divisors, and numerical analysis"},
            {"API & Web Services", "REST API calls, web services, and external data retrieval"}
        };

        for (const auto& [name, description] : classDefinitions) {
            PrimaryClass pc;
            pc.name = name;
            pc.description = description;
            pc.embedding = generateEmbedding(name + " " + description);
            primaryClasses.push_back(pc);
        }

        std::cout << "Initialized " << primaryClasses.size() << " primary classes." << std::endl;
    }

    // Classify tools into primary classes based on similarity
    void classifyTools() {
        std::cout << "\nClassifying tools into primary classes..." << std::endl;
        
        for (const auto& tool : tools) {
            float maxSim = -1.0f;
            int bestClass = 0;
            
            for (size_t i = 0; i < primaryClasses.size(); ++i) {
                float sim = cosineSimilarity(tool.embedding, primaryClasses[i].embedding);
                if (sim > maxSim) {
                    maxSim = sim;
                    bestClass = i;
                }
            }
            
            primaryClasses[bestClass].tools.push_back(tool);
        }

        // Print classification results
        for (const auto& pc : primaryClasses) {
            std::cout << "Class '" << pc.name << "': " << pc.tools.size() << " tools" << std::endl;
        }
    }

    // Cluster tools within each primary class
    void clusterWithinClasses(int clustersPerClass = 3) {
        std::cout << "\nClustering tools within each primary class..." << std::endl;
        
        for (auto& pc : primaryClasses) {
            if (pc.tools.empty()) {
                std::cout << "Class '" << pc.name << "': No tools to cluster." << std::endl;
                continue;
            }

            // Adjust number of clusters if there are fewer tools
            int k = std::min(clustersPerClass, static_cast<int>(pc.tools.size()));
            
            std::cout << "\nClass '" << pc.name << "' - Creating " << k << " clusters:" << std::endl;
            
            auto clusters = kMeansClustering(pc.tools, k);
            
            for (const auto& cluster : clusters) {
                std::cout << "  Cluster " << cluster.id << " (" << cluster.tools.size() << " tools):" << std::endl;
                for (size_t i = 0; i < std::min(size_t(5), cluster.tools.size()); ++i) {
                    std::cout << "    - " << cluster.tools[i].name << std::endl;
                }
                if (cluster.tools.size() > 5) {
                    std::cout << "    ... and " << (cluster.tools.size() - 5) << " more" << std::endl;
                }
            }
        }
    }

    // Save results to JSON
    void saveResults(const std::string& outputPath, int clustersPerClass = 3) {
        json output;
        output["summary"] = {
            {"total_tools", tools.size()},
            {"primary_classes", primaryClasses.size()},
            {"clusters_per_class", clustersPerClass}
        };

        json classesArray = json::array();
        
        for (auto& pc : primaryClasses) {
            json classObj;
            classObj["name"] = pc.name;
            classObj["description"] = pc.description;
            classObj["tool_count"] = pc.tools.size();
            
            if (!pc.tools.empty()) {
                int k = std::min(clustersPerClass, static_cast<int>(pc.tools.size()));
                auto clusters = kMeansClustering(pc.tools, k);
                
                json clustersArray = json::array();
                for (const auto& cluster : clusters) {
                    json clusterObj;
                    clusterObj["cluster_id"] = cluster.id;
                    clusterObj["tool_count"] = cluster.tools.size();
                    
                    json toolsArray = json::array();
                    for (const auto& tool : cluster.tools) {
                        json toolObj;
                        toolObj["id"] = tool.id;
                        toolObj["name"] = tool.name;
                        toolObj["description"] = tool.description;
                        toolsArray.push_back(toolObj);
                    }
                    clusterObj["tools"] = toolsArray;
                    clustersArray.push_back(clusterObj);
                }
                classObj["clusters"] = clustersArray;
            }
            
            classesArray.push_back(classObj);
        }
        
        output["classes"] = classesArray;

        std::ofstream outFile(outputPath);
        if (outFile.is_open()) {
            outFile << output.dump(2);
            outFile.close();
            std::cout << "\nResults saved to " << outputPath << std::endl;
        } else {
            std::cerr << "Error: Could not write to " << outputPath << std::endl;
        }
    }

    // Get statistics
    void printStatistics() {
        std::cout << "\n=== Classification & Clustering Statistics ===" << std::endl;
        std::cout << "Total tools loaded: " << tools.size() << std::endl;
        std::cout << "Primary classes: " << primaryClasses.size() << std::endl;
        std::cout << "\nDistribution across classes:" << std::endl;
        
        for (const auto& pc : primaryClasses) {
            float percentage = tools.empty() ? 0.0f : (100.0f * pc.tools.size() / tools.size());
            std::cout << "  " << pc.name << ": " << pc.tools.size() 
                      << " tools (" << std::fixed << std::setprecision(1) << percentage << "%)" << std::endl;
        }
    }
};

int main(int argc, char* argv[]) {
    std::cout << "=== Tool Classification and Clustering System ===" << std::endl;
    std::cout << "Using Gemma 300m embeddings (mock implementation)" << std::endl;
    std::cout << std::endl;

    // Parse command line arguments
    std::string inputPath = "../data/BFCL_v3_simple.json";
    std::string outputPath = "classification_results.json";
    int clustersPerClass = 3;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--input" && i + 1 < argc) {
            inputPath = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            outputPath = argv[++i];
        } else if (arg == "--clusters" && i + 1 < argc) {
            clustersPerClass = std::stoi(argv[++i]);
        } else if (arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  --input <path>      Input JSON file path (default: ../data/BFCL_v3_simple.json)" << std::endl;
            std::cout << "  --output <path>     Output JSON file path (default: classification_results.json)" << std::endl;
            std::cout << "  --clusters <num>    Number of clusters per class (default: 3)" << std::endl;
            std::cout << "  --help              Show this help message" << std::endl;
            return 0;
        }
    }

    ToolClassifier classifier;

    // Step 1: Load tools from JSON
    std::cout << "Step 1: Loading tools from JSON..." << std::endl;
    classifier.loadToolsFromJson(inputPath);

    // Step 2: Initialize primary classes
    std::cout << "\nStep 2: Initializing primary classes..." << std::endl;
    classifier.initializePrimaryClasses();

    // Step 3: Classify tools into primary classes
    std::cout << "\nStep 3: Classifying tools..." << std::endl;
    classifier.classifyTools();

    // Step 4: Cluster within each class
    std::cout << "\nStep 4: Clustering within classes..." << std::endl;
    classifier.clusterWithinClasses(clustersPerClass);

    // Step 5: Print statistics
    classifier.printStatistics();

    // Step 6: Save results
    std::cout << "\nStep 5: Saving results..." << std::endl;
    classifier.saveResults(outputPath, clustersPerClass);

    std::cout << "\n=== Processing Complete ===" << std::endl;
    
    return 0;
}
