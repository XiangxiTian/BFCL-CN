#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <cmath>
#include <algorithm>
#include <random>
#include <fstream>
#include <sstream>
#include <memory>
#include <numeric>
#include <limits>
#include <iomanip>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

// Tool structure to hold tool information
struct Tool {
    std::string name;
    std::string description;
    std::map<std::string, std::string> parameters;
    std::vector<float> embedding;
    int primaryClass = -1;
    int subCluster = -1;
};

// Primary class structure
struct PrimaryClass {
    std::string name;
    std::string description;
    std::vector<float> embedding;
    std::vector<Tool*> tools;
};

// Configuration structure
struct Config {
    std::string modelPath;
    std::vector<std::string> primaryClassNames;
    std::vector<std::string> primaryClassDescriptions;
    int numSubClusters = 3;
    float similarityThreshold = 0.6f;
    int maxIterations = 100;
    float convergenceThreshold = 0.001f;
    int embeddingDimension = 256;
};

// Simple text encoder using hash-based embeddings (replacement for Gemma)
class SimpleTextEncoder {
private:
    int embeddingDim;
    std::mt19937 gen;
    
public:
    SimpleTextEncoder(int dim = 256) : embeddingDim(dim), gen(42) {}
    
    // Generate deterministic embedding based on text
    std::vector<float> encode(const std::string& text) {
        std::vector<float> embedding(embeddingDim, 0.0f);
        
        // Use hash of words to create pseudo-embeddings
        std::hash<std::string> hasher;
        std::istringstream iss(text);
        std::string word;
        int wordCount = 0;
        
        while (iss >> word) {
            // Convert to lowercase
            std::transform(word.begin(), word.end(), word.begin(), ::tolower);
            
            size_t wordHash = hasher(word);
            gen.seed(wordHash);
            std::normal_distribution<float> dist(0.0f, 1.0f);
            
            // Add word contribution to embedding
            for (int i = 0; i < embeddingDim; ++i) {
                embedding[i] += dist(gen);
            }
            wordCount++;
        }
        
        // Average and normalize
        if (wordCount > 0) {
            for (float& val : embedding) {
                val /= wordCount;
            }
        }
        
        // L2 normalization
        float norm = 0.0f;
        for (float val : embedding) {
            norm += val * val;
        }
        norm = std::sqrt(norm);
        
        if (norm > 0) {
            for (float& val : embedding) {
                val /= norm;
            }
        }
        
        return embedding;
    }
};

// Similarity calculation functions
class SimilarityCalculator {
public:
    // Cosine similarity between two vectors
    static float cosineSimilarity(const std::vector<float>& a, const std::vector<float>& b) {
        if (a.size() != b.size()) {
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
    
    // Euclidean distance
    static float euclideanDistance(const std::vector<float>& a, const std::vector<float>& b) {
        if (a.size() != b.size()) {
            return std::numeric_limits<float>::max();
        }
        
        float sum = 0.0f;
        for (size_t i = 0; i < a.size(); ++i) {
            float diff = a[i] - b[i];
            sum += diff * diff;
        }
        
        return std::sqrt(sum);
    }
};

// K-Means clustering implementation
class KMeansClustering {
private:
    int k;
    int maxIterations;
    float convergenceThreshold;
    std::vector<std::vector<float>> centroids;
    
public:
    KMeansClustering(int numClusters, int maxIter = 100, float convThreshold = 0.001f)
        : k(numClusters), maxIterations(maxIter), convergenceThreshold(convThreshold) {}
    
    // Initialize centroids using K-Means++
    void initializeCentroids(const std::vector<std::vector<float>>& data) {
        if (data.empty()) return;
        
        std::mt19937 gen(42);
        centroids.clear();
        
        // Choose first centroid randomly
        std::uniform_int_distribution<> dis(0, data.size() - 1);
        centroids.push_back(data[dis(gen)]);
        
        // Choose remaining centroids using K-Means++
        for (int i = 1; i < k && i < static_cast<int>(data.size()); ++i) {
            std::vector<float> distances(data.size());
            float totalDistance = 0.0f;
            
            // Calculate distance to nearest centroid for each point
            for (size_t j = 0; j < data.size(); ++j) {
                float minDist = std::numeric_limits<float>::max();
                for (const auto& centroid : centroids) {
                    float dist = SimilarityCalculator::euclideanDistance(data[j], centroid);
                    minDist = std::min(minDist, dist);
                }
                distances[j] = minDist * minDist;
                totalDistance += distances[j];
            }
            
            // Choose next centroid with probability proportional to squared distance
            if (totalDistance > 0) {
                std::uniform_real_distribution<float> probDis(0.0f, totalDistance);
                float target = probDis(gen);
                float cumSum = 0.0f;
                
                for (size_t j = 0; j < data.size(); ++j) {
                    cumSum += distances[j];
                    if (cumSum >= target) {
                        centroids.push_back(data[j]);
                        break;
                    }
                }
            }
        }
    }
    
    // Perform clustering
    std::vector<int> cluster(const std::vector<std::vector<float>>& data) {
        if (data.empty()) {
            return std::vector<int>();
        }
        
        initializeCentroids(data);
        
        std::vector<int> assignments(data.size());
        bool converged = false;
        
        for (int iter = 0; iter < maxIterations && !converged; ++iter) {
            // Assign points to nearest centroid
            for (size_t i = 0; i < data.size(); ++i) {
                float minDist = std::numeric_limits<float>::max();
                int bestCluster = 0;
                
                for (size_t j = 0; j < centroids.size(); ++j) {
                    float dist = SimilarityCalculator::euclideanDistance(data[i], centroids[j]);
                    if (dist < minDist) {
                        minDist = dist;
                        bestCluster = j;
                    }
                }
                
                assignments[i] = bestCluster;
            }
            
            // Update centroids
            std::vector<std::vector<float>> newCentroids(centroids.size(), 
                std::vector<float>(data[0].size(), 0.0f));
            std::vector<int> counts(centroids.size(), 0);
            
            for (size_t i = 0; i < data.size(); ++i) {
                int cluster = assignments[i];
                for (size_t j = 0; j < data[i].size(); ++j) {
                    newCentroids[cluster][j] += data[i][j];
                }
                counts[cluster]++;
            }
            
            // Calculate average
            for (size_t i = 0; i < newCentroids.size(); ++i) {
                if (counts[i] > 0) {
                    for (float& val : newCentroids[i]) {
                        val /= counts[i];
                    }
                }
            }
            
            // Check convergence
            float maxChange = 0.0f;
            for (size_t i = 0; i < centroids.size(); ++i) {
                float change = SimilarityCalculator::euclideanDistance(centroids[i], newCentroids[i]);
                maxChange = std::max(maxChange, change);
            }
            
            centroids = newCentroids;
            converged = (maxChange < convergenceThreshold);
        }
        
        return assignments;
    }
};

// Tool Manager class
class ToolManager {
private:
    std::vector<Tool> tools;
    std::vector<PrimaryClass> primaryClasses;
    std::unique_ptr<SimpleTextEncoder> encoder;
    Config config;
    
public:
    ToolManager(const Config& cfg) : config(cfg) {
        encoder = std::make_unique<SimpleTextEncoder>(config.embeddingDimension);
        initializePrimaryClasses();
    }
    
    // Initialize primary classes with embeddings
    void initializePrimaryClasses() {
        for (size_t i = 0; i < config.primaryClassNames.size(); ++i) {
            PrimaryClass pc;
            pc.name = config.primaryClassNames[i];
            pc.description = (i < config.primaryClassDescriptions.size()) ? 
                config.primaryClassDescriptions[i] : config.primaryClassNames[i];
            
            // Encode primary class description
            pc.embedding = encoder->encode(pc.name + " " + pc.description);
            primaryClasses.push_back(pc);
        }
        
        std::cout << "Initialized " << primaryClasses.size() << " primary classes" << std::endl;
    }
    
    // Get tool list function
    void getToolList() {
        // Comprehensive list of example tools
        std::vector<json> toolsJson = {
            // Data Processing Tools
            {{"name", "file_reader"}, {"description", "Reads content from files on disk"}, 
             {"parameters", {{"path", "string"}, {"encoding", "string"}}}},
            {{"name", "csv_parser"}, {"description", "Parses CSV files and extracts data"}, 
             {"parameters", {{"file", "string"}, {"delimiter", "string"}}}},
            {{"name", "json_parser"}, {"description", "Parses JSON data structures"}, 
             {"parameters", {{"json_string", "string"}}}},
            {{"name", "xml_parser"}, {"description", "Parses XML documents"}, 
             {"parameters", {{"xml_string", "string"}}}},
            {{"name", "database_query"}, {"description", "Executes SQL database queries"}, 
             {"parameters", {{"query", "string"}, {"connection", "string"}}}},
            {{"name", "data_transformer"}, {"description", "Transforms data between formats"}, 
             {"parameters", {{"input", "object"}, {"format", "string"}}}},
            
            // Communication Tools
            {{"name", "web_scraper"}, {"description", "Scrapes data from websites"}, 
             {"parameters", {{"url", "string"}, {"selector", "string"}}}},
            {{"name", "api_caller"}, {"description", "Makes HTTP API calls"}, 
             {"parameters", {{"endpoint", "string"}, {"method", "string"}, {"headers", "object"}}}},
            {{"name", "email_sender"}, {"description", "Sends emails to recipients"}, 
             {"parameters", {{"to", "string"}, {"subject", "string"}, {"body", "string"}}}},
            {{"name", "slack_notifier"}, {"description", "Sends notifications to Slack channels"}, 
             {"parameters", {{"channel", "string"}, {"message", "string"}}}},
            {{"name", "webhook_sender"}, {"description", "Sends data to webhooks"}, 
             {"parameters", {{"url", "string"}, {"payload", "object"}}}},
            
            // Analysis Tools
            {{"name", "text_analyzer"}, {"description", "Analyzes text content for insights"}, 
             {"parameters", {{"text", "string"}, {"analysis_type", "string"}}}},
            {{"name", "sentiment_analyzer"}, {"description", "Analyzes sentiment of text"}, 
             {"parameters", {{"text", "string"}, {"language", "string"}}}},
            {{"name", "image_processor"}, {"description", "Processes and manipulates images"}, 
             {"parameters", {{"image_path", "string"}, {"operation", "string"}}}},
            {{"name", "calculator"}, {"description", "Performs mathematical calculations"}, 
             {"parameters", {{"expression", "string"}}}},
            {{"name", "statistics_calculator"}, {"description", "Calculates statistical metrics"}, 
             {"parameters", {{"data", "array"}, {"metric", "string"}}}},
            
            // External Services
            {{"name", "weather_fetcher"}, {"description", "Gets weather information for locations"}, 
             {"parameters", {{"location", "string"}, {"units", "string"}}}},
            {{"name", "translation_service"}, {"description", "Translates text between languages"}, 
             {"parameters", {{"text", "string"}, {"source_lang", "string"}, {"target_lang", "string"}}}},
            {{"name", "geocoding_service"}, {"description", "Converts addresses to coordinates"}, 
             {"parameters", {{"address", "string"}}}},
            {{"name", "stock_price_fetcher"}, {"description", "Gets stock market prices"}, 
             {"parameters", {{"symbol", "string"}, {"exchange", "string"}}}},
            
            // Utility Tools
            {{"name", "logger"}, {"description", "Logs messages to various outputs"}, 
             {"parameters", {{"message", "string"}, {"level", "string"}}}},
            {{"name", "cache_manager"}, {"description", "Manages cache storage"}, 
             {"parameters", {{"key", "string"}, {"value", "any"}, {"ttl", "number"}}}},
            {{"name", "scheduler"}, {"description", "Schedules tasks for execution"}, 
             {"parameters", {{"task", "string"}, {"cron", "string"}}}},
            {{"name", "validator"}, {"description", "Validates data against schemas"}, 
             {"parameters", {{"data", "object"}, {"schema", "object"}}}},
            {{"name", "formatter"}, {"description", "Formats data for display"}, 
             {"parameters", {{"data", "any"}, {"format", "string"}}}}
        };
        
        // Parse and create Tool objects
        for (const auto& toolJson : toolsJson) {
            Tool tool;
            tool.name = toolJson["name"];
            tool.description = toolJson["description"];
            
            if (toolJson.contains("parameters")) {
                for (auto& [key, value] : toolJson["parameters"].items()) {
                    tool.parameters[key] = value;
                }
            }
            
            // Generate embedding for tool
            std::string toolText = tool.name + " " + tool.description;
            tool.embedding = encoder->encode(toolText);
            
            tools.push_back(tool);
        }
        
        std::cout << "Loaded " << tools.size() << " tools" << std::endl;
    }
    
    // Classify tools into primary classes
    void classifyTools() {
        std::cout << "\nClassifying tools into primary classes..." << std::endl;
        
        for (auto& tool : tools) {
            float maxSimilarity = -1.0f;
            int bestClass = -1;
            
            // Calculate similarity with each primary class
            std::vector<float> similarities;
            for (size_t i = 0; i < primaryClasses.size(); ++i) {
                float similarity = SimilarityCalculator::cosineSimilarity(
                    tool.embedding, primaryClasses[i].embedding
                );
                similarities.push_back(similarity);
                
                if (similarity > maxSimilarity) {
                    maxSimilarity = similarity;
                    bestClass = i;
                }
            }
            
            // Assign to best matching class
            if (bestClass >= 0 && maxSimilarity >= config.similarityThreshold) {
                tool.primaryClass = bestClass;
            } else {
                // Assign to "Other" class (last one)
                tool.primaryClass = primaryClasses.size() - 1;
            }
            
            primaryClasses[tool.primaryClass].tools.push_back(&tool);
            
            // Print classification details
            std::cout << "  " << std::setw(25) << std::left << tool.name 
                      << " -> " << primaryClasses[tool.primaryClass].name
                      << " (similarity: " << std::fixed << std::setprecision(3) 
                      << maxSimilarity << ")" << std::endl;
        }
        
        // Print summary
        std::cout << "\n=== Primary Classification Summary ===" << std::endl;
        for (const auto& pc : primaryClasses) {
            if (!pc.tools.empty()) {
                std::cout << std::setw(25) << std::left << pc.name 
                          << ": " << pc.tools.size() << " tools" << std::endl;
            }
        }
    }
    
    // Cluster tools within each primary class
    void clusterWithinClasses() {
        std::cout << "\n=== Clustering within Primary Classes ===" << std::endl;
        
        for (auto& pc : primaryClasses) {
            if (pc.tools.size() <= 1) {
                if (!pc.tools.empty()) {
                    pc.tools[0]->subCluster = 0;
                }
                continue;
            }
            
            std::cout << "\nClustering " << pc.name << " (" << pc.tools.size() << " tools):" << std::endl;
            
            // Prepare embeddings for clustering
            std::vector<std::vector<float>> embeddings;
            for (const auto* tool : pc.tools) {
                embeddings.push_back(tool->embedding);
            }
            
            // Determine number of clusters
            int numClusters = std::min(config.numSubClusters, static_cast<int>(pc.tools.size()));
            
            // Perform K-means clustering
            KMeansClustering kmeans(numClusters, config.maxIterations, config.convergenceThreshold);
            std::vector<int> assignments = kmeans.cluster(embeddings);
            
            // Assign cluster IDs and organize results
            std::map<int, std::vector<std::string>> clusters;
            for (size_t i = 0; i < pc.tools.size(); ++i) {
                pc.tools[i]->subCluster = assignments[i];
                clusters[assignments[i]].push_back(pc.tools[i]->name);
            }
            
            // Print clustering results
            for (const auto& [clusterId, toolNames] : clusters) {
                std::cout << "  Sub-cluster " << clusterId + 1 << ":" << std::endl;
                for (const auto& name : toolNames) {
                    std::cout << "    - " << name << std::endl;
                }
            }
        }
    }
    
    // Calculate cluster quality metrics
    void calculateMetrics() {
        std::cout << "\n=== Clustering Quality Metrics ===" << std::endl;
        
        for (const auto& pc : primaryClasses) {
            if (pc.tools.size() <= 1) continue;
            
            std::cout << "\n" << pc.name << ":" << std::endl;
            
            // Calculate intra-cluster similarity (cohesion)
            std::map<int, std::vector<const Tool*>> clusters;
            for (const auto* tool : pc.tools) {
                clusters[tool->subCluster].push_back(tool);
            }
            
            float totalCohesion = 0.0f;
            int clusterCount = 0;
            
            for (const auto& [clusterId, clusterTools] : clusters) {
                if (clusterTools.size() > 1) {
                    float cohesion = 0.0f;
                    int pairCount = 0;
                    
                    for (size_t i = 0; i < clusterTools.size(); ++i) {
                        for (size_t j = i + 1; j < clusterTools.size(); ++j) {
                            cohesion += SimilarityCalculator::cosineSimilarity(
                                clusterTools[i]->embedding, 
                                clusterTools[j]->embedding
                            );
                            pairCount++;
                        }
                    }
                    
                    if (pairCount > 0) {
                        cohesion /= pairCount;
                        totalCohesion += cohesion;
                        clusterCount++;
                        std::cout << "  Cluster " << clusterId + 1 
                                  << " cohesion: " << std::fixed << std::setprecision(3) 
                                  << cohesion << std::endl;
                    }
                }
            }
            
            if (clusterCount > 0) {
                std::cout << "  Average cohesion: " << std::fixed << std::setprecision(3) 
                          << (totalCohesion / clusterCount) << std::endl;
            }
        }
    }
    
    // Export results to JSON
    void exportResults(const std::string& filename) {
        json output;
        output["timestamp"] = std::time(nullptr);
        output["config"] = {
            {"numSubClusters", config.numSubClusters},
            {"similarityThreshold", config.similarityThreshold},
            {"embeddingDimension", config.embeddingDimension}
        };
        
        output["summary"] = {
            {"totalTools", tools.size()},
            {"numPrimaryClasses", primaryClasses.size()}
        };
        
        json classesJson = json::array();
        for (const auto& pc : primaryClasses) {
            if (pc.tools.empty()) continue;
            
            json classJson;
            classJson["name"] = pc.name;
            classJson["description"] = pc.description;
            classJson["toolCount"] = pc.tools.size();
            
            // Organize tools by subclusters
            std::map<int, json> subclusters;
            for (const auto* tool : pc.tools) {
                if (subclusters.find(tool->subCluster) == subclusters.end()) {
                    subclusters[tool->subCluster] = json::object();
                    subclusters[tool->subCluster]["tools"] = json::array();
                }
                
                json toolJson;
                toolJson["name"] = tool->name;
                toolJson["description"] = tool->description;
                toolJson["parameters"] = tool->parameters;
                subclusters[tool->subCluster]["tools"].push_back(toolJson);
            }
            
            json subclustersArray = json::array();
            for (const auto& [clusterId, clusterData] : subclusters) {
                json subcluster;
                subcluster["id"] = clusterId;
                subcluster["tools"] = clusterData["tools"];
                subclustersArray.push_back(subcluster);
            }
            classJson["subclusters"] = subclustersArray;
            
            classesJson.push_back(classJson);
        }
        output["classes"] = classesJson;
        
        // Write to file
        std::ofstream file(filename);
        file << output.dump(2);
        file.close();
        
        std::cout << "\nResults exported to: " << filename << std::endl;
    }
    
    // Main processing pipeline
    void process() {
        std::cout << "=====================================" << std::endl;
        std::cout << "Tool Classification & Clustering System" << std::endl;
        std::cout << "=====================================" << std::endl;
        
        // Step 1: Get all tools
        std::cout << "\n[Step 1] Loading tools..." << std::endl;
        getToolList();
        
        // Step 2: Classify tools into primary classes
        std::cout << "\n[Step 2] Primary classification..." << std::endl;
        classifyTools();
        
        // Step 3: Cluster within each class
        std::cout << "\n[Step 3] Sub-clustering..." << std::endl;
        clusterWithinClasses();
        
        // Step 4: Calculate metrics
        std::cout << "\n[Step 4] Calculating metrics..." << std::endl;
        calculateMetrics();
        
        // Step 5: Export results
        std::cout << "\n[Step 5] Exporting results..." << std::endl;
        exportResults("tool_classification_results.json");
        
        std::cout << "\n=====================================" << std::endl;
        std::cout << "Processing Complete!" << std::endl;
        std::cout << "=====================================" << std::endl;
    }
};

// Load configuration
Config loadConfig(const std::string& configFile) {
    Config config;
    
    try {
        std::ifstream file(configFile);
        if (file.is_open()) {
            json configJson;
            file >> configJson;
            
            config.numSubClusters = configJson.value("numSubClusters", 3);
            config.similarityThreshold = configJson.value("similarityThreshold", 0.6f);
            config.maxIterations = configJson.value("maxIterations", 100);
            config.convergenceThreshold = configJson.value("convergenceThreshold", 0.001f);
            config.embeddingDimension = configJson.value("embeddingDimension", 256);
            
            if (configJson.contains("primaryClasses")) {
                for (const auto& pc : configJson["primaryClasses"]) {
                    config.primaryClassNames.push_back(pc["name"]);
                    config.primaryClassDescriptions.push_back(pc["description"]);
                }
            }
            
            std::cout << "Configuration loaded from: " << configFile << std::endl;
        } else {
            throw std::runtime_error("Cannot open config file");
        }
    } catch (const std::exception& e) {
        std::cout << "Using default configuration (config file error: " << e.what() << ")" << std::endl;
        
        // Default configuration
        config.embeddingDimension = 256;
        config.numSubClusters = 3;
        config.similarityThreshold = 0.6f;
        
        config.primaryClassNames = {
            "Data Processing",
            "Communication",
            "Analysis & Intelligence",
            "External Services",
            "Utility & Helpers",
            "Other"
        };
        
        config.primaryClassDescriptions = {
            "Tools for reading, writing, querying, and manipulating data",
            "Tools for sending messages and making network requests",
            "Tools for analyzing data and extracting insights",
            "Tools that interact with external APIs and services",
            "General utility and helper tools",
            "Miscellaneous tools that don't fit other categories"
        };
    }
    
    return config;
}

int main(int argc, char* argv[]) {
    try {
        // Load configuration
        std::string configFile = (argc > 1) ? argv[1] : "tool_config.json";
        Config config = loadConfig(configFile);
        
        // Create tool manager and process
        ToolManager manager(config);
        manager.process();
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}