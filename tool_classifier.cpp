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
#include <nlohmann/json.hpp>

// For Gemma model inference (using ONNX Runtime)
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>

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
    std::string tokenizerPath;
    std::vector<std::string> primaryClassNames;
    std::vector<std::string> primaryClassDescriptions;
    int numSubClusters = 3;
    float similarityThreshold = 0.7f;
    int maxIterations = 100;
    float convergenceThreshold = 0.001f;
};

// Gemma Model Wrapper for encoding text
class GemmaEncoder {
private:
    std::unique_ptr<Ort::Session> session;
    Ort::Env env;
    Ort::SessionOptions sessionOptions;
    Ort::AllocatorWithDefaultOptions allocator;
    
public:
    GemmaEncoder(const std::string& modelPath) 
        : env(ORT_LOGGING_LEVEL_WARNING, "GemmaEncoder") {
        
        // Configure session options
        sessionOptions.SetIntraOpNumThreads(4);
        sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        
        // Load the model
        #ifdef _WIN32
            std::wstring wideModelPath(modelPath.begin(), modelPath.end());
            session = std::make_unique<Ort::Session>(env, wideModelPath.c_str(), sessionOptions);
        #else
            session = std::make_unique<Ort::Session>(env, modelPath.c_str(), sessionOptions);
        #endif
    }
    
    // Tokenize text (simplified - in production, use proper tokenizer)
    std::vector<int64_t> tokenize(const std::string& text) {
        std::vector<int64_t> tokens;
        std::istringstream iss(text);
        std::string word;
        
        // Simple word-based tokenization (replace with proper Gemma tokenizer)
        while (iss >> word) {
            // Hash each word to create a token ID (simplified)
            std::hash<std::string> hasher;
            size_t hash = hasher(word);
            tokens.push_back(static_cast<int64_t>(hash % 50000)); // Assuming vocab size of 50000
        }
        
        // Add padding/special tokens as needed
        if (tokens.empty()) {
            tokens.push_back(0); // Empty token
        }
        
        return tokens;
    }
    
    // Encode text to embeddings
    std::vector<float> encode(const std::string& text) {
        try {
            // Tokenize the input text
            std::vector<int64_t> inputIds = tokenize(text);
            
            // Create input tensor
            std::vector<int64_t> inputShape = {1, static_cast<int64_t>(inputIds.size())};
            Ort::Value inputTensor = Ort::Value::CreateTensor<int64_t>(
                allocator, inputShape.data(), inputShape.size()
            );
            
            // Copy input data
            int64_t* inputData = inputTensor.GetTensorMutableData<int64_t>();
            for (size_t i = 0; i < inputIds.size(); ++i) {
                inputData[i] = inputIds[i];
            }
            
            // Run inference
            const char* inputNames[] = {"input_ids"};
            const char* outputNames[] = {"embeddings"};
            
            std::vector<Ort::Value> inputs;
            inputs.push_back(std::move(inputTensor));
            
            auto outputs = session->Run(
                Ort::RunOptions{nullptr},
                inputNames, inputs.data(), inputs.size(),
                outputNames, 1
            );
            
            // Extract embeddings from output
            float* outputData = outputs[0].GetTensorMutableData<float>();
            auto outputShape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
            
            size_t embeddingSize = 1;
            for (auto dim : outputShape) {
                embeddingSize *= dim;
            }
            
            std::vector<float> embedding(outputData, outputData + embeddingSize);
            
            // Normalize the embedding
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
            
        } catch (const Ort::Exception& e) {
            std::cerr << "ONNX Runtime error: " << e.what() << std::endl;
            // Return random embedding as fallback for demonstration
            return generateRandomEmbedding(256);
        }
    }
    
    // Generate random embedding (fallback for demonstration)
    std::vector<float> generateRandomEmbedding(size_t size) {
        static std::mt19937 gen(42);
        static std::normal_distribution<float> dist(0.0f, 1.0f);
        
        std::vector<float> embedding(size);
        for (float& val : embedding) {
            val = dist(gen);
        }
        
        // Normalize
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
            std::cerr << "Vector size mismatch in cosine similarity" << std::endl;
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
        for (int i = 1; i < k && i < data.size(); ++i) {
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
                
                for (int j = 0; j < k && j < centroids.size(); ++j) {
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
    std::unique_ptr<GemmaEncoder> encoder;
    Config config;
    
public:
    ToolManager(const Config& cfg) : config(cfg) {
        // Initialize encoder
        encoder = std::make_unique<GemmaEncoder>(config.modelPath);
        
        // Initialize primary classes
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
    }
    
    // Get tool list function (simulated - in production, this would fetch from actual source)
    void getToolList() {
        // Example tools - in production, this would fetch from an API or database
        std::vector<json> toolsJson = {
            {{"name", "file_reader"}, {"description", "Reads content from files"}, 
             {"parameters", {{"path", "string"}, {"encoding", "string"}}}},
            {{"name", "web_scraper"}, {"description", "Scrapes data from websites"}, 
             {"parameters", {{"url", "string"}, {"selector", "string"}}}},
            {{"name", "database_query"}, {"description", "Executes database queries"}, 
             {"parameters", {{"query", "string"}, {"connection", "string"}}}},
            {{"name", "api_caller"}, {"description", "Makes HTTP API calls"}, 
             {"parameters", {{"endpoint", "string"}, {"method", "string"}, {"headers", "object"}}}},
            {{"name", "text_analyzer"}, {"description", "Analyzes text content"}, 
             {"parameters", {{"text", "string"}, {"analysis_type", "string"}}}},
            {{"name", "image_processor"}, {"description", "Processes and manipulates images"}, 
             {"parameters", {{"image_path", "string"}, {"operation", "string"}}}},
            {{"name", "email_sender"}, {"description", "Sends emails"}, 
             {"parameters", {{"to", "string"}, {"subject", "string"}, {"body", "string"}}}},
            {{"name", "calculator"}, {"description", "Performs mathematical calculations"}, 
             {"parameters", {{"expression", "string"}}}},
            {{"name", "weather_fetcher"}, {"description", "Gets weather information"}, 
             {"parameters", {{"location", "string"}, {"units", "string"}}}},
            {{"name", "translation_service"}, {"description", "Translates text between languages"}, 
             {"parameters", {{"text", "string"}, {"source_lang", "string"}, {"target_lang", "string"}}}}
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
        for (auto& tool : tools) {
            float maxSimilarity = -1.0f;
            int bestClass = -1;
            
            // Find best matching primary class
            for (size_t i = 0; i < primaryClasses.size(); ++i) {
                float similarity = SimilarityCalculator::cosineSimilarity(
                    tool.embedding, primaryClasses[i].embedding
                );
                
                if (similarity > maxSimilarity) {
                    maxSimilarity = similarity;
                    bestClass = i;
                }
            }
            
            // Assign to class if similarity exceeds threshold
            if (maxSimilarity >= config.similarityThreshold && bestClass >= 0) {
                tool.primaryClass = bestClass;
                primaryClasses[bestClass].tools.push_back(&tool);
            } else {
                // Create or assign to "Other" class
                tool.primaryClass = primaryClasses.size() - 1;
                primaryClasses.back().tools.push_back(&tool);
            }
        }
        
        // Print classification results
        std::cout << "\nPrimary Classification Results:" << std::endl;
        for (const auto& pc : primaryClasses) {
            std::cout << "Class: " << pc.name << " - " << pc.tools.size() << " tools" << std::endl;
            for (const auto* tool : pc.tools) {
                std::cout << "  - " << tool->name << std::endl;
            }
        }
    }
    
    // Cluster tools within each primary class
    void clusterWithinClasses() {
        std::cout << "\nClustering within primary classes:" << std::endl;
        
        for (auto& pc : primaryClasses) {
            if (pc.tools.size() <= 1) {
                // No need to cluster single tool
                if (!pc.tools.empty()) {
                    pc.tools[0]->subCluster = 0;
                }
                continue;
            }
            
            std::cout << "\nClustering class: " << pc.name << " with " << pc.tools.size() << " tools" << std::endl;
            
            // Prepare data for clustering
            std::vector<std::vector<float>> embeddings;
            for (const auto* tool : pc.tools) {
                embeddings.push_back(tool->embedding);
            }
            
            // Determine optimal number of clusters (min of configured value and number of tools)
            int numClusters = std::min(config.numSubClusters, static_cast<int>(pc.tools.size()));
            
            // Perform K-means clustering
            KMeansClustering kmeans(numClusters, config.maxIterations, config.convergenceThreshold);
            std::vector<int> assignments = kmeans.cluster(embeddings);
            
            // Assign cluster IDs to tools
            for (size_t i = 0; i < pc.tools.size(); ++i) {
                pc.tools[i]->subCluster = assignments[i];
            }
            
            // Print clustering results
            std::map<int, std::vector<std::string>> clusters;
            for (size_t i = 0; i < pc.tools.size(); ++i) {
                clusters[assignments[i]].push_back(pc.tools[i]->name);
            }
            
            for (const auto& [clusterId, toolNames] : clusters) {
                std::cout << "  Cluster " << clusterId << ":" << std::endl;
                for (const auto& name : toolNames) {
                    std::cout << "    - " << name << std::endl;
                }
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
            {"primaryClasses", config.primaryClassNames}
        };
        
        json classesJson = json::array();
        for (const auto& pc : primaryClasses) {
            json classJson;
            classJson["name"] = pc.name;
            classJson["description"] = pc.description;
            classJson["toolCount"] = pc.tools.size();
            
            json toolsJson = json::array();
            for (const auto* tool : pc.tools) {
                json toolJson;
                toolJson["name"] = tool->name;
                toolJson["description"] = tool->description;
                toolJson["subCluster"] = tool->subCluster;
                toolJson["parameters"] = tool->parameters;
                toolsJson.push_back(toolJson);
            }
            classJson["tools"] = toolsJson;
            
            // Group by subclusters
            std::map<int, json> subclusters;
            for (const auto* tool : pc.tools) {
                if (subclusters.find(tool->subCluster) == subclusters.end()) {
                    subclusters[tool->subCluster] = json::array();
                }
                subclusters[tool->subCluster].push_back(tool->name);
            }
            classJson["subclusters"] = subclusters;
            
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
        std::cout << "Starting tool classification and clustering pipeline..." << std::endl;
        
        // Step 1: Get all tools
        std::cout << "\nStep 1: Getting tool list..." << std::endl;
        getToolList();
        
        // Step 2: Classify tools into primary classes
        std::cout << "\nStep 2: Classifying tools into primary classes..." << std::endl;
        classifyTools();
        
        // Step 3: Cluster within each class
        std::cout << "\nStep 3: Clustering tools within each class..." << std::endl;
        clusterWithinClasses();
        
        // Step 4: Export results
        exportResults("tool_classification_results.json");
        
        std::cout << "\nProcessing complete!" << std::endl;
    }
};

// Load configuration from file
Config loadConfig(const std::string& configFile) {
    Config config;
    
    try {
        std::ifstream file(configFile);
        json configJson;
        file >> configJson;
        
        config.modelPath = configJson.value("modelPath", "gemma_300m.onnx");
        config.tokenizerPath = configJson.value("tokenizerPath", "tokenizer.json");
        config.numSubClusters = configJson.value("numSubClusters", 3);
        config.similarityThreshold = configJson.value("similarityThreshold", 0.7f);
        config.maxIterations = configJson.value("maxIterations", 100);
        config.convergenceThreshold = configJson.value("convergenceThreshold", 0.001f);
        
        if (configJson.contains("primaryClasses")) {
            for (const auto& pc : configJson["primaryClasses"]) {
                config.primaryClassNames.push_back(pc["name"]);
                config.primaryClassDescriptions.push_back(pc["description"]);
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error loading config: " << e.what() << std::endl;
        
        // Use default configuration
        config.modelPath = "gemma_300m.onnx";
        config.primaryClassNames = {
            "Data Processing",
            "Communication",
            "Analysis",
            "Utility",
            "Other"
        };
        config.primaryClassDescriptions = {
            "Tools for reading, writing, and manipulating data",
            "Tools for sending messages and making network requests",
            "Tools for analyzing and processing information",
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
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }
}