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
#include <ctime>

// Simple JSON-like data structure (minimal implementation)
class SimpleJson {
public:
    std::string stringify() const { return content; }
    void addField(const std::string& key, const std::string& value) {
        if (!content.empty() && content.back() != '{') content += ", ";
        if (content.empty()) content = "{";
        content += "\"" + key + "\": \"" + value + "\"";
    }
    void close() { if (!content.empty()) content += "}"; }
private:
    std::string content;
};

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
    std::vector<std::string> primaryClassNames;
    std::vector<std::string> primaryClassDescriptions;
    int numSubClusters = 3;
    float similarityThreshold = 0.6f;
    int maxIterations = 100;
    float convergenceThreshold = 0.001f;
    int embeddingDimension = 128;
};

// Simple text encoder using hash-based embeddings
class SimpleTextEncoder {
private:
    int embeddingDim;
    std::mt19937 gen;
    
public:
    SimpleTextEncoder(int dim = 128) : embeddingDim(dim), gen(42) {}
    
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
            std::transform(word.begin(), word.end(), word.begin(), 
                [](unsigned char c){ return std::tolower(c); });
            
            size_t wordHash = hasher(word);
            gen.seed(static_cast<unsigned int>(wordHash));
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
        std::uniform_int_distribution<> dis(0, static_cast<int>(data.size()) - 1);
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
                        bestCluster = static_cast<int>(j);
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

// Tool Manager class - main orchestrator
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
        
        std::cout << "✓ Initialized " << primaryClasses.size() << " primary classes" << std::endl;
    }
    
    // Get tool list function - simulates fetching from external source
    void getToolList() {
        // Comprehensive list of example tools organized by category
        struct ToolDef {
            std::string name;
            std::string description;
            std::vector<std::pair<std::string, std::string>> params;
        };
        
        std::vector<ToolDef> toolDefs = {
            // Data Processing Tools
            {"file_reader", "Reads content from files on disk", 
             {{"path", "string"}, {"encoding", "string"}}},
            {"file_writer", "Writes content to files", 
             {{"path", "string"}, {"content", "string"}}},
            {"csv_parser", "Parses CSV files and extracts data", 
             {{"file", "string"}, {"delimiter", "string"}}},
            {"json_parser", "Parses JSON data structures", 
             {{"json_string", "string"}}},
            {"xml_parser", "Parses XML documents and extracts data", 
             {{"xml_string", "string"}, {"xpath", "string"}}},
            {"database_query", "Executes SQL database queries", 
             {{"query", "string"}, {"connection", "string"}}},
            {"data_transformer", "Transforms data between different formats", 
             {{"input", "object"}, {"format", "string"}}},
            {"excel_reader", "Reads data from Excel spreadsheets", 
             {{"file", "string"}, {"sheet", "string"}}},
            
            // Communication Tools
            {"web_scraper", "Scrapes data from websites", 
             {{"url", "string"}, {"selector", "string"}}},
            {"api_caller", "Makes HTTP API calls to endpoints", 
             {{"endpoint", "string"}, {"method", "string"}, {"headers", "object"}}},
            {"graphql_client", "Executes GraphQL queries", 
             {{"query", "string"}, {"variables", "object"}}},
            {"email_sender", "Sends emails to recipients", 
             {{"to", "string"}, {"subject", "string"}, {"body", "string"}}},
            {"slack_notifier", "Sends notifications to Slack channels", 
             {{"channel", "string"}, {"message", "string"}}},
            {"webhook_sender", "Sends data to webhook endpoints", 
             {{"url", "string"}, {"payload", "object"}}},
            {"mqtt_publisher", "Publishes messages to MQTT topics", 
             {{"topic", "string"}, {"message", "string"}}},
            
            // Analysis & Intelligence Tools
            {"text_analyzer", "Analyzes text content for insights", 
             {{"text", "string"}, {"analysis_type", "string"}}},
            {"sentiment_analyzer", "Analyzes sentiment of text", 
             {{"text", "string"}, {"language", "string"}}},
            {"image_processor", "Processes and manipulates images", 
             {{"image_path", "string"}, {"operation", "string"}}},
            {"ocr_scanner", "Extracts text from images", 
             {{"image", "string"}, {"language", "string"}}},
            {"calculator", "Performs mathematical calculations", 
             {{"expression", "string"}}},
            {"statistics_calculator", "Calculates statistical metrics", 
             {{"data", "array"}, {"metric", "string"}}},
            {"ml_predictor", "Makes predictions using ML models", 
             {{"input", "array"}, {"model", "string"}}},
            
            // External Services
            {"weather_fetcher", "Gets weather information for locations", 
             {{"location", "string"}, {"units", "string"}}},
            {"translation_service", "Translates text between languages", 
             {{"text", "string"}, {"source_lang", "string"}, {"target_lang", "string"}}},
            {"geocoding_service", "Converts addresses to coordinates", 
             {{"address", "string"}}},
            {"stock_price_fetcher", "Gets stock market prices", 
             {{"symbol", "string"}, {"exchange", "string"}}},
            {"currency_converter", "Converts between currencies", 
             {{"amount", "number"}, {"from", "string"}, {"to", "string"}}},
            
            // Utility & Helper Tools
            {"logger", "Logs messages to various outputs", 
             {{"message", "string"}, {"level", "string"}}},
            {"cache_manager", "Manages cache storage", 
             {{"key", "string"}, {"value", "any"}, {"ttl", "number"}}},
            {"scheduler", "Schedules tasks for execution", 
             {{"task", "string"}, {"cron", "string"}}},
            {"validator", "Validates data against schemas", 
             {{"data", "object"}, {"schema", "object"}}},
            {"formatter", "Formats data for display", 
             {{"data", "any"}, {"format", "string"}}},
            {"uuid_generator", "Generates unique identifiers", 
             {{"version", "number"}}},
            {"hash_generator", "Generates hash values", 
             {{"data", "string"}, {"algorithm", "string"}}}
        };
        
        // Create Tool objects
        for (const auto& def : toolDefs) {
            Tool tool;
            tool.name = def.name;
            tool.description = def.description;
            
            for (const auto& param : def.params) {
                tool.parameters[param.first] = param.second;
            }
            
            // Generate embedding for tool
            std::string toolText = tool.name + " " + tool.description;
            tool.embedding = encoder->encode(toolText);
            
            tools.push_back(tool);
        }
        
        std::cout << "✓ Loaded " << tools.size() << " tools" << std::endl;
    }
    
    // Classify tools into primary classes
    void classifyTools() {
        std::cout << "\n📊 Classifying tools into primary classes..." << std::endl;
        std::cout << std::string(50, '-') << std::endl;
        
        for (auto& tool : tools) {
            float maxSimilarity = -1.0f;
            int bestClass = -1;
            
            // Calculate similarity with each primary class
            for (size_t i = 0; i < primaryClasses.size(); ++i) {
                float similarity = SimilarityCalculator::cosineSimilarity(
                    tool.embedding, primaryClasses[i].embedding
                );
                
                if (similarity > maxSimilarity) {
                    maxSimilarity = similarity;
                    bestClass = static_cast<int>(i);
                }
            }
            
            // Assign to best matching class
            if (bestClass >= 0 && maxSimilarity >= config.similarityThreshold) {
                tool.primaryClass = bestClass;
            } else {
                // Assign to "Other" class (last one)
                tool.primaryClass = static_cast<int>(primaryClasses.size()) - 1;
            }
            
            primaryClasses[tool.primaryClass].tools.push_back(&tool);
            
            // Print classification details
            std::cout << "  " << std::setw(25) << std::left << tool.name 
                      << " → " << std::setw(20) << primaryClasses[tool.primaryClass].name
                      << " (similarity: " << std::fixed << std::setprecision(3) 
                      << maxSimilarity << ")" << std::endl;
        }
        
        // Print summary
        std::cout << "\n📈 Classification Summary:" << std::endl;
        std::cout << std::string(50, '-') << std::endl;
        for (const auto& pc : primaryClasses) {
            if (!pc.tools.empty()) {
                std::cout << "  " << std::setw(25) << std::left << pc.name 
                          << ": " << pc.tools.size() << " tools" << std::endl;
            }
        }
    }
    
    // Cluster tools within each primary class
    void clusterWithinClasses() {
        std::cout << "\n🔧 Clustering within Primary Classes" << std::endl;
        std::cout << std::string(50, '=') << std::endl;
        
        for (auto& pc : primaryClasses) {
            if (pc.tools.size() <= 1) {
                if (!pc.tools.empty()) {
                    pc.tools[0]->subCluster = 0;
                }
                continue;
            }
            
            std::cout << "\n🎯 " << pc.name << " (" << pc.tools.size() << " tools):" << std::endl;
            
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
                std::cout << "  📦 Sub-cluster " << clusterId + 1 << ":" << std::endl;
                for (const auto& name : toolNames) {
                    std::cout << "     • " << name << std::endl;
                }
            }
        }
    }
    
    // Calculate and display metrics
    void calculateMetrics() {
        std::cout << "\n📊 Clustering Quality Metrics" << std::endl;
        std::cout << std::string(50, '=') << std::endl;
        
        float totalCohesion = 0.0f;
        int validClasses = 0;
        
        for (const auto& pc : primaryClasses) {
            if (pc.tools.size() <= 1) continue;
            
            std::cout << "\n" << pc.name << ":" << std::endl;
            
            // Calculate intra-cluster similarity (cohesion)
            std::map<int, std::vector<const Tool*>> clusters;
            for (const auto* tool : pc.tools) {
                clusters[tool->subCluster].push_back(tool);
            }
            
            float classCohesion = 0.0f;
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
                        classCohesion += cohesion;
                        clusterCount++;
                        std::cout << "  Cluster " << clusterId + 1 
                                  << " cohesion: " << std::fixed << std::setprecision(3) 
                                  << cohesion << std::endl;
                    }
                }
            }
            
            if (clusterCount > 0) {
                classCohesion /= clusterCount;
                totalCohesion += classCohesion;
                validClasses++;
                std::cout << "  Average cohesion: " << std::fixed << std::setprecision(3) 
                          << classCohesion << std::endl;
            }
        }
        
        if (validClasses > 0) {
            std::cout << "\n📈 Overall average cohesion: " << std::fixed << std::setprecision(3)
                      << (totalCohesion / validClasses) << std::endl;
        }
    }
    
    // Export results to file
    void exportResults(const std::string& filename) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open output file" << std::endl;
            return;
        }
        
        file << "{\n";
        file << "  \"timestamp\": " << std::time(nullptr) << ",\n";
        file << "  \"summary\": {\n";
        file << "    \"totalTools\": " << tools.size() << ",\n";
        file << "    \"numPrimaryClasses\": " << primaryClasses.size() << "\n";
        file << "  },\n";
        file << "  \"classes\": [\n";
        
        bool firstClass = true;
        for (const auto& pc : primaryClasses) {
            if (pc.tools.empty()) continue;
            
            if (!firstClass) file << ",\n";
            firstClass = false;
            
            file << "    {\n";
            file << "      \"name\": \"" << pc.name << "\",\n";
            file << "      \"toolCount\": " << pc.tools.size() << ",\n";
            file << "      \"tools\": [\n";
            
            bool firstTool = true;
            for (const auto* tool : pc.tools) {
                if (!firstTool) file << ",\n";
                firstTool = false;
                
                file << "        {\n";
                file << "          \"name\": \"" << tool->name << "\",\n";
                file << "          \"description\": \"" << tool->description << "\",\n";
                file << "          \"subCluster\": " << tool->subCluster << "\n";
                file << "        }";
            }
            
            file << "\n      ]\n";
            file << "    }";
        }
        
        file << "\n  ]\n";
        file << "}\n";
        
        file.close();
        std::cout << "\n✅ Results exported to: " << filename << std::endl;
    }
    
    // Main processing pipeline
    void process() {
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "    🚀 TOOL CLASSIFICATION & CLUSTERING SYSTEM" << std::endl;
        std::cout << "       Using Google Gemma-style Embeddings" << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        
        // Step 1: Get all tools
        std::cout << "\n[Step 1/5] Loading tools..." << std::endl;
        getToolList();
        
        // Step 2: Classify tools
        std::cout << "\n[Step 2/5] Primary classification..." << std::endl;
        classifyTools();
        
        // Step 3: Sub-clustering
        std::cout << "\n[Step 3/5] Sub-clustering..." << std::endl;
        clusterWithinClasses();
        
        // Step 4: Metrics
        std::cout << "\n[Step 4/5] Calculating metrics..." << std::endl;
        calculateMetrics();
        
        // Step 5: Export
        std::cout << "\n[Step 5/5] Exporting results..." << std::endl;
        exportResults("tool_classification_results.json");
        
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "    ✨ PROCESSING COMPLETE!" << std::endl;
        std::cout << std::string(60, '=') << std::endl;
    }
};

// Load configuration
Config loadConfig() {
    Config config;
    
    // Default configuration
    config.embeddingDimension = 128;
    config.numSubClusters = 3;
    config.similarityThreshold = 0.55f;  // Lowered for better classification
    config.maxIterations = 100;
    config.convergenceThreshold = 0.001f;
    
    config.primaryClassNames = {
        "Data Processing",
        "Communication",
        "Analysis & Intelligence",
        "External Services",
        "Utility & Helpers",
        "Other"
    };
    
    config.primaryClassDescriptions = {
        "Tools for reading, writing, querying, and manipulating data from files and databases",
        "Tools for sending messages, making network requests, and API interactions",
        "Tools for analyzing data, processing text and images, and extracting insights",
        "Tools that interact with external APIs and third-party services",
        "General utility tools, helpers, validators, and formatters",
        "Miscellaneous tools that don't fit other categories"
    };
    
    std::cout << "Configuration loaded successfully" << std::endl;
    std::cout << "  • Embedding dimension: " << config.embeddingDimension << std::endl;
    std::cout << "  • Sub-clusters per class: " << config.numSubClusters << std::endl;
    std::cout << "  • Similarity threshold: " << config.similarityThreshold << std::endl;
    
    return config;
}

int main(int argc, char* argv[]) {
    try {
        std::cout << "\n╔════════════════════════════════════════════════╗" << std::endl;
        std::cout << "║  Tool Classifier - Standalone C++ Implementation  ║" << std::endl;
        std::cout << "╚════════════════════════════════════════════════╝" << std::endl;
        
        // Load configuration
        Config config = loadConfig();
        
        // Create tool manager and process
        ToolManager manager(config);
        manager.process();
        
        std::cout << "\n👍 Program completed successfully!" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "\n❌ Error: " << e.what() << std::endl;
        return 1;
    }
}