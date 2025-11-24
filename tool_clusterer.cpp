#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include <map>
#include <random>
#include "json.hpp"

using json = nlohmann::json;
using namespace std;

// --- Data Structures ---

struct Tool {
    string id;
    string name;
    string description;
    json parameters;
    vector<float> embedding;
    string primary_class;
    int cluster_id = -1;
};

struct PrimaryClass {
    string name;
    vector<float> embedding;
};

// --- Configuration ---
// Assuming 300 dimensions for the embedding if user meant dimension, 
// or just a standard size. The prompt says "gemma 300m model". 
// This implies model size, but for embeddings we define the dimension.
const int EMBEDDING_DIM = 256; 

// --- Mock / Simulated Gemma Model ---

class GemmaEncoder {
public:
    GemmaEncoder(const string& model_path = "gemma-300m.gguf") {
        // In a real production environment, you would initialize the model here.
        // Example with llama.cpp:
        // llama_init_backend();
        // ctx = llama_new_context_with_model(...);
        cout << "[INFO] Initializing Gemma Encoder (Simulated)..." << endl;
        cout << "[INFO] Note: Actual Gemma model loading requires 'llama.cpp' or TensorFlow C++ API linkage." << endl;
        cout << "[INFO] Using deterministic hash-based embedding for demonstration." << endl;
    }

    // Encodes text into a vector
    vector<float> encode(const string& text) {
        // In a real implementation:
        // 1. Tokenize 'text'
        // 2. Run inference to get hidden states
        // 3. Pool hidden states (e.g., mean pooling) to get embedding vector
        
        // Mock implementation:
        // Generate a deterministic vector based on the string content.
        // This ensures the same string always gets the same vector.
        vector<float> embedding(EMBEDDING_DIM, 0.0f);
        
        // Seed random generator with string hash
        size_t seed = hash<string>{}(text);
        mt19937 gen(seed);
        normal_distribution<float> d(0.0f, 1.0f);

        float norm_sq = 0.0f;
        for(int i=0; i<EMBEDDING_DIM; ++i) {
            embedding[i] = d(gen);
            norm_sq += embedding[i] * embedding[i];
        }

        // Normalize
        float norm = sqrt(norm_sq);
        for(int i=0; i<EMBEDDING_DIM; ++i) {
            embedding[i] /= norm;
        }
        
        return embedding;
    }
};

// --- Helper Functions ---

float cosine_similarity(const vector<float>& a, const vector<float>& b) {
    float dot = 0.0f;
    float norm_a_sq = 0.0f;
    float norm_b_sq = 0.0f;
    for(size_t i=0; i<a.size(); ++i) {
        dot += a[i] * b[i];
        norm_a_sq += a[i] * a[i];
        norm_b_sq += b[i] * b[i];
    }
    if (norm_a_sq == 0 || norm_b_sq == 0) return 0.0f;
    return dot / (sqrt(norm_a_sq) * sqrt(norm_b_sq));
}

// Function to get all tools information
vector<Tool> get_tool_list(const string& filepath) {
    vector<Tool> tools;
    ifstream f(filepath);
    if (!f.is_open()) {
        cerr << "[ERROR] Could not open file: " << filepath << endl;
        return tools;
    }
    
    cout << "[INFO] Loading tools from " << filepath << "..." << endl;
    string line;
    int count = 0;
    while(getline(f, line)) {
        try {
            // Check if line is empty
            if (line.empty()) continue;
            
            auto j = json::parse(line);
            
            // Handle BFCL format which usually has a 'function' array inside
            if(j.contains("function") && j["function"].is_array() && !j["function"].empty()) {
                auto func = j["function"][0];
                Tool t;
                t.id = j.value("id", "unknown_" + to_string(count));
                t.name = func.value("name", "unnamed");
                t.description = func.value("description", "");
                t.parameters = func.value("parameters", json::object());
                tools.push_back(t);
                count++;
            }
        } catch (const exception& e) {
            // Ignore parse errors for specific lines
            continue; 
        }
        // Limit for demonstration speed
        if (count >= 200) break;
    }
    cout << "[INFO] Loaded " << tools.size() << " tools." << endl;
    return tools;
}

// Simple K-Means Clustering
void cluster_tools(vector<Tool*>& tools, int k) {
    if (tools.empty() || k <= 0) return;
    if (k > tools.size()) k = tools.size();

    // Initialize centroids randomly from points
    vector<vector<float>> centroids;
    vector<int> centroid_indices;
    // Simple random pick without replacement logic
    for(int i=0; i<tools.size(); ++i) centroid_indices.push_back(i);
    random_device rd;
    mt19937 g(rd());
    shuffle(centroid_indices.begin(), centroid_indices.end(), g);
    
    for(int i=0; i<k; ++i) {
        centroids.push_back(tools[centroid_indices[i]]->embedding);
    }

    bool changed = true;
    int max_iters = 20;
    int iter = 0;

    while(changed && iter < max_iters) {
        changed = false;
        iter++;

        // Assignment step
        for(auto* t : tools) {
            int best_cluster = -1;
            float best_sim = -2.0f; // Cosine sim is [-1, 1]

            for(int c=0; c<k; ++c) {
                float sim = cosine_similarity(t->embedding, centroids[c]);
                if (sim > best_sim) {
                    best_sim = sim;
                    best_cluster = c;
                }
            }
            if (t->cluster_id != best_cluster) {
                t->cluster_id = best_cluster;
                changed = true;
            }
        }

        // Update step
        vector<vector<float>> new_centroids(k, vector<float>(EMBEDDING_DIM, 0.0f));
        vector<int> counts(k, 0);

        for(auto* t : tools) {
            if (t->cluster_id == -1) continue;
            for(int i=0; i<EMBEDDING_DIM; ++i) {
                new_centroids[t->cluster_id][i] += t->embedding[i];
            }
            counts[t->cluster_id]++;
        }

        for(int c=0; c<k; ++c) {
            if (counts[c] > 0) {
                float norm_sq = 0.0f;
                for(int i=0; i<EMBEDDING_DIM; ++i) {
                    new_centroids[c][i] /= counts[c];
                    norm_sq += new_centroids[c][i] * new_centroids[c][i];
                }
                // Re-normalize for cosine similarity compatibility (centroids should be unit vectors)
                float norm = sqrt(norm_sq);
                if (norm > 1e-6) {
                    for(int i=0; i<EMBEDDING_DIM; ++i) new_centroids[c][i] /= norm;
                }
                centroids[c] = new_centroids[c];
            } else {
                // If a cluster is empty, re-initialize it to a random point to avoid collapse
                // For simplicity, leave as is or pick random point
            }
        }
    }
}

// --- Main Script ---

int main() {
    // 1. Predefined Primary Classes List
    vector<string> class_names = {
        "Mathematics", 
        "Physics & Engineering", 
        "Biology & Genetics", 
        "Geography & Navigation", 
        "General Utility & Data Processing",
        "Finance & Business"
    };

    // 2. Initialize Encoder (Gemma 300M placeholder)
    GemmaEncoder encoder;

    // Encode Primary Classes
    vector<PrimaryClass> primary_classes;
    cout << "[STEP] Encoding Primary Classes..." << endl;
    for (const auto& name : class_names) {
        primary_classes.push_back({name, encoder.encode(name)});
    }

    // 3. Get Tool List
    string data_path = "data/BFCL_v3_simple.json";
    vector<Tool> tools = get_tool_list(data_path);
    if (tools.empty()) {
        cerr << "No tools found." << endl;
        return 1;
    }

    // 4. Encode Tools (Name + Description)
    cout << "[STEP] Encoding " << tools.size() << " tools..." << endl;
    for (auto& tool : tools) {
        string text_to_encode = tool.name + ": " + tool.description;
        tool.embedding = encoder.encode(text_to_encode);
    }

    // 5. Assign Tools to Primary Classes based on Similarity
    cout << "[STEP] Assigning tools to Primary Classes..." << endl;
    map<string, vector<Tool*>> class_buckets;
    
    for (auto& tool : tools) {
        float max_sim = -2.0f;
        string best_class = "Unclassified";
        
        for (const auto& pc : primary_classes) {
            float sim = cosine_similarity(tool.embedding, pc.embedding);
            if (sim > max_sim) {
                max_sim = sim;
                best_class = pc.name;
            }
        }
        tool.primary_class = best_class;
        class_buckets[best_class].push_back(&tool);
    }

    // 6. Within each primary class, further cluster tools
    cout << "[STEP] Clustering within primary classes..." << endl;
    for (auto& [pclass, p_tools] : class_buckets) {
        if (p_tools.empty()) continue;

        // Determine number of clusters (e.g., sqrt(n/2) or fixed)
        int k = max(2, (int)sqrt(p_tools.size())); 
        if (p_tools.size() < 3) k = 1;

        cout << "  > Class '" << pclass << "': " << p_tools.size() << " tools. Clustering into " << k << " groups..." << endl;
        
        cluster_tools(p_tools, k);

        // Display results for this class
        map<int, vector<Tool*>> clusters;
        for(auto* t : p_tools) {
            clusters[t->cluster_id].push_back(t);
        }

        for(auto& [cid, c_tools] : clusters) {
            cout << "    - Cluster " << cid << " (" << c_tools.size() << " items):" << endl;
            // Print up to 3 examples
            for(size_t i=0; i<min((size_t)3, c_tools.size()); ++i) {
                cout << "      * " << c_tools[i]->name << endl;
            }
            if(c_tools.size() > 3) cout << "      * ..." << endl;
        }
    }

    cout << "[DONE] Script completed successfully." << endl;
    return 0;
}
