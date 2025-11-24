#include "ToolClusterer.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <random>
#include <map>

using namespace std;

// --- Helper Functions Implementation ---

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
            if (line.empty()) continue;
            
            auto j = json::parse(line);
            
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
            continue; 
        }
        if (count >= 200) break;
    }
    cout << "[INFO] Loaded " << tools.size() << " tools." << endl;
    return tools;
}

void cluster_tools_kmeans(vector<Tool*>& tools, int k) {
    if (tools.empty() || k <= 0) return;
    if (k > tools.size()) k = tools.size();

    vector<vector<float>> centroids;
    vector<int> centroid_indices;
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
            float best_sim = -2.0f;

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
                float norm = sqrt(norm_sq);
                if (norm > 1e-6) {
                    for(int i=0; i<EMBEDDING_DIM; ++i) new_centroids[c][i] /= norm;
                }
                centroids[c] = new_centroids[c];
            }
        }
    }
}

// --- Main Pipeline Logic ---

void run_tool_clustering_pipeline(const string& data_path) {
    // 1. Predefined Primary Classes List
    vector<string> class_names = {
        "Mathematics", 
        "Physics & Engineering", 
        "Biology & Genetics", 
        "Geography & Navigation", 
        "General Utility & Data Processing",
        "Finance & Business"
    };

    // 2. Initialize Encoder
    GemmaEncoder encoder;

    // Encode Primary Classes
    vector<PrimaryClass> primary_classes;
    cout << "[STEP] Encoding Primary Classes..." << endl;
    for (const auto& name : class_names) {
        primary_classes.push_back({name, encoder.encode(name)});
    }

    // 3. Get Tool List
    vector<Tool> tools = get_tool_list(data_path);
    if (tools.empty()) {
        cerr << "No tools found." << endl;
        return;
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

        // Determine number of clusters
        int k = max(2, (int)sqrt(p_tools.size())); 
        if (p_tools.size() < 3) k = 1;

        cout << "  > Class '" << pclass << "': " << p_tools.size() << " tools. Clustering into " << k << " groups..." << endl;
        
        cluster_tools_kmeans(p_tools, k);

        // Display results
        map<int, vector<Tool*>> clusters;
        for(auto* t : p_tools) {
            clusters[t->cluster_id].push_back(t);
        }

        for(auto& [cid, c_tools] : clusters) {
            cout << "    - Cluster " << cid << " (" << c_tools.size() << " items):" << endl;
            for(size_t i=0; i<min((size_t)3, c_tools.size()); ++i) {
                cout << "      * " << c_tools[i]->name << endl;
            }
            if(c_tools.size() > 3) cout << "      * ..." << endl;
        }
    }
}
