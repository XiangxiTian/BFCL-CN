#include "ToolClusterer.h"
#include "dbscan.h"
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

// Replace K-Means with DBSCAN
void cluster_tools_dbscan(vector<Tool*>& tools, float eps, int min_pts) {
    if (tools.empty()) return;

    // Convert Tools to Points for DBSCAN
    vector<Point> points;
    points.reserve(tools.size());
    for(size_t i=0; i<tools.size(); ++i) {
        Point p;
        p.embedding = tools[i]->embedding;
        p.clusterID = UNCLASSIFIED;
        p.originalIndex = i; // Store index to map back
        points.push_back(p);
    }

    // Run DBSCAN
    DBSCAN dbscan(min_pts, eps, points);
    dbscan.run();

    // Map results back to Tools
    // Note: DBSCAN::m_points is modified with clusterIDs
    for(const auto& p : dbscan.m_points) {
        if(p.originalIndex >= 0 && p.originalIndex < tools.size()) {
            tools[p.originalIndex]->cluster_id = p.clusterID;
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

    // 6. Within each primary class, further cluster tools using DBSCAN
    cout << "[STEP] Clustering within primary classes (DBSCAN)..." << endl;
    
    // DBSCAN Parameters
    // Epsilon (Squared Euclidean Distance threshold)
    // For normalized vectors, d^2 = 2(1 - cos_sim). 
    // High dimensional random vectors are nearly orthogonal.
    // If cos_sim is near 0, d^2 is near 2.
    // To cluster random data, we need a very loose epsilon.
    // In real data, semantic clusters are tighter.
    float eps = 1.95f; 
    int min_pts = 2;

    for (auto& [pclass, p_tools] : class_buckets) {
        if (p_tools.empty()) continue;

        cout << "  > Class '" << pclass << "': " << p_tools.size() << " tools." << endl;
        
        cluster_tools_dbscan(p_tools, eps, min_pts);

        // Display results
        map<int, vector<Tool*>> clusters;
        for(auto* t : p_tools) {
            // Check for noise
            if (t->cluster_id == NOISE) {
                clusters[-1].push_back(t); // -1 for noise
            } else {
                clusters[t->cluster_id].push_back(t);
            }
        }

        for(auto& [cid, c_tools] : clusters) {
            if (cid == -1) {
                cout << "    - [NOISE/Unclustered] (" << c_tools.size() << " items)" << endl;
            } else {
                cout << "    - Cluster " << cid << " (" << c_tools.size() << " items):" << endl;
            }
            
            for(size_t i=0; i<min((size_t)3, c_tools.size()); ++i) {
                cout << "      * " << c_tools[i]->name << endl;
            }
            if(c_tools.size() > 3) cout << "      * ..." << endl;
        }
    }
}
