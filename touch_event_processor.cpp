#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <numeric>

// Structure to hold touch event data
// This corresponds to one line of data from the input file
struct TouchEvent {
    int field0, field1, field2, field3, field4, field5;
    float field6;
    int field7;
    float field8, field9;
    int field10;
    std::string field11, field12, field13, bundleName;
    int event_ID;
    int trace_ID;  // Will be populated later during trace construction
    float duration;
    
    // Constructor to initialize from parsed line data
    TouchEvent() : field0(0), field1(0), field2(0), field3(0), field4(0), field5(0),
                   field6(0.0f), field7(0), field8(0.0f), field9(0.0f), field10(0),
                   event_ID(0), trace_ID(-1), duration(0.0f) {}
};

// Structure to hold column indices for easier data access
struct ColumnIndices {
    int bundleName_idx;
    int type_idx;
    int event_ID_idx;
    int posX_idx;
    int posY_idx;
    int time_idx;
    int duration_idx;
    int trace_ID_idx;
};

// Parse a single line from the input file into a TouchEvent structure
TouchEvent parseLine(const std::string& line, int line_number, 
                     const std::string& prev_bundleName, int& event_counter) {
    TouchEvent event;
    std::istringstream iss(line);
    
    // Parse the first 6 integer fields
    iss >> event.field0 >> event.field1 >> event.field2 
        >> event.field3 >> event.field4 >> event.field5;
    
    // Parse float, int, float, float, int sequence
    iss >> event.field6 >> event.field7 >> event.field8 >> event.field9 >> event.field10;
    
    // Parse the remaining string fields
    iss >> event.field11 >> event.field12 >> event.field13 >> event.bundleName;
    
    // Reset event counter when bundleName changes (indicates new app session)
    if (event.bundleName != prev_bundleName) {
        event_counter = 0;
    }
    event.event_ID = event_counter;
    event_counter++;
    
    return event;
}

// Load all touch events from the input file
std::vector<TouchEvent> loadTouchEvents(const std::string& file_path, ColumnIndices& col_idx) {
    std::vector<TouchEvent> all_data;
    std::ifstream file(file_path);
    
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << file_path << std::endl;
        return all_data;
    }
    
    std::string line;
    // Skip the first line (header)
    std::getline(file, line);
    
    // Read column names from second line to determine column indices
    std::getline(file, line);
    std::istringstream header_iss(line);
    std::vector<std::string> column_names;
    std::string col_name;
    while (header_iss >> col_name) {
        column_names.push_back(col_name);
    }
    column_names.push_back("event_ID");
    column_names.push_back("trace_ID");
    
    // Find indices of important columns
    for (size_t i = 0; i < column_names.size(); i++) {
        if (column_names[i] == "bundleName") col_idx.bundleName_idx = i;
        else if (column_names[i] == "TYPE") col_idx.type_idx = i;
        else if (column_names[i] == "event_ID") col_idx.event_ID_idx = i;
        else if (column_names[i] == "POSX") col_idx.posX_idx = i;
        else if (column_names[i] == "POSY") col_idx.posY_idx = i;
        else if (column_names[i] == "timestamp") col_idx.time_idx = i;
        else if (column_names[i] == "duration") col_idx.duration_idx = i;
        else if (column_names[i] == "trace_ID") col_idx.trace_ID_idx = i;
    }
    
    // Parse all data lines
    std::string prev_bundleName = "";
    int event_counter = 0;
    int line_num = 0;
    
    while (std::getline(file, line)) {
        TouchEvent event = parseLine(line, line_num, prev_bundleName, event_counter);
        all_data.push_back(event);
        prev_bundleName = event.bundleName;
        line_num++;
    }
    
    file.close();
    return all_data;
}

// Filter events for a specific application
std::vector<TouchEvent> filterByApp(const std::vector<TouchEvent>& all_data, 
                                     const std::string& target_app) {
    std::vector<TouchEvent> interested_data;
    
    for (const auto& event : all_data) {
        if (event.bundleName == target_app) {
            interested_data.push_back(event);
        }
    }
    
    return interested_data;
}

// Get accessor functions for TouchEvent fields based on column index
// These functions map column indices to actual struct members
int getIntField(const TouchEvent& event, int col_idx) {
    switch(col_idx) {
        case 0: return event.field0;
        case 1: return event.field1;
        case 2: return event.field2;
        case 3: return event.field3;
        case 4: return event.field4;
        case 5: return event.field5;
        case 7: return event.field7;
        case 10: return event.field10;
        default: return 0;
    }
}

float getFloatField(const TouchEvent& event, int col_idx) {
    switch(col_idx) {
        case 6: return event.field6;
        case 8: return event.field8;
        case 9: return event.field9;
        default: return 0.0f;
    }
}

// Build traces from touch events
// A trace represents a sequence of touch operations in a single app session
// Traces are divided based on:
// 1. Event ID resetting to 0 (new app session)
// 2. Classification of operations as touch (small shift) or scroll (large shift)
std::map<int, std::vector<TouchEvent>> buildTraces(
    std::vector<TouchEvent>& interested_data,
    const ColumnIndices& col_idx,
    std::vector<TouchEvent>& traces_flat) {
    
    std::map<int, std::vector<TouchEvent>> traces;
    
    // Separate TYPE 1 (touch down) and TYPE 3 (touch up) events
    std::vector<int> type1_indices, type3_indices;
    for (size_t i = 0; i < interested_data.size(); i++) {
        int type = getIntField(interested_data[i], col_idx.type_idx);
        if (type == 1) type1_indices.push_back(i);
        else if (type == 3) type3_indices.push_back(i);
    }
    
    // Calculate shifting between TYPE 1 (touch down) and TYPE 3 (touch up)
    // Small shifting indicates a simple touch/click operation
    // Large shifting indicates a scroll/swipe operation
    std::vector<float> shifting;
    for (size_t i = 0; i < type1_indices.size(); i++) {
        int idx1 = type1_indices[i];
        int idx3 = type3_indices[i];
        
        float posX1 = getFloatField(interested_data[idx1], col_idx.posX_idx);
        float posY1 = getFloatField(interested_data[idx1], col_idx.posY_idx);
        float posX3 = getFloatField(interested_data[idx3], col_idx.posX_idx);
        float posY3 = getFloatField(interested_data[idx3], col_idx.posY_idx);
        
        float shift = std::abs(posX3 - posX1) + std::abs(posY3 - posY1);
        shifting.push_back(shift);
    }
    
    // Build traces by grouping consecutive operations
    int trace_id = 0;
    std::vector<TouchEvent> current_trace;
    
    for (size_t j = 0; j < shifting.size(); j++) {
        int idx_start = type1_indices[j];
        int idx_end = type3_indices[j];
        float shift = shifting[j];
        
        // Check if this is a new trace (event_ID reset to 0)
        if (interested_data[idx_start].event_ID == 0) {
            if (!current_trace.empty()) {
                traces[trace_id] = current_trace;
                trace_id++;
            }
            current_trace.clear();
        }
        
        // Update duration for the touch up event
        interested_data[idx_end].duration = 
            getFloatField(interested_data[idx_end], col_idx.time_idx) - 
            getFloatField(interested_data[idx_start], col_idx.time_idx);
        
        // Classify operation based on shifting threshold
        // shift < 10: Simple touch/click operation (only add start and end)
        // shift >= 10: Scroll operation (add start, middle, and end points)
        if (shift < 10) {
            // Simple touch operation
            interested_data[idx_start].trace_ID = trace_id;
            interested_data[idx_end].trace_ID = trace_id;
            
            current_trace.push_back(interested_data[idx_start]);
            current_trace.push_back(interested_data[idx_end]);
            
            traces_flat.push_back(interested_data[idx_start]);
            traces_flat.push_back(interested_data[idx_end]);
        } else {
            // Scroll operation - include middle point for better trajectory representation
            int idx_mid = (idx_start + idx_end) / 2;
            
            interested_data[idx_start].trace_ID = trace_id;
            interested_data[idx_mid].trace_ID = trace_id;
            interested_data[idx_end].trace_ID = trace_id;
            
            current_trace.push_back(interested_data[idx_start]);
            current_trace.push_back(interested_data[idx_mid]);
            current_trace.push_back(interested_data[idx_end]);
            
            traces_flat.push_back(interested_data[idx_start]);
            traces_flat.push_back(interested_data[idx_mid]);
            traces_flat.push_back(interested_data[idx_end]);
        }
    }
    
    // Add the last trace
    if (!current_trace.empty()) {
        traces[trace_id] = current_trace;
    }
    
    return traces;
}

// Structure to hold clustering results
// This would be populated by your existing DBSCAN implementation
struct ClusteringResult {
    std::vector<int> labels;  // Cluster label for each point (-1 for noise)
    std::set<int> cluster_ids; // Set of all unique cluster IDs
};

// Placeholder function for DBSCAN clustering
// Replace this with your actual DBSCAN implementation
ClusteringResult performDBSCANClustering(
    const std::vector<TouchEvent>& traces_flat,
    const ColumnIndices& col_idx,
    int num_traces,
    float eps = 60.0f,
    float min_samples_ratio = 0.8f) {
    
    ClusteringResult result;
    
    // TODO: Implement or call your DBSCAN clustering here
    // Input: traces_flat with features [posX, posY, event_ID]
    // Parameters: eps=60, min_samples=num_traces * 0.8
    // 
    // The clustering should group touch events that:
    // - Occur at similar positions (posX, posY)
    // - Have similar sequential order (event_ID)
    // 
    // This identifies frequent operations that users perform repeatedly
    // across different app sessions (traces)
    
    std::cout << "DBSCAN clustering to be implemented with your existing code" << std::endl;
    std::cout << "Expected input features: posX, posY, event_ID" << std::endl;
    std::cout << "Parameters: eps=" << eps << ", min_samples=" << (int)(num_traces * min_samples_ratio) << std::endl;
    
    // For now, return empty result
    result.labels.resize(traces_flat.size(), -1);
    
    return result;
}

// Filter clusters using voting method
// Only clusters that appear in a sufficient number of different traces (days)
// are considered as true frequent operations
std::map<int, std::vector<int>> filterClustersByVoting(
    const std::vector<TouchEvent>& traces_flat,
    const ClusteringResult& clustering,
    const ColumnIndices& col_idx,
    int num_traces,
    float voting_threshold_ratio = 0.8f) {
    
    // Calculate voting threshold
    // Only clusters appearing in at least this many different traces are valid
    int voting_tau = static_cast<int>(num_traces * voting_threshold_ratio);
    
    std::map<int, std::vector<int>> cluster_voting;
    
    // For each cluster, collect statistics:
    // [min_event_ID, max_event_ID, num_unique_traces]
    for (int cluster_id : clustering.cluster_ids) {
        if (cluster_id == -1) continue;  // Skip noise points
        
        std::vector<int> event_ids;
        std::set<int> unique_traces;
        
        // Gather all points in this cluster
        for (size_t i = 0; i < clustering.labels.size(); i++) {
            if (clustering.labels[i] == cluster_id) {
                event_ids.push_back(traces_flat[i].event_ID);
                unique_traces.insert(traces_flat[i].trace_ID);
            }
        }
        
        // Store cluster statistics: [min_event_ID, max_event_ID, num_unique_traces]
        int min_event_id = *std::min_element(event_ids.begin(), event_ids.end());
        int max_event_id = *std::max_element(event_ids.begin(), event_ids.end());
        int num_traces_with_cluster = unique_traces.size();
        
        cluster_voting[cluster_id] = {min_event_id, max_event_id, num_traces_with_cluster};
    }
    
    // Sort clusters by their sequential order (min_event_ID)
    // This maintains the natural operation order in the extracted sequence
    std::vector<int> valid_clusters;
    for (const auto& pair : cluster_voting) {
        int cluster_id = pair.first;
        int num_traces_with_cluster = pair.second[2];
        
        // Only include clusters that appear in enough different traces
        if (num_traces_with_cluster >= voting_tau) {
            valid_clusters.push_back(cluster_id);
        }
    }
    
    // Sort valid clusters by operation order (min_event_ID)
    std::sort(valid_clusters.begin(), valid_clusters.end(),
        [&cluster_voting](int a, int b) {
            return cluster_voting[a][0] < cluster_voting[b][0];
        });
    
    // Create filtered map with only valid clusters
    std::map<int, std::vector<int>> filtered_voting;
    for (int cluster_id : valid_clusters) {
        filtered_voting[cluster_id] = cluster_voting[cluster_id];
    }
    
    return filtered_voting;
}

// Extract frequent operation traces from validated clusters
// Computes centroid positions and average durations for each frequent operation
std::vector<TouchEvent> extractFrequentTraces(
    const std::vector<TouchEvent>& traces_flat,
    const ClusteringResult& clustering,
    const std::map<int, std::vector<int>>& valid_clusters,
    const ColumnIndices& col_idx) {
    
    std::vector<TouchEvent> frequent_traces;
    
    // Process each valid cluster in sequential order
    for (const auto& pair : valid_clusters) {
        int cluster_id = pair.first;
        
        // Collect all events in this cluster, excluding TYPE 2 (move events)
        std::vector<TouchEvent> cluster_events;
        for (size_t i = 0; i < clustering.labels.size(); i++) {
            if (clustering.labels[i] == cluster_id) {
                int event_type = getIntField(traces_flat[i], col_idx.type_idx);
                if (event_type != 2) {  // Exclude move events
                    cluster_events.push_back(traces_flat[i]);
                }
            }
        }
        
        if (cluster_events.empty()) continue;
        
        // Calculate centroid position (average of all positions in cluster)
        // This represents the typical location where this operation occurs
        float sum_posX = 0.0f, sum_posY = 0.0f;
        float sum_duration = 0.0f;
        int duration_count = 0;
        
        for (const auto& event : cluster_events) {
            sum_posX += getFloatField(event, col_idx.posX_idx);
            sum_posY += getFloatField(event, col_idx.posY_idx);
            
            // Only TYPE 3 events have meaningful duration
            if (getIntField(event, col_idx.type_idx) == 3) {
                sum_duration += event.duration;
                duration_count++;
            }
        }
        
        float centroid_x = sum_posX / cluster_events.size();
        float centroid_y = sum_posY / cluster_events.size();
        float avg_duration = (duration_count > 0) ? (sum_duration / duration_count) : 0.0f;
        
        // Create TYPE 1 event (touch down) with centroid position
        TouchEvent event1 = cluster_events[0];
        event1.field1 = static_cast<int>(centroid_x);  // posX
        event1.field2 = static_cast<int>(centroid_y);  // posY
        event1.field7 = 1;  // TYPE = 1
        
        // Create TYPE 3 event (touch up) with centroid position and average duration
        TouchEvent event3 = cluster_events.size() > 1 ? cluster_events[1] : cluster_events[0];
        event3.field1 = static_cast<int>(centroid_x);  // posX
        event3.field2 = static_cast<int>(centroid_y);  // posY
        event3.field7 = 3;  // TYPE = 3
        event3.duration = avg_duration;
        
        // Add the pair of events representing this frequent operation
        frequent_traces.push_back(event1);
        frequent_traces.push_back(event3);
    }
    
    return frequent_traces;
}

// Save frequent traces to output file for future replay
void saveFrequentTraces(const std::vector<TouchEvent>& frequent_traces,
                        const std::string& output_path,
                        const ColumnIndices& col_idx) {
    std::ofstream outfile(output_path);
    
    if (!outfile.is_open()) {
        std::cerr << "Error: Cannot create output file " << output_path << std::endl;
        return;
    }
    
    // Write header
    outfile << "# Frequent Touch Operation Traces\n";
    outfile << "# Format: TYPE POSX POSY DURATION BUNDLENAME\n";
    outfile << "# Each operation consists of two events: TYPE 1 (down) and TYPE 3 (up)\n\n";
    
    // Write each event
    for (const auto& event : frequent_traces) {
        int type = getIntField(event, col_idx.type_idx);
        float posX = getFloatField(event, col_idx.posX_idx);
        float posY = getFloatField(event, col_idx.posY_idx);
        
        outfile << type << " " 
                << posX << " " 
                << posY << " " 
                << event.duration << " "
                << event.bundleName << "\n";
    }
    
    outfile.close();
    std::cout << "Frequent traces saved to: " << output_path << std::endl;
}

int main() {
    // Configuration
    std::string root_directory = "D:\\agent\\trajectory\\data\\navigation_company\\2025-11-19-09-32-25";
    std::string file_path = root_directory + "\\touchEvent_0.txt";
    std::string target_app = "com.amap.hmapp";  // Target application to analyze
    std::string output_path = root_directory + "\\frequent_traces.txt";
    
    // Column indices will be populated during file loading
    ColumnIndices col_idx;
    
    // Step 1: Load all touch events from file
    std::cout << "Loading touch events from: " << file_path << std::endl;
    std::vector<TouchEvent> all_data = loadTouchEvents(file_path, col_idx);
    std::cout << "Loaded " << all_data.size() << " events" << std::endl;
    
    if (all_data.empty()) {
        std::cerr << "No data loaded. Exiting." << std::endl;
        return 1;
    }
    
    // Step 2: Filter for interested application
    std::cout << "Filtering events for app: " << target_app << std::endl;
    std::vector<TouchEvent> interested_data = filterByApp(all_data, target_app);
    std::cout << "Filtered to " << interested_data.size() << " events" << std::endl;
    
    if (interested_data.empty()) {
        std::cerr << "No events found for target app. Exiting." << std::endl;
        return 1;
    }
    
    // Step 3: Build traces by grouping operations and classifying touch vs scroll
    std::cout << "Building traces..." << std::endl;
    std::vector<TouchEvent> traces_flat;
    std::map<int, std::vector<TouchEvent>> traces = buildTraces(interested_data, col_idx, traces_flat);
    std::cout << "Built " << traces.size() << " traces with " << traces_flat.size() << " total events" << std::endl;
    
    // Step 4: Perform DBSCAN clustering (use your existing implementation)
    std::cout << "\nPerforming DBSCAN clustering..." << std::endl;
    ClusteringResult clustering = performDBSCANClustering(traces_flat, col_idx, traces.size());
    
    // Step 5: Filter clusters using voting method
    // Only operations that appear frequently across multiple sessions are kept
    std::cout << "Filtering clusters by voting..." << std::endl;
    std::map<int, std::vector<int>> valid_clusters = filterClustersByVoting(
        traces_flat, clustering, col_idx, traces.size());
    std::cout << "Found " << valid_clusters.size() << " valid frequent operation clusters" << std::endl;
    
    // Step 6: Extract frequent traces with averaged positions and durations
    std::cout << "Extracting frequent traces..." << std::endl;
    std::vector<TouchEvent> frequent_traces = extractFrequentTraces(
        traces_flat, clustering, valid_clusters, col_idx);
    std::cout << "Extracted " << frequent_traces.size() / 2 << " frequent operations" << std::endl;
    
    // Step 7: Save results for future replay
    saveFrequentTraces(frequent_traces, output_path, col_idx);
    
    std::cout << "\nProcessing complete!" << std::endl;
    std::cout << "Note: Integrate your DBSCAN implementation in performDBSCANClustering()" << std::endl;
    std::cout << "Note: Integrate your plotting code to visualize results" << std::endl;
    
    return 0;
}
