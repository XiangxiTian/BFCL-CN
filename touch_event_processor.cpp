#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <iomanip>

// Data structure to represent a single touch event
// Stores all fields as strings for flexible access, with parsed values for common operations
struct TouchEvent {
    std::vector<std::string> fields;  // All fields as strings (for flexible column access)
    int event_ID;                      // Sequential event ID within bundle
    int trace_ID;                      // ID of the trace this event belongs to
    
    // Helper functions to get typed values by column index
    int getInt(int idx) const { return std::stoi(fields[idx]); }
    double getDouble(int idx) const { return std::stod(fields[idx]); }
    std::string getString(int idx) const { return fields[idx]; }
    
    // Helper function to set a field value
    void setField(int idx, const std::string& value) {
        if (idx < static_cast<int>(fields.size())) {
            fields[idx] = value;
        }
    }
    
    TouchEvent() : event_ID(0), trace_ID(0) {}
};

// ============================================================================
// DBSCAN Clustering Function - USER MUST IMPLEMENT THIS
// ============================================================================
// This is a stub implementation. Replace with your own DBSCAN implementation.
// 
// Parameters:
//   points: Vector of points, where each point is [posX, posY, event_ID] as integers
//   eps: Maximum distance between points in the same cluster (e.g., 60.0)
//   min_samples: Minimum number of points required to form a cluster
//
// Returns:
//   Vector of cluster labels for each point (-1 indicates noise/outlier)
//
// TODO: Replace this stub with your actual DBSCAN implementation
std::vector<int> performDBSCAN(const std::vector<std::vector<int>>& points, 
                                double eps, int min_samples) {
    // STUB IMPLEMENTATION - Returns all points as noise (-1)
    // Replace this with your DBSCAN algorithm
    std::vector<int> labels(points.size(), -1);
    
    // Your DBSCAN implementation should:
    // 1. Calculate distances between points (using posX, posY, event_ID)
    // 2. Identify core points (points with >= min_samples neighbors within eps)
    // 3. Expand clusters from core points
    // 4. Assign cluster IDs (0, 1, 2, ...) or -1 for noise
    
    std::cerr << "Warning: Using stub DBSCAN implementation. Please implement performDBSCAN()!" << std::endl;
    
    return labels;
}

int main() {
    // Root directory and file path - user should modify this path
    std::string root_directory = R"(D:\agent\trajectory\data\navigation_company\2025-11-19-09-32-25)";
    std::string file_path = root_directory + R"(\touchEvent_0.txt)";

    // Column names from the file header
    std::vector<std::string> column_names;
    std::vector<TouchEvent> all_data;

    // ============================================================================
    // STEP 1: Load all events from touchEvent_0.txt
    // ============================================================================
    // We read the header line first, then parse each data line
    // For each bundle name change, we reset the event_ID counter to 0
    std::ifstream file(file_path);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << file_path << std::endl;
        return 1;
    }

    std::string line;
    
    // Read header line (first line) to get column names
    std::getline(file, line);
    std::istringstream header_stream(line);
    std::string column_name;
    while (header_stream >> column_name) {
        column_names.push_back(column_name);
    }
    column_names.push_back("event_ID");  // Add event_ID column name

    // Find bundleName column index for tracking bundle changes
    int bundleName_idx = -1;
    for (size_t i = 0; i < column_names.size() - 1; i++) {
        if (column_names[i] == "bundleName") {
            bundleName_idx = static_cast<int>(i);
            break;
        }
    }

    if (bundleName_idx == -1) {
        std::cerr << "Error: bundleName column not found" << std::endl;
        return 1;
    }

    // Track previous bundle name to detect changes and reset event_ID counter
    std::string prev_bundleName = "";
    int event_counter = 0;  // Sequential counter within each bundle

    // Parse each data line
    int line_index = 0;
    while (std::getline(file, line)) {
        if (line.empty()) continue;

        std::istringstream line_stream(line);
        std::vector<std::string> tokens;
        std::string token;

        // Split line into tokens (space-separated)
        while (line_stream >> token) {
            tokens.push_back(token);
        }

        if (tokens.size() < 15) {
            std::cerr << "Warning: Line has insufficient columns, skipping" << std::endl;
            continue;
        }

        TouchEvent event;
        event.fields = tokens;  // Store all fields as strings

        // Check if bundle name changed - if so, reset event counter
        std::string current_bundleName = tokens[bundleName_idx];
        if (current_bundleName != prev_bundleName) {
            event_counter = 0;
            prev_bundleName = current_bundleName;
        }

        // Assign sequential event_ID within this bundle
        event.event_ID = event_counter;
        event_counter++;

        all_data.push_back(event);
        line_index++;
    }

    file.close();

    // Get screen size from first event (fields at indices 3 and 4)
    std::vector<int> screen_size = {all_data[0].getInt(3), all_data[0].getInt(4)};

    // ============================================================================
    // STEP 2: Filter data for interested bundle name (e.g., com.amap.hmapp)
    // ============================================================================
    std::vector<TouchEvent> interested_data;
    std::string target_bundle = "com.amap.hmapp";

    for (const auto& data : all_data) {
        if (data.getString(bundleName_idx) == target_bundle) {
            interested_data.push_back(data);
        }
    }

    // Find column indices for frequently accessed fields
    int type_idx = -1, event_ID_idx = -1, posX_idx = -1, posY_idx = -1;
    int time_idx = -1, duration_idx = -1;

    for (size_t i = 0; i < column_names.size(); i++) {
        if (column_names[i] == "TYPE") type_idx = static_cast<int>(i);
        else if (column_names[i] == "event_ID") event_ID_idx = static_cast<int>(i);
        else if (column_names[i] == "POSX") posX_idx = static_cast<int>(i);
        else if (column_names[i] == "POSY") posY_idx = static_cast<int>(i);
        else if (column_names[i] == "timestamp") time_idx = static_cast<int>(i);
        else if (column_names[i] == "duration") duration_idx = static_cast<int>(i);
    }

    // Verify all required columns found
    if (type_idx == -1 || posX_idx == -1 || posY_idx == -1 || 
        time_idx == -1 || duration_idx == -1) {
        std::cerr << "Error: Required columns not found" << std::endl;
        return 1;
    }

    // ============================================================================
    // STEP 3: Classify operations as touch or scroll
    // ============================================================================
    // Find indices of TYPE 1 events (touch start) and TYPE 3 events (touch end)
    // Calculate shifting (distance) between TYPE 1 and TYPE 3 events
    // Small shifting (< 10) indicates a touch, large shifting indicates a scroll
    std::vector<int> interested_data_type1_idx;
    std::vector<int> interested_data_type3_idx;

    for (size_t i = 0; i < interested_data.size(); i++) {
        if (interested_data[i].getInt(type_idx) == 1) {
            interested_data_type1_idx.push_back(static_cast<int>(i));
        } else if (interested_data[i].getInt(type_idx) == 3) {
            interested_data_type3_idx.push_back(static_cast<int>(i));
        }
    }

    // Calculate shifting (Manhattan distance) between corresponding TYPE 1 and TYPE 3 events
    std::vector<double> shifting;
    int min_pairs = std::min(interested_data_type1_idx.size(), interested_data_type3_idx.size());
    
    for (int i = 0; i < min_pairs; i++) {
        int idx1 = interested_data_type1_idx[i];
        int idx3 = interested_data_type3_idx[i];
        
        double shift = std::abs(interested_data[idx3].getDouble(posX_idx) - 
                               interested_data[idx1].getDouble(posX_idx)) +
                      std::abs(interested_data[idx3].getDouble(posY_idx) - 
                               interested_data[idx1].getDouble(posY_idx));
        shifting.push_back(shift);
    }

    // ============================================================================
    // STEP 4: Divide trajectory into traces
    // ============================================================================
    // Traces are separated by event_ID == 0 (start of new trace)
    // For touches (shift < 10): include TYPE 1 and TYPE 3 events
    // For scrolls (shift >= 10): include TYPE 1, middle point, and TYPE 3 events
    column_names.push_back("trace_ID");  // Add trace_ID column name
    int trace_ID_idx = static_cast<int>(column_names.size() - 1);

    std::map<int, std::vector<TouchEvent>> traces;  // Map trace_ID to events
    std::vector<TouchEvent> traces_;  // Flat list of events with trace_ID
    int current_trace_id = 0;
    std::vector<TouchEvent> current_trace;

    for (size_t j = 0; j < shifting.size(); j++) {
        int idx_start = interested_data_type1_idx[j];
        int idx_end = interested_data_type3_idx[j];
        double shift = shifting[j];

        // Check if this is the start of a new trace (event_ID == 0)
        if (interested_data[idx_start].event_ID == 0) {
            // Save previous trace if it exists
            if (!current_trace.empty()) {
                traces[current_trace_id] = current_trace;
                current_trace_id++;
            }
            current_trace.clear();
        }

        // Calculate duration for TYPE 3 event
        double duration = interested_data[idx_end].getDouble(time_idx) - 
                         interested_data[idx_start].getDouble(time_idx);

        if (shift < 10) {
            // Touch operation: add TYPE 1 and TYPE 3 events
            TouchEvent event1 = interested_data[idx_start];
            event1.trace_ID = current_trace_id;
            current_trace.push_back(event1);
            traces_.push_back(event1);

            TouchEvent event3 = interested_data[idx_end];
            event3.trace_ID = current_trace_id;
            // Update duration field for TYPE 3 event
            std::ostringstream duration_str;
            duration_str << duration;
            event3.setField(duration_idx, duration_str.str());
            current_trace.push_back(event3);
            traces_.push_back(event3);
        } else {
            // Scroll operation: add TYPE 1, middle point, and TYPE 3 events
            TouchEvent event1 = interested_data[idx_start];
            event1.trace_ID = current_trace_id;
            current_trace.push_back(event1);
            traces_.push_back(event1);

            // Middle point (average of start and end)
            int middle_idx = (idx_start + idx_end) / 2;
            TouchEvent event_mid = interested_data[middle_idx];
            event_mid.trace_ID = current_trace_id;
            current_trace.push_back(event_mid);
            traces_.push_back(event_mid);

            TouchEvent event3 = interested_data[idx_end];
            event3.trace_ID = current_trace_id;
            std::ostringstream duration_str;
            duration_str << duration;
            event3.setField(duration_idx, duration_str.str());
            current_trace.push_back(event3);
            traces_.push_back(event3);
        }
    }

    // Save the last trace
    if (!current_trace.empty()) {
        traces[current_trace_id] = current_trace;
    }

    // ============================================================================
    // STEP 5: Prepare data for DBSCAN clustering
    // ============================================================================
    // Cluster based on posX, posY, and event_ID (for sequential ordering)
    // Convert traces_ to format suitable for clustering
    std::vector<std::vector<int>> clustering_points;
    
    for (const auto& event : traces_) {
        std::vector<int> point = {
            static_cast<int>(event.getDouble(posX_idx)),
            static_cast<int>(event.getDouble(posY_idx)),
            event.event_ID
        };
        clustering_points.push_back(point);
    }

    // Perform DBSCAN clustering
    // eps=60: maximum distance between points in same cluster
    // min_samples: minimum points required to form cluster (80% of traces)
    double eps = 60.0;
    int min_samples = static_cast<int>(traces.size() * 0.8);
    std::vector<int> labels = performDBSCAN(clustering_points, eps, min_samples);

    // ============================================================================
    // STEP 6: Filter clusters using voting method
    // ============================================================================
    // Only clusters that appear in multiple traces (>= 80% of traces) are considered
    // frequent operations. This filters out noise and one-off operations.
    std::set<int> cluster_ID_set;
    for (int label : labels) {
        if (label != -1) {  // -1 is noise in DBSCAN
            cluster_ID_set.insert(label);
        }
    }

    std::vector<int> cluster_ID(cluster_ID_set.begin(), cluster_ID_set.end());

    // Build cluster voting data structure
    // For each cluster, track: [min_event_ID, max_event_ID, num_unique_traces]
    std::map<int, std::vector<int>> cluster_voting;
    std::vector<TouchEvent> traces_filtered;

    for (int cluster : cluster_ID) {
        std::vector<TouchEvent> cluster_events;
        std::set<int> unique_traces;

        // Collect all events in this cluster
        for (size_t i = 0; i < traces_.size(); i++) {
            if (labels[i] == cluster) {
                cluster_events.push_back(traces_[i]);
                unique_traces.insert(traces_[i].trace_ID);
            }
        }

        if (!cluster_events.empty()) {
            // Find min and max event_ID in this cluster
            int min_event_ID = cluster_events[0].event_ID;
            int max_event_ID = cluster_events[0].event_ID;
            for (const auto& event : cluster_events) {
                min_event_ID = std::min(min_event_ID, event.event_ID);
                max_event_ID = std::max(max_event_ID, event.event_ID);
            }

            cluster_voting[cluster] = {min_event_ID, max_event_ID, 
                                      static_cast<int>(unique_traces.size())};
            traces_filtered.insert(traces_filtered.end(), 
                                  cluster_events.begin(), cluster_events.end());
        }
    }

    // Sort clusters by min_event_ID (operation order)
    std::vector<std::pair<int, std::vector<int>>> cluster_voting_vec(
        cluster_voting.begin(), cluster_voting.end());
    std::sort(cluster_voting_vec.begin(), cluster_voting_vec.end(),
              [](const std::pair<int, std::vector<int>>& a,
                 const std::pair<int, std::vector<int>>& b) {
                  return a.second[0] < b.second[0];  // Sort by min_event_ID
              });

    // Filter clusters by voting threshold (80% of traces)
    int voting_tau = static_cast<int>(traces.size() * 0.8);
    std::vector<int> valid_cluster_ID_sorted;

    for (const auto& pair : cluster_voting_vec) {
        int cluster_id = pair.first;
        int num_traces = pair.second[2];
        if (num_traces >= voting_tau) {
            valid_cluster_ID_sorted.push_back(cluster_id);
        }
    }

    // ============================================================================
    // STEP 7: Extract frequent traces
    // ============================================================================
    // For each valid cluster, create representative TYPE 1 and TYPE 3 events
    // Use centroid (average) position and average duration
    // In Python: tmp[0][:-2] removes trace_ID and event_ID to get original event structure
    // In C++: we create new events from the original fields, excluding trace_ID
    std::vector<TouchEvent> frequent_traces;

    for (int cluster_id : valid_cluster_ID_sorted) {
        // Collect all events in this cluster (excluding TYPE 2)
        // Find TYPE 1 and TYPE 3 events separately
        std::vector<TouchEvent> cluster_events;
        TouchEvent type1_template, type3_template;
        bool found_type1 = false, found_type3 = false;

        for (size_t i = 0; i < traces_.size(); i++) {
            if (labels[i] == cluster_id && traces_[i].getInt(type_idx) != 2) {
                cluster_events.push_back(traces_[i]);
                if (traces_[i].getInt(type_idx) == 1 && !found_type1) {
                    type1_template = traces_[i];
                    found_type1 = true;
                }
                if (traces_[i].getInt(type_idx) == 3 && !found_type3) {
                    type3_template = traces_[i];
                    found_type3 = true;
                }
            }
        }

        if (cluster_events.empty() || !found_type1 || !found_type3) continue;

        // Calculate centroid (average position) - using POSX and POSY indices
        // Note: In Python code, posX_idx and posY_idx refer to column indices
        // But the actual data access uses point[1] and point[2], which suggests
        // the data structure might be different. Let's use the column indices as defined.
        double sum_x = 0.0, sum_y = 0.0;
        double sum_duration = 0.0;
        int duration_count = 0;

        for (const auto& event : cluster_events) {
            sum_x += event.getDouble(posX_idx);
            sum_y += event.getDouble(posY_idx);
            if (event.getInt(type_idx) == 3) {
                sum_duration += event.getDouble(duration_idx);
                duration_count++;
            }
        }

        double centroid_x = sum_x / cluster_events.size();
        double centroid_y = sum_y / cluster_events.size();
        double avg_duration = (duration_count > 0) ? sum_duration / duration_count : 0.0;

        // Create TYPE 1 event (touch start) with centroid position
        // Copy all fields from template, then update position and type
        TouchEvent new_event1;
        new_event1.fields = type1_template.fields;  // Copy all original fields
        std::ostringstream x_str, y_str;
        x_str << std::fixed << std::setprecision(6) << centroid_x;
        y_str << std::fixed << std::setprecision(6) << centroid_y;
        new_event1.setField(posX_idx, x_str.str());
        new_event1.setField(posY_idx, y_str.str());
        std::ostringstream type1_str;
        type1_str << 1;
        new_event1.setField(type_idx, type1_str.str());
        // Note: We don't include trace_ID in the output (Python uses [:-2] to remove it)
        frequent_traces.push_back(new_event1);

        // Create TYPE 3 event (touch end) with centroid position and average duration
        TouchEvent new_event3;
        new_event3.fields = type3_template.fields;  // Copy all original fields
        new_event3.setField(posX_idx, x_str.str());
        new_event3.setField(posY_idx, y_str.str());
        std::ostringstream type3_str, duration_str;
        type3_str << 3;
        duration_str << std::fixed << std::setprecision(6) << avg_duration;
        new_event3.setField(type_idx, type3_str.str());
        new_event3.setField(duration_idx, duration_str.str());
        // Note: We don't include trace_ID in the output
        frequent_traces.push_back(new_event3);
    }

    // ============================================================================
    // Output results (user can add their own output/plotting code here)
    // ============================================================================
    std::cout << "Processing complete!" << std::endl;
    std::cout << "Total events loaded: " << all_data.size() << std::endl;
    std::cout << "Interested events: " << interested_data.size() << std::endl;
    std::cout << "Number of traces: " << traces.size() << std::endl;
    std::cout << "Valid frequent clusters: " << valid_cluster_ID_sorted.size() << std::endl;
    std::cout << "Frequent trace events: " << frequent_traces.size() << std::endl;

    // User can add plotting code here using their existing plotting/visualization functions
    // User can add DBSCAN implementation in performDBSCAN function

    return 0;
}
