#ifndef TOUCH_EVENT_PROCESSOR_H
#define TOUCH_EVENT_PROCESSOR_H

#include <vector>
#include <map>
#include <set>
#include <string>

// Structure to hold touch event data
struct TouchEvent {
    int field0, field1, field2, field3, field4, field5;
    float field6;
    int field7;
    float field8, field9;
    int field10;
    std::string field11, field12, field13, bundleName;
    int event_ID;
    int trace_ID;
    float duration;
    
    TouchEvent();
};

// Structure to hold column indices for data access
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

// Structure to hold clustering results from DBSCAN
struct ClusteringResult {
    std::vector<int> labels;      // Cluster label for each point (-1 for noise)
    std::set<int> cluster_ids;    // Set of all unique cluster IDs
};

// Function declarations

// Load all touch events from the input file
std::vector<TouchEvent> loadTouchEvents(const std::string& file_path, ColumnIndices& col_idx);

// Filter events for a specific application
std::vector<TouchEvent> filterByApp(const std::vector<TouchEvent>& all_data, 
                                    const std::string& target_app);

// Build traces from touch events
std::map<int, std::vector<TouchEvent>> buildTraces(
    std::vector<TouchEvent>& interested_data,
    const ColumnIndices& col_idx,
    std::vector<TouchEvent>& traces_flat);

// Perform DBSCAN clustering (integrate your implementation here)
ClusteringResult performDBSCANClustering(
    const std::vector<TouchEvent>& traces_flat,
    const ColumnIndices& col_idx,
    int num_traces,
    float eps = 60.0f,
    float min_samples_ratio = 0.8f);

// Filter clusters using voting method
std::map<int, std::vector<int>> filterClustersByVoting(
    const std::vector<TouchEvent>& traces_flat,
    const ClusteringResult& clustering,
    const ColumnIndices& col_idx,
    int num_traces,
    float voting_threshold_ratio = 0.8f);

// Extract frequent operation traces from validated clusters
std::vector<TouchEvent> extractFrequentTraces(
    const std::vector<TouchEvent>& traces_flat,
    const ClusteringResult& clustering,
    const std::map<int, std::vector<int>>& valid_clusters,
    const ColumnIndices& col_idx);

// Save frequent traces to output file
void saveFrequentTraces(const std::vector<TouchEvent>& frequent_traces,
                       const std::string& output_path,
                       const ColumnIndices& col_idx);

// Field accessor functions
int getIntField(const TouchEvent& event, int col_idx);
float getFloatField(const TouchEvent& event, int col_idx);

#endif // TOUCH_EVENT_PROCESSOR_H
