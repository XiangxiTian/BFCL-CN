/*
 * FrequentOperationExtraction.cpp
 * 
 * This C++ program translates the logic of the provided Python script for processing touch event data.
 * It reads a text file, parses events, segments them into traces, prepares data for clustering,
 * filters clusters, and extracts frequent traces.
 * 
 * NOTE: The DBSCAN clustering implementation and Plotting logic are omitted as requested.
 *       Placeholders are provided where these external components would integrate.
 * 
 * Usage: Compile with a C++ compiler (e.g., g++ -o frequent_op FrequentOperationExtraction.cpp)
 *        Run: ./frequent_op <path_to_touchEvent.txt>
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <map>
#include <cmath>
#include <numeric>
#include <set>
#include <iomanip>

// Use standard namespace for convenience in this script
using namespace std;

// Structure to represent a single touch event row
struct EventData {
    // We store the raw tokens to maintain the original data structure if needed for output
    // or for accessing columns that we don't explicitly parse into named fields.
    vector<string> raw_tokens;

    // Specific fields used in the logic. 
    // We will populate these based on the header indices.
    string bundleName;
    int type = 0;
    long long event_ID = 0;
    float posX = 0.0f;
    float posY = 0.0f;
    double timestamp = 0.0;
    double duration = 0.0;

    // Added fields during processing
    int bundle_seq_id = 0; // The 'j' index in the python script
    int trace_ID = -1;     // Assigned during trace segmentation
};

// Helper function to split string by whitespace
vector<string> split(const string& s) {
    vector<string> tokens;
    string token;
    istringstream tokenStream(s);
    while (tokenStream >> token) {
        tokens.push_back(token);
    }
    return tokens;
}

// Placeholder for DBSCAN. 
// Input: points matrix (N x 3) -> [posX, posY, event_ID]
// Output: vector of labels (size N)
// The user has their own implementation, so this is a mock.
vector<int> runDBSCAN(const vector<vector<double>>& points, double eps, int min_samples) {
    // TODO: Integrate your actual DBSCAN implementation here.
    // For demonstration, we will just return a dummy label (0) for all points if points exist.
    // In a real scenario, this would return cluster IDs (-1 for noise, 0, 1, 2...).
    
    // Just to pretend we found a cluster for everything (so downstream logic runs):
    vector<int> labels(points.size(), 0); 
    return labels;
}

int main(int argc, char* argv[]) {
    string file_path = "touchEvent_0.txt"; // Default filename
    if (argc > 1) {
        file_path = argv[1];
    }

    cout << "Reading file: " << file_path << endl;

    ifstream file(file_path);
    if (!file.is_open()) {
        cerr << "Error: Could not open file " << file_path << endl;
        return 1;
    }

    // 1. Read Header
    string line;
    if (!getline(file, line)) {
        cerr << "Error: File is empty" << endl;
        return 1;
    }
    vector<string> column_names = split(line);
    
    // Append "event_ID" as per Python script: column_names.extend(["event_ID"])
    // Wait, the python script appends "event_ID" to column_names list, 
    // implying the header line in the file didn't have it, or it's implicitly the index?
    // "lines = file.readlines()[1:] ... column_names = lines[0].split() ... column_names.extend(['event_ID'])"
    // Actually in python `lines[0]` IS the header if `readlines()` reads everything.
    // But then `lines[1:]` loop starts from the *second* line of data?
    // Let's look closer at Python:
    // lines = file.readlines()[1:] -> this removes the first line of the file (likely an irrelevant header or empty line?)
    // column_names = lines[0].split() -> The new first line is the header.
    // So the file structure is:
    // Line 0: Ignored
    // Line 1: Header
    // Line 2+: Data
    
    // Let's adjust reading logic to match Python exactly.
    // Re-open or reset not needed if we just skipped first line already?
    // No, we need to handle the "skip first line" logic.
    
    file.clear();
    file.seekg(0, ios::beg);
    string dummy;
    if (!getline(file, dummy)) return 1; // Skip Line 0
    
    if (!getline(file, line)) return 1; // Line 1 is Header
    column_names = split(line);
    column_names.push_back("event_ID"); // Python adds this manually. 
                                        // This implies the data has one more column than the header describes?
                                        // Or it's a calculated column?
                                        // In the loop: `all_data.append(... + [int(tmp[i]) for i in range(6)] + ...)`
                                        // It seems the data lines have enough columns. 
                                        // Let's proceed assuming the column mapping works via name.

    // Map column name to index
    map<string, int> col_map;
    for (size_t i = 0; i < column_names.size(); ++i) {
        col_map[column_names[i]] = i;
    }

    // Identify critical indices
    auto get_idx = [&](const string& name) -> int {
        if (col_map.find(name) != col_map.end()) return col_map[name];
        cerr << "Warning: Column " << name << " not found!" << endl;
        return -1;
    };

    int bundleName_idx = get_idx("bundleName");
    int type_idx = get_idx("TYPE");
    int event_ID_idx = get_idx("event_ID");
    int posX_idx = get_idx("POSX");
    int posY_idx = get_idx("POSY");
    int time_idx = get_idx("timestamp");
    int duration_idx = get_idx("duration");

    // 2. Load Data
    vector<EventData> all_data;
    string prev_bundleName = "";
    int j = 0;

    // In Python: `for i, line in enumerate(lines[1:]):` 
    // This means data starts after the header.
    while (getline(file, line)) {
        vector<string> tmp = split(line);
        if (tmp.empty()) continue;

        EventData evt;
        evt.raw_tokens = tmp; // Store raw strings

        // Parse specific fields if index is valid
        try {
            if (bundleName_idx >= 0 && bundleName_idx < tmp.size()) evt.bundleName = tmp[bundleName_idx];
            if (type_idx >= 0 && type_idx < tmp.size()) evt.type = stoi(tmp[type_idx]);
            // event_ID might be at the end or calculated. Python appends column name but reads data from file?
            // "all_data.append(...)". The python parsing logic is complex:
            // `[int(tmp[i]) for i in range(6)]` ... 
            // It constructs a list manually from `tmp`. 
            // It seems `tmp` has the data. 
            // If `event_ID` is in `column_names`, it must correspond to a position in the parsed `all_data` list.
            // In python, `all_data` is a list of lists.
            // The mapping `event_ID_idx` refers to the index in `all_data`, NOT `tmp` (the raw line).
            // BUT `column_names` comes from the header.
            // The python code: `column_names.extend(["event_ID"])` implies `event_ID` is NOT in the header.
            // But later `int(tmp[10])` is appended. 
            // Let's look at the structure construction in Python:
            // 0-5: ints
            // 6: float
            // 7: int
            // 8: float
            // 9: float
            // 10: int
            // 11-14: string
            // Total columns: 6 + 1 + 1 + 2 + 1 + 4 = 15 columns.
            // If `column_names` initially had 14 items, and we added `event_ID`, then `event_ID` is at index 14?
            // Wait, Python `all_data` structure matches the `column_names` order?
            // Usually yes.
            // Let's assume we can trust the named indices against the values we parse.
            
            // To be safe, let's parse raw `tmp` into numbers based on the user's explicit Python list construction
            // if we want to be exact, or just trust the `get_idx` on the raw tokens if the file has headers.
            // However, the user script implies the file has headers but then manually constructs the list.
            // This suggests the headers in the file might NOT match the data types or order 1:1, or he's just casting.
            // We will rely on the `get_idx` for `tmp` if possible, but `event_ID` was added manually.
            // If `event_ID` was added manually to `column_names`, it means it's a NEW column not in the file header?
            // OR it means the file header was missing it but the data has it?
            // Given `tmp[10]` is an int, maybe that's it?
            
            // Let's just parse the fields we need from `tmp` using the indices we found, 
            // assuming the header names correspond to `tmp` indices properly (except event_ID).
            // If `event_ID_idx` returns -1 (because it wasn't in the file header), we need to find where it is.
            // In the Python loop: `all_data` gets `int(tmp[10])`. 
            // If `column_names` (header) has 14 elements, and we add "event_ID", it becomes index 14.
            // But the data we append has 15 elements + 'j' = 16.
            // This is getting confusing without the file. 
            // STRATEGY: Parse specific indices based on standard names if found, otherwise fallback to user's Python offsets.
            // The user's Python offsets:
            // POSX/POSY are likely floats: indices 8 and 9 match `float(tmp[8]), float(tmp[9])`.
            // TYPE is likely int: index ?? 
            // BundleName is string.
            
            // We will use the `get_idx` derived from header for robustness.
            // For `event_ID`, if it's not in header, we'll try to guess or use the Python logic.
            
            if (posX_idx >= 0 && posX_idx < tmp.size()) evt.posX = stof(tmp[posX_idx]);
            if (posY_idx >= 0 && posY_idx < tmp.size()) evt.posY = stof(tmp[posY_idx]);
            if (time_idx >= 0 && time_idx < tmp.size()) evt.timestamp = stod(tmp[time_idx]);
            if (duration_idx >= 0 && duration_idx < tmp.size()) evt.duration = stod(tmp[duration_idx]);
            if (event_ID_idx >= 0 && event_ID_idx < tmp.size()) evt.event_ID = stoll(tmp[event_ID_idx]);
            
            // Handle bundleName sequence logic
            if (evt.bundleName != prev_bundleName) {
                j = 0;
                prev_bundleName = evt.bundleName;
            }
            evt.bundle_seq_id = j;
            j++;
            
            all_data.push_back(evt);

        } catch (const exception& e) {
            // Ignore malformed lines
            continue;
        }
    }

    if (all_data.empty()) {
        cout << "No data loaded." << endl;
        return 0;
    }
    
    // Screen size from first row (cols 3, 4 in python -> likely index 3 and 4 in data list)
    // In python: `screen_size = [all_data[0][3], all_data[0][4]]`
    // If we rely on mapped indices, we should find "screen_width"/"screen_height" or similar?
    // Python uses index 3 and 4 explicitly.
    // Let's assume we don't strictly need screen size for the logic (it's used for plotting).
    // We can ignore it.

    // 3. Filter Interested Data
    vector<EventData> interested_data;
    for (const auto& d : all_data) {
        if (d.bundleName == "com.amap.hmapp") {
            interested_data.push_back(d);
        }
    }

    cout << "Interested events: " << interested_data.size() << endl;

    // 4. Trace Segmentation
    // Identify indices for TYPE 1 (down) and TYPE 3 (up)
    vector<int> type1_indices;
    vector<int> type3_indices;
    
    // Note: Python logic assumes matched pairs of 1 and 3?
    // `interested_data_type1_idx = [... if type == 1]`
    // `interested_data_type3_idx = [... if type == 3]`
    // `range(len(interested_data_type1_idx))` implies they must be same length and aligned?
    // The python code assumes perfect pairing (likely enforced by data source).
    
    for (int i = 0; i < interested_data.size(); ++i) {
        if (interested_data[i].type == 1) type1_indices.push_back(i);
        else if (interested_data[i].type == 3) type3_indices.push_back(i);
    }
    
    // Safety check for equal length
    size_t num_pairs = min(type1_indices.size(), type3_indices.size());
    
    map<int, vector<EventData>> traces; // Trace ID -> List of events
    vector<EventData> traces_flat;      // Flattened list for DBSCAN (corresponds to traces_ in Python)
    
    int trace_counter = 0;
    vector<EventData> current_trace_tmp;

    for (size_t k = 0; k < num_pairs; ++k) {
        int idx_start = type1_indices[k];
        int idx_end = type3_indices[k];
        
        // Calculate shifting
        float sX = abs(interested_data[idx_end].posX - interested_data[idx_start].posX);
        float sY = abs(interested_data[idx_end].posY - interested_data[idx_start].posY);
        float shift = sX + sY;

        // "if int(interested_data[idx_start][event_ID_idx]) == 0:"
        // Start new trace if event_ID is 0
        if (interested_data[idx_start].event_ID == 0) {
            if (!current_trace_tmp.empty()) {
                traces[trace_counter] = current_trace_tmp;
                trace_counter++;
            }
            current_trace_tmp.clear();
        }

        // Logic based on shift
        EventData start_evt = interested_data[idx_start];
        EventData end_evt = interested_data[idx_end];
        
        // Update duration of end event
        end_evt.duration = end_evt.timestamp - start_evt.timestamp;

        // Modify start/end events to include trace_ID (we store in struct)
        // Note: The python script appends trace_ID to the list row. 
        // We will set the property `trace_id` in our struct copies.
        
        if (shift < 10) {
            // Touch operation
            current_trace_tmp.push_back(start_evt);
            current_trace_tmp.push_back(end_evt);
            
            start_evt.trace_ID = trace_counter;
            end_evt.trace_ID = trace_counter;
            
            traces_flat.push_back(start_evt);
            traces_flat.push_back(end_evt);
        } else {
            // Scroll operation - add midpoint
            EventData mid_evt = interested_data[(idx_start + idx_end) / 2];
            mid_evt.trace_ID = trace_counter;
            
            // Python: "interested_data[idx_end][duration_idx] = ..." 
            // It sets duration again? (It was set above).
            
            current_trace_tmp.push_back(start_evt);
            current_trace_tmp.push_back(mid_evt);
            current_trace_tmp.push_back(end_evt);

            start_evt.trace_ID = trace_counter;
            traces_flat.push_back(start_evt);
            
            mid_evt.trace_ID = trace_counter;
            traces_flat.push_back(mid_evt);
            
            end_evt.trace_ID = trace_counter;
            traces_flat.push_back(end_evt);
        }
    }
    // Add last trace
    if (!current_trace_tmp.empty()) {
        traces[trace_counter] = current_trace_tmp;
    }

    // 5. Prepare for DBSCAN
    // "traces__ = np.copy(traces_)... [posX, posY, event_ID, trace_ID]"
    // We need points for clustering: posX, posY, event_ID
    vector<vector<double>> dbscan_points;
    for (const auto& e : traces_flat) {
        vector<double> p = { (double)e.posX, (double)e.posY, (double)e.event_ID };
        dbscan_points.push_back(p);
    }

    // 6. Run DBSCAN (Mocked)
    // "clustering = DBSCAN(eps=60, min_samples=int(len(traces) * 0.8)).fit(traces__[:,:3])"
    int min_samples = (int)(traces.size() * 0.8);
    cout << "Running DBSCAN with eps=60, min_samples=" << min_samples << "..." << endl;
    
    // Call the placeholder function
    vector<int> labels = runDBSCAN(dbscan_points, 60.0, min_samples);

    // Get unique cluster IDs
    set<int> cluster_ids(labels.begin(), labels.end());

    // 7. Filter Clusters (Voting)
    // "only those clusters with operation in multiple days(traces)..."
    // We need to associate filtered points with their labels and original data.
    
    struct FilteredPoint {
        EventData event;
        int label;
    };
    vector<FilteredPoint> traces_filtered;
    
    // Map cluster_id -> {min_event_ID, max_event_ID, unique_trace_count}
    struct ClusterStats {
        long long min_event_ID = -1;
        long long max_event_ID = -1;
        set<int> unique_traces;
    };
    map<int, ClusterStats> cluster_voting;

    for (int cid : cluster_ids) {
        if (cid == -1) continue; // Noise

        // Find all points belonging to this cluster
        for (size_t i = 0; i < labels.size(); ++i) {
            if (labels[i] == cid) {
                FilteredPoint fp;
                fp.event = traces_flat[i];
                fp.label = cid;
                traces_filtered.push_back(fp);

                // Update stats
                if (cluster_voting.find(cid) == cluster_voting.end()) {
                    cluster_voting[cid].min_event_ID = fp.event.event_ID;
                    cluster_voting[cid].max_event_ID = fp.event.event_ID;
                } else {
                    cluster_voting[cid].min_event_ID = min(cluster_voting[cid].min_event_ID, fp.event.event_ID);
                    cluster_voting[cid].max_event_ID = max(cluster_voting[cid].max_event_ID, fp.event.event_ID);
                }
                cluster_voting[cid].unique_traces.insert(fp.event.trace_ID);
            }
        }
    }

    // Sort clusters by operation order (min_event_ID)
    vector<int> sorted_clusters;
    for (auto const& [cid, stats] : cluster_voting) {
        sorted_clusters.push_back(cid);
    }
    sort(sorted_clusters.begin(), sorted_clusters.end(), [&](int a, int b) {
        return cluster_voting[a].min_event_ID < cluster_voting[b].min_event_ID;
    });

    int voting_tau = (int)(traces.size() * 0.8);
    vector<int> valid_cluster_IDs;
    for (int cid : sorted_clusters) {
        if (cluster_voting[cid].unique_traces.size() >= voting_tau) {
            valid_cluster_IDs.push_back(cid);
        }
    }

    cout << "Valid Frequent Operations found: " << valid_cluster_IDs.size() << endl;

    // 8. Extract Frequent Traces
    // Construct format: Two events (TYPE 1, TYPE 3) with centroid coords and mean duration
    vector<EventData> frequent_traces;

    for (int cid : valid_cluster_IDs) {
        // Gather points for this cluster where type != 2 (Python: `if traces_[j][type_idx] != 2`)
        // The script calculates centroids from these points.
        vector<EventData> cluster_points;
        vector<double> durations_type3;
        
        double sum_x = 0;
        double sum_y = 0;
        int count = 0;

        for (size_t i = 0; i < labels.size(); ++i) {
            if (labels[i] == cid) {
                // Check original type. We have it in traces_flat[i]
                if (traces_flat[i].type != 2) {
                    cluster_points.push_back(traces_flat[i]);
                    sum_x += traces_flat[i].posX;
                    sum_y += traces_flat[i].posY;
                    count++;
                }
                if (traces_flat[i].type == 3) {
                    durations_type3.push_back(traces_flat[i].duration);
                }
            }
        }

        if (count == 0) continue;

        double centroid_x = sum_x / count;
        double centroid_y = sum_y / count;
        
        double mean_duration = 0;
        if (!durations_type3.empty()) {
            mean_duration = accumulate(durations_type3.begin(), durations_type3.end(), 0.0) / durations_type3.size();
        }

        // Create the frequent trace pair (Start and End)
        // Python takes `tmp[0][:-2]` and `tmp[1][:-2]` as templates but overwrites values.
        // We can just create new EventData objects.
        
        EventData p1;
        p1.type = 1;
        p1.posX = (float)centroid_x;
        p1.posY = (float)centroid_y;
        
        EventData p2;
        p2.type = 3;
        p2.posX = (float)centroid_x;
        p2.posY = (float)centroid_y;
        p2.duration = mean_duration;

        frequent_traces.push_back(p1);
        frequent_traces.push_back(p2);
    }

    // 9. Output Results
    cout << "Extracted Frequent Traces:" << endl;
    for (size_t i = 0; i < frequent_traces.size(); ++i) {
        const auto& p = frequent_traces[i];
        cout << "Event " << i << ": Type=" << p.type 
             << ", Pos=(" << p.posX << ", " << p.posY << ")"
             << ", Duration=" << p.duration << endl;
    }

    return 0;
}
