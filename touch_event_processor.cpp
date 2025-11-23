#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <map>
#include <set>
#include <algorithm>
#include <numeric>
#include <cmath>

// Structure to represent a touch event with all its attributes
struct TouchEvent {
    // Basic event attributes
    int data[6];              // First 6 integer fields
    double field6;            // 7th field (double)
    int field7;               // 8th field (int)
    double posX;              // POSX
    double posY;              // POSY
    int field10;              // 11th field (int)
    std::vector<std::string> stringFields; // String fields (11-14)
    
    // Derived attributes
    int eventID;              // Sequential event ID within the app session
    int traceID;              // ID of the trace this event belongs to
    double timestamp;         // Event timestamp
    double duration;          // Duration for TYPE 3 events
    int type;                 // Event type (1, 2, or 3)
    
    // Constructor
    TouchEvent() : eventID(0), traceID(-1), duration(0.0) {}
};

// Column indices for accessing specific fields
struct ColumnIndices {
    int bundleName;
    int type;
    int eventID;
    int posX;
    int posY;
    int timestamp;
    int duration;
    int traceID;
};

// Function to split a string by whitespace
std::vector<std::string> splitString(const std::string& str) {
    std::vector<std::string> tokens;
    std::istringstream iss(str);
    std::string token;
    while (iss >> token) {
        tokens.push_back(token);
    }
    return tokens;
}

// Function to load all events from the touchEvent file
std::vector<TouchEvent> loadTouchEvents(const std::string& filePath, std::vector<int>& screenSize) {
    std::vector<TouchEvent> allData;
    std::ifstream file(filePath);
    
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filePath << std::endl;
        return allData;
    }
    
    std::string line;
    // Skip the first line (header)
    std::getline(file, line);
    
    // Read column names from the second line
    std::getline(file, line);
    std::vector<std::string> columnNames = splitString(line);
    columnNames.push_back("event_ID");
    
    // Find bundleName index for tracking app changes
    int bundleNameIdx = std::find(columnNames.begin(), columnNames.end(), "bundleName") - columnNames.begin();
    std::string prevBundleName = "";
    int j = 0;
    
    // Process each data line
    while (std::getline(file, line)) {
        std::vector<std::string> tmp = splitString(line);
        TouchEvent event;
        
        // Parse first 6 integer fields
        for (int i = 0; i < 6; i++) {
            event.data[i] = std::stoi(tmp[i]);
        }
        
        // Parse remaining numeric fields
        event.field6 = std::stod(tmp[6]);
        event.field7 = std::stoi(tmp[7]);
        event.posX = std::stod(tmp[8]);
        event.posY = std::stod(tmp[9]);
        event.field10 = std::stoi(tmp[10]);
        
        // Store string fields
        for (int i = 11; i < 15 && i < tmp.size(); i++) {
            event.stringFields.push_back(tmp[i]);
        }
        
        // Assign event ID - resets when app changes
        if (tmp[bundleNameIdx] != prevBundleName) {
            j = 0;
            prevBundleName = tmp[bundleNameIdx];
        }
        event.eventID = j;
        j++;
        
        // Store common fields for easier access
        event.timestamp = event.field6;
        event.type = event.field7;
        
        allData.push_back(event);
    }
    
    // Extract screen size from first event
    if (!allData.empty()) {
        screenSize.push_back(allData[0].data[3]);
        screenSize.push_back(allData[0].data[4]);
    }
    
    file.close();
    return allData;
}

// Function to filter events for a specific app (bundle name)
std::vector<TouchEvent> filterByApp(const std::vector<TouchEvent>& allData, 
                                    const std::string& targetApp,
                                    const std::vector<std::string>& columnNames) {
    std::vector<TouchEvent> filteredData;
    
    // Find bundleName column index
    int bundleNameIdx = std::find(columnNames.begin(), columnNames.end(), "bundleName") - columnNames.begin();
    
    // Filter events matching the target app
    for (const auto& event : allData) {
        if (event.stringFields.size() > (bundleNameIdx - 11) && 
            event.stringFields[bundleNameIdx - 11] == targetApp) {
            filteredData.push_back(event);
        }
    }
    
    return filteredData;
}

// Structure to hold trace information
struct Trace {
    std::vector<TouchEvent> events;
    int traceID;
};

// Function to process touch/scroll detection and create traces
std::map<int, std::vector<TouchEvent>> createTraces(std::vector<TouchEvent>& interestedData,
                                                     std::vector<TouchEvent>& tracesFlat,
                                                     const ColumnIndices& indices) {
    std::map<int, std::vector<TouchEvent>> traces;
    
    // Find all TYPE 1 and TYPE 3 events
    std::vector<int> type1Indices;
    std::vector<int> type3Indices;
    
    for (int i = 0; i < interestedData.size(); i++) {
        if (interestedData[i].type == 1) {
            type1Indices.push_back(i);
        } else if (interestedData[i].type == 3) {
            type3Indices.push_back(i);
        }
    }
    
    // Calculate position shifts between TYPE 1 and TYPE 3 events
    // Small shift = touch operation, large shift = scroll operation
    std::vector<double> shifting;
    for (int i = 0; i < type1Indices.size() && i < type3Indices.size(); i++) {
        double xShift = std::abs(interestedData[type3Indices[i]].posX - 
                                interestedData[type1Indices[i]].posX);
        double yShift = std::abs(interestedData[type3Indices[i]].posY - 
                                interestedData[type1Indices[i]].posY);
        shifting.push_back(xShift + yShift);
    }
    
    // Create traces based on shifting and event sequences
    std::vector<TouchEvent> currentTrace;
    int traceID = 0;
    
    for (int j = 0; j < shifting.size(); j++) {
        int idxStart = type1Indices[j];
        int idxEnd = type3Indices[j];
        double shift = shifting[j];
        
        // Start new trace if this is the first event of a new app session
        if (interestedData[idxStart].eventID == 0) {
            if (!currentTrace.empty()) {
                traces[traceID] = currentTrace;
                traceID++;
            }
            currentTrace.clear();
        }
        
        // Process based on shift amount
        if (shift < 10) {
            // Touch operation - small movement
            interestedData[idxStart].traceID = traceID;
            interestedData[idxEnd].traceID = traceID;
            
            // Calculate duration for TYPE 3 event
            interestedData[idxEnd].duration = interestedData[idxEnd].timestamp - 
                                              interestedData[idxStart].timestamp;
            
            currentTrace.push_back(interestedData[idxStart]);
            currentTrace.push_back(interestedData[idxEnd]);
            
            tracesFlat.push_back(interestedData[idxStart]);
            tracesFlat.push_back(interestedData[idxEnd]);
        } else {
            // Scroll operation - large movement
            // Include start, middle, and end points
            int midIdx = (idxStart + idxEnd) / 2;
            
            interestedData[idxStart].traceID = traceID;
            interestedData[midIdx].traceID = traceID;
            interestedData[idxEnd].traceID = traceID;
            
            // Calculate duration for TYPE 3 event
            interestedData[idxEnd].duration = interestedData[idxEnd].timestamp - 
                                              interestedData[idxStart].timestamp;
            
            currentTrace.push_back(interestedData[idxStart]);
            currentTrace.push_back(interestedData[midIdx]);
            currentTrace.push_back(interestedData[idxEnd]);
            
            tracesFlat.push_back(interestedData[idxStart]);
            tracesFlat.push_back(interestedData[midIdx]);
            tracesFlat.push_back(interestedData[idxEnd]);
        }
    }
    
    // Don't forget the last trace
    if (!currentTrace.empty()) {
        traces[traceID] = currentTrace;
    }
    
    return traces;
}

// Structure to hold clustering results (placeholder for your implementation)
struct ClusterResult {
    std::vector<int> labels;
    // Add other fields as needed
};

// Placeholder for your DBSCAN implementation
ClusterResult performDBSCAN(const std::vector<std::vector<double>>& data, 
                            double eps, int minSamples) {
    // Your DBSCAN implementation goes here
    ClusterResult result;
    // ... your code ...
    return result;
}

// Structure to hold voting information for each cluster
struct ClusterVoting {
    int minEventID;
    int maxEventID;
    int uniqueTraceCount;
};

// Function to filter clusters based on voting mechanism
std::vector<int> filterClustersByVoting(const std::vector<TouchEvent>& tracesFlat,
                                        const std::vector<int>& labels,
                                        int numTraces,
                                        double votingThreshold) {
    std::map<int, ClusterVoting> clusterVoting;
    
    // Get unique cluster IDs
    std::set<int> uniqueLabels(labels.begin(), labels.end());
    
    // Calculate voting statistics for each cluster
    for (int clusterId : uniqueLabels) {
        if (clusterId == -1) continue; // Skip noise points
        
        std::vector<TouchEvent> clusterEvents;
        std::set<int> uniqueTraceIDs;
        int minEventID = INT_MAX;
        int maxEventID = INT_MIN;
        
        // Collect events belonging to this cluster
        for (int i = 0; i < labels.size(); i++) {
            if (labels[i] == clusterId) {
                clusterEvents.push_back(tracesFlat[i]);
                uniqueTraceIDs.insert(tracesFlat[i].traceID);
                minEventID = std::min(minEventID, tracesFlat[i].eventID);
                maxEventID = std::max(maxEventID, tracesFlat[i].eventID);
            }
        }
        
        // Store voting information
        clusterVoting[clusterId] = {minEventID, maxEventID, 
                                    static_cast<int>(uniqueTraceIDs.size())};
    }
    
    // Apply voting threshold - only keep clusters that appear in enough traces
    int votingTau = static_cast<int>(numTraces * votingThreshold);
    std::vector<int> validClusterIDs;
    
    for (const auto& [clusterId, voting] : clusterVoting) {
        if (voting.uniqueTraceCount >= votingTau) {
            validClusterIDs.push_back(clusterId);
        }
    }
    
    // Sort by order of appearance (minimum event ID)
    std::sort(validClusterIDs.begin(), validClusterIDs.end(),
              [&clusterVoting](int a, int b) {
                  return clusterVoting[a].minEventID < clusterVoting[b].minEventID;
              });
    
    return validClusterIDs;
}

// Function to extract frequent traces from filtered clusters
std::vector<TouchEvent> extractFrequentTraces(const std::vector<TouchEvent>& tracesFlat,
                                               const std::vector<int>& labels,
                                               const std::vector<int>& validClusterIDs) {
    std::vector<TouchEvent> frequentTraces;
    
    // Process each valid cluster
    for (int clusterId : validClusterIDs) {
        std::vector<TouchEvent> clusterEvents;
        
        // Collect events belonging to this cluster (excluding TYPE 2)
        for (int i = 0; i < labels.size(); i++) {
            if (labels[i] == clusterId && tracesFlat[i].type != 2) {
                clusterEvents.push_back(tracesFlat[i]);
            }
        }
        
        if (clusterEvents.empty()) continue;
        
        // Calculate centroid position for the cluster
        double centroidX = 0.0;
        double centroidY = 0.0;
        for (const auto& event : clusterEvents) {
            centroidX += event.posX;
            centroidY += event.posY;
        }
        centroidX /= clusterEvents.size();
        centroidY /= clusterEvents.size();
        
        // Calculate average duration for TYPE 3 events
        double avgDuration = 0.0;
        int type3Count = 0;
        for (const auto& event : clusterEvents) {
            if (event.type == 3) {
                avgDuration += event.duration;
                type3Count++;
            }
        }
        if (type3Count > 0) {
            avgDuration /= type3Count;
        }
        
        // Create representative TYPE 1 and TYPE 3 events for this cluster
        // Use the first events as templates
        TouchEvent type1Event = clusterEvents[0];
        TouchEvent type3Event = clusterEvents[0];
        
        // Find actual TYPE 3 event as template
        for (const auto& event : clusterEvents) {
            if (event.type == 3) {
                type3Event = event;
                break;
            }
        }
        
        // Update TYPE 1 event
        type1Event.type = 1;
        type1Event.posX = centroidX;
        type1Event.posY = centroidY;
        
        // Update TYPE 3 event
        type3Event.type = 3;
        type3Event.posX = centroidX;
        type3Event.posY = centroidY;
        type3Event.duration = avgDuration;
        
        frequentTraces.push_back(type1Event);
        frequentTraces.push_back(type3Event);
    }
    
    return frequentTraces;
}

int main() {
    // File paths
    std::string rootDirectory = "D:/agent/trajectory/data/navigation_company/2025-11-19-09-32-25";
    std::string filePath = rootDirectory + "/touchEvent_0.txt";
    
    // Load all touch events from the file
    std::vector<int> screenSize;
    std::vector<TouchEvent> allData = loadTouchEvents(filePath, screenSize);
    
    if (allData.empty()) {
        std::cerr << "Failed to load data from file." << std::endl;
        return 1;
    }
    
    std::cout << "Loaded " << allData.size() << " events." << std::endl;
    std::cout << "Screen size: " << screenSize[0] << "x" << screenSize[1] << std::endl;
    
    // Define column names and indices
    std::vector<std::string> columnNames = {"timestamp", "field1", "field2", "field3", "field4", 
                                            "field5", "field6", "TYPE", "POSX", "POSY", "field10",
                                            "bundleName", "field12", "field13", "field14", 
                                            "event_ID", "trace_ID"};
    
    // Set up column indices for easy access
    ColumnIndices indices;
    indices.bundleName = 11;
    indices.type = 7;
    indices.eventID = 15;
    indices.posX = 8;
    indices.posY = 9;
    indices.timestamp = 0;
    indices.duration = 16;  // Will be added later
    indices.traceID = 16;
    
    // Filter data for the specific app (com.amap.hmapp)
    std::vector<TouchEvent> interestedData = filterByApp(allData, "com.amap.hmapp", columnNames);
    std::cout << "Filtered to " << interestedData.size() << " events for com.amap.hmapp." << std::endl;
    
    // Create traces from the filtered data
    std::vector<TouchEvent> tracesFlat;
    std::map<int, std::vector<TouchEvent>> traces = createTraces(interestedData, tracesFlat, indices);
    std::cout << "Created " << traces.size() << " traces." << std::endl;
    
    // Prepare data for clustering (posX, posY, eventID)
    std::vector<std::vector<double>> clusteringData;
    for (const auto& event : tracesFlat) {
        clusteringData.push_back({event.posX, event.posY, static_cast<double>(event.eventID)});
    }
    
    // Perform DBSCAN clustering (using your implementation)
    double eps = 60.0;
    int minSamples = static_cast<int>(traces.size() * 0.8);
    ClusterResult clusterResult = performDBSCAN(clusteringData, eps, minSamples);
    
    // Filter clusters using voting mechanism
    // Only clusters that appear in at least 80% of traces are considered frequent
    double votingThreshold = 0.8;
    std::vector<int> validClusterIDs = filterClustersByVoting(tracesFlat, clusterResult.labels, 
                                                               traces.size(), votingThreshold);
    std::cout << "Found " << validClusterIDs.size() << " valid clusters after voting." << std::endl;
    
    // Extract frequent traces from the valid clusters
    std::vector<TouchEvent> frequentTraces = extractFrequentTraces(tracesFlat, clusterResult.labels, 
                                                                    validClusterIDs);
    std::cout << "Extracted " << frequentTraces.size() / 2 << " frequent operations." << std::endl;
    
    // Output frequent traces for verification
    std::cout << "\nFrequent operations:" << std::endl;
    for (int i = 0; i < frequentTraces.size(); i += 2) {
        std::cout << "Operation " << (i/2 + 1) << ": ";
        std::cout << "Position (" << frequentTraces[i].posX << ", " << frequentTraces[i].posY << ")";
        if (i + 1 < frequentTraces.size()) {
            std::cout << ", Duration: " << frequentTraces[i + 1].duration << "ms";
        }
        std::cout << std::endl;
    }
    
    return 0;
}