#ifndef TOUCH_EVENT_PROCESSOR_H
#define TOUCH_EVENT_PROCESSOR_H

#include <vector>
#include <string>
#include <map>

// Structure to represent a touch event with all its attributes
struct TouchEvent {
    // Basic event attributes from file
    int data[6];              // First 6 integer fields from the file
    double field6;            // 7th field (timestamp as double)
    int field7;               // 8th field (event type as int)
    double posX;              // X position on screen
    double posY;              // Y position on screen
    int field10;              // 11th field
    std::vector<std::string> stringFields; // String fields (bundleName, etc.)
    
    // Derived attributes for processing
    int eventID;              // Sequential event ID within the app session
    int traceID;              // ID of the trace this event belongs to
    double timestamp;         // Event timestamp (copy of field6)
    double duration;          // Duration for TYPE 3 events (calculated)
    int type;                 // Event type: 1=touch start, 2=move, 3=touch end
    
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

// Structure to hold DBSCAN clustering results
struct ClusterResult {
    std::vector<int> labels;  // Cluster label for each data point (-1 for noise)
};

// Structure to hold voting information for each cluster
struct ClusterVoting {
    int minEventID;           // Earliest event in the cluster
    int maxEventID;           // Latest event in the cluster
    int uniqueTraceCount;     // Number of different traces containing this cluster
};

// Function declarations
std::vector<std::string> splitString(const std::string& str);

std::vector<TouchEvent> loadTouchEvents(const std::string& filePath, 
                                       std::vector<int>& screenSize);

std::vector<TouchEvent> filterByApp(const std::vector<TouchEvent>& allData, 
                                   const std::string& targetApp,
                                   const std::vector<std::string>& columnNames);

std::map<int, std::vector<TouchEvent>> createTraces(std::vector<TouchEvent>& interestedData,
                                                    std::vector<TouchEvent>& tracesFlat,
                                                    const ColumnIndices& indices);

ClusterResult performDBSCAN(const std::vector<std::vector<double>>& data, 
                           double eps, int minSamples);

std::vector<int> filterClustersByVoting(const std::vector<TouchEvent>& tracesFlat,
                                       const std::vector<int>& labels,
                                       int numTraces,
                                       double votingThreshold);

std::vector<TouchEvent> extractFrequentTraces(const std::vector<TouchEvent>& tracesFlat,
                                             const std::vector<int>& labels,
                                             const std::vector<int>& validClusterIDs);

#endif // TOUCH_EVENT_PROCESSOR_H