# Touch Event Processor - C++ Implementation

This C++ code converts the Python touch event processing script for extracting frequent operations from touch event trajectories.

## Overview

The program processes touch event data to:
1. Load and parse touch event data from a text file
2. Filter events for a specific application (bundle name)
3. Classify operations as touches or scrolls based on movement distance
4. Divide trajectories into separate traces
5. Cluster operations using DBSCAN
6. Filter clusters using voting method (operations must appear in >=80% of traces)
7. Extract frequent traces with centroid positions and average durations

## Key Components

### Data Structure: `TouchEvent`
- Stores all event fields as strings for flexible column access
- Includes `event_ID` (sequential within bundle) and `trace_ID` (trace identifier)
- Provides helper functions `getInt()`, `getDouble()`, `getString()` for typed access

### Main Processing Steps

1. **File Loading**: Reads touch event data, tracks bundle name changes to assign sequential event IDs
2. **Filtering**: Extracts events for target bundle name (default: "com.amap.hmapp")
3. **Operation Classification**: Calculates Manhattan distance between TYPE 1 (touch start) and TYPE 3 (touch end) events
   - Shift < 10: Touch operation (includes TYPE 1 and TYPE 3)
   - Shift >= 10: Scroll operation (includes TYPE 1, middle point, and TYPE 3)
4. **Trace Division**: Separates trajectories into traces based on event_ID == 0 (new trace start)
5. **Clustering**: Prepares data for DBSCAN clustering on [posX, posY, event_ID]
6. **Voting Filter**: Only keeps clusters appearing in >=80% of traces
7. **Extraction**: Creates representative TYPE 1 and TYPE 3 events with centroid positions

## Integration

### DBSCAN Function

You need to implement the `performDBSCAN()` function. Add your implementation before `main()` or in a separate file:

```cpp
std::vector<int> performDBSCAN(const std::vector<std::vector<int>>& points, 
                                double eps, int min_samples) {
    // Your DBSCAN implementation here
    // Input: points[i] = [posX, posY, event_ID] as integers
    // Return: labels[i] = cluster ID for point i (-1 for noise)
    std::vector<int> labels(points.size(), -1);
    // ... your clustering logic ...
    return labels;
}
```

### Plotting/Visualization

The Python code includes matplotlib plotting code which has been excluded. Add your visualization code at the end of `main()` function where indicated:

```cpp
// ============================================================================
// Output results (user can add their own output/plotting code here)
// ============================================================================
// Add your plotting/visualization code here
// You have access to:
// - traces: map<int, vector<TouchEvent>> - all traces
// - traces_filtered: vector<TouchEvent> - filtered cluster results
// - frequent_traces: vector<TouchEvent> - extracted frequent operations
// - screen_size: vector<int> - screen dimensions
```

## Compilation

```bash
g++ -std=c++11 -O2 touch_event_processor.cpp -o touch_event_processor
```

Or with C++14/17 for better features:
```bash
g++ -std=c++14 -O2 touch_event_processor.cpp -o touch_event_processor
```

## Usage

1. Update the `root_directory` and `file_path` variables in `main()` to point to your data file
2. Optionally change `target_bundle` to filter for a different application
3. Implement `performDBSCAN()` function
4. Add your plotting/visualization code if needed
5. Compile and run

## Notes

- The code assumes space-separated values in the input file
- Column names are read from the first line of the file
- Required columns: TYPE, event_ID, POSX, POSY, timestamp, duration, bundleName
- The code handles UTF-8 encoding (though C++ iostream may need locale settings for non-ASCII)
- All floating-point comparisons use threshold of 10 pixels for touch vs scroll classification
