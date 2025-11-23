# Touch Event Processor - C++ Implementation

This C++ implementation converts the Python script for processing touch event data and extracting frequent operations.

## Overview

The code processes touch event data to:
1. Load and parse touch events from a text file
2. Filter events for a specific application
3. Detect touch vs. scroll operations based on position changes
4. Group events into traces (sequences of related operations)
5. Cluster similar operations across traces (DBSCAN - you need to implement this)
6. Filter clusters using a voting mechanism
7. Extract representative frequent operations

## Key Components

### Data Structures

- **TouchEvent**: Main structure holding all event information
  - Basic fields from the file (coordinates, timestamps, type)
  - Derived fields (eventID, traceID, duration)
  
- **ColumnIndices**: Stores column positions for easy field access

- **ClusterVoting**: Tracks cluster statistics for filtering

### Main Processing Steps

1. **Data Loading (`loadTouchEvents`)**
   - Reads the touchEvent file line by line
   - Parses numeric and string fields
   - Assigns sequential event IDs that reset when app changes
   - Extracts screen dimensions

2. **App Filtering (`filterByApp`)**
   - Filters events for specific bundle name (e.g., "com.amap.hmapp")
   - Uses the bundleName field from the string fields

3. **Touch/Scroll Detection (`createTraces`)**
   - Pairs TYPE 1 (touch start) with TYPE 3 (touch end) events
   - Calculates position shift between start and end
   - Small shift (<10 pixels) = touch operation
   - Large shift = scroll operation (includes middle point)
   - Groups events into traces based on app sessions

4. **Clustering (DBSCAN)**
   - You need to implement this part
   - Input: 3D data (posX, posY, eventID)
   - Output: cluster labels for each point

5. **Cluster Filtering (`filterClustersByVoting`)**
   - Counts how many different traces contain each cluster
   - Only keeps clusters appearing in ≥80% of traces
   - This ensures we find truly frequent operations

6. **Frequent Operation Extraction (`extractFrequentTraces`)**
   - For each valid cluster:
     - Calculates centroid position
     - Computes average duration
     - Creates representative TYPE 1 and TYPE 3 events

## Key Differences from Python

1. **Manual memory management**: Using vectors instead of numpy arrays
2. **Explicit type conversions**: C++ requires explicit casting
3. **No built-in plotting**: Visualization code removed
4. **Manual string parsing**: Using stringstream instead of split()
5. **Map/vector combinations**: Instead of Python dictionaries

## Usage

```bash
g++ -std=c++11 touch_event_processor.cpp -o touch_event_processor
./touch_event_processor
```

## Notes

- The code assumes a specific file format for touchEvent_0.txt
- Screen size is extracted from the first event's data[3] and data[4]
- Event types: 1 = touch start, 2 = move, 3 = touch end
- Duration is calculated only for TYPE 3 events
- The voting threshold (0.8) determines how frequent an operation must be

## DBSCAN Implementation

You need to implement the `performDBSCAN` function. It should:
- Accept 3D data points (posX, posY, eventID)
- Use eps=60 and minSamples=80% of trace count
- Return cluster labels (-1 for noise points)