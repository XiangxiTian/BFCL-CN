# Touch Event Processor - C++ Implementation

This is a C++ conversion of the Python touch event processing script, excluding DBSCAN clustering and plotting functionality as requested.

## Overview

The code processes touch event data to extract frequent user operations. The workflow is:

1. **Load Data** - Read touch events from file
2. **Filter Data** - Extract events for a specific application
3. **Build Traces** - Group events into operation sequences
4. **Cluster** - Use DBSCAN to find similar operations (YOUR CODE)
5. **Filter** - Validate clusters using voting method
6. **Extract** - Create averaged frequent operation sequences
7. **Save** - Output results for replay

## File Structure

- `touch_event_processor.cpp` - Main implementation with detailed comments
- `touch_event_processor.h` - Header file with structure and function declarations
- `TOUCH_EVENT_PROCESSOR_README.md` - This file

## Integration Points

### 1. DBSCAN Clustering Integration

Replace the placeholder function `performDBSCANClustering()` with your implementation:

```cpp
ClusteringResult performDBSCANClustering(
    const std::vector<TouchEvent>& traces_flat,
    const ColumnIndices& col_idx,
    int num_traces,
    float eps,
    float min_samples_ratio) {
    
    ClusteringResult result;
    
    // Prepare input data: [posX, posY, event_ID] for each event
    std::vector<std::vector<float>> features;
    for (const auto& event : traces_flat) {
        float posX = getFloatField(event, col_idx.posX_idx);
        float posY = getFloatField(event, col_idx.posY_idx);
        float event_id = static_cast<float>(event.event_ID);
        features.push_back({posX, posY, event_id});
    }
    
    // TODO: Call your DBSCAN implementation
    // YourDBSCAN dbscan(eps, min_samples);
    // result.labels = dbscan.fit(features);
    
    // Populate cluster_ids from labels
    for (int label : result.labels) {
        if (label != -1) {
            result.cluster_ids.insert(label);
        }
    }
    
    return result;
}
```

**Key Parameters:**
- `eps`: Distance threshold (60.0 pixels in original code)
- `min_samples`: Minimum points per cluster = `num_traces * 0.8`
- **Features**: [posX, posY, event_ID] as integers

**Expected Output:**
- `labels`: Vector of cluster IDs for each point (-1 for noise)
- `cluster_ids`: Set of unique cluster IDs (excluding -1)

### 2. Plotting Integration

You can add visualization at several points in `main()`:

```cpp
// After Step 3: Visualize raw traces
// Plot all traces with different colors
for (const auto& trace_pair : traces) {
    int trace_id = trace_pair.first;
    const auto& trace = trace_pair.second;
    // Extract x, y coordinates and plot
}

// After Step 4: Visualize clustering results
// Scatter plot colored by cluster labels
for (size_t i = 0; i < traces_flat.size(); i++) {
    if (clustering.labels[i] != -1) {
        float x = getFloatField(traces_flat[i], col_idx.posX_idx);
        float y = getFloatField(traces_flat[i], col_idx.posY_idx);
        int cluster = clustering.labels[i];
        // Plot point (x, y) with color based on cluster
    }
}

// After Step 5: Visualize filtered clusters
// Similar to above but only for valid clusters

// After Step 6: Visualize frequent traces
// Plot the extracted frequent operation sequence
for (const auto& event : frequent_traces) {
    float x = getFloatField(event, col_idx.posX_idx);
    float y = getFloatField(event, col_idx.posY_idx);
    // Plot with markers
}
```

## Key Data Structures

### TouchEvent
Represents a single touch event with:
- Position fields (`field1` = posX, `field2` = posY)
- Type field (`field7` = TYPE: 1=down, 2=move, 3=up)
- Timing fields (`field6` = timestamp, `duration`)
- Metadata (`bundleName`, `event_ID`, `trace_ID`)

### ColumnIndices
Maps column names to field indices for data access.

### ClusteringResult
Holds DBSCAN output:
- `labels`: Cluster assignment for each point
- `cluster_ids`: Set of unique cluster IDs

## Algorithm Details

### Operation Classification
- **Touch/Click**: Shift between TYPE 1 and TYPE 3 < 10 pixels
- **Scroll/Swipe**: Shift >= 10 pixels (includes midpoint)

### Trace Segmentation
Traces are separated when:
- `event_ID` resets to 0 (new app session)
- Implicit time gaps (handled by event_ID sequencing)

### Clustering Strategy
DBSCAN groups operations that are:
- **Spatially similar**: Close posX, posY values
- **Sequentially similar**: Similar event_ID (operation order)

### Voting Filter
Only clusters appearing in ≥80% of traces are considered frequent.
This ensures extracted operations are truly habitual, not isolated.

### Centroid Calculation
For each valid cluster:
- Average all posX, posY positions → representative location
- Average all TYPE 3 durations → typical operation duration

## Compilation

```bash
# Basic compilation
g++ -std=c++11 touch_event_processor.cpp -o touch_event_processor

# With optimization
g++ -std=c++11 -O2 touch_event_processor.cpp -o touch_event_processor

# With your DBSCAN library (example)
g++ -std=c++11 -O2 touch_event_processor.cpp your_dbscan.cpp -o touch_event_processor
```

## Usage

1. Update file paths in `main()`:
```cpp
std::string root_directory = "YOUR_PATH";
std::string target_app = "YOUR_APP_BUNDLE_NAME";
```

2. Integrate your DBSCAN implementation in `performDBSCANClustering()`

3. Add visualization code at marked points if needed

4. Compile and run:
```bash
./touch_event_processor
```

## Output Format

The program generates `frequent_traces.txt` with format:
```
# Frequent Touch Operation Traces
# Format: TYPE POSX POSY DURATION BUNDLENAME
# Each operation consists of two events: TYPE 1 (down) and TYPE 3 (up)

1 245.3 512.7 0.0 com.amap.hmapp
3 245.3 512.7 145.2 com.amap.hmapp
1 512.1 768.4 0.0 com.amap.hmapp
3 512.1 768.4 128.6 com.amap.hmapp
...
```

Each pair represents one frequent operation ready for replay.

## Debugging Tips

1. **Check data loading**: Print `all_data.size()` after Step 1
2. **Verify filtering**: Print `interested_data.size()` after Step 2
3. **Inspect traces**: Print `traces.size()` and `traces_flat.size()` after Step 3
4. **Monitor clustering**: Print cluster sizes and distributions after Step 4
5. **Validate filtering**: Print `valid_clusters.size()` after Step 5

## Notes

- The code preserves the exact logic from the Python implementation
- Column index mapping handles different field arrangements
- Event_ID tracking matches the Python version's behavior
- The 80% threshold for voting can be adjusted via parameters
- DBSCAN parameters (eps=60, min_samples) can be tuned for your data

## Differences from Python Version

1. **Type Safety**: Explicit types instead of Python's dynamic typing
2. **Memory Management**: Stack-based by default (no garbage collection)
3. **Data Access**: Accessor functions instead of dictionary-style access
4. **No NumPy**: Manual calculations instead of NumPy vectorization
5. **No Pandas**: Direct file parsing instead of DataFrame operations

## Next Steps

1. Integrate your DBSCAN implementation
2. Add visualization code if needed
3. Test with your actual data files
4. Tune parameters (eps, voting threshold) for optimal results
5. Consider adding multi-file processing capability
