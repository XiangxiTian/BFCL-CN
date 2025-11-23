# Touch Event Processor - Data Flow Explanation

This document explains the data transformation pipeline step by step.

## Input Data Format

**File:** `touchEvent_0.txt`

```
[header line 1]
field0 field1 field2 field3 field4 field5 field6 field7 field8 field9 field10 field11 field12 field13 bundleName
123    245    512    1080   1920   0      1234.5 1      0.0    0.0    0       ...     ...     ...     com.amap.hmapp
123    245    512    1080   1920   0      1234.6 2      10.5   15.2   1       ...     ...     ...     com.amap.hmapp
123    247    518    1080   1920   0      1234.7 3      10.5   15.2   2       ...     ...     ...     com.amap.hmapp
...
```

**Key Fields:**
- `field1`: POSX (X coordinate)
- `field2`: POSY (Y coordinate)  
- `field3`: Screen width
- `field4`: Screen height
- `field6`: timestamp
- `field7`: TYPE (1=touch down, 2=move, 3=touch up)
- `field8`: Additional position data
- `field9`: Additional position data
- `bundleName`: Application package name

## Step-by-Step Data Transformation

### Step 1: Load All Data
```
Input: touchEvent_0.txt (raw file)
Output: vector<TouchEvent> all_data

Example:
all_data[0] = {field1: 245, field2: 512, field7: 1, bundleName: "com.amap.hmapp", event_ID: 0, ...}
all_data[1] = {field1: 245, field2: 512, field7: 2, bundleName: "com.amap.hmapp", event_ID: 1, ...}
all_data[2] = {field1: 247, field2: 518, field7: 3, bundleName: "com.amap.hmapp", event_ID: 2, ...}
all_data[3] = {field1: 512, field2: 768, field7: 1, bundleName: "com.amap.hmapp", event_ID: 3, ...}
...
all_data[N] = {field1: 100, field2: 200, field7: 1, bundleName: "com.other.app", event_ID: 0, ...}
```

**Note:** `event_ID` resets to 0 when `bundleName` changes (new app session).

### Step 2: Filter by Application
```
Input: all_data (all apps)
Output: interested_data (only "com.amap.hmapp")

Example:
interested_data = [
    {posX: 245, posY: 512, TYPE: 1, event_ID: 0},  // Touch down
    {posX: 245, posY: 512, TYPE: 2, event_ID: 1},  // Move
    {posX: 247, posY: 518, TYPE: 3, event_ID: 2},  // Touch up
    {posX: 512, posY: 768, TYPE: 1, event_ID: 3},  // Next operation...
    ...
]
```

### Step 3: Build Traces

**Substep 3a: Extract TYPE 1 and TYPE 3 events**
```
TYPE 1 (touch down): indices [0, 3, 6, 9, ...]
TYPE 3 (touch up):   indices [2, 5, 8, 11, ...]
```

**Substep 3b: Calculate shifting**
```
shifting[i] = |posX_type3[i] - posX_type1[i]| + |posY_type3[i] - posY_type1[i]|

Example:
Operation 0: shift = |247 - 245| + |518 - 512| = 2 + 6 = 8 (< 10 → TOUCH)
Operation 1: shift = |600 - 512| + |800 - 768| = 88 + 32 = 120 (≥ 10 → SCROLL)
```

**Substep 3c: Group into traces**
```
traces_flat format:
[
    // Trace 0 (event_ID started at 0)
    {posX: 245, posY: 512, TYPE: 1, event_ID: 0, trace_ID: 0},  // Op1 down
    {posX: 247, posY: 518, TYPE: 3, event_ID: 2, trace_ID: 0},  // Op1 up
    {posX: 512, posY: 768, TYPE: 1, event_ID: 3, trace_ID: 0},  // Op2 down
    {posX: 520, posY: 770, TYPE: 3, event_ID: 5, trace_ID: 0},  // Op2 up
    
    // Trace 1 (event_ID reset to 0 - new session)
    {posX: 248, posY: 515, TYPE: 1, event_ID: 0, trace_ID: 1},  // Op1 down
    {posX: 250, posY: 517, TYPE: 3, event_ID: 2, trace_ID: 1},  // Op1 up
    {posX: 515, posY: 770, TYPE: 1, event_ID: 3, trace_ID: 1},  // Op2 down
    {posX: 518, posY: 772, TYPE: 3, event_ID: 5, trace_ID: 1},  // Op2 up
    ...
]

traces map:
traces[0] = [events from trace_ID 0]
traces[1] = [events from trace_ID 1]
...
```

### Step 4: DBSCAN Clustering

**Input Features:** `[posX, posY, event_ID]`

```
Point 0: [245, 512, 0]  → cluster 0 (Op1 in trace 0)
Point 1: [247, 518, 2]  → cluster 0 (Op1 in trace 0)
Point 2: [512, 768, 3]  → cluster 1 (Op2 in trace 0)
Point 3: [520, 770, 5]  → cluster 1 (Op2 in trace 0)
Point 4: [248, 515, 0]  → cluster 0 (Op1 in trace 1 - similar to trace 0 op1)
Point 5: [250, 517, 2]  → cluster 0 (Op1 in trace 1)
Point 6: [515, 770, 3]  → cluster 1 (Op2 in trace 1 - similar to trace 0 op2)
Point 7: [518, 772, 5]  → cluster 1 (Op2 in trace 1)
Point 8: [900, 100, 0]  → cluster -1 (Noise - unique operation)
...
```

**Why these features?**
- `posX, posY`: Groups operations at similar screen locations
- `event_ID`: Groups operations at similar positions in the operation sequence

**Result:**
```
clustering.labels = [0, 0, 1, 1, 0, 0, 1, 1, -1, ...]
clustering.cluster_ids = {0, 1}
```

### Step 5: Filter Clusters by Voting

**Purpose:** Ensure operations appear frequently across multiple sessions (traces)

```
Cluster analysis:
- Cluster 0: appears in traces {0, 1, 2, 3, 4, 5, 6, 7} = 8 traces
- Cluster 1: appears in traces {0, 1, 2, 3, 4, 5, 6} = 7 traces  
- Cluster 2: appears in traces {0, 1} = 2 traces (REJECTED - not frequent enough)
- Cluster -1: noise points (always rejected)

If total traces = 10, voting_tau = 10 * 0.8 = 8
Valid clusters: {0, 1} (both appear in ≥8 traces)

cluster_voting format:
{
    0: [min_event_ID: 0, max_event_ID: 2, num_traces: 8],
    1: [min_event_ID: 3, max_event_ID: 5, num_traces: 7]
}

Sorted by min_event_ID (operation order):
valid_clusters = {0, 1}  // Cluster 0 comes before cluster 1
```

### Step 6: Extract Frequent Traces

**For each valid cluster, compute centroid:**

```
Cluster 0 (all TYPE 1 and TYPE 3, excluding TYPE 2):
Points: [245,512], [247,518], [248,515], [250,517], [246,513], ...
Centroid: [246.8, 515.0]
Durations (TYPE 3 only): [0.15s, 0.14s, 0.16s, 0.15s, ...]
Avg duration: 0.15s

Cluster 1:
Points: [512,768], [520,770], [515,770], [518,772], [514,769], ...
Centroid: [515.8, 769.8]
Avg duration: 0.12s

frequent_traces output:
[
    // Operation 1 (cluster 0)
    {TYPE: 1, posX: 246.8, posY: 515.0, duration: 0.0},      // Touch down
    {TYPE: 3, posX: 246.8, posY: 515.0, duration: 0.15},     // Touch up
    
    // Operation 2 (cluster 1)
    {TYPE: 1, posX: 515.8, posY: 769.8, duration: 0.0},      // Touch down
    {TYPE: 3, posX: 515.8, posY: 769.8, duration: 0.12},     // Touch up
]
```

### Step 7: Save Output

**File:** `frequent_traces.txt`

```
# Frequent Touch Operation Traces
# Format: TYPE POSX POSY DURATION BUNDLENAME

1 246.8 515.0 0.0 com.amap.hmapp
3 246.8 515.0 0.15 com.amap.hmapp
1 515.8 769.8 0.0 com.amap.hmapp
3 515.8 769.8 0.12 com.amap.hmapp
```

**Usage for Replay:**
```cpp
// Read this file and replay the operations:
1. Touch down at (246.8, 515.0)
2. Wait 0.15 seconds
3. Touch up at (246.8, 515.0)
4. Touch down at (515.8, 769.8)
5. Wait 0.12 seconds
6. Touch up at (515.8, 769.8)
```

## Visual Example

### Raw Traces (Step 3 output)
```
Session 1: A → B → C → D
Session 2: A → B → C → D → E
Session 3: A → B → C → F
Session 4: A → B → C → D
Session 5: X → A → B → C
...
(10 sessions total)
```

### After Clustering (Step 4)
```
Cluster 0: Operation A (appears in sessions 1,2,3,4,5,6,7,8,9,10)
Cluster 1: Operation B (appears in sessions 1,2,3,4,5,6,7,8,9,10)  
Cluster 2: Operation C (appears in sessions 1,2,3,4,5,6,7,8,9,10)
Cluster 3: Operation D (appears in sessions 1,2,4,6,7,8) 
Cluster 4: Operation E (appears in session 2 only) ← Will be filtered
Cluster 5: Operation F (appears in session 3 only) ← Will be filtered
Cluster 6: Operation X (appears in session 5 only) ← Will be filtered
```

### After Voting (Step 5, threshold=8 sessions)
```
Valid clusters: {0, 1, 2} 
Sorted by order: A → B → C
```

### Final Output (Step 6)
```
Frequent operation sequence: A → B → C
With averaged positions and timings from all occurrences
```

## Key Insights

1. **Spatial + Temporal Clustering**: DBSCAN considers both where (posX, posY) and when (event_ID) operations occur

2. **Voting Robustness**: Only operations appearing in most sessions are kept, filtering out one-time or rare actions

3. **Centroid Averaging**: Final positions and timings represent typical user behavior, smoothing out minor variations

4. **Replay Ready**: Output format directly usable for automated UI testing or task replay

5. **Trace Segmentation**: Properly handles multiple app sessions and different operation lengths (touch vs scroll)
