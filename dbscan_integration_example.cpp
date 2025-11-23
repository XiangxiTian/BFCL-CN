// Example: How to integrate your DBSCAN implementation
// This file shows the integration pattern - adapt to your specific DBSCAN API

#include "touch_event_processor.h"
#include <vector>
#include <set>

// Example 1: If your DBSCAN class has a simple interface
// Replace "YourDBSCAN" with your actual class name
/*
#include "your_dbscan.h"  // Your DBSCAN header

ClusteringResult performDBSCANClustering(
    const std::vector<TouchEvent>& traces_flat,
    const ColumnIndices& col_idx,
    int num_traces,
    float eps,
    float min_samples_ratio) {
    
    ClusteringResult result;
    
    // Step 1: Prepare features [posX, posY, event_ID] for clustering
    std::vector<std::vector<double>> features;
    features.reserve(traces_flat.size());
    
    for (const auto& event : traces_flat) {
        float posX = getFloatField(event, col_idx.posX_idx);
        float posY = getFloatField(event, col_idx.posY_idx);
        double event_id = static_cast<double>(event.event_ID);
        
        features.push_back({
            static_cast<double>(posX),
            static_cast<double>(posY),
            event_id
        });
    }
    
    // Step 2: Configure DBSCAN parameters
    int min_samples = static_cast<int>(num_traces * min_samples_ratio);
    
    // Step 3: Run your DBSCAN implementation
    YourDBSCAN dbscan(eps, min_samples);
    result.labels = dbscan.fit(features);  // Assuming fit() returns vector<int>
    
    // Step 4: Extract unique cluster IDs
    for (int label : result.labels) {
        if (label != -1) {  // -1 represents noise points
            result.cluster_ids.insert(label);
        }
    }
    
    return result;
}
*/

// Example 2: If your DBSCAN uses separate fit and predict
/*
ClusteringResult performDBSCANClustering(
    const std::vector<TouchEvent>& traces_flat,
    const ColumnIndices& col_idx,
    int num_traces,
    float eps,
    float min_samples_ratio) {
    
    ClusteringResult result;
    result.labels.resize(traces_flat.size());
    
    // Prepare data matrix
    std::vector<std::vector<double>> data_matrix;
    for (const auto& event : traces_flat) {
        float posX = getFloatField(event, col_idx.posX_idx);
        float posY = getFloatField(event, col_idx.posY_idx);
        data_matrix.push_back({
            static_cast<double>(posX),
            static_cast<double>(posY),
            static_cast<double>(event.event_ID)
        });
    }
    
    // Run clustering
    int min_samples = static_cast<int>(num_traces * min_samples_ratio);
    YourDBSCAN dbscan;
    dbscan.set_epsilon(eps);
    dbscan.set_min_samples(min_samples);
    dbscan.fit(data_matrix);
    
    // Get labels
    for (size_t i = 0; i < traces_flat.size(); i++) {
        result.labels[i] = dbscan.get_label(i);
        if (result.labels[i] != -1) {
            result.cluster_ids.insert(result.labels[i]);
        }
    }
    
    return result;
}
*/

// Example 3: If your DBSCAN uses C-style arrays
/*
ClusteringResult performDBSCANClustering(
    const std::vector<TouchEvent>& traces_flat,
    const ColumnIndices& col_idx,
    int num_traces,
    float eps,
    float min_samples_ratio) {
    
    ClusteringResult result;
    int n_points = traces_flat.size();
    int n_features = 3;  // posX, posY, event_ID
    
    // Allocate data array: [point0_x, point0_y, point0_id, point1_x, ...]
    double* data = new double[n_points * n_features];
    
    for (size_t i = 0; i < traces_flat.size(); i++) {
        data[i * n_features + 0] = getFloatField(traces_flat[i], col_idx.posX_idx);
        data[i * n_features + 1] = getFloatField(traces_flat[i], col_idx.posY_idx);
        data[i * n_features + 2] = static_cast<double>(traces_flat[i].event_ID);
    }
    
    // Allocate labels array
    int* labels = new int[n_points];
    
    // Run your DBSCAN (example API)
    int min_samples = static_cast<int>(num_traces * min_samples_ratio);
    dbscan_cluster(data, n_points, n_features, eps, min_samples, labels);
    
    // Convert to result structure
    result.labels.assign(labels, labels + n_points);
    for (int i = 0; i < n_points; i++) {
        if (labels[i] != -1) {
            result.cluster_ids.insert(labels[i]);
        }
    }
    
    // Cleanup
    delete[] data;
    delete[] labels;
    
    return result;
}
*/

// Example 4: Using Eigen library for matrix representation
/*
#include <Eigen/Dense>

ClusteringResult performDBSCANClustering(
    const std::vector<TouchEvent>& traces_flat,
    const ColumnIndices& col_idx,
    int num_traces,
    float eps,
    float min_samples_ratio) {
    
    ClusteringResult result;
    
    // Create Eigen matrix
    int n_points = traces_flat.size();
    Eigen::MatrixXd features(n_points, 3);
    
    for (int i = 0; i < n_points; i++) {
        features(i, 0) = getFloatField(traces_flat[i], col_idx.posX_idx);
        features(i, 1) = getFloatField(traces_flat[i], col_idx.posY_idx);
        features(i, 2) = static_cast<double>(traces_flat[i].event_ID);
    }
    
    // Run DBSCAN
    int min_samples = static_cast<int>(num_traces * min_samples_ratio);
    YourDBSCAN dbscan(eps, min_samples);
    Eigen::VectorXi labels = dbscan.fit_predict(features);
    
    // Convert results
    result.labels.resize(n_points);
    for (int i = 0; i < n_points; i++) {
        result.labels[i] = labels(i);
        if (labels(i) != -1) {
            result.cluster_ids.insert(labels(i));
        }
    }
    
    return result;
}
*/

// Note: Choose the example that matches your DBSCAN implementation's API
// and uncomment/modify it accordingly. Update the Makefile to link your
// DBSCAN library.

/*
 * Key Points for Integration:
 * 
 * 1. Input Features: Always use [posX, posY, event_ID]
 *    - posX, posY: Spatial similarity (where the touch occurs)
 *    - event_ID: Temporal/sequential similarity (order of operations)
 * 
 * 2. Parameters:
 *    - eps: Distance threshold (60 pixels recommended)
 *    - min_samples: num_traces * 0.8 (80% of traces must contain the operation)
 * 
 * 3. Output Format:
 *    - labels: Integer array/vector, same length as input
 *    - -1 indicates noise/outlier points
 *    - Positive integers are cluster IDs (0, 1, 2, ...)
 * 
 * 4. Distance Metric:
 *    - You may want to normalize or weight the features:
 *      - posX, posY in pixel space (typically 0-1080 range)
 *      - event_ID in sequence space (typically 0-50 range)
 *    - Consider using weighted Euclidean distance if needed
 * 
 * 5. Performance Tips:
 *    - Use spatial indexing (KD-tree, R-tree) for faster neighbor queries
 *    - Pre-allocate result vectors to avoid reallocations
 *    - Consider parallel processing if you have many points
 */
