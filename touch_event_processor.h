#ifndef TOUCH_EVENT_PROCESSOR_H
#define TOUCH_EVENT_PROCESSOR_H

#include <vector>

// Forward declaration for DBSCAN clustering function
// User should implement this function with their own DBSCAN implementation
// 
// Parameters:
//   points: Vector of points, where each point is [posX, posY, event_ID] as integers
//   eps: Maximum distance between points in the same cluster (e.g., 60.0)
//   min_samples: Minimum number of points required to form a cluster
//
// Returns:
//   Vector of cluster labels for each point (-1 indicates noise/outlier)
//
// Example implementation signature:
// std::vector<int> performDBSCAN(const std::vector<std::vector<int>>& points, 
//                                 double eps, int min_samples) {
//     // Your DBSCAN implementation here
//     // Return vector of labels
// }
std::vector<int> performDBSCAN(const std::vector<std::vector<int>>& points, 
                                double eps, int min_samples);

#endif // TOUCH_EVENT_PROCESSOR_H
