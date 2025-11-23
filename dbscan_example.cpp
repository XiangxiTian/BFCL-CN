// Example DBSCAN implementation for reference
// You can replace this with your own implementation

#include <vector>
#include <cmath>
#include <queue>
#include <set>

// Calculate Euclidean distance between two points
double euclideanDistance(const std::vector<double>& p1, const std::vector<double>& p2) {
    double sum = 0.0;
    for (size_t i = 0; i < p1.size(); i++) {
        sum += (p1[i] - p2[i]) * (p1[i] - p2[i]);
    }
    return std::sqrt(sum);
}

// Find all points within eps distance of the given point
std::vector<int> findNeighbors(const std::vector<std::vector<double>>& data, 
                                int pointIdx, double eps) {
    std::vector<int> neighbors;
    for (int i = 0; i < data.size(); i++) {
        if (i != pointIdx && euclideanDistance(data[pointIdx], data[i]) <= eps) {
            neighbors.push_back(i);
        }
    }
    return neighbors;
}

// Example DBSCAN implementation
ClusterResult performDBSCAN(const std::vector<std::vector<double>>& data, 
                            double eps, int minSamples) {
    ClusterResult result;
    result.labels.resize(data.size(), -1);  // Initialize all as noise (-1)
    
    int currentCluster = 0;
    std::vector<bool> visited(data.size(), false);
    
    for (int i = 0; i < data.size(); i++) {
        if (visited[i]) continue;
        
        visited[i] = true;
        std::vector<int> neighbors = findNeighbors(data, i, eps);
        
        if (neighbors.size() < minSamples) {
            // Point is noise (already labeled as -1)
            continue;
        }
        
        // Start a new cluster
        result.labels[i] = currentCluster;
        
        // Process all points in the cluster
        std::queue<int> toProcess;
        for (int neighbor : neighbors) {
            toProcess.push(neighbor);
        }
        
        while (!toProcess.empty()) {
            int point = toProcess.front();
            toProcess.pop();
            
            if (result.labels[point] == -1) {
                // Change noise to border point
                result.labels[point] = currentCluster;
            }
            
            if (visited[point]) continue;
            visited[point] = true;
            
            result.labels[point] = currentCluster;
            
            std::vector<int> pointNeighbors = findNeighbors(data, point, eps);
            if (pointNeighbors.size() >= minSamples) {
                // This is a core point, add its neighbors
                for (int neighbor : pointNeighbors) {
                    if (!visited[neighbor]) {
                        toProcess.push(neighbor);
                    }
                }
            }
        }
        
        currentCluster++;
    }
    
    return result;
}