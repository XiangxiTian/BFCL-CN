#ifndef DBSCAN_H
#define DBSCAN_H

#include <vector>
#include <cmath>
#include <iostream>

#define UNCLASSIFIED -1
#define NOISE -2

#define SUCCESS 0
#define FAILURE -1

using namespace std;

struct Point {
    std::vector<float> embedding;
    int clusterID;
    int originalIndex; // To map back to tools
};

class DBSCAN {
public:
    // eps: radius (using squared euclidean distance in this implementation)
    // minPts: minimum points to form a cluster
    DBSCAN(unsigned int minPts, float eps, const vector<Point>& points) {
        m_minPoints = minPts;
        m_epsilon = eps;
        m_points = points;
    }
    int run();
    vector<int> getClusterLabels();
    vector<Point> m_points; 

private:
    unsigned int m_minPoints;
    float m_epsilon;
    
    int expandCluster(Point point, int clusterID);
    vector<int> calculateCluster(Point point);
    double calculateDistance(const Point& pointCore, const Point& pointTarget);
};

#endif // DBSCAN_H
