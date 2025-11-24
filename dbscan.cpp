#include "dbscan.h"

int DBSCAN::run()
{
    int clusterID = 1;
    vector<Point>::iterator iter;
    for(iter = m_points.begin(); iter != m_points.end(); ++iter)
    {
        if ( iter->clusterID == UNCLASSIFIED )
        {
            if ( expandCluster(*iter, clusterID) != FAILURE )
            {
                clusterID += 1;
            }
        }
    }

    return 0;
}

int DBSCAN::expandCluster(Point point, int clusterID)
{    
    vector<int> clusterSeeds = calculateCluster(point);

    if ( clusterSeeds.size() < m_minPoints )
    {
        // We need to find the specific point in m_points to update its ID
        // Since 'point' is a copy, we iterate or use index if available.
        // For efficiency in this specific implementation structure:
        // m_points[point.originalIndex] might be safer if sorted, but here we scan.
        // Or simpler: calculateCluster returned indices into m_points. 
        // We know 'point' corresponds to one of them if distance is 0, but 'point' passed here 
        // came from the outer loop iterator. 
        // Let's rely on finding the point by index or reference in a real optimize implementation.
        // For this port, let's find the point in m_points that matches 'point's originalIndex.
        
        // Optimization: In the run() loop, we are iterating. 
        // Ideally we would pass index to expandCluster.
        // Let's fix the logic below to update m_points correctly.
        
        for(size_t i=0; i<m_points.size(); ++i) {
            if(m_points[i].originalIndex == point.originalIndex) {
                 m_points[i].clusterID = NOISE;
                 break;
            }
        }
        return FAILURE;
    }
    else
    {
        int index = 0, indexCorePoint = 0;
        vector<int>::iterator iterSeeds;
        for( iterSeeds = clusterSeeds.begin(); iterSeeds != clusterSeeds.end(); ++iterSeeds)
        {
            m_points.at(*iterSeeds).clusterID = clusterID;
            if (m_points.at(*iterSeeds).originalIndex == point.originalIndex)
            {
                indexCorePoint = index;
            }
            ++index;
        }
        
        // Remove the core point itself from seeds to process neighbors
        if (!clusterSeeds.empty() && indexCorePoint < clusterSeeds.size()) {
             clusterSeeds.erase(clusterSeeds.begin()+indexCorePoint);
        }

        for( vector<int>::size_type i = 0, n = clusterSeeds.size(); i < n; ++i )
        {
            vector<int> clusterNeighors = calculateCluster(m_points.at(clusterSeeds[i]));

            if ( clusterNeighors.size() >= m_minPoints )
            {
                vector<int>::iterator iterNeighors;
                for ( iterNeighors = clusterNeighors.begin(); iterNeighors != clusterNeighors.end(); ++iterNeighors )
                {
                    if ( m_points.at(*iterNeighors).clusterID == UNCLASSIFIED || m_points.at(*iterNeighors).clusterID == NOISE )
                    {
                        if ( m_points.at(*iterNeighors).clusterID == UNCLASSIFIED )
                        {
                            clusterSeeds.push_back(*iterNeighors);
                            n = clusterSeeds.size();
                        }
                        m_points.at(*iterNeighors).clusterID = clusterID;
                    }
                }
            }
        }

        return SUCCESS;
    }
}

vector<int> DBSCAN::calculateCluster(Point point)
{
    int index = 0;
    vector<Point>::iterator iter;
    vector<int> clusterIndex;
    for( iter = m_points.begin(); iter != m_points.end(); ++iter)
    {
        if ( calculateDistance(point, *iter) <= m_epsilon )
        {
            clusterIndex.push_back(index);
        }
        index++;
    }
    return clusterIndex;
}

vector<int> DBSCAN::getClusterLabels() {
    vector<int> labels;
    for (const auto& p : m_points) {
        labels.push_back(p.clusterID == NOISE ? -1 : p.clusterID);
    }
    return labels;
}

// Modified to use Cosine Distance or Euclidean Distance for High Dimensions
// The provided implementation used 3D pow(x-x,2).
// Since we have N-dimensional embeddings, we must adapt this.
// Standard DBSCAN uses Euclidean distance.
// Epsilon is a distance threshold.
// High dimensional embeddings (normalized) are often compared with Cosine Similarity.
// Cosine Distance = 1 - Cosine Similarity.
// If using Euclidean on normalized vectors: ||a-b||^2 = 2(1 - cos(theta)).
// So Euclidean is monotonic with Cosine Distance for normalized vectors.
// We will use Squared Euclidean Distance here for performance (avoid sqrt)
// and strict adherence to the request of "embedding".
inline double DBSCAN::calculateDistance(const Point& pointCore, const Point& pointTarget )
{
    double distSq = 0.0;
    for(size_t i=0; i<pointCore.embedding.size(); ++i) {
        double d = pointCore.embedding[i] - pointTarget.embedding[i];
        distSq += d*d;
    }
    return distSq;
}
