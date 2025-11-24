# Tool Classification and Clustering System

A C++ implementation for classifying and clustering tools using semantic embeddings and similarity-based classification.

## Features

- **Tool Loading**: Parse tool definitions from JSON files (BFCL format)
- **Semantic Encoding**: Generate embeddings for tool names and descriptions (currently using mock implementation, ready for Gemma 300m integration)
- **Classification**: Classify tools into predefined primary classes based on semantic similarity
- **Clustering**: Perform K-means clustering within each primary class
- **Export Results**: Save classification and clustering results to JSON

## Architecture

```
┌─────────────────┐
│  JSON Tool Data │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Tool Loading   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌──────────────────┐
│    Encoding     │◄────│ Gemma 300m Model │
│   (Embeddings)  │     │   (Integration)  │
└────────┬────────┘     └──────────────────┘
         │
         ▼
┌─────────────────┐     ┌──────────────────┐
│ Classification  │◄────│ Primary Classes  │
│   (Similarity)  │     │   (Predefined)   │
└────────┬────────┘     └──────────────────┘
         │
         ▼
┌─────────────────┐
│   Clustering    │
│    (K-means)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Results Export  │
│      (JSON)     │
└─────────────────┘
```

## Prerequisites

- **C++ Compiler**: GCC 7.5+, Clang 7.0+, or MSVC 2019+
- **CMake**: Version 3.15 or higher
- **nlohmann/json**: Will be automatically downloaded if not found

## Building

### Option 1: Using CMake (Recommended)

```bash
cd cpp_tool_classifier
mkdir build
cd build
cmake ..
make
```

### Option 2: Manual Compilation with nlohmann/json

If you have nlohmann/json installed system-wide:

```bash
cd cpp_tool_classifier
g++ -std=c++17 -O3 tool_classifier.cpp -o tool_classifier
```

### Option 3: Docker Build (Coming Soon)

```bash
docker build -t tool-classifier .
docker run -v $(pwd):/workspace tool-classifier
```

## Usage

### Basic Usage

```bash
./tool_classifier
```

This will:
1. Load tools from `../data/BFCL_v3_simple.json`
2. Generate embeddings for each tool
3. Classify tools into 5 primary classes
4. Create 3 clusters within each class
5. Save results to `classification_results.json`

### Custom Parameters

```bash
./tool_classifier --input ../data/BFCL_v3_multiple.json \
                  --output results.json \
                  --clusters 5
```

### Command-Line Options

- `--input <path>`: Input JSON file path (default: `../data/BFCL_v3_simple.json`)
- `--output <path>`: Output JSON file path (default: `classification_results.json`)
- `--clusters <num>`: Number of clusters per class (default: 3)
- `--help`: Display help message

## Primary Classes

The system currently uses 5 predefined primary classes:

1. **Mathematics**: Mathematical operations including geometry, algebra, calculus
2. **Physics**: Physics-related calculations including mechanics, electromagnetism
3. **Travel & Navigation**: Travel planning, route finding, location services
4. **Data & Numbers**: Number theory, factorization, numerical analysis
5. **API & Web Services**: REST API calls, web services, external data retrieval

## Output Format

The output JSON file contains:

```json
{
  "summary": {
    "total_tools": 100,
    "primary_classes": 5,
    "clusters_per_class": 3
  },
  "classes": [
    {
      "name": "Mathematics",
      "description": "Mathematical operations...",
      "tool_count": 25,
      "clusters": [
        {
          "cluster_id": 0,
          "tool_count": 8,
          "tools": [
            {
              "id": "simple_0",
              "name": "calculate_triangle_area",
              "description": "Calculate the area of a triangle..."
            }
          ]
        }
      ]
    }
  ]
}
```

## Integrating with Gemma 300m

The current implementation uses a mock embedding generator. To integrate with the actual Gemma 300m model:

### Option 1: Use the Python Bridge (Recommended)

We provide a Python script that uses the Gemma 300m model:

```bash
# Install dependencies
pip install torch transformers

# Use the Python embedding service
python gemma_encoder.py --model google/gemma-300m
```

Then modify the C++ code to call the Python service via HTTP or file-based communication.

### Option 2: Use ONNX Runtime

Convert Gemma 300m to ONNX format and integrate with ONNX Runtime C++ API:

```bash
# Install ONNX Runtime
# Update CMakeLists.txt to link against ONNX Runtime
```

### Option 3: Use llama.cpp

Build Gemma 300m with llama.cpp and link against the C++ library:

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make
```

## Algorithm Details

### Cosine Similarity

Used to measure semantic similarity between tool embeddings and class embeddings:

```
similarity(A, B) = (A · B) / (||A|| × ||B||)
```

### K-means Clustering

Iterative algorithm to partition tools into K clusters:

1. Initialize K centroids randomly
2. Assign each tool to nearest centroid
3. Update centroids as mean of assigned tools
4. Repeat until convergence

## Performance

- **Loading**: ~1000 tools/second
- **Embedding**: Depends on model implementation
- **Classification**: O(N × C) where N=tools, C=classes
- **Clustering**: O(N × K × I) where K=clusters, I=iterations

## Extending the System

### Adding New Primary Classes

Edit `initializePrimaryClasses()` in `tool_classifier.cpp`:

```cpp
classDefinitions.push_back({"Class Name", "Description"});
```

### Custom Clustering Algorithms

Implement alternative clustering methods (hierarchical, DBSCAN):

```cpp
std::vector<Cluster> hierarchicalClustering(const std::vector<Tool>& tools);
```

### Different Similarity Metrics

Replace `cosineSimilarity()` with:

- Euclidean distance
- Manhattan distance
- Jaccard similarity

## Troubleshooting

### Build Errors

**Error**: `nlohmann/json.hpp: No such file or directory`
- **Solution**: CMake will automatically download it. Ensure you have internet access.

**Error**: `C++17 not supported`
- **Solution**: Update your compiler to GCC 7.5+, Clang 7.0+, or MSVC 2019+

### Runtime Errors

**Error**: `Could not open file`
- **Solution**: Check the input file path. Use absolute paths if necessary.

**Error**: `Vector dimensions don't match`
- **Solution**: Ensure all embeddings have the same dimension (default: 768)

## License

Apache 2.0 - See LICENSE file for details

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## Contact

For questions or issues, please open an issue on GitHub or contact the BFCL team.

## Acknowledgments

- Berkeley Function Calling Leaderboard (BFCL) project
- nlohmann/json library
- Gemma model by Google
