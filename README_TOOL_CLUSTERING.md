# Tool Classification and Clustering System

A C++ implementation for classifying and clustering tools using Google Gemma 300M model for embeddings.

## Features

- **Tool Parsing**: Extracts tool information (name, description, parameters) from JSON files
- **Embedding Generation**: Uses Google Gemma 300M model to encode tool names and descriptions
- **Primary Classification**: Classifies tools into predefined primary classes based on similarity
- **Sub-clustering**: Further clusters tools within each primary class using K-means

## Requirements

- C++17 compatible compiler (GCC 7+, Clang 5+, MSVC 2017+)
- CMake 3.15 or higher
- (Optional) ONNX Runtime for actual Gemma model inference
- (Optional) nlohmann/json for better JSON parsing

## Building

### Basic Build (Mock Implementation)

```bash
mkdir build
cd build
cmake ..
make
```

This will build with a mock embedding implementation (hash-based embeddings).

### Build with ONNX Runtime (For Real Model Inference)

1. Download ONNX Runtime:
   ```bash
   # Download from https://github.com/microsoft/onnxruntime/releases
   # Extract to third_party/onnxruntime/
   ```

2. Build with ONNX support:
   ```bash
   mkdir build
   cd build
   cmake -DUSE_ONNX=ON ..
   make
   ```

### Build with nlohmann/json (Better JSON Parsing)

1. Download nlohmann/json:
   ```bash
   git clone https://github.com/nlohmann/json.git third_party/json
   ```

2. Build with JSON support:
   ```bash
   mkdir build
   cd build
   cmake -DUSE_NLOHMANN_JSON=ON ..
   make
   ```

## Usage

### Basic Usage

```bash
./tool_clustering [json_file] [clusters_per_class] [output_csv]
```

Parameters:
- `json_file`: Path to JSON file containing tools (default: `data/BFCL_v3_simple.json`)
- `clusters_per_class`: Number of clusters per primary class (default: 3)
- `output_csv`: Optional CSV file to save results

### Example

```bash
# Use default settings
./tool_clustering

# Specify custom JSON file and clustering parameters
./tool_clustering data/BFCL_v3_simple.json 5 results.csv
```

## Input Format

The script expects JSON files in the following format:

```json
{
  "function": [
    {
      "name": "tool_name",
      "description": "Tool description",
      "parameters": {
        "type": "dict",
        "properties": {...}
      }
    }
  ]
}
```

Or a JSON array of such objects (one per line).

## Primary Classes

The default primary classes are:
- **Mathematics**: Mathematical operations, calculations, algebra, geometry, calculus
- **Physics**: Physical calculations, mechanics, electromagnetism, thermodynamics
- **Travel**: Travel planning, directions, routes, itineraries, locations
- **Data**: Data manipulation, analysis, storage, retrieval
- **Communication**: Messaging, posting, social media, user interactions

You can modify these in the `main()` function of `tool_clustering.cpp`.

## Output

The script outputs:
1. Console output showing:
   - Number of tools loaded
   - Classification progress
   - Clustering results organized by primary class and cluster

2. Optional CSV file with columns:
   - Tool Name
   - Primary Class
   - Cluster ID
   - Description

## Implementation Details

### Embedding Model

Currently uses a mock implementation. To use the actual Google Gemma 300M model:

1. Convert Gemma model to ONNX format
2. Load it using ONNX Runtime
3. Implement the `encode()` method in `GemmaEmbeddingModel` class

### Similarity Calculation

Uses cosine similarity between embedding vectors to:
- Match tools to primary classes
- Calculate distances for K-means clustering

### Clustering Algorithm

Uses K-means clustering with:
- Cosine similarity as distance metric
- Random initialization of centroids
- Iterative refinement until convergence

## Customization

### Adding Primary Classes

Edit the `main()` function:

```cpp
classifier.addPrimaryClass("YourClass", "Description of your class");
```

### Changing Embedding Dimensions

Modify the model initialization:

```cpp
auto model = std::make_shared<GemmaEmbeddingModel>(your_dimension);
```

### Adjusting Clustering Parameters

Modify the `KMeansClusterer` constructor:

```cpp
KMeansClusterer clusterer(k, max_iterations, tolerance);
```

## Integration with ONNX Runtime

To integrate with actual Gemma model:

1. Load the ONNX model:
```cpp
Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "GemmaEmbedding");
Ort::Session session(env, "gemma-300m.onnx", Ort::SessionOptions{nullptr});
```

2. Prepare input tensors with tokenized text
3. Run inference
4. Extract embeddings from output tensor

## Performance Notes

- Mock implementation is fast but not semantically meaningful
- Real model inference will be slower but provide accurate embeddings
- Consider batching embeddings for better performance
- Use GPU acceleration if available

## Troubleshooting

### JSON Parsing Errors

If you encounter JSON parsing issues:
- Use `-DUSE_NLOHMANN_JSON=ON` for better parsing
- Ensure JSON file is valid
- Check file encoding (should be UTF-8)

### Model Loading Errors

If ONNX Runtime fails:
- Verify model file path
- Check ONNX Runtime version compatibility
- Ensure model is in ONNX format

### Memory Issues

For large tool sets:
- Process tools in batches
- Reduce embedding dimensions
- Use streaming JSON parser

## License

This code is provided as-is for the BFCL project.

## Future Improvements

- [ ] Real ONNX Runtime integration for Gemma model
- [ ] Better JSON parsing with nlohmann/json
- [ ] Parallel processing for embeddings
- [ ] GPU acceleration support
- [ ] Hierarchical clustering options
- [ ] Visualization of clusters
- [ ] API integration for getting tools dynamically
