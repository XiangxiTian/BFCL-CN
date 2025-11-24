# Implementation Notes

## Overview

This C++ implementation provides a complete tool classification and clustering system that:

1. **Parses tools** from JSON files (supports JSONL format)
2. **Encodes tools** using embeddings (currently mock implementation, ready for Gemma 300M)
3. **Classifies tools** into primary classes based on similarity
4. **Clusters tools** within each primary class using K-means

## Key Components

### 1. Tool Structure (`struct Tool`)
- Stores tool name, description, parameters
- Holds embedding vector
- Tracks primary class and cluster assignment

### 2. JSON Parser (`SimpleJSONParser`)
- Handles JSONL format (one JSON object per line)
- Extracts function objects from "function" arrays
- Properly handles escaped strings and nested structures

### 3. Embedding Model (`GemmaEmbeddingModel`)
- **Current**: Mock implementation using hash-based embeddings
- **Ready for**: ONNX Runtime integration with actual Gemma 300M model
- Normalizes embeddings to unit vectors

### 4. Similarity Calculator (`SimilarityCalculator`)
- Implements cosine similarity between embedding vectors
- Used for both classification and clustering

### 5. K-Means Clusterer (`KMeansClusterer`)
- Clusters embeddings using cosine similarity
- Supports configurable number of clusters
- Iterative refinement until convergence

### 6. Tool Classifier (`ToolClassifier`)
- Manages primary classes
- Classifies tools into primary classes
- Performs sub-clustering within each class

## Integration with Google Gemma 300M

To integrate with the actual Gemma 300M model:

### Step 1: Convert Model to ONNX
```bash
# Use HuggingFace transformers or similar to export Gemma to ONNX
# Example: Use optimum library
```

### Step 2: Update `GemmaEmbeddingModel::encode()`

Replace the mock implementation with:

```cpp
std::vector<float> encode(const std::string& text) override {
    // 1. Tokenize text
    std::vector<int64_t> tokens = tokenize(text);
    
    // 2. Create input tensor
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);
    
    std::vector<int64_t> input_shape = {1, static_cast<int64_t>(tokens.size())};
    Ort::Value input_tensor = Ort::Value::CreateTensor<int64_t>(
        memory_info, tokens.data(), tokens.size(),
        input_shape.data(), input_shape.size());
    
    // 3. Run inference
    const char* input_names[] = {"input_ids"};
    const char* output_names[] = {"last_hidden_state"};
    
    auto output_tensors = session_.Run(
        Ort::RunOptions{nullptr},
        input_names, &input_tensor, 1,
        output_names, 1);
    
    // 4. Extract embeddings (mean pooling or use [CLS] token)
    float* output_data = output_tensors[0].GetTensorMutableData<float>();
    // ... process to get embedding vector
    
    // 5. Normalize
    return normalize(embedding);
}
```

### Step 3: Initialize ONNX Session

In `initialize()`:

```cpp
bool initialize() override {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "GemmaEmbedding");
    Ort::SessionOptions session_options;
    
    // Optional: Use GPU
    // OrtCUDAProviderOptions cuda_options{};
    // session_options.AppendExecutionProvider_CUDA(cuda_options);
    
    session_ = std::make_unique<Ort::Session>(
        env, "gemma-2b-it.onnx", session_options);
    
    return true;
}
```

## Performance Considerations

1. **Batching**: Process multiple tools in batches for better throughput
2. **Caching**: Cache embeddings for tools that don't change
3. **Parallel Processing**: Use threads to encode multiple tools simultaneously
4. **GPU Acceleration**: Use CUDA provider for ONNX Runtime if available

## Extending Primary Classes

To add custom primary classes, modify `main()`:

```cpp
classifier.addPrimaryClass("YourCategory", 
    "Detailed description of what tools belong to this category");
```

The description should be comprehensive as it's used for similarity matching.

## Clustering Parameters

Adjust clustering behavior:

```cpp
// In ToolClassifier::clusterToolsInClasses()
KMeansClusterer clusterer(
    k,                    // Number of clusters
    100,                  // Max iterations
    1e-4f                 // Convergence tolerance
);
```

## Future Enhancements

1. **Hierarchical Clustering**: Support for nested clustering
2. **DBSCAN**: Alternative clustering algorithm for variable cluster sizes
3. **Visualization**: Generate plots/diagrams of clusters
4. **API Integration**: Fetch tools dynamically from APIs
5. **Incremental Clustering**: Add new tools without re-clustering all
6. **Evaluation Metrics**: Silhouette score, Davies-Bouldin index

## Testing

Test with different datasets:

```bash
# Small dataset
./tool_clustering data/BFCL_v3_simple.json 3

# Large dataset
./tool_clustering data/BFCL_v3_multiple.json 5 results_large.csv
```

## Troubleshooting

### Low Quality Clusters
- Increase number of clusters per class
- Improve primary class descriptions
- Use better embedding model

### Slow Performance
- Enable batching
- Use GPU acceleration
- Reduce embedding dimensions
- Process in parallel

### Memory Issues
- Process tools in batches
- Reduce embedding dimensions
- Use streaming JSON parser
