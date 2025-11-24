# Tool Classification and Clustering System

A C++ implementation for classifying and clustering software tools using text embeddings and machine learning techniques. This system can encode tool descriptions using Google Gemma 300M model (or a simplified encoder), classify them into primary categories, and further cluster them within each category.

## Features

- **Tool Information Management**: Structures to store tool names, descriptions, and parameters
- **Text Encoding**: 
  - Full version: Integration with Google Gemma 300M model via ONNX Runtime
  - Simplified version: Hash-based pseudo-embeddings for testing
- **Primary Classification**: Classifies tools into predefined categories using cosine similarity
- **Sub-clustering**: K-means clustering within each primary category
- **Quality Metrics**: Calculates cluster cohesion and other quality metrics
- **JSON Export**: Exports classification and clustering results to JSON format

## Architecture

```
┌─────────────────────────────────────────────────┐
│                Tool Manager                      │
├─────────────────────────────────────────────────┤
│  1. Load Tools from Source                       │
│  2. Encode Tool Descriptions                     │
│  3. Classify into Primary Classes                │
│  4. Sub-cluster within Classes                   │
│  5. Calculate Metrics & Export                   │
└─────────────────────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│   Encoder    │ │  Similarity  │ │  Clustering  │
│  (Gemma/     │ │  Calculator  │ │  (K-Means)   │
│   Simple)    │ │   (Cosine)   │ │              │
└──────────────┘ └──────────────┘ └──────────────┘
```

## Prerequisites

### For Simplified Version
- C++17 compatible compiler (g++ 7.0+ or clang++ 5.0+)
- CMake 3.14+
- Git (for fetching dependencies)

### For Full Version (with Gemma)
All of the above plus:
- ONNX Runtime 1.16.3+
- Gemma 300M ONNX model file

## Building

### Quick Build (Simplified Version)

```bash
# Make build script executable
chmod +x build.sh

# Build simplified version (recommended for testing)
./build.sh --simple

# Or build with debug symbols
./build.sh --simple --debug
```

### Manual Build

```bash
# Create build directory
mkdir build && cd build

# Configure (simplified version)
cmake .. -DBUILD_SIMPLE_VERSION=ON

# Build
make -j$(nproc)
```

### Full Version Build (with ONNX Runtime)

```bash
# Build full version (requires ONNX Runtime)
./build.sh

# Or manually
mkdir build && cd build
cmake ..
make -j$(nproc)
```

## Usage

### Running the Tool Classifier

```bash
# Using default configuration
./build/tool_classifier_simple

# Using custom configuration
./build/tool_classifier_simple my_config.json
```

### Configuration File

The system uses a JSON configuration file (`tool_config.json`) with the following structure:

```json
{
  "modelPath": "gemma_300m.onnx",           // Path to ONNX model (full version only)
  "numSubClusters": 3,                      // Number of sub-clusters per primary class
  "similarityThreshold": 0.7,               // Minimum similarity for classification
  "maxIterations": 100,                     // K-means max iterations
  "convergenceThreshold": 0.001,            // K-means convergence threshold
  "embeddingDimension": 256,                // Embedding vector dimension
  "primaryClasses": [
    {
      "name": "Data Processing",
      "description": "Tools for reading, writing, and manipulating data"
    },
    {
      "name": "Communication",
      "description": "Tools for network requests and messaging"
    }
    // ... more classes
  ]
}
```

## Primary Classes

The system classifies tools into these primary categories:

1. **Data Processing**: File I/O, database operations, data transformation
2. **Communication**: HTTP requests, email, webhooks, notifications
3. **Analysis & Intelligence**: Text analysis, image processing, calculations
4. **External Services**: Weather, translation, geocoding, stock prices
5. **Utility & Helpers**: Logging, caching, scheduling, validation
6. **Other**: Tools that don't fit other categories

## Output

The system generates a JSON file (`tool_classification_results.json`) containing:

```json
{
  "timestamp": 1234567890,
  "config": { /* configuration used */ },
  "summary": {
    "totalTools": 25,
    "numPrimaryClasses": 6
  },
  "classes": [
    {
      "name": "Data Processing",
      "description": "...",
      "toolCount": 6,
      "subclusters": [
        {
          "id": 0,
          "tools": [
            {
              "name": "file_reader",
              "description": "Reads content from files",
              "parameters": { /* ... */ }
            }
          ]
        }
      ]
    }
  ]
}
```

## Algorithms

### Text Encoding

**Simplified Version**: Uses hash-based pseudo-embeddings
- Hashes each word in the text
- Generates deterministic embeddings based on word hashes
- L2 normalization for unit vectors

**Full Version**: Uses Google Gemma 300M model
- Tokenizes input text
- Runs inference through ONNX Runtime
- Returns normalized embeddings

### Classification

- Computes cosine similarity between tool embeddings and primary class embeddings
- Assigns tool to class with highest similarity above threshold
- Falls back to "Other" category if below threshold

### Clustering

Uses K-Means++ algorithm:
1. Smart initialization of centroids
2. Iterative assignment and update
3. Convergence based on centroid movement

## Example Tool List

The system includes example tools such as:
- `file_reader`: Reads content from files
- `web_scraper`: Scrapes data from websites
- `database_query`: Executes SQL queries
- `api_caller`: Makes HTTP API calls
- `text_analyzer`: Analyzes text content
- `email_sender`: Sends emails
- `calculator`: Performs calculations
- And many more...

## Customization

### Adding New Tools

Edit the `getToolList()` function in the source code to add new tools:

```cpp
toolsJson.push_back({
    {"name", "my_tool"},
    {"description", "Tool description"},
    {"parameters", {
        {"param1", "string"},
        {"param2", "number"}
    }}
});
```

### Modifying Primary Classes

Update the configuration file to change primary classes:

```json
"primaryClasses": [
    {
        "name": "Custom Category",
        "description": "Description of the category"
    }
]
```

### Adjusting Clustering Parameters

Modify these parameters in the config file:
- `numSubClusters`: Number of clusters within each primary class
- `similarityThreshold`: Classification sensitivity (0.0-1.0)
- `maxIterations`: K-means iteration limit
- `convergenceThreshold`: K-means convergence criterion

## Performance Considerations

- **Embedding Dimension**: Higher dimensions provide more accurate representations but increase computation
- **Number of Clusters**: More clusters provide finer granularity but may create sparse groups
- **Similarity Threshold**: Lower values include more tools in primary classes; higher values are more selective

## Troubleshooting

### Build Errors

1. **Missing nlohmann/json**: CMake should automatically fetch this dependency. Ensure you have internet connection.

2. **ONNX Runtime not found**: For full version, ensure ONNX Runtime is installed or use the simplified version:
   ```bash
   ./build.sh --simple
   ```

3. **Compiler errors**: Ensure you have C++17 support:
   ```bash
   g++ --version  # Should be 7.0+
   ```

### Runtime Issues

1. **Config file not found**: Ensure `tool_config.json` is in the same directory as the executable

2. **Segmentation fault**: Check that the config file is valid JSON

3. **Poor clustering results**: Try adjusting the similarity threshold and number of clusters

## Integration with Gemma Model

To use the actual Gemma 300M model:

1. Download the Gemma 300M model in ONNX format
2. Update the `modelPath` in the configuration file
3. Build the full version (without `--simple` flag)
4. Ensure ONNX Runtime is properly installed

## License

This implementation is provided as-is for educational and research purposes.

## Contributing

To extend this system:

1. Add new similarity metrics in `SimilarityCalculator`
2. Implement alternative clustering algorithms (hierarchical, DBSCAN)
3. Add support for other embedding models
4. Enhance the tool discovery mechanism
5. Implement incremental learning for dynamic tool addition

## Contact

For questions or issues, please refer to the source code documentation or create an issue in the project repository.