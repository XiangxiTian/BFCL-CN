# Integrating with Gemma 300m Model

This guide explains how to integrate the actual Gemma 300m model for text encoding in the C++ tool classifier.

## Overview

The current implementation uses a **mock embedding generator** that creates deterministic but semantically meaningless embeddings based on text hashing. For production use, you should integrate with the actual Gemma 300m model.

## Integration Options

### Option 1: Python Bridge (Recommended for Quick Setup)

Use the provided Python script as a bridge between C++ and the Gemma model.

#### Step 1: Install Dependencies

```bash
pip install torch transformers numpy
```

#### Step 2: Generate Embeddings

```bash
python gemma_encoder.py --input ../data/BFCL_v3_simple.json \
                        --output tool_embeddings.json \
                        --model google/gemma-300m \
                        --device cuda  # or cpu
```

#### Step 3: Modify C++ to Load Pre-computed Embeddings

Update `tool_classifier.cpp` to load embeddings from the JSON file:

```cpp
void loadToolsWithEmbeddings(const std::string& filepath) {
    std::ifstream file(filepath);
    json j = json::parse(file);
    
    for (const auto& tool_json : j) {
        Tool tool;
        tool.id = tool_json["id"];
        tool.name = tool_json["name"];
        tool.description = tool_json["description"];
        tool.embedding = tool_json["embedding"].get<std::vector<float>>();
        tools.push_back(tool);
    }
}
```

**Pros:**
- Easy to set up
- No C++ ML dependencies
- Can use latest HuggingFace models

**Cons:**
- Two-step process (Python then C++)
- Not real-time encoding

---

### Option 2: HTTP Service

Run Gemma as a microservice and call it from C++.

#### Step 1: Create Flask Service

```python
# gemma_service.py
from flask import Flask, request, jsonify
from gemma_encoder import GemmaEncoder

app = Flask(__name__)
encoder = GemmaEncoder()

@app.route('/encode', methods=['POST'])
def encode():
    texts = request.json['texts']
    embeddings = encoder.encode(texts)
    return jsonify({'embeddings': embeddings.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

#### Step 2: Update C++ to Call HTTP Service

```cpp
#include <curl/curl.h>

std::vector<float> generateEmbedding(const std::string& text) {
    CURL *curl = curl_easy_init();
    std::string response;
    
    if(curl) {
        json request_data = {{"texts", {text}}};
        std::string post_data = request_data.dump();
        
        curl_easy_setopt(curl, CURLOPT_URL, "http://localhost:5000/encode");
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, post_data.c_str());
        
        // ... perform request and parse response
    }
    
    return embedding;
}
```

**Pros:**
- Real-time encoding
- Clean separation of concerns
- Easy to scale

**Cons:**
- Network overhead
- Requires running service

---

### Option 3: ONNX Runtime (Production Ready)

Convert Gemma to ONNX format and use ONNX Runtime C++ API.

#### Step 1: Convert Model to ONNX

```python
import torch
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("google/gemma-300m")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-300m")

# Create dummy input
dummy_input = tokenizer("Hello", return_tensors="pt")

# Export to ONNX
torch.onnx.export(
    model,
    tuple(dummy_input.values()),
    "gemma_300m.onnx",
    input_names=['input_ids', 'attention_mask'],
    output_names=['last_hidden_state'],
    dynamic_axes={
        'input_ids': {0: 'batch', 1: 'sequence'},
        'attention_mask': {0: 'batch', 1: 'sequence'},
        'last_hidden_state': {0: 'batch', 1: 'sequence'}
    }
)
```

#### Step 2: Install ONNX Runtime

```bash
# Ubuntu/Debian
wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.0/onnxruntime-linux-x64-1.16.0.tgz
tar -xzf onnxruntime-linux-x64-1.16.0.tgz

# macOS
brew install onnxruntime
```

#### Step 3: Update CMakeLists.txt

```cmake
find_package(onnxruntime REQUIRED)
target_link_libraries(tool_classifier PRIVATE onnxruntime)
```

#### Step 4: Implement in C++

```cpp
#include <onnxruntime_cxx_api.h>

class GemmaEncoder {
private:
    Ort::Env env;
    Ort::Session session;
    
public:
    GemmaEncoder(const std::string& model_path) 
        : env(ORT_LOGGING_LEVEL_WARNING, "GemmaEncoder"),
          session(env, model_path.c_str(), Ort::SessionOptions{}) {}
    
    std::vector<float> encode(const std::string& text) {
        // Tokenize text
        auto tokens = tokenize(text);
        
        // Prepare input tensors
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<int64_t>(
            memory_info, tokens.data(), tokens.size(), 
            input_shape.data(), input_shape.size()
        );
        
        // Run inference
        auto output_tensors = session.Run(
            Ort::RunOptions{nullptr},
            input_names.data(), &input_tensor, 1,
            output_names.data(), 1
        );
        
        // Extract embeddings
        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        std::vector<float> embedding(output_data, output_data + embedding_dim);
        
        return embedding;
    }
};
```

**Pros:**
- Native C++ integration
- Fast inference
- Production-ready
- No external services needed

**Cons:**
- More complex setup
- Larger binary size

---

### Option 4: llama.cpp (Lightweight)

Use llama.cpp which supports Gemma models.

#### Step 1: Build llama.cpp

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make
```

#### Step 2: Convert Model

```bash
python convert.py /path/to/gemma-300m --outfile gemma-300m.gguf
```

#### Step 3: Link in C++

```cpp
#include "llama.h"

class GemmaEncoder {
private:
    llama_model* model;
    llama_context* ctx;
    
public:
    GemmaEncoder(const std::string& model_path) {
        llama_backend_init(false);
        
        llama_model_params model_params = llama_model_default_params();
        model = llama_load_model_from_file(model_path.c_str(), model_params);
        
        llama_context_params ctx_params = llama_context_default_params();
        ctx = llama_new_context_with_model(model, ctx_params);
    }
    
    std::vector<float> encode(const std::string& text) {
        // Tokenize
        std::vector<llama_token> tokens = llama_tokenize(ctx, text, true);
        
        // Get embeddings
        llama_eval(ctx, tokens.data(), tokens.size(), 0, 0);
        
        // Extract embedding from last token
        float* embedding = llama_get_embeddings(ctx);
        return std::vector<float>(embedding, embedding + embedding_dim);
    }
};
```

**Pros:**
- Lightweight
- CPU-optimized
- Easy to use

**Cons:**
- May not support all model features
- Quantization may reduce accuracy

---

## Recommended Approach

For **development and testing**: Use **Option 1 (Python Bridge)**

For **production deployment**: Use **Option 3 (ONNX Runtime)**

For **resource-constrained environments**: Use **Option 4 (llama.cpp)**

For **real-time applications**: Use **Option 2 (HTTP Service)**

## Performance Considerations

### Embedding Generation
- **Batch processing**: Encode multiple tools at once
- **Caching**: Cache embeddings to avoid recomputation
- **Quantization**: Use INT8 or FP16 for faster inference

### Memory Usage
- **Model size**: Gemma 300m ≈ 600MB in FP32, 300MB in FP16
- **Embedding storage**: 768 floats × 4 bytes = 3KB per tool
- For 10,000 tools: ~30MB for embeddings

### Inference Speed
- **CPU**: ~50-100 tools/second
- **GPU**: ~500-1000 tools/second
- **Batch size 32**: 3-5x speedup

## Testing Your Integration

### 1. Verify Embeddings

```cpp
auto embedding1 = generateEmbedding("calculate triangle area");
auto embedding2 = generateEmbedding("compute triangle space");
auto embedding3 = generateEmbedding("book flight ticket");

float sim_12 = cosineSimilarity(embedding1, embedding2);  // Should be high
float sim_13 = cosineSimilarity(embedding1, embedding3);  // Should be low

std::cout << "Similar: " << sim_12 << std::endl;    // Expected: > 0.7
std::cout << "Different: " << sim_13 << std::endl;  // Expected: < 0.3
```

### 2. Validate Dimensions

```cpp
assert(embedding.size() == 768);  // Gemma 300m embedding dimension
```

### 3. Check Normalization

```cpp
float norm = 0.0f;
for (float val : embedding) {
    norm += val * val;
}
norm = sqrt(norm);
assert(abs(norm - 1.0f) < 0.01);  // Should be normalized
```

## Troubleshooting

### Issue: Model not loading
- Check model path
- Verify model format (ONNX, GGUF, etc.)
- Ensure sufficient memory

### Issue: Slow inference
- Use GPU if available
- Increase batch size
- Enable model quantization

### Issue: Poor classification
- Verify embeddings are normalized
- Check similarity threshold
- Try different primary class definitions

## Additional Resources

- [ONNX Runtime Documentation](https://onnxruntime.ai/docs/)
- [llama.cpp GitHub](https://github.com/ggerganov/llama.cpp)
- [Gemma Model Card](https://huggingface.co/google/gemma-300m)
- [Transformers Documentation](https://huggingface.co/docs/transformers/)

## Next Steps

After integrating Gemma 300m:

1. **Benchmark**: Compare against baseline accuracy
2. **Optimize**: Tune batch size and model parameters
3. **Deploy**: Package with Docker or as standalone binary
4. **Monitor**: Track classification accuracy and performance
