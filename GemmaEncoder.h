#ifndef GEMMA_ENCODER_H
#define GEMMA_ENCODER_H

#include <vector>
#include <string>

// Configuration for embedding dimension
const int EMBEDDING_DIM = 256;

class GemmaEncoder {
public:
    GemmaEncoder(const std::string& model_path = "gemma-300m.gguf");
    
    // Encodes text into a vector
    std::vector<float> encode(const std::string& text);
};

#endif // GEMMA_ENCODER_H
