// Gemma embedding client.

#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "tool_taxonomy/options.h"

namespace tool_taxonomy {

class GemmaEmbeddingClient {
   public:
    explicit GemmaEmbeddingClient(ProgramOptions options);
    std::vector<float> Embed(const std::string& text);

   private:
    ProgramOptions options_;
    std::unordered_map<std::string, std::vector<float>> cache_;

    std::vector<float> RequestEmbedding(const std::string& text);
    std::string BuildUrl() const;
    std::vector<std::string> BuildHeaders() const;
};

}  // namespace tool_taxonomy
