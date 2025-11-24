// Gemma embedding client implementation.

#include "tool_taxonomy/embedder.h"

#include <curl/curl.h>

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include <boost/property_tree/json_parser.hpp>

#include "tool_taxonomy/utils.h"

namespace tool_taxonomy {
namespace {

using boost::property_tree::ptree;

struct HttpResponse {
    long status = 0;
    std::string body;
};

size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    const size_t total = size * nmemb;
    auto* buffer = static_cast<std::string*>(userp);
    buffer->append(static_cast<char*>(contents), total);
    return total;
}

HttpResponse Post(const std::string& url,
                  const std::string& payload,
                  const std::vector<std::string>& headers) {
    CURL* curl = curl_easy_init();
    if (!curl) {
        throw std::runtime_error("Failed to initialize libcurl");
    }

    std::string response_body;
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_POST, 1L);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, payload.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, payload.size());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_body);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 120L);

    struct curl_slist* header_list = nullptr;
    for (const auto& header : headers) {
        header_list = curl_slist_append(header_list, header.c_str());
    }
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, header_list);

    const CURLcode res = curl_easy_perform(curl);
    if (res != CURLE_OK) {
        std::string msg = "curl_easy_perform() failed: ";
        msg += curl_easy_strerror(res);
        curl_slist_free_all(header_list);
        curl_easy_cleanup(curl);
        throw std::runtime_error(msg);
    }

    long status = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &status);
    curl_slist_free_all(header_list);
    curl_easy_cleanup(curl);
    return HttpResponse{status, std::move(response_body)};
}

bool ExtractVector(const ptree& node, const std::string& key, std::vector<float>& out) {
    if (auto child = node.get_child_optional(key)) {
        for (const auto& entry : *child) {
            out.push_back(entry.second.get_value<float>());
        }
        return !out.empty();
    }
    return false;
}

std::vector<float> ExtractEmbedding(const ptree& root) {
    if (auto embedding = root.get_child_optional("embedding")) {
        std::vector<float> result;
        if (ExtractVector(*embedding, "value", result) || ExtractVector(*embedding, "values", result)) {
            return result;
        }
    }
    if (auto embeddings = root.get_child_optional("embeddings")) {
        for (const auto& entry : *embeddings) {
            std::vector<float> result;
            if (ExtractVector(entry.second, "value", result) ||
                ExtractVector(entry.second, "values", result)) {
                return result;
            }
        }
    }
    if (auto data = root.get_child_optional("data")) {
        for (const auto& entry : *data) {
            if (auto emb_node = entry.second.get_child_optional("embedding")) {
                std::vector<float> result;
                for (const auto& value : *emb_node) {
                    result.push_back(value.second.get_value<float>());
                }
                if (!result.empty()) {
                    return result;
                }
            }
        }
    }
    if (auto predictions = root.get_child_optional("predictions")) {
        for (const auto& prediction : *predictions) {
            if (auto emb_list = prediction.second.get_child_optional("embeddings")) {
                for (const auto& emb : *emb_list) {
                    std::vector<float> result;
                    if (ExtractVector(emb.second, "values", result) ||
                        ExtractVector(emb.second, "value", result)) {
                        return result;
                    }
                }
            }
        }
    }
    throw std::runtime_error("Embedding vector missing from response.");
}

}  // namespace

GemmaEmbeddingClient::GemmaEmbeddingClient(ProgramOptions options) : options_(std::move(options)) {
    if (options_.api_key.empty()) {
        if (const char* env = std::getenv("GOOGLE_API_KEY")) {
            options_.api_key = env;
        }
    }
    if (options_.api_key.empty()) {
        throw std::runtime_error("API key is required for Gemma embedding calls.");
    }
}

std::vector<float> GemmaEmbeddingClient::Embed(const std::string& text) {
    if (text.empty()) {
        return {};
    }
    if (auto it = cache_.find(text); it != cache_.end()) {
        return it->second;
    }
    auto embedding = RequestEmbedding(text);
    cache_.emplace(text, embedding);
    return embedding;
}

std::vector<float> GemmaEmbeddingClient::RequestEmbedding(const std::string& text) {
    ptree body;
    body.put("model", options_.gemma_model);
    ptree parts_array;
    ptree part;
    part.put("text", text);
    parts_array.push_back({"", part});
    ptree content;
    content.add_child("parts", parts_array);
    body.add_child("content", content);

    std::ostringstream oss;
    boost::property_tree::write_json(oss, body, false);

    const auto response = Post(BuildUrl(), oss.str(), BuildHeaders());
    if (response.status < 200 || response.status >= 300) {
        std::ostringstream err;
        err << "Gemma API call failed (" << response.status << "): " << response.body;
        throw std::runtime_error(err.str());
    }

    ptree parsed;
    std::istringstream iss(response.body);
    boost::property_tree::read_json(iss, parsed);
    return ExtractEmbedding(parsed);
}

std::string GemmaEmbeddingClient::BuildUrl() const {
    if (options_.auth_mode == ApiAuthMode::kQueryParam) {
        if (options_.api_key.empty()) {
            throw std::runtime_error("API key missing for query parameter auth.");
        }
        if (options_.gemma_endpoint.find('?') == std::string::npos) {
            return options_.gemma_endpoint + "?key=" + options_.api_key;
        }
        return options_.gemma_endpoint + "&key=" + options_.api_key;
    }
    return options_.gemma_endpoint;
}

std::vector<std::string> GemmaEmbeddingClient::BuildHeaders() const {
    std::vector<std::string> headers = {"Content-Type: application/json"};
    if (options_.auth_mode == ApiAuthMode::kBearerHeader) {
        std::string value = options_.api_key;
        if (!options_.api_key_prefix.empty()) {
            value = options_.api_key_prefix + options_.api_key;
        }
        headers.push_back(options_.api_key_header + ": " + value);
    }
    return headers;
}

}  // namespace tool_taxonomy
