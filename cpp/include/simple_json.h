#pragma once

#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace simplejson {

class JsonValue;
using JsonObject = std::map<std::string, JsonValue>;
using JsonArray = std::vector<JsonValue>;

class JsonValue {
  public:
    enum class Type { Null, Bool, Number, String, Object, Array };

    JsonValue() : value_(nullptr) {}
    JsonValue(std::nullptr_t) : value_(nullptr) {}
    JsonValue(bool b) : value_(b) {}
    JsonValue(double d) : value_(d) {}
    JsonValue(std::string s) : value_(std::move(s)) {}
    JsonValue(const char* s) : value_(std::string(s)) {}
    JsonValue(JsonObject obj) : value_(std::move(obj)) {}
    JsonValue(JsonArray arr) : value_(std::move(arr)) {}

    Type type() const { return static_cast<Type>(value_.index()); }

    bool IsNull() const { return std::holds_alternative<std::nullptr_t>(value_); }
    bool IsBool() const { return std::holds_alternative<bool>(value_); }
    bool IsNumber() const { return std::holds_alternative<double>(value_); }
    bool IsString() const { return std::holds_alternative<std::string>(value_); }
    bool IsObject() const { return std::holds_alternative<JsonObject>(value_); }
    bool IsArray() const { return std::holds_alternative<JsonArray>(value_); }

    bool AsBool() const { return std::get<bool>(value_); }
    double AsNumber() const { return std::get<double>(value_); }
    const std::string& AsString() const { return std::get<std::string>(value_); }
    const JsonObject& AsObject() const { return std::get<JsonObject>(value_); }
    const JsonArray& AsArray() const { return std::get<JsonArray>(value_); }
    JsonObject& AsObject() { return std::get<JsonObject>(value_); }
    JsonArray& AsArray() { return std::get<JsonArray>(value_); }

    bool Contains(const std::string& key) const {
        if (!IsObject()) {
            return false;
        }
        const auto& obj = AsObject();
        return obj.find(key) != obj.end();
    }

    const JsonValue& At(const std::string& key) const {
        const auto& obj = AsObject();
        auto it = obj.find(key);
        if (it == obj.end()) {
            throw std::runtime_error("key not found: " + key);
        }
        return it->second;
    }

    const JsonValue& operator[](size_t idx) const { return AsArray().at(idx); }

    std::string Dump(int indent = -1) const {
        std::string out;
        DumpImpl(out, indent, 0);
        return out;
    }

  private:
    std::variant<std::nullptr_t, bool, double, std::string, JsonObject, JsonArray> value_;

    static void AppendEscapedString(const std::string& input, std::string& out) {
        out.push_back('"');
        for (char ch : input) {
            switch (ch) {
                case '"':
                case '\\':
                    out.push_back('\\');
                    out.push_back(ch);
                    break;
                case '\b':
                    out += "\\b";
                    break;
                case '\f':
                    out += "\\f";
                    break;
                case '\n':
                    out += "\\n";
                    break;
                case '\r':
                    out += "\\r";
                    break;
                case '\t':
                    out += "\\t";
                    break;
                default:
                    if (static_cast<unsigned char>(ch) < 0x20) {
                        char buffer[7];
                        std::snprintf(buffer, sizeof(buffer), "\\u%04x", ch);
                        out += buffer;
                    } else {
                        out.push_back(ch);
                    }
            }
        }
        out.push_back('"');
    }

    void DumpImpl(std::string& out, int indent, int depth) const {
        auto indent_if_needed = [&](bool new_line) {
            if (indent >= 0 && new_line) {
                out.push_back('\n');
                out.append(depth * indent, ' ');
            }
        };

        switch (type()) {
            case Type::Null:
                out += "null";
                break;
            case Type::Bool:
                out += AsBool() ? "true" : "false";
                break;
            case Type::Number: {
                char buffer[64];
                std::snprintf(buffer, sizeof(buffer), "%.10g", AsNumber());
                out += buffer;
                break;
            }
            case Type::String:
                AppendEscapedString(AsString(), out);
                break;
            case Type::Array: {
                out.push_back('[');
                const auto& arr = AsArray();
                if (!arr.empty()) {
                    for (size_t i = 0; i < arr.size(); ++i) {
                        if (i != 0) {
                            out.push_back(',');
                        }
                        if (indent >= 0) {
                            out.push_back('\n');
                            out.append((depth + 1) * indent, ' ');
                        }
                        arr[i].DumpImpl(out, indent, depth + 1);
                    }
                    if (indent >= 0) {
                        out.push_back('\n');
                        out.append(depth * indent, ' ');
                    }
                }
                out.push_back(']');
                break;
            }
            case Type::Object: {
                out.push_back('{');
                const auto& obj = AsObject();
                if (!obj.empty()) {
                    bool first = true;
                    for (const auto& [key, value] : obj) {
                        if (!first) {
                            out.push_back(',');
                        }
                        first = false;
                        if (indent >= 0) {
                            out.push_back('\n');
                            out.append((depth + 1) * indent, ' ');
                        }
                        AppendEscapedString(key, out);
                        out.push_back(':');
                        if (indent >= 0) {
                            out.push_back(' ');
                        }
                        value.DumpImpl(out, indent, depth + 1);
                    }
                    if (indent >= 0) {
                        out.push_back('\n');
                        out.append(depth * indent, ' ');
                    }
                }
                out.push_back('}');
                break;
            }
        }
    }
};

class JsonParser {
  public:
    explicit JsonParser(std::string_view input) : input_(input) {}

    JsonValue Parse() {
        SkipWhitespace();
        JsonValue result = ParseValue();
        SkipWhitespace();
        if (pos_ != input_.size()) {
            throw std::runtime_error("Unexpected trailing characters in JSON");
        }
        return result;
    }

  private:
    std::string_view input_;
    size_t pos_ = 0;

    void SkipWhitespace() {
        while (pos_ < input_.size() && std::isspace(static_cast<unsigned char>(input_[pos_]))) {
            ++pos_;
        }
    }

    JsonValue ParseValue() {
        if (pos_ >= input_.size()) {
            throw std::runtime_error("Unexpected end of input");
        }
        char ch = input_[pos_];
        if (ch == '"') {
            return JsonValue(ParseString());
        }
        if (ch == '{') {
            return ParseObject();
        }
        if (ch == '[') {
            return ParseArray();
        }
        if (ch == 't') {
            ExpectLiteral("true");
            return JsonValue(true);
        }
        if (ch == 'f') {
            ExpectLiteral("false");
            return JsonValue(false);
        }
        if (ch == 'n') {
            ExpectLiteral("null");
            return JsonValue(nullptr);
        }
        if (ch == '-' || std::isdigit(static_cast<unsigned char>(ch))) {
            return JsonValue(ParseNumber());
        }
        throw std::runtime_error("Invalid character in JSON");
    }

    void ExpectLiteral(const char* literal) {
        size_t len = std::strlen(literal);
        if (pos_ + len > input_.size() || input_.substr(pos_, len) != literal) {
            throw std::runtime_error("Invalid literal in JSON");
        }
        pos_ += len;
    }

    JsonValue ParseObject() {
        if (input_[pos_] != '{') {
            throw std::runtime_error("Expected '{'");
        }
        ++pos_;
        SkipWhitespace();
        JsonObject obj;
        if (pos_ < input_.size() && input_[pos_] == '}') {
            ++pos_;
            return JsonValue(obj);
        }
        while (true) {
            SkipWhitespace();
            if (pos_ >= input_.size() || input_[pos_] != '"') {
                throw std::runtime_error("Expected string key");
            }
            std::string key = ParseString();
            SkipWhitespace();
            if (pos_ >= input_.size() || input_[pos_] != ':') {
                throw std::runtime_error("Expected ':' after key");
            }
            ++pos_;
            SkipWhitespace();
            JsonValue value = ParseValue();
            obj.emplace(std::move(key), std::move(value));
            SkipWhitespace();
            if (pos_ >= input_.size()) {
                throw std::runtime_error("Unexpected end of input in object");
            }
            char ch = input_[pos_];
            if (ch == ',') {
                ++pos_;
                continue;
            }
            if (ch == '}') {
                ++pos_;
                break;
            }
            throw std::runtime_error("Expected ',' or '}' in object");
        }
        return JsonValue(obj);
    }

    JsonValue ParseArray() {
        if (input_[pos_] != '[') {
            throw std::runtime_error("Expected '['");
        }
        ++pos_;
        SkipWhitespace();
        JsonArray arr;
        if (pos_ < input_.size() && input_[pos_] == ']') {
            ++pos_;
            return JsonValue(arr);
        }
        while (true) {
            SkipWhitespace();
            arr.emplace_back(ParseValue());
            SkipWhitespace();
            if (pos_ >= input_.size()) {
                throw std::runtime_error("Unexpected end of input in array");
            }
            char ch = input_[pos_];
            if (ch == ',') {
                ++pos_;
                continue;
            }
            if (ch == ']') {
                ++pos_;
                break;
            }
            throw std::runtime_error("Expected ',' or ']' in array");
        }
        return JsonValue(arr);
    }

    static uint32_t HexDigit(char ch) {
        if (ch >= '0' && ch <= '9') {
            return ch - '0';
        }
        if (ch >= 'a' && ch <= 'f') {
            return 10 + (ch - 'a');
        }
        if (ch >= 'A' && ch <= 'F') {
            return 10 + (ch - 'A');
        }
        throw std::runtime_error("Invalid hex digit in unicode escape");
    }

    static void EncodeUtf8(uint32_t codepoint, std::string& out) {
        if (codepoint <= 0x7F) {
            out.push_back(static_cast<char>(codepoint));
        } else if (codepoint <= 0x7FF) {
            out.push_back(static_cast<char>(0xC0 | ((codepoint >> 6) & 0x1F)));
            out.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
        } else if (codepoint <= 0xFFFF) {
            out.push_back(static_cast<char>(0xE0 | ((codepoint >> 12) & 0x0F)));
            out.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
        } else {
            out.push_back(static_cast<char>(0xF0 | ((codepoint >> 18) & 0x07)));
            out.push_back(static_cast<char>(0x80 | ((codepoint >> 12) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
        }
    }

    std::string ParseString() {
        if (input_[pos_] != '"') {
            throw std::runtime_error("Expected '\"' to start string");
        }
        ++pos_;
        std::string result;
        while (pos_ < input_.size()) {
            char ch = input_[pos_++];
            if (ch == '"') {
                return result;
            }
            if (ch == '\\') {
                if (pos_ >= input_.size()) {
                    throw std::runtime_error("Invalid escape sequence");
                }
                char esc = input_[pos_++];
                switch (esc) {
                    case '"':
                    case '\\':
                    case '/':
                        result.push_back(esc);
                        break;
                    case 'b':
                        result.push_back('\b');
                        break;
                    case 'f':
                        result.push_back('\f');
                        break;
                    case 'n':
                        result.push_back('\n');
                        break;
                    case 'r':
                        result.push_back('\r');
                        break;
                    case 't':
                        result.push_back('\t');
                        break;
                    case 'u': {
                        if (pos_ + 4 > input_.size()) {
                            throw std::runtime_error("Invalid unicode escape");
                        }
                        uint32_t value = (HexDigit(input_[pos_]) << 12) | (HexDigit(input_[pos_ + 1]) << 8) |
                                         (HexDigit(input_[pos_ + 2]) << 4) | HexDigit(input_[pos_ + 3]);
                        pos_ += 4;
                        if (value >= 0xD800 && value <= 0xDBFF) {
                            // High surrogate, expect following low surrogate.
                            if (pos_ + 6 > input_.size() || input_[pos_] != '\\' || input_[pos_ + 1] != 'u') {
                                throw std::runtime_error("Invalid surrogate pair");
                            }
                            pos_ += 2;
                            uint32_t low = (HexDigit(input_[pos_]) << 12) | (HexDigit(input_[pos_ + 1]) << 8) |
                                           (HexDigit(input_[pos_ + 2]) << 4) | HexDigit(input_[pos_ + 3]);
                            pos_ += 4;
                            if (low < 0xDC00 || low > 0xDFFF) {
                                throw std::runtime_error("Invalid low surrogate");
                            }
                            value = 0x10000 + ((value - 0xD800) << 10) + (low - 0xDC00);
                        }
                        EncodeUtf8(value, result);
                        break;
                    }
                    default:
                        throw std::runtime_error("Unknown escape sequence");
                }
            } else {
                result.push_back(ch);
            }
        }
        throw std::runtime_error("Unterminated string literal");
    }

    double ParseNumber() {
        size_t start = pos_;
        if (input_[pos_] == '-') {
            ++pos_;
        }
        if (pos_ >= input_.size()) {
            throw std::runtime_error("Invalid number");
        }
        if (input_[pos_] == '0') {
            ++pos_;
        } else if (std::isdigit(static_cast<unsigned char>(input_[pos_]))) {
            while (pos_ < input_.size() && std::isdigit(static_cast<unsigned char>(input_[pos_]))) {
                ++pos_;
            }
        } else {
            throw std::runtime_error("Invalid number");
        }
        if (pos_ < input_.size() && input_[pos_] == '.') {
            ++pos_;
            if (pos_ >= input_.size() || !std::isdigit(static_cast<unsigned char>(input_[pos_]))) {
                throw std::runtime_error("Invalid fraction in number");
            }
            while (pos_ < input_.size() && std::isdigit(static_cast<unsigned char>(input_[pos_]))) {
                ++pos_;
            }
        }
        if (pos_ < input_.size() && (input_[pos_] == 'e' || input_[pos_] == 'E')) {
            ++pos_;
            if (pos_ < input_.size() && (input_[pos_] == '+' || input_[pos_] == '-')) {
                ++pos_;
            }
            if (pos_ >= input_.size() || !std::isdigit(static_cast<unsigned char>(input_[pos_]))) {
                throw std::runtime_error("Invalid exponent in number");
            }
            while (pos_ < input_.size() && std::isdigit(static_cast<unsigned char>(input_[pos_]))) {
                ++pos_;
            }
        }
        double value = std::strtod(std::string(input_.substr(start, pos_ - start)).c_str(), nullptr);
        return value;
    }
};

inline JsonValue Parse(std::string_view input) { return JsonParser(input).Parse(); }

}  // namespace simplejson

