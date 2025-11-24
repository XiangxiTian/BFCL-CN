// Tool extraction helpers.

#pragma once

#include <cstddef>
#include <filesystem>
#include <vector>

#include "tool_taxonomy/types.h"

namespace tool_taxonomy {

std::vector<Tool> LoadTools(const std::filesystem::path& data_root,
                            std::size_t max_tools,
                            bool deduplicate);

}  // namespace tool_taxonomy
