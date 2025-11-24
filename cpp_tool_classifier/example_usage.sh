#!/bin/bash
# Example usage of the Tool Classifier

echo "=== Tool Classifier - Example Usage ==="
echo

# Build the project (if not already built)
if [ ! -f "build/tool_classifier" ]; then
    echo "Building the project first..."
    ./build.sh
    echo
fi

# Example 1: Basic usage with default parameters
echo "Example 1: Basic classification and clustering"
echo "----------------------------------------------"
./build/tool_classifier
echo
echo

# Example 2: Custom input file
echo "Example 2: Using a different input file"
echo "----------------------------------------"
./build/tool_classifier --input ../data/BFCL_v3_multiple.json \
                        --output results_multiple.json
echo
echo

# Example 3: More clusters per class
echo "Example 3: Creating 5 clusters per class"
echo "-----------------------------------------"
./build/tool_classifier --input ../data/BFCL_v3_simple.json \
                        --output results_5clusters.json \
                        --clusters 5
echo
echo

echo "=== Examples Complete ==="
echo
echo "Output files created:"
echo "  - classification_results.json (Example 1)"
echo "  - results_multiple.json (Example 2)"
echo "  - results_5clusters.json (Example 3)"
echo
echo "You can view the results using:"
echo "  cat classification_results.json | jq '.summary'"
echo "  cat classification_results.json | jq '.classes[] | {name, tool_count}'"
echo
