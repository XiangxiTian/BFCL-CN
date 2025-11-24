#!/bin/bash

# Example usage script for Tool Clustering System

echo "=== Tool Clustering Example Usage ==="
echo ""

# Check if executable exists
if [ ! -f "./tool_clustering" ] && [ ! -f "./build/tool_clustering" ]; then
    echo "Building tool_clustering..."
    ./build.sh
fi

EXEC="./tool_clustering"
if [ -f "./build/tool_clustering" ]; then
    EXEC="./build/tool_clustering"
fi

# Example 1: Basic usage with default file
echo "Example 1: Basic usage"
echo "Command: $EXEC"
echo ""
$EXEC data/BFCL_v3_simple.json 3 results.csv 2>&1 | head -50

echo ""
echo "========================================"
echo ""

# Example 2: Custom clustering
echo "Example 2: Custom number of clusters per class"
echo "Command: $EXEC data/BFCL_v3_simple.json 5"
echo ""
$EXEC data/BFCL_v3_simple.json 5 2>&1 | head -50

echo ""
echo "========================================"
echo ""
echo "Check results.csv for detailed output (if generated)"
