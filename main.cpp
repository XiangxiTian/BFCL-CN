#include <iostream>
#include "ToolClusterer.h"

int main() {
    std::string data_path = "data/BFCL_v3_simple.json";
    
    // Execute the pipeline
    run_tool_clustering_pipeline(data_path);
    
    std::cout << "[DONE] Script completed successfully." << std::endl;
    return 0;
}
