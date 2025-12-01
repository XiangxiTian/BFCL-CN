# How to Add Top-K Tool Retrieval to the Leaderboard

This guide explains how to contribute a top-k tool retrieval step that filters tools before passing them to the LLM for tool selection.

## Overview

Top-k tool retrieval is a technique where, before sending all available tools to the LLM, you first retrieve the top-k most relevant tools based on the user query. This can improve performance and reduce token usage when there are many available tools.

## Architecture Decision

Based on the codebase structure, you have two main approaches:

### Approach 1: Create a Wrapper Handler (Recommended)

Create a new handler that wraps an existing handler and adds top-k retrieval. This allows you to:
- Keep the original handler unchanged
- Create model variants like `model-name-topk` or `model-name-retrieval`
- Easily test and compare with/without retrieval

### Approach 2: Add Retrieval to Base Handler

Modify the `_compile_tools` method in `base_handler.py` to support optional retrieval. This would affect all models but requires more careful implementation.

**We recommend Approach 1** as it's cleaner, more maintainable, and follows the existing pattern of model variants (e.g., `-FC` suffix).

## Implementation Steps

### Step 1: Create a Retrieval Utility Module

Create a new file `bfcl/model_handler/utils/retrieval.py` (or add to `bfcl/model_handler/utils.py`) with your retrieval logic:

```python
from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class ToolRetriever:
    """Retrieves top-k most relevant tools based on query similarity."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", top_k: int = 10):
        """
        Args:
            model_name: Name of the sentence transformer model to use
            top_k: Number of top tools to retrieve
        """
        self.encoder = SentenceTransformer(model_name)
        self.top_k = top_k
    
    def retrieve_top_k(
        self, 
        query: str, 
        tools: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top-k most relevant tools based on query.
        
        Args:
            query: User query/question
            tools: List of tool definitions (function docs)
            
        Returns:
            Top-k most relevant tools
        """
        if len(tools) <= self.top_k:
            return tools
        
        # Create embeddings for query and tools
        query_embedding = self.encoder.encode([query])
        
        # Create text representation for each tool
        tool_texts = []
        for tool in tools:
            # Combine name, description, and parameter descriptions
            tool_text = f"{tool.get('name', '')} {tool.get('description', '')}"
            if 'parameters' in tool and 'properties' in tool['parameters']:
                for param_name, param_info in tool['parameters']['properties'].items():
                    tool_text += f" {param_name} {param_info.get('description', '')}"
            tool_texts.append(tool_text)
        
        tool_embeddings = self.encoder.encode(tool_texts)
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_embedding, tool_embeddings)[0]
        
        # Get top-k indices
        top_k_indices = np.argsort(similarities)[::-1][:self.top_k]
        
        # Return top-k tools
        return [tools[i] for i in top_k_indices]
```

**Alternative:** You can use other retrieval methods like:
- BM25 (keyword-based)
- Custom embedding models
- Hybrid retrieval (combining multiple methods)

### Step 2: Create a Wrapper Handler

Create a new handler file, e.g., `bfcl/model_handler/api_inference/openai_topk.py`:

```python
from bfcl.model_handler.api_inference.openai import OpenAIHandler
from bfcl.model_handler.utils.retrieval import ToolRetriever
from typing import Dict, Any

class OpenAITopKHandler(OpenAIHandler):
    """OpenAI handler with top-k tool retrieval."""
    
    def __init__(self, model_name: str, temperature: float, top_k: int = 10):
        super().__init__(model_name, temperature)
        self.retriever = ToolRetriever(top_k=top_k)
        # Extract top_k from model name if specified (e.g., "gpt-4o-topk-5")
        if "topk" in model_name.lower():
            try:
                # Parse top_k from model name if format is "model-topk-N"
                parts = model_name.lower().split("-topk-")
                if len(parts) > 1:
                    self.retriever.top_k = int(parts[1].split("-")[0])
            except:
                pass
    
    def _compile_tools(self, inference_data: dict, test_entry: dict) -> dict:
        """
        Override to add top-k retrieval before compiling tools.
        """
        # Get all functions from test entry
        functions: list = test_entry["function"]
        test_category: str = test_entry["id"].rsplit("_", 1)[0]
        
        # Extract user query for retrieval
        user_query = self._extract_user_query(test_entry)
        
        # Retrieve top-k tools
        if len(functions) > self.retriever.top_k:
            functions = self.retriever.retrieve_top_k(user_query, functions)
        
        # Continue with normal tool compilation
        functions = func_doc_language_specific_pre_processing(functions, test_category)
        tools = convert_to_tool(functions, GORILLA_TO_OPENAPI, self.model_style)
        
        inference_data["tools"] = tools
        inference_data["retrieved_tools_count"] = len(functions)  # For logging
        
        return inference_data
    
    def _extract_user_query(self, test_entry: dict) -> str:
        """Extract user query from test entry."""
        question = test_entry.get("question", [])
        if isinstance(question, list):
            # For single-turn, get the first user message
            if len(question) > 0 and isinstance(question[0], list):
                # Multi-turn format
                for msg_list in question:
                    for msg in msg_list:
                        if msg.get("role") == "user":
                            return msg.get("content", "")
            else:
                # Single-turn format
                for msg in question:
                    if msg.get("role") == "user":
                        return msg.get("content", "")
        return ""
```

### Step 3: Register Your Handler

Update `bfcl/model_handler/handler_map.py`:

```python
from bfcl.model_handler.api_inference.openai_topk import OpenAITopKHandler

# Add to api_inference_handler_map
api_inference_handler_map = {
    # ... existing entries ...
    "gpt-4o-2024-11-20-FC-topk": OpenAITopKHandler,
    "gpt-4o-2024-11-20-FC-topk-5": OpenAITopKHandler,  # With custom top_k=5
    "gpt-4o-2024-11-20-FC-topk-10": OpenAITopKHandler,  # With custom top_k=10
    # Add more variants as needed
}
```

### Step 4: Update Model Metadata

Update `bfcl/constants/model_metadata.py`:

```python
MODEL_METADATA_MAPPING = {
    # ... existing entries ...
    "gpt-4o-2024-11-20-FC-topk": [
        "GPT-4o-2024-11-20 (FC) + Top-K Retrieval",
        "https://openai.com/index/hello-gpt-4o/",
        "OpenAI",
        "Proprietary",
    ],
    "gpt-4o-2024-11-20-FC-topk-5": [
        "GPT-4o-2024-11-20 (FC) + Top-5 Retrieval",
        "https://openai.com/index/hello-gpt-4o/",
        "OpenAI",
        "Proprietary",
    ],
    # ... add more as needed ...
}
```

### Step 5: Update SUPPORTED_MODELS.md

Add your new model variants to the supported models table:

```markdown
| GPT-4o-2024-11-20 + Top-K Retrieval | Function Calling | OpenAI | gpt-4o-2024-11-20-FC-topk |
```

### Step 6: Add Dependencies

If you're using new dependencies (e.g., `sentence-transformers`), update `pyproject.toml`:

```toml
[project.optional-dependencies]
retrieval = [
    "sentence-transformers>=2.2.0",
    "scikit-learn>=1.0.0",
]
```

Then install with:
```bash
pip install -e .[retrieval]
```

## Testing Your Implementation

1. **Test on a small dataset first:**
   ```bash
   bfcl generate --model gpt-4o-2024-11-20-FC-topk --test-category simple
   ```

2. **Compare with baseline:**
   ```bash
   bfcl generate --model gpt-4o-2024-11-20-FC --test-category simple
   bfcl generate --model gpt-4o-2024-11-20-FC-topk --test-category simple
   ```

3. **Evaluate results:**
   ```bash
   bfcl evaluate --model gpt-4o-2024-11-20-FC-topk --test-category simple
   ```

## Considerations for Multi-Turn Conversations

For multi-turn conversations, you may want to:
- Use the current turn's query for retrieval
- Or combine all previous context for retrieval
- Re-retrieve at each turn if new tools are added

Modify `_compile_tools` in your handler to handle multi-turn scenarios appropriately.

## Submitting Your Contribution

1. **Create a Pull Request** with:
   - Your retrieval utility module
   - Your wrapper handler(s)
   - Updates to `handler_map.py`
   - Updates to `model_metadata.py`
   - Updates to `SUPPORTED_MODELS.md`
   - Documentation explaining your retrieval method

2. **Include in your PR description:**
   - What retrieval method you used
   - Why you chose that method
   - Performance improvements (if any)
   - Any limitations or trade-offs

3. **Ensure your code:**
   - Follows the existing code style
   - Has proper error handling
   - Works for both single-turn and multi-turn scenarios
   - Is well-documented

## Leaderboard Integration

Once your PR is merged:
- Your model variants will appear on the leaderboard
- Results will be tracked separately from the base models
- You can compare performance with/without retrieval

## Example: Complete Implementation for Gemini

Here's a complete example for Gemini models:

```python
# bfcl/model_handler/api_inference/gemini_topk.py
from bfcl.model_handler.api_inference.gemini import GeminiHandler
from bfcl.model_handler.utils.retrieval import ToolRetriever
from bfcl.model_handler.utils import (
    func_doc_language_specific_pre_processing,
    convert_to_tool,
)
from bfcl.constants.type_mappings import GORILLA_TO_OPENAPI

class GeminiTopKHandler(GeminiHandler):
    """Gemini handler with top-k tool retrieval."""
    
    def __init__(self, model_name: str, temperature: float, top_k: int = 10):
        super().__init__(model_name, temperature)
        self.retriever = ToolRetriever(top_k=top_k)
    
    def _compile_tools(self, inference_data: dict, test_entry: dict) -> dict:
        functions: list = test_entry["function"]
        test_category: str = test_entry["id"].rsplit("_", 1)[0]
        
        # Extract query and retrieve top-k
        user_query = self._extract_user_query(test_entry)
        if len(functions) > self.retriever.top_k:
            functions = self.retriever.retrieve_top_k(user_query, functions)
        
        # Continue with normal processing
        functions = func_doc_language_specific_pre_processing(functions, test_category)
        tools = convert_to_tool(functions, GORILLA_TO_OPENAPI, self.model_style)
        
        inference_data["tools"] = tools
        return inference_data
    
    def _extract_user_query(self, test_entry: dict) -> str:
        # Same implementation as OpenAI example
        ...
```

## Questions?

- Join the [Gorilla Discord](https://discord.gg/grXXvj9Whz) `#leaderboard` channel
- Open an issue on GitHub
- Review existing handlers for reference implementations
