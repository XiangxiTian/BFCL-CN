from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from bfcl.model_handler.api_inference.openai import OpenAIHandler
from bfcl.model_handler.utils import (
    func_doc_language_specific_pre_processing,
    convert_to_tool,
)
from bfcl.constants.type_mappings import GORILLA_TO_OPENAPI
from overrides import override


class OpenAITopKHandler(OpenAIHandler):
    def __init__(self, model_name, temperature) -> None:
        super().__init__(model_name, temperature)
        
        # Extract top_k from model name if specified (e.g., "gpt-4o-topk-5" or "gpt-4o-topk")
        top_k = 10  # Default value
        if "topk" in model_name.lower():
            try:
                # Parse top_k from model name if format is "model-topk-N"
                parts = model_name.lower().split("-topk-")
                if len(parts) > 1:
                    # Extract number after "-topk-"
                    top_k = int(parts[1].split("-")[0])
            except:
                pass
        
        self.top_k = top_k
        # Initialize the encoder lazily (only when needed)
        self._encoder = None
    
    def _get_encoder(self):
        """Lazy initialization of the encoder to avoid loading if not needed."""
        if self._encoder is None:
            self._encoder = SentenceTransformer("all-MiniLM-L6-v2")
        return self._encoder
    
    def _retrieve_top_k_tools(
        self, 
        query: str, 
        tools: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top-k most relevant tools based on query similarity.
        
        Args:
            query: User query/question
            tools: List of tool definitions (function docs)
            
        Returns:
            Top-k most relevant tools
        """
        if len(tools) <= self.top_k:
            return tools
        
        encoder = self._get_encoder()
        
        # Create embeddings for query and tools
        query_embedding = encoder.encode([query])
        
        # Create text representation for each tool
        tool_texts = []
        for tool in tools:
            # Combine name, description, and parameter descriptions
            tool_text = f"{tool.get('name', '')} {tool.get('description', '')}"
            if 'parameters' in tool and 'properties' in tool['parameters']:
                for param_name, param_info in tool['parameters']['properties'].items():
                    tool_text += f" {param_name} {param_info.get('description', '')}"
            tool_texts.append(tool_text)
        
        tool_embeddings = encoder.encode(tool_texts)
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_embedding, tool_embeddings)[0]
        
        # Get top-k indices
        top_k_indices = np.argsort(similarities)[::-1][:self.top_k]
        
        # Return top-k tools
        return [tools[i] for i in top_k_indices]
    
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

    @override
    def _compile_tools(self, inference_data: dict, test_entry: dict) -> dict:
        """
        Override to add top-k retrieval before compiling tools.
        """
        # Get all functions from test entry
        functions: list = test_entry["function"]
        test_category: str = test_entry["id"].rsplit("_", 1)[0]
        
        # Extract user query for retrieval
        user_query = self._extract_user_query(test_entry)
        
        # Retrieve top-k tools if we have more than top_k functions
        if len(functions) > self.top_k:
            functions = self._retrieve_top_k_tools(user_query, functions)
        
        # Continue with normal tool compilation
        functions = func_doc_language_specific_pre_processing(functions, test_category)
        tools = convert_to_tool(functions, GORILLA_TO_OPENAPI, self.model_style)
        
        inference_data["tools"] = tools
        inference_data["retrieved_tools_count"] = len(functions)  # For logging
        
        return inference_data
