#!/usr/bin/env python3
"""
Gemma 300m Encoding Service
This script provides a bridge between C++ and the Gemma 300m model for text encoding.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict
import numpy as np

try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("Warning: transformers not installed. Using mock embeddings.", file=sys.stderr)

class GemmaEncoder:
    """Text encoder using Gemma 300m model."""
    
    def __init__(self, model_name: str = "google/gemma-300m", device: str = "auto"):
        """
        Initialize the Gemma encoder.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to run model on ('cpu', 'cuda', or 'auto')
        """
        self.model_name = model_name
        
        if not HAS_TRANSFORMERS:
            print("Using mock encoder. Install transformers for real embeddings:", file=sys.stderr)
            print("  pip install torch transformers", file=sys.stderr)
            self.mock_mode = True
            return
        
        # Determine device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"Loading model {model_name} on {self.device}...", file=sys.stderr)
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
            self.model.eval()
            self.mock_mode = False
            print("Model loaded successfully!", file=sys.stderr)
        except Exception as e:
            print(f"Error loading model: {e}", file=sys.stderr)
            print("Falling back to mock embeddings.", file=sys.stderr)
            self.mock_mode = True
    
    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Encode a list of texts into embeddings.
        
        Args:
            texts: List of text strings to encode
            batch_size: Batch size for encoding
            
        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        if self.mock_mode:
            return self._mock_encode(texts)
        
        embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # Tokenize
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(self.device)
                
                # Get embeddings
                outputs = self.model(**inputs)
                
                # Use mean pooling over sequence
                attention_mask = inputs['attention_mask']
                token_embeddings = outputs.last_hidden_state
                
                # Mean pooling
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                batch_embeddings = (sum_embeddings / sum_mask).cpu().numpy()
                
                embeddings.append(batch_embeddings)
        
        return np.vstack(embeddings)
    
    def _mock_encode(self, texts: List[str]) -> np.ndarray:
        """Generate mock embeddings for testing without a real model."""
        embedding_dim = 768
        embeddings = []
        
        for text in texts:
            # Use hash for deterministic embeddings
            np.random.seed(hash(text) % (2**32))
            embedding = np.random.randn(embedding_dim).astype(np.float32)
            # Normalize
            embedding = embedding / np.linalg.norm(embedding)
            embeddings.append(embedding)
        
        return np.array(embeddings)

def load_tools_from_json(filepath: str) -> List[Dict]:
    """Load tools from BFCL JSON file."""
    tools = []
    
    with open(filepath, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                if "function" in data and isinstance(data["function"], list):
                    for func in data["function"]:
                        tools.append({
                            "id": data.get("id", "unknown"),
                            "name": func.get("name", ""),
                            "description": func.get("description", ""),
                            "parameters": func.get("parameters", {})
                        })
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON: {e}", file=sys.stderr)
    
    return tools

def encode_tools(encoder: GemmaEncoder, tools: List[Dict]) -> List[Dict]:
    """Encode tools with embeddings."""
    texts = [f"{tool['name']} {tool['description']}" for tool in tools]
    embeddings = encoder.encode(texts)
    
    for tool, embedding in zip(tools, embeddings):
        tool['embedding'] = embedding.tolist()
    
    return tools

def save_embeddings(tools: List[Dict], output_path: str):
    """Save tools with embeddings to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(tools, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Encode tools using Gemma 300m model")
    parser.add_argument("--input", type=str, default="../data/BFCL_v3_simple.json",
                        help="Input JSON file path")
    parser.add_argument("--output", type=str, default="tool_embeddings.json",
                        help="Output JSON file path")
    parser.add_argument("--model", type=str, default="google/gemma-300m",
                        help="HuggingFace model name")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"],
                        help="Device to run model on")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for encoding")
    
    args = parser.parse_args()
    
    print("=== Gemma 300m Tool Encoder ===")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print()
    
    # Initialize encoder
    encoder = GemmaEncoder(args.model, args.device)
    
    # Load tools
    print("Loading tools...", file=sys.stderr)
    tools = load_tools_from_json(args.input)
    print(f"Loaded {len(tools)} tools", file=sys.stderr)
    
    # Encode tools
    print("Encoding tools...", file=sys.stderr)
    tools_with_embeddings = encode_tools(encoder, tools)
    print("Encoding complete!", file=sys.stderr)
    
    # Save results
    print(f"Saving to {args.output}...", file=sys.stderr)
    save_embeddings(tools_with_embeddings, args.output)
    print("Done!", file=sys.stderr)

if __name__ == "__main__":
    main()
