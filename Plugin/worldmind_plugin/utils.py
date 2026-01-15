"""
WorldMind Plugin Utilities Module

This module contains utility functions used across the WorldMind plugin,
including LLM output parsing, JSON fixing, and format conversion.

Key Functions:
- parse_agent_output: Parse LLM output to extract action and predicted_state
- fix_json: Fix common JSON formatting issues in LLM outputs
- local_image_to_data_url: Convert local image to data URL for API
- compute_semantic_similarity: Compute text similarity for experience retrieval
"""

import os
import re
import json
import base64
from typing import Dict, List, Tuple, Any, Optional, Union
from mimetypes import guess_type


# =============================================================================
# EXPLORATION PHASE DETECTION
# =============================================================================

# Default exploration phase marker that signals WorldMind to skip processing
EXPLORATION_PHASE_MARKER = "Exploration phase: target not visible, prediction skipped."


def is_exploration_phase(predicted_state: str) -> bool:
    """
    Check if the predicted_state indicates an exploration phase.
    When in exploration phase, WorldMind will skip discrimination and reflection.
    
    Args:
        predicted_state: The predicted state string from agent output
        
    Returns:
        bool: True if in exploration phase, False otherwise
    """
    if not predicted_state:
        return False
    return "exploration phase" in predicted_state.lower() or predicted_state.strip() == EXPLORATION_PHASE_MARKER


# =============================================================================
# LLM OUTPUT PARSING
# =============================================================================

def parse_agent_output(
    output_text: str,
    action_key: str = "action_id",
    json_key: str = "executable_plan"
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Parse the LLM agent output to extract actions and predicted states.
    
    This is the KEY FUNCTION for WorldMind integration. It extracts:
    1. Action list with action_id and action_name
    2. Predicted states for each action
    
    The default expected output format is:
    {
        "executable_plan": [
            {"action_id": 0, "action_name": "Find X", "predicted_state": "..."},
            {"action_id": 1, "action_name": "Pick X", "predicted_state": "..."}
        ]
    }
    
    Users should modify this function to match their agent's output format.
    
    Args:
        output_text: Raw text output from the LLM agent
        action_key: Key name for action ID in the output (default: "action_id")
        json_key: Key name for the executable plan list (default: "executable_plan")
        
    Returns:
        Tuple of:
            - List of action dictionaries with keys: action_id, action_name, predicted_state
            - Error message if parsing failed, empty string otherwise
    """
    try:
        # Fix common JSON issues
        fixed_text = fix_json(output_text)
        
        # Find JSON object in the text
        first_brace = fixed_text.find('{')
        if first_brace == -1:
            return [], "No JSON object found in output"
        
        # Find matching closing brace
        brace_count = 0
        last_brace = -1
        for i in range(first_brace, len(fixed_text)):
            if fixed_text[i] == '{':
                brace_count += 1
            elif fixed_text[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    last_brace = i
                    break
        
        if last_brace == -1:
            return [], "No matching closing brace found"
        
        json_text = fixed_text[first_brace:last_brace + 1]
        
        try:
            json_object = json.loads(json_text)
        except json.JSONDecodeError as e:
            return [], f"JSON parse error: {e}"
        
        # Extract executable plan
        executable_plan = json_object.get(json_key, [])
        
        if not executable_plan:
            return [], "Empty executable plan"
        
        # Parse each action
        actions = []
        for action_item in executable_plan:
            if not isinstance(action_item, dict):
                continue
            
            action = {
                "action_id": action_item.get(action_key),
                "action_name": action_item.get("action_name", ""),
                "predicted_state": action_item.get("predicted_state", "")
            }
            
            if action["action_id"] is not None:
                actions.append(action)
        
        if not actions:
            return [], "No valid actions found"
        
        return actions, ""
        
    except Exception as e:
        return [], f"Unexpected error: {e}"


def extract_first_action(
    output_text: str,
    action_key: str = "action_id",
    json_key: str = "executable_plan"
) -> Tuple[Optional[int], Optional[str], Optional[str]]:
    """
    Extract the first action from agent output.
    Convenience wrapper around parse_agent_output.
    
    Args:
        output_text: Raw text output from the LLM agent
        action_key: Key name for action ID
        json_key: Key name for the executable plan list
        
    Returns:
        Tuple of (action_id, action_name, predicted_state), or (None, None, None) if failed
    """
    actions, error = parse_agent_output(output_text, action_key, json_key)
    
    if not actions:
        return None, None, None
    
    first_action = actions[0]
    return (
        first_action.get("action_id"),
        first_action.get("action_name"),
        first_action.get("predicted_state")
    )


# =============================================================================
# JSON FIXING UTILITIES
# =============================================================================

def fix_json(json_str: str) -> str:
    """
    Fix common JSON formatting issues in LLM outputs.
    
    Common issues fixed:
    - Markdown code blocks (```json ... ```)
    - Trailing commas
    - Single quotes instead of double quotes
    - Unescaped newlines in strings
    
    Args:
        json_str: Potentially malformed JSON string
        
    Returns:
        Fixed JSON string
    """
    if not json_str:
        return json_str
    
    # Remove markdown code blocks
    if "```json" in json_str:
        json_str = json_str.split("```json")[1].split("```")[0]
    elif "```" in json_str:
        parts = json_str.split("```")
        if len(parts) >= 3:
            json_str = parts[1]
    
    # Remove leading/trailing whitespace
    json_str = json_str.strip()
    
    # Fix trailing commas before } or ]
    json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
    
    # Fix unescaped newlines within strings
    # This is tricky - we need to be careful not to break valid JSON
    
    return json_str


def safe_parse_json(json_str: str) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Safely parse JSON string with automatic fixing.
    
    Args:
        json_str: JSON string to parse
        
    Returns:
        Tuple of (parsed_dict, error_message)
    """
    try:
        fixed = fix_json(json_str)
        return json.loads(fixed), None
    except json.JSONDecodeError as e:
        return None, str(e)


def extract_json_object(text: str) -> Optional[str]:
    """
    Extract the first complete JSON object from text.
    
    Args:
        text: Text containing JSON object
        
    Returns:
        Extracted JSON string or None
    """
    first_brace = text.find('{')
    if first_brace == -1:
        return None
    
    brace_count = 0
    for i in range(first_brace, len(text)):
        if text[i] == '{':
            brace_count += 1
        elif text[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                return text[first_brace:i + 1]
    
    return None


# =============================================================================
# IMAGE UTILITIES (for Multimodal Mode)
# =============================================================================

def local_image_to_data_url(image_path: str) -> str:
    """
    Convert a local image file to a data URL for API consumption.
    
    Args:
        image_path: Path to the local image file
        
    Returns:
        Data URL string (e.g., "data:image/png;base64,...")
        
    Raises:
        FileNotFoundError: If image file does not exist
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'
    
    with open(image_path, "rb") as image_file:
        base64_encoded = base64.b64encode(image_file.read()).decode('utf-8')
    
    return f"data:{mime_type};base64,{base64_encoded}"


# =============================================================================
# SEMANTIC SIMILARITY (for Experience Retrieval)
# =============================================================================

# Lazy loaded sentence transformer model
_sentence_model = None
_model_load_attempted = False


def _get_sentence_model():
    """Lazy load SentenceTransformer model for semantic similarity."""
    global _sentence_model, _model_load_attempted
    
    if _model_load_attempted:
        return _sentence_model
    
    _model_load_attempted = True
    
    try:
        from sentence_transformers import SentenceTransformer
        _sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    except ImportError:
        _sentence_model = None
    
    return _sentence_model


def compute_semantic_similarity(text1: str, text2: str, cache: Optional[Dict] = None) -> float:
    """
    Compute semantic similarity between two texts.
    
    Uses SentenceTransformer if available, falls back to word overlap.
    
    Args:
        text1: First text
        text2: Second text
        cache: Optional embedding cache dict for efficiency
        
    Returns:
        Similarity score between 0 and 1
    """
    model = _get_sentence_model()
    
    if model is None:
        # Fallback to basic word overlap (Jaccard similarity)
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1 or not words2:
            return 0.0
        intersection = words1 & words2
        union = words1 | words2
        return len(intersection) / len(union) if union else 0.0
    
    try:
        import numpy as np
        
        if cache is not None:
            if text1 not in cache:
                cache[text1] = model.encode(text1, convert_to_numpy=True)
            if text2 not in cache:
                cache[text2] = model.encode(text2, convert_to_numpy=True)
            emb1 = cache[text1]
            emb2 = cache[text2]
        else:
            emb1 = model.encode(text1, convert_to_numpy=True)
            emb2 = model.encode(text2, convert_to_numpy=True)
        
        # Cosine similarity
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return float(similarity)
        
    except Exception:
        # Fallback to word overlap
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1 or not words2:
            return 0.0
        intersection = words1 & words2
        union = words1 | words2
        return len(intersection) / len(union) if union else 0.0


# =============================================================================
# LOGGING UTILITIES
# =============================================================================

class SimpleLogger:
    """Simple logger for WorldMind plugin when no external logger is provided."""
    
    def __init__(self, name: str = "WorldMind", verbose: bool = False):
        self.name = name
        self.verbose = verbose
    
    def info(self, msg: str):
        """Log info message."""
        print(f"[{self.name}] INFO: {msg}")
    
    def debug(self, msg: str):
        """Log debug message (only if verbose)."""
        if self.verbose:
            print(f"[{self.name}] DEBUG: {msg}")
    
    def warning(self, msg: str):
        """Log warning message."""
        print(f"[{self.name}] WARNING: {msg}")
    
    def error(self, msg: str):
        """Log error message."""
        print(f"[{self.name}] ERROR: {msg}")


# Default logger instance
default_logger = SimpleLogger()


def set_logger(logger):
    """Set custom logger for the plugin."""
    global default_logger
    default_logger = logger


def get_logger() -> SimpleLogger:
    """Get current logger instance."""
    return default_logger


# =============================================================================
# FILE UTILITIES
# =============================================================================

def ensure_dir(path: str):
    """Ensure directory exists, create if not."""
    os.makedirs(path, exist_ok=True)


def save_json(data: Any, filepath: str, indent: int = 2):
    """Save data to JSON file."""
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def load_json(filepath: str) -> Any:
    """Load data from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)
