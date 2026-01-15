"""
WorldMind State Prediction Discriminator for Navigation Environment

LLM-based discriminator that compares the agent's predicted state with actual state.
Supports two modes:
1. Text mode: Compare predicted state text with state summarizer output text
2. Vision mode: Compare predicted state text with actual observation image
"""

import os
import json
import re
import base64
from typing import Dict, Optional
from openai import OpenAI
from embodiedbench.main import logger


DISCRIMINATOR_SYSTEM_PROMPT = """You are a **Logical Consistency Validator** for a navigation robot in an **indoor home environment**.
Your SOLE purpose is to detect **Factual Contradictions** between the Agent's Prediction and the Actual Observation.

## The Navigation Environment Context
- **Valid Actions**: Move forward/backward/left/right by 0.25 meters, Rotate left/right by 90 degrees, Tilt camera up/down by 30 degrees
- **Key Objects**: Furniture (chair, table, sofa, bed, desk), Appliances (TV, microwave, fridge), Fixtures (door, window, stairs)
- **Rooms**: Kitchen, living room, bedroom, bathroom, hallway, dining room

## The "Default True" Principle
**CRITICAL RULE**: You must maintain a "Presumption of Innocence" for the agent.
1. **Irrelevance is True**: If the prediction and actual state discuss different objects or aspects, and do not logically contradict, return `"match": true`.
2. **Key Consistency**: Compare ONLY the aspects that overlap between prediction and actual state.
3. **Conflict Driven**: **ONLY** return `false` if there is a direct, undeniable factual conflict.

## What Constitutes a Conflict in Navigation
- **Object Presence**: Prediction says "I will see a TV" but actual shows no TV visible
- **Spatial Relationship**: Prediction says "closer to the chair" but actual shows moved away
- **Direction**: Prediction says "facing the window" but actual shows facing a wall
- **Movement Effect**: Prediction says "new room visible" but still in same room

## What Does NOT Constitute a Conflict
- Different levels of detail in description
- Minor wording differences
- Prediction mentions objects not in view (could be out of frame)
- Prediction discusses intent, not state assertion

## Output Format
Output ONLY a JSON object:
{
    "match": true, // or false
    "reason": "Explain ONLY if there's a factual conflict. Otherwise state 'No factual contradictions found'."
}
"""

DISCRIMINATOR_USER_PROMPT_TEXT = """## Comparison Task

**Action Executed**: {action}

**Agent's Predicted State After Action**:
{predicted_state}

**Actual State After Action** (from state summarizer):
{actual_state}

Please analyze whether the agent's predicted state matches the actual state description.
Focus on factual contradictions, not stylistic differences.
Output your judgment in JSON format with keys: "match" (boolean), "reason" (string)."""


DISCRIMINATOR_USER_PROMPT_VISION = """## Visual Comparison Task

**Action Executed**: {action}

**Agent's Predicted State After Action**:
{predicted_state}

Look at the actual observation image and determine if the agent's prediction matches what is visible.
Focus on:
1. Are key objects mentioned in the prediction actually visible?
2. Are spatial relationships correctly predicted?
3. Did the action have the expected effect on the view?

Minor differences in description detail are acceptable.
Output your judgment in JSON format with keys: "match" (boolean), "reason" (string)."""


class WorldMindDiscriminator:
    """LLM-based discriminator for comparing predicted states with actual states."""
    
    def __init__(self, model_name: str = None, model_type: str = "remote", use_vision: bool = True):
        """Initialize the discriminator."""
        self.model_name = model_name or os.environ.get('WORLDMIND_DISCRIMINATOR_MODEL', 'gpt-4o-mini')
        self.model_type = model_type
        self.use_vision = use_vision
        
        self.api_key = os.environ.get('OPENAI_API_KEY')
        self.api_base = os.environ.get('OPENAI_API_BASE', None)
        
        self.client = None
        if self.api_key:
            try:
                if self.api_base:
                    self.client = OpenAI(api_key=self.api_key, base_url=self.api_base)
                else:
                    self.client = OpenAI(api_key=self.api_key)
            except Exception as e:
                logger.warning(f"Failed to initialize discriminator client: {e}")
        
        self.system_prompt = DISCRIMINATOR_SYSTEM_PROMPT
        
        self.total_judgments = 0
        self.match_count = 0
        self.mismatch_count = 0
        
        mode_str = "vision" if use_vision else "text-only"
        logger.info(f"WorldMind Navigation Discriminator initialized with model: {self.model_name}, mode: {mode_str}")
    
    def discriminate(
        self, 
        predicted_state: str, 
        actual_state_summary: str = "",
        actual_state_image: Optional[str] = None,
        action_description: str = ""
    ) -> Dict:
        """Compare the predicted state with the actual state."""
        if not self.client:
            return {"match": True, "reason": "Discriminator not available (no client)"}
        
        if not predicted_state or predicted_state.strip() == "":
            return {"match": True, "reason": "No prediction available to compare"}
        
        if "exploration phase" in predicted_state.lower() or "prediction skipped" in predicted_state.lower():
            return {"match": True, "reason": "Exploration phase - prediction skipped"}
        
        try:
            if self.use_vision and actual_state_image and os.path.exists(actual_state_image):
                return self._discriminate_with_vision(predicted_state, actual_state_image, action_description)
            else:
                return self._discriminate_text(predicted_state, actual_state_summary, action_description)
        except Exception as e:
            logger.error(f"WorldMind Discriminator error: {e}")
            return {"match": True, "reason": f"Discriminator error: {str(e)}"}
    
    def _discriminate_text(self, predicted_state: str, actual_state_summary: str, action_description: str) -> Dict:
        """Text mode discrimination: compare two text descriptions."""
        user_prompt = DISCRIMINATOR_USER_PROMPT_TEXT.format(
            action=action_description,
            predicted_state=predicted_state,
            actual_state=actual_state_summary or "No state summary available"
        )
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0,
            max_tokens=256
        )
        
        result_text = response.choices[0].message.content
        result = self._parse_result(result_text)
        self._update_statistics(result["match"])
        
        logger.debug(f"Discriminator (text mode) result: match={result['match']}, reason={result['reason'][:100]}...")
        
        return result
    
    def _discriminate_with_vision(self, predicted_state: str, image_path: str, action_description: str) -> Dict:
        """Vision mode discrimination: compare prediction with actual image."""
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
        
        user_prompt = DISCRIMINATOR_USER_PROMPT_VISION.format(
            action=action_description,
            predicted_state=predicted_state
        )
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}}
                ]
            }
        ]
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0,
            max_tokens=256
        )
        
        result_text = response.choices[0].message.content
        result = self._parse_result(result_text)
        self._update_statistics(result["match"])
        
        logger.debug(f"Discriminator (vision mode) result: match={result['match']}, reason={result['reason'][:100]}...")
        
        return result
    
    def _update_statistics(self, is_match: bool):
        """Update internal statistics."""
        self.total_judgments += 1
        if is_match:
            self.match_count += 1
        else:
            self.mismatch_count += 1
    
    def _parse_result(self, result_text: str) -> Dict:
        """Parse the LLM output to extract the judgment result."""
        clean_text = result_text.strip()
        if clean_text.startswith("```"):
            lines = clean_text.split("\n")
            clean_text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        
        try:
            json_match = re.search(r'\{[^{}]*\}', clean_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return {
                    "match": bool(result.get("match", True)),
                    "reason": str(result.get("reason", "No reason provided"))
                }
        except json.JSONDecodeError:
            pass
        
        match = "true" in result_text.lower() and "match" in result_text.lower()
        return {
            "match": match,
            "reason": result_text[:500]
        }
    
    def get_statistics(self) -> Dict:
        """Get discrimination statistics."""
        return {
            "total_judgments": self.total_judgments,
            "match_count": self.match_count,
            "mismatch_count": self.mismatch_count,
            "match_rate": self.match_count / self.total_judgments if self.total_judgments > 0 else 0
        }
    
    def reset_statistics(self):
        """Reset statistics counters."""
        self.total_judgments = 0
        self.match_count = 0
        self.mismatch_count = 0


def create_discriminator(model_name: str = None, use_vision: bool = True) -> WorldMindDiscriminator:
    """Factory function to create a WorldMind Discriminator for Navigation."""
    return WorldMindDiscriminator(model_name=model_name, use_vision=use_vision)
