"""
WorldMind State Summarizer Module for Alfred Environment

This module provides an LLM-based state summarization mechanism that converts
visual observations (images) into textual descriptions of the environment state.
Used for building experience trajectories and enhancing reflection.
"""

import os
import json
import base64
from typing import Dict, Optional, Tuple, Union, Any
from mimetypes import guess_type

from embodiedbench.main import logger


def local_image_to_data_url(image_path: str) -> str:
    """Convert a local image to a data URL for API consumption."""
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')
    return f"data:{mime_type};base64,{base64_encoded_data}"


# State summarization system prompt for Alfred environment
STATE_SUMMARIZER_SYSTEM_PROMPT = """You are a **Fact-Based Visual Validator** for a robot in the **Alfred simulated household environment**.
Your goal is NOT to summarize the whole scene, but to **verify the visual outcome** of a specific Action by comparing the state of relevant objects before and after the attempt.

## The Alfred Environment Context
- **Valid Receptacles** (Containers/Furniture): Fridge, Cabinet, Drawer, Microwave, CounterTop, DiningTable, SideTable, Sink, StoveBurner, GarbageCan, Shelf, Dresser, Desk, Bed, Toilet, Bathtub
- **Valid Objects**: Apple, Bread, Tomato, Lettuce, Potato, Egg, Knife, Fork, Spoon, Spatula, Cup, Mug, Bowl, Plate, Pot, Pan, Book, Pen, RemoteControl, CellPhone, etc.
- **Valid Actions**: Find, Pick up, Put down, Drop, Open, Close, Turn on, Turn off, Slice
- **Constraint**: Do NOT hallucinate objects not present in the Alfred environment. Use standard Alfred terminology.

## Your Strict Reasoning Process (Internal Monologue)
1. **Identify Targets**: Extract the specific object(s) and receptacle(s) mentioned in the input `Action`.
2. **Analyze 'Before' Image**: Locate these specific targets. Describe their position and state (e.g., "Apple is on CounterTop").
3. **Analyze 'After' Image**: Locate the SAME targets again. **Crucially check**: Did the state actually change?
    - If Action is "Pick up Apple": Is the apple really in the hand? Or is it still on the counter?
    - If Action is "Open Fridge": Is the door actually open? Or does it look closed?
4. **Report Facts**: Describe exactly what you see. **Do NOT assume the action succeeded.** If the action was "Pick up Apple" but the apple is still on the counter, you MUST report "The apple remains on the counter."

## Output Format
Output ONLY a JSON object:
{
    "state_before_action": "Description of the target object's position/state relative to receptacles/hand in the first image.",
    "state_after_action": "Description of the target object's position/state relative to receptacles/hand in the second image."
}

## Critical Rules (Anti-Hallucination)
- **TRUST YOUR EYES, NOT THE ACTION NAME.**
- If the action is "Pick up apple", but the image shows the hand is empty, state: "The hand is empty."
- If the action is "Find CounterTop", but the image shows a different area, state: "Facing [what you see]."
- Focus ONLY on the objects and receptacles involved in the action. Ignore irrelevant background details.

## Examples

### Example 1 (Successful Action)
**Action**: "Pick up the apple"
**Input Images**: [Img1: Apple on counter], [Img2: Apple in hand]
**Output**:
{
    "state_before_action": "The robot is facing the counter. An apple is visible resting on the counter surface. The hand is empty.",
    "state_after_action": "The robot's hand is now holding the apple. The counter surface where the apple was is now empty."
}

### Example 2 (Failed Action - VERY IMPORTANT)
**Action**: "Pick up the apple"
**Input Images**: [Img1: Apple on counter], [Img2: Apple STILL on counter, hand empty]
**Output**:
{
    "state_before_action": "The robot is facing the counter. An apple is visible on the counter. The hand is empty.",
    "state_after_action": "The hand is EMPTY. The apple REMAINS on the counter surface. The action failed to change the object's state."
}

### Example 3 (Navigation)
**Action**: "Find the Fridge"
**Input Images**: [Img1: Facing DiningTable], [Img2: Facing Fridge]
**Output**:
{
    "state_before_action": "The robot is facing a dining table.",
    "state_after_action": "The robot is now standing directly in front of the refrigerator. The fridge door is closed."
}
"""


# User prompt template
STATE_SUMMARIZER_USER_PROMPT = """## Task
Analyze the visual changes related to the following action.

**Action Executed**: {action_description}

## Input Data
- **Image 1 (Before)**: State prior to execution.
- **Image 2 (After)**: State post execution.

## Instructions
1. Focus specifically on the objects/receptacles mentioned in: "{action_description}".
2. Describe the state of these specific items in Image 1.
3. Describe the state of these specific items in Image 2.
4. **Be Honest**: If the object did not move, say it did not move. Do not invent success.

Output your analysis in JSON format with keys: "state_before_action", "state_after_action"."""


# Single state summarization prompt (current state only)
SINGLE_STATE_SUMMARIZER_USER_PROMPT = """## Current Observation Task
You are observing an **Alfred simulation environment**. Analyze the provided image.

## Focus Areas
1. **Robot's Position**: What furniture or receptacle is the robot directly facing? (e.g., "Facing CounterTop", "Facing the Fridge").
2. **Hand State**: Is the robot holding anything? If yes, identify the object clearly.
3. **Visible Objects**: List key interactable objects (e.g., Apple, Bowl, Knife) and their specific locations (e.g., "on the counter", "in the sink").
4. **Receptacle States**: Are doors (Fridge/Cabinet/Microwave) open or closed? Are appliances (Microwave/Stove) on or off?

## Constraint
- Only describe what is clearly visible.
- Use standard Alfred object names.

Output your analysis in JSON format with key: "current_state"."""


class WorldMindStateSummarizer:
    """
    LLM-based state summarization module for converting visual observations to text.
    Used for building experience trajectories and enhancing reflection.
    """
    
    def __init__(self, model_name: str = None, use_separate_summarization: bool = False):
        """
        Initialize the state summarizer.
        
        Args:
            model_name: The name of the VLM model to use. If None, reads from environment.
            use_separate_summarization: Whether to summarize images separately (one LLM call per image)
        """
        self.model_name = model_name or os.environ.get('WorldMind_SUMMARIZER_MODEL')
        self.api_key = os.environ.get('OPENAI_API_KEY')
        self.api_base = os.environ.get('OPENAI_API_BASE', None)
        self.use_separate_summarization = use_separate_summarization
        
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required for WorldMind State Summarizer")
        
        # Initialize OpenAI client
        try:
            from openai import OpenAI
            if self.api_base:
                self.client = OpenAI(api_key=self.api_key, base_url=self.api_base)
            else:
                self.client = OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("openai package is required for WorldMind State Summarizer")
        
        self.system_prompt = STATE_SUMMARIZER_SYSTEM_PROMPT
        
        # Statistics tracking
        self.total_summaries = 0
        
        summarization_mode = "separate" if use_separate_summarization else "combined"
        logger.info(f"WorldMind State Summarizer initialized with model: {self.model_name}, mode: {summarization_mode}")
    
    def summarize_states(
        self,
        action_description: str,
        before_image_path: str,
        after_image_path: str
    ) -> Dict[str, str]:
        """
        Summarize the states before and after an action.
        
        Args:
            action_description: Description of the action executed
            before_image_path: Path to the observation image before the action
            after_image_path: Path to the observation image after the action
            
        Returns:
            dict: {"state_before_action": str, "state_after_action": str}
        """
        if self.use_separate_summarization:
            return self._summarize_states_separately(action_description, before_image_path, after_image_path)
        else:
            return self._summarize_states_combined(action_description, before_image_path, after_image_path)
    
    def _summarize_states_combined(
        self,
        action_description: str,
        before_image_path: str,
        after_image_path: str
    ) -> Dict[str, str]:
        """Summarize states using a single LLM call with both images."""
        try:
            before_image_url = local_image_to_data_url(before_image_path)
            after_image_url = local_image_to_data_url(after_image_path)
            
            user_prompt = STATE_SUMMARIZER_USER_PROMPT.format(
                action_description=action_description
            )
            
            messages = [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": before_image_url}
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": after_image_url}
                        }
                    ]
                }
            ]
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0,
                max_tokens=512
            )
            
            result_text = response.choices[0].message.content
            result = self._parse_result(result_text)
            
            self.total_summaries += 1
            
            logger.debug(f"State summarization (combined) completed for action: {action_description}")
            
            return result
            
        except Exception as e:
            logger.error(f"WorldMind State Summarizer (combined) error: {e}")
            return {
                "state_before_action": f"[Summarization failed: {str(e)[:50]}]",
                "state_after_action": f"[Summarization failed: {str(e)[:50]}]"
            }
    
    def _summarize_states_separately(
        self,
        action_description: str,
        before_image_path: str,
        after_image_path: str
    ) -> Dict[str, str]:
        """Summarize states using separate LLM calls for each image."""
        try:
            state_before = self.summarize_single_state(before_image_path)
            
            state_after_prompt = f"""## Task
Analyze the current observation image AFTER executing the following action.

**Action Executed**: {action_description}

## Instructions
1. Describe what you observe in the current image.
2. Focus on the objects and receptacles mentioned in the action.
3. Describe the state of the robot's hand (holding something or empty).
4. Be honest about what you see - do not assume the action succeeded just because it was attempted.

Output your analysis in JSON format with key: "current_state"."""

            after_image_url = local_image_to_data_url(after_image_path)
            
            messages = [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": state_after_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": after_image_url}
                        }
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
            state_after = self._parse_single_state(result_text)
            
            self.total_summaries += 1
            
            return {
                "state_before_action": state_before,
                "state_after_action": state_after
            }
            
        except Exception as e:
            logger.error(f"WorldMind State Summarizer (separate) error: {e}")
            return {
                "state_before_action": f"[Summarization failed: {str(e)[:50]}]",
                "state_after_action": f"[Summarization failed: {str(e)[:50]}]"
            }
    
    def summarize_single_state(self, image_path: str) -> str:
        """
        Summarize a single observation image.
        
        Args:
            image_path: Path to the observation image
            
        Returns:
            str: Text description of the current state
        """
        try:
            image_url = local_image_to_data_url(image_path)
            
            messages = [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": SINGLE_STATE_SUMMARIZER_USER_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url}
                        }
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
            return self._parse_single_state(result_text)
            
        except Exception as e:
            logger.error(f"WorldMind State Summarizer (single) error: {e}")
            return f"[Summarization failed: {str(e)[:50]}]"
    
    def _parse_result(self, result_text: str) -> Dict[str, str]:
        """Parse the LLM output for combined state summarization."""
        import re
        try:
            json_match = re.search(r'\{[^{}]*\}', result_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return {
                    "state_before_action": str(result.get("state_before_action", "")),
                    "state_after_action": str(result.get("state_after_action", ""))
                }
        except json.JSONDecodeError:
            pass
        
        # Fallback: return raw text
        return {
            "state_before_action": result_text[:200],
            "state_after_action": result_text[200:400] if len(result_text) > 200 else ""
        }
    
    def _parse_single_state(self, result_text: str) -> str:
        """Parse the LLM output for single state summarization."""
        import re
        try:
            json_match = re.search(r'\{[^{}]*\}', result_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return str(result.get("current_state", result_text[:200]))
        except json.JSONDecodeError:
            pass
        
        return result_text[:300]
    
    def get_statistics(self) -> Dict:
        """Get summarization statistics."""
        return {
            "total_summaries": self.total_summaries
        }
    
    def reset_statistics(self):
        """Reset statistics counters."""
        self.total_summaries = 0


# Factory function
def create_state_summarizer(model_name: str = None, use_separate: bool = False) -> WorldMindStateSummarizer:
    """
    Factory function to create a WorldMind State Summarizer.
    
    Args:
        model_name: Model name for summarization
        use_separate: Whether to use separate summarization for each image
        
    Returns:
        WorldMindStateSummarizer instance
    """
    return WorldMindStateSummarizer(model_name=model_name, use_separate_summarization=use_separate)
