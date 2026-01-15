"""
WorldMind State Summarizer Module for Navigation Environment

LLM-based state summarization mechanism that converts visual observations (images)
into textual descriptions of the environment state.
"""

import os
import json
import base64
from typing import Dict, Optional
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


STATE_SUMMARIZER_SYSTEM_PROMPT = """You are a **Visual State Analyzer** for a navigation robot in an **indoor home environment**.
Your goal is to accurately describe what the robot observes to help with navigation planning.

## The Navigation Environment Context
- **Room Types**: Kitchen, living room, bedroom, bathroom, hallway, dining room, office
- **Common Objects**: Furniture (table, chair, sofa, bed, desk, cabinet), Appliances (TV, microwave, fridge), Fixtures (door, window, stairs, toilet, sink)
- **Spatial Elements**: Walls, floor, ceiling, doorways, open passages

## Analysis Guidelines
1. **Robot's Position**: Describe what the robot is facing (room type, main visible area)
2. **Key Objects**: List important objects and their approximate positions (left, center, right, near, far)
3. **Navigation Cues**: Note any obstacles, clear paths, doorways, or open areas
4. **Spatial Relationships**: Describe relative positions of objects to each other

## What to Focus On
- Objects that could be navigation targets (furniture, appliances)
- Potential obstacles in the movement path
- Doorways or passages to other rooms
- Clear areas suitable for navigation

## What NOT to Include
- Excessive detail about textures or colors
- Objects too small to affect navigation
- Speculative descriptions of things not clearly visible

## Output Format
For combined before/after analysis:
{
    "state_before_action": "Description of the scene before the action was executed.",
    "state_after_action": "Description of the scene after the action was executed."
}

For single state analysis:
{
    "current_state": "Description of the current visible scene."
}
"""


STATE_SUMMARIZER_USER_PROMPT_COMBINED = """## Task
Analyze the visual changes related to the following navigation action.

**Action Executed**: {action_description}

## Input Data
- **Image 1 (Before)**: State prior to action execution
- **Image 2 (After)**: State after action execution

## Instructions
1. Describe what's visible in Image 1 (before action)
2. Describe what's visible in Image 2 (after action)
3. Note any changes in visible objects, room area, or perspective
4. Be factual - describe what you see, not what you expect

Output your analysis in JSON format with keys: "state_before_action", "state_after_action"."""


STATE_SUMMARIZER_USER_PROMPT_SINGLE = """## Current Observation Task
Analyze the navigation observation image and provide a concise state description.

Focus on:
1. **Visible Objects**: What furniture, appliances, or fixtures are visible?
2. **Positions**: Where are key objects located (left, center, right, near, far)?
3. **Room Type**: What type of room or area is this?
4. **Navigation Cues**: Any obstacles, clear paths, or doorways visible?

Provide a brief 1-2 sentence description of the current state.
Output in JSON format with key: "current_state"."""


STATE_SUMMARIZER_USER_PROMPT_WITH_CONTEXT = """## Observation After Action
Analyze the current observation image AFTER executing the following action.

**Action Executed**: {action_description}

Focus on:
1. What is now visible in the scene?
2. How does this compare to what was expected?
3. Are there any notable objects, obstacles, or navigation opportunities?

Provide a brief 1-2 sentence description of the current state.
Output in JSON format with key: "current_state"."""


class WorldMindStateSummarizer:
    """LLM-based state summarization module for converting visual observations to text."""
    
    def __init__(self, model_name: str = None, model_type: str = "remote", use_separate_summarization: bool = False):
        """Initialize the state summarizer."""
        self.model_name = model_name or os.environ.get('WORLDMIND_SUMMARIZER_MODEL', 'gpt-4o-mini')
        self.model_type = model_type
        self.use_separate_summarization = use_separate_summarization
        
        self.api_key = os.environ.get('OPENAI_API_KEY')
        self.api_base = os.environ.get('OPENAI_API_BASE', None)
        
        self.client = None
        if self.api_key:
            try:
                from openai import OpenAI
                if self.api_base:
                    self.client = OpenAI(api_key=self.api_key, base_url=self.api_base)
                else:
                    self.client = OpenAI(api_key=self.api_key)
            except Exception as e:
                logger.warning(f"Failed to initialize summarizer client: {e}")
        
        self.system_prompt = STATE_SUMMARIZER_SYSTEM_PROMPT
        
        self.total_summaries = 0
        self.call_count = 0
        
        mode_str = "separate" if use_separate_summarization else "combined"
        logger.info(f"WorldMind Navigation State Summarizer initialized with model: {self.model_name}, mode: {mode_str}")
    
    def summarize_states(
        self,
        action_description: str,
        before_image_path: str,
        after_image_path: str
    ) -> Dict[str, str]:
        """Summarize the states before and after an action."""
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
        if not self.client:
            return {
                "state_before_action": "State summarizer not available",
                "state_after_action": "State summarizer not available"
            }
        
        try:
            before_image_url = local_image_to_data_url(before_image_path)
            after_image_url = local_image_to_data_url(after_image_path)
            
            user_prompt = STATE_SUMMARIZER_USER_PROMPT_COMBINED.format(
                action_description=action_description
            )
            
            messages = [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": before_image_url}},
                        {"type": "image_url", "image_url": {"url": after_image_url}}
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
            result = self._parse_combined_result(result_text)
            
            self.total_summaries += 1
            self.call_count += 1
            
            logger.debug(f"State summarization (combined) completed for action: {action_description}")
            
            return result
            
        except Exception as e:
            logger.error(f"State summarization (combined) error: {e}")
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
        state_before = self.summarize_state(before_image_path, f"Before: {action_description}")
        state_after = self.summarize_state(after_image_path, f"After: {action_description}")
        
        return {
            "state_before_action": state_before,
            "state_after_action": state_after
        }
    
    def summarize_state(self, image_path: str, action_context: str = "") -> str:
        """Summarize a single observation image."""
        if not self.client:
            return "State summarizer not available"
        
        try:
            image_url = local_image_to_data_url(image_path)
            
            if action_context:
                prompt = STATE_SUMMARIZER_USER_PROMPT_WITH_CONTEXT.format(
                    action_description=action_context
                )
            else:
                prompt = STATE_SUMMARIZER_USER_PROMPT_SINGLE
            
            messages = [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}}
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
            self.call_count += 1
            
            return self._parse_single_state(result_text)
            
        except Exception as e:
            logger.error(f"State summarization (single) error: {e}")
            return f"[Summarization failed: {str(e)[:50]}]"
    
    def _parse_combined_result(self, result_text: str) -> Dict[str, str]:
        """Parse the LLM output for combined state summarization."""
        import re
        
        clean_text = result_text.strip()
        if clean_text.startswith("```"):
            lines = clean_text.split("\n")
            clean_text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        
        try:
            json_match = re.search(r'\{[^{}]*\}', clean_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return {
                    "state_before_action": str(result.get("state_before_action", "")),
                    "state_after_action": str(result.get("state_after_action", ""))
                }
        except json.JSONDecodeError:
            pass
        
        return {
            "state_before_action": result_text[:200] if result_text else "",
            "state_after_action": result_text[200:400] if len(result_text) > 200 else ""
        }
    
    def _parse_single_state(self, result_text: str) -> str:
        """Parse the LLM output for single state summarization."""
        import re
        
        clean_text = result_text.strip()
        if clean_text.startswith("```"):
            lines = clean_text.split("\n")
            clean_text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        
        try:
            json_match = re.search(r'\{[^{}]*\}', clean_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return str(result.get("current_state", clean_text[:300]))
        except json.JSONDecodeError:
            pass
        
        return clean_text[:300]
    
    def get_statistics(self) -> Dict:
        """Get summarization statistics."""
        return {
            "total_summaries": self.total_summaries,
            "call_count": self.call_count
        }
    
    def reset_statistics(self):
        """Reset statistics counters."""
        self.total_summaries = 0
        self.call_count = 0


def create_state_summarizer(model_name: str = None, use_separate: bool = False) -> WorldMindStateSummarizer:
    """Factory function to create a WorldMind State Summarizer for Navigation."""
    return WorldMindStateSummarizer(model_name=model_name, use_separate_summarization=use_separate)
