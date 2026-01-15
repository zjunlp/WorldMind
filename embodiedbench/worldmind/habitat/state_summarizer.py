"""
WorldMind State Summarizer Module for Habitat Environment

LLM-based state summarization mechanism that converts visual observations
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


STATE_SUMMARIZER_SYSTEM_PROMPT = """You are a **Fact-Based Visual Validator** for a robot in the **Habitat simulated environment**. 
Your goal is NOT to summarize the whole scene, but to **verify the visual outcome** of a specific Action by comparing the state of relevant objects before and after the attempt.

## The Habitat Environment Context
- **Valid Receptacles** (Containers/Furniture): [Cabinet, Fridge, Counter, Table, Sofa, Sink, TV Stand, Chair, Drawer...]
- **Valid Objects**: [Apple, Bowl, Sponge, Spoon, Can, Cleanser, Lemon, Pear, Knife, Scissors, Toy Airplane, Banana...]
- **Constraint**: Do NOT hallucinate objects not present in the Habitat environment. Use standard Habitat terminology.

## Your Strict Reasoning Process (Internal Monologue)
1. **Identify Targets**: Extract the specific object(s) and receptacle(s) mentioned in the input `Action`.
2. **Analyze 'Before' Image**: Locate these specific targets. Describe their position and state (e.g., "Apple is on Table 1").
3. **Analyze 'After' Image**: Locate the SAME targets again. **Crucially check**: Did the state actually change?
    - If Action is "Pick Apple": Is the apple really in the gripper? Or is it still on the table?
    - If Action is "Open Fridge": Is the door actually open? Or does it look closed?
4. **Report Facts**: Describe exactly what you see. **Do NOT assume the action succeeded.** If the action was "Pick Apple" but the apple is still on the table, you MUST report "The apple remains on the table."

## Output Format
Output ONLY a JSON object:
{
    "state_before_action": "Description of the target object's position/state relative to receptacles/gripper in the first image.",
    "state_after_action": "Description of the target object's position/state relative to receptacles/gripper in the second image."
}

## Critical Rules (Anti-Hallucination)
- **TRUST YOUR EYES, NOT THE ACTION NAME.**
- If the action is "Pick up apple", but the image shows the gripper is empty, state: "The gripper is empty."
- If the action is "Navigate to table", but the image shows a wall, state: "Facing a wall."
- Focus ONLY on the objects and receptacles involved in the action. Ignore irrelevant background details.

## Examples

### Example 1 (Successful Action)
**Action**: "Pick up the apple"
**Input Images**: [Img1: Apple on table], [Img2: Apple in gripper]
**Output**:
{
    "state_before_action": "The robot is facing the table. An apple is visible resting on the table surface. The gripper is empty.",
    "state_after_action": "The robot's gripper is now holding the apple. The table surface where the apple was is now empty."
}

### Example 2 (Failed Action - VERY IMPORTANT)
**Action**: "Pick up the apple"
**Input Images**: [Img1: Apple on table], [Img2: Apple STILL on table, gripper closed on air]
**Output**:
{
    "state_before_action": "The robot is facing the table. An apple is visible on the table. The gripper is open.",
    "state_after_action": "The robot's gripper is closed but EMPTY. The apple REMAINS on the table surface. The action failed to change the object's state."
}

### Example 3 (Navigation)
**Action**: "Navigate to the fridge"
**Input Images**: [Img1: Facing Sofa], [Img2: Facing Fridge]
**Output**:
{
    "state_before_action": "The robot is facing a sofa.",
    "state_after_action": "The robot is now standing directly in front of the refrigerator. The fridge door is closed."
}
"""


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


SINGLE_STATE_SUMMARIZER_USER_PROMPT = """## Current Observation Task
You are observing a **Habitat simulation environment**. Analyze the provided image.

## Focus Areas
1. **Robot's Position**: What furniture or receptacle is the robot directly facing? (e.g., "Facing Table 1", "Facing the Fridge").
2. **Gripper State**: Is the robot holding anything? If yes, identify the object clearly.
3. **Visible Objects**: List key interactable objects (e.g., Apple, Bowl, Sponge) and their specific locations (e.g., "on the left side of the counter").
4. **Receptacle States**: Are doors (Fridge/Cabinet) open or closed?

## Constraint
- Only describe what is clearly visible.
- Use standard Habitat object names.

Output your analysis in JSON format with key: "current_state"."""


class WorldMindStateSummarizer:
    """LLM-based state summarization module for converting visual observations to text."""
    
    def __init__(self, model_name: str = None, use_separate_summarization: bool = False):
        """Initialize the state summarizer."""
        self.model_name = model_name or os.environ.get('WORLDMIND_SUMMARIZER_MODEL')
        self.api_key = os.environ.get('OPENAI_API_KEY')
        self.api_base = os.environ.get('OPENAI_API_BASE', None)
        self.use_separate_summarization = use_separate_summarization
        
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        try:
            from openai import OpenAI
            if self.api_base:
                self.client = OpenAI(api_key=self.api_key, base_url=self.api_base)
            else:
                self.client = OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("openai package is required")
        
        self.system_prompt = STATE_SUMMARIZER_SYSTEM_PROMPT
        self.total_summaries = 0
        
        summarization_mode = "separate" if use_separate_summarization else "combined"
        logger.info(f"WorldMind State Summarizer initialized with model: {self.model_name}, mode: {summarization_mode}")
    
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
            
            logger.debug(f"State summarization completed for action: {action_description}")
            
            return result
            
        except Exception as e:
            logger.error(f"State Summarizer error: {e}")
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
3. Describe the state of the robot's gripper (holding something or empty).
4. Be honest about what you see - do not assume the action succeeded just because it was attempted.

Output your analysis in JSON format with key: "current_state"."""
            
            state_after = self._summarize_single_state_with_prompt(after_image_path, state_after_prompt)
            
            self.total_summaries += 2
            
            logger.debug(f"State summarization (separate) completed for action: {action_description}")
            
            return {
                "state_before_action": state_before,
                "state_after_action": state_after
            }
            
        except Exception as e:
            logger.error(f"State Summarizer error: {e}")
            return {
                "state_before_action": f"[Summarization failed: {str(e)[:50]}]",
                "state_after_action": f"[Summarization failed: {str(e)[:50]}]"
            }
    
    def _summarize_single_state_with_prompt(self, image_path: str, prompt: str) -> str:
        """Summarize a single state with custom prompt."""
        try:
            image_url = local_image_to_data_url(image_path)
            
            messages = [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
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
            
            try:
                import re
                json_match = re.search(r'\{[^{}]*\}', result_text, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                    return result.get("current_state", result_text[:300])
            except:
                pass
            
            return result_text[:300]
            
        except Exception as e:
            logger.error(f"Single state summarization error: {e}")
            return f"[State summarization failed: {str(e)[:50]}]"

    def summarize_single_state(self, image_path: str) -> str:
        """Summarize a single observation state."""
        return self._summarize_single_state_with_prompt(image_path, SINGLE_STATE_SUMMARIZER_USER_PROMPT)
    
    def _parse_result(self, result_text: str) -> Dict[str, str]:
        """Parse the LLM output to extract state summaries."""
        import re
        
        default_result = {
            "state_before_action": "Unable to summarize state before action",
            "state_after_action": "Unable to summarize state after action"
        }
        
        try:
            json_match = re.search(r'\{[^{}]*\}', result_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return {
                    "state_before_action": result.get("state_before_action", default_result["state_before_action"]),
                    "state_after_action": result.get("state_after_action", default_result["state_after_action"])
                }
            
            result = json.loads(result_text)
            return {
                "state_before_action": result.get("state_before_action", default_result["state_before_action"]),
                "state_after_action": result.get("state_after_action", default_result["state_after_action"])
            }
            
        except json.JSONDecodeError:
            pass
        
        logger.warning("Failed to parse JSON from state summarizer response")
        return default_result
    
    def get_statistics(self) -> Dict:
        """Get summarization statistics."""
        return {
            "total_summaries": self.total_summaries,
            "mode": "separate" if self.use_separate_summarization else "combined"
        }
    
    def reset_statistics(self):
        """Reset statistics counters."""
        self.total_summaries = 0


def create_state_summarizer(model_name: str = None) -> WorldMindStateSummarizer:
    """Factory function to create a State Summarizer."""
    return WorldMindStateSummarizer(model_name=model_name)
