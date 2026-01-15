"""
WorldMind Reflector Module for Navigation Environment

LLM-based self-reflection mechanism that analyzes why an action's predicted state
did not match the actual observation. Outputs experience entries (process experience).
"""

import os
import json
import re
from typing import Dict, Optional, List
from openai import OpenAI

from embodiedbench.main import logger


REFLECTOR_SYSTEM_PROMPT = """## Role
You are a Root Cause Analyst for a Navigation Robot. Your goal is to extract concise **World Knowledge** and **Navigation Logic** by analyzing the gap between predictions and reality.

## The Navigation Environment Context
The robot operates in indoor home environments with:
- **Available Actions**: Move forward/backward/left/right (0.25m), Rotate left/right (90°), Tilt camera up/down (30°)
- **Environment Elements**: Rooms (kitchen, living room, bedroom, etc.), Furniture (table, chair, sofa, bed), Appliances, Doors, Windows, Walls
- **Navigation Constraints**: 
  * Movement can be blocked by walls, furniture, or obstacles
  * Rotation changes the entire visible scene (90° turn)
  * Moving forward is the primary navigation strategy
  * Objects may appear larger/smaller as you move closer/farther

## Task: Extract Environmental Navigation Logic
For every prediction failure, generate a logic entry that:
1. **First** explains what the agent predicted vs what actually happened
2. **Then** derives the **Universal Navigation Rule** that can help future tasks

### Mandatory Logic Structure:
"Navigation Logic: My prediction that [expected state] failed because [specific reason]. Therefore, [Universal Navigation Rule]."

### Key Navigation Rules to Consider:
1. **Visibility vs Position**: Objects visible in the distance don't mean you're close to them
2. **Obstacle Blocking**: Forward movement may fail if obstacles are directly ahead
3. **Rotation Effect**: A 90° rotation completely changes what's visible - previous targets may disappear
4. **Distance Perception**: In indoor environments, 0.25m is a small step - significant movement requires multiple steps
5. **Room Transitions**: Passing through doorways may suddenly change visible objects
6. **Camera Tilt**: Tilting down shows floor/low objects; tilting up shows ceiling/high objects

## Special Cases

### Movement Blocked
If movement failed due to an obstacle:
"Navigation Logic: My prediction of moving forward failed because there was [obstacle type] directly ahead. Therefore, when movement is blocked, try alternative directions (left/right) or rotate to find a clearer path."

### Target Lost After Rotation
If the target disappeared after rotation:
"Navigation Logic: My prediction of still seeing [target] failed because a 90° rotation completely changes the visible scene. Therefore, avoid unnecessary rotations when the target is already visible - use forward/lateral movement instead."

### Distance Misjudgment
If the agent thought it was closer/farther than it actually is:
"Navigation Logic: My prediction of being [close to/far from] the target failed because [reason]. Therefore, [rule about distance estimation in indoor environments]."

## Output Format (Strict JSON)
Output a JSON object with one key: `experience_entry` (a list of strings).
Each entry should be a complete, actionable navigation rule.

{
    "experience_entry": [
        "Navigation Logic: [specific failure analysis]. Therefore, [universal rule]."
    ]
}

## Examples

### Example 1: Movement Blocked
**Action**: Move forward by 0.25 meters
**Predicted**: "I will be closer to the table"
**Actual**: "The robot did not move. A chair is blocking the path."

**Output**:
{
    "experience_entry": [
        "Navigation Logic: My prediction of moving closer to the table failed because a chair was blocking the forward path. Therefore, before moving forward, check for obstacles in the immediate path. If blocked, use lateral movement (Move left/right) to navigate around obstacles."
    ]
}

### Example 2: Target Lost After Rotation
**Action**: Rotate right by 90 degrees
**Predicted**: "I will have a better view of the door"
**Actual**: "Now facing a wall. The door is no longer visible."

**Output**:
{
    "experience_entry": [
        "Navigation Logic: My prediction of seeing the door after rotation failed because a 90° rotation completely changes the visible scene. Therefore, use rotation sparingly and only when the target is not visible. If the target is already in view, prefer forward movement to approach it."
    ]
}

### Example 3: Spatial Misjudgment
**Action**: Move forward by 0.25 meters
**Predicted**: "I will reach the sofa"
**Actual**: "Still several steps away from the sofa"

**Output**:
{
    "experience_entry": [
        "Navigation Logic: My prediction of reaching the sofa failed because 0.25m is a small step and the sofa was farther than expected. Therefore, in indoor environments, estimate distances conservatively - reaching distant objects requires multiple forward movements."
    ]
}
"""


REFLECTOR_USER_PROMPT_TEMPLATE = """## Reflection Task
Analyze the navigation prediction failure and extract useful knowledge.

## 1. Human Instruction (Navigation Goal)
{human_instruction}

## 2. Trajectory (Recent History)
{experience_trajectory}

## 3. Failure Context
* **Observation BEFORE Action**: {state_before_action}
* **Action Executed**: {action_description}
* **Environment Feedback**: "{current_env_feedback}"
* **Agent's Prediction**: {predicted_state}
* **Actual Observation (AFTER Action)**: {state_after_action}

## Requirement
Output JSON with `experience_entry` (list of strings).
Each entry must combine the prediction failure analysis with a universal navigation rule.
Focus on knowledge that would help similar navigation tasks in the future.
"""


class WorldMindReflector:
    """LLM-based self-reflection module for analyzing navigation prediction errors."""
    
    def __init__(self, model_name: str = None, model_type: str = "remote"):
        """Initialize the reflector."""
        self.model_name = model_name or os.environ.get(
            'WORLDMIND_REFLECTOR_MODEL', 
            os.environ.get('WORLDMIND_DISCRIMINATOR_MODEL', 'gpt-4o-mini')
        )
        self.model_type = model_type
        
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
                logger.warning(f"Failed to initialize reflector client: {e}")
        
        self.system_prompt = REFLECTOR_SYSTEM_PROMPT
        
        self.total_reflections = 0
        self.reflection_history: List[Dict] = []
        
        logger.info(f"WorldMind Navigation Reflector initialized with model: {self.model_name}")
    
    def reflect(
        self,
        action_description: str,
        predicted_state: str,
        state_before_action: str,
        state_after_action: str,
        env_feedback: str = "",
        action_id: Optional[int] = None,
        experience_trajectory: Optional[str] = None,
        human_instruction: Optional[str] = None
    ) -> Dict:
        """Perform self-reflection on a prediction error."""
        if not self.client:
            return {
                "experience_entry": [],
                "reflexion": "Reflector not available",
                "lesson": ""
            }
        
        try:
            trajectory_text = experience_trajectory or "No previous trajectory available."
            instruction_text = human_instruction or "Navigate to target location."
            
            user_prompt = REFLECTOR_USER_PROMPT_TEMPLATE.format(
                human_instruction=instruction_text,
                experience_trajectory=trajectory_text,
                action_description=action_description,
                current_env_feedback=env_feedback or "No specific feedback",
                predicted_state=predicted_state,
                state_before_action=state_before_action or "Not available",
                state_after_action=state_after_action or "Not available"
            )
            
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0,
                max_tokens=512
            )
            
            result_text = response.choices[0].message.content
            result = self._parse_result(result_text)
            
            self.total_reflections += 1
            
            self.reflection_history.append({
                "action_id": action_id,
                "action_description": action_description,
                "predicted_state": predicted_state,
                "state_before": state_before_action,
                "state_after": state_after_action,
                "current_env_feedback": env_feedback,
                "human_instruction": human_instruction,
                "experience_trajectory_provided": bool(experience_trajectory),
                "experience_entry": result.get("experience_entry", [])
            })
            
            logger.debug(f"Reflector output: {result.get('experience_entry', [])}")
            
            return result
            
        except Exception as e:
            logger.error(f"WorldMind Reflector error: {e}")
            return {
                "experience_entry": [],
                "reflexion": f"Reflection error: {e}",
                "lesson": ""
            }
    
    def _parse_result(self, result_text: str) -> Dict:
        """Parse the LLM output to extract the reflection result."""
        clean_text = result_text.strip()
        if clean_text.startswith("```"):
            lines = clean_text.split("\n")
            clean_text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        
        try:
            json_match = re.search(r'\{[^{}]*\}', clean_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                
                experience_entry = result.get("experience_entry", [])
                if isinstance(experience_entry, str):
                    experience_entry = [experience_entry] if experience_entry else []
                elif not isinstance(experience_entry, list):
                    experience_entry = []
                
                reflexion = experience_entry[0] if experience_entry else ""
                
                return {
                    "experience_entry": experience_entry,
                    "reflexion": reflexion,
                    "lesson": result.get("lesson", "")
                }
        except json.JSONDecodeError:
            pass
        
        return {
            "experience_entry": [],
            "reflexion": result_text.strip()[:500],
            "lesson": ""
        }
    
    def get_statistics(self) -> Dict:
        """Get reflection statistics."""
        return {
            "total_reflections": self.total_reflections
        }
    
    def reset_statistics(self):
        """Reset statistics counters."""
        self.total_reflections = 0
        self.reflection_history = []


def create_reflector(model_name: str = None) -> WorldMindReflector:
    """Factory function to create a WorldMind Reflector for Navigation."""
    return WorldMindReflector(model_name=model_name)
