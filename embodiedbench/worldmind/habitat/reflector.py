"""
WorldMind Reflector Module for Habitat Environment

LLM-based self-reflection mechanism that analyzes prediction errors
and generates process experience entries.
"""

import os
import json
import re
from typing import Dict, Optional, List
from openai import OpenAI

from embodiedbench.main import logger


REFLECTOR_SYSTEM_PROMPT = """## Role
You are a Root Cause Analyst & Knowledge Extractor for an Embodied Agent. Your goal is to analyze the Agent's interaction history and failure context to generate **Process Experience Entries**. You extract **TRUTH** from the history and **LOGIC** from the failures, specifically focusing on physical constraints and task execution workflows.

## Task 1: Extract Object Locations (Success Analysis)
Scan the `Recent History` for any successful object interactions.
* **Logic**: If the agent successfully Picked object `Y`, look backwards for the last successful `Navigation to X`.
* **Conclusion**: Object `Y` is located at `X`.
* **Output Format**: "Location Knowledge: [Object Name] is located at [Receptacle Name]."

## Task 2: Extract Environmental & Workflow Logic (Failure Analysis)
Compare the `Planner's Prediction` vs. `Actual Observation`. Diagnose why the plan failed using these specific guidelines:

### Guidelines for Logic Extraction:
1. **The "All" Objects Workflow Violation**:
   - **Context**: Human Instruction involves "all" items (e.g., "Put all apples in sink").
   - **Refining Logic**: After you have found the location of a target object, you must strictly initiate a repetitive loop: [Navigate to Source -> Pick -> Navigate to Destination -> PLACE].
   - **Special Note**: Not all target objects are necessarily in the same place. For each object, you may need to search and repeat the loop separately.
   - **Root Cause Check**: Did the agent attempt to pick a second item without placing the first one?
   - **Entry**: "Workflow Logic: For 'all items' tasks, after you have found the location of a target object, you must execute the loop: [Navigate to Source -> Pick -> Navigate to Destination -> PLACE]. Not all objects are necessarily in the same place, so you may need to search and repeat for each. You failed because you skipped the 'Place' action. You MUST ensure your hand is empty before returning to search for or pick the next item."

2. **The Multi-Goal Sequential Rule**:
   - **Context**: Goal involves multiple distinct targets (e.g., "Ball to table, Hammer to box").
   - **Failure**: Starting a new sub-goal while the current one is physically incomplete (hand not empty).
   - **Entry**: "Strategic Logic: For multi-target instructions, you MUST complete goals one-by-one. Ensure the current item is successfully PLACED and the gripper is empty before navigating to the next target's location."

3. **Visibility & Proximity**:
   - **Context**: Feedback says "object not near".
   - **Entry**: "To interact with [Object], you must first Navigate to its specific Receptacle, not just the general area."

4. **Hand Limits & Preconditions**:
   - **Entry**: "To Pick items inside a [Container], you must Open it first." / "To Pick a new object, you must Place the current one to free the gripper."

5. **Ambiguous Pick Failure**:
   - **Context**: Environment Feedback is "Last action is invalid. Robot cannot pick any object that is not near the robot. Navigate to other place to find the object."
   - **Logic**: There are two possibilities:
     1. The agent has not yet found the object (needs to navigate to its location).
     2. The object is already in the agent's hand (check if a successful Pick action occurred earlier).
   - **Entry**: "Ambiguous Pick Failure: When receiving this feedback, check your action history. If you have already successfully picked up the object, do not attempt to pick it again. Otherwise, you need to navigate to the correct location to find the object."

## Output Format (Strict JSON)
Output a JSON object with exactly ONE key: `experience_entry`.
**experience_entry**: A **List of Strings** containing Location Knowledge and Environmental/Workflow Logic.

## Examples

### Example 1 (Failure in "All" Task - Missing Place, objects not all in one place)
**Instruction**: "Put all the apples into the sink."
**History**: 
- Step 1: Navigate to Table 1 -> Success. (Found Apple 1 at Table 1)
- Step 2: Pick Apple 1 -> Success.
- Step 3: Navigate to Sink -> Success.
- Step 4: Place Apple 1 -> Success.
- Step 5: Navigate to Counter -> Success. (Found Apple 2 at Counter)
- Step 6: Pick Apple 2 -> Success.
- Step 7: Navigate to Sink -> Success.
- Step 8: [Skipped Place] -> Navigate back to Table 1.
**Current Failure**: 
- Action: Pick Apple 3. 
- Feedback: "Invalid: Robot is already holding an object."
**Output**:
{
    "experience_entry": [
        "Location Knowledge: Apple 1 is located at Table 1.",
        "Location Knowledge: Apple 2 is located at Counter.",
        "Workflow Logic: For 'all items' tasks, after you have found the location of a target object, you must execute the loop: [Navigate to Source -> Pick -> Navigate to Destination -> PLACE]. Not all objects are necessarily in the same place, so you may need to search and repeat for each. You failed because you skipped the 'Place' action. You MUST ensure your hand is empty before returning to search for or pick the next item."
    ]
}

### Example 2 (Multi-Goal Sequential Failure)
**Instruction**: "Move the ball to the brown table and the hammer to the black table."
**History**: 
- Step 1: Navigate to Counter -> Success. 
- Step 2: Pick Ball -> Success. 
- Step 3: Navigate to Brown Table -> Success.
**Current Failure**: 
- Action: Navigate to TV Stand (to find hammer). 
- Feedback: "Task logic warning: Current object 'ball' has not been placed yet."
**Output**:
{
    "experience_entry": [
        "Strategic Logic: For multi-target instructions, you MUST complete goals one-by-one. Ensure the current item (ball) is successfully PLACED and the gripper is empty before navigating to the next target's location (hammer)."
    ]
}

### Example 3 (Container Precondition)
**Current Failure**: Pick Can (at Fridge). Feedback: "Action failed (receptacle closed)."
**Output**:
{
    "experience_entry": [
        "Environmental Logic: To Pick an object inside a container like Fridge, you must Open it first."
    ]
}

### Example 4 (Ambiguous Pick Failure)
**Current Failure**: Pick Apple (at Table 1).  
**Feedback**: "Last action is invalid. Robot cannot pick any object that is not near the robot. Navigate to other place to find the object."
**Output**:
{
    "experience_entry": [
        "Ambiguous Pick Failure: When receiving this feedback, check your action history. If you have already successfully picked up the object, do not attempt to pick it again. Otherwise, you need to navigate to the correct location to find the object."
    ]
}
"""

REFLECTOR_USER_PROMPT_TEMPLATE = """## Reflection Task
The Planner's prediction was wrong. Extract Process Experience (Object Locations) from history and generate Environmental Logic based on the failure.

## 1. Human Instruction (Goal)
{human_instruction}

## 2. Trajectory (Recent History)
{experience_trajectory}

## 3. Failure Context
* **Observation BEFORE Action**: 
{state_before_action}
* **Action Executed**: {action_description}
* **Environment Feedback**: "{current_env_feedback}"
* **Planner's Prediction**: {predicted_state}
* **Actual Observation (AFTER Action)**: {state_after_action}

## 4. Requirement
Generate a JSON response with key: `experience_entry`.

**experience_entry**: 
   - A list of strings.
   - **Type 1 (Location)**: "Location Knowledge: [Object] is at [Receptacle]" (Derived from successful history).
   - **Type 2 (Logic)**: "[Condition/Error] -> [Required Precondition/Workflow]" (Derived from current failure).
"""


class WorldMindReflector:
    """LLM-based self-reflection module for analyzing prediction errors."""
    
    def __init__(self, model_name: str = None):
        """Initialize the reflector."""
        self.model_name = model_name or os.environ.get(
            'WORLDMIND_REFLECTOR_MODEL', 
            os.environ.get('WORLDMIND_DISCRIMINATOR_MODEL', 'gpt-4o-mini')
        )
        self.api_key = os.environ.get('OPENAI_API_KEY')
        self.api_base = os.environ.get('OPENAI_API_BASE', None)
        
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        if self.api_base:
            self.client = OpenAI(api_key=self.api_key, base_url=self.api_base)
        else:
            self.client = OpenAI(api_key=self.api_key)
        
        self.system_prompt = REFLECTOR_SYSTEM_PROMPT
        
        self.total_reflections = 0
        self.reflection_history: List[Dict] = []
        
        logger.info(f"WorldMind Reflector initialized with model: {self.model_name}")
    
    def reflect(
        self,
        action_description: str,
        predicted_state: str,
        state_before_action: str,
        state_after_action: str,
        current_env_feedback: str = "",
        action_id: Optional[int] = None,
        experience_trajectory: Optional[str] = None,
        human_instruction: Optional[str] = None
    ) -> Dict:
        """Perform self-reflection on a prediction error."""
        try:
            trajectory_text = experience_trajectory or "No previous experience."
            instruction_text = human_instruction or "Not specified."
            
            user_prompt = REFLECTOR_USER_PROMPT_TEMPLATE.format(
                human_instruction=instruction_text,
                experience_trajectory=trajectory_text,
                action_description=action_description,
                current_env_feedback=current_env_feedback,
                predicted_state=predicted_state,
                state_before_action=state_before_action,
                state_after_action=state_after_action
            )
            
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0,
                max_tokens=1024
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
                "current_env_feedback": current_env_feedback,
                "human_instruction": human_instruction,
                "experience_trajectory_provided": bool(experience_trajectory),
                "experience_entry": result.get("experience_entry", [])
            })
            
            logger.debug(f"Process Experience Entry: {result.get('experience_entry', [])}")
            
            return result
            
        except Exception as e:
            logger.error(f"WorldMind Reflector error: {e}")
            return {
                "experience_entry": []
            }
    
    def _parse_result(self, result_text: str) -> Dict:
        """Parse the LLM output to extract the reflection result."""
        try:
            json_match = re.search(r'\{[^{}]*\}', result_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                
                experience_entry = result.get("experience_entry", [])
                if isinstance(experience_entry, str):
                    experience_entry = [experience_entry] if experience_entry else []
                elif not isinstance(experience_entry, list):
                    experience_entry = []
                
                return {
                    "experience_entry": experience_entry
                }
        except json.JSONDecodeError:
            pass
        
        return {
            "experience_entry": []
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
    """Factory function to create a WorldMind Reflector."""
    return WorldMindReflector(model_name=model_name)
