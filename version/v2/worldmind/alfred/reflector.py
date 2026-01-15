"""
WorldMind Reflector Module for Alfred Environment

This module provides an LLM-based self-reflection mechanism that analyzes
why an action's predicted state did not match the actual observation.
The reflection result outputs experience entries (process experience).
Note: The reflector does not use visual input, state_before and state_after use agent's visual_state_description.
Supports receiving experience trajectory as context to help LLM reflect based on historical trajectory.
"""

import os
import json
import re
from typing import Dict, Optional, List
from openai import OpenAI

from embodiedbench.main import logger


REFLECTOR_SYSTEM_PROMPT = """## Role
You are a Root Cause Analyst for an Alfred Agent. Your goal is to extract concise **World Knowledge** and **Universal Logic** by analyzing the gap between predictions and reality.

## Task: Integrated Environmental Logic & Workflows
For every failure, generate a logic entry that **first** explains the prediction failure in the specific context, and **then** derives the **Universal Corrective Rule/Workflow**.

### Mandatory Logic Structure:
"Environmental Logic: My prediction that [expected specific state] failed because [specific reason]. Therefore, [Universal Abstract Rule/Workflow]."

### Key Atomic Rules to Integrate:
1. **Indexing & Precision**: If feedback mentions a specific index (e.g., CounterTop_2), navigating to a generic name fails. You must use the exact indexed name.
2. **Hand Constraints**: Carrying two items is impossible. Must "Put down" current item before picking a new one.
3. **Container State**: Must "Open" enclosed receptacles (Fridge/Cabinet/Microwave) before interacting with contents.
4. **Destination Awareness Before Placement**: When analyzing the Trajectory (Recent History), if a "put down the object in hand" action is executed, check if the immediately preceding action was a "find" action targeting the intended destination. If not, in Environmental Logic, state: "Before executing 'put down the object in hand', you must use 'find' to explicitly navigate to the intended destination receptacle. This ensures the object will be placed at the correct location and avoids placement errors."

## Special Workflow Extraction (Universal Templates)
If the Human Instruction contains any of the following cases and the agent's prediction failed due to misunderstanding or missing the required workflow, you MUST append the corresponding workflow after the Environmental Logic.
**Before describing the workflow, you should briefly clarify the intended goal in your own words.**  

- **For "With" Placement Tasks** (e.g., "set A with B in it on C"):  
  If you want to place an object together with another object (e.g., put a plate with a spoon in it on the table), first clarify the goal (e.g., "The goal is to have the plate containing the spoon placed on the table.").
  Universal Workflow: First, find and pick up the **secondary object**; then, find the **main container object** and put down the secondary object into/on it; next, pick up the **main container object** (now containing the secondary object); finally, find the destination and put down the main container object.

- **For General Placement Tasks**:  
  If you want to place an object at a destination, first clarify the goal (e.g., "The goal is to have the soap placed on the shelf.").
  Universal Workflow: First, find the **target object**; then, pick up the object; next, find the **destination location**; finally, use 'put down' to place the object at the destination.

- **For Cleaning Tasks**:  
  If you want to clean an object before placing it, first clarify the goal (e.g., "The goal is to have a clean apple placed in the fridge.").
  Universal Workflow: Pick up the **dirty object**, find a **sink**, put down the object into the sink, find the **faucet** and turn it on and off (toggle), then pick up the object again. Only after this sequence is the object considered clean.

- **For Inferring Candidates** (Ambiguous Targets):  
  If the instruction refers to an ambiguous object, first clarify the goal (e.g., "The goal is to place the correct lamp as referred in the instruction.").
  Universal Workflow: Infer which available action_id objects correspond to the instruction (e.g., DeskLamp vs FloorLamp). Then, attempt the required operation on each plausible candidate sequentially until the task succeeds.

- **For Heating Tasks** ("warm", "heat", "microwaved" in instruction):  
  If the instruction requires a heated object, clarify the goal (e.g., "The goal is to have a heated slice of apple placed in the sink.").
  Universal Workflow: find the target object → pick up the object → find a Microwave → open the Microwave → put down the object in hand → close the Microwave → turn on the Microwave → turn off the Microwave → open the Microwave → find the object → pick up the object → close the Microwave.

- **For Cooling Tasks** ("cool", "cold", "chill", "fridge" in instruction):  
  If the instruction requires a cooled object, clarify the goal (e.g., "The goal is to have a cold piece of lettuce placed in the sink.").
  Universal Workflow: find the target object → pick up the object → find a Fridge → open the Fridge → put down the object in hand → close the Fridge → open the Fridge → find the object → pick up the object → close the Fridge.

- **For Slicing Tasks** ("slice", "piece of" in instruction):  
  If the instruction requires a sliced object, clarify the goal (e.g., "The goal is to have a sliced apple placed in the microwave.").
  Universal Workflow: find a Knife → pick up the Knife → find the object to slice → slice the object → find a CounterTop → put down the object in hand → find the sliced object → pick up the sliced object.

- **For Cleaning Tasks** ("clean", "cleaned", "wash" in instruction):  
  If the instruction requires a cleaned object, clarify the goal (e.g., "The goal is to have a cleaned plate placed on the table.").
  Universal Workflow: pick up the object → find a Sink → put down the object in hand into the sink → find the Faucet → turn on the Faucet → turn off the Faucet → pick up the object again.

**Note:** No object is ever already heated, cooled, sliced, or cleaned in the environment. If the instruction refers to an "already" processed object (e.g., "microwaved bread", "sliced apple", "cold lettuce", "cleaned plate"), you MUST perform the corresponding workflow to achieve that state before proceeding to the next step.

## Output Format (Strict JSON)
Output a JSON object with one key: `experience_entry` (a list of strings). 
Each logic string must begin with the prediction failure analysis.

## Examples

### Example 1 (Indexing Failure)
**Current Failure**: Pick up Ladle. 
**Feedback**: "Last action is invalid. Ladle is not visible because it is in CounterTop_2."
**Output**:
{
    "experience_entry": [
        "Environmental Logic: My prediction of holding the ladle failed because I navigated to a generic CounterTop instead of the specific CounterTop_2. Therefore, Navigation Precision Rule: You MUST navigate to the exact indexed receptacle mentioned in feedback to make objects visible."
    ]
}

### Example 2 (Special Workflow: 'With' Placement)
**Instruction**: "Put a mug with a pencil in it on the desk."
**Current Failure**: Pick up Pencil. Feedback: "Invalid: Hand is full (holding Mug)."
**Output**:
{
    "experience_entry": [
        "Environmental Logic: My prediction failed because I picked up the main container (Mug) first, blocking me from picking the secondary object (Pencil). Therefore, for 'with' instructions, strict sequencing is required. The goal is to have the mug containing the pencil placed on the desk. Universal Workflow: First, find and pick up the **secondary object**; then, find the **main container object** and put down the secondary object into/on it; next, pick up the **main container object**; finally, find the destination and put down the main container object."
    ]
}

### Example 3 (Special Workflow: Cleaning)
**Instruction**: "Put a clean apple in the fridge."
**Current Failure**: Task not complete (Apple is dirty).
**Output**:
{
    "experience_entry": [
        "Environmental Logic: My prediction that the task was done failed because the object was dirty. Therefore, cleaning requires a specific chain. The goal is to have a clean apple placed in the fridge. Universal Workflow: Pick up the **dirty object**, find a **sink**, put down the object into the sink, find the **faucet** and turn it on and off (toggle), then pick up the object again. Only after this sequence is the object considered clean."
    ]
}

### Example 4 (General Placement Workflow)
**Instruction**: "Put the soap on the shelf."
**Current Failure**: Put down Soap. Feedback: "Action failed (not at destination)."
**Output**:
{
    "experience_entry": [
        "Environmental Logic: My prediction failed because I attempted to put the object down before navigating to the correct destination. If you want to place an object at a target location, the correct workflow is: First, find the **target object**; then, pick up the object; next, find the **destination location**; finally, use 'put down' to place the object at the destination."
    ]
}

### Example 5 (Special Workflow: Heating)
**Instruction**: "Put a microwaved slice of bread in the fridge."
**Current Failure**: Task not complete (Bread is not microwaved).
**Output**:
{
    "experience_entry": [
        "Environmental Logic: My prediction that the bread was ready to be placed in the fridge failed because I did not heat it in the microwave. The goal is to have a microwaved slice of bread placed in the fridge. Universal Workflow: find the target object → pick up the object → find a Microwave → open the Microwave → put down the object in hand → close the Microwave → turn on the Microwave → turn off the Microwave → open the Microwave → find the object → pick up the object → close the Microwave."
    ]
}

### Example 6 (Special Workflow: Cooling)
**Instruction**: "Put a cold piece of lettuce in the sink."
**Current Failure**: Task not complete (Lettuce is not cold).
**Output**:
{
    "experience_entry": [
        "Environmental Logic: My prediction that the lettuce was ready to be placed in the sink failed because I did not chill it in the fridge. The goal is to have a cold piece of lettuce placed in the sink. Universal Workflow: find the target object → pick up the object → find a Fridge → open the Fridge → put down the object in hand → close the Fridge → open the Fridge → find the object → pick up the object → close the Fridge."
    ]
}

### Example 7 (Special Workflow: Slicing)
**Instruction**: "Put a sliced apple in the microwave."
**Current Failure**: Task not complete (Apple is not sliced).
**Output**:
{
    "experience_entry": [
        "Environmental Logic: My prediction that the apple was ready to be placed in the microwave failed because I did not slice it. The goal is to have a sliced apple placed in the microwave. Universal Workflow: find a Knife → pick up the Knife → find the object to slice → slice the object → find a CounterTop → put down the object in hand → find the sliced object → pick up the sliced object."
    ]
}

### Example 8 (Special Workflow: Cleaning)
**Instruction**: "Put a cleaned plate on the table."
**Current Failure**: Task not complete (Plate is not cleaned).
**Output**:
{
    "experience_entry": [
        "Environmental Logic: My prediction that the plate was ready to be placed on the table failed because I did not clean it. The goal is to have a cleaned plate placed on the table. Universal Workflow: pick up the object → find a Sink → put down the object in hand into the sink → find the Faucet → turn on the Faucet → turn off the Faucet → pick up the object again."
    ]
}
"""


REFLECTOR_USER_PROMPT_TEMPLATE = """## Reflection Task
Analyze the failure, compare the prediction with the feedback, and extract integrated logic.

## 1. Human Instruction (Goal)
{human_instruction}

## 2. Trajectory (Recent History)
{experience_trajectory}


## 3. Failure Context
* **Observation BEFORE Action**: {state_before_action}
* **Action Executed**: {action_description}
* **Environment Feedback**: "{current_env_feedback}"
* **Planner's Prediction**: {predicted_state}
* **Actual Observation (AFTER Action)**: {state_after_action}

## Requirement
Output JSON with `experience_entry`. 
Each entry must combine the **prediction failure analysis** and the **environmental rule**. 
Use specific indices (e.g., CounterTop_1) and avoid duplicating known location info.
"""

class WorldMindReflector:
    """
    LLM-based self-reflection module for analyzing prediction errors.
    Outputs experience_entry list (process experience entries).
    Note: The reflector does not use visual input, uses agent's visual_state_description as state description.
    Supports receiving experience trajectory as context (only includes action, env_feedback).
    """
    
    def __init__(self, model_name: str = None):
        """
        Initialize the reflector.
        
        Args:
            model_name: The name of the LLM model to use. If None, reads from environment.
        """
        self.model_name = model_name or os.environ.get(
            'WorldMind_REFLECTOR_MODEL', 
            os.environ.get('WorldMind_DISCRIMINATOR_MODEL', 'gpt-4o-mini')
        )
        self.api_key = os.environ.get('OPENAI_API_KEY')
        self.api_base = os.environ.get('OPENAI_API_BASE', None)
        
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required for WorldMind Reflector")
        
        # Initialize OpenAI client
        if self.api_base:
            self.client = OpenAI(api_key=self.api_key, base_url=self.api_base)
        else:
            self.client = OpenAI(api_key=self.api_key)
        
        self.system_prompt = REFLECTOR_SYSTEM_PROMPT
        
        # Statistics tracking
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
        human_instruction: Optional[str] = None,
        location_knowledge: Optional[str] = None
    ) -> Dict:
        """
        Perform self-reflection on a prediction error.
        
        Args:
            action_description: Description of the action that was executed
            predicted_state: The agent's predicted state after the action
            state_before_action: Agent's visual_state_description before action
            state_after_action: Agent's visual_state_description after action
            current_env_feedback: Environment feedback for the current action
            action_id: The action ID (optional)
            experience_trajectory: Formatted string of previous experience trajectory
            human_instruction: The human instruction/task goal
            location_knowledge: Accumulated location knowledge from previous steps
        
        Returns:
            dict: {"experience_entry": List[str]}
        """
        try:
            trajectory_text = experience_trajectory or "No previous experience."
            instruction_text = human_instruction or "Not specified."
            location_knowledge_text = location_knowledge or "No accumulated location knowledge yet."
            
            user_prompt = REFLECTOR_USER_PROMPT_TEMPLATE.format(
                human_instruction=instruction_text,
                experience_trajectory=trajectory_text,
                action_description=action_description,
                current_env_feedback=current_env_feedback,
                predicted_state=predicted_state,
                state_before_action=state_before_action,
                state_after_action=state_after_action
            )
            
            # Call LLM (text only, no vision)
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
            
            logger.debug(f"WorldMind Experience Entry: {result.get('experience_entry', [])}")
            
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
                
                # Handle experience_entry - ensure list format
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
        
        # Fallback: return empty list
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


# Factory function
def create_reflector(model_name: str = None) -> WorldMindReflector:
    """
    Factory function to create a WorldMind Reflector.
    
    Args:
        model_name: Model name for reflection
        
    Returns:
        WorldMindReflector instance
    """
    return WorldMindReflector(model_name=model_name)
