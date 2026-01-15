"""
WorldMind Navigation Prompts

Contains WorldMind prompt templates for navigation environment with predicted_state field.
"""

import json
import re
from typing import Dict, List, Any, Optional


NAVIGATION_ACTIONS = [
    "Move forward by 0.25 meters",
    "Move backward by 0.25 meters",
    "Move left by 0.25 meters",
    "Move right by 0.25 meters",
    "Rotate left by 90 degrees",
    "Rotate right by 90 degrees",
    "Tilt camera up by 30 degrees",
    "Tilt camera down by 30 degrees"
]


WORLDMIND_NAVIGATION_SYSTEM_PROMPT = '''## You are an intelligent navigation robot operating in a home environment. Your mission is to plan a sequence of actions to reach a specific target location or object based on visual observations.

## The available action id (0 ~ {}) and action names are: {}.

*** Strategy ***

1. Locate the Target Object Type: Clearly describe the spatial location of the target object 
from the observation image (i.e. in the front left side, a few steps from current standing point).

2. Navigate by *** Using Move forward and Move right/left as main strategy ***, since any point can be reached through a combination of those. \
When planning for movement, reason based on target object\'s location and obstacles around you. \

3. Focus on primary goal: Only address invalid action when it blocks you from moving closer in the direction to target object. In other words, \
do not overly focus on correcting invalid actions when direct movement towards target object can still bring you closer. \

4. *** Use Rotation Sparingly ***, only when you lose track of the target object and it\'s not in your view. If so, plan nothing but ONE ROTATION at a step until that object appears in your view. \
After the target object appears, start navigation and avoid using rotation until you lose sight of the target again.

5. *** Do not complete task too early until you can not move any closer to the object, i.e. try to be as close as possible.

*** WorldMind State Prediction ***
For each action in your plan, you must predict what you expect to observe after executing that action.
Include a "predicted_state" field in each action that describes:
- Expected changes in your view (objects appearing larger/smaller, new objects visible, etc.)
- Expected spatial relationship changes (closer to target, different room visible, etc.)
- This helps the system learn from prediction errors and improve navigation.


{}

----------
*** State Prediction ***
For each action in your plan, you must predict what you expect to observe after executing that action.
Include a "predicted_state" field in each action that describes:
- Expected changes in your view (objects appearing larger/smaller, new objects visible, etc.)
- Expected spatial relationship changes (closer to target, different room visible, etc.)
This helps the system learn from prediction errors and improve navigation.

The output json format should be {{"language_plan": str, "executable_plan": List[{{"action_id": int, "action_name": str, "predicted_state": str}}...]}}
For the "predicted_state" field:
If the target object or destination is NOT VISIBLE (Exploration Phase), you MUST output exactly the string: "Exploration phase: target not visible, prediction skipped."
Otherwise, describe the specific environmental change.
!!! Please do not output anything other than the above-mentioned JSON, do not include ```json and ```!!!
'''


WORLDMIND_NAVIGATION_TEMPLATE = """Current task: {instruction}

{history}

Based on the current observation, plan your next actions to complete the navigation task.
For each action, include a predicted_state describing what you expect to observe after executing it.
"""


def get_worldmind_first_prompt(num_actions: int, language_only: bool = False, multistep: bool = False, multiview: bool = False) -> str:
    """Get WorldMind first prompt (used for first step)."""
    from embodiedbench.planner.planner_utils import template, template_lang
    
    observation_desc = 'an first person view observation' if not multistep else 'latest 3 steps of the first person view observations'
    if multiview:
        observation_desc += ' and a overhead view of the house where the silver circle represents where you locates (Notice:The part hanging on the outside is your arm, and it is on your right side)'
    
    first_prompt = f'''\nTo achieve the task, 1. Reason about the current visual state and your final goal, and 2. Reflect on the effect of previous actions. 3. Summarize how you learn from the Strategy and Examples provided \
\nAim for about 1-2 actions in this step. !!!Notice: you cannot assess the situation until the whole plan in this planning step is finished executed, so plan accordingly.\
\nAt last, output the action id(s) (0 ~ {num_actions-1}) from the available actions to execute. 

The input given to you is {observation_desc}. Plan accordingly based on the visual observation.
'''
    
    return first_prompt


def get_worldmind_following_prompt(num_actions: int, language_only: bool = False, multistep: bool = False, multiview: bool = False) -> str:
    """Get WorldMind following prompt (used for subsequent steps)."""
    observation_desc = 'an first person view observation' if not multistep else 'latest 3 steps of the first person view observations'
    if multiview:
        observation_desc += ' and a overhead view of the house where the silver circle represents where you locates (Notice:The part hanging on the outside is your arm, and it is on your right side)'
    
    following_prompt = f'''\nTo achieve the task, 1. Reason about the current visual state and your final goal, and 2. Reflect on the effect of previous actions. 3. Summarize how you learn from the Strategy and Examples provided \
\nAim for about 5-6 actions in this step to be closer to the target object. !!!Notice: you cannot assess the situation until the whole plan in this planning step is finished executed, so plan accordingly.\
\nAt last, output the action id(s) (0 ~ {num_actions-1}) from the available actions to execute. 

The input given to you is {observation_desc}. Plan accordingly based on the visual observation.
'''
    
    return following_prompt


def format_actions_list() -> str:
    """Format actions list."""
    return "\n".join([f"{i}: {action}" for i, action in enumerate(NAVIGATION_ACTIONS)])


def parse_navigation_response(response: str) -> Dict[str, Any]:
    """Parse navigation model response, extracting actions and predicted states."""
    try:
        clean_response = response.strip()
        if clean_response.startswith("```"):
            lines = clean_response.split("\n")
            start_idx = 1 if lines[0].startswith("```") else 0
            end_idx = -1 if lines[-1].strip() == "```" else len(lines)
            clean_response = "\n".join(lines[start_idx:end_idx])
        
        result = json.loads(clean_response)
        return result
    except json.JSONDecodeError:
        pass
    
    json_pattern = r'\{[^{}]*"language_plan"[^{}]*"executable_plan"[^{}]*\[.*?\][^{}]*\}'
    match = re.search(json_pattern, response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    
    return {
        "language_plan": "",
        "executable_plan": []
    }


def extract_predicted_states(response_dict: Dict) -> List[str]:
    """Extract all predicted states from parsed response."""
    states = []
    executable_plan = response_dict.get("executable_plan", [])
    for action in executable_plan:
        if isinstance(action, dict):
            states.append(action.get("predicted_state", ""))
    return states


def extract_action_ids(response_dict: Dict) -> List[int]:
    """Extract all action IDs from parsed response."""
    action_ids = []
    executable_plan = response_dict.get("executable_plan", [])
    for action in executable_plan:
        if isinstance(action, dict):
            action_id = action.get("action_id")
            if action_id is not None:
                action_ids.append(action_id)
    return action_ids
