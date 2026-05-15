
import os
import json
import base64
from typing import Dict, Optional, Tuple
from mimetypes import guess_type

from embodiedbench.main import logger


def local_image_to_data_url(image_path: str) -> str:
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

    def __init__(self, model_name: str = None, use_separate_summarization: bool = False):

        self.model_name = model_name or os.environ.get('WorldMind_SUMMARIZER_MODEL')
        self.api_key = os.environ.get('OPENAI_API_KEY')
        self.api_base = os.environ.get('OPENAI_API_BASE', None)
        self.use_separate_summarization = use_separate_summarization

        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required for WorldMind State Summarizer")


        try:
            from openai import OpenAI
            if self.api_base:
                self.client = OpenAI(api_key=self.api_key, base_url=self.api_base)
            else:
                self.client = OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("openai package is required for WorldMind State Summarizer")

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
            logger.error(f"WorldMind State Summarizer (separate) error: {e}")

            return {
                "state_before_action": f"[Summarization failed: {str(e)[:50]}]",
                "state_after_action": f"[Summarization failed: {str(e)[:50]}]"
            }

    def _summarize_single_state_with_prompt(self, image_path: str, prompt: str) -> str:
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
        return self._summarize_single_state_with_prompt(image_path, SINGLE_STATE_SUMMARIZER_USER_PROMPT)

    def _parse_result(self, result_text: str) -> Dict[str, str]:
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
        return {
            "total_summaries": self.total_summaries,
            "mode": "separate" if self.use_separate_summarization else "combined"
        }

    def reset_statistics(self):
        self.total_summaries = 0





def create_state_summarizer(model_name: str = None) -> WorldMindStateSummarizer:
    return WorldMindStateSummarizer(model_name=model_name)


if __name__ == "__main__":
    """Test the state summarizer module."""
    import os


    if not os.environ.get('OPENAI_API_KEY'):
        print("Warning: OPENAI_API_KEY not set. Please set it to test the summarizer.")
        print("Example: export OPENAI_API_KEY='your-api-key'")
        exit(1)

    try:

        summarizer = WorldMindStateSummarizer()
        print(f"State Summarizer initialized with model: {summarizer.model_name}")
        print("State Summarizer is ready for use.")

    except Exception as e:
        print(f"Failed to initialize state summarizer: {e}")
        import traceback
        traceback.print_exc()




LOCATION_KNOWLEDGE_SYSTEM_PROMPT_VISION = """## Role
You are a **Location Knowledge Extractor** for a robot operating in the **EB-Habitat simulated environment**. Your goal is to analyze an observation image taken after a navigation action and summarize exactly what objects are present at that specific location.

## Task
Analyze the provided Navigation Action and the observation image to extract a comprehensive summary of visible objects.

## The Habitat Environment Context
- **Valid Receptacles**: [Cabinet, Fridge, Counter, Table, Sofa, Sink, TV Stand, Chair, Drawer, etc.]
- **Valid Objects**: [Apple, Bowl, Sponge, Spoon, Can, Cleanser, Lemon, Pear, Knife, Scissors, Toy Airplane, Banana, Bottle, Plate, Cup, Mug, Remote, Book, etc.]

## Output Format
Output ONLY a JSON object with exactly ONE string in the list:
{
    "location_knowledge": ["[Full Location Name]: There is one [object1], there are [n] [object2]s, ..."]
}

## Guidelines
1. **Location Name**: You MUST extract the **FULL string** immediately following "navigate to" from the provided navigation action.
   - *Example*: If action is "navigate to the left counter in the kitchen", the location name is "the left counter in the kitchen".
2. **Counting & Description Format**:
   - Use natural English descriptions for counts.
   - For a single object: Use "There is one [object]".
   - For multiple objects: Use "There are [n] [object]s".
   - **Do NOT** use the "[object] (count)" format.
3. **Single Entry Constraint**: The "location_knowledge" list MUST contain exactly one string.
4. **Comprehensiveness**: List ALL visible objects on or within the target receptacle as seen in the image.
5. **No Objects**: If no objects are visible, output: `{"location_knowledge": ["[Full Location Name]: no objects visible"]}`.

## Examples

### Example 1
**Navigation Action**: "navigate to table 1"
**Observation Image**: [Shows table with 1 apple and 1 bowl]
**Output**:
{
    "location_knowledge": ["table 1: There is one apple, there is one bowl"]
}

### Example 2
**Navigation Action**: "navigate to the left counter in the kitchen"
**Observation Image**: [Shows counter with 3 bananas and 1 knife]
**Output**:
{
    "location_knowledge": ["the left counter in the kitchen: There are three bananas, there is one knife"]
}

### Example 3
**Navigation Action**: "navigate to the refrigerator"
**Observation Image**: [Shows an empty open fridge]
**Output**:
{
    "location_knowledge": ["the refrigerator: no objects visible"]
}
"""


LOCATION_KNOWLEDGE_USER_PROMPT_VISION = """## Location Knowledge Extraction Task

**Navigation Action**: {nav_action}

Analyze the observation image taken AFTER this specific action.

**Instructions**:
1. Identify the location by using the **entire string** that follows "navigate to" in the action provided.
2. List all visible objects using the format: "There is one [object]" or "There are [n] [objects]".
3. Output ONLY the JSON object with the "location_knowledge" key.
"""





class LocationKnowledgeExtractor:

    def __init__(self, model_name: str = None):
        import os
        from openai import OpenAI


        self.model_name = model_name or os.environ.get('WorldMind_SUMMARIZER_MODEL', 'gpt-4o-mini')
        self.api_key = os.environ.get('OPENAI_API_KEY')
        self.api_base = os.environ.get('OPENAI_API_BASE', None)

        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required for Location Knowledge Extractor")


        if self.api_base:
            self.client = OpenAI(api_key=self.api_key, base_url=self.api_base)
        else:
            self.client = OpenAI(api_key=self.api_key)


        self.accumulated_knowledge: list = []


        self.visited_locations: set = set()


        self.total_extractions = 0

        from embodiedbench.main import logger
        logger.info(f"Location Knowledge Extractor initialized with model: {self.model_name} (vision mode)")

    @staticmethod
    def is_navigation_action(action_str: str) -> bool:
        if not action_str:
            return False
        action_lower = action_str.lower().strip()

        return (
            action_lower.startswith('navigate') or
            action_lower.startswith('nav(') or
            action_lower.startswith('nav ') or
            'navigate to' in action_lower
        )

    @staticmethod
    def extract_location_from_action(action_str: str) -> str:
        import re

        if not action_str:
            return ""

        action_lower = action_str.lower().strip()


        match = re.search(r'navigate\s+(?:to\s+)?(.+)', action_lower)
        if match:
            return match.group(1).strip()


        match = re.search(r'nav\s*\(\s*(.+?)\s*\)', action_lower)
        if match:
            return match.group(1).strip()


        match = re.search(r'nav\s+(.+)', action_lower)
        if match:
            return match.group(1).strip()

        return ""

    def is_first_visit(self, action_str: str) -> bool:
        if not self.is_navigation_action(action_str):
            return False

        location = self.extract_location_from_action(action_str)
        if not location:
            return False

        return location not in self.visited_locations

    def mark_location_visited(self, action_str: str):
        location = self.extract_location_from_action(action_str)
        if location:
            self.visited_locations.add(location)

    def find_first_visit_nav_actions(self, action_history: list) -> list:
        first_visits = []
        seen_locations = set()

        for idx, action in enumerate(action_history):

            if isinstance(action, dict):
                action_str = action.get('action', action.get('action_description', ''))
            else:
                action_str = str(action)

            if self.is_navigation_action(action_str):
                location = self.extract_location_from_action(action_str)
                if location and location not in seen_locations:
                    seen_locations.add(location)
                    first_visits.append((idx, action_str, location))

        return first_visits

    def extract_location_knowledge_from_image(
        self,
        nav_action: str,
        after_image_path: str
    ) -> list:
        import json
        import re
        from embodiedbench.main import logger

        if not after_image_path:
            logger.warning("Empty after_image_path provided")
            return []

        if not nav_action:
            logger.warning("No nav_action provided")
            return []

        try:

            image_url = local_image_to_data_url(after_image_path)


            user_prompt = LOCATION_KNOWLEDGE_USER_PROMPT_VISION.format(
                nav_action=nav_action
            )


            messages = [
                {"role": "system", "content": LOCATION_KNOWLEDGE_SYSTEM_PROMPT_VISION},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
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
                max_tokens=512
            )

            result_text = response.choices[0].message.content
            result = self._parse_result(result_text)


            self.total_extractions += 1

            if result:
                logger.info(f"Location knowledge extracted from image ({len(result)} entries): {result[0][:50] if result else ''}...")

            return result

        except Exception as e:
            from embodiedbench.main import logger
            logger.error(f"Location Knowledge Extractor (vision) error: {e}")
            return []

    def try_extract_on_first_visit(
        self,
        nav_action: str,
        after_image_path: str
    ) -> list:
        from embodiedbench.main import logger


        if not self.is_navigation_action(nav_action):
            return []


        if not self.is_first_visit(nav_action):
            location = self.extract_location_from_action(nav_action)
            logger.debug(f"Location '{location}' already visited, skipping extraction")
            return []


        self.mark_location_visited(nav_action)
        location = self.extract_location_from_action(nav_action)
        logger.info(f"First visit to '{location}', extracting location knowledge...")


        if not after_image_path:
            logger.warning("No after_image_path provided for location knowledge extraction")
            return []
        return self.extract_location_knowledge_from_image(nav_action, after_image_path)

    def _parse_result(self, result_text: str) -> list:
        import json
        import re

        try:

            json_match = re.search(r'\{[^{}]*\}', result_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                knowledge = result.get("location_knowledge", [])
                if isinstance(knowledge, list):
                    return knowledge
                elif isinstance(knowledge, str):
                    return [knowledge] if knowledge else []


            result = json.loads(result_text)
            knowledge = result.get("location_knowledge", [])
            if isinstance(knowledge, list):
                return knowledge
            elif isinstance(knowledge, str):
                return [knowledge] if knowledge else []

        except json.JSONDecodeError:
            pass

        return []

    def add_to_accumulated_knowledge(self, knowledge_list: list):
        for knowledge in knowledge_list:
            if knowledge and knowledge.strip() and knowledge not in self.accumulated_knowledge:
                self.accumulated_knowledge.append(knowledge)

    def get_accumulated_knowledge_str(self) -> str:
        if not self.accumulated_knowledge:
            return ""

        return "\n".join([f"- {k}" for k in self.accumulated_knowledge])

    def get_accumulated_knowledge_list(self) -> list:
        return self.accumulated_knowledge.copy()

    def get_visited_locations(self) -> set:
        return self.visited_locations.copy()

    def reset(self):
        self.accumulated_knowledge = []
        self.visited_locations = set()

    def get_statistics(self) -> dict:
        return {
            "total_extractions": self.total_extractions,
            "accumulated_count": len(self.accumulated_knowledge),
            "visited_locations_count": len(self.visited_locations),
            "visited_locations": list(self.visited_locations)
        }


def create_location_knowledge_extractor(
    model_name: str = None,
    language_only: bool = False
) -> LocationKnowledgeExtractor:
    return LocationKnowledgeExtractor(model_name=model_name, language_only=language_only)
