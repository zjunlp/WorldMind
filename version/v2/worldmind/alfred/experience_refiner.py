"""
WorldMind Experience Refiner Module for Alfred Environment

This module provides an LLM-based experience refinement mechanism that:
1. Receives retrieved goal experiences and process experience along with current task instruction
2. Uses LLM to remove duplicates and merge experiences
3. Outputs two keys: merged_experience (refined experiences) and initial_plan (suggested plan for current task)
4. The refined content is added to the system prompt for the next task
"""

import os
import json
import re
from typing import Dict, Optional, List, Tuple
from openai import OpenAI

from embodiedbench.main import logger


# Experience Refiner Prompts
EXPERIENCE_REFINER_SYSTEM_PROMPT = """You are an intelligent experience consolidation agent. Your goal is to distill specific past interactions into **Universal Rules** and **Actionable Facts**.

## Your Task
1. **Generalize Logic**: Convert specific past actions (e.g., "I opened the microwave to put the apple in") into **Universal Workflows** (e.g., "Placing items in enclosed receptacles like Microwaves requires Opening them first").
2. **Filter Facts**: Extract valid object locations, removing any conflicting or duplicate data.
3. **Plan Strategy**: Apply these universal rules to the current specific instruction.

## 1. Merged Experience Guidelines
The `merged_experience` field must consist of two distinct parts:
* **Part A: Universal Workflows & Rules**: Abstracted logic applicable to *any* similar object. DO NOT refer to the specific target object of the current task here. Use terms like "an object", "a container", "enclosed receptacles".
* **Part B: Valid Location Facts**: A clean list of verified object locations.
* **Conflict Resolution**: If Location A and Location B conflict for the same object, **OMIT** that object's location entirely.

## 2. Initial Plan Guidelines
* **Format**: `[Instruction Understanding]: [Simple Workflow Sequence]`
* **Logic**: Apply the **Universal Workflows** to the **Current Target**.

## 3. Few-Shot Demonstrations (Input -> Output)

### Example 1: Generalization & Conflict Resolution
**[Input]**
* **Instruction**: "Heat the egg and put it in the sink."
* **Goal Experiences**: "To heat the egg, I had to pick it, find microwave, open it, put egg in..."
* **Process Experience**: 
    1. "Location Knowledge: Egg is at Fridge."
    2. "Location Knowledge: Egg is at DiningTable." (CONFLICT! Discard.)
    3. "Environmental Logic: Failed because I didn't open microwave."

**[Output]**
{
    "merged_experience": "Heating workflow requires navigating to a Microwave, Opening it, Placing the object inside, Closing, and Turning on. Moving items requires Picking them up and navigating to the destination. \nLocation Facts: (Conflicting data for 'Egg' omitted).",
    "initial_plan": "Instruction Understanding: The user wants me to heat an egg and move it to the sink. Workflow: Find Egg (Explore) -> Pick Egg -> Find Microwave -> Open -> Put down Egg -> Close -> Turn on -> Open -> Pick Egg -> Find Sink -> Put down Egg"
}

### Example 2: Abstraction of Precision Rules
**[Input]**
* **Instruction**: "Pick up the ladle."
* **Process Experience**: 
    1. "Environmental Logic: Generic 'Find CounterTop' failed. Feedback states Ladle is at CounterTop_2. Must navigate to exact index."
    2. "Location Knowledge: Ladle is at CounterTop_2."

**[Output]**
{
    "merged_experience": "Navigation must be precise. If the environment or feedback specifies an indexed receptacle (e.g., CounterTop_2), generic navigation will fail to make the object visible. You must navigate to the specific instance.\nLocation Facts: Ladle is at CounterTop_2.",
    "initial_plan": "Instruction Understanding: The user wants to pick up a ladle located at a specific counter instance. Workflow: Find CounterTop_2 -> Pick Ladle"
}

### Example 3: Generalizing Complex Workflows (Clean & Store)
**[Input]**
* **Instruction**: "Put a clean pan in the refrigerator."
* **Goal Experiences**: "I cleaned the pan by putting it in sink, turning faucet on/off. Then I opened fridge to put pan in."
* **Process Experience**: 
    1. "Location Knowledge: Pan is at StoveBurner."
    2. "Environmental Logic: Cannot place item in Fridge because it is closed."

**[Output]**
{
    "merged_experience": "The standard cleaning workflow involves placing an object in a Sink, Turning the Faucet On, and then Turning it Off. Storing items in enclosed receptacles (like Fridges or Cabinets) requires the 'Open' action before placement.\nLocation Facts: Pan is at StoveBurner.",
    "initial_plan": "Instruction Understanding: The goal is to clean a dirty pan and store it inside the fridge. Workflow: Find StoveBurner -> Pick Pan -> Find Sink -> Put down Pan -> Turn on Faucet -> Turn off Faucet -> Pick Pan -> Find Fridge -> Open -> Put down Pan -> Close"
}

## Output Format
Output ONLY a JSON object with exactly two keys. Do not include markdown formatting or ```json ``` blocks.
{
    "merged_experience": "[Abstracted logic]... \nLocation Facts: [List of locations]...",
    "initial_plan": "[Instruction Understanding]: [Simple Workflow Sequence]"
}

!!! Please output only the raw JSON.
"""

EXPERIENCE_REFINER_USER_PROMPT = """## Experience Consolidation Task

### Current Task Instruction
{current_instruction}

### Retrieved Goal Experiences
{goal_experiences}

### Retrieved Process Experience
{process_experience}

Consolidate the knowledge by generalizing specific actions into universal rules and filtering location facts. Then generate the initial plan.

Output only the JSON.
"""


class ExperienceRefiner:
    """
    LLM-based experience refinement module for consolidating experiences at retrieval time.
    Consolidates goal experiences and process experience, removes duplicates.
    Generates initial plan for current task.
    Outputs consolidated experience and initial plan to add to system prompt.
    """
    
    def __init__(self, model_name: str = None, model_type: str = 'remote'):
        """
        Initialize the refiner.
        
        Args:
            model_name: LLM model name for refinement (uses agent's model if not specified)
            model_type: Model type ('remote' or 'custom')
        """
        self.model_name = model_name
        self.model_type = model_type
        
        # Initialize OpenAI client
        self.api_key = os.environ.get('OPENAI_API_KEY')
        self.api_base = os.environ.get('OPENAI_API_BASE', None)
        
        if not self.api_key:
            logger.warning("OPENAI_API_KEY not set, ExperienceRefiner may not work")
            self.client = None
        else:
            if self.api_base:
                self.client = OpenAI(api_key=self.api_key, base_url=self.api_base)
            else:
                self.client = OpenAI(api_key=self.api_key)
        
        # Statistics tracking
        self.total_refinements = 0
        self.refinement_history: List[Dict] = []
        
        logger.info(f"Experience Refiner initialized with model: {self.model_name}")
    
    def set_model(self, model_name: str, model_type: str = None):
        """
        Set or update the model for refinement.
        
        Args:
            model_name: Model name to use
            model_type: Model type (optional)
        """
        self.model_name = model_name
        if model_type:
            self.model_type = model_type
        logger.info(f"Experience Refiner model updated to: {self.model_name}")
    
    def refine_for_task(
        self,
        current_instruction: str,
        goal_experiences: List[Dict],
        process_experience: List[Dict]
    ) -> Dict:
        """
        Refine retrieved experiences and generate initial plan for the current task.
        
        Args:
            current_instruction: Current task's human instruction
            goal_experiences: List of retrieved goal experiences 
                             [{instruction: str, goal_experience: str}, ...]
            process_experience: List of retrieved process experience
                               [{instruction: str, knowledge: str}, ...]
        
        Returns:
            Dict with keys:
                - merged_experience: Consolidated experience summary
                - initial_plan: Preliminary plan for current task
                - success: Whether refinement succeeded
        """
        # If no experiences retrieved, return empty result
        if not goal_experiences and not process_experience:
            return {
                "merged_experience": "",
                "initial_plan": "",
                "success": True,
                "reason": "No experiences or knowledge to refine"
            }
        
        # If client unavailable, return simple concatenation
        if not self.client or not self.model_name:
            return self._fallback_refine(current_instruction, goal_experiences, process_experience)
        
        try:
            # Format goal experiences
            goal_exp_text = self._format_goal_experiences(goal_experiences)
            
            # Format process experience
            process_exp_text = self._format_process_experience(process_experience)
            
            # Build user prompt
            user_prompt = EXPERIENCE_REFINER_USER_PROMPT.format(
                current_instruction=current_instruction,
                goal_experiences=goal_exp_text if goal_exp_text else "No goal experiences retrieved.",
                process_experience=process_exp_text if process_exp_text else "No process experience retrieved."
            )
            
            # Call LLM
            messages = [
                {"role": "system", "content": EXPERIENCE_REFINER_SYSTEM_PROMPT},
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
            
            # Record statistics
            self.total_refinements += 1
            self.refinement_history.append({
                "instruction": current_instruction,
                "goal_exp_count": len(goal_experiences),
                "process_exp_count": len(process_experience),
                "result": result
            })
            
            logger.info(f"Experience refinement completed for task: {current_instruction[:50]}...")
            
            return result
            
        except Exception as e:
            logger.error(f"Experience refinement failed: {e}")
            return self._fallback_refine(current_instruction, goal_experiences, process_experience)
    
    def _format_goal_experiences(self, experiences: List[Dict]) -> str:
        """Format goal experiences for the prompt."""
        if not experiences:
            return ""
        
        formatted = []
        for i, exp in enumerate(experiences):
            instruction = exp.get('instruction', 'Unknown task')
            experience = exp.get('goal_experience', '')
            if not experience:
                # Fallback for old format
                experience = exp.get('success_experience', 'No experience available')
            formatted.append(f"{i+1}. [From task: {instruction}]\n   {experience}")
        
        return "\n\n".join(formatted)
    
    def _format_process_experience(self, knowledge_list: List[Dict]) -> str:
        """Format process experience for the prompt."""
        if not knowledge_list:
            return ""
        
        formatted = []
        for i, entry in enumerate(knowledge_list):
            instruction = entry.get('instruction', 'Unknown task')
            knowledge = entry.get('knowledge', 'No knowledge available')
            formatted.append(f"{i+1}. [From task: {instruction}]\n   {knowledge}")
        
        return "\n\n".join(formatted)
    
    def _parse_result(self, result_text: str) -> Dict:
        """Parse LLM output and extract merged_experience and initial_plan."""
        try:
            # Handle possible markdown code blocks
            json_text = result_text
            if "```json" in result_text:
                json_text = result_text.split("```json")[1].split("```")[0]
            elif "```" in result_text:
                json_text = result_text.split("```")[1].split("```")[0]
            
            # Find JSON object
            first_brace = json_text.find('{')
            if first_brace == -1:
                raise ValueError("No JSON object found")
            
            brace_count = 0
            last_brace = -1
            for i in range(first_brace, len(json_text)):
                if json_text[i] == '{':
                    brace_count += 1
                elif json_text[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        last_brace = i
                        break
            
            if last_brace == -1:
                raise ValueError("No matching closing brace")
            
            json_str = json_text[first_brace:last_brace+1]
            result = json.loads(json_str)
            
            merged_experience = result.get("merged_experience", "")
            initial_plan = result.get("initial_plan", "")
            
            return {
                "merged_experience": merged_experience,
                "initial_plan": initial_plan,
                "success": True,
                "reason": ""
            }
            
        except Exception as e:
            logger.warning(f"Failed to parse refiner result: {e}")
            return {
                "merged_experience": "",
                "initial_plan": "",
                "success": False,
                "reason": f"Parse failed: {e}"
            }
    
    def _fallback_refine(
        self,
        current_instruction: str,
        goal_experiences: List[Dict],
        process_experience: List[Dict]
    ) -> Dict:
        """
        Fallback refinement when LLM is not available.
        Simply concatenate experiences without LLM processing.
        """
        # Simple concatenation of goal experiences
        exp_parts = []
        for exp in goal_experiences:
            experience = exp.get('goal_experience', '') or exp.get('success_experience', '')
            if experience:
                exp_parts.append(experience)
        
        # Simple concatenation of process experience
        knowledge_parts = []
        for entry in process_experience:
            knowledge = entry.get('knowledge', '')
            if knowledge:
                knowledge_parts.append(knowledge)
        
        merged = ""
        if exp_parts:
            merged += "Goal Patterns: " + " | ".join(exp_parts[:3])
        if knowledge_parts:
            if merged:
                merged += " | "
            merged += "Process Rules: " + " | ".join(knowledge_parts[:3])
        
        return {
            "merged_experience": merged,
            "initial_plan": f"Apply the learned patterns to: {current_instruction}",
            "success": True,
            "reason": "Fallback mode - simple concatenation"
        }
    
    def format_for_prompt(self, refine_result: Dict) -> str:
        """
        Format the refinement result for inclusion in system prompt.
        
        Args:
            refine_result: Result from refine_for_task()
            
        Returns:
            Formatted string to add to system prompt
        """
        merged_exp = refine_result.get("merged_experience", "")
        initial_plan = refine_result.get("initial_plan", "")
        
        if not merged_exp and not initial_plan:
            return ""
        
        parts = []
        
        if merged_exp:
            parts.append("## Consolidated Experience from Similar Tasks")
            parts.append(merged_exp)
        
        if initial_plan:
            parts.append("\n## Preliminary Thinking")
            parts.append(initial_plan)
        
        return "\n".join(parts)
    
    def get_statistics(self) -> Dict:
        """Get refinement statistics."""
        return {
            "total_refinements": self.total_refinements,
            "refinement_history": self.refinement_history[-10:]
        }
    
    def reset_statistics(self):
        """Reset statistics counters."""
        self.total_refinements = 0
        self.refinement_history = []


# Factory function
def create_experience_refiner(model_name: str = None, model_type: str = 'remote') -> ExperienceRefiner:
    """
    Factory function to create an Experience Refiner.
    
    Args:
        model_name: LLM model name for refinement
        model_type: Model type ('remote' or 'custom')
        
    Returns:
        ExperienceRefiner instance
    """
    return ExperienceRefiner(model_name=model_name, model_type=model_type)
