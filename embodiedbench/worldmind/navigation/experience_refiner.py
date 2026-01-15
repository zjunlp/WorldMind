"""
WorldMind Experience Refiner Module for Navigation Environment

LLM-based experience refinement mechanism that:
1. Receives retrieved goal experiences and process experiences along with current task instruction
2. Uses LLM to remove duplicates and merge experiences
3. Outputs two keys: merged_experience (refined experiences) and initial_plan (suggested plan)
"""

import os
import json
import re
from typing import Dict, Optional, List
from openai import OpenAI

from embodiedbench.main import logger


EXPERIENCE_REFINER_SYSTEM_PROMPT = """You are an intelligent navigation strategist. Your goal is to consolidate past interactions into **Task-Specific Workflows** and **Critical Safety Rules**.

## Your Task
1. **Refine Goal Experiences**: The input contains entries like "Task: [Name] Experience: [Content]". You must extract the **Task Context** and convert the experience into a **Standardized Workflow** (using arrows `->`). Do NOT merge different task types into one; list them separately.
2. **Extract Process Experiences**: Convert specific error analysis (e.g., "My prediction failed because...") into objective **Critical Warnings**. Remove the first-person narrative and keep only the universal rule (the "Therefore..." part).
3. **Generate Initial Plan**: Create a concise, linear strategy for the *current* task based on the most relevant workflow.

## 1. Merged Experience Guidelines
The `merged_experience` field must be a text block containing two distinct sections:

* **Section A: Task-Specific Workflows**
    * For each retrieved experience, identify the specific target or context.
    * Format: `Workflow for [Specific Task/Target]: [Step 1] -> [Step 2] -> ...`
    * Example: `Workflow for [Finding TV]: Search -> Rotate -> Approach...`

* **Section B: Critical Warnings**
    * A list of key "Do's and Don'ts" derived from `Process Experiences`.
    * *Transformation Rule*: If input is "My prediction of seeing the door failed because rotation changes the scene... Therefore, use rotation sparingly...", output ONLY: "Use rotation sparingly and only when target is not visible."

## 2. Initial Plan Guidelines
* **Format**: `[Task Understanding]: [Concise Linear Workflow]`
* **Style**: Extremely concise. Use keywords and arrows.
* **Content**: First, state the current goal. Then, select the most relevant **Workflow** from Section A and adapt it linearly.

## Few-Shot Examples

### Example 1: Visible Target & Obstacle
**[Input]**
* **Instruction**: "Navigate to the TV"
* **Goal Experiences**: 
    1. "Task: Navigate to TV. Experience: I saw the TV on the left. I moved forward. Then I moved left to align. I stopped when I was close."
* **Process Experiences**: "My prediction of moving forward failed because a chair blocked me. Therefore, check for obstacles before moving. If blocked, side-step."

**[Output]**
{
    "merged_experience": "Workflow for [Navigate to TV]: Identify visible target -> Move Forward to close distance -> Use Lateral Movement (Left/Right) for alignment -> Stop when target occupies significant view. \nCritical Warnings: Check for obstacles before committing to forward movement. If blocked, use side-stepping immediately.",
    "initial_plan": "Task Understanding: Reach the visible TV. Strategy: Move Forward -> Adjust Left/Right to align -> Continue Forward until close."
}

### Example 2: Hidden Target (Search)
**[Input]**
* **Instruction**: "Navigate to the bed"
* **Goal Experiences**: 
    1. "Task: Navigate to Bed. Experience: Bed wasn't visible. I rotated right systematically until I saw it. Then moved forward."
    2. "Task: Navigate to Sofa. Experience: Used rotation to find sofa, then approached."
* **Process Experiences**: "My prediction failed because rotation changed the scene entirely. Therefore, stop rotating immediately once target is found."

**[Output]**
{
    "merged_experience": "Workflow for [Navigate to Bed]: Target Not Visible -> Execute systematic 90° Rotation -> Stop Rotation immediately upon detection -> Switch to Forward Approach.\nWorkflow for [Navigate to Sofa]: Rotate to search -> Approach once visible. \nCritical Warnings: Rotation drastically changes the scene; do not rotate if target is already in view.",
    "initial_plan": "Task Understanding: Locate and reach the bed. Strategy: Rotate 90° (Search) -> Detect Bed -> Stop Rotation -> Move Forward (Approach)."
}

## Output Format
Output ONLY a JSON object with exactly two keys. Do not include markdown formatting.
{
    "merged_experience": "Workflow for [Task A]: ... \nWorkflow for [Task B]: ... \nCritical Warnings: ...",
    "initial_plan": "[Task Understanding]: [Strategy Workflow]"
}

!!! Please output only the raw JSON.
"""

EXPERIENCE_REFINER_USER_PROMPT = """## Navigation Experience Consolidation Task

### Current Navigation Instruction
{current_instruction}

### Retrieved Goal Experiences
{goal_experiences}

### Retrieved Process Experiences
{process_experiences}

Refine the experiences into task-specific workflows and extract critical warnings. Then generate the concise initial plan.

Output only the JSON.
"""


class ExperienceRefiner:
    """LLM-based experience refinement module for consolidating navigation experiences at retrieval time."""
    
    def __init__(self, model_name: str = None, model_type: str = 'remote'):
        """Initialize the refiner."""
        self.model_name = model_name
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
                logger.warning(f"Failed to initialize ExperienceRefiner client: {e}")
        
        self.total_refinements = 0
        self.refinement_history: List[Dict] = []
        
        logger.info(f"Navigation Experience Refiner initialized with model: {self.model_name}")
    
    def set_model(self, model_name: str, model_type: str = None):
        """Set or update the model for refinement."""
        self.model_name = model_name
        if model_type:
            self.model_type = model_type
        logger.info(f"Experience Refiner model updated to: {self.model_name}")
    
    def refine_for_task(
        self,
        current_instruction: str,
        goal_experiences: List[Dict],
        process_experiences: List[Dict]
    ) -> Dict:
        """Refine retrieved experiences and generate initial plan for the current task."""
        if not goal_experiences and not process_experiences:
            return {
                "merged_experience": "",
                "initial_plan": "",
                "success": True,
                "reason": "No experiences to refine"
            }
        
        if not self.client or not self.model_name:
            return self._fallback_refine(current_instruction, goal_experiences, process_experiences)
        
        try:
            goal_exp_text = self._format_goal_experiences(goal_experiences)
            process_exp_text = self._format_process_experiences(process_experiences)
            
            user_prompt = EXPERIENCE_REFINER_USER_PROMPT.format(
                current_instruction=current_instruction,
                goal_experiences=goal_exp_text if goal_exp_text else "No goal experiences retrieved.",
                process_experiences=process_exp_text if process_exp_text else "No process experiences retrieved."
            )
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": EXPERIENCE_REFINER_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0,
                max_tokens=1024
            )
            
            result_text = response.choices[0].message.content
            result = self._parse_result(result_text)
            
            self.total_refinements += 1
            self.refinement_history.append({
                "instruction": current_instruction,
                "goal_exp_count": len(goal_experiences),
                "process_exp_count": len(process_experiences),
                "result": result
            })
            
            logger.info(f"Experience refinement completed for task: {current_instruction[:50]}...")
            
            return result
            
        except Exception as e:
            logger.error(f"Experience refinement failed: {e}")
            return self._fallback_refine(current_instruction, goal_experiences, process_experiences)
    
    def _format_goal_experiences(self, experiences: List[Dict]) -> str:
        """Format goal experiences for the prompt."""
        if not experiences:
            return ""
        
        formatted = []
        for i, exp in enumerate(experiences):
            instruction = exp.get('instruction', 'Unknown task')
            experience = exp.get('goal_experience', 'No experience available')
            formatted.append(f"{i+1}. [From task: {instruction}]\n   {experience}")
        
        return "\n\n".join(formatted)
    
    def _format_process_experiences(self, experience_list: List[Dict]) -> str:
        """Format process experiences for the prompt."""
        if not experience_list:
            return ""
        
        formatted = []
        for i, entry in enumerate(experience_list):
            instruction = entry.get('instruction', 'Unknown task')
            experience = entry.get('experience', 'No experience available')
            formatted.append(f"{i+1}. [From task: {instruction}]\n   {experience}")
        
        return "\n\n".join(formatted)
    
    def _parse_result(self, result_text: str) -> Dict:
        """Parse LLM output and extract merged_experience and initial_plan."""
        try:
            json_text = result_text
            if "```json" in result_text:
                json_text = result_text.split("```json")[1].split("```")[0]
            elif "```" in result_text:
                json_text = result_text.split("```")[1].split("```")[0]
            
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
            
            return {
                "merged_experience": result.get("merged_experience", ""),
                "initial_plan": result.get("initial_plan", ""),
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
        process_experiences: List[Dict]
    ) -> Dict:
        """Fallback refinement when LLM is not available."""
        exp_parts = []
        for exp in goal_experiences:
            experience = exp.get('goal_experience', '')
            if experience:
                exp_parts.append(experience)
        
        knowledge_parts = []
        for entry in process_experiences:
            experience = entry.get('experience', '')
            if experience:
                knowledge_parts.append(experience)
        
        merged = ""
        if exp_parts:
            merged += "Navigation Patterns: " + " | ".join(exp_parts[:3])
        if knowledge_parts:
            if merged:
                merged += " | "
            merged += "Navigation Rules: " + " | ".join(knowledge_parts[:3])
        
        return {
            "merged_experience": merged,
            "initial_plan": f"Apply navigation patterns to: {current_instruction}",
            "success": True,
            "reason": "Fallback mode - simple concatenation"
        }
    
    def format_for_prompt(self, refine_result: Dict) -> str:
        """Format the refinement result for inclusion in system prompt."""
        merged_exp = refine_result.get("merged_experience", "")
        initial_plan = refine_result.get("initial_plan", "")
        
        if not merged_exp and not initial_plan:
            return ""
        
        parts = []
        
        if merged_exp:
            parts.append("## Consolidated Navigation Experience from Similar Tasks")
            parts.append(merged_exp)
        
        if initial_plan:
            parts.append("\n## Preliminary Navigation Thinking")
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


def create_experience_refiner(model_name: str = None, model_type: str = 'remote') -> ExperienceRefiner:
    """Factory function to create an Experience Refiner for Navigation."""
    return ExperienceRefiner(model_name=model_name, model_type=model_type)
