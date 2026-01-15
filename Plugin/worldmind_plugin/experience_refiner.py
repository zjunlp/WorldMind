"""
WorldMind Plugin Experience Refiner Module

This module provides an LLM-based experience refinement mechanism that:
1. Receives retrieved goal experiences and process experiences
2. Uses LLM to consolidate and remove duplicates
3. Outputs merged_experience and initial_plan for current task

This is an optional module that can be enabled when experiences need to be
consolidated before being added to the prompt.

Usage:
    refiner = ExperienceRefiner(api_key="...", model_name="gpt-4o-mini")
    result = refiner.refine_for_task(
        current_instruction="Put the apple in the fridge",
        goal_experiences=[...],
        process_experience=[...]
    )
    # result = {"merged_experience": "...", "initial_plan": "..."}
"""

import json
import re
from typing import Dict, Optional, List

from worldmind_plugin.llm_client import LLMClient
from worldmind_plugin.prompts import (
    EXPERIENCE_REFINER_SYSTEM_PROMPT,
    EXPERIENCE_REFINER_USER_PROMPT
)
from worldmind_plugin.utils import get_logger


class ExperienceRefiner:
    """
    LLM-based experience refinement module for consolidating experiences.
    
    Consolidates goal experiences and process experiences into:
    1. merged_experience: Universal rules and location facts
    2. initial_plan: Preliminary plan for current task
    
    Attributes:
        llm_client: LLM client for API calls
        system_prompt: System prompt for refinement (customizable)
        total_refinements: Total number of refinements performed
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        model_name: str = "gpt-4o-mini",
        system_prompt: Optional[str] = None
    ):
        """
        Initialize the experience refiner.
        
        Args:
            api_key: OpenAI API key. If None, reads from environment.
            api_base: OpenAI API base URL.
            model_name: Model name for refinement.
            system_prompt: Custom system prompt. If None, uses default.
        """
        self.logger = get_logger()
        
        self.llm_client = None
        if api_key or model_name:
            try:
                self.llm_client = LLMClient(
                    api_key=api_key,
                    api_base=api_base,
                    model_name=model_name,
                    is_multimodal=False,
                    max_tokens=1024
                )
            except Exception as e:
                self.logger.warning(f"Failed to initialize refiner: {e}")
        
        self.system_prompt = system_prompt or EXPERIENCE_REFINER_SYSTEM_PROMPT
        
        # Statistics
        self.total_refinements = 0
        self.refinement_history: List[Dict] = []
        
        self.logger.info(f"ExperienceRefiner initialized with model: {model_name}")
    
    def refine_for_task(
        self,
        current_instruction: str,
        goal_experiences: List[Dict],
        process_experience: List[Dict]
    ) -> Dict:
        """
        Refine retrieved experiences and generate initial plan.
        
        Args:
            current_instruction: Current task's instruction
            goal_experiences: List of goal experiences
                             [{"instruction": str, "goal_experience": str}, ...]
            process_experience: List of process experiences
                               [{"instruction": str, "knowledge": str}, ...]
        
        Returns:
            dict: {
                "merged_experience": str,
                "initial_plan": str,
                "success": bool,
                "reason": str
            }
        """
        # If no experiences, return empty
        if not goal_experiences and not process_experience:
            return {
                "merged_experience": "",
                "initial_plan": "",
                "success": True,
                "reason": "No experiences to refine"
            }
        
        # If no client, use fallback
        if not self.llm_client:
            return self._fallback_refine(current_instruction, goal_experiences, process_experience)
        
        try:
            # Format experiences
            goal_text = self._format_goal_experiences(goal_experiences)
            process_text = self._format_process_experience(process_experience)
            
            # Build user prompt
            user_prompt = EXPERIENCE_REFINER_USER_PROMPT.format(
                current_instruction=current_instruction,
                goal_experiences=goal_text or "No goal experiences.",
                process_experience=process_text or "No process experiences."
            )
            
            # Call LLM
            response = self.llm_client.chat(
                user_message=user_prompt,
                system_prompt=self.system_prompt
            )
            
            # Parse result
            result = self._parse_result(response)
            
            # Update statistics
            self.total_refinements += 1
            self.refinement_history.append({
                "instruction": current_instruction,
                "goal_count": len(goal_experiences),
                "process_count": len(process_experience),
                "result": result
            })
            
            self.logger.info(f"Refinement completed for: {current_instruction[:50]}...")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Refinement failed: {e}")
            return self._fallback_refine(current_instruction, goal_experiences, process_experience)
    
    def _format_goal_experiences(self, experiences: List[Dict]) -> str:
        """Format goal experiences for prompt."""
        if not experiences:
            return ""
        
        formatted = []
        for i, exp in enumerate(experiences):
            instruction = exp.get('instruction', 'Unknown')
            experience = exp.get('goal_experience', '')
            formatted.append(f"{i + 1}. [From: {instruction}]\n   {experience}")
        
        return "\n\n".join(formatted)
    
    def _format_process_experience(self, knowledge_list: List[Dict]) -> str:
        """Format process experiences for prompt."""
        if not knowledge_list:
            return ""
        
        formatted = []
        for i, entry in enumerate(knowledge_list):
            instruction = entry.get('instruction', 'Unknown')
            knowledge = entry.get('knowledge', '')
            formatted.append(f"{i + 1}. [From: {instruction}]\n   {knowledge}")
        
        return "\n\n".join(formatted)
    
    def _parse_result(self, result_text: str) -> Dict:
        """Parse LLM output to extract merged_experience and initial_plan."""
        try:
            # Handle markdown code blocks
            json_text = result_text
            if "```json" in result_text:
                json_text = result_text.split("```json")[1].split("```")[0]
            elif "```" in result_text:
                parts = result_text.split("```")
                if len(parts) >= 3:
                    json_text = parts[1]
            
            # Find JSON object
            first_brace = json_text.find('{')
            if first_brace == -1:
                raise ValueError("No JSON found")
            
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
                raise ValueError("No closing brace")
            
            json_str = json_text[first_brace:last_brace + 1]
            result = json.loads(json_str)
            
            return {
                "merged_experience": result.get("merged_experience", ""),
                "initial_plan": result.get("initial_plan", ""),
                "success": True,
                "reason": ""
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to parse result: {e}")
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
        """Fallback refinement without LLM."""
        # Simple concatenation
        exp_parts = []
        for exp in goal_experiences[:3]:
            experience = exp.get('goal_experience', '')
            if experience:
                exp_parts.append(experience)
        
        knowledge_parts = []
        for entry in process_experience[:3]:
            knowledge = entry.get('knowledge', '')
            if knowledge:
                knowledge_parts.append(knowledge)
        
        merged = ""
        if exp_parts:
            merged += "Patterns: " + " | ".join(exp_parts)
        if knowledge_parts:
            if merged:
                merged += " | "
            merged += "Rules: " + " | ".join(knowledge_parts)
        
        return {
            "merged_experience": merged,
            "initial_plan": f"Apply patterns to: {current_instruction}",
            "success": True,
            "reason": "Fallback mode"
        }
    
    def format_for_prompt(self, refine_result: Dict) -> str:
        """
        Format refinement result for inclusion in system prompt.
        
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
            "recent_history": self.refinement_history[-5:]
        }
    
    def reset_statistics(self):
        """Reset statistics counters."""
        self.total_refinements = 0
        self.refinement_history = []
    
    def set_system_prompt(self, prompt: str):
        """Set a custom system prompt."""
        self.system_prompt = prompt
        self.logger.info("Refiner system prompt updated")


def create_experience_refiner(
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    model_name: str = "gpt-4o-mini",
    system_prompt: Optional[str] = None
) -> ExperienceRefiner:
    """
    Factory function to create an Experience Refiner.
    
    Args:
        api_key: OpenAI API key
        api_base: OpenAI API base URL
        model_name: Model name for refinement
        system_prompt: Custom system prompt
        
    Returns:
        ExperienceRefiner instance
    """
    return ExperienceRefiner(
        api_key=api_key,
        api_base=api_base,
        model_name=model_name,
        system_prompt=system_prompt
    )
