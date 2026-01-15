"""
Experience Refiner for WorldMind Habitat.
Processes and refines experiences based on reflection results.
"""

import json
from typing import Any
from openai import OpenAI


class ExperienceRefiner:
    """
    Refines experiences by extracting key patterns from reflection results.
    Converts raw reflection data into structured goal and process experiences.
    """
    
    def __init__(self, config: dict):
        """Initialize the refiner with configuration."""
        self.config = config
        api_key = config.get("api_key", "")
        base_url = config.get("base_url", "")
        self.model = config.get("model", "gpt-4")
        
        self.client = OpenAI(api_key=api_key, base_url=base_url)
    
    def refine_goal_experience(self, reflection_result: dict, task_info: dict) -> dict:
        """
        Refine reflection result into goal experience format.
        
        Args:
            reflection_result: Raw reflection output from Reflector
            task_info: Context about the current task
            
        Returns:
            Structured goal experience entry
        """
        experience = {
            "task_type": task_info.get("task_type", ""),
            "pattern": reflection_result.get("pattern", ""),
            "insight": reflection_result.get("insight", ""),
            "action_sequence": reflection_result.get("action_sequence", []),
            "success_indicators": reflection_result.get("success_indicators", [])
        }
        return experience
    
    def refine_process_experience(self, reflection_result: dict, context: dict) -> dict:
        """
        Refine reflection result into process experience format.
        
        Args:
            reflection_result: Raw reflection output from Reflector
            context: Environmental and task context
            
        Returns:
            Structured process experience entry
        """
        experience = {
            "situation": context.get("situation", ""),
            "action": reflection_result.get("action", ""),
            "outcome": reflection_result.get("outcome", ""),
            "lesson": reflection_result.get("lesson", ""),
            "applicability": reflection_result.get("applicability", [])
        }
        return experience
    
    def batch_refine(self, reflections: list, experience_type: str = "goal") -> list:
        """
        Refine multiple reflection results in batch.
        
        Args:
            reflections: List of reflection results
            experience_type: Either "goal" or "process"
            
        Returns:
            List of refined experiences
        """
        refined = []
        for reflection in reflections:
            if experience_type == "goal":
                refined.append(self.refine_goal_experience(
                    reflection.get("result", {}),
                    reflection.get("task_info", {})
                ))
            else:
                refined.append(self.refine_process_experience(
                    reflection.get("result", {}),
                    reflection.get("context", {})
                ))
        return refined


def create_experience_refiner(config: dict) -> ExperienceRefiner:
    """Factory function to create an ExperienceRefiner instance."""
    return ExperienceRefiner(config)
