"""
WorldMind Plugin Reflector Module

This module provides an LLM-based self-reflection mechanism that analyzes
why an action's predicted state did not match the actual observation.

The reflector generates process experience entries when prediction errors occur.

Usage:
    reflector = WorldMindReflector(api_key="...", model_name="gpt-4o-mini")
    result = reflector.reflect(
        action_description="Pick up the apple",
        predicted_state="I will be holding the apple",
        state_before="Apple on counter. Hand empty.",
        state_after="Hand empty. Apple on counter.",
        env_feedback="Action failed: object not reachable"
    )
    # result = {"experience_entry": ["Environmental Logic: ..."]}
"""

import json
import re
from typing import Dict, Optional, List

from worldmind_plugin.llm_client import LLMClient
from worldmind_plugin.prompts import REFLECTOR_SYSTEM_PROMPT, REFLECTOR_USER_PROMPT
from worldmind_plugin.utils import get_logger


class WorldMindReflector:
    """
    LLM-based self-reflection module for analyzing prediction errors.
    
    The reflector generates process experience entries that capture:
    1. The specific failure context
    2. Universal corrective rules/workflows
    
    Attributes:
        llm_client: LLM client for API calls
        system_prompt: System prompt for reflection (customizable)
        use_env_feedback: Whether to include environment feedback in reflection
        total_reflections: Total number of reflections performed
        reflection_history: History of reflection results
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        model_name: str = "gpt-4o-mini",
        system_prompt: Optional[str] = None,
        use_env_feedback: bool = True
    ):
        """
        Initialize the reflector.
        
        Args:
            api_key: OpenAI API key. If None, reads from environment.
            api_base: OpenAI API base URL.
            model_name: Model name for reflection.
            system_prompt: Custom system prompt. If None, uses default.
            use_env_feedback: Whether to include environment feedback.
        """
        self.llm_client = LLMClient(
            api_key=api_key,
            api_base=api_base,
            model_name=model_name,
            is_multimodal=False  # Reflector is text-only
        )
        
        self.system_prompt = system_prompt or REFLECTOR_SYSTEM_PROMPT
        self.use_env_feedback = use_env_feedback
        self.logger = get_logger()
        
        # Statistics
        self.total_reflections = 0
        self.reflection_history: List[Dict] = []
        
        self.logger.info(f"WorldMindReflector initialized with model: {model_name}")
    
    def reflect(
        self,
        action_description: str,
        predicted_state: str,
        state_before: str,
        state_after: str,
        env_feedback: str = "",
        human_instruction: str = "",
        action_history: str = ""
    ) -> Dict[str, any]:
        """
        Perform self-reflection on a prediction error.
        
        Args:
            action_description: Description of the executed action
            predicted_state: The agent's predicted state after action
            state_before: State description before action
            state_after: State description after action
            env_feedback: Environment feedback for the action (if available)
            human_instruction: The human instruction/task goal
            action_history: Formatted string of previous action history
            
        Returns:
            dict: {"experience_entry": List[str]}
                - experience_entry: List of process experience strings
        """
        try:
            # Build user prompt
            feedback_text = env_feedback if self.use_env_feedback and env_feedback else "Not available"
            
            user_prompt = REFLECTOR_USER_PROMPT.format(
                human_instruction=human_instruction or "Not specified",
                action_history=action_history or "No previous actions",
                action_description=action_description,
                predicted_state=predicted_state,
                state_before=state_before,
                state_after=state_after,
                env_feedback=feedback_text
            )
            
            # Call LLM
            response = self.llm_client.chat(
                user_message=user_prompt,
                system_prompt=self.system_prompt
            )
            
            # Parse result
            result = self._parse_result(response)
            
            # Update statistics
            self.total_reflections += 1
            self.reflection_history.append({
                "action_description": action_description,
                "predicted_state": predicted_state,
                "env_feedback": env_feedback,
                "experience_entry": result.get("experience_entry", [])
            })
            
            self.logger.debug(f"Reflection result: {len(result.get('experience_entry', []))} entries")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Reflection error: {e}")
            return {"experience_entry": []}
    
    def _parse_result(self, result_text: str) -> Dict[str, any]:
        """
        Parse the LLM output to extract experience entries.
        
        Args:
            result_text: Raw LLM response
            
        Returns:
            dict: {"experience_entry": List[str]}
        """
        try:
            # Try to find JSON in response
            json_match = re.search(r'\{[^{}]*\}', result_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                
                # Handle experience_entry - ensure list format
                experience_entry = result.get("experience_entry", [])
                if isinstance(experience_entry, str):
                    experience_entry = [experience_entry] if experience_entry else []
                elif not isinstance(experience_entry, list):
                    experience_entry = []
                
                return {"experience_entry": experience_entry}
                
        except json.JSONDecodeError:
            pass
        
        # Fallback: return empty list
        return {"experience_entry": []}
    
    def get_statistics(self) -> Dict:
        """
        Get reflection statistics.
        
        Returns:
            dict with total_reflections and recent history
        """
        return {
            "total_reflections": self.total_reflections,
            "recent_history": self.reflection_history[-5:]  # Last 5 reflections
        }
    
    def reset_statistics(self):
        """Reset statistics counters."""
        self.total_reflections = 0
        self.reflection_history = []
    
    def set_system_prompt(self, prompt: str):
        """
        Set a custom system prompt.
        
        Args:
            prompt: New system prompt for reflection
        """
        self.system_prompt = prompt
        self.logger.info("Reflector system prompt updated")
    
    def set_use_env_feedback(self, use_feedback: bool):
        """
        Set whether to use environment feedback.
        
        Args:
            use_feedback: Whether to include environment feedback in reflection
        """
        self.use_env_feedback = use_feedback
        self.logger.info(f"Reflector use_env_feedback set to: {use_feedback}")


def create_reflector(
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    model_name: str = "gpt-4o-mini",
    system_prompt: Optional[str] = None,
    use_env_feedback: bool = True
) -> WorldMindReflector:
    """
    Factory function to create a WorldMind Reflector.
    
    Args:
        api_key: OpenAI API key
        api_base: OpenAI API base URL
        model_name: Model name for reflection
        system_prompt: Custom system prompt
        use_env_feedback: Whether to include environment feedback
        
    Returns:
        WorldMindReflector instance
    """
    return WorldMindReflector(
        api_key=api_key,
        api_base=api_base,
        model_name=model_name,
        system_prompt=system_prompt,
        use_env_feedback=use_env_feedback
    )
