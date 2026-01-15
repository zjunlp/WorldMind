"""
WorldMind Plugin Discriminator Module

This module provides an LLM-based discriminator that compares the agent's predicted state
with the actual state after executing an action.

The discriminator determines if there is a prediction error, which can trigger reflection.

Usage:
    discriminator = WorldMindDiscriminator(api_key="...", model_name="gpt-4o-mini")
    result = discriminator.discriminate(
        predicted_state="I will be holding the apple",
        actual_state="Hand is empty. Apple on table."
    )
    # result = {"match": False, "reason": "Agent predicted holding apple, but hand is empty"}
"""

import json
import re
from typing import Dict, Optional

from worldmind_plugin.llm_client import LLMClient
from worldmind_plugin.prompts import DISCRIMINATOR_SYSTEM_PROMPT, DISCRIMINATOR_USER_PROMPT
from worldmind_plugin.utils import get_logger


class WorldMindDiscriminator:
    """
    LLM-based discriminator for comparing predicted states with actual states.
    
    The discriminator uses the "Default True" principle - it only returns match=False
    when there is a clear, undeniable factual contradiction.
    
    Attributes:
        llm_client: LLM client for API calls
        system_prompt: System prompt for discrimination (customizable)
        total_judgments: Total number of discriminations performed
        match_count: Number of matches
        mismatch_count: Number of mismatches
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        model_name: str = "gpt-4o-mini",
        system_prompt: Optional[str] = None
    ):
        """
        Initialize the discriminator.
        
        Args:
            api_key: OpenAI API key. If None, reads from environment.
            api_base: OpenAI API base URL.
            model_name: Model name for discrimination.
            system_prompt: Custom system prompt. If None, uses default.
        """
        self.llm_client = LLMClient(
            api_key=api_key,
            api_base=api_base,
            model_name=model_name,
            is_multimodal=False  # Discriminator is text-only
        )
        
        self.system_prompt = system_prompt or DISCRIMINATOR_SYSTEM_PROMPT
        self.logger = get_logger()
        
        # Statistics
        self.total_judgments = 0
        self.match_count = 0
        self.mismatch_count = 0
        
        self.logger.info(f"WorldMindDiscriminator initialized with model: {model_name}")
    
    def discriminate(
        self,
        predicted_state: str,
        actual_state: str,
        action_description: str = ""
    ) -> Dict[str, any]:
        """
        Compare the predicted state with the actual state.
        
        Args:
            predicted_state: The agent's predicted state after action execution
            actual_state: The actual observed state after action execution
            action_description: Description of the executed action (optional)
            
        Returns:
            dict: {"match": bool, "reason": str}
                - match: True if states are consistent, False if contradiction found
                - reason: Explanation of the judgment
        """
        try:
            # Build user prompt
            user_prompt = DISCRIMINATOR_USER_PROMPT.format(
                predicted_state=predicted_state,
                actual_state=actual_state
            )
            
            # Call LLM
            response = self.llm_client.chat(
                user_message=user_prompt,
                system_prompt=self.system_prompt
            )
            
            # Parse result
            result = self._parse_result(response)
            
            # Update statistics
            self._update_statistics(result["match"])
            
            self.logger.debug(f"Discrimination result: match={result['match']}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Discrimination error: {e}")
            # Default to match=True to avoid blocking the process
            return {
                "match": True,
                "reason": f"Discriminator error: {str(e)}"
            }
    
    def _parse_result(self, result_text: str) -> Dict[str, any]:
        """
        Parse the LLM output to extract the judgment result.
        
        Args:
            result_text: Raw LLM response
            
        Returns:
            dict: {"match": bool, "reason": str}
        """
        try:
            # Try to find JSON in response
            json_match = re.search(r'\{[^{}]*\}', result_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return {
                    "match": bool(result.get("match", True)),
                    "reason": str(result.get("reason", "No reason provided"))
                }
        except json.JSONDecodeError:
            pass
        
        # Fallback: determine from text
        lower_text = result_text.lower()
        if "false" in lower_text and ("match" in lower_text or "conflict" in lower_text):
            return {"match": False, "reason": result_text[:500]}
        
        return {"match": True, "reason": result_text[:500]}
    
    def _update_statistics(self, is_match: bool):
        """Update internal statistics."""
        self.total_judgments += 1
        if is_match:
            self.match_count += 1
        else:
            self.mismatch_count += 1
    
    def get_statistics(self) -> Dict:
        """
        Get discrimination statistics.
        
        Returns:
            dict with total_judgments, match_count, mismatch_count, match_rate
        """
        return {
            "total_judgments": self.total_judgments,
            "match_count": self.match_count,
            "mismatch_count": self.mismatch_count,
            "match_rate": self.match_count / self.total_judgments if self.total_judgments > 0 else 0
        }
    
    def reset_statistics(self):
        """Reset statistics counters."""
        self.total_judgments = 0
        self.match_count = 0
        self.mismatch_count = 0
    
    def set_system_prompt(self, prompt: str):
        """
        Set a custom system prompt.
        
        Args:
            prompt: New system prompt for discrimination
        """
        self.system_prompt = prompt
        self.logger.info("Discriminator system prompt updated")


def create_discriminator(
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    model_name: str = "gpt-4o-mini",
    system_prompt: Optional[str] = None
) -> WorldMindDiscriminator:
    """
    Factory function to create a WorldMind Discriminator.
    
    Args:
        api_key: OpenAI API key
        api_base: OpenAI API base URL
        model_name: Model name for discrimination
        system_prompt: Custom system prompt
        
    Returns:
        WorldMindDiscriminator instance
    """
    return WorldMindDiscriminator(
        api_key=api_key,
        api_base=api_base,
        model_name=model_name,
        system_prompt=system_prompt
    )
