"""
WorldMind Plugin State Summarizer Module

This module provides an LLM-based state summarization mechanism that converts
visual observations (images) into textual descriptions of the environment state.

ONLY used when is_multimodal=True in the configuration.

Usage:
    summarizer = WorldMindStateSummarizer(
        api_key="...", 
        model_name="gpt-4o",
        is_multimodal=True
    )
    result = summarizer.summarize_states(
        action_description="Pick up the apple",
        before_image_path="/path/to/before.png",
        after_image_path="/path/to/after.png"
    )
    # result = {"state_before_action": "...", "state_after_action": "..."}
"""

import json
import re
from typing import Dict, Optional

from worldmind_plugin.llm_client import LLMClient
from worldmind_plugin.prompts import (
    STATE_SUMMARIZER_SYSTEM_PROMPT,
    STATE_SUMMARIZER_USER_PROMPT,
    SINGLE_STATE_SUMMARIZER_PROMPT
)
from worldmind_plugin.utils import get_logger


class WorldMindStateSummarizer:
    """
    LLM-based state summarization module for converting visual observations to text.
    
    This module is ONLY used in multimodal (vision) environments.
    For text-only environments, the agent's own state description is used directly.
    
    Attributes:
        llm_client: LLM client with vision capabilities
        system_prompt: System prompt for summarization (customizable)
        total_summaries: Total number of summarizations performed
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        model_name: str = "gpt-4o",
        system_prompt: Optional[str] = None
    ):
        """
        Initialize the state summarizer.
        
        Args:
            api_key: OpenAI API key. If None, reads from environment.
            api_base: OpenAI API base URL.
            model_name: Model name for summarization (must support vision).
            system_prompt: Custom system prompt. If None, uses default.
        """
        self.llm_client = LLMClient(
            api_key=api_key,
            api_base=api_base,
            model_name=model_name,
            is_multimodal=True,  # State summarizer requires vision
            max_tokens=512
        )
        
        self.system_prompt = system_prompt or STATE_SUMMARIZER_SYSTEM_PROMPT
        self.logger = get_logger()
        
        # Statistics
        self.total_summaries = 0
        
        self.logger.info(f"WorldMindStateSummarizer initialized with model: {model_name}")
    
    def summarize_states(
        self,
        action_description: str,
        before_image_path: str,
        after_image_path: str
    ) -> Dict[str, str]:
        """
        Summarize the states before and after an action.
        
        Args:
            action_description: Description of the action executed
            before_image_path: Path to the observation image before action
            after_image_path: Path to the observation image after action
            
        Returns:
            dict: {"state_before_action": str, "state_after_action": str}
        """
        try:
            # Build user prompt
            user_prompt = STATE_SUMMARIZER_USER_PROMPT.format(
                action_description=action_description
            )
            
            # Call LLM with both images
            response = self.llm_client.chat_with_images(
                user_message=user_prompt,
                image_paths=[before_image_path, after_image_path],
                system_prompt=self.system_prompt
            )
            
            # Parse result
            result = self._parse_result(response)
            
            # Update statistics
            self.total_summaries += 1
            
            self.logger.debug(f"State summarization completed for: {action_description}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"State summarization error: {e}")
            return {
                "state_before_action": f"[Summarization failed: {str(e)[:50]}]",
                "state_after_action": f"[Summarization failed: {str(e)[:50]}]"
            }
    
    def summarize_single_state(self, image_path: str) -> str:
        """
        Summarize a single observation image.
        
        Args:
            image_path: Path to the observation image
            
        Returns:
            str: Text description of the current state
        """
        try:
            # Call LLM with single image
            response = self.llm_client.chat_with_images(
                user_message=SINGLE_STATE_SUMMARIZER_PROMPT,
                image_paths=[image_path],
                system_prompt=self.system_prompt
            )
            
            # Parse single state result
            return self._parse_single_state(response)
            
        except Exception as e:
            self.logger.error(f"Single state summarization error: {e}")
            return f"[Summarization failed: {str(e)[:50]}]"
    
    def _parse_result(self, result_text: str) -> Dict[str, str]:
        """
        Parse the LLM output for combined state summarization.
        
        Args:
            result_text: Raw LLM response
            
        Returns:
            dict: {"state_before_action": str, "state_after_action": str}
        """
        try:
            # Try to find JSON in response
            json_match = re.search(r'\{[^{}]*\}', result_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return {
                    "state_before_action": str(result.get("state_before_action", "")),
                    "state_after_action": str(result.get("state_after_action", ""))
                }
        except json.JSONDecodeError:
            pass
        
        # Fallback: split text roughly
        return {
            "state_before_action": result_text[:200],
            "state_after_action": result_text[200:400] if len(result_text) > 200 else ""
        }
    
    def _parse_single_state(self, result_text: str) -> str:
        """
        Parse the LLM output for single state summarization.
        
        Args:
            result_text: Raw LLM response
            
        Returns:
            str: State description
        """
        try:
            # Try to find JSON in response
            json_match = re.search(r'\{[^{}]*\}', result_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return str(result.get("current_state", result_text[:200]))
        except json.JSONDecodeError:
            pass
        
        return result_text[:300]
    
    def get_statistics(self) -> Dict:
        """
        Get summarization statistics.
        
        Returns:
            dict with total_summaries
        """
        return {
            "total_summaries": self.total_summaries
        }
    
    def reset_statistics(self):
        """Reset statistics counters."""
        self.total_summaries = 0
    
    def set_system_prompt(self, prompt: str):
        """
        Set a custom system prompt.
        
        Args:
            prompt: New system prompt for summarization
        """
        self.system_prompt = prompt
        self.logger.info("State summarizer system prompt updated")


def create_state_summarizer(
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    model_name: str = "gpt-4o",
    system_prompt: Optional[str] = None
) -> WorldMindStateSummarizer:
    """
    Factory function to create a WorldMind State Summarizer.
    
    Args:
        api_key: OpenAI API key
        api_base: OpenAI API base URL
        model_name: Model name for summarization (must support vision)
        system_prompt: Custom system prompt
        
    Returns:
        WorldMindStateSummarizer instance
    """
    return WorldMindStateSummarizer(
        api_key=api_key,
        api_base=api_base,
        model_name=model_name,
        system_prompt=system_prompt
    )
