"""
WorldMind Plugin LLM Client Module

This module provides a unified interface for calling LLM models via OpenAI-compatible APIs.
Supports both multimodal (vision) and text-only modes.

Usage:
    client = LLMClient(api_key="...", model_name="gpt-4o-mini")
    response = client.chat("Hello, world!")
    
    # For multimodal:
    client = LLMClient(api_key="...", model_name="gpt-4o", is_multimodal=True)
    response = client.chat_with_images("Describe this", ["/path/to/image.png"])
"""

import os
from typing import List, Dict, Optional, Union, Any

from worldmind_plugin.utils import local_image_to_data_url, get_logger


class LLMClient:
    """
    Unified LLM client for WorldMind plugin.
    
    Supports both text-only and multimodal (vision) modes via OpenAI-compatible API.
    
    Attributes:
        api_key: OpenAI API key
        api_base: OpenAI API base URL (optional)
        model_name: Name of the model to use
        is_multimodal: Whether to use vision capabilities
        temperature: Sampling temperature (default: 0)
        max_tokens: Maximum tokens in response (default: 1024)
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        model_name: str = "gpt-4o-mini",
        is_multimodal: bool = False,
        temperature: float = 0,
        max_tokens: int = 1024
    ):
        """
        Initialize the LLM client.
        
        Args:
            api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env var.
            api_base: OpenAI API base URL. If None, reads from OPENAI_API_BASE env var.
            model_name: Name of the model to use.
            is_multimodal: Whether the model supports vision input.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.
        """
        self.api_key = api_key or os.environ.get('OPENAI_API_KEY')
        self.api_base = api_base or os.environ.get('OPENAI_API_BASE')
        self.model_name = model_name
        self.is_multimodal = is_multimodal
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        self.logger = get_logger()
        
        if not self.api_key:
            raise ValueError("API key is required. Set OPENAI_API_KEY env var or pass api_key.")
        
        # Initialize OpenAI client
        try:
            from openai import OpenAI
            if self.api_base:
                self.client = OpenAI(api_key=self.api_key, base_url=self.api_base)
            else:
                self.client = OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("openai package is required. Install with: pip install openai")
        
        self.logger.info(f"LLMClient initialized: model={model_name}, multimodal={is_multimodal}")
    
    def chat(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
        message_history: Optional[List[Dict]] = None
    ) -> str:
        """
        Send a text-only chat request to the LLM.
        
        Args:
            user_message: The user's message
            system_prompt: Optional system prompt
            message_history: Optional list of previous messages
            
        Returns:
            The model's response text
        """
        messages = []
        
        # Add system prompt if provided
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Add message history if provided
        if message_history:
            messages.extend(message_history)
        
        # Add current user message
        messages.append({"role": "user", "content": user_message})
        
        return self._call_api(messages)
    
    def chat_with_images(
        self,
        user_message: str,
        image_paths: List[str],
        system_prompt: Optional[str] = None,
        message_history: Optional[List[Dict]] = None
    ) -> str:
        """
        Send a multimodal chat request with images to the LLM.
        
        Args:
            user_message: The user's message
            image_paths: List of paths to image files
            system_prompt: Optional system prompt
            message_history: Optional list of previous messages
            
        Returns:
            The model's response text
            
        Raises:
            ValueError: If is_multimodal is False
        """
        if not self.is_multimodal:
            raise ValueError("Multimodal mode is disabled. Set is_multimodal=True to use images.")
        
        messages = []
        
        # Add system prompt if provided
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Add message history if provided
        if message_history:
            messages.extend(message_history)
        
        # Build content with text and images
        content = [{"type": "text", "text": user_message}]
        
        for image_path in image_paths:
            image_url = local_image_to_data_url(image_path)
            content.append({
                "type": "image_url",
                "image_url": {"url": image_url}
            })
        
        messages.append({"role": "user", "content": content})
        
        return self._call_api(messages)
    
    def _call_api(self, messages: List[Dict]) -> str:
        """
        Make the actual API call.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            The model's response text
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            self.logger.error(f"LLM API call failed: {e}")
            raise
    
    def set_model(self, model_name: str, is_multimodal: Optional[bool] = None):
        """
        Change the model.
        
        Args:
            model_name: New model name
            is_multimodal: Whether the new model supports vision (optional)
        """
        self.model_name = model_name
        if is_multimodal is not None:
            self.is_multimodal = is_multimodal
        self.logger.info(f"Model changed to: {model_name}")


class MultimodalLLMClient(LLMClient):
    """
    LLM client specifically for multimodal (vision) tasks.
    Convenience wrapper with is_multimodal=True by default.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        model_name: str = "gpt-4o",
        temperature: float = 0,
        max_tokens: int = 1024
    ):
        """Initialize multimodal client."""
        super().__init__(
            api_key=api_key,
            api_base=api_base,
            model_name=model_name,
            is_multimodal=True,
            temperature=temperature,
            max_tokens=max_tokens
        )


class TextOnlyLLMClient(LLMClient):
    """
    LLM client specifically for text-only tasks.
    Convenience wrapper with is_multimodal=False by default.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0,
        max_tokens: int = 1024
    ):
        """Initialize text-only client."""
        super().__init__(
            api_key=api_key,
            api_base=api_base,
            model_name=model_name,
            is_multimodal=False,
            temperature=temperature,
            max_tokens=max_tokens
        )


def create_llm_client(
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    model_name: str = "gpt-4o-mini",
    is_multimodal: bool = False
) -> LLMClient:
    """
    Factory function to create an LLM client.
    
    Args:
        api_key: OpenAI API key
        api_base: OpenAI API base URL
        model_name: Model name to use
        is_multimodal: Whether to enable vision capabilities
        
    Returns:
        LLMClient instance
    """
    return LLMClient(
        api_key=api_key,
        api_base=api_base,
        model_name=model_name,
        is_multimodal=is_multimodal
    )
