"""
WorldMind Plugin Configuration Module

This module provides the configuration class for WorldMind plugin.
All parameters can be configured through the WorldMindConfig dataclass.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class WorldMindConfig:
    """
    Configuration class for WorldMind plugin.
    
    This class contains all configurable parameters for the WorldMind plugin.
    Users should create an instance of this class and pass it to the modules.
    
    Attributes:
        # API Configuration
        api_key: OpenAI API key or compatible API key
        api_base: API base URL (for OpenAI-compatible APIs)
        
        # Multimodal Configuration
        is_multimodal: Whether to use vision capabilities for state summarization
        
        # Model Configuration (per module)
        discriminator_model: Model for discrimination (comparing predictions)
        reflector_model: Model for reflection (analyzing errors)
        summarizer_model: Model for state summarization (vision model if multimodal)
        extractor_model: Model for goal experience extraction
        refiner_model: Model for experience refinement
        
        # Experience Configuration
        enable_experience_refine: Whether to refine/consolidate experiences
        goal_experience_top_k: Number of goal experiences to retrieve
        process_experience_top_k: Number of process experiences to retrieve
        goal_trajectory_include_observation: Whether to include observation in goal trajectory
        
        # Feedback Configuration
        use_env_feedback: Whether to use environment feedback in reflection
        
        # Save Configuration
        save_path: Base path for saving experiences and trajectories
        
        # Output Configuration
        detailed_output: Whether to output detailed logs
    """
    
    # API Configuration
    api_key: str = ""
    api_base: Optional[str] = None
    
    # Multimodal Configuration
    is_multimodal: bool = False
    
    # Model Configuration
    discriminator_model: str = "gpt-4o-mini"
    reflector_model: str = "gpt-4o-mini"
    summarizer_model: str = "gpt-4o"  # Vision model for multimodal
    extractor_model: str = "gpt-4o-mini"
    refiner_model: str = "gpt-4o-mini"
    
    # Experience Configuration
    enable_experience_refine: bool = True
    goal_experience_top_k: int = 3
    process_experience_top_k: int = 5
    goal_trajectory_include_observation: bool = True
    
    # Feedback Configuration
    use_env_feedback: bool = True
    
    # Save Configuration
    save_path: str = "./worldmind_output"
    
    # Output Configuration
    detailed_output: bool = True
    
    def validate(self):
        """
        Validate configuration parameters.
        
        Raises:
            ValueError: If required parameters are missing or invalid
        """
        if not self.api_key:
            raise ValueError("api_key is required")
        
        if self.goal_experience_top_k < 1:
            raise ValueError("goal_experience_top_k must be >= 1")
        
        if self.process_experience_top_k < 1:
            raise ValueError("process_experience_top_k must be >= 1")
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            "api_key": "***",  # Hide API key
            "api_base": self.api_base,
            "is_multimodal": self.is_multimodal,
            "discriminator_model": self.discriminator_model,
            "reflector_model": self.reflector_model,
            "summarizer_model": self.summarizer_model,
            "extractor_model": self.extractor_model,
            "refiner_model": self.refiner_model,
            "enable_experience_refine": self.enable_experience_refine,
            "goal_experience_top_k": self.goal_experience_top_k,
            "process_experience_top_k": self.process_experience_top_k,
            "goal_trajectory_include_observation": self.goal_trajectory_include_observation,
            "use_env_feedback": self.use_env_feedback,
            "save_path": self.save_path,
            "detailed_output": self.detailed_output
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "WorldMindConfig":
        """Create configuration from dictionary."""
        return cls(**{k: v for k, v in data.items() if k != "api_key" or v != "***"})
