"""
WorldMind Plugin Package

A modular experience learning framework for intelligent agents.

This plugin provides three independent modules:

1. ProcessExperienceModule: Extracts process experience from prediction errors
   - Input: task_instruction, trajectory (observation, action, predicted_state, env_feedback)
   - Output: process experience entries

2. GoalExperienceModule: Extracts goal experience from successful trajectories  
   - Input: task_instruction, trajectory (action, env_feedback, optional observation)
   - Output: goal experience

3. ExperienceRetrievalModule: Retrieves and optionally refines experiences
   - Input: task_instruction
   - Output: retrieved experiences

Usage:
    from worldmind_plugin import (
        WorldMindConfig,
        ProcessExperienceModule,
        GoalExperienceModule,
        ExperienceRetrievalModule,
        ProcessTrajectoryStep,
        GoalTrajectoryStep
    )
    
    # Create configuration
    config = WorldMindConfig(
        api_key="your-api-key",
        save_path="./output"
    )
    
    # Use modules independently
    process_module = ProcessExperienceModule(config)
    goal_module = GoalExperienceModule(config)
    retrieval_module = ExperienceRetrievalModule(config)
"""

__version__ = "2.0.0"
__author__ = "WorldMind Team"

# Configuration
from worldmind_plugin.config import WorldMindConfig

# Core modules
from worldmind_plugin.core import (
    # Data classes for trajectory
    ProcessTrajectoryStep,
    GoalTrajectoryStep,
    ProcessExperienceEntry,
    GoalExperienceEntry,
    
    # Three main modules
    ProcessExperienceModule,
    GoalExperienceModule,
    ExperienceRetrievalModule,
    
    # Convenience wrapper
    WorldMind,
    
    # Factory functions
    create_process_module,
    create_goal_module,
    create_retrieval_module,
    create_worldmind
)

# Prompts (for customization)
from worldmind_plugin.prompts import (
    EXPLORATION_PHASE_MARKER,
    PREDICTION_CONSTRAINT_PROMPT,
    DISCRIMINATOR_SYSTEM_PROMPT,
    DISCRIMINATOR_USER_PROMPT,
    REFLECTOR_SYSTEM_PROMPT,
    REFLECTOR_USER_PROMPT,
    STATE_SUMMARIZER_SYSTEM_PROMPT,
    STATE_SUMMARIZER_USER_PROMPT,
    GOAL_EXPERIENCE_EXTRACTION_PROMPT
)

# Utilities
from worldmind_plugin.utils import (
    parse_llm_output,
    is_exploration_phase
)

# LLM Client
from worldmind_plugin.llm_client import LLMClient

__all__ = [
    # Version
    "__version__",
    
    # Configuration
    "WorldMindConfig",
    
    # Data classes
    "ProcessTrajectoryStep",
    "GoalTrajectoryStep", 
    "ProcessExperienceEntry",
    "GoalExperienceEntry",
    
    # Main modules
    "ProcessExperienceModule",
    "GoalExperienceModule",
    "ExperienceRetrievalModule",
    
    # Convenience wrapper
    "WorldMind",
    
    # Factory functions
    "create_process_module",
    "create_goal_module",
    "create_retrieval_module",
    "create_worldmind",
    
    # Prompts
    "EXPLORATION_PHASE_MARKER",
    "PREDICTION_CONSTRAINT_PROMPT",
    
    # Utilities
    "parse_llm_output",
    "is_exploration_phase",
    
    # LLM Client
    "LLMClient"
]
