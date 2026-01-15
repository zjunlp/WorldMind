"""
WorldMind Navigation Module for EmbodiedBench

This module provides WorldMind capabilities for navigation tasks:
- Goal experience management (learning from successful navigation)
- Process experience management (learning from failures)
- LLM-based discrimination and reflection
- Experience refinement for better knowledge retrieval
"""

from .knowledge_manager import (
    GoalExperienceManager,
    ProcessExperienceManager,
    create_goal_experience_manager,
    create_process_experience_manager
)
from .discriminator import WorldMindDiscriminator, create_discriminator
from .reflector import WorldMindReflector, create_reflector
from .state_summarizer import WorldMindStateSummarizer, create_state_summarizer
from .experience_refiner import ExperienceRefiner, create_experience_refiner
from .planner_wrapper import WorldMindNavigationPlannerWrapper
from .prompts import WORLDMIND_NAVIGATION_SYSTEM_PROMPT
from .color_print import (
    Colors,
    print_separator,
    print_agent_input,
    print_agent,
    print_summarizer_input,
    print_summarizer_output,
    print_discriminator_input,
    print_discriminator_output,
    print_reflector_input,
    print_reflector_output,
    print_trajectory_entry,
    print_trajectory_retrieval
)

# Aliases for compatibility
Discriminator = WorldMindDiscriminator
Reflector = WorldMindReflector
StateSummarizer = WorldMindStateSummarizer


__all__ = [
    # Knowledge managers
    'GoalExperienceManager',
    'ProcessExperienceManager',
    'create_goal_experience_manager',
    'create_process_experience_manager',
    
    # Core components
    'WorldMindDiscriminator',
    'Discriminator',
    'create_discriminator',
    'WorldMindReflector',
    'Reflector',
    'create_reflector',
    'WorldMindStateSummarizer',
    'StateSummarizer',
    'create_state_summarizer',
    'ExperienceRefiner',
    'create_experience_refiner',
    
    # Planner wrapper
    'WorldMindNavigationPlannerWrapper',
    
    
    # Prompts
    'WORLDMIND_NAVIGATION_SYSTEM_PROMPT',
    
    # Color printing utilities
    'Colors',
    'print_separator',
    'print_agent_input',
    'print_agent',
    'print_summarizer_input',
    'print_summarizer_output',
    'print_discriminator_input',
    'print_discriminator_output',
    'print_reflector_input',
    'print_reflector_output',
    'print_trajectory_entry',
    'print_trajectory_retrieval',
]
