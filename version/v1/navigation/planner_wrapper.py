"""
WorldMind Planner Wrapper for Navigation Environment

Core component that wraps navigation planners with WorldMind capabilities:
- Goal experience management (learning from successful navigation)
- Process experience management (learning from failures)
- LLM-based discrimination for experience extraction
- Experience refinement for better knowledge retrieval
"""

import os
import json
import numpy as np
from typing import Dict, Optional, Any, List

from embodiedbench.main import logger

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


class WorldMindNavigationPlannerWrapper:
    """WorldMind wrapper for navigation planners with goal and process experience management."""
    
    def __init__(
        self,
        base_planner: Any,
        discriminator_model: str = None,
        reflector_model: str = None,
        log_path: str = '',
        enable_discrimination: bool = True,
        use_vision_discriminator: bool = False,
        use_experience_trajectory: bool = False,
        detailed_output: bool = False,
        
        # Goal experience settings
        enable_goal_experience: bool = True,
        goal_experience_top_k: int = 3,
        goal_experience_mode: str = 'eval',
        goal_experience_exp_file: str = '',
        
        # Process experience settings
        enable_process_experience: bool = True,
        enable_process_experience_accumulation: bool = True,
        process_experience_top_k: int = 3,
        process_experience_exp_file: str = '',
        
        # Experience refinement settings
        enable_experience_refine: bool = False,
        experience_refiner_model: str = 'gpt-4o-mini',
        
        **kwargs
    ):
        """Initialize the WorldMind planner wrapper."""
        self.base_planner = base_planner
        self.discriminator_model = discriminator_model
        self.reflector_model = reflector_model
        self.log_path = log_path
        self.enable_discrimination = enable_discrimination
        self.use_vision_discriminator = use_vision_discriminator
        self.use_experience_trajectory = use_experience_trajectory
        self.detailed_output = detailed_output
        
        # Goal experience settings
        self.enable_goal_experience = enable_goal_experience
        self.goal_experience_top_k = goal_experience_top_k
        self.goal_experience_mode = goal_experience_mode
        self.goal_experience_exp_file = goal_experience_exp_file
        
        # Process experience settings
        self.enable_process_experience = enable_process_experience
        self.enable_process_experience_accumulation = enable_process_experience_accumulation
        self.process_experience_top_k = process_experience_top_k
        self.process_experience_exp_file = process_experience_exp_file
        
        # Experience refinement settings
        self.enable_experience_refine = enable_experience_refine
        self.experience_refiner_model = experience_refiner_model
        
        # Episode state
        self.current_instruction: Optional[str] = None
        self.current_episode_id: Optional[str] = None
        self.eval_set: Optional[str] = None
        self.episode_history = []
        self.action_count = 0
        self.trajectory_data = []
        
        # Initialize WorldMind components
        self.goal_experience_manager: Optional[GoalExperienceManager] = None
        self.process_experience_manager: Optional[ProcessExperienceManager] = None
        self.discriminator: Optional[WorldMindDiscriminator] = None
        self.reflector: Optional[WorldMindReflector] = None
        self.state_summarizer: Optional[WorldMindStateSummarizer] = None
        self.experience_refiner: Optional[ExperienceRefiner] = None
        
        self._initialize_worldmind_components()
        
        logger.info(f"WorldMind Navigation Planner Wrapper initialized")
    
    def _initialize_worldmind_components(self):
        """Initialize all WorldMind components."""
        # Initialize goal experience manager
        if self.enable_goal_experience:
            self.goal_experience_manager = create_goal_experience_manager(
                log_path=self.log_path,
                top_k=self.goal_experience_top_k
            )
            logger.info("Goal Experience Manager initialized")
        
        # Initialize process experience manager
        if self.enable_process_experience:
            self.process_experience_manager = create_process_experience_manager(
                log_path=self.log_path,
                top_k=self.process_experience_top_k
            )
            logger.info("Process Experience Manager initialized")
        
        # Initialize discriminator
        if self.enable_discrimination and self.discriminator_model:
            self.discriminator = create_discriminator(
                model_name=self.discriminator_model,
                use_vision=self.use_vision_discriminator
            )
            logger.info(f"Discriminator initialized with model: {self.discriminator_model}")
        
        # Initialize reflector
        if self.reflector_model:
            self.reflector = create_reflector(
                model_name=self.reflector_model
            )
            logger.info(f"Reflector initialized with model: {self.reflector_model}")
        
        # Initialize state summarizer
        if self.discriminator_model:
            self.state_summarizer = create_state_summarizer(
                model_name=self.discriminator_model
            )
            logger.info(f"State Summarizer initialized")
        
        # Initialize experience refiner
        if self.enable_experience_refine:
            self.experience_refiner = create_experience_refiner(
                model_name=self.experience_refiner_model
            )
            logger.info(f"Experience Refiner initialized with model: {self.experience_refiner_model}")
    
    def set_log_path(self, log_path: str):
        """Set log path for saving experiences."""
        self.log_path = log_path
        if self.goal_experience_manager:
            self.goal_experience_manager.set_log_path(log_path)
        if self.process_experience_manager:
            self.process_experience_manager.set_log_path(log_path)
    
    def set_eval_set(self, eval_set: str):
        """Set current evaluation set."""
        self.eval_set = eval_set
        if self.goal_experience_manager:
            self.goal_experience_manager.set_eval_set(eval_set)
        if self.process_experience_manager:
            self.process_experience_manager.set_eval_set(eval_set)
    
    def reset(self):
        """Reset episode state."""
        self.episode_history = []
        self.action_count = 0
        self.trajectory_data = []
        
        # Reset discriminator statistics
        if self.discriminator:
            self.discriminator.reset_statistics()
        
        # Reset state summarizer statistics
        if self.state_summarizer:
            self.state_summarizer.reset_statistics()
        
        # Reset process experience manager for new episode
        if self.process_experience_manager:
            self.process_experience_manager.reset_episode()
        
        # Reset goal experience manager trajectory
        if self.goal_experience_manager:
            self.goal_experience_manager.reset_trajectory()
        
        # Reset base planner
        if hasattr(self.base_planner, 'reset'):
            self.base_planner.reset()
        
        logger.info(f"Episode reset")
    
    def act(self, observation, user_instruction):
        """Generate action using base planner with WorldMind enhancements."""
        self.current_instruction = user_instruction
        return self.base_planner.act(observation, user_instruction)
    
    def update_info(self, info):
        """Update episode info."""
        if hasattr(self.base_planner, 'update_info'):
            self.base_planner.update_info(info)
    
    def summarize_states(
        self,
        before_image_path: str,
        after_image_path: str,
        action_description: str
    ) -> Dict:
        """Summarize states before and after action."""
        if not self.state_summarizer:
            return {
                'state_before_action': '',
                'state_after_action': ''
            }
        
        try:
            state_before = self.state_summarizer.summarize_state(before_image_path)
            state_after = self.state_summarizer.summarize_state(after_image_path)
            
            return {
                'state_before_action': state_before,
                'state_after_action': state_after
            }
        except Exception as e:
            logger.error(f"State summarization failed: {e}")
            return {
                'state_before_action': '',
                'state_after_action': ''
            }
    
    def discriminate_prediction(
        self,
        predicted_state: str,
        actual_state_summary: str,
        action_index: int,
        action_id: int,
        action_description: str
    ) -> Dict:
        """Compare predicted state with actual state."""
        if not self.discriminator:
            return {'match': True}
        
        try:
            result = self.discriminator.discriminate(
                predicted_state=predicted_state,
                actual_state_summary=actual_state_summary,
                action_description=action_description
            )
            return result
        except Exception as e:
            logger.error(f"Discrimination failed: {e}")
            return {'match': True, 'error': str(e)}
    
    def trigger_reflection(
        self,
        discrimination_result: Dict,
        state_before_action: str,
        state_after_action: str,
        observation_path: str,
        env_feedback: str,
        action_id: int,
        action_description: str,
        human_instruction: str = None
    ) -> Dict:
        """Trigger reflection when prediction mismatch detected."""
        if not self.reflector:
            return None
        
        try:
            # Call reflect with correct parameter names
            reflection = self.reflector.reflect(
                action_description=action_description,
                predicted_state=discrimination_result.get('predicted_state', ''),
                state_before_action=state_before_action,
                state_after_action=state_after_action,
                env_feedback=env_feedback,
                action_id=action_id,
                human_instruction=human_instruction or self.current_instruction
            )
            
            # Store experience if valid
            if reflection and reflection.get('experience_entry'):
                experience_entries = reflection['experience_entry']
                
                if self.process_experience_manager and self.enable_process_experience_accumulation:
                    # Use add_entries method
                    entries = experience_entries if isinstance(experience_entries, list) else [experience_entries]
                    self.process_experience_manager.add_entries(
                        entries=entries,
                        instruction=human_instruction or self.current_instruction
                    )
                
                return {
                    'experience_entry': experience_entries,
                    'reflection': reflection
                }
            
            return reflection
            
        except Exception as e:
            logger.error(f"Reflection failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_trajectory_for_save(self) -> List[Dict]:
        """Get trajectory data for saving."""
        return self.trajectory_data
    
    def get_statistics(self) -> Dict:
        """Get WorldMind statistics."""
        stats = {
            'enable_discrimination': self.enable_discrimination,
            'current_episode_steps': self.action_count
        }
        
        if self.discriminator:
            stats['discriminator'] = self.discriminator.get_statistics()
        
        if self.experience_refiner:
            stats['experience_refiner'] = self.experience_refiner.get_statistics()
        
        return stats
    
    @property
    def planner_steps(self):
        """Get planner steps from base planner."""
        return getattr(self.base_planner, 'planner_steps', 0)
    
    @property
    def output_json_error(self):
        """Get output json error from base planner."""
        return getattr(self.base_planner, 'output_json_error', 0)
    
    def __getattr__(self, name):
        """Delegate attribute access to base planner."""
        return getattr(self.base_planner, name)


def create_worldmind_wrapper(
    base_planner: Any,
    config: Dict = None,
    **kwargs
) -> WorldMindNavigationPlannerWrapper:
    """Factory function to create a WorldMind Navigation Planner Wrapper."""
    if config:
        wrapper_kwargs = {**config, **kwargs}
    else:
        wrapper_kwargs = kwargs
    
    return WorldMindNavigationPlannerWrapper(base_planner=base_planner, **wrapper_kwargs)
