"""
WorldMind Navigation Planner Wrapper

Wraps base navigation planner with WorldMind capabilities:
- State prediction and discrimination
- Reflection and experience learning
- Goal experience retrieval
- Process experience management
- Experience refinement

Parameters aligned with Alfred's planner_wrapper.
"""

import json
import os
from typing import Dict, List, Optional, Tuple, Any

from embodiedbench.planner.nav_planner import EBNavigationPlanner
from embodiedbench.worldmind.navigation.discriminator import WorldMindDiscriminator
from embodiedbench.worldmind.navigation.reflector import WorldMindReflector
from embodiedbench.worldmind.navigation.state_summarizer import WorldMindStateSummarizer
from embodiedbench.worldmind.navigation.knowledge_manager import (
    GoalExperienceManager,
    ProcessExperienceManager
)
from embodiedbench.worldmind.navigation.color_print import (
    print_agent_input,
    print_agent,
    print_summarizer_input,
    print_summarizer_output,
    print_discriminator_input,
    print_discriminator_output,
    print_reflector_input,
    print_reflector_output,
    print_trajectory_entry,
    print_separator
)
from embodiedbench.worldmind.navigation.prompts import (
    parse_navigation_response,
    extract_predicted_states,
    extract_action_ids,
    get_worldmind_first_prompt,
    get_worldmind_following_prompt
)
from embodiedbench.main import logger


class ExperienceTrajectory:
    """Experience trajectory manager for recording step information."""
    
    def __init__(self):
        self.trajectory: List[Dict] = []
        
    def add_entry(
        self, 
        step: int, 
        action_id: int, 
        action_description: str,
        observation_before: str, 
        predicted_state: str,
        env_feedback: str = "", 
        agent_input_prompt: Optional[str] = None,
        agent_output: Optional[str] = None,
        # WorldMind related info
        discriminator_called: bool = False,
        discrimination_result: Optional[Dict] = None,
        has_prediction_error: bool = False,
        reflector_called: bool = False,
        reflection_result: Optional[Dict] = None,
        experience_entries: Optional[List[str]] = None
    ):
        """Add trajectory entry."""
        entry = {
            "step": step,
            "action_id": action_id,
            "action_description": action_description,
            "observation_before": observation_before,
            "predicted_state": predicted_state,
            "env_feedback": env_feedback,
            "agent_input_prompt": agent_input_prompt,
            "agent_output": agent_output,
            # WorldMind related
            "discriminator_called": discriminator_called,
            "discrimination_result": discrimination_result,
            "has_prediction_error": has_prediction_error,
            "reflector_called": reflector_called,
            "reflection_result": reflection_result,
            "experience_entries": experience_entries
        }
        self.trajectory.append(entry)
        
    def get_trajectory_for_save(self) -> List[Dict]:
        """Get trajectory for saving."""
        return self.trajectory.copy()
        
    def reset(self):
        """Reset trajectory."""
        self.trajectory = []
        
    def __len__(self):
        return len(self.trajectory)


class WorldMindNavigationPlannerWrapper:
    """
    WorldMind navigation planner wrapper.
    
    Core features:
    1. Extract predicted_state from model output
    2. Use StateSummarizer to generate state descriptions
    3. Use Discriminator to verify predictions
    4. Trigger Reflector on prediction errors
    5. Goal experience retrieval and management
    6. Process experience retrieval and management
    7. Experience refinement
    
    Parameters aligned with Alfred's WorldMindPlannerWrapper.
    """
    
    def __init__(
        self,
        base_planner: EBNavigationPlanner,
        # Model configuration
        discriminator_model: str = None,
        reflector_model: str = None,
        summarizer_model: str = None,
        # Basic feature switches
        enable_discrimination: bool = True,
        use_experience_trajectory: bool = False,
        use_state_summarizer: bool = True,
        detailed_output: bool = False,
        use_vision_discriminator: bool = True,
        worldmind_first_action_only: bool = True,
        # Goal experience parameters
        enable_goal_experience: bool = True,
        goal_experience_top_k: int = 3,
        # Process experience parameters
        enable_process_experience: bool = True,
        process_experience_top_k: int = 3,
        # Experience refinement parameters
        enable_experience_refine: bool = False,
        # Log path
        log_path: Optional[str] = None
    ):
        """Initialize WorldMind wrapper."""
        self.base_planner = base_planner
        self.log_path = log_path
        self.detailed_output = detailed_output
        
        # Feature switches
        self.enable_discrimination = enable_discrimination
        self.use_experience_trajectory = use_experience_trajectory
        self.use_state_summarizer = use_state_summarizer
        self.use_vision_discriminator = use_vision_discriminator
        self.worldmind_first_action_only = worldmind_first_action_only
        
        # Goal experience configuration
        self.enable_goal_experience = enable_goal_experience
        self.goal_experience_top_k = goal_experience_top_k
        
        # Process experience configuration
        self.enable_process_experience = enable_process_experience
        self.process_experience_top_k = process_experience_top_k
        
        # Experience refinement configuration
        self.enable_experience_refine = enable_experience_refine
        
        # Current eval_set
        self.current_eval_set = None
        
        # Initialize components
        if use_state_summarizer:
            self.state_summarizer = WorldMindStateSummarizer(
                model_name=summarizer_model or discriminator_model
            )
        else:
            self.state_summarizer = None
            
        if enable_discrimination:
            self.discriminator = WorldMindDiscriminator(
                model_name=discriminator_model,
                use_vision=use_vision_discriminator
            )
        else:
            self.discriminator = None
            
        self.reflector = WorldMindReflector(
            model_name=reflector_model or discriminator_model
        )
            
        # Goal experience manager
        if enable_goal_experience:
            self.goal_experience_manager = GoalExperienceManager(
                log_path=log_path,
                eval_set=None,
                top_k=goal_experience_top_k,
                extractor_model=discriminator_model
            )
        else:
            self.goal_experience_manager = None
            
        # Process experience manager
        if enable_process_experience:
            self.process_experience_manager = ProcessExperienceManager(
                log_path=log_path,
                eval_set=None,
                top_k=process_experience_top_k
            )
        else:
            self.process_experience_manager = None
        
        # Experience refiner (if enabled)
        self.experience_refiner = None
        if enable_experience_refine:
            try:
                from embodiedbench.worldmind.navigation.experience_refiner import ExperienceRefiner
                refiner_model = base_planner.model_name if hasattr(base_planner, 'model_name') else discriminator_model
                self.experience_refiner = ExperienceRefiner(
                    model_name=refiner_model,
                    model_type=base_planner.model_type if hasattr(base_planner, 'model_type') else 'remote'
                )
                logger.info(f"Experience refiner enabled with model: {refiner_model}")
            except Exception as e:
                logger.warning(f"Failed to initialize experience refiner: {e}")
                self.experience_refiner = None
            
        # Experience trajectory
        self.experience_trajectory = ExperienceTrajectory()
        
        # Current state
        self.current_predicted_state: Optional[str] = None
        self.current_predicted_states: List[str] = []
        self.current_action_id: Optional[int] = None
        self.current_action_description: Optional[str] = None
        self.current_obs_path: Optional[str] = None
        self.current_agent_input_prompt: Optional[str] = None
        self.current_agent_output: Optional[str] = None
        self.current_worldmind_feedback: Dict = {}
        self.current_human_instruction: Optional[str] = None
        
        # History
        self.discrimination_history: List[Dict] = []
        self.reflection_history: List[Dict] = []
        
        # Statistics
        self.stats = {
            "total_predictions": 0,
            "discrimination_matches": 0,
            "discrimination_mismatches": 0,
            "reflections_triggered": 0
        }
        
        logger.info(f"WorldMind Navigation Wrapper initialized")
        logger.info(f"  - enable_discrimination: {enable_discrimination}")
        logger.info(f"  - use_experience_trajectory: {use_experience_trajectory}")
        logger.info(f"  - use_vision_discriminator: {use_vision_discriminator}")
        logger.info(f"  - enable_goal_experience: {enable_goal_experience}")
        logger.info(f"  - enable_process_experience: {enable_process_experience}")
        logger.info(f"  - enable_experience_refine: {enable_experience_refine}")
        
    def reset(self):
        """Reset state."""
        self.base_planner.reset()
        self.experience_trajectory.reset()
        
        if self.goal_experience_manager:
            self.goal_experience_manager.reset_trajectory()
        
        if self.process_experience_manager:
            self.process_experience_manager.reset_episode()
        
        self.current_predicted_state = None
        self.current_predicted_states = []
        self.current_action_id = None
        self.current_action_description = None
        self.current_obs_path = None
        self.current_agent_input_prompt = None
        self.current_agent_output = None
        self.current_worldmind_feedback = {}
        self.current_human_instruction = None
        
        self.discrimination_history = []
        self.reflection_history = []
        
        if self.state_summarizer:
            self.state_summarizer.reset_statistics()
        if self.discriminator:
            self.discriminator.reset_statistics()
        if self.reflector:
            self.reflector.reset_statistics()

    def _build_worldmind_prompt(self, user_instruction: str) -> str:
        """
        Build WorldMind-specific prompt with goal experience retrieval.
        
        If experience refinement is enabled:
        1. Retrieve goal experiences
        2. Call refiner to consolidate experiences and generate initial plan
        3. Add consolidated experience and plan to prompt
        
        If refinement is not enabled, add retrieved experiences directly.
        Aligned with Alfred's _build_worldmind_prompt implementation.
        """
        user_instruction = user_instruction.rstrip('.')
        planner = self.base_planner
        
        # Base prompt
        if planner.n_shot >= 1:
            prompt = planner.system_prompt.format(
                len(planner.actions)-1, 
                planner.available_action_str, 
                '\n\n'.join([f'## Task Execution Example {i}: \n {x}' for i, x in enumerate(planner.examples[:planner.n_shot])])
            )
        else:
            prompt = planner.system_prompt.format(len(planner.actions)-1, planner.available_action_str, '')
        
        # Retrieve goal experiences
        relevant_experiences = []
        if self.enable_goal_experience and self.goal_experience_manager:
            relevant_experiences = self.goal_experience_manager.retrieve_similar(user_instruction, self.goal_experience_top_k)
        
        prompt += f'\n\n## Now the human instruction is: {user_instruction}.'
        
        # If experience refinement is enabled, use refiner to consolidate
        if self.enable_experience_refine and self.experience_refiner and relevant_experiences:
            try:
                # Also retrieve process experiences for refinement
                process_experiences = []
                if self.enable_process_experience and self.process_experience_manager:
                    process_exp = self.process_experience_manager.retrieve_experience(user_instruction)
                    process_experiences = process_exp if process_exp else []
                
                refine_result = self.experience_refiner.refine_for_task(
                    current_instruction=user_instruction,
                    goal_experiences=relevant_experiences,
                    process_experiences=process_experiences
                )
                
                refined_prompt = self.experience_refiner.format_for_prompt(refine_result)
                if refined_prompt:
                    prompt += f'\n\n{refined_prompt}'
                    logger.info(f"Experience refiner: added consolidated experience and initial plan to prompt")
            except Exception as e:
                logger.warning(f"Experience refiner failed: {e}, falling back to direct experience addition")
                if relevant_experiences:
                    experience_prompt = self._format_experiences_for_prompt(relevant_experiences)
                    prompt += f'\n\n{experience_prompt}'

        else:
            # No refinement, add experiences directly
            if relevant_experiences:
                experience_prompt = self._format_experiences_for_prompt(relevant_experiences)
                prompt += f'\n\n{experience_prompt}'

        # Add action history
        if len(planner.episode_act_feedback) > 0:
            prompt += '\n The action history:\n'
            for i, action_feedback in enumerate(planner.episode_act_feedback):
                prompt += '\nStep {}, action id {}, {}, env feedback: {}'.format(
                    i, action_feedback[0], planner.actions[action_feedback[0]], action_feedback[1]
                )
            
            # Use WorldMind-specific following_prompt (includes predicted_state)
            worldmind_following = get_worldmind_following_prompt(
                num_actions=len(planner.actions),
                language_only=getattr(planner, 'language_only', False),
                multistep=getattr(planner, 'multistep', False),
                multiview=getattr(planner, 'multiview', False)
            )
            prompt += f"\n\n{worldmind_following}"
        else:
            # Use WorldMind-specific first_prompt (includes predicted_state)
            worldmind_first = get_worldmind_first_prompt(
                num_actions=len(planner.actions),
                language_only=getattr(planner, 'language_only', False),
                multistep=getattr(planner, 'multistep', False),
                multiview=getattr(planner, 'multiview', False)
            )
            prompt += worldmind_first
        
        return prompt
    
    def _format_experiences_for_prompt(self, experiences: List[Dict]) -> str:
        """Format goal experiences for prompt."""
        if not experiences:
            return ""
        
        lines = ["## Relevant Goal Experiences:"]
        for i, exp in enumerate(experiences, 1):
            if isinstance(exp, dict):
                instruction = exp.get('instruction', '')
                experience = exp.get('goal_experience', '')
                lines.append(f"{i}. Task: {instruction}")
                lines.append(f"   Experience: {experience}")
            else:
                lines.append(f"{i}. {exp}")
        
        return "\n".join(lines)
    
    def _format_process_experience_for_prompt(self, experience: List[str]) -> str:
        """Format process experience for prompt."""
        if not experience:
            return ""
        
        lines = ["## Relevant Process Experience:"]
        for i, k in enumerate(experience, 1):
            lines.append(f"{i}. {k}")
        
        return "\n".join(lines)

    def act(self, observation, user_instruction: str) -> Tuple[Any, str]:
        """
        Execute planning.
        
        Args:
            observation: Current observation (image path or dict)
            user_instruction: Task instruction
            
        Returns:
            (action list or single action, raw model output)
        """
        # Save current human instruction
        self.current_human_instruction = user_instruction
        
        # Save current observation path
        if isinstance(observation, str):
            self.current_obs_path = observation
        elif isinstance(observation, dict):
            self.current_obs_path = observation.get('head_rgb', None)
            
        # Build WorldMind-specific prompt (with goal experience retrieval)
        worldmind_prompt = self._build_worldmind_prompt(user_instruction)
        
        # Save current agent input prompt
        self.current_agent_input_prompt = worldmind_prompt
        
        # Print Agent input (automatically extracts user-related parts)
        print_agent_input(worldmind_prompt, self.detailed_output)
        
        # Call base planner
        actions, output_text = self.base_planner.act(observation, user_instruction)
        
        # Save output
        self.current_agent_output = output_text
        
        # Print Agent output
        print_agent(output_text, self.detailed_output)
        
        # Parse response to extract predicted states
        response_dict = parse_navigation_response(output_text)
        self.current_predicted_states = extract_predicted_states(response_dict)
        
        # Set current action info
        if isinstance(actions, list) and len(actions) > 0:
            self.current_action_id = actions[0]
            self.current_predicted_state = self.current_predicted_states[0] if self.current_predicted_states else None
            if 0 <= self.current_action_id < len(self.base_planner.actions):
                self.current_action_description = self.base_planner.actions[self.current_action_id]
            else:
                self.current_action_description = f"action_{self.current_action_id}"
        elif isinstance(actions, int) and actions >= 0:
            self.current_action_id = actions
            self.current_predicted_state = self.current_predicted_states[0] if self.current_predicted_states else None
            if self.current_action_id < len(self.base_planner.actions):
                self.current_action_description = self.base_planner.actions[self.current_action_id]
            else:
                self.current_action_description = f"action_{self.current_action_id}"
                
        if self.current_predicted_state:
            self.stats["total_predictions"] += 1
            
        return actions, output_text
        
    def summarize_state(self, image_path: str, obs: Any = None) -> str:
        """Summarize current state (single image)."""
        if self.state_summarizer is None:
            return ""
            
        return self.state_summarizer.summarize_state(
            image_path,
            action_context=self.current_action_description or ""
        )
    
    def summarize_states(
        self, 
        before_image_path: str, 
        after_image_path: str,
        action_description: str = None
    ) -> Dict[str, str]:
        """
        Summarize states before and after action.
        Aligned with Alfred's WorldMindPlannerWrapper.summarize_states.
        
        Args:
            before_image_path: Image path before action
            after_image_path: Image path after action
            action_description: Action description
            
        Returns:
            dict: {
                "state_before_action": str,
                "state_after_action": str
            }
        """
        if self.state_summarizer is None:
            logger.warning("[WorldMind] State summarizer not configured, returning empty state summaries")
            return {'state_before_action': '', 'state_after_action': ''}
        
        action_desc = action_description or self.current_action_description or "unknown action"
        
        print_summarizer_input("(image)", "(image)", action_desc, self.detailed_output)
        
        result = self.state_summarizer.summarize_states(
            action_description=action_desc,
            before_image_path=before_image_path,
            after_image_path=after_image_path
        )
        
        print_summarizer_output(
            result.get('state_before_action', ''),
            result.get('state_after_action', ''),
            self.detailed_output
        )
        
        return result
        
    def discriminate(self, action: Any, predicted_state: str,
                    actual_state: str, action_success: int = 1) -> Tuple[bool, str]:
        """
        Discriminate between predicted and actual state.
        
        Returns:
            (is_match, reason)
        """
        if self.discriminator is None:
            return True, "Discrimination disabled"
            
        if not predicted_state:
            return True, "No prediction available"
            
        # Print discriminator input
        print_discriminator_input(
            predicted_state,
            actual_state,
            str(action),
            self.detailed_output
        )
        
        result = self.discriminator.discriminate(
            predicted_state=predicted_state,
            actual_state_summary=actual_state,
            actual_state_image=self.current_obs_path,
            action_description=self.current_action_description or str(action)
        )
        
        is_match = result.get("match", True)
        reason = result.get("reason", "")
        
        # Print discriminator output
        print_discriminator_output(is_match, reason, self.detailed_output)
        
        # Update statistics
        if is_match:
            self.stats["discrimination_matches"] += 1
        else:
            self.stats["discrimination_mismatches"] += 1
            
        # Save history
        self.discrimination_history.append({
            "action": str(action),
            "predicted_state": predicted_state,
            "actual_state": actual_state,
            "match": is_match,
            "reason": reason
        })
        
        return is_match, reason
    
    def discriminate_prediction(
        self, 
        predicted_state: str,
        actual_state_summary: str, 
        action_index: int = 0,
        action_id: int = None,
        action_description: str = None
    ) -> Dict:
        """
        Discriminate between predicted and actual state (Alfred-compatible interface).
        
        Args:
            predicted_state: Model's predicted state
            actual_state_summary: State summarizer's output
            action_index: Action index
            action_id: Action ID
            action_description: Action description
            
        Returns:
            dict: Discrimination result with 'match', 'reason' fields
        """
        if self.discriminator is None:
            return {"match": True, "reason": "Discrimination disabled"}
        
        if not predicted_state:
            return {"match": True, "reason": "No prediction available"}
        
        action_desc = action_description or self.current_action_description or f"action_{action_id}"
        
        # Print discriminator input
        print_discriminator_input(
            predicted_state,
            actual_state_summary,
            action_desc,
            self.detailed_output
        )
        
        # Save current predicted state (for reflection)
        self.current_predicted_state = predicted_state
        self.current_action_id = action_id
        self.current_action_description = action_desc
        
        result = self.discriminator.discriminate(
            predicted_state=predicted_state,
            actual_state_summary=actual_state_summary,
            actual_state_image=self.current_obs_path,
            action_description=action_desc
        )
        
        is_match = result.get("match", True)
        reason = result.get("reason", "")
        
        # Print discriminator output
        print_discriminator_output(is_match, reason, self.detailed_output)
        
        # Update statistics
        if is_match:
            self.stats["discrimination_matches"] += 1
        else:
            self.stats["discrimination_mismatches"] += 1
        
        # Save history
        self.discrimination_history.append({
            "action_index": action_index,
            "action_id": action_id,
            "action": action_desc,
            "predicted_state": predicted_state,
            "actual_state": actual_state_summary,
            "match": is_match,
            "reason": reason
        })
        
        return result
    
    def reflect(self, action: Any, predicted_state: str,
               actual_state: str, explanation: str = "") -> Optional[Dict]:
        """
        Generate reflection.
        
        Returns:
            Reflection result dict or None
        """
        if self.reflector is None:
            return None
            
        # Print reflector input
        print_reflector_input(
            self.current_action_description or str(action),
            predicted_state,
            "",  # state_before
            actual_state,
            explanation,
            self.detailed_output
        )
        
        # Build experience trajectory string
        trajectory_parts = []
        for fb in self.base_planner.episode_act_feedback:
            if len(fb) >= 2:
                trajectory_parts.append(f"Action: {self.base_planner.actions[fb[0]]}, Feedback: {fb[1]}")
        experience_trajectory_str = "\n".join(trajectory_parts) if trajectory_parts else "No previous experience."
        
        result = self.reflector.reflect(
            action_description=self.current_action_description or str(action),
            predicted_state=predicted_state,
            state_before_action="",
            state_after_action=actual_state,
            action_id=self.current_action_id,
            env_feedback=explanation,
            experience_trajectory=experience_trajectory_str,
            human_instruction=self.current_human_instruction
        )
        
        reflexion = result.get("reflexion", "")
        
        # Print reflector output
        print_reflector_output(reflexion, self.detailed_output)
        
        # Update statistics
        self.stats["reflections_triggered"] += 1
        
        # Save history
        self.reflection_history.append(result)
        
        # If process experience is enabled, add to manager
        if self.enable_process_experience and self.process_experience_manager and reflexion:
            self.process_experience_manager.add_entries([reflexion])
        
        return result
    
    def trigger_reflection(
        self, 
        discrimination_result: Dict,
        state_before_action: str,
        state_after_action: str,
        observation_path: Optional[str] = None,
        env_feedback: str = "",
        action_id: int = None,
        action_description: str = None,
        human_instruction: str = None
    ) -> Optional[Dict]:
        """
        Trigger reflection processing (Alfred-compatible interface).
        
        Only called on prediction mismatch, generates experience entries.
        
        Args:
            discrimination_result: Discrimination result
            state_before_action: State before action
            state_after_action: State after action
            observation_path: Observation image path
            env_feedback: Environment feedback
            action_id: Action ID
            action_description: Action description
            human_instruction: Human instruction
            
        Returns:
            dict: Reflection result with 'reflexion', 'experience_entry' fields
        """
        if self.reflector is None:
            return None
        
        # Check if reflection is needed (on mismatch)
        is_match = discrimination_result.get("match", True)
        if is_match:
            return None
        
        action_desc = action_description or self.current_action_description or f"action_{action_id}"
        predicted_state = discrimination_result.get("predicted_state", self.current_predicted_state or "")
        reason = discrimination_result.get("reason", "")
        
        # Print reflector input
        print_reflector_input(
            action_desc,
            predicted_state,
            state_before_action,
            state_after_action,
            reason,
            self.detailed_output
        )
        
        # Build experience trajectory string
        trajectory_parts = []
        for fb in self.base_planner.episode_act_feedback:
            if len(fb) >= 2:
                trajectory_parts.append(f"Action: {self.base_planner.actions[fb[0]]}, Feedback: {fb[1]}")
        experience_trajectory_str = "\n".join(trajectory_parts) if trajectory_parts else "No previous experience."
        
        instruction = human_instruction or self.current_human_instruction or "Not specified."
        
        # Call reflector
        result = self.reflector.reflect(
            action_description=action_desc,
            predicted_state=predicted_state,
            state_before_action=state_before_action,
            state_after_action=state_after_action,
            action_id=action_id,
            env_feedback=env_feedback,
            experience_trajectory=experience_trajectory_str,
            human_instruction=instruction
        )
        
        reflexion = result.get("reflexion", "")
        
        # Print reflector output
        print_reflector_output(reflexion, self.detailed_output)
        
        # Update statistics
        self.stats["reflections_triggered"] += 1
        
        # Save history
        self.reflection_history.append({
            "action_id": action_id,
            "action": action_desc,
            "predicted_state": predicted_state,
            "state_before": state_before_action,
            "state_after": state_after_action,
            "reflexion": reflexion,
            "human_instruction": human_instruction
        })
        
        # Generate experience entries
        experience_entries = []
        if reflexion:
            experience_entries.append(reflexion)
            
            # If process experience is enabled, add to manager
            if self.enable_process_experience and self.process_experience_manager:
                self.process_experience_manager.add_entries([reflexion])
        
        result['experience_entry'] = experience_entries
        
        # Update WorldMind feedback (for next planning)
        self.current_worldmind_feedback = {
            "observation_before": state_before_action,
            "predicted_state": predicted_state,
            "reflexion": reflexion
        }
        
        return result
    
    def retrieve_goal_experiences(self, query: str) -> List[Dict]:
        """Retrieve relevant goal experiences."""
        if not self.enable_goal_experience or self.goal_experience_manager is None:
            return []
        return self.goal_experience_manager.retrieve_similar(query, self.goal_experience_top_k)
        
    def retrieve_process_experience(self, query: str) -> List[str]:
        """Retrieve relevant process experience."""
        if not self.enable_process_experience or self.process_experience_manager is None:
            return []
        return self.process_experience_manager.retrieve_relevant(query, self.process_experience_top_k)
        
    def add_goal_experience(self, experience: Dict):
        """Add goal experience."""
        if self.enable_goal_experience and self.goal_experience_manager:
            self.goal_experience_manager.add_experience(experience)
            
    def set_log_path(self, log_path: str):
        """Set the log path for saving experience entries."""
        self.log_path = log_path
        if self.goal_experience_manager:
            self.goal_experience_manager.set_log_path(log_path)
        if self.process_experience_manager:
            self.process_experience_manager.set_log_path(log_path)
    
    def set_eval_set(self, eval_set: str):
        """Set the current eval set for experience management."""
        self.current_eval_set = eval_set
        if self.goal_experience_manager:
            self.goal_experience_manager.set_eval_set(eval_set)
        if self.process_experience_manager:
            self.process_experience_manager.set_eval_set(eval_set)
            
    def update_info(self, info: dict):
        """Update info to base planner."""
        # Add WorldMind feedback to info
        if self.current_worldmind_feedback:
            worldmind_parts = []
            if self.current_worldmind_feedback.get("observation_before"):
                worldmind_parts.append(f"[Observation] {self.current_worldmind_feedback['observation_before']}")
            if self.current_worldmind_feedback.get("predicted_state"):
                worldmind_parts.append(f"[Predicted State] {self.current_worldmind_feedback['predicted_state']}")
            if self.current_worldmind_feedback.get("reflexion"):
                worldmind_parts.append(f"[Reflexion] {self.current_worldmind_feedback['reflexion']}")
                
            if worldmind_parts:
                original_feedback = info.get('env_feedback', '')
                worldmind_feedback = "\n".join(worldmind_parts)
                info['env_feedback'] = f"{original_feedback}\n{worldmind_feedback}" if original_feedback else worldmind_feedback
                
        self.current_worldmind_feedback = {}
        self.base_planner.update_info(info)
        
    def get_trajectory_for_save(self) -> List[Dict]:
        """Get trajectory for saving."""
        return self.experience_trajectory.get_trajectory_for_save()
        
    def save_experiences(self, path: str):
        """Save experiences to file."""
        if self.goal_experience_manager:
            self.goal_experience_manager.save_experiences(path)
            
    def save_process_experience(self, path: str):
        """Save process experience to file."""
        if self.process_experience_manager:
            self.process_experience_manager.save_experience(path)
        
    def get_worldmind_stats(self) -> Dict:
        """Get WorldMind statistics."""
        stats = self.stats.copy()
        
        if self.discriminator:
            stats["discriminator_stats"] = self.discriminator.get_statistics()
        if self.reflector:
            stats["reflector_stats"] = self.reflector.get_statistics()
        if self.state_summarizer:
            stats["summarizer_stats"] = self.state_summarizer.get_statistics()
            
        stats["discrimination_history"] = self.discrimination_history
        stats["reflection_history"] = self.reflection_history
        
        total = stats["discrimination_matches"] + stats["discrimination_mismatches"]
        stats["match_rate"] = stats["discrimination_matches"] / total if total > 0 else 0
        
        return stats
        
    # Proxy properties to base planner
    @property
    def planner_steps(self):
        return self.base_planner.planner_steps
        
    @property
    def output_json_error(self):
        return self.base_planner.output_json_error
        
    @property
    def actions(self):
        return self.base_planner.actions

