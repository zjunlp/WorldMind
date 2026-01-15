"""
WorldMind Planner Wrapper for Habitat Environment

Provides a wrapper around VLMPlanner that adds state prediction capabilities
and integrates with WorldMind discriminator, reflector, and state summarizer.
"""

import os
import json
import re
from typing import Dict, List, Optional, Tuple, Any

from embodiedbench.planner.vlm_planner import VLMPlanner
from embodiedbench.worldmind.habitat.discriminator import WorldMindDiscriminator
from embodiedbench.worldmind.habitat.reflector import WorldMindReflector
from embodiedbench.worldmind.habitat.state_summarizer import WorldMindStateSummarizer
from embodiedbench.worldmind.habitat.knowledge_manager import (
    GoalExperienceManager, 
    ProcessExperienceManager,
    create_goal_experience_manager,
    create_process_experience_manager
)
from embodiedbench.worldmind.habitat.color_print import (
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
from embodiedbench.worldmind.habitat.prompts import (
    WorldMind_TEMPLATE, 
    WorldMind_TEMPLATE_LANG,
    get_worldmind_examples,
    format_worldmind_examples,
    fix_json_worldmind,
    parse_json_worldmind
)
from embodiedbench.main import logger


class ExperienceTrajectory:
    """Experience trajectory manager for recording action history with WorldMind information."""
    
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
        discriminator_called: bool = False,
        discrimination_result: Optional[Dict] = None,
        has_prediction_error: bool = False,
        reflector_called: bool = False,
        reflection_result: Optional[Dict] = None,
        experience_entries: Optional[List[str]] = None
    ):
        """Add an entry to the experience trajectory."""
        entry = {
            "step": step,
            "action_id": action_id,
            "action_description": action_description,
            "observation_before": observation_before,
            "predicted_state": predicted_state,
            "env_feedback": env_feedback,
            "agent_input_prompt": agent_input_prompt,
            "agent_output": agent_output,
            "worldmind_info": {
                "discriminator_called": discriminator_called,
                "discrimination_result": discrimination_result,
                "has_prediction_error": has_prediction_error,
                "reflector_called": reflector_called,
                "reflection_result": reflection_result,
                "experience_entries": experience_entries or []
            }
        }
        self.trajectory.append(entry)
    
    def update_last_entry(
        self,
        env_feedback: str = None,
        agent_input_prompt: str = None,
        agent_output: str = None,
        discriminator_called: bool = None,
        discrimination_result: Dict = None,
        has_prediction_error: bool = None,
        reflector_called: bool = None,
        reflection_result: Dict = None,
        experience_entries: List[str] = None
    ):
        """Update the last entry in the trajectory."""
        if not self.trajectory:
            return
        
        if env_feedback is not None:
            self.trajectory[-1]["env_feedback"] = env_feedback
        if agent_input_prompt is not None:
            self.trajectory[-1]["agent_input_prompt"] = agent_input_prompt
        if agent_output is not None:
            self.trajectory[-1]["agent_output"] = agent_output
        
        if "worldmind_info" not in self.trajectory[-1]:
            self.trajectory[-1]["worldmind_info"] = {}
        
        if discriminator_called is not None:
            self.trajectory[-1]["worldmind_info"]["discriminator_called"] = discriminator_called
        if discrimination_result is not None:
            self.trajectory[-1]["worldmind_info"]["discrimination_result"] = discrimination_result
        if has_prediction_error is not None:
            self.trajectory[-1]["worldmind_info"]["has_prediction_error"] = has_prediction_error
        if reflector_called is not None:
            self.trajectory[-1]["worldmind_info"]["reflector_called"] = reflector_called
        if reflection_result is not None:
            self.trajectory[-1]["worldmind_info"]["reflection_result"] = reflection_result
        if experience_entries is not None:
            self.trajectory[-1]["worldmind_info"]["experience_entries"] = experience_entries
    
    def get_trajectory_for_save(self) -> List[Dict]:
        """Get trajectory data for saving to file."""
        return self.trajectory.copy()
    
    def reset(self):
        """Reset the trajectory."""
        self.trajectory = []
    
    def __len__(self):
        return len(self.trajectory)


class WorldMindPlannerWrapper:
    """Wrapper around VLMPlanner that adds Action-World Modeling capabilities."""
    
    def __init__(
        self, 
        base_planner: VLMPlanner,
        discriminator_model: str = None,
        reflector_model: str = None,
        summarizer_model: str = None,
        enable_discrimination: bool = True,
        use_experience_trajectory: bool = False,
        use_state_summarizer: bool = True,
        detailed_output: bool = False,
        use_separate_summarization: bool = False,
        worldmind_first_action_only: bool = True,
        enable_goal_experience: bool = True,
        goal_experience_top_k: int = 3,
        enable_process_experience: bool = True,
        process_experience_top_k: int = 3,
        enable_experience_refine: bool = False
    ):
        """Initialize the WorldMind Planner Wrapper."""
        self.base_planner = base_planner
        self.enable_discrimination = enable_discrimination
        self.use_experience_trajectory = use_experience_trajectory
        self.use_state_summarizer = use_state_summarizer
        self.detailed_output = detailed_output
        self.worldmind_first_action_only = worldmind_first_action_only
        
        self.enable_goal_experience = enable_goal_experience
        self.goal_experience_top_k = goal_experience_top_k
        self.enable_process_experience = enable_process_experience
        self.process_experience_top_k = process_experience_top_k
        self.enable_experience_refine = enable_experience_refine
        
        self.current_human_instruction: Optional[str] = None
        
        default_model = None
        if hasattr(base_planner, 'model_name'):
            default_model = base_planner.model_name
        elif hasattr(base_planner, 'model') and hasattr(base_planner.model, 'model_name'):
            default_model = base_planner.model.model_name
        
        actual_discriminator_model = discriminator_model or default_model
        actual_reflector_model = reflector_model or discriminator_model or default_model
        actual_summarizer_model = summarizer_model or discriminator_model or default_model
        
        logger.info(f"WorldMind Models - Discriminator: {actual_discriminator_model}, Reflector: {actual_reflector_model}, Summarizer: {actual_summarizer_model}")
        
        if enable_discrimination and actual_discriminator_model:
            self.discriminator = WorldMindDiscriminator(model_name=actual_discriminator_model)
        else:
            self.discriminator = None
            if enable_discrimination:
                logger.warning("Discrimination enabled but no model configured - discrimination will be skipped")
        
        if actual_reflector_model:
            self.reflector = WorldMindReflector(model_name=actual_reflector_model)
        else:
            self.reflector = None
            logger.warning("No reflector model configured - reflection will be skipped")
        
        if actual_summarizer_model:
            self.state_summarizer = WorldMindStateSummarizer(
                model_name=actual_summarizer_model,
                use_separate_summarization=use_separate_summarization
            )
        else:
            self.state_summarizer = None
            logger.warning("No summarizer model configured - state summarization will be skipped")
        
        self.experience_trajectory = ExperienceTrajectory()
        
        self.goal_experience_manager = GoalExperienceManager(
            top_k=goal_experience_top_k,
            extractor_model=actual_reflector_model,
            model_type=base_planner.model_type if hasattr(base_planner, 'model_type') else 'remote'
        )
        
        self.process_experience_manager = ProcessExperienceManager(
            top_k=process_experience_top_k
        )
        
        self.experience_refiner = None
        if enable_experience_refine:
            from embodiedbench.worldmind.habitat.experience_refiner import ExperienceRefiner
            refiner_model = base_planner.model_name if hasattr(base_planner, 'model_name') else (discriminator_model or reflector_model)
            self.experience_refiner = ExperienceRefiner(
                model_name=refiner_model,
                model_type=base_planner.model_type if hasattr(base_planner, 'model_type') else 'remote'
            )
            logger.info(f"Experience refiner enabled with model: {refiner_model}")
        
        if base_planner.language_only:
            self.worldmind_template = WorldMind_TEMPLATE_LANG
        else:
            self.worldmind_template = WorldMind_TEMPLATE
        
        self.current_predicted_state: Optional[str] = None
        self.current_action_id: Optional[int] = None
        self.current_action_description: Optional[str] = None
        self.current_predicted_states: List[str] = []
        self.discrimination_history: List[Dict] = []
        self.reflection_history: List[Dict] = []
        self.current_state_summary: Optional[str] = None
        self.current_obs_path: Optional[str] = None
        self.current_worldmind_feedback: Dict = {}
        self.current_agent_input_prompt: Optional[str] = None
        self.current_agent_output: Optional[str] = None
        self.current_visual_state_description: Optional[str] = None
        self.current_eval_set: str = "base"
        
        self._cached_process_experience: Optional[List[Dict]] = None
        
        self.worldmind_stats = {
            "total_predictions": 0,
            "discrimination_matches": 0,
            "discrimination_mismatches": 0,
            "reflections_triggered": 0,
            "success_feedbacks": 0
        }
        
        logger.info(f"WorldMind Planner Wrapper initialized (goal_experience={enable_goal_experience}, process_experience={enable_process_experience}, experience_refine={enable_experience_refine})")

    def set_log_path(self, log_path: str):
        """Set the log path for saving experience entries."""
        self.goal_experience_manager.set_log_path(log_path)
        self.process_experience_manager.set_log_path(log_path)
    
    def set_eval_set(self, eval_set: str):
        """Set the current eval set for experience management."""
        self.current_eval_set = eval_set
        self.goal_experience_manager.set_eval_set(eval_set)
        self.process_experience_manager.set_eval_set(eval_set)

    def set_human_instruction(self, instruction: str):
        """Set the current human instruction for the task."""
        self.current_human_instruction = instruction
        self.goal_experience_manager.set_instruction(instruction)
        self.process_experience_manager.set_instruction(instruction)

    def update_trajectory_info(self, info: Dict):
        """Update trajectory with action info for goal experience extraction."""
        self.goal_experience_manager.update_trajectory(info)
        logger.debug(f"Trajectory updated with action_id={info.get('action_id')}, task_progress={info.get('task_progress')}")
    
    def extract_goal_experience(self, instruction: str, task_success: bool):
        """Extract and save goal experience from successful task."""
        if not self.enable_goal_experience:
            logger.debug("Goal experience extraction disabled")
            return
        
        logger.info(f"Attempting to extract goal experience: task_success={task_success}")
        
        result = self.goal_experience_manager.extract_experience(
            instruction=instruction,
            task_success=task_success
        )
        
        if result:
            logger.info(f"Successfully extracted experience: {result.get('goal_experience', '')[:100]}...")
            self.worldmind_stats["success_feedbacks"] += 1
        else:
            logger.debug("No experience extracted (task may have failed or extraction failed)")
    
    def save_process_experience(self, episode_num: int):
        """Save process experience entries for the current episode."""
        if not self.enable_process_experience:
            return
        self.process_experience_manager.save_episode_knowledge(episode_num)
    
    def reset(self):
        """Reset the planner state for a new episode."""
        self.base_planner.reset()
        self.current_predicted_state = None
        self.current_action_id = None
        self.current_action_description = None
        self.current_predicted_states = []
        self.discrimination_history = []
        self.reflection_history = []
        self.current_state_summary = None
        self.current_obs_path = None
        self.current_worldmind_feedback = {}
        self.current_agent_input_prompt = None
        self.current_agent_output = None
        self.current_visual_state_description = None
        self.current_human_instruction = None
        
        self._cached_process_experience = None
        self.experience_trajectory.reset()
        
        self.goal_experience_manager.reset_trajectory()
        self.goal_experience_manager.set_actions(self.base_planner.actions)
        self.process_experience_manager.reset_episode()
        
        if self.discriminator:
            self.discriminator.reset_statistics()
        if self.reflector:
            self.reflector.reset_statistics()
        if self.state_summarizer:
            self.state_summarizer.reset_statistics()
    
    def _extract_predicted_states(self, output_text: str) -> List[str]:
        """Extract all predicted_state values from the model output."""
        predicted_states = []
        try:
            json_object = json.loads(output_text)
            executable_plan = json_object.get('executable_plan', [])
            
            for action in executable_plan:
                state = action.get('predicted_state', '')
                predicted_states.append(state)
                
        except json.JSONDecodeError:
            pattern = r'"predicted_state"\s*:\s*"([^"]*)"'
            matches = re.findall(pattern, output_text)
            predicted_states = list(matches)
        
        return predicted_states
    
    def _extract_visual_state_description(self, output_text: str) -> Optional[str]:
        """Extract visual_state_description from agent output."""
        try:
            json_object = json.loads(output_text)
            return json_object.get('visual_state_description', None)
        except json.JSONDecodeError:
            pattern = r'"visual_state_description"\s*:\s*"([^"]*)"'
            match = re.search(pattern, output_text)
            if match:
                return match.group(1)
        return None

    def _is_exploration_phase(self, predicted_state: str) -> bool:
        """Check if the predicted_state indicates an Exploration phase."""
        if not predicted_state:
            return False
        return "exploration phase" in predicted_state.lower()
    
    def should_skip_worldmind(self) -> bool:
        """Check if WorldMind cycle should be skipped based on current predicted state."""
        if self.current_predicted_state and self._is_exploration_phase(self.current_predicted_state):
            logger.info("WorldMind: Exploration phase detected, skipping WorldMind cycle")
            return True
        return False

    def _json_to_action_with_prediction(self, output_text: str, json_key: str = 'executable_plan') -> Tuple[Any, List[str]]:
        """Parse the model output and extract both actions and predicted states."""
        try:
            output_text = fix_json_worldmind(output_text)
            
            first_brace = output_text.find('{')
            if first_brace == -1:
                logger.warning("No opening brace found in output")
                return -1, []
            
            brace_count = 0
            last_brace = -1
            for i in range(first_brace, len(output_text)):
                if output_text[i] == '{':
                    brace_count += 1
                elif output_text[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        last_brace = i
                        break
            
            if last_brace == -1:
                logger.warning("No matching closing brace found")
                return -1, []
            
            json_text = output_text[first_brace:last_brace+1]
            
            try:
                json_object = json.loads(json_text)
            except json.JSONDecodeError as e:
                logger.warning(f"First parse attempt failed: {e}")
                return -1, []
            
            executable_plan = json_object.get(json_key, [])
            
            if not executable_plan:
                logger.warning("Empty executable plan in output")
                return -2, []
            
            action_key = self.base_planner.action_key
            actions = []
            predicted_states = []
            
            for action_item in executable_plan:
                if not isinstance(action_item, dict):
                    continue
                    
                action_id = action_item.get(action_key)
                if action_id is not None:
                    actions.append(action_id)
                    
                predicted_state = action_item.get('predicted_state', '')
                predicted_states.append(predicted_state)
            
            if not actions:
                return -2, predicted_states
            
            return actions, predicted_states
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to decode JSON in WorldMind wrapper: {e}")
            return -1, []
        except Exception as e:
            logger.warning(f"Unexpected error parsing output in WorldMind wrapper: {e}")
            return -1, []

    def act(self, observation, user_instruction: str) -> Tuple[Any, str]:
        """Execute the planning with WorldMind enhancements."""
        self.current_human_instruction = user_instruction
        self.set_human_instruction(user_instruction)
        
        if isinstance(observation, str):
            self.current_obs_path = observation
        elif isinstance(observation, dict):
            self.current_obs_path = observation.get('head_rgb', None)
        
        worldmind_prompt = self._build_worldmind_prompt(user_instruction)
        self.current_agent_input_prompt = worldmind_prompt
        
        print_agent_input(worldmind_prompt, self.detailed_output)
        
        original_process_prompt = self.base_planner.process_prompt
        self.base_planner.process_prompt = lambda user_instruction, prev_act_feedback=[]: worldmind_prompt
        
        try:
            actions, output_text = self.base_planner.act(observation, user_instruction)
        finally:
            self.base_planner.process_prompt = original_process_prompt
        
        self.current_agent_output = output_text
        print_agent(output_text, self.detailed_output)
        
        self.current_visual_state_description = self._extract_visual_state_description(output_text)
        
        _, predicted_states = self._json_to_action_with_prediction(output_text)
        self.current_predicted_states = predicted_states
        
        if isinstance(actions, list) and len(actions) > 0:
            self.current_action_id = actions[0]
            self.current_predicted_state = predicted_states[0] if predicted_states else None
            if 0 <= self.current_action_id < len(self.base_planner.actions):
                self.current_action_description = self.base_planner.actions[self.current_action_id]
            else:
                self.current_action_description = f"action_{self.current_action_id}"
        elif isinstance(actions, int):
            if actions >= 0:
                self.current_action_id = actions
                self.current_predicted_state = predicted_states[0] if predicted_states else None
                if self.current_action_id < len(self.base_planner.actions):
                    self.current_action_description = self.base_planner.actions[self.current_action_id]
                else:
                    self.current_action_description = f"action_{self.current_action_id}"
            else:
                self.current_action_id = actions
                self.current_predicted_state = None
                self.current_action_description = "invalid" if actions == -1 else "empty_plan"
        
        if self.current_predicted_state:
            self.worldmind_stats["total_predictions"] += 1
        
        return actions, output_text
    
    def _build_worldmind_prompt(self, user_instruction: str) -> str:
        """Build WorldMind-specific prompt with goal experience and process experience retrieval."""
        user_instruction = user_instruction.rstrip('.')
        planner = self.base_planner
        
        if planner.n_shot >= 1:
            prompt = planner.system_prompt.format(
                len(planner.actions)-1, 
                planner.available_action_str, 
                '\n\n'.join([f'## Task Execution Example {i}: \n {x}' for i, x in enumerate(planner.examples[:planner.n_shot])])
            )
        else:
            prompt = planner.system_prompt.format(len(planner.actions)-1, planner.available_action_str, '')
        
        relevant_experiences = []
        if self.enable_goal_experience:
            relevant_experiences = self.goal_experience_manager.retrieve_experiences(user_instruction)
            logger.info(f"[DEBUG] Retrieved {len(relevant_experiences)} goal experiences for instruction: {user_instruction[:50]}...")
        
        relevant_knowledge = []
        if self.enable_process_experience:
            if self._cached_process_experience is None:
                self._cached_process_experience = self.process_experience_manager.retrieve_knowledge(user_instruction)
                logger.info(f"[DEBUG] Retrieved and cached {len(self._cached_process_experience)} process experience entries at episode start")
            relevant_knowledge = self._cached_process_experience
            logger.debug(f"[DEBUG] Using cached {len(relevant_knowledge)} process experience entries")
        
        prompt += f'\n\n## Now the human instruction is: {user_instruction}.'
        
        if self.enable_experience_refine and self.experience_refiner and (relevant_experiences or relevant_knowledge):
            refine_result = self.experience_refiner.refine_for_task(
                current_instruction=user_instruction,
                goal_experiences=relevant_experiences,
                process_experiences=relevant_knowledge
            )
            
            refined_prompt = self.experience_refiner.format_for_prompt(refine_result)
            if refined_prompt:
                prompt += f'\n\n{refined_prompt}'
                logger.info(f"Experience refiner: added consolidated experience and initial plan to prompt")
        else:
            if relevant_experiences:
                experience_prompt = self.goal_experience_manager.format_experiences_for_prompt(relevant_experiences)
                prompt += f'\n\n{experience_prompt}'
            
            if relevant_knowledge:
                knowledge_prompt = self.process_experience_manager.format_knowledge_for_prompt(relevant_knowledge)
                prompt += f'\n\n{knowledge_prompt}'
                
        if len(planner.episode_act_feedback) > 0:
            prompt += '\n The action history:\n'
            for i, action_feedback in enumerate(planner.episode_act_feedback):
                prompt += '\nStep {}, action id {}, {}, env feedback: {}'.format(
                    i, action_feedback[0], planner.actions[action_feedback[0]], action_feedback[1]
                )
            
            prompt += f"\n\n Considering the above interaction history, to achieve the human instruction: '{user_instruction}', you are supposed to output in json. You need to summarize interaction history {'and environment feedback ' if planner.use_feedback else ''}and reason why the last action or plan failed and did not finish the task, output your new plan to achieve the goal from current state. At the end, output the executable plan with action ids(0 ~ {len(planner.actions)-1}) from the available actions."
        else:
            prompt += f" You are supposed to output in json. You need to output your reasoning steps and plan. At the end, output the action id (0 ~ {len(planner.actions)-1}) from the available actions to execute."
        
        return prompt

    def get_state_description(self, summarizer_state: str = None) -> str:
        """Get the state description based on use_state_summarizer setting."""
        if self.use_state_summarizer:
            return summarizer_state or ""
        else:
            return self.current_visual_state_description or summarizer_state or ""
    
    def summarize_states(
        self, 
        before_image_path: str, 
        after_image_path: str,
        action_description: str = None
    ) -> Dict[str, str]:
        """Summarize the states before and after an action using WorldMindStateSummarizer."""
        if self.state_summarizer is None:
            logger.warning("State summarizer not configured, returning empty state summaries")
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
    
    def discriminate_prediction(
        self, 
        actual_state_summary: str, 
        action_index: int = 0,
        action_id: int = None,
        action_description: str = None
    ) -> Dict:
        """Use the discriminator to compare predicted state with actual state."""
        if not self.enable_discrimination or self.discriminator is None:
            return {"match": True, "reason": "Discrimination disabled"}
        
        if action_index < len(self.current_predicted_states):
            predicted_state = self.current_predicted_states[action_index]
        else:
            predicted_state = self.current_predicted_state
        
        if not predicted_state:
            return {"match": True, "reason": "No prediction available"}
        
        actual_action_id = action_id if action_id is not None else self.current_action_id
        action_desc = action_description if action_description is not None else (self.current_action_description or "")
        
        print_discriminator_input(
            predicted_state,
            actual_state_summary,
            action_desc,
            self.detailed_output
        )
        
        result = self.discriminator.discriminate(
            predicted_state=predicted_state,
            actual_state_summary=actual_state_summary,
            action_description=action_desc
        )
        
        print_discriminator_output(
            result["match"],
            result["reason"],
            self.detailed_output
        )
        
        if result["match"]:
            self.worldmind_stats["discrimination_matches"] += 1
        else:
            self.worldmind_stats["discrimination_mismatches"] += 1
        
        self.discrimination_history.append({
            "action_id": actual_action_id,
            "action_description": action_desc,
            "predicted_state": predicted_state,
            "actual_state_summary": actual_state_summary,
            "discrimination_result": result
        })
        
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
        """Trigger self-reflection based on discrimination result to generate process experience entries."""
        actual_action_id = action_id if action_id is not None else self.current_action_id
        actual_action_description = action_description if action_description is not None else (self.current_action_description or "unknown action")
        predicted_state = self.current_predicted_state or ""
        instruction = human_instruction or self.current_human_instruction or "Not specified."
        
        actual_state_before = self.get_state_description(state_before_action)
        actual_state_after = self.get_state_description(state_after_action)
        
        self.current_worldmind_feedback = {
            "observation_before": actual_state_before,
            "predicted_state": predicted_state,
            "env_feedback": env_feedback
        }
        
        if discrimination_result.get("match", True):
            self.worldmind_stats["success_feedbacks"] += 1
            logger.info(f"WorldMind: Action succeeded: {actual_action_description}")
            
            step = len(self.experience_trajectory)
            self.experience_trajectory.add_entry(
                step=step,
                action_id=actual_action_id,
                action_description=actual_action_description,
                observation_before=actual_state_before,
                predicted_state=predicted_state,
                env_feedback=env_feedback,
                agent_input_prompt=self.current_agent_input_prompt,
                agent_output=self.current_agent_output,
                discriminator_called=True,
                discrimination_result=discrimination_result,
                has_prediction_error=False,
                reflector_called=False,
                reflection_result=None,
                experience_entries=[]
            )
            
            if self.use_experience_trajectory:
                print_trajectory_entry(
                    step=step,
                    observation=actual_state_before,
                    action=actual_action_description,
                    predicted_state=predicted_state,
                    reflexion=None,
                    next_observation=actual_state_after,
                    detailed_output=self.detailed_output
                )
            return None
        
        trajectory_parts = []
        for fb in self.base_planner.episode_act_feedback:
            if len(fb) >= 2:
                trajectory_parts.append(f"Action: {self.base_planner.actions[fb[0]]}, Feedback: {fb[1]}")
        experience_trajectory_str = "\n".join(trajectory_parts) if trajectory_parts else "No previous experience."
        
        if self.reflector is None:
            logger.warning("Reflector not configured, skipping reflection")
            return None
        
        print_reflector_input(
            actual_action_description,
            predicted_state,
            actual_state_before,
            actual_state_after,
            env_feedback,
            self.detailed_output
        )
        
        reflection_result = self.reflector.reflect(
            action_description=actual_action_description,
            predicted_state=predicted_state,
            state_before_action=actual_state_before,
            state_after_action=actual_state_after,
            current_env_feedback=env_feedback,
            action_id=actual_action_id,
            experience_trajectory=experience_trajectory_str,
            human_instruction=instruction
        )
        
        experience_entries = reflection_result.get("experience_entry", [])
        
        print_reflector_output(str(experience_entries), self.detailed_output)
        
        if experience_entries and len(experience_entries) > 0:
            self.process_experience_manager.add_entries(experience_entries, instruction)
            logger.info(f"Added {len(experience_entries)} process experience entries")
        
        self.worldmind_stats["reflections_triggered"] += 1
        
        self.reflection_history.append({
            "action_id": actual_action_id,
            "action_description": actual_action_description,
            "experience_entries": experience_entries,
            "human_instruction": instruction
        })
        
        step = len(self.experience_trajectory)
        self.experience_trajectory.add_entry(
            step=step,
            action_id=actual_action_id,
            action_description=actual_action_description,
            observation_before=actual_state_before,
            predicted_state=predicted_state,
            env_feedback=env_feedback,
            agent_input_prompt=self.current_agent_input_prompt,
            agent_output=self.current_agent_output,
            discriminator_called=True,
            discrimination_result=discrimination_result,
            has_prediction_error=True,
            reflector_called=True,
            reflection_result=reflection_result,
            experience_entries=experience_entries
        )
        
        if self.use_experience_trajectory:
            print_trajectory_entry(
                step=step,
                observation=actual_state_before,
                action=actual_action_description,
                predicted_state=predicted_state,
                reflexion=str(experience_entries),
                next_observation=actual_state_after,
                detailed_output=self.detailed_output
            )
        
        logger.info(f"WorldMind Reflection triggered for action: {actual_action_description}")
        
        return reflection_result
    

    def save_knowledge(self, episode_num: int):
        """Save process experience entries for the current episode."""
        self.process_experience_manager.save_episode_knowledge(episode_num)
    
    def update_info(
        self, 
        info: dict, 
        discrimination_result: Optional[Dict] = None,
        reflection_result: Optional[Dict] = None
    ):
        """Update episode feedback history."""
        if 'action_id' in info:
            self.goal_experience_manager.update_trajectory(info)
        self.base_planner.update_info(info)

    def get_trajectory_for_save(self) -> List[Dict]:
        """Get trajectory data for saving to file."""
        return self.experience_trajectory.get_trajectory_for_save()
    
    def get_worldmind_stats(self) -> Dict:
        """Get WorldMind statistics."""
        stats = self.worldmind_stats.copy()
        stats["worldmind_first_action_only"] = self.worldmind_first_action_only
        
        if self.discriminator:
            stats["discriminator_stats"] = self.discriminator.get_statistics()
        if self.reflector:
            stats["reflector_stats"] = self.reflector.get_statistics()
        if self.state_summarizer:
            stats["summarizer_stats"] = self.state_summarizer.get_statistics()
        
        stats["discrimination_history"] = self.discrimination_history
        stats["reflection_history"] = self.reflection_history
        stats["experience_trajectory_length"] = len(self.experience_trajectory)
        stats["goal_experience_count"] = len(self.goal_experience_manager.experiences)
        stats["process_experience_entry_count"] = len(self.process_experience_manager.knowledge_entries)
        stats["process_experience_total_count"] = len(self.process_experience_manager.get_all_knowledge_flat())
        
        total = stats["discrimination_matches"] + stats["discrimination_mismatches"]
        stats["match_rate"] = stats["discrimination_matches"] / total if total > 0 else 0
        
        return stats
    
    @property
    def planner_steps(self):
        return self.base_planner.planner_steps
    
    @property
    def output_json_error(self):
        return self.base_planner.output_json_error
    
    @property
    def actions(self):
        return self.base_planner.actions
    
    def set_actions(self, actions):
        """Set the available actions for the planner."""
        self.base_planner.actions = actions
        self.goal_experience_manager.set_actions(actions)
    
    @property
    def history(self):
        return self.base_planner.history
    
    @property
    def language_only(self):
        return self.base_planner.language_only
