import re
import os
import numpy as np
from tqdm import tqdm
import json
from embodiedbench.envs.eb_navigation.EBNavEnv import EBNavigationEnv, ValidEvalSets
from embodiedbench.planner.nav_planner import EBNavigationPlanner
from embodiedbench.evaluator.summarize_result import average_json_values
import sys
import warnings

from time import sleep

from embodiedbench.evaluator.config.system_prompts import eb_navigation_system_prompt
from embodiedbench.evaluator.config.eb_navigation_example import examples
from embodiedbench.main import logger

system_prompt = eb_navigation_system_prompt
examples = examples

class EB_NavigationEvaluator():
    def __init__(self, config):

        self.model_name = config['model_name']
        self.eval_sets = config["eval_sets"]
        self.eval_set = None
        self.config = config

        self.env = None
        self.planner = None

    def save_episode_metric(self, episode_info, num_steps=None):
        """Save episode metrics with step count in filename."""
        episode_idx = self.env._current_episode_num if not len(self.env.selected_indexes) else self.env.selected_indexes[self.env._current_episode_num - 1] + 1
        step_count = num_steps if num_steps is not None else episode_info.get('num_steps', 0)
        filename = 'episode_{}_step_{}_final_res.json'.format(episode_idx, step_count)
        res_path = os.path.join(self.env.log_path, 'results')
        if not os.path.exists(res_path):
            os.makedirs(res_path)
        with open(os.path.join(res_path, filename), 'w', encoding='utf-8') as f:
            json.dump(episode_info, f, ensure_ascii=False, indent=2)

    def evaluate_main(self):

        valid_eval_sets = self.config.get('eval_sets', ValidEvalSets)
        self.eval_sets = list(valid_eval_sets)
        if type(self.eval_sets) == list and len(self.eval_sets) == 0:
            self.eval_sets = ValidEvalSets
            
        for eval_set in self.eval_sets:
            if self.env is not None:
                self.env.close()
            self.eval_set = eval_set
            logger.info(f'Current eval set: {eval_set}')
            exp_name = f"{self.model_name.split('/')[-1]}_{self.config['exp_name']}/{eval_set}" if len(self.config['exp_name']) else f"{self.model_name.split('/')[-1]}/{eval_set}"
            self.env = EBNavigationEnv(eval_set=self.eval_set, down_sample_ratio=self.config['down_sample_ratio'],
                                                      exp_name=exp_name, multiview=self.config['multiview'], boundingbox=self.config['detection_box'],
                                                      multistep = self.config['multistep'], resolution = self.config['resolution'])
            self.planner = EBNavigationPlanner(model_name=self.model_name, model_type = self.config['model_type'],
                                                      actions = self.env.language_skill_set, system_prompt = system_prompt,
                                                      examples = examples, n_shot=self.config['n_shots'], obs_key='head_rgb',
                                                      chat_history=self.config['chat_history'], language_only=self.config['language_only'],
                                                      multiview=self.config['multiview'], multistep = self.config['multistep'],
                                                      visual_icl = self.config['visual_icl'], truncate=self.config.get('truncate', False))
            self.evaluate()
            average_json_values(os.path.join(self.env.log_path, 'results'), selected_key = None)
            with open(os.path.join(self.env.log_path, 'config.txt'), 'w') as f:
                f.write(str(self.config))

    def evaluate(self):
        progress_bar = tqdm(total=self.env.number_of_episodes, desc="Episodes")
        while self.env._current_episode_num < self.env.number_of_episodes:
            logger.info(f"Evaluating episode {self.env._current_episode_num} ...")
            episode_info = {'reward': []}
            obs = self.env.reset()
            img_path = self.env.save_image(obs)
            user_instruction = self.env.episode_language_instruction
            print(f"Instruction: {user_instruction}")
            self.planner.reset()
            done = False
            while not done:
                try:
                    action, reasoning = self.planner.act(img_path, user_instruction)
                    print(f"Planner Output Action: {action}")
                    reasoning = json.loads(reasoning)
                    if type(action) == list:
                        for i, action_single in enumerate( action[:min(self.env._max_episode_steps - self.env._current_step + 1, len(action))] ):
                            if i==0:
                                obs, reward, done, info = self.env.step(action_single,reasoning,1)
                            else:
                                obs, reward, done, info = self.env.step(action_single,reasoning,0)
                            print(f"Executed action: {action_single}, Task success: {info['task_success']}")
                            logger.debug(f"reward: {reward}")
                            logger.debug(f"terminate: {done}\n")
                            self.planner.update_info(info)
                            img_path = self.env.save_image(obs)
                            episode_info['reward'].append(reward)

                            if done==True:
                                break

                            if info['last_action_success'] == 0:
                                print('invalid action, start replanning')
                                break
                    else:
                        obs, reward, done, info = self.env.step(action, reasoning, 1)
                        print(f"Executed action: {action}, Task success: {info['task_success']}")
                        logger.debug(f"reward: {reward}")
                        logger.debug(f"terminate: {done}\n")
                        self.planner.update_info(info)
                        img_path = self.env.save_image(obs)
                        episode_info['reward'].append(reward)

                except Exception as e:
                    sleep(1)
                    print(e)
                    print("retrying...")


            # evaluation metrics
            episode_info['instruction'] = user_instruction
            episode_info['reward'] = np.mean(episode_info['reward'])
            episode_info['task_success'] = info['task_success']
            episode_info['num_steps'] = info["env_step"]
            episode_info['planner_steps'] = self.planner.planner_steps
            episode_info['planner_output_error'] = self.planner.output_json_error
            episode_info["episode_elapsed_seconds"] = info["episode_elapsed_seconds"]
            self.save_episode_metric(episode_info)
            progress_bar.update()

    def check_config_valid(self):
        if self.config['multiview'] + self.config['multistep'] + self.config['visual_icl'] + self.config['chat_history'] > 1:
            raise ValueError("Only one of multiview, multistep, visual_icl, chat_history can be enabled at a time.")
        if self.config['language_only']:
            if self.config['multiview'] or self.config['multistep']:
                logger.warning("Language only mode should not have multiview or multistep enabled. Setting these arguments to False ...")
                self.config['multiview'] = 0
                self.config['multistep'] = 0


class EB_NavigationEvaluator_WorldMind(EB_NavigationEvaluator):
    """WorldMind version of Navigation Evaluator with experience learning and knowledge management."""
    
    def __init__(self, config):
        super().__init__(config)
        self.worldmind_wrapper = None
        
        # WorldMind parameters from config
        self.enable_worldmind = config.get('enable_worldmind', True)
        self.worldmind_discriminator_model = config.get('worldmind_discriminator_model', self.model_name)
        self.worldmind_reflector_model = config.get('worldmind_reflector_model', None)
        self.use_vision_discriminator = config.get('use_vision_discriminator', False)
        self.use_experience_trajectory = config.get('use_experience_trajectory', False)
        self.detailed_output = config.get('detailed_output', False)
        
        # Goal experience parameters (renamed from success_experience)
        self.enable_goal_experience = config.get('enable_goal_experience', True)
        self.goal_experience_top_k = config.get('goal_experience_top_k', 3)
        
        # Process experience parameters (renamed from world_knowledge)
        self.enable_process_experience = config.get('enable_process_experience', True)
        self.process_experience_top_k = config.get('process_experience_top_k', 3)
        
        # Experience refinement parameters
        self.enable_experience_refine = config.get('enable_experience_refine', False)
        
        # Use WorldMind template
        self.use_worldmind_template = config.get('use_worldmind_template', True)
        self.resume = config.get('resume', False)
        
        # WorldMind discrimination only for first action (default True)
        self.worldmind_first_action_only = config.get('worldmind_first_action_only', True)
        
    def evaluate_main(self):
        """Main evaluation entry with WorldMind wrapper initialization."""
        from embodiedbench.worldmind.navigation.planner_wrapper import WorldMindNavigationPlannerWrapper
        from embodiedbench.worldmind.navigation.prompts import WORLDMIND_NAVIGATION_SYSTEM_PROMPT
        from embodiedbench.worldmind.navigation.eb_navigation_example import examples as worldmind_examples
        from embodiedbench.planner.nav_planner import EBNavigationPlanner
        
        valid_eval_sets = self.config.get('eval_sets', ValidEvalSets)
        self.eval_sets = list(valid_eval_sets)
        if type(self.eval_sets) == list and len(self.eval_sets) == 0:
            self.eval_sets = ValidEvalSets
            
        for eval_set in self.eval_sets:
            if self.env is not None:
                self.env.close()
            self.eval_set = eval_set
            logger.info(f'[WorldMind] Current eval set: {eval_set}')
            
            exp_name = f"{self.model_name.split('/')[-1]}_{self.config['exp_name']}_worldmind/{eval_set}" if len(self.config['exp_name']) else f"{self.model_name.split('/')[-1]}_worldmind/{eval_set}"
            
            self.env = EBNavigationEnv(
                eval_set=self.eval_set, 
                down_sample_ratio=self.config['down_sample_ratio'], 
                exp_name=exp_name, 
                multiview=self.config['multiview'], 
                boundingbox=self.config['detection_box'],
                multistep=self.config['multistep'], 
                resolution=self.config['resolution']
            )

            # Create base planner with WorldMind visual_icl support
            base_planner = EBNavigationPlanner(
                model_name=self.model_name, 
                model_type=self.config['model_type'], 
                actions=self.env.language_skill_set, 
                system_prompt=WORLDMIND_NAVIGATION_SYSTEM_PROMPT,
                examples=worldmind_examples,
                n_shot=self.config['n_shots'], 
                obs_key='head_rgb', 
                chat_history=self.config['chat_history'], 
                language_only=self.config['language_only'],
                multiview=self.config['multiview'], 
                multistep=self.config['multistep'], 
                visual_icl=self.config['visual_icl'], 
                truncate=self.config.get('truncate', False),
                use_worldmind_icl=True
            )
            
            # Wrap with WorldMind planner
            self.worldmind_wrapper = WorldMindNavigationPlannerWrapper(
                base_planner=base_planner,
                discriminator_model=self.worldmind_discriminator_model,
                reflector_model=self.worldmind_reflector_model,
                log_path=self.env.log_path,
                enable_discrimination=self.enable_worldmind,
                use_vision_discriminator=self.use_vision_discriminator,
                use_experience_trajectory=self.use_experience_trajectory,
                detailed_output=self.detailed_output,
                enable_goal_experience=self.enable_goal_experience,
                goal_experience_top_k=self.goal_experience_top_k,
                enable_process_experience=self.enable_process_experience,
                process_experience_top_k=self.process_experience_top_k,
                enable_experience_refine=self.enable_experience_refine
            )
            
            self.worldmind_wrapper.set_log_path(self.env.log_path)
            self.worldmind_wrapper.set_eval_set(self.eval_set)
            
            self.planner = self.worldmind_wrapper
            
            self.evaluate()
            average_json_values(os.path.join(self.env.log_path, 'results'), selected_key=None)
            
            config_to_save = {**self.config, 'worldmind_enabled': True}
            with open(os.path.join(self.env.log_path, 'config.txt'), 'w') as f:
                f.write(str(config_to_save))

    def save_episode_trajectory(self, episode_num: int, trajectory_data: list, instruction: str):
        """Save episode trajectory to a separate trajectory folder."""
        trajectory_path = os.path.join(self.env.log_path, 'trajectory')
        if not os.path.exists(trajectory_path):
            os.makedirs(trajectory_path)
        
        trajectory_file_content = {
            "episode_num": episode_num,
            "instruction": instruction,
            "trajectory": trajectory_data
        }
        
        filename = f'episode_{episode_num}_trajectory.json'
        filepath = os.path.join(trajectory_path, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(trajectory_file_content, f, ensure_ascii=False, indent=2)
        
        logger.info(f"[WorldMind] Trajectory saved to {filepath}")
                
    def evaluate(self):
        """WorldMind enhanced evaluation loop."""
        progress_bar = tqdm(total=self.env.number_of_episodes, desc="[WorldMind] Episodes")
        
        while self.env._current_episode_num < self.env.number_of_episodes:
            episode_idx = self.env._current_episode_num + 1
            logger.info(f"[WorldMind] Evaluating episode {episode_idx} ...")
            
            episode_info = {
                'reward': [],
                'worldmind_stats': {
                    'discrimination_count': 0,
                    'reflection_count': 0,
                    'experience_entries': []
                },
                'worldmind_discrimination_results': [],
                'worldmind_reflection_results': []
            }
            
            obs = self.env.reset()
            img_path = self.env.save_image(obs)
            user_instruction = self.env.episode_language_instruction
            print(f"[WorldMind] Instruction: {user_instruction}")
            
            self.planner.reset()
            
            # Set current task instruction for experience extraction
            if hasattr(self.planner, 'goal_experience_manager') and self.planner.goal_experience_manager:
                self.planner.goal_experience_manager.set_instruction(user_instruction)
                self.planner.goal_experience_manager.set_actions(self.env.language_skill_set)
            if hasattr(self.planner, 'process_experience_manager') and self.planner.process_experience_manager:
                self.planner.process_experience_manager.set_instruction(user_instruction)
            
            done = False
            before_img_path = img_path
            
            while not done:
                try:
                    action, reasoning = self.planner.act(img_path, user_instruction)
                    print(f"[WorldMind] Planner Output Action: {action}")
                    
                    try:
                        reasoning_dict = json.loads(reasoning) if isinstance(reasoning, str) else reasoning
                    except json.JSONDecodeError:
                        reasoning_dict = {'raw_response': reasoning}
                    
                    predicted_states = self._extract_predicted_states(reasoning_dict)
                    
                    if isinstance(action, list):
                        for i, action_single in enumerate(action[:min(self.env._max_episode_steps - self.env._current_step + 1, len(action))]):
                            if i == 0:
                                obs, reward, done, info = self.env.step(action_single, reasoning_dict, 1)
                            else:
                                obs, reward, done, info = self.env.step(action_single, reasoning_dict, 0)
                            
                            print(f"[WorldMind] Executed action: {action_single}, Task success: {info['task_success']}")
                            logger.debug(f"reward: {reward}")
                            logger.debug(f"terminate: {done}\n")
                            
                            after_img_path = self.env.save_image(obs)
                            
                            action_str = self.env.language_skill_set[action_single] if isinstance(action_single, int) and action_single < len(self.env.language_skill_set) else str(action_single)
                            
                            # Determine whether to perform WorldMind cycle (only for first action if enabled)
                            should_perform_worldmind = (i == 0) if self.worldmind_first_action_only else True
                            
                            predicted_state = predicted_states[i] if i < len(predicted_states) else None
                            
                            if should_perform_worldmind and predicted_state and self.enable_worldmind:
                                discrimination_result, reflection_result = self._perform_worldmind_cycle(
                                    action_index=i,
                                    before_img_path=before_img_path,
                                    after_img_path=after_img_path,
                                    action_str=action_str,
                                    action_id=action_single,
                                    predicted_state=predicted_state,
                                    episode_info=episode_info,
                                    env_feedback=info.get('env_feedback', ''),
                                    human_instruction=user_instruction
                                )
                            
                            if hasattr(self.planner, 'goal_experience_manager') and self.planner.goal_experience_manager:
                                self.planner.goal_experience_manager.update_trajectory({
                                    'action_id': action_single,
                                    'env_feedback': info.get('env_feedback', ''),
                                    'last_action_success': info.get('last_action_success', 1)
                                })
                            
                            before_img_path = after_img_path
                            img_path = after_img_path
                            
                            self.planner.update_info(info)
                            episode_info['reward'].append(reward)

                            if done:
                                break

                            if info['last_action_success'] == 0:
                                print('[WorldMind] Invalid action, start replanning')
                                break
                    else:
                        obs, reward, done, info = self.env.step(action, reasoning_dict, 1)
                        print(f"[WorldMind] Executed action: {action}, Task success: {info['task_success']}")
                        logger.debug(f"reward: {reward}")
                        logger.debug(f"terminate: {done}\n")
                        
                        after_img_path = self.env.save_image(obs)
                        
                        action_str = self.env.language_skill_set[action] if isinstance(action, int) and action < len(self.env.language_skill_set) else str(action)
                        
                        predicted_state = predicted_states[0] if predicted_states else None
                        
                        if predicted_state and self.enable_worldmind:
                            discrimination_result, reflection_result = self._perform_worldmind_cycle(
                                action_index=0,
                                before_img_path=before_img_path,
                                after_img_path=after_img_path,
                                action_str=action_str,
                                action_id=action,
                                predicted_state=predicted_state,
                                episode_info=episode_info,
                                env_feedback=info.get('env_feedback', ''),
                                human_instruction=user_instruction
                            )
                        
                        if hasattr(self.planner, 'goal_experience_manager') and self.planner.goal_experience_manager:
                            self.planner.goal_experience_manager.update_trajectory({
                                'action_id': action,
                                'env_feedback': info.get('env_feedback', ''),
                                'last_action_success': info.get('last_action_success', 1)
                            })
                        
                        img_path = after_img_path
                        
                        self.planner.update_info(info)
                        episode_info['reward'].append(reward)

                except Exception as e:
                    sleep(1)
                    print(f"[WorldMind] Error: {e}")
                    import traceback
                    traceback.print_exc()
                    print("retrying...")

            # Evaluation metrics
            episode_info['instruction'] = user_instruction
            episode_info['reward'] = float(np.mean(episode_info['reward'])) if episode_info['reward'] else 0.0
            episode_info['task_success'] = info['task_success']
            episode_info['num_steps'] = info["env_step"]
            episode_info['planner_steps'] = self.planner.planner_steps
            episode_info['planner_output_error'] = self.planner.output_json_error
            episode_info["episode_elapsed_seconds"] = info["episode_elapsed_seconds"]
            
            self.save_episode_metric(episode_info, num_steps=info["env_step"])
            
            # Extract goal experience if task succeeded
            if info['task_success'] == 1:
                self._extract_and_save_goal_experience(user_instruction)
            
            # Save process experience
            self._save_episode_process_experience(episode_idx)
            
            # Save trajectory
            if hasattr(self.planner, 'get_trajectory_for_save'):
                trajectory_data = self.planner.get_trajectory_for_save()
                if trajectory_data:
                    self.save_episode_trajectory(
                        episode_num=episode_idx,
                        trajectory_data=trajectory_data,
                        instruction=user_instruction
                    )
            
            progress_bar.update()
            
    def _extract_predicted_states(self, reasoning_dict):
        """Extract predicted states list from reasoning."""
        predicted_states = []
        
        if 'executable_plan' in reasoning_dict:
            for step in reasoning_dict['executable_plan']:
                if isinstance(step, dict) and 'predicted_state' in step:
                    predicted_states.append(step['predicted_state'])
                else:
                    predicted_states.append(None)
        
        return predicted_states
    
    def _perform_worldmind_cycle(
        self,
        action_index: int,
        before_img_path: str,
        after_img_path: str,
        action_str: str,
        action_id: int,
        predicted_state: str,
        episode_info: dict,
        env_feedback: str = "",
        human_instruction: str = None
    ) -> tuple:
        """
        Perform complete WorldMind cycle: state summary -> discrimination -> reflection.
        
        Returns:
            (discrimination_result, reflection_result)
        """
        if not self.enable_worldmind:
            return None, None
        
        if not hasattr(self, 'worldmind_wrapper') or self.worldmind_wrapper is None:
            return None, None
            
        discrimination_result = None
        reflection_result = None
        
        try:
            # Step 1: State summarization
            state_summaries = self.worldmind_wrapper.summarize_states(
                before_image_path=before_img_path,
                after_image_path=after_img_path,
                action_description=action_str
            )
            
            state_before = state_summaries.get('state_before_action', '')
            state_after = state_summaries.get('state_after_action', '')
            
            logger.debug(f"[WorldMind] State before: {state_before[:100]}...")
            logger.debug(f"[WorldMind] State after: {state_after[:100]}...")
            
            # Step 2: Discrimination
            discrimination_result = self.worldmind_wrapper.discriminate_prediction(
                predicted_state=predicted_state,
                actual_state_summary=state_after,
                action_index=action_index,
                action_id=action_id,
                action_description=action_str
            )
            
            if discrimination_result:
                logger.info(f"[WorldMind] Discrimination: match={discrimination_result.get('match', True)}")
                episode_info['worldmind_stats']['discrimination_count'] += 1
                episode_info['worldmind_discrimination_results'].append({
                    'action': action_str,
                    'action_id': action_id,
                    'predicted_state': predicted_state,
                    'actual_state': state_after,
                    'result': discrimination_result
                })
                
                # Step 3: Trigger reflection if prediction mismatch
                if not discrimination_result.get('match', True):
                    reflection_result = self.worldmind_wrapper.trigger_reflection(
                        discrimination_result=discrimination_result,
                        state_before_action=state_before,
                        state_after_action=state_after,
                        observation_path=after_img_path,
                        env_feedback=env_feedback,
                        action_id=action_id,
                        action_description=action_str,
                        human_instruction=human_instruction
                    )
                    
                    if reflection_result:
                        logger.info(f"[WorldMind] Reflection triggered")
                        episode_info['worldmind_stats']['reflection_count'] += 1
                        
                        experience_entries = reflection_result.get('experience_entry', [])
                        if experience_entries:
                            if isinstance(experience_entries, str):
                                experience_entries = [experience_entries]
                            logger.info(f"[WorldMind] Experience Entries ({len(experience_entries)}):")
                            for i, entry in enumerate(experience_entries, 1):
                                entry_preview = entry[:100] if isinstance(entry, str) else str(entry)[:100]
                                logger.info(f"  {i}. {entry_preview}...")
                            
                            episode_info['worldmind_stats']['experience_entries'].extend(experience_entries)
                        
                        episode_info['worldmind_reflection_results'].append({
                            'action': action_str,
                            'action_id': action_id,
                            'experience_entries': experience_entries
                        })
        
        except Exception as e:
            logger.error(f"[WorldMind] WorldMind cycle error: {e}")
            import traceback
            traceback.print_exc()
        
        return discrimination_result, reflection_result
    
    def _extract_and_save_goal_experience(self, instruction: str):
        """Extract and save goal experience."""
        if not hasattr(self.planner, 'goal_experience_manager') or self.planner.goal_experience_manager is None:
            return
        
        try:
            experience = self.planner.goal_experience_manager.extract_experience(
                instruction=instruction,
                task_success=True
            )
            if experience:
                logger.info(f"[WorldMind] Goal experience extracted and saved")
        except Exception as e:
            logger.warning(f"[WorldMind] Failed to extract goal experience: {e}")
    
    def _save_episode_process_experience(self, episode_num: int):
        """Save current episode's process experience."""
        if not hasattr(self.planner, 'process_experience_manager') or self.planner.process_experience_manager is None:
            return
        
        try:
            self.planner.process_experience_manager.save_episode_experience(episode_num)
            logger.info(f"[WorldMind] Process experience saved for episode {episode_num}")
        except Exception as e:
            logger.warning(f"[WorldMind] Failed to save process experience: {e}")


if __name__ == '__main__':
    
    config = {
        'model_name': sys.argv[2],
        'down_sample_ratio': 1,
        'model_type': 'remote',
        'language_only': False,
        'dataset': sys.argv[1],
        'chat_history': True, 
        'action_num_per_plan': 5,
        'fov': 100,
        'n_shots' : int(sys.argv[4]),
        'sleep_time':  0,
        'multiview': 0,
        'boundingbox': 0,
        'target_only': 0,
        'multistep':0,
        'resolution': 500,
        'purpose': "retest",
        'exp_name': sys.argv[3],
        'icl_abl':0,
        'visual':0
    }
    evaluator = EB_NavigationEvaluator(config)
    evaluator.evaluate_main()
