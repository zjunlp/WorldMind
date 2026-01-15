import os
import numpy as np
from tqdm import tqdm
import time
import json
from embodiedbench.envs.eb_alfred.EBAlfEnv import EBAlfEnv, ValidEvalSets
from embodiedbench.planner.vlm_planner import VLMPlanner
from embodiedbench.evaluator.summarize_result import average_json_values
from embodiedbench.evaluator.evaluator_utils import load_saved_data, update_config_with_args
from embodiedbench.evaluator.config.system_prompts import alfred_system_prompt
from embodiedbench.main import logger

example_path = os.path.join(os.path.dirname(__file__), 'config/alfred_examples.json')
exploration_example_path = os.path.join(os.path.dirname(__file__), 'config/alfred_long_horizon_examples.json')
system_prompt = alfred_system_prompt


class EB_AlfredEvaluator():
    def __init__(self, config):
        self.model_name = config['model_name']
        self.eval_set = ValidEvalSets[0]
        self.config = config
        self.env = None
        self.planner = None

    def check_config_valid(self):
        if self.config['multistep'] + self.config['chat_history'] > 1:
            raise ValueError("Only one of multistep, chat_history can be enabled at a time.")
        
        if self.config['language_only']:
            if self.config['multistep']:
                logger.warning("Language only mode should not have multistep enabled. Setting these arguments to False ...")
                self.config['multistep'] = 0
        
    def save_episode_metric(self, episode_info):
        episode_idx = self.env._current_episode_num if not len(self.env.selected_indexes) else self.env.selected_indexes[self.env._current_episode_num - 1] + 1
        filename = 'episode_{}_final_res.json'.format(episode_idx)
        res_path = os.path.join(self.env.log_path, 'results')
        if not os.path.exists(res_path):
            os.makedirs(res_path)
        with open(os.path.join(res_path, filename), 'w', encoding='utf-8') as f:
            json.dump(episode_info, f, ensure_ascii=False, indent=2)

    def evaluate_main(self):
        valid_eval_sets = self.config.get('eval_sets', ValidEvalSets)
        valid_eval_sets = list(valid_eval_sets)
        if type(valid_eval_sets) == list and len(valid_eval_sets) == 0:
            valid_eval_sets = ValidEvalSets

        for eval_set in valid_eval_sets:
            if self.env is not None:
                self.env.close()
            self.eval_set = eval_set
            logger.info(f'Current eval set: {eval_set}')
            exp_name = f"{self.model_name.split('/')[-1]}_{self.config['exp_name']}/{eval_set}" if len(self.config['exp_name']) else f"{self.model_name.split('/')[-1]}/{eval_set}"
            self.env = EBAlfEnv(eval_set=self.eval_set, down_sample_ratio=self.config['down_sample_ratio'],
                                          exp_name=exp_name, selected_indexes=self.config.get('selected_indexes', []),
                                          detection_box=self.config.get('detection_box', False),
                                          resolution=self.config.get('resolution', 500), 
                                          )
            examples = json.load(open(example_path, 'r+')) if self.eval_set != 'long_horizon' else json.load(open(exploration_example_path, 'r+'))
            model_type = self.config.get('model_type', 'remote')
            self.planner = VLMPlanner(self.model_name, model_type, self.env.language_skill_set, system_prompt, examples, n_shot=self.config['n_shots'],
                                                    obs_key='head_rgb', chat_history=self.config['chat_history'], language_only=self.config['language_only'],
                                                    use_feedback=self.config.get('env_feedback', True), multistep=self.config.get('multistep', 0), tp=self.config.get('tp', 1))

            self.evaluate()
            average_json_values(os.path.join(self.env.log_path, 'results'), output_file='summary.json')
            with open(os.path.join(self.env.log_path, 'config.txt'), 'w') as f:
                f.write(str(self.config))

    def evaluate(self):
        progress_bar = tqdm(total=self.env.number_of_episodes, desc="Episodes")
        while self.env._current_episode_num < self.env.number_of_episodes:
            logger.info(f"Evaluating episode {self.env._current_episode_num} ...")
            episode_info = {'reward': [], 'num_invalid_actions': 0, 'empty_plan': 0}
            obs = self.env.reset()
            img_path = self.env.save_image(obs)
            user_instruction = self.env.episode_language_instruction
            print(f"Instruction: {user_instruction}")

            self.planner.reset()
            self.planner.set_actions(self.env.language_skill_set)
            done = False
            while not done:
                try: 
                    action, reasoning = self.planner.act(img_path, user_instruction)
                    print(f"Planner Output Action: {action}")
                    if action == -2:
                        episode_info['empty_plan'] = 1
                        self.env.episode_log.append({
                            'last_action_success': 0.0,
                            'action_id': -2,
                            'action_description': 'empty plan',
                            'reasoning': reasoning,
                        })
                        info = {
                            'task_success': episode_info.get('task_success', 0),
                            'task_progress': episode_info.get("task_progress", 0),
                            'env_step': self.env._current_step,
                        }
                        break 
                    if action == -1:
                        self.env._cur_invalid_actions += 1
                        episode_info['reward'].append(-1)
                        episode_info['num_invalid_actions'] += 1
                        self.env.episode_log.append({
                            'last_action_success': 0.0,
                            'action_id': -1,
                            'action_description': 'invalid action',
                            'reasoning': reasoning,
                        })
                        info = {
                            'task_success': episode_info.get('task_success', 0),
                            'task_progress': episode_info.get("task_progress", 0),
                            'env_step': self.env._current_step,
                        }
                        if self.env._cur_invalid_actions >= self.env._max_invalid_actions:
                            break
                        continue
                    
                    if type(action) == list:
                        for action_single in action[:min(self.env._max_episode_steps - self.env._current_step, len(action))]:
                            obs, reward, done, info = self.env.step(action_single, reasoning=reasoning)
                            action_str = action_single if type(action_single) == str else self.env.language_skill_set[action_single]
                            print(f"Executed action: {action_str}, Task success: {info['task_success']}")
                            logger.debug(f"reward: {reward}")
                            logger.debug(f"terminate: {done}\n")
                            self.planner.update_info(info)
                            img_path = self.env.save_image(obs)
                            episode_info['reward'].append(reward)
                            episode_info['num_invalid_actions'] += (info['last_action_success'] == 0)
                            if done or not info['last_action_success']:
                                print("Invalid action or task complete. If invalid then Replanning.")
                                break
                    else:
                        obs, reward, done, info = self.env.step(action, reasoning=reasoning)
                        action_str = action if type(action) == str else self.env.language_skill_set[action]
                        print(f"Executed action: {action_str}, Task success: {info['task_success']}")
                        logger.debug(f"reward: {reward}")
                        logger.debug(f"terminate: {done}\n")
                        
                        self.planner.update_info(info)
                        img_path = self.env.save_image(obs)
                        episode_info['reward'].append(reward)
                        episode_info['num_invalid_actions'] += (info['last_action_success'] == 0)
                
                except Exception as e: 
                    print(e)
                    time.sleep(30)

            episode_info['instruction'] = user_instruction
            episode_info['reward'] = np.mean(episode_info['reward'])
            episode_info['task_success'] = info['task_success']
            episode_info["task_progress"] = info['task_progress']
            episode_info['num_steps'] = info["env_step"]
            episode_info['planner_steps'] = self.planner.planner_steps
            episode_info['planner_output_error'] = self.planner.output_json_error
            episode_info["num_invalid_actions"] = episode_info['num_invalid_actions']
            episode_info["num_invalid_action_ratio"] = episode_info['num_invalid_actions'] / info["env_step"] if info['env_step'] > 0 else 0
            episode_info["episode_elapsed_seconds"] = info.get("episode_elapsed_seconds", time.time() - self.env._episode_start_time)

            self.env.save_episode_log()
            self.save_episode_metric(episode_info)
            progress_bar.update()


if __name__ == '__main__':
    import argparse
    def parse_arguments():
        parser = argparse.ArgumentParser(description='Change configuration parameters.')
        parser.add_argument('--model_name', type=str, help='Name of the model.')
        parser.add_argument('--n_shots', type=int, help='Number of examples')
        parser.add_argument('--down_sample_ratio', type=float, help='Down sample ratio.')
        parser.add_argument('--model_type', type=str, help='Type of the model.')
        parser.add_argument('--language_only', type=int, help='Set to True for language only mode.')
        parser.add_argument('--exp_name', type=str, help='Name of the experiment.')
        parser.add_argument('--chat_history', type=int, help='Set to True to enable chat history.')
        parser.add_argument('--detection_box', type=int, help='Set to True to enable detection.')
        parser.add_argument('--eval_sets', type=lambda s: s.split(','), help='Comma-separated list of evaluation sets.')
        parser.add_argument('--multistep', type=int, help='Number of steps for multi-step reasoning.')
        parser.add_argument('--resolution', type=int, help='Resolution for processing.')
        parser.add_argument('--env_feedback', type=int, help='Set to True to enable environment feedback.')
        parser.add_argument('--tp', type=int, help='number of tensor parallel splits of the model parameters')
        return parser.parse_args()


    config = {
        'model_name': 'gpt-4o-mini',
        'n_shots': 10,
        'down_sample_ratio': 1.0,
        'model_type': 'remote',
        'language_only': 0,
        'exp_name': 'vlm_10shots_imgsize500',
        'chat_history': 0, 
        'detection_box': 0,
        'eval_sets': ['base'], 
        'selected_indexes': [], 
        'multistep':0, 
        'resolution': 500, 
        'env_feedback': 1,
        'tp': 1,
    }

    args = parse_arguments()
    update_config_with_args(config, args)

    evaluator = EB_AlfredEvaluator(config)
    evaluator.evaluate_main()


worldmind_example_path = os.path.join(os.path.dirname(__file__), '../worldmind/alfred/alfred_examples_worldmind.json')
worldmind_long_horizon_example_path = os.path.join(os.path.dirname(__file__), '../worldmind/alfred/alfred_long_horizon_examples_worldmind.json')

worldmind_examples_raw = json.load(open(worldmind_example_path, 'r+'))
worldmind_long_horizon_examples_raw = json.load(open(worldmind_long_horizon_example_path, 'r+'))

def convert_worldmind_examples(examples_raw):
    examples = []
    for example in examples_raw:
        converted_example = {
            "user_instruction": example.get("Human instruction", ""),
            "language_plan": example.get("Output", {}).get("language_plan", ""),
            "executable_plan": example.get("Output", {}).get("executable_plan", [])
        }
        examples.append(converted_example)
    return examples

worldmind_examples = convert_worldmind_examples(worldmind_examples_raw)
worldmind_long_horizon_examples = convert_worldmind_examples(worldmind_long_horizon_examples_raw)


class EB_AlfredEvaluator_WorldMind(EB_AlfredEvaluator):
    """WorldMind-enhanced Alfred Evaluator with reflection and experience trajectory."""
    
    def __init__(self, config):
        super().__init__(config)
        self.enable_worldmind = config.get('enable_worldmind', True)
        self.worldmind_discriminator_model = config.get('worldmind_discriminator_model')
        self.worldmind_reflector_model = config.get('worldmind_reflector_model', None)
        self.use_experience_trajectory = config.get('use_experience_trajectory', False)
        self.detailed_output = config.get('detailed_output', False)
        
        self.enable_goal_experience = config.get('enable_goal_experience', True)
        self.goal_experience_top_k = config.get('goal_experience_top_k', 3)
        
        self.enable_process_experience = config.get('enable_process_experience', True)
        self.process_experience_top_k = config.get('process_experience_top_k', 3)
        
        self.enable_experience_refine = config.get('enable_experience_refine', False)
        
        self.include_reflection_in_history = config.get('include_reflection_in_history', True)
        
        from embodiedbench.worldmind.alfred.prompts import WORLDMIND_ALFRED_SYSTEM_PROMPT
        self.system_prompt = WORLDMIND_ALFRED_SYSTEM_PROMPT
        self.worldmind_examples = worldmind_examples
        self.worldmind_long_horizon_examples = worldmind_long_horizon_examples
    
    def check_config_valid(self):
        """Validate configuration for WorldMind mode."""
        super().check_config_valid()
        if self.enable_worldmind:
            logger.info("WorldMind (World Modeling) is enabled for Alfred")
            logger.info(f"WorldMind Discriminator Model: {self.worldmind_discriminator_model}")
            reflector_model = self.worldmind_reflector_model or self.worldmind_discriminator_model
            logger.info(f"WorldMind Reflector Model: {reflector_model}")
            logger.info(f"Use Experience Trajectory: {self.use_experience_trajectory}")
            logger.info(f"Detailed Output: {self.detailed_output}")
            logger.info(f"Enable Experience Refine: {self.enable_experience_refine}")
            logger.info(f"Include Reflection in History: {self.include_reflection_in_history}")

    def get_completed_episodes(self, log_path: str) -> set:
        """Get set of completed episode numbers from results folder."""
        completed = set()
        results_path = os.path.join(log_path, 'results')
        if not os.path.exists(results_path):
            return completed
        
        import re
        for filename in os.listdir(results_path):
            if filename.startswith('episode_') and filename.endswith('_final_res.json'):
                match = re.match(r'episode_(\d+)_final_res\.json', filename)
                if match:
                    completed.add(int(match.group(1)))
        return completed
    
    def should_skip_episode(self, episode_num: int, completed_episodes: set) -> bool:
        """Check if episode should be skipped (already completed)."""
        if not self.config.get('resume', False):
            return False
        actual_episode = episode_num if not len(self.env.selected_indexes) else self.env.selected_indexes[episode_num] + 1
        return actual_episode in completed_episodes
    
    def is_eval_set_completed(self, log_path: str, total_episodes: int) -> bool:
        """Check if all episodes in eval set are completed."""
        if not self.config.get('resume', False):
            return False
        completed = self.get_completed_episodes(log_path)
        return len(completed) >= total_episodes
    
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
        
        logger.info(f"Trajectory saved to {filepath}")
    
    def evaluate_main(self):
        """Main evaluation loop for WorldMind mode."""
        from embodiedbench.worldmind.alfred.planner_wrapper import WorldMindPlannerWrapper
        
        valid_eval_sets = self.config.get('eval_sets', ValidEvalSets)
        valid_eval_sets = list(valid_eval_sets)
        if type(valid_eval_sets) == list and len(valid_eval_sets) == 0:
            valid_eval_sets = ValidEvalSets
            
        for eval_set in valid_eval_sets:
            if self.env is not None:
                self.env.close()
            self.eval_set = eval_set
            logger.info(f'Current eval set: {eval_set}')
            
            base_exp_name = f"{self.model_name.split('/')[-1]}_{self.config['exp_name']}" if len(self.config['exp_name']) else f"{self.model_name.split('/')[-1]}"
            exp_name = f"worldmind_{base_exp_name}/{eval_set}"
            
            self.env = EBAlfEnv(
                eval_set=self.eval_set, 
                down_sample_ratio=self.config['down_sample_ratio'], 
                exp_name=exp_name, 
                selected_indexes=self.config.get('selected_indexes', []), 
                detection_box=self.config.get('detection_box', False),
                resolution=self.config.get('resolution', 500)
            )
            
            if self.is_eval_set_completed(self.env.log_path, self.env.number_of_episodes):
                logger.info(f"Skipping eval_set '{eval_set}' - all episodes already completed")
                continue

            examples_to_use = self.worldmind_long_horizon_examples if eval_set == 'long_horizon' else self.worldmind_examples
            
            model_type = self.config.get('model_type', 'remote')
            
            base_planner = VLMPlanner(
                self.model_name, 
                model_type, 
                self.env.language_skill_set, 
                self.system_prompt,
                examples_to_use,
                n_shot=self.config['n_shots'], 
                obs_key='head_rgb',
                chat_history=self.config['chat_history'], 
                language_only=self.config['language_only'],
                use_feedback=self.config.get('env_feedback', True), 
                multistep=self.config.get('multistep', 0), 
                tp=self.config.get('tp', 1),
                kwargs={'use_worldmind_template': True}
            )
            
            self.planner = WorldMindPlannerWrapper(
                base_planner=base_planner,
                discriminator_model=self.worldmind_discriminator_model,
                reflector_model=self.worldmind_reflector_model,
                enable_discrimination=self.enable_worldmind,
                use_experience_trajectory=self.use_experience_trajectory,
                detailed_output=self.detailed_output,
                worldmind_first_action_only=True, 
                enable_goal_experience=self.enable_goal_experience,
                goal_experience_top_k=self.goal_experience_top_k,
                enable_process_experience=self.enable_process_experience,
                process_experience_top_k=self.process_experience_top_k,
                enable_experience_refine=self.enable_experience_refine
            )
            
            self.planner.set_log_path(self.env.log_path)
            self.planner.set_eval_set(self.eval_set)
            
            logger.info("WorldMind Planner Wrapper initialized successfully")

            self.evaluate()
            average_json_values(os.path.join(self.env.log_path, 'results'), output_file='summary.json')
            with open(os.path.join(self.env.log_path, 'config.txt'), 'w') as f:
                f.write(str(self.config))
    
    def evaluate(self):
        """Evaluate all episodes with WorldMind discrimination and reflection."""
        completed_episodes = set()
        if self.config.get('resume', False):
            completed_episodes = self.get_completed_episodes(self.env.log_path)
            if completed_episodes:
                logger.info(f"Resume mode enabled. Found {len(completed_episodes)} completed episodes: {sorted(completed_episodes)}")
        
        progress_bar = tqdm(total=self.env.number_of_episodes, desc="Episodes")
        
        if completed_episodes:
            progress_bar.update(len(completed_episodes))
        
        while self.env._current_episode_num < self.env.number_of_episodes:
            current_episode = self.env._current_episode_num
            
            if self.should_skip_episode(current_episode, completed_episodes):
                logger.info(f"Skipping completed episode {current_episode}")
                try:
                    self.env.reset()
                except:
                    self.env._current_episode_num += 1
                continue
            
            logger.info(f"Evaluating episode {self.env._current_episode_num} ...")
            episode_info = {
                'reward': [], 
                'num_invalid_actions': 0, 
                'empty_plan': 0,
                'worldmind_discrimination_results': [],
                'worldmind_reflection_results': []
            }
            obs = self.env.reset()
            img_path = self.env.save_image(obs)
            user_instruction = self.env.episode_language_instruction
            print(f"Instruction: {user_instruction}")

            self.planner.reset()
            self.planner.set_actions(self.env.language_skill_set)
            done = False
            
            while not done:
                try: 
                    before_img_path = img_path
                    
                    action, reasoning = self.planner.act(img_path, user_instruction)
                    print(f"Planner Output Action: {action}")

                    if action == -2:
                        episode_info['empty_plan'] = 1
                        self.env.episode_log.append({
                            'last_action_success': 0.0,
                            'action_id': -2,
                            'action_description': 'empty plan',
                            'reasoning': reasoning,
                        })
                        info = {
                            'task_success': episode_info.get('task_success', 0),
                            'task_progress': episode_info.get("task_progress", 0),
                            'env_step': self.env._current_step,
                        }
                        break
                    
                    if action == -1:
                        self.env._cur_invalid_actions += 1
                        episode_info['reward'].append(-1)
                        episode_info['num_invalid_actions'] += 1
                        self.env.episode_log.append({
                            'last_action_success': 0.0,
                            'action_id': -1,
                            'action_description': 'invalid action',
                            'reasoning': reasoning,
                        })
                        info = {
                            'task_success': episode_info.get('task_success', 0),
                            'task_progress': episode_info.get("task_progress", 0),
                            'env_step': self.env._current_step,
                        }
                        if self.env._cur_invalid_actions >= self.env._max_invalid_actions:
                            break
                        continue
                    
                    if type(action) == list:
                        for idx, action_single in enumerate(action[:min(self.env._max_episode_steps - self.env._current_step, len(action))]):
                            obs, reward, done, info = self.env.step(action_single, reasoning=reasoning)
                            action_str = action_single if type(action_single) == str else self.env.language_skill_set[action_single]
                            print(f"Executed action: {action_str}, Task success: {info['task_success']}")
                            logger.debug(f"reward: {reward}")
                            logger.debug(f"terminate: {done}\n")
                            
                            after_img_path = self.env.save_image(obs)
                            
                            should_perform_worldmind = (idx == 0) if hasattr(self.planner, 'worldmind_first_action_only') and self.planner.worldmind_first_action_only else True
                            
                            if should_perform_worldmind and hasattr(self.planner, 'should_skip_worldmind') and self.planner.should_skip_worldmind():
                                should_perform_worldmind = False
                                logger.info(f"Skipping WorldMind cycle for action {idx} (Exploration phase detected)")
                            
                            if should_perform_worldmind:
                                discrimination_result, reflection_result = self._perform_worldmind_cycle(
                                    action_index=idx,
                                    before_img_path=before_img_path,
                                    after_img_path=after_img_path,
                                    action_str=action_str,
                                    episode_info=episode_info,
                                    env_feedback=info.get('env_feedback', ''),
                                    action_id=action_single if isinstance(action_single, int) else None,
                                    human_instruction=user_instruction
                                )
                            else:
                                discrimination_result = None
                                reflection_result = None
                            
                            self.planner.update_info(info, discrimination_result, reflection_result)
                            
                            before_img_path = after_img_path
                            img_path = after_img_path
                            episode_info['reward'].append(reward)
                            episode_info['num_invalid_actions'] += (info['last_action_success'] == 0)
                            
                            if done or info['last_action_success'] == 0:
                                print("Invalid action or task complete. If invalid then Replanning.")
                                break
                    else:
                        obs, reward, done, info = self.env.step(action, reasoning=reasoning)
                        action_str = action if type(action) == str else self.env.language_skill_set[action]
                        print(f"Executed action: {action_str}, Task success: {info['task_success']}")
                        logger.debug(f"reward: {reward}")
                        logger.debug(f"terminate: {done}\n")
                        
                        after_img_path = self.env.save_image(obs)
                        
                        if hasattr(self.planner, 'should_skip_worldmind') and self.planner.should_skip_worldmind():
                            discrimination_result = None
                            reflection_result = None
                            logger.info("Skipping WorldMind cycle (Exploration phase detected)")
                        else:
                            discrimination_result, reflection_result = self._perform_worldmind_cycle(
                                action_index=0,
                                before_img_path=before_img_path,
                                after_img_path=after_img_path,
                                action_str=action_str,
                                episode_info=episode_info,
                                env_feedback=info.get('env_feedback', ''),
                                action_id=action if isinstance(action, int) else None,
                                human_instruction=user_instruction
                            )
                        
                        self.planner.update_info(info, discrimination_result, reflection_result)
                        
                        img_path = after_img_path
                        episode_info['reward'].append(reward)
                        episode_info['num_invalid_actions'] += (info['last_action_success'] == 0)
                
                except Exception as e: 
                    logger.error(f"Error during evaluation: {e}")
                    import traceback
                    traceback.print_exc()
                    time.sleep(30)

            episode_info['instruction'] = user_instruction
            episode_info['reward'] = np.mean(episode_info['reward']) if episode_info['reward'] else 0
            episode_info['task_success'] = info['task_success']
            episode_info["task_progress"] = info['task_progress']
            episode_info['num_steps'] = info["env_step"]
            episode_info['planner_steps'] = self.planner.planner_steps
            episode_info['planner_output_error'] = self.planner.output_json_error
            episode_info["num_invalid_actions"] = episode_info['num_invalid_actions']
            episode_info["num_invalid_action_ratio"] = episode_info['num_invalid_actions'] / info["env_step"] if info['env_step'] > 0 else 0
            episode_info["episode_elapsed_seconds"] = info.get("episode_elapsed_seconds", time.time() - self.env._episode_start_time)
            
            if hasattr(self.planner, 'get_worldmind_stats'):
                episode_info['worldmind_stats'] = self.planner.get_worldmind_stats()
            
            self.env.save_episode_log()
            self.save_episode_metric(episode_info)
            
            if hasattr(self.planner, 'get_trajectory_for_save'):
                trajectory_data = self.planner.get_trajectory_for_save()
                episode_idx = self.env._current_episode_num if not len(self.env.selected_indexes) else self.env.selected_indexes[self.env._current_episode_num - 1] + 1
                self.save_episode_trajectory(
                    episode_num=episode_idx,
                    trajectory_data=trajectory_data,
                    instruction=user_instruction
                )
            
            if info['task_success'] == 1 and hasattr(self.planner, 'extract_goal_experience'):
                self.planner.extract_goal_experience(
                    instruction=user_instruction,
                    task_success=True
                )
            
            if hasattr(self.planner, 'save_process_experience'):
                episode_idx = self.env._current_episode_num if not len(self.env.selected_indexes) else self.env.selected_indexes[self.env._current_episode_num - 1] + 1
                self.planner.save_process_experience(episode_idx)
            
            progress_bar.update()
    
    def _perform_worldmind_cycle(
        self,
        action_index: int,
        before_img_path: str,
        after_img_path: str,
        action_str: str,
        episode_info: dict,
        env_feedback: str = "",
        action_id: int = None,
        human_instruction: str = None
    ) -> tuple:
        """Perform complete WorldMind cycle: state summary -> discrimination -> reflection."""
        if not self.enable_worldmind:
            return None, None
        
        discrimination_result = None
        reflection_result = None
        
        try:
            state_summaries = self.planner.summarize_states(
                before_image_path=before_img_path,
                after_image_path=after_img_path,
                action_description=action_str
            )
            
            state_before = state_summaries.get('state_before_action', '')
            state_after = state_summaries.get('state_after_action', '')
            
            logger.debug(f"State before: {state_before[:100]}...")
            logger.debug(f"State after: {state_after[:100]}...")
            
            discrimination_result = self.planner.discriminate_prediction(
                actual_state_summary=state_after,
                action_index=action_index,
                action_id=action_id,
                action_description=action_str
            )
            
            if discrimination_result:
                logger.info(f"WorldMind Discrimination: match={discrimination_result['match']}")
                episode_info['worldmind_discrimination_results'].append({
                    'action': action_str,
                    'result': discrimination_result
                })
                
                reflection_result = self.planner.trigger_reflection(
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
                    logger.info(f"WorldMind Reflexion triggered")
                    experience_entries = reflection_result.get('experience_entry', [])
                    if experience_entries:
                        logger.info(f"WorldMind Experience Entries ({len(experience_entries)}):")
                        for i, entry in enumerate(experience_entries, 1):
                            logger.info(f"  {i}. {entry[:100]}...")
                    
                    experience_entries_str = "\n".join(experience_entries) if experience_entries else ""
                    
                    episode_info['worldmind_reflection_results'].append({
                        'action': action_str,
                        'experience_entry': experience_entries_str,
                        'experience_entries_count': len(experience_entries)
                    })
        
        except Exception as e:
            logger.error(f"WorldMind cycle error: {e}")
            import traceback
            traceback.print_exc()
        
        return discrimination_result, reflection_result


if __name__ == '__main__' and 'EB_AlfredEvaluator_WorldMind' in dir():
    import argparse
    import sys
    
    def parse_arguments_worldmind():
        """Parse command line arguments for WorldMind mode."""
        parser = argparse.ArgumentParser(description='Alfred Evaluator with WorldMind support.')
        parser.add_argument('--model_name', type=str, help='Name of the model.')
        parser.add_argument('--n_shots', type=int, help='Number of examples')
        parser.add_argument('--down_sample_ratio', type=float, help='Down sample ratio.')
        parser.add_argument('--model_type', type=str, help='Type of the model.')
        parser.add_argument('--language_only', type=int, help='Set to True for language only mode.')
        parser.add_argument('--exp_name', type=str, help='Name of the experiment.')
        parser.add_argument('--chat_history', type=int, help='Set to True to enable chat history.')
        parser.add_argument('--detection_box', type=int, help='Set to True to enable detection.')
        parser.add_argument('--eval_sets', type=lambda s: s.split(','), help='Comma-separated list of evaluation sets.')
        parser.add_argument('--multistep', type=int, help='Number of steps for multi-step reasoning.')
        parser.add_argument('--resolution', type=int, help='Resolution for processing.')
        parser.add_argument('--env_feedback', type=int, help='Set to True to enable environment feedback.')
        parser.add_argument('--tp', type=int, help='Number of tensor parallel splits of the model parameters.')
        
        parser.add_argument('--enable_worldmind', type=int, default=0, help='Enable WorldMind mode.')
        parser.add_argument('--worldmind_discriminator_model', type=str, help='Model for WorldMind discrimination.')
        parser.add_argument('--worldmind_reflector_model', type=str, help='Model for WorldMind reflection.')
        parser.add_argument('--use_experience_trajectory', type=int, help='Use experience trajectory.')
        parser.add_argument('--detailed_output', type=int, help='Enable detailed colored output.')
        parser.add_argument('--resume', type=int, default=0, help='Resume from checkpoint. Skip completed episodes.')
        
        parser.add_argument('--enable_goal_experience', type=int, help='Enable goal experience.')
        parser.add_argument('--goal_experience_top_k', type=int, help='Top K experiences to retrieve.')
        
        parser.add_argument('--enable_process_experience', type=int, help='Enable process experience.')
        parser.add_argument('--process_experience_top_k', type=int, help='Top K knowledge entries to retrieve.')
        
        parser.add_argument('--enable_experience_refine', type=int, help='Enable experience refine.')
        
        return parser.parse_args()
    
    config = {
        'model_name': 'gpt-4o-mini',
        'n_shots': 10,
        'down_sample_ratio': 1.0, 
        'model_type': 'remote',
        'language_only': 0,
        'exp_name': 'vlm_10shots_imgsize500',
        'chat_history': 0,
        'detection_box': 0,
        'eval_sets': ['base', 'common_sense', 'complex_instruction', 'spatial_relationship', 'visual_appearance', 'long_horizon'],
        'selected_indexes': [],
        'multistep': 0, 
        'resolution': 500, 
        'env_feedback': 1,
        'tp': 1,
        'enable_worldmind': 0,
        'worldmind_discriminator_model': 'gpt-4o-mini',
        'worldmind_reflector_model': None,
        'use_experience_trajectory': 0,
        'detailed_output': 0,
        'resume': 0,
        'enable_goal_experience': 1,
        'goal_experience_top_k': 3,
        'enable_process_experience': 1,
        'process_experience_top_k': 3,
        'enable_experience_refine': 0,
    }
    
    args = parse_arguments_worldmind()
    update_config_with_args(config, args)

    if config.get('enable_worldmind', False):
        evaluator = EB_AlfredEvaluator_WorldMind(config)
        logger.info("Using WorldMind-enabled Alfred Evaluator")
    else:
        evaluator = EB_AlfredEvaluator(config)
        logger.info("Using standard Alfred Evaluator")
    
    evaluator.check_config_valid()
    evaluator.evaluate_main()
