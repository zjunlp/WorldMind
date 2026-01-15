import torch
import re
import os
import time
import numpy as np
import cv2
import json
from embodiedbench.planner.planner_config.generation_guide import llm_generation_guide, vlm_generation_guide
from embodiedbench.planner.planner_utils import local_image_to_data_url, template, template_lang, fix_json, safe_parse_json_or_dict

from embodiedbench.planner.remote_model import RemoteModel
from embodiedbench.planner.custom_model import CustomModel
from embodiedbench.main import logger


class VLMPlanner():
    def __init__(self, model_name, model_type, actions, system_prompt, examples, n_shot=0, obs_key='head_rgb', 
                chat_history=False, language_only=False, use_feedback=True, multistep=0, tp=1, kwargs={}):
        self.model_name = model_name
        self.obs_key = obs_key
        self.system_prompt = system_prompt
        self.examples = examples
        self.n_shot = n_shot
        self.chat_history = chat_history
        self.set_actions(actions)
        self.model_type = model_type
        if model_type == 'custom':
            self.model = CustomModel(model_name, language_only)
        else:
            self.model = RemoteModel(model_name, model_type, language_only, tp=tp)

        self.use_feedback = use_feedback
        self.multistep = multistep
        self.planner_steps = 0
        self.output_json_error = 0
        self.language_only = language_only
        self.kwargs = kwargs
        self.action_key = kwargs.pop('action_key', 'action_id')
        self.use_worldmind_template = True
    
    def set_actions(self, actions):
        self.actions = actions
        self.available_action_str = self.get_availabel_action_prompt(actions)

    def get_availabel_action_prompt(self, available_actions):
        available_action_str = ''
        for i in range(len(available_actions)):
            available_action_str += '\naction id ' + str(i) + ': ' + str(available_actions[i]) 
            if i < len(available_actions) - 1:
                available_action_str += ', '
        return available_action_str

    def process_prompt(self, user_instruction, prev_act_feedback=[]):
        user_instruction = user_instruction.rstrip('.')
        if len(prev_act_feedback) == 0:
            if self.n_shot >= 1:
                prompt = self.system_prompt.format(len(self.actions)-1, self.available_action_str, '\n\n'.join([f'## Task Execution Example {i}: \n {x}' for i,x in enumerate(self.examples[:self.n_shot])]))
            else:
                prompt = self.system_prompt.format(len(self.actions)-1, self.available_action_str, '')

            prompt += f'\n\n## Now the human instruction is: {user_instruction}.'
            if self.language_only:
                prompt += f" You are supposed to output in json. You need to output your reasoning steps and plan. At the end, output the action id (0 ~ {len(self.actions)-1}) from the available actions to excute."
            else:
                prompt += f" You are supposed to output in json. You need to describe current visual state from the image, output your reasoning steps and plan. At the end, output the action id (0 ~ {len(self.actions)-1}) from the available actions to excute."
            
        elif self.chat_history:
            prompt = f'The human instruction is: {user_instruction}.'
            prompt += '\n\n The action history:'
            for i, action_feedback in enumerate(prev_act_feedback):
                if self.use_feedback:
                    prompt += '\nStep {}, action id {}, {}, env feedback: {}'.format(i, action_feedback[0], self.actions[action_feedback[0]], action_feedback[1])
                else:
                    prompt += '\nStep {}, action id {}, {}'.format(i, action_feedback[0], self.actions[action_feedback[0]])

            if self.language_only:
                prompt += f'''\n\n Considering the above interaction history, to achieve the human instruction: '{user_instruction}', you are supposed to output in json. You need to summarize interaction history {'and environment feedback ' if self.use_feedback else ''}and reason why the last action or plan failed and did not finish the task, output your new plan to achieve the goal from current state. At the end, output the executable plan with action ids(0 ~ {len(self.actions)-1}) from the available actions.'''
            else:
                prompt += f'''\n\n Considering the above interaction history and the current image state, to achieve the human instruction: '{user_instruction}', you are supposed to output in json. You need to describe current visual state from the image, summarize interaction history {'and environment feedback ' if self.use_feedback else ''}and reason why the last action or plan failed and did not finish the task, output your new plan to achieve the goal from current state. At the end, output the excutable plan with action ids(0 ~ {len(self.actions)-1}) from the available actions.'''
        else:
            if self.n_shot >= 1:
                prompt = self.system_prompt.format(len(self.actions)-1, self.available_action_str, '\n\n'.join([f'## Task Execution Example  {i}: \n {x}' for i,x in enumerate(self.examples[:self.n_shot])]))
            else:
                prompt = self.system_prompt.format(len(self.actions)-1, self.available_action_str, '')
            prompt += f'\n\n## Now the human instruction is: {user_instruction}.'
            prompt += '\n\n The action history:'
            for i, action_feedback in enumerate(prev_act_feedback):
                if self.use_feedback:
                    prompt += '\nStep {}, action id {}, {}, env feedback: {}'.format(i, action_feedback[0], self.actions[action_feedback[0]], action_feedback[1])
                else:
                    prompt += '\nStep {}, action id {}, {}'.format(i, action_feedback[0], self.actions[action_feedback[0]])

            if self.language_only:
                prompt += f'''\n\n Considering the above interaction history, to achieve the human instruction: '{user_instruction}', you are supposed to output in json. You need to summarize interaction history {'and environment feedback ' if self.use_feedback else ''}and reason why the last action or plan failed and did not finish the task, output your new plan to achieve the goal from current state. At the end, output the excutable plan with action ids(0 ~ {len(self.actions)-1}) from the available actions.'''
            else:
                prompt += f'''\n\n Considering the above interaction history and the current image state, to achieve the human instruction: '{user_instruction}', you are supposed to output in json. You need to describe current visual state from the image, summarize interaction history {'and environment feedback ' if self.use_feedback else ''}and reason why the last action or plan failed and did not finish the task, output your new plan to achieve the goal from current state. At the end, output the excutable plan with action ids(0 ~ {len(self.actions)-1}) from the available actions.'''
        return prompt
    

    def get_message(self, image, prompt, messages=[]):
        if self.language_only:
            return messages + [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt}],
                }
            ]
        else:
            if type(image) == str:
                image_path = image 
            else:
                image_path = './evaluation/tmp_{}.png'.format(len(messages)//2)
                cv2.imwrite(image_path, image)

            if self.multistep:
                ind = int(image_path.split('step_')[-1].strip('.png'))
                content = [{"type": "text", "text": prompt}]
                for i in range(max(ind - self.multistep + 1, 0), ind +1):
                    temp_path = ''.join(image_path.split('step_')[:-1])+ f'step_{str(i)}.png'
                    temp_data_url = local_image_to_data_url(image_path=temp_path)
                    content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": temp_data_url,
                            }})
            else:
                data_url = local_image_to_data_url(image_path=image_path)
                content = [{ "type": "image_url", "image_url": { "url": data_url,}}, {"type": "text", "text": prompt}]

            return messages + [
                {
                    "role": "user",
                    "content": content,
                }
            ]

    def reset(self):
        self.episode_messages = []
        self.episode_act_feedback = []
        self.planner_steps = 0
        self.output_json_error = 0

    def language_to_action(self, output_text):
        pattern = r'\*\*\d+\*\*'
        match = re.search(pattern, output_text)
        if match:
            action = int(match.group().strip('*'))
        else:
            print('random action')
            action = np.random.randint(len(self.actions))
        return action

    def json_to_action(self, output_text, json_key='executable_plan'):
        import re
        try:
            if not output_text or not output_text.strip():
                logger.warning("Empty model output received")
                self.output_json_error += 1
                return -1
            
            if self.use_worldmind_template:
                from embodiedbench.worldmind.habitat.prompts import fix_json_worldmind
                output_text = fix_json_worldmind(output_text)
            else:
                output_text = fix_json(output_text)
            
            first_brace = output_text.find('{')
            if first_brace == -1:
                logger.warning("No opening brace found in output")
                self.output_json_error += 1
                return -1
            
            brace_count = 0
            last_brace = -1
            in_string = False
            escape_next = False
            string_char = None
            
            for i in range(first_brace, len(output_text)):
                char = output_text[i]
                
                if escape_next:
                    escape_next = False
                    continue
                
                if char == '\\':
                    escape_next = True
                    continue
                
                if char in ('"', "'"):
                    if not in_string:
                        in_string = True
                        string_char = char
                    elif char == string_char:
                        in_string = False
                        string_char = None
                    continue
                
                if not in_string:
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            last_brace = i
                            break
            
            if last_brace == -1:
                logger.warning("No matching closing brace found")
                self.output_json_error += 1
                return -1
            
            json_text = output_text[first_brace:last_brace+1]
            
            try:
                if self.use_worldmind_template:
                    import json
                    json_object = json.loads(json_text)
                else:
                    json_object = safe_parse_json_or_dict(json_text)
            except (ValueError, json.JSONDecodeError) as e:
                logger.warning(f"Failed to parse output: {e}")
                logger.debug(f"Problematic text (first 500 chars): {json_text[:500]}")
                self.output_json_error += 1
                return -1
            
            executable_plan = json_object.get(json_key, [])
            if not executable_plan:
                logger.warning('Empty plan in JSON output')
                return -2
            
            action = [x.get(self.action_key) for x in executable_plan if isinstance(x, dict)]
            action = [a for a in action if a is not None]
            
            if not len(action):
                logger.warning('Empty plan, stop here')
                action = -2
            else:
                for i, act in enumerate(action):
                    if act is None or act >= len(self.actions) or act < 0:
                        logger.warning(f'Found invalid action: {act}')
                        if i == 0:
                            action = -1
                        else:
                            action = action[:i]
                        break
                        
        except Exception as e:
            logger.warning(f"An unexpected error occurred: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            self.output_json_error += 1
            action = -1
            
        return action
        
    def act_custom(self, prompt, obs):
        assert type(obs) == str
        out = self.model.respond(prompt, obs)
        out = fix_json(out)
        logger.debug(f"Model Output:\n{out}\n")
        action = self.json_to_action(out)
        self.planner_steps += 1
        return action, out

    def act(self, observation, user_instruction):
        if type(observation) == dict:
            obs = observation[self.obs_key]
        else:
            obs = observation
        
        prompt = self.process_prompt(user_instruction, prev_act_feedback=self.episode_act_feedback)
        if 'claude' in self.model_name or 'InternVL' in self.model_name or 'Qwen2-VL' in self.model_name or 'Qwen2.5-VL' in self.model_name or self.model_type == 'custom':
            if self.use_worldmind_template:
                from embodiedbench.worldmind.habitat.prompts import WorldMind_TEMPLATE, WorldMind_TEMPLATE_LANG
                prompt = prompt + WorldMind_TEMPLATE_LANG if self.language_only else prompt + WorldMind_TEMPLATE
            else:
                prompt = prompt + template_lang if self.language_only else prompt + template

        if self.model_type == 'custom':
            return self.act_custom(prompt, obs) 

        if len(self.episode_messages) == 0:
             self.episode_messages = self.get_message(obs, prompt)
        else:
            if self.chat_history:
                self.episode_messages = self.get_message(obs, prompt, self.episode_messages)
            else:
                self.episode_messages = self.get_message(obs, prompt)
        
        for entry in self.episode_messages:
            for content_item in entry["content"]:
                if content_item["type"] == "text":
                    text_content = content_item["text"]
                    logger.debug(f"Model Input:\n{text_content}\n")

        if 'gemini-1.5-pro' in self.model_name or 'gemini-2.0-flash' in self.model_name:
            try: 
                out = self.model.respond(self.episode_messages)
                time.sleep(15)
            except Exception as e:
                print("An unexpected error occurred:", e)
                time.sleep(60)
                out = self.model.respond(self.episode_messages)
        else:
            try: 
                out = self.model.respond(self.episode_messages)
            except Exception as e:
                print("An unexpected error occurred:", e)

                if self.model_type != 'local':
                    time.sleep(60)
                else:
                    time.sleep(20)
                out = self.model.respond(self.episode_messages)
        
        logger.debug(f"Model Output:\n{out}\n")

        if self.chat_history:
            self.episode_messages.append(
                {
                "role": "assistant",
                "content": [{"type": "text", "text": out}],
                }
            )
        action = self.json_to_action(out)
        self.planner_steps += 1
        return action, out

    def update_info(self, info):
        """Update episode feedback history."""
        self.episode_act_feedback.append([
            info['action_id'],
            info['env_feedback']
        ])
