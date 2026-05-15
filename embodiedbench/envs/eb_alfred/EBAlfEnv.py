
import os
import time
import gym
import json
import colorsys
import numpy as np
from PIL import Image


import embodiedbench.envs.eb_alfred.utils as utils
from embodiedbench.envs.eb_alfred.utils import alfred_objs, alfred_open_obj, alfred_pick_obj, alfred_slice_obj, alfred_open_obj, alfred_toggle_obj, alfred_recep
from embodiedbench.envs.eb_alfred.thor_connector import ThorConnector
from embodiedbench.envs.eb_alfred.data.preprocess import Dataset
from embodiedbench.envs.eb_alfred.gen import constants
from embodiedbench.main import logger


X_DISPLAY = '1'
ALFRED_SPLIT_PATH = os.path.join(os.path.dirname(__file__), 'data/splits/splits.json')
ALFRED_REWARD_PATH = os.path.join(os.path.dirname(__file__), 'models/config/rewards.json')
ALFRED_DATASET_PATH = os.path.join(os.path.dirname(__file__), 'data/json_2.1.0')
ValidEvalSets = [
    'base', 'common_sense', 'complex_instruction', 'spatial',
    'visual_appearance', 'long_horizon'
    ]


def get_global_action_space():
    action_space = []


    findable_objs = alfred_objs
    action_space.extend([f"find a {obj}" for obj in findable_objs])


    pickup_objs = alfred_pick_obj
    for obj in pickup_objs:
        action_space.extend([
            f"pick up the {obj}",
        ])

    action_space.extend([
            f"put down the object in hand",
            f"drop the object in hand"
        ])


    open_objs = alfred_open_obj
    for obj in open_objs:
        action_space.extend([
            f"open the {obj}",
            f"close the {obj}"
        ])


    turn_on_objs = alfred_toggle_obj
    for obj in turn_on_objs:
        action_space.extend([
            f"turn on the {obj}",
            f"turn off the {obj}"
        ])


    slice_objs = alfred_slice_obj
    action_space.extend([f"slice the {obj}" for obj in slice_objs])

    return action_space


class EBAlfEnv(gym.Env):
    def __init__(self, eval_set='base', exp_name='', down_sample_ratio=1.0, selected_indexes=[], detection_box=False, resolution=500, task_order_seed=None, observation_perturb='none'):
        super().__init__()
        self.data_path = ALFRED_SPLIT_PATH
        self.reward_config_path = ALFRED_REWARD_PATH
        self.resolution = resolution
        self.env = ThorConnector(x_display=X_DISPLAY, player_screen_height=resolution, player_screen_width=resolution)


        assert eval_set in ValidEvalSets
        self.down_sample_ratio = down_sample_ratio
        self.dataset = self._load_dataset(eval_set)
        if len(selected_indexes):
            self.dataset = [self.dataset[i] for i in selected_indexes]


        self.task_order_seed = task_order_seed
        if task_order_seed is not None:
            import random as _random
            rng = _random.Random(task_order_seed)
            order = list(range(len(self.dataset)))
            rng.shuffle(order)
            self.dataset = [self.dataset[i] for i in order]

            if len(selected_indexes):
                self.selected_indexes = [selected_indexes[i] for i in order]
            else:
                self.selected_indexes = order


        self.observation_perturb = observation_perturb


        self.number_of_episodes = len(self.dataset)
        self._reset = False
        self._current_episode_num = 0
        if task_order_seed is None:
            self.selected_indexes = selected_indexes

        self._initial_episode_num = 0
        self._current_step = 0
        self._max_episode_steps = 30
        self._cur_invalid_actions = 0
        self._max_invalid_actions = 10
        self._episode_start_time = 0
        self.episode_log = []


        self.episode_language_instruction = ''
        self.episode_data = None

        self.language_skill_set = None
        self.action_space = None



        self.feedback_verbosity = 0
        self.log_path = 'running/eb_alfred/{}'.format(exp_name)

        self.detection = detection_box
        self.name_to_id_dict = None
        self.id_to_name_dict = None
        self.language_skill_set = get_global_action_space()
        self.action_space = gym.spaces.Discrete(len(self.language_skill_set))


    def generate_additional_action_space(self):

        add_findable_objs = []
        add_pickable_objs = []

        recept_obj_dict = {}
        pickable_obj_dict = {}
        name_to_id_dict = {}
        for obj in self.env.last_event.metadata['objects']:
            if obj['receptacle']:
                if obj['objectType'] in recept_obj_dict:
                    recept_obj_dict[obj['objectType']].append(obj['objectId'])
                else:
                    recept_obj_dict[obj['objectType']] = [obj['objectId']]
            elif obj['pickupable']:
                if obj['objectType'] in pickable_obj_dict:
                    pickable_obj_dict[obj['objectType']].append(obj['objectId'])
                else:
                    pickable_obj_dict[obj['objectType']] = [obj['objectId']]



        for key in recept_obj_dict:
            if len(recept_obj_dict[key]) >= 2:
                for i in range(len(recept_obj_dict[key])):
                    if i == 0:
                        name_to_id_dict[key] = recept_obj_dict[key][i]
                    else:
                        name_to_id_dict[key + '_{}'.format(i+1)] = recept_obj_dict[key][i]
                        add_findable_objs.append(key + '_{}'.format(i+1))

        for key in pickable_obj_dict:
            if len(pickable_obj_dict[key]) >= 2:
                for i in range(len(pickable_obj_dict[key])):
                    if i == 0:
                        name_to_id_dict[key] = pickable_obj_dict[key][i]
                    else:
                        name_to_id_dict[key + '_{}'.format(i+1)] = pickable_obj_dict[key][i]
                        add_pickable_objs.append(key + '_{}'.format(i+1))

        id_to_name_dict = {}
        for key in name_to_id_dict:
            id_to_name_dict[name_to_id_dict[key]] = key


        add_findable_objs = sorted(list(set(add_findable_objs)))
        add_pickable_objs = sorted(list(set(add_pickable_objs)))
        action_space = [f"find a {obj}" for obj in add_findable_objs]
        for obj in add_findable_objs:
            if obj.split('_')[0] in alfred_open_obj:
                action_space.extend([
                    f"open the {obj}",
                    f"close the {obj}"
                ])
        for obj in add_pickable_objs:
            if obj.split('_')[0] in alfred_pick_obj:
                action_space.extend([
                    f"find a {obj}",
                ])

        self.language_skill_set = get_global_action_space() + action_space
        self.action_space = gym.spaces.Discrete(len(self.language_skill_set))
        self.name_to_id_dict = name_to_id_dict
        self.id_to_name_dict = id_to_name_dict

    def _load_dataset(self, eval_set):
        with open(self.data_path) as f:
            dataset_split = json.load(f)
        dataset = dataset_split[eval_set]
        if 0 <= self.down_sample_ratio < 1:
            select_every = round(1 / self.down_sample_ratio)
            dataset = dataset[0:len(dataset):select_every]
        return dataset


    def current_episode(self):
        res = None
        try:
            res = utils.load_task_json(self.dataset[self._current_episode_num])
        except:
            print("episode failed to load trying next episode")
            self.current_episode_num += 1
            self.current_episode()
        return res

    def _reset_controller(self, task):
        traj_data = utils.load_task_json(task)
        traj_data['turk_annotations']['anns'][task['repeat_idx']]['task_desc'] = task["instruction"]
        self.episode_data = traj_data
        args_dict = {'data': ALFRED_DATASET_PATH, 'pframe': 300, 'fast_epoch': False,
                    'use_templated_goals': False, 'dout': 'exp/model', 'pp_folder': 'pp',
                    'reward_config': self.reward_config_path, 'max_steps': 1000}
        model_args = utils.dotdict(args_dict)


        scene_num = traj_data['scene']['scene_num']
        object_poses = traj_data['scene']['object_poses']
        dirty_and_empty = traj_data['scene']['dirty_and_empty']
        object_toggles = traj_data['scene']['object_toggles']

        scene_name = 'FloorPlan%d' % scene_num
        self.episode_language_instruction = task["instruction"]

        logger.info(f"Restoring scene {scene_name}...")
        self.env.reset(scene_name)
        self.env.restore_scene(object_poses, object_toggles, dirty_and_empty)
        if traj_data['scene']['init_action']['action'] == 'TeleportFull':
            del traj_data['scene']['init_action']["rotateOnTeleport"]
            traj_data['scene']['init_action']["standing"] = True
        self.env.step(dict(traj_data['scene']['init_action']))
        self.env.set_task(traj_data, model_args, reward_type='dense', max_episode_length=self._max_episode_steps)

        self.generate_additional_action_space()

    def reset(self):
        assert self._current_episode_num < self.number_of_episodes
        self._reset_controller(self.dataset[self._current_episode_num])
        self._current_step = 0
        self._cur_invalid_actions = 0
        self._current_episode_num += 1
        obs = {
            'head_rgb': self.env.last_event.frame,
        }
        self._reset = True
        self.episode_log = []
        self._episode_start_time = time.time()
        return obs

    def _alias_for_object(self, obj):
        object_id = obj.get('objectId')
        object_type = obj.get('objectType', '')
        if self.id_to_name_dict and object_id in self.id_to_name_dict:
            return self.id_to_name_dict[object_id]
        if object_type == 'BathtubBasin':
            return 'Bathtub'
        if object_type == 'SinkBasin':
            return 'Sink'
        return object_type

    @staticmethod
    def _coarse_color_label(rgb):
        r, g, b = [max(0, min(255, int(x))) / 255.0 for x in rgb[:3]]
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        hue = h * 360.0

        if v < 0.16:
            return 'black'
        if s < 0.16:
            if v > 0.78:
                return 'white'
            if v > 0.48:
                return 'silver/gray'
            return 'gray'
        if 18 <= hue < 48 and v < 0.68:
            return 'brown'
        if hue < 15 or hue >= 345:
            return 'red'
        if hue < 35:
            return 'orange'
        if hue < 70:
            return 'yellow'
        if hue < 165:
            return 'green'
        if hue < 195:
            return 'cyan'
        if hue < 255:
            return 'blue'
        if hue < 290:
            return 'purple'
        return 'pink'

    def _visible_pixels_for_object(self, event, object_id, bbox):
        frame = getattr(event, 'frame', None)
        if frame is None:
            return np.array([])

        object_color = getattr(event, 'object_id_to_color', {}).get(object_id)
        segmentation = getattr(event, 'instance_segmentation_frame', None)
        if object_color is not None and segmentation is not None:
            color = np.array(object_color, dtype=segmentation.dtype)
            mask = np.all(segmentation == color, axis=2)
            if np.any(mask):
                return frame[mask]

        if bbox is None:
            return np.array([])
        x1, y1, x2, y2 = [int(x) for x in bbox]
        height, width = frame.shape[:2]
        x1, x2 = max(0, x1), min(width, x2)
        y1, y2 = max(0, y1), min(height, y2)
        if x2 <= x1 or y2 <= y1:
            return np.array([])
        return frame[y1:y2, x1:x2].reshape(-1, 3)

    def _color_palette_for_object(self, event, object_id, bbox):
        pixels = self._visible_pixels_for_object(event, object_id, bbox)
        if pixels.size == 0:
            return [], None
        pixels = pixels.reshape(-1, 3)
        if len(pixels) > 4000:
            stride = max(1, len(pixels) // 4000)
            pixels = pixels[::stride]

        labels = {}
        for rgb in pixels:
            label = self._coarse_color_label(rgb)
            labels[label] = labels.get(label, 0) + 1

        ranked = sorted(labels.items(), key=lambda item: item[1], reverse=True)
        palette = [label for label, count in ranked if count >= max(8, len(pixels) * 0.04)]
        if not palette and ranked:
            palette = [ranked[0][0]]

        median_rgb = np.median(pixels, axis=0).astype(int).tolist()
        return palette[:3], median_rgb

    def get_visual_observation_summary(self, max_objects=30):
        event = self.env.last_event
        detections = getattr(event, 'instance_detections2D', None) or {}
        entries = []

        for obj in event.metadata.get('objects', []):
            object_id = obj.get('objectId')
            object_type = obj.get('objectType', '')
            if not obj.get('visible') or object_id not in detections:
                continue
            if (
                object_type not in alfred_objs
                and not obj.get('pickupable')
                and not obj.get('receptacle')
                and not obj.get('toggleable')
                and object_type not in {'BathtubBasin', 'SinkBasin'}
            ):
                continue

            bbox = detections.get(object_id)
            bbox_list = [int(x) for x in bbox] if bbox is not None else []
            bbox_area = 0
            if len(bbox_list) == 4:
                bbox_area = max(0, bbox_list[2] - bbox_list[0]) * max(0, bbox_list[3] - bbox_list[1])

            palette, median_rgb = self._color_palette_for_object(event, object_id, bbox)
            parents = []
            for parent_id in obj.get('parentReceptacles') or []:
                parent_alias = None
                if self.id_to_name_dict and parent_id in self.id_to_name_dict:
                    parent_alias = self.id_to_name_dict[parent_id]
                elif isinstance(parent_id, str) and '|' in parent_id:
                    parent_alias = parent_id.split('|')[0]
                if parent_alias and parent_alias not in parents:
                    parents.append(parent_alias)

            entries.append({
                'alias': self._alias_for_object(obj),
                'object_type': 'Bathtub' if object_type == 'BathtubBasin' else ('Sink' if object_type == 'SinkBasin' else object_type),
                'pickupable': bool(obj.get('pickupable')),
                'receptacle': bool(obj.get('receptacle')),
                'toggleable': bool(obj.get('toggleable')),
                'openable': bool(obj.get('openable')),
                'is_open': bool(obj.get('isOpen')),
                'distance': round(float(obj.get('distance', 0.0)), 2),
                'colors': palette,
                'median_rgb': median_rgb,
                'materials': obj.get('salientMaterials') or [],
                'parents': parents[:3],
                'bbox_area': bbox_area,
            })

        entries.sort(
            key=lambda item: (
                not item['pickupable'],
                not item['toggleable'],
                not item['receptacle'],
                item['distance'],
                -item['bbox_area'],
                item['alias'],
            )
        )
        entries = entries[:max_objects]

        if not entries:
            return {
                'text': 'No task-relevant ALFRED objects are currently visible in the camera metadata.',
                'objects': []
            }

        lines = [
            'Current camera-visible objects from simulator metadata and observed pixels only:'
        ]
        for item in entries:
            properties = []
            if item['pickupable']:
                properties.append('pickupable')
            if item['receptacle']:
                properties.append('receptacle')
            if item['toggleable']:
                properties.append('toggleable')
            if item['openable']:
                properties.append('open' if item['is_open'] else 'closed')
            colors = ', '.join(item['colors']) if item['colors'] else 'unknown'
            materials = ', '.join(item['materials']) if item['materials'] else 'unknown'
            parent_text = f", on/in {', '.join(item['parents'])}" if item['parents'] else ''
            lines.append(
                f"- {item['alias']}: type={item['object_type']}, "
                f"{'/'.join(properties) if properties else 'object'}, "
                f"observed colors={colors}, materials={materials}, "
                f"distance={item['distance']}{parent_text}"
            )

        return {
            'text': '\n'.join(lines),
            'objects': entries,
        }


    def step(self, action, reasoning=''):
        assert self._reset, 'Reset env before stepping'
        info = {}
        self._current_step += 1
        if type(action) == int:
            lang_action  = self.language_skill_set[action]
        elif type(action) == str:
            lang_action  = action
        else:
            raise NotImplementedError

        if 'find' in lang_action or 'open' in lang_action or 'close' in lang_action or 'pick up' in lang_action:
            lang_action_split = lang_action.split(' ')
            if (self.name_to_id_dict is not None) and lang_action_split[-1] in self.name_to_id_dict:
                lang_action = ' '.join(lang_action_split[:-1] + [self.name_to_id_dict[lang_action_split[-1]]])

        event = self.env.llm_skill_interact(lang_action)
        if not event['success']:
            self._cur_invalid_actions += 1


        reward, done = self.env.get_transition_reward()
        subgoal_met = self.env.get_goal_conditions_met()
        info['task_success'] = float(self.env.get_goal_satisfied())
        info['task_progress'] = subgoal_met[0] / subgoal_met[1]

        obs = {
            'head_rgb': self.env.last_event.frame,
        }

        if self._current_step >= self._max_episode_steps or info['task_success'] or self._cur_invalid_actions >= self._max_invalid_actions:
            done = True


        info['instruction'] = self.episode_language_instruction
        info['env_step'] = self._current_step
        info['env_feedback'] = self.get_env_feedback(event)

        if self.observation_perturb and self.observation_perturb != 'none':
            info['env_feedback'] = self._apply_observation_perturb(info['env_feedback'])
        info['episode_elapsed_seconds'] = time.time() - self._episode_start_time
        info['last_action_success'] = float(event['success'])
        info['object_states'] = {
                                    "cooled_objects" : self.env.cooled_objects,
                                    "heated_objects" : self.env.heated_objects,
                                    "cleaned_objects" : self.env.cleaned_objects,
                                    "visible_objs": [obj['objectType'] for obj in self.env.last_event.metadata['objects'] if obj['visible']]
                                }
        info['action_id'] = action
        info['action_description'] = self.language_skill_set[action] if type(action) == int else action
        info['reasoning'] = reasoning
        self.episode_log.append(info)
        return obs, reward, done, info


    _PERTURB_SYNONYMS = {
        'Mug': 'cup_object', 'Cup': 'drinking_vessel', 'Bowl': 'shallow_dish',
        'Apple': 'red_fruit_obj', 'Tomato': 'red_round_veg', 'Bread': 'baked_loaf',
        'Knife': 'sharp_tool', 'Fork': 'pronged_utensil', 'Spoon': 'curved_utensil',
        'Plate': 'flat_dishware', 'Pan': 'cooking_skillet', 'Pot': 'deep_pan',
        'Fridge': 'cold_storage_unit', 'Cabinet': 'storage_compartment',
        'Drawer': 'pull_compartment', 'Microwave': 'heating_box',
        'CounterTop': 'counter_surface', 'StoveBurner': 'cook_burner',
        'SinkBasin': 'wash_basin', 'GarbageCan': 'trash_bin',
        'Book': 'reading_material', 'Laptop': 'portable_computer',
        'Pillow': 'soft_cushion', 'Statue': 'figurine_object',
        'CellPhone': 'mobile_device', 'CD': 'circular_disc',
        'KeyChain': 'keyring_obj', 'CreditCard': 'payment_card',
        'AlarmClock': 'time_device', 'BaseballBat': 'wooden_bat',
        'TennisRacket': 'string_racket', 'Pen': 'writing_stick',
        'Pencil': 'graphite_stick', 'RemoteControl': 'tv_controller',
    }
    _PERTURB_DISTRACTORS = [
        ' (a vague shadow is also visible nearby)',
        ' (the ambient lighting flickers slightly)',
        ' (an unrelated object appears momentarily)',
        ' (background reflection adds visual noise)',
    ]

    def _apply_observation_perturb(self, text):
        if not isinstance(text, str) or not text:
            return text
        import random as _random
        rng = _random.Random(hash(text) & 0xFFFFFFFF)
        out = text
        if self.observation_perturb in ('rename', 'combined'):

            for name, syn in self._PERTURB_SYNONYMS.items():
                if name in out and rng.random() < 0.3:
                    out = out.replace(name, syn, 1)
        if self.observation_perturb in ('distractor', 'combined'):

            if rng.random() < 0.5:
                out = out + rng.choice(self._PERTURB_DISTRACTORS)
        return out

    def get_env_feedback(self, info):
        msg = ''
        if info["success"]:
            msg += f"Last action executed successfully."
        else:
            if 'is not visible' in info['message'] and '|' in info['message']:
                recep_id = info['message'].split('because it is in ')[1].split('. Note')[0]
                if recep_id not in self.id_to_name_dict:
                    pos = recep_id.split('|')[0]
                else:
                    pos = self.id_to_name_dict[recep_id]
                message = info['message'].split(recep_id)[0] + pos + '. Go there to pick the object instead.'
            else:
                message = info['message']
            msg += f"Last action is invalid. {message}"
        return msg

    def seed(self, seed=None):
        self.env.random_initilize(seed)

    def save_image(self, *args, **kwargs):
        episode_idx = self._current_episode_num if not len(self.selected_indexes) else self.selected_indexes[self._current_episode_num - 1] + 1

        folder = self.log_path + '/images/episode_{}'.format(episode_idx)
        if not os.path.exists(folder):
            os.makedirs(folder)
        img = Image.fromarray(self.env.last_event.frame)
        if self.detection:
            img = utils.draw_boxes(img, self.env.last_event.instance_detections2D, name_translation=self.id_to_name_dict)


        image_path = os.path.join(folder, 'episode_{}_step_{}.png'.format(episode_idx, self._current_step))
        img.save(image_path)
        return image_path

    def save_episode_log(self):
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

        episode_idx = self._current_episode_num if not len(self.selected_indexes) else self.selected_indexes[self._current_episode_num - 1] + 1
        filename = 'episode_{}_step_{}.json'.format(episode_idx, self._current_step)
        if len(self.episode_log):

            cleaned_log = []
            for item in self.episode_log:
                if 'object_states' in item:
                    item.pop('object_states')
                cleaned_log.append(item)

            with open(os.path.join(self.log_path, filename), 'w', encoding='utf-8') as f:
                json.dump(cleaned_log, f, ensure_ascii=False, indent=2)


    def close(self):
        self.env.stop()



if __name__ == "__main__":
    """
    Example usage of the EBAlfEnv environment.
    Demonstrates environment interaction with random actions.
    """
    env = EBAlfEnv(eval_set='base', down_sample_ratio=1.0, selected_indexes=[])
    env.reset()
    print([(i, name) for i, name in enumerate(env.language_skill_set)])
    for _ in range(30):

        action = int(input('action id: '))
        if action in env.language_skill_set:
            action = env.language_skill_set.index(action)
        else:
            action = int(action)
            if action < 0:
                break

        print(env.language_skill_set[action])


        obs, reward, done, info = env.step(action)
        print(reward, done, info)

        env.save_image()
        if done:
            break
    env.close()
