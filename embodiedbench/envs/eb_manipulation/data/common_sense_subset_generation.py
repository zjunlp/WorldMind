from openai import OpenAI 
import os
import argparse
from distutils.util import strtobool
from pathlib import Path
from amsolver.environment import Environment
from amsolver.action_modes import ArmActionMode, ActionMode
from amsolver.observation_config import ObservationConfig
import numpy as np
from amsolver.backend.utils import task_file_to_task_class
import pickle
import shutil

class Agent(object):

    def __init__(self, action_shape):
        self.action_shape = action_shape

    def act(self, obs, descriptions):
        arm = np.random.normal(0.0, 0.1, size=(self.action_shape-1,))
        gripper = [1.0]  # Always open
        return np.concatenate([arm, gripper], axis=-1)

def load_test_config(data_folder: Path, task_name):
    episode_list = []
    # import pdb;pdb.set_trace()
    for path in data_folder.rglob('task_base*'):
        t_name = path.parents[3].name
        
        if t_name == task_name:
            episode_list.append(path.parent)
    episode_list.sort()
    return episode_list

def copy_folder_with_new(src_folder, new_folder):
    # if not os.path.exists(new_folder):
    #     os.makedirs(os.path.dirname(new_folder), exist_ok=True)
    shutil.copytree(src_folder, new_folder)
    print("copy base folder successfully.")

def transform_common_sense(instruction, chat_history=None):
    system_prompt = '''## You are a helpful assistant. A Franka Panda robot with a parallel gripper needs to complete a specific task on a home desk based on the instruction. You need to help me create a modified instruction that requires the use of common sense in real-life scenarios.

    Here are some examples.

    Example 1: Instruction: 'Pick up the star and place it into the silver container.'
    Generated Context: Pick up the bright object that usually appears in the night sky alongside the moon and place it into the silver box used for storing things.

    Example 2: Instruction: 'Pick up the cylinder and place it into the lime container.'
    Generated Context: Pick up the object shaped like a soda can and place it into the lime storage bin.
    
    Example 3: Instruction: 'Put the red cube into the shape sorter.'
    Generated Context: Put the red object shaped like a 3D puzzle with six faces of nine colored squares that twist to align each side, into the toy box designed for sorting shapes.

    Example 4: Instruction: 'Put the green star into the shape sorter.'
    Generated Context: Place the green object resembling a star from Greek mythology constellations into the container designed for organizing various shapes.

    Example 5: Instruction: 'Stack the green triangular prism and the teal triangular prism in sequence.'
    Generated Context: Stack the teal object associated with physics experiments on light refraction and dispersion on top of the green object with the same shape

    Example 6: Instruction: 'Stack the green cylinder and the azure cylinder in sequence.'
    Generated Context: Stack the azure candle-like object used in ancient times before the invention of lamps on top of the green one.

    Example 7: Instruction: 'Wipe the horizontal area.'
    Generated Context: Clean the area that requires you to move the eraser left and right.

    Example 8: Instruction: 'Wipe the vertical area.'
    Generated Context: Wipe the object region that requires you to move the eraser back and forth to ensure it is spotless.

    ## Guidelines
    1. Generated common sense context should be concise.
    2. For similar instructions, ensure the generated context is diverse and the responses include a variety of sentence structures.
    3. You can consider diverse scenarios to construct context: replace item descriptions with everyday life items (e.g. battery, telescope tube) or use physics experiments, imagination and common sense for descriptions.
    4. Star should be associated with night/myth.
    5. For wiping tasks, the object to be wiped is the table, no other objects. For horizontal tasks, subtly emphasize the horizontal characteristic, such as requiring left-to-right wiping. Similarly, for vertical tasks, highlight the need for front-and-rear motion
    6. Avoid using vague terms like 'side-to-side', 'pushing and pulling' or 'from one end to the other'.
    7. Avoid using 'vertical' or 'horizontal' directly.
    8. You should include 'area' or 'object region' in wipe tasks.

    Now output the corresponding the corresponding context for the following instruction (Do not output 'Generated Context: '):
    '''
# 6. Do not directly describe shape like 'cube-shaped', 'cylindrical object' or 'object with three rectangular sides and two triangular bases' format.
    if chat_history is None:
        chat_history = []
        chat_history.insert(0, {"role": "system", "content": system_prompt})

    chat_history.append({"role": "user", "content": instruction})
    completion = client.chat.completions.create(
        model=MODEL,
        messages=chat_history,
        temperature=0.1
    )
    chat_history.append({"role": "assistant", "content": completion.choices[0].message.content})

    return completion, chat_history

task_dict = {
    # 'base': ['pick_cube_shape', 'place_into_shape_sorter_color', 'stack_cubes_color', 'wipe_table_size', 'open_drawer'],
    'base': ['wipe_table_direction']
}

data_folder = 'base/eval/'
save_folder = 'common_sense/eval/'
MODEL="gpt-4o"
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

if __name__=="__main__":
    obs_config = ObservationConfig()
    obs_config.set_all(True)
    obs_config.set_image_size([360,360])
    
    task_files = task_dict['base']
    eval_tasks = [task_file_to_task_class(t, parent_folder = 'vlm') for t in task_files]
    # copy_folder_with_new(data_folder, save_folder)
    save_folder = Path(save_folder)

    action_mode = ActionMode(ArmActionMode.ABS_JOINT_VELOCITY)
    env = Environment(action_mode, obs_config=obs_config, headless=False) # set headless=False, if user want to visualize the simulator
    env.launch()

    agent = Agent(env.action_size)
    need_test_numbers = 21
    action_steps = 2
    for i, task_to_use in enumerate(eval_tasks):
        chat_history = None
        task = env.get_task(task_to_use)
        print("task_name:\n", task_to_use)
        e_path = load_test_config(save_folder, task_files[i]) 
        print(e_path)
        for num, e in enumerate(e_path):
            if num >= need_test_numbers:
                break
            print("data_path:\n", e)
            task_base = str(e/"task_base.ttm")
            waypoint_sets = str(e/"waypoint_sets.ttm")
            config_load = str(e/"configs.pkl")
            with open(config_load, "rb") as f:
                config_data = pickle.load(f)
            instruction = config_data.high_level_descriptions
            print("original instruction:\n", instruction)
            completion, chat_history = transform_common_sense(instruction[0], chat_history)
            config_data.high_level_descriptions = [completion.choices[0].message.content]
            with open(config_load, 'wb') as f:
                pickle.dump(config_data, f)
            descriptions, obs = task.load_config(task_base, waypoint_sets, config_load)
            # waypoints_info = {name: obj for name, obj in obs.object_informations.items() if "waypoint" in name}
            print("descriptions:\n", descriptions)
            # print("waypoints_info", waypoints_info)
            # print("Common sense context:\n", completion.choices[0].message.content)
            # for _ in range(action_steps):
            #     action = agent.act(obs, descriptions)
            #     # print(action)
            #     obs, reward, terminate = task.step(action)

    env.shutdown()