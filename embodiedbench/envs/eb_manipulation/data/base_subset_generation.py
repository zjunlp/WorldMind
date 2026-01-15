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
    system_prompt = '''## You are a helpful assistant. A Franka Panda robot with a parallel gripper needs to complete a specific task on a home desk based on the instruction.You need to help me create simple real-life scenario for this instruction.

    Here are some examples.

    Example 7: Instruction: 'Wipe the larger area.'
    Generated Context: Would you mind wiping the larger area on the table to clean it up?

    Example 8: Instruction: 'Wipe the smaller area.'
    Generated Context: There's a spill on the table. Could you help by wiping the smaller area first?


    Now output the corresponding the corresponding context for the following instruction (Do not output 'Generated Context: '):
    '''

    if chat_history is None:
        chat_history = []
        chat_history.insert(0, {"role": "system", "content": system_prompt})

    chat_history.append({"role": "user", "content": instruction})
    completion = client.chat.completions.create(
        model=MODEL,
        messages=chat_history,
        temperature=1
    )
    chat_history.append({"role": "assistant", "content": completion.choices[0].message.content})

    return completion, chat_history

def swap_words(instruction):
    # Define the replacement rules
    replacements = {
        "left": "right",
        "right": "left",
        "front": "rear",
        "rear": "front"
    }
    
    # Replace words based on rules
    return " ".join(replacements.get(word, word) for word in instruction.split())

def reformat_instruction(instruction):

    parts = instruction.split(" and ")
    first_part = parts[0].split("the ", 1)[1].strip(" .")  
    second_part = parts[1].split("the ", 1)[1].replace(" in sequence", "").strip(" .") 

    return f"Stack the {second_part} on top of the {first_part}."

task_dict = {
    # 'base': ['pick_cube_shape', 'place_into_shape_sorter_color', 'stack_cubes_color', 'wipe_table_size', 'open_drawer'],
    'base': ['stack_cubes_color']
}

data_folder = 'base/eval/'
save_folder = 'base/eval/'
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
    need_test_numbers = 12
    action_steps = 2
    for i, task_to_use in enumerate(eval_tasks):
        chat_history = None
        task = env.get_task(task_to_use)
        print("task_name:\n", task_to_use)
        e_path = load_test_config(save_folder, task_files[i]) 
        for num, e in enumerate(e_path):
            if num >= need_test_numbers:
                break
            print("data_path:\n", e)
            task_base = str(e/"task_base.ttm")
            waypoint_sets = str(e/"waypoint_sets.ttm")
            config_load = str(e/"configs.pkl")
            with open(config_load, "rb") as f:
                config_data = pickle.load(f)
            new_instruction = reformat_instruction(config_data.high_level_descriptions[0])
            print("instruction:\n", config_data.high_level_descriptions)
            config_data.high_level_descriptions = [new_instruction]
            print("new instruction:\n", config_data.high_level_descriptions)
            # completion, chat_history = transform_common_sense(instruction[0], chat_history)
            # config_data.high_level_descriptions = [completion.choices[0].message.content]
            with open(config_load, 'wb') as f:
                pickle.dump(config_data, f)
            # descriptions, obs = task.load_config(task_base, waypoint_sets, config_load)
            # waypoints_info = {name: obj for name, obj in obs.object_informations.items() if "waypoint" in name}
            # print("descriptions:\n", descriptions)
            # print("waypoints_info", waypoints_info)
            # print("Common sense context:\n", completion.choices[0].message.content)
            # for _ in range(action_steps):
            #     action = agent.act(obs, descriptions)
            #     # print(action)
            #     obs, reward, terminate = task.step(action)

    env.shutdown()