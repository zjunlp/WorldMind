"""
WorldMind Navigation Visual ICL Examples

这个版本的ICL例子包含 predicted_state 字段，用于WorldMind的状态预测功能。
输出格式: [[action_id, 'action_name', 'predicted_state'], ...]
"""

from embodiedbench.planner.planner_utils import local_image_to_data_url
from copy import deepcopy
import os
import json

# WorldMind版本的模板 - 包含predicted_state
EXAMPLE_TEMPLATE = """
Task Description: {task_description}
Reasoning and reflection: {reasoning}
Executable Plan: {output}
Feedback: {env_feedback}
"""

# 使用当前目录的example文件 - WorldMind版本包含predicted_state
EXAMPLE_PATH = [
    os.path.join(os.path.dirname(__file__), f'example{i}.jsonl') for i in range(1)
]


def create_example(i, example_dict_list):
    """
    创建包含图片的ICL示例
    
    Args:
        i: 示例编号
        example_dict_list: 示例字典列表
        
    Returns:
        content列表，包含文本和图片
    """
    contents = [
        {
            "type": "text",
            "text": f"## Example {i} of a successful task completion with state prediction"
        },
    ]
    
    for example_dict in example_dict_list:
        img_url = local_image_to_data_url(
            os.path.join(os.path.dirname(__file__), example_dict["image_path"])
        )
        contents.append({
            "type": "image_url",
            "image_url": {
                "url": img_url
            }
        })
        
        example_text = EXAMPLE_TEMPLATE.format(
            task_description=example_dict["task_description"],
            reasoning=example_dict["reasoning"],
            output=example_dict["output"],
            env_feedback=example_dict["env_feedback"]
        )
        contents.append({
            "type": "text",
            "text": example_text
        })
    
    return contents


def create_example_no_image(i, example_dict_list):
    """
    创建不包含图片的ICL示例（纯文本）
    
    Args:
        i: 示例编号
        example_dict_list: 示例字典列表
        
    Returns:
        content列表，仅包含文本
    """
    contents = [
        {
            "type": "text",
            "text": f"## Example {i} of a successful task completion with state prediction"
        },
    ]
    
    for example_dict in example_dict_list:
        example_text = EXAMPLE_TEMPLATE.format(
            task_description=example_dict["task_description"],
            reasoning=example_dict["reasoning"],
            output=example_dict["output"],
            env_feedback=example_dict["env_feedback"]
        )
        contents.append({
            "type": "text",
            "text": example_text
        })
    
    return contents


def create_example_json_list(include_image=True):
    """
    创建WorldMind版本的ICL示例列表
    
    注意：WorldMind版本的示例包含predicted_state字段
    格式: [[action_id, 'action_name', 'predicted_state'], ...]
    
    Args:
        include_image: 是否包含图片
        
    Returns:
        content列表
    """
    example_content = []
    
    for i, path in enumerate(EXAMPLE_PATH):
        # 加载jsonl文件
        with open(path, 'r') as f:
            example_dict_list = [json.loads(line) for line in f]
        
        if include_image:
            example_content.extend(create_example(i, example_dict_list))
        else:
            example_content.extend(create_example_no_image(i, example_dict_list))
    
    return example_content


if __name__ == "__main__":
    print(create_example_json_list())
