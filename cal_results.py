import os
import json
import glob
from typing import Dict, List

def calculate_summary_stats(results_dir: str) -> Dict:
    """
    Calculate summary statistics from episode result files.
    从 episode 结果文件计算汇总统计信息
    
    Args:
        results_dir: 结果文件目录路径
        
    Returns:
        包含汇总统计的字典
    """
    # 查找所有 episode_*_final_res.json 文件
    pattern = os.path.join(results_dir, 'episode_*_final_res.json')
    json_files = glob.glob(pattern)
    
    if not json_files:
        print(f"No episode result files found in {results_dir}")
        return {}
    
    print(f"Found {len(json_files)} episode result files")
    
    # 用于累积统计的变量
    stats = {
        'reward': [],
        'task_success': [],
        'task_progress': [],
        'subgoal_reward': [],
        'num_steps': [],
        'planner_steps': [],
        'planner_output_error': [],
        'num_invalid_actions': [],
        'num_invalid_action_ratio': [],
        'episode_elapsed_seconds': [],
        'empty_plan': []
    }
    
    # 读取所有文件并累积数据
    for json_file in sorted(json_files):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 累积每个指标
            for key in stats.keys():
                if key in data:
                    value = data[key]
                    # 处理列表类型（如 episode_elapsed_seconds）
                    if isinstance(value, list):
                        if len(value) > 0:
                            stats[key].append(value[0] if len(value) == 1 else sum(value) / len(value))
                    else:
                        stats[key].append(value)
        except Exception as e:
            print(f"Error reading {json_file}: {e}")
            continue
    
    # 计算平均值
    summary = {}
    for key, values in stats.items():
        if values:
            summary[key] = sum(values) / len(values)
        else:
            summary[key] = 0.0
    
    # 添加额外的统计信息
    summary['total_episodes'] = len(json_files)
    summary['success_rate'] = summary.get('task_success', 0.0)  # task_success 已经是0或1，平均值就是成功率
    
    return summary


def print_summary_stats(summary: Dict):
    """
    Print summary statistics in a readable format.
    以可读格式打印汇总统计
    """
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    print(f"\n📊 Overall Performance:")
    print(f"  Total Episodes: {summary.get('total_episodes', 0)}")
    print(f"  Success Rate: {summary.get('success_rate', 0.0):.2%}")
    print(f"  Average Reward: {summary.get('reward', 0.0):.4f}")
    print(f"  Average Task Progress: {summary.get('task_progress', 0.0):.4f}")
    print(f"  Average Subgoal Reward: {summary.get('subgoal_reward', 0.0):.4f}")
    
    print(f"\n⚙️ Execution Metrics:")
    print(f"  Average Environment Steps: {summary.get('num_steps', 0.0):.2f}")
    print(f"  Average Planner Steps: {summary.get('planner_steps', 0.0):.2f}")
    print(f"  Average Episode Duration: {summary.get('episode_elapsed_seconds', 0.0):.2f}s")
    
    print(f"\n❌ Error Metrics:")
    print(f"  Average Invalid Actions: {summary.get('num_invalid_actions', 0.0):.2f}")
    print(f"  Average Invalid Action Ratio: {summary.get('num_invalid_action_ratio', 0.0):.2%}")
    print(f"  Average Planner Output Errors: {summary.get('planner_output_error', 0.0):.2f}")
    print(f"  Average Empty Plans: {summary.get('empty_plan', 0.0):.2f}")
    
    print("\n" + "="*60)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Calculate summary statistics from episode results.')
    parser.add_argument('--results_dir', type=str, default='/mnt/20t/rbc/Embodied/EmbodiedBench/running/eb_habitat/awm_deepseek-v3.2-exp_awm/base/results', help='Path to the results directory')
    args = parser.parse_args()
    
    # 检查目录是否存在
    if not os.path.exists(args.results_dir):
        print(f"Error: Directory {args.results_dir} does not exist")
        exit(1)
    
    # 计算并打印汇总统计
    summary = calculate_summary_stats(args.results_dir)
    if summary:
        print_summary_stats(summary)
        
        # 以 JSON 格式打印（方便复制）
        print("\n📋 JSON Format:")
        print(json.dumps(summary, indent=2))

### python cal_results.py