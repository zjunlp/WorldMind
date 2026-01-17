<div align="center">

# 🧠 WorldMind

### Aligning Agentic World Models via Knowledgeable Experience Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![arXiv](https://img.shields.io/badge/arXiv-2024.XXXXX-b31b1b.svg)](https://arxiv.org/abs/2024.XXXXX)

<p align="center">
  <img src="assets/framework.jpg" alt="WorldMind Framework" width="90%"/>
</p>

**WorldMind** is a novel framework for aligning agentic world models through knowledgeable experience learning, enabling agents to learn from both successful trajectories and prediction errors.

[Overview](#-overview) • [Installation](#-installation) • [Quick Start](#-quick-start) • [Environments](#-environments) • [Plugin](#-worldmind-plugin) • [Citation](#-citation)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
  - [Key Features](#key-features)
  - [Method](#method)
- [Installation](#-installation)
  - [Environment Setup](#environment-setup)
  - [Task-Specific Setup](#task-specific-setup)
- [Quick Start](#-quick-start)
  - [Running Experiments](#running-experiments)
  - [Configuration](#configuration)
- [Environments](#-environments)
  - [EB-ALFRED](#-eb-alfred-household-tasks)
  - [EB-Habitat](#-eb-habitat-rearrangement-tasks)
  - [EB-Navigation](#-eb-navigation-vision-and-language-navigation)
- [WorldMind Plugin](#-worldmind-plugin)
- [Project Structure](#-project-structure)
- [Results](#-results)
- [Citation](#-citation)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

---

## 📖 Overview

**WorldMind** introduces a paradigm shift in how embodied AI agents learn and adapt. Unlike traditional approaches that rely on extensive environment interaction or domain-specific fine-tuning, WorldMind enables agents to:

- **Learn from Experience**: Extract reusable knowledge from both successful task completions and prediction errors
- **Generalize Across Tasks**: Apply learned patterns to novel situations through semantic similarity-based retrieval
- **Continuously Improve**: Accumulate and refine knowledge throughout deployment

### Key Features

| Feature | Description |
|---------|-------------|
| 🎯 **Dual Experience Learning** | Combines *Goal Experience* from successful trajectories with *Process Experience* from prediction errors |
| 🔄 **Experience-Driven Alignment** | Uses discriminator and reflector components to align world model predictions with actual environment dynamics |
| 🔍 **Semantic Retrieval** | Employs SentenceTransformer-based semantic similarity for efficient experience retrieval during task execution |
| 🌐 **Environment Agnostic** | Designed to work across different embodied AI environments (ALFRED, Habitat, Navigation) |
| 🔌 **Modular Plugin** | Standalone plugin for easy integration into existing agent systems |

### Method

WorldMind introduces a two-stage approach for world model alignment:

#### Stage 1: Experience Acquisition

```
┌─────────────────────────────────────────────────────────────────┐
│                    Experience Acquisition                        │
├─────────────────────────────┬───────────────────────────────────┤
│     Goal Experience         │      Process Experience           │
│  (Successful Trajectories)  │    (Prediction Errors)            │
├─────────────────────────────┼───────────────────────────────────┤
│  • Extract action patterns  │  • Discriminator identifies       │
│  • Capture task workflows   │    prediction conflicts           │
│  • Generalize strategies    │  • Reflector generates            │
│                             │    corrective knowledge           │
└─────────────────────────────┴───────────────────────────────────┘
```

1. **Goal Experience Extraction**: From successful task trajectories, extract high-level action-outcome patterns and task-specific workflows
2. **Process Experience Extraction**: During task execution, when the world model makes incorrect predictions, use a discriminator to identify conflicts and a reflector to generate corrective knowledge

#### Stage 2: Experience-Guided Inference

```
┌─────────────────────────────────────────────────────────────────┐
│                 Experience-Guided Inference                      │
├─────────────────────────────────────────────────────────────────┤
│  Current Task Instruction                                        │
│           ↓                                                      │
│  Semantic Similarity Search                                      │
│           ↓                                                      │
│  ┌─────────────────┐    ┌─────────────────┐                     │
│  │ Goal Experiences│    │Process Knowledge│                     │
│  └────────┬────────┘    └────────┬────────┘                     │
│           └──────────┬───────────┘                              │
│                      ↓                                           │
│           Experience Refinement (Optional)                       │
│                      ↓                                           │
│              Augmented Prompt                                    │
│                      ↓                                           │
│            World Model Prediction                                │
└─────────────────────────────────────────────────────────────────┘
```

During inference, relevant experiences are retrieved based on semantic similarity with the current task instruction, providing the world model with:
- Task-relevant action patterns from goal experiences
- Error-corrective knowledge from process experiences

---

## 🖥️ Installation

> **Note**: We need to set up two conda environments:
> - `worldmind` for EB-ALFRED and EB-Habitat
> - `worldmind_nav` for EB-Navigation
> 
> Please use SSH download instead of HTTP to avoid errors during git lfs pull.

### Environment Setup

#### 1. Clone Repository

```bash
git clone https://github.com/zjunlp/WorldMind.git
cd WorldMind
```

#### 2. Create Conda Environments

<details>
<summary><b>Option 1: Environment for ALFRED and Habitat (High-Level Planning)</b></summary>

```bash
# Create environment named 'worldmind' 
conda env create -f conda_envs/environment.yaml -n worldmind
conda activate worldmind

# Install the package
pip install -e .
```

</details>

<details>
<summary><b>Option 2: Environment for Navigation (Low-Level Navigation)</b></summary>

```bash
# Create environment named 'worldmind_nav'
conda env create -f conda_envs/environment_eb-nav.yaml -n worldmind_nav
conda activate worldmind_nav

# Install the package
pip install -e .
```

</details>

#### 3. Start Headless Server

For headless servers, start the X server in a separate `tmux` window:

```bash
conda activate worldmind
python -m embodiedbench.envs.eb_alfred.scripts.startx 1
```

### Task-Specific Setup

<details>
<summary><b>🏠 EB-ALFRED (Household Tasks)</b></summary>

**Download Dataset:**
```bash
conda activate worldmind
git clone https://huggingface.co/datasets/EmbodiedBench/EB-ALFRED
mv EB-ALFRED embodiedbench/envs/eb_alfred/data/json_2.1.0
```

**Verify Installation:**
```bash
# Remember to start the headless server first!
python -m embodiedbench.envs.eb_alfred.EBAlfEnv
```

</details>

<details>
<summary><b>🛋️ EB-Habitat (Rearrangement Tasks)</b></summary>

**1. Install Habitat Sim & Lab:**
```bash
conda activate worldmind

# Install Habitat-Sim with Bullet physics support
conda install -y habitat-sim==0.3.0 withbullet headless -c conda-forge -c aihabitat

# Install Habitat-Lab
git clone -b 'v0.3.0' --depth 1 https://github.com/facebookresearch/habitat-lab.git ./habitat-lab
cd ./habitat-lab
pip install -e habitat-lab
cd ..
```

**2. Download Datasets:**
```bash
conda install -y -c conda-forge git-lfs
python -m habitat_sim.utils.datasets_download --uids rearrange_task_assets
mv data embodiedbench/envs/eb_habitat
```

**3. Verify Installation:**
```bash
python -m embodiedbench.envs.eb_habitat.EBHabEnv
```

</details>

<details>
<summary><b>🧭 EB-Navigation (Vision-and-Language Navigation)</b></summary>

**Verify Installation:**
```bash
conda activate worldmind_nav
python -m embodiedbench.envs.eb_navigation.EBNavEnv
```

</details>

---

## 🚀 Quick Start

### Running Experiments

#### Basic Usage

```bash
python -m embodiedbench.main \
    --agent worldmind \
    --env <environment> \
    --model <model_name> \
    --eval_set <eval_set>
```

#### Environment-Specific Commands

| Environment | Command |
|-------------|---------|
| **ALFRED** | `python -m embodiedbench.main --agent worldmind --env alfred --model gpt-4o --eval_set valid_seen` |
| **Habitat** | `python -m embodiedbench.main --agent worldmind --env habitat --model gpt-4o --eval_set val` |
| **Navigation** | `python -m embodiedbench.main --agent worldmind_nav --env navigation --model gpt-4o --eval_set test` |

### Configuration

WorldMind uses YAML configuration files for experiment settings:

```yaml
# configs/eb-nav.yaml
model_name: gpt-4o-mini
model_type: remote
exp_name: navigation_baseline

# WorldMind Settings
enable_worldmind: True
use_vision_discriminator: false
use_experience_trajectory: true
detailed_output: true

# Goal Experience Settings
enable_goal_experience: true
goal_experience_top_k: 2

# Process Experience Settings
enable_process_experience: true
process_experience_top_k: 2

# Experience Refinement
enable_experience_refine: true
use_worldmind_template: true
```

#### Key Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `enable_worldmind` | Enable WorldMind components | `True` |
| `enable_goal_experience` | Enable goal experience retrieval | `True` |
| `goal_experience_top_k` | Number of goal experiences to retrieve | `3` |
| `enable_process_experience` | Enable process experience retrieval | `True` |
| `process_experience_top_k` | Number of process experiences to retrieve | `3` |
| `enable_experience_refine` | Enable LLM-based experience refinement | `False` |
| `use_vision_discriminator` | Use vision-based discrimination | `False` |

---

## 🌍 Environments

### 🏠 EB-ALFRED (Household Tasks)

A benchmark for grounded language learning in 3D household environments. Tasks require agents to execute multi-step instructions involving object manipulation.

**Task Types:**
- Pick & Place
- Examine Objects
- Clean & Heat Objects
- Toggle Appliances

**Evaluation Sets:** `base`, `valid_seen`, `valid_unseen`, `long_horizon`

### 🛋️ EB-Habitat (Rearrangement Tasks)

A simulation platform for embodied AI research focusing on object rearrangement tasks in realistic indoor environments.

**Task Types:**
- Object Goal Navigation
- Rearrangement Planning
- Multi-Object Manipulation

**Evaluation Sets:** `val`, `test`

### 🧭 EB-Navigation (Vision-and-Language Navigation)

A discrete navigation environment where agents must reach target locations through natural language instructions.

**Task Types:**
- Point Goal Navigation
- Object Goal Navigation
- Vision-Language Navigation

**Evaluation Sets:** `base`, `test`

---

## 🔌 WorldMind Plugin

For easy integration into your own projects, we provide a standalone plugin with modular components:

```python
from worldmind_plugin import (
    WorldMindConfig,
    ProcessExperienceModule,
    GoalExperienceModule,
    ExperienceRetrievalModule,
    ProcessTrajectoryStep,
    GoalTrajectoryStep
)

# Create configuration
config = WorldMindConfig(
    api_key="your-api-key",
    save_path="./worldmind_output"
)

# Initialize modules independently
process_module = ProcessExperienceModule(config)
goal_module = GoalExperienceModule(config)
retrieval_module = ExperienceRetrievalModule(config)

# Extract goal experience from successful trajectory
trajectory = [
    GoalTrajectoryStep(
        action="navigate_to(kitchen)",
        env_feedback="Arrived at kitchen",
        observation="Kitchen counter visible"
    ),
    # ... more steps
]

experience = goal_module.extract_experience(
    task_instruction="Go to the kitchen and get an apple",
    trajectory=trajectory
)

# Retrieve experiences for a new task
result = retrieval_module.retrieve(
    task_instruction="Find the coffee mug",
    enable_refine=True
)

# Use in agent prompt
agent_prompt = f"""You are a helpful assistant.

{result['formatted_prompt']}

Task: Find the coffee mug
"""
```

See [Plugin/README.md](Plugin/README.md) for detailed documentation.

---

## 📁 Project Structure

```
WorldMind/
├── 📂 embodiedbench/
│   ├── 📂 envs/                    # Environment implementations
│   │   ├── eb_alfred/              # ALFRED environment
│   │   ├── eb_habitat/             # Habitat environment
│   │   └── eb_navigation/          # Navigation environment
│   ├── 📂 evaluator/               # Evaluation scripts
│   │   ├── eb_alfred_evaluator.py
│   │   ├── eb_habitat_evaluator.py
│   │   └── eb_navigation_evaluator.py
│   ├── 📂 planner/                 # Base planner implementations
│   └── 📂 worldmind/               # WorldMind core modules
│       ├── alfred/                 # ALFRED integration
│       │   ├── discriminator.py
│       │   ├── reflector.py
│       │   ├── knowledge_manager.py
│       │   └── planner_wrapper.py
│       ├── habitat/                # Habitat integration
│       │   ├── discriminator.py
│       │   ├── reflector.py
│       │   ├── knowledge_manager.py
│       │   └── planner_wrapper.py
│       └── navigation/             # Navigation integration
│           ├── discriminator.py
│           ├── reflector.py
│           ├── knowledge_manager.py
│           ├── experience_refiner.py
│           └── planner_wrapper.py
├── 📂 Plugin/                      # Standalone WorldMind Plugin
│   ├── worldmind_plugin/
│   │   ├── core.py
│   │   ├── config.py
│   │   └── prompts.py
│   ├── example.py
│   └── README.md
├── 📂 configs/                     # Configuration files
│   ├── eb-nav.yaml
│   ├── eb-alfred.yaml
│   └── eb-habitat.yaml
├── 📂 assets/                      # Images and resources
│   └── framework.jpg
└── 📄 README.md
```

---

## 📊 Results

| Environment | Baseline | + WorldMind | Improvement |
|-------------|----------|-------------|-------------|
| **ALFRED (valid_seen)** | XX.X% | **XX.X%** | +X.X% |
| **ALFRED (valid_unseen)** | XX.X% | **XX.X%** | +X.X% |
| **Habitat (val)** | XX.X% | **XX.X%** | +X.X% |
| **Navigation (test)** | XX.X% | **XX.X%** | +X.X% |

> Detailed results and ablation studies available in our paper.

---

## 📝 Citation

If you find this work useful, please cite:

```bibtex
@article{worldmind2024,
  title={Aligning Agentic World Models via Knowledgeable Experience Learning},
  author={...},
  journal={arXiv preprint arXiv:2024.XXXXX},
  year={2024}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

We thank the following projects and teams:

- [EmbodiedBench](https://github.com/embodiedbench) for the evaluation framework
- [ALFRED](https://askforalfred.com/) for the household task benchmark
- [Habitat](https://aihabitat.org/) for the simulation platform
- [SentenceTransformers](https://www.sbert.net/) for semantic similarity

---

<div align="center">

**Made with ❤️ by [ZJUNLP](https://github.com/zjunlp)**

</div>

