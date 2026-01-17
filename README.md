<div align="center">

# <img src="https://em-content.zobj.net/source/twitter/376/globe-with-meridians_1f310.png" width="35"/> WorldMind

### Aligning Agentic World Models via Knowledgeable Experience Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![arXiv](https://img.shields.io/badge/arXiv-2024.XXXXX-b31b1b.svg)](https://arxiv.org/abs/2024.XXXXX)

<p align="center">
  <img src="assets/framework.jpg" alt="WorldMind Framework" width="90%"/>
</p>

**WorldMind** is a novel framework for aligning agentic world models through knowledgeable experience learning, enabling agents to learn from both successful trajectories and prediction errors.

[📖 Overview](#-overview) • [🖥️ Installation](#️-installation) • [🚀 Quick Start](#-quick-start) • [🌍 Environments](#-environments) • [🔌 Plugin](#-worldmind-plugin) • [📊 Results](#-results) • [📝 Citation](#-citation)

</div>

---

## 📖 Overview

**WorldMind** introduces a paradigm shift in how embodied AI agents learn and adapt. Unlike traditional approaches that rely on extensive environment interaction or domain-specific fine-tuning, WorldMind enables agents to:

- **Learn from Experience**: Extract reusable knowledge from both successful task completions and prediction errors
- **Generalize Across Tasks**: Apply learned patterns to novel situations through semantic similarity-based retrieval
- **Continuously Improve**: Accumulate and refine knowledge throughout deployment

### Key Features

| Feature | Description |
|---------|-------------|
| �� **Dual Experience Learning** | Combines *Goal Experience* from successful trajectories with *Process Experience* from prediction errors |
| 🔄 **Experience-Driven Alignment** | Uses discriminator and reflector components to align world model predictions with actual environment dynamics |
| 🔍 **Semantic Retrieval** | Employs SentenceTransformer-based semantic similarity for efficient experience retrieval during task execution |
| 🌐 **Environment Agnostic** | Designed to work across different embodied AI environments (ALFRED, Habitat, Navigation) |
| 🔌 **Modular Plugin** | Standalone plugin for easy integration into existing agent systems |

### Method

WorldMind introduces a two-stage approach for world model alignment:

<table>
<tr>
<td width="50%">

#### 🔬 Stage 1: Experience Acquisition

```
╔══════════════════════════════════════════╗
║       EXPERIENCE ACQUISITION             ║
╠═══════════════════╦══════════════════════╣
║  �� Goal          ║  ⚙️ Process          ║
║  Experience       ║  Experience          ║
╠═══════════════════╬══════════════════════╣
║ ✓ Extract action  ║ ✓ Discriminator      ║
║   patterns        ║   identifies errors  ║
║ ✓ Capture task    ║ ✓ Reflector creates  ║
║   workflows       ║   corrections        ║
║ ✓ Generalize      ║ ✓ Build knowledge    ║
║   strategies      ║   base               ║
╚═══════════════════╩══════════════════════╝
```

</td>
<td width="50%">

#### 🚀 Stage 2: Experience-Guided Inference

```
╔══════════════════════════════════════════╗
║    EXPERIENCE-GUIDED INFERENCE           ║
╠══════════════════════════════════════════╣
║         📝 Task Instruction              ║
║                  ⬇                       ║
║         🔍 Semantic Search               ║
║           ╱         ╲                    ║
║    ┌─────────┐ ┌─────────┐              ║
║    │  Goal   │ │ Process │              ║
║    │   Exp   │ │   Exp   │              ║
║    └────┬────┘ └────┬────┘              ║
║         ╲         ╱                      ║
║          ⬇       ⬇                       ║
║    ✨ Experience Refinement              ║
║                  ⬇                       ║
║    📤 Augmented World Model              ║
╚══════════════════════════════════════════╝
```

</td>
</tr>
</table>

**Stage 1** extracts knowledge during task execution:
- **Goal Experience**: From successful trajectories, extract high-level action-outcome patterns
- **Process Experience**: When predictions fail, use discriminator to identify conflicts and reflector to generate corrections

**Stage 2** applies learned knowledge to new tasks:
- Retrieve relevant experiences via semantic similarity
- Optionally refine and merge experiences
- Augment world model prompts with learned patterns

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

**Task Types:** Pick & Place, Examine Objects, Clean & Heat Objects, Toggle Appliances

**Evaluation Sets:** `base`, `valid_seen`, `valid_unseen`, `long_horizon`

### 🛋️ EB-Habitat (Rearrangement Tasks)

A simulation platform for embodied AI research focusing on object rearrangement tasks in realistic indoor environments.

**Task Types:** Object Goal Navigation, Rearrangement Planning, Multi-Object Manipulation

**Evaluation Sets:** `val`, `test`

### 🧭 EB-Navigation (Vision-and-Language Navigation)

A discrete navigation environment where agents must reach target locations through natural language instructions.

**Task Types:** Point Goal Navigation, Object Goal Navigation, Vision-Language Navigation

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
│   └── 📂 worldmind/               # WorldMind core modules
│       ├── alfred/                 # ALFRED integration
│       ├── habitat/                # Habitat integration
│       └── navigation/             # Navigation integration
├── 📂 Plugin/                      # Standalone WorldMind Plugin
├── 📂 configs/                     # Configuration files
├── 📂 assets/                      # Images and resources
└── 📄 README.md
```

---

## 📊 Results

### EB-ALFRED Results

<table>
<thead>
<tr>
<th rowspan="2">Model</th>
<th colspan="6">Success Rate (SR) %</th>
<th colspan="6">Goal Condition (GC) %</th>
</tr>
<tr>
<th>Avg</th><th>Base</th><th>Common</th><th>Complex</th><th>Visual</th><th>Spatial</th>
<th>Avg</th><th>Base</th><th>Common</th><th>Complex</th><th>Visual</th><th>Spatial</th>
</tr>
</thead>
<tbody>
<tr><td colspan="13"><b>Proprietary Models</b></td></tr>
<tr><td>GPT-4o</td><td>56.8</td><td>64.0</td><td>54.0</td><td>68.0</td><td>46.0</td><td>52.0</td><td>65.1</td><td>74.0</td><td>60.3</td><td>74.0</td><td>58.3</td><td>61.3</td></tr>
<tr><td>GPT-4o-mini</td><td>28.8</td><td>34.0</td><td>28.0</td><td>36.0</td><td>24.0</td><td>22.0</td><td>34.3</td><td>47.8</td><td>35.3</td><td>43.5</td><td>33.3</td><td>29.0</td></tr>
<tr><td>Claude-3.7-Sonnet</td><td>67.2</td><td>68.0</td><td>68.0</td><td>70.0</td><td>68.0</td><td>62.0</td><td>65.3</td><td>72.0</td><td>66.0</td><td>76.7</td><td>63.0</td><td>59.7</td></tr>
<tr><td>Gemini-1.5-Pro</td><td>63.2</td><td>70.0</td><td>64.0</td><td>72.0</td><td>58.0</td><td>52.0</td><td>67.4</td><td>74.3</td><td>66.7</td><td>76.5</td><td>62.8</td><td>59.0</td></tr>
<tr><td>Llama-3.2-90B-Vis</td><td>35.2</td><td>38.0</td><td>34.0</td><td>44.0</td><td>28.0</td><td>32.0</td><td>37.6</td><td>43.7</td><td>37.3</td><td>49.2</td><td>35.3</td><td>36.0</td></tr>
<tr><td>InternVL2.5-78B</td><td>37.0</td><td>41.0</td><td>40.0</td><td>39.0</td><td>16.0</td><td>49.0</td><td>41.0</td><td>42.3</td><td>35.3</td><td>43.3</td><td>35.7</td><td>40.3</td></tr>
<tr><td colspan="13"><b>GPT-3.5 Based Methods</b></td></tr>
<tr><td>ReAct</td><td>44.4</td><td>52.0</td><td>48.0</td><td>52.0</td><td>32.0</td><td>38.0</td><td>50.4</td><td>55.3</td><td>53.5</td><td>55.3</td><td>42.7</td><td>45.0</td></tr>
<tr><td>BoN</td><td>42.8</td><td>46.0</td><td>42.0</td><td>50.0</td><td>42.0</td><td>34.0</td><td>50.4</td><td>54.2</td><td>46.5</td><td>56.5</td><td>52.0</td><td>42.8</td></tr>
<tr><td>SimuRA</td><td>45.2</td><td>50.0</td><td>42.0</td><td>54.0</td><td>38.0</td><td>42.0</td><td>53.6</td><td>57.8</td><td>47.8</td><td>59.7</td><td>48.5</td><td>54.3</td></tr>
<tr><td>ReasoningBank</td><td>41.6</td><td>50.0</td><td>36.0</td><td>44.0</td><td>36.0</td><td>42.0</td><td>47.6</td><td>57.5</td><td>41.5</td><td>47.0</td><td>44.2</td><td>48.0</td></tr>
<tr><td>Synapse</td><td>38.8</td><td>38.0</td><td>46.0</td><td>40.0</td><td>36.0</td><td>34.0</td><td>43.6</td><td>42.5</td><td>51.3</td><td>42.7</td><td>42.0</td><td>39.7</td></tr>
<tr><td>AWM</td><td>40.0</td><td>46.0</td><td>32.0</td><td>48.0</td><td>40.0</td><td>34.0</td><td>46.2</td><td>53.2</td><td>39.2</td><td>50.7</td><td>47.0</td><td>41.0</td></tr>
<tr><td><b>WorldMind</b></td><td><b>48.0</b></td><td><b>58.0</b></td><td><b>48.0</b></td><td><b>56.0</b></td><td>34.0</td><td><b>44.0</b></td><td><b>54.1</b></td><td><b>63.0</b></td><td>52.7</td><td><b>61.0</b></td><td>41.7</td><td><b>52.0</b></td></tr>
<tr><td colspan="13"><b>GPT-4.1 Based Methods</b></td></tr>
<tr><td>ReAct</td><td>41.2</td><td>50.0</td><td>40.0</td><td>46.0</td><td>38.0</td><td>32.0</td><td>47.5</td><td>55.3</td><td>42.8</td><td>52.2</td><td>47.2</td><td>39.8</td></tr>
<tr><td>BoN</td><td>44.4</td><td>46.0</td><td>44.0</td><td>50.0</td><td>42.0</td><td>40.0</td><td>49.5</td><td>50.8</td><td>48.3</td><td>54.7</td><td>48.7</td><td>45.0</td></tr>
<tr><td>SimuRA</td><td>45.6</td><td>52.0</td><td>44.0</td><td>54.0</td><td>38.0</td><td>40.0</td><td>52.2</td><td>61.0</td><td>50.3</td><td>58.2</td><td>45.3</td><td>46.3</td></tr>
<tr><td>ReasoningBank</td><td>38.0</td><td>42.0</td><td>36.0</td><td>42.0</td><td>34.0</td><td>36.0</td><td>42.6</td><td>46.7</td><td>38.8</td><td>45.8</td><td>41.5</td><td>40.3</td></tr>
<tr><td>Synapse</td><td>37.2</td><td>40.0</td><td>32.0</td><td>44.0</td><td>36.0</td><td>34.0</td><td>42.2</td><td>41.2</td><td>37.5</td><td>49.5</td><td>41.3</td><td>41.7</td></tr>
<tr><td>AWM</td><td>41.2</td><td>44.0</td><td>36.0</td><td>48.0</td><td>38.0</td><td>40.0</td><td>46.0</td><td>48.3</td><td>42.0</td><td>52.5</td><td>44.3</td><td>42.7</td></tr>
<tr><td><b>WorldMind</b></td><td><b>49.2</b></td><td>50.0</td><td><b>58.0</b></td><td><b>54.0</b></td><td><b>42.0</b></td><td><b>42.0</b></td><td><b>55.7</b></td><td><b>61.0</b></td><td><b>61.0</b></td><td><b>58.8</b></td><td><b>48.0</b></td><td><b>49.7</b></td></tr>
</tbody>
</table>

### EB-Habitat Results

<table>
<thead>
<tr>
<th rowspan="2">Model</th>
<th colspan="6">Success Rate (SR) %</th>
<th colspan="6">Goal Condition (GC) %</th>
</tr>
<tr>
<th>Avg</th><th>Base</th><th>Common</th><th>Complex</th><th>Visual</th><th>Spatial</th>
<th>Avg</th><th>Base</th><th>Common</th><th>Complex</th><th>Visual</th><th>Spatial</th>
</tr>
</thead>
<tbody>
<tr><td colspan="13"><b>Proprietary Models</b></td></tr>
<tr><td>GPT-4o</td><td>56.8</td><td>64.0</td><td>54.0</td><td>68.0</td><td>46.0</td><td>52.0</td><td>65.1</td><td>74.0</td><td>60.3</td><td>74.0</td><td>58.3</td><td>61.3</td></tr>
<tr><td>GPT-4o-mini</td><td>28.8</td><td>34.0</td><td>28.0</td><td>36.0</td><td>24.0</td><td>22.0</td><td>34.3</td><td>47.8</td><td>35.3</td><td>43.5</td><td>33.3</td><td>29.0</td></tr>
<tr><td>Claude-3.7-Sonnet</td><td>67.2</td><td>68.0</td><td>68.0</td><td>70.0</td><td>68.0</td><td>62.0</td><td>65.3</td><td>72.0</td><td>66.0</td><td>76.7</td><td>63.0</td><td>59.7</td></tr>
<tr><td>Gemini-1.5-Pro</td><td>63.2</td><td>70.0</td><td>64.0</td><td>72.0</td><td>58.0</td><td>52.0</td><td>67.4</td><td>74.3</td><td>66.7</td><td>76.5</td><td>62.8</td><td>59.0</td></tr>
<tr><td>Llama-3.2-90B-Vis</td><td>35.2</td><td>38.0</td><td>34.0</td><td>44.0</td><td>28.0</td><td>32.0</td><td>37.6</td><td>43.7</td><td>37.3</td><td>49.2</td><td>35.3</td><td>36.0</td></tr>
<tr><td>InternVL2.5-78B</td><td>37.0</td><td>41.0</td><td>40.0</td><td>39.0</td><td>16.0</td><td>49.0</td><td>41.0</td><td>42.3</td><td>35.3</td><td>43.3</td><td>35.7</td><td>40.3</td></tr>
<tr><td colspan="13"><b>GPT-3.5 Based Methods</b></td></tr>
<tr><td>ReAct</td><td>44.4</td><td>52.0</td><td>48.0</td><td>52.0</td><td>32.0</td><td>38.0</td><td>50.4</td><td>55.3</td><td>53.5</td><td>55.3</td><td>42.7</td><td>45.0</td></tr>
<tr><td>BoN</td><td>42.8</td><td>46.0</td><td>42.0</td><td>50.0</td><td>42.0</td><td>34.0</td><td>50.4</td><td>54.2</td><td>46.5</td><td>56.5</td><td>52.0</td><td>42.8</td></tr>
<tr><td>SimuRA</td><td>45.2</td><td>50.0</td><td>42.0</td><td>54.0</td><td>38.0</td><td>42.0</td><td>53.6</td><td>57.8</td><td>47.8</td><td>59.7</td><td>48.5</td><td>54.3</td></tr>
<tr><td>ReasoningBank</td><td>41.6</td><td>50.0</td><td>36.0</td><td>44.0</td><td>36.0</td><td>42.0</td><td>47.6</td><td>57.5</td><td>41.5</td><td>47.0</td><td>44.2</td><td>48.0</td></tr>
<tr><td>Synapse</td><td>38.8</td><td>38.0</td><td>46.0</td><td>40.0</td><td>36.0</td><td>34.0</td><td>43.6</td><td>42.5</td><td>51.3</td><td>42.7</td><td>42.0</td><td>39.7</td></tr>
<tr><td>AWM</td><td>40.0</td><td>46.0</td><td>32.0</td><td>48.0</td><td>40.0</td><td>34.0</td><td>46.2</td><td>53.2</td><td>39.2</td><td>50.7</td><td>47.0</td><td>41.0</td></tr>
<tr><td><b>WorldMind</b></td><td><b>48.0</b></td><td><b>58.0</b></td><td><b>48.0</b></td><td><b>56.0</b></td><td>34.0</td><td><b>44.0</b></td><td><b>54.1</b></td><td><b>63.0</b></td><td>52.7</td><td><b>61.0</b></td><td>41.7</td><td><b>52.0</b></td></tr>
<tr><td colspan="13"><b>GPT-4.1 Based Methods</b></td></tr>
<tr><td>ReAct</td><td>41.2</td><td>50.0</td><td>40.0</td><td>46.0</td><td>38.0</td><td>32.0</td><td>47.5</td><td>55.3</td><td>42.8</td><td>52.2</td><td>47.2</td><td>39.8</td></tr>
<tr><td>BoN</td><td>44.4</td><td>46.0</td><td>44.0</td><td>50.0</td><td>42.0</td><td>40.0</td><td>49.5</td><td>50.8</td><td>48.3</td><td>54.7</td><td>48.7</td><td>45.0</td></tr>
<tr><td>SimuRA</td><td>45.6</td><td>52.0</td><td>44.0</td><td>54.0</td><td>38.0</td><td>40.0</td><td>52.2</td><td>61.0</td><td>50.3</td><td>58.2</td><td>45.3</td><td>46.3</td></tr>
<tr><td>ReasoningBank</td><td>38.0</td><td>42.0</td><td>36.0</td><td>42.0</td><td>34.0</td><td>36.0</td><td>42.6</td><td>46.7</td><td>38.8</td><td>45.8</td><td>41.5</td><td>40.3</td></tr>
<tr><td>Synapse</td><td>37.2</td><td>40.0</td><td>32.0</td><td>44.0</td><td>36.0</td><td>34.0</td><td>42.2</td><td>41.2</td><td>37.5</td><td>49.5</td><td>41.3</td><td>41.7</td></tr>
<tr><td>AWM</td><td>41.2</td><td>44.0</td><td>36.0</td><td>48.0</td><td>38.0</td><td>40.0</td><td>46.0</td><td>48.3</td><td>42.0</td><td>52.5</td><td>44.3</td><td>42.7</td></tr>
<tr><td><b>WorldMind</b></td><td><b>49.2</b></td><td>50.0</td><td><b>58.0</b></td><td><b>54.0</b></td><td><b>42.0</b></td><td><b>42.0</b></td><td><b>55.7</b></td><td><b>61.0</b></td><td><b>61.0</b></td><td><b>58.8</b></td><td><b>48.0</b></td><td><b>49.7</b></td></tr>
</tbody>
</table>

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

## 🙏 Acknowledgments

We thank the following projects and teams:

- [EmbodiedBench](https://github.com/embodiedbench) for the evaluation framework
- [ALFRED](https://askforalfred.com/) for the household task benchmark
- [Habitat](https://aihabitat.org/) for the simulation platform

---

<div align="center">

**Made with ❤️ by [ZJUNLP](https://github.com/zjunlp)**

</div>

