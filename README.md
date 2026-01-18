<div align="center">

<h1 align="center"> 🌐 WorldMind </h1>
<h3 align="center"> Aligning Agentic World Models via Knowledgeable Experience Learning </h3>

</div>

<div align="center">
  <img src="https://readme-typing-svg.herokuapp.com?font=Orbitron&size=24&duration=3000&pause=1000&color=00D9FF&center=true&vCenter=true&width=600&lines=Welcome+to+WorldMind;Aligning+Agentic+World+Models;Knowledgeable+Experience+Learning" alt="Typing Animation" />
</div>

<div align="center">
  <img src="./demo/lightagent_demo.gif" width="800" alt="演示动画">
</div>



<div align="center">

  <a href='https://github.com/zjunlp/WorldMind'>
    <img src='https://img.shields.io/badge/🦄_Awesome-List-fc60a8?style=for-the-badge&logo=awesome-lists&logoColor=white&labelColor=1a1a2e'>
  </a>

  <a href='https://zjunlp.github.io/WorldMind/'>
    <img src='https://img.shields.io/badge/🔥_Project-Page-00d9ff?style=for-the-badge&logo=github&logoColor=white&labelColor=1a1a2e'>
  </a>

  <a href='https://arxiv.org/abs/2026.xxxxx'>
    <img src='https://img.shields.io/badge/📄_arXiv-Paper-ff6b6b?style=for-the-badge&logo=arxiv&logoColor=white&labelColor=1a1a2e'>
  </a>

  <a href="https://github.com/zjunlp/WorldMind/stargazers">
    <img src='https://img.shields.io/github/stars/zjunlp/WorldMind?color=00d9ff&style=for-the-badge&logo=star&logoColor=white&labelColor=1a1a2e' />
  </a>

  <a href="https://github.com/zjunlp/WorldMind/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-4ecdc4?style=for-the-badge&logo=open-source-initiative&logoColor=white&labelColor=1a1a2e">
  </a>

</div>

<div align="center">
  <div style="width: 100%; height: 2px; margin: 20px 0; background: linear-gradient(90deg, transparent, #00d9ff, transparent);"></div>
</div>

<p align="center">
  <img src="assets/framework.jpg" alt="WorldMind Framework" width="90%"/>
</p>


**WorldMind** is a framework for aligning agentic world models through knowledgeable experience learning, enabling agents to learn directly from the environment.
<div align="center">
  
[📖 Overview](#-overview) • [🖥️ Installation](#️-installation) • [🚀 Quick Start](#-quick-start) • [🌍 Environments](#-environments) • [🔌 Plugin](#-worldmind-plugin) • [📁 Project Structure](#-project-structure) • [📊 Results](#-results) • [📝 Citation](#-citation) • [🙏 Acknowledgments](#-acknowledgments)

</div>

---

## 📖 Overview

**WorldMind** introduces a paradigm shift in how embodied AI agents learn and adapt. Unlike traditional approaches that rely on extensive environment interaction or domain-specific fine-tuning, WorldMind operates as a **training-free** framework that enables agents to:

- **Learn from Experience**: Extract reusable symbolic knowledge from both successful task completions and prediction errors without gradient updates.
- **Generalize Across Tasks**: Apply learned causal rules and heuristics to novel situations through semantic similarity-based retrieval.
- **Continuously Improve**: Accumulate and refine the World Knowledge Repository (WKR) throughout deployment.

### Key Features

| Feature | Description |
|---------|-------------|
| 🧠 **Experience Learning** | Combines *Goal Experience* (heuristics) from successful trajectories with *Process Experience* (causal boundaries) from prediction errors |
| 🔄 **Experience-Driven Alignment** | Uses **State Abstraction** and **Verifier** components to align world model predictions with actual environment dynamics |
| 🌐 **Universal Adaptability** | Seamlessly generalizes across diverse embodied environments (ALFRED, Habitat, Navigation) and tasks without specific fine-tuning |
| 🔌 **Modular Plugin** | Standalone plugin for easy integration into existing agent systems |

### Method

WorldMind introduces a two-stage approach for world model alignment:

**Stage 1** extracts knowledge during task execution (World Knowledge Building):
- **Goal Experience**: From successful trajectories, distill procedural heuristics to guide task optimality.
- **Process Experience**: Employ a **Predict-Act-Verify** loop. When a **Verifier** detects a semantic discrepancy between the predicted and actual abstract states, a **Self-Reflexion** mechanism synthesizes corrective causal rules.

**Stage 2** applies learned knowledge to new tasks (Inference via Constrained Simulation):
- Retrieve relevant *Process* and *Goal* experiences via semantic similarity.
- **Gated Simulation**: Selectively simulate outcomes only when target objects are grounded, enhancing inference efficiency.
- Augment world model prompts with retrieved knowledge to constrain planning within physical feasibility.

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


<summary><b>1️⃣ Environment for ALFRED and Habitat (High-Level Planning)</b></summary>

```bash
# Create environment named 'worldmind' 
conda env create -f conda_envs/environment.yaml 
conda activate worldmind
pip install -e .
```


<summary><b>2️⃣ Environment for Navigation (Low-Level Navigation)</b></summary>

```bash
# Create environment named 'worldmind_nav'
conda env create -f conda_envs/environment_eb-nav.yaml 
conda activate worldmind_nav
pip install -e .
```


#### 3. Start Headless Server

For headless servers, start the X server in a separate `tmux` window:

```bash
conda activate worldmind
python -m embodiedbench.envs.eb_alfred.scripts.startx 1
```

### Task-Specific Setup

<details>
<summary><b>🏠 EB-ALFRED (Household Tasks)</b></summary>

**Verify Installation:**
```bash
conda activate worldmind

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
cd ./habitat-lab
pip install -e habitat-lab
cd ..
```

**2. Verify Installation:**
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

We provide a universal run script `run.sh` for easy experiment execution. Simply configure the script and run:

```bash
#!/bin/bash
# WorldMind Universal Run Script
# Supports all three environments: Alfred (eb-alf), Habitat (eb-hab), Navigation (eb-nav)

set -e

# ============================================================
# ENVIRONMENT VARIABLES (Export Section)
# ============================================================

export DISPLAY=":1"
export CUDA_VISIBLE_DEVICES="0"
export OPENAI_API_KEY="your-openai-api-key"
export OPENAI_BASE_URL="your-openai-base-url"

# ============================================================
# CONFIGURATION PARAMETERS (Edit here)
# ============================================================

MODEL_NAME="gpt-3.5-turbo"   # Choose your model
ENV="eb-hab"              # Options: eb-alf, eb-hab, eb-nav
EXP_NAME="test"       # Your experiment name
ENABLE_WORLDMIND="True"   # True or False

# WorldMind component models (fixed to MODEL_NAME)
export WORLDMIND_DISCRIMINATOR_MODEL="$MODEL_NAME"
export WORLDMIND_SUMMARIZER_MODEL="$MODEL_NAME"
export WORLDMIND_REFLECTOR_MODEL="$MODEL_NAME"
export WORLDMIND_REFINER_MODEL="$MODEL_NAME"

# ============================================================
# VALIDATION
# ============================================================

if [ -z "$OPENAI_API_KEY" ]; then
    echo "=========================================="
    echo "ERROR: OPENAI_API_KEY not set!"
    echo "=========================================="
    exit 1
fi

case "$ENV" in
    eb-alf|eb-hab|eb-nav)
        echo "✓ Valid environment: $ENV"
        ;;
    *)
        echo "=========================================="
        echo "ERROR: Invalid environment '$ENV'"
        echo "=========================================="
        echo "Valid options: eb-alf, eb-hab, eb-nav"
        exit 1
        ;;
esac

# ============================================================
# DISPLAY CONFIGURATION
# ============================================================

echo ""
echo "=========================================="
echo "WorldMind Experiment Configuration"
echo "=========================================="
echo "Environment:     $ENV"
echo "Model:           $MODEL_NAME"
echo "Experiment:      $EXP_NAME"
echo "WorldMind:       $ENABLE_WORLDMIND"
echo "----------------------------------------"
echo "GPU Device:      $CUDA_VISIBLE_DEVICES"
echo "Display:         $DISPLAY"
echo "API Base URL:    $OPENAI_BASE_URL"
echo "=========================================="
echo ""

# ============================================================
# RUN EXPERIMENT
# ============================================================

python -m embodiedbench.main \
    env="$ENV" \
    model_name="$MODEL_NAME" \
    exp_name="$EXP_NAME" \
    enable_worldmind="$ENABLE_WORLDMIND"
```

**Usage:**

```bash
bash run.sh
```

### Configuration

WorldMind uses YAML configuration files for experiment settings. You can find and customize these files in the **[`WorldMind/embodiedbench/configs`](embodiedbench/configs)** directory.

<details>
<summary><b>📄 Click to view example configuration (`configs/eb-nav.yaml`)</b></summary>

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
</details>

#### Key Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `enable_worldmind` | Enable WorldMind components | `True` |
| `enable_goal_experience` | Enable goal experience retrieval | `True` |
| `goal_experience_top_k` | Number of goal experiences to retrieve | `2` |
| `enable_process_experience` | Enable process experience retrieval | `True` |
| `process_experience_top_k` | Number of process experiences to retrieve | `2` |
| `enable_experience_refine` | Enable LLM-based experience refinement | `True` |

---

## 🌍 Environments

### 🏠 EB-ALFRED (Household Tasks)

A benchmark for grounded language learning in 3D household environments. Tasks require agents to execute multi-step instructions involving object manipulation.

**Evaluation Metrics:** Success Rate (SR) and Goal Condition (GC)

**Evaluation Sets:** `Base`, `Common`, `Complex`, `Visual`, `Spatial`

### 🛋️ EB-Habitat (Rearrangement Tasks)

A simulation platform for embodied AI research focusing on object rearrangement tasks in realistic indoor environments.

**Evaluation Metrics:** Success Rate (SR) and Goal Condition (GC)

**Evaluation Sets:** `Base`, `Common`, `Complex`, `Visual`, `Spatial`

### 🧭 EB-Navigation (Vision-and-Language Navigation)

A discrete navigation environment where agents must reach target locations through natural language instructions.

**Evaluation Metrics:** Success Rate (SR)

**Evaluation Sets:** `Base`, `Common`, `Complex`, `Visual`

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
<tr><td colspan="13"><b>Open-source and Proprietary Models</b></td></tr>
<tr><td>GPT-4o</td><td>56.8</td><td>64.0</td><td>54.0</td><td>68.0</td><td>46.0</td><td>52.0</td><td>65.1</td><td>74.0</td><td>60.3</td><td>74.0</td><td>58.3</td><td>61.3</td></tr>
<tr><td>GPT-4o-mini</td><td>28.8</td><td>34.0</td><td>28.0</td><td>36.0</td><td>24.0</td><td>22.0</td><td>34.3</td><td>47.8</td><td>35.3</td><td>43.5</td><td>33.3</td><td>29.0</td></tr>
<tr><td>Claude-3.7-Sonnet</td><td>67.2</td><td>68.0</td><td>68.0</td><td>70.0</td><td>68.0</td><td>62.0</td><td>65.3</td><td>72.0</td><td>66.0</td><td>76.7</td><td>63.0</td><td>59.7</td></tr>
<tr><td>Gemini-1.5-Pro</td><td>63.2</td><td>70.0</td><td>64.0</td><td>72.0</td><td>58.0</td><td>52.0</td><td>67.4</td><td>74.3</td><td>66.7</td><td>76.5</td><td>62.8</td><td>59.0</td></tr>
<tr><td>Llama-3.2-90B-Vis</td><td>35.2</td><td>38.0</td><td>34.0</td><td>44.0</td><td>28.0</td><td>32.0</td><td>37.6</td><td>43.7</td><td>37.3</td><td>49.2</td><td>35.3</td><td>36.0</td></tr>
<tr><td>InternVL2.5-78B</td><td>37.0</td><td>41.0</td><td>40.0</td><td>39.0</td><td>16.0</td><td>49.0</td><td>41.0</td><td>42.3</td><td>35.3</td><td>43.3</td><td>35.7</td><td>40.3</td></tr>
<tr><td colspan="13"><b>GPT-3.5-turbo Based Methods</b></td></tr>
<tr><td>ReAct</td><td>44.4</td><td>52.0</td><td>48.0</td><td>52.0</td><td>32.0</td><td>38.0</td><td>50.4</td><td>55.3</td><td>53.5</td><td>55.3</td><td>42.7</td><td>45.0</td></tr>
<tr><td>BoN</td><td>42.8</td><td>46.0</td><td>42.0</td><td>50.0</td><td>42.0</td><td>34.0</td><td>50.4</td><td>54.2</td><td>46.5</td><td>56.5</td><td>52.0</td><td>42.8</td></tr>
<tr><td>SimuRA</td><td>45.2</td><td>50.0</td><td>42.0</td><td>54.0</td><td>38.0</td><td>42.0</td><td>53.6</td><td>57.8</td><td>47.8</td><td>59.7</td><td>48.5</td><td>54.3</td></tr>
<tr><td>ReasoningBank</td><td>41.6</td><td>50.0</td><td>36.0</td><td>44.0</td><td>36.0</td><td>42.0</td><td>47.6</td><td>57.5</td><td>41.5</td><td>47.0</td><td>44.2</td><td>48.0</td></tr>
<tr><td>Synapse</td><td>38.8</td><td>38.0</td><td>46.0</td><td>40.0</td><td>36.0</td><td>34.0</td><td>43.6</td><td>42.5</td><td>51.3</td><td>42.7</td><td>42.0</td><td>39.7</td></tr>
<tr><td>AWM</td><td>40.0</td><td>46.0</td><td>32.0</td><td>48.0</td><td>40.0</td><td>34.0</td><td>46.2</td><td>53.2</td><td>39.2</td><td>50.7</td><td>47.0</td><td>41.0</td></tr>
<tr><td><b>WorldMind</b></td><td><b>48.0</b></td><td><b>58.0</b></td><td><b>48.0</b></td><td><b>56.0</b></td><td>34.0</td><td><b>44.0</b></td><td><b>54.1</b></td><td><b>63.0</b></td><td>52.7</td><td><b>61.0</b></td><td>41.7</td><td><b>52.0</b></td></tr>
<tr><td colspan="13"><b>GPT-4.1-mini Based Methods</b></td></tr>
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
<tr><td colspan="13"><b>Open-source and Proprietary Models</b></td></tr>
<tr><td>GPT-4o</td><td>56.8</td><td>64.0</td><td>54.0</td><td>68.0</td><td>46.0</td><td>52.0</td><td>65.1</td><td>74.0</td><td>60.3</td><td>74.0</td><td>58.3</td><td>61.3</td></tr>
<tr><td>GPT-4o-mini</td><td>28.8</td><td>34.0</td><td>28.0</td><td>36.0</td><td>24.0</td><td>22.0</td><td>34.3</td><td>47.8</td><td>35.3</td><td>43.5</td><td>33.3</td><td>29.0</td></tr>
<tr><td>Claude-3.7-Sonnet</td><td>67.2</td><td>68.0</td><td>68.0</td><td>70.0</td><td>68.0</td><td>62.0</td><td>65.3</td><td>72.0</td><td>66.0</td><td>76.7</td><td>63.0</td><td>59.7</td></tr>
<tr><td>Gemini-1.5-Pro</td><td>63.2</td><td>70.0</td><td>64.0</td><td>72.0</td><td>58.0</td><td>52.0</td><td>67.4</td><td>74.3</td><td>66.7</td><td>76.5</td><td>62.8</td><td>59.0</td></tr>
<tr><td>Llama-3.2-90B-Vis</td><td>35.2</td><td>38.0</td><td>34.0</td><td>44.0</td><td>28.0</td><td>32.0</td><td>37.6</td><td>43.7</td><td>37.3</td><td>49.2</td><td>35.3</td><td>36.0</td></tr>
<tr><td>InternVL2.5-78B</td><td>37.0</td><td>41.0</td><td>40.0</td><td>39.0</td><td>16.0</td><td>49.0</td><td>41.0</td><td>42.3</td><td>35.3</td><td>43.3</td><td>35.7</td><td>40.3</td></tr>
<tr><td colspan="13"><b>GPT-3.5-turbo Based Methods</b></td></tr>
<tr><td>ReAct</td><td>44.4</td><td>52.0</td><td>48.0</td><td>52.0</td><td>32.0</td><td>38.0</td><td>50.4</td><td>55.3</td><td>53.5</td><td>55.3</td><td>42.7</td><td>45.0</td></tr>
<tr><td>BoN</td><td>42.8</td><td>46.0</td><td>42.0</td><td>50.0</td><td>42.0</td><td>34.0</td><td>50.4</td><td>54.2</td><td>46.5</td><td>56.5</td><td>52.0</td><td>42.8</td></tr>
<tr><td>SimuRA</td><td>45.2</td><td>50.0</td><td>42.0</td><td>54.0</td><td>38.0</td><td>42.0</td><td>53.6</td><td>57.8</td><td>47.8</td><td>59.7</td><td>48.5</td><td>54.3</td></tr>
<tr><td>ReasoningBank</td><td>41.6</td><td>50.0</td><td>36.0</td><td>44.0</td><td>36.0</td><td>42.0</td><td>47.6</td><td>57.5</td><td>41.5</td><td>47.0</td><td>44.2</td><td>48.0</td></tr>
<tr><td>Synapse</td><td>38.8</td><td>38.0</td><td>46.0</td><td>40.0</td><td>36.0</td><td>34.0</td><td>43.6</td><td>42.5</td><td>51.3</td><td>42.7</td><td>42.0</td><td>39.7</td></tr>
<tr><td>AWM</td><td>40.0</td><td>46.0</td><td>32.0</td><td>48.0</td><td>40.0</td><td>34.0</td><td>46.2</td><td>53.2</td><td>39.2</td><td>50.7</td><td>47.0</td><td>41.0</td></tr>
<tr><td><b>WorldMind</b></td><td><b>48.0</b></td><td><b>58.0</b></td><td><b>48.0</b></td><td><b>56.0</b></td><td>34.0</td><td><b>44.0</b></td><td><b>54.1</b></td><td><b>63.0</b></td><td>52.7</td><td><b>61.0</b></td><td>41.7</td><td><b>52.0</b></td></tr>
<tr><td colspan="13"><b>GPT-4.1-mini Based Methods</b></td></tr>
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

```

---

## 🙏 Acknowledgments


We thank the following projects and teams for their open-source contributions:

- [EmbodiedBench](https://github.com/embodiedbench) for the evaluation tasks
- [ALFRED](https://github.com/askforalfred/alfred) and [AI2-THOR](https://github.com/allenai/ai2thor) for the household task benchmark and simulation environment
- [Habitat](https://github.com/facebookresearch/habitat-lab) for the rearrangement simulation platform
- [vLLM](https://github.com/vllm-project/vllm) for efficient LLM inference and serving


---


