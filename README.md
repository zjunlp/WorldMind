# Aligning Agentic World Models via Knowledgeable Experience Learning

<p align="center">
  <img src="assets/framework.jpg" alt="WorldMind Framework" width="100%"/>
</p>

## 📖 Overview

**WorldMind** is a novel framework for aligning agentic world models through knowledgeable experience learning. Unlike traditional approaches that rely on extensive environment interaction or domain-specific fine-tuning, WorldMind enables agents to learn from both successful trajectories and prediction errors, accumulating reusable knowledge that generalizes across tasks.

### Key Features

- **Dual Experience Learning**: Combines *Goal Experience* from successful trajectories with *Process Experience* from prediction errors
- **Experience-Driven Alignment**: Uses discriminator and reflector components to align world model predictions with actual environment dynamics
- **Semantic Retrieval**: Employs semantic similarity for efficient experience retrieval during task execution
- **Environment Agnostic**: Designed to work across different embodied AI environments

## 🔬 Method

WorldMind introduces a two-stage approach for world model alignment:

### Stage 1: Experience Acquisition

1. **Goal Experience Extraction**: From successful task trajectories, extract high-level action-outcome patterns and task-specific knowledge
2. **Process Experience Extraction**: During task execution, when the world model makes incorrect predictions, use a discriminator to identify conflicts and a reflector to generate corrective knowledge

### Stage 2: Experience-Guided Inference

During inference, relevant experiences are retrieved based on semantic similarity with the current task instruction, providing the world model with:
- Task-relevant action patterns from goal experiences
- Error-corrective knowledge from process experiences


## 📁 Project Structure

```
WorldMind/
├── embodiedbench/
│   └── worldmind/
│       ├── alfred/              # ALFRED environment integration
│       │   ├── agent.py         # WorldMind agent for ALFRED
│       │   └── knowledge_manager.py  # Knowledge management
│       ├── habitat/             # Habitat environment integration  
│       │   ├── agent.py         # WorldMind agent for Habitat
│       │   └── knowledge_manager.py  # Knowledge management
│       └── navigation/          # Navigation environment integration
│           ├── agent.py         # WorldMind agent for Navigation
│           └── knowledge_manager.py  # Knowledge management
├── Plugin/                      # Standalone WorldMind Plugin
│   ├── worldmind_plugin/        # Core plugin module
│   │   ├── core.py              # Main implementation
│   │   ├── config.py            # Configuration
│   │   └── prompts.py           # Prompt templates
│   ├── example.py               # Usage examples
│   └── README.md                # Plugin documentation
└── assets/                      # Images and resources
    └── framework.jpg            # Framework diagram
```

## 🚀 Quick Start

## 🖥️ Installation

**Note: We need to set up two conda environments: `worldmind` (for EB-ALFRED and EB-Habitat) and `worldmind_nav` (for EB-Navigation). Please use ssh download instead of HTTP download to avoid errors during git lfs pull.**

### 1. Clone Repository & Git LFS

```bash
git clone [https://github.com/zjunlp/WorldMind.git](https://github.com/zjunlp/WorldMind.git)
cd WorldMind

# Initialize Git LFS for large datasets
git lfs install
git lfs pull

```

### 2. Set up Environments

**Option 1: Environment for `Alfred and Habitat**`
This environment is used for high-level planning tasks.

```bash
# Create environment named 'worldmind' (forcing name with -n)
conda env create -f conda_envs/environment.yaml -n worldmind
conda activate worldmind

# Install the package
pip install -e .

# Install dependencies for Semantic Retrieval
pip install sentence-transformers

```

**Option 2: Environment for `Navigation**`
This environment is used for low-level navigation tasks.

```bash
# Create environment named 'worldmind_nav' (forcing name with -n)
conda env create -f conda_envs/environment_eb-nav.yaml -n worldmind_nav
conda activate worldmind_nav

# Install the package
pip install -e .

```

---

### 3. Start Headless Server

Please run the `startx.py` script before running experiments on headless servers. The server should be started in a separate `tmux` window. We use `X_DISPLAY id=1` by default.

```bash
# Ensure you are in the worldmind environment
conda activate worldmind
python -m embodiedbench.envs.eb_alfred.scripts.startx 1

```

---

### 4. Task-Specific Setup

#### 🏠 EB-ALFRED (Household Tasks)

Download the dataset from Hugging Face.

```bash
conda activate worldmind
git clone [https://huggingface.co/datasets/EmbodiedBench/EB-ALFRED](https://huggingface.co/datasets/EmbodiedBench/EB-ALFRED)
mv EB-ALFRED embodiedbench/envs/eb_alfred/data/json_2.1.0

```

Run the following code to verify the EB-ALFRED environment is working correctly (**remember to start the headless server first**):

```bash
conda activate worldmind
python -m embodiedbench.envs.eb_alfred.EBAlfEnv

```

#### 🛋️ EB-Habitat (Rearrangement Tasks)

1. **Install Habitat Sim & Lab:**

```bash
conda activate worldmind

# Install Habitat-Sim with Bullet physics support
conda install -y habitat-sim==0.3.0 withbullet headless -c conda-forge -c aihabitat

# Install Habitat-Lab
git clone -b 'v0.3.0' --depth 1 [https://github.com/facebookresearch/habitat-lab.git](https://github.com/facebookresearch/habitat-lab.git) ./habitat-lab
cd ./habitat-lab
pip install -e habitat-lab
cd ..

```

2. **Download YCB and ReplicaCAD datasets:**

```bash
conda install -y -c conda-forge git-lfs
python -m habitat_sim.utils.datasets_download --uids rearrange_task_assets
mv data embodiedbench/envs/eb_habitat

```

*Note: After this step, there should be a `data` folder under `embodiedbench/envs/eb_habitat`.*

3. **Verify Installation:**

```bash
conda activate worldmind
python -m embodiedbench.envs.eb_habitat.EBHabEnv

```

#### 🧭 EB-Navigation (Vision-and-Language Navigation)

Run the following code to ensure the EB-Navigation environment is working correctly:

```bash
conda activate worldmind_nav
python -m embodiedbench.envs.eb_navigation.EBNavEnv

```

```








### Running Experiments

#### ALFRED Environment

```bash
python -m embodiedbench.main \
    --agent worldmind \
    --env alfred \
    --model gpt-4o \
    --eval_set valid_seen
```

#### Habitat Environment

```bash
python -m embodiedbench.main \
    --agent worldmind \
    --env habitat \
    --model gpt-4o \
    --eval_set val
```

#### Navigation Environment

```bash
python -m embodiedbench.main \
    --agent worldmind_nav \
    --env navigation \
    --model gpt-4o \
    --eval_set test
```

## 🔌 WorldMind Plugin

For easy integration into your own projects, we provide a standalone plugin:

```python
from worldmind_plugin import (
    ProcessExperienceModule,
    GoalExperienceModule,
    ExperienceRetrievalModule,
    WorldMindConfig
)

# Initialize configuration
config = WorldMindConfig(
    model="gpt-4o",
    api_key="your-api-key"
)

# Initialize modules
process_module = ProcessExperienceModule(config)
goal_module = GoalExperienceModule(config)
retrieval_module = ExperienceRetrievalModule(config)

# Extract goal experience from successful trajectory
goal_exp = await goal_module.extract_goal_experience(
    instruction="Navigate to the kitchen",
    trajectory=[...]  # Your trajectory data
)

# Retrieve relevant experience for new task
experiences = await retrieval_module.retrieve_experience(
    instruction="Go to the bedroom",
    knowledge_base=knowledge_base
)
```

See [Plugin/README.md](Plugin/README.md) for detailed documentation.

## 📊 Environments

### ALFRED

A benchmark for grounded language learning in 3D household environments. Tasks require agents to execute multi-step instructions involving object manipulation.

### Habitat

A simulation platform for embodied AI research. We use the Object Goal Navigation task where agents navigate to target objects.

### Navigation

A discrete navigation environment where agents must reach target locations through natural language instructions.

## 📈 Results

| Environment | Agent | Success Rate |
|-------------|-------|--------------|
| ALFRED | GPT-4o | XX.X% |
| ALFRED | GPT-4o + WorldMind | **XX.X%** |
| Habitat | GPT-4o | XX.X% |
| Habitat | GPT-4o + WorldMind | **XX.X%** |
| Navigation | GPT-4o | XX.X% |
| Navigation | GPT-4o + WorldMind | **XX.X%** |

## 🛠️ Configuration

Key configuration options in `config.py`:

```python
WorldMindConfig(
    # Model settings
    model="gpt-4o",           # LLM model to use
    api_key="...",            # API key
    
    # Experience settings
    top_k_experiences=5,      # Number of experiences to retrieve
    similarity_threshold=0.5, # Minimum similarity for retrieval
    
    # Generation settings
    temperature=0.7,          # LLM temperature
    max_tokens=4096,          # Maximum response tokens
)
```

## 📝 Citation

If you find this work useful, please cite:

```bibtex
@article{worldmind2024,
  title={Aligning Agentic World Models via Knowledgeable Experience Learning},
  author={...},
  journal={...},
  year={2024}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [EmbodiedBench](https://github.com/embodiedbench) for the evaluation framework
- [ALFRED](https://askforalfred.com/) for the household task benchmark
- [Habitat](https://aihabitat.org/) for the simulation platform
