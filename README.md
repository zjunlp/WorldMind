<div align="center">
  <img src="assets/worldmind_logo.png" width="60" alt="WorldMind Logo" style="vertical-align: bottom; margin-right: 10px;" />
  <span style="font-size: 50px; font-weight: bold; vertical-align: bottom;">
    <span style="color: #333333;">World</span><span style="color: #7c3aed;">Mind</span>
  </span>
</div>

<br/>

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

<p align="center">
  <img src="assets/experience_flow.png" alt="Experience Flow" width="80%"/>
</p>

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

### Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/WorldMind.git
cd WorldMind

# Install dependencies
pip install -r requirements.txt

# For semantic retrieval support
pip install sentence-transformers
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
