<div align="center">

# 🔌 WorldMind Plugin

</div>
---

## 📋 Table of Contents

- [Overview](#-overview)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Modules](#-modules)
  - [ProcessExperienceModule](#module-1-processexperiencemodule)
  - [GoalExperienceModule](#module-2-goalexperiencemodule)
  - [ExperienceRetrievalModule](#module-3-experienceretrievalmodule)
- [Configuration](#configuration)
- [Complete Integration Example](#-complete-integration-example)
- [API Reference](#-api-reference)

---

## 📖 Overview

**WorldMind Plugin** provides three independent modules for experience-based learning that can be easily integrated into various agent systems:

| Module | Purpose | When to Use |
|--------|---------|-------------|
| 🔄 **ProcessExperienceModule** | Extracts process experience from prediction errors | During task execution when errors occur |
| 🎯 **GoalExperienceModule** | Extracts goal experience from successful trajectories | After successful task completion |
| 🔍 **ExperienceRetrievalModule** | Retrieves and optionally refines experiences | Before starting a new task |

Each module operates independently, allowing flexible integration into your existing agent architecture.

---

## 📦 Installation

```bash
# Install core dependencies
pip install openai

# Optional: For semantic similarity-based retrieval (recommended)
pip install sentence-transformers
```

---

## 🚀 Quick Start

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

# Use modules independently
process_module = ProcessExperienceModule(config)
goal_module = GoalExperienceModule(config)
retrieval_module = ExperienceRetrievalModule(config)
```

---

## 📚 Modules

### Module 1: ProcessExperienceModule

Extracts process experience from prediction errors during task execution. When the agent's predicted state differs from the actual environment feedback, this module generates corrective knowledge.

#### Input/Output

| Parameter | Type | Description |
|-----------|------|-------------|
| `task_instruction` | `str` | The task instruction being executed |
| `trajectory` | `List[ProcessTrajectoryStep]` | List of trajectory steps |
| **Returns** | `List[str]` | Extracted experience entries |

#### ProcessTrajectoryStep Schema

```python
ProcessTrajectoryStep(
    observation: str,      # Current state/observation (text description)
    action: str,           # Action executed (name or description)
    predicted_state: str,  # Agent's predicted state after action
    env_feedback: str      # Environment's actual feedback after action
)
```

#### Storage Format

```json
{
    "instruction": "task instruction text",
    "knowledge": [
        "Experience: ... Lesson: ...",
        "Experience: ... Lesson: ..."
    ]
}
```

#### Usage Example

```python
from worldmind_plugin import ProcessExperienceModule, ProcessTrajectoryStep

process_module = ProcessExperienceModule(config)

# Build trajectory
trajectory = [
    ProcessTrajectoryStep(
        observation="Form page loaded. All fields empty.",
        action="Submit form",
        predicted_state="Form submitted successfully",
        env_feedback="Error: Required fields are empty"
    )
]

# Process entire trajectory
experiences = process_module.process_trajectory(
    task_instruction="Complete the registration form",
    trajectory=trajectory
)

# Or process single step (for real-time use)
has_error, experiences = process_module.process_single_step(
    task_instruction="Complete the registration form",
    step=trajectory[0],
    action_history=["Previous action 1", "Previous action 2"]
)
```

---

### Module 2: GoalExperienceModule

Extracts goal experience from successful task trajectories. Call this module when a task completes successfully to capture the effective workflow.

#### Input/Output

| Parameter | Type | Description |
|-----------|------|-------------|
| `task_instruction` | `str` | The completed task instruction |
| `trajectory` | `List[GoalTrajectoryStep]` | List of trajectory steps |
| **Returns** | `str` | Extracted goal experience text |

#### GoalTrajectoryStep Schema

```python
GoalTrajectoryStep(
    action: str,          # Action executed
    env_feedback: str,    # Environment's feedback (can be empty)
    observation: str      # Optional observation (included if configured)
)
```

#### Storage Format

```json
{
    "instruction": "task instruction text",
    "goal_experience": "Workflow description..."
}
```

#### Usage Example

```python
from worldmind_plugin import GoalExperienceModule, GoalTrajectoryStep

goal_module = GoalExperienceModule(config)

# Build trajectory (call when task succeeds)
trajectory = [
    GoalTrajectoryStep(
        action="Open email client",
        env_feedback="Email client opened",
        observation="Main inbox displayed"  # Optional
    ),
    GoalTrajectoryStep(
        action="Compose new email",
        env_feedback="Draft opened"
    ),
    GoalTrajectoryStep(
        action="Send email",
        env_feedback="Email sent successfully"
    )
]

# Extract experience
experience = goal_module.extract_experience(
    task_instruction="Send email to user@example.com",
    trajectory=trajectory
)
```

---

### Module 3: ExperienceRetrievalModule

Retrieves and optionally refines experiences for injection into agent prompts. Use this before starting a new task to augment the agent's context.

#### Input/Output

| Parameter | Type | Description |
|-----------|------|-------------|
| `task_instruction` | `str` | Current task instruction |
| `enable_refine` | `bool` | Override config to enable/disable refinement |

| Output Field | Type | Description |
|--------------|------|-------------|
| `goal_experiences` | `List[Dict]` | Retrieved goal experiences |
| `process_experiences` | `List[Dict]` | Retrieved process experiences |
| `refined_experience` | `Dict` | Consolidated experience (if refine enabled) |
| `formatted_prompt` | `str` | Ready-to-inject prompt text |

#### Usage Example

```python
from worldmind_plugin import ExperienceRetrievalModule

retrieval_module = ExperienceRetrievalModule(config)

# Retrieve experiences
result = retrieval_module.retrieve(
    task_instruction="Book a hotel room in Paris",
    enable_refine=True
)

# Inject into agent prompt
agent_prompt = f"""You are a helpful assistant.

{result['formatted_prompt']}

Task: Book a hotel room in Paris
"""

# Reload after new experiences are saved
retrieval_module.reload_experiences()
```

---

## <a id="configuration"></a>⚙️ Configuration

```python
from worldmind_plugin import WorldMindConfig

config = WorldMindConfig(
    # ═══════════════════════════════════════════
    # API Configuration
    # ═══════════════════════════════════════════
    api_key="your-api-key",               # Required
    api_base="https://api.openai.com/v1", # Optional: custom endpoint
    
    # ═══════════════════════════════════════════
    # Model Configuration
    # ═══════════════════════════════════════════
    discriminator_model="gpt-4o-mini",    # For prediction comparison
    reflector_model="gpt-4o-mini",        # For error analysis
    summarizer_model="gpt-4o",            # Vision model for multimodal
    extractor_model="gpt-4o-mini",        # For goal experience extraction
    refiner_model="gpt-4o-mini",          # For experience consolidation
    
    # ═══════════════════════════════════════════
    # Multimodal Configuration
    # ═══════════════════════════════════════════
    is_multimodal=False,                  # Use vision for state summarization
    
    # ═══════════════════════════════════════════
    # Experience Retrieval Configuration
    # ═══════════════════════════════════════════
    enable_experience_refine=True,        # Consolidate retrieved experiences
    goal_experience_top_k=3,              # Number of goal experiences to retrieve
    process_experience_top_k=5,           # Number of process experiences to retrieve
    
    # ═══════════════════════════════════════════
    # Trajectory Configuration
    # ═══════════════════════════════════════════
    goal_trajectory_include_observation=True,  # Include observation in goal trajectory
    use_env_feedback=True,                # Use env feedback in reflection
    
    # ═══════════════════════════════════════════
    # Save Configuration
    # ═══════════════════════════════════════════
    save_path="./worldmind_output",       # Base path for all saves
    
    # ═══════════════════════════════════════════
    # Output Configuration
    # ═══════════════════════════════════════════
    detailed_output=True                  # Verbose logging
)
```

---

## 🔧 Complete Integration Example

```python
from worldmind_plugin import (
    WorldMindConfig,
    ProcessExperienceModule,
    GoalExperienceModule,
    ExperienceRetrievalModule,
    ProcessTrajectoryStep,
    GoalTrajectoryStep,
    PREDICTION_CONSTRAINT_PROMPT,
    EXPLORATION_PHASE_MARKER
)

# ═══════════════════════════════════════════════════════════════════
# Initialization
# ═══════════════════════════════════════════════════════════════════

config = WorldMindConfig(api_key="your-key", save_path="./output")

process_module = ProcessExperienceModule(config)
goal_module = GoalExperienceModule(config)
retrieval_module = ExperienceRetrievalModule(config)

task_instruction = "Complete the checkout process"

# ═══════════════════════════════════════════════════════════════════
# Phase 1: Before Task - Retrieve Experiences
# ═══════════════════════════════════════════════════════════════════

experiences = retrieval_module.retrieve(task_instruction)

agent_system_prompt = f"""You are a helpful assistant.

{PREDICTION_CONSTRAINT_PROMPT}

{experiences['formatted_prompt']}
"""

# ═══════════════════════════════════════════════════════════════════
# Phase 2: During Task - Process Each Step
# ═══════════════════════════════════════════════════════════════════

goal_trajectory = []
action_history = []

for step in agent_steps:  # Your agent's execution loop
    # Create process step
    process_step = ProcessTrajectoryStep(
        observation=current_observation,
        action=agent_action,
        predicted_state=agent_predicted_state,
        env_feedback=env_response
    )
    
    # Check for errors (skip if exploration phase)
    if agent_predicted_state != EXPLORATION_PHASE_MARKER:
        has_error, experiences = process_module.process_single_step(
            task_instruction=task_instruction,
            step=process_step,
            action_history=action_history
        )
    
    # Track for goal experience
    goal_trajectory.append(GoalTrajectoryStep(
        action=agent_action,
        env_feedback=env_response,
        observation=current_observation
    ))
    
    action_history.append(f"Action: {agent_action}, Feedback: {env_response}")

# ═══════════════════════════════════════════════════════════════════
# Phase 3: After Success - Extract Experience
# ═══════════════════════════════════════════════════════════════════

if task_success:
    goal_module.extract_experience(
        task_instruction=task_instruction,
        trajectory=goal_trajectory
    )
    
    # Reload to include new experiences
    retrieval_module.reload_experiences()
```

---

## 💡 Tips & Best Practices

### Exploration Phase Handling

When the agent cannot observe the target of an action, output the exploration marker to skip discrimination:

```python
from worldmind_plugin import EXPLORATION_PHASE_MARKER

# Agent outputs this when target is not visible
predicted_state = EXPLORATION_PHASE_MARKER
# Value: "Exploration phase: target not visible, prediction skipped."
```

### Directory Structure

After running, the following structure is created:

```
worldmind_output/
├── 📂 goal_experience/
│   └── goal_experiences.json       # All goal experiences
└── 📂 process_experience/
    └── process_experiences.json    # All process experiences
```

### Customizing Prompts

All prompts can be customized by modifying `worldmind_plugin/prompts.py`:

```python
from worldmind_plugin.prompts import (
    DISCRIMINATOR_SYSTEM_PROMPT,      # Customize comparison logic
    REFLECTOR_SYSTEM_PROMPT,          # Customize error analysis
    GOAL_EXPERIENCE_EXTRACTION_PROMPT # Customize experience extraction
)
```

---

## 📖 API Reference

### ProcessExperienceModule

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `process_trajectory` | `task_instruction`, `trajectory`, `before_images`, `after_images` | `List[str]` | Process entire trajectory |
| `process_single_step` | `task_instruction`, `step`, `action_history`, `state_before`, `before_image`, `after_image` | `Tuple[bool, List[str]]` | Process single step |
| `reset` | - | - | Reset module state |

### GoalExperienceModule

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `extract_experience` | `task_instruction`, `trajectory` | `str` | Extract experience from trajectory |
| `set_include_observation` | `include: bool` | - | Set whether to include observation |

### ExperienceRetrievalModule

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `retrieve` | `task_instruction`, `enable_refine` | `Dict` | Retrieve relevant experiences |
| `reload_experiences` | - | - | Reload experiences from files |
| `get_goal_experience_count` | - | `int` | Get number of stored goal experiences |
| `get_process_experience_count` | - | `int` | Get number of stored process experiences |

---


<div align="center">

**Part of the [WorldMind](https://github.com/zjunlp/WorldMind) Framework**

</div>
