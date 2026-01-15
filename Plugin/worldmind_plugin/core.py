"""
WorldMind Plugin Core Module

This module provides three independent modules for the WorldMind plugin:

1. ProcessExperienceModule: Extracts process experience from prediction errors
   - Input: task_instruction, trajectory (with observation, action, predicted_state, env_feedback)
   - Output: process experience entries

2. GoalExperienceModule: Extracts goal experience from successful trajectories
   - Input: task_instruction, trajectory (with action, env_feedback, optional observation)
   - Output: goal experience

3. ExperienceRetrievalModule: Retrieves and optionally refines experiences
   - Input: task_instruction
   - Output: retrieved experiences (goal + process)

Each module is independent and can be used separately.
"""

import os
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

from worldmind_plugin.config import WorldMindConfig
from worldmind_plugin.discriminator import WorldMindDiscriminator
from worldmind_plugin.reflector import WorldMindReflector
from worldmind_plugin.state_summarizer import WorldMindStateSummarizer
from worldmind_plugin.experience_refiner import ExperienceRefiner
from worldmind_plugin.llm_client import LLMClient
from worldmind_plugin.prompts import (
    PREDICTION_CONSTRAINT_PROMPT,
    EXPLORATION_PHASE_MARKER,
    GOAL_EXPERIENCE_EXTRACTION_PROMPT,
    GOAL_EXPERIENCE_RETRIEVAL_CONTEXT,
    GOAL_EXPERIENCE_ENTRY_FORMAT,
    PROCESS_EXPERIENCE_RETRIEVAL_CONTEXT,
    PROCESS_EXPERIENCE_ENTRY_FORMAT,
    TRAJECTORY_STEP_FORMAT,
    TRAJECTORY_STEP_FORMAT_WITH_OBSERVATION
)
from worldmind_plugin.utils import (
    is_exploration_phase,
    get_logger,
    set_logger,
    SimpleLogger,
    ensure_dir,
    save_json,
    load_json,
    compute_semantic_similarity
)


# =============================================================================
# Data Classes for Trajectory
# =============================================================================

@dataclass
class ProcessTrajectoryStep:
    """
    A single step in a process trajectory.
    
    Used for ProcessExperienceModule.
    Contains all information needed for discrimination and reflection.
    
    Attributes:
        observation: The state/observation before/after the action
        action: The action executed (action name or description)
        predicted_state: The agent's predicted state after action
        env_feedback: The environment's feedback after action
    """
    observation: str
    action: str
    predicted_state: str
    env_feedback: str = ""


@dataclass
class GoalTrajectoryStep:
    """
    A single step in a goal trajectory.
    
    Used for GoalExperienceModule.
    Contains action and feedback, optionally observation.
    
    Attributes:
        action: The action executed (action name or description)
        env_feedback: The environment's feedback after action
        observation: Optional observation (included based on config)
    """
    action: str
    env_feedback: str = ""
    observation: str = ""


@dataclass 
class ProcessExperienceEntry:
    """
    A process experience entry for storage.
    
    Attributes:
        instruction: The task instruction
        knowledge: List of experience entries
    """
    instruction: str
    knowledge: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "instruction": self.instruction,
            "knowledge": self.knowledge
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "ProcessExperienceEntry":
        return cls(
            instruction=data.get("instruction", ""),
            knowledge=data.get("knowledge", [])
        )


@dataclass
class GoalExperienceEntry:
    """
    A goal experience entry for storage.
    
    Attributes:
        instruction: The task instruction
        goal_experience: The extracted experience text
    """
    instruction: str
    goal_experience: str
    
    def to_dict(self) -> Dict:
        return {
            "instruction": self.instruction,
            "goal_experience": self.goal_experience
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "GoalExperienceEntry":
        return cls(
            instruction=data.get("instruction", ""),
            goal_experience=data.get("goal_experience", "")
        )


# =============================================================================
# Module 1: Process Experience Module
# =============================================================================

class ProcessExperienceModule:
    """
    Module for extracting process experience from prediction errors.
    
    This module:
    1. Compares predicted states with actual observations (discrimination)
    2. Reflects on prediction errors to extract lessons
    3. Saves process experiences paired with task instructions
    
    Input:
        - task_instruction: str - The task instruction
        - trajectory: List[ProcessTrajectoryStep] - List of steps, each containing:
            - observation: str - State/observation
            - action: str - Action executed
            - predicted_state: str - Agent's predicted state
            - env_feedback: str - Environment feedback
    
    Output:
        - process_experiences: List[str] - Extracted experience entries
    
    Storage Format:
        {
            "instruction": "task instruction",
            "knowledge": ["experience 1", "experience 2", ...]
        }
    """
    
    def __init__(self, config: WorldMindConfig):
        """
        Initialize the ProcessExperienceModule.
        
        Args:
            config: WorldMindConfig instance
        """
        self.config = config
        self.logger = SimpleLogger("ProcessExperienceModule", verbose=config.detailed_output)
        
        # Initialize components
        self.discriminator = WorldMindDiscriminator(
            api_key=config.api_key,
            api_base=config.api_base,
            model_name=config.discriminator_model
        )
        
        self.reflector = WorldMindReflector(
            api_key=config.api_key,
            api_base=config.api_base,
            model_name=config.reflector_model,
            use_env_feedback=config.use_env_feedback
        )
        
        # State summarizer for multimodal
        self.state_summarizer = None
        if config.is_multimodal:
            self.state_summarizer = WorldMindStateSummarizer(
                api_key=config.api_key,
                api_base=config.api_base,
                model_name=config.summarizer_model
            )
        
        # Ensure save directories exist
        self._init_save_dirs()
        
        self.logger.info("ProcessExperienceModule initialized")
    
    def _init_save_dirs(self):
        """Initialize save directories."""
        ensure_dir(self.config.save_path)
        ensure_dir(os.path.join(self.config.save_path, 'process_experience'))
    
    def process_trajectory(
        self,
        task_instruction: str,
        trajectory: List[ProcessTrajectoryStep],
        before_images: List[str] = None,
        after_images: List[str] = None
    ) -> List[str]:
        """
        Process a trajectory to extract process experiences.
        
        This is the main entry point for the module.
        
        Args:
            task_instruction: The task instruction string
            trajectory: List of ProcessTrajectoryStep objects
            before_images: List of before image paths (for multimodal)
            after_images: List of after image paths (for multimodal)
        
        Returns:
            List of extracted experience entries
        """
        all_experiences = []
        action_history = []
        
        for i, step in enumerate(trajectory):
            # Skip exploration phase
            if is_exploration_phase(step.predicted_state):
                action_history.append(f"Action: {step.action}, Feedback: {step.env_feedback}")
                continue
            
            # Get states
            if i == 0:
                state_before = step.observation
            else:
                state_before = trajectory[i-1].observation
            
            state_after = step.observation
            
            # Multimodal: use state summarizer
            if self.config.is_multimodal and self.state_summarizer and before_images and after_images:
                if i < len(before_images) and i < len(after_images):
                    summary = self.state_summarizer.summarize_states(
                        action_description=step.action,
                        before_image_path=before_images[i],
                        after_image_path=after_images[i]
                    )
                    state_before = summary.get("state_before_action", state_before)
                    state_after = summary.get("state_after_action", state_after)
            
            # Discriminate
            disc_result = self.discriminator.discriminate(
                predicted_state=step.predicted_state,
                actual_state=state_after,
                action_description=step.action
            )
            
            # If mismatch, reflect
            if not disc_result.get("match", True):
                reflection = self.reflector.reflect(
                    action_description=step.action,
                    predicted_state=step.predicted_state,
                    state_before=state_before,
                    state_after=state_after,
                    env_feedback=step.env_feedback if self.config.use_env_feedback else "",
                    human_instruction=task_instruction,
                    action_history="\n".join(action_history[-5:])
                )
                
                experience_entries = reflection.get("experience_entry", [])
                all_experiences.extend(experience_entries)
            
            action_history.append(f"Action: {step.action}, Feedback: {step.env_feedback}")
        
        # Save experiences if any
        if all_experiences:
            self._save_experiences(task_instruction, all_experiences)
        
        return all_experiences
    
    def process_single_step(
        self,
        task_instruction: str,
        step: ProcessTrajectoryStep,
        action_history: List[str] = None,
        state_before: str = None,
        before_image: str = None,
        after_image: str = None
    ) -> Tuple[bool, List[str]]:
        """
        Process a single step for real-time experience extraction.
        
        Args:
            task_instruction: The task instruction
            step: The current ProcessTrajectoryStep
            action_history: Previous action descriptions
            state_before: State before the action (if different from step.observation)
            before_image: Before image path (for multimodal)
            after_image: After image path (for multimodal)
        
        Returns:
            Tuple of (has_error: bool, experiences: List[str])
        """
        # Skip exploration phase
        if is_exploration_phase(step.predicted_state):
            return False, []
        
        state_before = state_before or step.observation
        state_after = step.observation
        
        # Multimodal processing
        if self.config.is_multimodal and self.state_summarizer and before_image and after_image:
            summary = self.state_summarizer.summarize_states(
                action_description=step.action,
                before_image_path=before_image,
                after_image_path=after_image
            )
            state_before = summary.get("state_before_action", state_before)
            state_after = summary.get("state_after_action", state_after)
        
        # Discriminate
        disc_result = self.discriminator.discriminate(
            predicted_state=step.predicted_state,
            actual_state=state_after,
            action_description=step.action
        )
        
        if disc_result.get("match", True):
            return False, []
        
        # Reflect
        history_str = "\n".join(action_history[-5:]) if action_history else "No previous actions"
        
        reflection = self.reflector.reflect(
            action_description=step.action,
            predicted_state=step.predicted_state,
            state_before=state_before,
            state_after=state_after,
            env_feedback=step.env_feedback if self.config.use_env_feedback else "",
            human_instruction=task_instruction,
            action_history=history_str
        )
        
        experiences = reflection.get("experience_entry", [])
        
        # Save if we have experiences
        if experiences:
            self._save_experiences(task_instruction, experiences)
        
        return True, experiences
    
    def _save_experiences(self, instruction: str, experiences: List[str]):
        """Save process experiences to file."""
        filepath = os.path.join(
            self.config.save_path,
            'process_experience',
            'process_experiences.json'
        )
        
        # Load existing
        existing = []
        if os.path.exists(filepath):
            existing = load_json(filepath)
        
        # Find or create entry for this instruction
        found = False
        for entry in existing:
            if entry.get("instruction") == instruction:
                # Add new experiences (avoid duplicates)
                for exp in experiences:
                    if exp not in entry.get("knowledge", []):
                        entry.setdefault("knowledge", []).append(exp)
                found = True
                break
        
        if not found:
            existing.append({
                "instruction": instruction,
                "knowledge": experiences
            })
        
        save_json(existing, filepath)
        self.logger.info(f"Saved {len(experiences)} process experiences")
    
    def reset(self):
        """Reset module state."""
        self.discriminator.reset_statistics()
        self.reflector.reset_statistics()
        if self.state_summarizer:
            self.state_summarizer.reset_statistics()


# =============================================================================
# Module 2: Goal Experience Module
# =============================================================================

class GoalExperienceModule:
    """
    Module for extracting goal experience from successful trajectories.
    
    This module:
    1. Accepts a successful task trajectory
    2. Uses LLM to extract reusable workflow/experience
    3. Saves goal experiences paired with task instructions
    
    Input:
        - task_instruction: str - The task instruction
        - trajectory: List[GoalTrajectoryStep] - List of steps, each containing:
            - action: str - Action executed
            - env_feedback: str - Environment feedback
            - observation: str (optional) - Observation
    
    Output:
        - goal_experience: str - Extracted experience
    
    Storage Format:
        {
            "instruction": "task instruction",
            "goal_experience": "extracted experience text"
        }
    """
    
    def __init__(self, config: WorldMindConfig):
        """
        Initialize the GoalExperienceModule.
        
        Args:
            config: WorldMindConfig instance
        """
        self.config = config
        self.logger = SimpleLogger("GoalExperienceModule", verbose=config.detailed_output)
        
        # LLM client for extraction
        self.extractor = LLMClient(
            api_key=config.api_key,
            api_base=config.api_base,
            model_name=config.extractor_model,
            is_multimodal=False,
            max_tokens=1024
        )
        
        # Whether to include observation in trajectory
        self.include_observation = config.goal_trajectory_include_observation
        
        # Ensure save directories exist
        self._init_save_dirs()
        
        self.logger.info("GoalExperienceModule initialized")
    
    def _init_save_dirs(self):
        """Initialize save directories."""
        ensure_dir(self.config.save_path)
        ensure_dir(os.path.join(self.config.save_path, 'goal_experience'))
    
    def extract_experience(
        self,
        task_instruction: str,
        trajectory: List[GoalTrajectoryStep]
    ) -> Optional[str]:
        """
        Extract goal experience from a successful trajectory.
        
        This is the main entry point for the module.
        Call this when a task is successfully completed.
        
        Args:
            task_instruction: The task instruction string
            trajectory: List of GoalTrajectoryStep objects
        
        Returns:
            Extracted experience string, or None if extraction failed
        """
        if not trajectory:
            self.logger.warning("Empty trajectory, cannot extract experience")
            return None
        
        # Build trajectory description
        trajectory_desc = self._format_trajectory(trajectory)
        
        # Build extraction prompt
        prompt = GOAL_EXPERIENCE_EXTRACTION_PROMPT.format(
            instruction=task_instruction,
            trajectory=trajectory_desc
        )
        
        try:
            # Call LLM
            response = self.extractor.chat(prompt)
            
            # Parse response
            import re
            json_match = re.search(r'\{[^{}]*"goal_experience"[^{}]*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                experience_text = result.get('goal_experience', '')
                
                if experience_text:
                    # Save experience
                    self._save_experience(task_instruction, experience_text)
                    self.logger.info(f"Extracted goal experience for: {task_instruction[:50]}...")
                    return experience_text
            
            self.logger.warning("Failed to parse extraction response")
            return None
            
        except Exception as e:
            self.logger.error(f"Experience extraction failed: {e}")
            return None
    
    def _format_trajectory(self, trajectory: List[GoalTrajectoryStep]) -> str:
        """Format trajectory for LLM prompt."""
        lines = []
        for i, step in enumerate(trajectory):
            if self.include_observation and step.observation:
                lines.append(TRAJECTORY_STEP_FORMAT_WITH_OBSERVATION.format(
                    step_num=i + 1,
                    observation=step.observation,
                    action_name=step.action,
                    env_feedback=step.env_feedback
                ))
            else:
                lines.append(TRAJECTORY_STEP_FORMAT.format(
                    step_num=i + 1,
                    action_name=step.action,
                    env_feedback=step.env_feedback
                ))
        return "".join(lines)
    
    def _save_experience(self, instruction: str, experience: str):
        """Save goal experience to file."""
        filepath = os.path.join(
            self.config.save_path,
            'goal_experience',
            'goal_experiences.json'
        )
        
        # Load existing
        existing = []
        if os.path.exists(filepath):
            existing = load_json(filepath)
        
        # Add new entry
        existing.append({
            "instruction": instruction,
            "goal_experience": experience
        })
        
        save_json(existing, filepath)
        self.logger.info("Saved goal experience")
    
    def set_include_observation(self, include: bool):
        """Set whether to include observation in trajectory."""
        self.include_observation = include


# =============================================================================
# Module 3: Experience Retrieval Module
# =============================================================================

class ExperienceRetrievalModule:
    """
    Module for retrieving and optionally refining experiences.
    
    This module:
    1. Retrieves relevant goal experiences based on semantic similarity
    2. Retrieves relevant process experiences based on semantic similarity
    3. Optionally refines/consolidates experiences using LLM
    
    Input:
        - task_instruction: str - The current task instruction
    
    Output:
        - experiences: Dict containing:
            - goal_experiences: List[Dict] - Retrieved goal experiences
            - process_experiences: List[Dict] - Retrieved process experiences
            - refined_experience: Dict (optional) - Consolidated experience
    """
    
    def __init__(self, config: WorldMindConfig):
        """
        Initialize the ExperienceRetrievalModule.
        
        Args:
            config: WorldMindConfig instance
        """
        self.config = config
        self.logger = SimpleLogger("ExperienceRetrievalModule", verbose=config.detailed_output)
        
        # Experience refiner (optional)
        self.refiner = None
        if config.enable_experience_refine:
            self.refiner = ExperienceRefiner(
                api_key=config.api_key,
                api_base=config.api_base,
                model_name=config.refiner_model
            )
        
        # Cache for embeddings
        self._embeddings_cache: Dict[str, Any] = {}
        
        # Load experiences
        self._goal_experiences: List[Dict] = []
        self._process_experiences: List[Dict] = []
        self._load_experiences()
        
        self.logger.info("ExperienceRetrievalModule initialized")
    
    def _load_experiences(self):
        """Load experiences from files."""
        # Load goal experiences
        goal_path = os.path.join(
            self.config.save_path,
            'goal_experience',
            'goal_experiences.json'
        )
        if os.path.exists(goal_path):
            self._goal_experiences = load_json(goal_path)
            self.logger.info(f"Loaded {len(self._goal_experiences)} goal experiences")
        
        # Load process experiences
        process_path = os.path.join(
            self.config.save_path,
            'process_experience',
            'process_experiences.json'
        )
        if os.path.exists(process_path):
            self._process_experiences = load_json(process_path)
            self.logger.info(f"Loaded {len(self._process_experiences)} process experience entries")
    
    def reload_experiences(self):
        """Reload experiences from files."""
        self._load_experiences()
        self._embeddings_cache.clear()
    
    def retrieve(self, task_instruction: str, enable_refine: bool = None) -> Dict:
        """
        Retrieve relevant experiences for the given task instruction.
        
        This is the main entry point for the module.
        
        Args:
            task_instruction: The current task instruction
            enable_refine: Whether to refine experiences (overrides config if set)
        
        Returns:
            Dict containing:
                - goal_experiences: List of relevant goal experiences
                - process_experiences: List of relevant process experiences  
                - refined_experience: Dict with merged_experience and initial_plan (if refine enabled)
                - formatted_prompt: Formatted string for injection into agent prompt
        """
        # Retrieve goal experiences
        goal_exps = self._retrieve_goal_experiences(task_instruction)
        
        # Retrieve process experiences
        process_exps = self._retrieve_process_experiences(task_instruction)
        
        result = {
            "goal_experiences": goal_exps,
            "process_experiences": process_exps,
            "refined_experience": None,
            "formatted_prompt": ""
        }
        
        # Refine if enabled
        should_refine = enable_refine if enable_refine is not None else self.config.enable_experience_refine
        
        if should_refine and self.refiner and (goal_exps or process_exps):
            refined = self.refiner.refine_for_task(
                current_instruction=task_instruction,
                goal_experiences=goal_exps,
                process_experience=process_exps
            )
            result["refined_experience"] = refined
            result["formatted_prompt"] = self.refiner.format_for_prompt(refined)
        else:
            # Format without refinement
            result["formatted_prompt"] = self._format_experiences(goal_exps, process_exps)
        
        return result
    
    def _retrieve_goal_experiences(self, instruction: str) -> List[Dict]:
        """Retrieve relevant goal experiences."""
        if not self._goal_experiences:
            return []
        
        # Exclude exact match
        instruction_norm = instruction.strip().lower()
        filtered = [
            exp for exp in self._goal_experiences
            if exp.get('instruction', '').strip().lower() != instruction_norm
        ]
        
        if not filtered:
            return []
        
        # Compute similarities
        similarities = []
        for exp in filtered:
            sim = compute_semantic_similarity(
                instruction,
                exp.get('instruction', ''),
                self._embeddings_cache
            )
            similarities.append((exp, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return [exp for exp, _ in similarities[:self.config.goal_experience_top_k]]
    
    def _retrieve_process_experiences(self, instruction: str) -> List[Dict]:
        """Retrieve relevant process experiences."""
        if not self._process_experiences:
            return []
        
        # Compute similarities
        similarities = []
        for entry in self._process_experiences:
            sim = compute_semantic_similarity(
                instruction,
                entry.get('instruction', ''),
                self._embeddings_cache
            )
            similarities.append((entry, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Flatten to individual knowledge entries
        result = []
        for entry, _ in similarities:
            for knowledge in entry.get('knowledge', []):
                result.append({
                    'instruction': entry.get('instruction', ''),
                    'knowledge': knowledge
                })
                if len(result) >= self.config.process_experience_top_k:
                    break
            if len(result) >= self.config.process_experience_top_k:
                break
        
        return result
    
    def _format_experiences(
        self,
        goal_exps: List[Dict],
        process_exps: List[Dict]
    ) -> str:
        """Format experiences for prompt injection."""
        parts = []
        
        # Format goal experiences
        if goal_exps:
            formatted = []
            for i, exp in enumerate(goal_exps):
                formatted.append(GOAL_EXPERIENCE_ENTRY_FORMAT.format(
                    index=i + 1,
                    instruction=exp.get('instruction', 'Unknown'),
                    experience=exp.get('goal_experience', 'No experience')
                ))
            parts.append(GOAL_EXPERIENCE_RETRIEVAL_CONTEXT.format(
                experiences="\n".join(formatted)
            ))
        
        # Format process experiences
        if process_exps:
            formatted = []
            for i, exp in enumerate(process_exps):
                formatted.append(PROCESS_EXPERIENCE_ENTRY_FORMAT.format(
                    index=i + 1,
                    instruction=exp.get('instruction', 'Unknown'),
                    knowledge=exp.get('knowledge', 'No knowledge')
                ))
            parts.append(PROCESS_EXPERIENCE_RETRIEVAL_CONTEXT.format(
                knowledge="\n".join(formatted)
            ))
        
        return "\n\n".join(parts)
    
    def get_goal_experience_count(self) -> int:
        """Get the number of stored goal experiences."""
        return len(self._goal_experiences)
    
    def get_process_experience_count(self) -> int:
        """Get the number of stored process experience entries."""
        return sum(len(e.get('knowledge', [])) for e in self._process_experiences)


# =============================================================================
# Unified WorldMind Class (Optional Convenience Wrapper)
# =============================================================================

class WorldMind:
    """
    Convenience wrapper that provides access to all three modules.
    
    This class is optional - you can use the three modules independently.
    
    Attributes:
        process_module: ProcessExperienceModule
        goal_module: GoalExperienceModule
        retrieval_module: ExperienceRetrievalModule
    """
    
    def __init__(self, config: WorldMindConfig):
        """
        Initialize WorldMind with all modules.
        
        Args:
            config: WorldMindConfig instance
        """
        self.config = config
        self.logger = SimpleLogger("WorldMind", verbose=config.detailed_output)
        set_logger(self.logger)
        
        # Validate configuration
        config.validate()
        
        # Initialize modules
        self.process_module = ProcessExperienceModule(config)
        self.goal_module = GoalExperienceModule(config)
        self.retrieval_module = ExperienceRetrievalModule(config)
        
        self.logger.info("WorldMind initialized with all modules")
    
    def get_prediction_constraint_prompt(self) -> str:
        """Get the prediction constraint prompt for agent system prompt."""
        return PREDICTION_CONSTRAINT_PROMPT
    
    def get_exploration_marker(self) -> str:
        """Get the exploration phase marker string."""
        return EXPLORATION_PHASE_MARKER


# =============================================================================
# Factory Functions
# =============================================================================

def create_process_module(config: WorldMindConfig) -> ProcessExperienceModule:
    """Create a ProcessExperienceModule instance."""
    return ProcessExperienceModule(config)


def create_goal_module(config: WorldMindConfig) -> GoalExperienceModule:
    """Create a GoalExperienceModule instance."""
    return GoalExperienceModule(config)


def create_retrieval_module(config: WorldMindConfig) -> ExperienceRetrievalModule:
    """Create an ExperienceRetrievalModule instance."""
    return ExperienceRetrievalModule(config)


def create_worldmind(config: WorldMindConfig) -> WorldMind:
    """Create a WorldMind instance with all modules."""
    return WorldMind(config)
