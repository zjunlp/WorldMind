"""
WorldMind Plugin Knowledge Manager Module

This module provides management for two types of learned knowledge:
1. Goal Experience: Extracted from successful task trajectories
2. Process Experience: Extracted from prediction errors during task execution

Both types of experience can be retrieved for future tasks using semantic similarity.

Usage:
    # Goal Experience
    goal_manager = GoalExperienceManager(
        save_path="./output",
        top_k=3
    )
    goal_manager.extract_experience(instruction, trajectory, task_success=True)
    experiences = goal_manager.retrieve_experiences(new_instruction)
    
    # Process Experience
    process_manager = ProcessExperienceManager(
        save_path="./output",
        top_k=3
    )
    process_manager.add_entries(["Environmental Logic: ..."], instruction)
    knowledge = process_manager.retrieve_knowledge(new_instruction)
"""

import os
import json
import re
from typing import Dict, List, Optional, Any

from worldmind_plugin.llm_client import LLMClient
from worldmind_plugin.prompts import (
    GOAL_EXPERIENCE_EXTRACTION_PROMPT,
    GOAL_EXPERIENCE_RETRIEVAL_CONTEXT,
    GOAL_EXPERIENCE_ENTRY_FORMAT,
    PROCESS_EXPERIENCE_RETRIEVAL_CONTEXT,
    PROCESS_EXPERIENCE_ENTRY_FORMAT,
    TRAJECTORY_STEP_FORMAT
)
from worldmind_plugin.utils import (
    compute_semantic_similarity,
    get_logger,
    ensure_dir,
    save_json,
    load_json
)


class GoalExperienceManager:
    """
    Manager for goal experiences extracted from successful task trajectories.
    
    Goal experiences capture successful action patterns that can guide future tasks.
    They are extracted by an LLM that analyzes successful execution trajectories.
    
    Attributes:
        save_path: Base path for saving experiences
        eval_set: Current evaluation set name (for organizing experiences)
        top_k: Number of experiences to retrieve
        experiences: List of stored experiences
    """
    
    def __init__(
        self,
        save_path: str = "./worldmind_output",
        eval_set: str = "default",
        top_k: int = 3,
        extractor_model: Optional[str] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None
    ):
        """
        Initialize the goal experience manager.
        
        Args:
            save_path: Base path for saving experiences
            eval_set: Current evaluation set name
            top_k: Number of experiences to retrieve
            extractor_model: LLM model for experience extraction
            api_key: OpenAI API key
            api_base: OpenAI API base URL
        """
        self.save_path = save_path
        self.eval_set = eval_set
        self.top_k = top_k
        self.experiences: List[Dict] = []
        self.episode_trajectory: List[Dict] = []
        self.current_instruction: str = ""
        self.actions: List[str] = []
        self._embeddings_cache: Dict[str, Any] = {}
        
        self.logger = get_logger()
        
        # Initialize extractor if model provided
        self.extractor = None
        if extractor_model:
            try:
                self.extractor = LLMClient(
                    api_key=api_key,
                    api_base=api_base,
                    model_name=extractor_model,
                    is_multimodal=False,
                    max_tokens=1024
                )
            except Exception as e:
                self.logger.warning(f"Failed to initialize extractor: {e}")
        
        # Load existing experiences
        self._load_experiences()
        
        self.logger.info(f"GoalExperienceManager initialized: eval_set={eval_set}, top_k={top_k}")
    
    def set_save_path(self, path: str):
        """Set save path and reload experiences."""
        self.save_path = path
        self._load_experiences()
    
    def set_eval_set(self, eval_set: str):
        """Set current eval set and reload experiences."""
        self.eval_set = eval_set
        self._load_experiences()
    
    def set_actions(self, actions: List[str]):
        """Set the action list for trajectory formatting."""
        self.actions = actions
    
    def set_instruction(self, instruction: str):
        """Set current task instruction."""
        self.current_instruction = instruction
    
    def _get_experience_path(self) -> str:
        """Get the path for experience file."""
        experience_dir = os.path.join(self.save_path, 'goal_experience')
        ensure_dir(experience_dir)
        return os.path.join(experience_dir, f'{self.eval_set}_goal_experiences.json')
    
    def _load_experiences(self):
        """Load experiences for current eval_set."""
        experience_path = self._get_experience_path()
        if os.path.exists(experience_path):
            try:
                self.experiences = load_json(experience_path)
                self._embeddings_cache.clear()
                self.logger.info(f"Loaded {len(self.experiences)} goal experiences")
            except Exception as e:
                self.logger.warning(f"Failed to load experiences: {e}")
                self.experiences = []
        else:
            self.experiences = []
    
    def _save_experiences(self):
        """Save experiences for current eval_set."""
        try:
            # Remove duplicates
            unique_experiences = []
            seen = set()
            for exp in self.experiences:
                exp_text = exp.get('goal_experience', '')
                if exp_text and exp_text not in seen:
                    seen.add(exp_text)
                    unique_experiences.append(exp)
            
            self.experiences = unique_experiences
            save_json(self.experiences, self._get_experience_path())
            self.logger.info(f"Saved {len(self.experiences)} goal experiences")
        except Exception as e:
            self.logger.error(f"Failed to save experiences: {e}")
    
    def reset_trajectory(self):
        """Reset episode trajectory for new episode."""
        self.episode_trajectory = []
        self.current_instruction = ""
    
    def update_trajectory(self, info: Dict):
        """
        Update episode trajectory with step info.
        
        Args:
            info: Dictionary with keys:
                - action_id: Action ID
                - env_feedback: Environment feedback (if available)
                - last_action_success: Whether action succeeded
                - task_progress: Current task progress (0-1)
        """
        entry = {
            'action_id': info.get('action_id', -1),
            'env_feedback': info.get('env_feedback', ''),
            'last_action_success': info.get('last_action_success', 1),
            'task_progress': info.get('task_progress', 0.0)
        }
        self.episode_trajectory.append(entry)
    
    def retrieve_experiences(self, instruction: str) -> List[Dict]:
        """
        Retrieve top-k most relevant goal experiences using semantic similarity.
        
        Args:
            instruction: Current task instruction
            
        Returns:
            List of relevant experience dictionaries
        """
        if not self.experiences:
            return []
        
        # Exclude current task's instruction
        instruction_normalized = instruction.strip().lower()
        filtered = [
            exp for exp in self.experiences
            if exp.get('instruction', '').strip().lower() != instruction_normalized
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
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return [exp for exp, _ in similarities[:self.top_k]]
    
    def format_experiences_for_prompt(self, experiences: List[Dict]) -> str:
        """
        Format retrieved goal experiences for inclusion in prompt.
        
        Args:
            experiences: List of experience dictionaries
            
        Returns:
            Formatted string for prompt
        """
        if not experiences:
            return ""
        
        formatted = []
        for i, exp in enumerate(experiences):
            formatted.append(GOAL_EXPERIENCE_ENTRY_FORMAT.format(
                index=i + 1,
                instruction=exp.get('instruction', 'Unknown'),
                experience=exp.get('goal_experience', 'No experience')
            ))
        
        return GOAL_EXPERIENCE_RETRIEVAL_CONTEXT.format(experiences="\n".join(formatted))
    
    def extract_experience(
        self,
        instruction: str,
        task_success: bool
    ) -> Optional[Dict]:
        """
        Extract goal experience from successful episode trajectory.
        
        Args:
            instruction: Task instruction
            task_success: Whether task was successful
            
        Returns:
            Extracted experience dict, or None if extraction failed
        """
        if not task_success:
            self.logger.debug("Task not successful, skipping extraction")
            return None
        
        if not self.episode_trajectory:
            self.logger.warning("No trajectory data for extraction")
            return None
        
        if not self.extractor:
            self.logger.warning("No extractor configured")
            return None
        
        try:
            # Build trajectory description
            trajectory_desc = ""
            prev_progress = 0.0
            for i, step in enumerate(self.episode_trajectory):
                action_id = step.get('action_id', -1)
                action_name = self.actions[action_id] if 0 <= action_id < len(self.actions) else "unknown"
                result = "Success" if step.get('last_action_success', 1) else "Failed"
                task_progress = step.get('task_progress', 0.0)
                
                progress_change = task_progress - prev_progress
                if progress_change > 0:
                    progress_note = f"{task_progress:.1%} (+{progress_change:.1%} PROGRESS!)"
                else:
                    progress_note = f"{task_progress:.1%}"
                prev_progress = task_progress
                
                trajectory_desc += TRAJECTORY_STEP_FORMAT.format(
                    step_num=i + 1,
                    action_name=action_name,
                    action_id=action_id,
                    env_feedback=step.get('env_feedback', ''),
                    task_progress=progress_note,
                    result=result
                )
            
            # Build extraction prompt
            prompt = GOAL_EXPERIENCE_EXTRACTION_PROMPT.format(
                instruction=instruction,
                trajectory=trajectory_desc
            )
            
            # Call extractor
            response = self.extractor.chat(prompt)
            
            # Parse response
            json_match = re.search(r'\{[^{}]*"goal_experience"[^{}]*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                experience_text = result.get('goal_experience', '')
                
                if experience_text:
                    experience_entry = {
                        'instruction': instruction,
                        'goal_experience': experience_text,
                        'eval_set': self.eval_set,
                        'trajectory_length': len(self.episode_trajectory)
                    }
                    
                    self.experiences.append(experience_entry)
                    self._save_experiences()
                    
                    # Save episode experience
                    self._save_episode_experience(experience_entry)
                    
                    self.logger.info(f"Extracted goal experience for: {instruction[:50]}...")
                    return experience_entry
            
            self.logger.warning("Failed to parse extraction response")
            return None
            
        except Exception as e:
            self.logger.error(f"Experience extraction failed: {e}")
            return None
    
    def _save_episode_experience(self, experience: Dict):
        """Save individual episode experience."""
        episode_dir = os.path.join(self.save_path, 'goal_experience', 'episodes')
        ensure_dir(episode_dir)
        
        episode_num = len([f for f in os.listdir(episode_dir) if f.endswith('.json')]) + 1
        filepath = os.path.join(episode_dir, f'episode_{episode_num}_goal.json')
        save_json(experience, filepath)


class ProcessExperienceManager:
    """
    Manager for process experiences extracted from prediction errors.
    
    Process experiences capture lessons learned from prediction failures,
    including environmental rules and corrective workflows.
    
    Attributes:
        save_path: Base path for saving experiences
        eval_set: Current evaluation set name
        top_k: Number of experiences to retrieve
        knowledge_entries: List of stored knowledge entries
    """
    
    def __init__(
        self,
        save_path: str = "./worldmind_output",
        eval_set: str = "default",
        top_k: int = 3
    ):
        """
        Initialize the process experience manager.
        
        Args:
            save_path: Base path for saving experiences
            eval_set: Current evaluation set name
            top_k: Number of experiences to retrieve
        """
        self.save_path = save_path
        self.eval_set = eval_set
        self.top_k = top_k
        self.knowledge_entries: List[Dict] = []
        self.current_episode_knowledge: List[str] = []
        self.current_instruction: str = ""
        self._embeddings_cache: Dict[str, Any] = {}
        
        self.logger = get_logger()
        
        # Load existing knowledge
        self._load_knowledge()
        
        self.logger.info(f"ProcessExperienceManager initialized: eval_set={eval_set}, top_k={top_k}")
    
    def set_save_path(self, path: str):
        """Set save path and reload knowledge."""
        self.save_path = path
        self._load_knowledge()
    
    def set_eval_set(self, eval_set: str):
        """Set current eval set and reload knowledge."""
        self.eval_set = eval_set
        self._load_knowledge()
    
    def set_instruction(self, instruction: str):
        """Set current task instruction."""
        self.current_instruction = instruction
    
    def _get_knowledge_path(self) -> str:
        """Get the path for knowledge file."""
        knowledge_dir = os.path.join(self.save_path, 'process_experience')
        ensure_dir(knowledge_dir)
        return os.path.join(knowledge_dir, f'{self.eval_set}_process_experiences.json')
    
    def _load_knowledge(self):
        """Load knowledge for current eval_set."""
        knowledge_path = self._get_knowledge_path()
        if os.path.exists(knowledge_path):
            try:
                loaded = load_json(knowledge_path)
                self.knowledge_entries = self._normalize_format(loaded)
                self._embeddings_cache.clear()
                total = sum(len(e.get('knowledge', [])) for e in self.knowledge_entries)
                self.logger.info(f"Loaded {total} process experience entries")
            except Exception as e:
                self.logger.warning(f"Failed to load knowledge: {e}")
                self.knowledge_entries = []
        else:
            self.knowledge_entries = []
    
    def _normalize_format(self, data: List) -> List[Dict]:
        """Normalize knowledge data format."""
        if not data:
            return []
        
        # Check if old format (knowledge as string)
        if data and isinstance(data[0].get('knowledge'), str):
            instruction_map = {}
            for entry in data:
                instr = entry.get('instruction', 'Unknown')
                knowledge = entry.get('knowledge', '')
                
                if instr not in instruction_map:
                    instruction_map[instr] = {
                        'instruction': instr,
                        'eval_set': self.eval_set,
                        'knowledge': []
                    }
                
                if knowledge and knowledge not in instruction_map[instr]['knowledge']:
                    instruction_map[instr]['knowledge'].append(knowledge)
            
            return list(instruction_map.values())
        
        # Already new format
        return data
    
    def _save_knowledge(self):
        """Save knowledge for current eval_set."""
        try:
            # Remove duplicates within each entry
            for entry in self.knowledge_entries:
                knowledge = entry.get('knowledge', [])
                entry['knowledge'] = list(dict.fromkeys(knowledge))
            
            save_json(self.knowledge_entries, self._get_knowledge_path())
            total = sum(len(e.get('knowledge', [])) for e in self.knowledge_entries)
            self.logger.info(f"Saved {total} process experience entries")
        except Exception as e:
            self.logger.error(f"Failed to save knowledge: {e}")
    
    def reset_episode(self):
        """Reset current episode entries."""
        self.current_episode_knowledge = []
        self.current_instruction = ""
    
    def add_entries(self, entries: List[str], instruction: str = None):
        """
        Add process experience entries.
        
        Args:
            entries: List of experience entry strings
            instruction: Task instruction (uses current if not provided)
        """
        if not entries:
            return
        
        # Clean and deduplicate
        entries = [e.strip() for e in entries if e and e.strip()]
        entries = list(dict.fromkeys(entries))
        
        if not entries:
            return
        
        instr = instruction or self.current_instruction or "Unknown"
        
        # Add to current episode
        for entry in entries:
            if entry not in self.current_episode_knowledge:
                self.current_episode_knowledge.append(entry)
        
        # Find or create knowledge entry
        knowledge_entry = None
        for e in self.knowledge_entries:
            if e.get('instruction') == instr:
                knowledge_entry = e
                break
        
        if not knowledge_entry:
            knowledge_entry = {
                'instruction': instr,
                'eval_set': self.eval_set,
                'knowledge': []
            }
            self.knowledge_entries.append(knowledge_entry)
        
        # Add entries
        for entry in entries:
            if entry not in knowledge_entry['knowledge']:
                knowledge_entry['knowledge'].append(entry)
        
        self.logger.info(f"Added {len(entries)} process entries")
        self._save_knowledge()
    
    def save_episode_knowledge(self, episode_num: int):
        """
        Save current episode's knowledge to separate file.
        
        Args:
            episode_num: Episode number for filename
        """
        episode_dir = os.path.join(self.save_path, 'process_experience', 'episodes')
        ensure_dir(episode_dir)
        
        episode_data = {
            'instruction': self.current_instruction or "Unknown",
            'eval_set': self.eval_set,
            'knowledge': self.current_episode_knowledge.copy()
        }
        
        filepath = os.path.join(episode_dir, f'episode_{episode_num}_process.json')
        save_json(episode_data, filepath)
        
        self.logger.info(f"Saved episode {episode_num} knowledge")
        self._save_knowledge()
    
    def retrieve_knowledge(self, instruction: str) -> List[Dict]:
        """
        Retrieve top-k most relevant process experiences.
        
        Args:
            instruction: Current task instruction
            
        Returns:
            List of knowledge dictionaries with keys: instruction, knowledge
        """
        if not self.knowledge_entries:
            return []
        
        # Compute similarities
        similarities = []
        for entry in self.knowledge_entries:
            sim = compute_semantic_similarity(
                instruction,
                entry.get('instruction', ''),
                self._embeddings_cache
            )
            similarities.append((entry, sim))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Flatten to individual knowledge entries
        result = []
        for entry, _ in similarities:
            for knowledge in entry.get('knowledge', []):
                result.append({
                    'instruction': entry.get('instruction', 'Unknown'),
                    'knowledge': knowledge
                })
                if len(result) >= self.top_k:
                    break
            if len(result) >= self.top_k:
                break
        
        return result
    
    def format_knowledge_for_prompt(self, knowledge_list: List[Dict]) -> str:
        """
        Format retrieved process experiences for inclusion in prompt.
        
        Args:
            knowledge_list: List of knowledge dictionaries
            
        Returns:
            Formatted string for prompt
        """
        if not knowledge_list:
            return ""
        
        formatted = []
        for i, entry in enumerate(knowledge_list):
            formatted.append(PROCESS_EXPERIENCE_ENTRY_FORMAT.format(
                index=i + 1,
                instruction=entry.get('instruction', 'Unknown'),
                knowledge=entry.get('knowledge', 'No knowledge')
            ))
        
        return PROCESS_EXPERIENCE_RETRIEVAL_CONTEXT.format(knowledge="\n".join(formatted))
    
    def get_all_knowledge_flat(self) -> List[str]:
        """Get all knowledge entries as a flat list."""
        all_knowledge = []
        for entry in self.knowledge_entries:
            all_knowledge.extend(entry.get('knowledge', []))
        return all_knowledge


def create_goal_experience_manager(
    save_path: str = "./worldmind_output",
    eval_set: str = "default",
    top_k: int = 3,
    extractor_model: Optional[str] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None
) -> GoalExperienceManager:
    """
    Factory function to create a Goal Experience Manager.
    
    Args:
        save_path: Base path for saving experiences
        eval_set: Current evaluation set name
        top_k: Number of experiences to retrieve
        extractor_model: LLM model for experience extraction
        api_key: OpenAI API key
        api_base: OpenAI API base URL
        
    Returns:
        GoalExperienceManager instance
    """
    return GoalExperienceManager(
        save_path=save_path,
        eval_set=eval_set,
        top_k=top_k,
        extractor_model=extractor_model,
        api_key=api_key,
        api_base=api_base
    )


def create_process_experience_manager(
    save_path: str = "./worldmind_output",
    eval_set: str = "default",
    top_k: int = 3
) -> ProcessExperienceManager:
    """
    Factory function to create a Process Experience Manager.
    
    Args:
        save_path: Base path for saving experiences
        eval_set: Current evaluation set name
        top_k: Number of experiences to retrieve
        
    Returns:
        ProcessExperienceManager instance
    """
    return ProcessExperienceManager(
        save_path=save_path,
        eval_set=eval_set,
        top_k=top_k
    )
