"""
WorldMind Knowledge Manager Module for Habitat Environment

Provides management for two types of learned knowledge:
1. Goal Experience: Extracted from successful task trajectories
2. Process Experience: Extracted from prediction errors during task execution
"""
import os
import json
import re
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

from embodiedbench.planner.remote_model import RemoteModel
from embodiedbench.planner.planner_utils import fix_json
from embodiedbench.main import logger

# Import SentenceTransformer - required dependency
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError(
        "sentence-transformers is required for WorldMind. "
        "Please install it with: pip install sentence-transformers"
    )

# Global sentence model instance
_sentence_model = None

def _get_sentence_model():
    """Get or initialize SentenceTransformer model for semantic similarity."""
    global _sentence_model
    if _sentence_model is None:
        _sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("SentenceTransformer model loaded for semantic retrieval")
    return _sentence_model


# Goal Experience Prompts
GOAL_EXPERIENCE_EXTRACTION_PROMPT = '''You are a task analysis expert. Given a successful task execution trajectory with task_progress information, extract a SINGLE reusable workflow that captures the key actions that led to success.

## Task Instruction
{instruction}

## Successful Execution Trajectory (with Task Progress)
{trajectory}

## Task Category
{task_category}

## CRITICAL WORKFLOW EXTRACTION GUIDELINES

### Focus on Progress-Boosting Actions
The trajectory includes "task_progress" after each step. Pay SPECIAL ATTENTION to:
1. **Actions that INCREASED task_progress** - These are the KEY ACTIONS that actually advanced the task
2. **The sequence of actions leading to progress increases** - This reveals the correct approach
3. **Actions that maintained or prepared for progress** - Supporting actions matter too

### Workflow Requirements
1. **Extract ONE concise, generalizable workflow** - Not multiple workflows
2. **Focus on the ACTION PATTERN, not specific objects** - Use generic terms like "target object", "destination location"
3. **Highlight the critical decision points** - When to navigate vs. pick vs. place
4. **Capture the logical order** - The sequence matters for success
5. **Be specific about SUCCESS CONDITIONS** - What made this approach work

### Abstraction Guidelines
- Replace specific object names (e.g., "apple", "table_42") with generic terms (e.g., "[TARGET_OBJECT]", "[DESTINATION]")
- Keep action types specific (navigate, pick, place, open, close)
- Preserve the causal relationships between actions

Please extract a SINGLE reusable workflow that captures the essence of this successful execution.

Output format:
{{"goal_experience": "A complete workflow description that includes: (1) The goal pattern, (2) The key action sequence with EMPHASIS on progress-boosting actions, (3) Critical decision points, (4) Success conditions. Should be 3-6 sentences that can guide similar tasks."}}

!!! Please output only the JSON without any markdown formatting or code blocks.
'''


GOAL_EXPERIENCE_RETRIEVAL_CONTEXT = '''
## Relevant Goal Experiences from Past Successful Executions
Based on similar tasks, here are goal experiences that led to task completion. Pay attention to the KEY ACTIONS that boosted task progress:
{experiences}
IMPORTANT: These experiences highlight the critical action sequences that led to progress. Adapt them to your current task, focusing on the same logical patterns and decision points.
'''

GOAL_EXPERIENCE_ENTRY_FORMAT = '''
### Goal Experience {index}
**Original Task**: {instruction}
**Experience**: {experience}
'''


# Process Experience Prompts
PROCESS_EXPERIENCE_RETRIEVAL_CONTEXT = '''
## Relevant Process Experiences from Past Executions
Based on similar tasks, here are process experiences learned from prediction errors:
{knowledge}
IMPORTANT: Pay special attention to these learned patterns for avoiding past mistakes.
'''

PROCESS_EXPERIENCE_ENTRY_FORMAT = '''
### Process Experience {index}
**Knowledge**: {knowledge}
'''


TRAJECTORY_STEP_FORMAT = '''Step {step_num}:
- Action: {action_name} (ID: {action_id})
- Environment Feedback: {env_feedback}
- Task Progress: {task_progress}
- Result: {result}
'''


class GoalExperienceManager:
    """Manager for goal experiences extracted from successful task trajectories."""
    
    def __init__(
        self, 
        log_path: str = None,
        eval_set: str = "base",
        top_k: int = 3,
        extractor_model: str = None,
        model_type: str = 'remote'
    ):
        """Initialize the goal experience manager."""
        self.log_path = log_path
        self.eval_set = eval_set
        self.top_k = top_k
        self.experiences: List[Dict] = []
        self._embeddings_cache: Dict[str, any] = {}
        self.episode_trajectory: List[Dict] = []
        self.actions: List[str] = []
        self.current_instruction: str = ""
        
        self.extractor = None
        if extractor_model:
            try:
                self.extractor = RemoteModel(
                    extractor_model,
                    model_type=model_type,
                    language_only=True
                )
                logger.info(f"GoalExperienceManager: Extractor model initialized: {extractor_model}")
            except Exception as e:
                logger.warning(f"Failed to initialize extractor model: {e}")
                self.extractor = None
        
        self._load_experiences()
        
        logger.info(f"GoalExperienceManager initialized for eval_set={eval_set}, top_k={top_k}")
    
    def set_log_path(self, log_path: str):
        """Set log path and reload experiences."""
        self.log_path = log_path
        self._load_experiences()
        logger.info(f"GoalExperienceManager: log_path set to {log_path}")
    
    def set_eval_set(self, eval_set: str):
        """Set current eval set and reload experiences."""
        self.eval_set = eval_set
        self._load_experiences()
        logger.info(f"GoalExperienceManager: eval_set set to {eval_set}")
    
    def set_actions(self, actions: List[str]):
        """Set the action list for trajectory formatting."""
        self.actions = actions
    
    def set_instruction(self, instruction: str):
        """Set current task instruction."""
        self.current_instruction = instruction
    
    def _get_experience_path(self) -> str:
        """Get the path for experience file of current eval_set."""
        if not self.log_path:
            logger.warning("GoalExperienceManager: log_path is None")
            return None
        experience_dir = os.path.join(self.log_path, 'goal_experience')
        if not os.path.exists(experience_dir):
            os.makedirs(experience_dir)
            logger.info(f"Created goal_experience directory: {experience_dir}")
        return os.path.join(experience_dir, f'{self.eval_set}_goal_experiences.json')
    
    def _load_experiences(self):
        """Load experiences for current eval_set."""
        experience_path = self._get_experience_path()
        if experience_path and os.path.exists(experience_path):
            try:
                with open(experience_path, 'r', encoding='utf-8') as f:
                    self.experiences = json.load(f)
                self._embeddings_cache.clear()
                logger.info(f"Loaded {len(self.experiences)} goal experiences from {experience_path}")
            except Exception as e:
                logger.warning(f"Failed to load goal experiences: {e}")
                self.experiences = []
        else:
            self.experiences = []
            if experience_path:
                logger.info(f"No existing goal experiences at {experience_path}")
    
    def _save_experiences(self):
        """Save experiences for current eval_set."""
        experience_path = self._get_experience_path()
        if not experience_path:
            logger.warning("GoalExperienceManager: Cannot save - log_path is None")
            return
        
        try:
            unique_experiences = []
            seen_experiences = set()
            for exp in self.experiences:
                exp_text = exp.get('goal_experience', '')
                if exp_text and exp_text not in seen_experiences:
                    seen_experiences.add(exp_text)
                    unique_experiences.append(exp)
            
            original_count = len(self.experiences)
            self.experiences = unique_experiences
            
            if original_count != len(self.experiences):
                logger.info(f"Removed {original_count - len(self.experiences)} duplicate goal experiences")
            
            os.makedirs(os.path.dirname(experience_path), exist_ok=True)
            with open(experience_path, 'w', encoding='utf-8') as f:
                json.dump(self.experiences, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved {len(self.experiences)} goal experiences to {experience_path}")
        except Exception as e:
            logger.error(f"Failed to save goal experiences: {e}")
    
    def reset_trajectory(self):
        """Reset episode trajectory for new episode."""
        self.episode_trajectory = []
        self.current_instruction = ""
        logger.debug("GoalExperienceManager: trajectory reset")
    
    def update_trajectory(self, info: Dict):
        """Update episode trajectory with step info."""
        entry = {
            'action_id': info.get('action_id', -1),
            'env_feedback': info.get('env_feedback', ''),
            'last_action_success': info.get('last_action_success', 1),
            'task_progress': info.get('task_progress', 0.0)
        }
        self.episode_trajectory.append(entry)
        logger.debug(f"GoalExperienceManager: trajectory updated, now {len(self.episode_trajectory)} steps")
    
    def _compute_semantic_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between two text strings using sentence embeddings."""
        model = _get_sentence_model()
        
        try:
            # Use cache for embeddings
            if text1 not in self._embeddings_cache:
                self._embeddings_cache[text1] = model.encode(text1, convert_to_numpy=True)
            if text2 not in self._embeddings_cache:
                self._embeddings_cache[text2] = model.encode(text2, convert_to_numpy=True)
            
            emb1 = self._embeddings_cache[text1]
            emb2 = self._embeddings_cache[text2]
            
            # Compute cosine similarity
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            return float(similarity)
        except Exception as e:
            logger.error(f"Error computing semantic similarity: {e}")
            raise
    
    def retrieve_experiences(self, instruction: str) -> List[Dict]:
        """Retrieve top-k most relevant goal experiences based on instruction similarity."""
        if not self.experiences:
            return []
        
        instruction_normalized = instruction.strip().lower()
        filtered_experiences = [
            exp for exp in self.experiences 
            if exp.get('instruction', '').strip().lower() != instruction_normalized
        ]
        
        if not filtered_experiences:
            logger.debug("No experiences left after excluding current task's experiences")
            return []
        
        similarities = []
        for exp in filtered_experiences:
            sim = self._compute_semantic_similarity(instruction, exp.get('instruction', ''))
            similarities.append((exp, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [exp for exp, _ in similarities[:self.top_k]]
    
    def format_experiences_for_prompt(self, experiences: List[Dict]) -> str:
        """Format retrieved goal experiences for inclusion in prompt."""
        if not experiences:
            return ""
        
        formatted = []
        for i, exp in enumerate(experiences):
            formatted.append(GOAL_EXPERIENCE_ENTRY_FORMAT.format(
                index=i+1,
                instruction=exp.get('instruction', 'Unknown'),
                experience=exp.get('goal_experience', 'No experience available')
            ))
        
        return GOAL_EXPERIENCE_RETRIEVAL_CONTEXT.format(experiences="\n".join(formatted))
    
    def extract_experience(self, instruction: str, task_success: bool) -> Optional[Dict]:
        """Extract goal experience from successful episode trajectory."""
        logger.info(f"GoalExperienceManager.extract_experience called: task_success={task_success}")
        
        if not task_success:
            logger.info("Task not successful, skipping experience extraction")
            return None
            
        if not self.episode_trajectory:
            logger.warning("No trajectory data available for experience extraction")
            return None
        
        if not self.extractor:
            logger.warning("No extractor model configured, cannot extract goal experience")
            return None
        
        trajectory_desc = ""
        prev_progress = 0.0
        for i, step in enumerate(self.episode_trajectory):
            action_id = step.get('action_id', -1)
            action_name = self.actions[action_id] if 0 <= action_id < len(self.actions) else "unknown"
            result = "Success" if step.get('last_action_success', 1) else "Failed"
            task_progress = step.get('task_progress', 0.0)
            
            progress_change = task_progress - prev_progress
            if progress_change > 0:
                progress_note = f"{task_progress:.1%} (+{progress_change:.1%} PROGRESS BOOST!)"
            else:
                progress_note = f"{task_progress:.1%}"
            prev_progress = task_progress
            
            trajectory_desc += TRAJECTORY_STEP_FORMAT.format(
                step_num=i+1,
                action_name=action_name,
                action_id=action_id,
                env_feedback=step.get('env_feedback', ''),
                task_progress=progress_note,
                result=result
            )
        
        logger.info(f"Built trajectory description with {len(self.episode_trajectory)} steps")
        
        extraction_prompt = GOAL_EXPERIENCE_EXTRACTION_PROMPT.format(
            instruction=instruction,
            trajectory=trajectory_desc,
            task_category=self.eval_set
        )
        
        messages = [{"role": "user", "content": [{"type": "text", "text": extraction_prompt}]}]
        
        try:
            response = self.extractor.respond(messages)
            response = fix_json(response)
            
            logger.debug(f"Extractor response: {response[:500]}...")
            
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
                    
                    logger.info(f"Successfully extracted and saved goal experience for: {instruction[:50]}...")
                    return experience_entry
                else:
                    logger.warning("Extracted experience text is empty")
            else:
                try:
                    result = json.loads(response)
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
                        logger.info(f"Successfully extracted goal experience: {instruction[:50]}...")
                        return experience_entry
                except json.JSONDecodeError:
                    logger.warning("Failed to parse goal_experience from response")
                    
        except Exception as e:
            logger.error(f"Goal experience extraction failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())
        
        return None


class ProcessExperienceManager:
    """Manager for process experiences extracted from prediction errors."""
    
    def __init__(
        self,
        log_path: str = None,
        eval_set: str = "base",
        top_k: int = 3
    ):
        """Initialize the process experience manager."""
        self.log_path = log_path
        self.eval_set = eval_set
        self.top_k = top_k
        self.knowledge_entries: List[Dict] = []
        self._embeddings_cache: Dict[str, any] = {}
        self.current_episode_knowledge: List[str] = []
        self.current_instruction: str = ""
        
        self._load_knowledge()
        
        logger.info(f"ProcessExperienceManager initialized for eval_set={eval_set}, top_k={top_k}")
    
    def set_log_path(self, log_path: str):
        """Set log path and reload knowledge."""
        self.log_path = log_path
        self._load_knowledge()
        logger.info(f"ProcessExperienceManager: log_path set to {log_path}")
    
    def set_eval_set(self, eval_set: str):
        """Set current eval set and reload knowledge."""
        self.eval_set = eval_set
        self._load_knowledge()
        logger.info(f"ProcessExperienceManager: eval_set set to {eval_set}")
    
    def set_instruction(self, instruction: str):
        """Set current task instruction."""
        self.current_instruction = instruction
    
    def _get_knowledge_path(self) -> str:
        """Get the path for knowledge file of current eval_set."""
        if not self.log_path:
            logger.warning("ProcessExperienceManager: log_path is None")
            return None
        knowledge_dir = os.path.join(self.log_path, 'process_experience')
        if not os.path.exists(knowledge_dir):
            os.makedirs(knowledge_dir)
            logger.info(f"Created process_experience directory: {knowledge_dir}")
        return os.path.join(knowledge_dir, f'{self.eval_set}_process_experiences.json')
    
    def _load_knowledge(self):
        """Load knowledge for current eval_set."""
        knowledge_path = self._get_knowledge_path()
        if knowledge_path and os.path.exists(knowledge_path):
            try:
                with open(knowledge_path, 'r', encoding='utf-8') as f:
                    loaded_data = json.load(f)
                
                self.knowledge_entries = self._normalize_knowledge_format(loaded_data)
                self._embeddings_cache.clear()
                
                total_knowledge = sum(len(entry.get('knowledge', [])) for entry in self.knowledge_entries)
                logger.info(f"Loaded {len(self.knowledge_entries)} process experience entries ({total_knowledge} total items)")
            except Exception as e:
                logger.warning(f"Failed to load process experiences: {e}")
                self.knowledge_entries = []
        else:
            self.knowledge_entries = []
            if knowledge_path:
                logger.info(f"No existing process experiences at {knowledge_path}")
    
    def _normalize_knowledge_format(self, data: List) -> List[Dict]:
        """Normalize knowledge data format."""
        if not data:
            return []
        
        if data and isinstance(data[0].get('knowledge'), str):
            instruction_knowledge_map = {}
            for entry in data:
                instr = entry.get('instruction', 'Unknown')
                knowledge = entry.get('knowledge', '')
                eval_set = entry.get('eval_set', self.eval_set)
                
                if instr not in instruction_knowledge_map:
                    instruction_knowledge_map[instr] = {
                        'instruction': instr,
                        'eval_set': eval_set,
                        'knowledge': []
                    }
                
                if knowledge and knowledge not in instruction_knowledge_map[instr]['knowledge']:
                    instruction_knowledge_map[instr]['knowledge'].append(knowledge)
            
            return list(instruction_knowledge_map.values())
        
        normalized = []
        for entry in data:
            knowledge = entry.get('knowledge', [])
            if isinstance(knowledge, str):
                knowledge = [knowledge] if knowledge else []
            normalized.append({
                'instruction': entry.get('instruction', 'Unknown'),
                'eval_set': entry.get('eval_set', self.eval_set),
                'knowledge': knowledge
            })
        return normalized
    
    def _save_knowledge(self):
        """Save knowledge for current eval_set."""
        knowledge_path = self._get_knowledge_path()
        if not knowledge_path:
            logger.warning("ProcessExperienceManager: Cannot save - log_path is None")
            return
        
        try:
            for entry in self.knowledge_entries:
                knowledge_list = entry.get('knowledge', [])
                unique_knowledge = list(dict.fromkeys(knowledge_list))
                entry['knowledge'] = unique_knowledge
            
            os.makedirs(os.path.dirname(knowledge_path), exist_ok=True)
            with open(knowledge_path, 'w', encoding='utf-8') as f:
                json.dump(self.knowledge_entries, f, ensure_ascii=False, indent=2)
            
            total_knowledge = sum(len(entry.get('knowledge', [])) for entry in self.knowledge_entries)
            logger.info(f"Saved {len(self.knowledge_entries)} process experience entries ({total_knowledge} total items)")
        except Exception as e:
            logger.error(f"Failed to save process experiences: {e}")
    
    def reset_episode(self):
        """Reset current episode entries."""
        self.current_episode_knowledge = []
        self.current_instruction = ""
        logger.debug("ProcessExperienceManager: episode reset")
    
    def _find_or_create_entry(self, instruction: str) -> Dict:
        """Find or create knowledge entry for the given instruction."""
        for entry in self.knowledge_entries:
            if entry.get('instruction') == instruction:
                return entry
        
        new_entry = {
            'instruction': instruction,
            'eval_set': self.eval_set,
            'knowledge': []
        }
        self.knowledge_entries.append(new_entry)
        return new_entry
    
    def add_entries(self, entries: List[str], instruction: str = None):
        """Add process experience entries from current episode."""
        if not entries:
            return
        
        entries = [e.strip() for e in entries if e and e.strip()]
        entries = list(dict.fromkeys(entries))
        if not entries:
            return
        
        instr = instruction or self.current_instruction or "Unknown instruction"
        
        for entry in entries:
            if entry not in self.current_episode_knowledge:
                self.current_episode_knowledge.append(entry)
        
        knowledge_entry = self._find_or_create_entry(instr)
        for entry in entries:
            if entry not in knowledge_entry['knowledge']:
                knowledge_entry['knowledge'].append(entry)
        
        logger.info(f"Added {len(entries)} process experience entries for instruction: {instr[:50]}...")
        
        self._save_knowledge()
    
    def save_episode_knowledge(self, episode_num: int):
        """Save current episode's knowledge entries to a separate file."""
        if not self.log_path:
            logger.warning("ProcessExperienceManager: Cannot save episode - log_path is None")
            return
        
        episode_knowledge_dir = os.path.join(self.log_path, 'process_experience', 'episodes')
        os.makedirs(episode_knowledge_dir, exist_ok=True)
        
        episode_file = os.path.join(episode_knowledge_dir, f'episode_{episode_num}_knowledge.json')
        
        episode_data = {
            'instruction': self.current_instruction or "Unknown instruction",
            'eval_set': self.eval_set,
            'knowledge': self.current_episode_knowledge.copy()
        }
        
        with open(episode_file, 'w', encoding='utf-8') as f:
            json.dump(episode_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved {len(self.current_episode_knowledge)} process experience entries for episode {episode_num}")
        
        self._save_knowledge()
    
    def _compute_semantic_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between two text strings using sentence embeddings."""
        model = _get_sentence_model()
        
        try:
            # Use cache for embeddings
            if text1 not in self._embeddings_cache:
                self._embeddings_cache[text1] = model.encode(text1, convert_to_numpy=True)
            if text2 not in self._embeddings_cache:
                self._embeddings_cache[text2] = model.encode(text2, convert_to_numpy=True)
            
            emb1 = self._embeddings_cache[text1]
            emb2 = self._embeddings_cache[text2]
            
            # Compute cosine similarity
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            return float(similarity)
        except Exception as e:
            logger.error(f"Error computing semantic similarity: {e}")
            raise
    
    def retrieve_knowledge(self, instruction: str) -> List[Dict]:
        """Retrieve top-k most relevant process experiences based on instruction similarity."""
        if not self.knowledge_entries:
            return []
        
        similarities = []
        for entry in self.knowledge_entries:
            sim = self._compute_semantic_similarity(instruction, entry.get('instruction', ''))
            similarities.append((entry, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        result = []
        for entry, _ in similarities:
            knowledge_list = entry.get('knowledge', [])
            for knowledge in knowledge_list:
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
        """Format retrieved process experiences for inclusion in prompt."""
        if not knowledge_list:
            return ""
        
        formatted = []
        for i, entry in enumerate(knowledge_list):
            formatted.append(PROCESS_EXPERIENCE_ENTRY_FORMAT.format(
                index=i+1,
                knowledge=entry.get('knowledge', 'No knowledge available')
            ))
        
        return PROCESS_EXPERIENCE_RETRIEVAL_CONTEXT.format(knowledge="\n".join(formatted))
    
    def get_all_knowledge_flat(self) -> List[str]:
        """Get all knowledge entries as a flat list."""
        all_knowledge = []
        for entry in self.knowledge_entries:
            knowledge_list = entry.get('knowledge', [])
            all_knowledge.extend(knowledge_list)
        return all_knowledge


def create_goal_experience_manager(
    log_path: str = None,
    eval_set: str = "base",
    top_k: int = 3,
    extractor_model: str = None,
    model_type: str = 'remote'
) -> GoalExperienceManager:
    """Factory function to create a GoalExperienceManager."""
    return GoalExperienceManager(
        log_path=log_path,
        eval_set=eval_set,
        top_k=top_k,
        extractor_model=extractor_model,
        model_type=model_type
    )


def create_process_experience_manager(
    log_path: str = None,
    eval_set: str = "base",
    top_k: int = 3
) -> ProcessExperienceManager:
    """Factory function to create a ProcessExperienceManager."""
    return ProcessExperienceManager(
        log_path=log_path,
        eval_set=eval_set,
        top_k=top_k
    )

