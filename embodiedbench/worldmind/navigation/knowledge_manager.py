"""
WorldMind Knowledge Manager Module for Navigation Environment

This module provides management for two types of learned knowledge:
1. Goal Experience: Extracted from successful task trajectories
2. Process Experience: Extracted from prediction errors during task execution

Save path structure:
- goal_experience/
    - {eval_set}_goal_experiences.json
- process_experience/
    - {eval_set}_process_experience.json
    - episodes/
        - episode_{num}_experience.json
"""

import os
import json
import re
from typing import Dict, List, Optional, Any


from embodiedbench.main import logger

# Lazy load sentence transformer to avoid import errors when not needed
_sentence_model = None

def _get_sentence_model():
    """Lazy load SentenceTransformer model for semantic similarity."""
    global _sentence_model
    if _sentence_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("SentenceTransformer model loaded for semantic retrieval")
        except ImportError:
            logger.warning("sentence-transformers not installed, falling back to basic similarity")
            _sentence_model = False
    return _sentence_model


GOAL_EXPERIENCE_EXTRACTION_PROMPT = """You are an expert in analyzing successful navigation trajectories and extracting reusable experience patterns.

## Task
Analyze the following successful navigation task execution and extract a **reusable experience** that could help similar future tasks.

## Task Category: {task_category}

## Human Instruction (Goal)
{instruction}

## Successful Trajectory
{trajectory}

## Extraction Guidelines

### What Makes a Good Navigation Experience
1. **Navigation Strategy**: How did the agent efficiently find and reach the target?
2. **Key Decision Points**: What critical choices led to success (e.g., choice of direction when blocked, decision to stop rotating)?
3. **Obstacle Handling**: How were specific obstacles (large furniture vs. walls) handled?
4. **Search Patterns**: If the target was not immediately visible, what search strategy worked?

### Key Navigation Patterns & Workflows to Look For
1. **Direct Approach Workflow**:
   - **Pattern**: When the target is visible, use primarily forward movement. Use small lateral adjustments (0.25m left/right) only for alignment or minor obstacle avoidance.
   - **Rule**: Avoid unnecessary rotations when the target is in view.

2. **Obstacle Bypass Workflow**:
   - **Pattern (Furniture)**: When blocked by large furniture (Tables, Sofas), execute a "side-step -> forward" sequence. Move laterally for 1-2 steps maximum, then immediately try forward again.
   - **Pattern (Boundaries)**: When blocked by walls/room boundaries, move parallel along the boundary or reverse direction.
   - **Anti-Oscillation**: If blocked repeatedly, switch to the opposite perpendicular direction (e.g., if left fails, try right) or rotate to find a new path.

3. **Search & Exploration Workflow**:
   - **Pattern**: When target is NOT visible, use systematic 90° rotations. Stop rotating immediately once the target appears.
   - **Rule**: Never rotate if the target is already visible.

4. **Distance & Progress Management**:
   - **Pattern**: Recognize that 0.25m steps are small; reaching distant objects requires multiple consecutive forward actions.
   - **Rule**: Implement a "3-step rule" for lateral movement: if moving sideways for 3+ steps increases distance to target, stop and reverse. The goal is proximity, not infinite avoidance.

### Format Requirements
- Extract ONE concise, generalizable navigation pattern.
- Focus on the ACTION STRATEGY and LOGIC, not just specific object names.
- Use generic terms like "target object", "obstacle", "boundary".
- Explicitly mention the **logic** behind the success (e.g., "by limiting lateral steps to 2, the agent avoided drifting away").

## Output Format
Output ONLY a JSON object:
{{"goal_experience": "A complete navigation pattern description (3-5 sentences) that includes: the goal type, the specific navigation workflow used (Direct Approach/Obstacle Bypass/Search), how critical obstacles were handled (e.g., side-stepping vs rotating), and the conditions that confirmed success (e.g., stopping when distance minimized)."}}

!!! Please output only the JSON without any markdown formatting or code blocks.
"""


GOAL_EXPERIENCE_RETRIEVAL_CONTEXT = '''
## Relevant Goal Experiences from Past Successful Navigation Tasks
Based on similar tasks, here are navigation experiences that led to success:

{experiences}

IMPORTANT: These experiences highlight effective navigation strategies. Adapt them to your current task, focusing on similar spatial reasoning and movement patterns.
'''


GOAL_EXPERIENCE_ENTRY_FORMAT = '''
### Goal Experience {index}
**Original Task**: {instruction}
**Experience**: {experience}
'''


PROCESS_EXPERIENCE_RETRIEVAL_CONTEXT = '''
## Relevant Process Experiences from Past Navigation
Based on similar tasks, here are process experience entries learned from prediction errors:

{knowledge}

IMPORTANT: Apply these navigation rules to avoid similar prediction errors.
'''


PROCESS_EXPERIENCE_ENTRY_FORMAT = '''
### Process Experience {index}
**Task Context**: {instruction}
**Experience**: {knowledge}
'''


TRAJECTORY_STEP_FORMAT = '''Step {step_num}:
- Action: {action_name} (ID: {action_id})
- Environment Feedback: {env_feedback}
- Result: {result}
'''


class GoalExperienceManager:
    """Manager for goal experiences extracted from successful navigation task trajectories."""
    
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
        self.episode_trajectory: List[Dict] = []
        self.actions: List[str] = []
        self.current_instruction: str = ""
        self._embeddings_cache: Dict[str, any] = {}
        
        self.extractor = None
        self.extractor_model = extractor_model
        if extractor_model:
            try:
                from openai import OpenAI
                api_key = os.environ.get('OPENAI_API_KEY')
                api_base = os.environ.get('OPENAI_API_BASE', None)
                if api_key:
                    if api_base:
                        self.extractor = OpenAI(api_key=api_key, base_url=api_base)
                    else:
                        self.extractor = OpenAI(api_key=api_key)
                    logger.info(f"GoalExperienceManager: Extractor model initialized: {extractor_model}")
            except Exception as e:
                logger.warning(f"Failed to initialize extractor model: {e}")
                self.extractor = None
        
        self._load_experiences()
        
        logger.info(f"Navigation GoalExperienceManager initialized for eval_set={eval_set}, top_k={top_k}")
    
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
            seen = set()
            for exp in self.experiences:
                exp_text = exp.get('goal_experience', '')
                if exp_text and exp_text not in seen:
                    seen.add(exp_text)
                    unique_experiences.append(exp)
            self.experiences = unique_experiences
            
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
            'last_action_success': info.get('last_action_success', 1)
        }
        self.episode_trajectory.append(entry)
        logger.debug(f"GoalExperienceManager: trajectory updated, now {len(self.episode_trajectory)} steps")
    
    def add_experience(self, experience: Dict):
        """Add a new experience."""
        self.experiences.append(experience)
        self._save_experiences()
    
    def _compute_semantic_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity using SentenceTransformer."""
        model = _get_sentence_model()
        if model is False or model is None:
            # Fallback to basic word overlap
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            if not words1 or not words2:
                return 0.0
            intersection = words1 & words2
            union = words1 | words2
            return len(intersection) / len(union) if union else 0.0
        
        try:
            import numpy as np
            
            # Cache embeddings for efficiency
            if text1 not in self._embeddings_cache:
                self._embeddings_cache[text1] = model.encode(text1, convert_to_numpy=True)
            if text2 not in self._embeddings_cache:
                self._embeddings_cache[text2] = model.encode(text2, convert_to_numpy=True)
            
            emb1 = self._embeddings_cache[text1]
            emb2 = self._embeddings_cache[text2]
            
            # Cosine similarity
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            return float(similarity)
        except Exception as e:
            logger.warning(f"Semantic similarity computation failed: {e}")
            return 0.0
    
    def retrieve_experiences(self, instruction: str) -> List[Dict]:
        """Retrieve top-k most relevant goal experiences based on instruction similarity."""
        if not self.experiences:
            return []
        
        similarities = []
        for exp in self.experiences:
            sim = self._compute_semantic_similarity(instruction, exp.get('instruction', ''))
            similarities.append((exp, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [exp for exp, _ in similarities[:self.top_k]]
    
    def retrieve_similar(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        """Retrieve similar experiences."""
        if top_k:
            old_top_k = self.top_k
            self.top_k = top_k
            result = self.retrieve_experiences(query)
            self.top_k = old_top_k
            return result
        return self.retrieve_experiences(query)
    
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
        logger.info(f"GoalExperienceManager.extract_experience called: task_success={task_success}, trajectory_len={len(self.episode_trajectory)}")
        
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
        for i, step in enumerate(self.episode_trajectory):
            action_id = step.get('action_id', -1)
            action_name = self.actions[action_id] if 0 <= action_id < len(self.actions) else "unknown"
            result = "Success" if step.get('last_action_success', 1) else "Failed"
            
            trajectory_desc += TRAJECTORY_STEP_FORMAT.format(
                step_num=i+1,
                action_name=action_name,
                action_id=action_id,
                env_feedback=step.get('env_feedback', ''),
                result=result
            )
        
        logger.info(f"Built trajectory description with {len(self.episode_trajectory)} steps")
        
        extraction_prompt = GOAL_EXPERIENCE_EXTRACTION_PROMPT.format(
            instruction=instruction,
            trajectory=trajectory_desc,
            task_category=self.eval_set
        )
        
        try:
            response = self.extractor.chat.completions.create(
                model=self.extractor_model,
                messages=[{"role": "user", "content": extraction_prompt}],
                temperature=0,
                max_tokens=512
            )
            
            result_text = response.choices[0].message.content
            logger.debug(f"Extractor response: {result_text[:500]}...")
            
            json_match = re.search(r'\{[^{}]*"goal_experience"[^{}]*\}', result_text, re.DOTALL)
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
                    
                    logger.info(f"Successfully extracted and saved experience for: {instruction[:50]}...")
                    return experience_entry
            else:
                try:
                    result = json.loads(result_text)
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
                        logger.info(f"Successfully extracted experience (direct parse): {instruction[:50]}...")
                        return experience_entry
                except json.JSONDecodeError:
                    logger.warning("Failed to parse goal_experience from response")
                    
        except Exception as e:
            logger.error(f"Goal experience extraction failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())
        
        return None
    
    def save_experiences(self, path: Optional[str] = None):
        """Save experiences to file."""
        if path:
            try:
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(self.experiences, f, ensure_ascii=False, indent=2)
                logger.info(f"Saved experiences to {path}")
            except Exception as e:
                logger.error(f"Failed to save experiences to {path}: {e}")
        else:
            self._save_experiences()


class ProcessExperienceManager:
    """Manager for process experience extracted from prediction errors."""
    
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
        self.experience_entries: List[Dict] = []
        self.current_episode_experience: List[str] = []
        self.current_instruction: str = ""
        self._embeddings_cache: Dict[str, any] = {}
        
        self._load_experience()
        
        logger.info(f"Navigation ProcessExperienceManager initialized for eval_set={eval_set}, top_k={top_k}")
    
    def set_log_path(self, log_path: str):
        """Set log path and reload experience."""
        self.log_path = log_path
        self._load_experience()
        logger.info(f"ProcessExperienceManager: log_path set to {log_path}")
    
    def set_eval_set(self, eval_set: str):
        """Set current eval set and reload experience."""
        self.eval_set = eval_set
        self._load_experience()
        logger.info(f"ProcessExperienceManager: eval_set set to {eval_set}")
    
    def set_instruction(self, instruction: str):
        """Set current task instruction."""
        self.current_instruction = instruction
    
    def _get_experience_path(self) -> str:
        """Get the path for experience file of current eval_set."""
        if not self.log_path:
            logger.warning("ProcessExperienceManager: log_path is None")
            return None
        experience_dir = os.path.join(self.log_path, 'process_experience')
        if not os.path.exists(experience_dir):
            os.makedirs(experience_dir)
        return os.path.join(experience_dir, f'{self.eval_set}_process_experience.json')
    
    def _load_experience(self):
        """Load experience for current eval_set."""
        experience_path = self._get_experience_path()
        if experience_path and os.path.exists(experience_path):
            try:
                with open(experience_path, 'r', encoding='utf-8') as f:
                    loaded_data = json.load(f)
                self.experience_entries = self._normalize_experience_format(loaded_data)
                self._embeddings_cache.clear()
                total = sum(len(e.get('experience', [])) for e in self.experience_entries)
                logger.info(f"Loaded {len(self.experience_entries)} process experience entries ({total} items) from {experience_path}")
            except Exception as e:
                logger.warning(f"Failed to load process experience: {e}")
                self.experience_entries = []
        else:
            self.experience_entries = []
            if experience_path:
                logger.info(f"No existing process experience at {experience_path}")
    
    def _normalize_experience_format(self, data: List) -> List[Dict]:
        """Normalize experience data to new format."""
        if not data:
            return []
        
        if data and isinstance(data[0].get('experience'), str):
            instruction_map = {}
            for entry in data:
                instr = entry.get('instruction', 'Unknown')
                experience = entry.get('experience', '')
                eval_set = entry.get('eval_set', self.eval_set)
                
                if instr not in instruction_map:
                    instruction_map[instr] = {
                        'instruction': instr,
                        'eval_set': eval_set,
                        'experience': []
                    }
                
                if experience and experience not in instruction_map[instr]['experience']:
                    instruction_map[instr]['experience'].append(experience)
            
            return list(instruction_map.values())
        
        normalized = []
        for entry in data:
            experience = entry.get('experience', [])
            if isinstance(experience, str):
                experience = [experience] if experience else []
            normalized.append({
                'instruction': entry.get('instruction', 'Unknown'),
                'eval_set': entry.get('eval_set', self.eval_set),
                'experience': experience
            })
        return normalized
    
    def _save_experience(self):
        """Save experience for current eval_set."""
        experience_path = self._get_experience_path()
        if not experience_path:
            logger.warning("ProcessExperienceManager: Cannot save - log_path is None")
            return
        
        try:
            for entry in self.experience_entries:
                entry['experience'] = list(dict.fromkeys(entry.get('experience', [])))
            
            os.makedirs(os.path.dirname(experience_path), exist_ok=True)
            with open(experience_path, 'w', encoding='utf-8') as f:
                json.dump(self.experience_entries, f, ensure_ascii=False, indent=2)
            
            total = sum(len(e.get('experience', [])) for e in self.experience_entries)
            logger.info(f"Saved {len(self.experience_entries)} process experience entries ({total} items) to {experience_path}")
        except Exception as e:
            logger.error(f"Failed to save process experience: {e}")
    
    def reset_episode(self):
        """Reset current episode entries."""
        self.current_episode_experience = []
        self.current_instruction = ""
        logger.debug("ProcessExperienceManager: episode reset")
    
    def _find_or_create_entry(self, instruction: str) -> Dict:
        """Find or create experience entry for instruction."""
        for entry in self.experience_entries:
            if entry.get('instruction') == instruction:
                return entry
        
        new_entry = {
            'instruction': instruction,
            'eval_set': self.eval_set,
            'experience': []
        }
        self.experience_entries.append(new_entry)
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
            if entry not in self.current_episode_experience:
                self.current_episode_experience.append(entry)
        
        experience_entry = self._find_or_create_entry(instr)
        for entry in entries:
            if entry not in experience_entry['experience']:
                experience_entry['experience'].append(entry)
        
        logger.info(f"Added {len(entries)} process experience entries for: {instr[:50]}...")
        self._save_experience()
    
    def _compute_semantic_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity using SentenceTransformer."""
        model = _get_sentence_model()
        if model is False or model is None:
            # Fallback to basic word overlap
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            if not words1 or not words2:
                return 0.0
            intersection = words1 & words2
            union = words1 | words2
            return len(intersection) / len(union) if union else 0.0
        
        try:
            import numpy as np
            
            # Cache embeddings for efficiency
            if text1 not in self._embeddings_cache:
                self._embeddings_cache[text1] = model.encode(text1, convert_to_numpy=True)
            if text2 not in self._embeddings_cache:
                self._embeddings_cache[text2] = model.encode(text2, convert_to_numpy=True)
            
            emb1 = self._embeddings_cache[text1]
            emb2 = self._embeddings_cache[text2]
            
            # Cosine similarity
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            return float(similarity)
        except Exception as e:
            logger.warning(f"Semantic similarity computation failed: {e}")
            return 0.0
    
    def retrieve_experience(self, instruction: str) -> List[Dict]:
        """Retrieve top-k most relevant process experience based on instruction similarity."""
        if not self.experience_entries:
            return []
        
        similarities = []
        for entry in self.experience_entries:
            sim = self._compute_semantic_similarity(instruction, entry.get('instruction', ''))
            similarities.append((entry, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        result = []
        for entry, _ in similarities:
            for experience in entry.get('experience', []):
                result.append({
                    'instruction': entry.get('instruction', 'Unknown'),
                    'experience': experience
                })
                if len(result) >= self.top_k:
                    break
            if len(result) >= self.top_k:
                break
        
        return result
    
    def retrieve_relevant(self, query: str, top_k: Optional[int] = None) -> List[str]:
        """Retrieve relevant experience entries."""
        old_top_k = self.top_k
        if top_k:
            self.top_k = top_k
        
        result = self.retrieve_experience(query)
        self.top_k = old_top_k
        
        return [r.get('experience', '') for r in result]
    
    def format_experience_for_prompt(self, experience_list: List[Dict]) -> str:
        """Format retrieved process experience for inclusion in prompt."""
        if not experience_list:
            return ""
        
        formatted = []
        for i, entry in enumerate(experience_list):
            formatted.append(PROCESS_EXPERIENCE_ENTRY_FORMAT.format(
                index=i+1,
                instruction=entry.get('instruction', 'Unknown'),
                knowledge=entry.get('experience', 'No experience available')
            ))
        
        return PROCESS_EXPERIENCE_RETRIEVAL_CONTEXT.format(knowledge="\n".join(formatted))
    
    def save_episode_experience(self, episode_num: int):
        """Save current episode's experience entries to a separate file."""
        if not self.log_path:
            logger.warning("ProcessExperienceManager: Cannot save episode - log_path is None")
            return
        
        episode_experience_dir = os.path.join(self.log_path, 'process_experience', 'episodes')
        os.makedirs(episode_experience_dir, exist_ok=True)
        
        episode_file = os.path.join(episode_experience_dir, f'episode_{episode_num}_experience.json')
        
        episode_data = {
            'instruction': self.current_instruction or "Unknown instruction",
            'eval_set': self.eval_set,
            'experience': self.current_episode_experience.copy()
        }
        
        with open(episode_file, 'w', encoding='utf-8') as f:
            json.dump(episode_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved {len(self.current_episode_experience)} experience entries for episode {episode_num}")
        self._save_experience()
    
    def save_experience(self, path: Optional[str] = None):
        """Save experience to file."""
        if path:
            try:
                to_save = [e for e in self.experience_entries if e.get('experience')]
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(to_save, f, ensure_ascii=False, indent=2)
                logger.info(f"Saved experience to {path}")
            except Exception as e:
                logger.error(f"Failed to save experience to {path}: {e}")
        else:
            self._save_experience()
    
    def get_all(self) -> List[str]:
        """Get all experience items as a flat list."""
        all_experience = []
        for entry in self.experience_entries:
            all_experience.extend(entry.get('experience', []))
        return all_experience


def create_goal_experience_manager(
    log_path: str = None,
    eval_set: str = "base",
    top_k: int = 3,
    extractor_model: str = None,
    model_type: str = 'remote'
) -> GoalExperienceManager:
    """Factory function to create a Goal Experience Manager."""
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
    """Factory function to create a Process Experience Manager."""
    return ProcessExperienceManager(
        log_path=log_path,
        eval_set=eval_set,
        top_k=top_k
    )
