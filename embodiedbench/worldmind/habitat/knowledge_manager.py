import os
import json
import re
from typing import Dict, List, Optional, Tuple, Any
from difflib import SequenceMatcher

from embodiedbench.planner.remote_model import RemoteModel
from embodiedbench.planner.planner_utils import fix_json
from embodiedbench.main import logger


_RETRIEVAL_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "into", "onto",
    "move", "bring", "take", "find", "need", "left", "right", "counter",
    "kitchen", "table", "object", "target", "place", "located", "navigate",
    "pick", "task", "instruction", "items", "item", "one"
}


def _retrieval_keywords(text: str) -> set:
    tokens = set(re.findall(r"[a-zA-Z][a-zA-Z_ -]*[a-zA-Z]|[a-zA-Z]", (text or "").lower()))
    expanded = set()
    for token in tokens:
        for part in re.split(r"[^a-zA-Z]+", token):
            part = part.strip().lower()
            if len(part) >= 3 and part not in _RETRIEVAL_STOPWORDS:
                expanded.add(part)
    return expanded


def _keyword_overlap_score(query: str, candidate: str) -> float:
    query_keywords = _retrieval_keywords(query)
    candidate_keywords = _retrieval_keywords(candidate)
    if not query_keywords or not candidate_keywords:
        return 0.0
    overlap = query_keywords & candidate_keywords
    if not overlap:
        return 0.0
    return len(overlap) / max(1, min(len(query_keywords), len(candidate_keywords)))





SUCCESS_EXPERIENCE_EXTRACTION_PROMPT = '''You are a task analysis expert. Given a successful task execution trajectory with ordinary environment feedback, extract a SINGLE reusable workflow that captures the key actions that led to success.

## Task Instruction
{instruction}

## Successful Execution Trajectory
{trajectory}

## Task Category
{task_category}

## CRITICAL WORKFLOW EXTRACTION GUIDELINES

### Focus on Executable Feedback
The trajectory includes only ordinary action results and environment feedback. Pay SPECIAL ATTENTION to:
1. **Actions that executed successfully** - These form the reliable workflow backbone
2. **Invalid-action feedback** - This reveals preconditions such as needing to navigate closer, pick before placing, or open before accessing
3. **The sequence of successful actions after failures** - This reveals how the agent recovered

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
{{"success_experience": "A complete workflow description that includes: (1) The goal pattern, (2) the key executable action sequence, (3) critical decision points and preconditions, (4) success conditions. Should be 3-6 sentences that can guide similar tasks."}}

!!! Please output only the JSON without any markdown formatting or code blocks. Output ONLY the "success_experience" field, no other fields needed.
'''



SUCCESS_EXPERIENCE_RETRIEVAL_CONTEXT = '''
## Relevant Success Experiences from Past Successful Executions
Based on similar tasks, here are success experiences that led to task completion. Pay attention to the key executable action sequence and preconditions:
{experiences}
IMPORTANT: Adapt them to your current task, focusing on the same logical patterns and decision points.
'''

SUCCESS_EXPERIENCE_ENTRY_FORMAT = '''
### Success Experience {index}
**Original Task**: {instruction}
**Experience**: {experience}
'''






WORLD_KNOWLEDGE_RETRIEVAL_CONTEXT = '''
## Relevant World Knowledge from Past Experiences
Based on similar tasks, here are world knowledge entries learned from prediction errors:
{knowledge}
IMPORTANT: Pay special attention to 'Location Knowledge' entries. These allow you to locate objects quickly and accurately, preventing blind navigation and random searching.
'''

WORLD_KNOWLEDGE_ENTRY_FORMAT = '''
### World Knowledge {index}
**Task Context**: {instruction}
**Knowledge**: {knowledge}
'''




WORLD_KNOWLEDGE_ENTRY_FORMAT = '''
### World Knowledge {index}
**Knowledge**: {knowledge}
'''






TRAJECTORY_STEP_FORMAT = '''Step {step_num}:
- Action: {action_name} (ID: {action_id})
- Environment Feedback: {env_feedback}
- Result: {result}
'''


class SuccessExperienceManager:

    def __init__(
        self,
        log_path: str = None,
        eval_set: str = "base",
        top_k: int = 3,
        extractor_model: str = None,
        model_type: str = 'remote',
        cross_eval_set_retrieval: bool = True,
        all_eval_sets: List[str] = None,
        enable_experience_transfer: bool = False,
        experience_transfer_path: str = None
    ):
        self.log_path = log_path
        self.eval_set = eval_set
        self.top_k = top_k
        self.cross_eval_set_retrieval = cross_eval_set_retrieval
        self.all_eval_sets = all_eval_sets or []
        self.experiences: List[Dict] = []
        self.all_experiences: List[Dict] = []
        self.episode_trajectory: List[Dict] = []
        self.actions: List[str] = []
        self.current_instruction: str = ""


        self.enable_experience_transfer = enable_experience_transfer
        self.experience_transfer_path = experience_transfer_path


        self.enable_experience_transfer = enable_experience_transfer
        self.experience_transfer_path = experience_transfer_path


        self.extractor = None
        if extractor_model:
            try:
                self.extractor = RemoteModel(
                    extractor_model,
                    model_type=model_type,
                    language_only=True
                )
                logger.info(f"SuccessExperienceManager: Extractor model initialized: {extractor_model}")
            except Exception as e:
                logger.warning(f"Failed to initialize extractor model: {e}")
                self.extractor = None


        if self.enable_experience_transfer and self.experience_transfer_path:

            self._load_transfer_experiences()
            logger.info(f"SuccessExperienceManager initialized in TRANSFER mode from {self.experience_transfer_path}")
        else:

            self._load_experiences()


            if self.cross_eval_set_retrieval:
                self._load_all_experiences()

        logger.info(f"SuccessExperienceManager initialized for eval_set={eval_set}, top_k={top_k}, cross_eval_set_retrieval={cross_eval_set_retrieval}, experience_transfer={enable_experience_transfer}")

    def set_log_path(self, log_path: str):
        self.log_path = log_path
        self._load_experiences()
        logger.info(f"SuccessExperienceManager: log_path set to {log_path}")

    def set_eval_set(self, eval_set: str):
        self.eval_set = eval_set

        if self.enable_experience_transfer and self.experience_transfer_path:
            self._load_transfer_experiences()
        else:
            self._load_experiences()

            if self.cross_eval_set_retrieval:
                self._load_all_experiences()
        logger.info(f"SuccessExperienceManager: eval_set set to {eval_set}")

    def set_all_eval_sets(self, all_eval_sets: List[str]):
        self.all_eval_sets = all_eval_sets
        if self.cross_eval_set_retrieval:
            self._load_all_experiences()
        logger.info(f"SuccessExperienceManager: all_eval_sets set to {all_eval_sets}")

    def set_cross_eval_set_retrieval(self, enabled: bool):
        self.cross_eval_set_retrieval = enabled
        if enabled:
            self._load_all_experiences()
        logger.info(f"SuccessExperienceManager: cross_eval_set_retrieval set to {enabled}")

    def set_actions(self, actions: List[str]):
        self.actions = actions

    def set_instruction(self, instruction: str):
        self.current_instruction = instruction


    def _get_experience_path(self) -> str:
        if not self.log_path:
            logger.warning("SuccessExperienceManager: log_path is None")
            return None
        experience_dir = os.path.join(self.log_path, 'success_experience')
        if not os.path.exists(experience_dir):
            os.makedirs(experience_dir)
            logger.info(f"Created success_experience directory: {experience_dir}")
        return os.path.join(experience_dir, f'{self.eval_set}_success_experiences.json')

    def _load_experiences(self):
        experience_path = self._get_experience_path()
        if experience_path and os.path.exists(experience_path):
            try:
                with open(experience_path, 'r', encoding='utf-8') as f:
                    self.experiences = json.load(f)
                logger.info(f"Loaded {len(self.experiences)} success experiences from {experience_path}")
            except Exception as e:
                logger.warning(f"Failed to load success experiences: {e}")
                self.experiences = []
        else:
            self.experiences = []
            if experience_path:
                logger.info(f"No existing success experiences at {experience_path}")

    def _load_all_experiences(self):
        if not self.log_path or not self.all_eval_sets:
            self.all_experiences = self.experiences.copy()
            return

        self.all_experiences = []
        experience_dir = os.path.join(self.log_path, 'success_experience')

        for eval_set_name in self.all_eval_sets:
            experience_file = os.path.join(experience_dir, f'{eval_set_name}_success_experiences.json')
            if os.path.exists(experience_file):
                try:
                    with open(experience_file, 'r', encoding='utf-8') as f:
                        experiences = json.load(f)
                        self.all_experiences.extend(experiences)
                    logger.debug(f"Loaded {len(experiences)} experiences from {eval_set_name}")
                except Exception as e:
                    logger.warning(f"Failed to load experiences from {eval_set_name}: {e}")

        logger.info(f"Cross-set retrieval: Loaded {len(self.all_experiences)} total experiences from {len(self.all_eval_sets)} eval_sets")

    def _load_transfer_experiences(self):
        if not self.experience_transfer_path:
            logger.warning("SuccessExperienceManager: experience_transfer_path is None")
            return


        transfer_experience_path = os.path.join(
            self.experience_transfer_path,
            self.eval_set,
            'success_experience',
            f'{self.eval_set}_success_experiences.json'
        )

        if os.path.exists(transfer_experience_path):
            try:
                with open(transfer_experience_path, 'r', encoding='utf-8') as f:
                    self.experiences = json.load(f)
                logger.info(f"[TRANSFER] Loaded {len(self.experiences)} success experiences from {transfer_experience_path}")
            except Exception as e:
                logger.warning(f"Failed to load transfer experiences: {e}")
                self.experiences = []
        else:
            logger.warning(f"Transfer experience file not found: {transfer_experience_path}")
            self.experiences = []


        if self.cross_eval_set_retrieval:
            self._load_all_transfer_experiences()

    def _load_all_transfer_experiences(self):
        if not self.experience_transfer_path or not self.all_eval_sets:
            self.all_experiences = self.experiences.copy()
            return

        self.all_experiences = []

        for eval_set_name in self.all_eval_sets:
            transfer_experience_path = os.path.join(
                self.experience_transfer_path,
                eval_set_name,
                'success_experience',
                f'{eval_set_name}_success_experiences.json'
            )

            if os.path.exists(transfer_experience_path):
                try:
                    with open(transfer_experience_path, 'r', encoding='utf-8') as f:
                        experiences = json.load(f)
                        self.all_experiences.extend(experiences)
                    logger.debug(f"[TRANSFER] Loaded {len(experiences)} experiences from {eval_set_name}")
                except Exception as e:
                    logger.warning(f"Failed to load transfer experiences from {eval_set_name}: {e}")

        logger.info(f"[TRANSFER] Cross-set retrieval: Loaded {len(self.all_experiences)} total experiences from {len(self.all_eval_sets)} eval_sets")

    def set_experience_transfer(self, enabled: bool, path: str = None):
        self.enable_experience_transfer = enabled
        self.experience_transfer_path = path

        if enabled and path:
            self._load_transfer_experiences()
            logger.info(f"SuccessExperienceManager: Experience transfer enabled from {path}")
        else:

            self._load_experiences()
            if self.cross_eval_set_retrieval:
                self._load_all_experiences()
            logger.info(f"SuccessExperienceManager: Experience transfer disabled, using local experiences")

    def reload_experiences_for_eval_set(self):
        if self.enable_experience_transfer and self.experience_transfer_path:
            self._load_transfer_experiences()
        else:
            self._load_experiences()
            if self.cross_eval_set_retrieval:
                self._load_all_experiences()

    def _save_experiences(self):
        experience_path = self._get_experience_path()
        if not experience_path:
            logger.warning("SuccessExperienceManager: Cannot save - log_path is None")
            return

        try:

            unique_experiences = []
            seen_experiences = set()
            for exp in self.experiences:

                exp_text = exp.get('success_experience', '')
                if exp_text and exp_text not in seen_experiences:
                    seen_experiences.add(exp_text)
                    unique_experiences.append(exp)

            original_count = len(self.experiences)
            self.experiences = unique_experiences

            if original_count != len(self.experiences):
                logger.info(f"Removed {original_count - len(self.experiences)} duplicate success experiences")


            os.makedirs(os.path.dirname(experience_path), exist_ok=True)
            with open(experience_path, 'w', encoding='utf-8') as f:
                json.dump(self.experiences, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved {len(self.experiences)} success experiences to {experience_path}")
        except Exception as e:
            logger.error(f"Failed to save success experiences: {e}")


    def reset_trajectory(self):
        self.episode_trajectory = []
        self.current_instruction = ""
        logger.debug("SuccessExperienceManager: trajectory reset")

    def update_trajectory(self, info: Dict):
        entry = {
            'action_id': info.get('action_id', -1),
            'env_feedback': info.get('env_feedback', ''),
            'last_action_success': info.get('last_action_success', 1)
        }
        self.episode_trajectory.append(entry)
        logger.debug(f"SuccessExperienceManager: trajectory updated, now {len(self.episode_trajectory)} steps")

    def _compute_text_similarity(self, text1: str, text2: str) -> float:
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

    def _get_prior_instructions(self, current_instruction: str) -> set:
        if not self.enable_experience_transfer or not self.experience_transfer_path:
            return set()

        results_dir = os.path.join(self.experience_transfer_path, self.eval_set, 'results')

        if not os.path.exists(results_dir):
            logger.warning(f"Results directory not found: {results_dir}")
            return set()


        episode_files = []
        try:
            for filename in os.listdir(results_dir):
                if filename.startswith('episode_') and filename.endswith('_final_res.json'):

                    parts = filename.replace('episode_', '').replace('_final_res.json', '')
                    try:
                        episode_num = int(parts)
                        episode_files.append((episode_num, filename))
                    except ValueError:
                        continue
        except Exception as e:
            logger.warning(f"Failed to list results directory: {e}")
            return set()


        episode_files.sort(key=lambda x: x[0])


        current_instruction_normalized = current_instruction.strip().lower()
        instructions_with_order = []
        current_episode_num = None

        for episode_num, filename in episode_files:
            filepath = os.path.join(results_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    instr = data.get('instruction', '')
                    instructions_with_order.append((episode_num, instr))


                    if instr.strip().lower() == current_instruction_normalized:
                        current_episode_num = episode_num
            except Exception as e:
                logger.debug(f"Failed to read episode file {filename}: {e}")
                continue

        if current_episode_num is None:

            logger.debug(f"Current instruction not found in results files: {current_instruction[:50]}...")
            return set()


        prior_instructions = set()
        for episode_num, instr in instructions_with_order:
            if episode_num < current_episode_num:
                prior_instructions.add(instr.strip().lower())

        logger.debug(f"Found {len(prior_instructions)} prior instructions before episode {current_episode_num}")
        return prior_instructions


    def retrieve_experiences(self, instruction: str, filter_by_temporal_order: bool = True) -> List[Dict]:

        if self.cross_eval_set_retrieval and self.all_experiences:
            experiences_to_search = self.all_experiences
            logger.debug(f"Cross-set retrieval: searching from {len(experiences_to_search)} experiences across all eval_sets")
        else:
            experiences_to_search = self.experiences
            logger.debug(f"Single-set retrieval: searching from {len(experiences_to_search)} experiences in {self.eval_set}")

        if not experiences_to_search:
            return []

        if not experiences_to_search:
            return []



        instruction_normalized = instruction.strip().lower()
        filtered_experiences = [
            exp for exp in experiences_to_search
            if exp.get('instruction', '').strip().lower() != instruction_normalized
        ]


        if self.enable_experience_transfer and filter_by_temporal_order:
            prior_instructions = self._get_prior_instructions(instruction)
            if prior_instructions:
                before_filter_count = len(filtered_experiences)
                filtered_experiences = [
                    exp for exp in filtered_experiences
                    if exp.get('instruction', '').strip().lower() in prior_instructions
                ]
                logger.debug(f"Temporal filtering: {before_filter_count} -> {len(filtered_experiences)} experiences")

        if len(filtered_experiences) < len(experiences_to_search):
            excluded_count = len(experiences_to_search) - len(filtered_experiences)
            logger.debug(f"Excluded {excluded_count} experiences with same instruction as current task")

        if not filtered_experiences:
            logger.debug("No experiences left after excluding current task's experiences")
            return []


        similarities = []
        for exp in filtered_experiences:
            exp_instruction = exp.get('instruction', '')
            exp_text = exp.get('success_experience', '')
            sim = self._compute_text_similarity(instruction, exp_instruction)
            sim += 0.35 * _keyword_overlap_score(instruction, exp_instruction)
            sim += 0.25 * _keyword_overlap_score(instruction, exp_text)
            similarities.append((exp, sim))


        similarities.sort(key=lambda x: x[1], reverse=True)
        return [exp for exp, _ in similarities[:self.top_k]]

    def format_experiences_for_prompt(self, experiences: List[Dict]) -> str:
        if not experiences:
            return ""

        formatted = []
        for i, exp in enumerate(experiences):
            formatted.append(SUCCESS_EXPERIENCE_ENTRY_FORMAT.format(
                index=i+1,
                instruction=exp.get('instruction', 'Unknown'),
                experience=exp.get('success_experience', 'No experience available')
            ))

        return SUCCESS_EXPERIENCE_RETRIEVAL_CONTEXT.format(experiences="\n".join(formatted))

    def extract_experience(self, instruction: str, task_success: bool) -> Optional[Dict]:
        logger.info(f"SuccessExperienceManager.extract_experience called: task_success={task_success}, trajectory_len={len(self.episode_trajectory)}")

        if not task_success:
            logger.info("Task not successful, skipping experience extraction")
            return None

        if not self.episode_trajectory:
            logger.warning("No trajectory data available for experience extraction")
            return None

        if not self.extractor:
            logger.warning("No extractor model configured, cannot extract success experience")
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


        extraction_prompt = SUCCESS_EXPERIENCE_EXTRACTION_PROMPT.format(
            instruction=instruction,
            trajectory=trajectory_desc,
            task_category=self.eval_set
        )

        messages = [{"role": "user", "content": [{"type": "text", "text": extraction_prompt}]}]

        try:
            response = self.extractor.respond(messages)
            response = fix_json(response)

            logger.debug(f"Extractor response: {response[:500]}...")


            json_match = re.search(r'\{[^{}]*"success_experience"[^{}]*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                experience_text = result.get('success_experience', '')

                if experience_text:

                    experience_entry = {
                        'instruction': instruction,
                        'success_experience': experience_text,
                        'eval_set': self.eval_set,
                        'trajectory_length': len(self.episode_trajectory)
                    }


                    self.experiences.append(experience_entry)


                    self._save_experiences()

                    logger.info(f"Successfully extracted and saved experience for: {instruction[:50]}...")
                    return experience_entry
                else:
                    logger.warning("Extracted experience text is empty")
            else:

                try:
                    result = json.loads(response)
                    experience_text = result.get('success_experience', '')
                    if experience_text:
                        experience_entry = {
                            'instruction': instruction,
                            'success_experience': experience_text,
                            'eval_set': self.eval_set,
                            'trajectory_length': len(self.episode_trajectory)
                        }
                        self.experiences.append(experience_entry)
                        self._save_experiences()
                        logger.info(f"Successfully extracted experience (direct parse): {instruction[:50]}...")
                        return experience_entry
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse success_experience from response")

        except Exception as e:
            logger.error(f"Success experience extraction failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())

        return None


class WorldKnowledgeManager:

    def __init__(
        self,
        log_path: str = None,
        eval_set: str = "base",
        top_k: int = 3,
        cross_eval_set_retrieval: bool = True,
        all_eval_sets: List[str] = None,
        enable_experience_transfer: bool = False,
        experience_transfer_path: str = None
    ):
        self.log_path = log_path
        self.eval_set = eval_set
        self.top_k = top_k
        self.cross_eval_set_retrieval = cross_eval_set_retrieval
        self.all_eval_sets = all_eval_sets or []
        self.knowledge_entries: List[Dict] = []
        self.all_knowledge_entries: List[Dict] = []
        self.current_episode_knowledge: List[str] = []
        self.current_instruction: str = ""


        self.enable_experience_transfer = enable_experience_transfer
        self.experience_transfer_path = experience_transfer_path


        if self.enable_experience_transfer and self.experience_transfer_path:

            self._load_transfer_knowledge()
            logger.info(f"WorldKnowledgeManager initialized in TRANSFER mode from {self.experience_transfer_path}")
        else:

            self._load_knowledge()


            if self.cross_eval_set_retrieval:
                self._load_all_knowledge()

        logger.info(f"WorldKnowledgeManager initialized for eval_set={eval_set}, top_k={top_k}, cross_eval_set_retrieval={cross_eval_set_retrieval}, experience_transfer={enable_experience_transfer}")

    def set_log_path(self, log_path: str):
        self.log_path = log_path
        self._load_knowledge()
        logger.info(f"WorldKnowledgeManager: log_path set to {log_path}")

    def set_eval_set(self, eval_set: str):
        self.eval_set = eval_set

        if self.enable_experience_transfer and self.experience_transfer_path:
            self._load_transfer_knowledge()
        else:
            self._load_knowledge()

            if self.cross_eval_set_retrieval:
                self._load_all_knowledge()
        logger.info(f"WorldKnowledgeManager: eval_set set to {eval_set}")

    def set_all_eval_sets(self, all_eval_sets: List[str]):
        self.all_eval_sets = all_eval_sets
        if self.cross_eval_set_retrieval:
            self._load_all_knowledge()
        logger.info(f"WorldKnowledgeManager: all_eval_sets set to {all_eval_sets}")

    def set_cross_eval_set_retrieval(self, enabled: bool):
        self.cross_eval_set_retrieval = enabled
        if enabled:
            self._load_all_knowledge()
        logger.info(f"WorldKnowledgeManager: cross_eval_set_retrieval set to {enabled}")

    def set_instruction(self, instruction: str):
        self.current_instruction = instruction


    def _get_knowledge_path(self) -> str:
        if not self.log_path:
            logger.warning("WorldKnowledgeManager: log_path is None")
            return None
        knowledge_dir = os.path.join(self.log_path, 'world_knowledge')
        if not os.path.exists(knowledge_dir):
            os.makedirs(knowledge_dir)
            logger.info(f"Created world_knowledge directory: {knowledge_dir}")
        return os.path.join(knowledge_dir, f'{self.eval_set}_world_knowledge.json')

    def _load_knowledge(self):
        knowledge_path = self._get_knowledge_path()
        if knowledge_path and os.path.exists(knowledge_path):
            try:
                with open(knowledge_path, 'r', encoding='utf-8') as f:
                    loaded_data = json.load(f)


                self.knowledge_entries = self._normalize_knowledge_format(loaded_data)

                total_knowledge = sum(len(entry.get('knowledge', [])) for entry in self.knowledge_entries)
                logger.info(f"Loaded {len(self.knowledge_entries)} knowledge entries ({total_knowledge} total knowledge items) from {knowledge_path}")
            except Exception as e:
                logger.warning(f"Failed to load world knowledge: {e}")
                self.knowledge_entries = []
        else:
            self.knowledge_entries = []
            if knowledge_path:
                logger.info(f"No existing world knowledge at {knowledge_path}")

    def _load_all_knowledge(self):
        if not self.log_path or not self.all_eval_sets:
            self.all_knowledge_entries = self.knowledge_entries.copy()
            return

        self.all_knowledge_entries = []
        knowledge_dir = os.path.join(self.log_path, 'world_knowledge')

        for eval_set_name in self.all_eval_sets:
            knowledge_file = os.path.join(knowledge_dir, f'{eval_set_name}_world_knowledge.json')
            if os.path.exists(knowledge_file):
                try:
                    with open(knowledge_file, 'r', encoding='utf-8') as f:
                        loaded_data = json.load(f)

                        normalized = self._normalize_knowledge_format(loaded_data)
                        self.all_knowledge_entries.extend(normalized)
                    logger.debug(f"Loaded {len(normalized)} knowledge entries from {eval_set_name}")
                except Exception as e:
                    logger.warning(f"Failed to load knowledge from {eval_set_name}: {e}")

        total_items = sum(len(entry.get('knowledge', [])) for entry in self.all_knowledge_entries)
        logger.info(f"Cross-set retrieval: Loaded {len(self.all_knowledge_entries)} entries ({total_items} total items) from {len(self.all_eval_sets)} eval_sets")

    def _load_transfer_knowledge(self):
        if not self.experience_transfer_path:
            logger.warning("WorldKnowledgeManager: experience_transfer_path is None")
            return


        transfer_knowledge_path = os.path.join(
            self.experience_transfer_path,
            self.eval_set,
            'world_knowledge',
            f'{self.eval_set}_world_knowledge.json'
        )

        if os.path.exists(transfer_knowledge_path):
            try:
                with open(transfer_knowledge_path, 'r', encoding='utf-8') as f:
                    loaded_data = json.load(f)
                self.knowledge_entries = self._normalize_knowledge_format(loaded_data)
                total_knowledge = sum(len(entry.get('knowledge', [])) for entry in self.knowledge_entries)
                logger.info(f"[TRANSFER] Loaded {len(self.knowledge_entries)} knowledge entries ({total_knowledge} total items) from {transfer_knowledge_path}")
            except Exception as e:
                logger.warning(f"Failed to load transfer knowledge: {e}")
                self.knowledge_entries = []
        else:
            logger.warning(f"Transfer knowledge file not found: {transfer_knowledge_path}")
            self.knowledge_entries = []


        if self.cross_eval_set_retrieval:
            self._load_all_transfer_knowledge()

    def _load_all_transfer_knowledge(self):
        if not self.experience_transfer_path or not self.all_eval_sets:
            self.all_knowledge_entries = self.knowledge_entries.copy()
            return

        self.all_knowledge_entries = []

        for eval_set_name in self.all_eval_sets:
            transfer_knowledge_path = os.path.join(
                self.experience_transfer_path,
                eval_set_name,
                'world_knowledge',
                f'{eval_set_name}_world_knowledge.json'
            )

            if os.path.exists(transfer_knowledge_path):
                try:
                    with open(transfer_knowledge_path, 'r', encoding='utf-8') as f:
                        loaded_data = json.load(f)
                        normalized = self._normalize_knowledge_format(loaded_data)
                        self.all_knowledge_entries.extend(normalized)
                    logger.debug(f"[TRANSFER] Loaded {len(normalized)} knowledge entries from {eval_set_name}")
                except Exception as e:
                    logger.warning(f"Failed to load transfer knowledge from {eval_set_name}: {e}")

        total_items = sum(len(entry.get('knowledge', [])) for entry in self.all_knowledge_entries)
        logger.info(f"[TRANSFER] Cross-set retrieval: Loaded {len(self.all_knowledge_entries)} entries ({total_items} total items) from {len(self.all_eval_sets)} eval_sets")

    def set_experience_transfer(self, enabled: bool, path: str = None):
        self.enable_experience_transfer = enabled
        self.experience_transfer_path = path

        if enabled and path:
            self._load_transfer_knowledge()
            logger.info(f"WorldKnowledgeManager: Experience transfer enabled from {path}")
        else:

            self._load_knowledge()
            if self.cross_eval_set_retrieval:
                self._load_all_knowledge()
            logger.info(f"WorldKnowledgeManager: Experience transfer disabled, using local knowledge")

    def reload_knowledge_for_eval_set(self):
        if self.enable_experience_transfer and self.experience_transfer_path:
            self._load_transfer_knowledge()
        else:
            self._load_knowledge()
            if self.cross_eval_set_retrieval:
                self._load_all_knowledge()

    def _normalize_knowledge_format(self, data: List) -> List[Dict]:
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
        knowledge_path = self._get_knowledge_path()
        if not knowledge_path:
            logger.warning("WorldKnowledgeManager: Cannot save - log_path is None")
            return

        try:

            for entry in self.knowledge_entries:
                knowledge_list = entry.get('knowledge', [])

                unique_knowledge = list(dict.fromkeys(knowledge_list))
                original_count = len(knowledge_list)
                entry['knowledge'] = unique_knowledge

                if original_count != len(unique_knowledge):
                    logger.debug(f"Removed {original_count - len(unique_knowledge)} duplicate knowledge items for instruction: {entry.get('instruction', 'Unknown')[:50]}...")


            os.makedirs(os.path.dirname(knowledge_path), exist_ok=True)
            with open(knowledge_path, 'w', encoding='utf-8') as f:
                json.dump(self.knowledge_entries, f, ensure_ascii=False, indent=2)

            total_knowledge = sum(len(entry.get('knowledge', [])) for entry in self.knowledge_entries)
            logger.info(f"Saved {len(self.knowledge_entries)} knowledge entries ({total_knowledge} total items) to {knowledge_path}")
        except Exception as e:
            logger.error(f"Failed to save world knowledge: {e}")

    def reset_episode(self):
        self.current_episode_knowledge = []
        self.current_instruction = ""
        logger.debug("WorldKnowledgeManager: episode reset")

    def _find_or_create_entry(self, instruction: str) -> Dict:
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

        logger.info(f"Added {len(entries)} world knowledge entries for instruction: {instr[:50]}...")


        self._save_knowledge()

    def save_episode_knowledge(self, episode_num: int):
        if not self.log_path:
            logger.warning("WorldKnowledgeManager: Cannot save episode - log_path is None")
            return


        episode_knowledge_dir = os.path.join(self.log_path, 'world_knowledge', 'episodes')
        os.makedirs(episode_knowledge_dir, exist_ok=True)

        episode_file = os.path.join(episode_knowledge_dir, f'episode_{episode_num}_knowledge.json')


        episode_data = {
            'instruction': self.current_instruction or "Unknown instruction",
            'eval_set': self.eval_set,
            'knowledge': self.current_episode_knowledge.copy()
        }

        with open(episode_file, 'w', encoding='utf-8') as f:
            json.dump(episode_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved {len(self.current_episode_knowledge)} knowledge entries for episode {episode_num}")


        self._save_knowledge()

    def _compute_text_similarity(self, text1: str, text2: str) -> float:
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

    def _get_prior_instructions(self, current_instruction: str) -> set:
        if not self.enable_experience_transfer or not self.experience_transfer_path:
            return set()

        results_dir = os.path.join(self.experience_transfer_path, self.eval_set, 'results')

        if not os.path.exists(results_dir):
            logger.warning(f"Results directory not found: {results_dir}")
            return set()


        episode_files = []
        try:
            for filename in os.listdir(results_dir):
                if filename.startswith('episode_') and filename.endswith('_final_res.json'):

                    parts = filename.replace('episode_', '').replace('_final_res.json', '')
                    try:
                        episode_num = int(parts)
                        episode_files.append((episode_num, filename))
                    except ValueError:
                        continue
        except Exception as e:
            logger.warning(f"Failed to list results directory: {e}")
            return set()


        episode_files.sort(key=lambda x: x[0])


        current_instruction_normalized = current_instruction.strip().lower()
        instructions_with_order = []
        current_episode_num = None

        for episode_num, filename in episode_files:
            filepath = os.path.join(results_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    instr = data.get('instruction', '')
                    instructions_with_order.append((episode_num, instr))


                    if instr.strip().lower() == current_instruction_normalized:
                        current_episode_num = episode_num
            except Exception as e:
                logger.debug(f"Failed to read episode file {filename}: {e}")
                continue

        if current_episode_num is None:

            logger.debug(f"Current instruction not found in results files: {current_instruction[:50]}...")
            return set()


        prior_instructions = set()
        for episode_num, instr in instructions_with_order:
            if episode_num < current_episode_num:
                prior_instructions.add(instr.strip().lower())

        logger.debug(f"Found {len(prior_instructions)} prior instructions before episode {current_episode_num}")
        return prior_instructions


    def retrieve_knowledge(self, instruction: str, filter_by_temporal_order: bool = True) -> List[Dict]:

        if self.cross_eval_set_retrieval and self.all_knowledge_entries:
            knowledge_to_search = self.all_knowledge_entries
            logger.debug(f"Cross-set retrieval: searching from {len(knowledge_to_search)} knowledge entries across all eval_sets")
        else:
            knowledge_to_search = self.knowledge_entries
            logger.debug(f"Single-set retrieval: searching from {len(knowledge_to_search)} knowledge entries in {self.eval_set}")

        if not knowledge_to_search:
            return []

        if not knowledge_to_search:
            return []


        if self.enable_experience_transfer and filter_by_temporal_order:
            prior_instructions = self._get_prior_instructions(instruction)
            if prior_instructions:
                before_filter_count = len(knowledge_to_search)
                knowledge_to_search = [
                    entry for entry in knowledge_to_search
                    if entry.get('instruction', '').strip().lower() in prior_instructions
                ]
                logger.debug(f"Temporal filtering: {before_filter_count} -> {len(knowledge_to_search)} knowledge entries")

        if not knowledge_to_search:
            return []


        similarities = []
        for entry in knowledge_to_search:
            entry_instruction = entry.get('instruction', '')
            knowledge_text = " ".join(entry.get('knowledge', []) or [])
            sim = self._compute_text_similarity(instruction, entry_instruction)
            sim += 0.35 * _keyword_overlap_score(instruction, entry_instruction)
            sim += 0.65 * _keyword_overlap_score(instruction, knowledge_text)
            similarities.append((entry, sim))


        similarities.sort(key=lambda x: x[1], reverse=True)


        expanded_items = []
        for entry, entry_score in similarities:
            knowledge_list = entry.get('knowledge', [])
            for knowledge in knowledge_list:
                item_score = entry_score + 0.80 * _keyword_overlap_score(instruction, knowledge)
                if str(knowledge).lower().startswith("location knowledge") and _keyword_overlap_score(instruction, knowledge) > 0:
                    item_score += 0.35
                expanded_items.append((item_score, entry, knowledge))

        expanded_items.sort(key=lambda x: x[0], reverse=True)
        result = []
        for _, entry, knowledge in expanded_items[:self.top_k]:
            result.append({
                'instruction': entry.get('instruction', 'Unknown'),
                'knowledge': knowledge
            })

        return result

    def format_knowledge_for_prompt(self, knowledge_list: List[Dict]) -> str:
        if not knowledge_list:
            return ""

        formatted = []
        for i, entry in enumerate(knowledge_list):
            formatted.append(WORLD_KNOWLEDGE_ENTRY_FORMAT.format(
                index=i+1,
                instruction=entry.get('instruction', 'Unknown'),
                knowledge=entry.get('knowledge', 'No knowledge available')
            ))

        return WORLD_KNOWLEDGE_RETRIEVAL_CONTEXT.format(knowledge="\n".join(formatted))

    def get_all_knowledge_flat(self) -> List[str]:
        all_knowledge = []
        for entry in self.knowledge_entries:
            knowledge_list = entry.get('knowledge', [])
            all_knowledge.extend(knowledge_list)
        return all_knowledge



def create_success_experience_manager(
    log_path: str = None,
    eval_set: str = "base",
    top_k: int = 3,
    extractor_model: str = None,
    model_type: str = 'remote',
    cross_eval_set_retrieval: bool = True,
    all_eval_sets: List[str] = None,
    enable_experience_transfer: bool = False,
    experience_transfer_path: str = None
) -> SuccessExperienceManager:
    return SuccessExperienceManager(
        log_path=log_path,
        eval_set=eval_set,
        top_k=top_k,
        extractor_model=extractor_model,
        model_type=model_type,
        cross_eval_set_retrieval=cross_eval_set_retrieval,
        all_eval_sets=all_eval_sets,
        enable_experience_transfer=enable_experience_transfer,
        experience_transfer_path=experience_transfer_path
    )


def create_world_knowledge_manager(
    log_path: str = None,
    eval_set: str = "base",
    top_k: int = 3,
    cross_eval_set_retrieval: bool = True,
    all_eval_sets: List[str] = None,
    enable_experience_transfer: bool = False,
    experience_transfer_path: str = None
) -> WorldKnowledgeManager:
    return WorldKnowledgeManager(
        log_path=log_path,
        eval_set=eval_set,
        top_k=top_k,
        cross_eval_set_retrieval=cross_eval_set_retrieval,
        all_eval_sets=all_eval_sets,
        enable_experience_transfer=enable_experience_transfer,
        experience_transfer_path=experience_transfer_path
    )
