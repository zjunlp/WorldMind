
import json
import re
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from difflib import SequenceMatcher

from embodiedbench.planner.remote_model import RemoteModel
from embodiedbench.planner.planner_utils import fix_json
from embodiedbench.main import logger






MULTI_SIMULATION_LLM_SELECTION_PROMPT = '''You are an expert evaluator for robot task planning. Given a task instruction and multiple candidate plans, you need to evaluate and select the best one.

## Task Instruction
{instruction}

## Available Actions
{available_actions}

## Candidate Plans
{candidate_plans}

## Evaluation Criteria
1. **Correctness**: Does the plan follow valid action rules and sequences?
2. **Efficiency**: Is the plan concise without unnecessary steps?
3. **Completeness**: Does the plan cover all necessary steps to complete the task?
4. **Feasibility**: Are all actions in the plan valid given the current state?

## Current State Description (from agent's visual observation)
{state_description}

## Action History
{action_history}

Please evaluate each candidate plan based on the criteria above and select the best one.

Output format:
{{"evaluation": "Your reasoning about each plan...", "best_plan_index": <index of best plan, 0-indexed>}}

!!! Please output only the JSON without any markdown formatting or code blocks.
'''


class MultiSimulationManager:

    def __init__(
        self,
        enable: bool = True,
        n_candidates: int = 3,
        use_llm_selection: bool = False,
        llm_selector_model: str = None,
        model_type: str = 'remote'
    ):
        self.enable = enable
        self.n_candidates = n_candidates
        self.use_llm_selection = use_llm_selection
        self.model_type = model_type


        self.llm_selector = None
        if use_llm_selection and llm_selector_model:
            self.llm_selector = RemoteModel(
                llm_selector_model,
                model_type=model_type,
                language_only=True
            )


        self.stats = {
            "total_selections": 0,
            "similarity_selections": 0,
            "llm_selections": 0,
            "single_valid_candidate": 0,
            "no_valid_candidates": 0
        }

        logger.info(f"MultiSimulationManager initialized: enable={enable}, n_candidates={n_candidates}, use_llm_selection={use_llm_selection}")

    def set_llm_selector(self, model_name: str, model_type: str = None):
        self.llm_selector = RemoteModel(
            model_name,
            model_type=model_type or self.model_type,
            language_only=True
        )
        logger.info(f"LLM selector set to {model_name}")

    def reset_stats(self):
        self.stats = {
            "total_selections": 0,
            "similarity_selections": 0,
            "llm_selections": 0,
            "single_valid_candidate": 0,
            "no_valid_candidates": 0
        }

    def get_stats(self) -> Dict:
        return self.stats.copy()

    def _action_to_str(self, action, actions_list: List[str] = None) -> str:
        if isinstance(action, list):
            if actions_list:
                names = []
                for a in action:
                    if isinstance(a, int) and 0 <= a < len(actions_list):
                        names.append(actions_list[a])
                    else:
                        names.append(str(a))
                return " -> ".join(names)
            return " ".join([str(a) for a in action])
        elif isinstance(action, int) and actions_list and 0 <= action < len(actions_list):
            return actions_list[action]
        return str(action)

    def _compute_similarity(self, action1, action2, actions_list: List[str] = None) -> float:
        str1 = self._action_to_str(action1, actions_list)
        str2 = self._action_to_str(action2, actions_list)
        return SequenceMatcher(None, str1, str2).ratio()

    def _select_by_similarity(self, candidates: List[Dict], actions_list: List[str] = None) -> int:
        if len(candidates) <= 1:
            return 0


        similarity_scores = []
        for i, cand_i in enumerate(candidates):
            total_sim = 0
            for j, cand_j in enumerate(candidates):
                if i != j:
                    total_sim += self._compute_similarity(
                        cand_i['action'], cand_j['action'], actions_list
                    )
            avg_sim = total_sim / (len(candidates) - 1)
            similarity_scores.append((i, avg_sim))


        max_sim = max(s for _, s in similarity_scores)
        max_sim_candidates = [i for i, sim in similarity_scores if sim == max_sim]


        selected_index = np.random.choice(max_sim_candidates)

        logger.info(f"Multi-simulation (similarity): selected candidate {selected_index} "
                   f"(similarity: {similarity_scores[selected_index][1]:.3f})")

        self.stats["similarity_selections"] += 1
        return selected_index

    def _select_by_llm(
        self,
        candidates: List[Dict],
        instruction: str,
        available_actions: str,
        action_history: str,
        state_description: str,
        actions_list: List[str] = None
    ) -> int:
        if not self.llm_selector:
            logger.warning("LLM selector not initialized, falling back to similarity selection")
            return self._select_by_similarity(candidates, actions_list)


        candidate_descriptions = []
        for i, cand in enumerate(candidates):
            action_str = self._action_to_str(cand['action'], actions_list)
            reasoning_preview = cand.get('reasoning', '')[:500]
            candidate_descriptions.append(
                f"Plan {i}: {action_str}\nReasoning: {reasoning_preview}"
            )


        eval_prompt = MULTI_SIMULATION_LLM_SELECTION_PROMPT.format(
            instruction=instruction,
            available_actions=available_actions,
            candidate_plans="\n\n".join(candidate_descriptions),
            state_description=state_description or "Current observation from robot's camera",
            action_history=action_history or "No previous actions"
        )

        messages = [{"role": "user", "content": [{"type": "text", "text": eval_prompt}]}]

        try:
            response = self.llm_selector.respond(messages)
            response = fix_json(response)


            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                best_index = result.get('best_plan_index', 0)
                if 0 <= best_index < len(candidates):
                    logger.info(f"Multi-simulation (LLM): selected candidate {best_index}")
                    self.stats["llm_selections"] += 1
                    return best_index
        except Exception as e:
            logger.warning(f"LLM selection failed: {e}, falling back to similarity selection")


        return self._select_by_similarity(candidates, actions_list)

    def select_best_candidate(
        self,
        candidates: List[Dict],
        instruction: str = "",
        available_actions: str = "",
        action_history: str = "",
        state_description: str = "",
        actions_list: List[str] = None
    ) -> Tuple[int, Dict]:
        self.stats["total_selections"] += 1


        valid_candidates = []
        valid_indices = []
        for i, cand in enumerate(candidates):
            action = cand.get('action')
            if action not in [-1, -2]:
                valid_candidates.append(cand)
                valid_indices.append(i)

        if not valid_candidates:
            self.stats["no_valid_candidates"] += 1
            logger.warning("No valid candidates, returning first candidate")
            return (0, candidates[0]) if candidates else (0, {'action': -2, 'reasoning': ''})

        if len(valid_candidates) == 1:
            self.stats["single_valid_candidate"] += 1
            return (valid_indices[0], valid_candidates[0])


        if self.use_llm_selection:
            selected_idx = self._select_by_llm(
                valid_candidates, instruction, available_actions,
                action_history, state_description, actions_list
            )
        else:
            selected_idx = self._select_by_similarity(valid_candidates, actions_list)


        original_idx = valid_indices[selected_idx]
        return (original_idx, valid_candidates[selected_idx])


def create_multi_simulation_manager(
    enable: bool = True,
    n_candidates: int = 3,
    use_llm_selection: bool = False,
    llm_selector_model: str = None,
    model_type: str = 'remote'
) -> MultiSimulationManager:
    return MultiSimulationManager(
        enable=enable,
        n_candidates=n_candidates,
        use_llm_selection=use_llm_selection,
        llm_selector_model=llm_selector_model,
        model_type=model_type
    )
