
import os
import json
import re
from typing import Dict, List, Optional, Tuple, Any

from embodiedbench.planner.vlm_planner import VLMPlanner
from embodiedbench.worldmind.habitat.discriminator import WorldMindDiscriminator
from embodiedbench.worldmind.habitat.reflector import WorldMindReflector
from embodiedbench.worldmind.habitat.state_summarizer import WorldMindStateSummarizer, LocationKnowledgeExtractor
from embodiedbench.worldmind.habitat.knowledge_manager import (
    SuccessExperienceManager,
    WorldKnowledgeManager,
    create_success_experience_manager,
    create_world_knowledge_manager
)
from embodiedbench.worldmind.habitat.color_print import (
    print_agent_input,
    print_agent,
    print_summarizer_input,
    print_summarizer_output,
    print_discriminator_input,
    print_discriminator_output,
    print_reflector_input,
    print_reflector_output,
    print_trajectory_entry,
    print_separator
)
from embodiedbench.worldmind.habitat.prompts import (
    WorldMind_TEMPLATE,
    WorldMind_TEMPLATE_LANG,
    get_worldmind_examples,
    format_worldmind_examples,
    fix_json_worldmind,
    parse_json_worldmind
)
from embodiedbench.worldmind.habitat.multi_simulation import MultiSimulationManager
from embodiedbench.main import logger



class ExperienceTrajectory:

    def __init__(self):
        self.trajectory: List[Dict] = []

    def add_entry(
        self,
        step: int,
        action_id: int,
        action_description: str,
        observation_before: str,
        predicted_state: str,
        env_feedback: str = "",
        agent_input_prompt: Optional[str] = None,
        agent_output: Optional[str] = None,

        discriminator_called: bool = False,
        discrimination_result: Optional[Dict] = None,
        has_prediction_error: bool = False,
        reflector_called: bool = False,
        reflection_result: Optional[Dict] = None,
        experience_entries: Optional[List[str]] = None
    ):
        entry = {
            "step": step,
            "action_id": action_id,
            "action_description": action_description,
            "observation_before": observation_before,
            "predicted_state": predicted_state,
            "env_feedback": env_feedback,
            "agent_input_prompt": agent_input_prompt,
            "agent_output": agent_output,

            "worldmind_info": {
                "discriminator_called": discriminator_called,
                "discrimination_result": discrimination_result,
                "has_prediction_error": has_prediction_error,
                "reflector_called": reflector_called,
                "reflection_result": reflection_result,
                "experience_entries": experience_entries or []
            }
        }
        self.trajectory.append(entry)

    def update_last_entry(
        self,
        env_feedback: str = None,
        agent_input_prompt: str = None,
        agent_output: str = None,
        discriminator_called: bool = None,
        discrimination_result: Dict = None,
        has_prediction_error: bool = None,
        reflector_called: bool = None,
        reflection_result: Dict = None,
        experience_entries: List[str] = None
    ):
        if not self.trajectory:
            return

        if env_feedback is not None:
            self.trajectory[-1]["env_feedback"] = env_feedback
        if agent_input_prompt is not None:
            self.trajectory[-1]["agent_input_prompt"] = agent_input_prompt
        if agent_output is not None:
            self.trajectory[-1]["agent_output"] = agent_output


        if "worldmind_info" not in self.trajectory[-1]:
            self.trajectory[-1]["worldmind_info"] = {}

        if discriminator_called is not None:
            self.trajectory[-1]["worldmind_info"]["discriminator_called"] = discriminator_called
        if discrimination_result is not None:
            self.trajectory[-1]["worldmind_info"]["discrimination_result"] = discrimination_result
        if has_prediction_error is not None:
            self.trajectory[-1]["worldmind_info"]["has_prediction_error"] = has_prediction_error
        if reflector_called is not None:
            self.trajectory[-1]["worldmind_info"]["reflector_called"] = reflector_called
        if reflection_result is not None:
            self.trajectory[-1]["worldmind_info"]["reflection_result"] = reflection_result
        if experience_entries is not None:
            self.trajectory[-1]["worldmind_info"]["experience_entries"] = experience_entries

    def get_trajectory_for_save(self) -> List[Dict]:
        return self.trajectory.copy()

    def reset(self):
        self.trajectory = []

    def __len__(self):
        return len(self.trajectory)

class WorldMindPlannerWrapper:

    def __init__(
        self,
        base_planner: VLMPlanner,
        discriminator_model: str = None,
        reflector_model: str = None,
        summarizer_model: str = None,
        enable_discrimination: bool = True,
        use_experience_trajectory: bool = False,
        use_state_summarizer: bool = True,
        detailed_output: bool = False,
        use_separate_summarization: bool = False,
        worldmind_first_action_only: bool = True,

        enable_success_experience: bool = True,
        success_experience_top_k: int = 3,

        enable_world_knowledge: bool = True,
        world_knowledge_top_k: int = 3,

        enable_experience_refine: bool = False,

        enable_multi_simulation: bool = True,
        multi_simulation_n_candidates: int = 3,
        multi_simulation_use_llm_selection: bool = False,

        cross_eval_set_retrieval: bool = True,
        all_eval_sets: List[str] = None,

        enable_location_knowledge: bool = True,
        enable_location_knowledge_accumulation: bool = True,

        enable_experience_transfer: bool = False,
        experience_transfer_path: str = None,
        use_fast_feedback_discrimination: bool = True
    ):
        self.base_planner = base_planner
        self.enable_discrimination = enable_discrimination
        self.use_experience_trajectory = use_experience_trajectory
        self.use_state_summarizer = use_state_summarizer
        self.detailed_output = detailed_output
        self.worldmind_first_action_only = worldmind_first_action_only
        self.use_fast_feedback_discrimination = use_fast_feedback_discrimination


        self.enable_success_experience = enable_success_experience
        self.success_experience_top_k = success_experience_top_k
        self.enable_world_knowledge = enable_world_knowledge
        self.world_knowledge_top_k = world_knowledge_top_k
        self.enable_experience_refine = enable_experience_refine


        self.cross_eval_set_retrieval = cross_eval_set_retrieval
        self.all_eval_sets = all_eval_sets or []


        self.current_human_instruction: Optional[str] = None


        self.enable_location_knowledge = enable_location_knowledge
        self.enable_location_knowledge_accumulation = enable_location_knowledge_accumulation


        self.enable_experience_transfer = enable_experience_transfer
        self.experience_transfer_path = experience_transfer_path


        default_model = None
        if hasattr(base_planner, 'model_name'):
            default_model = base_planner.model_name
        elif hasattr(base_planner, 'model') and hasattr(base_planner.model, 'model_name'):
            default_model = base_planner.model.model_name


        actual_discriminator_model = discriminator_model or default_model
        actual_reflector_model = reflector_model or discriminator_model or default_model
        actual_summarizer_model = summarizer_model or discriminator_model or default_model

        logger.info(f"WorldMind Models - Discriminator: {actual_discriminator_model}, Reflector: {actual_reflector_model}, Summarizer: {actual_summarizer_model}")


        if enable_discrimination and actual_discriminator_model:
            self.discriminator = WorldMindDiscriminator(model_name=actual_discriminator_model)
        else:
            self.discriminator = None
            if enable_discrimination:
                logger.warning("Discrimination enabled but no model configured - discrimination will be skipped")


        if actual_reflector_model:
            self.reflector = WorldMindReflector(model_name=actual_reflector_model)
        else:
            self.reflector = None
            logger.warning("No reflector model configured - reflection will be skipped")


        if actual_summarizer_model:
            self.state_summarizer = WorldMindStateSummarizer(
                model_name=actual_summarizer_model,
                use_separate_summarization=use_separate_summarization
            )
        else:
            self.state_summarizer = None
            logger.warning("No summarizer model configured - state summarization will be skipped")


        if self.enable_location_knowledge and actual_summarizer_model:
            try:
                self.location_knowledge_extractor = LocationKnowledgeExtractor(
                    model_name=actual_summarizer_model
                )
                logger.info("Location Knowledge Extractor initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Location Knowledge Extractor: {e}")
                self.location_knowledge_extractor = None
        else:
            self.location_knowledge_extractor = None


        self.experience_trajectory = ExperienceTrajectory()


        self.success_experience_manager = SuccessExperienceManager(
            top_k=success_experience_top_k,
            extractor_model=actual_reflector_model if not enable_experience_transfer else None,
            model_type=base_planner.model_type if hasattr(base_planner, 'model_type') else 'remote',
            cross_eval_set_retrieval=cross_eval_set_retrieval,
            all_eval_sets=all_eval_sets or [],
            enable_experience_transfer=enable_experience_transfer,
            experience_transfer_path=experience_transfer_path
        )


        self.world_knowledge_manager = WorldKnowledgeManager(
            top_k=world_knowledge_top_k,
            cross_eval_set_retrieval=cross_eval_set_retrieval,
            all_eval_sets=all_eval_sets or [],
            enable_experience_transfer=enable_experience_transfer,
            experience_transfer_path=experience_transfer_path
        )



        self.experience_refiner = None
        if enable_experience_refine:
            from embodiedbench.worldmind.habitat.experience_refiner import ExperienceRefiner

            refiner_model = base_planner.model_name if hasattr(base_planner, 'model_name') else (discriminator_model or reflector_model)
            self.experience_refiner = ExperienceRefiner(
                model_name=refiner_model,
                model_type=base_planner.model_type if hasattr(base_planner, 'model_type') else 'remote'
            )
            logger.info(f"Experience refiner enabled with model: {refiner_model}")


        if base_planner.language_only:
            self.worldmind_template = WorldMind_TEMPLATE_LANG
        else:
            self.worldmind_template = WorldMind_TEMPLATE


        self.current_predicted_state: Optional[str] = None
        self.current_action_id: Optional[int] = None
        self.current_action_description: Optional[str] = None


        self.current_predicted_states: List[str] = []


        self.discrimination_history: List[Dict] = []


        self.reflection_history: List[Dict] = []


        self.current_state_summary: Optional[str] = None


        self.current_obs_path: Optional[str] = None


        self.current_worldmind_feedback: Dict = {}


        self.current_agent_input_prompt: Optional[str] = None
        self.current_agent_output: Optional[str] = None


        self.current_visual_state_description: Optional[str] = None


        self.current_eval_set: str = "base"




        self._cached_world_knowledge: Optional[List[Dict]] = None


        self.worldmind_stats = {
            "total_predictions": 0,
            "discrimination_matches": 0,
            "discrimination_mismatches": 0,
            "reflections_triggered": 0,
            "success_feedbacks": 0
        }


        self.enable_multi_simulation = enable_multi_simulation
        self.multi_simulation_n_candidates = multi_simulation_n_candidates
        self.multi_simulation_use_llm_selection = multi_simulation_use_llm_selection

        if enable_multi_simulation:
            self.multi_simulation_manager = MultiSimulationManager(
                enable=enable_multi_simulation,
                n_candidates=multi_simulation_n_candidates,
                use_llm_selection=multi_simulation_use_llm_selection,
                llm_selector_model=discriminator_model,
                model_type=base_planner.model_type if hasattr(base_planner, 'model_type') else 'remote'
            )
        else:
            self.multi_simulation_manager = None

        logger.info(f"WorldMind Planner Wrapper initialized (success_experience={enable_success_experience}, world_knowledge={enable_world_knowledge}, experience_refine={enable_experience_refine}, experience_transfer={enable_experience_transfer}, fast_feedback_discrimination={use_fast_feedback_discrimination})")

        if enable_experience_transfer:
            logger.info(f"[EXPERIENCE TRANSFER MODE] Loading experiences from: {experience_transfer_path}")
            logger.info("[EXPERIENCE TRANSFER MODE] WorldMind cycle (summarizer, discriminator, reflector) is DISABLED")

    def set_log_path(self, log_path: str):
        self.success_experience_manager.set_log_path(log_path)
        self.world_knowledge_manager.set_log_path(log_path)

    def set_eval_set(self, eval_set: str):
        self.current_eval_set = eval_set
        self.success_experience_manager.set_eval_set(eval_set)
        self.world_knowledge_manager.set_eval_set(eval_set)

    def set_all_eval_sets(self, all_eval_sets: List[str]):
        self.all_eval_sets = all_eval_sets
        self.success_experience_manager.set_all_eval_sets(all_eval_sets)
        self.world_knowledge_manager.set_all_eval_sets(all_eval_sets)
        logger.info(f"All eval sets for cross-set retrieval: {all_eval_sets}")

    def set_cross_eval_set_retrieval(self, enabled: bool):
        self.cross_eval_set_retrieval = enabled
        self.success_experience_manager.set_cross_eval_set_retrieval(enabled)
        self.world_knowledge_manager.set_cross_eval_set_retrieval(enabled)
        logger.info(f"Cross eval set retrieval: {enabled}")



    def set_human_instruction(self, instruction: str):
        self.current_human_instruction = instruction

        self.success_experience_manager.set_instruction(instruction)

        self.world_knowledge_manager.set_instruction(instruction)

    def update_trajectory_info(self, info: Dict):

        self.success_experience_manager.update_trajectory(info)

        logger.debug(f"Trajectory updated with action_id={info.get('action_id')}")

    def extract_success_experience(self, instruction: str, task_success: bool):
        if not self.enable_success_experience:
            logger.debug("Success experience extraction disabled")
            return

        logger.info(f"Attempting to extract success experience: task_success={task_success}")

        result = self.success_experience_manager.extract_experience(
            instruction=instruction,
            task_success=task_success
        )

        if result:
            logger.info(f"Successfully extracted experience: {result.get('success_experience', '')[:100]}...")
            self.worldmind_stats["success_feedbacks"] += 1
        else:
            logger.debug("No experience extracted (task may have failed or extraction failed)")

    def save_world_knowledge(self, episode_num: int):
        if not self.enable_world_knowledge:
            return

        self.world_knowledge_manager.save_episode_knowledge(episode_num)





    def _get_last_nav_action_from_history(self) -> str:
        if not hasattr(self.base_planner, 'episode_act_feedback'):
            return ""


        for feedback in reversed(self.base_planner.episode_act_feedback):
            if len(feedback) >= 1:
                action_id = feedback[0]
                if 0 <= action_id < len(self.base_planner.actions):
                    action_desc = self.base_planner.actions[action_id]

                    if 'navigate' in action_desc.lower() or 'nav' in action_desc.lower():
                        return action_desc

        return ""

    def _format_feedback_search_hints(self) -> str:
        if not hasattr(self.base_planner, 'episode_act_feedback'):
            return ""

        visited_locations = []
        failed_pick_locations = []
        last_successful_nav = ""

        for feedback in self.base_planner.episode_act_feedback:
            if len(feedback) < 2:
                continue

            action_id, env_feedback = feedback[0], str(feedback[1])
            if not isinstance(action_id, int) or not (0 <= action_id < len(self.base_planner.actions)):
                continue

            action_desc = str(self.base_planner.actions[action_id])
            action_lower = action_desc.lower()
            feedback_lower = env_feedback.lower()

            if action_lower.startswith("navigate to ") and "executed successfully" in feedback_lower:
                last_successful_nav = action_desc[len("navigate to "):]
                if last_successful_nav not in visited_locations:
                    visited_locations.append(last_successful_nav)

            if action_lower.startswith("pick up the ") and "not near the robot" in feedback_lower:
                target = action_desc[len("pick up the "):]
                failed_pick_locations.append((target, last_successful_nav or "the current location"))

        if not visited_locations and not failed_pick_locations:
            return ""

        lines = [
            "## Feedback-Derived Search Notes",
            "- These notes are inferred only from prior action ids and environment feedback.",
            "- If a pickup failed because the object was not near the robot after navigating to a location, do not repeat the same pickup at that same location without first navigating to a different plausible source.",
            "- Avoid consecutive duplicate navigation actions; one successful navigation to a location is enough before trying the next required interaction."
        ]

        if visited_locations:
            lines.append("- Visited locations with successful navigation: " + "; ".join(visited_locations[-8:]))

        if failed_pick_locations:
            formatted_failures = [
                f"{target} was not reachable at {location}"
                for target, location in failed_pick_locations[-8:]
            ]
            lines.append("- Failed pickup evidence: " + "; ".join(formatted_failures))

        return "\n".join(lines)

    def extract_location_knowledge_after_action(
        self,
        nav_action: str,
        after_image_path: str
    ) -> list:
        if not self.enable_location_knowledge or not self.location_knowledge_extractor:
            return []

        if not nav_action:
            logger.debug("Location Knowledge: No nav_action provided")
            return []

        if not after_image_path:
            logger.debug("Location Knowledge: No after_image_path provided")
            return []


        if not self.location_knowledge_extractor.is_navigation_action(nav_action):
            logger.debug(f"Location Knowledge: '{nav_action}' is not a navigation action")
            return []


        knowledge_list = self.location_knowledge_extractor.try_extract_on_first_visit(
            nav_action=nav_action,
            after_image_path=after_image_path
        )

        if not knowledge_list:

            location = self.location_knowledge_extractor.extract_location_from_action(nav_action)
            if location in self.location_knowledge_extractor.visited_locations:
                logger.debug(f"Location Knowledge: '{location}' already visited, skipping")
            return []


        knowledge_list = [f"Location Knowledge: {k}" for k in knowledge_list]


        if knowledge_list:
            self.location_knowledge_extractor.add_to_accumulated_knowledge(knowledge_list)
            logger.info(f"Location Knowledge: Extracted {len(knowledge_list)} entries for first visit: {knowledge_list[:2]}...")

        return knowledge_list

    def extract_location_knowledge_after_planning(self) -> list:


        logger.debug("Location Knowledge: extract_location_knowledge_after_planning is deprecated, use extract_location_knowledge_after_action instead")
        return []

    def get_accumulated_location_knowledge(self) -> str:
        if not hasattr(self, 'location_knowledge_extractor') or not self.location_knowledge_extractor:
            return ""

        return self.location_knowledge_extractor.get_accumulated_knowledge_str()

    def get_location_knowledge_for_world_knowledge(self) -> list:
        if not hasattr(self, 'location_knowledge_extractor') or not self.location_knowledge_extractor:
            return []

        return self.location_knowledge_extractor.get_accumulated_knowledge_list()

    def add_location_knowledge_to_world_knowledge(self):
        if not self.enable_location_knowledge or not self.enable_world_knowledge:
            return

        location_knowledge_list = self.get_location_knowledge_for_world_knowledge()
        if location_knowledge_list:

            instruction = self.current_human_instruction or "Unknown instruction"

            self.world_knowledge_manager.add_entries(location_knowledge_list, instruction)
            logger.info(f"Added {len(location_knowledge_list)} location knowledge entries to world knowledge")

    def reset(self):
        self.base_planner.reset()
        self.current_predicted_state = None
        self.current_action_id = None
        self.current_action_description = None
        self.current_predicted_states = []
        self.discrimination_history = []
        self.reflection_history = []
        self.current_state_summary = None
        self.current_obs_path = None
        self.current_worldmind_feedback = {}
        self.current_agent_input_prompt = None
        self.current_agent_output = None
        self.current_visual_state_description = None
        self.current_human_instruction = None


        self._cached_world_knowledge = None


        self.experience_trajectory.reset()


        self.success_experience_manager.reset_trajectory()
        self.success_experience_manager.set_actions(self.base_planner.actions)


        self.world_knowledge_manager.reset_episode()


        if self.discriminator:
            self.discriminator.reset_statistics()
        if self.reflector:
            self.reflector.reset_statistics()
        if self.state_summarizer:
            self.state_summarizer.reset_statistics()


        if hasattr(self, 'location_knowledge_extractor') and self.location_knowledge_extractor:
            self.location_knowledge_extractor.reset()


        if self.multi_simulation_manager:
            self.multi_simulation_manager.reset_stats()

    def _extract_predicted_states(self, output_text: str) -> List[str]:
        predicted_states = []
        try:
            json_object = json.loads(output_text)
            executable_plan = json_object.get('executable_plan', [])

            for action in executable_plan:
                state = action.get('predicted_state', '')
                predicted_states.append(state)

        except json.JSONDecodeError:

            pattern = r'"predicted_state"\s*:\s*"([^"]*)"'
            matches = re.findall(pattern, output_text)
            predicted_states = list(matches)

        return predicted_states

    def _extract_visual_state_description(self, output_text: str) -> Optional[str]:
        try:
            json_object = json.loads(output_text)
            return json_object.get('visual_state_description', None)
        except json.JSONDecodeError:

            pattern = r'"visual_state_description"\s*:\s*"([^"]*)"'
            match = re.search(pattern, output_text)
            if match:
                return match.group(1)
        return None

    def _find_instruction_pick_actions(self, user_instruction: str) -> List[int]:
        instruction = (user_instruction or "").lower()
        if not instruction:
            return []

        candidates = []
        for idx, action in enumerate(self.base_planner.actions):
            action_text = str(action).lower()
            if not action_text.startswith("pick up the "):
                continue
            obj_name = action_text[len("pick up the "):].strip()
            if obj_name and re.search(rf"\b{re.escape(obj_name)}s?\b", instruction):
                candidates.append((len(obj_name), idx))

        if not candidates:
            return []

        candidates.sort(reverse=True)
        return [idx for _, idx in candidates]

    def _find_instruction_pick_action(self, user_instruction: str) -> Optional[int]:
        candidates = self._find_instruction_pick_actions(user_instruction)
        return candidates[0] if candidates else None

    def _pick_action_by_object_name(self, object_name: str) -> Optional[int]:
        object_name = self._normalize_location_text(object_name)
        for idx, action in enumerate(self.base_planner.actions):
            action_text = str(action).lower()
            if not action_text.startswith("pick up the "):
                continue
            candidate = self._normalize_location_text(action_text[len("pick up the "):])
            if candidate == object_name:
                return idx
        return None

    def _action_id_by_exact_name(self, action_name: str) -> Optional[int]:
        target = self._normalize_location_text(action_name)
        for idx, action in enumerate(self.base_planner.actions):
            if self._normalize_location_text(str(action)) == target:
                return idx
        return None

    def _infer_common_sense_pick_actions(self, user_instruction: str) -> List[int]:
        instruction = (user_instruction or "").lower()
        if not instruction:
            return []

        rules = [
            (r"\b(red|reddish)\b.*\bgreen top\b|\bgreen top\b.*\b(red|reddish)\b", ["strawberry"]),
            (r"\bflat\b.*\b(red|reddish)\b.*\bround\b|\b(red|reddish)\b.*\bflat\b.*\bround\b", ["plate", "lid"]),
            (r"\bcylindrical\b.*\bhandle\b|\bhandle\b.*\bcylindrical\b", ["mug", "cup"]),
            (r"\byellow\b.*\bcurved\b|\bcurved\b.*\byellow\b", ["banana"]),
            (r"\byellow\b.*\bround\b.*\bfruit\b|\bround\b.*\byellow\b.*\bfruit\b", ["lemon", "orange", "banana", "apple", "pear", "peach"]),
            (r"\byellow\b.*\bround\b|\bround\b.*\byellow\b", ["ball", "lemon", "orange"]),
            (r"\bpurple\b.*\bfruit\b|\bfruit\b.*\bpurple\b", ["plum"]),
            (r"\b(game|play|toy|fun|entertain)\b", ["rubriks cube", "lego", "ball", "toy airplane", "block", "book"]),
            (r"\b(pour|coffee|tea|drink|beverage|liquid)\b", ["mug", "cup", "bowl"]),
            (r"\b(eat|food|snack|fruit|hungry)\b", ["apple", "orange", "banana", "pear", "peach", "strawberry", "plum"]),
            (r"\b(clean|wipe|wash|scrub)\b", ["sponge", "cleanser"]),
            (r"\b(cut|slice|chop)\b", ["knife", "scissors"]),
            (r"\b(cook|stir|serve)\b", ["spatula", "bowl", "plate", "knife"]),
            (r"\b(read|study)\b", ["book"]),
            (r"\b(screw|loose)\b", ["screwdriver", "wrench", "drill", "hammer", "clamp"]),
            (r"\b(fix|repair|build|tool|rectify)\b", ["screwdriver", "wrench", "hammer", "drill", "clamp"]),
            (r"\b(lock|secure)\b", ["padlock"]),
        ]

        candidates = []
        seen = set()
        for pattern, object_names in rules:
            if not re.search(pattern, instruction):
                continue
            for object_name in object_names:
                action_id = self._pick_action_by_object_name(object_name)
                if action_id is not None and action_id not in seen:
                    candidates.append(action_id)
                    seen.add(action_id)

        return candidates

    def _find_candidate_pick_action(self, user_instruction: str) -> Optional[int]:
        exact_action = self._find_instruction_pick_action(user_instruction)
        if exact_action is not None:
            return exact_action

        inferred_actions = self._infer_common_sense_pick_actions(user_instruction)
        return inferred_actions[0] if inferred_actions else None

    def _find_contrastive_target_pick_action(self, user_instruction: str) -> Optional[int]:
        instruction = (user_instruction or "").lower()
        if "otherwise" in instruction:
            return None

        marker_positions = [instruction.rfind(" but "), instruction.rfind("instead")]
        marker_pos = max(marker_positions)
        if marker_pos < 0:
            return None

        contrast_clause = instruction[marker_pos:]
        candidates = []
        for idx, action in enumerate(self.base_planner.actions):
            action_text = str(action).lower()
            if not action_text.startswith("pick up the "):
                continue
            obj_name = action_text[len("pick up the "):].strip()
            if not obj_name:
                continue

            for match in re.finditer(rf"\b{re.escape(obj_name)}s?\b", contrast_clause):
                before = contrast_clause[max(0, match.start() - 80):match.start()]
                score = len(obj_name)

                if re.search(r"\b(instead of|rather than|not)\b[^,.;]*$", before):
                    score -= 150
                if re.search(r"\b(need|find|move|put|place|transfer|transport|relocate|deliver|bring)\b[^,.;]*$", before):
                    score += 100
                if "," in contrast_clause[:match.start()]:
                    score += 20

                candidates.append((score, len(obj_name), idx))

        if not candidates:
            return None

        candidates.sort(reverse=True)
        best_score, _, best_idx = candidates[0]
        return best_idx if best_score > 0 else None

    def _format_common_sense_target_hints(self, user_instruction: str) -> str:
        if self._find_instruction_pick_action(user_instruction) is not None:
            return ""

        inferred_actions = self._infer_common_sense_pick_actions(user_instruction)
        if not inferred_actions:
            return ""

        action_names = [
            f"{action_id}: {self.base_planner.actions[action_id]}"
            for action_id in inferred_actions[:6]
        ]
        return (
            "## Common-Sense Target Hints\n"
            "- The instruction describes the target by use rather than exact object name.\n"
            "- Prefer these available pickup actions over unrelated objects: "
            + "; ".join(action_names)
            + ".\n"
            "- After navigating to a plausible source, try a candidate pickup before placing at the destination."
        )

    def _last_successful_navigation_action_id(self) -> Optional[int]:
        for feedback in reversed(getattr(self.base_planner, "episode_act_feedback", []) or []):
            if len(feedback) < 2:
                continue
            action_id, env_feedback = feedback[0], str(feedback[1]).lower()
            if (
                isinstance(action_id, int)
                and self._is_navigation_action_id(action_id)
                and "executed successfully" in env_feedback
            ):
                return action_id
        return None

    def _failed_pick_actions_at_navigation(self, nav_action_id: int) -> set:
        failed_actions = set()
        current_nav_action = None
        for feedback in getattr(self.base_planner, "episode_act_feedback", []) or []:
            if len(feedback) < 2:
                continue
            action_id, env_feedback = feedback[0], str(feedback[1]).lower()
            if not isinstance(action_id, int):
                continue
            if self._is_navigation_action_id(action_id) and "executed successfully" in env_feedback:
                current_nav_action = action_id
                continue
            if (
                current_nav_action == nav_action_id
                and self._is_pick_action_id(action_id)
                and "not near the robot" in env_feedback
            ):
                failed_actions.add(action_id)
        return failed_actions

    def _completed_pick_actions_from_feedback(self) -> set:
        completed = set()
        held_pick_action = None
        for feedback in getattr(self.base_planner, "episode_act_feedback", []) or []:
            if len(feedback) < 2:
                continue
            action_id, env_feedback = feedback[0], str(feedback[1]).lower()
            if not isinstance(action_id, int):
                continue
            if "executed successfully" not in env_feedback:
                continue
            if self._is_pick_action_id(action_id):
                held_pick_action = action_id
            elif self._is_place_action_id(action_id):
                if held_pick_action is not None:
                    completed.add(held_pick_action)
                held_pick_action = None
        return completed

    def _pick_failure_counts_from_feedback(self) -> Dict[int, int]:
        counts: Dict[int, int] = {}
        for feedback in getattr(self.base_planner, "episode_act_feedback", []) or []:
            if len(feedback) < 2:
                continue
            action_id, env_feedback = feedback[0], str(feedback[1]).lower()
            if (
                isinstance(action_id, int)
                and self._is_pick_action_id(action_id)
                and "not near the robot" in env_feedback
            ):
                counts[action_id] = counts.get(action_id, 0) + 1
        return counts

    def _navigation_visit_counts_from_feedback(self) -> Dict[int, int]:
        counts: Dict[int, int] = {}
        for feedback in getattr(self.base_planner, "episode_act_feedback", []) or []:
            if len(feedback) < 2:
                continue
            action_id, env_feedback = feedback[0], str(feedback[1]).lower()
            if (
                isinstance(action_id, int)
                and self._is_navigation_action_id(action_id)
                and "executed successfully" in env_feedback
            ):
                counts[action_id] = counts.get(action_id, 0) + 1
        return counts

    def _select_exact_pick_for_navigation(
        self,
        nav_action_id: Optional[int],
        user_instruction: str,
        *,
        exclude_completed: bool = True,
        require_source_match: bool = False
    ) -> Optional[int]:
        exact_pick_actions = self._find_instruction_pick_actions(user_instruction)
        if not exact_pick_actions:
            return None

        completed = self._completed_pick_actions_from_feedback()
        active_exact = [
            action_id for action_id in exact_pick_actions
            if not exclude_completed or action_id not in completed
        ]
        if not active_exact:
            active_exact = exact_pick_actions

        failed_here = (
            self._failed_pick_actions_at_navigation(nav_action_id)
            if nav_action_id is not None
            else set()
        )

        scored = []
        for order, pick_action in enumerate(active_exact):
            if pick_action in failed_here:
                continue

            source_score = 0
            if nav_action_id is not None:
                source_candidates = self._source_navigation_candidates_for_pick(
                    pick_action,
                    user_instruction
                )
                if nav_action_id in source_candidates:
                    source_score = 1000 - source_candidates.index(nav_action_id)
                elif require_source_match:
                    continue

            completed_penalty = 100 if pick_action in completed else 0
            failure_penalty = self._pick_failure_counts_from_feedback().get(pick_action, 0) * 80
            scored.append((source_score - completed_penalty - failure_penalty, -order, pick_action))

        if not scored:
            return None
        scored.sort(reverse=True)
        return scored[0][2]

    def _last_action_was_failed_pick(self) -> bool:
        history = getattr(self.base_planner, "episode_act_feedback", []) or []
        if not history:
            return False
        last_feedback = history[-1]
        if len(last_feedback) < 2:
            return False
        action_id, env_feedback = last_feedback[0], str(last_feedback[1]).lower()
        return (
            isinstance(action_id, int)
            and self._is_pick_action_id(action_id)
            and "not near the robot" in env_feedback
        )

    def _repair_multi_target_pick_actions(self, actions: Any, user_instruction: str) -> Any:
        if not isinstance(actions, list) or len(actions) == 0:
            return actions

        exact_pick_actions = self._find_instruction_pick_actions(user_instruction)
        if len(exact_pick_actions) < 2:
            return actions
        exact_set = set(exact_pick_actions)
        completed = self._completed_pick_actions_from_feedback()
        failure_counts = self._pick_failure_counts_from_feedback()

        repaired = []
        changed = False
        current_nav = self._last_successful_navigation_action_id()
        for idx, action_id in enumerate(actions):
            if self._is_navigation_action_id(action_id):
                current_nav = action_id
                repaired.append(action_id)
                continue

            if self._is_pick_action_id(action_id) and action_id not in exact_set:
                replacement = self._select_exact_pick_for_navigation(
                    current_nav,
                    user_instruction,
                    exclude_completed=True,
                    require_source_match=False
                )
                if replacement is not None:
                    logger.info(
                        "WorldMind: replaced non-target pickup action "
                        f"{action_id} with exact multi-target pickup {replacement}"
                    )
                    repaired.append(replacement)
                    changed = True
                    continue

            if self._is_pick_action_id(action_id) and action_id in exact_set and action_id in completed:
                replacement = self._select_exact_pick_for_navigation(
                    current_nav,
                    user_instruction,
                    exclude_completed=True,
                    require_source_match=False
                )
                if replacement is not None and replacement != action_id:
                    logger.info(
                        "WorldMind: replaced already completed multi-target pickup "
                        f"{action_id} with remaining target {replacement}"
                    )
                    repaired.append(replacement)
                    changed = True
                    continue
                fallback_probe = self._next_source_probe_for_exact_targets(user_instruction)
                if fallback_probe is not None:
                    trim_idx = len(repaired)
                    while trim_idx > 0:
                        previous_action = repaired[trim_idx - 1]
                        if self._is_navigation_action_id(previous_action):
                            trim_idx -= 1
                            break
                        if self._is_pick_action_id(previous_action) or self._is_place_action_id(previous_action):
                            break
                        trim_idx -= 1
                    logger.info(
                        "WorldMind: replaced completed multi-target source probe "
                        f"ending in pickup {action_id} with remaining-target probe {fallback_probe}"
                    )
                    return repaired[:trim_idx] + fallback_probe + actions[idx + 1:]

            if (
                self._is_pick_action_id(action_id)
                and action_id in exact_set
                and failure_counts.get(action_id, 0) >= 3
            ):
                replacement = self._select_exact_pick_for_navigation(
                    current_nav,
                    user_instruction,
                    exclude_completed=True,
                    require_source_match=True
                )
                if replacement is not None and replacement != action_id:
                    logger.info(
                        "WorldMind: rotated over-tested exact target pickup "
                        f"{action_id} to less-tested target {replacement}"
                    )
                    repaired.append(replacement)
                    changed = True
                    continue

            repaired.append(action_id)

        return repaired if changed else actions

    def _repair_single_target_pick_actions(self, actions: Any, user_instruction: str) -> Any:
        if not isinstance(actions, list) or len(actions) == 0:
            return actions

        target_pick_actions = self._find_instruction_pick_actions(user_instruction)
        if len(target_pick_actions) == 1:
            target_pick_action = target_pick_actions[0]
        else:
            target_pick_action = self._find_contrastive_target_pick_action(user_instruction)
        if target_pick_action is None:
            return actions
        repaired = []
        changed = False
        for action_id in actions:
            if (
                isinstance(action_id, int)
                and 0 <= action_id < len(self.base_planner.actions)
                and str(self.base_planner.actions[action_id]).lower().startswith("pick up ")
                and action_id != target_pick_action
            ):
                logger.info(
                    f"WorldMind: replaced pickup action id {action_id} "
                    f"with instruction target pickup id {target_pick_action}"
                )
                repaired.append(target_pick_action)
                changed = True
            else:
                repaired.append(action_id)

        return repaired if changed else actions

    def _is_navigation_action_id(self, action_id: Any) -> bool:
        if not isinstance(action_id, int) or not (0 <= action_id < len(self.base_planner.actions)):
            return False
        return str(self.base_planner.actions[action_id]).lower().startswith("navigate to ")

    def _is_pick_action_id(self, action_id: Any) -> bool:
        if not isinstance(action_id, int) or not (0 <= action_id < len(self.base_planner.actions)):
            return False
        return str(self.base_planner.actions[action_id]).lower().startswith("pick up ")

    def _is_place_action_id(self, action_id: Any) -> bool:
        if not isinstance(action_id, int) or not (0 <= action_id < len(self.base_planner.actions)):
            return False
        return str(self.base_planner.actions[action_id]).lower().startswith("place at ")

    def _is_likely_destination_navigation(self, action_id: Any, user_instruction: str) -> bool:
        if not self._is_navigation_action_id(action_id):
            return False
        instruction = (user_instruction or "").lower()
        location = str(self.base_planner.actions[action_id]).lower()[len("navigate to "):].strip()
        location = re.sub(r"\s+in\s+the\s+.*$", "", location)
        if not location:
            return False
        loc_pattern = rf"(the\s+|an?\s+|allocated\s+|indicated\s+)*{re.escape(location)}\b"
        if re.search(
            rf"^\s*(on|in|at)\s+{loc_pattern}\s*,?\s+"
            r"(i\s+need|we\s+need|need|please|put|place|bring|move)",
            instruction,
        ):
            return True
        return re.search(rf"\b(to|into|onto|at)\s+{loc_pattern}", instruction) is not None

    def _last_successful_action_id(self) -> Optional[int]:
        for feedback in reversed(getattr(self.base_planner, "episode_act_feedback", []) or []):
            if len(feedback) < 2:
                continue
            action_id, env_feedback = feedback[0], str(feedback[1]).lower()
            if isinstance(action_id, int) and "executed successfully" in env_feedback:
                return action_id
        return None

    def _current_held_object_from_feedback(self) -> Optional[str]:
        for feedback in reversed(getattr(self.base_planner, "episode_act_feedback", []) or []):
            if len(feedback) < 2:
                continue
            action_id, env_feedback = feedback[0], str(feedback[1]).lower()
            if (
                isinstance(action_id, int)
                and self._is_place_action_id(action_id)
                and "executed successfully" in env_feedback
            ):
                return None
            if "holding nothing" in env_feedback or "not holding" in env_feedback:
                return None
            match = re.search(r"you are holding ([^.]+)", env_feedback)
            if match:
                return match.group(1).strip()
        return None

    def _last_feedback_indicates_empty_hand(self) -> bool:
        for feedback in reversed(getattr(self.base_planner, "episode_act_feedback", []) or []):
            if len(feedback) < 2:
                continue
            env_feedback = str(feedback[1]).lower()
            if "you are holding " in env_feedback and "holding nothing" not in env_feedback:
                return False
            if "holding nothing" in env_feedback or "not holding" in env_feedback:
                return True
        return False

    @staticmethod
    def _normalize_location_text(text: str) -> str:
        text = re.sub(r"\b(the|a|an)\b", " ", (text or "").lower())
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _find_navigation_for_place_action(self, place_action_id: int) -> Optional[int]:
        if not isinstance(place_action_id, int) or not (0 <= place_action_id < len(self.base_planner.actions)):
            return None

        place_text = str(self.base_planner.actions[place_action_id]).lower()
        if not place_text.startswith("place at "):
            return None

        place_location = self._normalize_location_text(place_text[len("place at "):])
        for idx, action in enumerate(self.base_planner.actions):
            action_text = str(action).lower()
            if not action_text.startswith("navigate to "):
                continue
            nav_location = self._normalize_location_text(action_text[len("navigate to "):])
            if place_location == nav_location or place_location in nav_location or nav_location in place_location:
                return idx

        return None

    def _find_navigation_for_location_name(self, location_name: str) -> Optional[int]:
        matches = self._find_navigation_actions_for_location_name(location_name)
        return matches[0] if matches else None

    def _find_navigation_actions_for_location_name(
        self,
        location_name: str,
        *,
        expand_generic: bool = False
    ) -> List[int]:
        target_location = self._normalize_location_text(location_name)
        if not target_location:
            return []

        exact_matches = []
        fuzzy_matches = []
        for idx, action in enumerate(self.base_planner.actions):
            action_text = str(action).lower()
            if not action_text.startswith("navigate to "):
                continue
            nav_location = self._normalize_location_text(action_text[len("navigate to "):])
            if target_location == nav_location:
                exact_matches.append(idx)
                continue
            if (
                target_location in nav_location
                or nav_location in target_location
            ):
                fuzzy_matches.append(idx)

        matches = exact_matches + fuzzy_matches
        if expand_generic:
            return list(dict.fromkeys(matches))
        return [matches[0]] if matches else []

    def _navigation_location_aliases(self, nav_action_id: int) -> set:
        if not self._is_navigation_action_id(nav_action_id):
            return set()

        action_text = str(self.base_planner.actions[nav_action_id]).lower()
        location = action_text[len("navigate to "):].strip()
        short_location = re.sub(r"\s+in\s+the\s+.*$", "", location)
        aliases = {location, short_location}
        normalized_short = self._normalize_location_text(short_location)

        if normalized_short == "refrigerator":
            aliases.add("fridge")
        elif normalized_short == "tv stand":
            aliases.add("television stand")
        elif normalized_short == "table 1":
            aliases.add("black table")
        elif normalized_short == "table 2":
            aliases.add("white table")
            aliases.add("brown table")
        elif normalized_short == "right counter":
            aliases.add("right side of the sink")
            aliases.add("right of the sink")
        elif normalized_short == "left counter":
            aliases.add("left side of the sink")
            aliases.add("left of the sink")

        return {alias for alias in aliases if alias}

    def _pick_object_name(self, pick_action_id: int) -> str:
        if not self._is_pick_action_id(pick_action_id):
            return ""
        action_text = str(self.base_planner.actions[pick_action_id]).lower()
        return action_text[len("pick up the "):].strip()

    def _extract_location_knowledge_pair(self, entry: str) -> Optional[Tuple[str, str]]:
        text = (entry or "").strip()
        if not text.lower().startswith("location knowledge:"):
            return None

        body = text.split(":", 1)[1].strip()
        match = re.search(
            r"^(.+?)\s+is\s+(?:located\s+)?(?:at|in|on|inside)\s+(.+?)(?:[.;]|$)",
            body,
            flags=re.IGNORECASE,
        )
        if not match:
            return None

        object_name = self._normalize_location_text(match.group(1))
        location_name = self._normalize_location_text(match.group(2))
        return (object_name, location_name) if object_name and location_name else None

    def _successful_pick_location_pairs_from_feedback(self) -> set:
        pairs = set()
        current_nav_action = None
        for feedback in getattr(self.base_planner, "episode_act_feedback", []) or []:
            if len(feedback) < 2:
                continue
            action_id, env_feedback = feedback[0], str(feedback[1]).lower()
            if not isinstance(action_id, int) or "executed successfully" not in env_feedback:
                continue

            if self._is_navigation_action_id(action_id):
                current_nav_action = action_id
                continue

            if self._is_pick_action_id(action_id) and current_nav_action is not None:
                object_name = self._normalize_location_text(self._pick_object_name(action_id))
                nav_text = str(self.base_planner.actions[current_nav_action]).lower()
                location_name = self._normalize_location_text(nav_text[len("navigate to "):])
                if object_name and location_name:
                    pairs.add((object_name, location_name))

        return pairs

    def _location_pair_supported_by_feedback(self, object_name: str, location_name: str) -> bool:
        object_name = self._normalize_location_text(object_name)
        location_name = self._normalize_location_text(location_name)
        if not object_name or not location_name:
            return False

        for supported_object, supported_location in self._successful_pick_location_pairs_from_feedback():
            if object_name != supported_object:
                continue
            if (
                location_name == supported_location
                or location_name in supported_location
                or supported_location in location_name
            ):
                return True
        return False

    def _filter_reflection_entries_for_clean_evidence(self, entries: List[str]) -> List[str]:
        filtered = []
        for entry in entries:
            parsed = self._extract_location_knowledge_pair(entry)
            if parsed is None:
                filtered.append(entry)
                continue

            object_name, location_name = parsed
            if self._location_pair_supported_by_feedback(object_name, location_name):
                filtered.append(entry)
            else:
                logger.info(
                    "WorldMind: dropped unsupported Location Knowledge entry "
                    f"'{entry}'"
                )

        return filtered

    def _source_navigation_candidates_from_instruction(
        self,
        user_instruction: str,
        destination_nav_action: Optional[int]
    ) -> List[int]:
        instruction = (user_instruction or "").lower()
        candidates = []

        for idx, _ in enumerate(self.base_planner.actions):
            if not self._is_navigation_action_id(idx):
                continue
            if destination_nav_action is not None and idx == destination_nav_action:
                continue

            best_score = 0
            for loc in self._navigation_location_aliases(idx):
                loc_pattern = rf"(the\s+|an?\s+)*{re.escape(loc)}\b"
                if re.search(rf"\b(from|inside|in|on|at)\s+{loc_pattern}", instruction):
                    best_score = max(best_score, len(loc))
            if best_score:
                candidates.append((best_score, idx))

        candidates.sort(reverse=True)
        return [idx for _, idx in candidates]

    def _source_navigation_candidates_from_world_knowledge(self, object_name: str) -> List[int]:
        object_name = self._normalize_location_text(object_name)
        if not object_name:
            return []

        candidates = []
        seen = set()
        for entry in self._cached_world_knowledge or []:
            knowledge = str(entry.get("knowledge", "")).lower()
            if object_name not in self._normalize_location_text(knowledge):
                continue

            for match in re.finditer(
                r"\b(?:located|found|stored|placed)\s+(?:at|in|on|inside)\s+([a-z0-9_ ]+)",
                knowledge,
            ):
                location = re.split(r"[.;,\n]", match.group(1), 1)[0].strip()
                nav_action = self._find_navigation_for_location_name(location)
                if nav_action is not None and nav_action not in seen:
                    candidates.append(nav_action)
                    seen.add(nav_action)

        return candidates

    def _common_source_navigation_candidates(self, object_name: str) -> List[int]:
        normalized_object = self._normalize_location_text(object_name)
        if not normalized_object:
            return []

        source_rules = [
            (
                {"orange", "banana", "lemon", "apple"},
                ["refrigerator", "cabinet", "left counter", "right counter", "table 1", "table 2", "tv stand"],
            ),
            (
                {"pear", "peach", "strawberry", "plum"},
                ["right counter", "sofa", "left counter", "table 1", "table 2", "refrigerator", "tv stand"],
            ),
            (
                {"can", "bottle"},
                ["refrigerator", "left counter", "right counter", "cabinet", "table 1", "table 2"],
            ),
            (
                {"rubriks cube"},
                ["table 1", "table 2", "sofa", "tv stand", "left counter", "right counter"],
            ),
            (
                {"toy airplane", "lego", "ball", "block", "book"},
                ["sofa", "table 1", "table 2", "tv stand", "left counter", "right counter"],
            ),
            (
                {"mug", "cup", "bowl", "plate", "spoon", "knife", "spatula"},
                ["left counter", "right counter", "table 1", "table 2", "cabinet", "right drawer", "left drawer", "sink"],
            ),
            (
                {"lid"},
                ["table 1", "table 2", "sofa", "tv stand", "cabinet", "right drawer", "left drawer", "right counter", "sink", "refrigerator", "left counter"],
            ),
            (
                {"sponge", "cleanser"},
                ["sink", "left counter", "right counter", "cabinet"],
            ),
            (
                {"hammer", "screwdriver", "wrench", "drill", "clamp", "scissors"},
                ["cabinet", "left drawer", "right drawer", "table 1", "table 2", "left counter", "right counter"],
            ),
        ]

        candidates = []
        seen = set()
        for object_names, source_locations in source_rules:
            if normalized_object not in object_names:
                continue
            for location in source_locations:
                expand_generic = (
                    normalized_object in {"hammer", "screwdriver", "wrench", "drill", "clamp", "scissors", "lid", "mug", "cup", "bowl", "plate", "spoon", "knife", "spatula"}
                    and location in {"cabinet", "drawer"}
                )
                for nav_action in self._find_navigation_actions_for_location_name(
                    location,
                    expand_generic=expand_generic
                ):
                    if nav_action not in seen:
                        candidates.append(nav_action)
                        seen.add(nav_action)
            break

        return candidates

    def _source_navigation_candidates_for_pick(
        self,
        pick_action_id: int,
        user_instruction: str
    ) -> List[int]:
        object_name = self._pick_object_name(pick_action_id)
        if not object_name:
            return []

        destination_nav_action = None
        destination_place_action = self._find_instruction_place_action(user_instruction)
        if destination_place_action is not None:
            destination_nav_action = self._find_navigation_for_place_action(destination_place_action)

        candidates = []
        seen = set()
        for candidate_list in [
            self._source_navigation_candidates_from_instruction(user_instruction, destination_nav_action),
            self._common_source_navigation_candidates(object_name),
            self._source_navigation_candidates_from_world_knowledge(object_name),
        ]:
            for nav_action in candidate_list:
                if nav_action == destination_nav_action:
                    continue
                if nav_action not in seen:
                    candidates.append(nav_action)
                    seen.add(nav_action)

        return candidates

    def _find_open_action_for_location_name(self, location_name: str) -> Optional[int]:
        target_location = self._normalize_location_text(location_name)
        if not target_location:
            return None

        for idx, action in enumerate(self.base_planner.actions):
            action_text = str(action).lower()
            if not action_text.startswith("open the "):
                continue
            open_target = self._normalize_location_text(action_text[len("open the "):])
            if (
                target_location == open_target
                or target_location in open_target
                or open_target in target_location
            ):
                return idx

        return None

    def _find_open_action_for_navigation(self, nav_action_id: int) -> Optional[int]:
        if not self._is_navigation_action_id(nav_action_id):
            return None
        nav_text = str(self.base_planner.actions[nav_action_id]).lower()
        nav_location = nav_text[len("navigate to "):].strip()
        return self._find_open_action_for_location_name(nav_location)

    def _receptacle_known_open(self, location_name: str) -> bool:
        target = self._normalize_location_text(location_name)
        if not target:
            return False

        is_refrigerator = "refrigerator" in target or "fridge" in target
        cabinet_match = re.search(r"cabinet\s+([a-z0-9_]+)", target)
        cabinet_id = cabinet_match.group(1) if cabinet_match else None

        for feedback in reversed(getattr(self.base_planner, "episode_act_feedback", []) or []):
            if len(feedback) < 2:
                continue
            env_feedback = str(feedback[1]).lower()
            if is_refrigerator:
                if "now refrigerator is closed" in env_feedback:
                    return False
                if "now refrigerator is open" in env_feedback:
                    return True
            if "cabinet" in target:
                if cabinet_id and f"now cabinet {cabinet_id} is closed" in env_feedback:
                    return False
                if cabinet_id and f"now cabinet {cabinet_id} is open" in env_feedback:
                    return True
                if "now cabinet" in env_feedback and "closed" in env_feedback:
                    return False
                if "now cabinet" in env_feedback and "open" in env_feedback:
                    return True

        return False

    def _find_instruction_place_action(self, user_instruction: str) -> Optional[int]:
        instruction = (user_instruction or "").lower()
        if not instruction:
            return None

        candidates = []
        contrast_positions = [
            instruction.rfind(marker)
            for marker in [" but ", " instead ", " otherwise ", " then "]
        ]
        contrast_pos = max(contrast_positions)

        movement_verbs = r"(move|bring|put|place|transfer|transport|relocate|deliver)"

        for idx, action in enumerate(self.base_planner.actions):
            action_text = str(action).lower()
            if not action_text.startswith("place at "):
                continue
            location = action_text[len("place at "):].strip()
            short_location = re.sub(r"\s+in\s+the\s+.*$", "", location)
            location_aliases = {location, short_location}
            normalized_short = self._normalize_location_text(short_location)
            if normalized_short == "table 1":
                location_aliases.add("black table")
            elif normalized_short == "table 2":
                location_aliases.add("white table")
                location_aliases.add("brown table")
            elif normalized_short == "sofa":
                location_aliases.add("large gray piece of furniture with a backrest")
                location_aliases.add("gray piece of furniture with a backrest")
                location_aliases.add("large gray furniture with a backrest")
            elif normalized_short == "right counter":
                location_aliases.add("right of the sink")
                location_aliases.add("right side of the sink")
            elif normalized_short == "left counter":
                location_aliases.add("left of the sink")
                location_aliases.add("left side of the sink")
            elif normalized_short == "sink":
                location_aliases.add("right of the left counter")
                location_aliases.add("right receptacle of the left counter")
                location_aliases.add("right receptacle of left counter")
                location_aliases.add("left of the right counter")
                location_aliases.add("left receptacle of the right counter")
                location_aliases.add("left receptacle of right counter")
            elif normalized_short == "right drawer of kitchen counter":
                location_aliases.add("right drawer of the left counter")
            elif normalized_short == "left drawer of kitchen counter":
                location_aliases.add("left drawer of the right counter")

            for loc in location_aliases:
                if not loc:
                    continue
                loc_pattern = rf"(the\s+|an?\s+|allocated\s+|indicated\s+)*{re.escape(loc)}\b"
                prep_pattern = rf"\b(to|into|onto|on|at|in)\s+{loc_pattern}"
                match = re.search(prep_pattern, instruction)
                if not match:
                    continue

                score = len(loc)
                if match.group(1) in {"to", "into", "onto"}:
                    score += 20



                if re.search(rf"{movement_verbs}[^.;,]*?\b(to|into|onto|on|at|in)\s+{loc_pattern}", instruction):
                    score += 100
                if re.search(rf"\b(on|at|in)\s+{loc_pattern}[^.;,]{{0,80}}\b(need|put|place|move|bring|transfer|transport|relocate|deliver)\b", instruction):
                    score += 80

                if contrast_pos >= 0:
                    if match.start() > contrast_pos:
                        score += 50
                    elif match.end() < contrast_pos:
                        score -= 50

                candidates.append((score, idx))
        if not candidates:
            return None
        candidates.sort(reverse=True)
        return candidates[0][1]

    def _find_instruction_source_place_action(self, user_instruction: str) -> Optional[int]:
        instruction = (user_instruction or "").lower()
        if not instruction:
            return None

        candidates = []
        for idx, action in enumerate(self.base_planner.actions):
            action_text = str(action).lower()
            if not action_text.startswith("place at "):
                continue
            location = action_text[len("place at "):].strip()
            short_location = re.sub(r"\s+in\s+the\s+.*$", "", location)
            aliases = {location, short_location}
            normalized_short = self._normalize_location_text(short_location)
            if normalized_short == "table 1":
                aliases.add("black table")
            elif normalized_short == "table 2":
                aliases.add("brown table")
                aliases.add("white table")

            for loc in aliases:
                if not loc:
                    continue
                loc_pattern = rf"(the\s+|an?\s+)*{re.escape(loc)}\b"
                score = len(loc)
                if re.search(rf"\bfrom\s+{loc_pattern}", instruction):
                    candidates.append((score + 100, idx))
                elif re.search(rf"\bon\s+{loc_pattern}\s*,?\s*[^.;]{{0,80}}\b(remove|displace|detach|move off)\b", instruction):
                    candidates.append((score + 80, idx))

        if not candidates:
            return None
        candidates.sort(reverse=True)
        return candidates[0][1]

    def _default_alternative_place_action(self, source_place_action: int) -> Optional[int]:
        preferred_locations = [
            "place at the table 2",
            "place at the left counter in the kitchen",
            "place at the right counter in the kitchen",
            "place at the sofa",
            "place at the table 1",
        ]
        for preferred in preferred_locations:
            for idx, action in enumerate(self.base_planner.actions):
                if idx == source_place_action:
                    continue
                if str(action).lower() == preferred:
                    return idx
        return None

    def _repair_destination_place_actions(self, actions: Any, user_instruction: str) -> Any:
        if not isinstance(actions, list) or len(actions) == 0:
            return actions

        target_place_action = self._find_instruction_place_action(user_instruction)
        if target_place_action is None:
            return actions

        repaired = []
        changed = False
        for action_id in actions:
            if self._is_place_action_id(action_id) and action_id != target_place_action:
                logger.info(
                    f"WorldMind: replaced place action id {action_id} "
                    f"with instruction destination place id {target_place_action}"
                )
                repaired.append(target_place_action)
                changed = True
            else:
                repaired.append(action_id)

        return repaired if changed else actions

    def _repair_remove_place_back_to_source(self, actions: Any, user_instruction: str) -> Any:
        if not isinstance(actions, list) or len(actions) == 0:
            return actions

        instruction = (user_instruction or "").lower()
        if not re.search(r"\b(remove|displace|detach|move off)\b", instruction):
            return actions

        source_place_action = self._find_instruction_source_place_action(user_instruction)
        if source_place_action is None:
            return actions

        alternative_place_action = self._default_alternative_place_action(source_place_action)
        if alternative_place_action is None:
            return actions

        repaired = []
        changed = False
        for action_id in actions:
            if action_id == source_place_action:
                logger.info(
                    "WorldMind: replaced place-back-to-source action "
                    f"{source_place_action} with alternative place {alternative_place_action}"
                )
                repaired.append(alternative_place_action)
                changed = True
            else:
                repaired.append(action_id)

        return repaired if changed else actions

    def _find_next_action_index(self, actions: List[Any], start_idx: int, action_kind: str) -> Optional[int]:
        for idx in range(start_idx, len(actions)):
            if action_kind == "pick" and self._is_pick_action_id(actions[idx]):
                return idx
            if action_kind == "place" and self._is_place_action_id(actions[idx]):
                return idx
        return None

    def _repair_one_hand_plan_sequence(self, actions: Any) -> Any:
        if not isinstance(actions, list) or len(actions) == 0:
            return actions

        repaired = []
        holding_unplaced_object = False
        changed = False
        idx = 0

        while idx < len(actions):
            action_id = actions[idx]

            if holding_unplaced_object:
                next_pick_idx = self._find_next_action_index(actions, idx, "pick")
                next_place_idx = self._find_next_action_index(actions, idx, "place")

                if self._is_place_action_id(action_id):
                    repaired.append(action_id)
                    holding_unplaced_object = False
                    idx += 1
                    continue

                if next_pick_idx is not None and (next_place_idx is None or next_pick_idx < next_place_idx):
                    logger.info(
                        "WorldMind: inserted destination placement between pickups "
                        "to satisfy one-hand constraint"
                    )

                    if next_place_idx is None:
                        return repaired if repaired else actions

                    place_action_id = actions[next_place_idx]
                    nav_action_id = self._find_navigation_for_place_action(place_action_id)
                    if nav_action_id is not None and (not repaired or repaired[-1] != nav_action_id):
                        repaired.append(nav_action_id)
                    repaired.append(place_action_id)
                    holding_unplaced_object = False
                    changed = True




                    if action_id == nav_action_id:
                        idx += 1
                    continue

            repaired.append(action_id)
            if self._is_pick_action_id(action_id):
                holding_unplaced_object = True
            elif self._is_place_action_id(action_id):
                holding_unplaced_object = False
            idx += 1

        return repaired if changed else actions

    def _repair_place_before_pick_plan(self, actions: Any, user_instruction: str) -> Any:
        if not isinstance(actions, list) or len(actions) == 0:
            return actions
        if self._current_held_object_from_feedback():
            return actions

        first_pick_index = self._find_next_action_index(actions, 0, "pick")
        first_place_index = self._find_next_action_index(actions, 0, "place")
        if first_place_index is None:
            return actions
        if first_pick_index is not None and first_pick_index < first_place_index:
            return actions

        pick_action = self._find_candidate_pick_action(user_instruction)
        if pick_action is None:
            return actions

        place_action = actions[first_place_index]
        nav_action = self._find_navigation_for_place_action(place_action)

        for action_id in actions[:first_place_index]:
            if not self._is_navigation_action_id(action_id):
                continue
            if self._is_likely_destination_navigation(action_id, user_instruction):
                continue
            if pick_action in self._failed_pick_actions_at_navigation(action_id):
                continue

            repaired = [action_id, pick_action]
            if nav_action is not None and nav_action != action_id:
                repaired.append(nav_action)
            repaired.append(place_action)
            repaired.extend(actions[first_place_index + 1:])
            logger.info(
                "WorldMind: inserted candidate pickup after source navigation "
                f"{action_id} before premature place action"
            )
            return repaired

        repaired = actions[:first_place_index]
        if not repaired or repaired[-1] != pick_action:
            repaired.append(pick_action)
        if nav_action is not None and (not repaired or repaired[-1] != nav_action):
            repaired.append(nav_action)
        repaired.append(place_action)
        repaired.extend(actions[first_place_index + 1:])

        logger.info(
            "WorldMind: inserted candidate pickup before premature place action "
            f"using action id {pick_action}"
        )
        return repaired

    def _repair_empty_hand_leading_place_plan(self, actions: Any, user_instruction: str) -> Any:
        if not isinstance(actions, list) or len(actions) == 0:
            return actions
        if not self._last_feedback_indicates_empty_hand():
            return actions

        first_pick_index = self._find_next_action_index(actions, 0, "pick")
        first_place_index = self._find_next_action_index(actions, 0, "place")
        if first_place_index is None:
            return actions
        if first_pick_index is not None and first_pick_index < first_place_index:
            return actions

        suffix = actions[first_place_index + 1:]
        if suffix:
            logger.info(
                "WorldMind: dropped empty-hand leading place sequence before "
                "continuing search"
            )
            return suffix

        fallback_probe = self._next_source_probe_for_exact_targets(user_instruction)
        if fallback_probe is not None:
            logger.info(
                "WorldMind: replaced empty-hand leading place sequence with "
                f"remaining-target probe {fallback_probe}"
            )
            return fallback_probe

        return actions

    def _repair_place_navigation_sequence(self, actions: Any) -> Any:
        if not isinstance(actions, list) or len(actions) == 0:
            return actions

        repaired = []
        changed = False
        for action_id in actions:
            if self._is_place_action_id(action_id):
                nav_action_id = self._find_navigation_for_place_action(action_id)
                previous_action = repaired[-1] if repaired else self._last_successful_action_id()
                if nav_action_id is not None and previous_action != nav_action_id:
                    logger.info(
                        "WorldMind: inserted matching navigation before place action "
                        f"{action_id}"
                    )
                    repaired.append(nav_action_id)
                    changed = True
            repaired.append(action_id)

        return repaired if changed else actions

    def _repair_pre_pick_source_probe_sequence(self, actions: Any, user_instruction: str) -> Any:
        if not isinstance(actions, list) or len(actions) == 0:
            return actions
        if self._current_held_object_from_feedback():
            return actions

        first_pick_index = self._find_next_action_index(actions, 0, "pick")
        if first_pick_index is None or first_pick_index <= 0:
            return actions

        pick_action = actions[first_pick_index]
        if not self._is_pick_action_id(pick_action):
            return actions

        for idx in range(first_pick_index):
            nav_action = actions[idx]
            if not self._is_navigation_action_id(nav_action):
                continue
            if self._is_likely_destination_navigation(nav_action, user_instruction):
                continue
            if pick_action in self._failed_pick_actions_at_navigation(nav_action):
                continue

            suffix = []
            place_action = None
            next_place_idx = self._find_next_action_index(actions, first_pick_index + 1, "place")
            if next_place_idx is not None:
                place_action = actions[next_place_idx]
            if place_action is None:
                place_action = self._find_instruction_place_action(user_instruction)

            if place_action is not None:
                nav_to_place = self._find_navigation_for_place_action(place_action)
                if nav_to_place is not None and nav_to_place != nav_action:
                    suffix.append(nav_to_place)
                suffix.append(place_action)
            else:
                suffix = actions[first_pick_index + 1:]

            logger.info(
                "WorldMind: inserted immediate pickup probe after source navigation "
                f"{nav_action} before continuing source search"
            )
            return [nav_action, pick_action] + suffix

        return actions

    def _repair_open_close_navigation_sequence(self, actions: Any) -> Any:
        if not isinstance(actions, list) or len(actions) == 0:
            return actions

        repaired = []
        changed = False
        for action_id in actions:
            if isinstance(action_id, int) and 0 <= action_id < len(self.base_planner.actions):
                action_text = str(self.base_planner.actions[action_id]).lower()
                target_name = ""
                for prefix in ("open the ", "close the "):
                    if action_text.startswith(prefix):
                        target_name = action_text[len(prefix):].strip()
                        break
                if target_name:
                    if action_text.startswith("open the ") and self._receptacle_known_open(target_name):
                        logger.info(
                            "WorldMind: removed repeated open action for known-open receptacle "
                            f"{action_id}"
                        )
                        changed = True
                        continue
                    nav_action_id = self._find_navigation_for_location_name(target_name)
                    previous_action = repaired[-1] if repaired else self._last_successful_action_id()
                    if nav_action_id is not None and previous_action != nav_action_id:
                        logger.info(
                            "WorldMind: inserted matching navigation before open/close action "
                            f"{action_id}"
                        )
                        repaired.append(nav_action_id)
                        changed = True
            repaired.append(action_id)

        return repaired if changed else actions

    def _explicit_refrigerator_state_plan(self, user_instruction: str) -> Optional[List[int]]:
        instruction = (user_instruction or "").lower()
        if "fridge" not in instruction and "refrigerator" not in instruction:
            return None

        wants_close = re.search(r"\b(close|shut)\b", instruction) is not None
        if not wants_close and "open" in instruction:
            wants_close = (
                re.search(r"\b(left|leave|leaving|mistake)\b.{0,80}\b(fridge|refrigerator)\b.{0,80}\bopen\b", instruction) is not None
                or re.search(r"\b(fridge|refrigerator)(?:\s+door)?\b.{0,40}\b(is|was|left)\b.{0,20}\bopen\b", instruction) is not None
            )
        wants_open = re.search(r"\bopen\b", instruction) is not None and not wants_close
        if not wants_close and not wants_open:
            return None

        target_action_name = "close the refrigerator" if wants_close else "open the refrigerator"
        state_action = self._action_id_by_exact_name(target_action_name)
        nav_action = self._find_navigation_for_location_name("refrigerator")
        if state_action is None:
            return None

        plan = []
        if nav_action is not None:
            plan.append(nav_action)
        plan.append(state_action)
        return plan

    def _repair_explicit_refrigerator_state_task(self, actions: Any, user_instruction: str) -> Any:
        repaired = self._explicit_refrigerator_state_plan(user_instruction)
        if repaired is None:
            return actions

        if repaired != actions:
            logger.info(
                "WorldMind: normalized explicit refrigerator state task to "
                f"{repaired}"
            )
        return repaired

    def _repair_container_source_open_sequence(self, actions: Any) -> Any:
        if not isinstance(actions, list) or len(actions) == 0:
            return actions

        repaired = []
        changed = False
        for action_id in actions:
            if self._is_pick_action_id(action_id):
                previous_action = repaired[-1] if repaired else self._last_successful_action_id()
                open_action = self._find_open_action_for_navigation(previous_action)
                previous_location = ""
                if self._is_navigation_action_id(previous_action):
                    previous_text = str(self.base_planner.actions[previous_action]).lower()
                    previous_location = previous_text[len("navigate to "):].strip()
                if (
                    open_action is not None
                    and not self._receptacle_known_open(previous_location)
                    and (len(repaired) < 2 or repaired[-2] != open_action)
                    and previous_action != open_action
                ):
                    logger.info(
                        "WorldMind: inserted source container open action "
                        f"{open_action} before pickup {action_id}"
                    )
                    repaired.append(open_action)
                    changed = True
            repaired.append(action_id)

        return repaired if changed else actions

    def _repair_failed_source_pick_sequence(self, actions: Any, user_instruction: str) -> Any:
        if not isinstance(actions, list) or len(actions) == 0:
            return actions
        if self._current_held_object_from_feedback():
            return actions

        first_pick_index = self._find_next_action_index(actions, 0, "pick")
        if first_pick_index is None:
            return actions
        pick_action = actions[first_pick_index]
        if not self._is_pick_action_id(pick_action):
            return actions

        source_nav = None
        for idx in range(first_pick_index - 1, -1, -1):
            candidate = actions[idx]
            if self._is_navigation_action_id(candidate):
                source_nav = candidate
                break
            if self._is_pick_action_id(candidate) or self._is_place_action_id(candidate):
                break

        if source_nav is None:
            return actions
        if self._is_likely_destination_navigation(source_nav, user_instruction):
            return actions

        if pick_action in self._completed_pick_actions_from_feedback():
            alternate_pick = self._select_exact_pick_for_navigation(
                source_nav,
                user_instruction,
                exclude_completed=True,
                require_source_match=False
            )
            if alternate_pick is not None and alternate_pick != pick_action:
                logger.info(
                    "WorldMind: replaced completed failed-source pickup "
                    f"{pick_action} with remaining target {alternate_pick}"
                )
                repaired = list(actions)
                repaired[first_pick_index] = alternate_pick
                return repaired
            return actions

        if pick_action not in self._failed_pick_actions_at_navigation(source_nav):
            return actions

        alternate_pick = self._select_exact_pick_for_navigation(
            source_nav,
            user_instruction,
            exclude_completed=True,
            require_source_match=True
        )
        if alternate_pick is not None and alternate_pick != pick_action:
            logger.info(
                "WorldMind: changed failed source probe pickup "
                f"{pick_action} to source-compatible exact target {alternate_pick}"
            )
            repaired = list(actions)
            repaired[first_pick_index] = alternate_pick
            return repaired

        for nav_action_id in self._source_navigation_candidates_for_pick(pick_action, user_instruction):
            if nav_action_id == source_nav:
                continue
            if pick_action in self._failed_pick_actions_at_navigation(nav_action_id):
                continue
            if self._is_likely_destination_navigation(nav_action_id, user_instruction):
                continue

            prefix = [nav_action_id]
            open_action = self._find_open_action_for_navigation(nav_action_id)
            nav_text = str(self.base_planner.actions[nav_action_id]).lower()
            nav_location = nav_text[len("navigate to "):].strip()
            if open_action is not None and not self._receptacle_known_open(nav_location):
                prefix.append(open_action)
            logger.info(
                "WorldMind: rotated failed source/pick pair "
                f"({source_nav}, {pick_action}) to source {nav_action_id}"
            )
            return prefix + [pick_action] + actions[first_pick_index + 1:]

        return actions

    def _repair_destination_as_source_pick_sequence(self, actions: Any, user_instruction: str) -> Any:
        if not isinstance(actions, list) or len(actions) == 0:
            return actions
        if self._current_held_object_from_feedback():
            return actions

        first_pick_index = self._find_next_action_index(actions, 0, "pick")
        if first_pick_index is None:
            return actions
        pick_action = actions[first_pick_index]
        if not self._is_pick_action_id(pick_action):
            return actions

        source_nav = None
        source_nav_index = None
        for idx in range(first_pick_index - 1, -1, -1):
            candidate = actions[idx]
            if self._is_navigation_action_id(candidate):
                source_nav = candidate
                source_nav_index = idx
                break
            if self._is_pick_action_id(candidate) or self._is_place_action_id(candidate):
                break

        if source_nav is None or source_nav_index is None:
            return actions
        if not self._is_likely_destination_navigation(source_nav, user_instruction):
            return actions

        for nav_action_id in self._source_navigation_candidates_for_pick(pick_action, user_instruction):
            if nav_action_id == source_nav:
                continue
            if pick_action in self._failed_pick_actions_at_navigation(nav_action_id):
                continue
            if self._is_likely_destination_navigation(nav_action_id, user_instruction):
                continue

            prefix = self._build_source_probe_plan(nav_action_id, pick_action)
            logger.info(
                "WorldMind: replaced destination-as-source pickup probe "
                f"({source_nav}, {pick_action}) with source probe {prefix}"
            )
            return actions[:source_nav_index] + prefix + actions[first_pick_index + 1:]

        return actions

    def _repair_low_priority_source_pick_sequence(self, actions: Any, user_instruction: str) -> Any:
        if not isinstance(actions, list) or len(actions) == 0:
            return actions
        if self._current_held_object_from_feedback():
            return actions

        first_pick_index = self._find_next_action_index(actions, 0, "pick")
        if first_pick_index is None:
            return actions
        pick_action = actions[first_pick_index]
        if not self._is_pick_action_id(pick_action):
            return actions
        object_name = self._normalize_location_text(self._pick_object_name(pick_action))
        if object_name not in {"lid", "hammer", "screwdriver", "wrench", "drill", "clamp", "scissors"}:
            return actions
        if self._pick_failure_counts_from_feedback().get(pick_action, 0) < 1:
            return actions

        source_nav = None
        source_nav_index = None
        for idx in range(first_pick_index - 1, -1, -1):
            candidate = actions[idx]
            if self._is_navigation_action_id(candidate):
                source_nav = candidate
                source_nav_index = idx
                break
            if self._is_pick_action_id(candidate) or self._is_place_action_id(candidate):
                break

        if source_nav is None or source_nav_index is None:
            return actions

        if pick_action in self._completed_pick_actions_from_feedback():
            alternate_pick = self._select_exact_pick_for_navigation(
                source_nav,
                user_instruction,
                exclude_completed=True,
                require_source_match=False
            )
            if alternate_pick is not None and alternate_pick != pick_action:
                logger.info(
                    "WorldMind: replaced completed low-priority pickup "
                    f"{pick_action} with remaining target {alternate_pick}"
                )
                repaired = list(actions)
                repaired[first_pick_index] = alternate_pick
                return repaired
            return actions

        for nav_action_id in self._source_navigation_candidates_for_pick(pick_action, user_instruction):
            if self._is_likely_destination_navigation(nav_action_id, user_instruction):
                continue
            if pick_action in self._failed_pick_actions_at_navigation(nav_action_id):
                continue
            if nav_action_id == source_nav:
                return actions

            prefix = self._build_source_probe_plan(nav_action_id, pick_action)
            logger.info(
                "WorldMind: replaced low-priority source probe "
                f"({source_nav}, {pick_action}) with best remaining probe {prefix}"
            )
            return actions[:source_nav_index] + prefix + actions[first_pick_index + 1:]

        return actions

    def _repair_pickup_source_navigation(self, actions: Any, user_instruction: str) -> Any:
        if not isinstance(actions, list) or len(actions) == 0:
            return actions
        first_action = actions[0]
        if not self._is_pick_action_id(first_action):
            return actions
        if self._current_held_object_from_feedback():
            return actions

        last_nav_action = self._last_successful_navigation_action_id()
        if (
            last_nav_action is not None
            and not self._is_likely_destination_navigation(last_nav_action, user_instruction)
            and first_action not in self._failed_pick_actions_at_navigation(last_nav_action)
        ):
            return actions

        for nav_action_id in self._source_navigation_candidates_for_pick(first_action, user_instruction):
            if last_nav_action is not None and nav_action_id == last_nav_action:
                continue
            if first_action in self._failed_pick_actions_at_navigation(nav_action_id):
                continue
            logger.info(
                "WorldMind: inserted source navigation before pickup action "
                f"{first_action} using navigation action {nav_action_id}"
            )
            prefix = [nav_action_id]
            open_action = self._find_open_action_for_navigation(nav_action_id)
            nav_text = str(self.base_planner.actions[nav_action_id]).lower()
            nav_location = nav_text[len("navigate to "):].strip()
            if open_action is not None and not self._receptacle_known_open(nav_location):
                prefix.append(open_action)
            return prefix + actions

        return actions

    def _pick_action_for_navigation_probe(self, nav_action_id: int, user_instruction: str) -> Optional[int]:
        exact_pick_actions = self._find_instruction_pick_actions(user_instruction)
        if len(exact_pick_actions) >= 2:
            source_matched = self._select_exact_pick_for_navigation(
                nav_action_id,
                user_instruction,
                exclude_completed=True,
                require_source_match=True
            )
            if source_matched is not None:
                return source_matched



            return None

        failed_here = self._failed_pick_actions_at_navigation(nav_action_id)
        for pick_action in exact_pick_actions:
            if pick_action not in failed_here:
                return pick_action

        candidate = self._find_candidate_pick_action(user_instruction)
        if candidate is not None and candidate not in failed_here:
            return candidate
        return None

    def _build_source_probe_plan(self, nav_action_id: int, pick_action_id: int) -> List[int]:
        prefix = [nav_action_id]
        open_action = self._find_open_action_for_navigation(nav_action_id)
        nav_text = str(self.base_planner.actions[nav_action_id]).lower()
        nav_location = nav_text[len("navigate to "):].strip()
        if open_action is not None and not self._receptacle_known_open(nav_location):
            prefix.append(open_action)
        return prefix + [pick_action_id]

    def _next_source_probe_for_exact_targets(self, user_instruction: str) -> Optional[List[int]]:
        if self._current_held_object_from_feedback():
            return None

        exact_pick_actions = self._find_instruction_pick_actions(user_instruction)
        if not exact_pick_actions:
            return None

        completed = self._completed_pick_actions_from_feedback()
        active_picks = [action_id for action_id in exact_pick_actions if action_id not in completed]
        if not active_picks:
            return None

        failure_counts = self._pick_failure_counts_from_feedback()
        visit_counts = self._navigation_visit_counts_from_feedback()
        scored = []

        for pick_order, pick_action in enumerate(active_picks):
            source_candidates = self._source_navigation_candidates_for_pick(
                pick_action,
                user_instruction
            )
            for source_order, nav_action in enumerate(source_candidates):
                if self._is_likely_destination_navigation(nav_action, user_instruction):
                    continue
                if pick_action in self._failed_pick_actions_at_navigation(nav_action):
                    continue

                score = (
                    1000
                    - source_order * 40
                    - failure_counts.get(pick_action, 0) * 35
                    - visit_counts.get(nav_action, 0) * 20
                    - pick_order
                )
                scored.append((score, -source_order, -pick_order, nav_action, pick_action))

        if not scored:
            return None

        scored.sort(reverse=True)
        _, _, _, nav_action, pick_action = scored[0]
        return self._build_source_probe_plan(nav_action, pick_action)

    def _repair_common_sense_candidate_rotation(self, actions: Any, user_instruction: str) -> Any:
        if not isinstance(actions, list) or len(actions) == 0:
            return actions
        if self._find_instruction_pick_action(user_instruction) is not None:
            return actions
        if self._is_pick_action_id(actions[0]):
            return actions
        if not self._last_action_was_failed_pick():
            return actions

        nav_action_id = self._last_successful_navigation_action_id()
        if nav_action_id is None:
            return actions

        inferred_actions = self._infer_common_sense_pick_actions(user_instruction)
        if not inferred_actions:
            return actions

        failed_here = self._failed_pick_actions_at_navigation(nav_action_id)
        for candidate in inferred_actions:
            if candidate not in failed_here:
                logger.info(
                    "WorldMind: rotating to next common-sense pickup candidate "
                    f"{candidate} before leaving current location"
                )
                return [candidate] + actions

        return actions

    def _repair_occupied_hand_plan(self, actions: Any, user_instruction: str) -> Any:
        if not isinstance(actions, list) or len(actions) == 0:
            return actions

        held_object = self._current_held_object_from_feedback()
        if not held_object:
            return actions

        first_pick_index = None
        first_place_index = None
        for idx, action_id in enumerate(actions):
            if not isinstance(action_id, int) or not (0 <= action_id < len(self.base_planner.actions)):
                continue
            action_text = str(self.base_planner.actions[action_id]).lower()
            if first_pick_index is None and action_text.startswith("pick up "):
                first_pick_index = idx
            if first_place_index is None and action_text.startswith("place at "):
                first_place_index = idx

        instruction_place_action = self._find_instruction_place_action(user_instruction)
        instruction_nav_action = (
            self._find_navigation_for_place_action(instruction_place_action)
            if instruction_place_action is not None
            else None
        )

        if first_pick_index is None:
            if instruction_place_action is None:
                return actions

            if first_place_index is None:
                prefix = []
                if (
                    instruction_nav_action is not None
                    and self._last_successful_action_id() != instruction_nav_action
                ):
                    prefix.append(instruction_nav_action)
                prefix.append(instruction_place_action)
                logger.info(
                    "WorldMind: replaced held-object search plan with "
                    f"instruction destination placement (holding {held_object})"
                )
                return prefix

            if (
                instruction_nav_action is not None
                and any(
                    self._is_navigation_action_id(action_id)
                    and action_id != instruction_nav_action
                    for action_id in actions[:first_place_index]
                )
            ):
                logger.info(
                    "WorldMind: shortened held-object plan to destination "
                    f"placement (holding {held_object})"
                )
                return [instruction_nav_action, instruction_place_action]

            return actions

        if first_place_index is not None and first_place_index < first_pick_index:
            return actions

        place_action_id = actions[first_place_index] if first_place_index is not None else None
        if place_action_id is None:
            place_action_id = instruction_place_action
        if place_action_id is None:
            return actions

        prefix = []
        nav_action_id = self._find_navigation_for_place_action(place_action_id)
        if nav_action_id is not None and self._last_successful_action_id() != nav_action_id:
            prefix.append(nav_action_id)
        prefix.append(place_action_id)

        logger.info(
            "WorldMind: inserted held-object placement before new pickup "
            f"(holding {held_object})"
        )
        return prefix + actions[first_pick_index + 1:]

    def _repair_navigation_only_search_plan(self, actions: Any, user_instruction: str) -> Any:
        if not isinstance(actions, list) or len(actions) == 0:
            return actions

        if not all(self._is_navigation_action_id(action) for action in actions):
            return actions

        for idx, nav_action in enumerate(actions):
            if self._is_likely_destination_navigation(nav_action, user_instruction):
                continue
            pick_action = self._pick_action_for_navigation_probe(nav_action, user_instruction)
            if pick_action is None:
                continue

            repaired = actions[:idx + 1] + [pick_action] + actions[idx + 1:]
            logger.info(
                "WorldMind: inserted pickup probe after navigation-only search plan "
                f"using action id {pick_action}"
            )
            return repaired

        fallback_probe = self._next_source_probe_for_exact_targets(user_instruction)
        if fallback_probe is not None:
            logger.info(
                "WorldMind: replaced navigation-only search with next clean "
                f"source probe {fallback_probe}"
            )
            return fallback_probe

        return actions

    def _repair_search_plan_without_pick(self, actions: Any, user_instruction: str) -> Any:
        if not isinstance(actions, list) or len(actions) == 0:
            return actions
        if self._current_held_object_from_feedback():
            return actions
        if any(self._is_pick_action_id(action_id) for action_id in actions):
            return actions
        if not self._find_instruction_pick_actions(user_instruction):
            return actions

        fallback_probe = self._next_source_probe_for_exact_targets(user_instruction)
        if fallback_probe is None:
            return actions

        logger.info(
            "WorldMind: replaced search plan without pickup with next clean "
            f"source probe {fallback_probe}"
        )
        return fallback_probe

    def _drop_leading_redundant_navigation(self, actions: Any) -> Any:
        self._dropped_leading_redundant_navigation = False
        if not isinstance(actions, list) or len(actions) <= 1:
            return actions

        last_successful_action = self._last_successful_action_id()
        while (
            len(actions) > 1
            and actions[0] == last_successful_action
            and self._is_navigation_action_id(actions[0])
        ):
            logger.info(f"WorldMind: dropped leading redundant navigation action id {actions[0]}")
            self._dropped_leading_redundant_navigation = True
            actions = actions[1:]

        return actions

    def _normalize_executable_actions(self, actions: Any, user_instruction: str) -> Any:
        if not isinstance(actions, list):
            return actions

        deduped_actions = []
        for action in actions:
            if deduped_actions and action == deduped_actions[-1]:
                logger.info(f"WorldMind: removed consecutive duplicate executable action id {action}")
                continue
            deduped_actions.append(action)

        fridge_state_plan = self._explicit_refrigerator_state_plan(user_instruction)
        if fridge_state_plan is not None:
            if fridge_state_plan != deduped_actions:
                logger.info(
                    "WorldMind: normalized explicit refrigerator state task to "
                    f"{fridge_state_plan}"
                )
            return fridge_state_plan

        deduped_actions = self._repair_single_target_pick_actions(deduped_actions, user_instruction)
        deduped_actions = self._repair_multi_target_pick_actions(deduped_actions, user_instruction)
        deduped_actions = self._repair_destination_place_actions(deduped_actions, user_instruction)
        deduped_actions = self._repair_remove_place_back_to_source(deduped_actions, user_instruction)
        deduped_actions = self._drop_leading_redundant_navigation(deduped_actions)
        if getattr(self, "_dropped_leading_redundant_navigation", False):
            last_nav_action = self._last_successful_navigation_action_id()
            pick_action = self._select_exact_pick_for_navigation(
                last_nav_action,
                user_instruction,
                exclude_completed=True,
                require_source_match=False
            )
            if pick_action is None:
                pick_action = self._find_candidate_pick_action(user_instruction)
            if (
                pick_action is not None
                and (
                    not deduped_actions
                    or self._is_likely_destination_navigation(deduped_actions[0], user_instruction)
                )
            ):
                logger.info(
                    "WorldMind: inserted pickup probe after dropping redundant source navigation "
                    f"using action id {pick_action}"
                )
                return [pick_action] + deduped_actions

        deduped_actions = self._repair_occupied_hand_plan(deduped_actions, user_instruction)
        deduped_actions = self._repair_one_hand_plan_sequence(deduped_actions)
        deduped_actions = self._repair_common_sense_candidate_rotation(deduped_actions, user_instruction)
        deduped_actions = self._repair_destination_as_source_pick_sequence(deduped_actions, user_instruction)
        deduped_actions = self._repair_low_priority_source_pick_sequence(deduped_actions, user_instruction)
        deduped_actions = self._repair_failed_source_pick_sequence(deduped_actions, user_instruction)
        deduped_actions = self._repair_pre_pick_source_probe_sequence(deduped_actions, user_instruction)
        deduped_actions = self._repair_pickup_source_navigation(deduped_actions, user_instruction)
        deduped_actions = self._repair_empty_hand_leading_place_plan(deduped_actions, user_instruction)
        deduped_actions = self._repair_place_before_pick_plan(deduped_actions, user_instruction)
        deduped_actions = self._repair_navigation_only_search_plan(deduped_actions, user_instruction)
        deduped_actions = self._repair_search_plan_without_pick(deduped_actions, user_instruction)


        deduped_actions = self._repair_multi_target_pick_actions(deduped_actions, user_instruction)
        deduped_actions = self._repair_failed_source_pick_sequence(deduped_actions, user_instruction)
        deduped_actions = self._repair_multi_target_pick_actions(deduped_actions, user_instruction)
        deduped_actions = self._repair_open_close_navigation_sequence(deduped_actions)
        deduped_actions = self._repair_container_source_open_sequence(deduped_actions)
        deduped_actions = self._repair_place_navigation_sequence(deduped_actions)
        return deduped_actions

    def _is_exploration_phase(self, predicted_state: str) -> bool:
        if not predicted_state:
            return False
        return "exploration phase" in predicted_state.lower()

    def should_skip_worldmind(self, action_index: int = 0, last_action_success: Any = 1) -> bool:
        if not bool(last_action_success):
            return False

        if action_index < len(self.current_predicted_states):
            predicted_state = self.current_predicted_states[action_index]
        else:
            predicted_state = self.current_predicted_state

        if predicted_state and self._is_exploration_phase(predicted_state):
            logger.info("WorldMind: Exploration phase detected, skipping WorldMind cycle")
            return True
        return False

    def _json_to_action_with_prediction(self, output_text: str, json_key: str = 'executable_plan') -> Tuple[Any, List[str]]:
        try:
            output_text = fix_json_worldmind(output_text)

            first_brace = output_text.find('{')
            if first_brace == -1:
                logger.warning("No opening brace found in output")
                return -1, []

            brace_count = 0
            last_brace = -1
            for i in range(first_brace, len(output_text)):
                if output_text[i] == '{':
                    brace_count += 1
                elif output_text[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        last_brace = i
                        break

            if last_brace == -1:
                logger.warning("No matching closing brace found")
                return -1, []

            json_text = output_text[first_brace:last_brace+1]

            try:
                json_object = json.loads(json_text)
            except json.JSONDecodeError as e:
                logger.warning(f"First parse attempt failed: {e}")
                return -1, []

            executable_plan = json_object.get(json_key, [])

            if not executable_plan:
                logger.warning("Empty executable plan in output")
                return -2, []

            action_key = self.base_planner.action_key
            actions = []
            predicted_states = []

            for action_item in executable_plan:
                if not isinstance(action_item, dict):
                    continue

                if hasattr(self.base_planner, 'resolve_action_item'):
                    action_id = self.base_planner.resolve_action_item(action_item)
                else:
                    action_id = action_item.get(action_key)
                if action_id is not None:
                    actions.append(action_id)

                predicted_state = action_item.get('predicted_state', '')
                predicted_states.append(predicted_state)

            if not actions:
                return -2, predicted_states

            deduped_actions = []
            deduped_predicted_states = []
            for idx, action in enumerate(actions):
                predicted_state = predicted_states[idx] if idx < len(predicted_states) else ""
                if deduped_actions and action == deduped_actions[-1]:
                    logger.info(f"WorldMind: removed consecutive duplicate action id {action}")
                    continue
                deduped_actions.append(action)
                deduped_predicted_states.append(predicted_state)

            return deduped_actions, deduped_predicted_states

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to decode JSON in WorldMind wrapper: {e}")
            return -1, []
        except Exception as e:
            logger.warning(f"Unexpected error parsing output in WorldMind wrapper: {e}")
            return -1, []

    def act(self, observation, user_instruction: str) -> Tuple[Any, str]:

        self.current_human_instruction = user_instruction
        self.set_human_instruction(user_instruction)


        if isinstance(observation, str):
            self.current_obs_path = observation
        elif isinstance(observation, dict):
            self.current_obs_path = observation.get('head_rgb', None)


        worldmind_prompt = self._build_worldmind_prompt(user_instruction)
        self.current_agent_input_prompt = worldmind_prompt


        print_agent_input(worldmind_prompt, self.detailed_output)



        original_process_prompt = self.base_planner.process_prompt
        self.base_planner.process_prompt = lambda user_instruction, prev_act_feedback=[]: worldmind_prompt

        try:

            if self.enable_multi_simulation and self.multi_simulation_manager and self.multi_simulation_manager.enable:
                actions, output_text = self._act_with_multi_simulation(observation, user_instruction)
            else:

                actions, output_text = self.base_planner.act(observation, user_instruction)
        finally:

            self.base_planner.process_prompt = original_process_prompt

        actions = self._normalize_executable_actions(actions, user_instruction)


        self.current_agent_output = output_text


        print_agent(output_text, self.detailed_output)


        self.current_visual_state_description = self._extract_visual_state_description(output_text)


        _, predicted_states = self._json_to_action_with_prediction(output_text)
        self.current_predicted_states = predicted_states


        if isinstance(actions, list) and len(actions) > 0:
            self.current_action_id = actions[0]
            self.current_predicted_state = predicted_states[0] if predicted_states else None
            if 0 <= self.current_action_id < len(self.base_planner.actions):
                self.current_action_description = self.base_planner.actions[self.current_action_id]
            else:
                self.current_action_description = f"action_{self.current_action_id}"
        elif isinstance(actions, int):
            if actions >= 0:
                self.current_action_id = actions
                self.current_predicted_state = predicted_states[0] if predicted_states else None
                if self.current_action_id < len(self.base_planner.actions):
                    self.current_action_description = self.base_planner.actions[self.current_action_id]
                else:
                    self.current_action_description = f"action_{self.current_action_id}"
            else:
                self.current_action_id = actions
                self.current_predicted_state = None
                self.current_action_description = "invalid" if actions == -1 else "empty_plan"

        if self.current_predicted_state:
            self.worldmind_stats["total_predictions"] += 1


        if self.enable_location_knowledge and self.location_knowledge_extractor:
            self.extract_location_knowledge_after_planning()

        return actions, output_text

    def _act_with_multi_simulation(self, observation, user_instruction: str) -> Tuple[Any, str]:
        candidates = []

        for i in range(self.multi_simulation_n_candidates):
            logger.debug(f"Generating candidate {i+1}/{self.multi_simulation_n_candidates}")

            saved_feedback = self.base_planner.episode_act_feedback.copy()
            saved_messages = self.base_planner.episode_messages.copy()

            action, reasoning = self.base_planner.act(observation, user_instruction)

            candidates.append({
                'index': i,
                'action': action,
                'reasoning': reasoning
            })

            self.base_planner.episode_act_feedback = saved_feedback
            self.base_planner.episode_messages = saved_messages


        action_history = ""
        for j, fb in enumerate(self.base_planner.episode_act_feedback):
            if len(fb) >= 2:
                action_id = fb[0]
                if isinstance(action_id, int) and 0 <= action_id < len(self.base_planner.actions):
                    action_name = self.base_planner.actions[action_id]
                else:
                    action_name = str(action_id)
                action_history += f"Step {j}: {action_name}\n"

        selected_idx, selected_candidate = self.multi_simulation_manager.select_best_candidate(
            candidates=candidates,
            instruction=user_instruction,
            available_actions=self.base_planner.available_action_str,
            action_history=action_history,
            state_description=self.current_visual_state_description or "",
            actions_list=self.base_planner.actions
        )

        logger.info(f"Multi-simulation selected candidate {selected_idx}")

        return selected_candidate['action'], selected_candidate['reasoning']

    def _build_worldmind_prompt(self, user_instruction: str) -> str:
        user_instruction = user_instruction.rstrip('.')
        planner = self.base_planner


        if planner.n_shot >= 1:
            prompt = planner.system_prompt.format(
                len(planner.actions)-1,
                planner.available_action_str,
                '\n\n'.join([f'## Task Execution Example {i}: \n {x}' for i, x in enumerate(planner.examples[:planner.n_shot])])
            )
        else:
            prompt = planner.system_prompt.format(len(planner.actions)-1, planner.available_action_str, '')


        relevant_experiences = []
        if self.enable_success_experience:
            relevant_experiences = self.success_experience_manager.retrieve_experiences(user_instruction)
            logger.info(f"[DEBUG] Retrieved {len(relevant_experiences)} success experiences for instruction: {user_instruction[:50]}...")
            if relevant_experiences:
                logger.info(f"[DEBUG] First experience instruction: {relevant_experiences[0].get('instruction', 'N/A')[:50]}...")
            else:
                logger.info(f"[DEBUG] No experiences retrieved. Manager has {len(self.success_experience_manager.experiences)} experiences, all_experiences: {len(self.success_experience_manager.all_experiences)}")




        relevant_knowledge = []
        if self.enable_world_knowledge:
            if self._cached_world_knowledge is None:

                self._cached_world_knowledge = self.world_knowledge_manager.retrieve_knowledge(user_instruction)
                logger.info(f"[DEBUG] Retrieved and cached {len(self._cached_world_knowledge)} world knowledge entries at episode start")
            relevant_knowledge = self._cached_world_knowledge
            logger.debug(f"[DEBUG] Using cached {len(relevant_knowledge)} world knowledge entries")




        if self.enable_location_knowledge_accumulation:
            location_knowledge = self.get_accumulated_location_knowledge()
            if location_knowledge:
                prompt += f'\n\n## Discovered Location Information (from previous navigation actions):\n{location_knowledge}'

        prompt += f'\n\n## Now the human instruction is: {user_instruction}.'
        common_sense_hints = self._format_common_sense_target_hints(user_instruction)
        if common_sense_hints:
            prompt += f'\n\n{common_sense_hints}'


        if self.enable_experience_refine and self.experience_refiner and (relevant_experiences or relevant_knowledge):

            refine_result = self.experience_refiner.refine_for_task(
                current_instruction=user_instruction,
                success_experiences=relevant_experiences,
                world_knowledge=relevant_knowledge
            )


            refined_prompt = self.experience_refiner.format_for_prompt(refine_result)
            if refined_prompt:
                prompt += f'\n\n{refined_prompt}'
                logger.info(f"Experience refiner: added consolidated experience and initial plan to prompt")
        else:

            if relevant_experiences:
                experience_prompt = self.success_experience_manager.format_experiences_for_prompt(relevant_experiences)
                prompt += f'\n\n{experience_prompt}'

            if relevant_knowledge:
                knowledge_prompt = self.world_knowledge_manager.format_knowledge_for_prompt(relevant_knowledge)
                prompt += f'\n\n{knowledge_prompt}'


        if len(planner.episode_act_feedback) > 0:
            prompt += '\n The action history:\n'
            for i, action_feedback in enumerate(planner.episode_act_feedback):
                prompt += '\nStep {}, action id {}, {}, env feedback: {}'.format(
                    i, action_feedback[0], planner.actions[action_feedback[0]], action_feedback[1]
                )

            feedback_search_hints = self._format_feedback_search_hints()
            if feedback_search_hints:
                prompt += f"\n\n{feedback_search_hints}"

            prompt += f"\n\n Considering the above interaction history, to achieve the human instruction: '{user_instruction}', you are supposed to output in json. You need to summarize interaction history {'and environment feedback ' if planner.use_feedback else ''}and reason why the last action or plan failed and did not finish the task, output your new plan to achieve the goal from current state. At the end, output the executable plan with action ids(0 ~ {len(planner.actions)-1}) from the available actions."
        else:
            prompt += f" You are supposed to output in json. You need to output your reasoning steps and plan. At the end, output the action id (0 ~ {len(planner.actions)-1}) from the available actions to execute."

        return prompt

    def get_state_description(self, summarizer_state: str = None) -> str:
        if self.use_state_summarizer:
            return summarizer_state or ""
        else:
            return self.current_visual_state_description or summarizer_state or ""

    def summarize_states(
        self,
        before_image_path: str,
        after_image_path: str,
        action_description: str = None
    ) -> Dict[str, str]:

        if self.state_summarizer is None:
            logger.warning("State summarizer not configured, returning empty state summaries")
            return {'state_before_action': '', 'state_after_action': ''}

        action_desc = action_description or self.current_action_description or "unknown action"

        print_summarizer_input("(image)", "(image)", action_desc, self.detailed_output)

        result = self.state_summarizer.summarize_states(
            action_description=action_desc,
            before_image_path=before_image_path,
            after_image_path=after_image_path
        )

        print_summarizer_output(
            result.get('state_before_action', ''),
            result.get('state_after_action', ''),
            self.detailed_output
        )

        return result

    def discriminate_prediction(
        self,
        actual_state_summary: str,
        action_index: int = 0,
        action_id: int = None,
        action_description: str = None
    ) -> Dict:
        if not self.enable_discrimination or self.discriminator is None:
            return {"match": True, "reason": "Discrimination disabled"}

        if action_index < len(self.current_predicted_states):
            predicted_state = self.current_predicted_states[action_index]
        else:
            predicted_state = self.current_predicted_state

        if not predicted_state:
            return {"match": True, "reason": "No prediction available"}

        actual_action_id = action_id if action_id is not None else self.current_action_id
        action_desc = action_description if action_description is not None else (self.current_action_description or "")

        print_discriminator_input(
            predicted_state,
            actual_state_summary,
            action_desc,
            self.detailed_output
        )

        result = self.discriminator.discriminate(
            predicted_state=predicted_state,
            actual_state_summary=actual_state_summary,
            action_description=action_desc
        )

        print_discriminator_output(
            result["match"],
            result["reason"],
            self.detailed_output
        )

        if result["match"]:
            self.worldmind_stats["discrimination_matches"] += 1
        else:
            self.worldmind_stats["discrimination_mismatches"] += 1

        self.discrimination_history.append({
            "action_id": actual_action_id,
            "action_description": action_desc,
            "predicted_state": predicted_state,
            "actual_state_summary": actual_state_summary,
            "discrimination_result": result
        })

        return result

    def describe_feedback_state(
        self,
        env_feedback: str = "",
        last_action_success: Any = None
    ) -> str:
        parts = []
        if env_feedback:
            parts.append(f"Environment feedback: {env_feedback}")
        if last_action_success is not None:
            parts.append(f"Last action success: {bool(last_action_success)}")
        return "\n".join(parts) if parts else "No structured feedback available."

    def discriminate_feedback(
        self,
        env_feedback: str = "",
        action_index: int = 0,
        action_id: int = None,
        action_description: str = None,
        last_action_success: Any = 1
    ) -> Dict:
        if not self.enable_discrimination:
            return {"match": True, "reason": "Discrimination disabled"}

        if action_index < len(self.current_predicted_states):
            predicted_state = self.current_predicted_states[action_index]
        else:
            predicted_state = self.current_predicted_state

        if not predicted_state:
            return {"match": True, "reason": "No prediction available"}

        actual_action_id = action_id if action_id is not None else self.current_action_id
        action_desc = action_description if action_description is not None else (self.current_action_description or "")
        actual_state_summary = self.describe_feedback_state(env_feedback, last_action_success)

        success = bool(last_action_success)
        if not success:
            result = {
                "match": False,
                "reason": env_feedback or "Action failed according to environment feedback.",
                "predicted_state": predicted_state,
                "fast_feedback_mode": True
            }
            self.worldmind_stats["discrimination_mismatches"] += 1
        elif self._is_exploration_phase(predicted_state):
            result = {
                "match": True,
                "reason": "Exploration phase prediction skipped after successful action.",
                "predicted_state": predicted_state,
                "fast_feedback_mode": True
            }
            self.worldmind_stats["discrimination_matches"] += 1
        else:
            result = {
                "match": True,
                "reason": "Action executed successfully according to environment feedback.",
                "predicted_state": predicted_state,
                "fast_feedback_mode": True
            }
            self.worldmind_stats["discrimination_matches"] += 1

        self.discrimination_history.append({
            "action_id": actual_action_id,
            "action_description": action_desc,
            "predicted_state": predicted_state,
            "actual_state_summary": actual_state_summary,
            "discrimination_result": result,
            "fast_feedback_mode": True
        })

        return result

    def trigger_reflection(
        self,
        discrimination_result: Dict,
        state_before_action: str,
        state_after_action: str,
        observation_path: Optional[str] = None,
        env_feedback: str = "",
        action_id: int = None,
        action_description: str = None,
        human_instruction: str = None
    ) -> Optional[Dict]:
        actual_action_id = action_id if action_id is not None else self.current_action_id
        actual_action_description = action_description if action_description is not None else (self.current_action_description or "unknown action")
        predicted_state = discrimination_result.get("predicted_state") or self.current_predicted_state or ""
        instruction = human_instruction or self.current_human_instruction or "Not specified."

        actual_state_before = self.get_state_description(state_before_action)
        actual_state_after = self.get_state_description(state_after_action)

        self.current_worldmind_feedback = {
            "observation_before": actual_state_before,
            "predicted_state": predicted_state,
            "env_feedback": env_feedback
        }


        if discrimination_result.get("match", True):
            self.worldmind_stats["success_feedbacks"] += 1
            logger.info(f"WorldMind: Action succeeded: {actual_action_description}")


            step = len(self.experience_trajectory)
            self.experience_trajectory.add_entry(
                step=step,
                action_id=actual_action_id,
                action_description=actual_action_description,
                observation_before=actual_state_before,
                predicted_state=predicted_state,
                env_feedback=env_feedback,
                agent_input_prompt=self.current_agent_input_prompt,
                agent_output=self.current_agent_output,

                discriminator_called=True,
                discrimination_result=discrimination_result,
                has_prediction_error=False,
                reflector_called=False,
                reflection_result=None,
                experience_entries=[]
            )

            if self.use_experience_trajectory:
                print_trajectory_entry(
                    step=step,
                    observation=actual_state_before,
                    action=actual_action_description,
                    predicted_state=predicted_state,
                    reflexion=None,
                    next_observation=actual_state_after,
                    detailed_output=self.detailed_output
                )
            return None



        trajectory_parts = []
        for fb in self.base_planner.episode_act_feedback:
            if len(fb) >= 2:
                trajectory_parts.append(f"Action: {self.base_planner.actions[fb[0]]}, Feedback: {fb[1]}")
        experience_trajectory_str = "\n".join(trajectory_parts) if trajectory_parts else "No previous experience."


        if self.reflector is None:
            logger.warning("Reflector not configured, skipping reflection")
            return None

        print_reflector_input(
            actual_action_description,
            predicted_state,
            actual_state_before,
            actual_state_after,
            env_feedback,
            self.detailed_output
        )


        location_knowledge_str = self.get_accumulated_location_knowledge() if hasattr(self, 'get_accumulated_location_knowledge') else ""

        reflection_result = self.reflector.reflect(
            action_description=actual_action_description,
            predicted_state=predicted_state,
            state_before_action=actual_state_before,
            state_after_action=actual_state_after,
            current_env_feedback=env_feedback,
            action_id=actual_action_id,
            experience_trajectory=experience_trajectory_str,
            human_instruction=instruction,
            location_knowledge=location_knowledge_str
        )

        experience_entries = self._filter_reflection_entries_for_clean_evidence(
            reflection_result.get("experience_entry", [])
        )
        reflection_result["experience_entry"] = experience_entries

        print_reflector_output(str(experience_entries), self.detailed_output)


        if experience_entries and len(experience_entries) > 0:


            self.world_knowledge_manager.add_entries(experience_entries, instruction)
            logger.info(f"Added {len(experience_entries)} world knowledge entries")

        self.worldmind_stats["reflections_triggered"] += 1

        self.reflection_history.append({
            "action_id": actual_action_id,
            "action_description": actual_action_description,
            "experience_entries": experience_entries,
            "human_instruction": instruction
        })


        step = len(self.experience_trajectory)
        self.experience_trajectory.add_entry(
            step=step,
            action_id=actual_action_id,
            action_description=actual_action_description,
            observation_before=actual_state_before,
            predicted_state=predicted_state,
            env_feedback=env_feedback,
            agent_input_prompt=self.current_agent_input_prompt,
            agent_output=self.current_agent_output,

            discriminator_called=True,
            discrimination_result=discrimination_result,
            has_prediction_error=True,
            reflector_called=True,
            reflection_result=reflection_result,
            experience_entries=experience_entries
        )

        if self.use_experience_trajectory:
            print_trajectory_entry(
                step=step,
                observation=actual_state_before,
                action=actual_action_description,
                predicted_state=predicted_state,
                reflexion=str(experience_entries),
                next_observation=actual_state_after,
                detailed_output=self.detailed_output
            )

        logger.info(f"WorldMind Reflection triggered for action: {actual_action_description}")

        return reflection_result


    def save_knowledge(self, episode_num: int):
        self.world_knowledge_manager.save_episode_knowledge(episode_num)

    def update_info(
        self,
        info: dict,
        discrimination_result: Optional[Dict] = None,
        reflection_result: Optional[Dict] = None
    ):

        self.base_planner.update_info(info)

    def get_trajectory_for_save(self) -> List[Dict]:
        return self.experience_trajectory.get_trajectory_for_save()

    def get_worldmind_stats(self) -> Dict:
        stats = self.worldmind_stats.copy()
        stats["worldmind_first_action_only"] = self.worldmind_first_action_only

        if self.discriminator:
            stats["discriminator_stats"] = self.discriminator.get_statistics()
        if self.reflector:
            stats["reflector_stats"] = self.reflector.get_statistics()
        if self.state_summarizer:
            stats["summarizer_stats"] = self.state_summarizer.get_statistics()

        stats["discrimination_history"] = self.discrimination_history
        stats["reflection_history"] = self.reflection_history
        stats["experience_trajectory_length"] = len(self.experience_trajectory)
        stats["success_experience_count"] = len(self.success_experience_manager.experiences)
        stats["world_knowledge_entry_count"] = len(self.world_knowledge_manager.knowledge_entries)
        stats["world_knowledge_total_count"] = len(self.world_knowledge_manager.get_all_knowledge_flat())

        total = stats["discrimination_matches"] + stats["discrimination_mismatches"]
        stats["match_rate"] = stats["discrimination_matches"] / total if total > 0 else 0

        if self.multi_simulation_manager:
            stats["multi_simulation_stats"] = self.multi_simulation_manager.get_stats()

        return stats


    @property
    def planner_steps(self):
        return self.base_planner.planner_steps

    @property
    def output_json_error(self):
        return self.base_planner.output_json_error

    @property
    def actions(self):
        return self.base_planner.actions

    def set_actions(self, actions):
        self.base_planner.actions = actions
        self.success_experience_manager.set_actions(actions)

    @property
    def history(self):
        return self.base_planner.history

    @property
    def language_only(self):
        return self.base_planner.language_only
