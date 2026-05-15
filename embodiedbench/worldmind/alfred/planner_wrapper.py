
import os
import json
import re
from typing import Dict, List, Optional, Tuple, Any

from embodiedbench.planner.vlm_planner import VLMPlanner
from embodiedbench.worldmind.alfred.discriminator import WorldMindDiscriminator
from embodiedbench.worldmind.alfred.reflector import WorldMindReflector
from embodiedbench.worldmind.alfred.state_summarizer import WorldMindStateSummarizer, LocationKnowledgeExtractor
from embodiedbench.worldmind.alfred.knowledge_manager import (
    SuccessExperienceManager,
    WorldKnowledgeManager,
    create_success_experience_manager,
    create_world_knowledge_manager
)
from embodiedbench.worldmind.alfred.color_print import (
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
from embodiedbench.worldmind.alfred.prompts import (
    WorldMind_TEMPLATE,
    WorldMind_TEMPLATE_LANG,
    get_worldmind_examples,
    format_worldmind_examples,
    get_instruction_workflow_guidance,
    fix_json_worldmind,
    parse_json_worldmind
)
from embodiedbench.worldmind.alfred.multi_simulation import MultiSimulationManager
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

        naive_memory_mode: bool = False,
        naive_memory_with_reflect: bool = False,
        use_fast_feedback_discrimination: bool = True,
        enable_visual_observation_summary: bool = True,

        memory_poison_ratio: float = 0.0
    ):
        self.base_planner = base_planner
        self.enable_discrimination = enable_discrimination
        self.use_experience_trajectory = use_experience_trajectory
        self.use_state_summarizer = use_state_summarizer
        self.detailed_output = detailed_output
        self.worldmind_first_action_only = worldmind_first_action_only


        self.enable_success_experience = enable_success_experience
        self.success_experience_top_k = success_experience_top_k
        self.enable_world_knowledge = enable_world_knowledge
        self.world_knowledge_top_k = world_knowledge_top_k
        self.enable_experience_refine = enable_experience_refine


        self.cross_eval_set_retrieval = cross_eval_set_retrieval
        self.all_eval_sets = all_eval_sets or []


        self.enable_experience_transfer = enable_experience_transfer
        self.experience_transfer_path = experience_transfer_path


        self.current_human_instruction: Optional[str] = None


        self.enable_location_knowledge = enable_location_knowledge
        self.enable_location_knowledge_accumulation = enable_location_knowledge_accumulation


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


        self.naive_memory_mode = naive_memory_mode
        self.naive_memory_with_reflect = naive_memory_with_reflect
        self.use_fast_feedback_discrimination = use_fast_feedback_discrimination
        self.enable_visual_observation_summary = enable_visual_observation_summary

        self.memory_poison_ratio = memory_poison_ratio



        self.success_experience_manager = SuccessExperienceManager(
            top_k=success_experience_top_k,
            extractor_model=None if naive_memory_mode else (actual_reflector_model if not enable_experience_transfer else None),
            model_type=base_planner.model_type if hasattr(base_planner, 'model_type') else 'remote',
            cross_eval_set_retrieval=cross_eval_set_retrieval,
            all_eval_sets=all_eval_sets or [],
            enable_experience_transfer=enable_experience_transfer,
            experience_transfer_path=experience_transfer_path,
            naive_memory_mode=naive_memory_mode
        )


        self.world_knowledge_manager = WorldKnowledgeManager(
            top_k=world_knowledge_top_k,
            cross_eval_set_retrieval=cross_eval_set_retrieval,
            all_eval_sets=all_eval_sets or [],
            enable_experience_transfer=enable_experience_transfer,
            experience_transfer_path=experience_transfer_path,
            memory_poison_ratio=memory_poison_ratio
        )



        self.experience_refiner = None
        if enable_experience_refine:
            from embodiedbench.worldmind.alfred.experience_refiner import ExperienceRefiner

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


        self.current_visual_observation_summary: str = ""
        self.current_visible_objects: List[Dict] = []


        self.current_eval_set: str = "base"


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


        self._cached_world_knowledge: Optional[List[Dict]] = None

        logger.info(f"WorldMind Planner Wrapper initialized (success_experience={enable_success_experience}, world_knowledge={enable_world_knowledge}, experience_refine={enable_experience_refine}, experience_transfer={enable_experience_transfer})")

        if enable_experience_transfer:
            logger.info(f"[EXPERIENCE TRANSFER MODE] Loading experiences from: {experience_transfer_path}")

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

    def set_visual_observation_summary(
        self,
        summary_text: str = "",
        visible_objects: Optional[List[Dict]] = None
    ):
        self.current_visual_observation_summary = summary_text or ""
        self.current_visible_objects = visible_objects or []



    def set_human_instruction(self, instruction: str):
        self.current_human_instruction = instruction

        self.success_experience_manager.set_instruction(instruction)

        self.world_knowledge_manager.set_instruction(instruction)

    def update_trajectory_info(self, info: Dict):

        self.success_experience_manager.update_trajectory(info)

        logger.debug(f"Trajectory updated with action_id={info.get('action_id')}, "
                    f"task_progress={info.get('task_progress')}")

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
        self.current_visual_observation_summary = ""
        self.current_visible_objects = []
        self.current_human_instruction = None


        self.experience_trajectory.reset()


        self.success_experience_manager.reset_trajectory()
        self.success_experience_manager.set_actions(self.base_planner.actions)


        self.world_knowledge_manager.reset_episode()


        self._cached_world_knowledge = None


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

    def _is_exploration_phase(self, predicted_state: str) -> bool:
        if not predicted_state:
            return False
        normalized = predicted_state.lower()
        exploration_markers = (
            "exploration phase",
            "exploring environment",
            "prediction skipped",
            "target not visible",
            "destination not visible",
            "object not visible",
            "location unknown",
        )
        return any(marker in normalized for marker in exploration_markers)

    def _find_action_id_by_name(self, action_name: str) -> Optional[int]:
        target = action_name.strip().lower()
        for idx, candidate in enumerate(self.base_planner.actions):
            if str(candidate).strip().lower() == target:
                return idx
        return None

    def _action_target(self, action_id: int, prefix: str) -> Optional[str]:
        if isinstance(action_id, str):
            action_name = action_id.strip()
        elif isinstance(action_id, int) and 0 <= action_id < len(self.base_planner.actions):
            action_name = str(self.base_planner.actions[action_id]).strip()
        else:
            return None
        if not action_name.lower().startswith(prefix.lower()):
            return None
        return action_name[len(prefix):].strip()

    @staticmethod
    def _canonical_text(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", value.lower())

    def _instruction_mentions_object(self, object_name: str) -> bool:
        if not object_name:
            return False
        instruction = (self.current_human_instruction or "").lower().replace("-", " ")
        instruction_key = self._canonical_text(instruction)
        object_key = self._canonical_text(self._base_object_name(object_name))
        if object_key and object_key in instruction_key:
            return True
        base_name = self._base_object_name(object_name)
        if base_name == "SoapBar" and re.search(r"\b(?:soap\s*bottles?|pump)\b", instruction):
            return False

        patterns = {
            "SoapBar": [
                r"\bsoaps?\b",
                r"\bsoap\s*bars?\b",
                r"\bbars?\s+of\s+soap\b",
                r"\bbar\s+soap\b",
            ],
            "CD": [
                r"\bcds?\b",
                r"\bcompact\s+discs?\b",
                r"\bmusic\s+discs?\b",
            ],
            "KeyChain": [
                r"\bkey\s*chains?\b",
                r"\bkey\s*rings?\b",
                r"\bsets?\s+of\s+keys\b",
                r"\bkeys?\b",
                r"\bjangling\s+items?\b",
            ],
            "DiningTable": [
                r"\bwooden\s+tables?\b",
                r"\bdining\s+tables?\b",
                r"\bkitchen\s+tables?\b",
                r"\btables?\b",
            ],
            "Desk": [
                r"\bdesks?\b",
                r"\bstudy(?:ing)?\s+(?:table|furniture|station)s?\b",
            ],
            "Book": [
                r"\bbooks?\b",
                r"\bbound\s+narratives?\b",
                r"\bread(?:ing)?\s+materials?\b",
                r"\bnovels?\b",
            ],
            "Newspaper": [
                r"\bnewspapers?\b",
                r"\bpapers?\b",
                r"\bread(?:ing)?\s+materials?\b",
            ],
            "CellPhone": [
                r"\bcell\s*phones?\b",
                r"\bphones?\b",
                r"\bcommunication\s+devices?\b",
            ],
            "RemoteControl": [
                r"\bremotes?\b",
                r"\bremote\s*controls?\b",
                r"\bdevices?\s+for\s+changing\s+channels?\b",
                r"\bchannel\s+changers?\b",
            ],
            "DeskLamp": [
                r"\bdesk\s*lamps?\b",
                r"\blamps?\b",
            ],
            "FloorLamp": [
                r"\bfloor\s*lamps?\b",
                r"\blamps?\b",
            ],
            "ButterKnife": [
                r"\bbutter\s*knives?\b",
                r"\bcutting\s+tools?\s+for\s+butter\b",
            ],
            "Spatula": [
                r"\bspatulas?\b",
                r"\bflipping\s+tools?\b",
                r"\bsoup\s+serving\s+tools?\b",
            ],
            "Cloth": [
                r"\bcloths?\b",
                r"\brags?\b",
                r"\bred\s+rags?\b",
                r"\bcleaning\s+rags?\b",
                r"\bcleaning\s+cloths?\b",
            ],
            "HandTowel": [
                r"\bhand\s*towels?\b",
                r"\btowels?\b",
            ],
            "Ladle": [
                r"\bladles?\b",
                r"\bsoup\s+serving\s+tools?\b",
                r"\bserving\s+tools?\b",
            ],
            "Lettuce": [
                r"\blettuces?\b",
                r"\bgreen\s+leaves?\b",
                r"\bleafy\s+(?:food|vegetable)s?\b",
            ],
            "Apple": [
                r"\bapples?\b",
                r"\bsnow\s+white\s+fruits?\b",
                r"\bred\s+crisp\s+fruits?\b",
            ],
            "Tomato": [
                r"\btomatoes?\b",
                r"\btomatos?\b",
                r"\bsauces?\s+and\s+salads?\s+fruits?\b",
            ],
            "Cup": [
                r"\bcups?\b",
                r"\b(?:glass|glasses)\b",
                r"\bdrink(?:ing)?\s+(?:vessel|container)s?\b",
            ],
            "Mug": [
                r"\bmugs?\b",
                r"\b(?:glass|glasses)\b",
                r"\bdrink(?:ing)?\s+(?:vessel|container)s?\b",
            ],
            "Glassbottle": [
                r"\bglass\s*bottles?\b",
                r"\b(?:glass|glasses)\b",
                r"\bjars?\b",
                r"\bglass\s*jars?\b",
            ],
            "Vase": [
                r"\bvases?\b",
                r"\bjars?\b",
                r"\bglass\s*jars?\b",
                r"\bvessels?\b",
            ],
            "Plate": [
                r"\bplates?\b",
                r"\beating\s+surfaces?\b",
            ],
            "Bowl": [
                r"\bbowls?\b",
                r"\bglass\s+bowls?\b",
                r"\bround\s+eating\s+(?:vessel|container)s?\b",
            ],
            "Bed": [
                r"\bbeds?\b",
                r"\bplaces?\s+where\s+you\s+sleep\b",
            ],
        }
        return any(re.search(pattern, instruction) for pattern in patterns.get(base_name, []))

    @staticmethod
    def _task_progress_from_feedback(feedback: str) -> Optional[float]:
        match = re.search(r"\[Task Progress\]\s*([0-9.]+)", str(feedback or ""))
        if not match:
            return None
        try:
            return float(match.group(1))
        except ValueError:
            return None

    def _latest_task_progress(self) -> float:
        progress = 0.0
        for entry in getattr(self.base_planner, "episode_act_feedback", []) or []:
            if len(entry) < 2:
                continue
            parsed = self._task_progress_from_feedback(str(entry[1]))
            if parsed is not None:
                progress = parsed
        return progress

    @staticmethod
    def _base_object_name(value: str) -> str:
        base_name = re.sub(r'_\d+$', '', str(value or '').strip())
        canonical_names = {
            "alarmclock": "AlarmClock",
            "baseballbat": "BaseballBat",
            "butterknife": "ButterKnife",
            "cellphone": "CellPhone",
            "coffeemachine": "CoffeeMachine",
            "coffeetable": "CoffeeTable",
            "countertop": "CounterTop",
            "creditcard": "CreditCard",
            "desklamp": "DeskLamp",
            "dishsponge": "DishSponge",
            "floorlamp": "FloorLamp",
            "garbagecan": "GarbageCan",
            "glassbottle": "Glassbottle",
            "handtowel": "HandTowel",
            "keychain": "KeyChain",
            "laptop": "Laptop",
            "microwave": "Microwave",
            "newspaper": "Newspaper",
            "papertowelroll": "PaperTowelRoll",
            "pepperShaker".lower(): "PepperShaker",
            "remotecontrol": "RemoteControl",
            "sidetable": "SideTable",
            "soapbar": "SoapBar",
            "soapbottle": "SoapBottle",
            "spraybottle": "SprayBottle",
            "stoveburner": "StoveBurner",
            "tennisracket": "TennisRacket",
            "tissuebox": "TissueBox",
            "toiletpaper": "ToiletPaper",
            "toiletpaperhanger": "ToiletPaperHanger",
            "wateringcan": "WateringCan",
            "winebottle": "WineBottle",
        }
        lookup_key = re.sub(r"[^a-z0-9]+", "", base_name.lower())
        return canonical_names.get(lookup_key, base_name)

    @staticmethod
    def _normalize_color_word(value: str) -> str:
        value = str(value or '').lower()
        if value == 'grey':
            return 'gray'
        return value

    def _instruction_visual_targets(self) -> Dict[str, set]:
        instruction = (self.current_human_instruction or "").lower().replace('-', ' ')
        tokens = re.findall(r"[a-z]+", instruction)
        color_words = {
            'red', 'green', 'blue', 'yellow', 'white', 'black', 'brown',
            'gray', 'grey', 'silver', 'orange', 'purple', 'pink'
        }
        targets: Dict[str, set] = {}

        def add_targets(types: List[str], color: str):
            color = self._normalize_color_word(color)
            if not types:
                targets.setdefault('*', set()).add(color)
                return
            for obj_type in types:
                targets.setdefault(obj_type, set()).add(color)

        for idx, token in enumerate(tokens):
            if token not in color_words:
                continue
            context_tokens = tokens[max(0, idx - 5): idx + 7]
            context = " ".join(context_tokens)
            types: List[str] = []
            if 'spray' in context and 'bottle' in context:
                types = ['SprayBottle']
            elif 'soap' in context and 'bottle' in context:
                types = ['SoapBottle']
            elif 'bottle' in context or 'pump' in context:
                types = ['SprayBottle', 'SoapBottle', 'WineBottle', 'Glassbottle']
            elif 'tissue' in context or 'box' in context:
                types = ['TissueBox', 'Box']
            elif 'lamp' in context:
                types = ['DeskLamp', 'FloorLamp']
            elif 'dresser' in context:
                types = ['Dresser']
            elif 'desk' in context:
                types = ['Desk']
            elif 'table' in context:
                types = ['DiningTable', 'CoffeeTable', 'SideTable', 'Desk']
            elif 'counter' in context:
                types = ['CounterTop']
            elif 'cabinet' in context:
                types = ['Cabinet']
            elif 'glass' in context and 'bowl' in context:
                types = ['Bowl']
            elif 'jar' in context:
                types = ['Vase', 'Glassbottle', 'Cup', 'Mug', 'Pot']
            elif 'vessel' in context or 'vase' in context:
                types = ['Vase', 'Cup', 'Mug', 'Bowl', 'Glassbottle']
            elif 'stand' in context or 'statue' in context:
                types = ['Statue']
            elif 'object' in context or 'thing' in context:
                types = []
            add_targets(types, token)

        return targets

    def _is_visual_descriptor_task(self) -> bool:
        instruction = (self.current_human_instruction or "").lower()
        if self._instruction_visual_targets():
            return True
        return any(term in instruction for term in (
            'wooden', 'metal', 'plastic', 'fabric', 'small ', 'large ', 'pump'
        ))

    def _is_counted_instruction(self) -> bool:
        text = f" {(self.current_human_instruction or '').lower()} "
        color_pair_object = re.search(
            r"\b(?:red|green|blue|yellow|white|black|brown|gray|grey|silver|orange|purple|pink)\s+and\s+"
            r"(?:red|green|blue|yellow|white|black|brown|gray|grey|silver|orange|purple|pink)\s+"
            r"(?:vases?|jars?|bottles?|knives?|objects?|things?)\b",
            text,
        )
        return (
            any(term in text for term in (" two ", " three ", " all ", " every ", " each ", " pair "))
            or bool(re.search(r"\b[2-9]\b", text))
            or bool(color_pair_object)
            or any(
                term in text
                for term in (
                    " bottles ",
                    " spray bottles ",
                    " soap bottles ",
                    " soap bars ",
                    " bars of soap ",
                    " tissue boxes ",
                    " vases ",
                    " jars ",
                )
            )
        )

    def _target_colors_for_type(self, target_type: str, targets: Dict[str, set]) -> set:
        base_type = self._base_object_name(target_type)
        colors = set(targets.get(base_type, set()))
        if not colors:
            colors.update(targets.get('*', set()))
        return colors

    def _entry_color_terms(self, entry: Dict) -> set:
        terms = set()
        for color in entry.get('colors') or []:
            for part in str(color).lower().replace(',', '/').split('/'):
                part = self._normalize_color_word(part.strip())
                if part:
                    terms.add(part)
        if 'silver' in terms:
            terms.add('gray')
        if 'gray' in terms:
            terms.add('grey')
        return terms

    def _entry_matches_any_color(self, entry: Dict, colors: set) -> bool:
        if not colors:
            return True
        entry_terms = self._entry_color_terms(entry)
        expanded = set()
        for color in colors:
            color = self._normalize_color_word(color)
            expanded.add(color)
            if color == 'silver':
                expanded.update({'gray', 'grey'})
            if color == 'gray':
                expanded.update({'grey', 'silver'})
        return bool(entry_terms & expanded)

    def _visible_entries_for_type(self, target_type: str) -> List[Dict]:
        base_type = self._base_object_name(target_type)
        entries = []
        for entry in self.current_visible_objects or []:
            alias_base = self._base_object_name(entry.get('alias', ''))
            object_type = self._base_object_name(entry.get('object_type', ''))
            if base_type in {alias_base, object_type}:
                entries.append(entry)
        return entries

    def _visible_entry_for_alias(self, alias: str) -> Optional[Dict]:
        alias = str(alias or "")
        alias_base = self._base_object_name(alias)
        for entry in self.current_visible_objects or []:
            if entry.get("alias") == alias:
                return entry
        if not alias_base:
            return None
        for entry in self.current_visible_objects or []:
            if self._base_object_name(entry.get("alias", "")) == alias_base:
                return entry
        return None

    def _entry_has_parent_type(self, entry: Dict, parent_type: str) -> bool:
        parent_base = self._base_object_name(parent_type)
        return any(
            self._base_object_name(parent) == parent_base
            for parent in entry.get("parents") or []
        )

    def _visible_target_at_destination(self, target_type: str, destination: str) -> bool:
        if not target_type or not destination:
            return False
        return any(
            self._entry_has_parent_type(entry, destination)
            for entry in self._visible_entries_for_type(target_type)
        )

    def _preferred_feedback_container_find_action(
        self,
        container_name: str,
        avoid_open_generic: bool = False,
    ) -> Tuple[Optional[int], Optional[str]]:
        container_base = self._base_object_name(container_name)
        candidates: List[Tuple[int, str, Dict]] = []
        for entry in self.current_visible_objects or []:
            alias = str(entry.get("alias") or "")
            if not alias or self._base_object_name(alias) != container_base:
                continue
            find_action_id = self._find_action_id_by_name(f"find a {alias}")
            if find_action_id is None:
                continue
            candidates.append((find_action_id, alias, entry))

        if avoid_open_generic:
            for find_action_id, alias, entry in candidates:
                if alias != container_base and not entry.get("is_open"):
                    return find_action_id, alias
            for find_action_id, alias, entry in candidates:
                if alias != container_base:
                    return find_action_id, alias

        generic_find_id = self._find_action_id_by_name(f"find a {container_base}")
        if generic_find_id is not None:
            return generic_find_id, container_base
        if candidates:
            find_action_id, alias, _ = candidates[0]
            return find_action_id, alias
        return None, None

    def _find_action_aliases_for_type(self, target_type: str) -> List[Tuple[int, str]]:
        base_type = self._base_object_name(target_type)
        pattern = re.compile(rf"^find a {re.escape(base_type)}(?:_\d+)?$", re.IGNORECASE)
        aliases = []
        for action_id, action_name in enumerate(self.base_planner.actions):
            action_name = str(action_name).strip()
            if pattern.match(action_name):
                aliases.append((action_id, action_name[len("find a "):]))
        return aliases

    def _tried_find_aliases_for_type(self, target_type: str) -> set:
        tried = set()
        for entry in getattr(self.base_planner, "episode_act_feedback", []) or []:
            if len(entry) < 1:
                continue
            action_id = entry[0]
            target = self._action_target(action_id, "find a ")
            if target and self._base_object_name(target) == self._base_object_name(target_type):
                tried.add(target)
        return tried

    def _select_next_visual_find_action(self, target_type: str) -> Optional[int]:
        candidates = self._find_action_aliases_for_type(target_type)
        if not candidates:
            return None

        tried_aliases = self._tried_find_aliases_for_type(target_type)
        current_aliases = {
            str(entry.get('alias'))
            for entry in self._visible_entries_for_type(target_type)
            if entry.get('alias')
        }

        for action_id, alias in candidates:
            if alias not in tried_aliases and alias not in current_aliases:
                return action_id
        for action_id, alias in candidates:
            if alias not in current_aliases:
                return action_id
        return None

    def _last_successful_find_target(self) -> Optional[str]:
        for entry in reversed(getattr(self.base_planner, "episode_act_feedback", []) or []):
            if len(entry) < 2:
                continue
            action_id, feedback = entry[0], str(entry[1]).lower()
            if "invalid" in feedback or "failed" in feedback:
                continue
            target = self._action_target(action_id, "find a ")
            if target:
                return target
        return None

    def _held_object_type_from_history(self) -> Optional[str]:
        held = None
        for entry in getattr(self.base_planner, "episode_act_feedback", []) or []:
            if len(entry) < 2:
                continue
            action_id, feedback = entry[0], str(entry[1]).lower()
            if "not holding" in feedback or "dropped the object instead" in feedback:
                held = None
                continue
            if "currently holding" in feedback:
                held_target = self._action_target(action_id, "pick up the ")
                if held_target:
                    held = self._base_object_name(held_target)
                    continue
            if (
                "invalid" in feedback
                or "failed" in feedback
                or "not visible" in feedback
                or "cannot " in feedback
            ):
                continue
            picked = self._action_target(action_id, "pick up the ")
            if picked:
                held = self._base_object_name(picked)
                continue
            action_name = (
                str(self.base_planner.actions[action_id]).strip().lower()
                if isinstance(action_id, int) and 0 <= action_id < len(self.base_planner.actions)
                else ""
            )
            if action_name.startswith("put down ") or action_name.startswith("drop "):
                held = None
        return held

    def _destination_find_action_from_instruction(self) -> Tuple[Optional[int], Optional[str]]:
        instruction = f" {(self.current_human_instruction or '').lower()} "
        table_mod = r"(?:(?:round|dining|kitchen|wooden|black|white|red|green|blue|yellow|brown|gray|grey|silver|metal|right|left|front|back|side)\s+){0,4}"
        shelf_mod = r"(?:(?:wooden|black|white|red|green|blue|yellow|brown|gray|grey|silver|metal|top|bottom|middle|upper|lower)\s+){0,4}"
        destination_patterns = [
            (r"\b(?:on top of|onto|on|to)\s+(?:the\s+|a\s+|an\s+)?toilet\b", "Toilet"),
            (r"\b(?:on top of|onto|on|to)\s+(?:the\s+|a\s+|an\s+)?(?:water|toilet)\s+tank\b", "Toilet"),
            (r"\b(?:to|onto|on)\s+(?:the\s+|a\s+|an\s+)?(?:back|front|left|right|top)\s+of\s+(?:the\s+|a\s+|an\s+)?toilet\b", "Toilet"),
            (r"\b(?:in|into|inside|to|onto|on)\s+(?:the\s+|a\s+|an\s+)?(?:(?:bottom|top|middle|lower|upper)\s+)?(?:(?:dresser|desk)\s+)?drawer\b", "Drawer"),
            (r"\b(?:in|into|inside|to|onto|on)\s+(?:the\s+|a\s+|an\s+)?(?:(?:bottom|top|middle|lower|upper)\s+)?drawer\s+of\s+(?:the\s+|a\s+|an\s+)?(?:dresser|desk)\b", "Drawer"),
            (r"\b(?:under|below)\s+(?:the\s+|a\s+|an\s+)?sink\b", "Cabinet"),
            (r"\b(?:in|into|inside|to)\s+(?:the\s+|a\s+|an\s+)?(?:cupboard|cabinet)\s*(?:under|below)\s+(?:the\s+|a\s+|an\s+)?sink\b", "Cabinet"),
            (r"\b(?:in|into|inside|to|under|below)\s+(?:the\s+|a\s+|an\s+)?cupboard\b", "Cabinet"),
            (r"\b(?:in|into|inside|on top of|onto|on|to)\s+(?:the\s+|a\s+|an\s+)?cabinet\b", "Cabinet"),
            (r"\b(?:under|below|beneath)\s+(?:the\s+|a\s+|an\s+)?coffee\s*(?:maker|machine)\b", "CounterTop"),
            (rf"\b(?:on top of|onto|on|to)\s+(?:the\s+|a\s+|an\s+)?{table_mod}counter\s*top\b", "CounterTop"),
            (rf"\b(?:on top of|onto|on|to)\s+(?:the\s+|a\s+|an\s+)?{table_mod}counter\b", "CounterTop"),
            (rf"\b(?:on top of|onto|on|to|return\s+it\s+to|return\s+them\s+to)\s+(?:the\s+|a\s+|an\s+)?{table_mod}table\b", "DiningTable"),
            (r"\b(?:on top of|onto|on|to)\s+(?:the\s+|a\s+|an\s+)?coffee\s+table\b", "CoffeeTable"),
            (r"\b(?:on top of|onto|on|to)\s+(?:the\s+|a\s+|an\s+)?side\s+table\b", "SideTable"),
            (rf"\b(?:on top of|onto|on|to)\s+(?:the\s+|a\s+|an\s+)?{shelf_mod}desk\b", "Desk"),
            (rf"\b(?:on top of|onto|on|to)\s+(?:the\s+|a\s+|an\s+)?{shelf_mod}dresser\b", "Dresser"),
            (rf"\b(?:on top of|onto|on|to)\s+(?:the\s+|a\s+|an\s+)?{shelf_mod}shelf\b", "Shelf"),
            (rf"\b(?:on top of|onto|on|to)\s+(?:the\s+|a\s+|an\s+)?{shelf_mod}shelves\b", "Shelf"),
            (r"\b(?:in|into|inside|on top of|onto|on|to)\s+(?:the\s+|a\s+|an\s+)?bowl\b", "Bowl"),
            (r"\b(?:on top of|onto|on|to)\s+(?:the\s+|a\s+|an\s+)?plate\b", "Plate"),
            (r"\b(?:in|into|inside|on top of|onto|on|to)\s+(?:the\s+|a\s+|an\s+)?sink\b", "Sink"),
            (r"\b(?:in|into|inside|on top of|onto|on|to)\s+(?:the\s+|a\s+|an\s+)?(?:bath\s*tub|bathtub|tub)\b", "Bathtub"),
            (r"\b(?:in|into|inside|on top of|onto|on|to)\s+(?:the\s+|a\s+|an\s+)?fridge\b", "Fridge"),
            (r"\b(?:in|into|inside|on top of|onto|on|to)\s+(?:the\s+|a\s+|an\s+)?microwave\b", "Microwave"),
            (r"\b(?:in|into|inside|on top of|onto|on|to)\s+(?:the\s+|a\s+|an\s+)?box\b", "Box"),
            (r"\b(?:on top of|onto|on|to)\s+(?:the\s+|a\s+|an\s+)?(?:metal\s+)?rack\b", "Shelf"),
            (r"\b(?:on top of|onto|on|to)\s+(?:the\s+|a\s+|an\s+)?bed\b", "Bed"),
            (r"\b(?:on top of|onto|on|to)\s+(?:the\s+|a\s+|an\s+)?place\s+where\s+you\s+sleep\b", "Bed"),
            (r"\b(?:on top of|onto|on|to)\s+(?:the\s+|a\s+|an\s+)?sofa\b", "Sofa"),
            (r"\b(?:on top of|onto|on|to)\s+(?:the\s+|a\s+|an\s+)?couch\b", "Sofa"),
            (r"\b(?:on top of|onto|on|to)\s+(?:the\s+|a\s+|an\s+)?arm\s*chair\b", "ArmChair"),
            (r"\b(?:on top of|onto|on|to)\s+(?:the\s+|a\s+|an\s+)?(?:tv|television)\s+stand\b", "SideTable"),
            (r"\b(?:in|into|inside|on top of|onto|on|to)\s+(?:the\s+|a\s+|an\s+)?garbage\s+can\b", "GarbageCan"),
        ]

        def match_destination(scope: str) -> Tuple[Optional[int], Optional[str]]:
            for pattern, target in destination_patterns:
                if not re.search(pattern, scope):
                    continue
                action_id = self._find_action_id_by_name(f"find a {target}")
                if action_id is not None:
                    return action_id, target
            return None, None

        placement_verb = r"\b(?:place|put|move|drop|set|return|organize|arrange)\b"
        clauses = re.split(r"\b(?:and\s+then|then|finally|and)\b", instruction)
        for clause in reversed(clauses):
            if not re.search(placement_verb, clause):
                continue
            action_id, target = match_destination(clause)
            if action_id is not None:
                return action_id, target

        return match_destination(instruction)

    def _destination_open_action_id(self, destination: Optional[str]) -> Optional[int]:
        if not destination:
            return None
        destination_base = self._base_object_name(destination)
        if destination_base not in {"Drawer", "Cabinet", "Fridge", "Microwave", "Box", "Safe"}:
            return None
        instruction = f" {(self.current_human_instruction or '').lower()} "
        if self._surface_destination_open_is_unneeded(destination_base, instruction):
            return None
        return self._find_action_id_by_name(f"open the {destination}") or self._find_action_id_by_name(
            f"open the {destination_base}"
        )

    @staticmethod
    def _surface_destination_open_is_unneeded(destination_base: str, instruction: str) -> bool:
        if destination_base not in {"Cabinet", "Box", "Safe"}:
            return False
        surface_pattern = rf"\b(?:on top of|onto|on)\s+(?:the\s+|a\s+|an\s+)?(?:[\w]+\s+){{0,4}}{re.escape(destination_base.lower())}\b"
        return bool(re.search(surface_pattern, instruction))

    def _history_receptacle_is_open(self, destination: Optional[str]) -> bool:
        if not destination:
            return False
        destination_base = self._base_object_name(destination).lower()
        is_open = False
        for entry in getattr(self.base_planner, "episode_act_feedback", []) or []:
            if len(entry) < 2:
                continue
            action_id, feedback = entry[0], str(entry[1]).lower()
            if not isinstance(action_id, int) or not (0 <= action_id < len(self.base_planner.actions)):
                continue
            action_name = str(self.base_planner.actions[action_id]).strip().lower()
            action_target = self._action_target(action_id, "open the ") or self._action_target(action_id, "close the ")
            if not action_target or self._base_object_name(action_target).lower() != destination_base:
                continue
            if action_name.startswith("open the "):
                if "already open" in feedback or ("invalid" not in feedback and "failed" not in feedback):
                    is_open = True
            elif action_name.startswith("close the ") and "invalid" not in feedback and "failed" not in feedback:
                is_open = False
        return is_open

    def _destination_placement_prefix(self) -> Tuple[List[int], List[str]]:
        find_action_id, destination = self._destination_find_action_from_instruction()
        put_down_action_id = self._find_action_id_by_name("put down the object in hand")
        if find_action_id is None or put_down_action_id is None:
            return [], []

        prefix = [find_action_id]
        states = ["Exploration phase: destination not visible, prediction skipped."]
        open_action_id = self._destination_open_action_id(destination)
        if open_action_id is not None and not self._history_receptacle_is_open(destination):
            prefix.append(open_action_id)
            states.append(f"The {self._base_object_name(destination)} is open for placement.")
        prefix.append(put_down_action_id)
        states.append("The held object has been placed at the goal destination.")
        return prefix, states

    def _repair_held_object_destination_priority(
        self,
        actions: List[int],
        predicted_states: List[str],
    ) -> Tuple[List[int], List[str]]:
        if (
            not actions
            or not self._history_indicates_holding_object()
            or self._is_keep_holding_action_goal()
        ):
            return actions, predicted_states

        instruction = (self.current_human_instruction or "").lower()
        if any(term in instruction for term in ("clean", "wash", "rinse", "rinsed", "slice", "sliced", "heat", "heated", "cool")):
            return actions, predicted_states
        main_find_id, _, _, sec_find_id, _, _ = self._containment_targets_from_instruction()
        if main_find_id is not None or sec_find_id is not None:
            return actions, predicted_states

        destination_find_id, destination = self._destination_find_action_from_instruction()
        if destination_find_id is None or not destination:
            return actions, predicted_states
        placement_prefix, placement_states = self._destination_placement_prefix()
        if not placement_prefix:
            return actions, predicted_states

        first_hand = self._first_hand_action(actions)
        if first_hand == "pick":
            logger.info(
                "Replacing pickup plan with destination placement because the hand is already occupied"
            )
            return placement_prefix, placement_states
        if first_hand != "put":
            return actions, predicted_states

        find_before_put = None
        for action_id in actions:
            target = self._action_target(action_id, "find a ")
            if target:
                find_before_put = target
                continue
            if self._action_target(action_id, "put down ") or self._action_target(action_id, "drop "):
                break

        if find_before_put and self._base_object_name(find_before_put) == self._base_object_name(destination):
            return actions, predicted_states
        if not find_before_put:
            last_find = self._last_successful_find_target()
            if last_find and self._base_object_name(last_find) == self._base_object_name(destination):
                return actions, predicted_states

        logger.info(
            "Redirecting held object from non-destination '%s' to goal destination '%s'",
            find_before_put or "current location",
            destination,
        )
        return placement_prefix, placement_states

    def _repair_missing_destination_before_putdown(
        self,
        actions: List[int],
        predicted_states: List[str],
    ) -> Tuple[List[int], List[str]]:
        if not actions:
            return actions, predicted_states
        first_action = (
            str(self.base_planner.actions[actions[0]]).strip().lower()
            if isinstance(actions[0], int) and 0 <= actions[0] < len(self.base_planner.actions)
            else ""
        )
        if not first_action.startswith("put down "):
            return actions, predicted_states
        if not self._is_visual_descriptor_task():
            return actions, predicted_states

        last_find = self._last_successful_find_target()
        held_object = self._held_object_type_from_history()
        if last_find and held_object and self._base_object_name(last_find) != held_object:
            return actions, predicted_states

        destination_find_id, destination = self._destination_find_action_from_instruction()
        if destination_find_id is None:
            return actions, predicted_states
        if last_find and self._base_object_name(last_find) == destination:
            return actions, predicted_states

        prefix, states = self._destination_placement_prefix()
        if not prefix:
            return actions, predicted_states
        logger.info(
            "Prepending destination navigation '%s' before putdown while holding '%s'",
            self.base_planner.actions[destination_find_id],
            held_object or "object",
        )
        remaining_actions = list(actions)
        remaining_states = list(predicted_states)
        if remaining_actions and remaining_actions[0] == self._find_action_id_by_name("put down the object in hand"):
            remaining_actions.pop(0)
            if remaining_states:
                remaining_states.pop(0)
        return prefix + remaining_actions, states + remaining_states

    def _is_keep_holding_action_goal(self) -> bool:
        text = (self.current_human_instruction or "").lower()
        lamp_inspection = (
            any(term in text for term in ("inspect", "examine", "look at", "look over", "check"))
            and any(term in text for term in ("lamp", "light", "lit"))
        )
        return (
            (
                "turn on" in text
                and (" with " in text or "while holding" in text or " as well as " in text)
            )
            or lamp_inspection
        )

    def _repair_keep_holding_goal_actions(
        self,
        actions: List[int],
        predicted_states: List[str],
    ) -> Tuple[List[int], List[str]]:
        if not actions or not self._is_keep_holding_action_goal():
            return actions, predicted_states

        held = self._held_object_type_from_history()
        first_pick_target = None
        for action_id in actions:
            first_pick_target = self._action_target(action_id, "pick up the ")
            if first_pick_target:
                break
        if held and first_pick_target and self._base_object_name(first_pick_target) == held:
            instruction = (self.current_human_instruction or "").lower()
            lamp_type = "FloorLamp" if any(term in instruction for term in ("floor lamp", "tall lamp", "corner")) else "DeskLamp"
            find_lamp_id = self._find_action_id_by_name(f"find a {lamp_type}") or self._find_action_id_by_name("find a DeskLamp")
            turn_lamp_id = self._find_action_id_by_name(f"turn on the {lamp_type}") or self._find_action_id_by_name("turn on the DeskLamp")
            fallback_actions = [action_id for action_id in (find_lamp_id, turn_lamp_id) if action_id is not None]
            if fallback_actions:
                logger.info(
                    "Skipping repeated pickup of already-held '%s' for keep-holding goal",
                    held,
                )
                return (
                    fallback_actions,
                    [
                        "Exploration phase: locating the lamp while keeping hold of the target object.",
                        "The lamp is on for inspecting the held object.",
                    ][: len(fallback_actions)],
                )

        repaired_actions: List[int] = []
        repaired_states: List[str] = []
        removed_hand_clear = False
        for idx, action_id in enumerate(actions):
            action_name = (
                str(self.base_planner.actions[action_id]).strip().lower()
                if isinstance(action_id, int) and 0 <= action_id < len(self.base_planner.actions)
                else ""
            )
            if action_name.startswith("put down ") or action_name.startswith("drop "):
                removed_hand_clear = True
                continue
            repaired_actions.append(action_id)
            repaired_states.append(
                predicted_states[idx] if idx < len(predicted_states) else "Continuing while holding the target object."
            )

        if repaired_actions:
            if removed_hand_clear:
                logger.info("Removed hand-clearing action for keep-holding inspection/toggle goal")
            return repaired_actions, repaired_states

        instruction = (self.current_human_instruction or "").lower()
        lamp_type = "FloorLamp" if any(term in instruction for term in ("floor lamp", "tall lamp", "corner")) else "DeskLamp"
        find_lamp_id = self._find_action_id_by_name(f"find a {lamp_type}") or self._find_action_id_by_name("find a DeskLamp")
        turn_lamp_id = self._find_action_id_by_name(f"turn on the {lamp_type}") or self._find_action_id_by_name("turn on the DeskLamp")
        fallback_actions = [action_id for action_id in (find_lamp_id, turn_lamp_id) if action_id is not None]
        if fallback_actions:
            logger.info("Replacing hand-clearing-only plan with lamp inspection actions")
            return (
                fallback_actions,
                [
                    "Exploration phase: locating the lamp while keeping hold of the target object.",
                    "The lamp is on for inspecting the held object.",
                ][: len(fallback_actions)],
            )
        return actions, predicted_states

    def _repair_drop_for_placement(
        self,
        actions: List[int],
        predicted_states: List[str],
    ) -> Tuple[List[int], List[str]]:
        if not actions:
            return actions, predicted_states
        instruction = (self.current_human_instruction or "").lower()
        if not any(term in instruction for term in (" on ", " onto ", " on top of ", " in ", " into ", " inside ", " to ")):
            return actions, predicted_states
        put_down_action_id = self._find_action_id_by_name("put down the object in hand")
        if put_down_action_id is None:
            return actions, predicted_states

        repaired_actions = list(actions)
        for idx, action_id in enumerate(repaired_actions):
            action_name = (
                str(self.base_planner.actions[action_id]).strip().lower()
                if isinstance(action_id, int) and 0 <= action_id < len(self.base_planner.actions)
                else ""
            )
            if action_name == "drop the object in hand":
                logger.info("Replacing drop with goal-directed putdown for placement task")
                repaired_actions[idx] = put_down_action_id
        return repaired_actions, predicted_states

    def _repair_drawer_destination_actions(
        self,
        actions: List[int],
        predicted_states: List[str],
    ) -> Tuple[List[int], List[str]]:
        instruction = (self.current_human_instruction or "").lower()
        if "drawer" not in instruction:
            return actions, predicted_states
        replacements = {
            "find a Dresser": "find a Drawer",
            "find a Cabinet": "find a Drawer",
            "open the Cabinet": "open the Drawer",
            "close the Cabinet": "close the Drawer",
        }
        repaired_actions = list(actions)
        for idx, action_id in enumerate(repaired_actions):
            if not isinstance(action_id, int) or not (0 <= action_id < len(self.base_planner.actions)):
                continue
            action_name = str(self.base_planner.actions[action_id]).strip()
            replacement_name = replacements.get(action_name)
            if not replacement_name:
                continue
            replacement_id = self._find_action_id_by_name(replacement_name)
            if replacement_id is None:
                continue
            logger.info(
                "Canonicalizing drawer destination action '%s' to '%s'",
                action_name,
                replacement_name,
            )
            repaired_actions[idx] = replacement_id
        return repaired_actions, predicted_states

    def _repair_open_without_navigation(
        self,
        actions: List[int],
        predicted_states: List[str],
    ) -> Tuple[List[int], List[str]]:
        if not actions:
            return actions, predicted_states
        repaired_actions: List[int] = []
        repaired_states: List[str] = []
        last_find = self._last_successful_find_target()

        for idx, action_id in enumerate(actions):
            target = None
            for prefix in ("open the ", "close the "):
                target = self._action_target(action_id, prefix)
                if target:
                    break

            if target:
                previous_find = None
                if repaired_actions:
                    previous_find = self._action_target(repaired_actions[-1], "find a ")
                elif last_find:
                    previous_find = last_find

                if not previous_find or self._base_object_name(previous_find) != self._base_object_name(target):
                    find_action_id = self._find_action_id_by_name(
                        f"find a {target}"
                    ) or self._find_action_id_by_name(
                        f"find a {self._base_object_name(target)}"
                    )
                    if find_action_id is not None:
                        logger.info(
                            "Inserting navigation '%s' before '%s'",
                            self.base_planner.actions[find_action_id],
                            self.base_planner.actions[action_id],
                        )
                        repaired_actions.append(find_action_id)
                        repaired_states.append(
                            "Exploration phase: receptacle not visible, prediction skipped."
                        )

            repaired_actions.append(action_id)
            repaired_states.append(
                predicted_states[idx] if idx < len(predicted_states) else "Exploring environment."
            )

        return repaired_actions, repaired_states

    def _repair_toggle_without_navigation(
        self,
        actions: List[int],
        predicted_states: List[str],
    ) -> Tuple[List[int], List[str]]:
        if not actions:
            return actions, predicted_states

        repaired_actions: List[int] = []
        repaired_states: List[str] = []
        last_find = self._last_successful_find_target()

        for idx, action_id in enumerate(actions):
            target = self._action_target(action_id, "turn on the ") or self._action_target(action_id, "turn off the ")
            if target:
                previous_find = None
                if repaired_actions:
                    previous_find = self._action_target(repaired_actions[-1], "find a ")
                elif last_find:
                    previous_find = last_find

                if not previous_find or self._base_object_name(previous_find) != self._base_object_name(target):
                    find_action_id = self._find_action_id_by_name(
                        f"find a {target}"
                    ) or self._find_action_id_by_name(
                        f"find a {self._base_object_name(target)}"
                    )
                    if find_action_id is not None:
                        logger.info(
                            "Inserting navigation '%s' before toggle '%s'",
                            self.base_planner.actions[find_action_id],
                            self.base_planner.actions[action_id],
                        )
                        repaired_actions.append(find_action_id)
                        repaired_states.append(
                            "Exploration phase: toggle target not visible, navigating to it first."
                        )

            repaired_actions.append(action_id)
            repaired_states.append(
                predicted_states[idx] if idx < len(predicted_states) else "Exploring environment."
            )

        return repaired_actions, repaired_states

    def _limit_counted_visual_transfer_cycle(
        self,
        actions: List[int],
        predicted_states: List[str],
    ) -> Tuple[List[int], List[str]]:
        if (
            not actions
            or not self._is_counted_instruction()
            or self._is_keep_holding_action_goal()
        ):
            return actions, predicted_states

        placement_prefix, placement_states = self._destination_placement_prefix()
        if not placement_prefix:
            return actions, predicted_states

        if self._history_indicates_holding_object() and self._first_hand_action(actions) == "pick":
            logger.info(
                "Replacing counted plan with destination placement because the hand is already occupied"
            )
            return placement_prefix, placement_states

        first_pick_idx = None
        for idx, action_id in enumerate(actions):
            action_name = (
                str(self.base_planner.actions[action_id]).strip().lower()
                if isinstance(action_id, int) and 0 <= action_id < len(self.base_planner.actions)
                else ""
            )
            if action_name.startswith("pick up "):
                first_pick_idx = idx
                break

        if first_pick_idx is None:
            return actions, predicted_states

        logger.info(
            "Normalizing counted plan to one transfer cycle with explicit destination placement"
        )
        return (
            list(actions[:first_pick_idx + 1]) + placement_prefix,
            list(predicted_states[:first_pick_idx + 1]) + placement_states,
        )

    def _repair_closed_receptacle_put_feedback(
        self,
        actions: List[int],
        predicted_states: List[str],
        feedback: str,
    ) -> Tuple[List[int], List[str]]:
        if not actions or not feedback or not self._history_indicates_holding_object():
            return actions, predicted_states
        if "putting the object on" not in feedback.lower() or "open" not in feedback.lower():
            return actions, predicted_states

        match = re.search(r"Putting the object on ([A-Za-z][\w]*) failed", feedback, re.IGNORECASE)
        if not match:
            return actions, predicted_states
        receptacle = match.group(1)
        open_action_id = self._find_action_id_by_name(f"open the {receptacle}")
        if open_action_id is None:
            open_action_id = self._find_action_id_by_name(f"open the {self._base_object_name(receptacle)}")
        put_down_action_id = self._find_action_id_by_name("put down the object in hand")
        if open_action_id is None or put_down_action_id is None:
            return actions, predicted_states

        first_action = (
            str(self.base_planner.actions[actions[0]]).strip().lower()
            if isinstance(actions[0], int) and 0 <= actions[0] < len(self.base_planner.actions)
            else ""
        )
        if first_action == str(self.base_planner.actions[open_action_id]).strip().lower():
            return actions, predicted_states

        remaining_actions = list(actions)
        remaining_states = list(predicted_states)
        while remaining_actions:
            candidate = (
                str(self.base_planner.actions[remaining_actions[0]]).strip().lower()
                if isinstance(remaining_actions[0], int) and 0 <= remaining_actions[0] < len(self.base_planner.actions)
                else ""
            )
            if candidate == "put down the object in hand":
                remaining_actions.pop(0)
                if remaining_states:
                    remaining_states.pop(0)
            else:
                break

        logger.info(
            "Opening receptacle '%s' before retrying putdown after placement feedback",
            receptacle,
        )
        return (
            [open_action_id, put_down_action_id] + remaining_actions,
            [
                f"The {self._base_object_name(receptacle)} is open for placement.",
                "The held object has been placed into/on the opened receptacle.",
            ] + remaining_states,
        )

    def _repair_receptacle_instance_put_feedback(
        self,
        actions: List[int],
        predicted_states: List[str],
        feedback: str,
    ) -> Tuple[List[int], List[str]]:
        if not feedback or not self._history_indicates_holding_object():
            return actions, predicted_states
        lowered = feedback.lower()
        if "putting the object on" not in lowered or "other instances" not in lowered:
            return actions, predicted_states

        match = re.search(r"Putting the object on ([A-Za-z][\w]*) failed", feedback, re.IGNORECASE)
        if not match:
            return actions, predicted_states
        receptacle = self._base_object_name(match.group(1))
        alternate_find_id = self._select_untried_counted_find_action(receptacle)
        put_down_action_id = self._find_action_id_by_name("put down the object in hand")
        if alternate_find_id is None or put_down_action_id is None:
            return actions, predicted_states

        logger.info(
            "Retrying placement on alternate receptacle instance '%s' after put failure",
            self.base_planner.actions[alternate_find_id],
        )
        return (
            [alternate_find_id, put_down_action_id],
            [
                "Exploration phase: locating another receptacle instance after placement feedback.",
                "The held object has been placed in/on the alternate receptacle instance.",
            ],
        )

    def _apply_visual_observation_policy(
        self,
        actions: Any,
        predicted_states: List[str],
    ) -> Tuple[Any, List[str]]:
        if (
            not self.enable_visual_observation_summary
            or not isinstance(actions, list)
            or not actions
            or not self._is_visual_descriptor_task()
        ):
            return actions, predicted_states

        repaired_actions = list(actions)
        repaired_states = list(predicted_states)
        repaired_actions, repaired_states = self._limit_counted_visual_transfer_cycle(
            repaired_actions, repaired_states
        )
        first_action_id = repaired_actions[0]
        first_action_name = (
            str(self.base_planner.actions[first_action_id]).strip()
            if isinstance(first_action_id, int) and 0 <= first_action_id < len(self.base_planner.actions)
            else ""
        )
        targets = self._instruction_visual_targets()
        should_probe_only = False

        target_type = None
        for prefix in ("pick up the ", "turn on the ", "turn off the ", "open the ", "close the ", "find a "):
            target_type = self._action_target(first_action_id, prefix)
            if target_type:
                break

        replacement_find_id = None
        if target_type:
            colors = self._target_colors_for_type(target_type, targets)
            visible_entries = self._visible_entries_for_type(target_type)
            if colors and visible_entries and not any(
                self._entry_matches_any_color(entry, colors) for entry in visible_entries
            ):
                replacement_find_id = self._select_next_visual_find_action(target_type)
                should_probe_only = True

            if first_action_name.lower().startswith("find a "):
                indexed_find_target = bool(re.search(r"_\d+$", target_type or ""))
                if not visible_entries:
                    should_probe_only = True
                elif colors and not any(self._entry_matches_any_color(entry, colors) for entry in visible_entries):
                    should_probe_only = True
                elif len(repaired_actions) > 1 and not indexed_find_target:
                    next_target = self._action_target(repaired_actions[1], "pick up the ")
                    if next_target and self._base_object_name(next_target) == self._base_object_name(target_type):
                        logger.info(
                            "Visual observation policy removing redundant find '%s' because matching target is already visible",
                            first_action_name,
                        )
                        repaired_actions = repaired_actions[1:]
                        repaired_states = repaired_states[1:] if len(repaired_states) > 1 else repaired_states[:0]
                        should_probe_only = False

        if first_action_name.lower().startswith("put down "):
            last_find_target = self._last_successful_find_target()
            if last_find_target:
                colors = self._target_colors_for_type(last_find_target, targets)
                visible_entries = self._visible_entries_for_type(last_find_target)
                if colors and visible_entries and not any(
                    self._entry_matches_any_color(entry, colors) for entry in visible_entries
                ):
                    replacement_find_id = self._select_next_visual_find_action(last_find_target)
                    should_probe_only = True

        if replacement_find_id is not None and replacement_find_id != first_action_id:
            logger.info(
                "Visual observation policy replacing '%s' with '%s' after color mismatch",
                first_action_name,
                self.base_planner.actions[replacement_find_id],
            )
            repaired_actions[0] = replacement_find_id
            if repaired_states:
                repaired_states[0] = "Exploration phase: target visual appearance did not match; inspecting another candidate."
            else:
                repaired_states = ["Exploration phase: target visual appearance did not match; inspecting another candidate."]

        if should_probe_only and len(repaired_actions) > 1:
            logger.info(
                "Visual observation policy executing one action before replanning from current-view metadata"
            )
            repaired_actions = repaired_actions[:1]
            repaired_states = repaired_states[:1] if repaired_states else repaired_states

        return repaired_actions, repaired_states

    def _container_repair_from_feedback(self, feedback: str) -> Tuple[Optional[int], Optional[int], Optional[str], Optional[str]]:
        if not feedback:
            return None, None, None, None

        match = re.search(
            r"\b([\w]+) is not visible because it is (?:in|on|inside) ([A-Za-z][\w]*)",
            feedback,
            re.IGNORECASE,
        )
        if not match:
            return None, None, None, None

        raw_target_name = match.group(1)
        raw_container_name = match.group(2)
        target_name = re.sub(r'_\d+$', '', raw_target_name)
        container_name = re.sub(r'_\d+$', '', raw_container_name)
        _, destination = self._destination_find_action_from_instruction()
        avoid_open_generic = (
            self._is_counted_instruction()
            and self._latest_task_progress() > 0
            and destination is not None
            and self._base_object_name(destination) == container_name
        )
        if avoid_open_generic:
            find_action_id, preferred_container = self._preferred_feedback_container_find_action(
                container_name,
                avoid_open_generic=True,
            )
            if preferred_container:
                raw_container_name = preferred_container
                container_name = preferred_container
        else:
            find_action_id = self._find_action_id_by_name(
                f"find a {raw_container_name}"
            ) or self._find_action_id_by_name(
                f"find a {container_name}"
            )
        pick_action_id = self._find_action_id_by_name(
            f"pick up the {raw_target_name}"
        ) or self._find_action_id_by_name(
            f"pick up the {target_name}"
        )
        return find_action_id, pick_action_id, target_name, container_name

    def _history_indicates_holding_object(self) -> bool:
        holding = False
        for entry in getattr(self.base_planner, "episode_act_feedback", []) or []:
            if len(entry) < 2:
                continue
            action_id, feedback = entry[0], str(entry[1]).lower()
            if not isinstance(action_id, int) or not (0 <= action_id < len(self.base_planner.actions)):
                continue
            if "not holding" in feedback or "dropped the object instead" in feedback:
                holding = False
                continue
            if "currently holding" in feedback:
                holding = True
                continue
            if (
                "invalid" in feedback
                or "failed" in feedback
                or "not visible" in feedback
                or "cannot " in feedback
            ):
                continue

            action_name = str(self.base_planner.actions[action_id]).strip().lower()
            if action_name.startswith("pick up "):
                holding = True
            elif action_name.startswith("put down ") or action_name.startswith("drop "):
                holding = False
        return holding

    def _last_visibility_container_feedback(self) -> Tuple[Optional[str], Optional[str]]:
        for entry in reversed(getattr(self.base_planner, "episode_act_feedback", []) or []):
            if len(entry) < 2:
                continue
            match = re.search(
                r"\b([\w]+) is not visible because it is (?:in|on|inside) ([A-Za-z][\w]*)",
                str(entry[1]),
                re.IGNORECASE,
            )
            if match:
                return self._base_object_name(match.group(1)), match.group(2)
        return None, None

    def _repair_after_feedback_container_navigation(
        self,
        actions: List[int],
        predicted_states: List[str],
    ) -> Tuple[List[int], List[str]]:
        if not actions or self._history_indicates_holding_object():
            return actions, predicted_states
        target_name, container_name = self._last_visibility_container_feedback()
        if not target_name or not container_name or not self._instruction_mentions_object(target_name):
            return actions, predicted_states
        last_find = self._last_successful_find_target()
        if not last_find or self._base_object_name(last_find) != self._base_object_name(container_name):
            return actions, predicted_states

        pick_action_id = self._find_action_id_by_name(f"pick up the {target_name}")
        if pick_action_id is None:
            return actions, predicted_states

        prefix_actions: List[int] = []
        prefix_states: List[str] = []
        open_action_id = self._find_action_id_by_name(
            f"open the {last_find}"
        ) or self._find_action_id_by_name(
            f"open the {self._base_object_name(last_find)}"
        )
        if (
            open_action_id is not None
            and self._base_object_name(last_find) in {"Cabinet", "Drawer", "Box", "Safe", "Fridge", "Microwave"}
            and not self._history_receptacle_is_open(last_find)
        ):
            prefix_actions.append(open_action_id)
            prefix_states.append(f"The {self._base_object_name(last_find)} is open for pickup.")
        prefix_actions.append(pick_action_id)
        prefix_states.append(f"I am holding the {target_name} from the referenced receptacle.")

        remaining_actions = list(actions)
        remaining_states = list(predicted_states)
        while remaining_actions:
            first = remaining_actions[0]
            first_find = self._action_target(first, "find a ")
            first_pick = self._action_target(first, "pick up the ")
            if (
                (first_find and self._base_object_name(first_find) in {self._base_object_name(container_name), target_name})
                or (first_pick and self._base_object_name(first_pick) == target_name)
            ):
                remaining_actions.pop(0)
                if remaining_states:
                    remaining_states.pop(0)
                continue
            break

        logger.info(
            "Following prior visibility feedback after container navigation by picking '%s'",
            self.base_planner.actions[pick_action_id],
        )
        return prefix_actions + remaining_actions, prefix_states + remaining_states

    def _remove_redundant_open_actions(
        self,
        actions: List[int],
        predicted_states: List[str],
    ) -> Tuple[List[int], List[str]]:
        if not actions:
            return actions, predicted_states
        instruction = f" {(self.current_human_instruction or '').lower()} "
        repaired_actions: List[int] = []
        repaired_states: List[str] = []
        for idx, action_id in enumerate(actions):
            target = self._action_target(action_id, "open the ")
            if target:
                target_base = self._base_object_name(target)
                if self._history_receptacle_is_open(target) or self._surface_destination_open_is_unneeded(target_base, instruction):
                    logger.info(
                        "Removing redundant open action '%s'",
                        self.base_planner.actions[action_id],
                    )
                    continue
            repaired_actions.append(action_id)
            repaired_states.append(
                predicted_states[idx] if idx < len(predicted_states) else "Exploring environment."
            )
        return repaired_actions, repaired_states

    def _first_hand_action(self, actions: List[int]) -> Optional[str]:
        for action_id in actions:
            if not isinstance(action_id, int) or not (0 <= action_id < len(self.base_planner.actions)):
                continue
            action_name = str(self.base_planner.actions[action_id]).strip().lower()
            if action_name.startswith("pick up "):
                return "pick"
            if action_name.startswith("put down ") or action_name.startswith("drop "):
                return "put"
        return None

    def _last_progress_drop_after_pick(self) -> Tuple[Optional[str], float, float]:
        previous_progress = 0.0
        for entry in getattr(self.base_planner, "episode_act_feedback", []) or []:
            if len(entry) < 2:
                continue
            parsed_progress = self._task_progress_from_feedback(str(entry[1]))
            if parsed_progress is None:
                continue
            action_id = entry[0]
            picked_target = self._action_target(action_id, "pick up the ")
            if (
                picked_target
                and parsed_progress + 1e-6 < previous_progress
                and self._instruction_mentions_object(picked_target)
            ):
                feedback = str(entry[1]).lower()
                if "invalid" not in feedback and "failed" not in feedback:
                    return picked_target, previous_progress, parsed_progress
            previous_progress = parsed_progress
        return None, previous_progress, previous_progress

    def _select_untried_counted_find_action(self, target_type: str) -> Optional[int]:
        candidates = self._find_action_aliases_for_type(target_type)
        if not candidates:
            return None
        tried_aliases = self._tried_find_aliases_for_type(target_type)
        current_aliases = {
            str(entry.get('alias'))
            for entry in self._visible_entries_for_type(target_type)
            if entry.get('alias')
        }
        for action_id, alias in candidates:
            if alias not in tried_aliases and alias not in current_aliases:
                return action_id
        for action_id, alias in candidates:
            if alias not in current_aliases:
                return action_id
        for action_id, alias in candidates:
            if alias not in tried_aliases:
                return action_id
        return None

    def _repair_counted_progress_drop(
        self,
        actions: List[int],
        predicted_states: List[str],
    ) -> Tuple[List[int], List[str]]:
        if not actions or not self._is_counted_instruction():
            return actions, predicted_states

        dropped_target, previous_progress, current_progress = self._last_progress_drop_after_pick()
        if not dropped_target:
            return actions, predicted_states

        placement_prefix, placement_states = self._destination_placement_prefix()
        if self._history_indicates_holding_object() and placement_prefix:
            logger.info(
                "Repairing counted progress drop for '%s' (%.3f -> %.3f) by returning the held object to the destination",
                dropped_target,
                previous_progress,
                current_progress,
            )
            return placement_prefix, placement_states

        if self._latest_task_progress() <= 0:
            return actions, predicted_states

        first_find_target = self._action_target(actions[0], "find a ")
        first_pick_target = self._action_target(actions[0], "pick up the ")
        target_name = first_find_target or first_pick_target
        if not target_name or self._base_object_name(target_name) != self._base_object_name(dropped_target):
            return actions, predicted_states

        alternate_find_id = self._select_untried_counted_find_action(target_name)
        if alternate_find_id is None or alternate_find_id == actions[0]:
            return actions, predicted_states

        logger.info(
            "Redirecting counted search from '%s' to alternate instance '%s' after prior progress drop",
            self.base_planner.actions[actions[0]],
            self.base_planner.actions[alternate_find_id],
        )
        repaired_actions = list(actions)
        repaired_states = list(predicted_states)
        if first_find_target:
            repaired_actions[0] = alternate_find_id
            if repaired_states:
                repaired_states[0] = "Exploration phase: locating a different counted instance after a prior placed-object pickup."
            return repaired_actions, repaired_states
        return (
            [alternate_find_id] + repaired_actions,
            ["Exploration phase: locating a different counted instance after a prior placed-object pickup."]
            + repaired_states,
        )

    def _repair_counted_destination_repickup(
        self,
        actions: List[int],
        predicted_states: List[str],
    ) -> Tuple[List[int], List[str]]:
        if (
            not actions
            or not self._is_counted_instruction()
            or self._latest_task_progress() <= 0
            or self._is_keep_holding_action_goal()
        ):
            return actions, predicted_states

        _, destination = self._destination_find_action_from_instruction()
        if not destination:
            return actions, predicted_states

        first_pick_idx = None
        pick_target = None
        for idx, action_id in enumerate(actions):
            pick_target = self._action_target(action_id, "pick up the ")
            if pick_target:
                first_pick_idx = idx
                break
        if first_pick_idx is None or not pick_target:
            return actions, predicted_states
        if not self._instruction_mentions_object(pick_target):
            return actions, predicted_states
        if not self._visible_target_at_destination(pick_target, destination):
            return actions, predicted_states

        alternate_find_id = self._select_untried_counted_find_action(pick_target)
        if alternate_find_id is None:
            return actions, predicted_states

        repaired_actions = list(actions)
        repaired_states = list(predicted_states)
        previous_find_idx = None
        if first_pick_idx > 0:
            prior_target = self._action_target(repaired_actions[first_pick_idx - 1], "find a ")
            if prior_target and self._base_object_name(prior_target) == self._base_object_name(pick_target):
                previous_find_idx = first_pick_idx - 1

        logger.info(
            "Avoiding pickup of already-placed counted target '%s'; redirecting to '%s'",
            pick_target,
            self.base_planner.actions[alternate_find_id],
        )
        if previous_find_idx is not None:
            repaired_actions[previous_find_idx] = alternate_find_id
            if previous_find_idx < len(repaired_states):
                repaired_states[previous_find_idx] = (
                    "Exploration phase: the visible target at the destination is already completed; "
                    "locating another counted instance."
                )
            return repaired_actions, repaired_states

        return (
            [alternate_find_id] + repaired_actions,
            [
                "Exploration phase: the visible target at the destination is already completed; locating another counted instance."
            ]
            + repaired_states,
        )

    def _replacement_find_for_visual_alias(self, target: str) -> Optional[int]:
        instruction = f" {(self.current_human_instruction or '').lower().replace('-', ' ')} "
        target_base = self._base_object_name(target)

        candidate_types: List[str] = []
        if " jar" in instruction or " jars" in instruction:
            if target_base in {"Box", "Cart"}:
                candidate_types = ["Vase", "Glassbottle", "Cup", "Mug", "Pot"]
        elif " stand" in instruction or " statue" in instruction:
            if target_base in {"Box", "Cart"}:
                candidate_types = ["Statue"]

        for candidate in candidate_types:
            find_action_id = self._find_action_id_by_name(f"find a {candidate}")
            if find_action_id is not None:
                return find_action_id
        return None

    def _repair_visual_alias_action_targets(
        self,
        actions: List[int],
        predicted_states: List[str],
    ) -> Tuple[List[int], List[str]]:
        if not actions or not self._is_visual_descriptor_task():
            return actions, predicted_states

        repaired_actions = list(actions)
        repaired_states = list(predicted_states)
        for idx, action_id in enumerate(repaired_actions):
            target = self._action_target(action_id, "find a ")
            if not target:
                continue
            replacement_id = self._replacement_find_for_visual_alias(target)
            if replacement_id is None or replacement_id == action_id:
                continue
            logger.info(
                "Replacing visual alias search '%s' with '%s'",
                self.base_planner.actions[action_id],
                self.base_planner.actions[replacement_id],
            )
            repaired_actions[idx] = replacement_id
            if idx < len(repaired_states):
                repaired_states[idx] = "Exploration phase: using the instruction's visual noun mapping to inspect a plausible canonical class."
            if idx == 0:
                break

        instruction = (self.current_human_instruction or "").lower()
        if "lamp" in instruction:
            desk_lamp_find = self._find_action_id_by_name("find a DeskLamp")
            if desk_lamp_find is not None:
                has_desk_lamp_interaction = any(
                    self._base_object_name(
                        self._action_target(action_id, "turn on the ")
                        or self._action_target(action_id, "turn off the ")
                        or ""
                    )
                    == "DeskLamp"
                    for action_id in repaired_actions
                )
                for idx, action_id in enumerate(repaired_actions):
                    find_target = self._action_target(action_id, "find a ")
                    if find_target == "FloorLamp" and ("desk lamp" in instruction or has_desk_lamp_interaction):
                        logger.info(
                            "Aligning lamp navigation '%s' with DeskLamp interaction",
                            self.base_planner.actions[action_id],
                        )
                        repaired_actions[idx] = desk_lamp_find
                        if idx < len(repaired_states):
                            repaired_states[idx] = "Exploration phase: locating the DeskLamp before toggling it."
                        break

        return repaired_actions, repaired_states

    def _repair_instruction_alias_action_targets(
        self,
        actions: List[int],
        predicted_states: List[str],
    ) -> Tuple[List[int], List[str]]:
        if not actions:
            return actions, predicted_states

        instruction = f" {(self.current_human_instruction or '').lower().replace('-', ' ')} "
        repaired_actions = list(actions)
        repaired_states = list(predicted_states)

        prefer_soap_bar = (
            "soap" in instruction
            and "bottle" not in instruction
            and "pump" not in instruction
            and "soapbar" not in instruction
        )
        if not prefer_soap_bar:
            return repaired_actions, repaired_states

        for idx, action_id in enumerate(repaired_actions):
            for prefix, replacement_name in (
                ("find a ", "find a SoapBar"),
                ("pick up the ", "pick up the SoapBar"),
            ):
                target = self._action_target(action_id, prefix)
                if self._base_object_name(target or "") != "SoapBottle":
                    continue
                replacement_id = self._find_action_id_by_name(replacement_name)
                if replacement_id is None:
                    continue
                logger.info(
                    "Replacing soap-bottle action '%s' with '%s' for non-bottle soap instruction",
                    self.base_planner.actions[action_id],
                    replacement_name,
                )
                repaired_actions[idx] = replacement_id
                if idx < len(repaired_states):
                    repaired_states[idx] = "Using SoapBar because the instruction asks for soap/bar of soap, not a soap bottle."
                break

        return repaired_actions, repaired_states

    def _repair_mismatched_pick_after_find(
        self,
        actions: List[int],
        predicted_states: List[str],
    ) -> Tuple[List[int], List[str]]:
        repaired_actions = list(actions)
        repaired_states = list(predicted_states)
        for idx in range(len(repaired_actions) - 1):
            find_target = self._action_target(repaired_actions[idx], "find a ")
            pick_target = self._action_target(repaired_actions[idx + 1], "pick up the ")
            if not find_target or not pick_target or find_target.lower() == pick_target.lower():
                continue
            if not self._instruction_mentions_object(find_target):
                continue
            if self._instruction_mentions_object(pick_target):
                continue

            corrected_pick_id = self._find_action_id_by_name(f"pick up the {find_target}")
            if corrected_pick_id is None:
                continue

            logger.info(
                "Correcting pickup after '%s': replacing '%s' with '%s'",
                self.base_planner.actions[repaired_actions[idx]],
                self.base_planner.actions[repaired_actions[idx + 1]],
                self.base_planner.actions[corrected_pick_id],
            )
            repaired_actions[idx + 1] = corrected_pick_id
            if idx + 1 < len(repaired_states):
                repaired_states[idx + 1] = f"I am holding the {find_target}."
        return repaired_actions, repaired_states

    def _repair_indexed_pick_after_find(
        self,
        actions: List[Any],
        predicted_states: List[str],
    ) -> Tuple[List[Any], List[str]]:
        repaired_actions = list(actions)
        repaired_states = list(predicted_states)
        for idx in range(len(repaired_actions) - 1):
            find_target = self._action_target(repaired_actions[idx], "find a ")
            pick_target = self._action_target(repaired_actions[idx + 1], "pick up the ")
            if not find_target or not pick_target or not re.search(r"_\d+$", find_target):
                continue
            if self._base_object_name(find_target) != self._base_object_name(pick_target):
                continue
            if not self._instruction_mentions_object(find_target):
                continue
            indexed_pick = f"pick up the {find_target}"
            logger.info(
                "Using indexed pickup '%s' after indexed find '%s'",
                indexed_pick,
                find_target,
            )
            repaired_actions[idx + 1] = indexed_pick
            if idx + 1 < len(repaired_states):
                repaired_states[idx + 1] = f"I am holding the {find_target}."
        return repaired_actions, repaired_states

    def _source_find_action_from_text(
        self,
        text: str,
        destination: Optional[str] = None,
    ) -> Tuple[Optional[int], Optional[str]]:
        lower_text = str(text or "").lower().replace("-", " ")
        if not lower_text:
            return None, None

        source_patterns = [
            (r"coffee\s+table", "CoffeeTable"),
            (r"side\s+table", "SideTable"),
            (r"dining\s+table|kitchen\s+table|wooden\s+table", "DiningTable"),
            (r"counter\s*top|counter", "CounterTop"),
            (r"sink\s+basin|sink", "Sink"),
            (r"desk", "Desk"),
            (r"drawer", "Drawer"),
            (r"cabinet", "Cabinet"),
            (r"dresser", "Dresser"),
            (r"shelf", "Shelf"),
            (r"toilet", "Toilet"),
            (r"fridge|refrigerator", "Fridge"),
            (r"microwave", "Microwave"),
            (r"box", "Box"),
            (r"bed", "Bed"),
            (r"sofa", "Sofa"),
            (r"arm\s*chair", "ArmChair"),
            (r"table", "DiningTable"),
        ]
        destination_base = self._base_object_name(destination).lower() if destination else ""
        for pattern, source in source_patterns:
            if self._base_object_name(source).lower() == destination_base:
                continue
            source_mention = rf"(?:on|in|inside|from|at|near)\s+(?:the\s+|a\s+|an\s+)?{pattern}\b"
            if not re.search(source_mention, lower_text):
                continue
            find_action_id = self._find_action_id_by_name(f"find a {source}")
            if find_action_id is not None:
                return find_action_id, source
        return None, None

    def _repair_counted_source_navigation(
        self,
        actions: List[int],
        predicted_states: List[str],
    ) -> Tuple[List[int], List[str]]:
        if (
            not actions
            or not self._is_counted_instruction()
            or self._latest_task_progress() <= 0
            or self._history_indicates_holding_object()
        ):
            return actions, predicted_states

        first_find_target = self._action_target(actions[0], "find a ")
        first_pick_target = self._action_target(actions[0], "pick up the ")
        target_name = first_find_target or first_pick_target
        if not target_name or not self._instruction_mentions_object(target_name):
            return actions, predicted_states

        has_matching_pick = False
        if first_pick_target:
            has_matching_pick = True
        else:
            for action_id in actions[1:]:
                if self._action_target(action_id, "put down ") or self._action_target(action_id, "drop "):
                    break
                pick_target = self._action_target(action_id, "pick up the ")
                if pick_target and self._base_object_name(pick_target) == self._base_object_name(target_name):
                    has_matching_pick = True
                    break
        if not has_matching_pick:
            return actions, predicted_states

        _, destination = self._destination_find_action_from_instruction()
        source_texts = []
        if predicted_states:
            source_texts.append(predicted_states[0])
        if self.current_agent_output:
            source_texts.append(self.current_agent_output)

        for source_text in source_texts:
            source_find_id, source = self._source_find_action_from_text(source_text, destination)
            if source_find_id is None or source_find_id == actions[0]:
                continue
            last_find = self._last_successful_find_target()
            if last_find and self._base_object_name(last_find) == self._base_object_name(source):
                return actions, predicted_states
            prefix_actions = [source_find_id]
            prefix_states = [
                "Exploration phase: moving to the predicted source receptacle for the next counted instance."
            ]
            if first_pick_target:
                target_find_id = self._find_action_id_by_name(
                    f"find a {target_name}"
                ) or self._find_action_id_by_name(
                    f"find a {self._base_object_name(target_name)}"
                )
                if target_find_id is not None:
                    prefix_actions.append(target_find_id)
                    prefix_states.append(
                        "Exploration phase: locating the next counted instance at the predicted source."
                    )
            logger.info(
                "Prepending counted source navigation '%s' before searching for '%s'",
                self.base_planner.actions[source_find_id],
                target_name,
            )
            return (
                prefix_actions + list(actions),
                prefix_states + list(predicted_states),
            )
        return actions, predicted_states

    def _is_clean_or_rinse_instruction(self) -> bool:
        instruction = (self.current_human_instruction or "").lower()
        return any(
            term in instruction
            for term in ("clean", "wash", "washed", "rins", "rinse", "rinsed")
        )

    def _instruction_marks_object_as_distractor(self, object_name: str) -> bool:
        instruction = (self.current_human_instruction or "").lower().replace("-", " ")
        base_name = self._base_object_name(object_name)
        spaced = re.sub(r"(?<!^)(?=[A-Z])", " ", base_name).lower()
        aliases = {spaced, spaced.replace(" ", ""), base_name.lower()}
        extra_aliases = {
            "Bowl": {"bowl", "round eating vessel", "container for rice"},
            "Plate": {"plate", "eating surface"},
            "SoapBar": {"soap bar", "bar of soap", "bars of soap"},
            "Ladle": {"ladle", "soup serving tool"},
            "Spatula": {"spatula", "flipping tool"},
            "Apple": {"apple", "snow white"},
            "Tomato": {"tomato"},
            "Book": {"book", "bound collection", "bound narratives"},
        }
        aliases.update(extra_aliases.get(base_name, set()))
        aliases = {re.escape(alias) for alias in aliases if alias}
        if not aliases:
            return False
        alias_pattern = r"(?:%s)" % "|".join(sorted(aliases, key=len, reverse=True))
        return bool(
            re.search(rf"\b(?:although|though|instead\s+of|rather\s+than)\b[^,.;!?]*\b{alias_pattern}s?\b", instruction)
        )

    def _instruction_action_context_score(self, object_name: str) -> int:
        instruction = (self.current_human_instruction or "").lower().replace("-", " ")
        base_name = self._base_object_name(object_name)
        aliases = {
            base_name.lower(),
            re.sub(r"(?<!^)(?=[A-Z])", " ", base_name).lower(),
        }
        context_aliases = {
            "Ladle": {"ladle", "soup serving tool"},
            "Spatula": {"spatula", "flipping tool"},
            "Plate": {"plate"},
            "SoapBar": {"soap bar", "bar of soap", "bars of soap"},
            "Book": {"book", "bound collection", "bound narratives"},
            "CellPhone": {"communication device", "phone", "cell phone"},
            "RemoteControl": {"remote", "remote control", "device for changing channels"},
        }
        aliases.update(context_aliases.get(base_name, set()))
        score = 0
        for alias in aliases:
            if not alias:
                continue
            escaped = re.escape(alias)
            if re.search(rf"\b(?:clean|wash|washed|rinse|rinsed|rinsing)\s+(?:off\s+)?(?:the\s+|a\s+|an\s+)?{escaped}s?\b", instruction):
                score += 20
            if re.search(rf"\b(?:get|move|put|place|pick\s+up|transfer|return)\b[^,.;!?]*\b{escaped}s?\b", instruction):
                score += 10
        if self._instruction_marks_object_as_distractor(object_name):
            score -= 100
        return score

    def _history_has_successful_faucet_cycle(self) -> bool:
        for entry in getattr(self.base_planner, "episode_act_feedback", []) or []:
            if len(entry) < 2:
                continue
            action_id, feedback = entry[0], str(entry[1]).lower()
            if not isinstance(action_id, int) or not (0 <= action_id < len(self.base_planner.actions)):
                continue
            action_name = str(self.base_planner.actions[action_id]).strip().lower()
            if not action_name.startswith(("turn on the faucet", "turn off the faucet")):
                continue
            if (
                "invalid" in feedback
                or "failed" in feedback
                or "not visible" in feedback
                or "cannot " in feedback
            ):
                continue
            return True
        return False

    def _primary_pick_target_from_instruction(self) -> Tuple[Optional[int], Optional[int], Optional[str]]:
        candidates = []
        for pick_action_id, action_name in enumerate(self.base_planner.actions):
            target = self._action_target(pick_action_id, "pick up the ")
            if not target or not self._instruction_mentions_object(target):
                continue
            find_action_id = self._find_action_id_by_name(
                f"find a {target}"
            ) or self._find_action_id_by_name(
                f"find a {self._base_object_name(target)}"
            )
            if find_action_id is not None:
                candidates.append((
                    self._instruction_action_context_score(target),
                    find_action_id,
                    pick_action_id,
                    target,
                ))
        if candidates:
            candidates.sort(key=lambda item: item[0], reverse=True)
            best = candidates[0]
            if best[0] > -100:
                return best[1], best[2], best[3]
        return None, None, None

    def _repair_cleaning_workflow_before_final_placement(
        self,
        actions: List[int],
        predicted_states: List[str],
    ) -> Tuple[List[int], List[str]]:
        instruction = (self.current_human_instruction or "").lower()
        if (
            not actions
            or not self._is_clean_or_rinse_instruction()
            or any(term in instruction for term in ("slice", "sliced", "piece of", "pieces of"))
            or self._history_has_successful_faucet_cycle()
        ):
            return actions, predicted_states

        destination_prefix, destination_states = self._destination_placement_prefix()
        if not destination_prefix:
            return actions, predicted_states
        target_find_id, target_pick_id, target = self._primary_pick_target_from_instruction()
        sink_find_id = self._find_action_id_by_name("find a Sink")
        put_down_id = self._find_action_id_by_name("put down the object in hand")
        faucet_on_id = self._find_action_id_by_name("turn on the Faucet")
        faucet_off_id = self._find_action_id_by_name("turn off the Faucet")
        if None in {target_find_id, target_pick_id, sink_find_id, put_down_id, faucet_on_id, faucet_off_id}:
            return actions, predicted_states

        hand_action = self._first_hand_action(actions)
        if hand_action == "put" and self._history_indicates_holding_object():
            prefix_actions = [sink_find_id, put_down_id, faucet_on_id, faucet_off_id]
            prefix_states = [
                "Exploration phase: locating the sink for the required cleaning step.",
                f"The {target} is in the sink for cleaning.",
                f"The Faucet is on and cleaning the {target}.",
                f"The Faucet is off and the {target} is clean.",
            ]
        else:
            prefix_actions = [
                target_find_id,
                target_pick_id,
                sink_find_id,
                put_down_id,
                faucet_on_id,
                faucet_off_id,
            ]
            prefix_states = [
                "Exploration phase: locating the target object that must be cleaned.",
                f"I am holding the {target}.",
                "Exploration phase: locating the sink for the required cleaning step.",
                f"The {target} is in the sink for cleaning.",
                f"The Faucet is on and cleaning the {target}.",
                f"The Faucet is off and the {target} is clean.",
            ]

        logger.info(
            "Rewriting clean/rinse plan to perform the required faucet workflow for '%s'",
            target,
        )
        return (
            prefix_actions + [target_find_id, target_pick_id] + destination_prefix,
            prefix_states
            + [
                "Exploration phase: locating the cleaned target object.",
                f"I am holding the cleaned {target}.",
            ]
            + destination_states,
        )

    @staticmethod
    def _text_mentions_object_alias(text: str, object_name: str) -> bool:
        text = str(text or "").lower().replace("-", " ")
        base_name = re.sub(r'_\d+$', '', str(object_name or '').strip())
        spaced = re.sub(r"(?<!^)(?=[A-Z])", " ", base_name).lower()
        aliases = {spaced, spaced.replace(" ", ""), base_name.lower()}
        aliases.update({
            "Bowl": {"bowl", "glass bowl", "round eating vessel", "container for rice"},
            "Plate": {"plate", "eating surface"},
            "Cup": {
                "cup",
                "glass",
                "beverage glass",
                "glass cup",
                "transparent beverage holder",
                "drinking vessel",
            },
            "Mug": {"mug", "glass"},
            "Pan": {"pan"},
            "Pot": {"pot"},
            "Vase": {"vase", "jar", "glass jar", "vessel"},
            "Pencil": {"pencil", "graphite writing tool", "graphite-writing tool"},
            "Pen": {"pen"},
            "Cloth": {"cloth", "rag", "red rag", "cleaning rag", "cleaning cloth"},
            "HandTowel": {"hand towel", "towel"},
            "Watch": {"watch"},
            "Spoon": {"spoon"},
            "SoapBar": {"soap", "wet soap", "soap bar", "bar of soap"},
            "RemoteControl": {"remote", "remote control", "channel changer"},
            "ButterKnife": {"butter knife", "cutting tool for butter"},
            "Apple": {"apple", "red crisp fruit"},
        }.get(base_name, set()))
        return any(re.search(rf"\b{re.escape(alias)}s?\b", text) for alias in aliases if alias)

    def _pickable_target_from_phrase(self, phrase: str) -> Tuple[Optional[int], Optional[int], Optional[str]]:
        candidates = []
        for pick_action_id, _ in enumerate(self.base_planner.actions):
            target = self._action_target(pick_action_id, "pick up the ")
            if not target or not self._text_mentions_object_alias(phrase, target):
                continue
            find_action_id = self._find_action_id_by_name(
                f"find a {target}"
            ) or self._find_action_id_by_name(
                f"find a {self._base_object_name(target)}"
            )
            if find_action_id is not None:
                candidates.append((len(self._base_object_name(target)), find_action_id, pick_action_id, target))
        if not candidates:
            return None, None, None
        candidates.sort(reverse=True)
        _, find_action_id, pick_action_id, target = candidates[0]
        return find_action_id, pick_action_id, target

    def _containment_targets_from_instruction(
        self,
    ) -> Tuple[Optional[int], Optional[int], Optional[str], Optional[int], Optional[int], Optional[str]]:
        instruction = (self.current_human_instruction or "").lower().replace("-", " ")
        patterns = [
            r"\b(?P<secondary>[a-z ]{2,40}?)\s+(?:inside\s+of|inside|in|on)\s+(?:the\s+|a\s+|an\s+)?(?P<main>[a-z ]{2,30}?)\s+(?:to|onto|on top of|on|at)\s+(?:the\s+|a\s+|an\s+)?[a-z ]{2,40}\b",
            r"\b(?P<main>[a-z ]{2,40}?)\s+with\s+(?:the\s+|a\s+|an\s+)?(?P<secondary>[a-z ]{2,40}?)\s+(?:inside\s+of\s+it|inside\s+it|in\s+it|on\s+it)\b",
            r"\b(?P<main>[a-z ]{2,40}?)\s+with\s+(?:the\s+|a\s+|an\s+)?(?P<secondary>[a-z ]{2,40}?)\s+(?:inside|in|on)\b",
            r"\b(?P<secondary>[a-z ]{2,40}?)\s+(?:inside\s+of|inside|in|on)\s+(?:the\s+|a\s+|an\s+)?(?P<main>[a-z ]{2,40}?)\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, instruction)
            if not match:
                continue
            main_phrase = match.group("main")
            secondary_phrase = match.group("secondary")
            main_phrase = re.sub(
                r"^\s*(?:place|put away|put|move|set|position|arrange)\s+",
                "",
                main_phrase,
            )
            main_phrase = re.split(r"\s+(?:to|onto|on top of|at)\s+", main_phrase, maxsplit=1)[0]
            secondary_phrase = re.sub(
                r"^\s*(?:place|put away|put|move|set|position|arrange)\s+",
                "",
                secondary_phrase,
            )
            main_find_id, main_pick_id, main_target = self._pickable_target_from_phrase(main_phrase)
            sec_find_id, sec_pick_id, sec_target = self._pickable_target_from_phrase(secondary_phrase)
            if main_find_id is not None and sec_find_id is not None and main_target != sec_target:
                return main_find_id, main_pick_id, main_target, sec_find_id, sec_pick_id, sec_target
        return None, None, None, None, None, None

    def _repair_containment_workflow(
        self,
        actions: List[int],
        predicted_states: List[str],
    ) -> Tuple[List[int], List[str]]:
        (
            main_find_id,
            main_pick_id,
            main_target,
            sec_find_id,
            sec_pick_id,
            sec_target,
        ) = self._containment_targets_from_instruction()
        if main_find_id is None or sec_find_id is None:
            return actions, predicted_states

        destination_prefix, destination_states = self._destination_placement_prefix()
        put_down_id = self._find_action_id_by_name("put down the object in hand")
        if not destination_prefix or put_down_id is None:
            return actions, predicted_states
        main_open_id = self._destination_open_action_id(main_target)
        main_needs_open = (
            main_open_id is not None
            and self._base_object_name(main_target) in {"Box", "Safe"}
            and not self._history_receptacle_is_open(main_target)
        )
        assemble_into_main_actions = [main_find_id]
        assemble_into_main_states = [f"Exploration phase: locating the {main_target}."]
        if main_needs_open:
            assemble_into_main_actions.append(main_open_id)
            assemble_into_main_states.append(f"The {self._base_object_name(main_target)} is open for containment.")
        assemble_into_main_actions.append(put_down_id)
        assemble_into_main_states.append(f"The {sec_target} is now inside/on the {main_target}.")

        if self._latest_task_progress() > 0 and not self._history_indicates_holding_object():
            logger.info(
                "Continuing completed containment by moving '%s' to the final destination",
                main_target,
            )
            return (
                [main_find_id, main_pick_id] + destination_prefix,
                [
                    f"Exploration phase: locating the {main_target} after containment is complete.",
                    f"I am holding the {main_target} containing the {sec_target}.",
                ] + destination_states,
            )

        if self._history_indicates_holding_object():
            held = self._held_object_type_from_history()
            if held and self._base_object_name(held) == self._base_object_name(main_target):
                logger.info(
                    "Putting down held main container '%s' before building containment",
                    main_target,
                )
                return (
                    [put_down_id, sec_find_id, sec_pick_id]
                    + assemble_into_main_actions
                    + [main_find_id, main_pick_id]
                    + destination_prefix,
                    [
                        f"The held {main_target} is placed down so the hand is free.",
                        f"Exploration phase: locating the {sec_target}.",
                        f"I am holding the {sec_target}.",
                    ]
                    + assemble_into_main_states
                    + [
                        f"Exploration phase: locating the assembled {main_target}.",
                        f"I am holding the {main_target} containing the {sec_target}.",
                    ] + destination_states,
                )
            return actions, predicted_states

        logger.info(
            "Rewriting containment plan as secondary-first assembly: '%s' into/on '%s'",
            sec_target,
            main_target,
        )
        return (
            [sec_find_id, sec_pick_id]
            + assemble_into_main_actions
            + [main_find_id, main_pick_id]
            + destination_prefix,
            [
                f"Exploration phase: locating the {sec_target}.",
                f"I am holding the {sec_target}.",
            ]
            + assemble_into_main_states
            + [
                f"Exploration phase: locating the assembled {main_target}.",
                f"I am holding the {main_target} containing the {sec_target}.",
            ] + destination_states,
        )

    def _repair_cleaned_object_final_placement(
        self,
        actions: List[int],
        predicted_states: List[str],
    ) -> Tuple[List[int], List[str]]:
        instruction = (self.current_human_instruction or "").lower()
        if (
            not actions
            or not self._is_clean_or_rinse_instruction()
            or any(term in instruction for term in ("slice", "sliced", "piece of", "pieces of"))
            or self._latest_task_progress() <= 0
            or self._history_indicates_holding_object()
            or not self._history_has_successful_faucet_cycle()
        ):
            return actions, predicted_states

        destination_prefix, destination_states = self._destination_placement_prefix()
        if not destination_prefix:
            return actions, predicted_states
        target_find_id, target_pick_id, target = self._primary_pick_target_from_instruction()
        if target_find_id is None or target_pick_id is None:
            return actions, predicted_states

        first_hand_action = self._first_hand_action(actions)
        if first_hand_action == "pick":
            return actions, predicted_states

        logger.info(
            "Rewriting post-clean plan to retrieve '%s' and place it at the final destination",
            target,
        )
        return (
            [target_find_id, target_pick_id] + destination_prefix,
            [
                "Exploration phase: locating the cleaned target object after washing.",
                f"I am holding the cleaned {target}.",
            ] + destination_states,
        )

    def _pickup_action_from_text(self, text: str) -> Tuple[Optional[int], Optional[str]]:
        text_key = self._canonical_text(text or "")
        matches = []
        for action_id, action_name in enumerate(self.base_planner.actions):
            target = self._action_target(action_id, "pick up the ")
            if not target:
                continue
            target_key = self._canonical_text(target)
            if target_key and target_key in text_key and self._instruction_mentions_object(target):
                matches.append((action_id, target))
        if len(matches) == 1:
            return matches[0]
        return None, None

    def _repair_putdown_predicted_as_pickup(
        self,
        actions: List[int],
        predicted_states: List[str],
    ) -> Tuple[List[int], List[str]]:
        repaired_actions = list(actions)
        repaired_states = list(predicted_states)
        holding_re = re.compile(r"\b(?:i am|robot is|the robot is|agent is) holding\b", re.IGNORECASE)
        for idx, action_id in enumerate(repaired_actions):
            action_name = str(self.base_planner.actions[action_id]).strip().lower() if 0 <= action_id < len(self.base_planner.actions) else ""
            if not (action_name.startswith("put down ") or action_name.startswith("drop ")):
                continue
            predicted_state = repaired_states[idx] if idx < len(repaired_states) else ""
            if not holding_re.search(predicted_state):
                continue
            pickup_action_id, target = self._pickup_action_from_text(predicted_state)
            if pickup_action_id is None:
                continue
            logger.info(
                "Replacing '%s' with '%s' because predicted state indicates holding",
                self.base_planner.actions[action_id],
                self.base_planner.actions[pickup_action_id],
            )
            repaired_actions[idx] = pickup_action_id
            repaired_states[idx] = f"I am holding the {target}."
        return repaired_actions, repaired_states

    def _repair_actions_from_feedback(
        self,
        actions: Any,
        predicted_states: List[str],
    ) -> Tuple[Any, List[str]]:
        if not isinstance(actions, list) or not actions:
            return actions, predicted_states

        put_down_action_id = self._find_action_id_by_name("put down the object in hand")
        first_action_name = (
            str(self.base_planner.actions[actions[0]]).strip().lower()
            if isinstance(actions[0], int) and 0 <= actions[0] < len(self.base_planner.actions)
            else ""
        )
        if (
            put_down_action_id is not None
            and not self._history_indicates_holding_object()
            and self._first_hand_action(actions) == "put"
        ):
            logger.info(
                "Removing hand-empty placement actions before replanning from object search"
            )
            filtered_actions = []
            filtered_states = []
            for idx, action_id in enumerate(actions):
                action_name = (
                    str(self.base_planner.actions[action_id]).strip().lower()
                    if isinstance(action_id, int) and 0 <= action_id < len(self.base_planner.actions)
                    else ""
                )
                if action_name.startswith("put down ") or action_name.startswith("drop "):
                    continue
                filtered_actions.append(action_id)
                filtered_states.append(
                    predicted_states[idx] if idx < len(predicted_states) else "Exploring environment."
                )
            actions = filtered_actions
            predicted_states = filtered_states
            if not actions:
                return actions, predicted_states
            first_action_name = (
                str(self.base_planner.actions[actions[0]]).strip().lower()
                if isinstance(actions[0], int) and 0 <= actions[0] < len(self.base_planner.actions)
                else ""
            )

        if (
            put_down_action_id is not None
            and self._history_indicates_holding_object()
            and self._first_hand_action(actions) == "pick"
            and not self._is_keep_holding_action_goal()
        ):
            first_action = self.base_planner.actions[actions[0]] if 0 <= actions[0] < len(self.base_planner.actions) else str(actions[0])
            placement_prefix, placement_states = self._destination_placement_prefix()
            if placement_prefix:
                logger.info(
                    "Prioritizing goal placement before plan starting with '%s' while hand is occupied",
                    first_action,
                )
                if self._is_counted_instruction() and self._is_visual_descriptor_task():
                    predicted_states = placement_states
                    actions = placement_prefix
                else:
                    predicted_states = placement_states + list(predicted_states)
                    actions = placement_prefix + actions
            else:
                logger.info(
                    "Prepending hand-clear action '%s' before plan starting with '%s'",
                    self.base_planner.actions[put_down_action_id],
                    first_action,
                )
                predicted_states = [
                    "The currently held object has been placed so the gripper is empty."
                ] + list(predicted_states)
                actions = [put_down_action_id] + actions

        last_feedback = ""
        if getattr(self.base_planner, "episode_act_feedback", None):
            last_entry = self.base_planner.episode_act_feedback[-1]
            if len(last_entry) >= 2:
                last_feedback = str(last_entry[1])

        actions, predicted_states = self._repair_mismatched_pick_after_find(actions, predicted_states)
        actions, predicted_states = self._repair_putdown_predicted_as_pickup(actions, predicted_states)
        actions, predicted_states = self._repair_instruction_alias_action_targets(actions, predicted_states)
        actions, predicted_states = self._repair_visual_alias_action_targets(actions, predicted_states)
        actions, predicted_states = self._repair_keep_holding_goal_actions(actions, predicted_states)
        actions, predicted_states = self._repair_drop_for_placement(actions, predicted_states)
        actions, predicted_states = self._repair_drawer_destination_actions(actions, predicted_states)
        actions, predicted_states = self._repair_held_object_destination_priority(actions, predicted_states)
        actions, predicted_states = self._repair_open_without_navigation(actions, predicted_states)
        actions, predicted_states = self._remove_redundant_open_actions(actions, predicted_states)
        actions, predicted_states = self._repair_toggle_without_navigation(actions, predicted_states)
        actions, predicted_states = self._repair_after_feedback_container_navigation(actions, predicted_states)
        actions, predicted_states = self._repair_containment_workflow(actions, predicted_states)
        actions, predicted_states = self._repair_counted_progress_drop(actions, predicted_states)
        actions, predicted_states = self._repair_counted_source_navigation(actions, predicted_states)
        actions, predicted_states = self._repair_cleaning_workflow_before_final_placement(actions, predicted_states)
        actions, predicted_states = self._repair_cleaned_object_final_placement(actions, predicted_states)
        actions, predicted_states = self._limit_counted_visual_transfer_cycle(actions, predicted_states)
        actions, predicted_states = self._repair_counted_destination_repickup(actions, predicted_states)
        actions, predicted_states = self._repair_closed_receptacle_put_feedback(
            actions, predicted_states, last_feedback
        )
        actions, predicted_states = self._repair_receptacle_instance_put_feedback(
            actions, predicted_states, last_feedback
        )
        actions, predicted_states = self._repair_missing_destination_before_putdown(actions, predicted_states)

        find_action_id, pick_action_id, target_name, container_name = self._container_repair_from_feedback(last_feedback)
        if find_action_id is None:
            return self._repair_indexed_pick_after_find(actions, predicted_states)
        instruction = (self.current_human_instruction or "").lower()
        if (
            target_name
            and not self._instruction_mentions_object(target_name)
            and not any(term in instruction for term in (" object", " thing", " item"))
        ):
            logger.info(
                "Ignoring feedback-directed pickup for unrelated target '%s'",
                target_name,
            )
            return actions, predicted_states

        target_find_action_id = (
            self._find_action_id_by_name(f"find a {target_name}") if target_name else None
        )
        remaining_actions = list(actions)
        remaining_states = list(predicted_states)
        repaired_actions = []
        repaired_states = []

        if remaining_actions and remaining_actions[0] == find_action_id:
            repaired_actions.append(remaining_actions.pop(0))
            repaired_states.append(
                remaining_states.pop(0) if remaining_states else "Exploring environment."
            )
        else:
            first_action = self.base_planner.actions[remaining_actions[0]] if remaining_actions and 0 <= remaining_actions[0] < len(self.base_planner.actions) else str(remaining_actions[0] if remaining_actions else "")
            find_action = self.base_planner.actions[find_action_id]
            logger.info(
                "Prepending feedback-directed navigation action '%s' before '%s'",
                find_action,
                first_action,
            )
            repaired_actions.append(find_action_id)
            repaired_states.append(
                "Exploring environment; following feedback to navigate to the referenced receptacle."
            )

        if container_name:
            open_action_id = self._find_action_id_by_name(
                f"open the {container_name}"
            ) or self._find_action_id_by_name(
                f"open the {self._base_object_name(container_name)}"
            )
            container_entry = self._visible_entry_for_alias(container_name)
            container_known_open = (
                bool(container_entry.get("is_open"))
                if container_entry is not None
                else self._history_receptacle_is_open(container_name)
            )
            if open_action_id is not None and not container_known_open:
                logger.info(
                    "Opening feedback-directed container '%s' before retrying pickup",
                    self.base_planner.actions[open_action_id],
                )
                repaired_actions.append(open_action_id)
                repaired_states.append(f"The {self._base_object_name(container_name)} is open for pickup.")
                last_find_target = self._last_successful_find_target()
                target_refind_action_id = None
                if last_find_target and target_name and self._base_object_name(last_find_target) == self._base_object_name(target_name):
                    target_refind_action_id = self._find_action_id_by_name(f"find a {last_find_target}")
                if target_refind_action_id is None:
                    target_refind_action_id = target_find_action_id
                if target_refind_action_id is not None:
                    logger.info(
                        "Re-finding target '%s' after opening feedback-directed container",
                        self.base_planner.actions[target_refind_action_id],
                    )
                    repaired_actions.append(target_refind_action_id)
                    repaired_states.append("Exploration phase: re-locating the target after opening its container.")

        while remaining_actions and remaining_actions[0] in {target_find_action_id, pick_action_id}:
            remaining_actions.pop(0)
            if remaining_states:
                remaining_states.pop(0)

        if pick_action_id is not None:
            logger.info(
                "Following container feedback by retrying pickup '%s' after navigation",
                self.base_planner.actions[pick_action_id],
            )
            repaired_actions.append(pick_action_id)
            repaired_states.append(f"I am holding the {target_name}.")

            if remaining_actions:
                next_action = str(self.base_planner.actions[remaining_actions[0]]).strip().lower() if 0 <= remaining_actions[0] < len(self.base_planner.actions) else ""
                if next_action.startswith("put down ") or next_action.startswith("drop "):
                    logger.info(
                        "Dropping stale immediate placement after feedback-directed pickup: '%s'",
                        self.base_planner.actions[remaining_actions[0]],
                    )
                    remaining_actions.pop(0)
                    if remaining_states:
                        remaining_states.pop(0)

        repaired_actions.extend(remaining_actions)
        repaired_states.extend(remaining_states)
        repaired_actions, repaired_states = self._repair_putdown_predicted_as_pickup(
            repaired_actions, repaired_states
        )
        return self._repair_indexed_pick_after_find(repaired_actions, repaired_states)

    def should_skip_worldmind(self) -> bool:
        if self.current_predicted_state and self._is_exploration_phase(self.current_predicted_state):
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

            return actions, predicted_states

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


        prompt_preview = self._build_worldmind_prompt(user_instruction)
        self.current_agent_input_prompt = prompt_preview


        print_agent_input(prompt_preview, self.detailed_output)



        original_process_prompt = self.base_planner.process_prompt
        worldmind_prompt = prompt_preview
        self.base_planner.process_prompt = lambda user_instruction, prev_act_feedback=[]: worldmind_prompt

        try:

            if self.enable_multi_simulation and self.multi_simulation_manager and self.multi_simulation_manager.enable:
                actions, output_text = self._act_with_multi_simulation(observation, user_instruction)
            else:

                actions, output_text = self.base_planner.act(observation, user_instruction)
        finally:

            self.base_planner.process_prompt = original_process_prompt


        self.current_agent_output = output_text


        print_agent(output_text, self.detailed_output)


        self.current_visual_state_description = self._extract_visual_state_description(output_text)


        _, predicted_states = self._json_to_action_with_prediction(output_text)
        actions, predicted_states = self._repair_actions_from_feedback(actions, predicted_states)
        actions, predicted_states = self._apply_visual_observation_policy(actions, predicted_states)
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




        relevant_knowledge = []
        if self.enable_world_knowledge:
            if self._cached_world_knowledge is None:

                self._cached_world_knowledge = self.world_knowledge_manager.retrieve_knowledge(user_instruction)
                logger.info(f"[DEBUG] Retrieved and cached {len(self._cached_world_knowledge)} world knowledge entries at episode start")
            relevant_knowledge = self._cached_world_knowledge
            logger.debug(f"[DEBUG] Using cached {len(relevant_knowledge)} world knowledge entries")

        prompt += f'\n\n## Now the human instruction is: {user_instruction}.'
        prompt += f'\n\n{get_instruction_workflow_guidance(user_instruction)}'
        if self.enable_visual_observation_summary and self.current_visual_observation_summary:
            prompt += (
                "\n\n## Current Visual Observation Summary\n"
                f"{self.current_visual_observation_summary}\n"
                "- For color/material/appearance goals, use these observed current-view facts before interacting.\n"
                "- If a visible candidate does not match the requested appearance, do not pick or use it; inspect another indexed instance or plausible canonical class.\n"
                "- The summary is only the current camera view, so after a `find` action, re-evaluate the updated summary before choosing the next interaction."
            )


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


        if self.enable_location_knowledge_accumulation:
            location_knowledge = self.get_accumulated_location_knowledge()
            if location_knowledge:
                prompt += f'\n\n## Discovered Location Information (from previous navigation actions):\n{location_knowledge}'




        if len(planner.episode_act_feedback) > 0:
            prompt += '\n The action history:\n'
            for i, action_feedback in enumerate(planner.episode_act_feedback):
                prompt += '\nStep {}, action id {}, {}, env feedback: {}'.format(
                    i, action_feedback[0], planner.actions[action_feedback[0]], action_feedback[1]
                )

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

        if not self.use_state_summarizer:
            return {'state_before_action': '', 'state_after_action': ''}

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

    def describe_feedback_state(self, env_feedback: str = "", step_info: Optional[Dict] = None) -> str:
        step_info = step_info or {}
        parts = []
        if env_feedback:
            parts.append(f"Environment feedback: {env_feedback}")
        if 'last_action_success' in step_info:
            parts.append(f"Last action success: {bool(step_info.get('last_action_success'))}")
        if 'task_progress' in step_info:
            parts.append(f"Task progress: {step_info.get('task_progress')}")
        if 'task_success' in step_info:
            parts.append(f"Task success: {bool(step_info.get('task_success'))}")
        object_states = step_info.get('object_states') or {}
        visible_objs = object_states.get('visible_objs') or []
        if visible_objs:
            unique_visible = sorted(set(str(obj) for obj in visible_objs))
            parts.append("Visible object types: " + ", ".join(unique_visible[:30]))
        return "\n".join(parts) if parts else "No structured feedback available."

    def discriminate_feedback(
        self,
        env_feedback: str = "",
        action_index: int = 0,
        action_id: int = None,
        action_description: str = None,
        step_info: Optional[Dict] = None
    ) -> Dict:
        if not self.enable_discrimination:
            return {"match": True, "reason": "Discrimination disabled"}

        if action_index < len(self.current_predicted_states):
            predicted_state = self.current_predicted_states[action_index]
        else:
            predicted_state = self.current_predicted_state

        if not predicted_state:
            return {"match": True, "reason": "No prediction available"}
        if self._is_exploration_phase(predicted_state):
            return {"match": True, "reason": "Exploration prediction skipped"}

        step_info = step_info or {}
        last_action_success = bool(step_info.get('last_action_success', True))
        actual_action_id = action_id if action_id is not None else self.current_action_id
        action_desc = action_description if action_description is not None else (self.current_action_description or "")

        if not last_action_success:
            result = {
                "match": False,
                "reason": f"Action failed according to environment feedback: {env_feedback or 'no feedback'}"
            }
            self.worldmind_stats["discrimination_mismatches"] += 1
        else:
            result = {
                "match": True,
                "reason": "Action executed successfully according to Alfred environment feedback"
            }
            self.worldmind_stats["discrimination_matches"] += 1

        self.discrimination_history.append({
            "action_id": actual_action_id,
            "action_description": action_desc,
            "predicted_state": predicted_state,
            "actual_state_summary": self.describe_feedback_state(env_feedback, step_info),
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
        predicted_state = self.current_predicted_state or ""
        instruction = human_instruction or self.current_human_instruction or "Not specified."

        actual_state_before = self.get_state_description(state_before_action)
        actual_state_after = self.get_state_description(state_after_action)

        self.current_worldmind_feedback = {
            "observation_before": actual_state_before,
            "predicted_state": predicted_state,
            "env_feedback": env_feedback,
            "reflexion": None
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

        experience_entries = reflection_result.get("experience_entry", [])


        if experience_entries:
            reflexion_text = "; ".join(experience_entries) if isinstance(experience_entries, list) else str(experience_entries)
            self.current_worldmind_feedback["reflexion"] = reflexion_text

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

        if 'action_id' in info:
            self.success_experience_manager.update_trajectory(info)

        progress_parts = []
        if 'task_progress' in info:
            try:
                progress_parts.append(f"[Task Progress] {float(info['task_progress']):.3f}")
            except (TypeError, ValueError):
                progress_parts.append(f"[Task Progress] {info['task_progress']}")
        if 'task_success' in info:
            progress_parts.append(f"[Task Success] {bool(info['task_success'])}")
        if progress_parts:
            original_feedback = info.get('env_feedback', '')
            progress_feedback = " ".join(progress_parts)
            info['env_feedback'] = f"{original_feedback}\n{progress_feedback}" if original_feedback else progress_feedback


        if self.current_worldmind_feedback:
            worldmind_parts = []
            if self.current_worldmind_feedback.get("observation_before"):
                worldmind_parts.append(f"[Observation] {self.current_worldmind_feedback['observation_before']}")
            if self.current_worldmind_feedback.get("predicted_state"):
                worldmind_parts.append(f"[Predicted State] {self.current_worldmind_feedback['predicted_state']}")
            if self.current_worldmind_feedback.get("reflexion"):
                worldmind_parts.append(f"[Reflexion] {self.current_worldmind_feedback['reflexion']}")

            if worldmind_parts:
                original_feedback = info.get('env_feedback', '')
                worldmind_feedback = "\n".join(worldmind_parts)
                info['env_feedback'] = f"{original_feedback}\n{worldmind_feedback}" if original_feedback else worldmind_feedback


        self.current_worldmind_feedback = {}


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
