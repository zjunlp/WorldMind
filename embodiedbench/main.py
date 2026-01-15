import os
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import yaml

logger = logging.getLogger("EB_logger")
if not logger.hasHandlers():
    formatter = logging.Formatter("[%(asctime)s][%(levelname)s] - %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

link_path = os.path.join(os.path.dirname(__file__), 'envs/eb_habitat/data')
try:
    os.symlink(link_path, 'data')
except FileExistsError:
    pass 

class_names = {
    "eb-alf": "EB_AlfredEvaluator",
    "eb-hab": "EB_HabitatEvaluator",
    "eb-nav": "EB_NavigationEvaluator",
    "eb-man": "EB_ManipulationEvaluator"
}

module_names = {
    "eb-alf": "eb_alfred_evaluator",
    "eb-hab": "eb_habitat_evaluator",
    "eb-nav": "eb_navigation_evaluator",
    "eb-man": "eb_manipulation_evaluator"
}


def get_evaluator(env_name: str, enable_worldmind: bool = False):
    """Get the evaluator class for the specified environment."""
    if env_name not in module_names:
        raise ValueError(f"Unknown environment: {env_name}")
    
    module_name = f"embodiedbench.evaluator.{module_names[env_name]}"
    
    if env_name == "eb-hab" and enable_worldmind:
        evaluator_name = "EB_HabitatEvaluator_WorldMind"
        logger.info("Using WorldMind-enabled Habitat Evaluator")
    elif env_name == "eb-alf" and enable_worldmind:
        evaluator_name = "EB_AlfredEvaluator_WorldMind"
        logger.info("Using WorldMind-enabled Alfred Evaluator")
    elif env_name == "eb-nav" and enable_worldmind:
        evaluator_name = "EB_NavigationEvaluator_WorldMind"
        logger.info("Using WorldMind-enabled Navigation Evaluator")
    else:
        evaluator_name = class_names[env_name]
    
    module = __import__(module_name, fromlist=[evaluator_name])
    return getattr(module, evaluator_name)


@hydra.main(config_path="./configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    logging.getLogger().handlers.clear()
    
    OmegaConf.set_struct(cfg, False)
    
    if 'log_level' not in cfg or cfg.log_level == "INFO":
        logger.setLevel(logging.INFO)
    if 'log_level' in cfg and cfg.log_level == "DEBUG":
        logger.setLevel(logging.DEBUG)

    env_name = cfg.env
    logger.info(f"Evaluating environment: {env_name}")
    
    config_file = f"embodiedbench/configs/{cfg.env}.yaml"
    if not os.path.exists(config_file):
        config_file = f"embodiedbench/configs/eb-hab.yaml"
    
    with open(config_file, 'r') as f:
        base_config = yaml.safe_load(f)

    override_config = {
        k: v for k, v in OmegaConf.to_container(cfg).items() 
        if k != 'env' and v is not None
    }
    
    config = OmegaConf.merge(
        OmegaConf.create(base_config),
        override_config
    )

    config = OmegaConf.to_container(config, resolve=True)
    
    print(config)
    logger.info("Starting evaluation")
    
    enable_worldmind = config.get('enable_worldmind', False)
    if enable_worldmind:
        logger.info("WorldMind mode is enabled")
    
    evaluator_class = get_evaluator(env_name, enable_worldmind=enable_worldmind)
    evaluator = evaluator_class(config)
    evaluator.check_config_valid()
    evaluator.evaluate_main()
    logger.info("Evaluation completed")


if __name__ == "__main__":
    main()
