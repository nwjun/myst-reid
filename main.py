from pathlib import Path
from datetime import datetime

import hydra
from omegaconf import DictConfig, OmegaConf
from loguru import logger

from src.data import DataManager
from src.models import ModelFactory
from src.evaluation import EvaluationPipeline
from src.utils import setup_logging


@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Main experiment entry point."""
    PROJECT_ROOT = Path(hydra.utils.get_original_cwd())
    cfg.dataset.root = PROJECT_ROOT / cfg.dataset.root

    # Setup logging
    output_dir = Path(cfg.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir / "experiment.log")
    
    # Log configuration
    logger.info("=" * 80)
    logger.info("Starting experiment")
    logger.info("=" * 80)
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    logger.info(f"Output directory: {output_dir}")
    
    # Save configuration
    with open(output_dir / "config.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg))
    
    # Initialize data manager
    logger.info(f"Loading dataset: {cfg.dataset.name}")
    
    data_manager = DataManager(cfg.dataset)
    full_df = data_manager.load_data()

    imposter_df = None
    if cfg.dataset.imposter_datasets:
        imposter_df = data_manager.load_imposter_data(cfg.dataset.imposter_datasets)

    # Initialize model
    logger.info(f"Initializing model: {cfg.model.name}")
    model = ModelFactory.create(cfg.model)
    
    # Run evaluation
    logger.info("Starting evaluation pipeline")
    evaluator = EvaluationPipeline(cfg.evaluation)
    
    if cfg.evaluation.mode == "standard":
        results = evaluator.evaluate(
            model=model,
            full_df=full_df,
            data_manager=data_manager,
            fpir_targets=cfg.evaluation.fpir_targets,
            imposter_df=imposter_df,
        )
    elif cfg.evaluation.mode == "per_view":
        results = evaluator.evaluate_per_view(
            model=model,
            full_df=full_df,
            data_manager=data_manager,
            fpir_targets=cfg.evaluation.fpir_targets,
            imposter_df=imposter_df,
        )
    else:
        raise ValueError(f"Unknown evaluation mode: {cfg.evaluation.mode}")

    results = evaluator.complete_evaluation(
        output_dir=output_dir,
    )

    logger.info("=" * 80)
    logger.info("Experiment completed successfully")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()