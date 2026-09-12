import torchvision.transforms as T
from timm import create_model
from transformers import AutoModel
from loguru import logger

from src.transform import CLAHETransform

from wildlife_tools.features import *
from wildlife_tools.similarity import MatchLOFTR, MatchLightGlue, CosineSimilarity
from wildlife_tools.similarity.wildfusion import SimilarityPipeline, WildFusion
from wildlife_tools.similarity.calibration import IsotonicCalibration


class ModelFactory:
    """Factory for creating wildlife re-identification models."""
    
    @staticmethod
    def create(cfg):
        """Create model based on configuration."""
        model_type = cfg.type
        
        if model_type == "local":
            matchers = ModelFactory._create_local_matchers(cfg)
        elif model_type == "global":
            matchers = ModelFactory._create_global_matchers(cfg)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        logger.info(f"Created {len(matchers)} matcher pipelines")
        return WildFusion(calibrated_pipelines=matchers)
    
    def _local_matcher_builder(name, transform, calibration=True):
        if name in ["superpoint", "aliked", "disk", "sift"]:
            matcher = MatchLightGlue(features=name)
            extractor_name = f"{name.title()}Extractor" if name != "superpoint" else "SuperPointExtractor"
            extractor = globals()[extractor_name]()
        elif name == "loftr":
            matcher = MatchLOFTR(pretrained=name)
            extractor = None
        
        calibration = IsotonicCalibration() if calibration else None
        pipeline = SimilarityPipeline(
            matcher=matcher,
            extractor=extractor,
            transform=transform,
            calibration=calibration,
        )
        
        return pipeline
    
    @staticmethod
    def _create_local_matchers(cfg):
        """Create local feature matchers."""
        logger.info("Creating local feature matchers")
        
        enabled = cfg.enabled
        calibration = cfg.get("calibration", True)
        use_clahe = cfg.get("use_clahe", False)
        pipeline_list = []
        for name in enabled:
            if name == "loftr":
                transform = T.Compose([
                    T.Resize([128, 256]),
                    T.Grayscale(),
                    CLAHETransform() if use_clahe else T.Lambda(lambda x: x),
                    T.ToTensor(),
                ])
            else:
                transform = T.Compose([
                    T.Resize(cfg.img_size),
                    CLAHETransform() if use_clahe else T.Lambda(lambda x: x),
                    T.ToTensor()
                ])
            pipeline = ModelFactory._local_matcher_builder(name, transform, calibration)
            pipeline_list.append(pipeline)

        return pipeline_list
    
    @staticmethod
    def _create_global_matchers(cfg):
        """Create global feature matchers."""
        model_name = cfg.name
        logger.info(f"Creating global feature matcher: {model_name}")
        
        if model_name == "hf-hub:BVRA/wildlife-mega-L-384":
            model = create_model(model_name, pretrained=True)
        elif model_name == "conservationxlabs/miewid-msv3":
            model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        else:
            raise ValueError(f"Unknown model_name: {model_name}")
        
        use_clahe = cfg.get("use_clahe", False)
        matchers = [
            SimilarityPipeline(
                matcher=CosineSimilarity(),
                extractor=DeepFeatures(model, device=cfg.get("device", "cuda")),
                transform=T.Compose([
                    T.Resize(cfg.img_size),
                    CLAHETransform() if use_clahe else T.Lambda(lambda x: x),
                    T.ToTensor(),
                    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ]),
            ),
        ]
        
        return matchers