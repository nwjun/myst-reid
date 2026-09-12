import os
import copy
from pathlib import Path

import pandas as pd
import torchvision.transforms as T
from loguru import logger

from wildlife_tools.data import ImageDataset
from wildlife_datasets.datasets import SeaTurtleIDHeads



WILDLIFEDATASET_REGISTRY = {
    "SeaTurtleIDHeads": SeaTurtleIDHeads,
}


class SubsettableImageDataset(ImageDataset):
    """Extends ImageDataset to support index-based subsetting while preserving label consistency."""
    
    def subset(self, indices):
        """Returns a new instance containing only specified indices, sharing the label mapping."""
        new_ds = copy.copy(self)
        new_ds.metadata = self.metadata.iloc[indices].reset_index(drop=True)
        new_ds.labels = self.labels[indices]
        return new_ds


class DataManager:
    """Manages dataset loading and preprocessing."""
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.root = Path(cfg.root)
        self.name = cfg.name
        self.seed = cfg.get("seed", 42)

    def load_data(self):
        """Load data"""
        logger.info(f"Loading data: {self.name}")

        full_df = self._load_data(self.name)
        full_df["split"] = "query"
        full_df["abs_diff_days"] = 0
        full_df["role"] = "genuine"
        
        # Filter existing paths
        full_df = full_df[full_df["path"].apply(lambda x: os.path.exists(x))]
        
        logger.info(f"Loaded {len(full_df)} images across {full_df['orientation'].nunique()} orientations")
        return full_df

    def _load_data(self, dataset_name):
        """Load dataset based on its type."""
        if dataset_name in WILDLIFEDATASET_REGISTRY:
            return self._load_wildlife_data(dataset_name)
        else:
            return self._load_ums_data(dataset_name)
    
    def _load_wildlife_data(self, dataset_name, test_split_only=True):
        """Load Wildlife dataset format."""
        data_dir = self.root / dataset_name
        full_df = pd.read_csv(data_dir / "metadata.csv")
        
        DataClass = WILDLIFEDATASET_REGISTRY[self.name]
        DataClass.get_data(str(data_dir))
        data = DataClass(str(data_dir))
        
        if self.name == "SeaTurtleIDHeads":
            orientation_df = data.df[["path", "orientation"]]
            orientation_df["path"] = orientation_df["path"].apply(
                lambda x: f"images/{x.split('/')[-2]}/{x.split('/')[-1]}"
            )
            full_df["path"] = full_df["path"].apply(
                lambda x: f"{x.split('_')[0]}.{x.split('.')[-1]}"
            )
            full_df = pd.merge(full_df, orientation_df, on=["path"], how="left")

        if test_split_only:
            full_df = full_df[full_df["split"] == "test"]

        full_df = full_df.rename(columns={"date": "timestamp"}, errors="ignore")
        full_df["timestamp"] = pd.to_datetime(full_df["timestamp"], errors="coerce")

        full_df = full_df.dropna(subset=["orientation"])
        full_df["path"] = full_df["path"].apply(lambda x: str(data_dir / x))
        
        return full_df
    
    def _load_ums_data(self, dataset_name):
        """Load UMS dataset format."""
        data_dir = self.root / dataset_name
        full_df = pd.read_csv(data_dir / "metadata.csv")
        
        date_format = "%d/%m/%Y %H:%M" if dataset_name == "mantanani" else "%Y-%m-%d %H:%M:%S"
        full_df["timestamp"] = pd.to_datetime(full_df["timestamp"], format=date_format)
        full_df = full_df.dropna(subset=["orientation"])
        full_df = full_df[["path", "identity", "orientation", "timestamp"]]
        
        if dataset_name.startswith("merged"):
            data_dir = self.root
        full_df["path"] = full_df["path"].apply(lambda x: str(data_dir / x))
        
        return full_df
    
    def load_imposter_data(self, dataset_names):
        """Load external imposter data."""
        logger.info(f"Loading imposter data from: {dataset_names}")
        
        imposter_df = pd.DataFrame()
        for ds_name in dataset_names:
            if ds_name == self.name:
                continue

            temp_df = self._load_data(ds_name)
            imposter_df = pd.concat([imposter_df, temp_df])
        
        imposter_df["split"] = "query"
        imposter_df["abs_diff_days"] = 0
        imposter_df["role"] = "imposter_ex"
        
        logger.info(f"Loaded {len(imposter_df)} imposter images")
        return imposter_df
    
    def prepare_gq_split(self, df):
        """Prepare gallery/query split."""
        group_size = df.groupby("identity")["identity"].transform("size")
        is_singleton = group_size < 2
        ns_mask = ~is_singleton
        
        df.loc[is_singleton, "role"] = "imposter_in"

        if "timestamp" in df.columns:
            df.sort_values(by=["identity", "timestamp"], inplace=True)
            gallery_idx = df[ns_mask].groupby("identity")["timestamp"].idxmin()
            df.loc[gallery_idx, "split"] = "gallery"
            
            gallery_ts = df.loc[gallery_idx, ["identity", "timestamp"]].set_index("identity")["timestamp"]
            df.loc[ns_mask, "gallery_timestamp"] = df.loc[ns_mask, "identity"].map(gallery_ts)
            df.loc[ns_mask, "abs_diff_days"] = (
                (df.loc[ns_mask, "timestamp"] - df.loc[ns_mask, "gallery_timestamp"])
                .abs()
                .dt.days
            )
        else:
            gallery_idx = (
                df[ns_mask]
                .groupby("identity", group_keys=False)
                .sample(1, random_state=self.seed)
                .index
            )
            df.loc[gallery_idx, "split"] = "gallery"
        
        return df[ns_mask] if not self.cfg.get("include_imposters", True) else df