from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, List

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path

@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_path: Path
    updated_base_model_path: Path
    params_image_size: list
    params_learning_rate: float
    params_include_top: bool
    params_weights: str
    params_classes: int

@dataclass(frozen=True)
class TrainingConfig:
    root_dir: str
    trained_model_path: str
    updated_base_model_path: str
    training_data: str
    params_epochs: int
    params_batch_size: int
    params_is_augmentation: bool
    params_image_size: list
    params_learning_rate: float  # 👈 Add this
    classes: int
    include_top: bool
    weights: str


@dataclass(frozen=True)
class EvaluationConfig:
    path_of_model: Path
    training_data: Path
    all_params: any
    params_image_size: List[int]
    params_batch_size: int
    mlflow_uri: Optional[str] = None  # optional override
