import os
import tensorflow as tf

# ✅ Force eager execution
tf.config.run_functions_eagerly(True)
tf.compat.v1.enable_eager_execution()

from cnnClassifier.constants import *
from cnnClassifier.utils.common import read_yaml, create_directories
from cnnClassifier.entity.config_entity import (DataIngestionConfig, PrepareBaseModelConfig, TrainingConfig, EvaluationConfig)


class ConfigurationManager:
    def __init__(
        self,
        config_filepath=CONFIG_FILE_PATH,
        params_filepath=PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])


    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config
    

    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model
        
        create_directories([config.root_dir])

        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            params_image_size=self.params.IMAGE_SIZE,
            params_learning_rate=self.params.LEARNING_RATE,
            params_include_top=self.params.INCLUDE_TOP,
            params_weights=self.params.WEIGHTS,
            params_classes=self.params.CLASSES
        )

        return prepare_base_model_config
    


    def get_training_config(self) -> TrainingConfig:
        training = self.config.training
        prepare_base_model = self.config.prepare_base_model
        params = self.params
        training_data = os.path.join(self.config.data_ingestion.unzip_dir, "dataset_mlops")
        create_directories([
            Path(training.root_dir)
        ])

        training_config = TrainingConfig(
            root_dir=Path(training.root_dir),
            trained_model_path=Path(training.trained_model_path),
            updated_base_model_path=Path(prepare_base_model.updated_base_model_path),
            training_data=Path(training_data),
            params_epochs=params.EPOCHS,
            params_batch_size=params.BATCH_SIZE,
            params_is_augmentation=params.AUGMENTATION,
            params_image_size=params.IMAGE_SIZE,
            params_learning_rate=params.LEARNING_RATE,  # 👈 Add this mapping
            classes=params.CLASSES,
            include_top=params.INCLUDE_TOP,
            weights=params.WEIGHTS

        )

        return training_config

    def get_evaluation_config(self) -> EvaluationConfig:
        config = self.config.evaluation  # 👈 use evaluation section from yaml
        trained_model_path = self.get_latest_model(Path(config["path_of_model"]))
        create_directories([os.path.dirname(config.metric_file_name)])  # ensure dir exists

        eval_config = EvaluationConfig(
            path_of_model=Path(self.config.training.trained_model_path),
            training_data=Path(self.config.data_ingestion.unzip_dir) / "dataset_mlops",
            all_params=self.params,
            params_image_size=self.params.IMAGE_SIZE,
            params_batch_size=self.params.BATCH_SIZE,
            mlflow_uri=None,
            trained_model_path=trained_model_path,
            test_data_path=Path(config.test_data_path),
            metric_file_name=Path(config.metric_file_name)  # 👈 pass scores.json path
        )
        return eval_config
    
    def get_latest_model(self, model_dir: Path) -> Path:
        models = list(model_dir.glob("*.h5"))  # or "*.pt"
        if not models:
            raise FileNotFoundError(f"No model files found in {model_dir}")
        # Sort by modified time and return latest
        latest_model = max(models, key=os.path.getmtime)
        return latest_model
