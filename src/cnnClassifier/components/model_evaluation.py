import tensorflow as tf
from pathlib import Path
import mlflow
from urllib.parse import urlparse
from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.utils.common import save_json


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.model = None
        self.valid_generator = None
        self.score = None  # [loss, accuracy]

    def _valid_generator(self):
        datagenerator_kwargs = dict(
            rescale=1.0 / 255,
            validation_split=0.30,
        )
        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear",
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs,
        )

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)

    def evaluation(self):
        self.model = self.load_model(self.config.path_of_model)
        self._valid_generator()
        self.score = self.model.evaluate(self.valid_generator, verbose=1)
        self.save_score()

    def save_score(self):
        scores = {"loss": float(self.score[0]), "accuracy": float(self.score[1])}
        save_json(path=Path("scores.json"), data=scores)

    def log_into_mlflow(self):
        """
        Logs params, metrics and model to MLflow (DagsHub).
        """
        tracking_scheme = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run(run_name="evaluation"):
            # log hyperparams (from params.yaml)
            mlflow.log_params(self.config.all_params)

            # log metrics
            mlflow.log_metrics(
                {"loss": float(self.score[0]), "accuracy": float(self.score[1])}
            )

            # ✅ For DagsHub → only log model file as artifact
            mlflow.log_artifact(
                local_path=str(self.config.path_of_model),
                artifact_path="model"
            )
