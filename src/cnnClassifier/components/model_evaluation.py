import os
import json
import tensorflow as tf
from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier import logger
from pathlib import Path


class Evaluation:
    def __init__(self, config, label_mode='int'):
        self.config = config
        # Load the trained model
        self.model = tf.keras.models.load_model(config.trained_model_path)
        
        # Load the test dataset with integer labels to match sparse_categorical_crossentropy
        self.test_data = tf.keras.preprocessing.image_dataset_from_directory(
            config.test_data_path,
            image_size=(config.params_image_size[0], config.params_image_size[1]),
            batch_size=config.params_batch_size,
            label_mode=label_mode,  # Use integer labels for sparse loss
            shuffle=False      # Important: don't shuffle for evaluation
        )
        self.scores = {}

        # Optional: print class mapping
        class_names = self.test_data.class_names
        logger.info(f"Class indices mapping: {dict(zip(class_names, range(len(class_names))))}")

    def evaluate_model(self):
        logger.info("🔍 Evaluating the model on test dataset...")
        loss, accuracy = self.model.evaluate(self.test_data)
        self.scores = {"loss": float(loss), "accuracy": float(accuracy)}
        logger.info(f"Model Evaluation Scores: {self.scores}")

    def save_score(self):
        os.makedirs(os.path.dirname(self.config.metric_file_name), exist_ok=True)
        with open(self.config.metric_file_name, "w") as f:
            json.dump(self.scores, f, indent=4)
        logger.info(f"Scores saved to: {self.config.metric_file_name}")

    def save_model_for_dvc(self):
        """Save model locally so DVC can version-control it"""
        save_path = Path(self.config.model_dvc_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(save_path)   # saves in Keras native format (.keras or .h5)
        logger.info(f"Model saved for DVC tracking at: {save_path}")
        return save_path

    def log_into_mlflow(self):
        import mlflow
        #import mlflow.keras

        logger.info("📦 Logging metrics and model into MLflow...")
        mlflow.log_metrics(self.scores)
        #mlflow.keras.log_model(self.model, "model")
        logger.info("Model & metrics logged into MLflow.")


if __name__ == "__main__":
    try:
        logger.info(">>>> Stage: Evaluation started <<<<")

        config = ConfigurationManager().get_evaluation_config()
        eval = Evaluation(config)

        eval.evaluate_model()
        eval.save_score()
        eval.log_into_mlflow()

        logger.info(">>>> Stage: Evaluation completed <<<<\n\n")

    except Exception as e:
        logger.exception(e)
        raise e
