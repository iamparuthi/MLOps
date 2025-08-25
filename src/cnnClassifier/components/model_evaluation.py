# import os
# import json
# import tensorflow as tf
# from cnnClassifier.config.configuration import ConfigurationManager
# from cnnClassifier import logger
# from pathlib import Path


# class Evaluation:
#     def __init__(self, config, label_mode='int'):
#         self.config = config
#         # Load the trained model
#         self.model = tf.keras.models.load_model(config.trained_model_path)
        
#         # Load the test dataset with integer labels to match sparse_categorical_crossentropy
#         self.test_data = tf.keras.preprocessing.image_dataset_from_directory(
#             config.test_data_path,
#             image_size=(config.params_image_size[0], config.params_image_size[1]),
#             batch_size=config.params_batch_size,
#             label_mode=label_mode,  # Use integer labels for sparse loss
#             shuffle=False      # Important: don't shuffle for evaluation
#         )
#         self.scores = {}

#         # Optional: print class mapping
#         class_names = self.test_data.class_names
#         logger.info(f"Class indices mapping: {dict(zip(class_names, range(len(class_names))))}")

#     def evaluate_model(self):
#         logger.info("🔍 Evaluating the model on test dataset...")
#         loss, accuracy = self.model.evaluate(self.test_data)
#         self.scores = {"loss": float(loss), "accuracy": float(accuracy)}
#         logger.info(f"Model Evaluation Scores: {self.scores}")

#     def save_score(self):
#         os.makedirs(os.path.dirname(self.config.metric_file_name), exist_ok=True)
#         with open(self.config.metric_file_name, "w") as f:
#             json.dump(self.scores, f, indent=4)
#         logger.info(f"Scores saved to: {self.config.metric_file_name}")

#     def save_model_for_dvc(self):
#         """Save model locally so DVC can version-control it"""
#         save_path = Path(self.config.model_dvc_path)
#         save_path.parent.mkdir(parents=True, exist_ok=True)
#         self.model.save(save_path)   # saves in Keras native format (.keras or .h5)
#         logger.info(f"Model saved for DVC tracking at: {save_path}")
#         return save_path

#     def log_into_mlflow(self):
#         import mlflow
#         #import mlflow.keras

#         logger.info("📦 Logging metrics and model into MLflow...")
#         mlflow.log_metrics(self.scores)
#         #mlflow.keras.log_model(self.model, "model")
#         logger.info("Model & metrics logged into MLflow.")


# if __name__ == "__main__":
#     try:
#         logger.info(">>>> Stage: Evaluation started <<<<")

#         config = ConfigurationManager().get_evaluation_config()
#         eval = Evaluation(config)

#         eval.evaluate_model()
#         eval.save_score()
#         eval.log_into_mlflow()

#         logger.info(">>>> Stage: Evaluation completed <<<<\n\n")

#     except Exception as e:
#         logger.exception(e)
#         raise e


import os
import json
import tensorflow as tf
import mlflow
from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier import logger
from pathlib import Path
import tempfile
import shutil

# Set up DagsHub MLflow tracking
os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/iamparuthi/MLOps.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "iamparuthi"  # Your DagsHub username
os.environ["MLFLOW_TRACKING_PASSWORD"] = "de772d0837361e7006f06c607f856e1a053e370c"  # Your DagsHub token

mlflow.set_tracking_uri("https://dagshub.com/iamparuthi/MLOps.mlflow")


class Evaluation:
    def __init__(self, config, label_mode='int'):
        self.config = config
        self.model = tf.keras.models.load_model(config.trained_model_path)
        
        self.test_data = tf.keras.preprocessing.image_dataset_from_directory(
            config.test_data_path,
            image_size=(config.params_image_size[0], config.params_image_size[1]),
            batch_size=config.params_batch_size,
            label_mode=label_mode,
            shuffle=False
        )
        self.scores = {}

        class_names = self.test_data.class_names
        logger.info(f"Class indices mapping: {dict(zip(class_names, range(len(class_names))))}")

    def evaluate_model(self):
        logger.info("Evaluating the model on test dataset...")
        loss, accuracy = self.model.evaluate(self.test_data)
        self.scores = {"loss": float(loss), "accuracy": float(accuracy)}
        logger.info(f"Model Evaluation Scores: {self.scores}")

    def save_score(self):
        os.makedirs(os.path.dirname(self.config.metric_file_name), exist_ok=True)
        with open(self.config.metric_file_name, "w") as f:
            json.dump(self.scores, f, indent=4)
        logger.info(f"Scores saved to: {self.config.metric_file_name}")

    # def log_into_mlflow(self):
    #     """
    #     DagsHub-compatible MLflow logging with multiple approaches
    #     """
    #     try:
    #         logger.info("Logging to MLflow (DagsHub compatible)...")
            
    #         experiment_name = "CNN_Classifier_Evaluation"
    #         mlflow.set_experiment(experiment_name)
    #         logger.info(f"Set MLflow experiment: {experiment_name}")
            
    #         with mlflow.start_run() as run:
    #             logger.info(f"Started MLflow run: {run.info.run_id}")
                
    #             # 1. Log metrics and parameters (WORKS with DagsHub)
    #             mlflow.log_metrics(self.scores)
    #             mlflow.log_param("test_data_path", str(self.config.test_data_path))
    #             mlflow.log_param("model_path", str(self.config.trained_model_path))
    #             mlflow.log_param("image_size", f"{self.config.params_image_size[0]}x{self.config.params_image_size[1]}")
    #             mlflow.log_param("batch_size", self.config.params_batch_size)
    #             mlflow.log_param("loss", self.scores["loss"])
    #             mlflow.log_param("accuracy", self.scores["accuracy"])
    #             logger.info("Metrics and parameters logged successfully")
                
    #             # 2. APPROACH 1: Log model as individual files (WORKS with DagsHub)
    #             self.log_model_as_artifacts()
                
    #             # 3. APPROACH 2: Try simplified mlflow.keras.log_model (may work)
    #             self.try_simple_model_logging()
                
    #             # 4. APPROACH 3: Log model weights and architecture separately
    #             self.log_model_components()
                
    #             # 5. Log additional metadata
    #             self.log_model_metadata()
                
    #             logger.info("All MLflow logging completed!")
                
    #     except Exception as e:
    #         logger.error(f"Error in MLflow logging: {str(e)}")
    #         logger.exception("Full traceback:")

    def log_into_mlflow(self):
        try:
            logger.info("Logging to MLflow (DagsHub compatible)...")
            
            experiment_name = "CNN_Classifier_Evaluation"
            mlflow.set_experiment(experiment_name)
            
            with mlflow.start_run() as run:
                logger.info(f"Started MLflow run: {run.info.run_id}")
                
                # Log metrics
                mlflow.log_metrics(self.scores)

                mlflow.log_param("EPOCHS", self.config.all_params["EPOCHS"])
                mlflow.log_param("BATCH_SIZE", self.config.all_params["BATCH_SIZE"])
                mlflow.log_param("IMAGE_SIZE", self.config.all_params["IMAGE_SIZE"])

                # Save params.yaml as artifact for reproducibility
                mlflow.log_artifact("params.yaml")
                
                logger.info("All MLflow logging completed!")

        except Exception as e:
            logger.error(f"Error in MLflow logging: {str(e)}")
            logger.exception("Full traceback:")


    # def log_model_as_artifacts(self):
    #     """
    #     APPROACH 1: Log model as individual artifact files
    #     This approach WORKS with DagsHub
    #     """
    #     try:
    #         logger.info("Approach 1: Logging model as artifacts...")
            
    #         # Create temporary directory
    #         with tempfile.TemporaryDirectory() as temp_dir:
    #             # Save model in different formats
    #             model_h5_path = os.path.join(temp_dir, "model.h5")
    #             model_weights_path = os.path.join(temp_dir, "model_weights.h5")
    #             model_config_path = os.path.join(temp_dir, "model_config.json")
                
    #             # Save complete model
    #             self.model.save(model_h5_path, save_format='h5')
    #             logger.info(f"Model saved as H5: {model_h5_path}")
                
    #             # Save only weights
    #             self.model.save_weights(model_weights_path)
    #             logger.info(f"Model weights saved: {model_weights_path}")
                
    #             # Save model configuration
    #             model_config = self.model.to_json()
    #             with open(model_config_path, 'w') as f:
    #                 f.write(model_config)
    #             logger.info(f"Model config saved: {model_config_path}")
                
    #             # Log artifacts to MLflow
    #             mlflow.log_artifact(model_h5_path, "model")
    #             mlflow.log_artifact(model_weights_path, "model")
    #             mlflow.log_artifact(model_config_path, "model")
                
    #             logger.info("SUCCESS: Model artifacts logged to DagsHub!")
                
    #     except Exception as e:
    #         logger.warning(f"Approach 1 failed: {e}")

    # def try_simple_model_logging(self):
    #     """
    #     APPROACH 2: Try simplified mlflow.keras.log_model
    #     Sometimes works with minimal parameters
    #     """
    #     try:
    #         logger.info("Approach 2: Trying simple model logging...")
            
    #         # Try with minimal parameters
    #         mlflow.keras.log_model(
    #             self.model,
    #             "simple_model"  # Just artifact path, no extra parameters
    #         )
            
    #         logger.info("SUCCESS: Simple model logging worked!")
            
    #     except Exception as e:
    #         logger.warning(f"Approach 2 failed: {e}")

    # def log_model_components(self):
    #     """
    #     APPROACH 3: Log model weights and architecture separately
    #     Most reliable approach for DagsHub
    #     """
    #     try:
    #         logger.info("Approach 3: Logging model components...")
            
    #         with tempfile.TemporaryDirectory() as temp_dir:
    #             # 1. Save and log model summary
    #             summary_path = os.path.join(temp_dir, "model_summary.txt")
    #             with open(summary_path, 'w') as f:
    #                 self.model.summary(print_fn=lambda x: f.write(x + '\n'))
    #             mlflow.log_artifact(summary_path, "model_info")
                
    #             # 2. Save and log model weights
    #             weights_path = os.path.join(temp_dir, "model_weights.h5")
    #             self.model.save_weights(weights_path)
    #             mlflow.log_artifact(weights_path, "model_components")
                
    #             # 3. Save and log model architecture
    #             arch_path = os.path.join(temp_dir, "architecture.json")
    #             with open(arch_path, 'w') as f:
    #                 f.write(self.model.to_json())
    #             mlflow.log_artifact(arch_path, "model_components")
                
    #             # 4. Save and log training config if available
    #             if hasattr(self.model, 'get_config'):
    #                 config_path = os.path.join(temp_dir, "model_config.json")
    #                 with open(config_path, 'w') as f:
    #                     json.dump(self.model.get_config(), f, indent=2, default=str)
    #                 mlflow.log_artifact(config_path, "model_components")
                
    #             logger.info("SUCCESS: Model components logged!")
                
    #     except Exception as e:
    #         logger.warning(f"Approach 3 failed: {e}")

    # def log_model_metadata(self):
    #     """
    #     Log detailed model metadata and instructions for reconstruction
    #     """
    #     try:
    #         logger.info("Logging model metadata...")
            
    #         class_names = self.test_data.class_names
            
    #         # Create comprehensive metadata
    #         metadata = {
    #             "model_info": {
    #                 "framework": "TensorFlow/Keras",
    #                 "model_type": "CNN_Classifier",
    #                 "input_shape": [None, self.config.params_image_size[0], self.config.params_image_size[1], 3],
    #                 "output_shape": [None, len(class_names)],
    #                 "num_parameters": self.model.count_params()
    #             },
    #             "training_info": {
    #                 "image_size": self.config.params_image_size,
    #                 "batch_size": self.config.params_batch_size,
    #                 "num_classes": len(class_names),
    #                 "class_names": class_names,
    #                 "class_mapping": dict(zip(class_names, range(len(class_names))))
    #             },
    #             "evaluation_results": self.scores,
    #             "reconstruction_instructions": {
    #                 "step1": "Load model architecture from architecture.json",
    #                 "step2": "Create model using tf.keras.models.model_from_json()",
    #                 "step3": "Load weights from model_weights.h5",
    #                 "step4": "Compile model with appropriate loss and metrics",
    #                 "example_code": [
    #                     "import tensorflow as tf",
    #                     "import json",
    #                     "# Load architecture",
    #                     "with open('architecture.json', 'r') as f:",
    #                     "    model_json = f.read()",
    #                     "model = tf.keras.models.model_from_json(model_json)",
    #                     "# Load weights", 
    #                     "model.load_weights('model_weights.h5')",
    #                     "# Compile model",
    #                     "model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])"
    #                 ]
    #             }
    #         }
            
    #         # Save and log metadata
    #         with tempfile.TemporaryDirectory() as temp_dir:
    #             metadata_path = os.path.join(temp_dir, "complete_model_metadata.json")
    #             with open(metadata_path, 'w') as f:
    #                 json.dump(metadata, f, indent=2)
                
    #             mlflow.log_artifact(metadata_path, "model_metadata")
                
    #         logger.info("Model metadata logged successfully!")
            
    #     except Exception as e:
    #         logger.warning(f"Metadata logging failed: {e}")


if __name__ == "__main__":
    try:
        logger.info(">>> Stage: Evaluation started <<<")

        config = ConfigurationManager().get_evaluation_config()
        eval_obj = Evaluation(config)

        eval_obj.evaluate_model()
        eval_obj.save_score()
        eval_obj.log_into_mlflow()

        logger.info(">>> Stage: Evaluation completed <<<")

    except Exception as e:
        logger.exception(e)
        raise e