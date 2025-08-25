# from cnnClassifier.config.configuration import ConfigurationManager
# from cnnClassifier.components.model_trainer import Training
# from cnnClassifier import logger



# STAGE_NAME = "Training"



# class ModelTrainingPipeline:
#     def __init__(self):
#         pass

#     def main(self):
#         config = ConfigurationManager()
#         training_config = config.get_training_config()
#         training = Training(config=training_config)
#         training.get_base_model()
#         training.train_valid_generator()
#         training.train()



# if __name__ == '__main__':
#     try:
#         logger.info(f"*******************")
#         logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
#         obj = ModelTrainingPipeline()
#         obj.main()
#         logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
#     except Exception as e:
#         logger.exception(e)
#         raise e
        
from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.model_evaluation import Evaluation
from cnnClassifier import logger
import mlflow
import mlflow.keras
import os


STAGE_NAME = "Evaluation"


class ModelEvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        eval_config = config.get_evaluation_config()

        evaluation = Evaluation(config=eval_config)

        # Step 1: Evaluate and save scores
        evaluation.evaluate_model()
        evaluation.save_score()

        # Step 2: Log to MLflow/Dagshub
        os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/iamparuthi/MLOps.mlflow"
        os.environ["MLFLOW_TRACKING_USERNAME"] = "iamparuthi"
        os.environ["MLFLOW_TRACKING_PASSWORD"] = "de772d0837361e7006f06c607f856e1a053e370c"

        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        mlflow.set_experiment("CNNClassifier_Evaluation")

        with mlflow.start_run(run_name="model_evaluation"):
            # Log params
            mlflow.log_params({
                "image_size": eval_config.params_image_size,
                "batch_size": eval_config.params_batch_size
            })

            # Log metrics
            mlflow.log_metrics(evaluation.scores)

            # Log model
            mlflow.keras.log_model(
                evaluation.model,
                artifact_path="model",
                registered_model_name="cnn_classifier_model"
            )

        logger.info("Evaluation completed and logged into MLflow/DagsHub.")


if __name__ == '__main__':
    try:
        logger.info("*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelEvaluationPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
