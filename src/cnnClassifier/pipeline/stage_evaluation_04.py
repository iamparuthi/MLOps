from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.model_evaluation import Evaluation
from cnnClassifier import logger

STAGE_NAME = "Evaluation stage"

class EvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        # 1️ Load configs
        config_manager = ConfigurationManager()
        eval_config = config_manager.get_evaluation_config()

        # 2️⃣ Initialize evaluation
        evaluation = Evaluation(eval_config)

        # 3️⃣ Run evaluation on test dataset
        logger.info(">>> Running evaluation (loss/accuracy)...")
        evaluation.evaluate_model()

        # 4️⃣ Save evaluation scores locally
        logger.info(">>> Saving evaluation scores locally...")
        evaluation.save_score()

        # # 5️⃣ Save model locally for DVC tracking
        # logger.info(">>> Saving model locally for DVC...")
        # evaluation.save_model_for_dvc()

        # 6️⃣ Log metrics to MLflow (DagsHub)
        logger.info(">>> Logging metrics to MLflow...")
        evaluation.log_into_mlflow()

        logger.info(">>> Evaluation pipeline completed ")

if __name__ == "__main__":
    try:
        logger.info(f">>>>>> Stage: {STAGE_NAME} started <<<<<<")
        pipeline = EvaluationPipeline()
        pipeline.main()
        logger.info(f">>>>>> Stage: {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
