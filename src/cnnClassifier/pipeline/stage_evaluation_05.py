from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.model_evaluation import Evaluation
from cnnClassifier import logger

STAGE_NAME = "Evaluation stage"


class EvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        # Load configs
        config = ConfigurationManager()
        eval_config = config.get_evaluation_config()

        # Initialize evaluation
        evaluation = Evaluation(eval_config)

        # Run evaluation steps
        logger.info(">>> Running evaluation (loss/accuracy)...")
        evaluation.evaluation()

        logger.info(">>> Saving evaluation scores locally...")
        evaluation.save_score()

        logger.info(">>> Logging metrics & model to MLflow/Dagshub...")
        evaluation.log_into_mlflow()

        logger.info(">>> Evaluation pipeline completed ✅")