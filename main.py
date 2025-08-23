from cnnClassifier import logger
from cnnClassifier.pipeline.stage_dataingestion_01 import DataIngestionTrainingPipeline
from cnnClassifier.pipeline.stage_base_model_prepare_02 import PrepareBaseModelTrainingPipeline
from cnnClassifier.pipeline.stage_model_train_03 import ModelTrainingPipeline
from cnnClassifier.pipeline.stage_evaluation_05 import EvaluationPipeline
import dagshub
dagshub.init(repo_owner='iamparuthi', repo_name='MLOps', mlflow=True)


STAGE_NAME = "Data Ingestion stage"


try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e



STAGE_NAME = "Prepare base model"
try: 
   logger.info(f"*******************")
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   prepare_base_model = PrepareBaseModelTrainingPipeline()
   prepare_base_model.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e



STAGE_NAME = "Training"
try: 
   logger.info(f"*******************")
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   model_trainer = ModelTrainingPipeline()
   model_trainer.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e



STAGE_NAME = "Evaluation stage"

try:
    logger.info("*******************")
    logger.info(f">>>>>> Stage: {STAGE_NAME} started <<<<<<")

    obj = EvaluationPipeline()
    obj.main()

    logger.info(f">>>>>> Stage: {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e