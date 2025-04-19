import os
import sys
from dataclasses import dataclass
from surprise import SVD, KNNBaseline
from surprise.model_selection import cross_validate
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_data, test_data):
        try:
            logging.info("Starting model training")
            models = {
                "SVD": SVD(),
                "KNNBaseline": KNNBaseline(),
            }
            model_report = {}

            for name, model in models.items():
                results = cross_validate(model, train_data, measures=['RMSE'], cv=3, verbose=False)
                rmse = results['test_rmse'].mean()
                model_report[name] = rmse
                logging.info(f"{name} RMSE: {rmse}")

            best_model_name = min(model_report, key=model_report.get)
            best_model = models[best_model_name]
            logging.info(f"Best model: {best_model_name} with RMSE {model_report[best_model_name]}")

            trainset = train_data.build_full_trainset()
            best_model.fit(trainset)

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            return model_report[best_model_name]

        except Exception as e:
            raise CustomException(e, sys)