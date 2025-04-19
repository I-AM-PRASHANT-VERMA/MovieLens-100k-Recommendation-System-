from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

class TrainPipeline:
    def __init__(self):
        pass

    def main(self):
        ingestion = DataIngestion()
        train_path, test_path = ingestion.initiate_data_ingestion()
        transformation = DataTransformation()
        train_data, test_data = transformation.initiate_data_transformation(train_path, test_path)
        trainer = ModelTrainer()
        rmse, precision, recall = trainer.initiate_model_trainer(train_data, test_data)
        return rmse, precision, recall

if __name__ == "__main__":
    pipeline = TrainPipeline()
    rmse, precision, recall = pipeline.main()
    print(f"Best model RMSE: {rmse}")
    print(f"Precision@5: {precision}")
    print(f"Recall@5: {recall}")