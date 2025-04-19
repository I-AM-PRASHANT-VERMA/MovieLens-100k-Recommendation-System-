import sys
from dataclasses import dataclass
import pandas as pd
from src.exception import CustomException
from src.logger import logging
import os
from surprise import Dataset, Reader

@dataclass
class DataTransformationConfig:
    pass

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data completed")

            reader = Reader(rating_scale=(1, 5))
            train_data = Dataset.load_from_df(train_df[['user_id', 'item_id', 'rating']], reader)
            test_data = Dataset.load_from_df(test_df[['user_id', 'item_id', 'rating']], reader)

            logging.info("Converted data to surprise format")
            return train_data, test_data

        except Exception as e:
            raise CustomException(e, sys)