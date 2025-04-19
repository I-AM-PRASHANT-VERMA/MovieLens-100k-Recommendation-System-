import sys
import pandas as pd
import os
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, user_id, n=5):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            print("Before Loading")
            model = load_object(file_path=model_path)
            print("After Loading")

            all_items = pd.read_csv('artifacts/ratings.csv')['item_id'].unique()
            predictions = []

            for item_id in all_items:
                pred = model.predict(user_id, item_id)
                predictions.append((item_id, pred.est))

            predictions.sort(key=lambda x: x[1], reverse=True)
            top_n = predictions[:n]
            return [item_id for item_id, _ in top_n]

        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self, user_id: int):
        self.user_id = user_id

    def get_data_as_data_frame(self):
        try:
            return pd.DataFrame({"user_id": [self.user_id]})
        except Exception as e:
            raise CustomException(e, sys)