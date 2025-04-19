import sys
import pandas as pd
import os
from typing import List, Tuple
from src.exception import CustomException
from src.utils import load_object
from pathlib import Path

class PredictPipeline:
    def __init__(self):
        # Initialize paths
        self.model_path = os.path.join("artifacts", "model.pkl")
        self.ratings_path = os.path.join("artifacts", "ratings.csv")
        self.items_path = os.path.join("notebook", "data", "ml-100k", "u.item")
        
        # Validate paths during initialization
        self._validate_paths()

    def _validate_paths(self):
        """Check if required files exist"""
        required_files = {
            "Model": self.model_path,
            "Ratings data": self.ratings_path,
            "Items data": self.items_path
        }
        
        missing_files = []
        for name, path in required_files.items():
            if not os.path.exists(path):
                missing_files.append(f"{name} not found at {path}")
        
        if missing_files:
            raise FileNotFoundError("\n".join(missing_files))

    def predict(self, user_id: int, n: int = 5) -> List[Tuple[str, float]]:
        """
        Predict top n movie recommendations for a user
        
        Args:
            user_id: ID of the user to make predictions for
            n: Number of recommendations to return
            
        Returns:
            List of tuples containing (movie_name, predicted_rating)
        """
        try:
            # Load model
            print("Loading recommendation model...")
            model = load_object(file_path=self.model_path)
            
            # Load movie metadata
            items = self._load_movie_metadata()
            
            # Get all unique items from ratings
            all_items = pd.read_csv(self.ratings_path)['item_id'].unique()
            
            # Generate predictions
            predictions = []
            for item_id in all_items:
                pred = model.predict(user_id, item_id)
                predictions.append((item_id, pred.est))
            
            # Sort and get top n predictions
            predictions.sort(key=lambda x: x[1], reverse=True)
            top_n = predictions[:n]
            
            # Map item_ids to movie names and include ratings
            top_n_recommendations = []
            for item_id, rating in top_n:
                movie_name = items.loc[items['item_id'] == item_id, 'movie_name'].values[0]
                top_n_recommendations.append((movie_name, round(rating, 2)))
            
            return top_n_recommendations

        except Exception as e:
            raise CustomException(e, sys)

    def _load_movie_metadata(self) -> pd.DataFrame:
        """Load movie names and IDs from u.item file"""
        columns = [
            'item_id', 'movie_name', 'release_date', 'video_release_date',
            'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation',
            'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
            'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
            'Thriller', 'War', 'Western'
        ]
        
        items = pd.read_csv(
            self.items_path,
            sep='|',
            encoding='latin-1',
            header=None,
            names=columns
        )
        
        return items[['item_id', 'movie_name']]


class CustomData:
    def __init__(self, user_id: int):
        self.user_id = user_id

    def get_data_as_data_frame(self) -> pd.DataFrame:
        """Convert user input to DataFrame format"""
        try:
            return pd.DataFrame({"user_id": [self.user_id]})
        except Exception as e:
            raise CustomException(e, sys)


# Example usage
if __name__ == "__main__":
    try:
        # Initialize pipeline
        pipeline = PredictPipeline()
        
        # Get recommendations for user 123
        user_id = 123
        recommendations = pipeline.predict(user_id, n=5)
        
        print(f"\nTop 5 recommendations for user {user_id}:")
        for i, (movie, rating) in enumerate(recommendations, 1):
            print(f"{i}. {movie} (Predicted rating: {rating})")
            
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)