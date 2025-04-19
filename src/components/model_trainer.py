import os
import sys
from dataclasses import dataclass
from typing import Dict, Tuple
from surprise import SVD, KNNBaseline, Dataset
from surprise.model_selection import cross_validate
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")
    metrics_file_path: str = os.path.join("artifacts", "training_metrics.txt")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_file_path), exist_ok=True)

    def _evaluate_models(self, models: Dict[str, object], train_data: Dataset) -> Dict[str, float]:
        """Evaluate multiple models using cross-validation"""
        model_report = {}
        for name, model in models.items():
            logging.info(f"Evaluating model: {name}")
            results = cross_validate(
                model, train_data, 
                measures=['RMSE', 'MAE'], 
                cv=3, 
                verbose=False,
                n_jobs=-1  # Parallel processing
            )
            model_report[name] = {
                'rmse': results['test_rmse'].mean(),
                'mae': results['test_mae'].mean()
            }
            logging.info(f"{name} - RMSE: {model_report[name]['rmse']:.4f}, MAE: {model_report[name]['mae']:.4f}")
        return model_report

    def _compute_top_n_metrics(self, model: object, trainset: object, test_ratings: list, 
                             k: int = 5, threshold: float = 3.5) -> Tuple[float, float]:
        """Compute precision@k and recall@k metrics"""
        testset = trainset.build_anti_testset()
        predictions = model.test(testset)
        
        user_recommendations = {}
        user_relevant_items = {}
        
        for pred in predictions:
            user_id, item_id, est = pred.uid, pred.iid, pred.est
            user_recommendations.setdefault(user_id, []).append((item_id, est))
        
        for user_id, item_id, rating, _ in test_ratings:
            if rating >= threshold:
                user_relevant_items.setdefault(user_id, set()).add(item_id)
        
        precision_sum = recall_sum = num_users = 0
        
        for user_id in user_recommendations:
            if user_id not in user_relevant_items:
                continue
            
            top_k = [item_id for item_id, _ in 
                    sorted(user_recommendations[user_id], 
                          key=lambda x: x[1], reverse=True)[:k]]
            
            relevant_items = user_relevant_items[user_id]
            relevant_recommended = set(top_k) & relevant_items
            
            precision = len(relevant_recommended) / k
            recall = len(relevant_recommended) / len(relevant_items) if relevant_items else 0
            
            precision_sum += precision
            recall_sum += recall
            num_users += 1
        
        avg_precision = precision_sum / num_users if num_users else 0
        avg_recall = recall_sum / num_users if num_users else 0
        
        return avg_precision, avg_recall

    def _save_training_metrics(self, metrics: Dict[str, float]):
        """Save training metrics to file"""
        with open(self.model_trainer_config.metrics_file_path, 'w') as f:
            for metric, value in metrics.items():
                if isinstance(value, (float, int)):
                    f.write(f"{metric}: {value:.4f}\n")
                else:
                    f.write(f"{metric}: {value}\n")

    def initiate_model_trainer(self, train_data: Dataset, test_data: Dataset) -> Tuple[float, float, float]:
        """
        Train and evaluate recommendation models
        
        Args:
            train_data: Dataset for training
            test_data: Dataset for evaluation
            
        Returns:
            Tuple of (best_rmse, avg_precision, avg_recall)
        """
        try:
            logging.info("Starting model training pipeline")
            
            models = {
                "SVD": SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02),
                "KNNBaseline": KNNBaseline(k=40, sim_options={'name': 'pearson_baseline', 'user_based': False}),
            }
            
            model_report = self._evaluate_models(models, train_data)
            best_model_name = min(model_report, key=lambda x: model_report[x]['rmse'])
            best_model = models[best_model_name]
            best_rmse = model_report[best_model_name]['rmse']
            
            logging.info(f"Selected best model: {best_model_name} with RMSE: {best_rmse:.4f}")
            
            trainset = train_data.build_full_trainset()
            best_model.fit(trainset)
            
            test_ratings = list(test_data.raw_ratings)
            avg_precision, avg_recall = self._compute_top_n_metrics(
                best_model, trainset, test_ratings)
            
            logging.info(f"Evaluation metrics - Precision@5: {avg_precision:.4f}, Recall@5: {avg_recall:.4f}")
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            training_metrics = {
                'best_model': best_model_name,
                'rmse': best_rmse,
                'precision@5': avg_precision,
                'recall@5': avg_recall,
                'mae': model_report[best_model_name]['mae']
            }
            self._save_training_metrics(training_metrics)
            
            return best_rmse, avg_precision, avg_recall

        except Exception as e:
            logging.error("Error during model training", exc_info=True)
            raise CustomException(e, sys)