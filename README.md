🎥 MovieLens 100k Recommendation System
🌟 Project Overview
This project develops a Recommendation System for movies using the MovieLens 100k dataset. The goal is to recommend movies to users based on collaborative filtering, leveraging user ratings to predict preferences. The system is built as a web application using Flask, allowing users to input a user_id and receive personalized movie recommendations.
I created this project to help movie enthusiasts discover films tailored to their tastes, addressing the challenge of finding relevant movies in a vast catalog. By using collaborative filtering, the system learns from user behavior to suggest movies they’re likely to enjoy, enhancing their viewing experience.
🎯 Objectives

Build a recommendation system that predicts top-N movies for a given user. 🔍
Evaluate model performance using metrics like RMSE, precision@5, and recall@5. 📊
Provide a user-friendly web interface to interact with the recommendation system. 🌐
Ensure the codebase is robust with proper error handling, logging, and modularity. 💻

📂 Dataset
The dataset used is the MovieLens 100k dataset, sourced from the GroupLens Research Project at the University of Minnesota. It contains 100,000 ratings from 943 users on 1,682 movies, with ratings ranging from 1 to 5.

Source: GroupLens MovieLens 100k Dataset 🌍
Link: https://grouplens.org/datasets/movielens/100k/


Location: The dataset is stored in notebook/data/ml-100k/, with key files:
u.data: User ratings (format: user_id, item_id, rating, timestamp).
u.item: Movie metadata (format: item_id, movie_title, etc.).



The data was split into training (80%) and test (20%) sets, saved as artifacts/train.csv and artifacts/test.csv, with the full dataset saved as artifacts/ratings.csv.
🤖 Machine Learning Models
We used the surprise library for collaborative filtering, implementing two models:

SVD (Singular Value Decomposition): A matrix factorization technique that decomposes the user-item rating matrix into latent factors.
KNNBaseline: A k-Nearest Neighbors approach with baseline ratings to account for user and item biases.

Purpose and Why These Algorithms? 🎓
I chose SVD and KNNBaseline for their proven effectiveness in collaborative filtering tasks:

SVD excels at uncovering latent patterns in user preferences, making it ideal for sparse datasets like MovieLens 100k, where users rate only a small fraction of movies. It provides accurate rating predictions, which is crucial for reliable recommendations.
KNNBaseline leverages item similarity with baseline adjustments, ensuring recommendations are based on movies similar to those a user already likes, adding diversity to the suggestions.

Despite many ML algorithms available (e.g., neural networks, content-based methods), SVD and KNNBaseline were ideal because they are lightweight, interpretable, and well-suited for collaborative filtering, offering a balance of performance and simplicity for this project.
⚙️ Hyperparameter Tuning
Initially, we used default parameters for both models, but later tuned them to improve recommendation quality:

SVD:

Tuned Parameters: n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02
These control the number of latent factors, training iterations, learning rate, and regularization strength.


KNNBaseline:

Tuned Parameters: k=40, sim_options={'name': 'pearson_baseline', 'user_based': False}
These set the number of neighbors, use Pearson baseline similarity, and focus on item-based filtering.



📈 Tuning Impact

Initial Metrics (Default Parameters):

RMSE: 0.9434 📉
Precision@5: 0.0024 (only 0.24% of top-5 recommendations were relevant) 🚫
Recall@5: 0.0012 (captured 0.12% of relevant items in top-5) 🚫


After Tuning:

RMSE: 0.9430 (slight improvement of ~0.0004, better rating prediction accuracy) ✅
Precision@5: 0.0630 (improved to 6.3%, ~0.3 items out of 5 are relevant) ✅
Recall@5: 0.0202 (improved to 2.02%, capturing more relevant items) ✅



The tuning significantly enhanced the ranking quality for top-N recommendations, making them more relevant to users, though there’s still room for improvement.

🚀 How to Run

Activate the Environment:
conda activate "E:\Coding\recommendation_model\venv"


Train the Model:
python src/pipeline/train_pipeline.py


Run the Web App:
python app.py


Visit http://localhost:5000/predictdata, enter a user_id (1–943), and get recommendations! 🖥️



