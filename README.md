🎥 MovieLens 100k Recommendation System
🌟 Project Overview
This project builds a Recommendation System for movies using the MovieLens 100k dataset, driven by a passion to help movie enthusiasts discover films they’ll love! 💖 The system leverages collaborative filtering to recommend movies based on user ratings, predicting preferences through a web app built with Flask. Users can input their user_id and get personalized movie suggestions instantly.
🎯 Purpose
I created this project to explore how machine learning can enhance personalized movie discovery, making it easier for users to find films that match their tastes. The goal was to build an efficient system that balances accuracy and usability, delivering relevant recommendations through a user-friendly interface.
📋 Objectives

Build a system to predict the top-N movies for a user based on their ratings. 🔍
Evaluate performance using RMSE, precision@5, and recall@5 metrics. 📊
Create a user-friendly web interface for seamless interaction. 🌐
Ensure robust code with error handling, logging, and modularity. 💻

📂 Dataset
The dataset used is the MovieLens 100k dataset, sourced from the GroupLens Research Project at the University of Minnesota. It includes 100,000 ratings from 943 users on 1,682 movies, with ratings from 1 to 5.

Source: GroupLens MovieLens 100k Dataset 🌍
Download Location: Manually downloaded and extracted to notebook/data/ml-100k/.
Key Files:
u.data: User ratings (user_id, item_id, rating, timestamp).
u.item: Movie metadata (item_id, movie_title, etc.).


Preprocessing: Split into 80% training and 20% test sets, saved as artifacts/train.csv and artifacts/test.csv, with the full dataset saved as artifacts/ratings.csv.

🤖 Machine Learning Models
I used the surprise library for collaborative filtering, selecting two models: SVD and KNNBaseline.

SVD (Singular Value Decomposition): A matrix factorization technique that decomposes the user-item rating matrix into latent factors, capturing hidden patterns in user preferences.
KNNBaseline: A k-Nearest Neighbors approach with baseline ratings to account for user and item biases, focusing on item-based similarity.

Why These Algorithms? 🎓
I chose SVD and KNNBaseline because they are well-suited for collaborative filtering tasks like movie recommendations:

SVD: Ideal for capturing latent factors in sparse datasets like MovieLens 100k, where users rate only a small subset of movies. It excels at predicting ratings (low RMSE) and generalizing user preferences, making it a strong choice for personalized recommendations.
KNNBaseline: Effective for leveraging item similarity, especially with baseline adjustments to account for user and item biases. It’s intuitive for recommending movies similar to those a user has liked, complementing SVD’s approach.

Despite other ML algorithms (e.g., neural networks, content-based methods), I prioritized SVD and KNNBaseline because they are lightweight, interpretable, and well-documented for collaborative filtering, balancing performance and implementation simplicity.
⚙️ Hyperparameter Tuning
Initially, I used default parameters, but later tuned them to boost recommendation quality:

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



The tuning significantly enhanced the ranking quality for top-N recommendations, though precision@5 and recall@5 remain low, indicating potential for further optimization (e.g., GridSearchCV or hybrid models).
🏆 Achievements

Built a functional recommendation system that delivers personalized movie suggestions. 🎉
Created a user-friendly Flask web app for easy interaction. 🌐
Improved recommendation quality through hyperparameter tuning, increasing precision@5 from 0.24% to 6.3%. 📈
Developed a robust codebase with error handling, logging, and modularity. 💻

🚀 How to Run

Activate the Environment:
conda activate "E:\Coding\recommendation_model\venv"


Train the Model:
python src/pipeline/train_pipeline.py


Run the Web App:
python app.py


Visit http://localhost:5000/predictdata, enter a user_id (1–943), and get recommendations! 🖥️



