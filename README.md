---

# 🎥 MovieLens 100k Recommendation System  

![Python](https://img.shields.io/badge/Python-3.8-blue)  
![Flask](https://img.shields.io/badge/Flask-2.0.1-lightgrey)  
![Surprise](https://img.shields.io/badge/Surprise-1.1.1-orange)  

## 🌟 Project Overview  
**Goal**: Build a movie recommendation system using collaborative filtering  
**Solution**: Flask web app that suggests personalized movies based on user ratings  
**Impact**: Helps users discover films aligned with their preferences  

## 📊 Dataset  
**Source**: [MovieLens 100k](https://grouplens.org/datasets/movielens/100k/)  
- **100,000 ratings** from 943 users on 1,682 movies  
- **Rating scale**: 1-5 stars  
- **Key Files**:  
  - `u.data`: User-item-rating-timestamp tuples  
  - `u.item`: Movie metadata with titles and genres  

**Storage**:  
- Raw data: `notebook/data/ml-100k/`  
- Processed: `artifacts/ratings.csv` (full), `train.csv`/`test.csv` (80/20 split)  

## 🤖 Machine Learning Approach  

| Model | Type | Why Chosen |  
|-------|------|------------|  
| **SVD** | Matrix Factorization | Uncovers latent user preferences |  
| **KNNBaseline** | Neighborhood-Based | Preserves item similarity relationships |  

**Key Decision**: Chose these over neural networks for:  
- Interpretability  
- Faster training on medium-sized data  
- Proven effectiveness in collaborative filtering  

## ⚙️ Hyperparameter Tuning  

### Optimized Configurations  
| Model | Key Parameters | Impact |  
|-------|---------------|--------|  
| SVD | n_factors=100, n_epochs=20 | Captures deeper patterns |  
| KNNBaseline | k=40, pearson_baseline | Better similarity metrics |  

### Performance Impact  

| Metric | Before Tuning | After Tuning |  
|--------|--------------|-------------|  
| RMSE | 0.9434 | 0.9430 |  
| Precision@5 | 0.24% | 6.3% |  
| Recall@5 | 0.12% | 2.02% |  

## 🚀 Getting Started  

1. **Setup Environment**  
   ```bash
   conda activate "your_env_path"
   pip install -r requirements.txt
   ```

2. **Train Model**  
   ```bash
   python src/pipeline/train_pipeline.py
   ```

3. **Launch Web App**  
   ```bash
   python app.py
   ```
   Access at: `http://localhost:5000/predictdata`  

## 📈 Results  
| Model | RMSE | Precision@5 | Recall@5 |  
|-------|------|------------|---------|  
| SVD | 0.943 | 6.3% | 2.02% |  
| KNNBaseline | 0.951 | 5.8% | 1.87% |  

## 💡 Key Features  
- **Web Interface**: Input user IDs (1-943) for instant recommendations  
- **Modular Design**: Separated data, training, and prediction pipelines  
- **Production-Ready**: Logging and error handling throughout  

---
