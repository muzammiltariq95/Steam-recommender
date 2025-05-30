# Steam Recommender System using PySpark & ALS

This project builds a collaborative filtering recommendation engine for Steam games using the ALS algorithm from PySpark MLlib. Based on implicit feedback (hours played), the model suggests top games for users.

## Key Features

- Cleaned and preprocessed 200k+ user-game interactions using PySpark
- Built ALS model with implicitPrefs enabled for feedback-based ranking
- Hyperparameter tuning across 8 combinations (rank, regParam, alpha)
- Tracked model runs and metrics using MLflow on Databricks
- Generated Top-N game recommendations per user

## Model Performance

- Best RMSE: **211.86**
- Best Config: `rank=20`, `regParam=0.01`, `alpha=10.0`

## Files

- `steam_recommender_notebook.html`: Full notebook with markdowns, code, and visualizations
- `outputs/screenshots/`: MLflow experiment tracking visuals
- `data/steam-200k.csv`: (Include if allowed, otherwise share source link)

## Tools & Libraries

- PySpark
- ALS (Alternating Least Squares)
- MLflow (Experiment tracking)
- Databricks
- Matplotlib (for plotting)

## Screenshots

### Sample Output: RMSE from MLflow Tracking
![MLflow RMSE](screenshots/mlflow.png)

### Top 10 Game Recommendations
![Game Recs](screenshots/top_game_recommendations.png)


## Dataset Source

[Steam-200k Dataset](https://www.kaggle.com/datasets/tamber/steam-video-games)
