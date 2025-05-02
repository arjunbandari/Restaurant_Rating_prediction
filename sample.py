# ----------------- Imports -----------------
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, VotingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import catboost as cb
import lightgbm as lgb
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import PolynomialFeatures
import time
from flask import Flask, render_template, request
import json

# ------------- Flask App Init -------------
app = Flask(__name__)
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# ------------- Function to Preprocess and Train -------------
def process_dataset(path, prefix):
    # Start the timer for the process
    start_time = time.time()

    # Read the dataset
    df = pd.read_csv(path)
    
    # Check if there are any missing values in the dataset
    missing_data = df.isnull().sum()
    
    # Fill missing values for categorical features with 'Unknown'
    df.fillna({'Reviews': "No review", 'Cuisine_Type': "Unknown", 'Location': "Unknown"}, inplace=True)
    
    # Drop duplicate rows from dataset
    df.drop_duplicates(inplace=True)
    
    # Sentiment Analysis: Apply Vader Sentiment Analysis to Reviews
    df['Sentiment_Score'] = df['Reviews'].apply(lambda r: sia.polarity_scores(str(r))['compound'])
    
    # Feature engineering: Create new features
    df['Service_Quality'] = df['Staff_Service'] / (df['Average_Wait_Time'] + 1)
    df['Cost_Efficiency'] = df['Rating'] / (df['Cost_for_Two'] + 1)
    
    # Categorical features encoding
    label_encoders = {}
    for col in ['Location', 'Cuisine_Type', 'Online_Delivery', 'Book_Table']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    # Feature Selection: Selecting Top-K Features based on f_regression score
    X = df[['Location', 'Cuisine_Type', 'Cost_for_Two', 'Online_Delivery', 'Book_Table', 'Average_Wait_Time', 'Staff_Service', 'Sentiment_Score', 'Service_Quality', 'Cost_Efficiency']]
    y = df['Rating']
    
    # Train-test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model Initialization: Instantiate models
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.1),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
        "Extra Trees": ExtraTreesRegressor(n_estimators=100, random_state=42),
        "AdaBoost": AdaBoostRegressor(n_estimators=100, random_state=42),
        "SVR": SVR(C=100),
        "KNN": KNeighborsRegressor(n_neighbors=5),
        "XGBoost": xgb.XGBRegressor(n_estimators=100, random_state=42),
        "CatBoost": cb.CatBoostRegressor(n_estimators=100, verbose=0, random_state=42),
        "LightGBM": lgb.LGBMRegressor(n_estimators=100, random_state=42)
    }

    # Perform training and evaluation
    performances = {}
    predictions = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        pred = model.predict(X_test_scaled)
        predictions[name] = pred
        performances[name] = {
            'MSE': mean_squared_error(y_test, pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, pred)),
            'MAE': mean_absolute_error(y_test, pred),
            'R2': r2_score(y_test, pred)
        }

    # Ensemble Model: Voting Regressor
    top_models = sorted(performances.items(), key=lambda x: x[1]['RMSE'])[:5]
    top_names = [name for name, _ in top_models]
    voting_models = [(name, models[name]) for name in top_names]
    voting_model = VotingRegressor(estimators=voting_models)
    voting_model.fit(X_train_scaled, y_train)
    voting_pred = voting_model.predict(X_test_scaled)

    # Save and plot graphs
    os.makedirs("static", exist_ok=True)

    # Graph 1: RMSE Bar Plot
    plt.figure(figsize=(12, 6))
    sns.barplot(x=list(performances.keys()), y=[v['RMSE'] for v in performances.values()], palette='magma')
    plt.xticks(rotation=45)
    plt.title(f'{prefix} - Model RMSE Comparison')
    plt.tight_layout()
    plt.savefig(f'static/{prefix}_rmse.png')
    plt.close()

    # Graph 2: Prediction vs Actual Scatter
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, voting_pred, alpha=0.6, color='royalblue')
    plt.plot([0, 5], [0, 5], 'r--')
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f'{prefix} - Voting Regressor Prediction')
    plt.tight_layout()
    plt.savefig(f'static/{prefix}_pred_vs_actual.png')
    plt.close()

    # Graph 3: Feature Importance Plot (Random Forest)
    rf_model = models["Random Forest"]
    feature_importance = rf_model.feature_importances_
    plt.figure(figsize=(12, 6))
    sns.barplot(x=X.columns, y=feature_importance, palette='viridis')
    plt.title(f'{prefix} - Feature Importance (Random Forest)')
    plt.tight_layout()
    plt.savefig(f'static/{prefix}_feature_importance.png')
    plt.close()

    # Graph 4: PCA Visualization (Principal Component Analysis)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_train_scaled)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y_train, palette='deep')
    plt.title(f'{prefix} - PCA Visualization')
    plt.tight_layout()
    plt.savefig(f'static/{prefix}_pca.png')
    plt.close()

    # End the timer and log the duration
    elapsed_time = time.time() - start_time
    print(f"[{prefix}] Processing Time: {elapsed_time:.2f} seconds")

    return df, voting_model, scaler, label_encoders

# ------------- Load and Process Datasets -------------
df_main, voting_model_main, scaler_main, label_enc_main = process_dataset("restaurant.csv", "main")
df_s1, _, _, _ = process_dataset("s1.csv", "s1")
df_s2, _, _, _ = process_dataset("s2.csv", "s2")

# ------------- Restaurant List for UI -------------
restaurant_names = list(df_main['Restaurant_Name'].unique())

# ------------- Helper Function -------------
def get_rating_category(rating):
    if rating >= 4.5: return "Excellent 🌟"
    elif rating >= 4.0: return "Good 👍"
    elif rating >= 3.0: return "Average ⚠️"
    else: return "Poor ❌"

# ------------- Routes -------------
@app.route('/')
def home():
    return render_template("index.html", restaurants=restaurant_names)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        name = request.form.get('restaurant')
        row = df_main[df_main['Restaurant_Name'] == name]

        if row.empty:
            return render_template("result.html", error="Restaurant not found.")

        X = row[['Location', 'Cuisine_Type', 'Cost_for_Two', 'Online_Delivery', 'Book_Table', 'Average_Wait_Time', 'Staff_Service', 'Sentiment_Score', 'Service_Quality', 'Cost_Efficiency']]
        X_scaled = scaler_main.transform(X)
        pred = voting_model_main.predict(X_scaled)[0]
        actual = row['Rating'].values[0]

        sentiment = row['Sentiment_Score'].values[0]
        sentiment_text = "Positive 😊" if sentiment > 0.5 else "Negative 😞" if sentiment < -0.5 else "Neutral 🤔"

        # Prediction Plot
        plt.figure(figsize=(6, 4))
        sns.barplot(x=["Predicted", "Actual"], y=[pred, actual], palette=["blue", "green"])
        plt.title(f"{name} - Predicted vs Actual")
        plt.ylim(0, 5)
        plt.tight_layout()
        plt.savefig("static/result_plot.png")
        plt.close()

        return render_template("result.html",
                               restaurant=name,
                               predicted_rating=round(pred, 2),
                               actual_rating=round(actual, 2),
                               rating_category=get_rating_category(pred),
                               location=label_enc_main['Location'].inverse_transform([row['Location'].values[0]])[0],
                               cuisine=label_enc_main['Cuisine_Type'].inverse_transform([row['Cuisine_Type'].values[0]])[0],
                               sentiment=sentiment_text,
                               plot_path="static/result_plot.png",
                               rmse_main="static/main_rmse.png",
                               pred_actual="static/main_pred_vs_actual.png",
                               rmse_s1="static/s1_rmse.png",
                               rmse_s2="static/s2_rmse.png",
                               feature_importance="static/main_feature_importance.png",
                               pca="static/main_pca.png")
    except Exception as e:
        return render_template("result.html", error=str(e))

# ------------- Run the App -------------
if __name__ == "__main__":
    app.run(debug=True)
