import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, VotingRegressor, AdaBoostRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
import xgboost as xgb
import catboost as cb
import lightgbm as lgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import time
import logging
from flask import Flask, render_template, request, jsonify
import json
import uuid
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ------------- Logging Configuration -------------
logging.basicConfig(
    filename='restaurant_rating_app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ------------- Flask App Initialization -------------
app = Flask(__name__, template_folder='templates', static_folder='static')
nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()

# ------------- Utility Functions -------------
def generate_uuid():
    return str(uuid.uuid4())

def save_plot(fig, filename, prefix):
    os.makedirs("static", exist_ok=True)
    filepath = f"static/{prefix}_{filename}.png"
    fig.savefig(filepath, bbox_inches='tight')
    plt.close(fig)
    return filepath

def save_plotly_fig(fig, filename, prefix):
    os.makedirs("static", exist_ok=True)
    filepath = f"static/{prefix}_{filename}.html"
    fig.write_html(filepath)
    return filepath

# ------------- Advanced Feature Engineering -------------
def advanced_feature_engineering(df):
    logger.info("Performing advanced feature engineering")
    
    # Existing features
    df['Service_Quality'] = df['Staff_Service'] / (df['Average_Wait_Time'] + 1)
    df['Cost_Efficiency'] = df['Rating'] / (df['Cost_for_Two'] + 1)
    
    # New features
    df['Wait_Time_Ratio'] = df['Average_Wait_Time'] / (df['Staff_Service'] + 1)
    df['Cost_per_Rating'] = df['Cost_for_Two'] / (df['Rating'] + 0.1)
    df['Online_Delivery_Impact'] = df['Online_Delivery'] * df['Sentiment_Score']
    df['Booking_Efficiency'] = df['Book_Table'] / (df['Average_Wait_Time'] + 1)
    
    # Interaction features
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    interaction_features = poly.fit_transform(df[['Cost_for_Two', 'Staff_Service', 'Average_Wait_Time']])
    interaction_df = pd.DataFrame(interaction_features, columns=[f'interaction_{i}' for i in range(interaction_features.shape[1])])
    df = pd.concat([df.reset_index(drop=True), interaction_df.reset_index(drop=True)], axis=1)
    
    return df

# ------------- Neural Network Model -------------
def create_neural_network(input_dim):
    model = Sequential([
        Dense(128, activation='relu', input_dim=input_dim),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# ------------- Function to Preprocess and Train -------------
def process_dataset(path, prefix, hyperparameter_tuning=False):
    start_time = time.time()
    logger.info(f"Starting processing for dataset: {path} with prefix: {prefix}")
    
    try:
        # Read the dataset
        df = pd.read_csv(path)
        logger.info(f"Dataset loaded with shape: {df.shape}")
        
        # Data Cleaning
        missing_data = df.isnull().sum()
        logger.info(f"Missing data: {missing_data.to_dict()}")
        
        df.fillna({
            'Reviews': "No review",
            'Cuisine_Type': "Unknown",
            'Location': "Unknown",
            'Cost_for_Two': df['Cost_for_Two'].median(),
            'Average_Wait_Time': df['Average_Wait_Time'].median(),
            'Staff_Service': df['Staff_Service'].median()
        }, inplace=True)
        
        df.drop_duplicates(inplace=True)
        logger.info(f"Dataset after cleaning: {df.shape}")
        
        # Sentiment Analysis
        df['Sentiment_Score'] = df['Reviews'].apply(lambda r: sia.polarity_scores(str(r))['compound'])
        
        # Advanced Feature Engineering
        df = advanced_feature_engineering(df)
        
        # Categorical Encoding
        label_encoders = {}
        for col in ['Location', 'Cuisine_Type', 'Online_Delivery', 'Book_Table']:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
        
        # Feature Selection
        features = [
            'Location', 'Cuisine_Type', 'Cost_for_Two', 'Online_Delivery', 'Book_Table',
            'Average_Wait_Time', 'Staff_Service', 'Sentiment_Score', 'Service_Quality',
            'Cost_Efficiency', 'Wait_Time_Ratio', 'Cost_per_Rating', 'Online_Delivery_Impact',
            'Booking_Efficiency'
        ] + [col for col in df.columns if 'interaction_' in col]
        
        X = df[features]
        y = df['Rating']
        
        # Train-test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Feature Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # MinMax Scaling for Neural Network
        minmax_scaler = MinMaxScaler()
        X_train_mm = minmax_scaler.fit_transform(X_train)
        X_test_mm = minmax_scaler.transform(X_test)
        
        # Model Initialization
        models = {
            "Linear Regression": LinearRegression(),
            "Ridge": Ridge(alpha=1.0),
            "Lasso": Lasso(alpha=0.1),
            "Decision Tree": DecisionTreeRegressor(random_state=42),
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
        
        # Neural Network
        nn_model = create_neural_network(X_train_scaled.shape[1])
        models["Neural Network"] = nn_model
        
        # Hyperparameter Tuning (Optional)
        if hyperparameter_tuning:
            logger.info("Performing hyperparameter tuning")
            param_grid_rf = {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5]
            }
            grid_search_rf = GridSearchCV(
                RandomForestRegressor(random_state=42),
                param_grid_rf,
                cv=5,
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            grid_search_rf.fit(X_train_scaled, y_train)
            models["Random Forest"] = grid_search_rf.best_estimator_
            logger.info(f"Best Random Forest params: {grid_search_rf.best_params_}")
        
        # Model Training and Evaluation
        performances = {}
        predictions = {}
        cv_scores = {}
        
        for name, model in models.items():
            logger.info(f"Training model: {name}")
            try:
                if name == "Neural Network":
                    model.fit(X_train_mm, y_train, epochs=50, batch_size=32, verbose=0)
                    pred = model.predict(X_test_mm).flatten()
                else:
                    model.fit(X_train_scaled, y_train)
                    pred = model.predict(X_test_scaled)
                
                predictions[name] = pred
                performances[name] = {
                    'MSE': mean_squared_error(y_test, pred),
                    'RMSE': np.sqrt(mean_squared_error(y_test, pred)),
                    'MAE': mean_absolute_error(y_test, pred),
                    'R2': r2_score(y_test, pred),
                    'MAPE': mean_absolute_percentage_error(y_test, pred)
                }
                
                # Cross-validation
                if name != "Neural Network":
                    cv_score = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
                    cv_scores[name] = cv_score.mean()
                else:
                    cv_scores[name] = None
                
            except Exception as e:
                logger.error(f"Error training {name}: {str(e)}")
                performances[name] = {'MSE': None, 'RMSE': None, 'MAE': None, 'R2': None, 'MAPE': None}
        
        # Ensemble Models
        top_models = sorted(
            [(name, perf) for name, perf in performances.items() if perf['RMSE'] is not None],
            key=lambda x: x[1]['RMSE']
        )[:5]
        top_names = [name for name, _ in top_models]
        
        # Voting Regressor
        voting_models = [(name, models[name]) for name in top_names if name != "Neural Network"]
        voting_model = VotingRegressor(estimators=voting_models)
        voting_model.fit(X_train_scaled, y_train)
        voting_pred = voting_model.predict(X_test_scaled)
        
        # Stacking Regressor
        stacking_model = StackingRegressor(
            estimators=voting_models,
            final_estimator=Ridge()
        )
        stacking_model.fit(X_train_scaled, y_train)
        stacking_pred = stacking_model.predict(X_test_scaled)
        
        # Add ensemble models to performances
        performances['Voting Regressor'] = {
            'MSE': mean_squared_error(y_test, voting_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, voting_pred)),
            'MAE': mean_absolute_error(y_test, voting_pred),
            'R2': r2_score(y_test, voting_pred),
            'MAPE': mean_absolute_percentage_error(y_test, voting_pred)
        }
        performances['Stacking Regressor'] = {
            'MSE': mean_squared_error(y_test, stacking_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, stacking_pred)),
            'MAE': mean_absolute_error(y_test, stacking_pred),
            'R2': r2_score(y_test, stacking_pred),
            'MAPE': mean_absolute_percentage_error(y_test, stacking_pred)
        }
        
        # Visualizations
        # 1. RMSE Bar Plot
        fig1, ax1 = plt.subplots(figsize=(14, 6))
        sns.barplot(x=list(performances.keys()), y=[v['RMSE'] for v in performances.values()], palette='magma', ax=ax1)
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
        ax1.set_title(f'{prefix} - Model RMSE Comparison')
        rmse_path = save_plot(fig1, 'rmse', prefix)
        
        # 2. Prediction vs Actual Scatter (Interactive Plotly)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=y_test, y=voting_pred, mode='markers', name='Voting Regressor',
                                 marker=dict(color='royalblue', size=8, opacity=0.6)))
        fig2.add_trace(go.Scatter(x=[0, 5], y=[0, 5], mode='lines', name='Ideal', line=dict(color='red', dash='dash')))
        fig2.update_layout(title=f'{prefix} - Voting Regressor Prediction vs Actual',
                          xaxis_title='Actual Rating', yaxis_title='Predicted Rating',
                          width=600, height=600)
        pred_vs_actual_path = save_plotly_fig(fig2, 'pred_vs_actual', prefix)
        
        # 3. Feature Importance (Random Forest)
        rf_model = models["Random Forest"]
        feature_importance = rf_model.feature_importances_
        fig3, ax3 = plt.subplots(figsize=(14, 6))
        sns.barplot(x=features, y=feature_importance, palette='viridis', ax=ax3)
        ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45)
        ax3.set_title(f'{prefix} - Feature Importance (Random Forest)')
        feature_importance_path = save_plot(fig3, 'feature_importance', prefix)
        
        # 4. PCA Visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_train_scaled)
        fig4, ax4 = plt.subplots(figsize=(8, 6))
        scatter = ax4.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train, cmap='viridis', alpha=0.6)
        plt.colorbar(scatter, label='Rating')
        ax4.set_title(f'{prefix} - PCA Visualization')
        pca_path = save_plot(fig4, 'pca', prefix)
        
        # 5. Correlation Heatmap
        fig5, ax5 = plt.subplots(figsize=(12, 10))
        corr_matrix = df[features + ['Rating']].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax5)
        ax5.set_title(f'{prefix} - Correlation Heatmap')
        corr_path = save_plot(fig5, 'correlation', prefix)
        
        # 6. Residual Plot
        residuals = y_test - voting_pred
        fig6, ax6 = plt.subplots(figsize=(8, 6))
        sns.scatterplot(x=voting_pred, y=residuals, color='purple', alpha=0.6, ax=ax6)
        ax6.axhline(0, color='red', linestyle='--')
        ax6.set_title(f'{prefix} - Residual Plot (Voting Regressor)')
        ax6.set_xlabel('Predicted Rating')
        ax6.set_ylabel('Residuals')
        residual_path = save_plot(fig6, 'residual', prefix)
        
        # 7. Learning Curves (Random Forest)
        from sklearn.model_selection import learning=sklearn.model_selection.learning_curve
        train_sizes, train_scores, val_scores = learning_curve(
            RandomForestRegressor(n_estimators=100, random_state=42),
            X_train_scaled, y_train, cv=5, scoring='r2'
        )
        fig7, ax7 = plt.subplots(figsize=(8, 6))
        ax7.plot(train_sizes, train_scores.mean(axis=1), label='Training Score')
        ax7.plot(train_sizes, val_scores.mean(axis=1), label='Validation Score')
        ax7.set_title(f'{prefix} - Learning Curves (Random Forest)')
        ax7.set_xlabel('Training Examples')
        ax7.set_ylabel('R2 Score')
        ax7.legend()
        learning_curve_path = save_plot(fig7, 'learning_curve', prefix)
        
        # 8. Distribution of Ratings
        fig8 = px.histogram(df, x='Rating', nbins=20, title=f'{prefix} - Rating Distribution')
        rating_dist_path = save_plotly_fig(fig8, 'rating_distribution', prefix)
        
        # Processing Time
        elapsed_time = time.time() - start_time
        logger.info(f"[{prefix}] Processing Time: {elapsed_time:.2f} seconds")
        
        return {
            'df': df,
            'voting_model': voting_model,
            'stacking_model': stacking_model,
            'scaler': scaler,
            'minmax_scaler': minmax_scaler,
            'label_encoders': label_encoders,
            'performances': performances,
            'cv_scores': cv_scores,
            'plot_paths': {
                'rmse': rmse_path,
                'pred_vs_actual': pred_vs_actual_path,
                'feature_importance': feature_importance_path,
                'pca': pca_path,
                'correlation': corr_path,
                'residual': residual_path,
                'learning_curve': learning_curve_path,
                'rating_distribution': rating_dist_path
            }
        }
    
    except Exception as e:
        logger.error(f"Error processing dataset {path}: {str(e)}")
        raise

# ------------- Load and Process Datasets -------------
try:
    main_data = process_dataset("restaurant.csv", "main", hyperparameter_tuning=True)
    s1_data = process_dataset("s1.csv", "s1")
    s2_data = process_dataset("s2.csv", "s2")
    
    df_main = main_data['df']
    voting_model_main = main_data['voting_model']
    stacking_model_main = main_data['stacking_model']
    scaler_main = main_data['scaler']
    minmax_scaler_main = main_data['minmax_scaler']
    label_enc_main = main_data['label_encoders']
    performances_main = main_data['performances']
    cv_scores_main = main_data['cv_scores']
    plot_paths_main = main_data['plot_paths']
    
except Exception as e:
    logger.error(f"Failed to load datasets: {str(e)}")
    raise

# ------------- Restaurant List for UI -------------
restaurant_names = list(df_main['Restaurant_Name'].unique())

# ------------- Helper Functions -------------
def get_rating_category(rating):
    if rating >= 4.5: return "Excellent 🌟"
    elif rating >= 4.0: return "Good 👍"
    elif rating >= 3.0: return "Average ⚠️"
    else: return "Poor ❌"

def get_sentiment_text(score):
    if score > 0.5: return "Positive 😊"
    elif score < -0.5: return "Negative 😞"
    else: return "Neutral 🤔"

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
            logger.warning(f"Restaurant not found: {name}")
            return render_template("result.html", error="Restaurant not found.")
        
        features = [
            'Location', 'Cuisine_Type', 'Cost_for_Two', 'Online_Delivery', 'Book_Table',
            'Average_Wait_Time', 'Staff_Service', 'Sentiment_Score', 'Service_Quality',
            'Cost_Efficiency', 'Wait_Time_Ratio', 'Cost_per_Rating', 'Online_Delivery_Impact',
            'Booking_Efficiency'
        ] + [col for col in row.columns if 'interaction_' in col]
        
        X = row[features]
        X_scaled = scaler_main.transform(X)
        X_mm = minmax_scaler_main.transform(X)
        
        voting_pred = voting_model_main.predict(X_scaled)[0]
        stacking_pred = stacking_model_main.predict(X_scaled)[0]
        actual = row['Rating'].values[0]
        
        sentiment = row['Sentiment_Score'].values[0]
        sentiment_text = get_sentiment_text(sentiment)
        
        # Result Visualization
        fig = go.Figure()
        fig.add_trace(go.Bar(x=["Voting Predicted", "Stacking Predicted", "Actual"],
                           y=[voting_pred, stacking_pred, actual],
                           marker_color=['blue', 'purple', 'green']))
        fig.update_layout(title=f"{name} - Rating Comparison",
                         yaxis_range=[0, 5],
                         height=400)
        result_plot_path = save_plotly_fig(fig, 'result_plot', 'result')
        
        return render_template("result.html",
                             restaurant=name,
                             voting_predicted=round(voting_pred, 2),
                             stacking_predicted=round(stacking_pred, 2),
                             actual_rating=round(actual, 2),
                             rating_category=get_rating_category(voting_pred),
                             location=label_enc_main['Location'].inverse_transform([row['Location'].values[0]])[0],
                             cuisine=label_enc_main['Cuisine_Type'].inverse_transform([row['Cuisine_Type'].values[0]])[0],
                             sentiment=sentiment_text,
                             cost_for_two=row['Cost_for_Two'].values[0],
                             wait_time=row['Average_Wait_Time'].values[0],
                             staff_service=row['Staff_Service'].values[0],
                             plot_paths=plot_paths_main,
                             result_plot=result_plot_path)
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return render_template("result.html", error=str(e))

@app.route('/model_performance')
def model_performance():
    return render_template("performance.html",
                         performances=performances_main,
                         cv_scores=cv_scores_main,
                         plot_paths=plot_paths_main)

@app.route('/api/restaurants', methods=['GET'])
def api_restaurants():
    return jsonify({'restaurants': restaurant_names})

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json()
        name = data.get('restaurant')
        row = df_main[df_main['Restaurant_Name'] == name]
        
        if row.empty:
            return jsonify({'error': 'Restaurant not found'}), 404
        
        features = [
            'Location', 'Cuisine_Type', 'Cost_for_Two', 'Online_Delivery', 'Book_Table',
            'Average_Wait_Time', 'Staff_Service', 'Sentiment_Score', 'Service_Quality',
            'Cost_Efficiency', 'Wait_Time_Ratio', 'Cost_per_Rating', 'Online_Delivery_Impact',
            'Booking_Efficiency'
        ] + [col for col in row.columns if 'interaction_' in col]
        
        X = row[features]
        X_scaled = scaler_main.transform(X)
        voting_pred = voting_model_main.predict(X_scaled)[0]
        stacking_pred = stacking_model_main.predict(X_scaled)[0]
        
        return jsonify({
            'restaurant': name,
            'voting_predicted': round(voting_pred, 2),
            'stacking_predicted': round(stacking_pred, 2),
            'actual': round(row['Rating'].values[0], 2),
            'rating_category': get_rating_category(voting_pred),
            'sentiment': get_sentiment_text(row['Sentiment_Score'].values[0])
        })
    
    except Exception as e:
        logger.error(f"API prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

# ------------- Run the App -------------
if __name__ == "__main__":
    logger.info("Starting Flask application")
    app.run(debug=True, host='0.0.0.0', port=5000)