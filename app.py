import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, VotingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
import catboost as cb
import joblib
from flask import Flask, request, render_template

# Initialize Flask app
app = Flask(__name__)

# Download sentiment model
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Load dataset
DATASET_PATH = "restaurant.csv"
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError("Dataset not found! Please provide 'restaurant.csv'.")
    
df = pd.read_csv(DATASET_PATH)
df.drop_duplicates(inplace=True)
df.fillna({'Reviews': "No reviews"}, inplace=True)

# Sentiment Analysis
def analyze_sentiment(review):
    scores = sia.polarity_scores(str(review))
    return scores['compound']

df['Sentiment_Score'] = df['Reviews'].apply(analyze_sentiment)

# Encode categorical data
label_encoders = {}
categorical_columns = ['Location', 'Cuisine_Type', 'Online_Delivery', 'Book_Table']
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Feature Engineering
df['Service_Quality'] = df['Staff_Service'] / (df['Average_Wait_Time'] + 1)
df['Cost_Efficiency'] = df['Rating'] / (df['Cost_for_Two'] + 1)

# Define Features & Target
X = df[['Location', 'Cuisine_Type', 'Cost_for_Two', 'Online_Delivery', 'Book_Table', 
        'Average_Wait_Time', 'Staff_Service', 'Sentiment_Score', 'Service_Quality', 
        'Cost_Efficiency']]
y = df['Rating']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define Models
models = {
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "Extra Trees": ExtraTreesRegressor(n_estimators=100, random_state=42),
    "XGBoost": xgb.XGBRegressor(n_estimators=100, random_state=42),
    "CatBoost": cb.CatBoostRegressor(n_estimators=100, verbose=0, random_state=42),
    "SVR": SVR(kernel='rbf', C=100),
    "KNN": KNeighborsRegressor(n_neighbors=5),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.1),
    "Linear Regression": LinearRegression(),
    "AdaBoost": AdaBoostRegressor(n_estimators=100, random_state=42)
}

# Train and evaluate models
model_performance = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    model_performance[name] = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
    }
    print(f"{name}: RMSE={rmse:.3f}, MAE={mae:.3f}")

# Create Voting Regressor with top performers
top_models = sorted(model_performance.items(), key=lambda x: x[1]['RMSE'])[:5]
voting_estimators = [(name, models[name]) for name, _ in top_models]
voting_regressor = VotingRegressor(estimators=voting_estimators)
voting_regressor.fit(X_train_scaled, y_train)

# Evaluate Voting Regressor
y_pred_voting = voting_regressor.predict(X_test_scaled)
voting_performance = {
    'MSE': mean_squared_error(y_test, y_pred_voting),
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_voting)),
    'MAE': mean_absolute_error(y_test, y_pred_voting)
}
print(f"\nVoting Regressor: RMSE={voting_performance['RMSE']:.3f}, MAE={voting_performance['MAE']:.3f}")

# Save the best model
joblib.dump(voting_regressor, 'best_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Graphical Representations
# 1. Model Comparison Plot (RMSE Scores)
plt.figure(figsize=(12, 6))
model_names = list(model_performance.keys())
rmse_scores = [model_performance[name]['RMSE'] for name in model_names]
sns.barplot(x=model_names, y=rmse_scores, palette='viridis')
plt.xticks(rotation=45)
plt.title('Model Comparison - RMSE Scores')
plt.ylabel('RMSE')
plt.tight_layout()
plt.savefig('static/model_comparison_rmse.png')
plt.close()

# 2. Feature Importance (using Random Forest)
rf_model = models['Random Forest']
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance, palette='magma')
plt.title('Feature Importance (Random Forest)')
plt.tight_layout()
plt.savefig('static/feature_importance.png')
plt.close()

# 3. Actual vs Predicted Scatter Plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_voting, alpha=0.5, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Rating')
plt.ylabel('Predicted Rating')
plt.title('Actual vs Predicted Ratings (Voting Regressor)')
plt.tight_layout()
plt.savefig('static/actual_vs_predicted.png')
plt.close()

# 4. Separate Model Performance Graph (Box Plot of Cross-Validation Scores)
cv_data = {name: cross_val_score(models[name], X_train_scaled, y_train, cv=5, scoring='neg_mean_absolute_error') 
          for name in model_names}
cv_df = pd.DataFrame(cv_data)

plt.figure(figsize=(12, 6))
sns.boxplot(data=cv_df, palette='Set2')
plt.xticks(rotation=45)
plt.title('Cross-Validation MAE Scores Across Models')
plt.ylabel('MAE')
plt.xlabel('Model')
plt.tight_layout()
plt.savefig('static/model_cv_boxplot.png')
plt.close()

# Function to Categorize Ratings
def categorize_rating(rating):
    if rating >= 4.5:
        return "Excellent 🌟"
    elif rating >= 4.0:
        return "Good ✔️"
    elif rating >= 3.0:
        return "Average ⚠️"
    else:
        return "Bad ❌"

# Flask Routes
df_restaurants = list(df['Restaurant_Name'].unique())[:50]

@app.route('/')
def home():
    return render_template('index.html', restaurants=df_restaurants)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        restaurant_name = request.form.get('restaurant')
        restaurant = df[df['Restaurant_Name'] == restaurant_name]
        if restaurant.empty:
            return render_template('result.html', error="Restaurant not found.")

        # Prepare Features for Prediction
        features = restaurant[['Location', 'Cuisine_Type', 'Cost_for_Two', 'Online_Delivery', 
                              'Book_Table', 'Average_Wait_Time', 'Staff_Service', 
                              'Sentiment_Score', 'Service_Quality', 'Cost_Efficiency']]
        features_scaled = scaler.transform(features)
        predicted_rating = voting_regressor.predict(features_scaled)[0]

        # Sentiment Analysis
        sentiment_score = restaurant['Sentiment_Score'].values[0]
        sentiment_text = (
            "Positive customer experience! 😊" if sentiment_score > 0.5 else
            "Mostly negative reviews. 😞" if sentiment_score < -0.5 else
            "Mixed reviews from customers. 🤔"
        )

        # Generate Prediction Graph
        plt.figure(figsize=(6, 4))
        sns.barplot(x=["Predicted", "Actual"], y=[predicted_rating, restaurant['Rating'].values[0]], 
                   palette=['blue', 'green'])
        plt.ylim(0, 5)
        plt.xlabel('Type')
        plt.ylabel('Rating')
        plt.title(f'Predicted vs Actual Rating - {restaurant_name}')
        for i, v in enumerate([predicted_rating, restaurant['Rating'].values[0]]):
            plt.text(i, v + 0.1, f'{v:.2f}', ha='center')
        
        if not os.path.exists('static'):
            os.makedirs('static')
        plot_path = os.path.join('static', 'plot.png')
        plt.savefig(plot_path)
        plt.close()

        # Decode categorical features for display
        location_decoded = label_encoders['Location'].inverse_transform([restaurant['Location'].values[0]])[0]
        cuisine_decoded = label_encoders['Cuisine_Type'].inverse_transform([restaurant['Cuisine_Type'].values[0]])[0]

        # Render Result Page
        return render_template(
            'result.html',
            restaurant=restaurant_name,
            predicted_rating=round(predicted_rating, 2),
            rating_category=categorize_rating(predicted_rating),
            cost=restaurant['Cost_for_Two'].values[0],
            location=location_decoded,
            cuisine=cuisine_decoded,
            sentiment_text=sentiment_text,
            plot_url='static/plot.png',
            model_comparison_url='static/model_comparison_rmse.png',
            feature_importance_url='static/feature_importance.png',
            actual_vs_predicted_url='static/actual_vs_predicted.png',
            model_cv_boxplot_url='static/model_cv_boxplot.png'
        )
    except Exception as e:
        return render_template('result.html', error=f"Error: {str(e)}")

@app.route('/restaurant_finder')
def restaurant_finder():
    locations = sorted(df['Location'].map(lambda x: label_encoders['Location'].inverse_transform([x])[0]).unique())
    cuisines = sorted(df['Cuisine_Type'].map(lambda x: label_encoders['Cuisine_Type'].inverse_transform([x])[0]).unique())
    return render_template('restaurant_finder.html', locations=locations, cuisines=cuisines)

@app.route('/filter_restaurants', methods=['POST'])
def filter_restaurants():
    try:
        filters = {
            'location': request.form.get('location'),
            'cuisine': request.form.get('cuisine'),
            'cost': request.form.get('cost'),
            'online_delivery': request.form.get('online_delivery'),
            'book_table': request.form.get('book_table')
        }

        filtered_df = df.copy()
        if filters['location']:
            location_encoded = label_encoders['Location'].transform([filters['location']])[0]
            filtered_df = filtered_df[filtered_df['Location'] == location_encoded]
        if filters['cuisine']:
            cuisine_encoded = label_encoders['Cuisine_Type'].transform([filters['cuisine']])[0]
            filtered_df = filtered_df[filtered_df['Cuisine_Type'] == cuisine_encoded]
        if filters['cost']:
            filtered_df = filtered_df[filtered_df['Cost_for_Two'] <= float(filters['cost'])]
        if filters['online_delivery']:
            delivery_encoded = label_encoders['Online_Delivery'].transform([filters['online_delivery']])[0]
            filtered_df = filtered_df[filtered_df['Online_Delivery'] == delivery_encoded]
        if filters['book_table']:
            table_encoded = label_encoders['Book_Table'].transform([filters['book_table']])[0]
            filtered_df = filtered_df[filtered_df['Book_Table'] == table_encoded]

        # Decode categorical columns for display
        filtered_df['Location'] = filtered_df['Location'].map(lambda x: label_encoders['Location'].inverse_transform([x])[0])
        filtered_df['Cuisine_Type'] = filtered_df['Cuisine_Type'].map(lambda x: label_encoders['Cuisine_Type'].inverse_transform([x])[0])
        filtered_df['Online_Delivery'] = filtered_df['Online_Delivery'].map(lambda x: label_encoders['Online_Delivery'].inverse_transform([x])[0])
        filtered_df['Book_Table'] = filtered_df['Book_Table'].map(lambda x: label_encoders['Book_Table'].inverse_transform([x])[0])

        locations = sorted(df['Location'].map(lambda x: label_encoders['Location'].inverse_transform([x])[0]).unique())
        cuisines = sorted(df['Cuisine_Type'].map(lambda x: label_encoders['Cuisine_Type'].inverse_transform([x])[0]).unique())
        return render_template('restaurant_finder.html', 
                             restaurants=filtered_df.to_dict('records'),
                             locations=locations,
                             cuisines=cuisines)
    except Exception as e:
        locations = sorted(df['Location'].map(lambda x: label_encoders['Location'].inverse_transform([x])[0]).unique())
        cuisines = sorted(df['Cuisine_Type'].map(lambda x: label_encoders['Cuisine_Type'].inverse_transform([x])[0]).unique())
        return render_template('restaurant_finder.html', 
                             error=f"Error filtering restaurants: {str(e)}",
                             locations=locations,
                             cuisines=cuisines)

if __name__ == '__main__':
    app.run(debug=True)