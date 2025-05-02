import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Download NLP model
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Load dataset
file_path = "restaurant.csv"
df = pd.read_csv(file_path)

# Data Cleaning
df.columns = df.columns.str.strip()
df.drop_duplicates(inplace=True)
df.fillna({'Reviews': "No reviews"}, inplace=True)

# Sentiment Analysis for Reviews
def analyze_sentiment(review):
    score = sia.polarity_scores(review)['compound']
    return score

df['Sentiment_Score'] = df['Reviews'].apply(analyze_sentiment)

# Convert categorical data to numerical
label_encoders = {}
categorical_columns = ['Location', 'Cuisine_Type', 'Online_Delivery', 'Book_Table']

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Feature Engineering: Create new feature based on service quality
df['Service_Quality'] = df['Staff_Service'] / df['Average_Wait_Time']

# Features & Target
X = df[['Location', 'Cuisine_Type', 'Cost_for_Two', 'Online_Delivery', 'Book_Table', 'Average_Wait_Time', 'Staff_Service', 'Sentiment_Score', 'Service_Quality']]
y = df['Rating']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize Models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Lasso Regression": Lasso(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "AdaBoost": AdaBoostRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
}

# Train & Evaluate Models
model_performance = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    model_performance[name] = {'MSE': mse, 'R2 Score': r2}

# Convert performance results to DataFrame
performance_df = pd.DataFrame(model_performance).T

# Select Best Model
best_model_name = performance_df['R2 Score'].idxmax()
best_model = models[best_model_name]

print(f"\nBest Model Selected: {best_model_name}")

# Function to Categorize Ratings
def categorize_rating(rating):
    if rating >= 4.0:
        return "Good ✔️"
    elif rating >= 3.0:
        return "Average ⚠️"
    else:
        return "Bad ❌"

# Function to Predict Rating for a Given Restaurant
def predict_restaurant_rating(restaurant_name):
    restaurant = df[df['Restaurant_Name'] == restaurant_name]
    
    if restaurant.empty:
        return "Restaurant not found in database."
    
    features = restaurant[['Location', 'Cuisine_Type', 'Cost_for_Two', 'Online_Delivery', 'Book_Table', 'Average_Wait_Time', 'Staff_Service', 'Sentiment_Score', 'Service_Quality']]
    features_scaled = scaler.transform(features)
    
    predicted_rating = best_model.predict(features_scaled)[0]
    sentiment_score = restaurant['Sentiment_Score'].values[0]
    
    if sentiment_score > 0.5:
        sentiment_text = "Positive customer experience!"
    elif sentiment_score < -0.5:
        sentiment_text = "Mostly negative reviews."
    else:
        sentiment_text = "Mixed reviews from customers."

    return {
        "Restaurant": restaurant_name,
        "Predicted Rating": round(predicted_rating, 2),
        "Rating Category": categorize_rating(predicted_rating),
        "Cost for Two": restaurant['Cost_for_Two'].values[0],
        "Location": label_encoders['Location'].inverse_transform([restaurant['Location'].values[0]])[0],
        "Cuisine Type": label_encoders['Cuisine_Type'].inverse_transform([restaurant['Cuisine_Type'].values[0]])[0],
        "Customer Sentiment": sentiment_text
    }

# Example Predictions
restaurant_to_predict = "Mehfil"
prediction_result = predict_restaurant_rating(restaurant_to_predict)
print("\nPrediction Result:")
for key, value in prediction_result.items():
    print(f"{key}: {value}")

# Improved Visualization with Annotations
plt.figure(figsize=(12,6))
ax = sns.barplot(x=performance_df.index, y=performance_df['R2 Score'], palette="coolwarm")
plt.xticks(rotation=45)
plt.title("Model Performance Comparison (R2 Score)", fontsize=14, fontweight='bold')
plt.ylabel("R2 Score", fontsize=12)
plt.xlabel("Machine Learning Models", fontsize=12)

# Adding value labels on bars
for p in ax.patches:
    ax.annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2, p.get_height()), ha='center', va='bottom', fontsize=11, fontweight='bold', color='black')

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()