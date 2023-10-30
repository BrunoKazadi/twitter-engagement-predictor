from src.data_preprocessing import load_and_preprocess_data
from src.model import split_data, train_model, evaluate_model, predict_best_hour_for_engagement
from datetime import datetime, timedelta
from src.utils import plot_predictions
import pandas as pd
import numpy as np
import random
import config



# Set a random seed for reproducibility
random.seed(42)

# Generate 1000 timestamps over the past month
timestamps = [datetime.now() - timedelta(hours=i) for i in range(1000)]

# Generate engagement scores with a pattern (higher engagement during the day)
engagement = [random.randint(50, 150) if 8 <= timestamp.hour <= 16 else random.randint(5, 50) for timestamp in timestamps]

# Create a DataFrame
data = pd.DataFrame({'timestamp': timestamps, 'engagement': engagement})

# Save to CSV
data.to_csv('twitter_data.csv', index=False)


def main():
    # Load and preprocess data
    X, y = load_and_preprocess_data('twitter_data.csv')

    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train model
    model = train_model(X_train, y_train)

    # Evaluate model
    evaluate_model(model, X_test, y_test)

    # Predict best hour for engagement
    best_hour = predict_best_hour_for_engagement(model)
    print('The hour of the day that maximizes engagement is:', best_hour)

    # Optional: Plot the predictions
    plot_predictions(model)

if __name__ == "__main__":
    main()
