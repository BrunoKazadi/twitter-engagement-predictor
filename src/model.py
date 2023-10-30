from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import config

def split_data(X, y):
    return train_test_split(X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE)

def train_model(X_train, y_train):
    model = RandomForestRegressor(random_state=config.RANDOM_STATE)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print('Mean Squared Error:', mse)

def predict_best_hour_for_engagement(model):
    hours = range(24)
    engagement_predictions = model.predict([[hour] for hour in hours])
    best_hour = hours[np.argmax(engagement_predictions)]
    return best_hour
