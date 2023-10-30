import pandas as pd

def load_and_preprocess_data(filepath):
    data = pd.read_csv(filepath)
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data['hour_of_day'] = data['timestamp'].dt.hour
    X = data[['hour_of_day']]
    y = data['engagement']
    return X, y
