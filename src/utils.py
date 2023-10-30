import matplotlib.pyplot as plt
import numpy as np

def plot_predictions(model):
    hours = np.array(range(24)).reshape(-1, 1)
    engagement_predictions = model.predict(hours)
    plt.plot(hours, engagement_predictions)
    plt.xlabel('Hour of Day')
    plt.ylabel('Predicted Engagement')
    plt.title('Predicted Engagement Throughout the Day')
    plt.show()
