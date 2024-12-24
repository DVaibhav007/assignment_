import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

# Load the data
data = pd.read_csv("data/data.csv", parse_dates=['timestamp'])

# Sort the data by part_id and timestamp
data = data.sort_values(by=['part_id', 'timestamp'])

# Define the sequence window size (k)
k = 5

# Prepare the windowed dataset
def create_windowed_dataset(data, k):
    X, y = [], []
    grouped = data.groupby('part_id')
    
    for _, group in grouped:
        if len(group) >= k:
            group = group[['temperature_1', 'temperature_2', 'temperature_3', 'temperature_4', 'label']].values
            for i in range(len(group) - k + 1):
                window = group[i:i+k, :-1]  # All columns except label
                label = group[i + k - 1, -1]  # Label of the kth part
                features = np.concatenate([
                    window.max(axis=0),
                    window.min(axis=0),
                    window.std(axis=0)
                ])
                X.append(features)
                y.append(label)
    
    return np.array(X), np.array(y)

X, y = create_windowed_dataset(data, k)

# Scale the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Train a Random Forest classifier
rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
rf_model.fit(X_train, y_train)

# Evaluate the model
y_pred = rf_model.predict(X_test)
print("Random Forest Classification Report:\n")
print(classification_report(y_test, y_pred))
