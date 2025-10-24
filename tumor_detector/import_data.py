import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler


def transform_delta(X_train: pd.DataFrame, X_test: pd.DataFrame):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def main():
    dataset = fetch_ucirepo(id=696)

    x = dataset.data.features.copy()
    y = dataset.data.targets["Target"]

    models = {
        "Random Forest":{
            "model": RandomForestClassifier(random_state=42),
            "params":{
                "n_estimators": [100, 200],
                "max_depth": [None, 10, 20]
            }

        }

    }

