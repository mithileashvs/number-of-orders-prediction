from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

import matplotlib.pyplot as plt
import joblib
import os


def train_models(df):

    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    X = df.drop(columns=["Order Date", "Orders"])
    y = df["Orders"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            random_state=42,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            random_state=42,
        ),
    }

    results = {}

    best_model = None
    best_predictions = None
    best_score = float("-inf")
    best_model_name = ""

    for name, model in models.items():

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        rmse = mean_squared_error(y_test, predictions) ** 0.5
        r2 = r2_score(y_test, predictions)

        results[name] = {
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
        }

        if r2 > best_score:
            best_score = r2
            best_model = model
            best_predictions = predictions
            best_model_name = name

    joblib.dump(best_model, "models/best_model.pkl", compress=3)

    # -------------------------
    # Actual vs Predicted Plot
    # -------------------------
    plt.figure(figsize=(8,6))

    plt.scatter(
        y_test,
        best_predictions,
        alpha=0.7
    )
    plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    "r--",
    linewidth=2,
    label="Ideal Prediction"
)

    plt.legend()

    plt.xlabel("Actual Orders")
    plt.ylabel("Predicted Orders")
    plt.title(f"Actual vs Predicted ({best_model_name})")

    plt.tight_layout()

    plt.savefig("outputs/actual_vs_predicted.png")

    plt.close()

    # -------------------------
    # Feature Importance
    # -------------------------

    if hasattr(best_model, "feature_importances_"):

        importance = best_model.feature_importances_

        plt.figure(figsize=(8,5))

        plt.bar(X.columns, importance)

        plt.xticks(rotation=45)

        plt.title("Feature Importance")

        plt.tight_layout()

        plt.savefig("outputs/feature_importance.png")

        plt.close()

    return results