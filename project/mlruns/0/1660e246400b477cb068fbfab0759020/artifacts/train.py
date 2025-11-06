import argparse
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
DATA_PATH = "../data/winequality-white.csv"

def load_and_prepare_data():
    df = pd.read_csv(DATA_PATH, sep=';')
    X = df.drop("quality", axis=1)
    y = df["quality"]
    return train_test_split(X, y, test_size=0.2, random_state=42)


def main(n_estimators, max_depth, random_state):

    with mlflow.start_run(nested=True):  

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", random_state)

        X_train, X_test, y_train, y_test = load_and_prepare_data()
        feature_names = X_train.columns.tolist()

        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="wine_quality_model",
            registered_model_name="WineQualityModel"
        )

        mlflow.log_artifact(__file__)

        print(f"Modelo registrado. RMSE: {rmse:.4f}, R²: {r2:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=10)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    main(args.n_estimators, args.max_depth, args.random_state)
