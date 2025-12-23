from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from datetime import datetime 
import argparse
import pandas as pd
import mlflow
import mlflow.sklearn

# Load dataset
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="diamond_preprocessing.csv")
args = parser.parse_args()

try:
    df = pd.read_csv(args.data_path)
    X = df[['carat', 'cut_enc', 'color_enc', 'clarity_enc']]
    y = df['price'].to_numpy()

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standarisasi
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # MLflow setup
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Diamond_RF_Tuning")
    mlflow.sklearn.autolog()

    # Grid search
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None]
    }

    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(rf, param_grid=param_grid, cv=3, scoring='r2')
    run_name = "RF_GridSearch_" + datetime.now().strftime("%Y%m%d_%H%M%S")

    with mlflow.start_run(run_name=run_name) as run:
        try:
            grid_search.fit(X_train_scaled, y_train)
            best_model = grid_search.best_estimator_

            y_pred = best_model.predict(X_test_scaled)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)

            print(f"Run ID: {run.info.run_id}")
            print(f"Run Name: {run.info.run_name}")
            print(f"Best Hyperparameters: {grid_search.best_params_}")
            print(f"R2 Score: {r2:.4f}")
            print(f"Mean Absolute Error: {mae:.2f}")

        except Exception as e:
            # Log error ke MLflow
            mlflow.log_text(str(e), "error.log")
            print("Terjadi error saat training / grid search:")
            print(e)

except Exception as e:
    print("Terjadi error saat load data atau setup MLflow:")
    print(e)