import os
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Import our preprocessor from data_preprocessing.py
from data_preprocessing import get_preprocessing_pipeline


def train_and_evaluate_model():
  # 1. Load Dataset
  base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  csv_path = os.path.join(base_dir, "data", "raw", "house_prices.csv")

  df = pd.read_csv(csv_path)

  # 2. Separate Features and Target
  X = df.drop(["Price", "Property_ID"], axis=1)
  y = df["Price"]

  # 3. Create full pipeline (Preprocessor + Regressor)
  preprocessor = get_preprocessing_pipeline()
  pipeline = Pipeline(
      steps=[
          ("preprocessor", preprocessor),
          (
              "regressor",
              RandomForestRegressor(
                  n_estimators=100, random_state=42, n_jobs=-1
              ),
          ),
      ]
  )

  # 4. Train-Test Split (80% training, 20% testing)
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=42
  )

  print("🔄 Training the Machine Learning model...")
  pipeline.fit(X_train, y_train)

  # 5. Evaluate the Model
  y_pred = pipeline.predict(X_test)
  mae = mean_absolute_error(y_test, y_pred)
  r2 = r2_score(y_test, y_pred)

  print("\n📊 MODEL PERFORMANCE REPORT:")
  print(f"- Mean Absolute Error (MAE): ₹{mae:,.2f}")
  print(f"- R² Score: {r2:.3f}")

  # 6. Save the Trained Model to models/ folder
  models_dir = os.path.join(base_dir, "models")
  os.makedirs(models_dir, exist_ok=True)
  model_path = os.path.join(models_dir, "model_v1.0.pkl")

  with open(model_path, "wb") as f:
    pickle.dump(pipeline, f)

  print(f"\n✅ Model successfully saved at: {model_path}")
  return pipeline, mae, r2


if __name__ == "__main__":
  train_and_evaluate_model()