import os
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def get_preprocessing_pipeline():
  """Creates and returns a preprocessing pipeline for numeric and categorical features."""
  numeric_features = ["Area", "Bedrooms", "Bathrooms", "Age"]
  categorical_features = ["Location", "Property_Type"]

  numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
  categorical_transformer = Pipeline(
      steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
  )

  preprocessor = ColumnTransformer(
      transformers=[
          ("num", numeric_transformer, numeric_features),
          ("cat", categorical_transformer, categorical_features),
      ]
  )
  return preprocessor


if __name__ == "__main__":
  # Safely locate the CSV file relative to project root
  base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  csv_path = os.path.join(base_dir, "data", "raw", "house_prices.csv")

  df = pd.read_csv(csv_path)
  X = df.drop(["Price", "Property_ID"], axis=1)

  preprocessor = get_preprocessing_pipeline()
  transformed_data = preprocessor.fit_transform(X)

  print("✅ Preprocessing pipeline executed successfully!")
  print(f"Transformed data shape: {transformed_data.shape}")