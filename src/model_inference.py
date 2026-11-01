import os
import pickle
import pandas as pd


def load_model():
  """Loads the trained machine learning pipeline from the models folder."""
  base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  model_path = os.path.join(base_dir, "models", "model_v1.0.pkl")

  if not os.path.exists(model_path):
    raise FileNotFoundError(
        f"Model file not found at {model_path}. Please run model_training.py"
        " first."
    )

  with open(model_path, "rb") as f:
    model = pickle.load(f)
  return model


def predict_house_price(house_features: dict):
  """Takes a dictionary of house features and returns the predicted price."""
  model = load_model()

  # Convert the input dictionary into a pandas DataFrame (expected by the pipeline)
  input_df = pd.DataFrame([house_features])

  # Make prediction
  prediction = model.predict(input_df)
  return prediction[0]


if __name__ == "__main__":
  # Test with a sample house input
  sample_house = {
      "Area": 2500,  # sqft
      "Bedrooms": 3,
      "Bathrooms": 2,
      "Age": 8,  # years
      "Location": "City Center",  # e.g., City Center, Suburb, Rural
      "Property_Type": "Apartment",  # e.g., Apartment, House, Villa
  }

  print("🔄 Making prediction for sample house...")
  predicted_price = predict_house_price(sample_house)

  print("\n🏠 PREDICTION RESULT:")
  print(f"- Input Features: {sample_house}")
  print(f"- Estimated House Price: ₹{predicted_price:,.2f}")