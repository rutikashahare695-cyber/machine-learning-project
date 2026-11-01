import os
import sys
import pandas as pd
from flask import Flask, render_template, request

# Add 'src' directory explicitly to Python system path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, "..", "src"))
if src_dir not in sys.path:
  sys.path.insert(0, src_dir)

from model_inference import load_model

app = Flask(__name__)

# Load model on startup
model = load_model()


@app.route("/", methods=["GET", "POST"])
def index():
  prediction = None
  error = None

  if request.method == "POST":
    try:
      # Extract data from form inputs
      area = float(request.form["area"])
      bedrooms = int(request.form["bedrooms"])
      bathrooms = int(request.form["bathrooms"])
      age = int(request.form["age"])
      location = request.form["location"]
      property_type = request.form["property_type"]

      # Prepare input dataframe
      input_data = pd.DataFrame([{
          "Area": area,
          "Bedrooms": bedrooms,
          "Bathrooms": bathrooms,
          "Age": age,
          "Location": location,
          "Property_Type": property_type,
      }])

      # Predict price
      pred_price = model.predict(input_data)[0]
      prediction = f"₹{pred_price:,.2f}"

    except Exception as e:
      error = f"Error in prediction: {str(e)}"

  return render_template(
      "index.html", prediction=prediction, error=error
  )


if __name__ == "__main__":
  app.run(debug=True, port=5000)