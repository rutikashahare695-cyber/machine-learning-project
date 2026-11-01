# 🏡 House Price Prediction System

A complete End-to-End Machine Learning web application built as part of Data Science Internship (Month 4). This system predicts residential property prices based on features like area, bedrooms, bathrooms, age, and location using a trained Random Forest Regressor pipeline.

## 🚀 Project Architecture
- **Data Preprocessing:** Handled using `scikit-learn` Pipelines, `StandardScaler`, and `OneHotEncoder`.
- **Machine Learning Model:** Random Forest Regressor achieving an **$R^2$ Score of ~0.97**.
- **Web Application:** Built using Flask framework with a clean, responsive HTML/CSS interface.
- **Model Persistence:** Serialized using Python `pickle` for fast loading and inference.

## 📂 Project Structure
```text
machine-learning-project/
├── data/raw/              # Raw dataset (house_prices.csv)
├── notebooks/             # Exploratory Data Analysis notebooks
├── src/                   # Core scripts (preprocessing, training, inference)
├── models/                # Saved trained model (.pkl)
├── app/                   # Flask app and HTML templates
├── tests/                 # Unit and integration tests
├── config/                # Configuration files
└── docs/                  # Documentation and data dictionary
```

## 🛠️ Quick Start & Installation
Clone the repository and navigate to the folder.

Install dependencies:

Bash
pip install -r requirements.txt
Run the Flask web application:

Bash
python app/app.py
Open your browser and go to: http://127.0.0.1:5000


---

### 2. `config/config.yaml` File
Apni `config/config.yaml` file mein yeh settings dal dein:

```yaml
model_params:
  n_estimators: 100
  random_state: 42
  test_size: 0.2

features:
  numeric:
    - "Area"
    - "Bedrooms"
    - "Bathrooms"
    - "Age"
  categorical:
    - "Location"
    - "Property_Type"
  target: "Price"
3. docs/data_dictionary.md File
Apni docs/data_dictionary.md file mein yeh details dal dein:

Markdown
# Data Dictionary - House Price Dataset

| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| **Property_ID** | String | Unique identifier for each property. |
| **Area** | Integer | Total floor area of the property in square feet. |
| **Bedrooms** | Integer | Number of bedrooms in the house. |
| **Bathrooms** | Integer | Number of bathrooms. |
| **Age** | Integer | Age of the property in years. |
| **Location** | String | Geographic location category (City Center, Suburb, Rural). |
| **Property_Type** | String | Type of housing structure (Apartment, House, Villa). |
| **Price** | Integer | Market sale price of the property in INR (Target Variable). |
