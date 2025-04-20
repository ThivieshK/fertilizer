from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model and scaler
scaler = joblib.load("scaler.pkl")
rf_model = joblib.load("random_forest_model.pkl")

# Fertilizer Mapping
fertilizer_mapping = {0: "10-26-26", 1: "14-35-14", 2: "17-17-17", 3: "20-20", 4: "28-28", 5: "DAP", 6: "Urea"}

# Fertilizer Recommendations and Amounts
fertilizer_recommendations = {
    '10-26-26': {"amount": "120-150 kg/ha (NPK Ratio: 10% Nitrogen (N), 26% Phosphorus (P), 26% Potassium (K))", "suggestion": "Use for balanced crop growth.This fertilizer ensures a balanced supply of phosphorus and potassium with a moderate nitrogen level."},
    '14-35-14': {"amount": "100-130 kg/ha (NPK Ratio: 14% Nitrogen, 35% Phosphorus, 14% Potassium)", "suggestion": "Best for phosphorus-rich soil.High phosphorus content makes it ideal for crops requiring significant root and flower development."},
    '17-17-17': {"amount": "150-180 kg/ha (NPK Ratio: 17% Nitrogen, 17% Phosphorus, 17% Potassium)", "suggestion": "Maintains nitrogen balance in crops.A well-balanced fertilizer that ensures uniform plant growth."},
    '20-20': {"amount": "90-120 kg/ha (NPK Ratio: 20% Nitrogen, 20% Phosphorus)", "suggestion": "Ideal for moderate nutrient needs.Primarily provides nitrogen and phosphorus, making it useful for moderate fertilization."},
    '28-28': {"amount": "80-100 kg/ha (NPK Ratio: 28% Nitrogen, 28% Phosphorus)", "suggestion": "Best for phosphorus-demanding crops.High in both nitrogen and phosphorus, useful for crops that require strong early growth."},
    'DAP': {"amount": "50-70 kg/ha (NPK Ratio: 18% Nitrogen, 46% Phosphorus)", "suggestion": "Good for early-stage crop growth.One of the most widely used phosphorus fertilizers, helping in early-stage plant establishment."},
    'Urea': {"amount": "50-100 kg/ha (Composition: 46% Nitrogen)", "suggestion": "Helps nitrogen-deficient soil.A high-nitrogen fertilizer that helps in correcting nitrogen deficiency."}
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = [float(request.form[feature]) for feature in ["Temperature", "Humidity", "Moisture", "Soil_Type", "Crop_Type", "Nitrogen", "Potassium", "Phosphorous"]]
        data_scaled = scaler.transform([data])
        
        prediction_index = rf_model.predict(data_scaled)[0]
        predicted_fertilizer = fertilizer_mapping[prediction_index]
        recommendation = fertilizer_recommendations.get(predicted_fertilizer, {"amount": "N/A", "suggestion": "General soil maintenance recommended."})
        
        return render_template('result.html', 
                               fertilizer=predicted_fertilizer, 
                               amount=recommendation["amount"], 
                               suggestion=recommendation["suggestion"])
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
