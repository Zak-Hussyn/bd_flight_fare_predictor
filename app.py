from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Load the model
model = joblib.load('fare_prediction_model.pkl')

# Initialize Flask app
app = Flask(__name__)

# Define expected columns in the same order as training
expected_cols = ['Airline', 'Source', 'Source Name', 'Destination',
'Destination Name', 'Duration (hrs)', 'Stopovers', 'Aircraft Type',
'Class', 'Booking Source', 'Base Fare (BDT)', 'Tax & Surcharge (BDT)',
'Seasonality', 'Days Before Departure', 'Departure_Hour', 'Departure_Day',
'Departure_Month', 'Departure_Weekend'
]

# Preprocessing function
def preprocess_input(data):
    df = pd.DataFrame([data])

    # Convert datetime
    df['Departure Date & Time'] = pd.to_datetime(df['Departure Date & Time'])
    df['Arrival Date & Time'] = pd.to_datetime(df['Arrival Date & Time'])

    # Extract datetime features
    df['Departure_Hour'] = df['Departure Date & Time'].dt.hour
    df['Departure_Day'] = df['Departure Date & Time'].dt.dayofweek
    df['Departure_Month'] = df['Departure Date & Time'].dt.month
    df['Departure_Weekend'] = (df['Departure_Day'] >= 5).astype(int)

    # Drop raw datetime fields
    df.drop(columns=['Departure Date & Time', 'Arrival Date & Time'], inplace=True)

    # Encode categorical features (replace with your own mappings if label encoder was used)
    label_mappings = {
        'Airline': { 'Air Arabia': 0, 'Air Astra': 1, 'Air India': 2, 'AirAsia': 3, 'Biman Bangladesh Airlines': 4, 'British Airways': 5, 'Cathay Pacific': 6, 'Emirates': 7, 'Etihad Airways': 8, 'FlyDubai': 9, 'Gulf Air': 10, 'IndiGo': 11, 'Kuwait Airways': 12, 'Lufthansa': 13, 'Malaysian Airlines': 14, 'NovoAir': 15, 'Qatar Airways': 16, 'Saudia': 17, 'Singapore Airlines': 18, 'SriLankan Airlines': 19, 'Thai Airways': 20, 'Turkish Airlines': 21, 'US-Bangla Airlines': 22, 'Vistara': 23},
        'Source': {'BZL': 0, 'CGP': 1, 'CXB': 2, 'DAC': 3, 'JSR': 4, 'RJH': 5, 'SPD': 6, 'ZYL': 7},
    'Source Name': {'Barisal Airport': 0, "Cox's Bazar Airport": 1, 'Hazrat Shahjalal International Airport, Dhaka': 2, 'Jessore Airport': 3, 'Osmani International Airport, Sylhet': 4, 'Saidpur Airport': 5, 'Shah Amanat International Airport, Chittagong': 6, 'Shah Makhdum Airport, Rajshahi': 7, 'nan': 8},
    'Destination': {'BKK': 0, 'BZL': 1, 'CCU': 2, 'CGP': 3, 'CXB': 4, 'DAC': 5, 'DEL': 6, 'DOH': 7, 'DXB': 8, 'IST': 9, 'JED': 10, 'JFK': 11, 'JSR': 12, 'KUL': 13, 'LHR': 14, 'RJH': 15, 'SIN': 16, 'SPD': 17, 'YYZ': 18, 'ZYL': 19, 'nan': 20},
    'Destination Name': {'Barisal Airport': 0, "Cox's Bazar Airport": 1, 'Dubai International Airport': 2, 'Hamad International Airport, Doha': 3, 'Hazrat Shahjalal International Airport, Dhaka': 4, 'Indira Gandhi International Airport, Delhi': 5, 'Istanbul Airport': 6, 'Jessore Airport': 7, 'John F. Kennedy International Airport, New York': 8, 'King Abdulaziz International Airport, Jeddah': 9, 'Kuala Lumpur International Airport': 10, 'London Heathrow Airport': 11, 'Netaji Subhas Chandra Bose International Airport, Kolkata': 12, 'Osmani International Airport, Sylhet': 13, 'Saidpur Airport': 14, 'Shah Amanat International Airport, Chittagong': 15, 'Shah Makhdum Airport, Rajshahi': 16, 'Singapore Changi Airport': 17, 'Suvarnabhumi Airport, Bangkok': 8, 'Toronto Pearson International Airport': 19, 'nan': 20},
    'Stopovers': {'1 Stop': 0, '2 Stops': 1, 'Direct': 2, 'nan': 3},
    'Airline Type': {'Airbus A320': 0, 'Airbus A350': 1, 'Boeing 737': 2, 'Boeing 777': 3, 'Boeing 787': 4, 'nan': 5},
    'Class': {'Business': 0, 'Economy': 1, 'First Class': 2, 'nan': 3},
    'Booking Source': {'Direct Booking': 0, 'Online Website': 1, 'Travel Agency': 2, 'nan': 3},
    'Seasonality': {'Eid': 0, 'Hajj': 1, 'Regular': 2, 'Winter Holidays': 3, 'nan': 4}
    }
    for col in label_mappings:
        df[col] = df[col].map(label_mappings[col])

    # Ensure correct column order
    df = df.reindex(columns=model.feature_names_in_)

    return df

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.json
        processed = preprocess_input(input_data)
        prediction = model.predict(processed)[0]
        return jsonify({'predicted_fare': round(prediction, 2)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)