from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS
import mysql.connector
from datetime import datetime

app = Flask(__name__)
CORS(app, origins=["https://dialogflow.cloud.google.com"])

# Load your trained model
model = joblib.load('random_forest_model.pkl')

# MySQL Database configuration
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'Deepa78*',
    'database': 'healthdata'
}

# Define mappings for categorical variables
gender_mapping = {"Male": 1, "Female": 0}
bmi_mapping = {"Normal Weight": 0, "Overweight": 2, "Obese": 1}
sleep_disorder_mapping = {"No Disorder": 1, "Insomnia": 0, "Sleep Apnea": 2}
food_habits_mapping = {"Healthy": 0, "Moderate": 1, "Unhealthy": 2}
smoking_status_mapping = {"Non-smoker": 0, "Smoker": 1}

# Define a mapping for predicted diseases
disease_mapping = {
    0: "Cardiovascular Disease",
    1: "Hypertension",
    2: "Lung Cancer or Cancer",
    3: "Mental Health Issues",
    4: "Minimal or No Risks",
    5: "No Significant Risks Found"
}

def connect_db():
    return mysql.connector.connect(**db_config)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    
    try:
        # Extract and encode features from the Dialogflow request
        features = np.array([
            gender_mapping[data['queryResult']['parameters']['Gender']],                  # Gender
            int(data['queryResult']['parameters']['Age']),                               # Age
            int(data['queryResult']['parameters']['SleepDuration']),                     # Sleep Duration
            int(data['queryResult']['parameters']['PhysicalActivityLevel']),             # Physical Activity Level
            int(data['queryResult']['parameters']['StressLevel']),                       # Stress Level
            bmi_mapping[data['queryResult']['parameters']['BMICategory']],               # BMI Category
            int(data['queryResult']['parameters']['DailySteps']),                        # Daily Steps
            sleep_disorder_mapping[data['queryResult']['parameters']['Sleep_Disorder']], # Sleep Disorder
            food_habits_mapping[data['queryResult']['parameters']['Food_Habits']],       # Food Habits
            smoking_status_mapping[data['queryResult']['parameters']['Smoking_Status']]  # Smoking Status
        ])
        
        # Reshape for the model
        features = features.reshape(1, -1)
        
        # Make a prediction
        prediction = model.predict(features)
        
        # Decode the prediction
        disease_name = disease_mapping.get(prediction[0], "Unknown Disease")

        # Store the input data and prediction in the MySQL database
        db = connect_db()
        cursor = db.cursor()

        insert_query = """
        INSERT INTO UserPredictions (Gender, Age, SleepDuration, PhysicalActivityLevel, StressLevel, BMICategory, DailySteps, Sleep_Disorder, Food_Habits, Smoking_Status, Predicted_Disease, Prediction_Time)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        cursor.execute(insert_query, (
            data['queryResult']['parameters']['Gender'],
            data['queryResult']['parameters']['Age'],
            data['queryResult']['parameters']['SleepDuration'],
            data['queryResult']['parameters']['PhysicalActivityLevel'],
            data['queryResult']['parameters']['StressLevel'],
            data['queryResult']['parameters']['BMICategory'],
            data['queryResult']['parameters']['DailySteps'],
            data['queryResult']['parameters']['Sleep_Disorder'],
            data['queryResult']['parameters']['Food_Habits'],
            data['queryResult']['parameters']['Smoking_Status'],
            disease_name,
            datetime.now()
        ))
        
        db.commit()
        cursor.close()
        db.close()

        # Prepare a response for Dialogflow
        response = {
            'fulfillmentText': f'The predicted disease is: {disease_name}'
        }
    
    except KeyError as e:
        response = {
            'fulfillmentText': f"Error: Missing or invalid data for field: {e.args[0]}"
        }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
