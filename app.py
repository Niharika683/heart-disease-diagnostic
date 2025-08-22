
# app.py

# Import necessary libraries for the Flask application and machine learning
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.exceptions import BadRequest
import os
import logging
# Import CORS specifically for more granular control
from flask_cors import CORS

# IMPORTS FOR PLOTTING: Ensure these are present
import matplotlib.pyplot as plt
import io
import base64

# Initialize Flask application
app = Flask(__name__)

# Enable CORS for the /predict endpoint, explicitly allowing all origins.
# This ensures that preflight OPTIONS requests are handled correctly.
CORS(app, resources={r"/predict": {"origins": "*"}})

# Configure logging for the Flask app
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Machine Learning Model Loading and Training (executed once on app startup) ---

DATA_FILE = 'cardio_train.csv'

model = None
scaler = None
X_train_columns = None
model_loaded_successfully = False
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def load_and_train_model():
    """
    Loads the dataset, preprocesses it, and trains the XGBClassifier model
    and StandardScaler. This function is called once when the Flask app starts.
    """
   
    global model, scaler, X_train_columns, model_loaded_successfully

    app.logger.info("Loading and training model...")
    try:
        data = pd.read_csv(DATA_FILE, delimiter=';')
        app.logger.info(f"Dataset '{DATA_FILE}' loaded successfully. Shape: {data.shape}")

        data['age'] = (data['age'] / 365).astype(int)

        X = data.drop(columns=['id', 'cardio'])
        y = data['cardio']

        X_train_columns = X.columns.tolist()

        # Standardize numerical features
        scaler = StandardScaler()
        X[['height', 'weight', 'ap_hi', 'ap_lo']] = scaler.fit_transform(X[['height', 'weight', 'ap_hi', 'ap_lo']])
        app.logger.info("Data preprocessing and standardization complete.")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        app.logger.info(f"Data split into training ({X_train.shape}) and testing ({X_test.shape}) sets.")

        # Initialize and train XGBoost Classifier
        # Removed 'use_label_encoder=False' as it's no longer needed and causes a warning
        model = XGBClassifier(eval_metric='logloss', random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        app.logger.info("XGBoost Classifier model trained successfully.")

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        app.logger.info(f"Model Accuracy on test set: {accuracy:.4f}")

# ✅ Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        app.logger.info(f"Confusion Matrix:\n{cm}")
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Disease", "Heart Disease"])
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix - Heart Disease Prediction")
        plt.savefig("confusion_matrix.png")
        plt.close()
        model_loaded_successfully = True

    except FileNotFoundError:
        app.logger.critical(f"Error: Dataset file '{DATA_FILE}' not found. Please ensure it is in the same directory as app.py. Exiting.")
        model_loaded_successfully = False
    except Exception as e:
        app.logger.critical(f"An unexpected error occurred during model loading or training: {e}", exc_info=True)
        model_loaded_successfully = False


# Call the function to load and train the model when the app starts
with app.app_context():
    load_and_train_model()


# --- Flask Routes ---

@app.route('/')
def index():
    app.logger.info("Serving index.html")
    # This route is mainly for serving the HTML if Flask were to serve the entire app.
    # In a typical local development, you might open index.html directly,
    # but having this route is good practice for completeness.
    return send_from_directory('templates', 'index.html')

@app.route('/predict', methods=['POST'])
def predict_heart_disease_api():
    # Check if the model and scaler are loaded before proceeding
    if not model_loaded_successfully or model is None or scaler is None or X_train_columns is None:
        app.logger.error("Prediction requested but model/scaler/columns are not loaded. Service unavailable.")
        return jsonify({"error": "Prediction service not ready. Model failed to load during startup."}), 503

    try:
        # Get JSON data from the request body
        data = request.get_json(force=False, silent=False)

        # Validate that JSON data was received
        if data is None:
            app.logger.error("Received request with missing or invalid JSON payload.")
            return jsonify({"error": "Invalid request: Expected JSON payload."}), 400

        app.logger.info(f"Received prediction request with data: {data}")

        # Extract individual features from the JSON payload
        try:
            age = data['age']
            gender = data['gender']
            height = data['height']
            weight = data['weight']
            ap_hi = data['ap_hi']
            ap_lo = data['ap_lo']
            cholesterol = data['cholesterol']
            gluc = data['gluc']
            smoke = data['smoke']
            alco = data['alco']
            active = data['active']
        except KeyError as e:
            app.logger.error(f"Missing key in JSON payload: {e}")
            return jsonify({"error": f"Missing required data field: {e}. Please ensure all fields are sent."}), 400

        # Create a DataFrame for the input, ensuring column order matches training data
        # This is crucial for the model to interpret the features correctly.
        input_df = pd.DataFrame([[age, gender, height, weight, ap_hi, ap_lo,
                                  cholesterol, gluc, smoke, alco, active]],
                                columns=X_train_columns)

        # Standardize numerical features using the *trained* scaler
        # The scaler must be the same one used during model training.
        input_df[['height', 'weight', 'ap_hi', 'ap_lo']] = scaler.transform(
            input_df[['height', 'weight', 'ap_hi', 'ap_lo']])

        # Make prediction using the trained model
        prediction_value = model.predict(input_df)[0]
        # Get prediction probabilities for confidence score
        # proba will be [probability_of_class_0, probability_of_class_1] for binary classification
        proba = model.predict_proba(input_df)[0]
        confidence = round(max(proba) * 100, 1) # Get the highest probability as confidence

        # Determine diagnosis label based on the prediction value
        prediction_label = 'High Risk' if prediction_value == 1 else 'Low Risk'

        # Initialize list to store identified risk factors for display
        risk_factors_present = []
        
        # Calculate BMI (Body Mass Index)
        bmi = weight / ((height / 100) * (height / 100)) # Height is in cm, convert to meters

        # Prepare data for graph comparison (0 = healthy/absent, 1 = unhealthy/present)
        # Define the set of risk factors to be visualized consistently
        risk_factor_labels = [
            'High BP', 'Elevated Cholesterol', 'High Glucose',
            'Smoking', 'Alcohol Use', 'Physical Inactivity', 'High BMI'
        ]

        # Initialize lists for 'Healthy Condition' (all zeros for risk factors)
        healthy_condition_values = [0] * len(risk_factor_labels)
        # Initialize list to store the 'Your Condition' values (binary: 0 or 1)
        user_condition_values = []

        # Populate user's condition and risk_factors_present based on input
        # High BP
        if (ap_hi > 140 or ap_lo > 90):
            user_condition_values.append(1)
            risk_factors_present.append('High Blood Pressure')
        else:
            user_condition_values.append(0)
        
        # Elevated Cholesterol (cholesterol > 1 means elevated)
        if cholesterol > 1:
            user_condition_values.append(1)
            risk_factors_present.append('Elevated Cholesterol')
        else:
            user_condition_values.append(0)
        
        # High Glucose (gluc > 1 means elevated)
        if gluc > 1:
            user_condition_values.append(1)
            risk_factors_present.append('High Glucose')
        else:
            user_condition_values.append(0)
        
        # Smoking (smoke == 1 means yes)
        if smoke == 1:
            user_condition_values.append(1)
            risk_factors_present.append('Smoking')
        else:
            user_condition_values.append(0)
        
        # Alcohol Use (alco == 1 means yes)
        if alco == 1:
            user_condition_values.append(1)
            risk_factors_present.append('Alcohol Use')
        else:
            user_condition_values.append(0)
        
        # Physical Inactivity (active == 0 means inactive)
        if active == 0:
            user_condition_values.append(1)
            risk_factors_present.append('Physical Inactivity')
        else:
            user_condition_values.append(0)
        
        # High BMI (bmi >= 25 is considered overweight/obese)
        if bmi >= 25:
            user_condition_values.append(1)
            risk_factors_present.append('High BMI / Overweight')
        else:
            user_condition_values.append(0)

        # Generate the comparison bar chart using Matplotlib
        plot_url = None
        
        # Create a figure and an axes for the plot
        fig, ax = plt.subplots(figsize=(10, 5)) # Set figure size for better readability
        
        # Set background color of the plot area and the figure itself
        ax.set_facecolor('#f9f9f9')
        fig.patch.set_facecolor('#f9f9f9')

        bar_width = 0.35 # Width of each bar
        index = np.arange(len(risk_factor_labels)) # X-axis locations for groups of bars

        # Plot bars for the 'Healthy Condition'
        healthy_bars = ax.bar(index, healthy_condition_values, bar_width, label='Healthy Condition', color='#3b82f6', alpha=0.8) # Blue color
        
        # Plot bars for 'Your Condition' (user's input)
        user_bars = ax.bar(index + bar_width, user_condition_values, bar_width, label='Your Condition', color='#ef4444', alpha=0.8) # Red color

        # Set y-axis limits to clearly show binary (0 or 1) status
        ax.set_ylim(-0.2, 1.2)
        ax.set_yticks([0, 1]) # Only show ticks at 0 and 1
        ax.set_yticklabels(['Absent', 'Present']) # Label ticks as Absent/Present

        # Add title and labels for axes
        ax.set_title('Comparison of Risk Factors: Healthy vs. Your Condition', fontsize=14, color='#374151')
        ax.set_ylabel('Status', fontsize=12, color='#4b5563')
        # Set x-axis tick locations and labels (rotated for readability)
        ax.set_xticks(index + bar_width / 2)
        ax.set_xticklabels(risk_factor_labels, rotation=45, ha='right', fontsize=10) # Removed labelcolor here

        # --- NEW: Set x-tick label color directly by iterating ---
        for label in ax.get_xticklabels():
            label.set_color('#4b5563')

        # Customize y-axis tick parameters
        ax.tick_params(axis='y', labelsize=10) # Removed labelcolor here

        # --- NEW: Set y-tick label color directly by iterating ---
        for label in ax.get_yticklabels():
            label.set_color('#4b5563')
        
        # Add a legend to differentiate the two sets of bars
        ax.legend()
        plt.tight_layout() # Adjust layout to prevent labels from overlapping

        # Save the plot to a BytesIO object in PNG format
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight') # bbox_inches='tight' removes extra whitespace
        buf.seek(0) # Rewind the buffer to the beginning
        # Encode the image data to base64 for embedding in HTML
        plot_url = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close(fig) # Close the plot to free up memory resources

        # Define recommendations based on prediction
        recommendations_text = ''
        if prediction_value == 1: # High Risk
            recommendations_text = (
                "• Consult a cardiologist immediately\n"
                "• Monitor blood pressure daily\n"
                "• Begin aggressive lifestyle modifications (diet, exercise)\n"
                "• Consider medical evaluation for medication."
            )
        else: # Low Risk
            recommendations_text = (
                "• Regular cardiovascular checkups are advised\n"
                "• Maintain a healthy BMI and balanced diet\n"
                "• Continue preventive measures and active lifestyle."
            )

        # Return the results as JSON, including the base64 encoded plot
        response = {
            "prediction": prediction_label,
            "confidence": f"{confidence}%",
            "risk_factors": risk_factors_present if risk_factors_present else ["No significant risk factors identified based on current inputs."],
            "recommendations": recommendations_text,
            "risk_factors_graph": plot_url # The image data for the graph
        }
        app.logger.info(f"Prediction successful. Sending response: {response}")
        return jsonify(response)

    except BadRequest as e:
        app.logger.error(f"Bad JSON request received: {e.description}", exc_info=True)
        return jsonify({"error": f"Invalid JSON payload: {e.description}"}), 400
    except Exception as e:
        app.logger.error(f"An unexpected error occurred during prediction API call: {e}", exc_info=True)
        return jsonify({"error": f"An internal server error occurred during prediction: {str(e)}"}), 500

# Run the Flask app
def main():
    app.run(debug=False, port=5000)

if __name__ == '__main__':
    main()
    # Before running the app, ensure the data file exists and model loaded successfully
    if not os.path.exists(DATA_FILE):
        app.logger.critical(f"Error: '{DATA_FILE}' not found. Please ensure it is in the same directory as 'app.py'.")
        app.logger.critical("The Flask application cannot start effectively without the dataset.")
        exit(1) # Exit if the data file is missing
    if not model_loaded_successfully:
        app.logger.critical("Application will not start because the model/data failed to load during initialization.")
        exit(1) # Exit if the model failed to load

    app.run(debug=True, host='0.0.0.0', port=5000)
import webbrowser
import threading
import time

def open_browser():
    time.sleep(1)
    webbrowser.open("http://127.0.0.1:5000")

if __name__ == '__main__':
    threading.Thread(target=open_browser).start()
    main()

