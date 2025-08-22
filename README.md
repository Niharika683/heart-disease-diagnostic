🚀 Features

Frontend (Tailwind CSS + Vanilla JS)

Clean, responsive UI with sliders & dropdowns for user inputs (age, BP, cholesterol, glucose, smoking, alcohol, activity, etc.).

Dynamic diagnosis report with prediction, confidence score, key risk factors, and recommendations.

Visualization of personal risk factors vs healthy baseline (auto-generated bar chart).

Backend (Flask + XGBoost)

Model trained on the Cardio dataset (cardio_train.csv).

Preprocessing with scaling for numeric features.

Uses XGBoost Classifier for prediction.

Provides REST API (/predict) returning JSON with prediction, risk factors, and recommendations.

Model Performance

Achieved strong classification performance.

Confusion Matrix results:

True Negatives (No Disease predicted correctly): 5413

False Positives (Predicted Heart Disease but actually no): 1575

False Negatives (Missed cases of Heart Disease): 2109

True Positives (Heart Disease predicted correctly): 4903

⚙️ Tech Stack

Frontend: HTML, Tailwind CSS, JavaScript

Backend: Python (Flask), scikit-learn, XGBoost, matplotlib

Dataset: Cardio Dataset

Deployment Ready: API + static frontend
