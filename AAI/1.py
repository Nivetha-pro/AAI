import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, brier_score_loss
from scipy.stats import entropy  # For KL Divergence
import time

# Load the dementia dataset with the correct file path
file_path = r'C:\Users\Student\Desktop\AAI\datasets\dementia_data-MRI-features.csv'
data = pd.read_csv(file_path)

# Drop unnecessary columns
data = data.drop(columns=['Subject ID', 'MRI ID', 'MR Delay', 'Hand'])

# Fill missing values in SES with mode and in MMSE with mean
data['SES'] = data['SES'].fillna(data['SES'].mode()[0])  # Replace inplace with assignment
data['MMSE'] = data['MMSE'].fillna(data['MMSE'].mean())  # Replace inplace with assignment

# Encode 'M/F' column (gender) into numerical values
label_encoder = LabelEncoder()
data['M/F'] = label_encoder.fit_transform(data['M/F'])  # 'M' -> 0, 'F' -> 1

# Encode target variable 'Group' into numerical labels for consistency
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(data['Group'])

# Define features
X = data.drop(columns=['Group'])

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define Gaussian Process model
model = GaussianProcessClassifier(kernel=C(1.0, (1e-3, 1e3)) * RBF(1, (1e-2, 1e2)), random_state=42)

# Function to evaluate the model with additional metrics
def evaluate_model(model, X_train, y_train, X_test, y_test):
    # Start training time
    start_train_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_train_time

    # Start testing time
    start_test_time = time.time()
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)  # Probabilities for all classes
    test_time = time.time() - start_test_time

    # Calculate essential metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
    logloss = log_loss(y_test, y_pred_proba)

    # Multiclass Brier Score
    brier_scores = []
    for i in range(y_pred_proba.shape[1]):  # For each class
        true_class = (y_test == i).astype(int)  # Binary indicator for class `i`
        prob_class = y_pred_proba[:, i]  # Predicted probability for class `i`
        brier_scores.append(brier_score_loss(true_class, prob_class))
    brier_score = np.mean(brier_scores)  # Average Brier score across all classes

    # KL Divergence (average divergence from predicted to actual)
    kl_divergence = np.mean([entropy((y_test == i).astype(int), y_pred_proba[:, i]) for i in range(y_pred_proba.shape[1])])

    return {
        'Accuracy': accuracy,
        'AUC': auc,
        'Log Loss': logloss,
        'Brier Score': brier_score,
        'KL Divergence': kl_divergence,
        'Train Time (s)': train_time,
        'Test Time (s)': test_time
    }

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Evaluate the Gaussian Process model
model_results = evaluate_model(model, X_train, y_train, X_test, y_test)

# Display the results with four decimal places
print("\nModel Evaluation Results:")
for metric, value in model_results.items():
    print(f"{metric}: {value:.4f}")

# Queries for prediction
queries = [
    {'Age': 88, 'EDUC': 14, 'SES': 2, 'MMSE': 30, 'CDR': 0, 'eTIV': 2004, 'nWBV': 0.681, 'ASF': 0.876, 'M/F': 1, 'Visit': 2},
    {'Age': 80, 'EDUC': 12, 'SES': 2, 'MMSE': 22, 'CDR': 0.5, 'eTIV': 1698, 'nWBV': 0.701, 'ASF': 1.034, 'M/F': 1, 'Visit': 3}
]

# Process each query
for i, query_data in enumerate(queries, start=1):
    # Convert to DataFrame
    query_df = pd.DataFrame([query_data])

    # Ensure the order of the columns matches the training set columns (same order as X)
    query_df = query_df[X.columns]  # Reorder columns to match training data

    # Preprocess query data (same as training data)
    query_df['SES'] = query_df['SES'].fillna(data['SES'].mode()[0])  # Fill missing SES
    query_df['MMSE'] = query_df['MMSE'].fillna(data['MMSE'].mean())  # Fill missing MMSE

    # Standardize the query data using the same scaler as the training data
    query_scaled = scaler.transform(query_df)

    # Predict using the trained model
    prediction = model.predict(query_scaled)
    prediction_proba = model.predict_proba(query_scaled)

    # Display the prediction result with four decimal places
    print(f"\nQuery {i} Prediction Results:")
    print(f"Predicted class: {target_encoder.inverse_transform(prediction)[0]}")
    print("Prediction probabilities:", [f"{prob:.4f}" for prob in prediction_proba[0]])

import matplotlib.pyplot as plt

# Assuming `model_results` dictionary is already obtained from the evaluation function.

# Separate out the metrics and times for plotting
metrics = {k: v for k, v in model_results.items() if 'Time' not in k}
times = {k: v for k, v in model_results.items() if 'Time' in k}

# Plotting the evaluation metrics (excluding time)
plt.figure(figsize=(10, 5))
plt.bar(metrics.keys(), metrics.values(), color='skyblue')
plt.title("Model Evaluation Metrics")
plt.ylabel("Score")
for i, (metric, value) in enumerate(metrics.items()):
    plt.text(i, value + 0.01, f"{value:.4f}", ha='center', va='bottom')
plt.ylim(0, 1.1)  # Set y limit for better readability
plt.show()

# Plotting training and test times
plt.figure(figsize=(6, 4))
plt.bar(times.keys(), times.values(), color='salmon')
plt.title("Training and Test Times")
plt.ylabel("Time (seconds)")
for i, (time, value) in enumerate(times.items()):
    plt.text(i, value + 0.01, f"{value:.4f}", ha='center', va='bottom')
plt.show()
