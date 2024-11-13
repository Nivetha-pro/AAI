import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.metrics import classification_report, roc_auc_score, brier_score_loss, log_loss
from sklearn.metrics import accuracy_score
from scipy.stats import entropy
import time

# Load the dataset
file_path = r'C:\Users\Student\Desktop\AAI\datasets\parkinsons_data-VOICE-features.csv'
data = pd.read_csv(file_path)

# Drop unnecessary columns
if 'name' in data.columns:
    data = data.drop(columns=['name'])

# Define features and target variable
X = data.drop(columns=['status'])
y = data['status']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define Gaussian Process model
model = GaussianProcessClassifier(kernel=C(1.0, (1e-3, 1e3)) * RBF(1, (1e-2, 1e2)), random_state=42)

# Function to evaluate the model
def evaluate_model(model, X_train, y_train, X_test, y_test):
    # Start training time
    start_train_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_train_time

    # Start testing time
    start_test_time = time.time()
    y_pred = model.predict(X_test)
    test_time = time.time() - start_test_time

    # Probabilities for AUC and Brier Score (in case of classification)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of the positive class

    # Calculate various metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    brier = brier_score_loss(y_test, y_pred_proba)
    logloss = log_loss(y_test, y_pred_proba)

    # Kullback-Leibler Divergence (KL-Divergence)
    try:
        gp_proba = model.predict_proba(X_test)[:, 1]
        kl_divergence = entropy(gp_proba, y_pred_proba)
    except Exception as e:
        kl_divergence = None
        print(f"Error calculating KL Divergence: {e}")

    return {
        'accuracy': accuracy,
        'AUC': auc,
        'Brier Score': brier,
        'Log Loss': logloss,
        'KL Divergence': kl_divergence,
        'Train Time (s)': train_time,
        'Test Time (s)': test_time
    }

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Evaluate the Gaussian Process model
model_results = evaluate_model(model, X_train, y_train, X_test, y_test)

# Display the results
print("\nModel Evaluation Results:")
for metric, value in model_results.items():
    print(f"{metric}: {value:.4f}")

# Now, add the query prediction part:

# Query 1: P(status=0 | given features)
query_data_1 = {
    'MDVP:Fo(Hz)': 197.076,
    'MDVP:Fhi(Hz)': 206.896,
    'MDVP:Flo(Hz)': 192.055,
    'MDVP:Jitter(%)': 0.00289,
    'MDVP:Jitter(Abs)': 0.00001,
    'MDVP:RAP': 0.00166,
    'MDVP:PPQ': 0.00168,
    'Jitter:DDP': 0.00498,
    'MDVP:Shimmer': 0.01098,
    'MDVP:Shimmer(dB)': 0.097,
    'Shimmer:APQ3': 0.00563,
    'Shimmer:APQ5': 0.0068,
    'MDVP:APQ': 0.00802,
    'Shimmer:DDA': 0.01689,
    'NHR': 0.00339,
    'HNR': 26.775
}

# Query 2: P(status=1 | given features)
query_data_2 = {
    'MDVP:Fo(Hz)': 162.568,
    'MDVP:Fhi(Hz)': 198.346,
    'MDVP:Flo(Hz)': 77.63,
    'MDVP:Jitter(%)': 0.00502,
    'MDVP:Jitter(Abs)': 0.00003,
    'MDVP:RAP': 0.0028,
    'MDVP:PPQ': 0.00253,
    'Jitter:DDP': 0.00841,
    'MDVP:Shimmer': 0.01791,
    'MDVP:Shimmer(dB)': 0.168,
    'Shimmer:APQ3': 0.00793,
    'Shimmer:APQ5': 0.01057,
    'MDVP:APQ': 0.01799,
    'Shimmer:DDA': 0.0238,
    'NHR': 0.0117,
    'HNR': 25.678
}

# Process query data and predict probabilities
def process_query(query_data):
    query_df = pd.DataFrame([query_data])

    # Align query columns with the model's feature columns
    missing_cols = set(X.columns) - set(query_df.columns)

    # Add missing columns with default value (0 or NaN)
    for col in missing_cols:
        query_df[col] = 0  # You can choose to use 0 or np.nan for missing features

    # Reorder columns to match the training data feature order
    query_df = query_df[X.columns]

    # Scale the query data using the scaler fitted on the training data
    query_scaled = scaler.transform(query_df)
    return query_scaled

# Process query 1 and predict
query_scaled_1 = process_query(query_data_1)
probabilities_1 = model.predict_proba(query_scaled_1)
prob_status_0 = probabilities_1[0][0]  # Probability of status = 0 (no Parkinson's)
print("P(status=0 | given features for query 1) =", prob_status_0)

# Process query 2 and predict
query_scaled_2 = process_query(query_data_2)
probabilities_2 = model.predict_proba(query_scaled_2)
prob_status_1 = probabilities_2[0][1]  # Probability of status = 1 (Parkinson's)
print("P(status=1 | given features for query 2) =", prob_status_1)
