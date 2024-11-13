import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.metrics import classification_report

# Load the dataset
file_path = r'C:\Users\Student\Desktop\AAI\Gaussian process\parkinsons_data-VOICE-features.csv'

# Check if the file exists and load data
try:
    data = pd.read_csv(file_path)
    print("Data loaded successfully!")
except FileNotFoundError:
    print(f"Error: The file at {file_path} was not found.")
    exit()  # Exit if the file is not found

# Drop unnecessary columns
if 'name' in data.columns:
    data = data.drop(columns=['name'])
else:
    print("Column 'name' not found in the data. Proceeding without dropping it.")

# Define features and target variable
X = data.drop(columns=['status'])
y = data['status']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define Gaussian Process kernel
kernel = C(1.0, (1e-3, 1e3)) * RBF(1, (1e-2, 1e2))

# Initialize the Gaussian Process Classifier
gpc = GaussianProcessClassifier(kernel=kernel, random_state=42)

# 5-Fold Cross-validation with StratifiedKFold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation
cv_results = cross_validate(gpc, X_scaled, y, cv=cv, scoring='accuracy', return_train_score=False)

# Output the results of cross-validation
print(f"Cross-validation Accuracy: {np.mean(cv_results['test_score']):.2f}")
print(f"Cross-validation Standard Deviation: {np.std(cv_results['test_score']):.2f}")

# After cross-validation, let's train on the entire dataset
gpc.fit(X_scaled, y)

# Classification report on full dataset
y_pred = gpc.predict(X_scaled)
print("Classification Report on full data (after training on entire dataset):")
print(classification_report(y, y_pred))

# Now, the query part:

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

# Function to process query data
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
probabilities_1 = gpc.predict_proba(query_scaled_1)
prob_status_0 = probabilities_1[0][0]  # Probability of status = 0 (no Parkinson's)
print("P(status=0 | given features for query 1) =", prob_status_0)

# Process query 2 and predict
query_scaled_2 = process_query(query_data_2)
probabilities_2 = gpc.predict_proba(query_scaled_2)
prob_status_1 = probabilities_2[0][1]  # Probability of status = 1 (Parkinson's)
print("P(status=1 | given features for query 2) =", prob_status_1)
