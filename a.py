import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np
from collections import Counter
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
# from sklearn.
from sklearn.preprocessing import FunctionTransformer

# Load the dataset (replace 'your_dataset.csv' with the actual file path)
data = pd.read_csv('dataset.csv')

# # 1. List all columns and their data types
# print("Columns and their data types:")
# for column in data.columns:
#     print(f"- {column}: {data[column].dtype}")

# 2. Categorize columns as numerical or categorical
numerical_cols = data.select_dtypes(include=['number']).columns.tolist()
categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()

# print("\nNumerical columns:", numerical_cols)
# print("Categorical columns:", categorical_cols)

# 3. Total number of columns and rows
num_cols = len(data.columns)
num_rows = len(data)
print("\nTotal number of columns:", num_cols)
print("Total number of rows:", num_rows)

# 4. Number of empty entries in each column
missing_values = data.isnull().sum()
# print("\nMissing values per column:")
# print(missing_values)

# # 5. Other information about the dataset
# print("\nDescriptive statistics for numerical columns:")
# print(data.describe())  # Provides summary statistics for numerical columns

# # Unique values in categorical columns (if not too many)
# for col in categorical_cols:
#     if len(data[col].unique()) <= 10: # Limit to avoid printing too many values
#         print(f"\nUnique values in '{col}':", data[col].unique())

# # Correlation matrix for numerical columns
# print("\nCorrelation matrix for numerical columns:")
# print(data[numerical_cols].corr())

###############################################################################
# Impute missing values using KNN imputation
numerical_cols = data.select_dtypes(include=['number']).columns.tolist()
categorical_cols = data.select_dtypes(include=['object', 'category', 'bool']).columns.tolist() 

# Scale numerical features (important for KNN)
scaler = StandardScaler()
data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

# Create a KNN imputer object (you can adjust the number of neighbors 'n_neighbors')
imputer = KNNImputer(n_neighbors=5)  # Use 5 nearest neighbors (you can experiment with this value)

# Impute missing values for numerical columns
data[numerical_cols] = imputer.fit_transform(data[numerical_cols])

# Impute missing values for categorical columns (using KNN for categorical features is less common)
for col in categorical_cols:
    if data[col].isnull().any():  # Check if the column has missing values
        # Create a mapping from categories to numerical labels
        mapping = {category: i for i, category in enumerate(data[col].unique())}
        # Convert the column to numerical labels
        data[col] = data[col].map(mapping)
        # Impute missing values using KNN
        data[col] = imputer.fit_transform(data[col].values.reshape(-1, 1))
        # Convert the imputed numerical labels back to categories
        reverse_mapping = {i: category for category, i in mapping.items()}
        data[col] = data[col].map(reverse_mapping)

# Inverse transform numerical features to get the original scale
data[numerical_cols] = scaler.inverse_transform(data[numerical_cols])

# Apply floor function to approach_year, approach_month, and approach_day
data['approach_year'] = data['approach_year'].apply(np.floor).astype(int) 
data['approach_month'] = data['approach_month'].apply(np.floor).astype(int)
data['approach_day'] = data['approach_day'].apply(np.floor).astype(int)

# # Save the imputed dataset (optional)
# data.to_csv('imputed_dataset.csv', index=False)

# print("\nImputation completed. Imputed dataset saved as 'imputed_dataset.csv'.")

# Calculate statistics for "Relative Velocity km per hr"
min_velocity = data['Relative Velocity km per hr'].min()
max_velocity = data['Relative Velocity km per hr'].max()
mean_velocity = data['Relative Velocity km per hr'].mean()
std_velocity = data['Relative Velocity km per hr'].std()  # You can also consider standard deviation

# Define bins based on statistics (example - adjust as needed)
# Define bins based on statistics (example - adjust as needed)
bins = [min_velocity, 
        mean_velocity - std_velocity,  
        mean_velocity, 
        mean_velocity + std_velocity,  
        max_velocity] 
labels = ['Very Slow', 'Slow', 'Medium', 'Fast']  # Removed one label

# Create a new column with binned velocity categories
data['binned_velocity'] = pd.cut(data['Relative Velocity km per hr'], bins=bins, labels=labels, include_lowest=True)

# Impute missing values in "Relative Velocity km per sec" using the binned categories
data['Relative Velocity km per sec'].fillna(data['binned_velocity'], inplace=True)

# Drop the temporary "binned_velocity" column
data.drop('binned_velocity', axis=1, inplace=True)

# # Print the bins and labels being used
# print("Bins for 'Relative Velocity km per hr':")
# for i in range(len(bins) - 1):
#     print(f"- {labels[i]}: [{bins[i]:.2f}, {bins[i+1]:.2f})")

# # Print the "Relative Velocity km per sec" column
# print("\n'Relative Velocity km per sec' column after imputation:")
# print(data['Relative Velocity km per sec'])


# Save the final imputed dataset (optional)
data.to_csv('final_imputed_dataset.csv', index=False)
print("\nFinal imputed dataset saved as 'final_imputed_dataset.csv'.")


# Print the imputed columns for verification
print("\n'Orbital Period':")
print(data['Orbital Period'])
print("\n'Orbit Uncertainity' column after imputation:")
print(data['Orbit Uncertainity'])