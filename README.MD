# Asteroid Close Approach Data Analysis and Prediction

## Overview
This project focuses on analyzing a dataset of asteroid close approaches to Earth. The primary goal is to preprocess the data, handle missing values, engineer relevant features, and build a machine learning model to predict whether an asteroid is hazardous to Earth.

## Dataset
The dataset contains information about asteroid close approaches, such as orbital parameters, velocity, approach date, and distances from Earth. Some fields were incomplete, which required imputation and data preprocessing. 

- **Total Rows:** 4534
- **Total Columns:** 24 (including numerical and categorical features)

Key columns include:
- **Numerical Columns:** Asteroid velocity, miss distances, orbital characteristics.
- **Categorical Columns:** Velocity descriptors, orbital period descriptors, hazardous labels.

## Data Preprocessing
### Handling Missing Data
Missing values were imputed using **K-Nearest Neighbors (KNN)** imputation for numerical fields and mode imputation for categorical fields. KNN finds the nearest records with complete data and imputes missing values based on the average of the nearest neighbors. This method was suitable given the nature of the dataset.

### Feature Engineering
- **Binning of Continuous Variables:** Continuous variables like relative velocity were categorized into bins (e.g., `Very Slow`, `Slow`, `Fast`, `Very Fast`) to improve interpretability.
- **Correlation Analysis:** A correlation matrix was generated to identify relationships between the numerical variables.

## Model Building
After data preprocessing, several models were trained on the dataset to predict whether an asteroid is hazardous or not. The final model chosen was an **ensemble method**, which achieved the best performance.

### Key Steps:
1. Data was split into training and testing sets.
2. Categorical variables were one-hot encoded for machine learning models.
3. The final model achieved an accuracy of **85%** on the test set.

## Key Files
- `final_imputed_dataset.csv`: The dataset after KNN imputation.
- `Preprocessed_dataset.csv`: The preprocessed dataset, ready for model training.
- `Astronomy.ipynb`: Jupyter notebook containing the entire workflow, including data cleaning, imputation, feature engineering, and model training.
- `report.pdf`: A detailed report summarizing the project methodology, results, and conclusions.

## Results
- The model achieved an accuracy of **85%** in predicting hazardous asteroids.
- Velocity and miss distances played a critical role in identifying potentially hazardous objects.

## Data Visualization
- **Correlation Heatmap:** Visualizes relationships between key features.
- **Histograms:** Show the distribution of velocities, orbital periods, and miss distances.

## Conclusion
This project successfully demonstrates how to handle missing data using KNN imputation, perform feature engineering, and apply machine learning techniques to classify asteroids. The final model shows promising performance in identifying hazardous asteroids, which can help in early detection and monitoring.

## Future Work
- Explore alternative imputation methods, such as MICE or deep learning-based approaches.
- Implement additional feature engineering techniques to enhance prediction accuracy.
- Experiment with more advanced models, such as gradient boosting or neural networks, to further improve classification performance.

