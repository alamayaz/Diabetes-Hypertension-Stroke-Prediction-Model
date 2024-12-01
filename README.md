# Diabetes, Hypertension, and Stroke Prediction Model

This repository contains a comprehensive machine learning pipeline to predict the likelihood of diabetes, hypertension, and stroke using structured data. The models explore various machine learning algorithms and evaluate their performance using multiple metrics.

## Features
- **Data Processing:** Includes preprocessing steps such as scaling, dimensionality reduction, and train-test splitting.
- **Modeling:** Implements various machine learning models:
  - Logistic Regression
  - Random Forest
  - XGBoost
  - K-Nearest Neighbors (KNN)
  - Linear Regression (as a baseline comparison)
- **Evaluation Metrics:** Uses accuracy, classification reports, and confusion matrices to evaluate model performance.
- **Visualization:** Generates detailed plots, including confusion matrices and bar charts comparing model accuracies.

---

## Repository Contents

### 1. **Dataset**
The notebook assumes a dataset (`merged_data.csv`) with labeled features for diabetes, hypertension, and stroke. Ensure your dataset is preprocessed to align with the notebook requirements.

### 2. **Notebook**
The primary code for data processing, model training, evaluation, and visualization is implemented in `FinalCode.ipynb`.

### 3. **Modeling Steps**
- **Data Loading:** Loads the dataset into a pandas DataFrame.
- **Preprocessing:** Applies scaling and dimensionality reduction using `StandardScaler` and PCA (optional).
- **Model Training:** Trains multiple machine learning models on the dataset.
- **Evaluation:** Compares models using classification reports, confusion matrices, and accuracy scores.

### 4. **Visualization**
- Confusion matrices for individual target columns.
- Bar charts for model comparison based on accuracy scores.

---

## Getting Started

### Prerequisites
Install the required Python libraries:
```bash
pip install pandas scikit-learn xgboost matplotlib seaborn
```

### Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/repo-name.git
   ```
2. Place your dataset (`merged_data.csv`) in the repository's root directory.
3. Open and run `FinalCode.ipynb` using Jupyter Notebook or JupyterLab.

---

## Results

### Models Evaluated
- Logistic Regression
- Random Forest
- XGBoost
- K-Nearest Neighbors (KNN)
- Linear Regression

### Performance Metrics
- Accuracy scores
- Classification reports
- Confusion matrices

### Visualization Examples
- Accuracy comparison bar charts.
- Confusion matrices for each target.

---

## Contributing
Contributions are welcome! Please create an issue to discuss any enhancements or fixes.

---

Feel free to update this document to include specific results or additional context about the project. Let me know if you'd like to further customize any section!
