# Credit Card Fraud Detection

This project uses a machine learning model to detect fraudulent credit card transactions. It is based on a real-world dataset containing anonymized features and a binary fraud label.

## Dataset

This project uses the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) available on Kaggle.

Once downloaded, place the `creditcard.csv` file in the root folder of this project (same folder as the Python files).

> **Note** The dataset is large (~150 MB) and is not included in this repository to keep the repo lightweight.

## How It Works

1. Loads the dataset using Pandas
2. Splits the data into training and testing sets
3. Trains a Random Forest Classifier
4. Evaluates the model using accuracy, confusion matrix, and classification report
5. Visualizes the confusion matrix with seaborn

## Libraries Used

- pandas
- matplotlib
- seaborn
- scikit-learn

Install them (if not already installed) using:

```bash
pip install pandas matplotlib seaborn scikit-learn
