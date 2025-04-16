import pandas as pd
import sys
import os

file_path = 'creditcard.csv'

if not os.path.exists(file_path):
    print(f"'{file_path}' not found.")
    print("Please download the dataset from:")
    print("https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
    print("and place the 'creditcard.csv' file in the project folder.")
    sys.exit()

df = pd.read_csv('creditcard.csv')