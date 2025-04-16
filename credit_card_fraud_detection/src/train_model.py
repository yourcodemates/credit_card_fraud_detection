import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from credit_card_fraud_detection.src.train_model import X_test
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from pyexpat import model

file_path = 'creditcard.csv'
df = pd.read_csv(file_path)


X = df.drop('Class', axis=1)
y = df['Class']

y_pred = model.predict(X_test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
