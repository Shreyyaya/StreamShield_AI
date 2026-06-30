import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression

X_train = pd.read_csv("data/processed/X_train.csv")
y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()


print("Classes in y_train:", set(y_train))

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

joblib.dump(model, "models/logistic.pkl")

print(" Logistic Regression model saved!")