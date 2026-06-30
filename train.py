import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


def train_model():
    print("Loading data...")

    X_train = pd.read_csv("data/processed/X_train.csv")
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()
    y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()

    print("Training model...")

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        random_state=42
    )

    model.fit(X_train, y_train)

    print("Evaluating model...")

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)

    print(f"\n Accuracy: {acc:.4f}\n")
    print("Classification Report:\n")
    print(classification_report(y_test, preds))

    # Save model
    joblib.dump(model, "models/random_forest.pkl")

    print("Model saved at models/random_forest.pkl")

    #confusion matrix
    cm = confusion_matrix(y_test, preds)

    plt.figure()
    plt.figure(figsize=(18, 14))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.savefig("confusion_matrix.png")
    plt.close()

    print("Confusion matrix saved")


if __name__ == "__main__":
    train_model()