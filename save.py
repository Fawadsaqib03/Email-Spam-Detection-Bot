import os, joblib
from load_data import load_data
from clean import clean_data
from vectorize import vectorize
from train import train_model
from evaluate import evaluate

os.makedirs("artifacts", exist_ok=True)

df = load_data()
df = clean_data(df)
X_train_v, X_test_v, y_train, y_test, vectorizer = vectorize(df)
model = train_model(X_train_v, y_train)
evaluate(model, X_test_v, y_test)

joblib.dump(model,      "artifacts/spam_model.pkl")
joblib.dump(vectorizer, "artifacts/vectorizer.pkl")
print("\nSaved → artifacts/spam_model.pkl")
print("Saved → artifacts/vectorizer.pkl")