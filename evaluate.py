from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from load_data import load_data
from clean import clean_data
from vectorize import vectorize
from train import train_model

def evaluate(model, X_test_v, y_test):
    y_pred = model.predict(X_test_v)
    acc = accuracy_score(y_test, y_pred)

    print("=" * 40)
    print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print("=" * 40)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    return acc

if __name__ == "__main__":
    df = load_data()
    df = clean_data(df)
    X_train_v, X_test_v, y_train, y_test, vectorizer = vectorize(df)
    model = train_model(X_train_v, y_train)
    evaluate(model, X_test_v, y_test)