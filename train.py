from sklearn.naive_bayes import MultinomialNB
from load_data import load_data
from clean import clean_data
from vectorize import vectorize

def train_model(X_train_v, y_train):
    model = MultinomialNB()
    model.fit(X_train_v, y_train)
    print("Model trained successfully!")
    return model

if __name__ == "__main__":
    df = load_data()
    df = clean_data(df)
    X_train_v, X_test_v, y_train, y_test, vectorizer = vectorize(df)
    model = train_model(X_train_v, y_train)
    print("Training complete!")