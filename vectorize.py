from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from load_data import load_data
from clean import clean_data

def vectorize(df):
    X = df['clean_message']
    y = df['label_num']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train_v = vectorizer.fit_transform(X_train)
    X_test_v  = vectorizer.transform(X_test)

    print(f"Train size: {X_train_v.shape}, Test size: {X_test_v.shape}")
    return X_train_v, X_test_v, y_train, y_test, vectorizer

if __name__ == "__main__":
    df = load_data()
    df = clean_data(df)
    vectorize(df)