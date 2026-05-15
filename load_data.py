import pandas as pd

def load_data():
    # Load SMS Spam dataset from URL
    url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
    df = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])
    print(f"Dataset loaded: {df.shape[0]} rows")
    print(df['label'].value_counts())
    return df

if __name__ == "__main__":
    df = load_data()
    print(df.head())