import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from src.config import DATA_DIR, VECTORIZE_FEATURES, MODEL_DIR
from pathlib import Path

def load_data():
    data = {'review': [], 'sentiment': []}
    for sentiment in ['pos', 'neg']:
        folder = Path(DATA_DIR) / 'train' / sentiment
        if not folder.exists():
            raise FileNotFoundError(f"Directory not found: {folder}")
        for file_name in os.listdir(folder):
            file_path = folder / file_name
            with open(file_path, 'r', encoding='utf-8') as f:
                data['review'].append(f.read())
                data['sentiment'].append(1 if sentiment == 'pos' else 0)
    return pd.DataFrame(data)


def preprocess_data(df):
    vectorizer = TfidfVectorizer(max_features=VECTORIZE_FEATURES)
    X = vectorizer.fit_transform(df['review']).toarray()
    y = df['sentiment']
    
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl'))
    return X, y

def split_data(X, y, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test
