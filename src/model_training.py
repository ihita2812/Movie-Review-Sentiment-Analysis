from sklearn.linear_model import LogisticRegression
from src.config import MODEL_DIR
import joblib
import os

def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, os.path.join(MODEL_DIR, 'logistic_regression.pkl'))
    return model