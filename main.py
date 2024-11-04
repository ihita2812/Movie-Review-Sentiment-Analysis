from src.data_preparation import load_data, preprocess_data, split_data
from src.model_training import train_model
from src.model_evaluation import evaluate_model

df = load_data()
X, y = preprocess_data(df)

X_train, X_test, y_train, y_test = split_data(X, y)

train_model(X_train, y_train)

evaluate_model(X_test, y_test)