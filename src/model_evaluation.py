from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os
from src.config import MODEL_DIR, RESULTS_DIR

def evaluate_model(X_test, y_test):
    model = joblib.load(os.path.join(MODEL_DIR, 'logistic_regression.pkl'))
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Negative', 'Positive'])
    cm = confusion_matrix(y_test, y_pred)
    
    print("Accuracy:", accuracy)
    print("\nClassification Report:\n", report)
    print("\nConfusion Matrix:\n", cm)
    
    with open(os.path.join(RESULTS_DIR, 'evaluation_report.txt'), 'w') as f:
        f.write(f"Accuracy: {accuracy}\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(str(cm))
