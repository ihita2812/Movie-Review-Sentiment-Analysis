import pandas as pd

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        print(f"Data loaded successfully. Shape: {data.shape}")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def explore_data(data):
    print("First 5 rows of the dataset:")
    print(data.head())
    
    print("Dataset columns:")
    print(data.columns)
    
    print("Class distribution (Sentiments):")
    print(data['sentiment'].value_counts())  # Assuming 'sentiment' column

if __name__ == "__main__":
    data_path = 'data/raw/reviews.csv'  # Path to your dataset
    data = load_data(data_path)
    
    if data is not None:
        explore_data(data)
