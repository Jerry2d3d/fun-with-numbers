import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

class NumberPattern:
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.data = self.read_data()

    def read_data(self):
        df = pd.read_csv(self.csv_file)
        return df.to_numpy()

    def write_data(self):
        df = pd.DataFrame(self.data, columns=[f"num{i+1}" for i in range(self.data.shape[1])])
        df.to_csv(self.csv_file, index=False)

    def add_data(self, new_data):
        new_data_parsed = np.array([list(map(int, d.split(','))) for d in new_data])
        self.data = np.vstack([self.data, new_data_parsed])
        self.write_data()
        print(f"Data updated: {self.data}")

    def find_pattern(self):
        models = []
        for i in range(self.data.shape[1]):
            X = np.arange(len(self.data)).reshape(-1, 1)
            y = self.data[:, i]
            model = LinearRegression()
            model.fit(X, y)
            models.append(model)
        return models

    def predict_next(self, n=1):
        models = self.find_pattern()
        X_new = np.arange(len(self.data), len(self.data) + n).reshape(-1, 1)
        y_new = np.zeros((n, self.data.shape[1]))
        for i, model in enumerate(models):
            y_new[:, i] = model.predict(X_new)
        
        # Constrain predictions to the required ranges
        y_new[:, :-1] = np.clip(y_new[:, :-1], 1, 69)  # First five numbers between 1 and 69
        y_new[:, -1] = np.clip(y_new[:, -1], 1, 26)   # Last number between 1 and 26
        
        return np.rint(y_new).astype(int)

# Example usage
if __name__ == "__main__":
    # Create a CSV file with initial data if it doesn't exist
    initial_csv = 'initial_data.csv'
    try:
        df = pd.read_csv(initial_csv)
    except FileNotFoundError:
        initial_data = ["1,2,3,4,5,6", "7,8,9,10,11,12", "13,14,15,16,17,18"]
        df = pd.DataFrame([list(map(int, d.split(','))) for d in initial_data], columns=[f"num{i+1}" for i in range(6)])
        df.to_csv(initial_csv, index=False)

    number_pattern = NumberPattern(initial_csv)
    number_pattern.add_data(["19,20,21,22,23,24", "25,26,27,28,29,30"])
    next_numbers = number_pattern.predict_next(3)
    next_numbers_str = [','.join(map(str, nums)) for nums in next_numbers]
    print(f"Next numbers: {next_numbers_str}")
