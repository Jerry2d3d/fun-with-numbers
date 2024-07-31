import numpy as np
import pandas as pd

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

    def calculate_frequencies(self):
        frequencies = []
        for i in range(self.data.shape[1]):
            counts = np.bincount(self.data[:, i])
            frequencies.append(counts / np.sum(counts))
        return frequencies

    def predict_next_by_frequency(self):
        frequencies = self.calculate_frequencies()
        next_numbers = []
        for freq in frequencies:
            next_numbers.append(np.argmax(freq))
        return next_numbers

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
    next_numbers = number_pattern.predict_next_by_frequency()
    next_numbers_str = ','.join(map(str, next_numbers))
    print(f"Next numbers by frequency: {next_numbers_str}")
