import numpy as np
import pandas as pd

class NumberPattern:
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.data = self.clean_and_read_data()
        self.sort_data_by_date()

    def clean_and_read_data(self):
        # Read CSV file
        df = pd.read_csv(self.csv_file)
        
        # Ensure the date is in the correct format
        df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y')
        
        # Ensure number columns are integers and fix any formatting issues
        for i in range(1, 8):
            col = f'num{i}'
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        
        # For the seventh number, ensure it's within the valid set {2, 3, 4, 5, 10}
        df['num7'] = df['num7'].apply(lambda x: x if x in {2, 3, 4, 5, 10} else 2)  # Default to 2 if invalid
        
        # Sort by date
        df = df.sort_values(by='date')
        
        # Return as numpy array
        return df.to_numpy()

    def write_data(self):
        df = pd.DataFrame(self.data, columns=["date"] + [f"num{i+1}" for i in range(7)])
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%m/%d/%Y')
        df.to_csv(self.csv_file, index=False)

    def add_data(self, new_data):
        new_data_parsed = np.array([[pd.to_datetime(d.split(',')[0], format='%m/%d/%Y')] + list(map(int, d.split(',')[1:])) for d in new_data])
        self.data = np.vstack([self.data, new_data_parsed])
        self.sort_data_by_date()
        self.write_data()
        print(f"Data updated:\n{self.data}")

    def sort_data_by_date(self):
        self.data = self.data[self.data[:, 0].argsort()]

    def calculate_frequencies(self):
        frequencies = []
        for i in range(1, 6):  # Skip the date column
            counts = np.bincount(self.data[:, i].astype(int), minlength=70)
            frequencies.append(counts / np.sum(counts))
        # For the sixth number
        counts = np.bincount(self.data[:, 6].astype(int), minlength=27)
        frequencies.append(counts / np.sum(counts))
        # For the seventh number (2, 3, 4, 5, 10)
        counts = np.bincount(self.data[:, 7].astype(int), minlength=11)
        specific_counts = [counts[2], counts[3], counts[4], counts[5], counts[10]]
        frequencies.append(np.array(specific_counts) / np.sum(specific_counts))
        return frequencies

    def predict_next_by_frequency(self):
        frequencies = self.calculate_frequencies()
        top_five_numbers = []
        next_numbers = []
        for i in range(5):
            top_five = np.argsort(frequencies[i])[::-1][:5]
            top_five_numbers.append(top_five)
            next_numbers.append(np.argmax(frequencies[i]))
        top_sixth_number = np.argmax(frequencies[5])
        next_numbers.append(top_sixth_number)
        top_seventh_number = np.argmax(frequencies[6]) + 2  # Adjust index for specific counts [2, 3, 4, 5, 10]
        if top_seventh_number == 6:  # If it points to index 4 which is 5 in specific counts
            top_seventh_number = 10
        return top_five_numbers, top_sixth_number, top_seventh_number, next_numbers

    def calculate_percentage_first_five(self):
        total_counts = np.zeros(70)
        for i in range(1, 6):  # Skip the date column
            counts = np.bincount(self.data[:, i].astype(int), minlength=70)
            total_counts += counts
        
        total_occurrences = np.sum(total_counts)
        percentages = (total_counts / total_occurrences) * 100
        
        return percentages

    def calculate_percentage_sixth(self):
        counts = np.bincount(self.data[:, 6].astype(int), minlength=27)
        total_occurrences = np.sum(counts)
        percentages = (counts / total_occurrences) * 100
        
        return percentages

    def calculate_percentage_seventh(self):
        counts = np.bincount(self.data[:, 7].astype(int), minlength=11)
        specific_counts = [counts[2], counts[3], counts[4], counts[5], counts[10]]
        total_occurrences = np.sum(specific_counts)
        percentages = (np.array(specific_counts) / total_occurrences) * 100
        
        return percentages

# Example usage
if __name__ == "__main__":
    # Create a CSV file with initial data if it doesn't exist
    initial_csv = 'initial_data.csv'
    try:
        df = pd.read_csv(initial_csv)
    except FileNotFoundError:
        initial_data = ["07/01/2023,1,2,3,4,5,6,2", "07/02/2023,7,8,9,10,11,12,3", "07/03/2023,13,14,15,16,17,18,4"]
        df = pd.DataFrame([d.split(',') for d in initial_data], columns=["date"] + [f"num{i+1}" for i in range(7)])
        df.to_csv(initial_csv, index=False)

    number_pattern = NumberPattern(initial_csv)
    #number_pattern.add_data(["07/04/2023,19,20,21,22,23,24,5", "07/05/2023,25,26,27,28,29,30,10"])
    
    # Predict next numbers
    top_five_numbers, top_sixth_number, top_seventh_number, next_numbers = number_pattern.predict_next_by_frequency()
    
    print("\nTop 5 most frequently used numbers for the first five positions:")
    for i, top_five in enumerate(top_five_numbers):
        print(f"Position {i+1}: {', '.join(map(str, top_five))}")

    print(f"\nTop most frequently used number for the sixth position: {top_sixth_number}")
    print(f"Top most frequently used number for the seventh position: {top_seventh_number}")

    # Display the predicted next numbers
    next_numbers_str = ', '.join(map(str, next_numbers))
    print(f"\nPredicted next numbers: {next_numbers_str}")

    # Display percentages for first five numbers
    print("\nPercentages for numbers 1-69 in the first five positions:")
    percentages_first_five = number_pattern.calculate_percentage_first_five()
    for i, perc in enumerate(percentages_first_five):
        if i > 0:
            print(f"Number {i:2d}: {perc:12.9f}%")

    # Display percentages for the sixth number
    print("\nPercentages for numbers 1-26 in the sixth position:")
    percentages_sixth = number_pattern.calculate_percentage_sixth()
    for i, perc in enumerate(percentages_sixth):
        if i > 0:
            print(f"Number {i:2d}: {perc:12.9f}%")
    
    # Display percentages for the seventh number
    print("\nPercentages for numbers 2, 3, 4, 5, 10 in the seventh position:")
    percentages_seventh = number_pattern.calculate_percentage_seventh()
    for val, perc in zip([2, 3, 4, 5, 10], percentages_seventh):
        print(f"Number {val:2d}: {perc:12.9f}%")