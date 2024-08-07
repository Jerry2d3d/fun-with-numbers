import pandas as pd

def reformat_and_sort_csv(input_file, output_file):
    # Read the CSV file
    df = pd.read_csv(input_file, header=None, names=['date', 'numbers', 'num7'])
    
    # Ensure the date is in the correct format
    df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y')
    
    # Split the 'numbers' column into separate columns
    numbers_split = df['numbers'].str.split(' ', expand=True)
    numbers_split.columns = [f'num{i+1}' for i in range(numbers_split.shape[1])]
    
    # Combine the split numbers and the other columns
    df = pd.concat([df['date'], numbers_split, df['num7']], axis=1)
    
    # Sort by date from newest to oldest
    df = df.sort_values(by='date', ascending=False)
    
    # Convert the date back to the required string format
    df['date'] = df['date'].dt.strftime('%m/%d/%Y')
    
    # Save the reformatted data to the output file
    df.to_csv(output_file, index=False, header=False)

# Example usage
input_file = 'raw_data.csv'
output_file = 'reformatted_data.csv'
reformat_and_sort_csv(input_file, output_file)