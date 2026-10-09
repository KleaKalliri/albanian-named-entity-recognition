import pandas as pd

# Define the input and output file paths
input_file = 'korpusi.txt'
output_file = 'korpusi.csv'

# Read the text file into a pandas DataFrame
# Assuming the columns are separated by whitespace (e.g., tab or spaces)
data = pd.read_csv(input_file, sep='\s+', header=None, names=['Token', 'Label'])

# Save the DataFrame to a CSV file
data.to_csv(output_file, index=False)

print(f"Converted {input_file} to {output_file}")
