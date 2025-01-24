import ast

# Define the path to your file
file_path = '../output/results.txt'

# Initialize an empty list to hold all elements
merged_list = []

# Open and read the file line by line
with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        # Remove newline characters and any potential leading/trailing spaces
        line = line.strip()
        if line:  # Check if the line is not empty
            try:
                # Convert string representation of list to actual list
                line_list = ast.literal_eval(line)
                if isinstance(line_list, list):  # Ensure the parsed object is a list
                    merged_list.extend(line_list)
            except ValueError as e:
                print(f"Error processing line: {line} with error {e}")

# The 'merged_list' now contains all items merged together from the lists in the file
print(merged_list[:])  # Print the first 10 items to verify
