# Sample input data
input_data = """
My O
name O
is O
Char fname
Dil lname
"""

# Split data into lines
lines = input_data.strip().split('\n')

# Initialize two datasets
dataset_fname = []
dataset_lname = []

# Initialize variables to keep track of the current entity being processed
current_entity = []
current_entity_type = None

# Process the lines and split the data
for line in lines:
    parts = line.split()
    if len(parts) == 2:
        word, entity = parts
        if entity == 'O':
            # If the entity is 'O', add it directly to both datasets
            dataset_fname.append(f"{word} {entity}")
            dataset_lname.append(f"{word} {entity}")
        elif entity == 'fname':
            # If the entity is 'O', add it directly to both datasets
            dataset_fname.append(f"{word} fname")
            dataset_lname.append(f"{word} 0")
        elif entity == 'lname':
            # If the entity is 'O', add it directly to both datasets
            dataset_fname.append(f"{word} 0")
            dataset_lname.append(f"{word} lname")
    else:
        # If the line doesn't have two parts, something went wrong
        print(f"Error: Invalid line format: {line}")



# Print the resulting datasets
print("Dataset with 'fname':\n", "\n".join(dataset_fname))
print("\nDataset with 'lname':\n", "\n".join(dataset_lname))
