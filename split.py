# Sample input data
input_data = """
My O
RANDOM kname
name O
is O
ABC fname
XYZ lname
XYT pname
"""

# Split data into lines
lines = input_data.strip().split('\n')

# Initialize a dictionary to store datasets dynamically
datasets = {}

for line in lines:
    parts = line.split()
    if len(parts) == 2:
        word, entity = parts
        if entity != 'O':
            if entity not in datasets:
                datasets[entity] = []


# Process the lines and split the data
for line in lines:
    parts = line.split()
    if len(parts) == 2:
        word, entity = parts
        # If the entity is not 'O', distribute it to the datasets accordingly
        if entity != 'O':
            datasets[entity].append(f"{word} {entity}")
            for dataset_name in datasets:
                if dataset_name != entity:
                    datasets[dataset_name].append(f"{word} 0")
        else:
            # If the entity is 'O', add it directly to all datasets
            for dataset_name in datasets:
                datasets[dataset_name].append(f"{word} {entity}")
    else:
        # If the line doesn't have two parts, something went wrong
        print(f"Error: Invalid line format: {line}")


# Print the resulting datasets
for dataset_name, dataset in datasets.items():
    print(f"\nDataset with '{dataset_name}':\n", "\n".join(dataset))