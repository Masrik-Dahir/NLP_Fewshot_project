# Sample input data
input_data ="""

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
        else:
            # If the entity is not 'O', track the current entity and entity type
            current_entity.append(word)
            current_entity_type = entity
    else:
        # If the line doesn't have two parts, something went wrong
        print(f"Error: Invalid line format: {line}")

# Add the tracked entity to the appropriate dataset
if current_entity_type == 'fname':
    dataset_fname.extend([f"{word} {current_entity_type}" for word in current_entity])
    dataset_lname.extend([f"{word} O" for word in current_entity])
elif current_entity_type == 'lname':
    dataset_fname.extend([f"{word} O" for word in current_entity])
    dataset_lname.extend([f"{word} {current_entity_type}" for word in current_entity])

# Print the resulting datasets
print("Dataset with 'fname':\n", "\n".join(dataset_fname))
print("\nDataset with 'lname':\n", "\n".join(dataset_lname))
