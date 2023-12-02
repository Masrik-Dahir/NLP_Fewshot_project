import os

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

folders = [r'F:\Repo\PycharmProjects\NLP_Fewshot_project\Entity\dev_data',
           r'F:\Repo\PycharmProjects\NLP_Fewshot_project\Entity\train_data',
           r'F:\Repo\PycharmProjects\NLP_Fewshot_project\Entity\code_test']

def get_files_recursively(folder_path):
    file_list = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list
def pick_ann(given_list: list):
    lis = []
    for i in given_list:
        if i.endswith(".ann"):
            lis.append(i)
    return lis

def process(input_data):
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
    return datasets

def read(file_path):
    try:
        with open(file_path, 'r') as file:
            content = file.read()
        return content
    except FileNotFoundError:
        return f"File not found: {file_path}"
    except Exception as e:
        return f"An error occurred: {e}"

def global_dictionary():
    dictionary = {}
    for file in get_files_recursively(folders[0]):
        dictionary[file] = process(read(file))