import argparse
from split import global_dictionary
def main():
    parser = argparse.ArgumentParser(
                    prog='python main.py',
                    description='Baseline NER system for CHEMU dataset')
    parser.add_argument("--train_path", help="Path to training data")
    parser.add_argument("--test_path", help="path to the test data")
    parser.add_argument("--dev_path", help="path to the dev data")
    args = parser.parse_args()
    train_path = args.train_path
    dev_path = args.dev_path
    test_path = args.test_path

    train_files = global_dictionary(train_path)
    dev_files = global_dictionary(dev_path)
    test_files = global_dictionary(test_path)

    
    for file in train_files:
        print(file)
        print(train_files[file])
        exit()


if __name__ == '__main__':
    main()