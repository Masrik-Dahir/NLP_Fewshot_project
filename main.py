import argparse
from preprocess import sentencize, filter_tasks
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

    n_tasks_to_sents_and_labels = {}

    for file in train_files:
        for entity in train_files[file]:
            if entity not in n_tasks_to_sents_and_labels:
                n_tasks_to_sents_and_labels[entity] = {"sents":[],"labels":[]}
            sents, labels = sentencize(train_files[file][entity])
            n_tasks_to_sents_and_labels[entity]["sents"].extend(sents)
            n_tasks_to_sents_and_labels[entity]["labels"].extend(labels)

    n_tasks_to_sents_and_labels = filter_tasks(n_tasks_to_sents_and_labels)
    for task in n_tasks_to_sents_and_labels:
        print(task)
        print(len(n_tasks_to_sents_and_labels[task]["sents"]))
        assert(len(n_tasks_to_sents_and_labels[task]["sents"])==len(n_tasks_to_sents_and_labels[task]["labels"]))
        
if __name__ == '__main__':
    main()