import argparse
from preprocess import sentencize, filter_tasks
from split import global_dictionary
from task import TrainTask, TestTask, DevTask
from tokenize_and_stuff import tokenize_bert
import torch
from torch.utils.data import TensorDataset, DataLoader
meta_train_tasks = ["B-EXAMPLE_LABEL", "B-REACTION_PRODUCT", "B-STARTING_MATERIAL", "B-SOLVENT", "I-TIME", "B-WORKUP", "I-YIELD_OTHER", "B-YIELD_PERCENT", "I-REAGENT_CATALYST"]
meta_dev_tasks = ["B-REAGENT_CATALYST", "B-REACTION_STEP", "B-TEMPERATURE", "I-TEMPERATURE"]
meta_test_tasks = ["B-TIME", "I-STARTING_MATERIAL", "I-REACTION_PRODUCT", "B-YIELD_OTHER"]


def main():
    parser = argparse.ArgumentParser(
                    prog='python main.py',
                    description='Baseline NER system for CHEMU dataset')
    parser.add_argument("--train_path", help="Path to training data")
    parser.add_argument("--test_path", help="path to the test data")
    parser.add_argument("--dev_path", help="path to the dev data")
    parser.add_argument("--k", help="number of support examples")
    parser.add_argument("--q", help="number of query set examples")
    parser.add_argument("--max_len", help="max_len of tokens for bert")
    parser.add_argument("--batch_size", help="specify batch_size (ANYTHING OVER 5 IS BIGGER THAN SUPPORT DATASET SIZE FOR K=5 AND WILL CAUSE ERRORS)")
    args = parser.parse_args()
    train_path = args.train_path
    dev_path = args.dev_path
    test_path = args.test_path
    k = args.k
    if k is None:
        k=5
    q=args.q
    if q is None:
        q=k*3
    max_len = args.max_len
    if max_len is None:
        max_len = 512
    batch_size = args.batch_size
    if batch_size is None:
        batch_size=5
    train_files = global_dictionary(train_path)
    dev_files = global_dictionary(dev_path)
    test_files = global_dictionary(test_path)

    train_n_tasks_sents_and_labels = {}

    for file in train_files:
        for entity in train_files[file]:
            if entity not in train_n_tasks_sents_and_labels:
                train_n_tasks_sents_and_labels[entity] = {"sents":[],"labels":[]}
            sents, labels = sentencize(train_files[file][entity])
            train_n_tasks_sents_and_labels[entity]["sents"].extend(sents)
            train_n_tasks_sents_and_labels[entity]["labels"].extend(labels)

    train_n_tasks_sents_and_labels = filter_tasks(train_n_tasks_sents_and_labels)
    print("TRAIN STATISTICS - counts of sentences with each entity in them")
    for task in train_n_tasks_sents_and_labels:
        print(task)
        print(len(train_n_tasks_sents_and_labels[task]["sents"]))
        assert(len(train_n_tasks_sents_and_labels[task]["sents"])==len(train_n_tasks_sents_and_labels[task]["labels"]))

    dev_n_tasks_sents_and_labels = {}

    for file in dev_files:
        for entity in dev_files[file]:
            if entity not in dev_n_tasks_sents_and_labels:
                dev_n_tasks_sents_and_labels[entity] = {"sents":[],"labels":[]}
            sents, labels = sentencize(dev_files[file][entity])
            dev_n_tasks_sents_and_labels[entity]["sents"].extend(sents)
            dev_n_tasks_sents_and_labels[entity]["labels"].extend(labels)

    print("DEV STATISTICS - counts of sentences for each task (not necessarily having an entity in them)")
    for task in dev_n_tasks_sents_and_labels:
        print(task)
        print(len(dev_n_tasks_sents_and_labels[task]["sents"]))
        assert(len(dev_n_tasks_sents_and_labels[task]["sents"])==len(dev_n_tasks_sents_and_labels[task]["labels"]))

    test_n_tasks_sents_and_labels = {}

    for file in test_files:
        for entity in test_files[file]:
            if entity not in test_n_tasks_sents_and_labels:
                test_n_tasks_sents_and_labels[entity] = {"sents":[],"labels":[]}
            sents, labels = sentencize(test_files[file][entity])
            test_n_tasks_sents_and_labels[entity]["sents"].extend(sents)
            test_n_tasks_sents_and_labels[entity]["labels"].extend(labels)

    print("TEST STATISTICS - counts of sentences for each task (not necessarily having an entity in them)")
    for task in test_n_tasks_sents_and_labels:
        print(task)
        print(len(test_n_tasks_sents_and_labels[task]["sents"]))
        assert(len(test_n_tasks_sents_and_labels[task]["sents"])==len(test_n_tasks_sents_and_labels[task]["labels"]))    

    text_train_tasks = {}
    text_dev_tasks = {}
    text_test_tasks = {}
    for task in meta_train_tasks:
        text_train_tasks[task] = TrainTask(train_n_tasks_sents_and_labels[task], k, q)
    for task in meta_dev_tasks:
        text_dev_tasks[task] = DevTask(dev_n_tasks_sents_and_labels[task], k)
    for task in meta_test_tasks:
        text_test_tasks[task] = TestTask(train_n_tasks_sents_and_labels[task], test_n_tasks_sents_and_labels[task], k)
    train_dataloader_support = {}
    train_dataloader_query = {}

    dev_dataloader_support = {}
    dev_dataloader_query = {}

    test_dataloader_support = {}
    test_dataloader_query = {}
    for t in text_train_tasks:

        support_sent_ids, support_labels, support_attention = tokenize_bert(text_train_tasks[t].support_sentences, text_train_tasks[t].support_labels, max_len)

        support_dataset = TensorDataset(torch.LongTensor(support_sent_ids), torch.LongTensor(support_labels), torch.LongTensor(support_attention))
        train_dataloader_support[t] = DataLoader(support_dataset, batch_size=batch_size, shuffle=True)

        query_sent_ids, query_labels, query_attention = tokenize_bert(text_train_tasks[t].query_sentences, text_train_tasks[t].query_labels, max_len)

        query_dataset = TensorDataset(torch.LongTensor(query_sent_ids), torch.LongTensor(query_labels), torch.LongTensor(query_attention))
        train_dataloader_query[t] = DataLoader(query_dataset, batch_size=batch_size, shuffle=True)
    
    for t in text_dev_tasks:
        support_sent_ids, support_labels, support_attention = tokenize_bert(text_dev_tasks[t].support_sentences, text_dev_tasks[t].support_labels, max_len)

        support_dataset = TensorDataset(torch.LongTensor(support_sent_ids), torch.LongTensor(support_labels), torch.LongTensor(support_attention))
        dev_dataloader_support[t] = DataLoader(support_dataset, batch_size=batch_size, shuffle=True)

        query_sent_ids, query_labels, query_attention = tokenize_bert(text_dev_tasks[t].query_sentences, text_dev_tasks[t].query_labels, max_len)

        query_dataset = TensorDataset(torch.LongTensor(query_sent_ids), torch.LongTensor(query_labels), torch.LongTensor(query_attention))
        dev_dataloader_query[t] = DataLoader(query_dataset, batch_size=batch_size, shuffle=True)

    for t in text_test_tasks:
        support_sent_ids, support_labels, support_attention = tokenize_bert(text_test_tasks[t].support_sentences, text_test_tasks[t].support_labels, max_len)

        support_dataset = TensorDataset(torch.LongTensor(support_sent_ids), torch.LongTensor(support_labels), torch.LongTensor(support_attention))
        test_dataloader_support[t] = DataLoader(support_dataset, batch_size=batch_size, shuffle=True)

        query_sent_ids, query_labels, query_attention = tokenize_bert(text_test_tasks[t].query_sentences, text_test_tasks[t].query_labels, max_len)

        query_dataset = TensorDataset(torch.LongTensor(query_sent_ids), torch.LongTensor(query_labels), torch.LongTensor(query_attention))
        test_dataloader_query[t] = DataLoader(query_dataset, batch_size=batch_size, shuffle=True)      
    
    ##MAML CODE GOES HERE

    


if __name__ == '__main__':
    main()