import argparse
import matplotlib.pyplot as plt
from model import MetaLearner, NERModel
from preprocess import sentencize, filter_tasks
from sklearn.metrics import classification_report
from split import global_dictionary
from task import TrainTask, TestTask, DevTask
from tokenize_and_stuff import tokenize_bert
import torch
import torch.nn as nn
import torch.optim as optim
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
    parser.add_argument("--e", help="number of epochs")
    parser.add_argument("--meta_lr", help="lr for metalearning")
    parser.add_argument("--lr", help="lr for inner loop")
    parser.add_argument("--train", action="store_true", help="meta-training are you doing it")
    parser.add_argument("--test", action="store_true", help="are you doin meta-testing")
    parser.add_argument("--model_path",help="path to the model")
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_path = args.train_path
    dev_path = args.dev_path
    test_path = args.test_path
    train = args.train
    test = args.test
    model_path = args.model_path
    k = args.k
    if k is None:
        k=5
    else:
        k = int(k)
    q=args.q
    if q is None:
        q=k*3
    else:
        q = int(q)
    max_len = args.max_len
    if max_len is None:
        max_len = 512
    else:
        max_len = int(max_len)
    batch_size = args.batch_size
    if batch_size is None:
        batch_size=5
    else:
        batch_size=int(batch_size)
    epochs = args.e
    if epochs is None:
        epochs = 100
    else:
        epochs = int(epochs)
    meta_lr = args.meta_lr
    if meta_lr is None:
        meta_lr = .001
    else:
        meta_lr = float(meta_lr)
    lr = args.lr
    if lr is None:
        lr = .1
    else:
        lr = float(lr)
    
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

    meta_learner = MetaLearner()
    meta_learner.to(device)
    if train:
        meta_optim = optim.Adam(meta_learner.parameters(), lr=meta_lr)
        loss_function = nn.BCELoss()
        train_losses = []
        dev_losses = []
        for epoch in range(epochs):
            print("Epoch number: "+str(epoch))
            meta_loss_total = 0.0
            for task in train_dataloader_support:
                learner = MetaLearner()
                learner.load_state_dict(meta_learner.state_dict())
                learner.to(device)
                optimizer = optim.Adam(learner.parameters(), lr=lr)
                for sent_ids, labels, attention in train_dataloader_support[task]:
                    sent_ids = sent_ids.to(device)
                    labels = labels.to(device)
                    attention = attention.to(device)
                    learner.zero_grad()
                    y_pred = learner(sent_ids, attention)
                    loss = loss_function(y_pred.squeeze().float(), labels.float())
                    loss.backward()
                    optimizer.step()
                meta_loss = 0.0
                for sent_ids, labels, attention in train_dataloader_query[task]:
                    sent_ids = sent_ids.to(device)
                    labels = labels.to(device)
                    attention = attention.to(device)
                    y_pred = learner(sent_ids, attention)
                    loss = loss_function(y_pred.squeeze().float(), labels.float())
                    meta_loss+=loss
                meta_loss_total += meta_loss.item()
            
            meta_loss_avg = meta_loss_total / len(train_dataloader_query)
            meta_optim.zero_grad()
            meta_loss_avg.backward()
            meta_optim.step()
            train_losses.append(meta_loss_avg)
            print("Training loss average: "+str(meta_loss_avg))
            meta_loss_total = 0.0
            for task in dev_dataloader_support:
                learner = MetaLearner()
                learner.load_state_dict(meta_learner.state_dict())
                learner.to(device)
                optimizer = optim.Adam(learner.parameters(), lr=lr)
                for sent_ids, labels, attention in dev_dataloader_support[task]:
                    sent_ids = sent_ids.to(device)
                    labels = labels.to(device)
                    attention = attention.to(device)
                    learner.zero_grad()
                    y_pred = learner(sent_ids, attention)
                    loss = loss_function(y_pred.squeeze().float(), labels.float())
                    loss.backward()
                    optimizer.step()
                meta_loss = 0.0
                with torch.no_grad():
                    for sent_ids, labels, attention in dev_dataloader_query[task]:
                        sent_ids = sent_ids.to(device)
                        labels = labels.to(device)
                        attention = attention.to(device)
                        y_pred = learner(sent_ids, attention)
                        loss = loss_function(y_pred.squeeze().float(), labels.float())
                        meta_loss+=loss
                    meta_loss_total+=meta_loss.item()
            meta_loss_avg = meta_loss_total/len(dev_dataloader_query)
            dev_losses.append(meta_loss_avg)
            print("Dev loss average: "+str(meta_loss_avg))
            torch.save(meta_learner.state_dict(), "model_dumps/"+str(epoch)+".pth")
        min_val = None
        min_index = None
        for i in range(dev_losses):
            if min_val is None or dev_losses[i]<min_val:
                min_val = dev_losses[i]
                min_index=i
        print("Best model is at epoch: "+str(min_index))
        plt.plot(train_losses, label="Avg Training Loss")
        plt.plot(dev_losses, label= "Avg Dev Loss")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(str(k)+"-shotLoss.png")
        plt.close()
    if test:
        if not train:
            meta_learner = MetaLearner()
            meta_learner.load_state_dict(torch.load(model_path))
            meta_learner.to(device)

        for task in test_dataloader_support:
            learner = MetaLearner()
            learner.load_state_dict(meta_learner.state_dict())
            learner.to(device)
            optimizer = optim.Adam(learner.parameters(), lr=lr)
            for sent_ids, labels, attention in test_dataloader_support[task]:
                sent_ids = sent_ids.to(device)
                labels = labels.to(device)
                attention = attention.to(device)
                optimizer.zero_grad()
                y_pred = learner(sent_ids, attention)
                loss = loss_function(y_pred.squeeze().float(), labels.float())
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                all_preds = []
                all_labels = []
                for sent_ids, labels, attention in test_dataloader_query[task]:
                    sent_ids = sent_ids.to(device)
                    labels = labels.to(device)
                    attention = attention.to(device)
                    y_pred = learner(sent_ids, attention)
                    binary_preds = (y_pred>.5).squeeze().cpu().numpy()
                    labels = labels.cpu().numpy()
                    all_preds.extend(binary_preds[labels!=-100])
                    all_labels.extend(labels[labels!=-100])

                with open(str(k)+"-shot"+task+"Report.txt", "w") as out:
                    out.write(classification_report(all_labels, all_preds, target_names=["O", task], labels=[0, 1]))
        
            

    


if __name__ == '__main__':
    main()