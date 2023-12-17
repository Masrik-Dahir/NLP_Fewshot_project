"""
Charie Dil, Nourchene Bargaoui, Masrik Dahir - 12/17/2023

Problem statement: NER is the task of identifying key terms in text, and it typically requires large columes of data. Acquiring annotated data is expensive and time consuming. We can use few-shot learing to address this problem! Specifically, we address the problem from a meta-learning standpoint. We have meta-training tasks and meta-testing tasks, and using k examples for n classes in each meta-training task, we want to generalize our model to handle new unseen tasks with limited support (k labeled examples). We implement an optimization-based approach called MAML (Model Agnostic Meta Learning) to address this problem. We experimen with N 2-way 5-shot and N 2-way 25-shot frameworks.

Example of input file:
Example	O
194	B-EXAMPLE_LABEL
3-Isobutyl-5-methyl-1-(oxetan-2-ylmethyl)-6-[(2-oxoimidazolidin-1-yl)methyl]thieno[2,3-d]pyrimidine-2,4(1H,3H)-dione	B-REACTION_PRODUCT
(racemate)	I-REACTION_PRODUCT
813	O
...

Example of output: See reports folder

Usage Instructions: see README.md

Architecture/Algorithm
BASE MODEL:
    Frozen BERT. BiLSTM. Linear Layer

META_TRAINING:
    start with randomly initialized parameters for model. copy that model for each task. train on k support examples. evaluate meta-loss on query set. average the losses, backpropagate on the general model.

META_VALIDATION:
    take model at end of each epoch, make a copy of the overall model for each task, train on the validation support k-examples, evaluate meta-loss on the query set. average te losses, output. NO BACKPROP ON GENERAL MODEL

META_TESTING:
    Same as validation, except instead of meta_loss, just pass in predicted versus actual to classification report to get scores

"""

import argparse
import matplotlib.pyplot as plt
from model import MetaLearner, NERModel
from preprocess import sentencize, filter_tasks, clean_logits_and_labels
from sklearn.metrics import classification_report
from split import global_dictionary
from task import TrainTask, TestTask, DevTask
from tokenize_and_stuff import tokenize_bert, calculate_class_weights
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

#random task distribution
meta_train_tasks = ["B-EXAMPLE_LABEL", "B-REACTION_PRODUCT", "B-STARTING_MATERIAL", "B-SOLVENT", "I-TIME", "B-WORKUP", "I-YIELD_OTHER", "B-YIELD_PERCENT", "I-REAGENT_CATALYST"]
meta_dev_tasks = ["B-REAGENT_CATALYST", "B-REACTION_STEP", "B-TEMPERATURE", "I-TEMPERATURE"]
meta_test_tasks = ["B-TIME", "I-STARTING_MATERIAL", "I-REACTION_PRODUCT", "B-YIELD_OTHER"]


def main():
    #command line stuff
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
    #mapping files to different task "datasets"
    train_files = global_dictionary(train_path)
    dev_files = global_dictionary(dev_path)
    test_files = global_dictionary(test_path)
    #used for storing the actual generalized sentence and labels after sentencizing them
    train_n_tasks_sents_and_labels = {}

    for file in train_files:
        for entity in train_files[file]:
            if entity not in train_n_tasks_sents_and_labels:
                train_n_tasks_sents_and_labels[entity] = {"sents":[],"labels":[]}
            sents, labels = sentencize(train_files[file][entity])
            train_n_tasks_sents_and_labels[entity]["sents"].extend(sents)
            train_n_tasks_sents_and_labels[entity]["labels"].extend(labels)
#we just want sentences with target labels in them in our training dataset
    train_n_tasks_sents_and_labels = filter_tasks(train_n_tasks_sents_and_labels)
    #print("TRAIN STATISTICS - counts of sentences with each entity in them")
    for task in train_n_tasks_sents_and_labels:
        #print(task)
        #print(len(train_n_tasks_sents_and_labels[task]["sents"]))
        assert(len(train_n_tasks_sents_and_labels[task]["sents"])==len(train_n_tasks_sents_and_labels[task]["labels"]))

    dev_n_tasks_sents_and_labels = {}
#same thing as above fo dev, except wihtout the filtration step
    for file in dev_files:
        for entity in dev_files[file]:
            if entity not in dev_n_tasks_sents_and_labels:
                dev_n_tasks_sents_and_labels[entity] = {"sents":[],"labels":[]}
            sents, labels = sentencize(dev_files[file][entity])
            dev_n_tasks_sents_and_labels[entity]["sents"].extend(sents)
            dev_n_tasks_sents_and_labels[entity]["labels"].extend(labels)

    #print("DEV STATISTICS - counts of sentences for each task (not necessarily having an entity in them)")
    for task in dev_n_tasks_sents_and_labels:
        #print(task)
        #print(len(dev_n_tasks_sents_and_labels[task]["sents"]))
        assert(len(dev_n_tasks_sents_and_labels[task]["sents"])==len(dev_n_tasks_sents_and_labels[task]["labels"]))
#same thing with test
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
        #print(task)
        #print(len(test_n_tasks_sents_and_labels[task]["sents"]))
        assert(len(test_n_tasks_sents_and_labels[task]["sents"])==len(test_n_tasks_sents_and_labels[task]["labels"]))    
#build objects (task should consist of a support set and query set)
    text_train_tasks = {}
    text_dev_tasks = {}
    text_test_tasks = {}
    for task in meta_train_tasks:
        text_train_tasks[task] = TrainTask(train_n_tasks_sents_and_labels[task], k, q)
    for task in meta_dev_tasks:
        text_dev_tasks[task] = DevTask(dev_n_tasks_sents_and_labels[task], k)
    for task in meta_test_tasks:
        text_test_tasks[task] = TestTask(train_n_tasks_sents_and_labels[task], test_n_tasks_sents_and_labels[task], k)
        #store dataloaders
    train_dataloader_support = {}
    train_dataloader_query = {}

    dev_dataloader_support = {}
    dev_dataloader_query = {}

    test_dataloader_support = {}
    test_dataloader_query = {}
    for t in text_train_tasks:
#tokenize w bert tokenizer
        support_sent_ids, support_labels, support_attention = tokenize_bert(text_train_tasks[t].support_sentences, text_train_tasks[t].support_labels, max_len)
#create tensor datasets
        support_dataset = TensorDataset(torch.LongTensor(support_sent_ids), torch.LongTensor(support_labels), torch.LongTensor(support_attention))
        #create dataloader for train support
        train_dataloader_support[t] = DataLoader(support_dataset, batch_size=batch_size, shuffle=True)
        #same process with query train
        query_sent_ids, query_labels, query_attention = tokenize_bert(text_train_tasks[t].query_sentences, text_train_tasks[t].query_labels, max_len)

        query_dataset = TensorDataset(torch.LongTensor(query_sent_ids), torch.LongTensor(query_labels), torch.LongTensor(query_attention))
        train_dataloader_query[t] = DataLoader(query_dataset, batch_size=batch_size, shuffle=True)
    #now same thing on dev tasks
    for t in text_dev_tasks:
        support_sent_ids, support_labels, support_attention = tokenize_bert(text_dev_tasks[t].support_sentences, text_dev_tasks[t].support_labels, max_len)

        support_dataset = TensorDataset(torch.LongTensor(support_sent_ids), torch.LongTensor(support_labels), torch.LongTensor(support_attention))
        dev_dataloader_support[t] = DataLoader(support_dataset, batch_size=batch_size, shuffle=True)

        query_sent_ids, query_labels, query_attention = tokenize_bert(text_dev_tasks[t].query_sentences, text_dev_tasks[t].query_labels, max_len)

        query_dataset = TensorDataset(torch.LongTensor(query_sent_ids), torch.LongTensor(query_labels), torch.LongTensor(query_attention))
        dev_dataloader_query[t] = DataLoader(query_dataset, batch_size=batch_size, shuffle=True)
#now same thing for test
    for t in text_test_tasks:
        support_sent_ids, support_labels, support_attention = tokenize_bert(text_test_tasks[t].support_sentences, text_test_tasks[t].support_labels, max_len)

        support_dataset = TensorDataset(torch.LongTensor(support_sent_ids), torch.LongTensor(support_labels), torch.LongTensor(support_attention))
        test_dataloader_support[t] = DataLoader(support_dataset, batch_size=batch_size, shuffle=True)

        query_sent_ids, query_labels, query_attention = tokenize_bert(text_test_tasks[t].query_sentences, text_test_tasks[t].query_labels, max_len)

        query_dataset = TensorDataset(torch.LongTensor(query_sent_ids), torch.LongTensor(query_labels), torch.LongTensor(query_attention))
        test_dataloader_query[t] = DataLoader(query_dataset, batch_size=batch_size, shuffle=True)
    meta_learner = MetaLearner()
    meta_learner.to(device)
    if train: #training flag is true
        meta_optim = optim.Adam(meta_learner.parameters(), lr=meta_lr)#adam optim
        train_losses = []#saving losses for graphing
        dev_losses = []
        for epoch in range(epochs):
            with open(str(k)+"-shotlog.txt", "a+") as f:
                f.write("\nEpoch number: "+str(epoch))#custom log file
            meta_loss_total = 0.0#we sum up the losses on the query set here
            for task in train_dataloader_support:
                learner = MetaLearner()#init metaleraner
                learner.load_state_dict(meta_learner.state_dict())#copy
                learner.to(device)#gpu time! if possible
                optimizer = optim.Adam(learner.parameters(), lr=lr)#optimizer for learner
                for sent_ids, labels, attention in train_dataloader_support[task]:#go through the dataloader
                    sent_ids = sent_ids.to(device)#send everything to the gpu if applicable
                    labels = labels.to(device)
                    attention = attention.to(device)
                    learner.zero_grad()#zero grad
                    y_pred = learner(sent_ids, attention)#predict
                    y_pred, labels = clean_logits_and_labels(y_pred, labels)#remove -100
                    cw = calculate_class_weights(labels)#calculate class weights
                    loss_function = nn.BCELoss(weight = cw)#init loss function with custom weights

                    loss = loss_function(y_pred.float(), labels.float())#calculate the loss
                    loss.backward()#calculate grads
                    optimizer.step()#update accordingly
                meta_loss = 0.0
                for sent_ids, labels, attention in train_dataloader_query[task]:
                    #same idea here but no immediate backprop
                    sent_ids = sent_ids.to(device)
                    labels = labels.to(device)
                    attention = attention.to(device)
                    y_pred = learner(sent_ids, attention)
                    y_pred, labels = clean_logits_and_labels(y_pred, labels)
                    cw = calculate_class_weights(labels)
                    loss_function = nn.BCELoss(weight=cw)

                    loss = loss_function(y_pred.float(), labels.float())
                    meta_loss+=loss#sum up the losses per the whole task
                meta_loss_total += meta_loss#add up the sum for each task
            #NOTE here we take the average
            meta_loss_avg = meta_loss_total / len(train_dataloader_query)
            meta_optim.zero_grad()#zero grad
            meta_loss_avg.backward()#backprop on the avg
            meta_optim.step()#grad descent
            train_losses.append(meta_loss_avg)
            with open(str(k)+"-shotlog.txt", "a+") as f:
                f.write("\nTraining loss average: "+str(meta_loss_avg))
            meta_loss_total = 0.0#same sort of idea, except we don't do the outer loop gradient descent thing
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
                    y_pred, labels = clean_logits_and_labels(y_pred, labels)
                    cw = calculate_class_weights(labels)
                    loss_function = nn.BCELoss(weight=cw)


                    loss = loss_function(y_pred.float(), labels.float())
                    loss.backward()
                    optimizer.step()
                meta_loss = 0.0
                with torch.no_grad():
                    for sent_ids, labels, attention in dev_dataloader_query[task]:
                        sent_ids = sent_ids.to(device)
                        labels = labels.to(device)
                        attention = attention.to(device)
                        y_pred = learner(sent_ids, attention)
                        y_pred, labels = clean_logits_and_labels(y_pred, labels)
                        cw = calculate_class_weights(labels)
                        loss_function = nn.BCELoss(weight=cw)

                        loss = loss_function(y_pred.float(), labels.float())
                        meta_loss+=loss
                    meta_loss_total+=meta_loss
            meta_loss_avg = meta_loss_total/len(dev_dataloader_query)
            dev_losses.append(meta_loss_avg)
            with open(str(k)+"-shotlog.txt", "a+") as f:
                f.write("\nDev loss average: "+str(meta_loss_avg))
            torch.save(meta_learner.state_dict(), "model_dumps/"+str(k)+"-shot-"+str(epoch)+".pth")
        min_val = None
        min_index = None
        for i in range(len(dev_losses)):
            if min_val is None or dev_losses[i]<min_val:
                min_val = dev_losses[i]
                min_index=i
        with open(str(k)+"-shotlog.txt", "a+") as f:
            f.write("\nBest model is at epoch: "+str(min_index))
    if test:#test flag true
        if not train:#if not train, we need a model path
            meta_learner = MetaLearner()
            meta_learner.load_state_dict(torch.load(model_path))
            meta_learner.to(device)

        for task in test_dataloader_support:
            learner = MetaLearner()#same idea as dev, except no loss in the outer loop just preictions
            learner.load_state_dict(meta_learner.state_dict())
            learner.to(device)
            optimizer = optim.Adam(learner.parameters(), lr=lr)
            for sent_ids, labels, attention in test_dataloader_support[task]:
                sent_ids = sent_ids.to(device)
                labels = labels.to(device)
                attention = attention.to(device)
                optimizer.zero_grad()
                y_pred = learner(sent_ids, attention)
                y_pred, labels = clean_logits_and_labels(y_pred, labels)
                cw = calculate_class_weights(labels)
                loss_function = nn.BCELoss(weight=cw)

                loss = loss_function(y_pred.float(), labels.float())
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
                    binary_preds = (y_pred>.5).cpu().numpy()
                    labels = labels.cpu().numpy()
                    all_preds.extend(binary_preds[labels!=-100])
                    all_labels.extend(labels[labels!=-100])
                with open(str(k)+"-shot"+task+"Report.txt", "w") as out:
                    out.write(classification_report(all_labels, all_preds, target_names=["O", task], labels=[0, 1]))
        
            

    


if __name__ == '__main__':
    main()
