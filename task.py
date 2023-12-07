import random
class TrainTask:
    def __init__(self, task_data, num_support, num_query):
        # task_data: A list of tuples (sentence, entity_label)
        # num_classes: Number of unique entities in each task
        # num_support: Number of instances in the support set
        # num_query: Number of instances in the query set

        self.support_sentences = []
        self.support_labels = []
        self.query_sentences = []
        self.query_labels = []

        sents = task_data["sents"]
        labels = task_data["labels"]

        zipped_list = list(zip(sents, labels))
        random.seed(42)
        random.shuffle(zipped_list)

        sents, labels = zip(*zipped_list)
        
        self.support_sentences = sents[:num_support]
        self.support_labels = labels[:num_support]

        self.query_sentences = sents[num_support:num_support+num_query]
        self.query_labels = labels[num_support:num_support+num_query]
        


class DevTask:
    def __init__(self, task_data, num_support):
        self.support_sentences = []
        self.support_labels = []
        self.query_sentences = []
        self.query_labels = []

        counter = 0
        sents = task_data["sents"]
        labels = task_data["labels"]

        zipped_list = list(zip(sents, labels))
        random.seed(42)
        random.shuffle(zipped_list)

        sents, labels = zip(*zipped_list)
        
        for i in range(len(labels)):
            for label in labels[i]:   
                if label!="O" and counter<num_support:
                    self.support_sentences.append(sents[i])
                    self.support_labels.append(labels[i])
                    counter+=1
                else:
                    self.query_sentences.append(sents[i])
                    self.query_labels.append(labels[i])
class TestTask:
    def __init__(self, task_support_source, task_data, num_support):
        self.support_sentences = []
        self.support_labels = []
        self.query_sentences = []
        self.query_labels = []

        sents = task_support_source["sents"]
        labels = task_support_source["labels"]

        zipped_list = list(zip(sents, labels))
        random.seed(42)
        random.shuffle(zipped_list)

        sents, labels = zip(*zipped_list)

        self.support_sentences = sents[:num_support]
        self.support_labels = labels[:num_support]

        self.query_sentences = task_data["sents"]
        self.query_labels = task_data["labels"]


                   
