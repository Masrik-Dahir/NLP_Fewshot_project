def sentencize(biotext):
    sents = []
    labels = []

    current_sent = []
    current_labels = []
    for i in range(len(biotext)):
        row = biotext[i]
        tok = row.split()[0]
        label = row.split()[1]

        if tok.endswith("."):
            current_sent.append(tok)
            current_labels.append(label)

            sents.append(current_sent)
            labels.append(current_labels)

            current_sent = []
            current_labels = []
        else:
            current_sent.append(tok)
            current_labels.append(label)
        if i==len(biotext)-1 and len(current_sent)!=0:
            sents.append(current_sent)
            labels.append(current_labels)
    return sents, labels

def filter_tasks(task_dict):
    out = {}
    for task in task_dict:
        if task not in out:
            out[task] = {"sents": [], "labels":[]}
        for i in range(len(task_dict[task]["labels"])):
            if task in task_dict[task]["labels"][i]:
                out[task]["sents"].append(task_dict[task]["sents"][i])
                out[task]["labels"].append(task_dict[task]["labels"][i])
    return out

