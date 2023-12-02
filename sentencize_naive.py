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
            labels.append(label)

            current_sent = []
            current_labels = []
        if i==len(biotext)-1 and len(current_sent)!=0:
            sents.append(current_sent)
            labels.append(current_labels)
    return sents, labels

