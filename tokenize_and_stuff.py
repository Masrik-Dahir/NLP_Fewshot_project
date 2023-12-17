import torch
import transformers
transformers.logging.set_verbosity_error()  # Silence HuggingFace Transformers Warnings/Info Statements
from transformers import BertTokenizerFast

def tokenize_bert(sentences, labels, max_len):
    all_sentence_ids = []
    all_sentence_labels = []
    all_attention_masks = []
    tokenizer = BertTokenizerFast.from_pretrained( "bert-base-uncased", do_lower_case=False)
    
    for sentidx in range(len(sentences)):
        sent = sentences[sentidx]
        tokenized_sequences = tokenizer(sent)
        input_ids, attention_masks, token_type_ids = tokenized_sequences['input_ids'], tokenized_sequences['attention_mask'], tokenized_sequences['token_type_ids']
        sentence_ids = [101]
        sentence_labels = [[0]]
        attention_mask = [1]
        for idx in range(len(input_ids)):
            assert(len(input_ids)==len(labels[sentidx]))
            word = input_ids[idx]
            for id in word:
                if len(sentence_ids)==max_len-1:
                    sentence_ids.append(102)
                    sentence_labels.append([0])
                    attention_mask.append(1)
                elif len(sentence_ids) < max_len:
                    if id not in [101,102]:
                        sentence_ids.append(id)
                        if labels[sentidx][idx] == "O":
                            sentence_labels.append([0])
                        else:
                            sentence_labels.append([1])
                        attention_mask.append(1)
        assert(len(sentence_ids)==len(sentence_labels))
        assert(len(sentence_ids)==len(attention_mask))

        while len(sentence_ids)!= max_len:
            sentence_ids.append(0)
            sentence_labels.append([-100])
            attention_mask.append(0)
        assert(len(sentence_ids)==max_len)
        all_sentence_ids.append(sentence_ids)
        all_sentence_labels.append(sentence_labels)
        all_attention_masks.append(attention_mask)
    return all_sentence_ids,all_sentence_labels,all_attention_masks
    
def calculate_class_weights(labels):
    labels_flat = labels.detach().cpu().numpy().flatten()
    class_weight_dict = {0:0,1:0}
    total_num_samples = 0

    for l in labels_flat:
        class_weight_dict[int(l)]+=1
        total_num_samples +=1
    try:
        for key in class_weight_dict.keys():
            class_weight_dict[key] = total_num_samples/(2*class_weight_dict[key])
    except:
        for key in class_weight_dict.keys():
            class_weight_dict[key] = 1

    class_weights = [class_weight_dict[key] for key in labels_flat]
    class_weights = torch.FloatTensor(class_weights).reshape(labels.shape)
    class_weights = class_weights.to(labels.device)
    return class_weights
