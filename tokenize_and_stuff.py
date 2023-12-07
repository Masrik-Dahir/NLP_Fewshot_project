import transformers
transformers.logging.set_verbosity_error()  # Silence HuggingFace Transformers Warnings/Info Statements
from transformers import BertTokenizerFast

def tokenize_bert(sentences, labels, max_len):
    all_sentence_ids = []
    all_sentence_labels = []
    all_attention_masks = []
    tokenizer = BertTokenizerFast.from_pretrained( "bert-base-cased", do_lower_case=False)
    
    for sentidx in range(len(sentences)):
        sent = sentences[sentidx]
        tokenized_sequences = tokenizer(sent)
        input_ids, attention_masks, token_type_ids = tokenized_sequences['input_ids'], tokenized_sequences['attention_mask'], tokenized_sequences['token_type_ids']
        sentence_ids = [101]
        sentence_labels = [0]
        attention_mask = [1]
        for idx in range(len(input_ids)):
            assert(len(input_ids)==len(labels[sentidx]))
            word = input_ids[idx]
            for id in word:
                if len(sentence_ids)==max_len-1:
                    sentence_ids.append(102)
                    sentence_labels.append(0)
                    attention_mask.append(1)
                elif len(sentence_ids) < max_len:
                    if id not in [101,102]:
                        sentence_ids.append(id)
                        if labels[sentidx][idx] == "O":
                            sentence_labels.append(0)
                        else:
                            sentence_labels.append(1)
                        attention_mask.append(1)
        assert(len(sentence_ids)==len(sentence_labels))
        assert(len(sentence_ids)==len(attention_mask))

        while len(sentence_ids)!= max_len:
            sentence_ids.append(0)
            sentence_labels.append(-100)
            attention_mask.append(0)
        assert(len(sentence_ids)==max_len)
        all_sentence_ids.append(sentence_ids)
        all_sentence_labels.append(sentence_labels)
        all_attention_masks.append(attention_mask)
    return all_sentence_ids,all_sentence_labels,all_attention_masks
    


    