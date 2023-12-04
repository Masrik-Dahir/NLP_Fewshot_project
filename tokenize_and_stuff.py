import transformers
transformers.logging.set_verbosity_error()  # Silence HuggingFace Transformers Warnings/Info Statements
from transformers import BertTokenizerFast

def tokenize_bert(task):
    tokenizer = BertTokenizerFast.from_pretrained( "bert-base-cased", do_lower_case=False)
    tokenized_sequences = tokenizer( task.support_sentences[0] )
    input_ids, attention_masks, token_type_ids = tokenized_sequences['input_ids'], tokenized_sequences['attention_mask'], tokenized_sequences['token_type_ids']
    print(tokenizer.convert_ids_to_tokens(input_ids[0]))
    print(task.support_sentences[0])
    print(input_ids)
    