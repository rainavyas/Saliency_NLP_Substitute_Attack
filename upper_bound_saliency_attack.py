'''
Use upper bound saliency at the word embedding level to choose word to substitute
with synonym.

This is currently designed for BERT encoder based models only
'''

import torch
import torch.nn as nn
import nltk
from nltk.corpus import wordnet as wn
from layer_handler import Bert_Layer_Handler
from models import BertSequenceClassifier
from data_prep_sentences import get_test
import json
from transformers import BertTokenizer
import sys
import os
import argparse

def get_token_saliencies(sentence, label, handler, criterion, tokenizer):
    '''
    Returns tensor of saliencies in token order

    Saliency is an upperbound saliency, given by the size of the vector of the
    loss functions derivative wrt to the word embedding.
    Word embeddings are taken from the input embedding layer before the encoder
    Note that the label should be the true label (1 or 0)
    '''

    encoded_inputs = tokenizer([sentence], padding=True, truncation=True, return_tensors="pt")
    ids = encoded_inputs['input_ids']
    mask = encoded_inputs['attention_mask']

    target = torch.LongTensor([label])

    model.eval()
    embeddings = handler.get_layern_outputs(ids, mask)
    embeddings.retain_grad()
    logits = handler.pass_through_rest(embeddings, mask)
    loss = criterion(logits, target)

    # Determine embedding token saliencies
    loss.backward()
    embedding_grads = embeddings.grad
    saliencies = torch.linalg.norm(embedding_grads, dim=-1).squeeze()

    return saliencies


def attack_sentence(sentence, label, model, handler, criterion, tokenizer, max_syn=5, N=1):
    '''
    Identifies the N most salient words (by upper bound saliency)
    Finds synonyms for these words using WordNet
    Selects the best synonym to replace with based on Forward Pass to maximise
    the loss function, sequentially starting with most salient word

    Returns the original_sentence, updated_sentence, original_logits, updated_logits
    '''
    model.eval()

    token_saliencies = get_token_saliencies(sentence, label, handler, criterion, tokenizer)
    token_saliencies[0] = 0
    token_saliencies[-1] = 0

    inds = torch.argsort(token_saliencies, descending=True)
    inds = inds[:N]

    encoded_inputs = tokenizer([sentence], padding=True, truncation=True, return_tensors="pt")
    ids = encoded_inputs['input_ids'].squeeze()
    mask = encoded_inputs['attention_mask']

    assert len(token_saliencies) == len(ids), "tokens and saliencies mismatch"

    for i, ind in enumerate(inds):
        target_id = ids[ind]
        word_token = tokenizer.convert_ids_to_tokens(target_id.item())

        synonyms = wn.synset(word_token+'.n.01').lemma_names()
        if len(synonyms) > max_syn+1:
            synonyms = synonyms[:max_syn+1]

        best = (None, 0) # (id, loss)
        for j, syn in enumerate(synonyms):
            try:
                new_id = tokenizer.convert_tokens_to_ids(syn)
            except:
                print(syn+" is not a token")
                continue

            ids[ind] = new_id
            with torch.no_grad():
                logits = model(torch.unsqueeze(ids, dim=0), mask)
                loss = criterion(logits, torch.LongTensor([label])).item()

            if i==0 and j==0:
                original_logits = logits.squeeze()
            if loss > best[1]:
                best = (new_id, loss)
                updated_logits = logits.squeeze()
        ids[ind] = best[0]

    updated_sentence = tokenizer.decode(ids)

    return sentence, updated_sentence, original_logits, updated_logits

if __name__ == '__main__':

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('MODEL', type=str, help='trained .th model')
    commandLineParser.add_argument('DIR', type=str, help='data base directory')
    commandLineParser.add_argument('--max_syn', type=int, default=5, help="Number of synonyms to search")
    commandLineParser.add_argument('--N', type=int, default=1, help="Number of words to substitute")
    commandLineParser.add_argument('--ind', type=int, default=0, help="IMDB file index for both pos and neg review")

    args = commandLineParser.parse_args()
    model_path = args.MODEL
    base_dir = args.DIR
    max_syn = args.max_syn
    N = args.N
    file_ind = args.ind

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/upper_bound_saliency_attack.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    nltk.download('wordnet')

    # Load the model
    model = BertSequenceClassifier()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    # Create model handler
    handler = Bert_Layer_Handler(model, layer_num=0)

    # Get the relevant data
    neg_review_list, pos_review_list, neg_labels, pos_labels = get_test(base_dir)
    neg_sentence = neg_review_list[file_ind]
    pos_sentence = pos_review_list[file_ind]
    neg_label = neg_labels[file_ind]
    pos_label = pos_labels[file_ind]

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    criterion = nn.CrossEntropyLoss()
    softmax = nn.Softmax()

    # Create directory to save files in
    if not os.path.isdir('Attacked_Data'):
        os.mkdir('Attacked_Data')

    # Attack and save the negative sentence attack
    sentence, updated_sentence, original_logits, updated_logits = attack_sentence(neg_sentence, neg_label, model, handler, criterion, tokenizer, max_syn=max_syn, N=N)
    original_probs = softmax(original_logits).tolist()
    updated_probs = softmax(updated_logits).tolist()
    info = {"sentence":sentence, "updated sentence":updated_sentence, "true label":neg_label, "original prob":original_probs, "updated prob":updated_probs}
    filename = 'Attacked_Data/neg'+str(file_ind)+'.txt'
    with open(filename, 'w') as f:
        f.write(json.dumps(info))

    # Attack and save the positive sentence attack
    sentence, updated_sentence, original_logits, updated_logits = attack_sentence(pos_sentence, pos_label, model, handler, criterion, tokenizer, max_syn=max_syn, N=N)
    original_probs = softmax(original_logits).tolist()
    updated_probs = softmax(updated_logits).tolist()
    info = {"sentence":sentence, "updated sentence":updated_sentence, "true label":pos_label, "original prob":original_probs, "updated prob":updated_probs}
    filename = 'Attacked_Data/pos'+str(file_ind)+'.txt'
    with open(filename, 'w') as f:
        f.write(json.dumps(info))
