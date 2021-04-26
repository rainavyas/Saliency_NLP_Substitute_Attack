'''
Use first order saliency at the word embedding level to choose word to substitute
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
from collections import OrderedDict

class Best_Tokens():
    '''
    Maintains an ordered list, ordered by saliency
    Each item consists of a dict:
        token_index: val
        original_token_id: val
        new_token_id: val # The synonym that gives highest first order saliency
        saliency: val
    '''
    def __init__(self, N):
        self.data = [{'token_index':None, 'original_token_id':None, 'new_token_id':None, 'saliency':-0.1}]*N

    def check_data_to_be_added(self, new_saliency):
        if new_saliency > self.data[-1]['saliency']:
            return True
        else:
            return False

    def add_data(self, token_index, original_token_id, new_token_id, saliency):
        new_data = {'token_index':token_index, 'original_token_id':original_token_id, 'new_token_id':new_token_id, 'saliency':saliency}
        self.data.append(new_data)
        # Sort from highest to lowest
        self.data = sorted(self.data, reverse=True, key = lambda x: x['saliency'])
        # Drop the worst
        self.data = self.data[:-1]

def get_token_gradient_vectors(sentence, label, handler, criterion, tokenizer):
    '''
    Returns gradient vectors of loss differentiated wrt embedding layer tokens
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

    # Determine embedding token gradient vectors
    loss.backward()
    embedding_grads = embeddings.grad
    embedding_grads = embedding_grads.squeeze(dim=0)

    return embedding_grads

def attack_sentence(sentence, label, model, handler, criterion, tokenizer, max_syn=5, N=1):
    '''
    Identifies the N most salient words (by first order saliency)
    using embeddings of synonyms for these words from WordNet
    Selects the best synonym to replace with based on maximising closeness to embedding gradient
    Replaces the N highest first order saliency words (saliency calculated for original sentence)

    Returns the original_sentence, updated_sentence, original_logits, updated_logits
    '''
    model.eval()
    token_gradient_vectors = get_token_gradient_vectors(sentence, label, handler, criterion, tokenizer)
    best = Best_Tokens(N)

    encoded_inputs = tokenizer([sentence], padding=True, truncation=True, return_tensors="pt")
    ids = encoded_inputs['input_ids'].squeeze()
    mask = encoded_inputs['attention_mask']

    for ind, grad_vec in enumerate(token_gradient_vectors):
        ids_copy = ids.clone()
        original_id = ids_copy[ind]
        # Calculate original token embedding
        with torch.no_grad():
            embeddings = handler.get_layern_outputs(ids_copy.unsqueeze(dim=0), mask).squeeze(dim=0)
            original_embedding = embeddings[ind]

        word_token = tokenizer.convert_ids_to_tokens(original_id.item())

        synonyms = []
        for syn in wn.synsets(word_token):
            for lemma in syn.lemmas():
                synonyms.append(lemma.name())
        if len(synonyms)==0:
            print(original_id, "has no synonyms")
            continue
        # Remove duplicates
        synonyms = list(OrderedDict.fromkeys(synonyms))

        if len(synonyms) > max_syn+1:
            synonyms = synonyms[:max_syn+1]

        best_syn = [original_id, 0] # (new id, first order saliency)
        for syn in synonyms:
            try:
                new_id = tokenizer.convert_tokens_to_ids(syn)
            except:
                print(syn+" is not a token")
                continue
            ids_copy[ind] = new_id
            # calculate new id token embedding
            with torch.no_grad():
                embeddings = handler.get_layern_outputs(ids_copy.unsqueeze(dim=0), mask).squeeze(dim=0)
                new_embedding = embeddings[ind]

            # calculate first order saliency
            with torch.no_grad():
                if new_id != original_id:
                    diff = torch.norm(new_embedding - original_embedding)
                    saliency = torch.dot(diff, grad_vec).item()
                    print(saliency)
                else:
                    saliency = 0

            # Compare and update best syn
            if saliency > best_syn[1]:
                best_syn = [new_id, saliency]

        # Compare with best N substitutions data
        if best.check_data_to_be_added(best_syn[1]):
            best.add_data(ind, original_id, best_syn[0], best_syn[1])

    # Determine original logits
    with torch.no_grad():
        original_logits = model(ids.unsqueeze(dim=0), mask).squeeze()

    # Use best N data to make substitutions
    for item in best.data:
        print(item)
        ind  = item['token_index']
        ids[ind] = item['new_token_id']

    # Determine updated logits
    with torch.no_grad():
        updated_logits = model(ids.unsqueeze(dim=0), mask).squeeze()

    updated_sentence = tokenizer.decode(ids)
    updated_sentence = updated_sentence.replace('[CLS] ', '')
    updated_sentence = updated_sentence.replace(' [SEP]', '')
    updated_sentence = updated_sentence.replace('[UNK]', '')

    return sentence, updated_sentence, original_logits, updated_logits

if __name__ == '__main__':

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('MODEL', type=str, help='trained .th model')
    commandLineParser.add_argument('DIR', type=str, help='data base directory')
    commandLineParser.add_argument('--max_syn', type=int, default=5, help="Number of synonyms to search")
    commandLineParser.add_argument('--N', type=int, default=1, help="Number of words to substitute")
    commandLineParser.add_argument('--start_ind', type=int, default=0, help="start IMDB file index for both pos and neg review")
    commandLineParser.add_argument('--end_ind', type=int, default=100, help=" end IMDB file index for both pos and neg review")

    args = commandLineParser.parse_args()
    model_path = args.MODEL
    base_dir = args.DIR
    max_syn = args.max_syn
    N = args.N
    start_ind = args.start_ind
    end_ind = args.end_ind

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/first_order_saliency_attack.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    nltk.download('wordnet')

    # Load the model
    model = BertSequenceClassifier()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    # Create model handler
    handler = Bert_Layer_Handler(model, layer_num=0)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    criterion = nn.CrossEntropyLoss()
    softmax = nn.Softmax(dim=0)

    # Create directory to save files in
    dir_name = 'First_Order_Attacked_Data_N'+str(N)
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)

    # Get all data
    neg_review_list, pos_review_list, neg_labels, pos_labels = get_test(base_dir)

    for file_ind in range(start_ind, end_ind):

        # Get the relevant data
        neg_sentence = neg_review_list[file_ind]
        pos_sentence = pos_review_list[file_ind]
        neg_label = neg_labels[file_ind]
        pos_label = pos_labels[file_ind]

        # Attack and save the negative sentence attack
        sentence, updated_sentence, original_logits, updated_logits = attack_sentence(neg_sentence, neg_label, model, handler, criterion, tokenizer, max_syn=max_syn, N=N)
        original_probs = softmax(original_logits).tolist()
        updated_probs = softmax(updated_logits).tolist()
        info = {"sentence":sentence, "updated sentence":updated_sentence, "true label":neg_label, "original prob":original_probs, "updated prob":updated_probs}
        filename = dir_name+'/neg'+str(file_ind)+'.txt'
        with open(filename, 'w') as f:
            f.write(json.dumps(info))

        # Attack and save the positive sentence attack
        sentence, updated_sentence, original_logits, updated_logits = attack_sentence(pos_sentence, pos_label, model, handler, criterion, tokenizer, max_syn=max_syn, N=N)
        original_probs = softmax(original_logits).tolist()
        updated_probs = softmax(updated_logits).tolist()
        info = {"sentence":sentence, "updated sentence":updated_sentence, "true label":pos_label, "original prob":original_probs, "updated prob":updated_probs}
        filename = dir_name+'/pos'+str(file_ind)+'.txt'
        with open(filename, 'w') as f:
            f.write(json.dumps(info))
