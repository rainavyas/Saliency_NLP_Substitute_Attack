'''
Select N words randomly to substitute and then find best of K synonyms to replace
with to maximise loss function
'''

import torch
import torch.nn as nn
import nltk
from nltk.corpus import wordnet as wn
from models import BertSequenceClassifier
from data_prep_sentences import get_test
import json
from transformers import BertTokenizer
import sys
import os
import argparse
from collections import OrderedDict
import random

def attack_sentence(sentence, label, model, criterion, tokenizer, max_syn=5, N=1):
    '''
    Selects N words randomly to substitute
    Finds best synoynm to replace with

    (sentence, label, model, handler, criterion, tokenizer, max_syn=5, N=1):
    '''
    model.eval()
    encoded_inputs = tokenizer([sentence], padding=True, truncation=True, return_tensors="pt")
    ids = encoded_inputs['input_ids'].squeeze()
    mask = encoded_inputs['attention_mask']

    # Randomly select indices
    inds = random.sample(range(1, ids.size(0)-1), max_syn)

    for i, ind in enumerate(inds):
        target_id = ids[ind]
        word_token = tokenizer.convert_ids_to_tokens(target_id.item())

        synonyms = []
        for syn in wn.synsets(word_token):
            for lemma in syn.lemmas():
                synonyms.append(lemma.name())
        if len(synonyms)==0:
            # print("No synonyms for ", word_token)
            updated_logits = model(torch.unsqueeze(ids, dim=0), mask).squeeze()
            if i==0:
                original_logits = updated_logits.clone()
            continue

        # Remove duplicates
        synonyms = list(OrderedDict.fromkeys(synonyms))

        if len(synonyms) > max_syn+1:
            synonyms = synonyms[:max_syn+1]

        best = (target_id, 0) # (id, loss)
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
    with open('CMDs/random_attack.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    nltk.download('wordnet')

    # Load the model
    model = BertSequenceClassifier()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    criterion = nn.CrossEntropyLoss()
    softmax = nn.Softmax(dim=0)

    # Create directory to save files in
    dir_name = 'Random_Attacked_Data_N'+str(N)
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
        sentence, updated_sentence, original_logits, updated_logits = attack_sentence(neg_sentence, neg_label, model, criterion, tokenizer, max_syn=max_syn, N=N)
        original_probs = softmax(original_logits).tolist()
        updated_probs = softmax(updated_logits).tolist()
        info = {"sentence":sentence, "updated sentence":updated_sentence, "true label":neg_label, "original prob":original_probs, "updated prob":updated_probs}
        filename = dir_name+'/neg'+str(file_ind)+'.txt'
        with open(filename, 'w') as f:
            f.write(json.dumps(info))

        # Attack and save the positive sentence attack
        sentence, updated_sentence, original_logits, updated_logits = attack_sentence(pos_sentence, pos_label, model, criterion, tokenizer, max_syn=max_syn, N=N)
        original_probs = softmax(original_logits).tolist()
        updated_probs = softmax(updated_logits).tolist()
        info = {"sentence":sentence, "updated sentence":updated_sentence, "true label":pos_label, "original prob":original_probs, "updated prob":updated_probs}
        filename = dir_name+'/pos'+str(file_ind)+'.txt'
        with open(filename, 'w') as f:
            f.write(json.dumps(info))
