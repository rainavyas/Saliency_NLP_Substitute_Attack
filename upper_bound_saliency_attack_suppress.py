'''
Identical to upper_bound_saliency_attack.py
but also attempts to choose synonyms that avoid detection using a linear
adv attack classifier that has been trained to detect adversarial attacks at
a specific layer
'''
import torch
import torch.nn as nn
import nltk
from nltk.corpus import wordnet as wn
from layer_handler import Bert_Layer_Handler
from models import BertSequenceClassifier
from linear_pca_classifier import LayerClassifier
from data_prep_sentences import get_test
import json
from transformers import BertTokenizer
import sys
import os
import argparse
from collections import OrderedDict
from upper_bound_saliency_attack import get_token_saliencies

def is_suppressed(ids, mask, detector, handler_for_pca, eigenvectors, correction_mean, num_comps):
    '''
    Checks to see if linear adv attack detector fails to detect
    the adv attack. If this is the case, returns True, to indicate the residual
    components have been suppressed in the attack, making it hard to detect.

    ids is for a single sentence squeezed; mask assumed to already be unsqueezed.
    '''
    with torch.no_grad():
        embeddings = handler_for_pca.get_layern_outputs(torch.unsqueeze(ids, dim=0), mask)
        CLS_embedding = embeddings[:,0,:].squeeze()
        CLS_embedding = CLS_embedding - correction_mean
        v_map = eigenvectors[:num_comps]
        CLS_pca_projected = torch.einsum('i,ji->j', CLS_embedding, v_map)
        logits = detector(torch.unsqueeze(CLS_pca_projected, dim=0)).squeeze()
    return (logits[0]>logits[1])


def attack_sentence(sentence, label, model, handler_for_saliency, handler_for_pca, detector, eigenvectors, correction_mean, num_comps, criterion, tokenizer, max_syn=5, N=1):
    '''
    Identifies the N most salient words (by upper bound saliency)
    Finds synonyms for these words using WordNet
    Selects the best synonym to replace with based on Forward Pass to maximise
    the loss function, sequentially starting with most salient word

    Returns the original_sentence, updated_sentence, original_logits, updated_logits
    '''
    model.eval()

    token_saliencies = get_token_saliencies(sentence, label, handler_for_saliency, criterion, tokenizer)
    token_saliencies[0] = 0
    token_saliencies[-1] = 0

    inds = torch.argsort(token_saliencies, descending=True)
    if len(inds) > N:
        inds = inds[:N]

    encoded_inputs = tokenizer([sentence], padding=True, truncation=True, return_tensors="pt")
    ids = encoded_inputs['input_ids'].squeeze()
    mask = encoded_inputs['attention_mask']

    assert len(token_saliencies) == len(ids), "tokens and saliencies mismatch"

    for i, ind in enumerate(inds):
        target_id = ids[ind]
        word_token = tokenizer.convert_ids_to_tokens(target_id.item())

        synonyms = []
        for syn in wn.synsets(word_token):
            for lemma in syn.lemmas():
                synonyms.append(lemma.name())
        updated_logits = model(torch.unsqueeze(ids, dim=0), mask).squeeze()
        if len(synonyms)==0:
            # print("No synonyms for ", word_token)
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
            if loss > best[1] and is_suppressed(ids, mask, detector, handler_for_pca, eigenvectors, correction_mean, num_comps):
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
    commandLineParser.add_argument('MODEL', type=str, help='trained sentiment classifier .th model')
    commandLineParser.add_argument('DIR', type=str, help='data base directory')
    commandLineParser.add_argument('MODEL_DETECTOR', type=str, help='trained adv attack detector')
    commandLineParser.add_argument('EIGENVECTORS', type=str, help='Learnt eigenvectors .pt file for PCA projection')
    commandLineParser.add_argument('CORRECTION_MEAN', type=str, help='Learnt correction mean.pt file for PCA projection')
    commandLineParser.add_argument('--layer_num', type=int, default=12, help="Layer at which to use detector")
    commandLineParser.add_argument('--num_comps', type=int, default=100, help="Number of PCA components to use")
    commandLineParser.add_argument('--max_syn', type=int, default=5, help="Number of synonyms to search")
    commandLineParser.add_argument('--N', type=int, default=1, help="Number of words to substitute")
    commandLineParser.add_argument('--start_ind', type=int, default=0, help="start IMDB file index for both pos and neg review")
    commandLineParser.add_argument('--end_ind', type=int, default=100, help=" end IMDB file index for both pos and neg review")

    args = commandLineParser.parse_args()
    model_path = args.MODEL
    base_dir = args.DIR
    detector_path = args.MODEL_DETECTOR
    eigenvectors_path = args.EIGENVECTORS
    correction_mean_path = args.CORRECTION_MEAN
    layer_num = args.layer_num
    num_comps = args.num_comps
    max_syn = args.max_syn
    N = args.N
    start_ind = args.start_ind
    end_ind = args.end_ind

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/upper_bound_saliency_attack_suppress.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    nltk.download('wordnet')

    # Load the Sentiment Classifier model
    model = BertSequenceClassifier()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    # Load the Adv Attack Detector model
    detector = LayerClassifier(num_comps)
    detector.load_state_dict(torch.load(detector_path, map_location=torch.device('cpu')))
    detector.eval()

    # Load the eigenvectors for PCA decomposition and the correction mean
    eigenvectors = torch.load(eigenvectors_path)
    correction_mean = torch.load(correction_mean_path)

    # Create model handler for saliency
    handler_for_saliency = Bert_Layer_Handler(model, layer_num=0)

    # Create model handler for PCA layer detection check
    handler_for_pca = Bert_Layer_Handler(model, layer_num=layer_num)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    criterion = nn.CrossEntropyLoss()
    softmax = nn.Softmax(dim=0)

    # Create directory to save files in
    dir_name = 'Suppressed_Attacked_Data_N'+str(N)
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
        sentence, updated_sentence, original_logits, updated_logits = attack_sentence(neg_sentence, neg_label, model, handler_for_saliency, handler_for_pca, detector, eigenvectors, correction_mean, num_comps, criterion, tokenizer, max_syn=max_syn, N=N)
        original_probs = softmax(original_logits).tolist()
        updated_probs = softmax(updated_logits).tolist()
        info = {"sentence":sentence, "updated sentence":updated_sentence, "true label":neg_label, "original prob":original_probs, "updated prob":updated_probs}
        filename = dir_name+'/neg'+str(file_ind)+'.txt'
        with open(filename, 'w') as f:
            f.write(json.dumps(info))

        # Attack and save the positive sentence attack
        sentence, updated_sentence, original_logits, updated_logits = attack_sentence(pos_sentence, pos_label, model, handler_for_saliency, handler_for_pca, detector, eigenvectors, correction_mean, num_comps, criterion, tokenizer, max_syn=max_syn, N=N)
        original_probs = softmax(original_logits).tolist()
        updated_probs = softmax(updated_logits).tolist()
        info = {"sentence":sentence, "updated sentence":updated_sentence, "true label":pos_label, "original prob":original_probs, "updated prob":updated_probs}
        filename = dir_name+'/pos'+str(file_ind)+'.txt'
        with open(filename, 'w') as f:
            f.write(json.dumps(info))