'''
Use training data to get PCA basis of CLS embedding (layer 12)
Take first 2 axes (or more)
Plot authentic and attacked plots, with separate colours per true class
'''

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from pca_tools import get_covariance_matrix, get_e_v
from models import BertSequenceClassifier
from transformers import BertTokenizer
from layer_handler import Bert_Layer_Handler
from tools import AverageMeter
import sys
import os
import argparse
import matplotlib.pyplot as plt
from data_prep_sentences import get_test
from data_prep_tensors import get_train
from pca_bert_layer import get_layer_embedding
import json


def get_pca_principal_components(eigenvectors, correction_mean, X):
    '''
    Returns components in two most principal directions
    Dim 0 of X should be the batch dimension
    '''
    comps = []
    with torch.no_grad():
        # Correct by pre-calculated authentic data mean
        X = X - correction_mean.repeat(X.size(0), 1)

        for i in range(2):
            v = eigenvectors[i]
            comp = torch.einsum('bi,i->b', X, v) # project to pca axis
            comps.append(comp.tolist())
    return comps[0], comps[1]

def get_sentence(fname):
    failed = False
    try:
        with open(fname, 'r') as f:
            item = json.load(f)
    except:
        print("Failed to load", fname)
        failed = True
    if not failed:
        original_prob = item['original prob']
        updated_prob = item['updated prob']
        original_pred = original_prob.index(max(original_prob))
        updated_pred = updated_prob.index(max(updated_prob))
        label = int(item['true label'])
        if (original_pred == label) and (updated_pred != original_pred):
            original = item['sentence']
            attack = item['updated sentence']
        else:
            return None, None
    else:
        return None, None
    return original, attack

def load_test_adapted_data_sentences(base_dir, num_test):
    '''
    Excludes data points with incorrect original predictions
    '''
    original_list_neg = []
    original_list_pos = []
    attack_list_neg = [] # Was originally negative
    attack_list_pos = [] # Was originally positive
    for i in range(num_test):
        fname = base_dir + '/neg'+str(i)+'.txt'
        original, attack = get_sentence(fname)
        if original is not None:
            original_list_neg.append(original)
            attack_list_neg.append(attack)

        fname = base_dir + '/pos'+str(i)+'.txt'
        original, attack = get_sentence(fname)
        if original is not None:
            original_list_pos.append(original)
            attack_list_pos.append(attack)

    return original_list_neg, original_list_pos, attack_list_neg, attack_list_pos

def plot_comparison(original_neg1, original_neg2, original_pos1, original_pos2, attack_neg1, attack_neg2, attack_pos1, attack_pos2, filename):

    plt.plot(original_neg1, original_neg2, marker='x', linestyle='None', label='Original Negative')
    plt.plot(original_pos1, original_pos2, marker='x', linestyle='None', label='Original Positive')
    plt.plot(attack_neg1, attack_neg2, marker='o', linestyle='None', label='Attack neg->pos')
    plt.plot(attack_pos1, attack_pos2, marker='o', linestyle='None', label='Attack pos->neg')
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.savefig(filename)
    plt.clf()


if __name__ == '__main__':

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('MODEL', type=str, help='trained .th model')
    commandLineParser.add_argument('TRAIN_DIR', type=str, help='training data base directory')
    commandLineParser.add_argument('TEST_DIR', type=str, help='attacked test data base directory')
    commandLineParser.add_argument('--layer_num', type=int, default=1, help="BERT layer to perturb")
    commandLineParser.add_argument('--num_points_train', type=int, default=25000, help="number of data points to use train")
    commandLineParser.add_argument('--num_points_test', type=int, default=12500, help="number of pairs data points to use test")
    commandLineParser.add_argument('--N', type=int, default=10, help="Num word substitutions used in attack")

    args = commandLineParser.parse_args()
    model_path = args.MODEL
    train_base_dir = args.TRAIN_DIR
    test_base_dir = args.TEST_DIR
    layer_num = args.layer_num
    num_points_train = args.num_points_train
    num_points_test = args.num_points_test
    N = args.N

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/pca_component_comparison_plot.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # Load the model
    model = BertSequenceClassifier()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    # Create model handler
    handler = Bert_Layer_Handler(model, layer_num=layer_num)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Use training data to get eigenvector basis of CLS token at correct layer
    input_ids, mask, _ = get_train('bert', train_base_dir)
    input_ids = input_ids[:num_points_train]
    mask = mask[:num_points_train]
    with torch.no_grad():
        layer_embeddings = handler.get_layern_outputs(input_ids, mask)
        CLS_embeddings = layer_embeddings[:,0,:].squeeze()
        correction_mean = torch.mean(CLS_embeddings, dim=0)
        cov = get_covariance_matrix(CLS_embeddings)
        e, v = get_e_v(cov)

    # Load the test data
    original_list_neg, original_list_pos, attack_list_neg, attack_list_pos = load_test_adapted_data_sentences(test_base_dir, num_points_test)

    # Get all points to plot
    embeddings = get_layer_embedding(original_list_neg, handler, tokenizer)
    original_neg1, original_neg2 = get_pca_principal_components(v, correction_mean, embeddings)

    embeddings = get_layer_embedding(original_list_pos, handler, tokenizer)
    original_pos1, original_pos2 = get_pca_principal_components(v, correction_mean, embeddings)

    embeddings = get_layer_embedding(attack_list_neg, handler, tokenizer)
    attack_neg1, attack_neg2 = get_pca_principal_components(v, correction_mean, embeddings)

    embeddings = get_layer_embedding(attack_list_pos, handler, tokenizer)
    attack_pos1, attack_pos2 = get_pca_principal_components(v, correction_mean, embeddings)

    # plot all the data
    filename = 'comparison_plot_layer'+str(layer_num)+"_N"+str(N)+".png"
    plot_comparison(original_neg1, original_neg2, original_pos1, original_pos2, attack_neg1, attack_neg2, attack_pos1, attack_pos2, filename)
