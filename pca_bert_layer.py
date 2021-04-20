'''
Use training data to perform pca decomposition in BERT layer
embedding.
Compare evaluation data authentic and attacked sentences in
the defined PCA space.
'''

import torch
import torch.nn as nn
from torch.nn import CosineSimilarity
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
import json

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
        pred = original_prob.index(max(original_prob))
        label = int(item['true label'])
        if pred == label:
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
    original_list = []
    attack_list = []
    for i in range(num_test):
        fname = base_dir + '/neg'+str(i)+'.txt'
        original, attack = get_sentence(fname)
        if original is not None:
            original_list.append(original)
            attack_list.append(attack)

        fname = base_dir + '/pos'+str(i)+'.txt'
        original, attack = get_sentence(fname)
        if original is not None
            original_list.append(original)
            attack_list.append(attack)

    return original_list, attack_list

def get_eigenvector_decomposition_magnitude(eigenvectors, eigenvalues, X, correction_mean):
    '''
    Mean average of magnitude of cosine distance to each eigenevector
    '''
    cos_dists = []
    whitened_cos_dists = []
    ranks = []

    with torch.no_grad():
        # Correct by pre-calculated authentic data mean
        X = X - correction_mean.repeat(X.size(0), 1)

        cos = CosineSimilarity(dim=1)
        for i in range(eigenvectors.size(0)):
            ranks.append(i)
            v = eigenvectors[i]
            v_repeat = v.repeat(X.size(0), 1)
            abs_cos_dist = torch.abs(cos(X, v_repeat))
            whitened_abs_cos_dist = abs_cos_dist/(eigenvalues[i]**0.5)
            cos_dists.append(torch.mean(abs_cos_dist).item())
            whitened_cos_dists.append(torch.mean(whitened_abs_cos_dist).item())

    return ranks, cos_dists, whitened_cos_dists

def get_layer_embedding(sentences_list, handler, tokenizer):
    '''
    Return the CLS token embedding at layer specified in handler
    [batch_size x 768]
    '''
    encoded_inputs = tokenizer(sentences_list, padding=True, truncation=True, return_tensors="pt")
    ids = encoded_inputs['input_ids']
    mask = encoded_inputs['attention_mask']

    with torch.no_grad():
        layer_embeddings = handler.get_layern_outputs(ids, mask)
        CLS_embeddings = layer_embeddings[:,0,:].squeeze()
    return CLS_embeddings

def plot_decomposition(ranks, cos_dists_auth, cos_dists_attack, filename):

    plt.plot(ranks, cos_dists_attack, label="Attacked")
    plt.plot(ranks, cos_dists_auth, label="Original")
    plt.xlabel("Eigenvalue Rank")
    plt.ylabel("Average Absolute Cosine Distance")
    plt.yscale('log')
    plt.legend()
    plt.savefig(filename)
    plt.clf()

def plot_pca_whitened_decomposition(ranks, cos_dists_auth, cos_dists_attack, filename, log_it=False):

    plt.plot(ranks, cos_dists_attack, label="Attacked")
    plt.plot(ranks, cos_dists_auth, label="Original")
    plt.xlabel("Eigenvalue Rank")
    plt.ylabel("Whitened Average Absolute Cosine Distance")
    if log_it:
        plt.yscale('log')
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
    with open('CMDs/pca_bert_layer.cmd', 'a') as f:
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
    original_list, attack_list = load_test_adapted_data_sentences(test_base_dir, num_points_test)

    # Get authentic pca decomposition
    embeddings = get_layer_embedding(original_list, handler, tokenizer)
    ranks, cos_dists_auth, whitened_cos_dists_auth = get_eigenvector_decomposition_magnitude(v, e, embeddings, correction_mean)

    # Get attacked pca decomposition
    embeddings = get_layer_embedding(attack_list, handler, tokenizer)
    ranks, cos_dists_attack, whitened_cos_dists_attack = get_eigenvector_decomposition_magnitude(v, e, embeddings, correction_mean)

    # Plot the data
    filename = "pca_decomp_layer"+str(layer_num)+"_N"+str(N)+".png"
    plot_decomposition(ranks, cos_dists_auth, cos_dists_attack, filename)
    filename = "pca_whitened_layer"+str(layer_num)+"_N"+str(N)+".png"
    plot_pca_whitened_decomposition(ranks, whitened_cos_dists_auth, whitened_cos_dists_attack, filename)
    filename = "log_pca_whitened_layer"+str(layer_num)+"_N"+str(N)+".png"
    plot_pca_whitened_decomposition(ranks, whitened_cos_dists_auth, whitened_cos_dists_attack, filename, log_it=True)
