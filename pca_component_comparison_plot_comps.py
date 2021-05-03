'''
Same as pca_component_comparison_plot but plots all pairs of plots
for entered number of components
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
from pca_component_comparison_plot import load_test_adapted_data_sentences

def get_pca_principal_components(eigenvectors, correction_mean, X, num_comps):
    '''
    Returns components in num_comps most principal directions
    Dim 0 of X should be the batch dimension
    '''
    comps = []
    with torch.no_grad():
        # Correct by pre-calculated authentic data mean
        X = X - correction_mean.repeat(X.size(0), 1)

        for i in range(num_comps):
            v = eigenvectors[i]
            comp = torch.einsum('bi,i->b', X, v) # project to pca axis
            comps.append(comp.tolist())
    return comps[:num_comps]

def plot_comparison(original_negs, original_poss, attack_negs, attack_poss, filename, compx=0, compy=1):

    plt.plot(original_negs[compx], original_negs[compy], marker='x', linestyle='None', label='Original Negative')
    plt.plot(original_pos[compx], original_pos[compy], marker='x', linestyle='None', label='Original Positive')
    plt.plot(attack_negs[compx], attack_neg[compy], marker='o', linestyle='None', label='Attack neg->pos')
    plt.plot(attack_poss[compx], attack_pos[compy], marker='o', linestyle='None', label='Attack pos->neg')
    plt.xlabel('PCA'+str(compx))
    plt.ylabel('PCA'+str(compy))
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
    commandLineParser.add_argument('--num_comps', type=int, default=2, help="number of PCA components")

    args = commandLineParser.parse_args()
    model_path = args.MODEL
    train_base_dir = args.TRAIN_DIR
    test_base_dir = args.TEST_DIR
    layer_num = args.layer_num
    num_points_train = args.num_points_train
    num_points_test = args.num_points_test
    N = args.N
    num_comps = args.num_comps

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/pca_component_comparison_plot_comps.cmd', 'a') as f:
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
    indices = torch.randperm(len(input_ids))[:num_points_train]
    input_ids = input_ids[indices]
    mask = mask[indices]
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
    original_negs = get_pca_principal_components(v, correction_mean, embeddings)

    embeddings = get_layer_embedding(original_list_pos, handler, tokenizer)
    original_poss = get_pca_principal_components(v, correction_mean, embeddings)

    embeddings = get_layer_embedding(attack_list_neg, handler, tokenizer)
    attack_negs = get_pca_principal_components(v, correction_mean, embeddings)

    embeddings = get_layer_embedding(attack_list_pos, handler, tokenizer)
    attack_poss = get_pca_principal_components(v, correction_mean, embeddings)

    # plot all the data
    for i in range(num_comps):
        for j in range(i+1, num_comps):
            filename = 'pca_layer'+str(layer_num)+"_N"+str(N)+"_comp"+str(i)+"_vs_"+str(j)+".png"
            plot_comparison(original_negs, original_poss, attack_negs, attack_poss, filename, compx=i, compy=j)
