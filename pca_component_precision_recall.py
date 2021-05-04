'''
Use a particular pca component to generate pr curve for detecting
adversarial samples
Analysis at a particular CLS token at  particular layer
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
from pca_component_comparison_plot import load_test_adapted_data_sentences



def get_pca_component(eigenvectors, correction_mean, X, comp_num):
    '''
    Returns components of X (dim 0 is batch) in pca component comp
    '''
    with torch.no_grad():
        # Correct by pre-calculated authentic data mean
        X = X - correction_mean.repeat(X.size(0), 1)
        v = eigenvectors[comp_num]
        comps = torch.einsum('bi,i->b', X, v).tolist() # project to pca axis
    return comps

def pr(list_auth, list_attack, start, stop, num):
    precision = []
    recall = []
    f05 = (0, 0, 0) # (F0.5, prec, recall)

    for thresh in np.linspace(start, stop, num):
        TP = 0 # true positive
        FP = 0 # false positive
        T = len(list_attack) # Number of True Attack examples

        for val in list_auth:
            if val < thresh:
                FP += 1
        for val in list_attack:
            if val < thresh:
                TP += 1

        if (TP+FP > 0) and TP>0:
            prec = TP/(TP+FP)
            rec = TP/T

            curr_f05 = (1.25*prec*rec)/((0.25*prec) + rec)
            if curr_f05 > f05[0]:
                f05 = (curr_f05, prec, rec)

            precision.append(prec)
            recall.append(rec)

    return precision, recall, f05

def plot_precision_recall(precision, recall, f05_data, filename):
    plt.plot(recall, precision)
    plt.plot(f05_data[2], f05_data[1], "s:k", label='F0.5='+str(round(f05_data[0], 3)))
    plt.xlabel("Recall")
    plt.ylabel("Precision")
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
    commandLineParser.add_argument('--comp_num', type=int, default=1, help="PCA component to use")

    args = commandLineParser.parse_args()
    model_path = args.MODEL
    train_base_dir = args.TRAIN_DIR
    test_base_dir = args.TEST_DIR
    layer_num = args.layer_num
    num_points_train = args.num_points_train
    num_points_test = args.num_points_test
    N = args.N
    comp_num = args.comp_num

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/pca_component_precision_recall.cmd', 'a') as f:
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

    # Get all the components to test
    embeddings = get_layer_embedding(original_list_neg, handler, tokenizer)
    original_negs = get_pca_component(v, correction_mean, embeddings, comp_num)

    embeddings = get_layer_embedding(original_list_pos, handler, tokenizer)
    original_poss = get_pca_component(v, correction_mean, embeddings, comp_num)

    embeddings = get_layer_embedding(attack_list_neg, handler, tokenizer)
    attack_negs = get_pca_component(v, correction_mean, embeddings, comp_num)

    embeddings = get_layer_embedding(attack_list_pos, handler, tokenizer)
    attack_poss = get_pca_component(v, correction_mean, embeddings, comp_num)

    original = original_negs + original_poss
    attacked = attack_negs + attack_poss

    # Plot precision and recall values
    precision, recall, f05 = pr(original, attacked, start=-8, stop=8, num=1000)
    filename = 'pr_layer'+str(layer_num)+'_N'+str(N)+'_comp'+str(comp_num)+'.png'
    plot_precision_recall(precision, recall, f05_data, filename)
