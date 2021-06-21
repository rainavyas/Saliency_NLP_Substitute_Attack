'''
Generate precision-recall curve for linear adversarial attack classifier
'''

import torch
import torch.nn as nn
import sys
import os
import argparse
from tools import get_default_device
import numpy as np
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from pca_component_comparison_plot import load_test_adapted_data_sentences
from models import BertSequenceClassifier
from transformers import BertTokenizer
from layer_handler import Bert_Layer_Handler
from linear_pca_classifier import batched_get_layer_embedding, get_pca_principal_components, LayerClassifier




if __name__ == '__main__':

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('MODEL', type=str, help='trained sentiment classifier .th model')
    commandLineParser.add_argument('DIR', type=str, help='attacked data base directory')
    commandLineParser.add_argument('MODEL_DETECTOR', type=str, help='trained adv attack detector')
    commandLineParser.add_argument('EIGENVECTORS', type=str, help='Learnt eigenvectors .pt file for PCA projection')
    commandLineParser.add_argument('CORRECTION_MEAN', type=str, help='Learnt correction mean.pt file for PCA projection')
    commandLineParser.add_argument('OUT_FILE', type=str, help='.png file to save plot to')
    commandLineParser.add_argument('--layer_num', type=int, default=12, help="Layer at which to use detector")
    commandLineParser.add_argument('--num_comps', type=int, default=100, help="Number of PCA components to use")
    commandLineParser.add_argument('--N', type=int, default=1, help="Number of words substituted")
    commandLineParser.add_argument('--cpu', type=str, default='no', help="force cpu use")
    
    args = commandLineParser.parse_args()
    model_path = args.MODEL
    base_dir = args.DIR
    detector_path = args.MODEL_DETECTOR
    eigenvectors_path = args.EIGENVECTORS
    correction_mean_path = args.CORRECTION_MEAN
    out_file = args.out_file
    layer_num = args.layer_num
    num_comps = args.num_comps
    N = args.N
    cpu_use = args.cpu

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/precision_recall_linear_classifier.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # Get device
    if cpu_use == 'yes':
        device = torch.device('cpu')
    else:
        device = get_default_device()

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

    # Create model handler for PCA layer detection check
    handler_for_pca = Bert_Layer_Handler(model, layer_num=layer_num)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Load the test data
    original_list_neg, original_list_pos, attack_list_neg, attack_list_pos = load_test_adapted_data_sentences(base_dir, num_points_test)

    # Prepare input tensors (mapped to pca components)
    embeddings = batched_get_layer_embedding(original_list_neg, handler, tokenizer, device)
    original_negs = get_pca_principal_components(v, correction_mean, embeddings, num_comps, start)

    embeddings = batched_get_layer_embedding(original_list_pos, handler, tokenizer, device)
    original_poss = get_pca_principal_components(v, correction_mean, embeddings, num_comps, start)

    embeddings = batched_get_layer_embedding(attack_list_neg, handler, tokenizer, device)
    attack_negs = get_pca_principal_components(v, correction_mean, embeddings, num_comps, start)

    embeddings = batched_get_layer_embedding(attack_list_pos, handler, tokenizer, device)
    attack_poss = get_pca_principal_components(v, correction_mean, embeddings, num_comps, start)

    with torch.no_grad():
        original = torch.cat((original_negs, original_poss))
        attack = torch.cat((attack_negs, attack_poss))
    
    labels = np.asarray([0]*original.size(0) + [1]*attack.size(0))
    X = torch.cat((original, attack))

    # get predicted logits of being adversarial attack
    with torch.no_grad():
        logits = detector(X)
        adv_logits = logits[:,1].squeeze().cpu().detach().numpy()
    
    # get precision recall values and highest F1 score
    precision, recall, _ = precision_recall_curve(labels, adv_logits)

    # plot all the data
