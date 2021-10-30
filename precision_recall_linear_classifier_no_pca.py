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
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
from pca_component_comparison_plot import load_test_adapted_data_sentences
from models import BertSequenceClassifier
from transformers import BertTokenizer
from layer_handler import Bert_Layer_Handler
from linear_pca_classifier import batched_get_layer_embedding, LayerClassifier


def get_best_f_score(precisions, recalls, beta=1.0):
    f_scores = (1+beta**2)*((precisions*recalls)/((precisions*(beta**2))+recalls))
    ind = np.argmax(f_scores)
    return precisions[ind], recalls[ind], f_scores[ind]

if __name__ == '__main__':

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('MODEL', type=str, help='trained sentiment classifier .th model')
    commandLineParser.add_argument('DIR', type=str, help='attacked data base directory')
    commandLineParser.add_argument('MODEL_DETECTOR', type=str, help='trained adv attack detector')
    commandLineParser.add_argument('OUT_FILE', type=str, help='.png file to save plot to')
    commandLineParser.add_argument('--layer_num', type=int, default=12, help="Layer at which to use detector")
    commandLineParser.add_argument('--N', type=int, default=1, help="Number of words substituted")
    commandLineParser.add_argument('--cpu', type=str, default='no', help="force cpu use")
    commandLineParser.add_argument('--num_points_test', type=int, default=12500, help="number of pairs data points to use test")
    
    args = commandLineParser.parse_args()
    model_path = args.MODEL
    base_dir = args.DIR
    detector_path = args.MODEL_DETECTOR
    out_file = args.OUT_FILE
    layer_num = args.layer_num
    N = args.N
    cpu_use = args.cpu
    num_points_test = args.num_points_test

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


    # Create model handler for detection check
    handler = Bert_Layer_Handler(model, layer_num=layer_num)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Load the test data
    original_list_neg, original_list_pos, attack_list_neg, attack_list_pos = load_test_adapted_data_sentences(base_dir, num_points_test)
    print("Loaded data")

    # Prepare input tensors (mapped to pca components)
    original_negs = batched_get_layer_embedding(original_list_neg, handler, tokenizer, device)
    original_poss = batched_get_layer_embedding(original_list_pos, handler, tokenizer, device)
    attack_negs = batched_get_layer_embedding(attack_list_neg, handler, tokenizer, device)
    attack_poss = batched_get_layer_embedding(attack_list_pos, handler, tokenizer, device)
    print("Got embeddings")

    with torch.no_grad():
        original = torch.cat((original_negs, original_poss))
        attack = torch.cat((attack_negs, attack_poss))
    
    labels = np.asarray([0]*original.size(0) + [1]*attack.size(0))
    X = torch.cat((original, attack))

    # get predicted logits of being adversarial attack
    with torch.no_grad():
        logits = detector(X)
        s = nn.Softmax(dim=1)
        probs = s(logits)
        adv_probs = probs[:,1].squeeze().cpu().detach().numpy()
    
    print("Got prediction probs")
    # get precision recall values and highest F1 score (with associated prec and rec)
    precision, recall, _ = precision_recall_curve(labels, adv_probs)
    best_precision, best_recall, best_f1 =  get_best_f_score(precision, recall)

    # plot all the data
    plt.plot(recall, precision, 'r-')
    plt.plot(best_recall,best_precision,'bo')
    plt.annotate(F"F1={best_f1:.2f}", (best_recall,best_precision))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.savefig(out_file)
