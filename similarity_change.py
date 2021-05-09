'''
Aim to analyze how the similarity between original and adversarial samples changes
from input layer to the output layer of the model.

Assess similarity using absolute distance between CLS token vectors
in the transformer layers.

This file returns the similarity score at each stage of the model
'''

from collections import OrderedDict
import torch
from models import BertSequenceClassifier
from transformers import BertTokenizer
from layer_handler import Bert_Layer_Handler
from pca_component_comparison_plot import load_test_adapted_data_sentences
from linear_pca_classifier import batched_get_layer_embedding
from tools import get_default_device
import sys
import os
import argparse

if __name__ == '__main__':

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('MODEL', type=str, help='trained .th model')
    commandLineParser.add_argument('TEST_DIR', type=str, help='attacked test data base directory')
    commandLineParser.add_argument('--max_layer_num', type=int, default=13, help="BERT layer to investigate")
    commandLineParser.add_argument('--num_points_test', type=int, default=12500, help="number of pairs data points to use test")
    commandLineParser.add_argument('--N', type=int, default=25, help="Num word substitutions used in attack")
    commandLineParser.add_argument('--cpu', type=str, default='no', help="force cpu use")

    args = commandLineParser.parse_args()
    model_path = args.MODEL
    test_base_dir = args.TEST_DIR
    max_layer_num = args.max_layer_num
    num_points_test = args.num_points_test
    N = args.N
    cpu_use = args.cpu

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/similarity_change.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # Get device
    if cpu_use == 'yes':
        device = torch.device('cpu')
    else:
        device = get_default_device()

    # Load the model
    model = BertSequenceClassifier()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.to(device)
    model.eval()

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Load the test data
    original_list_neg, original_list_pos, attack_list_neg, attack_list_pos = load_test_adapted_data_sentences(test_base_dir, num_points_test)
    original_list = original_list_neg + original_list_pos
    attack_list = attack_list_neg + attack_list_pos
    assert len(original_list)==len(attack_list), "Mismatched samples"

    # Create ordered dict to store similarity results
    layer_to_similarity = OrderedDict()

    for layer_num in range(1, max_layer_num+1):

        # Create model handler
        handler = Bert_Layer_Handler(model, layer_num=layer_num)

        # Obtain CLS token embeddings at layer
        original_embeddings = batched_get_layer_embedding(original_list, handler, tokenizer, device)
        attack_embeddings = batched_get_layer_embedding(attack_list, handler, tokenizer, device)

        # Calculate l2 distance between samples
        l2 = torch.norm((original_embeddings - attack_embeddings), dim=1)
        avg_l2 = torch.mean(l2)

        layer_to_similarity[layer_num] = avg_l2
        print(layer_to_similarity)
