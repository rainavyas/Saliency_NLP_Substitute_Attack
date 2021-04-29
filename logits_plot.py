'''
Visualize the output logits (not normalised)
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import sys
import os
import argparse
from models import BertSequenceClassifier
from transformers import BertTokenizer
from pca_component_comparison_plot import load_test_adapted_data_sentences


def get_logits(sentences_list, model, tokenizer):
    encoded_inputs = tokenizer(sentences_list, padding=True, truncation=True, return_tensors="pt")
    ids = encoded_inputs['input_ids']
    mask = encoded_inputs['attention_mask']

    with torch.no_grad():
        logits = model(ids, mask)
    return logits

if __name__ == '__main__':

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('MODEL', type=str, help='trained .th model')
    commandLineParser.add_argument('TEST_DIR', type=str, help='attacked test data base directory')
    commandLineParser.add_argument('--num_points_test', type=int, default=12500, help="number of pairs data points to use test")
    commandLineParser.add_argument('--N', type=int, default=10, help="Num word substitutions used in attack")

    args = commandLineParser.parse_args()
    model_path = args.MODEL
    test_base_dir = args.TEST_DIR
    num_points_test = args.num_points_test
    N = args.N


    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/logits_plot.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # Load the model
    model = BertSequenceClassifier()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Load test data
    original_list_neg, original_list_pos, attack_list_neg, attack_list_pos = load_test_adapted_data_sentences(test_base_dir, num_points_test)

    # Get logit outputs and put in dataframe
    logits_original_neg = get_logits(original_list_neg, model, tokenizer)
    logits_original_pos = get_logits(original_list_pos, model, tokenizer)
    logits_attack_neg = get_logits(attack_list_neg, model, tokenizer)
    logits_attack_pos = get_logits(attack_list_pos, model, tokenizer)

    X = torch.cat([logits_original_neg, logits_original_pos, logits_attack_neg, logits_attack_pos])
    feat_cols = [str(i) for i in range(X.size(1))]
    df = pd.DataFrame(X, columns=feat_cols)
    df['label'] = ['Original Negative']*len(original_list_neg) + ['Original Positive']*len(original_list_pos) + ['Attack neg->pos']*len(attack_list_neg) + ['Attack pos->neg']*len(attack_list_pos)

    # Plot the data
    df['logit-one'] = X[:,0]
    df['logit-two'] = X[:,1]

    plt.figure(figsize=(16,10))
    sns.scatterplot(
        x="logit-one", y="logit-two",
        hue="label",
        palette=sns.color_palette("bright", 4),
        data=df,
        legend="full",
        alpha=0.5
    )
    filename = 'logit_plot_N'+str(N)+".png"
    plt.savefig(filename)
