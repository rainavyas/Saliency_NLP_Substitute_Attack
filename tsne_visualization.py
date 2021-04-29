'''
Visualize the CLS token embedding of layer 12 using T-SNE

Original Paper:
https://jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf
'''
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import sys
import os
import argparse
from pca_bert_layer import get_layer_embedding
from models import BertSequenceClassifier
from transformers import BertTokenizer
from layer_handler import Bert_Layer_Handler
from pca_component_comparison_plot import load_test_adapted_data_sentences


if __name__ == '__main__':

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('MODEL', type=str, help='trained .th model')
    commandLineParser.add_argument('TEST_DIR', type=str, help='attacked test data base directory')
    commandLineParser.add_argument('--layer_num', type=int, default=1, help="BERT layer to perturb")
    commandLineParser.add_argument('--num_points_test', type=int, default=12500, help="number of pairs data points to use test")
    commandLineParser.add_argument('--N', type=int, default=10, help="Num word substitutions used in attack")

    args = commandLineParser.parse_args()
    model_path = args.MODEL
    test_base_dir = args.TEST_DIR
    layer_num = args.layer_num
    num_points_test = args.num_points_test
    N = args.N

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/tsne_visualization.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # Load the model
    model = BertSequenceClassifier()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    # Create model handler
    handler = Bert_Layer_Handler(model, layer_num=layer_num)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Load the test data, convert to embeddings and place in dataframe
    original_list_neg, original_list_pos, attack_list_neg, attack_list_pos = load_test_adapted_data_sentences(test_base_dir, num_points_test)
    embeddings_original_neg = get_layer_embedding(original_list_neg, handler, tokenizer)
    embeddings_original_pos = get_layer_embedding(original_list_pos, handler, tokenizer)
    embeddings_attack_neg = get_layer_embedding(attack_list_neg, handler, tokenizer)
    embeddings_attack_pos = get_layer_embedding(attack_list_pos, handler, tokenizer)
    X = torch.cat([embeddings_original_neg, embeddings_original_pos, embeddings_attack_neg, embeddings_attack_pos])

    feat_cols = [str(i) for i in range(X.size(1))]
    df = pd.DataFrame(X, columns=feat_cols)
    df['label'] = ['Original Negative']*len(original_list_neg) + ['Original Positive']*len(original_list_pos) + ['Attack neg->pos']*len(attack_list_neg) + ['Attack pos->neg']*len(attack_list_pos)

    # Perform t-SNE
    data = df[feat_cols].values
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(data)

    # Plot the data

    df['tsne-2d-one'] = tsne_results[:,0]
    df['tsne-2d-two'] = tsne_results[:,1]

    plt.figure(figsize=(16,10))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="label",
        palette=sns.color_palette("bright", 4),
        data=df,
        legend="full",
        alpha=0.3
    )
    filename = 'tsne_layer'+str(layer_num)+"_N"+str(N)+".png"
    plt.savefig(filename)
