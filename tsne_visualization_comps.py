'''
Same as tsne_visualization.py but also plots every pair of component
graphs
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

def plot_pair(x_name, y_name, filename):
    plt.figure(figsize=(16,10))
    sns.scatterplot(
        x=x_name, y=y_name,
        hue="label",
        palette=sns.color_palette("bright", 4),
        data=df,
        legend="full",
        alpha=0.5
    )
    plt.savefig(filename)


if __name__ == '__main__':

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('MODEL', type=str, help='trained .th model')
    commandLineParser.add_argument('TEST_DIR', type=str, help='attacked test data base directory')
    commandLineParser.add_argument('--layer_num', type=int, default=1, help="BERT layer to perturb")
    commandLineParser.add_argument('--num_points_test', type=int, default=12500, help="number of pairs data points to use test")
    commandLineParser.add_argument('--N', type=int, default=10, help="Num word substitutions used in attack")
    commandLineParser.add_argument('--iter', type=int, default=300, help="TSNE iterations")
    commandLineParser.add_argument('--comps', type=int, default=2, help="TSNE components")

    args = commandLineParser.parse_args()
    model_path = args.MODEL
    test_base_dir = args.TEST_DIR
    layer_num = args.layer_num
    num_points_test = args.num_points_test
    N = args.N
    iter = args.iter
    comps = args.comps

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/tsne_visualization_comps.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    np.random.seed(1)

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
    tsne = TSNE(n_components=comps, verbose=1, perplexity=40, n_iter=iter)
    tsne_results = tsne.fit_transform(data)

    # Plot the data
    for comp in range(comps):
        df['tsne'+str(comp)] = tsne_results[:,comp]

    for i in range(comps):
        for j in range(i+1, comps):
            x_name = 'tsne'+str(i)
            y_name = 'tsne'+str(j)
            filename = 'tsne_layer'+str(layer_num)+"_N"+str(N)+"_comp"+str(i)+"_vs_"+str(j)+".png"
            plot_pair(x_name, y_name, filename)
