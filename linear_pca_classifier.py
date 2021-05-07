'''
Use the PCA reduced dimensions CLS token at a specific layer as an
input into a simple linear classifier to distinguish between original and
adversarial samples
'''

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from pca_tools import get_covariance_matrix, get_e_v
from models import BertSequenceClassifier
from transformers import BertTokenizer
from layer_handler import Bert_Layer_Handler
from tools import AverageMeter, get_default_device, accuracy_topk
import sys
import os
import argparse
import matplotlib.pyplot as plt
from data_prep_sentences import get_test
from data_prep_tensors import get_train
from pca_bert_layer import get_layer_embedding
import json
from pca_component_comparison_plot import load_test_adapted_data_sentences

def train(train_loader, model, criterion, optimizer, epoch, device, print_freq=1):
    '''
    Run one train epoch
    '''
    losses = AverageMeter()
    accs = AverageMeter()

    # switch to train mode
    model.train()

    for i, (x, target) in enumerate(train_loader):

        x = x.to(device)
        target = target.to(device)

        # Forward pass
        logits = model(x)
        loss = criterion(logits, target)

        # Backward pass and update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        acc = accuracy_topk(logits.data, target)
        accs.update(acc.item(), x.size(0))
        losses.update(loss.item(), x.size(0))

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Accuracy {prec.val:.3f} ({prec.avg:.3f})'.format(
                      epoch, i, len(train_loader),
                      loss=losses, prec=accs))

def eval(val_loader, model, criterion, device):
    '''
    Run evaluation
    '''
    losses = AverageMeter()
    accs = AverageMeter()

    # switch to eval mode
    model.eval()

    with torch.no_grad():
        for i, (x, target) in enumerate(val_loader):

            x = x.to(device)
            target = target.to(device)

            # Forward pass
            logits = model(x)
            loss = criterion(logits, target)

            # measure accuracy and record loss
            acc = accuracy_topk(logits.data, target)
            accs.update(acc.item(), x.size(0))
            losses.update(loss.item(), x.size(0))

    print('Test\t Loss ({loss.avg:.4f})\t'
            'Accuracy ({prec.avg:.3f})\n'.format(
              loss=losses, prec=accs))

class LayerClassifier(nn.Module):
    '''
    Simple Linear classifier
    '''
    def __init__(self, dim, classes=2):
        super().__init__()
        self.layer = nn.Linear(dim, classes)
    def forward(self, X):
        return self.layer(X)

def get_pca_principal_components(eigenvectors, correction_mean, X, num_comps, start):
    '''
    Returns components in num_comps most principal directions
    Dim 0 of X should be the batch dimension
    '''
    with torch.no_grad():
        # Correct by pre-calculated authentic data mean
        X = X - correction_mean.repeat(X.size(0), 1)

        v_map = eigenvectors[start:start+num_comps]
        comps = torch.einsum('bi,ji->bj', X, v_map)
    return comps

if __name__ == '__main__':

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('MODEL', type=str, help='trained .th model')
    commandLineParser.add_argument('TRAIN_DIR', type=str, help='training data base directory')
    commandLineParser.add_argument('TEST_DIR', type=str, help='attacked test data base directory')
    commandLineParser.add_argument('--layer_num', type=int, default=1, help="BERT layer to investigate")
    commandLineParser.add_argument('--num_points_train', type=int, default=25000, help="number of data points to use train")
    commandLineParser.add_argument('--num_points_test', type=int, default=12500, help="number of pairs data points to use test")
    commandLineParser.add_argument('--N', type=int, default=25, help="Num word substitutions used in attack")
    commandLineParser.add_argument('--num_comps', type=int, default=10, help="number of PCA components")
    commandLineParser.add_argument('--start', type=int, default=0, help="start of PCA components")
    commandLineParser.add_argument('--B', type=int, default=100, help="Specify batch size")
    commandLineParser.add_argument('--epochs', type=int, default=3, help="Specify epochs")
    commandLineParser.add_argument('--lr', type=float, default=0.0001, help="Specify learning rate")
    commandLineParser.add_argument('--seed', type=int, default=1, help="Specify seed")

    args = commandLineParser.parse_args()
    model_path = args.MODEL
    train_base_dir = args.TRAIN_DIR
    test_base_dir = args.TEST_DIR
    layer_num = args.layer_num
    num_points_train = args.num_points_train
    num_points_test = args.num_points_test
    N = args.N
    num_comps = args.num_comps
    start = args.start
    batch_size = args.B
    epochs = args.epochs
    lr = args.lr
    seed = args.seed

    torch.manual_seed(seed)

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/linear_pca_classifier.cmd', 'a') as f:
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
        if layer_num == 13 or layer_num == 14:
            CLS_embeddings = layer_embeddings
        else:
            CLS_embeddings = layer_embeddings[:,0,:].squeeze()
        correction_mean = torch.mean(CLS_embeddings, dim=0)
        cov = get_covariance_matrix(CLS_embeddings)
        e, v = get_e_v(cov)

    # Load the test data
    original_list_neg, original_list_pos, attack_list_neg, attack_list_pos = load_test_adapted_data_sentences(test_base_dir, num_points_test)

    # Prepare input tensors (mapped to pca components)
    embeddings = get_layer_embedding(original_list_neg, handler, tokenizer)
    original_negs = get_pca_principal_components(v, correction_mean, embeddings, num_comps, start)

    embeddings = get_layer_embedding(original_list_pos, handler, tokenizer)
    original_poss = get_pca_principal_components(v, correction_mean, embeddings, num_comps, start)

    embeddings = get_layer_embedding(attack_list_neg, handler, tokenizer)
    attack_negs = get_pca_principal_components(v, correction_mean, embeddings, num_comps, start)

    embeddings = get_layer_embedding(attack_list_pos, handler, tokenizer)
    attack_poss = get_pca_principal_components(v, correction_mean, embeddings, num_comps, start)

    with torch.no_grad():
        original = torch.cat((original_negs, original_poss))
        attack = torch.cat((attack_negs, attack_poss))

    labels = torch.LongTensor([0]*original.size(0) + [1]*attack.size(0))
    X = torch.cat((original, attack))

    ds = TensorDataset(X, labels)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    # Get device
    device = get_default_device()

    # Model
    model = LayerClassifier(num_comps)
    model.to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Criterion
    criterion = nn.CrossEntropyLoss().to(device)

    # Train
    for epoch in range(epochs):

        # train for one epoch
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        train(dl, model, criterion, optimizer, epoch, device)

    # evaluate once trained
    eval(dl, model, criterion, device)
