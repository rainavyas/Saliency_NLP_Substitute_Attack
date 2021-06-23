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
import json
from pca_component_comparison_plot import load_test_adapted_data_sentences


def batched_get_layer_embedding(sentences_list, handler, tokenizer, device, bs=8):
    '''
    Performs the function of get_layer_embedding from pca_bert_layer
    in batches and allows gpu use
    '''
    encoded_inputs = tokenizer(sentences_list, padding=True, truncation=True, return_tensors="pt")
    ids = encoded_inputs['input_ids']
    mask = encoded_inputs['attention_mask']
    return batched_get_handler_embeddings(ids, mask, handler, device, bs=bs)


def batched_get_handler_embeddings(input_ids, mask, handler, device, bs=8):
    '''
    Input is a tensor of input ids and mask
    Returns tensor of CLS embeddings at the correct layer
    Does this in batches

    If layer_num = 13, use pooler output instead
    If layer_num = 14, use logits output instead
    '''
    CLS = []
    ds = TensorDataset(input_ids, mask)
    dl = DataLoader(ds, batch_size=bs)
    with torch.no_grad():
        for id, m in dl:
            id = id.to(device)
            m = m.to(device)
            layer_embeddings = handler.get_layern_outputs(id, m, device=device)
            if handler.layer_num == 13 or handler.layer_num == 14:
                CLS_embeddings = layer_embeddings
            else:
                CLS_embeddings = layer_embeddings[:,0,:].squeeze(dim=1)
            CLS.append(CLS_embeddings.cpu())
    embeddings = torch.cat(CLS)
    return embeddings

def train(train_loader, model, criterion, optimizer, epoch, device, out_file, print_freq=1):
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
            text = '\n Epoch: [{0}][{1}/{2}]\t Loss {loss.val:.4f} ({loss.avg:.4f})\t Accuracy {prec.val:.3f} ({prec.avg:.3f})'.format(epoch, i, len(train_loader), loss=losses, prec=accs)
            print(text)
            with open(out_file, 'a') as f:
                f.write(text)

def eval(val_loader, model, criterion, device, out_file):
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

    text ='\n Test\t Loss ({loss.avg:.4f})\t Accuracy ({prec.avg:.3f})\n'.format(loss=losses, prec=accs)
    print(text)
    with open(out_file, 'a') as f:
        f.write(text)

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
    commandLineParser.add_argument('OUT', type=str, help='file to print results to')
    commandLineParser.add_argument('CLASSIFIER_OUT', type=str, help='.th to save linear adv attack classifier to')
    commandLineParser.add_argument('PCA_OUT', type=str, help='.pt to save original PCA eigenvector directions to')
    commandLineParser.add_argument('PCA_MEAN_OUT', type=str, help='.pt to save PCA correction mean to')
    commandLineParser.add_argument('--layer_num', type=int, default=1, help="BERT layer to investigate")
    commandLineParser.add_argument('--num_points_train', type=int, default=25000, help="number of data points to use train")
    commandLineParser.add_argument('--num_points_test', type=int, default=12500, help="number of pairs data points to use test")
    commandLineParser.add_argument('--num_points_val', type=int, default=1200, help="number of test data points to use for validation")
    commandLineParser.add_argument('--N', type=int, default=25, help="Num word substitutions used in attack")
    commandLineParser.add_argument('--num_comps', type=int, default=10, help="number of PCA components")
    commandLineParser.add_argument('--start', type=int, default=0, help="start of PCA components")
    commandLineParser.add_argument('--B', type=int, default=100, help="Specify batch size")
    commandLineParser.add_argument('--epochs', type=int, default=3, help="Specify epochs")
    commandLineParser.add_argument('--lr', type=float, default=0.0001, help="Specify learning rate")
    commandLineParser.add_argument('--seed', type=int, default=1, help="Specify seed")
    commandLineParser.add_argument('--cpu', type=str, default='no', help="force cpu use")

    args = commandLineParser.parse_args()
    model_path = args.MODEL
    train_base_dir = args.TRAIN_DIR
    test_base_dir = args.TEST_DIR
    out_file = args.OUT
    classifier_out_file = args.CLASSIFIER_OUT
    pca_out_file = args.PCA_OUT
    pca_mean_out_file = args.PCA_MEAN_OUT
    layer_num = args.layer_num
    num_points_train = args.num_points_train
    num_points_test = args.num_points_test
    num_points_val = args.num_points_val
    N = args.N
    num_comps = args.num_comps
    start = args.start
    batch_size = args.B
    epochs = args.epochs
    lr = args.lr
    seed = args.seed
    cpu_use = args.cpu

    torch.manual_seed(seed)

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/linear_pca_classifier.cmd', 'a') as f:
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

    # Create model handler
    handler = Bert_Layer_Handler(model, layer_num=layer_num)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Use training data to get eigenvector basis of CLS token at correct layer
    input_ids, mask, _ = get_train('bert', train_base_dir)
    indices = torch.randperm(len(input_ids))[:num_points_train]
    input_ids = input_ids[indices]
    mask = mask[indices]
    CLS_embeddings = batched_get_handler_embeddings(input_ids, mask, handler, device)
    with torch.no_grad():
        correction_mean = torch.mean(CLS_embeddings, dim=0)
        cov = get_covariance_matrix(CLS_embeddings)
        e, v = get_e_v(cov)
    
    # Save the PCA embedding eigenvectors and correction mean
    torch.save(v, pca_out_file)
    torch.save(correction_mean, pca_mean_out_file)

    # Load the test data
    original_list_neg, original_list_pos, attack_list_neg, attack_list_pos = load_test_adapted_data_sentences(test_base_dir, num_points_test)

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

    labels = torch.LongTensor([0]*original.size(0) + [1]*attack.size(0))
    X = torch.cat((original, attack))

    # Shuffle all the data
    indices = torch.randperm(len(labels))
    labels = labels[indices]
    X = X[indices]

    # Split data
    X_val = X[:num_points_val]
    labels_val = labels[:num_points_val]
    X_train = X[num_points_val:]
    labels_train = labels[num_points_val:]

    ds_train = TensorDataset(X_train, labels_train)
    ds_val = TensorDataset(X_val, labels_val)
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=batch_size)

    # Model
    model = LayerClassifier(num_comps)
    model.to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Criterion
    criterion = nn.CrossEntropyLoss().to(device)

    # Create file
    with open(out_file, 'w') as f:
        text = f'Layer {layer_num}, Comps {num_comps}, N {N}\n'
        f.write(text)

    # Train
    for epoch in range(epochs):

        # train for one epoch
        text = '\n current lr {:.5e}'.format(optimizer.param_groups[0]['lr'])
        with open(out_file, 'a') as f:
            f.write(text)
        print(text)
        train(dl_train, model, criterion, optimizer, epoch, device, out_file)

        # evaluate
        eval(dl_val, model, criterion, device, out_file)
    
    # Save the trained model for identifying adversarial attacks
    torch.save(model.state_dict(), classifier_out_file)
