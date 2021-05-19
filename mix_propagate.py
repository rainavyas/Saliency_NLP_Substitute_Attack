'''
Aim to analyse adversarial attack propagation split beween CLS token
and other tokens.

For a particular layer, CLS token can be swapped out with original or adv.
Similarly, the other embeddings can be swapped out with original or adv.

This way, we can plot results (loss vs layer number) for:
1) orig-orig
2) adv-adv
3) adv-orig
4) orig-adv
'''
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from models import BertSequenceClassifier
from transformers import BertTokenizer
from layer_handler import Bert_Layer_Handler
from tools import get_default_device, accuracy_topk
import sys
import os
import argparse
import matplotlib.pyplot as plt
from pca_component_comparison_plot import load_test_adapted_data_sentences

def eval_sentences(sentences_list, model, device, bs=8):
    '''
    Returns logits after passing sentences through the model
    Allows batching and gpu use
    '''
    encoded_inputs = tokenizer(sentences_list, padding=True, truncation=True, return_tensors="pt")
    ids = encoded_inputs['input_ids']
    mask = encoded_inputs['attention_mask']

    model.eval()

    logits = []
    ds = TensorDataset(ids, mask)
    dl = DataLoader(ds, batch_size=bs)
    with torch.no_grad():
        for id, m in dl:
            id = id.to(device)
            m = m.to(device)
            curr_logits = model(id, m)
            logits.append(curr_logits.cpu())
    logits = torch.cat(logits)
    return logits



def batched_get_layer_embedding(sentences_list, handler, tokenizer, device, bs=8):
    '''
    Performs the function of preparing sentence list and gets all layer embeddings
    in batches and allows gpu use
    '''
    encoded_inputs = tokenizer(sentences_list, padding=True, truncation=True, return_tensors="pt")
    ids = encoded_inputs['input_ids']
    mask = encoded_inputs['attention_mask']
    return batched_get_handler_embeddings(ids, mask, handler, device, bs=bs)

def batched_get_handler_embeddings(input_ids, mask, handler, device, bs=8):
    '''
    Input is a tensor of input ids and mask
    Returns tensor of all embeddings at the correct layer
    Does this in batches
    '''
    embeddings = []
    ds = TensorDataset(input_ids, mask)
    dl = DataLoader(ds, batch_size=bs)
    with torch.no_grad():
        for id, m in dl:
            id = id.to(device)
            m = m.to(device)
            layer_embeddings = handler.get_layern_outputs(id, m, device=device)
            embeddings.append(layer_embeddings.cpu())
    embeddings = torch.cat(embeddings)
    return embeddings

def batched_propagate_handler_embeddings(hidden_states, mask, handler, device, bs=8):
    '''
    Passes layer embeddings through the remainder of the model
    Does this in a batched manner with gpu use allowed
    Returns logits tensor
    '''
    logits = []
    ds = TensorDataset(hidden_states, mask)
    dl = DataLoader(ds, batch_size=bs)
    with torch.no_grad():
        for h, m in dl:
            h = h.to(device)
            m = m.to(device)
            curr_logits = handler.pass_through_rest(h, m, device=device)
            logits.append(curr_logits.cpu())
    logits = torch.cat(logits)
    return logits

def mix_lists_propagate(sentences_list1, sentence_list2, handler, tokenizer, layer_num, device, bs=8):
    '''
    CLS used from list 1
    embeddings used from list 2
    After the layer specified
    returns logits tensor
    '''

    handler.layer_num = layer_num
    embeddings1 = batched_get_layer_embedding(sentences_list1, handler, tokenizer, device, bs=bs)
    embeddings2 = batched_get_layer_embedding(sentences_list2, handler, tokenizer, device, bs=bs)

    with torch.no_grad():
        mixed_embeddings = embeddings2.clone()
        mixed_embeddings[:,0,:] = embeddings1[:,0,:]
    return batched_propagate_handler_embeddings(mixed_embeddings, mask, handler, device, bs=bs)

def generate_plot(layers, orig_orig, adv_adv, orig_adv, adv_orig, ylabel, filename):
    plt.plot(layers, orig_orig, label='orig-orig')
    plt.plot(layers, adv_adv, label='adv-adv')
    plt.plot(layers, orig_adv, label='orig-adv')
    plt.plot(layers, adv_orig, label='adv-orig')
    plt.xlabel("Layer Output Manipulated")
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(filename)
    plt.clf()



if __name__ == '__main__':

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('MODEL', type=str, help='trained .th model')
    commandLineParser.add_argument('TEST_DIR', type=str, help='attacked test data base directory')
    commandLineParser.add_argument('OUT', type=str, help='file base name for saving plots')
    commandLineParser.add_argument('--num_points_test', type=int, default=12500, help="number of pairs data points to use test")
    commandLineParser.add_argument('--cpu', type=str, default='no', choice=['no', 'yes'], help="force cpu use")

    args = commandLineParser.parse_args()
    model_path = args.MODEL
    test_base_dir = args.TEST_DIR
    out_file = args.OUT
    num_points_test = args.num_points_test
    cpu_use = args.cpu

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/mix_propagate.cmd', 'a') as f:
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
    handler = Bert_Layer_Handler(model, layer_num=1)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Load the test data
    original_list_neg, original_list_pos, attack_list_neg, attack_list_pos = load_test_adapted_data_sentences(test_base_dir, num_points_test)
    original_list = original_list_neg + original_list_pos
    labels = torch.LongTensor([0]*len(original_list_neg) + [1]*len(original_list_pos))
    attack_list = attack_list_neg + attack_list_pos

    loss_criterion = nn.CrossEntropyLoss()

    # Calculate orig-orig loss and accuracy
    logits = eval_sentences(original_list, model, device)
    orig_acc = accuracy_topk(logits, labels)
    orig_loss = loss_criterion(logits, labels)

    # Calculate adv-adv loss and accuracy
    logits = eval_sentences(attack_list, model, device)
    attack_acc = accuracy_topk(logits, labels)
    attack_loss = loss_criterion(logits, labels)

    # Get loss and accuracy with mixing applied at each layer
    orig_orig_accs = []
    orig_orig_losss = []
    adv_adv_accs = []
    adv_adv_losss = []
    orig_adv_accs = []
    orig_adv_losss = []
    adv_orig_accs = []
    adv_orig_losss = []
    layers = []

    for layer_num in range(13):
        print("On layer", layer_num)
        orig_adv_logits = mix_lists_propagate(original_list, attack_list, handler, tokenizer, layer_num, device)
        orig_adv_acc = accuracy_topk(orig_adv_logits, labels)
        orig_adv_loss = loss_criterion(orig_adv_logits, labels)

        adv_orig_logits = mix_lists_propagate(attack_list, original_list, handler, tokenizer, layer_num, device)
        adv_orig_acc = accuracy_topk(adv_orig_logits, labels)
        adv_orig_loss = loss_criterion(adv_orig_logits, labels)

        orig_orig_accs.append(orig_acc)
        orig_orig_losses.append(orig_loss)
        adv_adv_accs.append(attack_acc)
        adv_adv_losss.append(attack_loss)
        orig_adv_accs.append(orig_adv_acc)
        orig_adv_losss.append(orig_adv_loss)
        adv_orig_accs.append(adv_orig_acc)
        adv_orig_losss.append(adv_orig_loss)
        layers.append(layer_num)

    # Plot accuracy vs layer
    filename = out_file + '_accuracy.png'
    ylabel = 'Accuracy'
    generate_plot(layers, orig_orig_accs, adv_adv_accs, orig_adv_accs, adv_orig_accs, ylabel, filename)

    # Plot loss vs layer
    filename = out_file + '_loss.png'
    ylabel = 'Loss'
    generate_plot(layers, orig_orig_losss, adv_adv_losss, orig_adv_losss, adv_orig_losss, ylabel, filename)
