'''
Use upper bound saliency at the word embedding level to choose word to substitute
with synonym.

This is currently designed for Electra encoder based models only
'''

import torch
from nltk.corpus import wordnet as wn

def get_word_saliencies(sentence, label, model, criterion, tokenizer):
    '''
    Returns list ordered by word order, where each item is a tuple
    (word, saliency).

    Saliency is an upperbound saliency, given by the size of the vector of the
    loss functions derivative wrt to the word embedding.

    Word embeddings are taken from the second last layer of the encoder - only
    the sentence embedding is used from the final layer.
    '''

    encoded_inputs = tokenizer([sentence], padding=True, truncation=True, return_tensors="pt")
    ids = encoded_inputs['input_ids']
    mask = encoded_inputs['attention_mask']

    target = torch.LongTensor([label])

    model.eval()
    all_layers_hidden_states = self.model.electra(ids, mask)
    second_last_layer = all_layers_hidden_states[1]
    second_last_layer.retain_grad()
    final_layer = all_layers_hidden_states[0]
    logits = self.model.classifier(final_layer)
    loss = criterion(logits, target)

    # Determine embedding gradient
    loss.backward()
    embedding_grad = second_last_layer.grad().squeeze()

    word_saliencies = []
    for k, id in enumerate(ids):
        saliency = torch.norm(embedding_grad[k]).item()
        word_token =  tokenizer.convert_ids_to_tokens(id.item())
        word_saliencies.append(word_token, saliency)

    return word_saliencies


def attack_sentence(sentence, label, model, criterion, tokenizer, max_syn=5):
    '''
    Identifies the most salient word (by upper bound saliency)
    Finds synonyms for this word using WordNet
    Selects the best synonym to replace with based on Forward Pass to maximise
    the loss function.

    Returns the updated sentence, the original token, new token and
    its position in the sentence
    '''

    word_saliencies = get_word_saliencies(sentence, label, model, criterion, tokenizer)

    best = {'word':'none', 'saliency': 0, 'token_position':-1}
    for i, (word, saliency) in enumerate(word_saliencies[1:-1]):
        if saliency > best[saliency]:
            best['word'] = word
            best['saliency'] = saliency
            best['token_position'] = i

    # Find the token synonyms
    synonyms = wn.synset(best['word']+'.n.01').lemma_names()
    if len(synonyms) > max_syn:
        synonyms = synonyms[:max_syn]

    # Find best synonym for attack
    best_syn = {'original_token':None, 'new_token':None, 'token_position':None, 'loss':0}

    encoded_inputs = tokenizer([sentence], padding=True, truncation=True, return_tensors="pt")
    ids = encoded_inputs['input_ids']
    mask = encoded_inputs['attention_mask']

    target = torch.LongTensor([label])

    model.eval()
    with torch.no_grad():
        for synonym in synonyms:
            syn_id = tokenizer.convert_tokens_to_ids(synonym)
            attacked_ids = ids.clone()
            attacked_ids[best['token_position']] = syn_id
            logits = model(attacked_ids, mask)
            loss = criterion(logits, target).item()

            if loss > best_syn['loss']:
                best_syn['original_token'] = best['word']
                best_syn['new_token'] = synonym
                best_syn['token_position'] = best['token_position']
                best_syn['loss'] = loss
                best_syn['syn_id'] = syn_id

    # Make attacked sentence
    attacked_ids = ids.clone()
    attacked_ids[best['token_position']] = best_syn['syn_id']
    attack_sentence = tokenizer.decode(attacked_ids)


    return attack_sentence, best_syn['original_token'], best_syn['new_token'], best_syn['token_position']

def N_step_attack(sentence, label, model, criterion, tokenizer, max_syn=5, N=1):
    '''
    Apply the substitute attack N-times greedily
    Returns a lot of information
    '''
    updated = sentence
    for n in range(N):
        (updated, org_tkn, new_tkn, pos) = attack_sentence(updated, label, model, criterion, tokenizer, max_syn)
    return (updated, org_tkn, new_tkn, pos)
