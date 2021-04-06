'''
Use upper bound saliency at the word embedding level to choose word to substitute
with synonym.
'''

def get_word_saliencies():
    '''
    Returns list ordered by word order, where each item is a tuple
    (word, saliency).

    Saliency is an upperbound saliency, given by the size of the vector of the
    loss functions derivative wrt to the word embedding.

    Word embeddings are taken from the second last layer of the encoder - only
    the sentence embedding is used from the final layer.
    '''

def attack_sentence(sentence, model, criterion):
    '''
    Identifies the most salient word (by upper bound saliency)
    Finds synonyms for this word using WordNet
    Selects the best synonym to replace with based on Forward Pass to maximise
    the loss function.

    Returns the updated sentence
    '''

    word_saliencies = get_word_saliencies(sentence, model, criterion)

def N_step_attack(sentence, model, criterion, N=1):
    '''
    Apply the substitute attack N-times greedily
    '''
    updated = sentence
    for n in range(N):
        updated = attack_sentence(updated, model, criterion)
    return updated
