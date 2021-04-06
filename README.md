# Objective

Use word level saliency to perform NLP adversarial attack.

Traditionally, saliency at a word embedding level (by back propagating the loss to the word embeddings) is used to determine which word to substitute in an adversarial attack. However, this approach is only an upper bound approximation for the true saliency, as it does not capture the discrete nature of the embedding space. This work seeks to explore how a higher order saliency approximation compares to the traditional upperbound saliency, when performing word level substitution NLP adversarial attacks.

This work focuses on attacking models trained for sentiment classification of IMDB reviews.

Note, that the implementation strategy used here performs all substitutions at a token-level, as opposed to a strict word-level.

# Requirements

## Install using PyPI

pip install torch

pip install transformers
