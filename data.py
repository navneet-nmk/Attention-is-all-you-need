import torch
import torch.nn as nn
import torchvision
import numpy as np
import unicodedata
import re
import random

from io import open

# Creating the vocabulary for our corpus
class Lang(object):
    def __init__(self, name):
        super(Lang, self).__init__()
        self.name = name
        self.index2word = {0:'SOS', 1:'EOS'}
        self.word2index=  {}
        self.word2count = {}
        self.n_words =2


    def addSentence(self, sentence):
        words  = sentence.split(' ')
        for w in words:
            self.addWord(w)


    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.word2count[word] = 1
            self.n_words += 1
        else:
            self.word2count[word]+=1


# The files are all in Unicode, to simplify we will turn Unicode characters to
            # ASCII, make everything lowercase, and trim most punctuation.


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open('%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
        read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs



def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


def get_embedding_weights(embedding_size):
    input_lang, output_lang, pairs = prepareData('eng', 'deu', True)

    # Create the embeddings for the words
    vocab_size = input_lang.n_words
    print(vocab_size)
    # Standard deviation to use
    sd = 1 / np.sqrt(embedding_size)
    weights = np.random.normal(0, sd, size=[vocab_size, embedding_size])
    weights = weights.astype(np.float32)

    # Use the Glove vectors for the words available
    file = '/media/kumar/Data/Pretrained_vectors/glove.840B.300d.txt'

    # Extract desired glove vectors from the file
    with open(file, encoding='utf', mode='r') as text_file:
        for line in text_file:
            line = line.split()
            word = line[0]

            if word in input_lang.word2index:
                id = input_lang.word2index(word)
                weights[id] = np.array(line[1:], dtype=np.float32)

    return weights


def get_vocab():
    input_lang, output_lang, pairs = prepareData('eng', 'deu', True)
    input_vocab_size = input_lang.n_words
    output_vocab_size = output_lang.n_words
    return input_vocab_size, output_vocab_size


def get_output_embeddding_weights(embedding_input_size):
    input_lang, output_lang, pairs = prepareData('eng', 'deu', True)

    # Create the embedding for the words
    vocab_size = output_lang.n_words
    print(vocab_size)
    #Standard deviation to use
    sd = 1/np.sqrt(embedding_input_size)
    weights = np.random.normal(0, sd, size=[vocab_size, embedding_input_size])
    weights = weights.astype(np.float32)

    # Since we dont have any Glove vectors for the german words available
    return weights





