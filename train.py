import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
use_cuda = torch.cuda.is_available()
from torch.autograd import Variable
import time
import math
import data
import random
import model_transformer

# Defining the End of sentence and the starting
EOS_TOKEN =1
SOS_TOKEN =0



# Preparing the training data
def indexes_from_sentences(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def VariableFromSentence(lang, sentence):
    indexes = indexes_from_sentences(lang, sentence)
    indexes.append(EOS_TOKEN)
    result = Variable(torch.LongTensor(indexes).view(-1,1))
    if use_cuda:
        result = result.cuda()
    return result

def variables_from_pairs(input_lang, output_lang, pair):
    input_variable = VariableFromSentence(input_lang, pair[0])
    target_variable = VariableFromSentence(output_lang, pair[1])
    return (input_variable, target_variable)


def train(model,optimizer, input_variable, target_variable, criterion):

    optimizer.zero_grad()
    output_sequence = model(input_variable, target_variable)
    loss = criterion(output_sequence, target_variable)
    loss.backward()
    optimizer.step()

    return loss.data[0]



# Helper function to print elapsed time and estimated time remaining
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))



# Actually training process
def train_iters(model, n_epochs, print_every=1000, learning_rate=0.01):
    start = time.time()
    print_loss_total = 0

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    input_lang, output_lang, pairs = data.prepareData('eng', 'deu', True)
    training_pairs = [variables_from_pairs(input_lang, output_lang, random.choice(pairs)) for i in range(n_epochs)]
    criterion  = nn.CrossEntropyLoss()

    for epoch in range(n_epochs):
        training_pair = training_pairs[epoch]
        input_variable = training_pair[0]
        target_variable = training_pair[1]


        print(input_variable)
        print(target_variable)

        loss = train(model=model, optimizer=optimizer,
                     input_variable=input_variable, target_variable=target_variable,criterion=criterion )
        print_loss_total+=loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_epochs),
                                         iter, iter / n_epochs * 100, print_loss_avg))




source_vocab, target_vocab  = data.get_vocab()

# Actually training the model
model = model_transformer.Transformer(queries_dim=512, keys_dim=512, values_dim=512, model_dim=512,
                                      num_encoder_layers=6, num_decoder_layers=6, n_source_vocab=source_vocab,
                                      n_target_vocab=target_vocab, num_encoder_heads=8, num_decoder_heads=8)

if use_cuda:
    model = model.cuda()

train_iters(model, n_epochs=20)


