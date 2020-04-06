import tensorflow as tf

import keras

import numpy as np
import pandas as pd


def build_data(data, Tx = 5, stride = 2):
    """
    Create a training set by scanning a window of size Tx over the event sequenses, with a given stride.

    Arguments:
    X -- events array
    Tx -- sequence length, number of time-steps in one training example
    stride -- how much the window shifts itself while scanning

    Returns:
    X -- list of training examples
    Y -- list of training labels
    """

    X = []
    Y = []

    for i in range(0, len(data) - 2*Tx, stride):
        X.append(['<start>']+data[i: i + Tx])
        Y.append(['<start>']+data[i + Tx:i+2*Tx])

    print('number of training examples:', len(X))

    return X, Y


def tokenize(seq,seq_tokenizer):
    """
    Making a tokenized dataset given inputs of event sequences and tokenzier object

    Arguments:
    seq -- an array or list of sequences
    tokenizer -- a Tokenizer object fitted on events combinations

    Returns:
    tensor -- a tokenized tensor dataset
    """

    tensor = seq_tokenizer.texts_to_sequences(seq)

    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')

    return tensor


def load_dataset(input_seq,target_seq,seq_tokenizer):

    input_tensor = tokenize(input_seq,seq_tokenizer)
    target_tensor = tokenize(target_seq,seq_tokenizer)

    return input_tensor, target_tensor

def convert(tensor,seq):
    '''
    convert a tensor back to event sequence
    '''
    for t in tensor:
        if t!=0:
            print ("%d ----> %s" % (t, seq.index_word[t]))
