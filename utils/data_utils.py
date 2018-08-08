#!/usr/bin/env python
""" Helper functions for character-based features"""

import os
import random

import pandas as pd
import numpy as np

from tqdm import tqdm  # percentage bar for tasks
from sklearn.model_selection import StratifiedShuffleSplit

TWEET_MAX_LEN = 140
FEATURES = list(
    "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:’’’/\|_@#$%ˆ&*~‘+-=<>()[]{}")

NUM_CLASSES = 3
CLASSES = ('none', 'racism', 'sexism')

NPY_DIR = './npys/'


def load_data(csvfile, x_npy, y_npy, word_cnn=False):
    """
    Read each text and label into a one-hot matrix and save as .npy
    ((num_tweets, tweet_max_len, num_quantization), (num_tweets, 3))
    """
    df = pd.read_csv(csvfile, error_bad_lines=False)

    # if word_cnn:
    #     return df['Text'], df['Label']

    # Convert to 1-hot matrices and vectors, and zip them
    x_data = []
    y_data = []

    if not (os.path.exists(x_npy) and os.path.exists(y_npy)):
        texts = df['Text']
        labels = df['Label']

        for text, label in tqdm(zip(texts, labels)):
            x_data.append(text)
            y_data.append(label)

        x_data = np.array(x_data)
        y_data = np.array(y_data)
        np.save(x_npy, x_data)
        np.save(y_npy, y_data)
    else:
        x_data = np.load(x_npy)
        y_data = np.load(y_npy)

    print(x_data.shape)
    print(y_data.shape)

    return x_data, y_data


def split_dataset(X, Y, num_folds=10):
    """
    train : test = 90 : 10
    use k-fold (k=10) for validation
    """
    return StratifiedShuffleSplit(n_splits=num_folds, test_size=0.1).split(X, Y)


def balanced_batch_gen(x, y, batch_size=30, num_classes=3):
    """
    Randomly generate mini batches for training.
    Sample from each label uniformly when creating a batch.
    """
    is_hybrid = False
    if type(x) is tuple:
        is_hybrid = True
    classes = np.unique(y, axis=0)

    assert len(classes) == num_classes

    idx = []
    for i in range(num_classes):
        idx.append(np.where(y == classes[i])[0])

    balance = 1 / num_classes
    while True:
        sample_idx = []
        for cl in range(num_classes):
            sample_idx.append(random.choices(
                idx[cl], k=int(balance * batch_size)))

        if is_hybrid:
            batch_x = ([], [])
        else:
            batch_x = []
        batch_y = []
        for cl in range(num_classes):
            if is_hybrid:
                batch_x_char = batch_x[0] + [x[0][i] for i in sample_idx[cl]]
                batch_x_word = batch_x[1] + [x[1][i] for i in sample_idx[cl]]
                batch_x = (batch_x_char, batch_x_word)
            else:
                batch_x += [x[i] for i in sample_idx[cl]]
            batch_y += [y[i] for i in sample_idx[cl]]

        yield batch_x, batch_y


def one_hot_to_char(row):
    """
    One hot row -> text
    """
    for i, c in enumerate(row):
        if c == 1:
            return FEATURES[i]
    return ''


def one_hot_to_chars(mat):
    """
    One hot 2d matrix -> vectors of texts
    """
    return [one_hot_to_char(_row) for _row in mat]


def text_to_1hot_matrix(text, max_len=TWEET_MAX_LEN, features=FEATURES):
    if features is None:
        features = FEATURES
    tokens = list(filter(lambda x: x in features, list(text)))
    matrix = np.zeros((max_len, len(features)))
    for i, t in enumerate(tokens):
        if i < max_len:
            row = np.zeros(len(features))
            try:
                j = features.index(t)
            except ValueError:
                j = -1
            if j >= 0:
                row[j] = 1
            matrix[i] = row
        else:
            return matrix

    return matrix


def label_to_1hot_vector(label, n_classes=NUM_CLASSES, classes=CLASSES):
    vector = np.zeros(n_classes)
    for i in range(n_classes):
        if label == classes[i]:
            vector[i] = 1
            break

    return vector
