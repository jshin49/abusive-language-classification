from gensim.models.keyedvectors import KeyedVectors
from sklearn.feature_extraction.text import CountVectorizer  # , TfidfTransformer
from tensorflow.contrib import learn

import pandas as pd
import numpy as np


def load_word_vectors(path):
    return KeyedVectors.load_word2vec_format(path, binary=True)


def build_vocabulary(texts):
    max_document_length = max([len(text.split(" ")) for text in texts])
    vocab_processor = learn.preprocessing.VocabularyProcessor(
        max_document_length)
    text2idx = np.array(list(vocab_processor.fit_transform(texts)))
    vocab = vocab_processor.vocabulary_
    return vocab, text2idx


def embed_to_word_vectors(word_vectors, vocab, emb_size=300, trainable=False):
    vocab_size = len(vocab)
    embedding_matrix = np.zeros((vocab_size, emb_size))

    for word, idx in vocab._mapping.items():
        if word in word_vectors.wv.vocab:
            embedding_matrix[idx] = word_vectors.word_vec(word)

    return embedding_matrix.astype(np.float32)


if __name__ == '__main__':
    preprocessed_file = 'data/preprocessed/wassem_preprocessed.csv'
    df = pd.read_csv(preprocessed_file, sep=',',
                     error_bad_lines=False, header=0)
    texts = df['Text'].astype('U')
    # texts = df.values.astype('U').T[0]
    vocab = build_vocabulary(texts)
    print('created vocabulary of %d words from %d tweets.' %
          (len(vocab_arr), len(texts)))

    pretrained_word_vectors = './data/GoogleNews-vectors-negative300.bin'
    word_vectors = load_word_vectors(pretrained_word_vectors)
    print(word_vectors.wv.most_similar(
        positive=['woman', 'king'], negative=['man']))
    print(word_vectors.wv.most_similar(
        positive=['Paris', 'Italy'], negative=['France']))

    embedding_matrix = embed_to_word_vectors(word_vectors, vocab)
    print(embedding_matrix.shape)
