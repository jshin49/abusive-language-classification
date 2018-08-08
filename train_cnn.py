import sys
import random

import tensorflow as tf
import pandas as pd
import numpy as np
from tqdm import tqdm  # percentage bar for tasks

import utils.data_utils as du
from utils.prepare_data_word_cnn import load_word_vectors, build_vocabulary, embed_to_word_vectors

from models.char_cnn import CharCNN
from models.word_cnn import WordCNN
from models.hybrid_cnn import HybridCNN

from metric_helper import ClassificationReport

NPY_DIR = './npys/'


def load_and_split_data(csvfile, x_npy, y_npy, num_folds, word_cnn=False):
    X, Y = du.load_data(csvfile, x_npy, y_npy, word_cnn)
    folds = du.split_dataset(X, Y, num_folds)
    return X, Y, folds


def next_batch(batches, sequence_length=None, vocab=None):
    batch_x, batch_y = next(batches)

    is_hybrid = False
    if type(batch_x) is tuple:
        is_hybrid = True

    if vocab is None:
        batch_x = np.array(list(
            map(lambda text: du.text_to_1hot_matrix(text), batch_x)))

    if is_hybrid:
        batch_x = (np.array(list(
            map(lambda text: du.text_to_1hot_matrix(text), batch_x[0]))), batch_x[1])

    batch_y = np.array(list(
        map(lambda label: du.label_to_1hot_vector(label), batch_y)))

    return batch_x, batch_y


def train_batch(model, sess, summary_writer, batch_x, batch_y):
    is_hybrid = False
    if type(batch_x) is tuple:
        is_hybrid = True

    if is_hybrid:
        feed_dict = {
            model.input_chars: batch_x[0],
            model.input_words: batch_x[1],
            model.Y: batch_y,
            model.training: True
        }
    else:
        feed_dict = {
            model.X: batch_x,
            model.Y: batch_y,
            model.training: True
        }

    _, loss, summary = sess.run(
        [model.train_op, model.loss, model.merge_summary],
        feed_dict=feed_dict)
    step = tf.train.global_step(sess, model.global_step)
    summary_writer.add_summary(summary, step)
    return step, loss


def evaluate_epoch(model, sess, epoch, eval_x, eval_y, sequence_length=None, vocab=None):
    is_hybrid = False
    if type(eval_x) is tuple:
        is_hybrid = True

    if vocab is None:
        eval_x = np.array(list(
            map(lambda text: du.text_to_1hot_matrix(text), eval_x)))

    if is_hybrid:
        eval_x = (np.array(list(
            map(lambda text: du.text_to_1hot_matrix(text), eval_x[0]))), eval_x[1])

    eval_y = np.array(list(
        map(lambda label: du.label_to_1hot_vector(label), eval_y)))

    if is_hybrid:
        feed_dict = {
            model.input_chars: eval_x[0],
            model.input_words: eval_x[1],
            model.Y: eval_y,
            model.training: False
        }
    else:
        feed_dict = {
            model.X: eval_x,
            model.Y: eval_y,
            model.training: False
        }

    preds = sess.run(
        [model.prediction],
        feed_dict=feed_dict)

    classes = ['none', 'racism', 'sexism']
    evaluator = ClassificationReport(
        model, classes, np.argmax(eval_y, axis=1), np.array(preds[0]))
    return evaluator.on_epoch_end(epoch)


def train(model, sess, summary_writer, X, Y, fold=None, num_iter=None, batch_size=30, epoches=30, test_idx=None, sequence_length=None, vocab=None):
    is_hybrid = False
    if type(X) is tuple:
        is_hybrid = True
        X_char = X[0]
        X_word = X[1]

    if fold is None:
        if is_hybrid:
            train_x_char = X_char
            train_x_word = X_word
            train_y = Y
            test_x_char = X_char[test_idx]
            test_x_word = X_word[test_idx]
            test_y = Y[test_idx]

            train_x = (train_x_char, train_x_word)
            test_x = (test_x_char, test_x_word)
        else:
            train_x = X
            train_y = Y
            test_x = X[test_idx]
            test_y = Y[test_idx]
    else:
        if is_hybrid:
            train_x_char = X_char[fold[0]]
            train_x_word = X_word[fold[0]]
            train_y = Y[fold[0]]
            test_x_char = X_char[fold[1]]
            test_x_word = X_word[fold[1]]
            test_y = Y[fold[1]]

            train_x = (train_x_char, train_x_word)
            test_x = (test_x_char, test_x_word)
        else:
            train_x = X[fold[0]]
            train_y = Y[fold[0]]
            test_x = X[fold[1]]
            test_y = Y[fold[1]]

    if num_iter is None:
        num_iter = int(train_y.shape[0] / batch_size)

    for epoch in range(epoches):
        batches = du.balanced_batch_gen(train_x, train_y, batch_size)

        for i in tqdm(range(num_iter)):
            batch_x, batch_y = next_batch(batches, sequence_length, vocab)
            step, loss = train_batch(
                model, sess, summary_writer, batch_x, batch_y)

            # if step % 20 == 0:
            #     print('Epoch:', epoch)
            #     print('Step:', step)
            #     print('Loss:', loss)

        precision, recall, f1, precision_avg, recall_avg, f1_avg = evaluate_epoch(
            model, sess, epoch, test_x, test_y, sequence_length, vocab)

    return precision, recall, f1, (precision_avg, recall_avg, f1_avg)


def generate_report(precisions, recalls, fscores, totals):
    classes = ['none', 'racism', 'sexism']
    digits = 3

    last_line_heading = 'avg / total'

    if classes is None:
        classes = [u'%s' % l for l in labels]
    name_width = max(len(cn) for cn in classes)
    width = max(name_width, len(last_line_heading), digits)

    headers = ["precision", "recall", "f1-score"]
    head_fmt = u'{:>{width}s} ' + u' {:>9}' * len(headers)
    report = head_fmt.format(u'', *headers, width=width)
    report += u'\n\n'

    row_fmt = u'{:>{width}s} ' + u' {:>9.{digits}f}' * 2 + u' {:>9}\n'
    rows = zip(classes, precisions, recalls, fscores)

    for row in rows:
        report += row_fmt.format(*row, width=width, digits=digits)

    report += u'\n'

    # compute averages
    report += row_fmt.format(last_line_heading, *totals,
                             width=width, digits=digits)

    return report


def print_report(precisions, recalls, fscores, totals, num_folds):
    print('Generating average scores for %d folds:' % (num_folds))

    avg_precisions = list(
        map(lambda p: round(np.asscalar(p), 3), np.mean(precisions, axis=0)))

    avg_recalls = list(
        map(lambda r: round(np.asscalar(r), 3), np.mean(recalls, axis=0)))

    avg_fscores = list(
        map(lambda f: round(np.asscalar(f), 3), np.mean(fscores, axis=0)))

    avg_totals = list(
        map(lambda t: round(np.asscalar(t), 3), np.mean(totals, axis=0)))

    report = generate_report(
        avg_precisions, avg_recalls, avg_fscores, avg_totals)
    print(report)


def main(argv):
    csvfile = 'data/preprocessed/wassem_preprocessed.csv'
    x_npy = NPY_DIR + 'wassem.X.npy'
    y_npy = NPY_DIR + 'wassem.Y.npy'

    # General configs
    num_folds = 10
    num_classes = 3

    X, Y, folds = load_and_split_data(csvfile, x_npy, y_npy, num_folds, True)

    if argv[1] == 'char':
        # CharCNN configs
        num_epoches = 30
        batch_size = 30
        learning_rate = 0.00005
        len_tweet = 140
        num_quantization = 70

        # For compatibility with other training models
        X_train = X
        sequence_length = None
        vocab = None
    elif argv[1] == 'word':
        # WordCNN configs
        num_epoches = 30
        batch_size = 30
        learning_rate = 0.0005
        vocab, text2idx = build_vocabulary(X)
        print('created vocabulary of %d words from %d tweets.' %
              (len(vocab), len(X)))
        pretrained_word_vectors = './data/GoogleNews-vectors-negative300.bin'
        word_vectors = load_word_vectors(pretrained_word_vectors)
        print('Loaded Google News pre-trained Word Vectors')
        vocab_size = len(vocab)
        sequence_length = text2idx.shape[1]
        embedding_matrix = embed_to_word_vectors(word_vectors, vocab)
        del word_vectors
        X_train = text2idx
    elif argv[1] == 'hybrid':
        # HybridCNN configs
        num_epoches = 20
        batch_size = 30
        learning_rate = 0.001

        # CharCNN configs
        len_tweet = 140
        num_quantization = 70

        # WordCNN configs
        vocab, text2idx = build_vocabulary(X)
        print('created vocabulary of %d words from %d tweets.' %
              (len(vocab), len(X)))
        pretrained_word_vectors = './data/GoogleNews-vectors-negative300.bin'
        word_vectors = load_word_vectors(pretrained_word_vectors)
        print('Loaded Google News pre-trained Word Vectors')
        vocab_size = len(vocab)
        sequence_length = text2idx.shape[1]
        embedding_matrix = embed_to_word_vectors(word_vectors, vocab)
        del word_vectors
        X_train_char = X
        X_train_word = text2idx
        X_train = (X, text2idx)
    else:
        raise ValueError('Wrong model')

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    sess_config = tf.ConfigProto(
        allow_soft_placement=True,
        gpu_options=gpu_options)
    sess_config.gpu_options.allow_growth = True

    def create_model():
        if argv[1] == 'char':
            return CharCNN(num_classes, len_tweet,
                           num_quantization, learning_rate=learning_rate)
        elif argv[1] == 'word':
            return WordCNN(sequence_length=sequence_length,
                           n_classes=num_classes,
                           vocab_size=vocab_size,
                           embedding_matrix=embedding_matrix,
                           learning_rate=learning_rate)
        elif argv[1] == 'hybrid':
            return HybridCNN(char_len=len_tweet,
                             word_len=sequence_length,
                             n_classes=num_classes,
                             num_quantization=num_quantization,
                             vocab_size=vocab_size,
                             learning_rate=0.001,
                             embedding_matrix=embedding_matrix)
        else:
            raise ValueError('Wrong model')

    precisions, recalls, fscores, totals = ([], [], [], [])

    try:
        do_cross_validation = True if argv[2] == 'cross_validate' else False
    except IndexError:
        do_cross_validation = False

    if do_cross_validation:
        fold_num = 0
        # 10 fold cross validation
        for fold in folds:
            print("Evaluating with Fold", fold_num)
            tf.reset_default_graph()
            graph = tf.Graph()

            with graph.as_default():
                model = create_model()

                with tf.Session(config=sess_config, graph=graph) as sess:
                    with tf.device('/gpu:0'):
                        summary_writer = tf.summary.FileWriter(
                            '/tmp/tensorboard/', graph=sess.graph)
                        sess.run(tf.global_variables_initializer())
                        precision, recall, fscore, total = train(
                            model, sess, summary_writer, X_train, Y, fold,
                            batch_size=batch_size, epoches=num_epoches,
                            sequence_length=sequence_length, vocab=vocab)
                        precisions.append(precision)
                        recalls.append(recall)
                        fscores.append(fscore)
                        totals.append(total)

                    with tf.device('/cpu:0'):
                        tf.train.Saver().save(sess, './ckpts/%s-cnn-fold-%d.ckpt' %
                                              (argv[1], fold_num))
                        fold_num += 1
                        # if fold_num > 1:
                        #     break

        print("Cross Validation Training Completed, Generating report")
        print_report(precisions, recalls, fscores, totals, num_folds)

    else:
        print("Fully training on all data")
        tf.reset_default_graph()
        graph = tf.Graph()

        with graph.as_default():
            model = create_model()

            with tf.Session(config=sess_config, graph=graph) as sess:
                with tf.device('/gpu:0'):
                    summary_writer = tf.summary.FileWriter(
                        '/tmp/tensorboard/', graph=sess.graph)
                    sess.run(tf.global_variables_initializer())
                    precision, recall, fscore, total = train(
                        model, sess, summary_writer, X_train, Y, None,
                        num_iter=int(Y.shape[0] / batch_size),
                        batch_size=batch_size, epoches=num_epoches,
                        test_idx=random.choices(np.arange(Y.shape[0]), k=1000),
                        sequence_length=sequence_length, vocab=vocab)

                with tf.device('/cpu:0'):
                    tf.train.Saver().save(
                        sess, './ckpts/%s-cnn-fully-trained.ckpt' % (argv[1]))

    summary_writer.close()

if __name__ == '__main__':
    main(sys.argv)
