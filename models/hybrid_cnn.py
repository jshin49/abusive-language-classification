#!/usr/bin/env python
"""
HybridCNN Model
"""

import tensorflow as tf
import numpy as np


class HybridCNN(object):

    def __init__(
            self, word_len, char_len, n_classes,
            num_quantization, vocab_size,
            char_filter_sizes=(3, 4, 5),
            word_filter_sizes=(1, 2, 3),
            num_filters=50,
            embedding_size=300,
            dropout_prob=0.5,
            use_embedding_layer=True,
            train_embedding=False,
            embedding_matrix=None,
            learning_rate=0.001,
            l2=1.0):

        initializer = tf.contrib.layers.xavier_initializer()
        regularizer = tf.contrib.layers.l2_regularizer(l2)

        # ================ Input Layer ================
        with tf.name_scope('input-layer'):
            self.input_chars = tf.placeholder(
                tf.float32, [None, char_len, num_quantization])
            self.input_words = tf.placeholder(
                tf.int32, [None, word_len])
            self.Y = tf.placeholder(tf.float32, [None, n_classes])
            self.embedded_words = tf.nn.embedding_lookup(
                embedding_matrix, self.input_words)
            self.training = tf.placeholder(tf.bool, shape=())

        # ================ Conv Layers ================
        with tf.name_scope('char-cnn-layers'):
            # Create a convolution + maxpool layer for each filter size
            pooled_outputs = []
            for i, char_filter_size in enumerate(char_filter_sizes):
                with tf.name_scope("conv-maxpool-%s" % char_filter_size):
                    # Convolution Layer
                    conv = tf.layers.conv1d(
                        inputs=self.input_chars,
                        filters=num_filters,
                        kernel_size=char_filter_size,
                        padding='valid',
                        kernel_initializer=initializer,
                        kernel_regularizer=regularizer,
                        use_bias=True,
                        bias_initializer=initializer,
                        bias_regularizer=regularizer,
                        activation=tf.nn.relu)
                    pool = tf.layers.max_pooling1d(
                        inputs=conv, pool_size=char_len - char_filter_size + 1, strides=1)
                    pooled_outputs.append(pool)

            # Combine all the pooled features
            # [batch_size, 1, num_filters] * len(filter_sizes)
            num_filters_total = num_filters * len(char_filter_sizes)
            print("char num filters", num_filters_total)
            # [batch_size, 1, num_filters_total]
            pooled = tf.concat(pooled_outputs, 2)
            print("char pooled concatenated", pooled.shape)
            # [batch_size, num_filters_total]
            char_flattened = tf.reshape(pooled, [-1, num_filters_total])
            print("char flattened", char_flattened.shape)

        with tf.name_scope('word-cnn-layers'):
            # Create a convolution + maxpool layer for each filter size
            pooled_outputs = []
            for i, word_filter_size in enumerate(word_filter_sizes):
                with tf.name_scope("conv-maxpool-%s" % word_filter_size):
                    # Convolution Layer
                    conv = tf.layers.conv1d(
                        inputs=self.embedded_words,
                        filters=num_filters,
                        kernel_size=word_filter_size,
                        padding='valid',
                        kernel_initializer=initializer,
                        kernel_regularizer=regularizer,
                        use_bias=True,
                        bias_initializer=initializer,
                        bias_regularizer=regularizer,
                        activation=tf.nn.relu)
                    pool = tf.layers.max_pooling1d(
                        inputs=conv, pool_size=word_len - word_filter_size + 1, strides=1)
                    pooled_outputs.append(pool)

            # Combine all the pooled features
            # [batch_size, 1, num_filters] * len(filter_sizes)
            num_filters_total = num_filters * len(word_filter_sizes)
            print("word num filters", num_filters_total)
            # [batch_size, 1, num_filters_total]
            pooled = tf.concat(pooled_outputs, 2)
            print("word pooled concatenated", pooled.shape)
            # [batch_size, num_filters_total]
            word_flattened = tf.reshape(pooled, [-1, num_filters_total])
            print("word flattened", word_flattened.shape)

        flattened = tf.concat(list([char_flattened, word_flattened]), 1)
        print("concatenated both word and char", flattened.shape)

        # ================ Output Layer ================
        with tf.name_scope('output-dense-layer'):
            logits = tf.layers.dense(
                inputs=flattened,
                units=n_classes,
                activation=None,  # tf.nn.softmax,
                kernel_initializer=initializer,
                kernel_regularizer=regularizer,
                use_bias=True,
                bias_initializer=initializer,
                bias_regularizer=regularizer)
            logits = tf.layers.dropout(
                inputs=logits,
                rate=dropout_prob,
                training=self.training)

        with tf.name_scope("training"):
            # using weighted cross-entropy loss since we have imbalanced
            # dataset. put more emphasis on positive training examples
            self.global_step = tf.Variable(
                0, name="global_step", trainable=False)
            print("\nUsing softmax cross-entropy loss")
            self.positive_weight = tf.constant(1.0, dtype="float32")
            self.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=self.Y,
                                                        logits=logits),
                name="softmax_cross_entropy_loss")
            # self.loss = tf.reduce_mean(
            #     tf.nn.weighted_cross_entropy_with_logits(self.Y,
            #                                              logits, self.positive_weight),
            #     name="cross_entropy_loss")

            print("\nUsing AdamOptimizer with learning_rate=%s"
                  % learning_rate)
            self.optimizer = tf.train.AdamOptimizer(learning_rate)
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=self.global_step, name="train_op")
            tf.summary.scalar("loss", self.loss)

        with tf.name_scope("prediction"):
            self.prediction = tf.argmax(logits, axis=1, name="prediction")

        self.merge_summary = tf.summary.merge_all()
