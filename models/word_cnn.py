#!/usr/bin/env python
"""
WordCNN Model from
Yoon Kim. 2014. Convolutional Neural Networks for Sentence Classification

referenced https://github.com/dennybritz/cnn-text-classification-tf
"""

import tensorflow as tf
import numpy as np


class WordCNN(object):

    def __init__(
            self, sequence_length, n_classes, vocab_size,
            filter_sizes=(1, 2, 3), num_filters=50,
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
            self.X = tf.placeholder(
                tf.int32, [None, sequence_length])  # , vocab_size])
            self.Y = tf.placeholder(tf.float32, [None, n_classes])
            self.embedded_X = tf.nn.embedding_lookup(embedding_matrix, self.X)
            self.training = tf.placeholder(tf.bool, shape=())

        # ================ Conv Layers ================
        with tf.name_scope('cnn-layers'):
            # Create a convolution + maxpool layer for each filter size
            pooled_outputs = []
            for i, filter_size in enumerate(filter_sizes):
                with tf.name_scope("conv-maxpool-%s" % filter_size):
                    # Convolution Layer
                    conv = tf.layers.conv1d(
                        inputs=self.embedded_X,
                        filters=num_filters,
                        kernel_size=filter_size,
                        padding='valid',
                        kernel_initializer=initializer,
                        kernel_regularizer=regularizer,
                        use_bias=True,
                        bias_initializer=initializer,
                        bias_regularizer=regularizer,
                        activation=tf.nn.relu)
                    pool = tf.layers.max_pooling1d(
                        inputs=conv, pool_size=sequence_length - filter_size + 1, strides=1)
                    pooled_outputs.append(pool)

            # Combine all the pooled features
            # [batch_size, 1, num_filters] * len(filter_sizes)
            num_filters_total = num_filters * len(filter_sizes)
            # [batch_size, 1, num_filters_total]
            pooled = tf.concat(pooled_outputs, 2)
            # [batch_size, num_filters_total]
            flattened = tf.reshape(pooled, [-1, num_filters_total])

        # ================ Output Layer ================
        with tf.name_scope('output-dense-layer'):
            logits = tf.layers.dense(
                inputs=flattened,
                units=n_classes,
                activation=tf.nn.softmax,
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
            # self.loss = tf.reduce_mean(
            #     tf.nn.softmax_cross_entropy_with_logits(labels=self.Y,
            #                                             logits=logits),
            #     name="softmax_cross_entropy_loss")
            self.loss = tf.reduce_mean(
                tf.nn.weighted_cross_entropy_with_logits(self.Y,
                                                         logits, self.positive_weight),
                name="cross_entropy_loss")

            print("\nUsing AdamOptimizer with learning_rate=%s"
                  % learning_rate)
            self.optimizer = tf.train.AdamOptimizer(learning_rate)
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=self.global_step, name="train_op")
            tf.summary.scalar("loss", self.loss)

        with tf.name_scope("prediction"):
            self.prediction = tf.argmax(logits, axis=1, name="prediction")

        self.merge_summary = tf.summary.merge_all()
