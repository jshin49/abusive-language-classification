#!/usr/bin/env python
"""
CharCNN Model from
Zhang, X.; Zhao, J.; and LeCun, Y. 2015. Character-level Convolutional
Networks for Text Classification. In Proceedings of NIPS.

also referenced https://github.com/johnb30/py_crepe
"""

import tensorflow as tf
import numpy as np


class CharCNN(object):
    N_FILTERS = {"small": 256, "large": 1024}
    FILTER_KERNELS = [7, 7, 3, 3, 3, 3]
    FULLY_CONNECTED_OUTPUT = {"small": 1024, "large": 2048}
    INIT_VAR = {"small": 0.05, "large": 0.02}

    def __init__(self, n_classes, len_tweets, n_quantized_chars, positive_weight=1,
                 kernel_size=4, learning_rate=0.0001, fc_dropout=0.5, l1=0, l2=1.0,
                 model_size='large', model_depth='shallow', max_pool_type=''):

        # ================ Config ================
        if kernel_size > 0:
            self.FILTER_KERNELS = np.ones(6, dtype=int)
            self.FILTER_KERNELS.fill(kernel_size)
            self.FILTER_KERNELS = list(
                map(lambda x: (x,), self.FILTER_KERNELS))
            print("Using Kernel Size:", self.FILTER_KERNELS)
        if max_pool_type == "normal_6":
            pool_size = [6, 6, 6]
        elif max_pool_type == "half_6":
            pool_size = [6, 3, 3]
        else:
            pool_size = [3, 3, 3]
        print("Using max_pool size:", pool_size)

        initializer = tf.random_normal_initializer(
            stddev=self.INIT_VAR[model_size])
        regularizer = tf.contrib.layers.l2_regularizer(l2)

        # ================ Input Layer ================
        with tf.name_scope('input-layer'):
            self.X = tf.placeholder(
                tf.float32, [None, len_tweets, n_quantized_chars])
            self.Y = tf.placeholder(tf.float32, [None, n_classes])
            self.training = tf.placeholder(tf.bool, shape=())

        # ================ Conv Layers ================
        with tf.name_scope('cnn-layers'):
            # ================ Layer 1 ================
            with tf.name_scope('cnn-layer-0'):
                conv0 = tf.layers.conv1d(
                    inputs=self.X,
                    filters=self.N_FILTERS[model_size],
                    kernel_size=self.FILTER_KERNELS[0],
                    padding='valid',
                    kernel_initializer=initializer,
                    kernel_regularizer=regularizer,
                    use_bias=True,
                    bias_initializer=initializer,
                    bias_regularizer=regularizer,
                    activation=tf.nn.relu)
                pool0 = tf.layers.max_pooling1d(
                    inputs=conv0, pool_size=pool_size[0], strides=pool_size[0])

            # ================ Layer 2 ================
            with tf.name_scope('cnn-layer-1'):
                conv1 = tf.layers.conv1d(
                    inputs=pool0,
                    filters=self.N_FILTERS[model_size],
                    kernel_size=self.FILTER_KERNELS[1],
                    padding='valid',
                    kernel_initializer=initializer,
                    kernel_regularizer=regularizer,
                    use_bias=True,
                    bias_initializer=initializer,
                    bias_regularizer=regularizer,
                    activation=tf.nn.relu)
                pool1 = tf.layers.max_pooling1d(
                    inputs=conv1, pool_size=pool_size[1], strides=pool_size[1])

            if model_depth == "deep":
                # ================ Layer 3 ================
                # ================ Layer 4 ================
                # ================ Layer 5 ================
                # ================ Layer 6 ================
                # ================ Layer 7 ================
                pass

        with tf.name_scope('flatten'):
            if model_depth == "shallow":
                print(pool1.shape)
                flatten = tf.reshape(pool1, [-1, 14 * 1024])

        with tf.name_scope('fc-layers'):
            # ================ Layer 8 ================
            with tf.name_scope('fc-layer-0'):
                fc0 = tf.layers.dense(
                    inputs=flatten,
                    units=self.FULLY_CONNECTED_OUTPUT[model_size],
                    activation=tf.nn.relu,
                    kernel_initializer=initializer,
                    kernel_regularizer=regularizer,
                    use_bias=True,
                    bias_initializer=initializer,
                    bias_regularizer=regularizer)
                fc0 = tf.layers.dropout(
                    inputs=fc0,
                    rate=fc_dropout,
                    training=self.training)

            if model_depth == "deep":
                # ================ Layer 9 ================
                pass
            else:
                fc_output = fc0

        with tf.name_scope('softmax'):
            logits = tf.layers.dense(
                inputs=fc_output,
                units=n_classes,
                activation=tf.nn.softmax,
                kernel_initializer=initializer)

        with tf.name_scope("training"):
            # using weighted cross-entropy loss since we have imbalanced
            # dataset. put more emphasis on positive training examples
            self.global_step = tf.Variable(
                0, name="global_step", trainable=False)
            print("\nUsing weighted cross-entropy loss with positive_weight=%s"
                  % positive_weight)
            self.positive_weight = tf.constant(
                positive_weight, dtype="float32")
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
