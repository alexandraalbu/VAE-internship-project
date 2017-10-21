import tensorflow as tf


def weight_variable(name, shape):
    return tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())
    # return tf.get_variable(name, shape=shape, initializer=tf.truncated_normal_initializer())


def bias_variable(shape):
    return tf.Variable(tf.zeros(shape=shape))
