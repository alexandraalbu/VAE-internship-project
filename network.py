from init_variables import *
import tensorflow as tf


def encoder(x, dim_latent, activation, width, height, nr_channels, dim_hidden=500):
    dim_input = width * height * nr_channels
    w_enc1 = weight_variable('w_enc1', [dim_input, dim_hidden])
    b_enc1 = bias_variable([dim_hidden])
    h1 = activation(tf.matmul(x, w_enc1) + b_enc1)

    w_enc2 = weight_variable('w_enc2', [dim_hidden, dim_hidden])
    b_enc2 = bias_variable([dim_hidden])
    h2 = activation(tf.matmul(h1, w_enc2) + b_enc2)

    w_latent = weight_variable('w_latent', [dim_hidden, 2 * dim_latent])
    b_latent = bias_variable([2 * dim_latent])

    z = tf.matmul(h2, w_latent) + b_latent

    return z[:, :dim_latent], z[:, dim_latent:]


def decoder(z, dim_latent, activation,  width, height, nr_channels,  dim_hidden=500):
    dim_input = width * height * nr_channels
    w_dec1 = weight_variable('w_dec1', [dim_latent, dim_hidden])
    b_dec1 = bias_variable([dim_hidden])
    h1 = activation(tf.matmul(z, w_dec1) + b_dec1)

    w_dec2 = weight_variable('w_dec2', [dim_hidden, dim_hidden])
    b_dec2 = bias_variable([dim_hidden])
    h2 = activation(tf.matmul(h1, w_dec2) + b_dec2)

    w_mu = weight_variable('w_mu', [dim_hidden, dim_input])
    b_mu = bias_variable([dim_input])
    return tf.matmul(h2, w_mu) + b_mu

