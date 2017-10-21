import argparse
import tensorflow as tf
import data_reader
from vae import VariationalAutoencoder
import matplotlib.pyplot as plt
import numpy as np
import os


def set_labels_axes_plot(ax, title, x_label, y_label, x_coordinates, values):
    ax.set_xticks(x_coordinates)
    ax.set_xticklabels(range(1, len(values) + 1))
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    label_height = values[0]/100
    for i in range(len(values)):
        ax.text(x_coordinates[i], values[i] + label_height, '%.3f' % values[i], ha='center', va='bottom')


def plot_singular_values(sv):
    idx = np.arange(len(sv))
    fig, ax = plt.subplots()
    ax.bar(idx, sv, 0.2)
    set_labels_axes_plot(ax, 'Singular values - Latent space mean(z_mu) matrix', 'singular value number',
                         'value', idx, sv)
    plt.show()


def plot_singular_value_ratio(sv):
    idx = np.arange(len(sv))
    total_sum = np.sum(sv)
    partial_sum = 0
    partial_sums = []
    fig, ax = plt.subplots()
    for i in range(len(sv)):
        partial_sum += sv[i]
        partial_sums.append(partial_sum / total_sum)
    ax.bar(idx, partial_sums, 0.2)
    set_labels_axes_plot(ax, 'Sum of first k singular values / Total sum', 'singular value number', 'ratio value',
                         idx, partial_sums)
    plt.show()


def make_singular_values_plots(vae, x):
    Z = vae.sess.run(vae.z_mu, feed_dict={vae.x: x})
    _, s, _ = np.linalg.svd(Z, full_matrices=False)
    plot_singular_value_ratio(s)
    plot_singular_values(s)


def restore_model(vae, checkpoint_path):
    saver = tf.train.Saver()
    saver.restore(vae.sess, checkpoint_path)
    return vae


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', default='mnist',
                        help='dataset name, choose from: mnist, frey_faces, cifar10, cifar10_full, svhn')
    parser.add_argument('--vae_type', default='flat_binarized', help='')
    parser.add_argument('--latent_dim', type=int, default=20, help='latent dimensions')
    parser.add_argument('--num_epochs', type=int, default=500, help='number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=100, help='batch size')
    parser.add_argument('--hidden_size', type=int, default=200, help='hidden size of network layers')

    args = parser.parse_args()
    data_set_name = args.dataset
    vae_type = args.vae_type

    input_values, width, height, nr_channels = data_reader.read_data_set(data_set_name)

    v = VariationalAutoencoder(width, height, nr_channels, dim_latent=args.latent_dim, input_values=input_values,
                               type=vae_type, learning_rate=args.learning_rate, batch_size=args.batch_size,
                               dim_hidden=args.hidden_size)
    model_path = '../checkpoints/' + data_set_name + '_' + vae_type + '_' + str(v.dim_latent)
    v = restore_model(v, model_path)

    make_singular_values_plots(v, input_values.train.images)

