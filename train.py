import argparse
import tensorflow as tf
import data_reader
import os
from vae import VariationalAutoencoder


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', default='mnist',
                        help='dataset name, choose from: mnist, frey_faces, cifar10, cifar10_full, svhn')
    parser.add_argument('--vae_type', default='flat_binarized',
                        help='choose from: flat_binarized, flat_cont_normal, flat_cont_logit, conv3')
    parser.add_argument('--latent_dim', type=int, default=20, help='latent dimensions')
    parser.add_argument('--num_epochs', type=int, default=500, help='number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=100, help='batch size')
    parser.add_argument('--hidden_size', type=int, default=200, help='hidden size of network layers')
    parser.add_argument('--restore', type=bool, default=False,
                        help='True if you want to continue training an already saved model, False otherwise')

    args = parser.parse_args()
    data_set_name = args.dataset
    vae_type = args.vae_type

    input_values, width, height, nr_channels = data_reader.read_data_set(data_set_name)

    vae = VariationalAutoencoder(width, height, nr_channels, dim_latent=args.latent_dim, input_values=input_values,
                                 type=vae_type, learning_rate=args.learning_rate, batch_size=args.batch_size,
                                 dim_hidden=args.hidden_size)
    model_path = '../checkpoints/' + data_set_name + '_' + vae_type + '_' + str(vae.dim_latent)
    saver = tf.train.Saver()
    if args.restore:
        saver.restore(vae.sess, model_path)

    vae.train(n_epochs=args.num_epochs)
    saver.save(vae.sess, model_path)
